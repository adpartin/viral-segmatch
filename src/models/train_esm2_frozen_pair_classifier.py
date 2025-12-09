"""
Modeling approach:
1. Binary classifier: Do (protein A, protein B) come from same isolate?
2. Start with frozen ESM-2 embeddings + MLP baseline

Requirements:
- Master embeddings file: master_esm2_embeddings.h5 (HDF5 format with 'emb' dataset)
- Parquet index file: master_esm2_embeddings.parquet (required for brc_fea_id to row mapping)
  Both files must be in the same directory and are created together by compute_esm2_embeddings.py
"""
import argparse
import sys
import random
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    precision_recall_curve, precision_score, recall_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# print(f'project_root: {project_root}')

from src.utils.timer_utils import Timer
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds
from src.utils.path_utils import resolve_run_suffix, build_training_paths, build_embeddings_paths
from src.utils.torch_utils import determine_device, create_optimizer, create_lr_scheduler
from src.utils.esm2_utils import load_esm2_embedding, get_esm2_embedding_dim, validate_embeddings_metadata
from src.utils.learning_verification_utils import (
    check_initialization_loss,
    compute_baseline_metrics,
    plot_learning_curves
)
import h5py

total_timer = Timer()


def find_optimal_threshold_pr(y_true, y_probs, metric='f1'):
    """
    Find optimal threshold using Precision-Recall curve.
    TODO: allow an option to generate a plot of the PR curve, showing the optimal threshold and the best score.

    This method is preferred for imbalanced datasets and when optimizing F1.
    It's faster than grid search and directly optimizes F1 score.

    Args:
        y_true: True binary labels (array-like)
        y_probs: Predicted probabilities (array-like)
        metric: Metric to optimize ('f1', 'f0.5', 'f2')
            - 'f1': Maximize F1 score (harmonic mean of precision and recall)
            - 'f0.5': Emphasize precision more (F0.5 = (1+0.5²) * P*R / (0.5²*P + R))
            - 'f2': Emphasize recall more (F2 = (1+2²) * P*R / (2²*P + R))

    Returns:
        optimal_threshold: Threshold that maximizes the specified metric
        best_score: Best score achieved at optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    # Handle edge case: no thresholds (all predictions same class)
    if len(thresholds) == 0:
        return 0.5, 0.0

    # Calculate F-beta scores
    if metric == 'f1':
        # F1 = 2 * (precision * recall) / (precision + recall)
        # Add small epsilon to avoid division by zero
        f_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    elif metric == 'f0.5':
        # F0.5 emphasizes precision: beta = 0.5
        beta_sq = 0.5 ** 2
        f_scores = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-10)
    elif metric == 'f2':
        # F2 emphasizes recall: beta = 2
        beta_sq = 2 ** 2
        f_scores = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-10)
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from 'f1', 'f0.5', 'f2'")

    # Find best F-score (excluding last point which has threshold=None)
    # The last point in precision_recall_curve has threshold=None and corresponds
    # to the case where all predictions are positive
    optimal_idx = np.argmax(f_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    best_score = f_scores[optimal_idx]

    return optimal_threshold, best_score


class SegmentPairDataset(Dataset):
    """
    Dataset for segment pairs with ESM-2 embeddings, supporting optional
    interaction features. Uses row-based indexing for master cache access.
    """
    # Shared cache so multiple datasets (train/val/test) can reuse the same embeddings
    _shared_embedding_cache: dict[str, np.ndarray] = {}

    def __init__(
        self,
        pairs: pd.DataFrame,
        embeddings_file: str,
        use_diff: bool = False,
        use_prod: bool = False,
        use_parquet: bool = True,
        preload_embeddings: bool = True
        ) -> None:
        """
        Args:
            pairs (pd.DataFrame): DataFrame with ['brc_a', 'brc_b', 'label'].
            embeddings_file (str): Path to master HDF5 cache file.
            use_diff (bool): Include absolute difference (|emb_a - emb_b|). Default: False.
            use_prod (bool): Include element-wise product (emb_a * emb_b). Default: False.
            use_parquet (bool): Use parquet index file for brc_fea_id to row mapping. Default: True.
            preload_embeddings (bool): Preload all required embeddings into memory for faster access. Default: True.
        """
        self.pairs = pairs
        self.embeddings_file = embeddings_file
        self.use_diff = use_diff
        self.use_prod = use_prod
        self.use_parquet = use_parquet
        self.preload_embeddings = preload_embeddings
        
        # Build id_to_row mapping (must be done before opening H5)
        self.id_to_row = self._build_id_to_row()
        
        # Open H5 file to read metadata and optionally preload embeddings
        self.h5 = h5py.File(embeddings_file, 'r')
        
        # Require master cache format (strict - no old format support)
        if 'emb' not in self.h5 or 'emb_keys' not in self.h5:
            raise ValueError(
                f"❌ Invalid embeddings file format: {embeddings_file}. "
                "Master cache format required (with 'emb' and 'emb_keys' datasets). "
                "Old format is not supported."
            )
        
        # Display metadata (required for master cache format)
        if 'model_name' in self.h5.attrs:
            print(f"📋 Embedding metadata: model={self.h5.attrs.get('model_name', 'unknown')}, "
                  f"pooling={self.h5.attrs.get('pooling', 'unknown')}, "
                  f"layer={self.h5.attrs.get('layer', 'unknown')}, "
                  f"max_length={self.h5.attrs.get('max_length', 'unknown')}, "
                  f"precision={self.h5.attrs.get('precision', 'unknown')}")
        
        # Preload embeddings into memory (shared across datasets) for faster access
        if self.preload_embeddings:
            self.embeddings_cache = self._get_or_preload_shared_embeddings()
            self.h5.close()
            self.h5 = None
        else:
            self.embeddings_cache = None
    
    def _build_id_to_row(self) -> dict:
        """
        Build mapping from brc_fea_id to row index in master cache.
        
        Returns:
            dict: Mapping {brc_fea_id: row_index}
        """
        if self.use_parquet:
            # Use parquet index file
            index_file = Path(self.embeddings_file).with_suffix('.parquet')
            if index_file.exists():
                df = pd.read_parquet(index_file)
                return dict(zip(df['brc_fea_id'], df['row']))
            else:
                print(f"⚠️  Parquet index not found: {index_file}. Falling back to H5 key scan.")
                self.use_parquet = False
        
        # Master cache format requires parquet index
        raise ValueError(
            f"❌ Parquet index not found: {index_file}. "
            "Master cache format requires parquet index for brc_fea_id to row mapping. "
            "Ensure the index file exists or regenerate embeddings."
        )

    def _get_or_preload_shared_embeddings(self) -> np.ndarray:
        """
        Preload embeddings into a shared in-memory cache so multiple datasets
        (train/val/test) reuse the same arrays without paying the cost per dataset.
        """
        if self.embeddings_file in self._shared_embedding_cache:
            return self._shared_embedding_cache[self.embeddings_file]

        print("Pre-loading entire embeddings matrix into memory (shared cache)...")
        emb_matrix = self.h5['emb'][:].astype(np.float32)
        self._shared_embedding_cache[self.embeddings_file] = emb_matrix
        print(f"✅ Pre-loading complete ({emb_matrix.shape[0]:,} embeddings cached).")
        return emb_matrix

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the aggregated embedding vector for a segment pair and its label.
        Uses row-based indexing for fast access to master cache.
        """
        row = self.pairs.iloc[idx]
        
        # Get row indices for brc_a and brc_b
        row_a = self.id_to_row.get(row['brc_a'], -1)
        row_b = self.id_to_row.get(row['brc_b'], -1)
        
        if row_a == -1 or row_b == -1:
            missing = []
            if row_a == -1:
                missing.append(f"brc_a={row['brc_a']}")
            if row_b == -1:
                missing.append(f"brc_b={row['brc_b']}")
            raise KeyError(f"❌ Missing embeddings for: {', '.join(missing)}")
        
        # Access embeddings from cache (preloaded) or master cache (on-demand)
        if self.embeddings_cache is not None:
            emb_a = self.embeddings_cache[row_a]
            emb_b = emb_a if row_a == row_b else self.embeddings_cache[row_b]
        else:
            # Access from HDF5 file directly (slower, but uses less memory)
            emb_a = self.h5['emb'][row_a].astype(np.float32)
            if row_a == row_b:
                emb_b = emb_a.copy()
            else:
                emb_b = self.h5['emb'][row_b].astype(np.float32)
    
        # Start with the standard concatenation
        features = [emb_a, emb_b]

        # Add absolute difference |A - B| as interaction feature conditionally
        # Element-wise Difference
        # Measures the degree of difference between two embeddings
        if self.use_diff:
            diff = np.abs(emb_a - emb_b)
            features.append(diff)

        # Add element-wise product A * B as interaction feature conditionally
        # Hadamard Product (or Element-wise Multiplication)
        # Captures how much two embeddings agree or correlate on each feature
        if self.use_prod:    
            prod = emb_a * emb_b
            features.append(prod)

        # Concatenate all chosen features
        emb = np.concatenate(features)

        # Convert to torch tensor
        emb = torch.tensor(emb, dtype=torch.float)
        label = torch.tensor(row['label'], dtype=torch.float)
        return emb, label
    
    def __del__(self):
        """Close H5 file when dataset is destroyed."""
        if hasattr(self, 'h5') and self.h5 is not None:
            self.h5.close()


class MLPClassifier(nn.Module):
    """
    MLP classifier for segment pairs.
    """
    def __init__(
        self,
        input_dim: int=2560,
        hidden_dims: list[int]=[512, 256, 64],
        dropout: float=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    output_dir,
    early_stopping_metric='loss',
    threshold_metric='f1',
    lr_scheduler=None
    ) -> tuple[str, float]:
    """
    Train the MLP classifier with early stopping based on a configurable metric.
    
    Args:
        early_stopping_metric: Metric to use for early stopping ('loss', 'f1', 'auc')
            - 'loss': Lower is better (default, backward compatible)
            - 'f1': Higher is better
            - 'auc': Higher is better
        threshold_metric: Metric to optimize for threshold selection ('f1', 'f0.5', 'f2', or None)
            - 'f1': Maximize F1 score (default)
            - 'f0.5': Emphasize precision more
            - 'f2': Emphasize recall more
            - None: Skip optimization, use default threshold 0.5
        lr_scheduler: Optional learning rate scheduler (ReduceLROnPlateau, CosineAnnealingLR, StepLR, or None)
            - If ReduceLROnPlateau: step() called with metric value
            - If other schedulers: step() called after each epoch
            - If None: No learning rate scheduling
    
    Returns:
        best_model_path: Path to best model checkpoint
        th: Optimal threshold found on validation set
    """
    # Karpathy-style initialization check
    check_initialization_loss(model, train_loader, criterion, device)
    
    # Compute baseline metrics for comparison
    # Get validation labels for baseline computation
    val_labels_for_baseline = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            val_labels_for_baseline.extend(batch_y.cpu().numpy())
    baseline_metrics = compute_baseline_metrics(val_labels_for_baseline)
    
    print(f"\n{'='*60}")
    print("BASELINE METRICS (for comparison)")
    print('='*60)
    print(f"Random classifier F1: {baseline_metrics['random_f1']:.4f}")
    print(f"Majority class ({baseline_metrics['majority_class']}) F1: {baseline_metrics['majority_f1']:.4f}")
    print(f"Majority class accuracy: {baseline_metrics['majority_acc']:.4f}")
    print(f"Class balance: {baseline_metrics['class_balance']['positive']} positive, {baseline_metrics['class_balance']['negative']} negative ({baseline_metrics['class_balance']['ratio']:.2%})")
    print('='*60 + "\n")
    
    # Track training history for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],  # F1 for positive class (same isolate) - binary
        'train_f1_macro': [],  # F1 macro (average of both classes)
        'train_precision': [],  # Precision for positive class (measures FP)
        'train_recall': [],  # Recall for positive class (measures FN)
        'train_auc': [],
        'val_f1': [],  # F1 for positive class (same isolate) - binary
        'val_f1_macro': [],  # F1 macro (average of both classes)
        'val_precision': [],  # Precision for positive class (measures FP)
        'val_recall': [],  # Recall for positive class (measures FN)
        'val_auc': [],
        'learning_rate': []  # Track learning rate over epochs
    }
    
    patience_counter = 0
    best_model_path = output_dir / 'best_model.pt'
    best_val_probs = None
    best_val_labels = None
    
    # Initialize best metric tracking based on metric type
    if early_stopping_metric == 'loss':
        best_metric_value = float('inf')
        is_higher_better = False
    elif early_stopping_metric in ['f1', 'auc']:
        best_metric_value = -1.0
        is_higher_better = True
    else:
        raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}. Choose from 'loss', 'f1', 'auc'")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}',
            dynamic_ncols=True, leave=False)
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            # Forward pass
            preds = model(batch_x).squeeze() 
            # Backward pass
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Compute training metrics AFTER training (with model in eval mode for consistency)
        # This ensures all training predictions come from the same model state
        model.eval()
        train_preds, train_probs, train_labels = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).squeeze()
                train_probs.extend(torch.sigmoid(preds).cpu().numpy())
                train_preds.extend((torch.sigmoid(preds) > 0.5).float().cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())
        train_preds = np.array(train_preds)
        train_probs = np.array(train_probs)
        # Compute metrics for positive class (label=1, same isolate)
        # Explicit parameters: average='binary', pos_label=1
        train_f1 = f1_score(train_labels, train_preds, average='binary', pos_label=1)
        train_f1_macro = f1_score(train_labels, train_preds, average='macro')
        train_precision = precision_score(train_labels, train_preds, average='binary', pos_label=1, zero_division=0)
        train_recall = recall_score(train_labels, train_preds, average='binary', pos_label=1, zero_division=0)
        train_auc = roc_auc_score(train_labels, train_probs)

        model.eval()
        val_loss = 0
        val_preds, val_probs, val_labels = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).squeeze()
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_probs.extend(torch.sigmoid(preds).cpu().numpy())
                val_preds.extend((torch.sigmoid(preds) > 0.5).float().cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_preds = np.array(val_preds)
        val_probs = np.array(val_probs)
        # Compute metrics for positive class (label=1, same isolate)
        # Explicit parameters: average='binary', pos_label=1
        val_f1 = f1_score(val_labels, val_preds, average='binary', pos_label=1)
        val_f1_macro = f1_score(val_labels, val_preds, average='macro')
        val_precision = precision_score(val_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        val_auc = roc_auc_score(val_labels, val_probs)

        # Select metric value for early stopping
        if early_stopping_metric == 'loss':
            current_metric_value = val_loss
        elif early_stopping_metric == 'f1':
            current_metric_value = val_f1
        elif early_stopping_metric == 'auc':
            current_metric_value = val_auc

        # Update learning rate scheduler if provided
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau uses metric value, not loss
                lr_scheduler.step(current_metric_value if is_higher_better else -current_metric_value)
            else:
                lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Track history for plotting
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['train_f1_macro'].append(train_f1_macro)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_auc'].append(train_auc)
        history['val_f1'].append(val_f1)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_auc'].append(val_auc)
        history['learning_rate'].append(current_lr)
        
        # Print simplified metrics (all metrics still saved to history/CSV)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, '
              f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f} '
              f'[{early_stopping_metric.upper()}: {current_metric_value:.4f}, LR: {current_lr:.6f}]')

        # Check if current metric is better than best
        is_better = False
        if is_higher_better:
            is_better = current_metric_value > best_metric_value
        else:
            is_better = current_metric_value < best_metric_value

        if is_better:
            best_metric_value = current_metric_value
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # Save validation probabilities and labels for threshold optimization
            best_val_probs = val_probs.copy()
            best_val_labels = np.array(val_labels)
            print(f'  ✓ New best {early_stopping_metric}: {best_metric_value:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered (no improvement in {early_stopping_metric} for {patience} epochs).')
                break

    # Save training history to CSV
    if len(history['train_loss']) > 0:
        history_df = pd.DataFrame({
            'epoch': range(1, len(history['train_loss']) + 1),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_f1': history.get('train_f1', [None] * len(history['train_loss'])),
            'train_f1_macro': history.get('train_f1_macro', [None] * len(history['train_loss'])),
            'train_precision': history.get('train_precision', [None] * len(history['train_loss'])),
            'train_recall': history.get('train_recall', [None] * len(history['train_loss'])),
            'train_auc': history.get('train_auc', [None] * len(history['train_loss'])),
            'val_f1': history.get('val_f1', [None] * len(history['train_loss'])),
            'val_f1_macro': history.get('val_f1_macro', [None] * len(history['train_loss'])),
            'val_precision': history.get('val_precision', [None] * len(history['train_loss'])),
            'val_recall': history.get('val_recall', [None] * len(history['train_loss'])),
            'val_auc': history.get('val_auc', [None] * len(history['train_loss'])),
            'learning_rate': history.get('learning_rate', [None] * len(history['train_loss']))
        })
        history_csv = output_dir / 'training_history.csv'
        history_df.to_csv(history_csv, index=False)
        print(f"Training history saved to: {history_csv}")
    
    # Plot learning curves
    if len(history['train_loss']) > 0:
        plot_learning_curves(history, output_dir)
        
        # Print learning summary
        print(f"\n{'='*60}")
        print("LEARNING SUMMARY")
        print('='*60)
        print(f"Initial train loss: {history['train_loss'][0]:.4f}")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Initial val F1 (binary): {history['val_f1'][0]:.4f}")
        print(f"Best val F1 (binary): {max(history['val_f1']):.4f}")
        if 'val_f1_macro' in history and len(history['val_f1_macro']) > 0:
            print(f"Best val F1 (macro): {max(history['val_f1_macro']):.4f}")
        print(f"Baseline (majority class) F1: {baseline_metrics['majority_f1']:.4f}")
        if max(history['val_f1']) > baseline_metrics['majority_f1']:
            print("✅ Model learned! (F1 > baseline)")
        else:
            print("⚠️  Model did not beat baseline - may indicate learning issues")
        print('='*60 + "\n")

    # Find optimal threshold on validation set using best model's predictions
    # If threshold_metric is None, skip optimization and use default 0.5
    if threshold_metric is None:
        th = 0.5
        print(f'\nUsing default threshold: {th:.4f} (threshold optimization disabled)')
    elif best_val_probs is not None and best_val_labels is not None:
        th, best_metric_score = find_optimal_threshold_pr(
            best_val_labels, best_val_probs, metric=threshold_metric
        )
        print(f'\nOptimal threshold (optimizing {threshold_metric}): {th:.4f}')
        print(f'Best {threshold_metric} score on validation: {best_metric_score:.4f}')

        # Save optimal threshold
        threshold_file = output_dir / 'optimal_threshold.txt'
        with open(threshold_file, 'w') as f:
            f.write(f'{th}\n')
            f.write(f'metric: {threshold_metric}\n')
            f.write(f'best_score: {best_metric_score:.4f}\n')
    else:
        th = 0.5
        print(f'\nUsing default threshold: {th:.4f} (no validation data available)')

    return best_model_path, th


def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    test_pairs,
    threshold=0.5
    ) -> pd.DataFrame:
    """
    Evaluate the model on the test set and compute metrics.

    Args:
        threshold: Classification threshold (default: 0.5)
    """
    model.eval()
    test_loss = 0
    test_preds, test_probs, test_labels = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x).squeeze()
            loss = criterion(preds, batch_y)
            test_loss += loss.item() * batch_x.size(0)
            test_probs.extend(torch.sigmoid(preds).cpu().numpy())
            test_preds.extend((torch.sigmoid(preds) > threshold).float().cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_probs = np.array(test_probs)
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    # Compute metrics for positive class (label=1, same isolate)
    # Explicit parameters: average='binary', pos_label=1
    test_f1 = f1_score(test_labels, test_preds, average='binary', pos_label=1)
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_precision = precision_score(test_labels, test_preds, average='binary', pos_label=1, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='binary', pos_label=1, zero_division=0)
    test_auc = roc_auc_score(test_labels, test_probs)

    # Classification report
    print('\nClassification Report:')
    print(classification_report(test_labels, test_preds, target_names=['Negative', 'Positive']))

    # Create test_res_df
    test_res_df = test_pairs.copy()
    test_res_df['pred_label'] = test_preds
    test_res_df['pred_prob'] = test_probs

    print(f'Test Loss: {test_loss:.4f}, Test F1 (binary): {test_f1:.4f}, Test F1 (macro): {test_f1_macro:.4f}')
    print(f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test AUC: {test_auc:.4f}')
    print(f'Using threshold: {threshold:.4f}')
    print(f'Note: Precision measures False Positives (FP), Recall measures False Negatives (FN)')
    print(f'      F1 (binary) focuses on positive class, F1 (macro) averages both classes')
    return test_res_df


# Parser
parser = argparse.ArgumentParser(description='Train ESM-2 frozen pair classifier')
parser.add_argument(
    '--config_bundle',
    type=str, default=None,
    help='Config bundle to use (e.g., flu_a, bunya).'
)
parser.add_argument(
    '--cuda_name', '-c',
    type=str, default='cuda:7',
    help='CUDA device to use (default: cuda:7)'
)
parser.add_argument(
    '--dataset_dir',
    type=str, default=None,
    help='Path to directory containing train_pairs.csv, val_pairs.csv, test_pairs.csv'
)
parser.add_argument(
    '--embeddings_file',
    type=str, default=None,
    help='Path to master HDF5 cache file (default: auto-detect from config). '
         'Note: Requires corresponding .parquet index file in same directory for brc_fea_id to row mapping.'
)
parser.add_argument(
    '--output_dir',
    type=str, default=None,
    help='Path to output directory for models'
)
parser.add_argument(
    '--run_output_subdir',
    type=str, default=None,
    help='Optional subdirectory name under default output_dir (e.g., experiment/run id).'
)
args = parser.parse_args()

# Load config
config_path = str(project_root / 'conf')  # Pass the config path explicitly
config_bundle = args.config_bundle
if config_bundle is None:
    raise ValueError("❌ Must provide --config_bundle")
config = get_virus_config_hydra(config_bundle, config_path=config_path)
print_config_summary(config)

# Extract config values
VIRUS_NAME = config.virus.virus_name
DATA_VERSION = config.virus.data_version
RANDOM_SEED = resolve_process_seed(config, 'training')
# TASK_NAME = config.dataset.task_name
MODEL_CKPT = config.embeddings.model_ckpt
ESM2_MAX_RESIDUES = config.embeddings.esm2_max_residues
POOLING = config.embeddings.pooling
LAYER = config.embeddings.layer
EMB_STORAGE_PRECISION = config.embeddings.emb_storage_precision
MAX_ISOLATES_TO_PROCESS = getattr(config.dataset, 'max_isolates_to_process', None)

# Training hyperparameters from config
BATCH_SIZE = config.training.batch_size
LEARNING_RATE = config.training.learning_rate
EPOCHS = config.training.epochs
HIDDEN_DIMS = config.training.hidden_dims
USE_DIFF = config.training.use_diff
USE_PROD = config.training.use_prod
DROPOUT = config.training.dropout
PATIENCE = config.training.patience
EARLY_STOPPING_METRIC = config.training.early_stopping_metric
THRESHOLD_METRIC = config.training.threshold_metric
USE_LR_SCHEDULER = getattr(config.training, 'use_lr_scheduler', False)
LR_SCHEDULER_TYPE = getattr(config.training, 'lr_scheduler', 'reduce_on_plateau')
LR_SCHEDULER_PATIENCE = getattr(config.training, 'lr_scheduler_patience', 5)
LR_SCHEDULER_FACTOR = getattr(config.training, 'lr_scheduler_factor', 0.5)
LR_SCHEDULER_MIN_LR = getattr(config.training, 'lr_scheduler_min_lr', 1e-6)

print(f"\n{'='*40}")
print(f"Virus: {VIRUS_NAME}")
print(f"Config bundle: {config_bundle}")
print(f"{'='*40}")

# Resolve run suffix (manual override in config or auto-generate from sampling params)
# This ensures consistency with preprocessing, embeddings, and dataset scripts
RUN_SUFFIX = resolve_run_suffix(
    config=config,
    max_isolates=MAX_ISOLATES_TO_PROCESS,
    seed=RANDOM_SEED,
    auto_timestamp=True
)

# Set deterministic seeds for training
if RANDOM_SEED is not None:
    set_deterministic_seeds(RANDOM_SEED, cuda_deterministic=True)
    print(f'Set deterministic seeds for training (seed: {RANDOM_SEED})')
else:
    print('No seed set - training will be non-deterministic')

# Build paths using the new path utilities
paths = build_training_paths(
    project_root=project_root,
    virus_name=VIRUS_NAME,
    data_version=DATA_VERSION,
    # run_suffix=RUN_SUFFIX,
    run_suffix="",  # Not used - kept for backward compatibility
    config=config
)

default_embeddings_file = paths['embeddings_file']
default_output_dir = paths['output_dir']

# Apply CLI overrides if provided
# Dataset directory MUST be provided explicitly (not built from config)
if not args.dataset_dir:
    raise ValueError(
        "❌ --dataset_dir is required. "
        "Datasets are in runs/ subdirectories: "
        "data/datasets/{virus}/{data_version}/runs/dataset_{config_bundle}_{timestamp}/"
    )
dataset_dir = Path(args.dataset_dir)

embeddings_file = Path(args.embeddings_file) if args.embeddings_file else default_embeddings_file

# Output directory: always use runs/ subdirectory structure
if args.output_dir:
    output_dir = Path(args.output_dir)
elif args.run_output_subdir:
    # Structure: models/{virus}/{data_version}/runs/{run_id}/
    output_dir = default_output_dir / 'runs' / args.run_output_subdir
else:
    # Fallback: create a run directory with config bundle name
    # This shouldn't happen if shell script is used correctly
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_run_id = f"training_{config_bundle}_{timestamp}"
    output_dir = default_output_dir / 'runs' / fallback_run_id
    print(f"⚠️  Warning: No run_output_subdir provided, using fallback: {fallback_run_id}")
output_dir.mkdir(parents=True, exist_ok=True)

print(f'\nConfig bundle:    {config_bundle}')
print(f'Run suffix:       {RUN_SUFFIX if RUN_SUFFIX else "(none)"}')
print(f'Dataset dir:      {dataset_dir}')
print(f'Embeddings file:  {embeddings_file}')
print(f'Output dir:       {output_dir}')  # This line is parsed by stage4_train.sh
print(f'Run ID:           {args.run_output_subdir if args.run_output_subdir else "auto-generated"}')
print(f'model:           {MODEL_CKPT}')
print(f'batch_size:      {BATCH_SIZE}')

print('\nLoad pair datasets.')
train_pairs = pd.read_csv(dataset_dir / 'train_pairs.csv')
val_pairs   = pd.read_csv(dataset_dir / 'val_pairs.csv')
test_pairs  = pd.read_csv(dataset_dir / 'test_pairs.csv')

# Validate embeddings metadata matches config
print('\nValidate embeddings metadata.')
validate_embeddings_metadata(
    embeddings_file=str(embeddings_file),
    model_name=MODEL_CKPT,
    max_length=ESM2_MAX_RESIDUES + 2,
    pooling=POOLING,
    layer=LAYER,
    emb_storage_precision=EMB_STORAGE_PRECISION
)

# Validate that all required embeddings exist (check parquet index)
print('Validate embedding availability.')
index_file = Path(embeddings_file).with_suffix('.parquet')
if not index_file.exists():
    raise ValueError(
        f"❌ Parquet index not found: {index_file}. "
        "Master cache format requires parquet index for validation."
    )

index_df = pd.read_parquet(index_file)
available_ids = set(index_df['brc_fea_id'].unique())
for df_name, df in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
    required_ids = set(df['brc_a']).union(set(df['brc_b']))
    missing = required_ids - available_ids
    if missing:
        raise ValueError(
            f"❌ Missing embeddings for {len(missing)} IDs in {df_name} set: {list(missing)[:5]}..."
        )
print(f"All required embeddings available ({len(available_ids)} total embeddings)")

# Create datasets
train_dataset = SegmentPairDataset(train_pairs, embeddings_file,
    use_diff=USE_DIFF, use_prod=USE_PROD)
val_dataset = SegmentPairDataset(val_pairs, embeddings_file,
    use_diff=USE_DIFF, use_prod=USE_PROD)
test_dataset = SegmentPairDataset(test_pairs, embeddings_file,
    use_diff=USE_DIFF, use_prod=USE_PROD)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CUDA device
CUDA_NAME = args.cuda_name
device = determine_device(CUDA_NAME)

# Get embedding dimension from model checkpoint
EMBED_DIM = get_esm2_embedding_dim(MODEL_CKPT)

# Compute input dimension based on feature flags
mlp_input_dim = 2 * EMBED_DIM  # Default: concat [emb_a, emb_b] = 2 * EMBED_DIM
feature_desc = "2"
if USE_DIFF:
    mlp_input_dim += EMBED_DIM  # add |A - B|
    feature_desc += "+1"
if USE_PROD:
    mlp_input_dim += EMBED_DIM  # add A * B
    feature_desc += "+1"
print(f"MLP Input Dimension: {mlp_input_dim} ({feature_desc} * {EMBED_DIM})") 

# Initialize model
model = MLPClassifier(
    input_dim=mlp_input_dim,
    hidden_dims=HIDDEN_DIMS,
    dropout=DROPOUT,
)
model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Get optimizer parameters from config
OPTIMIZER_NAME = getattr(config.training, 'optimizer', 'adam')
OPTIMIZER_WEIGHT_DECAY = getattr(config.training, 'weight_decay', 0.0)
OPTIMIZER_MOMENTUM = getattr(config.training, 'momentum', 0.9)

# Create optimizer using utility function
optimizer = create_optimizer(
    model=model,
    optimizer_name=OPTIMIZER_NAME,
    learning_rate=LEARNING_RATE,
    weight_decay=OPTIMIZER_WEIGHT_DECAY,
    momentum=OPTIMIZER_MOMENTUM
)

# Create learning rate scheduler using utility function
lr_scheduler = create_lr_scheduler(
    optimizer=optimizer,
    scheduler_type=LR_SCHEDULER_TYPE if USE_LR_SCHEDULER else None,
    early_stopping_metric=EARLY_STOPPING_METRIC,
    epochs=EPOCHS,
    patience=LR_SCHEDULER_PATIENCE,
    factor=LR_SCHEDULER_FACTOR,
    min_lr=LR_SCHEDULER_MIN_LR
)

# Train
print('\nTrain model.')
print(f'Early stopping metric: {EARLY_STOPPING_METRIC}')
print(f'Threshold optimization metric: {THRESHOLD_METRIC}')
best_model_path, optimal_threshold = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=EPOCHS,
    patience=PATIENCE,
    output_dir=output_dir,
    early_stopping_metric=EARLY_STOPPING_METRIC,
    threshold_metric=THRESHOLD_METRIC,
    lr_scheduler=lr_scheduler
)

# Evaluate
print('\nEvaluate model.')
model.load_state_dict(torch.load(best_model_path))
test_res_df = evaluate_model(
    model, test_loader, criterion, device, test_pairs,
    threshold=optimal_threshold
)

# Save raw predictions
preds_file = output_dir / 'test_predicted.csv'
print(f'\nSave raw predictions to: {preds_file}')
test_res_df.to_csv(preds_file, index=False)

print(f'\n✅ Finished {Path(__file__).name}!')
total_timer.display_timer()
