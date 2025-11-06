"""
Modeling approach:
1. Binary classifier: Do (protein A, protein B) come from same isolate?
2. Start with frozen ESM-2 embeddings + MLP baseline
"""
import argparse
import sys
import random
from pathlib import Path

import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    precision_recall_curve
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
from src.utils.path_utils import resolve_run_suffix, build_training_paths
from src.utils.torch_utils import determine_device
from src.utils.esm2_utils import load_esm2_embedding, get_esm2_embedding_dim

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
    interaction features.
    """
    def __init__(
        self,
        pairs: pd.DataFrame,
        embeddings_file: str,
        use_diff: bool = False,
        use_prod: bool = False
        ) -> None:
        """
        Args:
            pairs (pd.DataFrame): DataFrame with ['brc_a', 'brc_b', 'label'].
            embeddings_file (str): Path to ESM-2 embeddings file.
            use_diff (bool): Include absolute difference (|emb_a - emb_b|). Default: False.
            use_prod (bool): Include element-wise product (emb_a * emb_b). Default: False.
        """
        self.pairs = pairs
        self.embeddings_file = embeddings_file
        self.use_diff = use_diff
        self.use_prod = use_prod
        # Preload for efficiency
        # self.embeddings = self._preload_embeddings(embeddings_file, pairs_df)

    def _preload_embeddings(self, embeddings_file: str, pairs: pd.DataFrame) -> dict:
        unique_brcs = set(pairs['brc_a']).union(set(pairs['brc_b']))
        print(f"Pre-loading {len(unique_brcs):,} embeddings into memory...")
        self.embeddings = {}
        with h5py.File(embeddings_file, 'r') as f:
            for brc in tqdm(unique_brcs, desc="Pre-loading embeddings"):
                try:
                    if brc in f:
                        self.embeddings[brc] = load_esm2_embedding(brc, embeddings_file)  # Uses mmap if enabled
                    else:
                        print(f"Warning: Missing embedding for ID {brc}")
                        # Handle missing case (e.g., skip, or use zero vector)
                except KeyError:
                    print(f"Warning: Missing embedding for ID {brc}")
                    continue
        print("✅ Pre-loading complete.")
        return self.embeddings

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the aggregated embedding vector for a segment pair and its label.
        """
        row = self.pairs.iloc[idx]

        # Slower method: load embeddings from file for each pair
        emb_a = load_esm2_embedding(row['brc_a'], self.embeddings_file)
        emb_b = load_esm2_embedding(row['brc_b'], self.embeddings_file)

        # Faster method: use pre-loaded embeddings from memory for each pair
        # emb_a = self.embeddings[row['brc_a']]
        # emb_b = self.embeddings[row['brc_b']]
    
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
    threshold_metric='f1'
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
    
    Returns:
        best_model_path: Path to best model checkpoint
        th: Optimal threshold found on validation set
    """
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
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)

        # Select metric value for early stopping
        if early_stopping_metric == 'loss':
            current_metric_value = val_loss
        elif early_stopping_metric == 'f1':
            current_metric_value = val_f1
        elif early_stopping_metric == 'auc':
            current_metric_value = val_auc

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '\
              f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, '\
              f'Val AUC: {val_auc:.4f} [{early_stopping_metric.upper()}: {current_metric_value:.4f}]')

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
    test_f1 = f1_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)

    # Classification report
    print('\nClassification Report:')
    print(classification_report(test_labels, test_preds, target_names=['Negative', 'Positive']))

    # Create test_res_df
    test_res_df = test_pairs.copy()
    test_res_df['pred_label'] = test_preds
    test_res_df['pred_prob'] = test_probs

    print(f'Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}')
    print(f'Using threshold: {threshold:.4f}')
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
    '--embeddings_dir',
    type=str, default=None,
    help='Path to directory containing esm2_embeddings.h5'
)
parser.add_argument(
    '--output_dir',
    type=str, default=None,
    help='Path to output directory for models'
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
MAX_ISOLATES_TO_PROCESS = config.max_isolates_to_process

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
    run_suffix=RUN_SUFFIX,
    config=config
)

default_dataset_dir = paths['dataset_dir']
default_embeddings_file = paths['embeddings_file']
default_output_dir = paths['output_dir']

# Apply CLI overrides if provided
dataset_dir = Path(args.dataset_dir) if args.dataset_dir else default_dataset_dir
embeddings_file = Path(args.embeddings_dir) / 'esm2_embeddings.h5' if args.embeddings_dir else default_embeddings_file
output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
output_dir.mkdir(parents=True, exist_ok=True)

print(f'\nRun directory:   {DATA_VERSION}{RUN_SUFFIX}')
print(f'dataset_dir:     {dataset_dir}')
print(f'embeddings_file: {embeddings_file}')
print(f'output_dir:      {output_dir}')
print(f'model:           {MODEL_CKPT}')
print(f'batch_size:      {BATCH_SIZE}')

print('\nLoad pair datasets.')
train_pairs = pd.read_csv(dataset_dir / 'train_pairs.csv')
val_pairs   = pd.read_csv(dataset_dir / 'val_pairs.csv')
test_pairs  = pd.read_csv(dataset_dir / 'test_pairs.csv')

# Validate embeddings
print('\nValidate embeddings.')
with h5py.File(embeddings_file, 'r') as f:
    emb_ids = set(f.keys())
    for df in [train_pairs, val_pairs, test_pairs]:
        assert set(df['brc_a']).union(set(df['brc_b'])).issubset(emb_ids), 'Missing embeddings'

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
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
    threshold_metric=THRESHOLD_METRIC
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
