"""
Modeling approach:
1. Binary classifier: Do (protein A, protein B) come from same isolate?
2. Start with frozen ESM-2 embeddings + MLP baseline

Requirements:
- Master embeddings file: master_esm2_embeddings.h5 (HDF5 format with 'emb' dataset)
- Parquet index file: master_esm2_embeddings.parquet (required for brc_fea_id to row mapping)
  Both files must be in the same directory and are created together by compute_esm2_embeddings.py
"""
# --- Prevent TensorFlow from grabbing GPU memory ---
# TF is installed in the Polaris system conda env and gets loaded transitively
# (HuggingFace transformers checks for TF availability on import via esm2_utils).
# TF's default behavior is to eagerly allocate ALL GPU memory, which can starve
# PyTorch and cause OOM at model.to(device). These env vars must be set BEFORE
# any import that could trigger TF initialization.
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')          # Suppress TF C++ logs
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')   # Don't pre-allocate all GPU memory
# --- End TensorFlow prevention ---

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.timer_utils import Timer
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary, save_config
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds
from src.utils.path_utils import resolve_run_suffix, build_training_paths, build_embeddings_paths
from src.utils.torch_utils import determine_device, create_optimizer, create_lr_scheduler
from src.utils.esm2_utils import load_esm2_embedding, get_esm2_embedding_dim, validate_embeddings_metadata
from src.utils.learning_verification_utils import (
    check_initialization_loss,
    compute_baseline_metrics,
    plot_learning_curves
)
from src.utils.kmer_utils import load_kmer_index, load_kmer_matrix, get_kmer_pair_features
from src.models._pair_metrics import (
    compute_pair_metrics,
    find_optimal_threshold_pr,
    swap_pairs_df_columns,
)
import h5py

total_timer = Timer()


def parse_interaction_flags(interaction: str) -> tuple[bool, bool, bool, bool]:
    """
    Parse interaction spec string into (use_concat, use_diff, use_prod, use_unit_diff).
    Accepts: "concat", "diff", "prod", "unit_diff", or combinations like "concat+diff" (any order).

    unit_diff: L2-normalized difference (emb_a - emb_b) / (||emb_a - emb_b|| + eps).
    Preserves only the *direction* of the difference, removing magnitude information.
    Use as a diagnostic to test whether the model relies on diff magnitude vs direction.
    """
    if interaction is None:
        return False, False, False, False
    tokens = [t.strip().lower() for t in str(interaction).split('+') if t.strip()]
    allowed = {"concat", "diff", "prod", "unit_diff"}
    unknown = [t for t in tokens if t not in allowed]
    if unknown:
        raise ValueError(f"Unknown interaction tokens: {unknown} (allowed: {sorted(allowed)})")
    return ("concat" in tokens, "diff" in tokens, "prod" in tokens, "unit_diff" in tokens)


class ESMPairDataset(Dataset):
    """
    Dataset for segment pairs with ESM-2 protein embeddings.

    Always returns (emb_a, emb_b), label. Interaction features (concat / diff /
    unit_diff / prod) are computed inside the MLP, not here. Naming parallels
    KmerPairDataset so it's clear which feature source each dataset wraps.
    Uses row-based indexing into the master HDF5 cache.
    """
    # Shared cache so multiple datasets (train/val/test) can reuse the same embeddings
    _shared_embedding_cache: dict[str, np.ndarray] = {}

    def __init__(
        self,
        pairs: pd.DataFrame,
        embeddings_file: str,
        use_parquet: bool = True,
        preload_embeddings: bool = True,
        ) -> None:
        """
        Args:
            pairs (pd.DataFrame): DataFrame with ['brc_a', 'brc_b', 'label'].
            embeddings_file (str): Path to master HDF5 cache file.
            use_parquet (bool): Use parquet index file for brc_fea_id to row mapping. Default: True.
            preload_embeddings (bool): Preload all required embeddings into memory for faster access. Default: True.
        """
        self.pairs = pairs
        self.embeddings_file = embeddings_file
        self.use_parquet = use_parquet
        self.preload_embeddings = preload_embeddings
        
        # Build id_to_row mapping (must be done before opening H5)
        self.id_to_row = self._build_id_to_row()
        
        # Open H5 file to read metadata and optionally preload embeddings
        self.h5 = h5py.File(embeddings_file, 'r')
        
        # Require master cache format (strict - no old format support)
        if 'emb' not in self.h5 or 'emb_keys' not in self.h5:
            raise ValueError(
                f"Invalid embeddings file format: {embeddings_file}. "
                "Master cache format required (with 'emb' and 'emb_keys' datasets). "
                "Old format is not supported."
            )
        
        # Display metadata (required for master cache format)
        if 'model_name' in self.h5.attrs:
            print(f"Embedding metadata: model={self.h5.attrs.get('model_name', 'unknown')}, "
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
                print(f"WARNING: Parquet index not found: {index_file}. Falling back to H5 key scan.")
                self.use_parquet = False
        
        # Master cache format requires parquet index
        raise ValueError(
            f"Parquet index not found: {index_file}. "
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
        print(f"Pre-loading complete ({emb_matrix.shape[0]:,} embeddings cached).")
        return emb_matrix

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the aggregated embedding vector for a segment pair and its label.
        Uses row-based indexing for fast access to master cache.
        """
        # breakpoint()
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
            raise KeyError(f"Missing embeddings for: {', '.join(missing)}")

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

        emb_a_t = torch.tensor(emb_a, dtype=torch.float)
        emb_b_t = torch.tensor(emb_b, dtype=torch.float)
        label = torch.tensor(row['label'], dtype=torch.float)
        return (emb_a_t, emb_b_t), label

    def __del__(self):
        """Close H5 file when dataset is destroyed."""
        if hasattr(self, 'h5') and self.h5 is not None:
            self.h5.close()


class KmerPairDataset(Dataset):
    """Dataset for segment pairs with k-mer features.

    Returns (emb_a, emb_b), label — same interface as ESMPairDataset
    so the training loop and MLPClassifier work unchanged.

    Pairs are looked up via composite key (assembly_id::genbank_ctg_id)
    using ctg_a/ctg_b columns in the pair CSV.
    """

    def __init__(
        self,
        pairs: pd.DataFrame,
        kmer_matrix,   # scipy sparse CSR
        key_to_row: dict,
        ) -> None:
        # Pre-compute row indices for each pair
        keys_a = pairs['assembly_id_a'].astype(str) + '::' + pairs['ctg_a'].astype(str)
        keys_b = pairs['assembly_id_b'].astype(str) + '::' + pairs['ctg_b'].astype(str)
        rows_a = keys_a.map(key_to_row).astype(int).values
        rows_b = keys_b.map(key_to_row).astype(int).values

        # Subset the sparse matrix to only the rows used by this fold's pairs.
        # Full matrix is 868K×4096 (14.2 GB dense). Each fold uses ~100-200K unique
        # rows, so subsetting cuts memory 4-8x and dramatically improves cache locality.
        # Without subsetting, 4 concurrent folds on one node = 56.9 GB → OOM.
        # With subsetting, 4 folds × ~3.5 GB = 14 GB → fits comfortably.
        unique_rows = np.unique(np.concatenate([rows_a, rows_b]))
        old_to_new = np.empty(kmer_matrix.shape[0], dtype=np.int64)
        old_to_new[unique_rows] = np.arange(len(unique_rows))
        self.features = np.asarray(
            kmer_matrix[unique_rows].todense(), dtype=np.float32)
        self.rows_a = old_to_new[rows_a]
        self.rows_b = old_to_new[rows_b]

        # --- Original code (before subsetting optimization) ---
        # Densified the ENTIRE sparse matrix upfront regardless of which rows this
        # fold actually uses. Caused OOM on Polaris (4 folds × 14.2 GB = 56.9 GB)
        # and severe cache thrashing (14.2 GB >> L3 cache → ~15x per-batch slowdown).
        # self.features = np.asarray(kmer_matrix.todense(), dtype=np.float32)
        # self.rows_a = keys_a.map(key_to_row).astype(int).values
        # self.rows_b = keys_b.map(key_to_row).astype(int).values
        # --- End original code ---

        # Pre-extract labels as numpy array (avoids pandas iloc overhead per item).
        # Original: self.pairs.iloc[idx]['label'] in __getitem__ — pandas iloc has
        # ~50-100us overhead per call, adding ~10-18s per epoch at 177K train pairs.
        self.labels = pairs['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        # torch.tensor() copies data; torch.from_numpy() shares memory (zero-copy).
        # Zero-copy is safe with num_workers=0 (single process, no shared-memory
        # races). If num_workers>0, workers fork the process and shared numpy memory
        # can cause data corruption — switch back to torch.tensor() in that case.
        emb_a = torch.from_numpy(self.features[self.rows_a[idx]])
        emb_b = torch.from_numpy(self.features[self.rows_b[idx]])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return (emb_a, emb_b), label


class MLPClassifier(nn.Module):
    """
    MLP classifier for segment pairs.
    Expects (emb_a, emb_b) from dataset; computes interaction in forward.
    """
    def __init__(
        self,
        input_dim: int = 2560,
        hidden_dims: list[int] = [512, 256, 64],
        dropout: float = 0.3,
        slot_transform: str = "none",
        slot_transform_dims: Optional[list[int]] = None,
        adapter_dims: Optional[list[int]] = None,
        use_concat: bool = True,
        use_diff: bool = False,
        use_prod: bool = False,
        use_unit_diff: bool = False,
        embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.slot_transform = slot_transform
        self.use_concat = use_concat
        self.use_diff = use_diff
        self.use_prod = use_prod
        self.use_unit_diff = use_unit_diff

        self.slot_transform_shared = None
        self.slot_transform_a = None
        self.slot_transform_b = None
        self.adapter_a = None
        self.adapter_b = None
        self.norm_a = None
        self.norm_b = None

        def _build_mlp(dims: list[int]) -> nn.Sequential:
            # breakpoint()
            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                in_dim = dims[i]
                out_dim = dims[i + 1]
                layers.append(nn.Linear(in_dim, out_dim))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        def _build_adapter(in_out_dim: int, adapter_dims_list: list[int]) -> nn.Sequential:
            # breakpoint()
            dims = [in_out_dim] + adapter_dims_list + [in_out_dim]
            return _build_mlp(dims)

        # breakpoint()
        if self.slot_transform != "none" and embed_dim is None:
            raise ValueError("embed_dim is required when slot_transform != 'none'")

        if self.slot_transform == "shared":
            # Shared slot transform where both segments share the same slot transform
            if not slot_transform_dims:
                raise ValueError("slot_transform_dims must be set for slot_transform='shared'")
            self.slot_transform_shared = _build_mlp(slot_transform_dims)

        elif self.slot_transform == "slot_specific":
            # Slot-specific slot transform where each segment has its own slot transform
            if not slot_transform_dims:
                raise ValueError("slot_transform_dims must be set for slot_transform='slot_specific'")
            self.slot_transform_a = _build_mlp(slot_transform_dims)
            self.slot_transform_b = _build_mlp(slot_transform_dims)

        elif self.slot_transform == "shared_adapter":
            # Shared slot transform with adapters where both segments share the same slot transform and adapters
            if not slot_transform_dims:
                raise ValueError("slot_transform_dims must be set for slot_transform='shared_adapter'")
            if not adapter_dims:
                raise ValueError("adapter_dims must be set for slot_transform='shared_adapter'")
            self.slot_transform_shared = _build_mlp(slot_transform_dims)
            out_dim = slot_transform_dims[-1]
            self.adapter_a = _build_adapter(out_dim, adapter_dims)
            self.adapter_b = _build_adapter(out_dim, adapter_dims)

        elif self.slot_transform == "slot_norm":
            # Slot-specific norm where each segment has its own norm
            if slot_transform_dims:
                self.slot_transform_shared = _build_mlp(slot_transform_dims)
                out_dim = slot_transform_dims[-1]
            else:
                out_dim = embed_dim
            self.norm_a = nn.LayerNorm(out_dim)
            self.norm_b = nn.LayerNorm(out_dim)

        elif self.slot_transform != "none":
            raise ValueError(f"Unknown slot_transform: {self.slot_transform!r}")

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
    
    def _apply_slot_transform(self, a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply slot transform blocks to embeddings a and b.

        Args:
            a: Embedding tensor for segment A
            b: Embedding tensor for segment B

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Slot-transform embeddings for segments A and B
        """
        # breakpoint()
        if self.slot_transform == "none":
            return a, b

        if self.slot_transform == "shared":
            # Both embeddings are passed through the same slot transform block
            return self.slot_transform_shared(a), self.slot_transform_shared(b)

        if self.slot_transform == "slot_specific":
            # Each embedding is passed through its own slot transform block
            return self.slot_transform_a(a), self.slot_transform_b(b)

        if self.slot_transform == "shared_adapter":
            # Both embeddings are passed through the same slot transform block and then through the same adapter block
            z_a = self.slot_transform_shared(a)
            z_b = self.slot_transform_shared(b)
            return z_a + self.adapter_a(z_a), z_b + self.adapter_b(z_b)

        if self.slot_transform == "slot_norm":
            # Both embeddings are passed through the same slot transform block and then through the same norm block
            if self.slot_transform_shared is not None:
                a = self.slot_transform_shared(a)
                b = self.slot_transform_shared(b)
            return self.norm_a(a), self.norm_b(b)
        raise ValueError(f"Unknown slot_transform: {self.slot_transform!r}")

    def _compute_interaction(self, a: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute interaction features between embeddings a and b.
        Args:
            a: Embedding tensor for segment A  (batch, D)
            b: Embedding tensor for segment B  (batch, D)

        Returns:
            torch.Tensor: Concatenated interaction features (batch, K*D)
        """
        features = []
        if self.use_concat:
            features.extend([a, b])
        if self.use_diff:
            features.append(torch.abs(a - b))
        if self.use_unit_diff:
            diff = a - b
            norm = torch.linalg.norm(diff, dim=1, keepdim=True).clamp(min=1e-8)
            features.append(diff / norm)
        if self.use_prod:
            features.append(a * b)
        if not features:
            raise ValueError("At least one of concat/diff/prod/unit_diff must be enabled for interaction.")
        return torch.cat(features, dim=1)

    def forward(self, x: torch.Tensor, x_b: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_b is None:
            raise ValueError("MLPClassifier expects (emb_a, emb_b); x_b is required")
        a, b = self._apply_slot_transform(x, x_b)
        feats = self._compute_interaction(a, b)
        return self.mlp(feats)


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
    lr_scheduler=None,
    eval_train_metrics=False,
    train_eval_loader=None,
    use_amp=False
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
        eval_train_metrics: If True, run a full forward pass over training data each epoch
            to compute train F1/AUC/precision/recall/Brier. If False, only train_loss is tracked.
        train_eval_loader: DataLoader for train eval pass (can use larger batch size).
            If None, falls back to train_loader.
        use_amp: If True, use automatic mixed precision (float16 forward pass on GPU).
    
    Returns:
        best_model_path: Path to best model checkpoint
        th: Optimal threshold found on validation set
    """
    # Karpathy-style initialization check
    # breakpoint()
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
        'val_f1': [],  # F1 for positive class (same isolate) - binary
        'val_f1_macro': [],  # F1 macro (average of both classes)
        'val_precision': [],  # Precision for positive class (measures FP)
        'val_recall': [],  # Recall for positive class (measures FN)
        'val_auc': [],
        'val_brier': [],
        'learning_rate': [],  # Track learning rate over epochs
        'epoch_time_sec': [],  # Wall-clock time per epoch (seconds)
        # Level 1 profiling: per-epoch timing breakdown (seconds).
        # Identifies whether data loading, GPU compute, or evaluation is the bottleneck.
        'data_time_sec': [],      # Time in DataLoader (loading + collating batches)
        'compute_time_sec': [],   # Time in forward + backward + optimizer step
        'eval_time_sec': [],      # Time in validation (+ optional train eval) pass
    }
    # Train metrics keys only present when eval_train_metrics is enabled.
    # Plotting and CSV code check for key presence, so omitting them is safe.
    if eval_train_metrics:
        history.update({
            'train_f1': [],
            'train_f1_macro': [],
            'train_precision': [],
            'train_recall': [],
            'train_auc': [],
            'train_brier': [],
        })
    
    patience_counter = 0
    best_model_path = output_dir / 'best_model.pt'
    best_val_probs = None
    best_val_labels = None
    
    # AMP: GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Initialize best metric tracking based on metric type
    if early_stopping_metric == 'loss':
        best_metric_value = float('inf')
        is_higher_better = False
    elif early_stopping_metric in ['f1', 'auc']:
        best_metric_value = -1.0
        is_higher_better = True
    else:
        raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}. Choose from 'loss', 'f1', 'auc'")

    # --- Level 2 diagnostic: micro-benchmark first 10 batches ---
    # Confirmed that pin_memory=True causes ~300x data-loading slowdown with 4 concurrent
    # folds on Polaris (cudaHostAlloc serializes across processes). pin_memory=False in the
    # master bundle fixes this. This diagnostic can be removed once the fix is validated
    # in a full production run (Phase 3). To re-enable, uncomment the block below.
    #
    # print("\nLevel 2 diagnostic: benchmarking first 10 batches...")
    # _diag_loader_pinned = DataLoader(
    #     train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True,
    #     num_workers=0, pin_memory=True)
    # _diag_loader_unpinned = DataLoader(
    #     train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True,
    #     num_workers=0, pin_memory=False)
    # for _label, _loader in [("pin_memory=True", _diag_loader_pinned),
    #                          ("pin_memory=False", _diag_loader_unpinned)]:
    #     _t0 = time.time()
    #     for _i, (_bx, _by) in enumerate(_loader):
    #         if _i >= 9:
    #             break
    #     _elapsed = time.time() - _t0
    #     print(f"  {_label}: 10 batches in {_elapsed:.3f}s ({_elapsed/10*1000:.1f} ms/batch)")
    # del _diag_loader_pinned, _diag_loader_unpinned
    # print()
    # --- End Level 2 diagnostic ---

    for epoch in range(epochs):
        epoch_start = time.time()
        # Training phase — profiling separates data loading from GPU compute
        model.train()
        train_loss = 0
        epoch_data_time = 0.0     # Cumulative DataLoader time this epoch
        epoch_compute_time = 0.0  # Cumulative forward+backward+optimizer time
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}',
            dynamic_ncols=True, leave=False, miniters=50)
        data_start = time.time()
        for batch_x, batch_y in progress_bar:
            epoch_data_time += time.time() - data_start
            compute_start = time.time()
            is_pair = isinstance(batch_x, (tuple, list)) and len(batch_x) == 2
            if is_pair:
                batch_a, batch_b = batch_x
                batch_a, batch_b = batch_a.to(device), batch_b.to(device)
            else:
                batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                if is_pair:
                    preds = model(batch_a, batch_b).squeeze(-1)
                else:
                    preds = model(batch_x).squeeze(-1)
                loss = criterion(preds, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * batch_y.size(0)
            epoch_compute_time += time.time() - compute_start
            data_start = time.time()
        train_loss /= len(train_loader.dataset)
        
        # Evaluation phase — timing covers both optional train-eval and val-eval
        eval_start = time.time()

        # Compute training metrics AFTER training (with model in eval mode for consistency)
        # This ensures all training predictions come from the same model state.
        # Skipped when eval_train_metrics=False to save ~50% of epoch time.
        if eval_train_metrics:
            _train_eval_loader = train_eval_loader or train_loader
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for batch_x, batch_y in _train_eval_loader:
                    if isinstance(batch_x, (tuple, list)) and len(batch_x) == 2:
                        batch_a, batch_b = batch_x
                        batch_a, batch_b = batch_a.to(device), batch_b.to(device)
                        batch_y = batch_y.to(device)
                        preds = model(batch_a, batch_b).squeeze(-1)
                    else:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        preds = model(batch_x).squeeze(-1)
                    all_logits.append(preds)
                    all_labels.append(batch_y)
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            train_probs = torch.sigmoid(all_logits).cpu().numpy()
            train_preds = (train_probs > 0.5).astype(np.float32)
            train_labels = all_labels.cpu().numpy()
            # Compute metrics for positive class (label=1, same isolate)
            # Explicit parameters: average='binary', pos_label=1
            train_f1 = f1_score(train_labels, train_preds, average='binary', pos_label=1)
            train_f1_macro = f1_score(train_labels, train_preds, average='macro')
            train_precision = precision_score(train_labels, train_preds, average='binary', pos_label=1, zero_division=0)
            train_recall = recall_score(train_labels, train_preds, average='binary', pos_label=1, zero_division=0)
            try:
                train_auc = roc_auc_score(train_labels, train_probs)
            except ValueError:
                # See val_auc comment: degenerate predictions break roc_auc_score.
                train_auc = 0.5
            train_brier = float(np.mean((train_probs - np.array(train_labels)) ** 2))
        else:
            train_f1 = train_f1_macro = train_precision = train_recall = train_auc = train_brier = None

        model.eval()
        val_loss = 0
        all_logits, all_labels = [], []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for batch_x, batch_y in val_loader:
                if isinstance(batch_x, (tuple, list)) and len(batch_x) == 2:
                    batch_a, batch_b = batch_x
                    batch_a, batch_b = batch_a.to(device), batch_b.to(device)
                    batch_y = batch_y.to(device)
                    preds = model(batch_a, batch_b).squeeze(-1)
                else:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    preds = model(batch_x).squeeze(-1)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_y.size(0)
                all_logits.append(preds)
                all_labels.append(batch_y)
        val_loss /= len(val_loader.dataset)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        val_probs = torch.sigmoid(all_logits).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(np.float32)
        val_labels = all_labels.cpu().numpy()
        # Compute metrics for positive class (label=1, same isolate)
        # Explicit parameters: average='binary', pos_label=1
        val_f1 = f1_score(val_labels, val_preds, average='binary', pos_label=1)
        val_f1_macro = f1_score(val_labels, val_preds, average='macro')
        val_precision = precision_score(val_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            # roc_auc_score measures how well the model ranks positives above
            # negatives across all thresholds. When predictions are near-constant
            # (degenerate model), the FPR values contain ties that break
            # monotonicity, causing sklearn to raise ValueError. Fall back to
            # AUC=0.5 (equivalent to random ranking) so training can continue.
            val_auc = 0.5
        val_brier = float(np.mean((np.array(val_probs) - np.array(val_labels)) ** 2))

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
        
        epoch_eval_time = time.time() - eval_start
        epoch_time = time.time() - epoch_start

        # Track history for plotting
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['data_time_sec'].append(round(epoch_data_time, 2))
        history['compute_time_sec'].append(round(epoch_compute_time, 2))
        history['eval_time_sec'].append(round(epoch_eval_time, 2))
        if eval_train_metrics:
            history['train_f1'].append(train_f1)
            history['train_f1_macro'].append(train_f1_macro)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_auc'].append(train_auc)
            history['train_brier'].append(train_brier)
        history['val_f1'].append(val_f1)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_auc'].append(val_auc)
        history['val_brier'].append(val_brier)
        history['learning_rate'].append(current_lr)
        history['epoch_time_sec'].append(round(epoch_time, 2))

        # Print simplified metrics + timing breakdown (all metrics still saved to history/CSV)
        timing_str = f'time: {epoch_time:.1f}s (data={epoch_data_time:.1f} compute={epoch_compute_time:.1f} eval={epoch_eval_time:.1f})'
        if eval_train_metrics:
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, '
                  f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f} '
                  f'[{early_stopping_metric.upper()}: {current_metric_value:.4f}, LR: {current_lr:.6f}] {timing_str}')
        else:
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f} '
                  f'[{early_stopping_metric.upper()}: {current_metric_value:.4f}, LR: {current_lr:.6f}] {timing_str}')

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
            print(f'  New best {early_stopping_metric}: {best_metric_value:.4f}')
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
            'train_brier': history.get('train_brier', [None] * len(history['train_loss'])),
            'val_f1': history.get('val_f1', [None] * len(history['train_loss'])),
            'val_f1_macro': history.get('val_f1_macro', [None] * len(history['train_loss'])),
            'val_precision': history.get('val_precision', [None] * len(history['train_loss'])),
            'val_recall': history.get('val_recall', [None] * len(history['train_loss'])),
            'val_auc': history.get('val_auc', [None] * len(history['train_loss'])),
            'val_brier': history.get('val_brier', [None] * len(history['train_loss'])),
            'learning_rate': history.get('learning_rate', [None] * len(history['train_loss'])),
            'epoch_time_sec': history.get('epoch_time_sec', [None] * len(history['train_loss'])),
            'data_time_sec': history.get('data_time_sec', [None] * len(history['train_loss'])),
            'compute_time_sec': history.get('compute_time_sec', [None] * len(history['train_loss'])),
            'eval_time_sec': history.get('eval_time_sec', [None] * len(history['train_loss'])),
        })
        history_csv = output_dir / 'training_history.csv'
        history_df.to_csv(history_csv, index=False)
        print(f"Training history saved to: {history_csv}")
    
    # Plot learning curves
    if len(history['train_loss']) > 0:
        plot_learning_curves(history, output_dir, bundle_name=config_bundle)
        
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
            print("Model learned! (F1 > baseline)")
        else:
            print("WARNING: Model did not beat baseline - may indicate learning issues")
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


def evaluate_on_split(
    model,
    data_loader,
    criterion,
    device,
    pairs_df,
    threshold=0.5,
    split_name: str = "test",
    use_amp=False,
    ) -> pd.DataFrame:
    """
    Evaluate the model on a split and compute metrics.

    Args:
        threshold: Classification threshold (default: 0.5)
        use_amp: If True, use automatic mixed precision for inference.
    """
    model.eval()
    loss_sum = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for batch_x, batch_y in data_loader:
            if isinstance(batch_x, (tuple, list)) and len(batch_x) == 2:
                batch_a, batch_b = batch_x
                batch_a, batch_b = batch_a.to(device), batch_b.to(device)
                batch_y = batch_y.to(device)
                batch_logits = model(batch_a, batch_b).squeeze(-1)
            else:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_logits = model(batch_x).squeeze(-1)
            loss = criterion(batch_logits, batch_y)
            loss_sum += loss.item() * batch_y.size(0)
            all_logits.append(batch_logits.view(-1))
            all_labels.append(batch_y)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    mean_loss = loss_sum / len(data_loader.dataset)
    logits = all_logits.float().cpu().numpy().astype(np.float32, copy=False)
    probs = torch.sigmoid(all_logits).cpu().numpy()
    true_labels = all_labels.cpu().numpy()

    metrics, res_df = compute_pair_metrics(
        true_labels, probs, threshold, pairs_df, logits=logits,
    )

    split_title = split_name.strip() if split_name is not None else "split"
    print(f'{split_title} Loss: {mean_loss:.4f}, {split_title} F1 (binary): {metrics["f1"]:.4f}, {split_title} F1 (macro): {metrics["f1_macro"]:.4f}')
    print(f'{split_title} Precision: {metrics["precision"]:.4f}, {split_title} Recall: {metrics["recall"]:.4f}, {split_title} AUC: {metrics["auc"]:.4f}')
    print(f'Using threshold: {threshold:.4f}')
    print(f'Note: Precision measures False Positives (FP), Recall measures False Negatives (FN)')
    print(f'      F1 (binary) focuses on positive class, F1 (macro) averages both classes')
    return res_df


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
    '--fold_id',
    type=int, default=None,
    help='CV fold ID. If provided, appends fold_{fold_id}/ to --dataset_dir.'
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
parser.add_argument(
    '--override',
    type=str, nargs='+', default=None,
    help='Hydra-style dotlist overrides applied on top of the bundle (e.g., '
         'dataset.hn_subtype=H3N2). Useful for filter sweeps without creating new bundles.'
)
parser.add_argument(
    '--skip_post_hoc',
    action='store_true',
    help='Skip the post-hoc analysis (analyze_stage4_train.py) that normally runs '
         'after training completes. Useful for fast debugging iterations.'
)
args = parser.parse_args()

# Load config
config_path = str(project_root / 'conf')  # Pass the config path explicitly
config_bundle = args.config_bundle
if config_bundle is None:
    raise ValueError("Must provide --config_bundle")
config = get_virus_config_hydra(config_bundle, config_path=config_path)
if args.override:
    from omegaconf import OmegaConf
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    print(f"Applied CLI overrides: {args.override}")
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
DROPOUT = config.training.dropout
PATIENCE = config.training.patience
EARLY_STOPPING_METRIC = config.training.early_stopping_metric
THRESHOLD_METRIC = config.training.threshold_metric
SLOT_TRANSFORM = getattr(config.training, 'slot_transform', 'none')
SLOT_TRANSFORM_DIMS = getattr(config.training, 'slot_transform_dims', None)
ADAPTER_DIMS = getattr(config.training, 'adapter_dims', None)
# Interaction spec: "concat", "diff", "unit_diff", "prod", or combinations like "concat+unit_diff"
INTERACTION_SPEC = getattr(config.training, 'interaction', 'concat')
FEATURE_SOURCE = getattr(config.training, 'feature_source', 'esm2')
FEATURE_SCALING = getattr(config.training, 'feature_scaling', 'none')
if FEATURE_SCALING not in {'none', 'standard'}:
    raise ValueError(
        f"training.feature_scaling must be 'none' or 'standard'; got {FEATURE_SCALING!r}"
    )
EVAL_TRAIN_METRICS = getattr(config.training, 'eval_train_metrics', False)
INFER_BATCH_SIZE = getattr(config.training, 'infer_batch_size', None) or BATCH_SIZE
# Hard-coded to 0. Do NOT change without addressing both issues below:
#   1. Performance: num_workers>0 was benchmarked at 87% slower for this workload
#      (see speed_up.md). Our __getitem__ is in-memory array indexing — the IPC
#      overhead of forking workers (pickling, queue management) far exceeds any
#      benefit since there is no disk I/O to overlap with GPU computation.
#   2. Correctness: KmerPairDataset uses torch.from_numpy() for zero-copy tensor
#      creation. With num_workers>0, forked workers share numpy memory pages, which
#      can cause data corruption or segfaults. If num_workers>0 is ever needed,
#      switch torch.from_numpy() to torch.tensor() in KmerPairDataset.__getitem__.
# Previously read from config: NUM_WORKERS = getattr(config.training, 'num_workers', 0)
NUM_WORKERS = 0
PIN_MEMORY = getattr(config.training, 'pin_memory', False)
USE_AMP = getattr(config.training, 'use_amp', False)
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
        "--dataset_dir is required. "
        "Datasets are in runs/ subdirectories: "
        "data/datasets/{virus}/{data_version}/runs/dataset_{config_bundle}_{timestamp}/"
    )
dataset_dir = Path(args.dataset_dir)
if args.fold_id is not None:
    dataset_dir = dataset_dir / f"fold_{args.fold_id}"
    print(f"CV mode: using fold directory: {dataset_dir}")

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
    print(f"WARNING: No run_output_subdir provided, using fallback: {fallback_run_id}")
output_dir.mkdir(parents=True, exist_ok=True)

# Save resolved config snapshot for reproducibility
save_config(config, str(output_dir / 'resolved_config.yaml'))
print(f'Saved resolved config snapshot to: {output_dir / "resolved_config.yaml"}')

print(f'\nConfig bundle:    {config_bundle}')
print(f'Run suffix:       {RUN_SUFFIX if RUN_SUFFIX else "(none)"}')
print(f'Dataset dir:      {dataset_dir}')
print(f'Embeddings file:  {embeddings_file}')
print(f'Output dir:       {output_dir}')  # This line is parsed by stage4_train.sh
print(f'Run ID:           {args.run_output_subdir if args.run_output_subdir else "auto-generated"}')
print(f'model:           {MODEL_CKPT}')
print(f'batch_size:      {BATCH_SIZE}')
print(f'infer_batch_size: {INFER_BATCH_SIZE}')
print(f'eval_train_metrics: {EVAL_TRAIN_METRICS}')
print(f'num_workers:     {NUM_WORKERS} (hard-coded)')
print(f'pin_memory:      {PIN_MEMORY}')
print(f'use_amp:         {USE_AMP}')

total_timer.begin_phase('load_data')
print('\nLoad pair datasets.')
# Use engine='python' to avoid a pandas C parser segfault triggered by certain
# protein sequence byte patterns in large CSVs (observed on Polaris, fold 11 of
# PB1/HA and fold 6 of PB2/PA). The Python engine is slower but only adds ~seconds.
CSV_ENGINE = 'python'
train_pairs = pd.read_csv(dataset_dir / 'train_pairs.csv', engine=CSV_ENGINE)
val_pairs   = pd.read_csv(dataset_dir / 'val_pairs.csv', engine=CSV_ENGINE)
test_pairs  = pd.read_csv(dataset_dir / 'test_pairs.csv', engine=CSV_ENGINE)

# CUDA device
CUDA_NAME = args.cuda_name
device = determine_device(CUDA_NAME)

print(f'\nFeature source: {FEATURE_SOURCE}')

if FEATURE_SOURCE == 'kmer':
    # --- K-mer feature path ---
    from omegaconf import OmegaConf
    if hasattr(config, 'kmer') and config.get('kmer') is not None:
        KMER_K = int(config.kmer.get('k', 6))
    else:
        KMER_K = 6
    EMBED_DIM = 4 ** KMER_K

    # K-mer features live alongside ESM-2 embeddings
    kmer_dir = build_embeddings_paths(
        project_root=project_root, virus_name=VIRUS_NAME,
        data_version=DATA_VERSION, run_suffix="", config=config,
    )['output_dir']
    print(f'K-mer dir: {kmer_dir}')
    print(f'K-mer k={KMER_K}, embed_dim={EMBED_DIM}')

    kmer_key_to_row = load_kmer_index(kmer_dir, KMER_K)
    kmer_matrix = load_kmer_matrix(kmer_dir, KMER_K)
    print(f'Loaded k-mer matrix: {kmer_matrix.shape}')

    # Validate that pair CSVs have ctg_a/ctg_b columns
    for name, pdf in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        for col in ['ctg_a', 'ctg_b']:
            if col not in pdf.columns:
                raise ValueError(
                    f"Pair CSV ({name}) missing '{col}' column. "
                    "Re-run Stage 3 (dataset_segment_pairs.py) to add ctg columns."
                )

    train_dataset = KmerPairDataset(train_pairs, kmer_matrix, kmer_key_to_row)
    val_dataset = KmerPairDataset(val_pairs, kmer_matrix, kmer_key_to_row)
    test_dataset = KmerPairDataset(test_pairs, kmer_matrix, kmer_key_to_row)

else:
    # --- ESM-2 embedding path (default) ---
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
            f"Parquet index not found: {index_file}. "
            "Master cache format requires parquet index for validation."
        )

    index_df = pd.read_parquet(index_file)
    available_ids = set(index_df['brc_fea_id'].unique())
    for df_name, df in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        required_ids = set(df['brc_a']).union(set(df['brc_b']))
        missing = required_ids - available_ids
        if missing:
            raise ValueError(
                f"Missing embeddings for {len(missing)} IDs in {df_name} set: {list(missing)[:5]}..."
            )
    print(f"All required embeddings available ({len(available_ids)} total embeddings)")

    # Create datasets (always returns (emb_a, emb_b), label)
    train_dataset = ESMPairDataset(train_pairs, embeddings_file)
    val_dataset = ESMPairDataset(val_pairs, embeddings_file)
    test_dataset = ESMPairDataset(test_pairs, embeddings_file)

    # Get embedding dimension from model checkpoint
    EMBED_DIM = get_esm2_embedding_dim(MODEL_CKPT)

# ── Optional StandardScaler ────────────────────────────────────────────
# Fit on TRAIN slot vectors only (no val/test leakage). The same fitted
# scaler transforms train/val/test. K-mer datasets store features in a
# per-fold dense matrix at `self.features`, so we mutate them in place.
# ESM-2 datasets share a global embedding cache; per-dataset scaling there
# requires a `__getitem__`-level hook that hasn't been wired yet.
feature_scaler = None
if FEATURE_SCALING == 'standard':
    if FEATURE_SOURCE == 'kmer':
        from sklearn.preprocessing import StandardScaler
        import joblib
        print('\nFitting StandardScaler on training k-mer features (per-feature mean/std)...')
        feature_scaler = StandardScaler()
        feature_scaler.fit(train_dataset.features)
        train_dataset.features = feature_scaler.transform(train_dataset.features).astype(np.float32)
        val_dataset.features   = feature_scaler.transform(val_dataset.features).astype(np.float32)
        test_dataset.features  = feature_scaler.transform(test_dataset.features).astype(np.float32)
        scaler_path = output_dir / 'feature_scaler.joblib'
        joblib.dump(feature_scaler, scaler_path)
        print(f'  fitted on {len(train_dataset.features):,} train rows '
              f'({train_dataset.features.shape[1]:,} features)')
        print(f'  saved scaler to: {scaler_path}')
    else:
        raise NotImplementedError(
            "feature_scaling='standard' is currently implemented for feature_source='kmer' "
            "only. ESM-2 support requires a per-dataset scaler hook in ESMPairDataset "
            "(the embedding cache is shared across train/val/test, so in-place mutation "
            "would leak across runs). TODO before running ESM ablations."
        )

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
train_eval_loader = DataLoader(
    train_dataset, batch_size=INFER_BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(
    val_dataset, batch_size=INFER_BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(
    test_dataset, batch_size=INFER_BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# Resolve interaction flags from training.interaction
USE_CONCAT_RAW, USE_DIFF_RAW, USE_PROD_RAW, USE_UNIT_DIFF_RAW = parse_interaction_flags(INTERACTION_SPEC)

# Compute input dimension based on feature flags and slot transform
if SLOT_TRANSFORM in {"shared", "slot_specific", "shared_adapter"}:
    if not SLOT_TRANSFORM_DIMS:
        raise ValueError(f"slot_transform_dims must be set for slot_transform='{SLOT_TRANSFORM}'")
    out_dim = SLOT_TRANSFORM_DIMS[-1]
elif SLOT_TRANSFORM == "slot_norm" and SLOT_TRANSFORM_DIMS:
    out_dim = SLOT_TRANSFORM_DIMS[-1]
else:
    out_dim = EMBED_DIM

mlp_input_dim = 0
feature_desc_parts = []
if USE_CONCAT_RAW:
    mlp_input_dim += 2 * out_dim
    feature_desc_parts.append("2")
if USE_DIFF_RAW:
    mlp_input_dim += out_dim
    feature_desc_parts.append("1")
if USE_UNIT_DIFF_RAW:
    mlp_input_dim += out_dim
    feature_desc_parts.append("1")
if USE_PROD_RAW:
    mlp_input_dim += out_dim
    feature_desc_parts.append("1")
if mlp_input_dim == 0:
    raise ValueError("At least one interaction term must be enabled (training.interaction).")
feature_desc = "+".join(feature_desc_parts) if feature_desc_parts else "0"
print(f"MLP Input Dimension: {mlp_input_dim} ({feature_desc} * {out_dim})")

total_timer.end_phase('load_data')
total_timer.begin_phase('train')

# Initialize model
model = MLPClassifier(
    input_dim=mlp_input_dim,
    hidden_dims=HIDDEN_DIMS,
    dropout=DROPOUT,
    slot_transform=SLOT_TRANSFORM,
    slot_transform_dims=SLOT_TRANSFORM_DIMS,
    adapter_dims=ADAPTER_DIMS,
    use_concat=USE_CONCAT_RAW,
    use_diff=USE_DIFF_RAW,
    use_prod=USE_PROD_RAW,
    use_unit_diff=USE_UNIT_DIFF_RAW,
    embed_dim=EMBED_DIM,
)

# GPU memory diagnostic: log free/total memory BEFORE model.to(device).
# If a fold OOMs here, this tells us whether the GPU was already occupied
# (e.g., by TensorFlow pre-allocation or another process).
if 'cuda' in str(device):
    free_mem, total_mem = torch.cuda.mem_get_info(torch.device(device))
    print(f"GPU memory before model.to(device): "
          f"{free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total")
    if free_mem < 1e9:  # Less than 1 GB free
        print(f"WARNING: GPU has only {free_mem / 1e6:.0f} MB free — OOM likely")

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
    lr_scheduler=lr_scheduler,
    eval_train_metrics=EVAL_TRAIN_METRICS,
    train_eval_loader=train_eval_loader,
    use_amp=USE_AMP
)

total_timer.end_phase('train')
total_timer.begin_phase('inference')

# Evaluate
print('\nEvaluate model.')
model.load_state_dict(torch.load(best_model_path))
test_res_df = evaluate_on_split(
    model, test_loader, criterion, device, test_pairs,
    threshold=optimal_threshold,
    split_name="Test",
    use_amp=USE_AMP
)

# Save raw predictions
preds_file = output_dir / 'test_predicted.csv'
print(f'\nSave raw predictions to: {preds_file}')
test_res_df.to_csv(preds_file, index=False)

EVAL_SWAPPED_TEST = getattr(config.training, 'eval_swapped_test', False)
if EVAL_SWAPPED_TEST:
    print('\nSwap-test diagnostic: evaluate on swapped test inputs (B,A) using SAME checkpoint.')
    swapped_test_pairs = swap_pairs_df_columns(test_pairs)
    if FEATURE_SOURCE == 'kmer':
        swapped_test_dataset = KmerPairDataset(swapped_test_pairs, kmer_matrix, kmer_key_to_row)
    else:
        swapped_test_dataset = ESMPairDataset(swapped_test_pairs, embeddings_file)
    swapped_test_loader = DataLoader(swapped_test_dataset, batch_size=INFER_BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    swapped_test_res_df = evaluate_on_split(
        model, swapped_test_loader, criterion, device, swapped_test_pairs,
        threshold=optimal_threshold,
        split_name="Swapped Test",
        use_amp=USE_AMP
    )
    swapped_preds_file = output_dir / 'test_predicted_swapped.csv'
    print(f'\nSave swapped-test predictions to: {swapped_preds_file}')
    swapped_test_res_df.to_csv(swapped_preds_file, index=False)

total_timer.end_phase('inference')

# Save training provenance
training_info = {
    'config_bundle': config_bundle,
    'dataset_dir': str(dataset_dir),
    'embeddings_file': str(embeddings_file),
    'feature_source': FEATURE_SOURCE,
    'feature_scaling': FEATURE_SCALING,
    'interaction': INTERACTION_SPEC,
    'slot_transform': SLOT_TRANSFORM,
    'hidden_dims': list(HIDDEN_DIMS) if HIDDEN_DIMS else None,
    'dropout': DROPOUT,
    'batch_size': BATCH_SIZE,
    'infer_batch_size': INFER_BATCH_SIZE,
    'eval_train_metrics': EVAL_TRAIN_METRICS,
    'num_workers': NUM_WORKERS,
    'pin_memory': PIN_MEMORY,
    'use_amp': USE_AMP,
    'learning_rate': LEARNING_RATE,
    'patience': PATIENCE,
    'epochs': EPOCHS,
    'early_stopping_metric': EARLY_STOPPING_METRIC,
    'threshold_metric': THRESHOLD_METRIC,
    'optimal_threshold': optimal_threshold,
    'seed': RANDOM_SEED,
    'cuda_device': CUDA_NAME,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}
training_info_file = output_dir / 'training_info.json'
with open(training_info_file, 'w') as f:
    json.dump(training_info, f, indent=2)
print(f'\nSaved training provenance to: {training_info_file}')

print(f'\nFinished {Path(__file__).name}!')
total_timer.stop_timer()
total_timer.display_timer()
total_timer.save_timer(output_dir)

# ── Post-hoc analysis (guardrailed: failure never breaks training) ──────────
# Runs analyze_stage4_train.py as a subprocess so any failure (import error,
# plotting crash, stratification bug) is isolated from the training exit code.
# Training artifacts (best_model.pt, test_predicted.csv, optimal_threshold.txt)
# are already on disk at this point; the model is validly trained regardless of
# post-hoc outcome. To refresh post_hoc later, use scripts/run_allpairs_post_hoc.sh.
if not args.skip_post_hoc:
    import subprocess
    post_hoc_log = output_dir / 'post_hoc_run.log'
    print(f'\nRunning post-hoc analysis -> {output_dir}/post_hoc/')
    post_hoc_cmd = [
        sys.executable,
        str(project_root / 'src' / 'analysis' / 'analyze_stage4_train.py'),
        '--config_bundle', config_bundle,
        '--model_dir', str(output_dir),
    ]
    try:
        with open(post_hoc_log, 'w') as logf:
            rc = subprocess.call(post_hoc_cmd, stdout=logf, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f'WARNING: post-hoc analysis exited with code {rc} for {output_dir}.')
            print(f'         Training artifacts are unaffected. See log: {post_hoc_log}')
            print(f'         To regenerate later: bash scripts/run_allpairs_post_hoc.sh <TAG>')
        else:
            print(f'Done. Post-hoc analysis complete. Log: {post_hoc_log}')
    except Exception as e:
        import traceback
        print(f'WARNING: post-hoc analysis subprocess failed to launch for {output_dir}: {e}')
        print(f'         Training artifacts are unaffected.')
        traceback.print_exc()
