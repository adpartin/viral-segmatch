"""Pair feature loader for non-MLP baselines (logistic regression, etc.).

Returns dense numpy arrays (`X`, `y`) per split plus an optional fitted
``StandardScaler``. The MLP path keeps its torch ``Dataset``-based pipeline;
this module exists so sklearn-style baselines can avoid that machinery and
operate on plain matrices.

Scope (intentional, per plan):
- k-mer features only (ESM-2 raises ``NotImplementedError``).
- Concat-only interaction (no ``unit_diff``/``diff``/``prod``).
- ``slot_transform='none'`` only — a numpy LayerNorm placeholder lives here
  for future use but is not currently called.
"""
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.kmer_utils import (
    get_kmer_pair_features,
    load_kmer_index,
    load_kmer_matrix,
)


def _apply_slot_norm(emb: np.ndarray) -> np.ndarray:
    """Numpy LayerNorm placeholder. Not currently called.

    NOTE: mirrors ``torch.nn.LayerNorm(D)(emb)`` along the feature axis —
    zero-mean, unit-variance per row. Reserved for future ESM-on-LR
    baselines where slot-level normalization matters (k-mer features
    don't benefit from it). Keep here so the future hook lives next to
    the rest of the baseline feature pipeline.
    """
    eps = 1e-5
    mu = emb.mean(axis=-1, keepdims=True)
    sd = emb.std(axis=-1, keepdims=True)
    return (emb - mu) / (sd + eps)


def load_pair_features_for_baselines(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    *,
    feature_source: str,
    feature_scaling: str,
    kmer_dir: Path,
    kmer_k: int,
    output_dir: Path,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        Optional[StandardScaler],
    ]:
    """Materialize concat'd pair features per split for sklearn baselines.

    Args:
        train_pairs / val_pairs / test_pairs: pair CSVs as DataFrames.
        feature_source: ``'kmer'``. ``'esm2'`` raises ``NotImplementedError``.
        feature_scaling: ``'none'`` or ``'standard'``. When ``'standard'``,
            fit ``StandardScaler`` on the train split only, transform all
            splits, and persist the scaler to
            ``output_dir / 'feature_scaler.joblib'``.
        kmer_dir: directory containing ``kmer_features_k{k}.npz`` and the
            matching ``...index.parquet``.
        kmer_k: k-mer size (e.g. 6).
        output_dir: where the fitted scaler is saved (only when scaling is
            ``'standard'``).

    Returns:
        ``((X_train, y_train), (X_val, y_val), (X_test, y_test), scaler)``.
        ``scaler`` is ``None`` when ``feature_scaling == 'none'``.
    """
    if feature_source != 'kmer':
        raise NotImplementedError(
            f"Baseline feature loader supports feature_source='kmer' only; "
            f"got {feature_source!r}. ESM-2 baselines are not in scope yet."
        )
    if feature_scaling not in {'none', 'standard'}:
        raise ValueError(
            f"feature_scaling must be 'none' or 'standard'; got {feature_scaling!r}"
        )

    print(f'\nLoading k-mer features for baselines (k={kmer_k})...')
    key_to_row = load_kmer_index(Path(kmer_dir), kmer_k)
    kmer_matrix = load_kmer_matrix(Path(kmer_dir), kmer_k)
    print(f'  k-mer matrix: {kmer_matrix.shape}')

    X_train, y_train = get_kmer_pair_features(train_pairs, kmer_matrix, key_to_row, interaction='concat')
    X_val,   y_val   = get_kmer_pair_features(val_pairs,   kmer_matrix, key_to_row, interaction='concat')
    X_test,  y_test  = get_kmer_pair_features(test_pairs,  kmer_matrix, key_to_row, interaction='concat')
    print(f'  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}')

    scaler: Optional[StandardScaler] = None
    if feature_scaling == 'standard':
        print('Fitting StandardScaler on training features (per-feature mean/std)...')
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32, copy=False)
        X_val   = scaler.transform(X_val).astype(np.float32,   copy=False)
        X_test  = scaler.transform(X_test).astype(np.float32,  copy=False)
        scaler_path = Path(output_dir) / 'feature_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        print(f'  fitted on {len(X_train):,} train rows ({X_train.shape[1]:,} features)')
        print(f'  saved scaler to: {scaler_path}')

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
