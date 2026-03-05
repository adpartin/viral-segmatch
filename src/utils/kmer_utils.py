"""
K-mer feature loading and pair feature construction utilities.

Parallel to embedding_utils.py but for sparse k-mer features stored as
scipy .npz + parquet index.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from scipy import sparse


def load_kmer_index(kmer_dir: Path, k: int) -> Dict[str, int]:
    """Load (assembly_id::genbank_ctg_id) → row index mapping from parquet.

    Args:
        kmer_dir: Directory containing kmer_features_k{k}_index.parquet
        k: k-mer size

    Returns:
        dict mapping "assembly_id::genbank_ctg_id" → row index
    """
    index_file = kmer_dir / f'kmer_features_k{k}_index.parquet'
    if not index_file.exists():
        raise FileNotFoundError(f"K-mer index not found: {index_file}")

    idx_df = pd.read_parquet(index_file)
    # Build composite key: assembly_id::genbank_ctg_id
    keys = idx_df['assembly_id'].astype(str) + '::' + idx_df['genbank_ctg_id'].astype(str)
    return dict(zip(keys, idx_df['row']))


def load_kmer_matrix(kmer_dir: Path, k: int) -> sparse.csr_matrix:
    """Load sparse k-mer feature matrix from .npz file.

    Args:
        kmer_dir: Directory containing kmer_features_k{k}.npz
        k: k-mer size

    Returns:
        scipy CSR matrix of shape (N_segments, 4^k)
    """
    npz_file = kmer_dir / f'kmer_features_k{k}.npz'
    if not npz_file.exists():
        raise FileNotFoundError(f"K-mer features not found: {npz_file}")
    return sparse.load_npz(npz_file)


def get_kmer_pair_features(
    pairs_df: pd.DataFrame,
    kmer_matrix: sparse.csr_matrix,
    key_to_row: Dict[str, int],
    interaction: str = 'concat',
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Build pair feature matrix from k-mer features and pair CSV.

    Pairs are looked up using composite keys built from
    (assembly_id_a, ctg_a) and (assembly_id_b, ctg_b) columns.

    Args:
        pairs_df: DataFrame with assembly_id_a, ctg_a, assembly_id_b, ctg_b, label
        kmer_matrix: sparse CSR matrix from load_kmer_matrix
        key_to_row: mapping from load_kmer_index
        interaction: 'concat', 'diff', 'unit_diff', or combinations like 'concat+unit_diff'

    Returns:
        features: dense (N, D) float32 array
        labels: (N,) int array
    """
    # Build composite keys for both sides
    keys_a = pairs_df['assembly_id_a'].astype(str) + '::' + pairs_df['ctg_a'].astype(str)
    keys_b = pairs_df['assembly_id_b'].astype(str) + '::' + pairs_df['ctg_b'].astype(str)

    # Map to row indices
    rows_a = keys_a.map(key_to_row)
    rows_b = keys_b.map(key_to_row)
    valid = rows_a.notna() & rows_b.notna()

    n_invalid = (~valid).sum()
    if n_invalid > 0:
        print(f"Warning: {n_invalid} pairs have missing k-mer features (skipped)")
    if valid.sum() == 0:
        raise ValueError("No valid pairs found — check that k-mer features match pair CSV keys")

    idx_a = rows_a[valid].astype(int).values
    idx_b = rows_b[valid].astype(int).values
    labels = pairs_df.loc[valid, 'label'].values

    # Extract dense rows (k-mer matrices for k=6 are only 4096-dim, fine to densify)
    emb_a = np.asarray(kmer_matrix[idx_a].todense(), dtype=np.float32)
    emb_b = np.asarray(kmer_matrix[idx_b].todense(), dtype=np.float32)

    # Build interaction features (same logic as ESM-2 pair features)
    tokens = {t.strip().lower() for t in interaction.split('+')}
    allowed = {'concat', 'diff', 'unit_diff', 'prod'}
    unknown = tokens - allowed
    if unknown:
        raise ValueError(f"Unknown interaction tokens: {unknown}")

    features = []
    if 'concat' in tokens:
        features.append(emb_a)
        features.append(emb_b)
    if 'diff' in tokens:
        features.append(np.abs(emb_a - emb_b))
    if 'unit_diff' in tokens:
        diff = emb_a - emb_b
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features.append(diff / norms)
    if 'prod' in tokens:
        features.append(emb_a * emb_b)
    if not features:
        raise ValueError("At least one interaction term must be enabled")

    return np.concatenate(features, axis=1), labels
