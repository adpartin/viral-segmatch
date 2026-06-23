"""
K-mer feature loading and pair feature construction utilities.

Parallel to embedding_utils.py but for sparse k-mer features stored as
scipy .npz + parquet index. Supports both alphabets:
    nt: occurrence keyed by (assembly_id, genbank_ctg_id); pair-table
        lookup uses (assembly_id_a, ctg_a) / (assembly_id_b, ctg_b).
    aa: occurrence keyed by (assembly_id, brc_fea_id); pair-table
        lookup uses (assembly_id_a, brc_a) / (assembly_id_b, brc_b).

See docs/plans/2026-05-13_aa_kmer_and_cache_symmetry_plan.md.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from scipy import sparse

from src.utils import schema


def _occurrence_col(alphabet: str) -> str:
    """k-mer matrix INDEX key for this alphabet (per the schema registry)."""
    return schema.require(alphabet).occurrence_col


def _pair_side_col(alphabet: str, side: str) -> str:
    """Return the pair-table column for the occurrence key on one side."""
    return schema.pair_occ_col(alphabet, side)


def load_kmer_index(kmer_dir: Path, k: int, alphabet: str = 'nt_ctg'
                    ) -> Dict[Tuple[str, str], int]:
    """Load (assembly_id, occurrence_id) -> row mapping from parquet.

    Args:
        kmer_dir: Directory containing the alphabet-tagged k-mer cache.
        k: k-mer size.
        alphabet: 'nt_ctg' or 'aa'.

    Returns:
        dict mapping (assembly_id, occurrence_id) tuple -> row index.
        For nt, occurrence_id is genbank_ctg_id (normalized through
        str(float(...))); for aa, it's brc_fea_id.
    """
    index_file = kmer_dir / f'kmer_features_{alphabet}_k{k}_index.parquet'
    if not index_file.exists():
        raise FileNotFoundError(f"K-mer index not found: {index_file}")

    idx_df = pd.read_parquet(index_file)
    occ_col = _occurrence_col(alphabet)

    if alphabet == 'nt_ctg':
        # Normalize genbank_ctg_id through float round-trip so keys match
        # pair CSVs. Stage 3 writes ctg columns as float, so "1564510.10"
        # becomes "1564510.1".
        occ_normalized = idx_df[occ_col].astype(str).apply(
            lambda x: str(float(x)) if x.replace('.', '', 1).isdigit() else x
        )
    else:
        occ_normalized = idx_df[occ_col].astype(str)

    keys = list(zip(idx_df['assembly_id'].astype(str), occ_normalized))
    return dict(zip(keys, idx_df['row']))


def load_kmer_matrix(kmer_dir: Path, k: int, alphabet: str = 'nt_ctg'
                     ) -> sparse.csr_matrix:
    """Load sparse k-mer feature matrix from .npz file.

    Args:
        kmer_dir: Directory containing the alphabet-tagged k-mer cache.
        k: k-mer size.
        alphabet: 'nt_ctg' or 'aa'.

    Returns:
        scipy CSR matrix of shape (N_rows, len(alphabet)**k). For aa the
        matrix is sequence-deduplicated (N_rows = unique sequences); for
        nt it has one row per occurrence (Phase 6 will migrate this).
    """
    npz_file = kmer_dir / f'kmer_features_{alphabet}_k{k}.npz'
    if not npz_file.exists():
        raise FileNotFoundError(f"K-mer features not found: {npz_file}")
    return sparse.load_npz(npz_file)


def get_kmer_pair_features(
    pairs_df: pd.DataFrame,
    kmer_matrix: sparse.csr_matrix,
    key_to_row: Dict[Tuple[str, str], int],
    interaction: str = 'concat',
    slot_transform: str = 'none',
    alphabet: str = 'nt_ctg',
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Build pair feature matrix from k-mer features and pair CSV.

    Composite (assembly_id, occurrence_id) tuples drive the lookup. For
    `alphabet='nt_ctg'` the occurrence column is `ctg_a/b`; for
    `alphabet='aa'` it is `brc_a/b`.

    Args:
        pairs_df: DataFrame with assembly_id_a/b plus either ctg_a/b
            (nt) or brc_a/b (aa), and a label column.
        kmer_matrix: sparse CSR matrix from load_kmer_matrix.
        key_to_row: mapping from load_kmer_index (composite tuples).
        interaction: 'concat', 'diff', 'unit_diff', 'prod', 'unit_prod',
            or '+'-separated combinations (e.g., 'unit_diff+prod').
            Semantics mirror the MLP path
            (`train_pair_classifier._compute_interaction`).
        slot_transform: 'none' (default) or 'unit_norm'. With
            'unit_norm', each row of emb_a and emb_b is L2-normalized
            before the interaction (matches MLP
            `slot_transform='unit_norm'`).
        alphabet: 'nt_ctg' or 'aa'.

    Returns:
        features: dense (N, D) float32 array.
        labels: (N,) int array.
    """
    # Build composite tuple keys based on alphabet-specific columns.
    occ_col_a = _pair_side_col(alphabet, 'a')
    occ_col_b = _pair_side_col(alphabet, 'b')
    keys_a = list(zip(pairs_df['assembly_id_a'].astype(str),
                      pairs_df[occ_col_a].astype(str)))
    keys_b = list(zip(pairs_df['assembly_id_b'].astype(str),
                      pairs_df[occ_col_b].astype(str)))

    # Map to row indices
    rows_a = pd.Series([key_to_row.get(k) for k in keys_a], index=pairs_df.index)
    rows_b = pd.Series([key_to_row.get(k) for k in keys_b], index=pairs_df.index)
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

    # Optional per-slot L2 row normalization (mirrors MLP slot_transform='unit_norm').
    if slot_transform == 'unit_norm':
        emb_a = emb_a / np.maximum(np.linalg.norm(emb_a, axis=1, keepdims=True), 1e-8)
        emb_b = emb_b / np.maximum(np.linalg.norm(emb_b, axis=1, keepdims=True), 1e-8)
    elif slot_transform != 'none':
        raise ValueError(
            f"get_kmer_pair_features: slot_transform={slot_transform!r} not supported "
            f"(use 'none' or 'unit_norm'). Non-negative count vectors don't benefit "
            f"from LayerNorm-style slot_norm."
        )

    # Build interaction features (mirrors the MLP path's _compute_interaction).
    tokens = {t.strip().lower() for t in interaction.split('+')}
    allowed = {'concat', 'diff', 'unit_diff', 'prod', 'unit_prod'}
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
        diff_abs = np.abs(emb_a - emb_b)
        norms = np.maximum(np.linalg.norm(diff_abs, axis=1, keepdims=True), 1e-8)
        features.append(diff_abs / norms)
    if 'prod' in tokens:
        features.append(emb_a * emb_b)
    if 'unit_prod' in tokens:
        prod = emb_a * emb_b
        norms = np.maximum(np.linalg.norm(prod, axis=1, keepdims=True), 1e-8)
        features.append(prod / norms)
    if not features:
        raise ValueError("At least one interaction term must be enabled")

    return np.concatenate(features, axis=1), labels
