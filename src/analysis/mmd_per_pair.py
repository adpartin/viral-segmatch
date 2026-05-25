"""Per-pair (S2) MMD on the model's pair representation for one Stage 3 dataset.

S2 sibling to mmd_per_slot.py. The pair entity is (HA, NA); the pair
feature is exactly what the MLP sees — currently Test 3:
`slot_transform=unit_norm`, `interaction=unit_diff+prod`. Matches the
active HA/NA and PB2/PB1 production bundles.

Two partition modes:

- `fresh_random` (default): ignore the dataset's CSV labels and do a
  random per-pair 50/50 split, repeated across N seeds. Used as the
  wiring sanity check — MMD^2 should be near zero on a random partition.

- `dataset_labels`: use the dataset's own train / test labels. Each
  unique (seq_hash_a, seq_hash_b) is assigned the split of its first
  occurrence; pairs whose hash combination appears in multiple splits
  are flagged ambiguous and filtered out (only happens under random
  per-pair routing).

Pairs are restricted to positives (label==1) — same convention as the
per-slot script. Positives are the real co-occurring (HA, NA)
combinations; negatives are constructed by the dataset builder and
their distribution depends on the negative-sampling regime, not the
biological signal we are probing.

Subsample seed is fixed across experiments. The (HA, NA) pair set
loaded is reproducible.

Run with:

    # Wiring sanity (Phase 1):
    python src/analysis/mmd_per_pair.py \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_20260520_211534 \\
        --partition_mode fresh_random \\
        --out_csv results/flu/July_2025/runs/split_separation_mmd/phase1_random_HA_NA_pair_esm2_test3.csv

    # Real comparison (Phase 2 — one run per routing):
    python src/analysis/mmd_per_pair.py \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_seq_disjoint_20260520_211109 \\
        --partition_mode dataset_labels \\
        --routing_label seq_disjoint \\
        --sigma <fixed_sigma_from_phase1> \\
        --out_csv results/flu/July_2025/runs/split_separation_mmd/phase2_seq_disjoint_HA_NA_pair_esm2_test3.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.embedding_utils import load_embedding_index, load_embeddings_by_ids
from src.utils.kmer_utils import load_kmer_index, load_kmer_matrix
from src.utils.dim_reduction_utils import compute_pca_reduction
# Reuse the kernel / MMD / permutation primitives from the per-slot script.
from src.analysis.mmd_per_slot import (
    rbf_kernel, mmd2_biased, permutation_pvalue, median_bandwidth,
)


def load_unique_pairs(dataset_dir: Path, n_isolates: int,
                       subsample_seed: int, dedup_by: str = 'protein',
                       ) -> pd.DataFrame:
    """Load positives, subsample isolates, dedup by pair hash combination.

    `dedup_by='protein'` (default): one row per unique
    (seq_hash_a, seq_hash_b). Used for ESM-2 and aa k-mer.
    `dedup_by='dna'`: one row per unique (dna_hash_a, dna_hash_b). Used
    for nt k-mer (the same protein pair may have multiple DNA encodings
    via synonymous codons, so DNA-level dedup gives slightly more pairs
    than protein-level).

    Returns a DataFrame with all lookup keys carried through:
    [assembly_id_a, brc_a, ctg_a, seq_hash_a, dna_hash_a,
     assembly_id_b, brc_b, ctg_b, seq_hash_b, dna_hash_b,
     orig_split, is_ambiguous]. `is_ambiguous` flags pairs whose hash
    combination appears in multiple splits (only under random per-pair
    routing — entity-disjoint routings have zero by construction).
    """
    if dedup_by == 'protein':
        hash_a_col, hash_b_col = 'seq_hash_a', 'seq_hash_b'
    elif dedup_by == 'dna':
        hash_a_col, hash_b_col = 'dna_hash_a', 'dna_hash_b'
    else:
        raise ValueError(f"dedup_by must be 'protein' or 'dna', got {dedup_by}")

    frames = []
    for split_name in ('train', 'val', 'test'):
        df = pd.read_csv(dataset_dir / f'{split_name}_pairs.csv', low_memory=False)
        df['orig_split'] = split_name
        frames.append(df)
    pos = pd.concat(frames, ignore_index=True)
    pos = pos[pos['label'] == 1].reset_index(drop=True)

    # Fixed subsample seed → the pair set is reproducible across runs.
    isolates = pos['assembly_id_a'].drop_duplicates().sample(
        n=min(n_isolates, pos['assembly_id_a'].nunique()),
        random_state=subsample_seed,
    )
    pos = pos[pos['assembly_id_a'].isin(set(isolates))].reset_index(drop=True)

    # Pair-level ambiguity: same hash combination straddles splits.
    pair_key = list(zip(pos[hash_a_col], pos[hash_b_col]))
    pos = pos.assign(_pair_key=pair_key)
    multi_split = (pos.groupby('_pair_key')['orig_split'].nunique())
    ambiguous = set(multi_split[multi_split > 1].index)

    cols = ['assembly_id_a', 'brc_a', 'ctg_a', 'seq_hash_a', 'dna_hash_a',
            'assembly_id_b', 'brc_b', 'ctg_b', 'seq_hash_b', 'dna_hash_b',
            'orig_split', '_pair_key']
    pairs = (pos
             .drop_duplicates(subset='_pair_key')
             .loc[:, cols]
             .reset_index(drop=True))
    pairs['is_ambiguous'] = pairs['_pair_key'].isin(ambiguous)
    return pairs


def _apply_interaction(emb_a: np.ndarray, emb_b: np.ndarray,
                        interaction: str, slot_transform: str) -> np.ndarray:
    """Per-slot L2 row-normalize (optional) then build interaction stack.

    Mirrors `kmer_utils.get_kmer_pair_features` and the MLP path's
    `train_pair_classifier._compute_interaction`. Test 3 = `unit_norm`
    slot transform + `unit_diff+prod` interaction.
    """
    emb_a = emb_a.astype(np.float32)
    emb_b = emb_b.astype(np.float32)

    if slot_transform == 'unit_norm':
        emb_a = emb_a / np.maximum(np.linalg.norm(emb_a, axis=1, keepdims=True), 1e-8)
        emb_b = emb_b / np.maximum(np.linalg.norm(emb_b, axis=1, keepdims=True), 1e-8)
    elif slot_transform != 'none':
        raise ValueError(f"slot_transform must be 'none' or 'unit_norm'; got {slot_transform!r}")

    tokens = {t.strip().lower() for t in interaction.split('+')}
    allowed = {'concat', 'diff', 'unit_diff', 'prod', 'unit_prod'}
    unknown = tokens - allowed
    if unknown:
        raise ValueError(f"Unknown interaction tokens: {unknown}")

    parts = []
    if 'concat' in tokens:
        parts.append(emb_a)
        parts.append(emb_b)
    if 'diff' in tokens:
        parts.append(np.abs(emb_a - emb_b))
    if 'unit_diff' in tokens:
        diff_abs = np.abs(emb_a - emb_b)
        norms = np.maximum(np.linalg.norm(diff_abs, axis=1, keepdims=True), 1e-8)
        parts.append(diff_abs / norms)
    if 'prod' in tokens:
        parts.append(emb_a * emb_b)
    if 'unit_prod' in tokens:
        prod = emb_a * emb_b
        norms = np.maximum(np.linalg.norm(prod, axis=1, keepdims=True), 1e-8)
        parts.append(prod / norms)
    if not parts:
        raise ValueError("At least one interaction term must be enabled")

    return np.concatenate(parts, axis=1).astype(np.float32)


def load_pair_features(pairs: pd.DataFrame, feature_space: str,
                        interaction: str, slot_transform: str,
                        embedding_path: Path, kmer_dir: Path, kmer_k: int,
                        ) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Load per-pair features for one of {esm2, kmer_aa, kmer_nt}.

    ESM-2 and aa k-mer use (assembly_id, brc_fea_id) lookup for both
    slots. nt k-mer uses (assembly_id, normalized_ctg_id) for both
    slots (numeric ctg ids are normalized through str(float(x)) to
    match the kmer index — same trick as the per-slot script).

    Returns (features, filtered_pairs, n_missing).
    """
    if feature_space == 'esm2':
        id_to_row = load_embedding_index(embedding_path)
        keys_a = list(zip(pairs['assembly_id_a'].astype(str),
                          pairs['brc_a'].astype(str)))
        keys_b = list(zip(pairs['assembly_id_b'].astype(str),
                          pairs['brc_b'].astype(str)))
        rows_a = [id_to_row.get(k) for k in keys_a]
        rows_b = [id_to_row.get(k) for k in keys_b]
        valid = np.array([(a is not None and b is not None)
                          for a, b in zip(rows_a, rows_b)])
        n_missing = int((~valid).sum())
        if n_missing:
            pairs = pairs[valid].reset_index(drop=True)
            keys_a = [k for k, v in zip(keys_a, valid) if v]
            keys_b = [k for k, v in zip(keys_b, valid) if v]
        emb_a, _ = load_embeddings_by_ids(keys_a, embedding_path, id_to_row=id_to_row)
        emb_b, _ = load_embeddings_by_ids(keys_b, embedding_path, id_to_row=id_to_row)

    elif feature_space in ('kmer_aa', 'kmer_nt'):
        alphabet = 'aa' if feature_space == 'kmer_aa' else 'nt'
        key_to_row = load_kmer_index(kmer_dir, k=kmer_k, alphabet=alphabet)
        kmer_csr = load_kmer_matrix(kmer_dir, k=kmer_k, alphabet=alphabet)

        if alphabet == 'aa':
            keys_a = list(zip(pairs['assembly_id_a'].astype(str),
                              pairs['brc_a'].astype(str)))
            keys_b = list(zip(pairs['assembly_id_b'].astype(str),
                              pairs['brc_b'].astype(str)))
        else:
            def _norm_ctg(x):
                x = str(x)
                return str(float(x)) if x.replace('.', '', 1).isdigit() else x
            keys_a = list(zip(pairs['assembly_id_a'].astype(str),
                              pairs['ctg_a'].apply(_norm_ctg)))
            keys_b = list(zip(pairs['assembly_id_b'].astype(str),
                              pairs['ctg_b'].apply(_norm_ctg)))

        rows_a = [key_to_row.get(k) for k in keys_a]
        rows_b = [key_to_row.get(k) for k in keys_b]
        valid = np.array([(a is not None and b is not None)
                          for a, b in zip(rows_a, rows_b)])
        n_missing = int((~valid).sum())
        if n_missing:
            pairs = pairs[valid].reset_index(drop=True)
            rows_a = [r for r, v in zip(rows_a, valid) if v]
            rows_b = [r for r, v in zip(rows_b, valid) if v]
        emb_a = kmer_csr[rows_a].toarray().astype(np.float32)
        emb_b = kmer_csr[rows_b].toarray().astype(np.float32)
    else:
        raise ValueError(f"unknown feature_space: {feature_space}")

    features = _apply_interaction(emb_a, emb_b, interaction, slot_transform)
    return features, pairs, n_missing


def run_fresh_random(emb_pca: np.ndarray, sigma: float,
                      n_split_seeds: int) -> list:
    """Fresh-random mode: ignore dataset labels, do N random 50/50 splits."""
    rows = []
    n = emb_pca.shape[0]
    half = n // 2
    print(f'\nFresh-random 50/50 splits × {n_split_seeds} seeds:')
    for split_seed in range(n_split_seeds):
        rng = np.random.default_rng(split_seed)
        idx = rng.permutation(n)
        A, B = emb_pca[idx[:half]], emb_pca[idx[half:]]
        mmd2 = mmd2_biased(A, B, sigma)
        rows.append({
            'split_seed': split_seed,
            'n_A': len(A), 'n_B': len(B),
            'mmd2': mmd2,
            'p_value': np.nan,
            'n_extreme': np.nan,
            'n_permutations': np.nan,
            'n_ambiguous_dropped': 0,
        })
        print(f'  seed={split_seed:>2d}  n_A={len(A):>4d}  n_B={len(B):>4d}  '
              f'mmd2={mmd2:+.6f}')
    return rows


def run_dataset_labels(pairs: pd.DataFrame, emb_pca: np.ndarray,
                        sigma: float, n_permutations: int, perm_seed: int
                        ) -> list:
    """Dataset-labels mode: filter ambiguous, slice by orig_split, one MMD."""
    n_ambig = int(pairs['is_ambiguous'].sum())
    if n_ambig:
        print(f'\nFiltering {n_ambig} ambiguous pairs (multi-split) ...')
        keep = ~pairs['is_ambiguous'].values
        pairs = pairs[keep].reset_index(drop=True)
        emb_pca = emb_pca[keep]

    train_mask = (pairs['orig_split'] == 'train').values
    test_mask = (pairs['orig_split'] == 'test').values
    A, B = emb_pca[train_mask], emb_pca[test_mask]

    print(f'\nDataset-labels MMD: train={len(A)} vs test={len(B)} ...')
    t0 = time.time()
    p_value, n_extreme, mmd2 = permutation_pvalue(
        A, B, sigma, n_permutations=n_permutations, seed=perm_seed,
    )
    perm_wall = time.time() - t0
    print(f'  mmd2={mmd2:+.6f}  p_value={p_value:.4f}  '
          f'({n_extreme}/{n_permutations} permutations >= observed)  '
          f'[{perm_wall:.1f}s]')
    return [{
        'split_seed': -1,
        'n_A': len(A), 'n_B': len(B),
        'mmd2': mmd2,
        'p_value': p_value,
        'n_extreme': n_extreme,
        'n_permutations': n_permutations,
        'n_ambiguous_dropped': n_ambig,
    }]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--dataset_dir', required=True, type=Path)
    p.add_argument('--partition_mode',
                   default='fresh_random',
                   choices=['fresh_random', 'dataset_labels'])
    p.add_argument('--routing_label', default=None)
    p.add_argument('--n_isolates', type=int, default=1000)
    p.add_argument('--subsample_seed', type=int, default=42,
                   help='Fixed across experiments — controls the (HA, NA) '
                        'pair set and the median-bandwidth subsample.')
    p.add_argument('--n_split_seeds', type=int, default=10,
                   help='fresh_random only — number of random 50/50 split repeats.')
    p.add_argument('--pca_dim', type=int, default=50)
    p.add_argument('--sigma', type=float, default=None,
                   help='RBF bandwidth. Pass a fixed value to make MMD^2 '
                        'directly comparable across routings.')
    p.add_argument('--n_permutations', type=int, default=500)
    p.add_argument('--permutation_seed', type=int, default=0)
    p.add_argument('--interaction', default='unit_diff+prod',
                   help='Pair interaction (Test 3 = unit_diff+prod).')
    p.add_argument('--slot_transform', default='unit_norm',
                   choices=['none', 'unit_norm'],
                   help='Per-slot L2 row normalize before the interaction.')
    p.add_argument('--feature_space',
                   default='esm2',
                   choices=['esm2', 'kmer_aa', 'kmer_nt'],
                   help='esm2 / kmer_aa: protein-level (dedup by '
                        '(seq_hash_a, seq_hash_b), lookup by brc). '
                        'kmer_nt: DNA-level (dedup by '
                        '(dna_hash_a, dna_hash_b), lookup by ctg).')
    p.add_argument('--embedding_path', type=Path,
                   default=Path('data/embeddings/flu/July_2025/master_esm2_embeddings.h5'),
                   help='Master ESM-2 HDF5 cache (used when feature_space=esm2).')
    p.add_argument('--kmer_dir', type=Path,
                   default=Path('data/embeddings/flu/July_2025'),
                   help='Directory holding the k-mer cache (used when '
                        'feature_space=kmer_aa or kmer_nt).')
    p.add_argument('--kmer_k', type=int, default=3,
                   help='k for the k-mer cache. Defaults: aa k=3, nt k=6.')
    p.add_argument('--out_csv', required=True, type=Path)
    args = p.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    routing_label = args.routing_label or args.dataset_dir.name

    # Pair entity dedup: protein-level for ESM-2 and aa k-mer; DNA-level
    # for nt k-mer (nt features can differ between synonymous codons).
    dedup_by = 'dna' if args.feature_space == 'kmer_nt' else 'protein'

    print(f'Loading unique (HA, NA) pairs (dedup_by={dedup_by}) ...')
    pairs = load_unique_pairs(args.dataset_dir, args.n_isolates,
                               args.subsample_seed, dedup_by=dedup_by)
    n_ambig = int(pairs['is_ambiguous'].sum())
    hash_label = '(seq_hash_a, seq_hash_b)' if dedup_by == 'protein' else '(dna_hash_a, dna_hash_b)'
    print(f'  {len(pairs)} unique {hash_label} pairs '
          f'(subsample_seed={args.subsample_seed}, n_isolates={args.n_isolates})')
    if n_ambig:
        print(f'  {n_ambig} pairs ambiguous (multi-split) — handling depends on mode')

    print(f'Loading {args.feature_space} features '
          f'(interaction={args.interaction}, slot_transform={args.slot_transform}) ...')
    t0 = time.time()
    emb, pairs, n_missing = load_pair_features(
        pairs, args.feature_space,
        interaction=args.interaction, slot_transform=args.slot_transform,
        embedding_path=args.embedding_path,
        kmer_dir=args.kmer_dir, kmer_k=args.kmer_k,
    )
    if n_missing:
        print(f'  WARNING: filtered {n_missing} pairs missing from {args.feature_space} cache')
    else:
        print(f'  all pairs found in cache (0 missing)')
    print(f'  pair features {emb.shape} in {time.time() - t0:.1f}s')

    print(f'PCA to {args.pca_dim} dims ...')
    emb_pca, _ = compute_pca_reduction(
        emb, n_components=args.pca_dim, random_state=args.subsample_seed,
    )
    print(f'  reduced to {emb_pca.shape}')

    if args.sigma is not None:
        sigma = args.sigma
        print(f'Bandwidth sigma = {sigma:.4f}  (fixed via --sigma)')
    else:
        sigma = median_bandwidth(emb_pca, seed=args.subsample_seed)
        print(f'Median bandwidth sigma = {sigma:.4f}  (computed)')

    if args.partition_mode == 'fresh_random':
        rows = run_fresh_random(emb_pca, sigma, args.n_split_seeds)
    else:
        rows = run_dataset_labels(
            pairs, emb_pca, sigma,
            n_permutations=args.n_permutations,
            perm_seed=args.permutation_seed,
        )

    for row in rows:
        row['partition_mode'] = args.partition_mode
        row['routing_label'] = routing_label
        row['slot'] = 'pair'
        row['feature_space'] = args.feature_space
        row['interaction'] = args.interaction
        row['slot_transform'] = args.slot_transform
        row['subsample_seed'] = args.subsample_seed
        row['n_pairs'] = len(pairs)
        row['n_isolates'] = args.n_isolates
        row['sigma'] = sigma

    out_df = pd.DataFrame(rows)
    col_order = ['partition_mode', 'routing_label', 'slot', 'feature_space',
                 'interaction', 'slot_transform',
                 'split_seed', 'n_A', 'n_B', 'mmd2', 'p_value', 'n_extreme',
                 'n_permutations', 'n_ambiguous_dropped',
                 'subsample_seed', 'n_pairs', 'n_isolates', 'sigma']
    out_df = out_df[col_order]
    out_df.to_csv(args.out_csv, index=False)

    if args.partition_mode == 'fresh_random':
        vals = out_df['mmd2'].values
        print(f'\nSummary across {len(vals)} split seeds:')
        print(f'  mean = {vals.mean():+.6f}')
        print(f'  std  = {vals.std():.6f}')
        print(f'  min  = {vals.min():+.6f}')
        print(f'  max  = {vals.max():+.6f}')

    print(f'\nWrote {args.out_csv}')
    print('Done.')


if __name__ == '__main__':
    main()
