"""Per-slot MMD on ESM-2 embeddings for one Stage 3 dataset and slot.

Two partition modes:

- `fresh_random` (default): ignore the dataset's CSV labels and do a
  random per-entity 50/50 split, repeated across N seeds. Used as the
  wiring sanity check — MMD^2 should be near zero on a random partition.

- `dataset_labels`: use the dataset's own train / test labels. Each
  unique seq_hash is assigned the split of its first occurrence; entities
  appearing in pairs from multiple splits are flagged ambiguous and
  filtered out (only happens under random per-pair routing — entity-
  disjoint routings like seq_disjoint and cluster_disjoint have zero
  ambiguous entities by construction).

Subsample seed is fixed across experiments — the entity set is
reproducible. In `fresh_random` mode, only the random split varies
between split seeds.

Run with:

    # Wiring sanity (Phase 1):
    python src/analysis/mmd_per_slot.py \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_20260520_211534 \\
        --slot a --partition_mode fresh_random \\
        --out_csv results/flu/July_2025/runs/split_separation_mmd/phase1_random_HA.csv

    # Real comparison (Phase 2 — one run per routing):
    python src/analysis/mmd_per_slot.py \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_seq_disjoint_20260520_211109 \\
        --slot a --partition_mode dataset_labels \\
        --routing_label seq_disjoint \\
        --out_csv results/flu/July_2025/runs/split_separation_mmd/phase2_seq_disjoint_HA.csv
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

from src.utils.embedding_utils import load_embeddings_by_ids
from src.utils.kmer_utils import load_kmer_index, load_kmer_matrix
from src.utils.dim_reduction_utils import compute_pca_reduction


SLOT_NAME = {'a': 'HA', 'b': 'NA'}


def load_unique_slot_entities(dataset_dir: Path, slot: str,
                               n_isolates: int, subsample_seed: int
                               ) -> pd.DataFrame:
    """Load positives, subsample isolates, dedup by seq_hash on one slot.

    Returns a DataFrame [assembly_id, brc_fea_id, seq_hash, orig_split,
    is_ambiguous] — one row per unique protein. `orig_split` is the first
    split the entity was seen in; `is_ambiguous` flags entities whose
    pairs straddle multiple splits (only happens under random per-pair
    routing).
    """
    frames = []
    for split_name in ('train', 'val', 'test'):
        df = pd.read_csv(dataset_dir / f'{split_name}_pairs.csv', low_memory=False)
        df['orig_split'] = split_name        # tag source CSV for downstream use
        frames.append(df)
    pos = pd.concat(frames, ignore_index=True)
    pos = pos[pos['label'] == 1].reset_index(drop=True)

    # Fixed subsample seed → the entity set is reproducible across runs.
    isolates = pos['assembly_id_a'].drop_duplicates().sample(
        n=min(n_isolates, pos['assembly_id_a'].nunique()),
        random_state=subsample_seed,
    )
    pos = pos[pos['assembly_id_a'].isin(set(isolates))].reset_index(drop=True)

    # Detect entities whose pairs straddle multiple splits. Zero for
    # entity-disjoint routings (seq_disjoint, cluster_disjoint); ~1% for
    # random per-pair on this corpus.
    multi_split = (pos
                   .groupby(f'seq_hash_{slot}')['orig_split']
                   .nunique())
    ambiguous = set(multi_split[multi_split > 1].index)

    # One row per unique seq_hash. Pair-CSV columns are slot-suffixed
    # (brc_a, seq_hash_a, ...); rename to plain names for the lookup.
    cols = {
        f'assembly_id_{slot}': 'assembly_id',
        f'brc_{slot}': 'brc_fea_id',
        f'seq_hash_{slot}': 'seq_hash',
        'orig_split': 'orig_split',
    }
    entities = (pos
                .drop_duplicates(subset=f'seq_hash_{slot}')
                .loc[:, list(cols.keys())]
                .rename(columns=cols)
                .reset_index(drop=True))
    entities['is_ambiguous'] = entities['seq_hash'].isin(ambiguous)
    return entities


def load_features(entities: pd.DataFrame, feature_space: str,
                   embedding_path: Path, kmer_dir: Path, kmer_k: int,
                   ) -> tuple[np.ndarray, pd.DataFrame]:
    """Load per-entity features. Returns (features, filtered_entities).

    Both ESM-2 and aa k-mer caches use the same (assembly_id, brc_fea_id)
    composite key, so the entity set is identical across feature spaces
    in principle. If some entities are missing from a cache, we filter
    `entities` to match the loaded subset (in row order) and report the
    miss count.
    """
    keys = list(zip(entities['assembly_id'].astype(str),
                    entities['brc_fea_id'].astype(str)))

    if feature_space == 'esm2':
        emb, valid_keys = load_embeddings_by_ids(keys, embedding_path)
    elif feature_space == 'kmer_aa':
        key_to_row = load_kmer_index(kmer_dir, k=kmer_k, alphabet='aa')
        kmer_csr = load_kmer_matrix(kmer_dir, k=kmer_k, alphabet='aa')
        # Pick the rows for our entities; track which keys hit.
        row_idx, valid_keys = [], []
        for key in keys:
            r = key_to_row.get(key)
            if r is not None:
                row_idx.append(r)
                valid_keys.append(key)
        emb = kmer_csr[row_idx].toarray().astype(np.float32)
    else:
        raise ValueError(f"unknown feature_space: {feature_space}")

    n_missing = len(keys) - len(valid_keys)
    if n_missing:
        valid_set = set(valid_keys)
        keep = [k in valid_set for k in keys]
        entities = entities[keep].reset_index(drop=True)
    return emb, entities, n_missing


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """RBF kernel K(x, y) = exp(-||x - y||^2 / (2 sigma^2))."""
    sq = (np.sum(X ** 2, axis=1, keepdims=True)
          + np.sum(Y ** 2, axis=1)
          - 2 * X @ Y.T)
    # Numerical noise can push close-pair squared distances slightly
    # below zero; clip so exp(...) stays well-defined.
    sq = np.maximum(sq, 0.0)
    return np.exp(-sq / (2 * sigma ** 2))


def mmd2_biased(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Biased MMD^2 = mean(K(X,X)) + mean(K(Y,Y)) - 2 mean(K(X,Y))."""
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def permutation_pvalue(X: np.ndarray, Y: np.ndarray, sigma: float,
                        n_permutations: int, seed: int
                        ) -> tuple[float, int, float]:
    """Two-sample permutation test for MMD^2 under H0 'X and Y are from the
    same distribution'.

    Shuffles the X / Y assignment n_permutations times, recomputes MMD^2,
    counts how many shuffled values are >= the observed one. Returns
    (p_value, n_extreme, observed_mmd2). p_value uses add-one smoothing
    (Phipson & Smyth 2010) so the minimum reportable p is 1/(N+1) instead
    of zero.
    """
    observed = mmd2_biased(X, Y, sigma)
    Z = np.vstack([X, Y])
    n_total = Z.shape[0]
    n_A = X.shape[0]
    rng = np.random.default_rng(seed)
    n_extreme = 0
    for _ in range(n_permutations):
        idx = rng.permutation(n_total)
        X_perm = Z[idx[:n_A]]
        Y_perm = Z[idx[n_A:]]
        if mmd2_biased(X_perm, Y_perm, sigma) >= observed:
            n_extreme += 1
    p_value = (n_extreme + 1) / (n_permutations + 1)
    return p_value, n_extreme, observed


def median_bandwidth(X: np.ndarray, max_sample: int = 2000,
                      seed: int = 0) -> float:
    """Median heuristic (Gretton 2012): sigma = median pairwise L2 distance.

    Computed on a random subsample if X is large; pairwise distances are
    O(N^2) in memory.
    """
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_sample:
        idx = rng.choice(X.shape[0], max_sample, replace=False)
        X = X[idx]
    sq = (np.sum(X ** 2, axis=1, keepdims=True)
          + np.sum(X ** 2, axis=1)
          - 2 * X @ X.T)
    sq = np.maximum(sq, 0.0)
    dists = np.sqrt(sq)
    # Exclude self-pairs so the median isn't dragged down by zeros.
    np.fill_diagonal(dists, np.nan)
    return float(np.nanmedian(dists))


def run_fresh_random(emb_pca: np.ndarray, sigma: float,
                      n_split_seeds: int) -> list:
    """Fresh-random mode: ignore dataset labels, do N random 50/50 splits.

    Permutation test isn't run here — the N random seeds ARE the null
    distribution (each seed is a fresh draw under H0 'same distribution').
    """
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
            'p_value': np.nan,             # not run in this mode
            'n_extreme': np.nan,
            'n_permutations': np.nan,
            'n_ambiguous_dropped': 0,
        })
        print(f'  seed={split_seed:>2d}  n_A={len(A):>4d}  n_B={len(B):>4d}  '
              f'mmd2={mmd2:+.6f}')
    return rows


def run_dataset_labels(entities: pd.DataFrame, emb_pca: np.ndarray,
                        sigma: float, n_permutations: int, perm_seed: int
                        ) -> list:
    """Dataset-labels mode: filter ambiguous, slice by orig_split, one MMD.

    Also runs a permutation test (n_permutations shuffles of the train /
    test labels) and reports a p-value so the MMD^2 magnitude can be
    compared to its null distribution.
    """
    n_ambig = int(entities['is_ambiguous'].sum())
    if n_ambig:
        print(f'\nFiltering {n_ambig} ambiguous entities (multi-split) ...')
        keep = ~entities['is_ambiguous'].values
        entities = entities[keep].reset_index(drop=True)
        emb_pca = emb_pca[keep]

    train_mask = (entities['orig_split'] == 'train').values
    test_mask = (entities['orig_split'] == 'test').values
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
        'split_seed': -1,        # sentinel: not a seeded split
        'n_A': len(A), 'n_B': len(B),
        'mmd2': mmd2,
        'p_value': p_value,
        'n_extreme': n_extreme,
        'n_permutations': n_permutations,
        'n_ambiguous_dropped': n_ambig,
    }]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--dataset_dir',
                   required=True, type=Path,
                   help='Stage 3 dataset dir.')
    p.add_argument('--slot',
                   default='a', choices=['a', 'b'],
                   help="'a' = HA, 'b' = NA on the standard ha_na bundles.")
    p.add_argument('--partition_mode',
                   default='fresh_random',
                   choices=['fresh_random', 'dataset_labels'],
                   help='fresh_random: ignore dataset CSV labels, do N random '
                        'splits (wiring sanity). dataset_labels: use the '
                        'dataset\'s train/test labels (real comparison).')
    p.add_argument('--routing_label',
                   default=None,
                   help='Label for the output CSV. Defaults to the dataset '
                        'directory basename.')
    p.add_argument('--n_isolates',
                   type=int, default=1000,
                   help='Isolate subsample size. ~900 unique HA proteins at 1000.')
    p.add_argument('--subsample_seed',
                   type=int, default=42,
                   help='Fixed across experiments — controls which entities '
                        'we include and the median-bandwidth subsample.')
    p.add_argument('--n_split_seeds',
                   type=int, default=10,
                   help='fresh_random only — number of random 50/50 split repeats.')
    p.add_argument('--pca_dim',
                   type=int, default=50,
                   help='PCA target dimension. ESM-2 is 1280-dim.')
    p.add_argument('--sigma',
                   type=float, default=None,
                   help='RBF kernel bandwidth. If omitted, computed from the '
                        'data via median heuristic. Pass a fixed value to make '
                        'MMD^2 directly comparable across runs with different '
                        'entity sets (otherwise each run picks its own sigma).')
    p.add_argument('--n_permutations',
                   type=int, default=500,
                   help='dataset_labels only — number of train/test label '
                        'shuffles for the permutation p-value. 500 gives '
                        'p-value resolution of ~1/500.')
    p.add_argument('--permutation_seed',
                   type=int, default=0,
                   help='Seed for the permutation test shuffles.')
    p.add_argument('--feature_space',
                   default='esm2',
                   choices=['esm2', 'kmer_aa'],
                   help='esm2: 1280-dim ESM-2 protein embeddings (default). '
                        'kmer_aa: aa k-mer count vectors (vocab_size = 20^k).')
    p.add_argument('--embedding_path',
                   type=Path,
                   default=Path('data/embeddings/flu/July_2025/master_esm2_embeddings.h5'),
                   help='Master ESM-2 HDF5 cache (used when feature_space=esm2).')
    p.add_argument('--kmer_dir',
                   type=Path,
                   default=Path('data/embeddings/flu/July_2025'),
                   help='Directory holding the k-mer cache (used when '
                        'feature_space=kmer_aa).')
    p.add_argument('--kmer_k',
                   type=int, default=3,
                   help='k for the k-mer cache. aa k=3 has vocab_size 8000.')
    p.add_argument('--out_csv', required=True, type=Path)
    args = p.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    slot_name = SLOT_NAME[args.slot]
    routing_label = args.routing_label or args.dataset_dir.name

    print(f'Loading unique slot_{args.slot} ({slot_name}) entities ...')
    entities = load_unique_slot_entities(
        args.dataset_dir, args.slot, args.n_isolates, args.subsample_seed,
    )
    n_ambig = int(entities['is_ambiguous'].sum())
    print(f'  {len(entities)} unique {slot_name} seq_hashes '
          f'(subsample_seed={args.subsample_seed}, n_isolates={args.n_isolates})')
    if n_ambig:
        print(f'  {n_ambig} entities ambiguous (multi-split) — handling depends on mode')

    print(f'Loading features (feature_space={args.feature_space}) ...')
    t0 = time.time()
    emb, entities, n_missing = load_features(
        entities, args.feature_space,
        embedding_path=args.embedding_path,
        kmer_dir=args.kmer_dir, kmer_k=args.kmer_k,
    )
    if n_missing:
        print(f'  WARNING: filtered {n_missing} entities missing from {args.feature_space} cache')
    else:
        print(f'  all entities found in cache (0 missing)')
    print(f'  loaded {emb.shape} in {time.time() - t0:.1f}s')

    print(f'PCA to {args.pca_dim} dims ...')
    emb_pca, _ = compute_pca_reduction(
        emb, n_components=args.pca_dim, random_state=args.subsample_seed,
    )
    print(f'  reduced to {emb_pca.shape}')

    # Bandwidth: either fixed via CLI (for cross-run comparability) or
    # computed per-run via the median heuristic.
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
            entities, emb_pca, sigma,
            n_permutations=args.n_permutations,
            perm_seed=args.permutation_seed,
        )

    # Uniform metadata on every row → CSVs from different runs can be
    # concatenated for aggregation.
    for row in rows:
        row['partition_mode'] = args.partition_mode
        row['routing_label'] = routing_label
        row['slot'] = args.slot
        row['feature_space'] = args.feature_space
        row['subsample_seed'] = args.subsample_seed
        row['n_entities'] = len(entities)
        row['n_isolates'] = args.n_isolates
        row['sigma'] = sigma

    out_df = pd.DataFrame(rows)
    col_order = ['partition_mode', 'routing_label', 'slot', 'feature_space',
                 'split_seed', 'n_A', 'n_B', 'mmd2', 'p_value', 'n_extreme',
                 'n_permutations', 'n_ambiguous_dropped',
                 'subsample_seed', 'n_entities', 'n_isolates', 'sigma']
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
