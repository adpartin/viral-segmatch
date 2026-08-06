"""Diagnose DataSAIL's similarity-matrix distribution.

Why this exists. Initial L(π) runs with `similarity='mmseqspp'` returned
~0.34 on ALL three routings (cluster_disjoint, seq_disjoint, random)
on both slots. 0.34 = 1 − (0.8² + 0.1² + 0.1²) = the fraction of pair
cells outside the diagonal blocks under an 80/10/10 split. That value
is exactly what L(π) collapses to **when the similarity matrix has low
variance** (most pairs ≈ same value), because then
`leakage / total ≈ #cross-split-cells / #total-cells`.

Hypothesis: Flu A HA/NA proteins live in a tight ~70–99% identity
band, so the mmseqspp fident matrix has low variance, and L(π) becomes
uninformative.

This script extracts DataSAIL's `dataset.cluster_similarity` matrix
after running its internal `cluster()` pipeline, computes summary
statistics on the off-diagonal pair similarities, and writes a
histogram PNG. One slot at a time; reuses the same data loading as
`datasail_lpi_measure.py`.

Run with:

    conda run -n datasail python src/analysis/datasail_lpi_diagnose_sim.py \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_20260520_211534 \\
        --slot a \\
        --similarity mmseqspp \\
        --n_isolates 1000 \\
        --out_dir results/flu/July_2025/runs/datasail_lpi/diag

Outputs:
    diag_<routing>_<slot>_<sim>_n<N>.csv  — one-row stats table
    diag_<routing>_<slot>_<sim>_n<N>.png  — histogram of off-diagonal sims
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from datasail.reader.read import read_data_type
    from datasail.cluster.clustering import cluster
    from datasail.settings import KW_OUTDIR, KW_THREADS, KW_LOGDIR, KW_LINKAGE
except ImportError as e:
    sys.stderr.write(
        f"ERROR: cannot import datasail ({e}). Run in 'datasail' conda env.\n"
    )
    sys.exit(1)


def load_positives(dataset_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ('train_pairs.csv', 'val_pairs.csv', 'test_pairs.csv'):
        df = pd.read_csv(dataset_dir / split, low_memory=False)
        df['orig_split'] = split.replace('_pairs.csv', '')
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    return full[full['label'] == 1].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--dataset_dir', required=True, type=Path)
    p.add_argument('--slot', required=True, choices=['a', 'b'])
    p.add_argument('--similarity', default='mmseqspp',
                   choices=['mmseqs', 'mmseqspp'])
    p.add_argument('--n_isolates', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', required=True, type=Path)
    p.add_argument('--routing_label', default=None,
                   help='Optional label for output filenames; defaults from dataset_dir name.')
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    routing_label = args.routing_label or args.dataset_dir.name.split('_', 2)[-1][:40]
    slot_name = {'a': 'HA', 'b': 'NA'}[args.slot]

    print(f"Loading positives from {args.dataset_dir} ...")
    pos = load_positives(args.dataset_dir)
    if args.n_isolates is not None:
        isolates = pos['assembly_id_a'].drop_duplicates().sample(
            n=min(args.n_isolates, pos['assembly_id_a'].nunique()),
            random_state=args.seed,
        )
        pos = pos[pos['assembly_id_a'].isin(set(isolates))].reset_index(drop=True)
    print(f"  {len(pos):,} positives, slot {args.slot} ({slot_name})")

    data = dict(zip(pos[f'seq_hash_{args.slot}'], pos[f'seq_{args.slot}']))
    print(f"  {len(data):,} unique entities")

    print(f"Building DataSAIL dataset and clustering with similarity='{args.similarity}' ...")
    t0 = time.time()
    dataset = read_data_type('P')(
        data=data, weights=None, sim=args.similarity, dist=None,
        num_clusters=np.inf, detect_duplicates=False,
    )
    dataset = cluster(
        dataset,
        **{KW_THREADS: 1, KW_LOGDIR: None, KW_LINKAGE: 'average', KW_OUTDIR: None},
    )
    wall = time.time() - t0
    print(f"  Clustering done in {wall:.1f}s")

    sim_matrix = dataset.cluster_similarity
    if sim_matrix is None:
        print(f"ERROR: cluster_similarity is None. mode='dist'? "
              f"distance shape={dataset.cluster_distance.shape if dataset.cluster_distance is not None else None}")
        sys.exit(1)
    if not isinstance(sim_matrix, np.ndarray):
        print(f"ERROR: cluster_similarity is not ndarray (type={type(sim_matrix)})")
        sys.exit(1)

    n = sim_matrix.shape[0]
    print(f"Similarity matrix: {n}x{n} ({sim_matrix.dtype})")

    # Off-diagonal upper-triangular cells (the unique pairs)
    iu = np.triu_indices(n, k=1)
    sims = sim_matrix[iu]
    diag = np.diag(sim_matrix)

    stats = {
        'routing_label': routing_label,
        'slot': args.slot,
        'slot_name': slot_name,
        'similarity_method': args.similarity,
        'n_isolates_requested': args.n_isolates,
        'n_entities': n,
        'n_off_diag_pairs': len(sims),
        'wall_clustering_sec': round(wall, 2),
        # Off-diagonal distribution
        'off_diag_mean': float(sims.mean()),
        'off_diag_std': float(sims.std()),
        'off_diag_min': float(sims.min()),
        'off_diag_p05': float(np.percentile(sims, 5)),
        'off_diag_p25': float(np.percentile(sims, 25)),
        'off_diag_median': float(np.median(sims)),
        'off_diag_p75': float(np.percentile(sims, 75)),
        'off_diag_p95': float(np.percentile(sims, 95)),
        'off_diag_max': float(sims.max()),
        'off_diag_frac_zero': float((sims == 0).mean()),
        'off_diag_frac_lt_0p1': float((sims < 0.1).mean()),
        'off_diag_frac_gt_0p9': float((sims > 0.9).mean()),
        # Diagonal (self-sim, expected = 1)
        'diag_mean': float(diag.mean()),
        'diag_min': float(diag.min()),
    }

    tag = f"{routing_label}_{slot_name}_{args.similarity}_n{args.n_isolates}"
    csv_path = args.out_dir / f"diag_{tag}.csv"
    pd.DataFrame([stats]).to_csv(csv_path, index=False)
    print(f"\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {csv_path}")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1.0001, 51) if sims.max() <= 1.0 else 50
    ax.hist(sims, bins=bins, edgecolor='black', color='#4477AA')
    ax.set_yscale('log')
    ax.set_xlabel(f'pairwise similarity ({args.similarity})')
    ax.set_ylabel('# pairs (log)')
    ax.set_title(
        f'{slot_name} similarity distribution ({args.similarity}, '
        f'{routing_label}, n_isolates={args.n_isolates}, {n:,} entities, '
        f'{len(sims):,} pairs)\n'
        f'mean={stats["off_diag_mean"]:.3f}  std={stats["off_diag_std"]:.3f}  '
        f'p5={stats["off_diag_p05"]:.3f}  p95={stats["off_diag_p95"]:.3f}',
        fontsize=9,
    )
    ax.axvline(stats['off_diag_mean'], color='red', linestyle='--', linewidth=1,
               label=f'mean = {stats["off_diag_mean"]:.3f}')
    ax.legend()
    fig.tight_layout()
    png_path = args.out_dir / f"diag_{tag}.png"
    fig.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {png_path}")

    print('\nDone.')


if __name__ == '__main__':
    main()
