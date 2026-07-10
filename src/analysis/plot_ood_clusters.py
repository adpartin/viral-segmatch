"""Regenerable QC figures for OOD cluster sets (reads existing artifacts; recomputes nothing).

Reads the cluster parquet + easy-search hits TSV that `build_ood_clusters.py`
wrote, and renders per-function figures into `<clusters_root>/figures/`. It never
re-clusters: delete a PNG and re-run to regenerate it. Kept separate from the
builder so the production builder stays figure-free.

Figures:
  separation map -- the cluster set's `>= t`/cov similarity matrix, with sequences
    ordered by cluster (largest first). Each dot is one `>= t`/cov hit, colored by
    % identity. Within-cluster hits fill the block-diagonal blocks; any dot OFF the
    block diagonal is a cross-cluster link (a separation violation -- expected 0,
    since a cluster IS a connected component of this hit graph). Drawn as a scatter
    (not a shrunk image) with violations ringed in red, so a lone violation can
    never be hidden -- see the rendering note in `plot_separation_map`. The visual
    companion to src/analysis/verify_ood_clusters.py, which certifies the same 0.

Cluster / mega-cluster here = single-segment SIMILARITY-graph component (nodes =
sequences), NOT the bipartite CC / mega-CC of 2D-CD routing (docs/methods/glossary.md).

CLI:
    python -m src.analysis.plot_ood_clusters \\
        --clusters_root data/processed/flu/July_2025/clusters_aa_ood \\
        --threshold 0.99 --functions M1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # headless backend; must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import schema  # noqa: E402
from src.utils.clustering_utils import read_hits_tsv, threshold_label  # noqa: E402


def _cluster_int(cluster_id: str) -> int:
    """Rank suffix of a cluster_id, e.g. 'M1_7' -> 7 (0 = largest cluster)."""
    return int(str(cluster_id).rsplit('_', 1)[-1])


def plot_separation_map(
    clusters_root: Path,
    threshold: float,
    short_name: str,
    out_dir: Path,
    *,
    dpi: int = 200,
    ) -> tuple:
    """Render one function's `>= t`/cov similarity matrix, sequences ordered by cluster.

    Reads `t<NN>/<short>_cluster.parquet` (hash -> cluster_id) and
    `t<NN>/<short>_hits.tsv` (the `>= t`/cov edges). Lays sequences out by cluster
    (largest first) and draws each hit at (row, col) colored by % identity.
    Off-block-diagonal dots are cross-cluster links; there should be none.

    Returns `(fig_path, stats)` where stats = {n_seqs, n_clusters, n_cross}.
    """
    clusters_root = Path(clusters_root)
    tlabel = threshold_label(threshold)
    tdir = clusters_root / tlabel
    parquet = tdir / f"{short_name}_cluster.parquet"
    hits_tsv = tdir / f"{short_name}_hits.tsv"
    for path in (parquet, hits_tsv):
        if not path.exists():
            raise FileNotFoundError(f"missing OOD artifact (build it first): {path}")

    clusters = pd.read_parquet(parquet)
    alphabet = str(clusters['alphabet'].iloc[0])
    hash_col = schema.SCHEMA[alphabet].hash_col

    # Order sequences by cluster: largest first (the cluster_id suffix is the size
    # rank), then by hash within a cluster -- a stable, cluster-contiguous layout.
    clusters = clusters.assign(_cint=clusters['cluster_id'].map(_cluster_int))
    clusters = clusters.sort_values(['_cint', hash_col]).reset_index(drop=True)
    n = len(clusters)
    n_clusters = int(clusters['_cint'].nunique())
    pos = dict(zip(clusters[hash_col], range(n)))          # hash -> row/col index
    cl = dict(zip(clusters[hash_col], clusters['_cint']))  # hash -> cluster rank

    # Edges: every hit already meets the `>= t`/cov rule. Map endpoints to grid
    # positions, drop any unmapped row and the trivial self-identity diagonal.
    hits = read_hits_tsv(hits_tsv, usecols=['query', 'target', 'fident'])
    qi = hits['query'].map(pos).to_numpy()
    ti = hits['target'].map(pos).to_numpy()
    fid = hits['fident'].astype(float).to_numpy()
    clq = hits['query'].map(cl).to_numpy()
    clt = hits['target'].map(cl).to_numpy()
    keep = ~(np.isnan(qi) | np.isnan(ti)) & (qi != ti)
    qi, ti, fid = qi[keep].astype(int), ti[keep].astype(int), fid[keep]
    clq, clt = clq[keep], clt[keep]
    cross = clq != clt          # endpoints in different clusters = a violation
    within = ~cross
    n_cross = int(cross.sum())  # 0 by construction; shown in the title for QC

    # --- How this is drawn, and why (read before trusting the picture) ---
    # We draw the matrix as a SCATTER -- one dot per hit -- NOT as an image shrunk
    # to fit the page. A shrunk image can silently drop a single off-diagonal dot,
    # which is exactly the point we most need to see: a cross-cluster hit (a
    # violation). A scatter draws every hit, and we draw any cross-cluster hit LAST,
    # in red and larger, so a violation can never hide behind other dots. The count
    # in the title (n_cross) is the exact, authoritative number; the picture is its
    # companion. verify_ood_clusters.py certifies the same count independently.
    fig, ax = plt.subplots(figsize=(9, 8))
    # Faint identity diagonal: one gray dot per sequence, so singletons (which have
    # no other hit and would otherwise be invisible) still show, and orientation is clear.
    ax.scatter(range(n), range(n), s=1, c='lightgray', marker='s',
               linewidths=0, rasterized=True, zorder=0)
    # Within-cluster hits, colored by % identity -- these fill the block-diagonal blocks.
    mappable = None
    if within.any():
        mappable = ax.scatter(ti[within], qi[within], c=fid[within], cmap='viridis',
                              vmin=float(threshold), vmax=1.0, s=2, marker='s',
                              linewidths=0, rasterized=True, zorder=1)
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, label='sequence identity (fident)')
    # Cross-cluster hits = violations: drawn last, ringed in red so they can't be missed.
    if n_cross:
        ax.scatter(ti[cross], qi[cross], s=20, facecolors='none', edgecolors='red',
                   linewidths=1.2, zorder=3, label=f'cross-cluster hit ({n_cross:,})')
        ax.legend(loc='upper right', fontsize=8)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # origin top-left (matrix convention)
    ax.set_aspect('equal')
    status = 'holds' if n_cross == 0 else 'VIOLATED'
    ax.set_title(f"{short_name} {alphabet} {tlabel} -- cluster separation map\n"
                 f"{n_clusters:,} clusters, {n:,} seqs; cross-cluster hits "
                 f"(>= t identity, >= cov): {n_cross:,}  ->  separation {status}")
    ax.set_xlabel('sequence (ordered by cluster, largest first)')
    ax.set_ylabel('sequence (ordered by cluster, largest first)')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{short_name}_{tlabel}_separation.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return fig_path, {'n_seqs': n, 'n_clusters': n_clusters, 'n_cross': n_cross}


def _available_functions(tdir: Path) -> list:
    """Function short-names with a cluster parquet in this t<NN> dir (excludes combined)."""
    shorts = []
    for path in sorted(tdir.glob('*_cluster.parquet')):
        stem = path.name[:-len('_cluster.parquet')]
        if stem != 'combined':
            shorts.append(stem)
    return shorts


def main() -> None:
    p = argparse.ArgumentParser(
        description="Regenerable QC figures for OOD cluster sets (reads artifacts; no recompute).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--clusters_root', required=True,
                   help='OOD cluster set root, e.g. .../clusters_aa_ood')
    p.add_argument('--threshold', type=float, required=True, help='identity threshold, e.g. 0.99')
    p.add_argument('--functions', nargs='+',
                   help='function short names (default: all present in the t<NN> dir)')
    p.add_argument('--out_subdir', default='figures', help='figure dir under --clusters_root')
    p.add_argument('--dpi', type=int, default=200, help='figure DPI')
    args = p.parse_args()

    clusters_root = Path(args.clusters_root)
    tdir = clusters_root / threshold_label(args.threshold)
    if not tdir.is_dir():
        raise SystemExit(f"threshold dir not found: {tdir}")
    functions = args.functions or _available_functions(tdir)
    if not functions:
        raise SystemExit(f"no *_cluster.parquet found in {tdir}")
    out_dir = clusters_root / args.out_subdir

    for short in functions:
        fig_path, s = plot_separation_map(
            clusters_root, args.threshold, short, out_dir, dpi=args.dpi)
        print(f"  [{short}] {s['n_clusters']:,} clusters, {s['n_seqs']:,} seqs, "
              f"{s['n_cross']:,} cross-cluster hits -> {fig_path}")


if __name__ == '__main__':
    main()
