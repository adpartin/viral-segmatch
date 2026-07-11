"""Regenerable figures for a cluster set (reads existing artifacts; recomputes nothing).

Reads a cluster set's `t<NN>/<short>_cluster.parquet` and renders per-function
figures into `<clusters_root>/figures/`. It never re-clusters: delete a PNG and
re-run to regenerate it. Kept separate from the cluster builders so they stay
figure-free. `--clusters_root` can point at any cluster set (OOD `clusters_*_ood/`
or set-cover `clusters_*/`); which figures apply depends on what that set carries.

Figures (`--plots`), and which cluster sets they apply to:
  separation  -- the cluster set's `>= t`/cov similarity matrix, sequences ordered
    by cluster (largest first). Each dot is one `>= t`/cov hit, colored by % identity.
    Within-cluster hits fill the block-diagonal blocks; any dot OFF the block diagonal
    is a cross-cluster link (a separation violation -- expected 0, since a cluster IS a
    connected component of this hit graph). Drawn as a scatter (not a shrunk image) with
    violations ringed in red, so a lone violation can never be hidden. The visual
    companion to src/analysis/verify_ood_clusters.py, which certifies the same 0.
    Needs the all-vs-all `<short>_hits.tsv` -> OOD sets only (build_ood_clusters).
  umap        -- 2-D UMAP of the ESM-2 embeddings of the cluster sequences, colored by
    cluster. Qualitative check that same-cluster sequences group together in ESM-2 space.
    Uses the existing master embedding cache (no embeddings computed) -> aa sets only
    (ESM-2 is a protein model).
  barplot     -- top-N unique-weighted cluster-size barplot (bars = the largest
    clusters; height = unique-sequence count, labeled with its % of the slot). Needs
    only the cluster parquet -> any cluster set / alphabet.

Cluster / mega-cluster here = single-segment SIMILARITY-graph component (nodes =
sequences), NOT the bipartite CC / mega-CC of 2D-CD routing (docs/methods/glossary.md).

CLI:
    # separation map (default)
    python -m src.analysis.plot_clusters \\
        --clusters_root data/processed/flu/July_2025/clusters_aa_ood \\
        --threshold 0.99 --functions M1

    # add the ESM-2 UMAP (aa only)
    python -m src.analysis.plot_clusters \\
        --clusters_root data/processed/flu/July_2025/clusters_aa_ood \\
        --threshold 0.99 --functions M1 --plots separation umap \\
        --protein_final data/processed/flu/July_2025/protein_final.parquet \\
        --embeddings    data/embeddings/flu/July_2025/master_esm2_embeddings.h5
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
from src.utils.clustering_utils import (  # noqa: E402
    cluster_sizes_unique,
    read_hits_tsv,
    threshold_decimal,
    threshold_label,
)
from src.utils.plot_config import get_protein_color  # noqa: E402


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

    Reads `t<NN>/<short>_cluster.parquet` (hash -> cluster_id) and the `>= t`/cov edges
    from `t<NN>/<short>_hits.parquet` (or `.tsv`). Lays sequences out by cluster
    (largest first) and draws each hit at (row, col) colored by % identity.
    Off-block-diagonal dots are cross-cluster links; there should be none.

    Returns `(fig_path, stats)` where stats = {n_seqs, n_clusters, n_cross}.
    """
    clusters_root = Path(clusters_root)
    tlabel = threshold_label(threshold)
    tdir = clusters_root / tlabel
    parquet = tdir / f"{short_name}_cluster.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"missing OOD artifact (build it first): {parquet}")
    hits_parquet = tdir / f"{short_name}_hits.parquet"
    hits_tsv = tdir / f"{short_name}_hits.tsv"

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
    if hits_parquet.exists():
        hits = pd.read_parquet(hits_parquet, columns=['query', 'target', 'fident'])
    elif hits_tsv.exists():
        hits = read_hits_tsv(hits_tsv, usecols=['query', 'target', 'fident'])
    else:
        raise FileNotFoundError(
            f"no hits for the separation map ({hits_parquet.name} / {hits_tsv.name}); "
            f"rebuild without --delete_hits")
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
    if within.any():
        sc = ax.scatter(ti[within], qi[within], c=fid[within], cmap='viridis',
                        vmin=float(threshold), vmax=1.0, s=2, marker='s',
                        linewidths=0, rasterized=True, zorder=1)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='sequence identity (fident)')
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


def _esm2_model_sig(attrs: dict) -> str:
    """Rebuild the ESM-2 cache-key signature from the master file's attrs.

    Mirrors `get_model_sig()` in src/utils/esm2_utils.py, but rebuilt from the h5
    attrs so this plotting script does not import torch/transformers (esm2_utils does).
    The master cache key for a sequence is `sha1(esm2_ready_seq)::<this sig>`.
    """
    return (f"{attrs['model_name']}|{attrs['pooling']}|{attrs['layer']}"
            f"|{int(attrs['max_length']) - 2}|{attrs['precision']}")


def _load_cluster_embeddings(
    clusters: pd.DataFrame,
    hash_col: str,
    full_name: str,
    *,
    protein_final: Path,
    embeddings_h5: Path,
    embeddings_index: Path,
    ) -> tuple:
    """Return `(X, cint, n_missing)`: one ESM-2 embedding per cluster sequence + its cluster rank.

    Joins cluster nodes (`prot_hash` -> cluster_id) to ESM-2 embeddings through the
    EXACT string Stage 2 embedded: `protein_final.esm2_ready_seq`, whose master-cache
    key is `sha1(esm2_ready_seq)::<model_sig>`. aa only (ESM-2 is a protein model).
    """
    import hashlib

    import h5py  # lazy: only the UMAP path needs HDF5 (matches the codebase pattern)

    if hash_col != 'prot_hash':
        raise ValueError(f"UMAP projection is aa-only (needs prot_hash + ESM-2); got {hash_col!r}")

    # prot_hash -> the exact embedded string (esm2_ready_seq), from protein_final.
    pf = pd.read_parquet(protein_final, columns=['function', 'prot_hash', 'esm2_ready_seq'])
    pf = pf[pf['function'] == full_name].drop_duplicates('prot_hash')
    seq_of = dict(zip(pf['prot_hash'], pf['esm2_ready_seq']))

    # Each node's cache key = sha1(its embedded string) :: model_sig.
    with h5py.File(embeddings_h5, 'r') as f:
        sig = _esm2_model_sig({k: f.attrs[k] for k in f.attrs})
    nodes = clusters[[hash_col, '_cint']].copy()
    nodes['seq'] = nodes[hash_col].map(seq_of)
    nodes['cache_key'] = nodes['seq'].map(
        lambda s: f"{hashlib.sha1(s.encode()).hexdigest()}::{sig}" if isinstance(s, str) else None)

    # Map cache_key -> row in the master embedding matrix (keep only our keys).
    my_keys = set(nodes['cache_key'].dropna())
    index = pd.read_parquet(embeddings_index, columns=['cache_key', 'row'])
    index = index[index['cache_key'].isin(my_keys)].drop_duplicates('cache_key')
    key2row = dict(zip(index['cache_key'], index['row']))
    nodes['row'] = nodes['cache_key'].map(key2row)

    have = nodes.dropna(subset=['row'])
    n_missing = len(nodes) - len(have)
    if have.empty:
        raise ValueError("no cluster sequence had an ESM-2 embedding -- check the embeddings file/index.")

    # Load only the rows we need (h5py fancy-indexing wants sorted-unique indices).
    rows = have['row'].astype(int).to_numpy()
    uniq, inv = np.unique(rows, return_inverse=True)
    with h5py.File(embeddings_h5, 'r') as f:
        emb_uniq = f['emb'][uniq.tolist()]        # (n_uniq, 1280) fp16
    X = emb_uniq[inv].astype(np.float32)          # (n_nodes, 1280), one row per cluster seq
    cint = have['_cint'].astype(int).to_numpy()
    return X, cint, n_missing


def plot_cluster_umap(
    clusters_root: Path,
    threshold: float,
    short_name: str,
    out_dir: Path,
    *,
    protein_final: Path,
    embeddings_h5: Path,
    embeddings_index: Path,
    top_n: int = 12,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    seed: int = 42,
    dpi: int = 200,
    ) -> tuple:
    """UMAP of the ESM-2 embeddings of one function's cluster sequences, colored by cluster.

    Qualitative companion to the separation map: sequences that cluster together at
    threshold `t` should land near each other in ESM-2 space. The `top_n` largest
    clusters get distinct colors (+ a legend); the rest are gray. aa only.

    Returns `(fig_path, stats)` where stats = {n_plotted, n_missing, n_clusters}.
    """
    import umap  # lazy: heavy import (numba), only needed for this figure

    clusters_root = Path(clusters_root)
    tlabel = threshold_label(threshold)
    parquet = clusters_root / tlabel / f"{short_name}_cluster.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"missing OOD artifact (build it first): {parquet}")
    clusters = pd.read_parquet(parquet)
    alphabet = str(clusters['alphabet'].iloc[0])
    hash_col = schema.SCHEMA[alphabet].hash_col
    full_name = str(clusters['function'].iloc[0])
    clusters = clusters.assign(_cint=clusters['cluster_id'].map(_cluster_int))
    n_total = len(clusters)
    n_clusters = int(clusters['_cint'].nunique())

    X, cint, n_missing = _load_cluster_embeddings(
        clusters, hash_col, full_name, protein_final=protein_final,
        embeddings_h5=embeddings_h5, embeddings_index=embeddings_index)
    n_plotted = len(cint)

    # 1280-dim ESM-2 -> 2-D. random_state fixes the layout (and makes UMAP single-threaded).
    xy = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                   random_state=seed).fit_transform(X)

    # Color the top_n largest clusters distinctly (cluster rank 0 = largest); rest gray.
    sizes = pd.Series(cint).value_counts()  # cluster rank -> n plotted
    top = list(sizes.sort_values(ascending=False).index[:top_n])
    palette = plt.get_cmap('tab20')
    color_of = {c: palette(i % 20) for i, c in enumerate(top)}

    fig, ax = plt.subplots(figsize=(9, 8))
    other = ~np.isin(cint, top)
    if other.any():
        ax.scatter(xy[other, 0], xy[other, 1], s=6, c='lightgray',
                   linewidths=0, rasterized=True, label=f'other ({int(other.sum())})')
    for c in top:
        m = cint == c
        ax.scatter(xy[m, 0], xy[m, 1], s=12, color=color_of[c],
                   linewidths=0, rasterized=True, label=f'{short_name}_{c} (n={int(m.sum())})')
    ax.legend(loc='best', fontsize=7, framealpha=0.9,
              title=f'top {len(top)} of {n_clusters:,} clusters')
    missing_note = f" ({n_missing:,} without embedding)" if n_missing else ""
    ax.set_title(f"{short_name} {alphabet} {tlabel} -- ESM-2 UMAP, colored by cluster\n"
                 f"{n_plotted:,} of {n_total:,} seqs embedded{missing_note}; "
                 f"metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist}, seed={seed}")
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{short_name}_{tlabel}_umap.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return fig_path, {'n_plotted': n_plotted, 'n_missing': n_missing, 'n_clusters': n_clusters}


def plot_cluster_size_barplot(
    sizes: pd.Series,
    *,
    protein: str,
    alphabet: str,
    threshold_id: str,
    top_n: int,
    out_png: Path,
    ) -> None:
    """Top-N unique-weighted cluster-size barplot for one slot.

    Each bar is one cluster (largest first); height = unique-sequence count, labeled
    with the raw count and its % of the slot's total unique sequences. Also reused by
    the multi-slice sweep in src/analysis/cluster_size_barplot.py.

    Args:
        sizes: cluster_id -> unique-seq count, descending (from cluster_sizes_unique).
        protein: short name (e.g. 'HA') -- sets the bar color (via get_protein_color).
        alphabet: 'aa' / 'nt_cds' / 'nt_ctg' -- title/label only.
        threshold_id: 'tXXX' -- title/label only.
        top_n: number of largest clusters to draw.
        out_png: output PNG path.
    """
    n_unique = int(sizes.sum())
    n_clusters = int(len(sizes))
    top = sizes.head(top_n)
    pcts = top.values / n_unique * 100.0
    top_cov = float(pcts.sum())

    fig, ax = plt.subplots(figsize=(max(9.0, len(top) * 0.55), 5.6))
    xs = np.arange(len(top))
    ax.bar(xs, top.values, color=get_protein_color(protein), edgecolor='black', linewidth=0.5)
    for x, c, p in zip(xs, top.values, pcts):
        ax.annotate(f'{int(c):,}\n{p:.1f}%', xy=(x, c), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7, color='#222')
    ax.set_xticks(xs)
    ax.set_xticklabels(top.index, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('cluster_id (rank-ordered, largest first)', fontsize=9)
    ax.set_ylabel('unique sequences in cluster', fontsize=9)
    ax.set_ylim(0, top.values.max() * 1.18)  # headroom for the count+% labels
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(
        f'{protein} — {alphabet} — {threshold_id} (id={threshold_decimal(threshold_id):.2f})\n'
        f'top {len(top)} of {n_clusters:,} clusters  ·  '
        f'total unique seqs: {n_unique:,}  ·  '
        f'top {len(top)} cover {top_cov:.1f}% of unique',
        fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _alphabet_from_root(clusters_root: Path) -> str:
    """Infer the alphabet from a cluster-set root name (`clusters_<alphabet>[_ood]`)."""
    name = Path(clusters_root).name
    for a in ('nt_ctg', 'nt_cds', 'aa'):
        if a in name:
            return a
    return '?'


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
    p.add_argument('--plots', nargs='+', default=['separation'], choices=['separation', 'umap', 'barplot'],
                   help='which figures to make')
    p.add_argument('--out_subdir', default='figures', help='figure dir under --clusters_root')
    p.add_argument('--dpi', type=int, default=200, help='figure DPI')

    # UMAP-only (aa): the exact embedded string comes from protein_final.esm2_ready_seq,
    # and the vectors from the master ESM-2 cache (no embeddings are computed here).
    p.add_argument('--protein_final', help='UMAP: protein_final.parquet (source of esm2_ready_seq)')
    p.add_argument('--embeddings', help='UMAP: master ESM-2 HDF5 (emb + emb_keys)')
    p.add_argument('--embeddings_index',
                   help='UMAP: ESM-2 index parquet (default: --embeddings with .parquet suffix)')
    p.add_argument('--top_n', type=int, default=12,
                   help='umap/barplot: the N largest clusters to color / draw as bars')
    p.add_argument('--n_neighbors', type=int, default=15, help='UMAP n_neighbors')
    p.add_argument('--min_dist', type=float, default=0.1, help='UMAP min_dist')
    p.add_argument('--metric', default='cosine', help='UMAP metric (cosine suits ESM-2 embeddings)')
    p.add_argument('--seed', type=int, default=42, help='UMAP random_state (fixes the layout)')
    args = p.parse_args()

    clusters_root = Path(args.clusters_root)
    tdir = clusters_root / threshold_label(args.threshold)
    if not tdir.is_dir():
        raise SystemExit(f"threshold dir not found: {tdir}")
    functions = args.functions or _available_functions(tdir)
    if not functions:
        raise SystemExit(f"no *_cluster.parquet found in {tdir}")
    out_dir = clusters_root / args.out_subdir

    emb_index = None
    if 'umap' in args.plots:
        if not (args.protein_final and args.embeddings):
            raise SystemExit("--plots umap requires --protein_final and --embeddings")
        emb_index = args.embeddings_index or str(Path(args.embeddings).with_suffix('.parquet'))

    for short in functions:
        if 'separation' in args.plots:
            fig_path, s = plot_separation_map(clusters_root, args.threshold, short, out_dir, dpi=args.dpi)
            print(f"  [{short}] separation: {s['n_clusters']:,} clusters, {s['n_seqs']:,} seqs, "
                  f"{s['n_cross']:,} cross-cluster hits -> {fig_path}")
        if 'umap' in args.plots:
            fig_path, s = plot_cluster_umap(
                clusters_root, args.threshold, short, out_dir,
                protein_final=args.protein_final, embeddings_h5=args.embeddings,
                embeddings_index=emb_index, top_n=args.top_n, n_neighbors=args.n_neighbors,
                min_dist=args.min_dist, metric=args.metric, seed=args.seed, dpi=args.dpi)
            print(f"  [{short}] umap: {s['n_plotted']:,} of {s['n_plotted'] + s['n_missing']:,} "
                  f"seqs embedded -> {fig_path}")
        if 'barplot' in args.plots:
            parquet = tdir / f"{short}_cluster.parquet"
            if not parquet.exists():
                raise FileNotFoundError(f"missing cluster parquet: {parquet}")
            sizes = cluster_sizes_unique(parquet)
            fig_path = out_dir / f"{short}_{tdir.name}_barplot.png"
            plot_cluster_size_barplot(
                sizes, protein=short, alphabet=_alphabet_from_root(clusters_root),
                threshold_id=tdir.name, top_n=args.top_n, out_png=fig_path)
            print(f"  [{short}] barplot: top {min(args.top_n, len(sizes))} of "
                  f"{len(sizes):,} clusters -> {fig_path}")


if __name__ == '__main__':
    main()
