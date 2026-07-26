"""k-mer UMAPs of CC structure: single-side clusters, mega-CC sides, and fragment sides.

Consumes the CC-structure artifacts (`build_cc_structure.py`) + the single-side cluster sets,
and renders k-mer UMAPs via the shared `plot_utils.umap_scatter`. The representation is LOADED
(never recomputed) from the exact k-mer matrix training consumes -- `kmer_utils.load_kmer_matrix`
keyed by `(assembly_id, occurrence_col)`, bridged to our `cds_dna_hash` plot points through
`kmer_utils.build_hash_to_kmer_row` (via `*_final`). Sparse 4096-dim counts are TruncatedSVD-reduced
before UMAP (the reduction the repo prescribes for count features), so what we plot is what the model
sees. aa (ESM-2) is a later step; this module is nt_cds / nt_ctg.

Modes (`--mode`):
  single_side  -- one slot's cluster sequences (points = sequences), colored by cluster (4a).
  megacc       -- each side of the natural mega-CC (largest CC), colored by cluster (4b).
  fragment     -- each side of the top fragments (edge-cut CCs), colored by fragment (4c).

CLI:
    python -m src.analysis.umap_cc --mode single_side \\
        --clusters_root data/processed/flu/July_2025/clusters_nt_cds_ood --threshold t099 --short HA \\
        --out_dir data/processed/flu/July_2025/clusters_nt_cds_ood/figures
    python -m src.analysis.umap_cc --mode megacc \\
        --cc_dir data/processed/flu/July_2025/cc_nt_cds_ood/HA-NA/t099 \\
        --universe data/processed/flu/July_2025/pair_universe_nt_cds/HA-NA/pairs.parquet --pair HA-NA
    python -m src.analysis.umap_cc --mode fragment \\
        --cc_dir data/processed/flu/July_2025/cc_nt_cds_ood/HA-NA/t099/fragmented \\
        --universe data/processed/flu/July_2025/pair_universe_nt_cds/HA-NA/pairs.parquet --pair HA-NA
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils import schema  # noqa: E402
from src.utils.clustering_utils import threshold_label  # noqa: E402
from src.utils.dim_reduction_utils import compute_truncated_svd_reduction  # noqa: E402
from src.utils.kmer_utils import build_hash_to_kmer_row, load_kmer_matrix  # noqa: E402
from src.utils.plot_utils import umap_scatter  # noqa: E402

KMER_DIR = PROJ / 'data/embeddings/flu/July_2025'
# the alphabet's *_final table (carries the sequence-hash + occurrence key for the hash->row bridge).
FINAL = {
    'nt_cds': PROJ / 'data/processed/flu/July_2025/cds_dna_final.parquet',
    'nt_ctg': PROJ / 'data/processed/flu/July_2025/ctg_dna_final.parquet',
}
# Categorical colors for the CC UMAPs -- seaborn 'muted' palette (distinct from the dark-gray
# reserved for 'Others').
_COLOR_PALETTE = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4',
                  '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c6e2']
# UMAP / SVD reduction params -- fixed here so cached coords stay consistent with the layout.
_UMAP_NEIGHBORS, _UMAP_MIN_DIST, _SVD_DIM = 15, 0.1, 50


def cluster_vectors(hashes, alphabet: str, *, kmer_k: int = 6, svd_dim: int = 50, seed: int = 42):
    """`hashes` (canonical sequence-hashes) -> `(X, keep)`: TruncatedSVD-reduced k-mer vectors.

    Loads the k-mer matrix training consumes and bridges each hash to its row
    (`build_hash_to_kmer_row`); hashes without a k-mer row are dropped. `keep` is a boolean mask
    aligned to `hashes` (so the caller subsets its labels the same way); `X` is `(keep.sum(), D)`
    in the kept-hash order. A degenerate input of <= 2 sequences skips SVD (returns raw counts).
    nt_cds / nt_ctg only (aa ESM-2 is a later step).
    """
    if alphabet not in FINAL:
        raise NotImplementedError(f"cluster_vectors: alphabet {alphabet!r} not wired (aa=ESM-2 later)")
    h2r = build_hash_to_kmer_row(KMER_DIR, kmer_k, alphabet, FINAL[alphabet])
    mat = load_kmer_matrix(KMER_DIR, kmer_k, alphabet=alphabet)
    rows = np.array([h2r.get(h, -1) for h in hashes], dtype=int)
    keep = rows >= 0
    xs = mat[rows[keep]]                                     # sparse (M, vocab)
    if xs.shape[0] == 0:
        raise ValueError("no plot hash resolved to a k-mer row -- check alphabet / final table")
    d = min(svd_dim, xs.shape[0] - 1, xs.shape[1] - 1)
    x = xs.toarray() if d < 2 else compute_truncated_svd_reduction(xs, n_components=d, random_state=seed)[0]
    return x, keep


def _side_sequences(pw: pd.DataFrame, uni: pd.DataFrame, slot: str, alphabet: str) -> pd.DataFrame:
    """Per-UNIQUE-sequence `(seq_hash, cluster_id, cc_id)` for one slot of a pair set.

    Joins `pairs_with_cc` (`pair_key`, `cluster_id_{slot}`, `cc_id`) to the universe
    (`pair_key`, `<hash_col>_{slot}`) and dedups on the sequence-hash -- a sequence sits in one
    cluster, and a cluster in one CC, so the dedup is exact. The hash column is the alphabet's
    (`cds_dna_hash` for nt_cds, `ctg_dna_hash` for nt_ctg), from the schema registry.
    """
    hcol, ccol = f'{schema.SCHEMA[alphabet].hash_col}_{slot}', f'cluster_id_{slot}'
    m = pw.merge(uni[['pair_key', hcol]], on='pair_key', how='left')
    side = m[[hcol, ccol, 'cc_id']].drop_duplicates(hcol)
    return side.rename(columns={hcol: 'seq_hash', ccol: 'cluster_id'})


def _umap_coords(hashes, alphabet, *, kmer_k, seed, metric, cache_dir):
    """`(xy 2-D, keep-mask)` for `hashes`, cached to disk (skips the k-mer load + SVD + UMAP on a
    hit). The cache key fingerprints everything that fixes the layout -- the hashes, the reduction
    params, and the k-mer matrix mtime -- so a data or param change misses and recomputes; only
    cosmetic changes (color / legend / title) reuse the coords. Delete `_umap_coords/` to rebuild.
    """
    import hashlib

    from src.utils.dim_reduction_utils import compute_umap_reduction
    key = None
    if cache_dir is not None:
        npz = KMER_DIR / f'kmer_features_{alphabet}_k{kmer_k}.npz'
        mtime = int(npz.stat().st_mtime) if npz.exists() else 0
        sig = f'{alphabet}|{kmer_k}|{seed}|{metric}|{_UMAP_NEIGHBORS}|{_UMAP_MIN_DIST}|{_SVD_DIM}|{mtime}'
        digest = hashlib.md5(('|'.join(map(str, hashes)) + '||' + sig).encode()).hexdigest()[:16]
        key = Path(cache_dir) / f'coords_{digest}.npz'
        if key.exists():
            d = np.load(key)
            return d['xy'], d['keep']
    x, keep = cluster_vectors(hashes, alphabet, kmer_k=kmer_k, svd_dim=_SVD_DIM, seed=seed)
    xy = compute_umap_reduction(x, n_components=2, n_neighbors=_UMAP_NEIGHBORS,
                                min_dist=_UMAP_MIN_DIST, metric=metric, random_state=seed)[0]
    if key is not None:
        key.parent.mkdir(parents=True, exist_ok=True)
        np.savez(key, xy=xy, keep=keep)
    return xy, keep


def _kmer_umap(hashes, categories, alphabet, out_png, title, *, kmer_k, min_share, cap,
               metric, seed, legend_title, category_labeler=None, alpha=None):
    """Shared tail: hashes -> cached 2-D UMAP coords -> umap_scatter (dropping unresolved hashes).
    Default legend label is '<cat>, <share>%, n=<sequences>'."""
    out_png = Path(out_png)
    xy, keep = _umap_coords(list(hashes), alphabet, kmer_k=kmer_k, seed=seed, metric=metric,
                            cache_dir=out_png.parent / '_umap_coords')
    cats = np.asarray(categories)[keep]
    if category_labeler is None:
        def category_labeler(c, n, sh):
            return f'{c}, {sh:.1%}, n={n:,}'
    return umap_scatter(xy, cats, out_png=out_png, title=title, min_share=min_share, cap=cap,
                        palette=_COLOR_PALETTE, metric=metric, seed=seed, legend_title=legend_title,
                        category_labeler=category_labeler, alpha=alpha)


def plot_single_side_clusters(clusters_root, threshold, short_name, out_dir, *, kmer_k=6,
                              min_share=0.01, cap=12, metric='euclidean', seed=42, alpha=None):
    """4a: k-mer UMAP of one slot's cluster sequences (points = sequences), colored by cluster."""
    tl = threshold if str(threshold).startswith('t') else threshold_label(threshold)
    clusters = pd.read_parquet(Path(clusters_root) / tl / f'{short_name}_cluster.parquet')
    alphabet = str(clusters['alphabet'].iloc[0])
    hcol = schema.SCHEMA[alphabet].hash_col
    out_png = Path(out_dir) / f'{short_name}_{tl}_{alphabet}_kmer_umap.png'
    n_clusters = int(clusters['cluster_id'].nunique())
    title = (f'{short_name} -- {alphabet} -- {tl} -- {kmer_k}-mer UMAP\n'
             f'{len(clusters):,} sequences; {n_clusters:,} clusters')
    stats = _kmer_umap(clusters[hcol].tolist(), clusters['cluster_id'].to_numpy(), alphabet, out_png,
                       title, kmer_k=kmer_k, min_share=min_share, cap=cap, metric=metric, seed=seed,
                       legend_title=f'clusters >= {min_share:.0%}', alpha=alpha)
    print(f'  single_side {short_name} {tl}: {stats["n_points"]:,} pts, {stats["n_selected"]} colored -> {out_png}')
    return out_png


def _plot_cc_sides(pw, uni, pair, out_dir, tag, tlabel, *, alphabet, kmer_k, min_share, cap, metric,
                   seed, color_by, cc_ids=None):
    """Shared 4b/4c body: for each slot draw the sequences of the selected CC(s), colored by
    `color_by` ('cluster' -> cluster_id; 'fragment' -> which CC/fragment). `cc_ids` (fragment mode)
    maps a cc_id to a rank label CC_1A/CC_1B..."""
    sa, sb = pair.split('-')
    outs = []
    for slot, sname in (('a', sa), ('b', sb)):
        side = _side_sequences(pw, uni, slot, alphabet)
        if color_by == 'cluster':
            cats = side['cluster_id'].to_numpy()
            n = int(side['cluster_id'].nunique())
            legend = f'clusters >= {min_share:.0%}'
            ms, cp, labeler = min_share, cap, None
            desc = f'{n:,} clusters'
        else:  # fragment
            rank = {cc: f'{sname}_frag{i + 1}' for i, cc in enumerate(cc_ids)}
            cats = side['cc_id'].map(rank).to_numpy()
            legend = 'fragment (edge-cut CC)'
            ms, cp, labeler = 0.0, len(cc_ids), None  # None -> _kmer_umap's '<cat>, <share>%, n=' label
            desc = f'{len(cc_ids)} fragments {list(cc_ids)}'
        out_png = Path(out_dir) / f'{tag}_{sname}_{pair}_{alphabet}_kmer_umap.png'
        title = (f'{pair} {tag} -- {sname}-side -- {alphabet} -- {tlabel} -- {kmer_k}-mer UMAP\n'
                 f'{len(side):,} sequences; {desc}')
        stats = _kmer_umap(side['seq_hash'].tolist(), cats, alphabet, out_png, title, kmer_k=kmer_k,
                           min_share=ms, cap=cp, metric=metric, seed=seed, legend_title=legend,
                           category_labeler=labeler)
        print(f'  {tag} {sname}: {stats["n_points"]:,} pts, {stats["n_selected"]} colored -> {out_png}')
        outs.append(out_png)
    return outs


def plot_megacc_sides(cc_dir, universe, out_dir, pair, *, alphabet='nt_cds', kmer_k=6,
                      min_share=0.01, cap=12, metric='euclidean', seed=42):
    """4b: each side of the natural mega-CC (the largest CC), colored by cluster."""
    pw = pd.read_parquet(Path(cc_dir) / 'pairs_with_cc.parquet')
    uni = pd.read_parquet(universe)
    mega = int(pw.groupby('cc_id').size().idxmax())
    return _plot_cc_sides(pw[pw['cc_id'] == mega], uni, pair, out_dir, f'megacc_cc{mega}',
                          Path(cc_dir).name, alphabet=alphabet, kmer_k=kmer_k, min_share=min_share,
                          cap=cap, metric=metric, seed=seed, color_by='cluster')


def plot_fragment_sides(frag_dir, universe, out_dir, pair, *, alphabet='nt_cds', kmer_k=6,
                        n_frags=2, min_cc_frac=0.0, metric='euclidean', seed=42):
    """4c: each side of the top fragments (edge-cut CCs), colored by fragment. Keeps the largest
    fragments carrying >= `min_cc_frac` of the fragmented pairs, capped at `n_frags`."""
    pw = pd.read_parquet(Path(frag_dir) / 'pairs_with_cc.parquet')
    uni = pd.read_parquet(universe)
    sizes = pw.groupby('cc_id').size().sort_values(ascending=False)
    top = sizes[sizes >= min_cc_frac * int(sizes.sum())].head(n_frags).index.tolist()
    return _plot_cc_sides(pw[pw['cc_id'].isin(top)], uni, pair, out_dir, 'fragments',
                          f'{Path(frag_dir).parent.name} fragmented', alphabet=alphabet,
                          kmer_k=kmer_k, min_share=0.0, cap=len(top), metric=metric, seed=seed,
                          color_by='fragment', cc_ids=top)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--mode', required=True, choices=['single_side', 'megacc', 'fragment'])
    p.add_argument('--alphabet', default='nt_cds', help='nt_cds / nt_ctg (default nt_cds)')
    p.add_argument('--kmer_k', type=int, default=6)
    p.add_argument('--min_share', type=float, default=0.01, help='min share of points for a distinct color')
    p.add_argument('--cap', type=int, default=12, help='max distinctly-colored categories')
    p.add_argument('--alpha', type=float, default=None,
                   help='point transparency 0..1 (default opaque); use e.g. 0.5 to reveal overplot/overlap')
    p.add_argument('--metric', default='euclidean', help='UMAP metric (default euclidean for k-mer SVD)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', type=Path, default=None)
    # single_side
    p.add_argument('--clusters_root', type=Path)
    p.add_argument('--threshold', help='tXXX (single_side)')
    p.add_argument('--short', help='slot short name, e.g. HA (single_side)')
    # megacc / fragment
    p.add_argument('--cc_dir', type=Path, help='tXXX (megacc) or tXXX/fragmented (fragment) artifact dir')
    p.add_argument('--universe', type=Path, help='pair_universe pairs.parquet')
    p.add_argument('--pair', help='slot pair, e.g. HA-NA')
    p.add_argument('--n_frags', type=int, default=2, help='fragment mode: how many top fragments to color')
    p.add_argument('--min_cc_frac', type=float, default=0.0,
                   help='fragment mode: keep only fragments with >= this share of pairs (default 0.0)')
    args = p.parse_args()

    if args.mode == 'single_side':
        out_dir = args.out_dir or (args.clusters_root / 'figures')
        plot_single_side_clusters(args.clusters_root, args.threshold, args.short, out_dir,
                                  kmer_k=args.kmer_k, min_share=args.min_share, cap=args.cap,
                                  metric=args.metric, seed=args.seed, alpha=args.alpha)
    elif args.mode == 'megacc':
        out_dir = args.out_dir or (args.cc_dir / 'figures')
        plot_megacc_sides(args.cc_dir, args.universe, out_dir, args.pair, alphabet=args.alphabet,
                          kmer_k=args.kmer_k, min_share=args.min_share, cap=args.cap,
                          metric=args.metric, seed=args.seed)
    else:
        out_dir = args.out_dir or (args.cc_dir / 'figures')
        plot_fragment_sides(args.cc_dir, args.universe, out_dir, args.pair, alphabet=args.alphabet,
                            kmer_k=args.kmer_k, n_frags=args.n_frags, min_cc_frac=args.min_cc_frac,
                            metric=args.metric, seed=args.seed)


if __name__ == '__main__':
    main()
