"""Per-slot cluster-size barplot (UNIQUE-WEIGHTED view) for one (protein, alphabet, t).

For a given (protein, alphabet, threshold), rank the protein's mmseqs clusters
by the number of *unique sequences* they hold and draw the top-N as a barplot.
Each bar is one cluster; bar height = unique-sequence count; the bar label shows
the raw count and its % of the slot's total unique sequences. The title carries
the protein, alphabet, threshold, and total unique sequences.

This is the **unique-weighted** cluster view of `docs/methods/clusters.md` §6.0 —
the slot's intrinsic sequence-space structure, before any pairing or isolate
weighting. It shows the "possibilities and limitations" of a slot: a few giant
clusters + a long singleton tail means whole-cluster routing has little freedom.

It is the sibling of the other two §6.0 weightings, kept as separate scripts:
  - records-weighted (isolates/cluster): `cluster_analysis_summary.py`
    (`compute_cluster_diversity_stats` — Gini / n_eff / top-1 %, stats only).
  - pair-weighted (positive pairs/cluster): `cluster_pair_weight_topk.py`
    (top-K + concentration curves; the constraint the splitter actually sees).
This script is the missing per-cluster *barplot* of the unique-weighted head;
`cluster_analysis_summary.py`'s `redundancy_stats.csv` already carries the
*summary stats* of the same distribution (largest_cluster, p90/p99/median,
fraction_singletons) but draws no per-cluster bars.

Reads the live Phase-2 cluster layout: `clusters_{aa,nt_cds}/tXXX/<PROT>_cluster.parquet`
(each row = one unique sequence; `cluster_id.value_counts()` is the unique-weighted
size). NOT the archived pre-Phase-2 `clusters_nt` / `id{XXX}` layout.

CLI:
    python -m src.analysis.cluster_size_barplot \\
        [--clusters_aa     data/processed/flu/July_2025/clusters_aa] \\
        [--clusters_nt_cds data/processed/flu/July_2025/clusters_nt_cds] \\
        [--out_dir         results/flu/July_2025/runs/1D_cluster_sizes] \\
        [--top_n 20] \\
        [--proteins HA NA PB2 PB1] \\
        [--alphabets aa nt_cds] \\
        [--thresholds t100 t099 t095 t090]

Defaults span the 8 major proteins × both alphabets × every tXXX on disk.

Outputs (under --out_dir):
    plots/barplot_{protein}_{alphabet}_{tXXX}.png   one barplot per slice
    cluster_size_top{N}.csv                         long-form top-N table, all
                                                    slices: protein, alphabet,
                                                    threshold, rank, cluster_id,
                                                    n_unique_in_cluster, pct,
                                                    cum_pct, n_unique_total,
                                                    n_clusters_total

The `protein` column contains the literal string 'NA' (Neuraminidase), so read
this CSV with `keep_default_na=False, na_values=['']` — a default `read_csv`
parses 'NA' as NaN and silently drops every Neuraminidase row (CLAUDE.md
Conventions, "Reading CSVs with function_short").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.config_hydra import load_function_metadata  # noqa: E402

# 8 ML-relevant majors, sourced from conf/virus/flu.yaml so the default protein
# set stays in sync with the rest of the pipeline.
_SHORT_ORDER = list(load_function_metadata(PROJ / 'conf' / 'virus' / 'flu.yaml').selected_short_names)
_FUNCTION_COLORS = {
    'PB2': '#1f77b4', 'PB1': '#ff7f0e', 'PA':  '#2ca02c', 'HA':  '#d62728',
    'NP':  '#9467bd', 'NA':  '#8c564b', 'M1':  '#e377c2', 'NS1': '#7f7f7f',
}
# alphabet key -> default cluster-root attr name (set in argparse).
_ALPHABETS = ('aa', 'nt_cds')


def _threshold_decimal(threshold_id: str) -> float:
    """'t095' -> 0.95. Threshold dirs are zero-padded tXXX (CLAUDE.md notation)."""
    return int(threshold_id[1:]) / 100.0


def _list_thresholds(root: Path) -> list[str]:
    """Sorted (loosest-last) tXXX subdir names present under a cluster root."""
    if not root.exists():
        return []
    names = [d.name for d in root.iterdir()
             if d.is_dir() and d.name.startswith('t') and d.name[1:].isdigit()]
    return sorted(names, reverse=True)  # t100 first


def cluster_sizes_unique(cluster_pq: Path) -> pd.Series:
    """Unique-weighted cluster sizes for one (protein, alphabet, t) parquet.

    Each parquet row is one unique sequence — keyed by `seq_hash` (protein hash)
    for aa clusters and `cds_dna_hash` (DNA hash) for nt_cds clusters — so
    `cluster_id.value_counts()` counts unique sequences per cluster regardless of
    the key column's name. Only `cluster_id` is read, so this is alphabet-
    agnostic. Returned descending (largest cluster first).
    """
    df = pd.read_parquet(cluster_pq, columns=['cluster_id'])
    return df['cluster_id'].value_counts()


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

    Args:
        sizes: cluster_id -> unique-seq count, descending (from cluster_sizes_unique).
        protein: short name (e.g. 'HA') — sets bar color and title.
        alphabet: 'aa' or 'nt_cds' — title/label only.
        threshold_id: 'tXXX' — title/label only.
        top_n: number of largest clusters to draw.
        out_png: output PNG path.
    """
    n_unique = int(sizes.sum())
    n_clusters = int(len(sizes))
    top = sizes.head(top_n)
    pcts = top.values / n_unique * 100.0
    top_cov = float(pcts.sum())
    color = _FUNCTION_COLORS.get(protein, '#1f77b4')

    fig, ax = plt.subplots(figsize=(max(9.0, len(top) * 0.55), 5.6))
    xs = np.arange(len(top))
    ax.bar(xs, top.values, color=color, edgecolor='black', linewidth=0.5)
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
        f'{protein} — {alphabet} — {threshold_id} (id={_threshold_decimal(threshold_id):.2f})\n'
        f'top {len(top)} of {n_clusters:,} clusters  ·  '
        f'total unique seqs: {n_unique:,}  ·  '
        f'top {len(top)} cover {top_cov:.1f}% of unique',
        fontsize=10,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--clusters_aa',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_aa'),
                   help='Root containing tXXX/{PROTEIN}_cluster.parquet for aa.')
    p.add_argument('--clusters_nt_cds',
                   default=str(PROJ / 'data/processed/flu/July_2025/clusters_nt_cds'),
                   help='Root containing tXXX/{PROTEIN}_cluster.parquet for nt_cds.')
    p.add_argument('--out_dir',
                   default=str(PROJ / 'results/flu/July_2025/runs/1D_cluster_sizes'),
                   help='Output directory for plots/ and the long-form CSV.')
    p.add_argument('--top_n', type=int, default=20,
                   help='Number of largest clusters to draw per slice (default 20).')
    p.add_argument('--proteins', nargs='*', default=None,
                   help='Protein short names (default: the 8 majors from flu.yaml).')
    p.add_argument('--alphabets', nargs='*', default=None, choices=list(_ALPHABETS),
                   help="Alphabets (default: aa nt_cds).")
    p.add_argument('--thresholds', nargs='*', default=None,
                   help='tXXX dirs (default: every tXXX present under each root).')
    args = p.parse_args()

    proteins = list(args.proteins) if args.proteins else list(_SHORT_ORDER)
    alphabets = list(args.alphabets) if args.alphabets else list(_ALPHABETS)
    roots = {'aa': Path(args.clusters_aa), 'nt_cds': Path(args.clusters_nt_cds)}

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict] = []
    n_plots = 0
    for alphabet in alphabets:
        root = roots[alphabet]
        thresholds = list(args.thresholds) if args.thresholds else _list_thresholds(root)
        if not thresholds:
            print(f'WARNING: no tXXX dirs under {root}; skipping alphabet={alphabet}.')
            continue
        for protein in proteins:
            for t in thresholds:
                cluster_pq = root / t / f'{protein}_cluster.parquet'
                if not cluster_pq.exists():
                    continue
                sizes = cluster_sizes_unique(cluster_pq)
                if len(sizes) == 0:
                    continue
                out_png = plots_dir / f'barplot_{protein.lower()}_{alphabet}_{t}.png'
                plot_cluster_size_barplot(
                    sizes, protein=protein, alphabet=alphabet, threshold_id=t,
                    top_n=args.top_n, out_png=out_png,
                )
                n_plots += 1
                print(f'wrote {out_png}')

                n_unique = int(sizes.sum())
                n_clusters = int(len(sizes))
                top = sizes.head(args.top_n)
                cum = 0.0
                for rank, (cid, sz) in enumerate(top.items(), start=1):
                    pct = sz / n_unique * 100.0
                    cum += pct
                    long_rows.append({
                        'protein': protein,
                        'alphabet': alphabet,
                        'threshold': t,
                        'rank': rank,
                        'cluster_id': cid,
                        'n_unique_in_cluster': int(sz),
                        'pct': round(pct, 4),
                        'cum_pct': round(cum, 4),
                        'n_unique_total': n_unique,
                        'n_clusters_total': n_clusters,
                    })

    if long_rows:
        long_df = pd.DataFrame(long_rows)
        long_csv = out_dir / f'cluster_size_top{args.top_n}.csv'
        long_df.to_csv(long_csv, index=False)
        print(f'\nwrote {long_csv} ({len(long_df):,} rows)')
    print(f'\nDone. {n_plots} barplot(s).')


if __name__ == '__main__':
    main()
