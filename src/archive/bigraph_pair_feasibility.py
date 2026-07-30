"""2D schema-pair feasibility barplot: connected-component sizes of the cluster bigraph.

The schema-pair analog of `cluster_size_barplot.py`. Where that script bars single
clusters by unique-sequence count for ONE slot, this bars **connected components
(CCs)** by **pair-count** for the slot PAIR — the joint co-occurrence structure.

For a given (schema_pair, alphabet, t): build the cluster-level bigraph (side A =
slot-a clusters, side B = slot-b clusters, an edge per positive pair) and draw the
top-N CCs by the number of positive pairs they carry. The **mega-CC's dominance is
the 2D-CD K-fold feasibility signal**: a cluster-disjoint K-fold needs every CC
(atom) to fit a fold, so the largest CC must drop to ~1/K of the pairs — a dashed
line marks the 1/K budget. When the largest CC towers over it (e.g. 98% at t095),
the split is infeasible without a cut.

Self-contained: reuses `load_pair_universe` + the membership-backed
`load_cluster_map` + `build_bipartite_multigraph`, then `nx.connected_components`.

CLI:
    python -m src.analysis.bigraph_pair_feasibility \\
        [--schema_pairs HA-NA PB2-PB1] \\
        [--alphabets aa nt_cds] \\
        [--thresholds t100 t099 t098 t097 t096 t095] \\
        [--top_n 20] [--k_folds 5] \\
        [--out_dir results/flu/July_2025/runs/2D_cluster_sizes]

Outputs (under --out_dir):
    barplot_{A}_{B}_{alphabet}_{tXXX}.png   one feasibility barplot per slice
    cluster_pair_feasibility_top{N}.csv     long-form: schema_pair, alphabet,
                                            threshold, rank, cc_pairs, pct, cum_pct,
                                            n_pairs_total, n_cells_total, n_ccs_total
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.bigraph_properties import build_bipartite_multigraph, load_cluster_map  # noqa: E402
from src.analysis.cluster_pair_weight_topk import load_pair_universe  # noqa: E402
from src.utils.clustering_utils import threshold_decimal  # noqa: E402

_DEFAULT_PAIRS = ['HA-NA', 'PB2-PB1']
_DEFAULT_ALPHABETS = ['aa', 'nt_cds']
_DEFAULT_THRESHOLDS = ['t100', 't099', 't098', 't097', 't096', 't095']
_ROOT = {'aa': PROJ / 'data/processed/flu/July_2025/clusters_aa',
         'nt_cds': PROJ / 'data/processed/flu/July_2025/clusters_nt_cds'}


def cc_pair_sizes(G: nx.MultiGraph) -> tuple[list[int], int]:
    """Per-CC positive-pair counts (descending) and the number of CCs.

    Each multigraph edge is one positive pair, so a CC's edge count is its pair
    mass. Counted in one O(E) pass via a node->CC map.
    """
    comps = list(nx.connected_components(G))
    node_cc = {n: i for i, c in enumerate(comps) for n in c}
    cnt: Counter = Counter()
    for u, v in G.edges():
        cnt[node_cc[u]] += 1
    return sorted(cnt.values(), reverse=True), len(comps)


def plot_feasibility_barplot(
    sizes: list[int],
    *,
    pair_label: str,
    alphabet: str,
    threshold_id: str,
    top_n: int,
    n_ccs: int,
    n_pairs: int,
    k_folds: int,
    out_png: Path,
) -> None:
    top = sizes[:top_n]
    pcts = [s / n_pairs * 100.0 for s in top]
    largest_pct = sizes[0] / n_pairs * 100.0
    budget_pairs = n_pairs / k_folds
    budget_pct = 100.0 / k_folds

    fig, ax = plt.subplots(figsize=(max(9.0, len(top) * 0.55), 5.8))
    xs = np.arange(len(top))
    colors = ['#d62728' if i == 0 else '#4c72b0' for i in range(len(top))]  # mega-CC red
    ax.bar(xs, top, color=colors, edgecolor='black', linewidth=0.5)
    for x, s, p in zip(xs, top, pcts):
        ax.annotate(f'{int(s):,}\n{p:.1f}%', xy=(x, s), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7, color='#222')

    ax.axhline(budget_pairs, linestyle='--', color='#555', linewidth=1.2,
               label=f'1/{k_folds} K-fold budget ({budget_pct:.0f}% of pairs)')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'CC{i + 1}' for i in range(len(top))], rotation=0, fontsize=7)
    ax.set_xlabel('connected component (rank-ordered, largest first)', fontsize=9)
    ax.set_ylabel('positive pairs in CC', fontsize=9)
    ax.set_ylim(0, top[0] * 1.18)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9)

    ax.set_title(
        f'{pair_label} — {alphabet} — {threshold_id} (id={threshold_decimal(threshold_id):.2f})\n'
        f'top {len(top)} of {n_ccs:,} CCs  ·  {n_pairs:,} pairs  ·  '
        f'largest CC {largest_pct:.1f}%',
        fontsize=10,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cds_final',
                   default=str(PROJ / 'data/processed/flu/July_2025/cds_dna_final.parquet'))
    p.add_argument('--schema_pairs', nargs='+', default=_DEFAULT_PAIRS,
                   help="Pairs as A-B (default: HA-NA PB2-PB1).")
    p.add_argument('--alphabets', nargs='+', default=_DEFAULT_ALPHABETS, choices=['aa', 'nt_cds'])
    p.add_argument('--thresholds', nargs='+', default=_DEFAULT_THRESHOLDS)
    p.add_argument('--top_n', type=int, default=20)
    p.add_argument('--k_folds', type=int, default=5,
                   help='K for the 1/K feasibility budget line (default 5).')
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/2D_cluster_sizes')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    universe_cache: dict[str, pd.DataFrame] = {}
    long_rows: list[dict] = []
    n_plots = 0
    for pair in args.schema_pairs:
        slot_a, slot_b = pair.split('-')
        if pair not in universe_cache:
            universe_cache[pair] = load_pair_universe(Path(args.cds_final), slot_a, slot_b)
        universe = universe_cache[pair]
        for alphabet in args.alphabets:
            for t in args.thresholds:
                cmap_a = load_cluster_map(_ROOT[alphabet], slot_a, t)
                cmap_b = load_cluster_map(_ROOT[alphabet], slot_b, t)
                if not cmap_a or not cmap_b:
                    print(f"  [{pair} {alphabet} {t}] missing cluster map; skipping.")
                    continue
                G, n_unmapped = build_bipartite_multigraph(universe, cmap_a, cmap_b, alphabet)
                if n_unmapped:
                    print(f"  WARNING: {pair} {alphabet} {t} dropped {n_unmapped} unmapped pairs.")
                n_pairs = G.number_of_edges()
                n_cells = len(set(G.edges()))
                sizes, n_ccs = cc_pair_sizes(G)

                slug = f'{slot_a.lower()}_{slot_b.lower()}'
                out_png = plots_dir / f'barplot_{slug}_{alphabet}_{t}.png'
                plot_feasibility_barplot(
                    sizes, pair_label=pair, alphabet=alphabet, threshold_id=t,
                    top_n=args.top_n, n_ccs=n_ccs, n_pairs=n_pairs, k_folds=args.k_folds,
                    out_png=out_png)
                n_plots += 1
                print(f"  [{pair} {alphabet} {t}] {n_pairs:,} pairs, {n_cells:,} cells, "
                      f"{n_ccs:,} CCs, largest {sizes[0] / n_pairs:.1%}; wrote {out_png.name}")

                cum = 0.0
                for rank, s in enumerate(sizes[:args.top_n], start=1):
                    pct = s / n_pairs * 100.0
                    cum += pct
                    long_rows.append({
                        'schema_pair': pair, 'alphabet': alphabet, 'threshold': t,
                        'rank': rank, 'cc_pairs': int(s), 'pct': round(pct, 4),
                        'cum_pct': round(cum, 4), 'n_pairs_total': int(n_pairs),
                        'n_cells_total': int(n_cells), 'n_ccs_total': int(n_ccs),
                    })

    if long_rows:
        csv = out_dir / f'cluster_pair_feasibility_top{args.top_n}.csv'
        pd.DataFrame(long_rows).to_csv(csv, index=False)
        print(f"\nwrote {csv} ({len(long_rows):,} rows)")
    print(f"\nDone. {n_plots} barplot(s).")


if __name__ == '__main__':
    main()
