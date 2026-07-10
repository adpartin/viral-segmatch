"""1-D cluster count vs identity threshold, per major protein.

The 1-D analog of `bigraph_cc_count_vs_threshold` (which is 2-D / per pair). For
each major protein, count its mmseqs clusters at each identity threshold and draw
a small-multiples grid — one barplot per protein — so the per-protein 1-D
granularity trend across t is one figure. Complements the per-(protein, t)
size barplots in `cluster_size_barplot.py`.

Like the 2-D CC count, n_clusters is NOT monotonic in t (mmseqs clusters each
threshold independently, no refinement hierarchy; e.g. NA 791 at t091 -> 1,077
at t090).

Reads the live cluster parquets `clusters_{aa,nt_cds}/tXXX/<PROT>_cluster.parquet`
(one row per unique sequence; n_clusters = `cluster_id.nunique()`).

CLI:
    python -m src.analysis.cluster_count_vs_threshold \\
        [--clusters_aa ...] [--clusters_nt_cds ...] [--alphabet aa] \\
        [--proteins HA NA ...] [--thresholds t099 ... t090] \\
        [--out_dir results/flu/July_2025/runs/1D_cluster_sizes]

t100 is excluded by default (exact-sequence clustering dwarfs the rest of the
axis); pass it explicitly via --thresholds to include it.

Outputs (under --out_dir):
    plots/cluster_count_vs_threshold_{alphabet}.png   8-panel small multiples
    cluster_count_vs_threshold_{alphabet}.csv         protein, threshold, n_clusters
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # headless backend; must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.cluster_size_barplot import _SHORT_ORDER  # noqa: E402
from src.utils.clustering_utils import cluster_sizes_unique  # noqa: E402
from src.utils.plot_config import get_protein_color  # noqa: E402

_DEFAULT_THRESHOLDS = [f't{i:03d}' for i in range(99, 89, -1)]  # t099..t090
_ROOT = {'aa': PROJ / 'data/processed/flu/July_2025/clusters_aa',
         'nt_cds': PROJ / 'data/processed/flu/July_2025/clusters_nt_cds'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--clusters_aa', default=str(_ROOT['aa']))
    p.add_argument('--clusters_nt_cds', default=str(_ROOT['nt_cds']))
    p.add_argument('--alphabet', default='aa', choices=['aa', 'nt_cds'])
    p.add_argument('--proteins', nargs='*', default=None,
                   help='Protein short names (default: the 8 majors from flu.yaml).')
    p.add_argument('--thresholds', nargs='+', default=_DEFAULT_THRESHOLDS,
                   help='Strict-first; default t099..t090 (t100 excluded).')
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/1D_cluster_sizes')
    args = p.parse_args()

    proteins = list(args.proteins) if args.proteins else list(_SHORT_ORDER)
    root = Path(args.clusters_aa if args.alphabet == 'aa' else args.clusters_nt_cds)
    thresholds = list(args.thresholds)

    rows = []
    counts: dict[str, dict[str, int]] = {}
    for prot in proteins:
        counts[prot] = {}
        for t in thresholds:
            pq = root / t / f'{prot}_cluster.parquet'
            if not pq.exists():
                continue
            n = int(len(cluster_sizes_unique(pq)))  # n unique clusters
            counts[prot][t] = n
            rows.append({'protein': prot, 'alphabet': args.alphabet,
                         'threshold': t, 'n_clusters': n})

    ncol = 4
    nrow = math.ceil(len(proteins) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    x = np.arange(len(thresholds))
    for i, prot in enumerate(proteins):
        ax = axes[i // ncol][i % ncol]
        ys = [counts[prot].get(t, 0) for t in thresholds]
        color = get_protein_color(prot)
        ax.bar(x, ys, color=color, edgecolor='black', linewidth=0.4, width=0.7, zorder=2)
        ax.set_yscale('log')
        ymax = max([y for y in ys if y > 0] + [1])
        ax.set_ylim(1, ymax * 2.6)
        for xi, y in zip(x, ys):
            if y > 0:
                ax.annotate(f'{y:,}', xy=(xi, y), xytext=(0, 2),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=6, color='#222')
        ax.set_xticks(x)
        ax.set_xticklabels(thresholds, fontsize=6, rotation=45, ha='right')
        ax.set_title(prot, fontsize=11, color=color)
        ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        if i % ncol == 0:
            ax.set_ylabel('n clusters (log)', fontsize=8)
    for j in range(len(proteins), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')

    fig.suptitle(f'1-D cluster count vs identity threshold  ·  {args.alphabet}',
                 fontsize=12, y=1.0)
    fig.supxlabel('cluster identity threshold', fontsize=9)
    fig.tight_layout()
    out_png = Path(args.out_dir) / 'plots' / f'cluster_count_vs_threshold_{args.alphabet}.png'
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')

    csv = Path(args.out_dir) / f'cluster_count_vs_threshold_{args.alphabet}.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f'wrote {csv} ({len(rows)} rows)')
    print('Done.')


if __name__ == '__main__':
    main()
