"""Cross-threshold CC-count trend of the cluster bigraph.

The 2D_cluster_sizes / 2D_cluster_maps / 2D_cluster_metadata views each show ONE
threshold at a time. This is the cross-t summary for one schema pair: how many
connected components (CCs) the cluster bigraph has at each identity threshold,
t099 -> t090, with the largest-CC pair share overlaid so the "many CCs but one
dominates" structure is visible in one view.

t100 is excluded by default: at 100% identity clustering is exact-sequence, n_CCs
is an order of magnitude larger (HA-NA 11,743; PB2-PB1 14,898) and dwarfs the
rest of the axis. Pass it explicitly via --thresholds to include it.

Note: n_CCs is NOT monotonic in t. The cluster bigraph's edge set (the pair
universe) is fixed across t; only the cluster labels change, and mmseqs2 clusters
each threshold independently (no refinement hierarchy), so a CC count can tick up
as t loosens (e.g., HA-NA 19 at t091 -> 20 at t090).

Reads the 2D_cluster_sizes feasibility CSV (the producer is
`bigraph_pair_feasibility.py`), which already carries `n_ccs_total`,
`n_pairs_total`, and per-rank `cc_pairs` (rank-1 = mega-CC) per
(schema_pair, alphabet, threshold). No bigraph rebuild needed.

CLI:
    python -m src.analysis.bigraph_cc_count_vs_threshold \\
        [--feasibility_csv results/flu/July_2025/runs/2D_cluster_sizes/cluster_pair_feasibility_top20.csv] \\
        [--alphabet aa] \\
        [--schema_pairs HA-NA PB2-PB1] \\
        [--thresholds t099 t098 ... t090] \\
        [--out_dir results/flu/July_2025/runs/2D_cluster_sizes]

Outputs (under --out_dir), one plot per schema pair:
    plots/cc_count_vs_threshold_{a}_{b}_{alphabet}.png
    cc_count_vs_threshold_{alphabet}.csv   tidy: schema_pair, alphabet, threshold,
                                           n_ccs, largest_cc_pairs, n_pairs,
                                           largest_cc_pct
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

_DEFAULT_CSV = PROJ / 'results/flu/July_2025/runs/2D_cluster_sizes/cluster_pair_feasibility_top20.csv'
_DEFAULT_PAIRS = ['HA-NA', 'PB2-PB1']
# t099..t090 — t100 excluded by default (see module docstring).
_DEFAULT_THRESHOLDS = [f't{i:03d}' for i in range(99, 89, -1)]

_BAR_COLOR = '#4c72b0'     # blue   — n_CCs bars (left axis)
_LINE_COLOR = '#ff7f0e'    # orange — largest-CC % (right axis)
_LINE_LABEL = '#9a4a00'    # darker orange for the % annotations (legible on blue)


def summarize(df: pd.DataFrame, alphabet: str, pair: str,
              thresholds: list[str]) -> pd.DataFrame:
    """Per-threshold n_CCs + largest-CC% for one (pair, alphabet), t-ordered.

    largest_cc_pairs is the rank-1 `cc_pairs` (the mega-CC); n_ccs_total and
    n_pairs_total are constant across ranks within a slice, so `first` is exact.
    """
    sub = df[(df['schema_pair'] == pair) & (df['alphabet'] == alphabet)]
    rows = []
    for t in thresholds:
        s = sub[sub['threshold'] == t]
        if s.empty:
            continue
        n_pairs = int(s['n_pairs_total'].iloc[0])
        largest = int(s['cc_pairs'].max())
        rows.append({
            'schema_pair': pair, 'alphabet': alphabet, 'threshold': t,
            'n_ccs': int(s['n_ccs_total'].iloc[0]),
            'largest_cc_pairs': largest, 'n_pairs': n_pairs,
            'largest_cc_pct': round(100.0 * largest / n_pairs, 2),
        })
    return pd.DataFrame(rows)


def plot_one(d: pd.DataFrame, *, pair: str, alphabet: str, out_png: Path) -> None:
    """n_CCs bars (log-y, left) + largest-CC% line (right) vs threshold, one pair.

    Args:
        d: summary DataFrame from `summarize`, t-ordered strict-first.
    """
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    x = np.arange(len(d))

    ax.bar(x, d['n_ccs'], color=_BAR_COLOR, edgecolor='black', linewidth=0.5,
           width=0.62, zorder=2)
    ax.set_yscale('log')
    ax.set_ylim(1, d['n_ccs'].max() * 2.4)
    for xi, v in zip(x, d['n_ccs']):
        ax.annotate(f'{int(v):,}', xy=(xi, v), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7.5, color='#222', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(d['threshold'], fontsize=8)
    ax.set_xlabel('cluster identity threshold', fontsize=9)
    ax.set_ylabel('number of CCs (log)', fontsize=9, color=_BAR_COLOR)
    ax.tick_params(axis='y', labelcolor=_BAR_COLOR)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(d) - 0.4)

    ax2 = ax.twinx()
    ax2.plot(x, d['largest_cc_pct'], color=_LINE_COLOR, marker='o', markersize=4.5,
             linewidth=1.8, zorder=5)
    for xi, v in zip(x, d['largest_cc_pct']):
        # to the RIGHT of each marker (not above/below) so the % labels never
        # collide with the centered bar-top n_CCs labels, wherever the line
        # crosses the bars.
        ax2.annotate(f'{v:.0f}%', xy=(xi, v), xytext=(7, 0),
                     textcoords='offset points', ha='left', va='center',
                     fontsize=6.5, color=_LINE_LABEL, zorder=6,
                     bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=0.4))
    ax2.set_ylim(0, 108)
    ax2.set_ylabel('largest-CC % of pairs', fontsize=9, color=_LINE_COLOR)
    ax2.tick_params(axis='y', labelcolor=_LINE_COLOR)

    ax.set_title(f'CC count vs identity threshold  ·  {pair} ({alphabet})', fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--feasibility_csv', type=Path, default=_DEFAULT_CSV,
                   help='2D_cluster_sizes feasibility CSV (carries n_ccs_total etc.).')
    p.add_argument('--alphabet', default='aa', choices=['aa', 'nt_cds'])
    p.add_argument('--schema_pairs', nargs='+', default=_DEFAULT_PAIRS)
    p.add_argument('--thresholds', nargs='+', default=_DEFAULT_THRESHOLDS,
                   help='Strict-first; default t099..t090 (t100 excluded — see docstring).')
    p.add_argument('--out_dir', type=Path,
                   default=PROJ / 'results/flu/July_2025/runs/2D_cluster_sizes')
    args = p.parse_args()

    csv = Path(args.feasibility_csv)
    if not csv.exists():
        raise FileNotFoundError(
            f'feasibility CSV not found: {csv}\n'
            f'Generate it first: python -m src.analysis.bigraph_pair_feasibility '
            f'--schema_pairs {" ".join(args.schema_pairs)} --alphabets {args.alphabet} '
            f'--thresholds {" ".join(args.thresholds)}')
    df = pd.read_csv(csv)

    out_dir = Path(args.out_dir)
    tidy_all = []
    for pair in args.schema_pairs:
        d = summarize(df, args.alphabet, pair, list(args.thresholds))
        if d.empty:
            print(f'WARNING: no rows for {pair} {args.alphabet} at the requested thresholds; skipping.')
            continue
        slug = pair.lower().replace('-', '_')
        out_png = out_dir / 'plots' / f'cc_count_vs_threshold_{slug}_{args.alphabet}.png'
        plot_one(d, pair=pair, alphabet=args.alphabet, out_png=out_png)
        print(f'wrote {out_png}')
        tidy_all.append(d)

    if tidy_all:
        tidy = pd.concat(tidy_all, ignore_index=True)
        out_csv = out_dir / f'cc_count_vs_threshold_{args.alphabet}.csv'
        tidy.to_csv(out_csv, index=False)
        print(f'wrote {out_csv} ({len(tidy)} rows)')
    print('\nDone.')


if __name__ == '__main__':
    main()
