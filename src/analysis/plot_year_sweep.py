"""F1 macro per collection year for the H3N2 single-year random-split CV sweep.

Reads `metrics_summary.json` from each fold's baseline run dir and draws one x-position per year:
the individual fold scores as open dots plus a filled mean marker with a std error bar. With k=4
the raw folds carry information a symmetric error bar hides, so both are drawn.

Every year in the sweep is a separate population with its own dataset size, and size plausibly
drives part of the score difference, so the unique-positive-pair count is printed in a text box
rather than encoded as marker size (which no reader converts back to a number). The counts come
from the fold CSVs, not from a hard-coded table.

The y-axis spans 0.5-1.0 by default: roughly chance to perfect for balanced F1 macro, so a small
spread reads as small instead of being magnified by a cropped axis.

CLI:
    python -m src.analysis.plot_year_sweep \\
        --years 2022 2023 2024 2025 \\
        --out_png tmp/score/h3n2_year_sweep_f1_macro.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402

# Sampled from tmp/score/h3n2_f1_macro_within_fold.png so the two figures read as one series.
FOLD_COLOR = '#4C7CAB'
MEAN_COLOR = '#CF8793'
MARKER_EDGE = '#222222'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024, 2025],
                   help='collection years to plot, left to right')
    p.add_argument('--model_run_template', default='lgbm_ha_na_h3n2_{year}_random_cv4',
                   help='baseline run dir name minus the _fold{k} suffix')
    p.add_argument('--dataset_run_template', default='dataset_ha_na_h3n2_{year}_random_cv4',
                   help='dataset run dir name holding the fold_{k} dirs')
    p.add_argument('--models_root', type=Path, default=PROJ / 'models/flu/July_2025/runs')
    p.add_argument('--datasets_root', type=Path, default=PROJ / 'data/datasets/flu/July_2025/runs')
    p.add_argument('--n_folds', type=int, default=4)
    p.add_argument('--metric', default='f1_macro', help="key in metrics_summary.json['test']")
    p.add_argument('--metric_label', default='F1 macro', help='y-axis label')
    p.add_argument('--ylim', nargs=2, type=float, default=[0.5, 1.0],
                   help='y-axis limits; the default spans chance to perfect for a balanced metric')
    p.add_argument('--out_png', type=Path, default=PROJ / 'tmp/score/h3n2_year_sweep_f1_macro.png')
    p.add_argument('--figsize', nargs=2, type=float, default=[8.0, 5.65],
                   help='figure size in inches; the default matches h3n2_f1_macro_within_fold.png')
    p.add_argument('--dpi', type=int, default=150)
    args = p.parse_args()

    rows = []
    for year in args.years:
        model_stem = args.model_run_template.format(year=year)
        dataset_dir = args.datasets_root / args.dataset_run_template.format(year=year)
        scores, n_pairs = [], 0
        for k in range(args.n_folds):
            metrics_path = args.models_root / f'{model_stem}_fold{k}' / 'metrics_summary.json'
            if not metrics_path.exists():
                raise SystemExit(f"missing {metrics_path}")
            scores.append(float(json.loads(metrics_path.read_text())['test'][args.metric]))
            test_path = dataset_dir / f'fold_{k}' / 'test_pairs.parquet'
            if not test_path.exists():
                raise SystemExit(f"missing {test_path}")
            test = pd.read_parquet(test_path, columns=['label'])
            n_pairs += int((test['label'] == 1).sum())  # each positive is tested exactly once
        rows.append({'year': year, 'scores': scores, 'n_pairs': n_pairs})

    setup_plot_style()
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    x = list(range(len(rows)))

    for xi, r in zip(x, rows):
        s = pd.Series(r['scores'])
        ax.scatter([xi] * len(s), s, s=34, color=FOLD_COLOR, edgecolors=MARKER_EDGE,
                   linewidths=0.7, zorder=3)
        ax.errorbar(xi, s.mean(), yerr=s.std(ddof=1), fmt='o', markersize=7.5,
                    color=MEAN_COLOR, markeredgecolor=MARKER_EDGE, markeredgewidth=0.7,
                    ecolor=MARKER_EDGE, elinewidth=1.1, capsize=4, zorder=4)
        ax.annotate(f'{s.mean():.3f}', (xi, s.mean()), textcoords='offset points',
                    xytext=(11, -3), fontsize=9, color=MARKER_EDGE)

    ax.set_xticks(x)
    ax.set_xticklabels([str(r['year']) for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(*args.ylim)
    ax.set_xlabel('collection year (H3N2 isolates only)')
    ax.set_ylabel(args.metric_label)
    ax.set_title(
        f'HA-NA {args.metric_label} by collection year\n'
        f'H3N2, random split, {args.n_folds}-fold CV, within_fold negatives, '
        f'LGBM on nt_cds k-mers (k=6)')
    ax.grid(axis='y', alpha=0.3)

    size_lines = ['unique positive pairs'] + [f"  {r['year']}   {r['n_pairs']:,}" for r in rows]
    ax.text(0.02, 0.03, '\n'.join(size_lines), transform=ax.transAxes, fontsize=9,
            family='monospace', va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='0.7', alpha=0.9))

    handles = [
        plt.Line2D([], [], marker='o', linestyle='none', color=FOLD_COLOR,
                   markeredgecolor=MARKER_EDGE, markeredgewidth=0.7, markersize=6,
                   label='individual fold'),
        plt.Line2D([], [], marker='o', linestyle='none', color=MEAN_COLOR,
                   markeredgecolor=MARKER_EDGE, markeredgewidth=0.7, markersize=7,
                   label='mean ± std'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=9, framealpha=0.9)

    out = savefig(args.out_png, dpi=args.dpi)
    for r in rows:
        s = pd.Series(r['scores'])
        print(f"  {r['year']}: n={r['n_pairs']:,} pairs, {args.metric} "
              f"{s.mean():.4f} +/- {s.std(ddof=1):.4f}")
    print(f"Done. Wrote {out}")


if __name__ == '__main__':
    main()
