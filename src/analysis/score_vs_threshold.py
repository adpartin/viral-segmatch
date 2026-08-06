"""Metric score vs mmseqs identity threshold for a Stage-4 experiment (ONE metric per figure).

For each `--series` (a run family `{prefix}_{tXXX}_fold{f}` under `--runs_root`), compute the chosen
metric per fold from `test_predicted.csv`, then plot its mean with min-max error bars across folds
vs the threshold t. One line per series; one metric per figure. The `--title` is caller-supplied so
it can name the schema pair / clustering / alphabet / model (e.g. "HA-NA, cm0, nt_cds, LGBM").

Series are given as `label=run_prefix`, so the same tool compares slots
(`"HA held out=lgbm_1dcd_cm0_slota" "NA held out=lgbm_1dcd_cm0_slotb"`) OR models
(`"LGBM=lgbm_..." "MLP=mlp_..."`). Optional `--floor` draws a single reference line (e.g. the
AUC-PR chance floor 0.50 at neg:pos = 1:1); omit it for metrics with no clean chance value.

CLI:
    python -m src.analysis.score_vs_threshold \\
        --series "HA held out=lgbm_1dcd_cm0_slota" "NA held out=lgbm_1dcd_cm0_slotb" \\
        --thresholds t099 t098 t097 t096 t095 --metric aucpr --floor 0.5 \\
        --title "AUC-PR -- HA-NA, cm0, nt_cds, LGBM" \\
        --out_png tmp/score/score_aucpr_1dcd_cm0.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402

# metric key -> (y-axis label, fn(y_true, pred_prob, pred_label) -> score).
_METRICS = {
    'aucpr':    ('AUC-PR',   lambda y_true, prob, pred: average_precision_score(y_true, prob)),
    'aucroc':   ('AUC-ROC',  lambda y_true, prob, pred: roc_auc_score(y_true, prob)),
    'f1_macro': ('F1 macro', lambda y_true, prob, pred: f1_score(y_true, pred, average='macro')),
    'f1':       ('F1',       lambda y_true, prob, pred: f1_score(y_true, pred)),
    'mcc':      ('MCC',      lambda y_true, prob, pred: matthews_corrcoef(y_true, pred)),
}
# Wong colorblind-safe order (slot a / slot b == blue / vermillion, matching the prior figures).
# Full 8-colour Wong set so up to 8 series get distinct colours (black + yellow appended; the
# first six are unchanged, so 2-series figures render identically).
_SERIES_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9',
                  '#000000', '#F0E442']


def _parse_series(items):
    """Split each `label=run_prefix` CLI item into its two parts.

    Args:
        items: Strings of the form 'label=run_prefix'.

    Returns:
        List of (label, run_prefix) tuples, both stripped of surrounding whitespace.
    """
    series = []
    for item in items:
        if '=' not in item:
            raise SystemExit(f"--series item must be 'label=run_prefix'; got {item!r}")
        label, run_prefix = item.split('=', 1)
        series.append((label.strip(), run_prefix.strip()))
    return series


def series_curve(runs_root, prefix, thresholds, n_folds, metric_fn):
    """Mean, min and max of one metric across folds, at each threshold.

    Scores each run dir `{prefix}_{threshold}_fold{f}` under `runs_root` from its
    `test_predicted.csv`. Folds with no run dir on disk are skipped; a threshold with no
    folds at all scores NaN and logs a warning.

    Args:
        runs_root: Directory holding the Stage-4 run dirs.
        prefix: Run-family prefix — the run dir name without its `_{threshold}_fold{f}` tail.
        thresholds: Threshold tokens as they appear in run dir names (e.g. 't099').
        n_folds: How many folds to look for at each threshold.
        metric_fn: Callable (y_true, pred_prob, pred_label) -> float.

    Returns:
        Three float arrays with one entry per threshold: the mean, the min and the max of
        `metric_fn` over the folds found.
    """
    means, mins, maxes = [], [], []
    for threshold in thresholds:
        fold_scores = []
        for fold in range(n_folds):
            run_dir = Path(runs_root) / f'{prefix}_{threshold}_fold{fold}'
            pred_csv = run_dir / 'test_predicted.csv'
            if not pred_csv.exists():
                continue
            preds = pd.read_csv(pred_csv, usecols=['label', 'pred_prob', 'pred_label'],
                                low_memory=False)
            score = metric_fn(preds['label'], preds['pred_prob'], preds['pred_label'])
            fold_scores.append(score)
        if not fold_scores:
            print(f'WARNING: no run dirs found for {prefix}_{threshold}_fold[0-{n_folds - 1}] '
                  f'under {runs_root}; this threshold plots as NaN')
            fold_scores = [np.nan]
        scores = np.array(fold_scores)
        means.append(scores.mean())
        mins.append(scores.min())
        maxes.append(scores.max())
    mean_curve = np.array(means)
    min_curve = np.array(mins)
    max_curve = np.array(maxes)
    return mean_curve, min_curve, max_curve


def main() -> None:
    """Parse the CLI, plot one metric vs threshold for every series, and write the PNG."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs_root', default=str(PROJ / 'models/flu/July_2025/runs'))
    ap.add_argument('--series', nargs='+', required=True, help="'label=run_prefix' (repeatable).")
    ap.add_argument('--thresholds', nargs='+', required=True, help='tXXX values, strict-first.')
    ap.add_argument('--n_folds', type=int, default=4)
    ap.add_argument('--metric', default='aucpr', choices=list(_METRICS))
    ap.add_argument('--title', required=True, help='Figure title -- name the pair / clustering / alphabet / model.')
    ap.add_argument('--out_png', required=True, type=Path)
    ap.add_argument('--floor', type=float, default=None,
                    help='Optional chance-floor reference line (e.g. 0.5 for AUC-PR at 1:1); omit for none.')
    ap.add_argument('--xlabel', default='mmseqs identity threshold  t')
    ap.add_argument('--ylim', nargs=2, type=float, default=None)
    ap.add_argument('--no_errorbars', action='store_true',
                    help='Plot mean lines only (no min-max error bars); clearer with many series.')
    args = ap.parse_args()

    setup_plot_style(use_seaborn_palette=False)
    ylabel, metric_fn = _METRICS[args.metric]
    series = _parse_series(args.series)

    # x position per threshold: the identity value carried by a `tXXX` token (t099 -> 0.99),
    # or the token's position in the list when it isn't in that form.
    x_positions = []
    for i, token in enumerate(args.thresholds):
        digits = token[1:]
        is_threshold_token = token.startswith('t') and digits.isdigit()
        x_positions.append(int(digits) / 100 if is_threshold_token else float(i))
    threshold_x = np.array(x_positions)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for i, (label, prefix) in enumerate(series):
        mean_curve, min_curve, max_curve = series_curve(
            args.runs_root, prefix, args.thresholds, args.n_folds, metric_fn)
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        if args.no_errorbars:
            ax.plot(threshold_x, mean_curve, '-o', color=color, lw=2.2, ms=6, label=label, zorder=3)
        else:
            spread = np.vstack([mean_curve - min_curve, max_curve - mean_curve])
            ax.errorbar(threshold_x, mean_curve, yerr=spread, fmt='-o', color=color, lw=2.2, ms=6,
                        capsize=4, elinewidth=1.3, capthick=1.3, label=label, zorder=3)
    if args.floor is not None:
        ax.axhline(args.floor, ls='--', color='#999999', lw=1.3,
                   label=f'chance floor ({args.floor:.2f})', zorder=2)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(threshold_x)
    ax.set_xticklabels([f'{v:.2f}' for v in threshold_x])
    ax.invert_xaxis()
    if args.ylim:
        ax.set_ylim(*args.ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(args.title, fontsize=11)
    savefig(args.out_png, dpi=180)
    print(f'wrote {args.out_png}')


if __name__ == '__main__':
    main()
