"""Metric score vs mmseqs identity threshold for a Stage-4 experiment (ONE metric per figure).

For each `--series`, compute the chosen metric per fold from `test_predicted.csv`, then plot its
mean with min-max error bars across folds vs the threshold t. One line per series; one metric per
figure. The `--title` is caller-supplied so it can name the schema pair / clustering / alphabet /
model (e.g. "HA-NA, cm0, nt_cds, LGBM").

A series is `label=run_pattern`, where the pattern names the run dirs under `--runs_root` with
`{t}` and `{fold}` placeholders. Spelling the whole dir name out lets series differ anywhere, not
only in a leading prefix -- the 2D-CD and random arms differ by a suffix sitting between the
threshold and the fold. Optional `--floor` draws a single reference line (e.g. the AUC-PR chance
floor 0.50 at neg:pos = 1:1); omit it for metrics with no clean chance value.

CLI:
    python -m src.analysis.score_vs_threshold \\
        --series "2D-CD=lgbm_cc_nt_cds_cm0_{t}_fold{fold}" \\
                 "random=lgbm_cc_nt_cds_cm0_{t}_random_fold{fold}" \\
        --thresholds t099 t098 t097 --metric f1_macro \\
        --title "F1 macro -- HA-NA, cm0, nt_cds, LGBM" \\
        --out_png tmp/score/score_f1_macro_2dcd_vs_random.png
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
# Paul Tol's high-contrast triple first (deep blue / muted red / amber), then his muted set, so
# two-series figures get the most legible pair and up to eight series stay colorblind-safe.
_SERIES_COLORS = ['#004488', '#BB5566', '#DDAA33', '#332288', '#117733', '#88CCEE',
                  '#882255', '#999933']


def _parse_series(items):
    """Split each `label=run_pattern` CLI item into its two parts.

    Args:
        items: Strings of the form 'label=run_pattern', the pattern carrying `{t}` and `{fold}`.

    Returns:
        List of (label, run_pattern) tuples, both stripped of surrounding whitespace.

    Raises:
        SystemExit: on a missing '=', or a pattern lacking either placeholder -- without them
            every threshold and fold would resolve to the same run dir and the curve would be flat
            rather than empty, which is the harder mistake to notice.
    """
    series = []
    for item in items:
        if '=' not in item:
            raise SystemExit(f"--series item must be 'label=run_pattern'; got {item!r}")
        label, run_pattern = item.split('=', 1)
        label, run_pattern = label.strip(), run_pattern.strip()
        missing = [p for p in ('{t}', '{fold}') if p not in run_pattern]
        if missing:
            raise SystemExit(f'--series pattern {run_pattern!r} is missing {" and ".join(missing)}')
        series.append((label, run_pattern))
    return series


def series_curve(runs_root, run_pattern, thresholds, n_folds, metric_fn):
    """Mean, min and max of one metric across folds, at each threshold.

    Scores the run dir `run_pattern.format(t=..., fold=...)` under `runs_root` from its
    `test_predicted.csv`. Folds with no run dir on disk are skipped; a threshold with no
    folds at all scores NaN and logs a warning.

    Args:
        runs_root: Directory holding the Stage-4 run dirs.
        run_pattern: Run dir name with `{t}` and `{fold}` placeholders.
        thresholds: Threshold tokens as they appear in run dir names (e.g. 't099').
        n_folds: How many folds to look for at each threshold.
        metric_fn: Callable (y_true, pred_prob, pred_label) -> float.

    Returns:
        `(mean_curve, per_threshold_scores)`: the mean per threshold, and the list of raw fold
        scores behind each -- summarising is the caller's, so the spread it draws and the mean it
        plots come from one place.
    """
    means, per_threshold = [], []
    for th in thresholds:
        fold_scores = []
        for fold in range(n_folds):
            run_dir = Path(runs_root) / run_pattern.format(t=th, fold=fold)
            pred_csv = run_dir / 'test_predicted.csv'
            if not pred_csv.exists():
                continue
            preds = pd.read_csv(pred_csv, usecols=['label', 'pred_prob', 'pred_label'],
                                low_memory=False)
            score = metric_fn(preds['label'], preds['pred_prob'], preds['pred_label'])
            fold_scores.append(score)
        if not fold_scores:
            print(f'WARNING: no run dirs found for '
                  f'{run_pattern.format(t=th, fold=f"[0-{n_folds - 1}]")} '
                  f'under {runs_root}; this threshold plots as NaN')
            fold_scores = [np.nan]
        scores = np.array(fold_scores)
        means.append(scores.mean())
        per_threshold.append(scores)
    mean_curve = np.array(means)
    return mean_curve, per_threshold


def main() -> None:
    """Parse the CLI, plot one metric vs threshold for every series, and write the PNG."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs_root', default=str(PROJ / 'models/flu/July_2025/runs'))
    ap.add_argument('--series', nargs='+', required=True,
                    help="'label=run_pattern' with {t} and {fold} placeholders (repeatable).")
    ap.add_argument('--thresholds', nargs='+', required=True, help='tXXX values, strict-first.')
    ap.add_argument('--n_folds', type=int, default=4)
    ap.add_argument('--metric', default='aucpr', choices=list(_METRICS))
    ap.add_argument('--title', required=True, help='Figure title -- name the pair / clustering / alphabet / model.')
    ap.add_argument('--out_png', required=True, type=Path)
    ap.add_argument('--floor', type=float, default=None,
                    help='Optional chance-floor reference line (e.g. 0.5 for AUC-PR at 1:1); omit for none.')
    ap.add_argument('--xlabel', default='MMseqs identity threshold  t')
    ap.add_argument('--xlim', nargs=2, type=float, default=None,
                    help='x range as given, e.g. 1.00 0.96; the order sets the direction.')
    ap.add_argument('--ylim', nargs=2, type=float, default=None)
    ap.add_argument('--marker_size', type=float, default=5.0,
                    help='point size (default 5); smaller lets a tight error bar show.')
    ap.add_argument('--spread', choices=('folds', 'errorbar', 'none'), default='folds',
                    help="How to show fold-to-fold variation: 'folds' scatters every fold's score "
                         "(default), 'errorbar' draws min-max bars, 'none' plots means only.")
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
    # Offset each series slightly on x so their fold points do not overlap at a shared threshold.
    x_span = abs(threshold_x.max() - threshold_x.min()) or 1.0
    for i, (label, run_pattern) in enumerate(series):
        mean_curve, per_threshold = series_curve(
            args.runs_root, run_pattern, args.thresholds, args.n_folds, metric_fn)
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        offset = (i - (len(series) - 1) / 2) * 0.012 * x_span
        x = threshold_x + offset
        if args.spread == 'folds':
            # Every fold as its own point and no summary line: with four folds the eye averages
            # them, and a mean would hide structure like t097's two-low/two-high split.
            xs = np.repeat(x, [len(scores) for scores in per_threshold])
            ys = np.concatenate(per_threshold)
            ax.scatter(xs, ys, s=55, facecolors=color, alpha=0.7, edgecolors='black',
                       linewidths=0.8, label=label, zorder=3)
            continue
        ax.plot(x, mean_curve, '-o', color=color, lw=2.2, ms=args.marker_size,
                label=label, zorder=3)
        if args.spread == 'errorbar':
            lows = np.array([s.min() for s in per_threshold])
            highs = np.array([s.max() for s in per_threshold])
            ax.errorbar(x, mean_curve, yerr=np.vstack([mean_curve - lows, highs - mean_curve]),
                        fmt='none', ecolor=color, capsize=4, elinewidth=1.3, capthick=1.3, zorder=3)
    if args.floor is not None:
        ax.axhline(args.floor, ls='--', color='#999999', lw=1.3,
                   label=f'chance floor ({args.floor:.2f})', zorder=2)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(threshold_x)
    ax.set_xticklabels([f'{v:.2f}' for v in threshold_x])
    # --xlim states the direction itself (1.00 -> 0.96), so inverting is only for the default
    # range, where strict-first thresholds should still read left to right.
    if args.xlim:
        ax.set_xlim(*args.xlim)
    else:
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
