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

# metric key -> (y-axis label, fn(y_true, prob, y_pred) -> score).
_METRICS = {
    'aucpr':    ('AUC-PR',   lambda y, p, yh: average_precision_score(y, p)),
    'aucroc':   ('AUC-ROC',  lambda y, p, yh: roc_auc_score(y, p)),
    'f1_macro': ('F1 macro', lambda y, p, yh: f1_score(y, yh, average='macro')),
    'f1':       ('F1',       lambda y, p, yh: f1_score(y, yh)),
    'mcc':      ('MCC',      lambda y, p, yh: matthews_corrcoef(y, yh)),
}
# Wong colorblind-safe order (slot a / slot b == blue / vermillion, matching the prior figures).
_SERIES_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']


def _parse_series(items):
    out = []
    for it in items:
        if '=' not in it:
            raise SystemExit(f"--series item must be 'label=run_prefix'; got {it!r}")
        label, prefix = it.split('=', 1)
        out.append((label.strip(), prefix.strip()))
    return out


def series_curve(runs_root, prefix, thresholds, n_folds, fn):
    """(mean, min, max) of `fn` over folds at each threshold; NaN where no runs exist."""
    means, los, his = [], [], []
    for t in thresholds:
        vals = []
        for f in range(n_folds):
            p = Path(runs_root) / f'{prefix}_{t}_fold{f}' / 'test_predicted.csv'
            if not p.exists():
                continue
            d = pd.read_csv(p, usecols=['label', 'pred_prob', 'pred_label'], low_memory=False)
            vals.append(fn(d['label'], d['pred_prob'], d['pred_label']))
        a = np.array(vals) if vals else np.array([np.nan])
        means.append(a.mean()); los.append(a.min()); his.append(a.max())
    return np.array(means), np.array(los), np.array(his)


def main() -> None:
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
    args = ap.parse_args()

    setup_plot_style(use_seaborn_palette=False)
    ylab, fn = _METRICS[args.metric]
    series = _parse_series(args.series)
    tx = np.array([int(t[1:]) / 100 if t.startswith('t') and t[1:].isdigit() else float(i)
                   for i, t in enumerate(args.thresholds)])

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for i, (label, prefix) in enumerate(series):
        m, lo, hi = series_curve(args.runs_root, prefix, args.thresholds, args.n_folds, fn)
        c = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        ax.errorbar(tx, m, yerr=np.vstack([m - lo, hi - m]), fmt='-o', color=c, lw=2.2, ms=6,
                    capsize=4, elinewidth=1.3, capthick=1.3, label=label, zorder=3)
    if args.floor is not None:
        ax.axhline(args.floor, ls='--', color='#999999', lw=1.3,
                   label=f'chance floor ({args.floor:.2f})', zorder=2)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(ylab)
    ax.set_xticks(tx); ax.set_xticklabels([f'{v:.2f}' for v in tx]); ax.invert_xaxis()
    if args.ylim:
        ax.set_ylim(*args.ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(args.title, fontsize=11)
    savefig(args.out_png, dpi=180)
    print(f'wrote {args.out_png}')


if __name__ == '__main__':
    main()
