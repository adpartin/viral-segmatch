"""Aggregate an nt_cds within_fold LGBM cluster-threshold sweep.

Reads the per-(threshold, fold) LGBM runs produced by the sweep, computes
per-threshold mean +/- std over the 5 folds for the test split, and writes a
summary CSV plus a metric-vs-threshold plot with the #atoms (CC) count overlaid
so the m=1 size confound stays visible.

Run dirs:      <RUNS>/<run_prefix>_<t>_fold<k>
Dataset dirs:  <DATASETS>/<dataset_prefix>_<t>/fold_0/dataset_stats.json
Output:        results/flu/July_2025/runs/<out_name>/

Test-split metrics per fold:
  - metrics_summary.json: auc_roc, f1, f1_macro, precision, recall
  - test_predicted.csv:   MCC (label vs pred_label) and AUC-PR
                          (average_precision over label vs pred_prob)

#atoms(t) = train_pos + val_pos + test_pos from fold_0 dataset_stats.json (at
m=1 each atom contributes exactly one positive, partitioned across the splits).

Usage:
    # drop=true sweep (defaults)
    python -m src.analysis.aggregate_cc_threshold_sweep
    # drop=false sweep
    python -m src.analysis.aggregate_cc_threshold_sweep \
        --run_prefix cc_ntcds_wf_drop0_concat \
        --dataset_prefix dataset_cc_nt_cds_wf_drop0 \
        --out_name cc_ntcds_wf_drop0_threshold_sweep
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, average_precision_score

PROJ = Path(__file__).resolve().parents[2]
RUNS = PROJ / 'models' / 'flu' / 'July_2025' / 'runs'
DATASETS = PROJ / 'data' / 'datasets' / 'flu' / 'July_2025' / 'runs'
RESULTS = PROJ / 'results' / 'flu' / 'July_2025' / 'runs'

FOLDS = [0, 1, 2, 3, 4]
METRIC_KEYS = ['auc_roc', 'auc_pr', 'f1', 'f1_macro', 'mcc', 'precision', 'recall']


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run_prefix', default='cc_ntcds_wf_concat',
                   help='Run-dir prefix; run = <run_prefix>_<t>_fold<k>.')
    p.add_argument('--dataset_prefix', default='dataset_cc_nt_cds_wf',
                   help='Dataset-dir prefix; dataset = <dataset_prefix>_<t>.')
    p.add_argument('--out_name', default='cc_ntcds_wf_threshold_sweep',
                   help='Output subdir under results/flu/July_2025/runs/.')
    p.add_argument('--thresholds', nargs='+',
                   default=['t099', 't098', 't097', 't096', 't095'])
    return p.parse_args()


def _read_fold(run_prefix: str, t: str, fold: int) -> dict:
    run = RUNS / f'{run_prefix}_{t}_fold{fold}'
    summary = json.loads((run / 'metrics_summary.json').read_text())['test']
    pred = pd.read_csv(run / 'test_predicted.csv')
    y, yhat, prob = pred['label'], pred['pred_label'], pred['pred_prob']
    return {
        'auc_roc': summary['auc_roc'],
        'auc_pr': average_precision_score(y, prob),
        'f1': summary['f1'],
        'f1_macro': summary['f1_macro'],
        'mcc': matthews_corrcoef(y, yhat),
        'precision': summary['precision'],
        'recall': summary['recall'],
    }


def _dataset_size(dataset_prefix: str, t: str) -> dict:
    stats = json.loads((DATASETS / f'{dataset_prefix}_{t}' / 'fold_0'
                        / 'dataset_stats.json').read_text())
    # At m=1 each atom (CC) contributes exactly one positive, so the atom count
    # is the positives summed across the three splits.
    stats['n_atoms'] = stats['train_pos'] + stats['val_pos'] + stats['test_pos']
    return stats


def main() -> None:
    args = _parse_args()
    out = RESULTS / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    per_fold_rows, summary_rows = [], []

    for t in args.thresholds:
        folds = [_read_fold(args.run_prefix, t, k) for k in FOLDS]
        for k, fm in zip(FOLDS, folds):
            per_fold_rows.append({'threshold': t, 'fold': k, **fm})

        row = {'threshold': t}
        for m in METRIC_KEYS:
            vals = np.array([fm[m] for fm in folds], dtype=float)
            row[f'{m}_mean'] = vals.mean()
            row[f'{m}_std'] = vals.std(ddof=1)
        size = _dataset_size(args.dataset_prefix, t)
        row['n_atoms'] = size['n_atoms']
        row['n_train'] = size['train_pairs']
        row['n_test'] = size['test_pairs']
        summary_rows.append(row)

    per_fold = pd.DataFrame(per_fold_rows)
    summary = pd.DataFrame(summary_rows)
    per_fold.to_csv(out / 'per_fold_metrics.csv', index=False)
    summary.to_csv(out / 'threshold_sweep_summary.csv', index=False)

    # Plot: test metrics (AUC-ROC, F1 macro) vs 2D cluster id, mean +/- std over
    # the 5 folds.
    x = np.arange(len(args.thresholds))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m, color, lbl in [('auc_roc', 'C0', 'AUC-ROC'), ('f1_macro', 'C1', 'F1 macro')]:
        ax.errorbar(x, summary[f'{m}_mean'], yerr=summary[f'{m}_std'],
                    marker='o', capsize=4, label=lbl, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(args.thresholds)
    ax.set_xlabel('2D cluster id')
    ax.set_ylabel('Test Set Metric (5-folds CV)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')

    ax.set_title('nt_cds LGBM (concat, m=1): metric vs cluster threshold')
    fig.tight_layout()
    fig.savefig(out / 'threshold_sweep_auc.png', dpi=150)

    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 30)
    cols = ['threshold', 'n_atoms', 'n_train', 'n_test', 'auc_roc_mean', 'auc_roc_std',
            'auc_pr_mean', 'f1_mean', 'mcc_mean', 'mcc_std']
    print(summary[cols].to_string(index=False))
    print(f'\nWrote: {out}')


if __name__ == '__main__':
    main()
