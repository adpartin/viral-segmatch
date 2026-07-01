"""Aggregate the nt_cds within_fold LGBM cluster-threshold sweep.

Reads the per-(threshold, fold) LGBM runs produced by the sweep
(``cc_ntcds_wf_concat_t0XX_foldK``), computes per-threshold mean +/- std
over the 5 folds for the test split, and writes a summary CSV plus an
AUC-vs-threshold plot.

Test-split metrics come from two sources per fold:
  - metrics_summary.json: auc_roc, f1, f1_macro, precision, recall
  - test_predicted.csv:   MCC (label vs pred_label) and AUC-PR
                          (average_precision over label vs pred_prob)

Dataset size per threshold (n_train / n_test, fold 0) is read from the
fold's dataset_stats.json so the m=1 size confound stays visible.

Usage:
    python -m src.analysis.aggregate_cc_threshold_sweep
"""
from pathlib import Path
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
OUT = PROJ / 'results' / 'flu' / 'July_2025' / 'runs' / 'cc_ntcds_wf_threshold_sweep'

THRESHOLDS = ['t099', 't098', 't097', 't096', 't095']
FOLDS = [0, 1, 2, 3, 4]
METRIC_KEYS = ['auc_roc', 'auc_pr', 'f1', 'f1_macro', 'mcc', 'precision', 'recall']


def _read_fold(t: str, fold: int) -> dict:
    run = RUNS / f'cc_ntcds_wf_concat_{t}_fold{fold}'
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


def _dataset_size(t: str) -> dict:
    stats = json.loads((DATASETS / f'dataset_cc_nt_cds_wf_{t}' / 'fold_0'
                        / 'dataset_stats.json').read_text())
    return stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_fold_rows, summary_rows = [], []

    for t in THRESHOLDS:
        folds = [_read_fold(t, k) for k in FOLDS]
        for k, fm in zip(FOLDS, folds):
            per_fold_rows.append({'threshold': t, 'fold': k, **fm})

        row = {'threshold': t}
        for m in METRIC_KEYS:
            vals = np.array([fm[m] for fm in folds], dtype=float)
            row[f'{m}_mean'] = vals.mean()
            row[f'{m}_std'] = vals.std(ddof=1)
        size = _dataset_size(t)
        row['n_train'] = size['train_pairs']
        row['n_test'] = size['test_pairs']
        summary_rows.append(row)

    per_fold = pd.DataFrame(per_fold_rows)
    summary = pd.DataFrame(summary_rows)
    per_fold.to_csv(OUT / 'per_fold_metrics.csv', index=False)
    summary.to_csv(OUT / 'threshold_sweep_summary.csv', index=False)

    # Plot: test AUC-ROC and F1 vs threshold (mean +/- std).
    x = np.arange(len(THRESHOLDS))
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, color in [('auc_roc', 'C0'), ('f1', 'C1'), ('mcc', 'C2')]:
        ax.errorbar(x, summary[f'{m}_mean'], yerr=summary[f'{m}_std'],
                    marker='o', capsize=4, label=m.replace('_', '-').upper(), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n(n={int(n)})' for t, n in zip(THRESHOLDS, summary['n_train'])])
    ax.set_xlabel('cluster identity threshold (n_train per fold)')
    ax.set_ylabel('test metric (mean +/- std over 5 folds)')
    ax.set_title('nt_cds within_fold LGBM (concat, m=1): metric vs cluster threshold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'threshold_sweep_auc.png', dpi=150)

    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 30)
    cols = ['threshold', 'n_train', 'n_test', 'auc_roc_mean', 'auc_roc_std',
            'auc_pr_mean', 'f1_mean', 'mcc_mean', 'mcc_std']
    print(summary[cols].to_string(index=False))
    print(f'\nWrote: {OUT}')


if __name__ == '__main__':
    main()
