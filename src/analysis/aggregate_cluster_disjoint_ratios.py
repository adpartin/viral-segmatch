"""Aggregate cluster_disjoint id099 results across negative-ratio variants.

Compares ratio=1.5 vs ratio=3.0 on both schema pairs and both models (LGBM,
MLP). Recomputes test metrics from `test_predicted.csv` for each run so
the table is consistent (some MLP runs store metrics_summary.json with
zero MCC due to a writer bug, but test_predicted.csv is authoritative).

Usage:
    python -m src.analysis.aggregate_cluster_disjoint_ratios

Writes:
    results/flu/July_2025/runs/cluster_disjoint_id99_ratio_sweep.csv
    results/flu/July_2025/runs/cluster_disjoint_id99_ratio_sweep.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / 'models' / 'flu' / 'July_2025' / 'runs'


def metrics_from_predictions(p: Path) -> dict:
    df = pd.read_csv(p)
    y = df['label']
    yp = df['pred_label']
    ys = df['pred_prob']
    n_fp = int(((yp == 1) & (y == 0)).sum())
    n_fn = int(((yp == 0) & (y == 1)).sum())
    return {
        'F1': float(f1_score(y, yp)),
        'AUC': float(roc_auc_score(y, ys)),
        'Precision': float(precision_score(y, yp, zero_division=0)),
        'Recall': float(recall_score(y, yp)),
        'MCC': float(matthews_corrcoef(y, yp)),
        'n_test': len(df),
        'n_pos': int((y == 1).sum()),
        'n_neg': int((y == 0).sum()),
        'n_FP': n_fp,
        'n_FN': n_fn,
        'FPR': n_fp / max(int((y == 0).sum()), 1),
    }


def latest_run(prefix: str, root: Path = MODELS_ROOT) -> Optional[Path]:
    candidates = sorted(root.glob(f'{prefix}*'))
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_csv',
                        default='results/flu/July_2025/runs/cluster_disjoint_id99_ratio_sweep.csv')
    parser.add_argument('--out_png',
                        default='results/flu/July_2025/runs/cluster_disjoint_id99_ratio_sweep.png')
    args = parser.parse_args()

    # Each row identified by (schema, ratio, model)
    spec = [
        ('HA/NA',   1.5, 'LGBM',
         'baseline_lgbm_flu_ha_na_cluster_id99_20260514_220604'),
        ('HA/NA',   1.5, 'MLP',
         'training_flu_ha_na_cluster_id99_20260514_225013'),
        ('PB2/PB1', 1.5, 'LGBM',
         'baseline_lgbm_flu_pb2_pb1_cluster_id99_20260514_220606'),
        ('PB2/PB1', 1.5, 'MLP',
         'training_flu_pb2_pb1_cluster_id99_20260514_225015'),
        ('HA/NA',   3.0, 'LGBM',
         'baseline_lgbm_flu_ha_na_cluster_id99_r3_'),
        ('HA/NA',   3.0, 'MLP',
         'training_flu_ha_na_cluster_id99_r3_'),
        ('PB2/PB1', 3.0, 'LGBM',
         'baseline_lgbm_flu_pb2_pb1_cluster_id99_r3_'),
        ('PB2/PB1', 3.0, 'MLP',
         'training_flu_pb2_pb1_cluster_id99_r3_'),
    ]

    rows = []
    for schema, ratio, model, run_or_prefix in spec:
        # Resolve to a concrete run dir
        if (MODELS_ROOT / run_or_prefix).exists():
            run = MODELS_ROOT / run_or_prefix
        else:
            run = latest_run(run_or_prefix)
        if run is None:
            print(f'WARNING: no run found for {schema} ratio={ratio} {model} (prefix={run_or_prefix})')
            continue
        p = run / 'test_predicted.csv'
        if not p.exists():
            print(f'WARNING: missing {p}')
            continue
        m = metrics_from_predictions(p)
        m.update({'schema_pair': schema, 'ratio': ratio, 'model': model, 'run': run.name})
        rows.append(m)
        print(f'{schema:8s} r={ratio} {model:5s}: F1={m["F1"]:.4f} AUC={m["AUC"]:.4f} '
              f'P={m["Precision"]:.4f} R={m["Recall"]:.4f} FP={m["n_FP"]} FPR={m["FPR"]:.4f}')

    df = pd.DataFrame(rows)
    cols = ['schema_pair', 'ratio', 'model', 'F1', 'AUC', 'Precision', 'Recall',
            'MCC', 'n_test', 'n_pos', 'n_neg', 'n_FP', 'n_FN', 'FPR', 'run']
    df = df[[c for c in cols if c in df.columns]]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote: {out_csv}')

    if df.empty:
        print('No data to plot.')
        return

    # 2x2 panels: F1, AUC, Precision, n_FP — bar per (schema, model) group, colored by ratio
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    schemas = ['HA/NA', 'PB2/PB1']
    models = ['LGBM', 'MLP']
    ratios = [1.5, 3.0]
    group_labels = [f'{s}\n{m}' for s in schemas for m in models]
    colors = {1.5: '#dd8452', 3.0: '#4c72b0'}
    metric_panels = [
        ('F1', 'Test F1', (0, 1.0)),
        ('AUC', 'Test AUC', (0, 1.0)),
        ('Precision', 'Test Precision', (0, 1.0)),
        ('n_FP', 'False positives (test, thr=0.5)', None),
    ]
    for ax, (metric, title, ylim) in zip(axes, metric_panels):
        x = list(range(len(group_labels)))
        width = 0.36
        for i, r in enumerate(ratios):
            vals = []
            for s in schemas:
                for m in models:
                    sub = df[(df.schema_pair == s) & (df.model == m) & (df.ratio == r)]
                    vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
            bars = ax.bar([xi + i*width - width/2 for xi in x], vals, width,
                          label=f'ratio={r}', color=colors[r])
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    label = f'{v:.3f}' if metric != 'n_FP' else f'{int(v)}'
                    ax.text(b.get_x() + b.get_width()/2, v, label,
                            ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(group_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        if ylim is not None: ax.set_ylim(*ylim)
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)
        ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
    fig.suptitle('cluster_disjoint @ id099 — effect of neg_to_pos_ratio (1.5 vs 3.0)\n'
                 'same routing, same features (k-mer nt k=6, unit_norm + unit_diff+prod), single seed=42',
                 fontsize=11, y=1.04)
    fig.tight_layout()
    out_png = Path(args.out_png)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
