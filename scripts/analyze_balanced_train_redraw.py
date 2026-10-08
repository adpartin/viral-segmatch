"""Summarize fixed-test prediction variation across training-negative redraws."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Summarize score and metric variation on the same test pairs.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--run-prefix', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()

    predictions = []
    metrics = []
    reference = None
    for seed in args.seeds:
        run = args.model_root / f'{args.run_prefix}_seed{seed}'
        pred = pd.read_csv(run / 'test_predicted.csv',
                           usecols=['pair_key', 'label', 'pred_label', 'pred_prob'])
        if pred.pair_key.duplicated().any():
            raise ValueError(f'Seed {seed}: duplicate test pair_key')
        pred = pred.sort_values('pair_key').reset_index(drop=True)
        if reference is None:
            reference = pred[['pair_key', 'label']]
        elif not pred[['pair_key', 'label']].equals(reference):
            raise ValueError(f'Seed {seed}: test pairs or labels differ')
        predictions.append(pred.pred_prob.to_numpy())
        row = {'seed': seed}
        row.update(json.loads((run / 'metrics_summary.json').read_text())['test'])
        metrics.append(row)

    probabilities = np.stack(predictions, axis=1)
    report = reference.copy()
    report['pred_prob_mean'] = probabilities.mean(axis=1)
    report['pred_prob_sd'] = probabilities.std(axis=1, ddof=1)
    report['pred_prob_min'] = probabilities.min(axis=1)
    report['pred_prob_max'] = probabilities.max(axis=1)
    report['n_predicted_positive'] = (probabilities > 0.5).sum(axis=1)
    report['prediction_flips'] = report.n_predicted_positive.between(1, len(args.seeds) - 1)
    metric_table = pd.DataFrame(metrics)

    summary = {'seeds': args.seeds, 'n_test': len(report), 'n_models': len(args.seeds),
               'n_prediction_flip_rows': int(report.prediction_flips.sum()),
               'fraction_prediction_flip_rows': float(report.prediction_flips.mean()),
               'metrics': {col: {'mean': float(metric_table[col].mean()),
                                 'sd': float(metric_table[col].std(ddof=1)),
                                 'min': float(metric_table[col].min()),
                                 'max': float(metric_table[col].max())}
                           for col in ('f1', 'f1_macro', 'auc_roc')},
               'by_true_label': {}}
    for label in (0, 1):
        subset = report.loc[report.label == label]
        summary['by_true_label'][str(label)] = {
            'n': len(subset),
            'pred_prob_sd_median': float(subset.pred_prob_sd.median()),
            'pred_prob_sd_p90': float(subset.pred_prob_sd.quantile(0.9)),
            'pred_prob_sd_max': float(subset.pred_prob_sd.max()),
            'fraction_prediction_flip_rows': float(subset.prediction_flips.mean()),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / 'test_prediction_variability.csv', index=False)
    metric_table.to_csv(args.output_dir / 'test_metrics_by_seed.csv', index=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
