"""Summarize fixed-test prediction variation across training-negative redraws.

Example:
    python scripts/analyze_balanced_train_redraw.py \
        --model-root models/flu/July_2025/runs \
        --run-prefix lgbm_exp6_negdraw_..._pb2_ha \
        --folds 0 1 2 3 --seeds 42 43 44 45 46 47 48 49 50 51 \
        --output-dir results/flu/July_2025/exp6_negative_redraw_..._pb2_ha

Reads `{run-prefix}_fold{k}_seed{s}/test_predicted.csv`, the layout written by
run_balanced_train_redraw.py. The design is Experiment 6 of
docs/plans/2026-09-14_codon_site_features_plan.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.plot_utils import grouped_boxplot, savefig

CLASSES = {1: 'positive', 0: 'negative'}  # test-pair label -> class name, positive first
SCORES = {'prob': 'probability', 'logit': 'raw score (logit)'}  # SD column stem -> axis name


def _load_fold(model_root: Path, run_prefix: str, fold: int,
               seeds: list[int]) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Read one fold's test predictions and test metrics from every redraw model.

    Args:
        model_root: Directory holding the model run directories.
        run_prefix: Run name before `_fold{k}_seed{s}`.
        fold: Fold index.
        seeds: Redraw seeds, one model each.

    Returns:
        The test pairs (`pair_key`, `label`) sorted by `pair_key`; {column: array of shape
        (n_test, n_seeds)} for `pred_prob`, `pred_logit` and `pred_label`; and one row of test
        metrics per seed.
    """
    reference = None
    columns = ('pred_prob', 'pred_logit', 'pred_label')
    stacked = {col: [] for col in columns}
    metrics = []
    for seed in seeds:
        run = model_root / f'{run_prefix}_fold{fold}_seed{seed}'
        wanted = {'pair_key', 'label', *columns}
        pred = pd.read_csv(run / 'test_predicted.csv', usecols=lambda col: col in wanted,
                           keep_default_na=False, na_values=[''])
        if 'pred_logit' not in pred.columns or pred.pred_logit.isna().any():
            raise ValueError(f'{run.name}: no pred_logit values; the run predates raw-score export')
        if pred.pair_key.duplicated().any():
            raise ValueError(f'{run.name}: duplicate test pair_key')
        pred = pred.sort_values('pair_key').reset_index(drop=True)
        if reference is None:
            reference = pred[['pair_key', 'label']]
        elif not pred[['pair_key', 'label']].equals(reference):
            raise ValueError(f'{run.name}: test pairs or labels differ from seed {seeds[0]}')
        for col in columns:
            stacked[col].append(pred[col].to_numpy())

        row = {'fold': fold, 'seed': seed}
        row.update(json.loads((run / 'metrics_summary.json').read_text())['test'])
        metrics.append(row)

    arrays = {col: np.stack(values, axis=1) for col, values in stacked.items()}
    return reference, arrays, pd.DataFrame(metrics)


def _per_pair_report(reference: pd.DataFrame, arrays: dict) -> pd.DataFrame:
    """Describe each test pair's scores across the redraw models.

    Args:
        reference: Test pairs (`pair_key`, `label`), in the row order of `arrays`.
        arrays: {column: array of shape (n_test, n_seeds)} from `_load_fold`.

    Returns:
        One row per test pair with the mean and sample SD of its probabilities and raw scores,
        the number of models that classify it as positive, and its count of probabilities at
        exactly 0 or 1.
    """
    report = reference.copy()
    for stem in SCORES:
        values = arrays[f'pred_{stem}']
        report[f'pred_{stem}_mean'] = values.mean(axis=1)
        report[f'pred_{stem}_sd'] = values.std(axis=1, ddof=1)
    report['pred_prob_min'] = arrays['pred_prob'].min(axis=1)
    report['pred_prob_max'] = arrays['pred_prob'].max(axis=1)
    report['n_predicted_positive'] = arrays['pred_label'].sum(axis=1)
    probs = arrays['pred_prob']
    report['n_prob_at_0_or_1'] = ((probs == 0.0) | (probs == 1.0)).sum(axis=1)
    return report


def _sd_by_class(report: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Summarize the per-pair SDs of one fold by class.

    Args:
        report: Output of `_per_pair_report`.
        fold: Fold index, copied into every row.

    Returns:
        One row per score and class with n, mean, median, quartiles and IQR of the per-pair SD.
    """
    rows = []
    for stem in SCORES:
        for label, name in CLASSES.items():
            sd = report.loc[report.label == label, f'pred_{stem}_sd']
            q25, q75 = sd.quantile(0.25), sd.quantile(0.75)
            rows.append({'fold': fold, 'score': stem, 'class': name, 'n': len(sd),
                         'mean_sd': sd.mean(), 'median_sd': sd.median(),
                         'q25_sd': q25, 'q75_sd': q75, 'iqr_sd': q75 - q25})
    return pd.DataFrame(rows)


def _class_difference(sd_table: pd.DataFrame) -> pd.DataFrame:
    """Subtract the negative-class mean SD from the positive-class mean SD, per fold and score.

    Args:
        sd_table: Rows from `_sd_by_class`.

    Returns:
        One row per fold and score with both class means and their difference.
    """
    wide = sd_table.pivot_table(index=['fold', 'score'], columns='class', values='mean_sd')
    wide = wide.rename(columns={'positive': 'positive_mean_sd', 'negative': 'negative_mean_sd'})
    wide['positive_minus_negative'] = wide.positive_mean_sd - wide.negative_mean_sd
    table = wide.reset_index()[['fold', 'score', 'positive_mean_sd', 'negative_mean_sd',
                                'positive_minus_negative']]
    return table


def _plot_per_pair_sd(report: pd.DataFrame, fold: int, title: str, out_png: Path) -> None:
    """Plot each class's per-pair SDs, one panel for probabilities and one for raw scores.

    Args:
        report: Output of `_per_pair_report`.
        fold: Fold index, used as the group name.
        title: Figure title.
        out_png: Destination file.
    """
    group = f'fold {fold}'
    fig, axes = plt.subplots(1, len(SCORES), figsize=(4.2 * len(SCORES), 4.2))
    for ax, (stem, axis_name) in zip(axes, SCORES.items()):
        values = {(group, name): report.loc[report.label == label, f'pred_{stem}_sd']
                  for label, name in CLASSES.items()}
        grouped_boxplot(ax, values, groups=[group], categories=list(CLASSES.values()))
        ax.set_ylabel(f'SD of {axis_name} across draws')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    savefig(out_png)


def _plot_draw_probabilities(reference: pd.DataFrame, probs: np.ndarray, seeds: list[int],
                             title: str, out_png: Path) -> None:
    """Plot each draw's class-1 probabilities, one box per class within each draw.

    Args:
        reference: Test pairs (`pair_key`, `label`), in the row order of `probs`.
        probs: Probabilities of shape (n_test, n_seeds).
        seeds: Redraw seeds, in the column order of `probs`.
        title: Figure title.
        out_png: Destination file.
    """
    groups = [f'seed {seed}' for seed in seeds]
    values = {(group, name): probs[reference.label.to_numpy() == label, i]
              for i, group in enumerate(groups) for label, name in CLASSES.items()}
    fig, ax = plt.subplots(figsize=(16, 4.2))
    grouped_boxplot(ax, values, groups=groups, categories=list(CLASSES.values()),
                    group_width=0.8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylabel('class-1 probability')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    savefig(out_png)


def main() -> None:
    """Write per-pair, per-fold and per-class summaries and the two figures for each fold.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--run-prefix', required=True)
    parser.add_argument('--folds', type=int, nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--label', default='', help='Pair name for figure titles, e.g. PB2-HA')
    args = parser.parse_args()

    sd_tables, metric_tables, fold_summaries = [], [], {}
    for fold in args.folds:
        reference, arrays, metrics = _load_fold(args.model_root, args.run_prefix, fold, args.seeds)
        report = _per_pair_report(reference, arrays)
        fold_dir = args.output_dir / f'fold_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(fold_dir / 'test_prediction_variability.csv', index=False)

        title = f'{args.label} fold {fold}, {len(args.seeds)} negative draws'.strip()
        _plot_per_pair_sd(report, fold, title, fold_dir / 'per_pair_sd_boxplot.png')
        _plot_draw_probabilities(reference, arrays['pred_prob'], args.seeds, title,
                                 fold_dir / 'draw_probability_boxplot.png')

        # How many of the models call each pair positive, by class: 0 or n_seeds means no change.
        votes = (report.groupby(['label', 'n_predicted_positive']).size()
                 .unstack(fill_value=0).rename(index=CLASSES))
        fold_summaries[fold] = {
            'n_test': len(report),
            'n_prob_at_0_or_1': int(report.n_prob_at_0_or_1.sum()),
            'n_predicted_positive_by_class': {name: {int(k): int(v) for k, v in row.items()}
                                              for name, row in votes.iterrows()},
        }
        sd_tables.append(_sd_by_class(report, fold))
        metric_tables.append(metrics)

    sd_table = pd.concat(sd_tables, ignore_index=True)
    difference = _class_difference(sd_table)
    metric_table = pd.concat(metric_tables, ignore_index=True)
    sd_table.to_csv(args.output_dir / 'per_pair_sd_by_class.csv', index=False)
    difference.to_csv(args.output_dir / 'per_pair_sd_class_difference.csv', index=False)
    metric_table.to_csv(args.output_dir / 'test_metrics_by_seed.csv', index=False)

    metric_spread = (metric_table.groupby('fold')[['f1', 'f1_macro', 'auc_roc']]
                     .agg(['mean', 'std', 'min', 'max']))
    summary = {'run_prefix': args.run_prefix, 'folds': args.folds, 'seeds': args.seeds,
               'by_fold': fold_summaries,
               'test_metrics_across_seeds': {
                   int(fold): {f'{metric}_{stat}': float(value) for (metric, stat), value
                               in row.items()}
                   for fold, row in metric_spread.iterrows()}}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(difference.to_string(index=False))
    print(f'Done. Wrote {args.output_dir}')


if __name__ == '__main__':
    main()
