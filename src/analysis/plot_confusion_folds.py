"""Confusion matrix pooled across CV folds, plus the per-fold spread.

`analyze_stage4_train.py` already writes a confusion matrix, but per run and only when the post-hoc
pass is not skipped. A k-fold experiment needs the folds together: one matrix over all test rows,
and the per-fold counts beside it so a pooled number is not mistaken for a stable one.

Pooling counts rather than averaging rates is what makes the matrix add up. Folds differ slightly in
size, so averaging per-fold rates would weight a small fold like a large one and the four cells
would no longer sum to the row count.

Reads the saved `test_predicted.csv` of each run, so it retrains nothing and works for any feature
source. The label and prediction columns are validated as 0/1 rather than rounded, because a
non-binary value would mean the run wrote something other than a hard decision.

Outputs (to `--out_dir`, by default derived from the first run dir):
    confusion_folds_{tag}.png   pooled matrix, and precision / recall / F1 per fold
    confusion_folds_{tag}.csv   one row per fold plus a pooled row: tp, fp, tn, fn, precision,
                                recall, f1, f1_macro, accuracy, n

CLI:
    python -m src.analysis.plot_confusion_folds \\
        --run_dirs models/flu/July_2025/runs/lgbm_ha_na_h3n2_2024_random_cv4_site_codon_fold{0,1,2,3} \\
        --tag site_codon
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.plot_utils import savefig, setup_plot_style  # noqa: E402

POSITIVE_COLOR = '#4C7CAB'
NEGATIVE_COLOR = '#CF8793'


def read_predictions(run_dir: Path) -> tuple:
    """Read one run's saved test labels and hard predictions.

    Args:
      run_dir: model run directory holding `test_predicted.csv`.

    Returns:
      `(y_true, y_pred)` as int arrays of 0 and 1.

    Raises:
      FileNotFoundError: the run has no `test_predicted.csv`.
      ValueError: `label` or `pred_label` holds a value other than 0 or 1.
    """
    path = run_dir / 'test_predicted.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Train the run first with "
            f"`python src/models/train_pair_baselines.py --baseline lgbm ...`.")
    # keep_default_na: pair tables carry protein names, and 'NA' (Neuraminidase) is a real value.
    table = pd.read_csv(path, keep_default_na=False, na_values=[''], low_memory=False)
    columns = {}
    for name in ('label', 'pred_label'):
        values = table[name].astype(float)
        unexpected = set(values.unique()) - {0.0, 1.0}
        if unexpected:
            raise ValueError(f"{path}: {name} must be 0 or 1; found {sorted(unexpected)[:5]}.")
        columns[name] = values.astype(int).to_numpy()
    return columns['label'], columns['pred_label']


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Count the four confusion cells and the metrics derived from them.

    F1 macro is the mean of the positive-class and negative-class F1, which is what
    `metrics.csv` reports, so the two can be checked against each other.

    Args:
      y_true: true labels, 0 or 1.
      y_pred: predicted labels, 0 or 1.

    Returns:
      `{'tp', 'fp', 'tn', 'fn', 'n', 'precision', 'recall', 'f1', 'f1_macro', 'accuracy'}`.
      Rate entries are `nan` where the denominator is zero.
    """
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float('nan')

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    # The negative class read as the positive one, so macro F1 is the mean of the two.
    negative_f1 = ratio(2 * tn, 2 * tn + fn + fp)
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'n': tp + fp + tn + fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'f1_macro': np.nanmean([f1, negative_f1]),
            'accuracy': ratio(tp + tn, tp + fp + tn + fn)}


def fold_table(run_dirs: list) -> pd.DataFrame:
    """Build the per-fold rows plus a pooled row.

    The pooled row sums the four cells across folds and recomputes the rates from those sums, so
    it describes every test row once and its cells stay consistent with its rates.

    Args:
      run_dirs: one model run directory per fold.

    Returns:
      One row per fold, then a `pooled` row.
    """
    rows, all_true, all_pred = [], [], []
    for run_dir in run_dirs:
        y_true, y_pred = read_predictions(run_dir)
        all_true.append(y_true)
        all_pred.append(y_pred)
        rows.append({'run': run_dir.name, **confusion_counts(y_true, y_pred)})
    pooled = confusion_counts(np.concatenate(all_true), np.concatenate(all_pred))
    rows.append({'run': 'pooled', **pooled})
    return pd.DataFrame(rows)


def plot_confusion(table: pd.DataFrame, tag: str, out_path: Path, dpi: int) -> Path:
    """Draw the pooled matrix and the per-fold metric spread.

    Args:
      table: rows from `fold_table`, including the pooled row.
      tag: label for the arm, used in the title.
      out_path: where the PNG goes.
      dpi: raster resolution.

    Returns:
      The written path.
    """
    setup_plot_style()
    pooled = table[table['run'] == 'pooled'].iloc[0]
    folds = table[table['run'] != 'pooled']

    fig, (ax_matrix, ax_metrics) = plt.subplots(1, 2, figsize=(11, 4.2))

    cells = np.array([[pooled['tn'], pooled['fp']], [pooled['fn'], pooled['tp']]], dtype=float)
    ax_matrix.imshow(cells / cells.sum(), cmap='Blues', vmin=0, vmax=cells.max() / cells.sum())
    for row in range(2):
        for col in range(2):
            count = int(cells[row, col])
            ax_matrix.text(col, row, f"{count:,}\n{count / cells.sum():.1%}",
                           ha='center', va='center', fontsize=12)
    ax_matrix.set_xticks([0, 1]); ax_matrix.set_xticklabels(['predicted 0', 'predicted 1'])
    ax_matrix.set_yticks([0, 1]); ax_matrix.set_yticklabels(['true 0', 'true 1'])
    ax_matrix.set_title(f"{tag}: pooled over {len(folds)} folds, {int(pooled['n']):,} test rows")

    metrics = ['precision', 'recall', 'f1', 'f1_macro']
    positions = np.arange(len(metrics))
    ax_metrics.bar(positions, [pooled[m] for m in metrics], color=POSITIVE_COLOR,
                   edgecolor='#222222', linewidth=0.6, zorder=2)
    for position, metric in zip(positions, metrics):
        ax_metrics.scatter([position] * len(folds), folds[metric], s=26, color=NEGATIVE_COLOR,
                           edgecolor='#222222', linewidth=0.5, zorder=3)
        ax_metrics.text(position, pooled[metric] + 0.02, f"{pooled[metric]:.3f}",
                        ha='center', va='bottom', fontsize=9)
    ax_metrics.set_xticks(positions); ax_metrics.set_xticklabels(metrics)
    ax_metrics.set_ylim(0, 1.12)
    ax_metrics.set_ylabel('pooled value; dots are folds')
    ax_metrics.set_title('Pooled metrics, with the per-fold spread')
    ax_metrics.grid(axis='y', alpha=0.3)

    fig.text(0.995, 0.002, f'src/analysis/{Path(__file__).name}', ha='right', va='bottom',
             fontsize=7, color='#666666')
    fig.tight_layout()
    return savefig(out_path, dpi=dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run_dirs', type=Path, nargs='+', required=True,
                        help='one model run directory per fold, each holding test_predicted.csv')
    parser.add_argument('--tag', required=True,
                        help='short name for this arm, used in the filenames and the title')
    parser.add_argument('--out_dir', type=Path, default=None)
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    table = fold_table(args.run_dirs)
    out_dir = args.out_dir or args.run_dirs[0] / 'confusion_folds'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'confusion_folds_{args.tag}.csv'
    table.to_csv(csv_path, index=False)
    figure_path = plot_confusion(table, args.tag, out_dir / f'confusion_folds_{args.tag}.png',
                                 args.dpi)

    shown = ['run', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1', 'f1_macro']
    print(table[shown].to_string(index=False,
                                 float_format=lambda v: f'{v:.4f}'))
    pooled = table[table['run'] == 'pooled'].iloc[0]
    print(f"\nPooled precision {pooled['precision']:.4f} against recall {pooled['recall']:.4f}: "
          f"{int(pooled['fp']):,} false positives against {int(pooled['fn']):,} false negatives.")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {figure_path}")
    print('Done.')


if __name__ == '__main__':
    main()
