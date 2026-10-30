"""Per-metadata-group counts and test metrics for a population that holds several groups.

A metadata group is one exact `(host, hn_subtype, year)` combination (see
`docs/methods/glossary.md`). `allpairs_summary.csv` keeps one row per schema pair for the whole
population, so on a population with several groups that row pools them. These two tables say what
each group contributed and how the model scored on it:

- `metadata_group_counts.csv`, from the saved pair files: one row per schema pair, fold, split and
  group, with its positives and negatives. Within a split the group counts add up to the split's
  totals, which the writer checks.
- `test_metrics_by_metadata.csv`, from each fit's `test_predicted.csv`: one row per schema pair,
  fold and grouping, scored with `compute_pair_metrics` at the fit's own decision threshold, so
  the numbers carry the same definitions as `allpairs_summary.csv`.

Both tables also verify the match the matched negative sampler is supposed to produce: every
saved negative's two endpoints must agree on host, HN subtype and year. The check reads the saved
endpoint metadata rather than the `neg_regime` column or the derived `same_*` flags, because those
record what was asked for rather than what was written.

A row's grouping is `joint` for one metadata group, or `host`, `hn_subtype` or `year` for all the
groups sharing that one value. Design: docs/plans/2026-10-08_paper_plan.md section 2.

CLI:
    python -m src.analysis.metadata_group_report \\
        --dataset_manifest data/datasets/flu/July_2025/runs/<prefix>_manifest.json \\
        --out_dir results/flu/July_2025/<dir>
    python -m src.analysis.metadata_group_report \\
        --run_manifest models/flu/July_2025/training_runs/<name>/training_run_manifest.json \\
        --out_dir results/flu/July_2025/<dir>

Outputs (to `--out_dir`):
    metadata_group_counts.csv          with `--dataset_manifest`
    counts_by_group.png                the same counts as a (group x schema pair) grid
    test_metrics_by_metadata.csv       with `--run_manifest`
    metrics_by_group_<metric>.png      one row per metadata group
    metrics_by_axis_<metric>.png       one row per host, HN subtype and year value

The schema-pair matrix for the whole population is the fourth figure Task 2 reports, and
`aggregate_allpairs_results.py` already writes it as `matrix_f1_macro.png`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.models._pair_metrics import compute_pair_metrics  # noqa: E402
from src.utils.dataset_manifest import load_dataset_manifest  # noqa: E402

# The axes a metadata group is defined on, and the per-slot columns the pair tables carry for them.
GROUP_AXES = ('host', 'hn_subtype', 'year')
SPLITS = ('train', 'val', 'test')
COUNT_COLUMNS = ['population', 'pair', 'fold', 'split', *GROUP_AXES, 'n_pos', 'n_neg']
# The metric names `allpairs_summary.csv` uses, recomputed here per group.
GROUP_METRICS = ('f1_macro', 'auc_roc', 'precision', 'recall')
METRIC_COLUMNS = ['population', 'pair', 'fold', 'group_by', *GROUP_AXES,
                  'n_pos', 'n_neg', 'tp', 'tn', 'fp', 'fn', *GROUP_METRICS]


def read_pairs(path: Path) -> pd.DataFrame:
    """Read a saved pair table, keeping 'NA' (Neuraminidase) a string.

    Args:
        path: a `{train,val,test}_pairs.csv` or `test_predicted.csv`.

    Returns:
        The table as written.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def check_axes_match(pairs: pd.DataFrame, source: str) -> None:
    """Raise unless every negative's two endpoints agree on host, HN subtype and year.

    Positives are same-isolate pairs, so only the negatives can disagree. This reads the saved
    per-slot metadata, which is what the group assignment below relies on.

    Args:
        pairs: a pair table carrying `label` and `<axis>_a` / `<axis>_b` for `GROUP_AXES`.
        source: the file the table came from, used in the error message.

    Raises:
        ValueError: a column is missing, or a negative's endpoints disagree on an axis.
    """
    missing = [f'{axis}_{side}' for axis in GROUP_AXES for side in ('a', 'b')
               if f'{axis}_{side}' not in pairs.columns]
    if missing:
        raise ValueError(f"check_axes_match: {source} has no {missing} columns, so the metadata "
                         f"of a negative's two endpoints cannot be compared.")
    negatives = pairs[pairs['label'] == 0]
    for axis in GROUP_AXES:
        differs = negatives[f'{axis}_a'].astype(str) != negatives[f'{axis}_b'].astype(str)
        if differs.any():
            examples = negatives.loc[differs, [f'{axis}_a', f'{axis}_b']].head(3).to_dict('records')
            raise ValueError(
                f"check_axes_match: {source} has {int(differs.sum()):,} negatives whose two "
                f"endpoints differ on {axis}, for example {examples}. Matched negatives must "
                f"share every axis in {list(GROUP_AXES)}.")


def group_values(pairs: pd.DataFrame) -> pd.DataFrame:
    """The metadata group of each row, taken from its slot-a endpoint.

    Args:
        pairs: a pair table whose endpoints have already passed `check_axes_match`.

    Returns:
        One column per axis in `GROUP_AXES`, as strings, with the row index of `pairs`.
    """
    return pd.DataFrame({axis: pairs[f'{axis}_a'].astype(str) for axis in GROUP_AXES},
                        index=pairs.index)


def count_one_split(pairs: pd.DataFrame, source: str) -> pd.DataFrame:
    """Positives and negatives per metadata group for one split.

    Args:
        pairs: the split's pair table.
        source: the file it came from, used in error messages.

    Returns:
        One row per group, with `GROUP_AXES`, `n_pos` and `n_neg`.

    Raises:
        ValueError: the per-group counts do not add up to the split's totals.
    """
    check_axes_match(pairs, source)
    table = group_values(pairs)
    table['n_pos'] = (pairs['label'] == 1).astype(int)
    table['n_neg'] = (pairs['label'] == 0).astype(int)
    counts = table.groupby(list(GROUP_AXES), as_index=False)[['n_pos', 'n_neg']].sum()
    for column, expected in (('n_pos', int((pairs['label'] == 1).sum())),
                             ('n_neg', int((pairs['label'] == 0).sum()))):
        total = int(counts[column].sum())
        if total != expected:
            raise ValueError(f"count_one_split: {source} has {expected:,} {column} but its "
                             f"groups add up to {total:,}.")
    return counts


def metadata_group_counts(dataset_manifest: dict) -> pd.DataFrame:
    """Per-group positive and negative counts for every pair, fold and split of one build.

    Args:
        dataset_manifest: from `load_dataset_manifest`.

    Returns:
        A table in `COUNT_COLUMNS`, sorted by pair, fold, split and group.
    """
    from src.analysis.aggregate_allpairs_results import _pair_labels

    labels = _pair_labels(dataset_manifest)
    rows = []
    for entry in dataset_manifest['datasets'].values():
        pair, population = labels[entry['pair']]
        dataset_dir = Path(entry['dataset_dir'])
        for split_dir in entry['split_dirs']:
            fold = Path(split_dir).name if split_dir != '.' else '.'
            for split in SPLITS:
                path = dataset_dir / split_dir / f'{split}_pairs.csv'
                counts = count_one_split(read_pairs(path), str(path))
                counts.insert(0, 'split', split)
                counts.insert(0, 'fold', fold)
                counts.insert(0, 'pair', pair)
                counts.insert(0, 'population', population)
                rows.append(counts)
    table = pd.concat(rows, ignore_index=True)
    ordered = table.sort_values(['pair', 'fold', 'split', *GROUP_AXES]).reset_index(drop=True)
    return ordered[COUNT_COLUMNS]


def subset_metrics(subset: pd.DataFrame, threshold: float) -> dict:
    """The counts and metrics of one subset of a fit's test predictions.

    The metric values come from `compute_pair_metrics`, the same function the trainer and
    `aggregate_allpairs_results` use, so they carry one definition. A subset holding one class has
    no precision, recall, F1 or AUC-ROC, so those are NaN rather than the chance value
    `compute_pair_metrics` substitutes.

    Args:
        subset: rows of a `test_predicted.csv`, carrying `label` and `pred_prob`.
        threshold: the fit's decision threshold, from its `training_info.json`.

    Returns:
        `n_pos`, `n_neg`, `tp`, `tn`, `fp`, `fn` and each name in `GROUP_METRICS`.
    """
    labels = subset['label'].to_numpy()
    probabilities = subset['pred_prob'].to_numpy()
    predicted = (probabilities > threshold).astype(int)
    counts = {
        'n_pos': int((labels == 1).sum()),
        'n_neg': int((labels == 0).sum()),
        'tp': int(((labels == 1) & (predicted == 1)).sum()),
        'tn': int(((labels == 0) & (predicted == 0)).sum()),
        'fp': int(((labels == 0) & (predicted == 1)).sum()),
        'fn': int(((labels == 1) & (predicted == 0)).sum()),
    }
    if counts['n_pos'] == 0 or counts['n_neg'] == 0:
        return {**counts, **{name: np.nan for name in GROUP_METRICS}}
    metrics, _ = compute_pair_metrics(labels, probabilities, threshold, subset,
                                      print_classification_report=False)
    return {**counts, **{name: float(metrics[name]) for name in GROUP_METRICS}}


def fit_group_metrics(model_dir: Path, pair: str, fold: str, population: str) -> list:
    """One fit's test metrics per metadata group and per single axis value.

    Args:
        model_dir: a finished fit's run directory.
        pair: the schema pair label, e.g. `PB2-PB1`.
        fold: the fold the fit trained on.
        population: the population label the dataset was built for.

    Returns:
        Rows in `METRIC_COLUMNS`: one `joint` row per metadata group, then one row per value of
        each axis in `GROUP_AXES`.

    Raises:
        ValueError: the metadata groups do not account for every test pair of the fit.
    """
    predictions = read_pairs(model_dir / 'test_predicted.csv')
    check_axes_match(predictions, str(model_dir / 'test_predicted.csv'))
    info = json.loads((model_dir / 'training_info.json').read_text())
    threshold = float(info['optimal_threshold'])
    groups = group_values(predictions)

    rows = []
    joint = groups.apply(tuple, axis=1)
    for values in sorted(set(joint)):
        subset = predictions[joint == values]
        rows.append({'population': population, 'pair': pair, 'fold': fold, 'group_by': 'joint',
                     **dict(zip(GROUP_AXES, values)),
                     **subset_metrics(subset, threshold)})
    for column, label in (('n_pos', 1), ('n_neg', 0)):
        grouped = sum(row[column] for row in rows)
        total = int((predictions['label'] == label).sum())
        if grouped != total:
            raise ValueError(
                f"fit_group_metrics: {model_dir.name} has {total:,} test {column} but its "
                f"metadata groups account for {grouped:,}.")

    for axis in GROUP_AXES:
        for value in sorted(set(groups[axis])):
            subset = predictions[groups[axis] == value]
            rows.append({'population': population, 'pair': pair, 'fold': fold, 'group_by': axis,
                         **{column: '' for column in GROUP_AXES}, axis: value,
                         **subset_metrics(subset, threshold)})
    return rows


def test_metrics_by_metadata(run_manifest_path: Path) -> pd.DataFrame:
    """Per-group test metrics for every fit of one training run.

    Args:
        run_manifest_path: a `training_run_manifest.json`.

    Returns:
        A table in `METRIC_COLUMNS`, sorted by pair, fold, grouping and group.
    """
    from src.analysis.aggregate_allpairs_results import _load_run_manifests, _pair_labels

    run_manifest, dataset_manifest = _load_run_manifests(run_manifest_path)
    labels = _pair_labels(dataset_manifest)
    rows = []
    for pair_token, plan in run_manifest['pairs'].items():
        pair, population = labels[pair_token]
        for fold, fit in plan['folds'].items():
            rows.extend(fit_group_metrics(Path(fit['model_dir']), pair, str(fold), population))
    table = pd.DataFrame(rows)
    ordered = table.sort_values(['pair', 'fold', 'group_by', *GROUP_AXES]).reset_index(drop=True)
    return ordered[METRIC_COLUMNS]


def group_label(values) -> str:
    """One metadata group's axis values as a row label, e.g. `Human-H3N2-2022`.

    Args:
        values: the group's values, in `GROUP_AXES` order.

    Returns:
        The values joined by `-`, which is how `population_label` names a population.
    """
    return '-'.join(str(value) for value in values)


def pair_columns(pairs) -> list:
    """Schema pair labels in canonical protein order, so a figure's columns read as segments.

    Args:
        pairs: the pair labels present, e.g. `{'PB2-PB1', 'NP-M1'}`.

    Returns:
        The labels sorted by their two proteins' positions in `PROTEINS`, with any label whose
        proteins are not both known sorted after them by name.
    """
    from src.analysis.aggregate_allpairs_results import PROTEINS

    def position(label: str):
        parts = label.split('-')
        if len(parts) == 2 and all(part in PROTEINS for part in parts):
            return (0, PROTEINS.index(parts[0]), PROTEINS.index(parts[1]))
        return (1, 0, 0)

    return sorted(pairs, key=lambda label: (position(label), label))


def fold_summary(metrics: pd.DataFrame, metric: str) -> tuple:
    """Each (row, pair) cell's fold mean, fold SD and contributing fold count.

    A fold where the metric is NaN, which is a group with one class in that fold's test split,
    is left out of its own cell rather than counted as a zero.

    Args:
        metrics: rows of `test_metrics_by_metadata.csv` for one grouping, with a `row` column
            holding the label to use.
        metric: the metric column to summarize.

    Returns:
        (mean, sd, folds) as DataFrames indexed by row label with one column per schema pair.
    """
    grouped = metrics.groupby(['row', 'pair'])[metric]
    mean = grouped.mean().unstack('pair')
    sd = grouped.std(ddof=1).unstack('pair')
    folds = grouped.count().unstack('pair')
    columns = pair_columns(mean.columns)
    return mean[columns], sd[columns], folds[columns]


def plot_metric_grid(mean: pd.DataFrame, sd: pd.DataFrame, folds: pd.DataFrame, *,
                     out_png: Path, title: str, ylabel: str, cbar_label: str,
                     scale: tuple, blank: pd.DataFrame = None) -> None:
    """Draw a (row x schema pair) metric grid, each cell the fold mean over the fold SD.

    Args:
        mean: fold means, from `fold_summary`.
        sd: fold SDs, same layout.
        folds: contributing fold counts, same layout. A cell with one fold has no SD and says so
            instead; a cell resting on fewer folds than the grid's best-covered one says how many.

        out_png: where to write the PNG.
        title: figure title, which carries the population.
        ylabel: what the rows are.
        cbar_label: color bar label.
        scale: (vmin, vmax) shared with the figure's sibling, so the two are comparable by eye.
        blank: True where a cell is masked out, or None to draw every cell.

    Returns:
        None. Writes `out_png`.
    """
    from src.utils.plot_utils import annotated_heatmap

    values = mean.to_numpy(dtype=float)
    most_folds = int(np.nanmax(folds.to_numpy(dtype=float))) if folds.size else 0
    text = [['' for _ in mean.columns] for _ in mean.index]
    for i in range(len(mean.index)):
        for j in range(len(mean.columns)):
            if np.isnan(values[i, j]):
                continue
            n_folds = int(folds.iat[i, j])
            lines = [f'{values[i, j]:.3f}']
            lines.append(f'± {sd.iat[i, j]:.3f}' if n_folds > 1 else '1 fold')
            if 1 < n_folds < most_folds:
                lines.append(f'{n_folds}/{most_folds} folds')
            text[i][j] = '\n'.join(lines)
    annotated_heatmap(
        values, row_labels=[str(label) for label in mean.index],
        col_labels=[str(label) for label in mean.columns], out_png=out_png, title=title,
        xlabel='Schema pair', ylabel=ylabel, cbar_label=cbar_label, cmap='YlGnBu',
        vmin=scale[0], vmax=scale[1], cell_text=text,
        blank_mask=None if blank is None else blank.to_numpy(bool),
        figsize=(1.6 + 0.70 * len(mean.columns), 2.2 + 0.52 * len(mean.index)),
        cell_fontsize=7, row_fontsize=9, col_rotation=45)
    print(f'Saved {out_png}')


def plot_count_grid(counts: pd.DataFrame, *, out_png: Path, title: str) -> None:
    """Draw the (metadata group x schema pair) grid of test positives summed over folds.

    Args:
        counts: `metadata_group_counts.csv` rows.
        out_png: where to write the PNG.
        title: figure title, which carries the population.

    Returns:
        None. Writes `out_png`.
    """
    from src.utils.plot_utils import annotated_heatmap

    test = counts[counts['split'] == 'test'].copy()
    test['row'] = test[list(GROUP_AXES)].apply(group_label, axis=1)
    table = test.groupby(['row', 'pair'])['n_pos'].sum().unstack('pair')
    table = table[pair_columns(table.columns)]
    values = table.to_numpy(dtype=float)
    text = [[('' if np.isnan(values[i, j]) else f'{int(values[i, j]):,}')
             for j in range(values.shape[1])] for i in range(values.shape[0])]
    annotated_heatmap(
        values, row_labels=[str(label) for label in table.index],
        col_labels=[str(label) for label in table.columns], out_png=out_png, title=title,
        xlabel='Schema pair', ylabel='Metadata group', cbar_label='Test positives, all folds',
        cmap='YlGnBu', cell_text=text,
        figsize=(1.6 + 0.70 * len(table.columns), 2.2 + 0.52 * len(table.index)),
        cell_fontsize=7, row_fontsize=9, col_rotation=45)
    print(f'Saved {out_png}')


def write_metric_figures(metrics: pd.DataFrame, out_dir: Path, metric: str,
                         min_test_pairs: int = 0) -> None:
    """Write the per-group and per-axis metric figures for one population.

    The two share one color scale, so a group's score can be read against its host's or its
    year's. `min_test_pairs` masks the cells too thin to read, which Open Question 1 of
    docs/plans/2026-10-08_paper_plan.md has to settle before it is used; at 0 nothing is masked.

    Args:
        metrics: a `test_metrics_by_metadata.csv` table.
        out_dir: where the PNGs go.
        metric: the metric to draw, e.g. `f1_macro`.
        min_test_pairs: mask a cell whose test pairs, summed over folds, fall below this.

    Returns:
        None. Writes `metrics_by_group_<metric>.png` and `metrics_by_axis_<metric>.png`.
    """
    population = ', '.join(sorted(set(metrics['population'])))
    joint = metrics[metrics['group_by'] == 'joint'].copy()
    joint['row'] = joint[list(GROUP_AXES)].apply(group_label, axis=1)
    axis_rows = metrics[metrics['group_by'].isin(GROUP_AXES)].copy()
    # An axis row carries its value in its own axis column and blanks in the others.
    axis_rows['row'] = axis_rows.apply(lambda row: str(row[row['group_by']]), axis=1)

    group_mean, group_sd, group_folds = fold_summary(joint, metric)
    axis_mean, axis_sd, axis_folds = fold_summary(axis_rows, metric)
    # Keep the axis rows in GROUP_AXES order, so hosts sit together, then subtypes, then years.
    axis_order = [value for axis in GROUP_AXES
                  for value in sorted(set(axis_rows.loc[axis_rows['group_by'] == axis, 'row']))]
    axis_mean = axis_mean.reindex(axis_order)
    axis_sd = axis_sd.reindex(axis_order)
    axis_folds = axis_folds.reindex(axis_order)
    finite = np.concatenate([group_mean.to_numpy(dtype=float).ravel(),
                             axis_mean.to_numpy(dtype=float).ravel()])
    finite = finite[~np.isnan(finite)]
    scale = (float(finite.min()), float(finite.max())) if finite.size else (None, None)

    blank = None
    masked = 0
    if min_test_pairs > 0:
        pairs_per_cell = (joint.assign(n=joint['n_pos'] + joint['n_neg'])
                          .groupby(['row', 'pair'])['n'].sum().unstack('pair'))
        blank = pairs_per_cell.reindex(index=group_mean.index,
                                       columns=group_mean.columns) < min_test_pairs
        masked = int(blank.to_numpy(bool).sum())
    threshold_note = (f'; cells under {min_test_pairs} test pairs left blank ({masked})'
                      if min_test_pairs > 0 else '')
    display = metric.replace('_', ' ')

    plot_metric_grid(
        group_mean, group_sd, group_folds, out_png=out_dir / f'metrics_by_group_{metric}.png',
        title=f'{display} per metadata group, {population}\nfold mean over fold SD'
              f'{threshold_note}',
        ylabel='Metadata group', cbar_label=f'{display} (fold mean)', scale=scale, blank=blank)
    plot_metric_grid(
        axis_mean, axis_sd, axis_folds, out_png=out_dir / f'metrics_by_axis_{metric}.png',
        title=f'{display} per host, HN subtype and year, {population}\n'
              f'each row pools every metadata group with that value',
        ylabel='Metadata value', cbar_label=f'{display} (fold mean)', scale=scale)


def main() -> None:
    """Write the per-group counts, the per-group test metrics, or both, and their figures.

    Returns:
        None. Exits non-zero when neither manifest is given or a table cannot be built.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset_manifest', type=Path, default=None,
                        help='dataset manifest to count the saved pair files of')
    parser.add_argument('--run_manifest', type=Path, default=None,
                        help='training run manifest whose fits to score per metadata group')
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--metric', default='f1_macro', choices=list(GROUP_METRICS),
                        help='the metric the figures draw (default: f1_macro)')
    parser.add_argument('--min_test_pairs', type=int, default=0,
                        help='mask figure cells with fewer test pairs than this, summed over '
                             'folds (default: 0, mask nothing)')
    args = parser.parse_args()
    if args.dataset_manifest is None and args.run_manifest is None:
        parser.error('give --dataset_manifest, --run_manifest, or both')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_manifest is not None:
        counts = metadata_group_counts(load_dataset_manifest(args.dataset_manifest))
        path = args.out_dir / 'metadata_group_counts.csv'
        counts.to_csv(path, index=False)
        print(f"Wrote {path} ({len(counts):,} rows, "
              f"{counts[list(GROUP_AXES)].drop_duplicates().shape[0]} metadata groups)")
        population = ', '.join(sorted(set(counts['population'])))
        plot_count_grid(counts, out_png=args.out_dir / 'counts_by_group.png',
                        title=f'Test positives per metadata group, {population}\n'
                              f'summed over folds')
    if args.run_manifest is not None:
        metrics = test_metrics_by_metadata(args.run_manifest)
        path = args.out_dir / 'test_metrics_by_metadata.csv'
        metrics.to_csv(path, index=False)
        joint = metrics[metrics['group_by'] == 'joint']
        print(f"Wrote {path} ({len(metrics):,} rows, "
              f"{joint[list(GROUP_AXES)].drop_duplicates().shape[0]} metadata groups)")
        write_metric_figures(metrics, args.out_dir, args.metric,
                             min_test_pairs=args.min_test_pairs)
    print('Done.')


if __name__ == '__main__':
    main()
