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
    metadata_group_counts.csv      with `--dataset_manifest`
    test_metrics_by_metadata.csv   with `--run_manifest`
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


def main() -> None:
    """Write the per-group counts, the per-group test metrics, or both.

    Returns:
        None. Exits non-zero when neither manifest is given or a table cannot be built.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset_manifest', type=Path, default=None,
                        help='dataset manifest to count the saved pair files of')
    parser.add_argument('--run_manifest', type=Path, default=None,
                        help='training run manifest whose fits to score per metadata group')
    parser.add_argument('--out_dir', type=Path, required=True)
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
    if args.run_manifest is not None:
        metrics = test_metrics_by_metadata(args.run_manifest)
        path = args.out_dir / 'test_metrics_by_metadata.csv'
        metrics.to_csv(path, index=False)
        joint = metrics[metrics['group_by'] == 'joint']
        print(f"Wrote {path} ({len(metrics):,} rows, "
              f"{joint[list(GROUP_AXES)].drop_duplicates().shape[0]} metadata groups)")
    print('Done.')


if __name__ == '__main__':
    main()
