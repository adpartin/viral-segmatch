"""Check built pair datasets against the capacity audit and the configuration they claim.

Run between building the 28 datasets and training on them, as the gate named in "Dataset checks
every experiment must pass" in docs/plans/2026-09-14_codon_site_features_plan.md. A dataset that
holds the right number of rows built from the wrong features, or the right features built from the
wrong population, passes every check a training run makes and produces a plausible score.

How?

- Each run's pair comes from its own `resolved_config.yaml`, not from its directory name, so a
  directory renamed by hand cannot make a run look like a pair it is not.
- Counts come from what the builder already persisted, so nothing is recomputed:
  `Eligible isolates` from `fold_*/duplicate_stats.json` `pos_dedup.n_pos_before_dedup`, and the
  other four from `positive_pair_selection.json`. The fold files must agree with each other, since
  that count is a property of the population rather than of a fold.
- Settings come from `resolved_config.yaml`. The ones in `DATASET_EXPECT` decide what the rows
  are, so a mismatch fails. The ones in `TRAINING_EXPECT` are applied when a model is trained,
  never when a dataset is built, so a mismatch is reported as a note.
- Each protein's site-feature cache is checked to have been built at the pinned length the run
  uses. A cache built against an earlier pin has the right file name and the wrong columns.

CLI:
    python -m src.analysis.audit_pair_datasets --runs_glob 'exp3_28p_codon_*'
    python -m src.analysis.audit_pair_datasets --runs_glob 'exp3_28p_codon_*' --expect_pairs 28

Notes:

- Exits non-zero if any check fails, so it can gate a training script.
- `--capacity_csv` defaults to Experiment 2's output. Its `Schema pair` column is the join key.
- Read that CSV with `keep_default_na=False, na_values=['']`, since `NA` is Neuraminidase.

Outputs:
    A per-run table on stdout, one line per check group, and a summary count of failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.config_hydra import get_function_short_name_map  # noqa: E402

# capacity column -> where the builder persisted the same count
COUNT_SOURCES = {
    'Unique positives': ('positive_pair_selection.json', 'input_pairs'),
    'Unique slot-A': ('positive_pair_selection.json', 'input_unique_a'),
    'Unique slot-B': ('positive_pair_selection.json', 'input_unique_b'),
    'HK selected': ('positive_pair_selection.json', 'selected_pairs'),
}

# What the rows are: change any of these and the dataset itself is a different dataset. Read from
# the run's own resolved_config.yaml, and a mismatch fails the audit.
DATASET_EXPECT = {
    'dataset.host': ['Human'],
    'dataset.hn_subtype': ['H3N2'],
    'dataset.year': [2024],
    'dataset.n_folds': 4,
    'dataset.split_strategy.pair_key_alphabet': 'nt_cds',
    'dataset.split_strategy.negative_scope': 'within_fold',
    'dataset.positive_pair_selection.method': 'hopcroft_karp',
    'dataset.require_complete_cds_at_pinned_length': True,
    'dataset.neg_to_pos_ratio': 1.0,
}

# What the rows will be turned into. A dataset directory holds pair keys and labels, never feature
# matrices, so these are decided by whichever bundle the training run uses and do not change the
# dataset. They are reported rather than failed, because a dataset built from a k-mer bundle is
# still the right dataset to train codon features on.
TRAINING_EXPECT = {
    'training.feature_source': 'site',
    'site.unit': 'codon',
    'site.encoding': 'ordinal',
}


def read_counts(run_dir: Path) -> dict:
    """The five capacity counts a built dataset persisted.

    Args:
      run_dir: a dataset run directory, holding `positive_pair_selection.json` and `fold_*/`.

    Returns:
      Capacity column name -> count.

    Raises:
      ValueError: a required file is absent, or the fold files disagree on the pre-dedup count.
    """
    selection_path = run_dir / 'positive_pair_selection.json'
    if not selection_path.exists():
        raise ValueError(f"read_counts: {selection_path} is absent; was the run interrupted?")
    selection = json.loads(selection_path.read_text())
    counts = {column: selection[key] for column, (_, key) in COUNT_SOURCES.items()}

    # One count per fold, all of the same population-level quantity, so disagreement is a bug
    # rather than something to average over.
    fold_counts = set()
    for stats_path in sorted(run_dir.glob('fold_*/duplicate_stats.json')):
        fold_counts.add(json.loads(stats_path.read_text())['pos_dedup']['n_pos_before_dedup'])
    if len(fold_counts) != 1:
        raise ValueError(
            f"read_counts: {run_dir.name} folds disagree on n_pos_before_dedup: "
            f"{sorted(fold_counts)}.")
    counts['Eligible isolates'] = fold_counts.pop()
    return counts


def read_settings(config, keys: list) -> dict:
    """The values a run resolved for each dotted key.

    Args:
      config: the run's loaded `resolved_config.yaml`.
      keys: dotted paths, e.g. `dataset.n_folds`.

    Returns:
      Dotted path -> value, with list values converted so they compare against plain lists.
    """
    found = {}
    for key in keys:
        value = OmegaConf.select(config, key)
        found[key] = list(value) if OmegaConf.is_list(value) else value
    return found


def check_caches(config, unit: str) -> list:
    """Site-feature caches that are absent or built at a different pinned length.

    A cache keeps its file name when a pin changes, so a run can load columns that no longer mean
    the position they are indexed by. The pin is recorded in each cache's metadata.

    Args:
      config: the run's loaded `resolved_config.yaml`.
      unit: site unit the run uses, e.g. `codon`.

    Returns:
      One message per protein whose cache is missing or disagrees with the run's pin; empty when
      every pinned protein checks out.
    """
    cache_dir = (PROJ / 'data/embeddings' / config.virus_name / config.virus.data_version)
    short_of = get_function_short_name_map(config)
    pins = {str(k): int(v['nt']) for k, v in dict(config.virus.cds_length).items()}

    problems = []
    for function in config.dataset.schema_pair:
        protein = short_of[function]
        metadata_path = cache_dir / f'site_features_{unit}_{protein}_metadata.json'
        if not metadata_path.exists():
            problems.append(f'{protein}: no {unit} cache at {metadata_path.name}')
            continue
        cached_nt = json.loads(metadata_path.read_text())['pinned_nt']
        if cached_nt != pins[protein]:
            problems.append(
                f'{protein}: cache built at {cached_nt} nt, run pins {pins[protein]} nt')
    return problems


def audit_run(run_dir: Path, capacity: pd.DataFrame, expect: dict) -> tuple:
    """Check one run's counts, settings and caches.

    Args:
      run_dir: the dataset run directory.
      capacity: the capacity table, indexed by `Schema pair`.
      expect: dotted setting path -> required value, for the settings that decide what the rows
          are.

    Returns:
      The pair label, the failure messages, and the notes about settings that do not determine the
      dataset. Both lists are empty when the run matches in every respect.
    """
    config = OmegaConf.load(run_dir / 'resolved_config.yaml')
    short_of = get_function_short_name_map(config)
    pair = '-'.join(short_of[f] for f in config.dataset.schema_pair)

    failures = []
    if pair not in capacity.index:
        return pair, [f'{pair}: absent from the capacity table'], []

    expected_counts = capacity.loc[pair]
    for column, count in read_counts(run_dir).items():
        if count != expected_counts[column]:
            failures.append(f'{column}: run has {count:,}, capacity says '
                            f'{expected_counts[column]:,}')

    for key, value in read_settings(config, list(expect)).items():
        if value != expect[key]:
            failures.append(f'{key}: run has {value!r}, expected {expect[key]!r}')

    failures.extend(check_caches(config, str(config.site.unit)))

    notes = [f'{key}: run has {value!r}, Experiment 3 trains with {TRAINING_EXPECT[key]!r}'
             for key, value in read_settings(config, list(TRAINING_EXPECT)).items()
             if value != TRAINING_EXPECT[key]]
    return pair, failures, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs_dir', type=Path,
                        default=PROJ / 'data/datasets/flu/July_2025/runs')
    parser.add_argument('--runs_glob', default='exp3_28p_codon_*',
                        help='glob over --runs_dir selecting the run directories to audit')
    parser.add_argument('--capacity_csv', type=Path,
                        default=PROJ / 'results/flu/July_2025/pair_capacity_8_proteins'
                                       '/pair_capacity.csv')
    parser.add_argument('--expect_pairs', type=int, default=None,
                        help='fail unless exactly this many runs were audited')
    args = parser.parse_args()

    capacity = pd.read_csv(args.capacity_csv, keep_default_na=False, na_values=[''])
    if 'Schema pair' not in capacity.columns:
        raise ValueError(
            f"main: {args.capacity_csv} has no 'Schema pair' column; its columns are "
            f"{list(capacity.columns)}. A table written before that column was renamed cannot be "
            f"joined on.")
    capacity = capacity.set_index('Schema pair')

    run_dirs = sorted(d for d in args.runs_dir.glob(args.runs_glob) if d.is_dir())
    if not run_dirs:
        raise ValueError(f"main: no run directories match {args.runs_glob!r} in {args.runs_dir}.")

    failed = 0
    for run_dir in run_dirs:
        # One unreadable run must not abandon the rest, since the point of the audit is to see
        # every run before any training starts.
        try:
            pair, failures, notes = audit_run(run_dir, capacity, DATASET_EXPECT)
        except Exception as error:
            failed += 1
            print(f'FAIL {"?":<10} {run_dir.name}\n       {type(error).__name__}: {error}')
            continue
        if failures:
            failed += 1
        print(f'{"FAIL" if failures else "OK  "} {pair:<10} {run_dir.name}')
        for message in failures:
            print(f'       {message}')
        for message in notes:
            print(f'       note: {message}')

    print(f"\nAudited {len(run_dirs)} runs against {args.capacity_csv.name}: "
          f"{len(run_dirs) - failed} passed, {failed} failed.")
    if args.expect_pairs is not None and len(run_dirs) != args.expect_pairs:
        print(f"ERROR: expected {args.expect_pairs} runs, audited {len(run_dirs)}.")
        sys.exit(1)
    if failed:
        sys.exit(1)
    print('Done.')


if __name__ == '__main__':
    main()
