"""Check built pair datasets against their manifest, the experiment spec and the capacity audit.

Run between building the pair datasets with `scripts/run_allpairs_datasets.py` and training on
them. A dataset that holds the right number of rows built from the wrong features, or the right
features built from the wrong population, passes every check a training run makes and produces a
plausible score.

How?

- The datasets come from the dataset manifest, not from a directory glob, and the expected pair
  set is the manifest's pair filter: exactly one dataset per selected pair and none besides. A
  manifest built without a pair filter must cover every schema pair the capacity table holds for
  the population.
- The experiment spec must be the one the manifest recorded. Its `expect` section is checked
  against each dataset's `resolved_config.yaml`, and its fold count against the manifest: a
  dataset must hold exactly the split directories that fold count implies.
- Every file the manifest records, the datasets' own files and the site-feature cache files, must
  still have its recorded checksum.
- Each run's pair and population come from its own `resolved_config.yaml`, so a directory renamed
  by hand cannot make a run look like a pair it is not. The capacity table must hold exactly one
  row for that population and pair.
- Counts come from what the builder already persisted, so nothing is recomputed:
  `Eligible isolates` from each split's `duplicate_stats.json` `pos_dedup.n_pos_before_dedup`, and
  the other four from `positive_pair_selection.json`. The split files must agree with each other,
  since that count is a property of the population rather than of a fold.
- Each protein's site-feature cache must load the way a fit loads it. A pinned cache must have
  been built at the pinned length the run uses, because a cache built against an earlier pin has
  the right file name and the wrong columns. An aligned cache must still match the final
  alignment beside it.
- With balanced-usage negatives at a 1:1 ratio over Hopcroft-Karp positives, every split's
  negatives must use each positive sequence exactly once per slot (`check_balanced_usage`).
- With matched balanced-usage negatives, every saved negative must join two isolates with equal
  values on each matching axis. The check reads both endpoints' metadata columns, not the
  negative's `neg_regime` label (`check_matched_negatives`).
- With an active positive-pair selection method, the test splits, pooled, must hold each selected
  positive exactly once, less the positives of the metadata groups the builder dropped when
  `dataset.min_metadata_group_positives` is set (`check_selected_test_positives`).
- With `--reference_dataset_prefix`, each pair's positive test pair keys, pooled over its folds,
  must equal those of the reference dataset of the same pair. Fold assignments and sampled
  negatives may differ.

CLI:
    python -m src.analysis.audit_pair_datasets \
        --dataset_manifest data/datasets/flu/July_2025/runs/<prefix>_manifest.json \
        --experiment_spec conf/experiments/human_h3n2_2024_28p_codon_matched_cv5.yaml \
        --reference_dataset_prefix exp2_28p_codon_balanced_20261007_180037_

Notes:

- Exits non-zero if any check fails, so it can gate a training script.
- The capacity CSV is the one the manifest recorded. Its `population` and `Schema pair` columns
  are the join key, and it is read with `keep_default_na=False, na_values=['']`, since `NA` is
  Neuraminidase.

Outputs:
    A per-run table on stdout, one line per failed check, and a summary count of failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets._negative_regime_sampling import DEFAULT_AXES  # noqa: E402
from src.datasets.dataset_segment_pairs_v2 import balanced_sampler_name  # noqa: E402
from src.utils.config_hydra import get_function_short_name_map  # noqa: E402
from src.utils.dataset_manifest import (  # noqa: E402
    changed_files,
    fold_count_failures,
    load_dataset_manifest,
    pair_token,
    read_split_dirs,
)
from src.utils.experiment_spec import (  # noqa: E402
    expected_setting_failures,
    load_experiment_spec,
)
from src.utils.experiment_utils import file_sha256  # noqa: E402
from src.utils.metadata_enrichment import population_label  # noqa: E402
from src.utils.site_utils import load_configured_site_cache  # noqa: E402

DATASET_RUNS = PROJ / 'data/datasets/flu/July_2025/runs'

# capacity column -> where the builder persisted the same count
COUNT_SOURCES = {
    'Unique positives': ('positive_pair_selection.json', 'input_pairs'),
    'Unique slot-A': ('positive_pair_selection.json', 'input_unique_a'),
    'Unique slot-B': ('positive_pair_selection.json', 'input_unique_b'),
    'HK selected': ('positive_pair_selection.json', 'selected_pairs'),
}


def read_counts(run_dir: Path, split_dirs: list) -> dict:
    """The five capacity counts a built dataset persisted.

    Args:
      run_dir: a dataset run directory, holding `positive_pair_selection.json`.
      split_dirs: its split directories, each holding `duplicate_stats.json`.

    Returns:
      Capacity column name -> count.

    Raises:
      ValueError: a required file is absent, or the split files disagree on the pre-dedup count.
    """
    selection_path = run_dir / 'positive_pair_selection.json'
    if not selection_path.exists():
        raise ValueError(f"read_counts: {selection_path} is absent; was the run interrupted?")
    selection = json.loads(selection_path.read_text())
    counts = {column: selection[key] for column, (_, key) in COUNT_SOURCES.items()}

    # One count per split, all of the same population-level quantity, so disagreement is a bug
    # rather than something to average over.
    split_counts = set()
    for split_dir in split_dirs:
        stats = json.loads((run_dir / split_dir / 'duplicate_stats.json').read_text())
        split_counts.add(stats['pos_dedup']['n_pos_before_dedup'])
    if len(split_counts) != 1:
        raise ValueError(
            f"read_counts: {run_dir.name} splits disagree on n_pos_before_dedup: "
            f"{sorted(split_counts)}.")
    counts['Eligible isolates'] = split_counts.pop()
    return counts


def run_pair_and_population(config) -> tuple[str, str]:
    """The pair label and population label a dataset was built for.

    Args:
      config: the run's loaded `resolved_config.yaml`.

    Returns:
      The pair label, e.g. `PB2-HA`, and the population label, e.g. `Human-H3N2-2024`, built the
      way the capacity table builds its `population` column.
    """
    short_of = get_function_short_name_map(config)
    pair = '-'.join(short_of[f] for f in config.dataset.schema_pair)
    population = population_label(
        host=config.dataset.host, hn_subtype=config.dataset.hn_subtype,
        year=config.dataset.year, year_range=config.dataset.year_range,
        geo_location=config.dataset.get('geo_location'), passage=config.dataset.get('passage'))
    return pair, population


def check_caches(config) -> list:
    """Site-feature caches that are missing or fail their reader's checks.

    Each protein's cache is loaded the way a fit loads it, with `load_configured_site_cache`. A
    pinned cache must exist at the run's pin, and its metadata must record that pin, since only
    the metadata is written from the build itself. An aligned cache must pass
    `read_aligned_codon_cache`, which checks it against the final alignment beside it.

    Args:
      config: the run's loaded `resolved_config.yaml`.

    Returns:
      One message per protein whose cache is missing or fails a check; empty when every cache
      loads.
    """
    short_of = get_function_short_name_map(config)
    problems = []
    for function in config.dataset.schema_pair:
        protein = short_of[function]
        try:
            load_configured_site_cache(config, protein)
        except (OSError, ValueError, KeyError) as error:
            problems.append(f'{protein}: {type(error).__name__}: {error}')
    return problems


def check_balanced_usage(run_dir: Path, split_dirs: list, hash_col: str) -> list[str]:
    """Check one negative use per positive sequence in every split.

    Args:
      run_dir: Dataset run directory.
      split_dirs: its split directories, each holding train, val and test pairs.
      hash_col: Per-slot hash field before its `_a` or `_b` suffix.

    Returns:
      One failure message per broken split or pair exclusion.
    """
    failures = []
    for split_dir in split_dirs:
        fold_dir = run_dir / split_dir
        observed = set(pd.read_csv(fold_dir / 'cooccurring_sequence_pairs.csv',
                                   usecols=['pair_key'], keep_default_na=False,
                                   na_values=['']).pair_key)
        seen_neg = set()
        for split in ('train', 'val', 'test'):
            columns = ['pair_key', 'label', f'{hash_col}_a', f'{hash_col}_b']
            pairs = pd.read_csv(fold_dir / f'{split}_pairs.csv', usecols=columns,
                                keep_default_na=False, na_values=[''])
            pos = pairs.loc[pairs.label == 1]
            neg = pairs.loc[pairs.label == 0]
            name = f'{split_dir}/{split}'
            if len(pos) != len(neg):
                failures.append(f'{name}: {len(pos)} positives, {len(neg)} negatives')
            for slot in ('a', 'b'):
                col = f'{hash_col}_{slot}'
                if Counter(pos[col]) != Counter(neg[col]):
                    failures.append(f'{name}: {slot} sequence-use counts differ by class')
            keys = set(neg.pair_key)
            if len(keys) != len(neg):
                failures.append(f'{name}: duplicate negative pair keys')
            if keys & observed:
                failures.append(f'{name}: {len(keys & observed)} observed pairs used as negatives')
            if keys & seen_neg:
                failures.append(f'{name}: {len(keys & seen_neg)} negative pairs reused across splits')
            seen_neg.update(keys)
    return failures


def check_matched_negatives(run_dir: Path, split_dirs: list, axes: list) -> list[str]:
    """Check that every saved negative joins two isolates with equal values on each matching axis.

    Reads each negative's `<axis>_a` and `<axis>_b` columns, which the builder fills from each
    endpoint's own isolate, and not its `neg_regime` label. Matched negatives use exact years, so
    years are compared as values. A missing value on either side counts as a difference.

    Args:
      run_dir: a dataset run directory built with matched balanced-usage negatives.
      split_dirs: its split directories, each holding train, val and test pairs.
      axes: the matching axes, `dataset.negative_sampling.axes`.

    Returns:
      One message per split file holding a negative whose endpoints differ; empty when none do.
    """
    failures = []
    columns = ['label'] + [f'{axis}_{slot}' for axis in axes for slot in ('a', 'b')]
    for split_dir in split_dirs:
        for split in ('train', 'val', 'test'):
            pairs = pd.read_csv(run_dir / split_dir / f'{split}_pairs.csv', usecols=columns,
                                keep_default_na=False, na_values=[''])
            negatives = pairs.loc[pairs.label == 0]
            # NaN != NaN, so a missing value counts as a difference.
            differing = {axis: int((negatives[f'{axis}_a'] != negatives[f'{axis}_b']).sum())
                         for axis in axes}
            differing = {axis: count for axis, count in differing.items() if count}
            if differing:
                failures.append(f'{split_dir}/{split}: negatives whose endpoints differ, by axis: '
                                f'{differing}')
    return failures


def check_selected_test_positives(run_dir: Path, split_dirs: list, config) -> list[str]:
    """Check that the test splits hold each selected positive of the kept metadata groups once.

    The selected positives are the pair keys of `positive_pair_selection.csv`. With
    `dataset.min_metadata_group_positives` set, the builder drops the metadata groups listed in
    `metadata_groups_dropped.json` before making folds. The test splits must then hold no
    positive of a dropped group, and as many positives as were selected minus those dropped. A
    test positive's group is read from its `<axis>_a` columns, for the axes the file names.
    Without the minimum, that file must be absent and the test splits must hold every selected
    positive.

    Args:
      run_dir: a dataset run directory built with an active positive-pair selection method.
      split_dirs: its split directories, each holding `test_pairs.csv`.
      config: the run's loaded `resolved_config.yaml`.

    Returns:
      One message per failed check; empty when the pooled test positives are exactly the
      selected positives of the kept groups, each once.
    """
    failures = []
    selected_keys = set(pd.read_csv(run_dir / 'positive_pair_selection.csv', usecols=['pair_key'],
                                    keep_default_na=False, na_values=['']).pair_key)
    minimum = config.dataset.get('min_metadata_group_positives')
    drop_path = run_dir / 'metadata_groups_dropped.json'
    if minimum is None:
        if drop_path.exists():
            failures.append(f'{drop_path.name} exists, but dataset.min_metadata_group_positives '
                            f'is null')
        axes, dropped_groups, positives_dropped = [], set(), 0
    else:
        if not drop_path.exists():
            failures.append(f'{drop_path.name} is absent, but dataset.min_metadata_group_positives '
                            f'is {minimum}')
            return failures
        record = json.loads(drop_path.read_text())
        if record['min_metadata_group_positives'] != minimum:
            failures.append(f"{drop_path.name} records a minimum of "
                            f"{record['min_metadata_group_positives']}, the config {minimum}")
        if record['positives_in'] != len(selected_keys):
            failures.append(f"{drop_path.name} counts {record['positives_in']:,} selected "
                            f"positives, positive_pair_selection.csv holds {len(selected_keys):,}")
        axes = record['axes']
        dropped_groups = {tuple(entry['group']) for entry in record['dropped_groups']}
        positives_dropped = record['positives_dropped']

    # Pool the test positives over the splits, counting repeats and those of dropped groups.
    group_columns = [f'{axis}_a' for axis in axes]
    test_key_counts = Counter()
    in_dropped_groups = 0
    for split_dir in split_dirs:
        test = pd.read_csv(run_dir / split_dir / 'test_pairs.csv',
                           usecols=['pair_key', 'label'] + group_columns,
                           keep_default_na=False, na_values=[''])
        positives = test.loc[test.label == 1]
        test_key_counts.update(positives.pair_key)
        groups = positives[group_columns].itertuples(index=False, name=None)
        in_dropped_groups += sum(group in dropped_groups for group in groups)

    repeated = sum(count > 1 for count in test_key_counts.values())
    if repeated:
        failures.append(f'{repeated:,} positives appear in more than one test split')
    not_selected = len(set(test_key_counts) - selected_keys)
    if not_selected:
        failures.append(f'{not_selected:,} test positives are not in positive_pair_selection.csv')
    if in_dropped_groups:
        failures.append(f'{in_dropped_groups:,} test positives belong to a dropped metadata group')
    expected = len(selected_keys) - positives_dropped
    if len(test_key_counts) != expected:
        failures.append(f'the test splits hold {len(test_key_counts):,} positives, expected '
                        f'{expected:,} ({len(selected_keys):,} selected minus '
                        f'{positives_dropped:,} dropped)')
    return failures


def positive_test_keys(run_dir: Path, split_dirs: list) -> set:
    """The positive pair keys of every split's test set, pooled.

    Args:
      run_dir: a dataset run directory.
      split_dirs: its split directories.

    Returns:
      The union of the positive `pair_key`s in each split's `test_pairs.csv`.
    """
    keys = set()
    for split_dir in split_dirs:
        test = pd.read_csv(run_dir / split_dir / 'test_pairs.csv', usecols=['pair_key', 'label'],
                           keep_default_na=False, na_values=[''])
        keys.update(test.loc[test.label == 1, 'pair_key'])
    return keys


def audit_run(entry: dict, capacity: pd.DataFrame, spec: dict,
              reference_dir: Optional[Path]) -> tuple:
    """Check one run's files, counts, settings, caches and negatives.

    Args:
      entry: the run's entry in the dataset manifest's `datasets` section.
      capacity: the capacity table, with `population` and `Schema pair` columns.
      spec: the experiment spec, from `load_experiment_spec`.
      reference_dir: the reference dataset of the same pair, or None to skip that comparison.

    Returns:
      The pair label and the failure messages; the list is empty when the run passes.
    """
    run_dir = Path(entry['dataset_dir'])
    failures = changed_files(entry['files'], base=run_dir)
    config = OmegaConf.load(run_dir / 'resolved_config.yaml')
    pair, population = run_pair_and_population(config)
    if pair_token(pair) != entry['pair']:
        failures.append(f"the run holds {pair}, the manifest lists it as {entry['pair']}")
    if config.dataset.n_folds != entry['n_folds']:
        failures.append(f"resolved_config.yaml has n_folds={config.dataset.n_folds}, the manifest "
                        f"{entry['n_folds']}")
    split_dirs = entry['split_dirs']
    if read_split_dirs(run_dir) != split_dirs:
        failures.append(f'the run holds splits {read_split_dirs(run_dir)}, the manifest '
                        f'lists {split_dirs}')

    rows = capacity[(capacity['population'] == population) & (capacity['Schema pair'] == pair)]
    if len(rows) != 1:
        failures.append(f'the capacity table has {len(rows)} rows for {population} {pair}, '
                        f'expected exactly 1')
    else:
        expected_counts = rows.iloc[0]
        for column, count in read_counts(run_dir, split_dirs).items():
            if count != expected_counts[column]:
                failures.append(f'{column}: run has {count:,}, capacity says '
                                f'{expected_counts[column]:,}')

    failures.extend(expected_setting_failures(config, spec['expect']))
    failures.extend(check_caches(config))
    if (config.dataset.split_strategy.negative_scope == 'balanced'
            and config.dataset.positive_pair_selection.method == 'hopcroft_karp'
            and float(config.dataset.neg_to_pos_ratio) == 1.0):
        hash_col = {'aa': 'prot_hash', 'nt_cds': 'cds_dna_hash',
                    'nt_ctg': 'ctg_dna_hash'}[config.dataset.split_strategy.pair_key_alphabet]
        failures.extend(check_balanced_usage(run_dir, split_dirs, hash_col))
    if balanced_sampler_name(config) == 'matched_balanced_usage_negatives':
        axes = list(config.dataset.negative_sampling.get('axes') or DEFAULT_AXES)
        failures.extend(check_matched_negatives(run_dir, split_dirs, axes))
    if config.dataset.positive_pair_selection.method != 'all':
        failures.extend(check_selected_test_positives(run_dir, split_dirs, config))

    if reference_dir is not None:
        ours = positive_test_keys(run_dir, split_dirs)
        theirs = positive_test_keys(reference_dir, read_split_dirs(reference_dir))
        if ours != theirs:
            failures.append(f'positive test pair keys differ from {reference_dir.name}: '
                            f'{len(ours - theirs)} only here, {len(theirs - ours)} only there')
    return pair, failures


def manifest_failures(manifest: dict, spec: dict, capacity: pd.DataFrame) -> list:
    """Messages for problems that concern the manifest as a whole rather than one run.

    Args:
      manifest: from `load_dataset_manifest`.
      spec: from `load_experiment_spec`.
      capacity: the capacity table.

    Returns:
      One message per problem; empty when the manifest is complete and consistent.
    """
    failures = []
    if manifest['experiment_spec']['sha256'] != spec['sha256']:
        failures.append(f"the experiment spec differs from the one the datasets were built with "
                        f"({manifest['experiment_spec']['path']})")
    failures.extend(fold_count_failures(manifest, spec))
    failures.extend(changed_files(manifest['cache_files']))
    failures.extend(f'{bundle}: build exited {code}'
                    for bundle, code in manifest['results'].items() if code != 0)

    # Exactly one dataset per selected pair, and none besides.
    built = Counter(entry['pair'] for entry in manifest['datasets'].values())
    expected = (set(manifest['pairs']) if manifest['pairs'] is not None
                else {bundle[len(manifest['bundle_prefix']):]
                      for bundle in manifest['dataset_directories']})
    failures.extend(f'{pair}: {count} datasets' for pair, count in built.items() if count > 1)
    failures.extend(f'{pair}: no dataset' for pair in sorted(expected - set(built)))
    failures.extend(f'{pair}: not a selected pair' for pair in sorted(set(built) - expected))

    # Without a pair filter, the datasets must cover every pair the capacity table holds for the
    # population.
    if manifest['pairs'] is None and manifest['datasets']:
        first = next(iter(manifest['datasets'].values()))
        _, population = run_pair_and_population(
            OmegaConf.load(Path(first['dataset_dir']) / 'resolved_config.yaml'))
        capacity_pairs = {pair_token(label) for label in
                          capacity.loc[capacity['population'] == population, 'Schema pair']}
        missing = sorted(capacity_pairs - set(built))
        if missing:
            failures.append(f'capacity pairs of {population} without a dataset: {missing}')
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset_manifest', type=Path, required=True,
                        help='the {prefix}_manifest.json written by scripts/run_allpairs_datasets.py')
    parser.add_argument('--experiment_spec', type=Path, required=True,
                        help='the experiment spec the datasets were built with')
    parser.add_argument('--reference_dataset_prefix', default=None,
                        help='dataset prefix whose {prefix}{pair} datasets must hold the same '
                             'positive test pair keys, pooled over folds')
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.dataset_manifest)
    spec = load_experiment_spec(args.experiment_spec)
    capacity_path = Path(manifest['capacity_csv']['path'])
    capacity = pd.read_csv(capacity_path, keep_default_na=False, na_values=[''])
    missing_columns = {'population', 'Schema pair'} - set(capacity.columns)
    if missing_columns:
        raise ValueError(f"main: {capacity_path} has no {sorted(missing_columns)} columns; its "
                         f"columns are {list(capacity.columns)}.")

    failed = 0
    global_failures = manifest_failures(manifest, spec, capacity)
    if file_sha256(capacity_path) != manifest['capacity_csv']['sha256']:
        global_failures.append(f'{capacity_path} changed since the manifest was written')
    for message in global_failures:
        print(f'FAIL manifest: {message}')
    failed += bool(global_failures)

    for bundle, entry in sorted(manifest['datasets'].items()):
        run_dir = Path(entry['dataset_dir'])
        reference_dir = (DATASET_RUNS / f"{args.reference_dataset_prefix}{entry['pair']}"
                         if args.reference_dataset_prefix else None)
        # One unreadable run must not abandon the rest, since the point of the audit is to see
        # every run before any training starts.
        try:
            pair, failures = audit_run(entry, capacity, spec, reference_dir)
        except Exception as error:
            failed += 1
            print(f'FAIL {"?":<10} {run_dir.name}\n       {type(error).__name__}: {error}')
            continue
        if failures:
            failed += 1
        print(f'{"FAIL" if failures else "OK  "} {pair:<10} {run_dir.name}')
        for message in failures:
            print(f'       {message}')

    n_runs = len(manifest['datasets'])
    print(f"\nAudited {n_runs} runs against {Path(spec['path']).name} and {capacity_path.name}: "
          f"{failed} failures (counting the manifest as one).")
    if failed:
        sys.exit(1)
    print('Done.')


if __name__ == '__main__':
    main()
