"""Build the 28 pair datasets of one bundle set under one dataset prefix, and write their manifest.

How?

- The bundles are the 28 child bundles `conf/bundles/{bundle_prefix}*.yaml`, and each pair token
  is the bundle name after the prefix. `--pairs` builds only the listed pairs, so a one-pair pilot
  runs the same path as the full screen.
- Every build receives the experiment spec's override list as `--override`. The spec's `expect`
  section is not checked here; `src/analysis/audit_pair_datasets.py` checks it.
- An existing dataset directory or manifest stops the script before anything is built, and the
  manifest is written once, after every build has finished. Its contents are described in
  `src/utils/dataset_manifest.py`.
- The Git commit must not change while the builds run, and every dataset's `cv_info.json` must
  record the commit the script started at.

CLI:
    python scripts/run_allpairs_datasets.py \
        --dataset-prefix h3n2_2024_28p_codon_cv5_20261010_120000_ \
        --experiment-spec conf/experiments/human_h3n2_2024_28p_codon_balanced_cv5.yaml \
        --capacity-csv results/flu/July_2025/pair_capacity_8_proteins/pair_capacity.csv \
        --pairs pb2_ha

Outputs:
    data/datasets/flu/July_2025/runs/{prefix}{pair}/        one dataset per pair
    data/datasets/flu/July_2025/runs/{prefix minus trailing _}_manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_hydra import get_function_short_name_map  # noqa: E402
from src.utils.dataset_manifest import (  # noqa: E402
    dataset_file_checksums,
    pair_token,
    read_split_dirs,
    site_cache_files,
)
from src.utils.experiment_spec import load_experiment_spec  # noqa: E402
from src.utils.experiment_utils import file_sha256, get_git_info  # noqa: E402

DATASET_RUNS = PROJECT_ROOT / 'data/datasets/flu/July_2025/runs'
N_PAIRS = 28


def select_bundles(bundle_prefix: str, pairs: Optional[list]) -> dict:
    """The child bundles to build, keyed by pair token.

    Args:
        bundle_prefix: bundle name minus the pair token, e.g. `flu_28p_codon_`.
        pairs: pair tokens to keep, or None for all of them.

    Returns:
        Pair token -> bundle name, sorted by pair token.

    Raises:
        ValueError: the prefix does not match exactly 28 bundles, or a requested pair has no
            bundle.
    """
    bundles = sorted(p.stem for p in (PROJECT_ROOT / 'conf/bundles').glob(f'{bundle_prefix}*.yaml'))
    if len(bundles) != N_PAIRS:
        raise ValueError(f'select_bundles: expected {N_PAIRS} child bundles matching '
                         f'{bundle_prefix}*.yaml; found {len(bundles)}')
    by_pair = {bundle[len(bundle_prefix):]: bundle for bundle in bundles}
    if pairs:
        unknown = sorted(set(pairs) - set(by_pair))
        if unknown:
            raise ValueError(f'select_bundles: no {bundle_prefix}* bundle for pairs {unknown}')
        by_pair = {pair: by_pair[pair] for pair in sorted(set(pairs))}
    return by_pair


def _build_one(bundle: str, output_dir: Path, overrides: list) -> tuple[str, int]:
    """Build one pair dataset and retain the Stage 3 log.

    Args:
        bundle: Child bundle name.
        output_dir: New dataset run directory; must not exist yet.
        overrides: dotlist entries passed to the builder as `--override`.

    Returns:
        Bundle name and process exit code.
    """
    output_dir.mkdir(parents=True)
    command = [sys.executable, str(PROJECT_ROOT / 'src/datasets/dataset_segment_pairs.py'),
               '--config_bundle', bundle, '--output_dir', str(output_dir)]
    if overrides:
        command += ['--override', *overrides]
    with (output_dir / 'stage3.log').open('w') as log:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log,
                                stderr=subprocess.STDOUT)
    return bundle, result.returncode


def describe_dataset(pair: str, dataset_dir: Path, commit: str) -> tuple[dict, list]:
    """The manifest entry for one built dataset, and the cache files its fits will read.

    Args:
        pair: the pair token the dataset was built for.
        dataset_dir: the dataset run directory.
        commit: the commit the builds started at.

    Returns:
        The entry (`pair`, `dataset_dir`, `n_folds`, `split_dirs`, `files`) and the absolute
        paths of the site-feature cache files.

    Raises:
        ValueError: the dataset's schema pair is not the pair its bundle is named for, or its
            `cv_info.json` records a different commit.
    """
    config = OmegaConf.load(dataset_dir / 'resolved_config.yaml')
    short_of = get_function_short_name_map(config)
    label = '-'.join(short_of[function] for function in config.dataset.schema_pair)
    if pair_token(label) != pair:
        raise ValueError(f'describe_dataset: {dataset_dir.name} holds {label}, not {pair}')

    cv_info_path = dataset_dir / 'cv_info.json'
    if cv_info_path.exists():
        built_at = json.loads(cv_info_path.read_text())['code']['commit']
        if built_at != commit:
            raise ValueError(f'describe_dataset: {dataset_dir.name} was built at {built_at}, '
                             f'the builds started at {commit}')

    split_dirs = read_split_dirs(dataset_dir)
    entry = {
        'pair': pair,
        'dataset_dir': str(dataset_dir),
        'n_folds': config.dataset.n_folds,
        'split_dirs': split_dirs,
        'files': dataset_file_checksums(dataset_dir, split_dirs),
    }
    return entry, site_cache_files(config)


def main() -> None:
    """Build the selected pair datasets and write their manifest.

    Returns:
        None. Exits non-zero when a build fails or the manifest cannot be completed.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset-prefix', required=True,
                        help='Run-directory prefix ending in _, including YYYYMMDD_HHMMSS')
    parser.add_argument('--experiment-spec', type=Path, required=True,
                        help='experiment spec YAML whose override list every build receives')
    parser.add_argument('--capacity-csv', type=Path, required=True,
                        help='pair-capacity table the audit will check these datasets against')
    parser.add_argument('--bundle-prefix', default='flu_28p_codon_',
                        help='bundle name minus the pair token (default: flu_28p_codon_)')
    parser.add_argument('--pairs', nargs='+', default=None,
                        help='pair tokens to build, e.g. pb2_ha; default all 28')
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error('--max-workers must be positive')
    if not args.dataset_prefix.endswith('_'):
        parser.error('--dataset-prefix must end in _')
    if not args.capacity_csv.exists():
        parser.error(f'--capacity-csv {args.capacity_csv} does not exist')

    spec = load_experiment_spec(args.experiment_spec)
    by_pair = select_bundles(args.bundle_prefix, args.pairs)
    jobs = {pair: (bundle, DATASET_RUNS / f'{args.dataset_prefix}{pair}')
            for pair, bundle in by_pair.items()}
    manifest_path = DATASET_RUNS / f'{args.dataset_prefix.rstrip("_")}_manifest.json'
    existing = [str(directory) for _, directory in jobs.values() if directory.exists()]
    if manifest_path.exists():
        existing.append(str(manifest_path))
    if existing:
        raise FileExistsError(f'Dataset outputs already exist: {existing}')

    code = get_git_info()
    print(f'Building {len(jobs)} datasets with {args.max_workers} workers at commit '
          f'{code["commit_short"]} ({code["status"]}); overrides: {spec["overrides"]}', flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(_build_one, bundle, directory, spec['overrides'])
                   for bundle, directory in jobs.values()]
        for future in as_completed(futures):
            bundle, exit_code = future.result()
            results[bundle] = exit_code
            print(f'{"Done." if exit_code == 0 else "ERROR:"} {bundle}: exit {exit_code}',
                  flush=True)

    # Describe every dataset that built, even when another failed, so the manifest shows what
    # exists; the failures still stop the script below.
    errors = []
    if get_git_info()['commit'] != code['commit']:
        errors.append('the commit changed while the datasets were being built')
    datasets, cache_paths = {}, set()
    for pair, (bundle, directory) in jobs.items():
        if results[bundle] != 0:
            errors.append(f'{bundle}: build exited {results[bundle]}; see {directory}/stage3.log')
            continue
        try:
            entry, caches = describe_dataset(pair, directory, code['commit'])
        except ValueError as error:
            errors.append(str(error))
            continue
        datasets[bundle] = entry
        cache_paths.update(caches)

    fold_counts = {entry['n_folds'] for entry in datasets.values()}
    if len(fold_counts) > 1:
        errors.append(f'the datasets disagree on n_folds: {sorted(fold_counts, key=str)}')
    missing_caches = sorted(str(path) for path in cache_paths if not path.exists())
    if missing_caches:
        errors.append(f'site-feature cache files are missing: {missing_caches}')
    cache_checksums = {str(path): file_sha256(path)
                       for path in sorted(cache_paths) if path.exists()}
    manifest = {
        'dataset_prefix': args.dataset_prefix,
        'results': results,
        'dataset_directories': {bundle: str(directory) for bundle, directory in jobs.values()},
        'experiment_spec': {'path': spec['path'], 'sha256': spec['sha256']},
        'capacity_csv': {'path': str(args.capacity_csv.resolve()),
                         'sha256': file_sha256(args.capacity_csv)},
        'overrides': spec['overrides'],
        'bundle_prefix': args.bundle_prefix,
        'pairs': sorted(args.pairs) if args.pairs else None,
        'n_folds': fold_counts.pop() if len(fold_counts) == 1 else None,
        'code': code,
        'datasets': datasets,
        'cache_files': cache_checksums,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if errors:
        print('\n'.join(f'ERROR: {message}' for message in errors), flush=True)
        raise SystemExit(f'{len(errors)} problems; manifest written to {manifest_path}')
    print(f'Done. Manifest: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
