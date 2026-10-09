#!/usr/bin/env python3
"""Redraw the balanced training negatives of every fold in a dataset manifest and fit each draw.

For every pair and fold, the training negatives are redrawn once per seed while the training
positives, the validation and test pairs and the model seed stay fixed. One model is fitted per
draw, and each pair's draws are then summarized. The steps are the existing scripts:
`resample_balanced_train_negatives.py` redraws, `train_pair_baselines.py` fits (through
`src/models/baseline_fits.py`), and `analyze_balanced_train_redraw.py` summarizes.

How?

- The pairs, datasets, split directories and fold count come from the dataset manifest; `--pairs`
  selects a subset. The experiment spec must be the one the manifest recorded, its fold count
  must match the manifest's, and every pair's planned config must match its `expect` section.
- The current commit must equal the dataset manifest's, and every dataset file and site-feature
  cache file must still have its recorded checksum.
- Before redrawing, each dataset must meet the redraw script's assumptions: Hopcroft-Karp
  positives keyed on the CDS hash, balanced-usage negatives at a 1:1 ratio, no metadata-matched
  negative sampling, and a passed one-use audit (`check_balanced_usage` in
  `src/analysis/audit_pair_datasets.py`). The redraw always draws balanced-usage negatives, so on
  a metadata-matched dataset it would produce unmatched ones.
- The first call writes `redraw_run_manifest.json` with the dataset manifest, the seeds, the
  commit and every redraw, model and analysis directory. As each fold's redraw finishes, the
  checksums of its files are added. A resume call names that file with `--resume`, reuses its run
  prefix and requires the current commit to equal the recorded one.
- An existing redraw directory is skipped only when its seed directories match the recorded seeds
  and all its files match the recorded checksums; an existing fit only when
  `existing_fit_problems` finds nothing, with `neg_redraw_seed` set to the draw's seed. Anything
  else stops the call without overwriting it.
- If two seeds of one fold draw identical sets of training negatives, the seed pair is recorded
  and the call fails.
- The analysis outputs are summaries and are rebuilt on every call; redraw directories and fits
  are never overwritten. The call exits non-zero when any redraw, fit or analysis fails.

CLI:
    python scripts/run_allpairs_balanced_redraws.py \
        --dataset_manifest data/datasets/flu/July_2025/runs/<prefix>_manifest.json \
        --experiment_spec conf/experiments/human_h3n2_2024_28p_codon_balanced_cv5.yaml \
        --seeds 42 43 --pairs pb2_ha --dry_run

    python scripts/run_allpairs_balanced_redraws.py \
        --resume models/flu/July_2025/training_runs/<name>/redraw_run_manifest.json

Outputs:
    data/datasets/{virus}/{version}/runs/{run_prefix}{pair}/fold_{k}/seed_{s}/   redraws
    models/{virus}/{version}/runs/{baseline}_{run_prefix}{pair}_fold{k}_seed{s}/  fits
    results/{virus}/{version}/{run_prefix}{pair}/                                 summaries
    models/{virus}/{version}/training_runs/{run_prefix}{baseline}/redraw_run_manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.audit_pair_datasets import check_balanced_usage  # noqa: E402
from src.models.baseline_fits import (  # noqa: E402
    concurrent_fits,
    existing_fit_problems,
    fit_command,
    run_fit,
)
from src.utils.config_hydra import compose_config, get_function_short_name_map  # noqa: E402
from src.utils.dataset_manifest import (  # noqa: E402
    SPLIT_FILES,
    changed_files,
    fold_count_failures,
    load_dataset_manifest,
    pair_cache_checksums,
)
from src.utils.experiment_spec import (  # noqa: E402
    expected_setting_failures,
    load_experiment_spec,
)
from src.utils.experiment_utils import file_sha256, get_git_info  # noqa: E402

REDRAW_SCRIPT = PROJECT_ROOT / 'scripts' / 'resample_balanced_train_negatives.py'
ANALYSIS_SCRIPT = PROJECT_ROOT / 'scripts' / 'analyze_balanced_train_redraw.py'
REDRAW_SUMMARY = 'redraw_summary.json'


def redraw_assumption_failures(dataset_dir: Path, split_dirs: list) -> list:
    """Reasons a built dataset does not meet the redraw script's assumptions.

    Args:
      dataset_dir: the dataset run directory.
      split_dirs: its split directories.

    Returns:
      One message per unmet assumption; empty when the dataset may be redrawn.
    """
    config = OmegaConf.load(dataset_dir / 'resolved_config.yaml')
    required = {
        'dataset.positive_pair_selection.method': 'hopcroft_karp',
        'dataset.split_strategy.pair_key_alphabet': 'nt_cds',
        'dataset.split_strategy.negative_scope': 'balanced',
        'dataset.neg_to_pos_ratio': 1.0,
        'dataset.negative_sampling': None,
    }
    failures = expected_setting_failures(config, required)
    if not failures:
        failures = check_balanced_usage(dataset_dir, split_dirs, 'cds_dna_hash')
    return failures


def plan_redraw_run(dataset_manifest: dict, spec: dict, baseline: str, pairs: list, seeds: list,
                    timestamp: str) -> dict:
    """The redraw run manifest for a first call.

    Args:
      dataset_manifest: from `load_dataset_manifest`.
      spec: from `load_experiment_spec`.
      baseline: baseline name.
      pairs: pair tokens to redraw.
      seeds: redraw seeds, the same for every fold.
      timestamp: `YYYYMMDD_HHMMSS` of this call.

    Returns:
      The manifest to write as `redraw_run_manifest.json`, with every fold's `redraw_files`
      still None.
    """
    by_pair = {entry['pair']: (bundle, entry)
               for bundle, entry in dataset_manifest['datasets'].items()}
    probe = compose_config(by_pair[pairs[0]][0], spec['overrides'])
    virus, version = probe.virus.virus_name, probe.virus.data_version
    run_prefix = f"{dataset_manifest['dataset_prefix']}negredraw_{timestamp}_"
    planned = {}
    for pair in pairs:
        bundle, entry = by_pair[pair]
        pair_dir = PROJECT_ROOT / 'data' / 'datasets' / virus / version / 'runs' / f'{run_prefix}{pair}'
        model_prefix = f'{baseline}_{run_prefix}{pair}'
        folds = {}
        for fold_id, split_dir in enumerate(entry['split_dirs']):
            redraw_dir = pair_dir / f'fold_{fold_id}'
            folds[str(fold_id)] = {
                'source_fold': str(Path(entry['dataset_dir']) / split_dir),
                'redraw_dir': str(redraw_dir),
                'redraw_files': None,
                'identical_seed_pairs': [],
                'seeds': {str(seed): {
                    'seed_dir': str(redraw_dir / f'seed_{seed}'),
                    'model_dir': str(PROJECT_ROOT / 'models' / virus / version / 'runs'
                                     / f'{model_prefix}_fold{fold_id}_seed{seed}')}
                          for seed in seeds},
            }
        planned[pair] = {
            'bundle': bundle, 'dataset_dir': entry['dataset_dir'],
            'analysis_dir': str(PROJECT_ROOT / 'results' / virus / version / f'{run_prefix}{pair}'),
            'analysis_run_prefix': model_prefix,
            'folds': folds,
        }
    run_manifest = {
        'dataset_manifest': {'path': dataset_manifest['manifest_path'],
                             'sha256': dataset_manifest['manifest_sha256']},
        'experiment_spec': {'path': spec['path'], 'sha256': spec['sha256']},
        'overrides': spec['overrides'],
        'baseline': baseline,
        'seeds': list(seeds),
        'run_prefix': run_prefix,
        'timestamp': timestamp,
        'code': get_git_info(),
        'pairs': planned,
    }
    return run_manifest


def redraw_files(redraw_dir: Path, seeds: list) -> dict:
    """Checksums of every file a fold's redraw wrote.

    Args:
      redraw_dir: the fold's redraw directory.
      seeds: the redraw seeds.

    Returns:
      Path relative to `redraw_dir` -> SHA-256, for each seed directory's `SPLIT_FILES` and for
      `redraw_summary.json`.
    """
    keys = [f'seed_{seed}/{name}' for seed in seeds for name in SPLIT_FILES] + [REDRAW_SUMMARY]
    checksums = {key: file_sha256(redraw_dir / key) for key in keys}
    return checksums


def identical_seed_pairs(redraw_dir: Path, seeds: list) -> list:
    """Pairs of seeds whose draws hold exactly the same training negatives.

    Args:
      redraw_dir: the fold's redraw directory.
      seeds: the redraw seeds.

    Returns:
      `[seed_a, seed_b]` lists, empty when every draw differs.
    """
    draws = {}
    for seed in seeds:
        train = pd.read_csv(redraw_dir / f'seed_{seed}' / 'train_pairs.csv',
                            usecols=['pair_key', 'label'], keep_default_na=False, na_values=[''])
        draws[seed] = frozenset(train.loc[train.label == 0, 'pair_key'])
    pairs = [[a, b] for i, a in enumerate(seeds) for b in seeds[i + 1:] if draws[a] == draws[b]]
    return pairs


def existing_redraw_problems(fold: dict, seeds: list) -> list:
    """Reasons an existing redraw directory is not the recorded, finished redraw.

    Args:
      fold: the fold's entry in the redraw run manifest.
      seeds: the recorded seeds.

    Returns:
      One message per problem; empty when the directory may be skipped.
    """
    redraw_dir = Path(fold['redraw_dir'])
    if fold['redraw_files'] is None:
        return [f'{redraw_dir}: exists but the run manifest records no finished redraw there']
    present = sorted(p.name for p in redraw_dir.glob('seed_*'))
    expected = sorted(f'seed_{seed}' for seed in seeds)
    if present != expected:
        return [f'{redraw_dir}: holds {present}, the run manifest records {expected}']
    problems = changed_files(fold['redraw_files'], base=redraw_dir)
    return problems


def run_redraw(fold: dict, seeds: list) -> int:
    """Run the redraw script for one fold, keeping its log next to the redraw directory.

    Args:
      fold: the fold's entry in the redraw run manifest.
      seeds: the redraw seeds.

    Returns:
      The script's exit code.
    """
    redraw_dir = Path(fold['redraw_dir'])
    redraw_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(REDRAW_SCRIPT), '--source-fold', fold['source_fold'],
               '--output-root', str(redraw_dir), '--seeds', *map(str, seeds)]
    with open(redraw_dir.parent / f'{redraw_dir.name}_redraw.log', 'w') as log_file:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log_file,
                                   stderr=subprocess.STDOUT)
    return completed.returncode


def fit_jobs(run_manifest: dict, dataset_manifest: dict, spec: dict) -> tuple[list, list]:
    """Every fit of the run, with its planned config, expected inputs and command.

    Args:
      run_manifest: the redraw run manifest, with `redraw_files` filled for every fold.
      dataset_manifest: from `load_dataset_manifest`.
      spec: from `load_experiment_spec`.

    Returns:
      The jobs, and failure messages for folds whose redraw is not recorded.
    """
    entries = {entry['pair']: entry for entry in dataset_manifest['datasets'].values()}
    jobs, failures = [], []
    for pair, plan in run_manifest['pairs'].items():
        planned_config = compose_config(plan['bundle'], spec['overrides'])
        dataset_config = Path(plan['dataset_dir']) / 'resolved_config.yaml'
        shared_inputs = {str(dataset_config.resolve()): entries[pair]['files']['resolved_config.yaml'],
                         **pair_cache_checksums(dataset_manifest, planned_config)}
        for fold_id, fold in plan['folds'].items():
            if fold['redraw_files'] is None:
                failures.append(f'{pair} fold {fold_id}: no finished redraw recorded')
                continue
            redraw_dir = Path(fold['redraw_dir'])
            for seed, paths in fold['seeds'].items():
                seed_dir = Path(paths['seed_dir'])
                expected_inputs = {str((redraw_dir / key).resolve()): checksum
                                   for key, checksum in fold['redraw_files'].items()
                                   if key.startswith(f'seed_{seed}/')}
                expected_inputs.update(shared_inputs)
                model_dir = Path(paths['model_dir'])
                jobs.append({
                    'pair': pair, 'fold_id': int(fold_id), 'seed': int(seed),
                    'model_dir': model_dir, 'planned_config': planned_config,
                    'expected_inputs': expected_inputs,
                    'command': fit_command(plan['bundle'], run_manifest['baseline'], seed_dir,
                                          model_dir, spec['overrides'], dataset_config,
                                          neg_redraw_seed=int(seed)),
                })
    return jobs, failures


def load_run_inputs(args) -> tuple[dict, dict, dict, Path]:
    """The redraw run manifest, dataset manifest and experiment spec for this call.

    Args:
      args: the parsed command line.

    Returns:
      The redraw run manifest, the dataset manifest, the experiment spec, and the path the run
      manifest is (or will be) written to.

    Raises:
      SystemExit: a recorded file changed, or the current commit differs from the recorded one.
    """
    if args.resume:
        run_manifest_path = args.resume.resolve()
        run_manifest = json.loads(run_manifest_path.read_text())
        dataset_manifest = load_dataset_manifest(Path(run_manifest['dataset_manifest']['path']))
        spec = load_experiment_spec(Path(run_manifest['experiment_spec']['path']))
        if dataset_manifest['manifest_sha256'] != run_manifest['dataset_manifest']['sha256']:
            raise SystemExit('ERROR: the dataset manifest changed since this redraw run started')
        if spec['sha256'] != run_manifest['experiment_spec']['sha256']:
            raise SystemExit('ERROR: the experiment spec changed since this redraw run started')
        if get_git_info()['commit'] != run_manifest['code']['commit']:
            raise SystemExit(f"ERROR: the current commit differs from the redraw run's "
                             f"{run_manifest['code']['commit']}; resume at that commit")
        return run_manifest, dataset_manifest, spec, run_manifest_path

    dataset_manifest = load_dataset_manifest(args.dataset_manifest)
    spec = load_experiment_spec(args.experiment_spec)
    if spec['sha256'] != dataset_manifest['experiment_spec']['sha256']:
        raise SystemExit(f"ERROR: {Path(spec['path']).name} differs from the experiment spec the "
                         f"datasets were built with ({dataset_manifest['experiment_spec']['path']})")
    available = sorted(entry['pair'] for entry in dataset_manifest['datasets'].values())
    pairs = sorted(set(args.pairs)) if args.pairs else available
    unknown = sorted(set(pairs) - set(available))
    if unknown:
        raise SystemExit(f'ERROR: the dataset manifest has no dataset for pairs {unknown}')
    if not pairs:
        raise SystemExit('ERROR: the dataset manifest lists no datasets')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_manifest = plan_redraw_run(dataset_manifest, spec, args.baseline, pairs, args.seeds,
                                   timestamp)
    first_model = next(iter(next(iter(run_manifest['pairs'].values()))['folds'].values()))
    models_base = Path(next(iter(first_model['seeds'].values()))['model_dir']).parents[1]
    run_name = f"{run_manifest['run_prefix']}{args.baseline}"
    run_manifest_path = models_base / 'training_runs' / run_name / 'redraw_run_manifest.json'
    return run_manifest, dataset_manifest, spec, run_manifest_path


def pair_label(bundle: str) -> str:
    """The pair label of a bundle, e.g. `PB2-HA`, for figure titles.

    Args:
      bundle: the pair's child bundle.

    Returns:
      Slot-A and slot-B short names joined by `-`.
    """
    config = compose_config(bundle)
    short_of = get_function_short_name_map(config)
    label = '-'.join(short_of[function] for function in config.dataset.schema_pair)
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset_manifest', type=Path, default=None,
                        help='the {prefix}_manifest.json written by run_allpairs_datasets.py')
    parser.add_argument('--experiment_spec', type=Path, default=None,
                        help='the experiment spec the datasets were built with')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='redraw seeds, the same for every fold, e.g. 42 43 ... 51')
    parser.add_argument('--resume', type=Path, default=None,
                        help='an existing redraw_run_manifest.json to continue')
    parser.add_argument('--baseline', default='lgbm')
    parser.add_argument('--pairs', nargs='+', default=None,
                        help='pair tokens to redraw; default every dataset in the manifest')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='concurrent fit processes; default keeps total threads under the '
                             'CPU count')
    parser.add_argument('--dry_run', action='store_true',
                        help='run every check and print the plan, without writing anything')
    args = parser.parse_args()
    if args.resume and (args.dataset_manifest or args.experiment_spec or args.pairs or args.seeds):
        parser.error('--resume reads the dataset manifest, spec, pairs and seeds from the redraw '
                     'run manifest; pass none of them')
    if not args.resume and not (args.dataset_manifest and args.experiment_spec and args.seeds):
        parser.error('a first call needs --dataset_manifest, --experiment_spec and --seeds')
    if args.seeds and len(args.seeds) != len(set(args.seeds)):
        parser.error('--seeds must be distinct')

    run_manifest, dataset_manifest, spec, run_manifest_path = load_run_inputs(args)
    seeds = run_manifest['seeds']

    # Checks that concern the whole call. All of them run before any redraw or fit starts.
    failures = fold_count_failures(dataset_manifest, spec)
    if get_git_info()['commit'] != dataset_manifest['code']['commit']:
        failures.append(f"the current commit differs from the datasets' "
                        f"{dataset_manifest['code']['commit']}")
    failures.extend(changed_files(dataset_manifest['cache_files']))
    entries = {entry['pair']: entry for entry in dataset_manifest['datasets'].values()}
    to_redraw = []
    for pair, plan in run_manifest['pairs'].items():
        entry = entries[pair]
        dataset_dir = Path(entry['dataset_dir'])
        failures.extend(f'{pair}: {m}' for m in changed_files(entry['files'], base=dataset_dir))
        planned_config = compose_config(plan['bundle'], spec['overrides'])
        failures.extend(f'{pair}: {m}'
                        for m in expected_setting_failures(planned_config, spec['expect']))
        failures.extend(f'{pair}: {m}'
                        for m in redraw_assumption_failures(dataset_dir, entry['split_dirs']))
        for fold_id, fold in plan['folds'].items():
            if not Path(fold['redraw_dir']).exists():
                if fold['redraw_files'] is not None:
                    failures.append(f"{pair} fold {fold_id}: recorded redraw {fold['redraw_dir']} "
                                    f"is missing")
                else:
                    to_redraw.append((pair, fold_id))
                continue
            failures.extend(f'{pair} fold {fold_id}: {m}'
                            for m in existing_redraw_problems(fold, seeds))
    if failures:
        print('\n'.join(f'ERROR: {message}' for message in failures))
        sys.exit(1)

    n_fits = sum(len(fold['seeds']) for plan in run_manifest['pairs'].values()
                 for fold in plan['folds'].values())
    print(f"Pairs:        {len(run_manifest['pairs'])}")
    print(f'Seeds:        {seeds}')
    print(f'Redraws:      {len(to_redraw)} folds to redraw')
    print(f'Fits:         {n_fits} planned')
    print(f'Run manifest: {run_manifest_path}\n')
    if args.dry_run:
        for pair, fold_id in to_redraw:
            fold = run_manifest['pairs'][pair]['folds'][fold_id]
            print(f"  redraw {fold['source_fold']} -> {fold['redraw_dir']}")
        print('\nDry run: nothing redrawn, fitted or written.')
        return

    def write_run_manifest() -> None:
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + '\n')

    if not args.resume:
        run_manifest_path.parent.mkdir(parents=True, exist_ok=False)
        write_run_manifest()

    # Redraw. Each finished fold's checksums are recorded at once, so an interrupted call can
    # be resumed without redrawing it.
    redraw_failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers or 4) as pool:
        futures = {pool.submit(run_redraw, run_manifest['pairs'][pair]['folds'][fold_id], seeds):
                   (pair, fold_id) for pair, fold_id in to_redraw}
        for future in as_completed(futures):
            pair, fold_id = futures[future]
            fold = run_manifest['pairs'][pair]['folds'][fold_id]
            code = future.result()
            if code != 0:
                redraw_failures.append(f"{pair} fold {fold_id}: redraw exited {code}; see "
                                       f"{fold['redraw_dir']}_redraw.log")
                continue
            redraw_dir = Path(fold['redraw_dir'])
            fold['redraw_files'] = redraw_files(redraw_dir, seeds)
            fold['identical_seed_pairs'] = identical_seed_pairs(redraw_dir, seeds)
            if fold['identical_seed_pairs']:
                redraw_failures.append(f"{pair} fold {fold_id}: seeds "
                                       f"{fold['identical_seed_pairs']} drew identical negatives")
            write_run_manifest()
            print(f'Done. redraw {pair} fold {fold_id}', flush=True)
    if redraw_failures:
        print('\n'.join(f'ERROR: {message}' for message in redraw_failures))
        sys.exit(1)

    # Fit. Every existing fit directory is checked before any fit starts.
    jobs, failures = fit_jobs(run_manifest, dataset_manifest, spec)
    to_fit = []
    for job in jobs:
        if not job['model_dir'].exists():
            to_fit.append(job)
            continue
        failures.extend(f"{job['pair']} fold {job['fold_id']} seed {job['seed']}: {m}"
                        for m in existing_fit_problems(job['model_dir'], job['planned_config'],
                                                       job['expected_inputs'],
                                                       dataset_manifest['code']['commit'],
                                                       neg_redraw_seed=job['seed']))
    if failures:
        print('\n'.join(f'ERROR: {message}' for message in failures))
        sys.exit(1)
    print(f'Fits:         {len(to_fit)} to run, {len(jobs) - len(to_fit)} already finished')

    exit_codes = {}
    if to_fit:
        max_workers = concurrent_fits(to_fit[0]['planned_config'], run_manifest['baseline'],
                                      args.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_fit, job['command'], job['model_dir']): job for job in to_fit}
            for future in as_completed(futures):
                job = futures[future]
                exit_codes[job['model_dir']] = future.result()
                print(f"{'OK  ' if exit_codes[job['model_dir']] == 0 else 'FAIL'} "
                      f"{job['model_dir'].name}", flush=True)

    # A fit that exited 0 must still have left a complete, matching run directory.
    failed_pairs = set()
    for job in jobs:
        code = exit_codes.get(job['model_dir'], 0)
        problems = [f'exit {code}'] if code else existing_fit_problems(
            job['model_dir'], job['planned_config'], job['expected_inputs'],
            dataset_manifest['code']['commit'], neg_redraw_seed=job['seed'])
        if problems:
            failed_pairs.add(job['pair'])
            print(f"ERROR: {job['model_dir'].name}: {problems}")

    # Summarize each pair whose fits all finished.
    for pair, plan in run_manifest['pairs'].items():
        if pair in failed_pairs:
            continue
        analysis_dir = Path(plan['analysis_dir'])
        command = [sys.executable, str(ANALYSIS_SCRIPT),
                   '--model-root', str(Path(jobs[0]['model_dir']).parent),
                   '--run-prefix', plan['analysis_run_prefix'],
                   '--folds', *plan['folds'], '--seeds', *map(str, seeds),
                   '--output-dir', str(analysis_dir), '--label', pair_label(plan['bundle'])]
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / 'analysis.log', 'w') as log_file:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log_file,
                                       stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            failed_pairs.add(pair)
            print(f'ERROR: {pair}: analysis exited {completed.returncode}; see '
                  f'{analysis_dir / "analysis.log"}')

    cache_changes = changed_files(dataset_manifest['cache_files'])
    for message in cache_changes:
        print(f'ERROR: after the fits, {message}')
    print(f"\n{len(run_manifest['pairs']) - len(failed_pairs)} pairs complete, "
          f"{len(failed_pairs)} failed.")
    if failed_pairs or cache_changes:
        sys.exit(1)
    print(f'Done. Run manifest: {run_manifest_path}')


if __name__ == '__main__':
    main()
