#!/usr/bin/env python3
"""Train a sklearn-style baseline on every fold of the pair datasets in a dataset manifest.

The GPU launcher `run_cv_lambda.py` runs `train_pair_classifier.py`, the MLP path, and cannot
run a baseline. This runs `train_pair_baselines.py` instead, over datasets that
`scripts/run_allpairs_datasets.py` already built and recorded in a dataset manifest, and produces
the same CV output layout the GPU launcher does, so `aggregate_cv_results.py` and
`src/analysis/aggregate_allpairs_results.py` read it unchanged.

Scope: it trains folds and aggregates them. It does not build datasets and does not generate
bundles.

How?

- The pairs, bundles, dataset directories, split directories and fold count come from the dataset
  manifest. `--pairs` trains a subset of them.
- The experiment spec must be the one the manifest recorded, its fold count must match the
  manifest's, and every pair's planned config must match its `expect` section. The planned config
  is the bundle composed with the spec's override list, the same composition the trainer applies.
- The current commit must equal the dataset manifest's, and every dataset file and site-feature
  cache file must still have its recorded checksum: before any fit, and the cache files again
  after all fits.
- The first call writes `training_run_manifest.json`, which assigns one model directory per fold
  and one CV directory per pair. A resume call names that file with `--resume` and reuses its
  directories; the current commit must equal the one it recorded.
- An existing fit directory is skipped only when it holds every required output and its saved
  config, input checksums and commit match the planned fit (`src/models/baseline_fits.py`).
  Anything else stops the call before any fit starts, because the trainer would overwrite it. A
  fit that crashed leaves such a directory; clearing it is a deletion and needs approval.
- `training.threshold_metric` must be null in every planned config, since a threshold tuned per
  fold would replace the 0.5 decision threshold the comparison across pairs assumes.
- After the fits, each pair's CV directory gets `cv_run_manifest.json`, `status.json` and the
  outputs of `aggregate_cv_results.py --strict`. These are the only files a resume overwrites.
- The call exits non-zero when a fit fails, a finished fit is incomplete, or an aggregation fails.
- `--cv_dir_bundle` names the CV directory. `aggregate_allpairs_results.py` reads the protein
  pair out of that name, from the two tokens after `flu_28p_`, and selects a variant by the
  suffix after them, so a bundle whose name carries the variant elsewhere needs a CV directory
  name in that form.

CLI:
    python scripts/run_allpairs_baselines.py \
        --dataset_manifest data/datasets/flu/July_2025/runs/<prefix>_manifest.json \
        --experiment_spec conf/experiments/human_h3n2_2024_28p_codon_balanced_cv5.yaml \
        --cv_dir_bundle 'flu_28p_{pair}_codon' --pairs pb2_ha --dry_run

    python scripts/run_allpairs_baselines.py \
        --resume models/flu/July_2025/training_runs/<name>/training_run_manifest.json

Outputs:
    models/{virus}/{version}/runs/{baseline}_{dataset_dir}_fold{k}/        per fold
    models/{virus}/{version}/cv_runs/cv_{cv_dir_bundle}_{timestamp}/       per pair
    models/{virus}/{version}/training_runs/{prefix}{baseline}_{timestamp}/training_run_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_fits import (  # noqa: E402
    concurrent_fits,
    existing_fit_problems,
    fit_command,
    run_fit,
)
from src.utils.config_hydra import compose_config  # noqa: E402
from src.utils.dataset_manifest import (  # noqa: E402
    changed_files,
    fit_input_checksums,
    fold_count_failures,
    load_dataset_manifest,
    pair_cache_checksums,
)
from src.utils.experiment_spec import (  # noqa: E402
    expected_setting_failures,
    load_experiment_spec,
)
from src.utils.experiment_utils import get_git_info  # noqa: E402


def plan_training_run(dataset_manifest: dict, spec: dict, baseline: str, pairs: list,
                      cv_dir_bundle: str, timestamp: str) -> dict:
    """The training run manifest for a first call: one model directory per fold, one CV directory
    per pair.

    Args:
      dataset_manifest: from `load_dataset_manifest`.
      spec: from `load_experiment_spec`.
      baseline: baseline name.
      pairs: pair tokens to train.
      cv_dir_bundle: CV directory name pattern with `{pair}` and `{bundle}` placeholders.
      timestamp: `YYYYMMDD_HHMMSS` of this call, shared by all its CV directories.

    Returns:
      The manifest to write as `training_run_manifest.json`.
    """
    by_pair = {entry['pair']: (bundle, entry)
               for bundle, entry in dataset_manifest['datasets'].items()}
    probe = compose_config(by_pair[pairs[0]][0], spec['overrides'])
    models_base = PROJECT_ROOT / 'models' / probe.virus.virus_name / probe.virus.data_version
    planned = {}
    for pair in pairs:
        bundle, entry = by_pair[pair]
        dataset_dir = Path(entry['dataset_dir'])
        cv_name = cv_dir_bundle.format(pair=pair, bundle=bundle)
        folds = {}
        for fold_id, split_dir in enumerate(entry['split_dirs']):
            run_id = f'{baseline}_{dataset_dir.name}_fold{fold_id}'
            folds[str(fold_id)] = {'split_dir': split_dir, 'run_id': run_id,
                                   'model_dir': str(models_base / 'runs' / run_id)}
        planned[pair] = {'bundle': bundle, 'dataset_dir': str(dataset_dir),
                         'cv_dir': str(models_base / 'cv_runs' / f'cv_{cv_name}_{timestamp}'),
                         'folds': folds}
    run_manifest = {
        'dataset_manifest': {'path': dataset_manifest['manifest_path'],
                             'sha256': dataset_manifest['manifest_sha256']},
        'experiment_spec': {'path': spec['path'], 'sha256': spec['sha256']},
        'overrides': spec['overrides'],
        'baseline': baseline,
        'timestamp': timestamp,
        'code': get_git_info(),
        'pairs': planned,
    }
    return run_manifest


def check_fit_jobs(run_manifest: dict, dataset_manifest: dict, spec: dict,
                   force: bool) -> tuple[list, list, list]:
    """Check every planned fit before any starts, and sort the folds into skip and run.

    Args:
      run_manifest: the training run manifest.
      dataset_manifest: from `load_dataset_manifest`.
      spec: from `load_experiment_spec`.
      force: refit folds whose run directory exists, overwriting it.

    Returns:
      The jobs to run, the jobs to skip, and the failure messages. A job holds the pair, fold id,
      split directory, model directory, planned config, expected input checksums and command.
    """
    entries = {entry['pair']: entry for entry in dataset_manifest['datasets'].values()}
    to_run, to_skip, failures = [], [], []
    for pair, plan in run_manifest['pairs'].items():
        entry = entries[pair]
        failures.extend(f'{pair}: {message}'
                        for message in changed_files(entry['files'], base=Path(entry['dataset_dir'])))
        planned_config = compose_config(plan['bundle'], spec['overrides'])
        failures.extend(f'{pair}: {message}'
                        for message in expected_setting_failures(planned_config, spec['expect']))
        if planned_config.dataset.n_folds != dataset_manifest['n_folds']:
            failures.append(f"{pair}: planned n_folds={planned_config.dataset.n_folds}, the "
                            f"datasets were built with {dataset_manifest['n_folds']}")
        threshold_metric = planned_config.training.get('threshold_metric')
        if threshold_metric is not None:
            failures.append(f'{pair}: training.threshold_metric={threshold_metric!r}; this sweep '
                            f'compares pairs at the 0.5 decision threshold')
        cache_checksums = pair_cache_checksums(dataset_manifest, planned_config)
        dataset_dir = Path(entry['dataset_dir'])
        for fold_id, fold in plan['folds'].items():
            model_dir = Path(fold['model_dir'])
            job = {
                'pair': pair, 'fold_id': int(fold_id), 'model_dir': model_dir,
                'planned_config': planned_config,
                'expected_inputs': fit_input_checksums(entry, fold['split_dir'], cache_checksums),
                'command': fit_command(plan['bundle'], run_manifest['baseline'],
                                       dataset_dir / fold['split_dir'], model_dir,
                                       spec['overrides'], dataset_dir / 'resolved_config.yaml'),
            }
            if not model_dir.exists() or force:
                to_run.append(job)
                continue
            problems = existing_fit_problems(model_dir, planned_config, job['expected_inputs'],
                                             dataset_manifest['code']['commit'])
            if problems:
                failures.extend(f'{pair} fold {fold_id}: {message}' for message in problems)
            else:
                to_skip.append(job)
    return to_run, to_skip, failures


def write_cv_outputs(cv_dir: Path, plan: dict, n_folds: int, failed_folds: list) -> bool:
    """Write the manifest and status a CV directory carries, then aggregate the folds strictly.

    Args:
      cv_dir: the CV results directory for this pair.
      plan: the pair's entry in the training run manifest.
      n_folds: number of folds.
      failed_folds: fold ids whose fit failed or is incomplete.

    Returns:
      True when every fold is complete and aggregation succeeded. Writes `cv_run_manifest.json`,
      `status.json`, and whatever `aggregate_cv_results.py` writes.
    """
    cv_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'config_bundle': plan['bundle'],
        'n_folds': n_folds,
        'dataset_run_dir': plan['dataset_dir'],
        'training_run_ids': {fold_id: fold['run_id'] for fold_id, fold in plan['folds'].items()},
        'launched_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    manifest_path = cv_dir / 'cv_run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    done = [int(fold_id) for fold_id in plan['folds'] if int(fold_id) not in failed_folds]
    status = {
        'status': 'FAILED' if failed_folds else 'COMPLETE',
        'config_bundle': plan['bundle'],
        'n_folds': n_folds,
        'folds_done': sorted(done),
        'folds_failed': sorted(failed_folds),
        'folds_running': [],
        'n_done': len(done),
        'n_failed': len(failed_folds),
        'n_running': 0,
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    (cv_dir / 'status.json').write_text(json.dumps(status, indent=2))

    if failed_folds:
        print(f'  ERROR: {len(failed_folds)} folds failed; skipping aggregation.')
        return False
    aggregate_cmd = [sys.executable, str(PROJECT_ROOT / 'scripts' / 'aggregate_cv_results.py'),
                     '--manifest', str(manifest_path), '--output_dir', str(cv_dir), '--strict']
    log_path = cv_dir / 'aggregate.log'
    with open(log_path, 'w') as log_file:
        completed = subprocess.run(aggregate_cmd, cwd=PROJECT_ROOT, stdout=log_file,
                                   stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        print(f'  ERROR: aggregation exited {completed.returncode}; see {log_path}')
        return False
    return True


def load_run_inputs(args) -> tuple[dict, dict, dict, Path]:
    """The training run manifest, dataset manifest and experiment spec for this call.

    A first call plans a new training run; a resume call reads the one it names and requires the
    files it recorded to be unchanged.

    Args:
      args: the parsed command line.

    Returns:
      The training run manifest, the dataset manifest, the experiment spec, and the path the
      training run manifest is (or will be) written to.

    Raises:
      SystemExit: a recorded file changed, or the current commit differs from the recorded one.
    """
    if args.resume:
        run_manifest_path = args.resume.resolve()
        run_manifest = json.loads(run_manifest_path.read_text())
        dataset_manifest = load_dataset_manifest(Path(run_manifest['dataset_manifest']['path']))
        spec = load_experiment_spec(Path(run_manifest['experiment_spec']['path']))
        if dataset_manifest['manifest_sha256'] != run_manifest['dataset_manifest']['sha256']:
            raise SystemExit('ERROR: the dataset manifest changed since this training run started')
        if spec['sha256'] != run_manifest['experiment_spec']['sha256']:
            raise SystemExit('ERROR: the experiment spec changed since this training run started')
        if get_git_info()['commit'] != run_manifest['code']['commit']:
            raise SystemExit(f"ERROR: the current commit differs from the training run's "
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
    run_manifest = plan_training_run(dataset_manifest, spec, args.baseline, pairs,
                                     args.cv_dir_bundle, timestamp)
    models_base = Path(next(iter(run_manifest['pairs'].values()))['cv_dir']).parents[1]
    run_name = f"{dataset_manifest['dataset_prefix']}{args.baseline}_{timestamp}"
    run_manifest_path = models_base / 'training_runs' / run_name / 'training_run_manifest.json'
    return run_manifest, dataset_manifest, spec, run_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset_manifest', type=Path, default=None,
                        help='the {prefix}_manifest.json written by run_allpairs_datasets.py')
    parser.add_argument('--experiment_spec', type=Path, default=None,
                        help='the experiment spec the datasets were built with')
    parser.add_argument('--resume', type=Path, default=None,
                        help='an existing training_run_manifest.json to continue')
    parser.add_argument('--baseline', default='lgbm')
    parser.add_argument('--pairs', nargs='+', default=None,
                        help='pair tokens to train; default every dataset in the manifest')
    parser.add_argument('--cv_dir_bundle', default='{bundle}',
                        help="CV directory name, with {pair} and {bundle} placeholders "
                             "(default: the bundle name, as run_cv_lambda.py writes it)")
    parser.add_argument('--max_workers', type=int, default=None,
                        help='concurrent fold processes; default keeps total threads under '
                             'the CPU count')
    parser.add_argument('--with_post_hoc', action='store_true',
                        help='run analyze_stage4_train.py per fold (off by default: this '
                             'sweep is one host, subtype and year, so its strata are single)')
    parser.add_argument('--force', action='store_true',
                        help='refit folds whose run directory exists, overwriting it')
    parser.add_argument('--dry_run', action='store_true',
                        help='run every check and print the commands, without writing anything')
    args = parser.parse_args()
    if args.resume and (args.dataset_manifest or args.experiment_spec or args.pairs):
        parser.error('--resume reads the dataset manifest, spec and pairs from the training run '
                     'manifest; pass none of them')
    if not args.resume and not (args.dataset_manifest and args.experiment_spec):
        parser.error('a first call needs --dataset_manifest and --experiment_spec')

    run_manifest, dataset_manifest, spec, run_manifest_path = load_run_inputs(args)

    # Checks that concern the whole call. All of them run before any fit starts.
    failures = fold_count_failures(dataset_manifest, spec)
    if get_git_info()['commit'] != dataset_manifest['code']['commit']:
        failures.append(f"the current commit differs from the datasets' "
                        f"{dataset_manifest['code']['commit']}")
    failures.extend(changed_files(dataset_manifest['cache_files']))
    to_run, to_skip, fit_failures = check_fit_jobs(run_manifest, dataset_manifest, spec, args.force)
    failures.extend(fit_failures)
    if failures:
        print('\n'.join(f'ERROR: {message}' for message in failures))
        sys.exit(1)

    max_workers = (concurrent_fits(to_run[0]['planned_config'], run_manifest['baseline'],
                                   args.max_workers) if to_run else 1)
    print(f"Pairs:        {len(run_manifest['pairs'])}")
    print(f"Baseline:     {run_manifest['baseline']}")
    print(f'Workers:      {max_workers} concurrent folds on {os.cpu_count()} CPUs')
    print(f'Folds:        {len(to_run) + len(to_skip)} total, {len(to_run)} to run, '
          f'{len(to_skip)} already finished')
    print(f'Run manifest: {run_manifest_path}\n')

    if args.dry_run:
        for job in to_run:
            print('  $ ' + ' '.join(job['command']))
        print('\nDry run: no folds trained, nothing written.')
        return

    if not args.resume:
        run_manifest_path.parent.mkdir(parents=True, exist_ok=False)
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + '\n')

    exit_codes = {}
    if to_run:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_fit, job['command'], job['model_dir']): job
                       for job in to_run}
            for future, job in futures.items():
                code = future.result()
                exit_codes[(job['pair'], job['fold_id'])] = code
                print(f"{'OK  ' if code == 0 else 'FAIL'} {job['model_dir'].name} (exit {code})")

    # A fit that exited 0 must still have left a complete, matching run directory.
    failed_folds = {}
    for job in to_run + to_skip:
        key = (job['pair'], job['fold_id'])
        problems = [] if exit_codes.get(key, 0) == 0 else [f'exit {exit_codes[key]}']
        if not problems:
            problems = existing_fit_problems(job['model_dir'], job['planned_config'],
                                             job['expected_inputs'],
                                             dataset_manifest['code']['commit'])
        if problems:
            failed_folds.setdefault(job['pair'], []).append(job['fold_id'])
            print(f"ERROR: {job['pair']} fold {job['fold_id']}: {problems}")

    cache_changes = changed_files(dataset_manifest['cache_files'])
    for message in cache_changes:
        print(f'ERROR: after the fits, {message}')

    n_failed_pairs = 0
    for pair, plan in run_manifest['pairs'].items():
        print(f"\n{pair}: {Path(plan['cv_dir']).name}")
        complete = write_cv_outputs(Path(plan['cv_dir']), plan, dataset_manifest['n_folds'],
                                    failed_folds.get(pair, []))
        n_failed_pairs += not complete

    print(f"\n{len(run_manifest['pairs']) - n_failed_pairs} pairs complete, {n_failed_pairs} "
          f"failed.")
    if n_failed_pairs or cache_changes:
        sys.exit(1)
    print(f'Done. Run manifest: {run_manifest_path}')


if __name__ == '__main__':
    main()
