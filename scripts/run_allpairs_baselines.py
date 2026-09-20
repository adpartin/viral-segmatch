#!/usr/bin/env python3
"""Train a sklearn-style baseline on every fold of a set of already-built pair datasets.

The GPU launcher `run_cv_lambda.py` runs `train_pair_classifier.py`, the MLP path, and cannot
run a baseline. This runs `train_pair_baselines.py` instead, over datasets that Stage 3 has
already written, and produces the same CV output layout the GPU launcher does, so
`aggregate_cv_results.py` and `src/analysis/aggregate_allpairs_results.py` read it unchanged.

Scope: it trains folds and aggregates them. It does not build datasets and does not generate
bundles.

How?

- Each pair is one dataset directory `{dataset_prefix}{pair}` holding `fold_0/ ... fold_{N-1}/`,
  and one bundle `{bundle_prefix}{pair}`. The pair token comes from the dataset directory name.
- `training.threshold_metric` must be null in every bundle, since a threshold tuned per fold
  would silently replace the 0.5 decision threshold the comparison across pairs assumes.
- A fold is skipped when its run directory already holds `test_predicted.csv`, so an
  interrupted sweep resumes by re-running the same command. `--force` retrains regardless.
- Fold processes run concurrently, capped so that concurrent fits times the baseline's own
  thread count stays under the machine's CPU count.
- `--cv_dir_bundle` names the CV directory. `aggregate_allpairs_results.py` reads the protein
  pair out of that name, from the two tokens after `flu_28p_`, and selects a variant by the
  suffix after them, so a bundle whose name carries the variant elsewhere needs a CV directory
  name in that form. The manifest still records the real bundle.

CLI:
    python scripts/run_allpairs_baselines.py \
        --bundle_prefix flu_28p_codon_ --dataset_prefix exp3_28p_codon_ \
        --cv_dir_bundle 'flu_28p_{pair}_codon' --pairs pb1_ns1 --dry_run

Outputs:
    models/{virus}/{version}/runs/{baseline}_{dataset_dir}_fold{k}/   per fold
    models/{virus}/{version}/cv_runs/cv_{cv_dir_bundle}_{timestamp}/  manifest, status, cv_summary
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

from src.utils.config_hydra import get_virus_config_hydra  # noqa: E402

CONF_PATH = str(PROJECT_ROOT / 'conf')


def load_bundle(bundle: str):
    """The resolved config for one bundle.

    Args:
      bundle: bundle name, e.g. `flu_28p_codon_pb1_ns1`.

    Returns:
      The resolved OmegaConf config.
    """
    config = get_virus_config_hydra(bundle, config_path=CONF_PATH)
    return config


def baseline_threads(config, baseline: str) -> int:
    """How many threads one fit of this baseline asks for.

    The number is read off the estimator the baseline would build, rather than off the bundle,
    because each baseline reads its own config section and supplies its own default when the
    bundle sets nothing (`src/models/baselines/lgbm.py` defaults `n_jobs` to 16).

    Args:
      config: a resolved bundle config.
      baseline: baseline name, e.g. `lgbm`.

    Returns:
      The estimator's `n_jobs`, or 1 when it has none. A negative `n_jobs` means every core,
      which is reported as the full CPU count so that only one fit runs at a time.
    """
    from src.models.train_pair_baselines import _resolve_baseline_module
    estimator = _resolve_baseline_module(baseline).get_estimator(config, random_state=0)
    n_jobs = int(getattr(estimator, 'n_jobs', 1) or 1)
    if n_jobs < 0:
        return os.cpu_count() or 1
    return max(1, n_jobs)


def fold_jobs(pair: str, bundle: str, dataset_dir: Path, n_folds: int, baseline: str,
              models_runs: Path) -> list:
    """One job per fold of one pair.

    Args:
      pair: pair token, e.g. `pb1_ns1`.
      bundle: the bundle to train with.
      dataset_dir: the pair's dataset run directory, holding `fold_*/`.
      n_folds: number of folds to run.
      baseline: baseline name.
      models_runs: `models/{virus}/{version}/runs`.

    Returns:
      Dicts holding the fold id, the fold's dataset directory, its run id and its output
      directory.

    Raises:
      ValueError: a fold directory is absent.
    """
    jobs = []
    for fold_id in range(n_folds):
        fold_dir = dataset_dir / f'fold_{fold_id}'
        if not fold_dir.is_dir():
            raise ValueError(f"fold_jobs: {pair} has no {fold_dir}.")
        run_id = f'{baseline}_{dataset_dir.name}_fold{fold_id}'
        jobs.append({'pair': pair, 'bundle': bundle, 'fold_id': fold_id, 'fold_dir': fold_dir,
                     'run_id': run_id, 'output_dir': models_runs / run_id})
    return jobs


def run_fold(job: dict, baseline: str, with_post_hoc: bool, dry_run: bool) -> int:
    """Train one fold in a subprocess.

    Args:
      job: one entry from `fold_jobs`.
      baseline: baseline name passed to the trainer.
      with_post_hoc: run `analyze_stage4_train.py` after training.
      dry_run: print the command and return 0 without running it.

    Returns:
      The trainer's exit code, or 0 in dry-run.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / 'src' / 'models' / 'train_pair_baselines.py'),
           '--config_bundle', job['bundle'],
           '--baseline', baseline,
           '--dataset_dir', str(job['fold_dir']),
           '--run_output_subdir', job['run_id']]
    if not with_post_hoc:
        cmd.append('--skip_post_hoc')
    if dry_run:
        print('  $ ' + ' '.join(cmd))
        return 0
    job['output_dir'].mkdir(parents=True, exist_ok=True)
    log_path = job['output_dir'] / 'train.log'
    with open(log_path, 'w') as log_file:
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log_file,
                                   stderr=subprocess.STDOUT)
    return completed.returncode


def write_cv_outputs(cv_dir: Path, bundle: str, dataset_dir: Path, n_folds: int,
                     jobs: list, failed_folds: list) -> None:
    """Write the manifest and status a CV directory carries, then aggregate the folds.

    Args:
      cv_dir: the CV results directory for this pair.
      bundle: the bundle the folds were trained with.
      dataset_dir: the pair's dataset run directory.
      n_folds: number of folds.
      jobs: this pair's fold jobs.
      failed_folds: fold ids whose training exited non-zero.

    Returns:
      None. Writes `cv_run_manifest.json`, `status.json`, and whatever
      `aggregate_cv_results.py` writes.
    """
    cv_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'config_bundle': bundle,
        'n_folds': n_folds,
        'dataset_run_dir': str(dataset_dir),
        'training_run_ids': {str(j['fold_id']): j['run_id'] for j in jobs},
        'launched_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    manifest_path = cv_dir / 'cv_run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    done = [j['fold_id'] for j in jobs if j['fold_id'] not in failed_folds]
    status = {
        'status': 'FAILED' if failed_folds else 'COMPLETE',
        'config_bundle': bundle,
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
        print(f'  WARNING: {len(failed_folds)} folds failed; skipping aggregation.')
        return
    aggregate_cmd = [sys.executable, str(PROJECT_ROOT / 'scripts' / 'aggregate_cv_results.py'),
                     '--manifest', str(manifest_path), '--output_dir', str(cv_dir)]
    log_path = cv_dir / 'aggregate.log'
    with open(log_path, 'w') as log_file:
        completed = subprocess.run(aggregate_cmd, cwd=PROJECT_ROOT, stdout=log_file,
                                   stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        print(f'  ERROR: aggregation exited {completed.returncode}; see {log_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bundle_prefix', required=True,
                        help="bundle name minus the pair token, e.g. 'flu_28p_codon_'")
    parser.add_argument('--dataset_prefix', required=True,
                        help="dataset run directory name minus the pair token, e.g. "
                             "'exp3_28p_codon_'")
    parser.add_argument('--baseline', default='lgbm')
    parser.add_argument('--pairs', nargs='+', default=None,
                        help='pair tokens to run; default every dataset matching the prefix')
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
                        help='retrain folds that already hold test_predicted.csv')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    # The dataset and model directories come from the virus and data version a bundle resolves,
    # so one bundle has to be read before anything can be located.
    pairs = args.pairs
    if pairs:
        probe_pair = pairs[0]
    else:
        candidates = sorted(PROJECT_ROOT.glob(f'conf/bundles/{args.bundle_prefix}*.yaml'))
        if not candidates:
            raise ValueError(f"main: no bundles match conf/bundles/{args.bundle_prefix}*.yaml.")
        probe_pair = candidates[0].stem[len(args.bundle_prefix):]
    probe = load_bundle(f'{args.bundle_prefix}{probe_pair}')

    datasets_base = (PROJECT_ROOT / 'data' / 'datasets' / probe.virus.virus_name
                     / probe.virus.data_version / 'runs')
    models_base = PROJECT_ROOT / 'models' / probe.virus.virus_name / probe.virus.data_version
    models_runs = models_base / 'runs'

    if pairs is None:
        pairs = sorted(d.name[len(args.dataset_prefix):]
                       for d in datasets_base.glob(f'{args.dataset_prefix}*') if d.is_dir())
    if not pairs:
        raise ValueError(f"main: no dataset directories match {args.dataset_prefix}* in "
                         f"{datasets_base}.")

    threads_per_fit = baseline_threads(probe, args.baseline)
    cpu_count = os.cpu_count() or 1
    max_workers = args.max_workers or max(1, min(8, cpu_count // threads_per_fit))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'Pairs:        {len(pairs)}')
    print(f'Baseline:     {args.baseline} ({threads_per_fit} threads per fit)')
    print(f'Workers:      {max_workers} concurrent folds on {cpu_count} CPUs')
    print(f'Post-hoc:     {"on" if args.with_post_hoc else "off"}')
    print(f'Datasets:     {datasets_base}')
    print(f'Models:       {models_runs}\n')

    # Build every job first, so a bad bundle or a missing fold stops the sweep before any
    # training starts.
    per_pair = {}
    for pair in pairs:
        bundle = f'{args.bundle_prefix}{pair}'
        config = load_bundle(bundle)
        threshold_metric = getattr(config.training, 'threshold_metric', None)
        if threshold_metric is not None:
            raise ValueError(
                f"main: {bundle} sets training.threshold_metric={threshold_metric!r}. This "
                f"sweep compares pairs at the 0.5 decision threshold, which a per-fold tuned "
                f"threshold would replace.")
        dataset_dir = datasets_base / f'{args.dataset_prefix}{pair}'
        if not dataset_dir.is_dir():
            raise ValueError(f"main: {pair} has no dataset directory at {dataset_dir}.")
        n_folds = int(config.dataset.n_folds)
        per_pair[pair] = {
            'bundle': bundle, 'dataset_dir': dataset_dir, 'n_folds': n_folds,
            'jobs': fold_jobs(pair, bundle, dataset_dir, n_folds, args.baseline, models_runs),
        }

    pending = []
    for pair, entry in per_pair.items():
        for job in entry['jobs']:
            if not args.force and (job['output_dir'] / 'test_predicted.csv').exists():
                continue
            pending.append(job)
    n_total = sum(len(e['jobs']) for e in per_pair.values())
    print(f'Folds: {n_total} total, {len(pending)} to run, '
          f'{n_total - len(pending)} already trained.\n')

    results = {}
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_fold, job, args.baseline, args.with_post_hoc,
                                   args.dry_run): job for job in pending}
            for future, job in futures.items():
                code = future.result()
                results[(job['pair'], job['fold_id'])] = code
                mark = 'OK  ' if code == 0 else 'FAIL'
                print(f'{mark} {job["run_id"]} (exit {code})')

    if args.dry_run:
        print('\nDry run: no folds trained, no CV directories written.')
        return

    n_failed_pairs = 0
    for pair, entry in per_pair.items():
        failed_folds = [fold for (p, fold), code in results.items() if p == pair and code != 0]
        cv_name = args.cv_dir_bundle.format(pair=pair, bundle=entry['bundle'])
        cv_dir = models_base / 'cv_runs' / f'cv_{cv_name}_{timestamp}'
        print(f'\n{pair}: {cv_dir.name}')
        write_cv_outputs(cv_dir, entry['bundle'], entry['dataset_dir'], entry['n_folds'],
                         entry['jobs'], failed_folds)
        if failed_folds:
            n_failed_pairs += 1

    print(f'\n{len(per_pair) - n_failed_pairs} pairs complete, {n_failed_pairs} with failed '
          f'folds.')
    if n_failed_pairs:
        sys.exit(1)
    print('Done.')


if __name__ == '__main__':
    main()
