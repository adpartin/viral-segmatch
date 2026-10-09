"""Launch one `train_pair_baselines.py` fit, and decide whether an existing fit may be skipped.

Shared by `scripts/run_allpairs_baselines.py`, `scripts/run_balanced_train_redraw.py` and
`scripts/run_allpairs_balanced_redraws.py`, so every runner builds the same command and applies
the same skip rule.

The trainer reuses an existing run directory and overwrites its files, so a runner must decide
before launching it: an absent directory is fitted; a complete directory whose saved config, input
checksums, code commit and redraw seed match the planned job is skipped; anything else stops the
runner without touching the directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER = PROJECT_ROOT / 'src' / 'models' / 'train_pair_baselines.py'

# What a finished fit leaves in its run directory.
REQUIRED_FIT_OUTPUTS = ('best_model.joblib', 'resolved_config.yaml', 'metrics_summary.json',
                        'training_info.json', 'train_predicted.csv', 'val_predicted.csv',
                        'test_predicted.csv')


def baseline_threads(config, baseline: str) -> int:
    """How many threads one fit of this baseline asks for.

    The number is read off the estimator the baseline would build, rather than off the bundle,
    because each baseline reads its own config section and supplies its own default when the
    bundle sets nothing (`src/models/baselines/lgbm.py` defaults `n_jobs` to 16).

    Args:
      config: a planned fit config.
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


def concurrent_fits(config, baseline: str, max_workers: Optional[int]) -> int:
    """How many fits to run at once.

    Args:
      config: a planned fit config.
      baseline: baseline name.
      max_workers: an explicit cap, or None to keep total threads under the CPU count (at most 8).

    Returns:
      The number of concurrent fit processes.
    """
    if max_workers:
        return max_workers
    workers = max(1, min(8, (os.cpu_count() or 1) // baseline_threads(config, baseline)))
    return workers


def fit_command(bundle: str, baseline: str, dataset_dir: Path, output_dir: Path, overrides: list,
                dataset_resolved_config: Path, neg_redraw_seed: Optional[int] = None,
                with_post_hoc: bool = False) -> list:
    """The command line for one fit.

    Args:
      bundle: the child bundle to train with.
      baseline: baseline name, e.g. `lgbm`.
      dataset_dir: the split directory holding the pair CSVs.
      output_dir: the fit's run directory.
      overrides: dotlist entries passed as `--override`.
      dataset_resolved_config: the dataset's `resolved_config.yaml`, whose checksum the trainer
          records with the fit's other inputs.
      neg_redraw_seed: seed of the redraw that wrote `dataset_dir`'s training negatives, or None
          when they are the dataset builder's own.
      with_post_hoc: run `analyze_stage4_train.py` after the fit.

    Returns:
      The argument list, starting with the Python interpreter.
    """
    command = [sys.executable, str(TRAINER),
               '--config_bundle', bundle,
               '--baseline', baseline,
               '--dataset_dir', str(dataset_dir),
               '--output_dir', str(output_dir),
               '--dataset_resolved_config', str(dataset_resolved_config)]
    if overrides:
        command += ['--override', *overrides]
    if neg_redraw_seed is not None:
        command += ['--neg_redraw_seed', str(neg_redraw_seed)]
    if not with_post_hoc:
        command.append('--skip_post_hoc')
    return command


def run_fit(command: list, output_dir: Path, log_name: str = 'train.log') -> int:
    """Run one fit and keep its log in the run directory.

    Args:
      command: from `fit_command`.
      output_dir: the fit's run directory; created if absent.
      log_name: file name of the log inside `output_dir`.

    Returns:
      The trainer's exit code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / log_name, 'w') as log_file:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log_file,
                                   stderr=subprocess.STDOUT)
    return completed.returncode


def existing_fit_problems(output_dir: Path, planned_config, expected_inputs: dict,
                          expected_commit: str, neg_redraw_seed: Optional[int] = None) -> list:
    """Reasons an existing run directory is not the finished fit of the planned job.

    Args:
      output_dir: the fit's run directory, which exists.
      planned_config: the config the runner composed for this fit.
      expected_inputs: absolute input path -> SHA-256 the fit must have recorded.
      expected_commit: the code commit the fit must have recorded.
      neg_redraw_seed: the redraw seed the fit must have recorded; None for the builder's own
          negatives.

    Returns:
      One message per problem; empty when the directory holds the finished planned fit.
    """
    missing = [name for name in REQUIRED_FIT_OUTPUTS if not (output_dir / name).exists()]
    if missing:
        return [f'{output_dir.name}: missing {missing}']

    problems = []
    saved = OmegaConf.to_container(OmegaConf.load(output_dir / 'resolved_config.yaml'), resolve=True)
    planned = OmegaConf.to_container(planned_config, resolve=True)
    if saved != planned:
        differing = sorted(key for key in set(saved) | set(planned)
                           if saved.get(key) != planned.get(key))
        problems.append(f'{output_dir.name}: saved config differs from the planned one in '
                        f'{differing}')

    info = json.loads((output_dir / 'training_info.json').read_text())
    if info.get('code', {}).get('commit') != expected_commit:
        problems.append(f"{output_dir.name}: fitted at commit {info.get('code', {}).get('commit')}, "
                        f"expected {expected_commit}")
    if info.get('input_checksums') != expected_inputs:
        problems.append(f'{output_dir.name}: input checksums differ from the planned inputs')
    if info.get('neg_redraw_seed') != neg_redraw_seed:
        problems.append(f"{output_dir.name}: neg_redraw_seed is {info.get('neg_redraw_seed')}, "
                        f"expected {neg_redraw_seed}")
    return problems
