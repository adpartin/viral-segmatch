"""Tests for `src/models/baseline_fits.py`.

Covers:
  1. fit_command passes the output directory, dataset config, overrides and redraw seed
  2. existing_fit_problems accepts only a complete directory whose saved config, commit, input
     checksums and redraw seed all match the planned fit, and names each mismatch

Run: python -m pytest tests/test_baseline_fits.py
"""
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.models.baseline_fits import (  # noqa: E402
    REQUIRED_FIT_OUTPUTS,
    existing_fit_problems,
    fit_command,
)

PLANNED = OmegaConf.create({'dataset': {'n_folds': 5}, 'master_seed': 42,
                            'derived': '${dataset.n_folds}'})
INPUTS = {'/data/fold_0/train_pairs.csv': 'a' * 64}


def _finished_fit(tmp_path, *, config=PLANNED, commit='abc', inputs=INPUTS, seed=None) -> Path:
    """A run directory holding every required output, with the given recorded values."""
    run = tmp_path / 'run'
    run.mkdir()
    for name in REQUIRED_FIT_OUTPUTS:
        (run / name).write_text('x\n')
    OmegaConf.save(config, run / 'resolved_config.yaml')
    (run / 'training_info.json').write_text(json.dumps(
        {'code': {'commit': commit}, 'input_checksums': inputs, 'neg_redraw_seed': seed}))
    return run


def test_fit_command():
    command = fit_command('flu_28p_codon_pb2_ha', 'lgbm', Path('/d/seed_42'), Path('/m/run'),
                          ['dataset.n_folds=5'], Path('/d/resolved_config.yaml'), neg_redraw_seed=42)
    joined = ' '.join(command)
    assert '--output_dir /m/run' in joined
    assert '--dataset_resolved_config /d/resolved_config.yaml' in joined
    assert '--override dataset.n_folds=5' in joined
    assert '--neg_redraw_seed 42' in joined
    assert command[-1] == '--skip_post_hoc'
    assert '--neg_redraw_seed' not in fit_command('b', 'lgbm', Path('/d'), Path('/m'), [],
                                                  Path('/c'))


def test_matching_fit_is_accepted(tmp_path):
    run = _finished_fit(tmp_path, seed=7)
    assert existing_fit_problems(run, PLANNED, INPUTS, 'abc', neg_redraw_seed=7) == []


def test_missing_output_is_a_problem(tmp_path):
    run = _finished_fit(tmp_path)
    (run / 'test_predicted.csv').unlink()
    problems = existing_fit_problems(run, PLANNED, INPUTS, 'abc')
    assert len(problems) == 1 and 'test_predicted.csv' in problems[0]


def test_each_mismatch_is_named(tmp_path):
    other_config = OmegaConf.create({'dataset': {'n_folds': 4}, 'master_seed': 42,
                                     'derived': '${dataset.n_folds}'})
    run = _finished_fit(tmp_path, config=other_config, commit='old',
                        inputs={'/data/fold_0/train_pairs.csv': 'b' * 64}, seed=3)
    problems = existing_fit_problems(run, PLANNED, INPUTS, 'abc', neg_redraw_seed=None)
    assert len(problems) == 4
    joined = ' '.join(problems)
    for fragment in ('saved config', 'commit', 'input checksums', 'neg_redraw_seed'):
        assert fragment in joined
