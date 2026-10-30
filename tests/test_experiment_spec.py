"""Tests for `src/utils/experiment_spec.py`.

Covers:
  1. A valid spec loads with its overrides, expect section, path and checksum
  2. A spec with an unknown section, a malformed override or a non-mapping expect is rejected
  3. A key in both sections must have the same value, compared after parsing the override
  4. expected_setting_failures reports mismatched and absent keys, and compares lists as lists
  5. expected_n_folds requires dataset.n_folds under expect
  6. split_dir_names gives fold directories for CV and the root for a single split
  7. Both committed experiment specs load

Run: python -m pytest tests/test_experiment_spec.py
"""
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.utils.experiment_spec import (  # noqa: E402
    expected_n_folds,
    expected_setting_failures,
    load_experiment_spec,
    split_dir_names,
)


def _write(tmp_path, text):
    path = tmp_path / 'spec.yaml'
    path.write_text(text)
    return path


def test_valid_spec_loads(tmp_path):
    path = _write(tmp_path, 'overrides:\n  - dataset.n_folds=5\n'
                            'expect:\n  dataset.n_folds: 5\n  dataset.host: [Human]\n')
    spec = load_experiment_spec(path)
    assert spec['overrides'] == ['dataset.n_folds=5']
    assert spec['expect'] == {'dataset.n_folds': 5, 'dataset.host': ['Human']}
    assert spec['path'] == str(path.resolve())
    assert len(spec['sha256']) == 64


@pytest.mark.parametrize('text', [
    'overrides: []\nexpect: {}\nnotes: x\n',              # unknown section
    'overrides:\n  - dataset.n_folds\nexpect: {}\n',      # override without '='
    'overrides: []\nexpect:\n  - dataset.n_folds\n',      # expect is a list
])
def test_malformed_spec_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        load_experiment_spec(_write(tmp_path, text))


def test_conflicting_sections_rejected(tmp_path):
    path = _write(tmp_path, 'overrides:\n  - dataset.n_folds=5\nexpect:\n  dataset.n_folds: 4\n')
    with pytest.raises(ValueError, match='disagrees with itself'):
        load_experiment_spec(path)


def test_expected_setting_failures():
    config = OmegaConf.create({'dataset': {'n_folds': 4, 'host': ['Human'], 'negative_sampling': None}})
    assert expected_setting_failures(config, {'dataset.host': ['Human'],
                                              'dataset.negative_sampling': None}) == []
    failures = expected_setting_failures(config, {'dataset.n_folds': 5, 'dataset.year': [2024]})
    assert len(failures) == 2
    assert any('n_folds' in f and 'expected 5' in f for f in failures)
    assert any('dataset.year' in f and 'absent' in f for f in failures)


def test_expected_n_folds(tmp_path):
    spec = load_experiment_spec(_write(tmp_path, 'expect:\n  dataset.n_folds: 5\n'))
    assert expected_n_folds(spec) == 5
    spec = load_experiment_spec(_write(tmp_path, 'expect:\n  dataset.host: [Human]\n'))
    with pytest.raises(ValueError):
        expected_n_folds(spec)


def test_split_dir_names():
    assert split_dir_names(5) == ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
    assert split_dir_names(None) == ['.']
    assert split_dir_names(1) == ['.']


@pytest.mark.parametrize('name, n_folds, overrides', [
    ('human_h3n2_2024_28p_codon_balanced_cv5.yaml', 5, ['dataset.n_folds=5']),
    ('human_h3n2_2024_28p_codon_balanced_cv4.yaml', 4, []),
    ('human_pig_h3n2_h1n1_2022_2023_21p_codon_matched_cv5.yaml', 5, ['dataset.n_folds=5']),
])
def test_committed_specs_load(name, n_folds, overrides):
    spec = load_experiment_spec(PROJ / 'conf' / 'experiments' / name)
    assert expected_n_folds(spec) == n_folds
    assert spec['overrides'] == overrides
