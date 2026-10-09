"""Tests for `src/utils/dataset_manifest.py` and the trainer's input checksums.

Covers:
  1. read_split_dirs reads cv_info.json, falls back to the root for a single split, and rejects
     an empty directory
  2. dataset_file_checksums keys files by their path in the run directory, and changed_files
     reports missing and changed files
  3. fold_count_failures compares the manifest's fold count and split directories with the spec
  4. fit_input_checksums produces exactly the dict the trainer's input_checksums records
  5. pair_token turns a pair label into the token used in names

Run: python -m pytest tests/test_dataset_manifest.py
"""
import json
import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.models.train_pair_baselines import input_checksums  # noqa: E402
from src.utils.dataset_manifest import (  # noqa: E402
    SPLIT_FILES,
    changed_files,
    dataset_file_checksums,
    fit_input_checksums,
    fold_count_failures,
    pair_token,
    read_split_dirs,
)


def _make_dataset(root: Path, n_folds: int) -> Path:
    """A minimal CV dataset: resolved_config.yaml, cv_info.json and the split files per fold."""
    root.mkdir()
    (root / 'resolved_config.yaml').write_text('dataset:\n  n_folds: %d\n' % n_folds)
    fold_dirs = [f'fold_{i}' for i in range(n_folds)]
    (root / 'cv_info.json').write_text(json.dumps({'fold_dirs': fold_dirs}))
    for fold in fold_dirs:
        (root / fold).mkdir()
        for name in SPLIT_FILES:
            (root / fold / name).write_text(f'{fold}/{name}\n')
    return root


def test_read_split_dirs(tmp_path):
    assert read_split_dirs(_make_dataset(tmp_path / 'cv', 3)) == ['fold_0', 'fold_1', 'fold_2']
    single = tmp_path / 'single'
    single.mkdir()
    (single / 'train_pairs.csv').write_text('x\n')
    assert read_split_dirs(single) == ['.']
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ValueError):
        read_split_dirs(empty)


def test_checksums_and_changes(tmp_path):
    root = _make_dataset(tmp_path / 'ds', 2)
    files = dataset_file_checksums(root, ['fold_0', 'fold_1'])
    assert 'resolved_config.yaml' in files and 'fold_1/test_pairs.csv' in files
    assert len(files) == 1 + 2 * len(SPLIT_FILES)
    assert changed_files(files, base=root) == []
    (root / 'fold_0' / 'train_pairs.csv').write_text('edited\n')
    (root / 'fold_1' / 'val_pairs.csv').unlink()
    problems = changed_files(files, base=root)
    assert len(problems) == 2
    assert any('fold_0/train_pairs.csv' in p and 'changed' in p for p in problems)
    assert any('fold_1/val_pairs.csv' in p and 'missing' in p for p in problems)


def test_fold_count_failures():
    spec = {'path': 'spec.yaml', 'expect': {'dataset.n_folds': 5}}
    good = {'n_folds': 5, 'datasets': {'b': {'split_dirs': [f'fold_{i}' for i in range(5)]}}}
    assert fold_count_failures(good, spec) == []
    bad = {'n_folds': 4, 'datasets': {'b': {'split_dirs': [f'fold_{i}' for i in range(4)]}}}
    assert len(fold_count_failures(bad, spec)) == 2


def test_fit_inputs_match_trainer(tmp_path):
    root = _make_dataset(tmp_path / 'ds', 2)
    entry = {'dataset_dir': str(root), 'files': dataset_file_checksums(root, ['fold_0', 'fold_1'])}
    expected = fit_input_checksums(entry, 'fold_1', {})
    recorded = input_checksums(root / 'fold_1', None, 'codon', None, root / 'resolved_config.yaml')
    assert recorded == expected


def test_pair_token():
    assert pair_token('PB2-HA') == 'pb2_ha'
    assert pair_token('M1-NS1') == 'm1_ns1'
