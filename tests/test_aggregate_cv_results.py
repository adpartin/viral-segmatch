"""Tests for the strict mode of `scripts/aggregate_cv_results.py`.

Covers:
  1. Without strict mode, a fold without test_predicted.csv is skipped and the rest aggregated
  2. In strict mode, the same input exits non-zero and writes nothing

Run: python -m pytest tests/test_aggregate_cv_results.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'scripts'))

from aggregate_cv_results import aggregate  # noqa: E402


def _folds(tmp_path) -> list:
    """Three fold directories; the last has no test_predicted.csv."""
    dirs = []
    for i in range(3):
        fold = tmp_path / f'fold{i}'
        fold.mkdir()
        if i < 2:
            pd.DataFrame({'label': [1, 0, 1, 0], 'pred_prob': [0.9, 0.2, 0.4, 0.6]}).to_csv(
                fold / 'test_predicted.csv', index=False)
        dirs.append(fold)
    return dirs


def test_lenient_mode_skips_missing_fold(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    aggregate(_folds(tmp_path), out)
    summary = pd.read_csv(out / 'cv_summary.csv')
    assert list(summary['fold_id'][:2].astype(int)) == [0, 1]


def test_strict_mode_fails_on_missing_fold(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.raises(SystemExit) as raised:
        aggregate(_folds(tmp_path), out, strict=True)
    assert raised.value.code == 1
    assert not (out / 'cv_summary.csv').exists()
