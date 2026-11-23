"""Tests for `src/analysis/_importance_helpers.py`.

Covers:
  1. rank_column / share_column build the right names and reject an unknown measure
  2. rank_columns orders by the requested measure, and ranking by gain differs from ranking by
     SHAP when the two disagree
  3. rank_columns raises when the table lacks the measure's rank column
  4. load_importance keeps the literal string 'NA' (Neuraminidase) as a protein value

Run: python tests/test_importance_helpers.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.analysis._importance_helpers import (  # noqa: E402
    IMPORTANCE_MEASURES,
    load_importance,
    rank_column,
    rank_columns,
    share_column,
)


def _importance_table():
    """A four-column table where gain and SHAP disagree on the ordering."""
    return pd.DataFrame({
        'column': [0, 1, 2, 3],
        'protein': ['HA', 'HA', 'NA', 'NA'],
        'site': [1, 2, 1, 2],
        'gain_frac': [0.4, 0.3, 0.2, 0.1],
        'shap_frac': [0.1, 0.2, 0.3, 0.4],
        'gain_rank': [1, 2, 3, 4],
        'shap_rank': [4, 3, 2, 1],
    })


def test_column_names_and_validation():
    assert rank_column('gain') == 'gain_rank'
    assert share_column('shap') == 'shap_frac'
    assert set(IMPORTANCE_MEASURES) == {'gain', 'shap', 'perm'}
    for bad in ('entropy', 'GAIN', ''):
        with pytest.raises(ValueError, match='importance measure must be one of'):
            rank_column(bad)
        with pytest.raises(ValueError, match='importance measure must be one of'):
            share_column(bad)


def test_rank_columns_follows_the_requested_measure():
    table = _importance_table()
    # The two measures are deliberately opposed, so a script that ranked by the wrong one would
    # get exactly the reverse order rather than something subtly off.
    assert list(rank_columns(table, 'gain')) == [0, 1, 2, 3]
    assert list(rank_columns(table, 'shap')) == [3, 2, 1, 0]


def test_rank_columns_raises_on_a_missing_rank_column():
    table = _importance_table().drop(columns=['gain_rank'])
    with pytest.raises(ValueError, match="no 'gain_rank' column"):
        rank_columns(table, 'gain')


def test_load_importance_keeps_neuraminidase():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'site_importance_codon.csv'
        _importance_table().to_csv(path, index=False)
        loaded = load_importance(path)
    # A default pd.read_csv turns the literal 'NA' into NaN and the NA rows vanish.
    assert set(loaded['protein']) == {'HA', 'NA'}
    assert loaded['protein'].isna().sum() == 0

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match='plot_site_importance'):
            load_importance(Path(tmp) / 'absent.csv')


if __name__ == '__main__':
    tests = [
        test_column_names_and_validation,
        test_rank_columns_follows_the_requested_measure,
        test_rank_columns_raises_on_a_missing_rank_column,
        test_load_importance_keeps_neuraminidase,
    ]
    failed = 0
    for t in tests:
        try:
            print(f'... {t.__name__}')
            t()
            print('    OK')
        except Exception as e:
            failed += 1
            print(f'    FAIL: {e}')
    if failed:
        print(f'\n{failed} test(s) failed')
        sys.exit(1)
    print(f'\nAll {len(tests)} tests passed.')
