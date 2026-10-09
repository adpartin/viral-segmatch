"""Tests for the negative-redraw overlap counts and the coordinator's draw checks.

Covers:
  1. draw_overlap_counts gives the largest two-draw intersection and the pairs in two or more draws
  2. With one draw there is no intersection to report
  3. identical_seed_pairs finds seeds whose training negatives are the same set

Run: python -m pytest tests/test_redraw_overlap.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'scripts'))

from resample_balanced_train_negatives import draw_overlap_counts  # noqa: E402
from run_allpairs_balanced_redraws import identical_seed_pairs  # noqa: E402


def test_overlap_counts():
    draws = [{'a', 'b', 'c'}, {'a', 'b', 'd'}, {'a', 'e', 'f'}]
    counts = draw_overlap_counts(draws)
    assert counts['max_negative_pairs_shared_by_two_draws'] == 2   # {a, b}
    assert counts['negative_pairs_in_multiple_draws'] == 2          # a (3 draws), b (2 draws)


def test_single_draw():
    counts = draw_overlap_counts([{'a', 'b'}])
    assert counts['max_negative_pairs_shared_by_two_draws'] is None
    assert counts['negative_pairs_in_multiple_draws'] == 0


def test_identical_seed_pairs(tmp_path):
    negatives = {42: ['x', 'y'], 43: ['y', 'x'], 44: ['x', 'z']}
    for seed, keys in negatives.items():
        seed_dir = tmp_path / f'seed_{seed}'
        seed_dir.mkdir()
        rows = [{'pair_key': key, 'label': 0} for key in keys] + [{'pair_key': 'p', 'label': 1}]
        pd.DataFrame(rows).to_csv(seed_dir / 'train_pairs.csv', index=False)
    assert identical_seed_pairs(tmp_path, [42, 43, 44]) == [[42, 43]]
