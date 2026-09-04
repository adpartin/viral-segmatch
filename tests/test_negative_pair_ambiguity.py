"""Tests for `src/analysis/plot_negative_pair_ambiguity.py`.

Covers:
  1. Bin labels and bin assignment, including the boundaries and the open-ended last bin
  2. check_bin_edges rejects empty, negative and non-increasing edges
  3. site_distance counts differing sites
  4. slot_distances returns one distance per slot, and None for a slot with no known partner
  5. summarize_slots handles min / mean / max, a single available slot, and an unknown summary
  6. positive_universe reads the co-occurrence file, recovers which hash belongs to which slot
     regardless of the sorted pair_key order, skips rows outside the caches, and raises when the
     file is missing or nothing maps
  7. distance_column names each summary distinctly, so a mean run cannot pose as a min run
  8. bin_false_positive_rate counts negatives and false positives per bin

Run: python tests/test_negative_pair_ambiguity.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.analysis.plot_negative_pair_ambiguity import (  # noqa: E402
    assign_bin,
    bin_false_positive_rate,
    bin_labels,
    check_bin_edges,
    distance_column,
    positive_universe,
    site_distance,
    slot_distances,
    summarize_slots,
)
from src.utils.site_utils import SiteCache  # noqa: E402

EDGES = (2, 5, 10)


def _caches():
    """Two tiny per-site caches, one per slot.

    Slot A rows differ from `ha1` by 0, 1 and 4 sites; slot B rows differ from `na1` by 0, 2 and 3.
    `aa9` is a slot-B hash that sorts before the slot-A hashes, which is what exercises the
    pair_key order swap in `positive_universe`.
    """
    cache_a = SiteCache(
        protein='HA', unit='nt',
        codes=np.array([[1, 1, 1, 1], [1, 1, 1, 2], [2, 2, 2, 2]], dtype=np.uint8),
        hash_to_row={'ha1': 0, 'ha2': 1, 'ha3': 2},
        metadata={'other_code': 4})
    cache_b = SiteCache(
        protein='NA', unit='nt',
        codes=np.array([[1, 1, 1], [1, 2, 2], [2, 2, 2], [3, 3, 3]], dtype=np.uint8),
        hash_to_row={'na1': 0, 'na2': 1, 'na3': 2, 'aa9': 3},
        metadata={'other_code': 4})
    return cache_a, cache_b


def test_bin_labels_and_assignment():
    assert bin_labels(EDGES) == ['0-2', '3-5', '6-10', '>10']
    # Boundaries land in the bin whose upper edge they equal, and anything past the last edge
    # falls in the trailing open-ended bin.
    assigned = assign_bin(np.array([0, 2, 3, 5, 6, 10, 11, 999]), EDGES)
    assert list(assigned) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_check_bin_edges_rejects_bad_input():
    check_bin_edges(EDGES)  # the good case must not raise
    with pytest.raises(ValueError, match='at least one'):
        check_bin_edges(())
    with pytest.raises(ValueError, match='non-negative'):
        check_bin_edges((-1, 4))
    with pytest.raises(ValueError, match='strictly increasing'):
        check_bin_edges((5, 2))
    with pytest.raises(ValueError, match='strictly increasing'):
        check_bin_edges((2, 2))


def test_site_distance_counts_differing_sites():
    cache_a, cache_b = _caches()
    assert site_distance(cache_a, 'ha1', 'ha1') == 0
    assert site_distance(cache_a, 'ha1', 'ha2') == 1
    assert site_distance(cache_a, 'ha1', 'ha3') == 4
    assert site_distance(cache_b, 'na1', 'na2') == 2


def test_slot_distances_reports_each_slot_independently():
    cache_a, cache_b = _caches()
    partners_of_a = {'ha1': ['na1']}          # ha1 truly pairs with na1
    partners_of_b = {'na1': ['ha1']}

    # Negative (ha1, na2): slot B was substituted, na2 is 2 sites from ha1's true partner na1.
    # Slot A cannot be scored, because na2 has no known true partner.
    distance_a, distance_b = slot_distances('ha1', 'na2', partners_of_a, partners_of_b,
                                            cache_a, cache_b)
    assert (distance_a, distance_b) == (None, 2)

    # Negative (ha2, na1): the mirror. ha2 is 1 site from na1's true partner ha1.
    distance_a, distance_b = slot_distances('ha2', 'na1', partners_of_a, partners_of_b,
                                            cache_a, cache_b)
    assert (distance_a, distance_b) == (1, None)

    # Neither sequence known: both sides are unmeasurable rather than zero.
    assert slot_distances('ha3', 'na3', partners_of_a, partners_of_b,
                          cache_a, cache_b) == (None, None)


def test_slot_distances_takes_the_closest_of_several_partners():
    cache_a, cache_b = _caches()
    # ha1 pairs with both na1 and na3, so the distance is to whichever is closer to the given na2.
    partners_of_a = {'ha1': ['na1', 'na3']}
    _, distance_b = slot_distances('ha1', 'na2', partners_of_a, {}, cache_a, cache_b)
    assert distance_b == 1  # na2 vs na3 differs at 1 site, vs na1 at 2


def test_summarize_slots():
    assert summarize_slots(2, 8, 'min') == 2
    assert summarize_slots(2, 8, 'max') == 8
    assert summarize_slots(2, 8, 'mean') == 5
    # One slot missing: summarize over what is available rather than treating it as zero.
    assert summarize_slots(None, 8, 'min') == 8
    assert summarize_slots(3, None, 'mean') == 3
    assert summarize_slots(None, None, 'min') is None
    with pytest.raises(ValueError, match="'min', 'mean' or 'max'"):
        summarize_slots(1, 2, 'median')


def test_positive_universe_recovers_slots_from_sorted_pair_keys():
    cache_a, cache_b = _caches()
    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = Path(tmp)
        pd.DataFrame({
            # 'ha1__na1' has slot A first; 'aa9__ha2' has slot B first, because the pair_key is
            # sorted lexicographically and 'aa9' precedes 'ha2'.
            'pair_key': ['ha1__na1', 'aa9__ha2', 'ghost__na3'],
            'num_isolates': [3, 1, 1],
        }).to_csv(dataset_dir / 'cooccurring_sequence_pairs.csv', index=False)
        partners_of_a, partners_of_b = positive_universe(dataset_dir, cache_a, cache_b)

    assert partners_of_a == {'ha1': ['na1'], 'ha2': ['aa9']}
    assert partners_of_b == {'na1': ['ha1'], 'aa9': ['ha2']}
    # The 'ghost' row names a sequence in neither cache and is skipped, not guessed at.
    assert 'ghost' not in partners_of_a and 'ghost' not in partners_of_b


def test_positive_universe_raises_when_unusable():
    cache_a, cache_b = _caches()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match='cooccurring_sequence_pairs.csv'):
            positive_universe(Path(tmp), cache_a, cache_b)

    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = Path(tmp)
        pd.DataFrame({'pair_key': ['ghost1__ghost2'], 'num_isolates': [1]}).to_csv(
            dataset_dir / 'cooccurring_sequence_pairs.csv', index=False)
        with pytest.raises(ValueError, match='different proteins'):
            positive_universe(dataset_dir, cache_a, cache_b)


def test_distance_column_names_the_summary():
    # The summary is in the name so a mean run cannot be read as, or overwrite, a min run.
    assert distance_column('min') == 'distance_min'
    assert distance_column('mean') == 'distance_mean'
    assert distance_column('max') == 'distance_max'
    assert len({distance_column(h) for h in ('min', 'mean', 'max')}) == 3


def test_bin_false_positive_rate():
    negatives = pd.DataFrame({
        'distance_min': [0, 1, 4, 4, 7, 30],
        'is_false_positive': [True, True, True, False, False, False],
    })
    bins = bin_false_positive_rate(negatives, EDGES, 'run0', 'distance_min')

    assert list(bins['bin_label']) == ['0-2', '3-5', '6-10', '>10']
    assert list(bins['n_negatives']) == [2, 2, 1, 1]
    assert list(bins['n_false_positives']) == [2, 1, 0, 0]
    assert list(bins['false_positive_rate']) == [1.0, 0.5, 0.0, 0.0]
    assert set(bins['run']) == {'run0'}


if __name__ == '__main__':
    tests = [
        test_bin_labels_and_assignment,
        test_check_bin_edges_rejects_bad_input,
        test_site_distance_counts_differing_sites,
        test_slot_distances_reports_each_slot_independently,
        test_slot_distances_takes_the_closest_of_several_partners,
        test_summarize_slots,
        test_distance_column_names_the_summary,
        test_positive_universe_recovers_slots_from_sorted_pair_keys,
        test_positive_universe_raises_when_unusable,
        test_bin_false_positive_rate,
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
