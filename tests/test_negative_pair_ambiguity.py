"""Tests for the negative-pair ambiguity analysis."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.analysis.plot_negative_pair_ambiguity import (  # noqa: E402
    bin_false_positive_rate,
    bin_labels,
    check_bin_edges,
    positive_universe,
    site_distance,
    slot_distances,
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


def test_bin_labels():
    assert bin_labels(EDGES) == ['0-2', '3-5', '6-10', '>10']


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


def test_bin_false_positive_rate():
    negatives = pd.DataFrame({
        'distance_min': [0, 2, 3, 5, 6, 10, 11, 999],
        'is_false_positive': [True, False, True, False, True, False, True, False],
    })
    bins = bin_false_positive_rate(negatives, EDGES, 'run0')

    assert list(bins['bin_label']) == ['0-2', '3-5', '6-10', '>10']
    assert list(bins['n_negatives']) == [2, 2, 2, 2]
    assert list(bins['n_false_positives']) == [1, 1, 1, 1]
    assert list(bins['false_positive_rate']) == [0.5, 0.5, 0.5, 0.5]
    assert set(bins['run']) == {'run0'}
