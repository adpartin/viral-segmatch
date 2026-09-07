"""Tests for restricted and unrestricted negative-pair distances."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.analysis.compare_negative_pair_distances import (  # noqa: E402
    hamming_count_matrix,
    measure_distances,
)
from src.utils.site_utils import SiteCache  # noqa: E402


def _cache(protein: str, hashes: list[str], codes: list[list[int]]) -> SiteCache:
    return SiteCache(
        protein=protein,
        unit='nt',
        codes=np.asarray(codes, dtype=np.uint8),
        hash_to_row={h: i for i, h in enumerate(hashes)},
        metadata={'other_code': 4},
    )


def test_hamming_count_matrix_returns_site_counts():
    cache = _cache('HA', ['a', 'b', 'c'], [[0, 0, 0], [1, 1, 1], [0, 0, 1]])
    distances = hamming_count_matrix(cache, ['a'], ['a', 'b', 'c'])
    assert distances.tolist() == [[0, 3, 1]]


def test_unrestricted_distance_can_be_smaller_than_single_slot_distance():
    cache_a = _cache('HA', ['a1', 'a2', 'a3'], [[0, 0, 0], [1, 1, 1], [0, 0, 1]])
    cache_b = _cache('NA', ['b1', 'b2', 'b3'], [[0, 0, 0], [1, 1, 1], [1, 1, 0]])
    partners_of_a = {'a1': ['b1'], 'a2': ['b2'], 'a3': ['b3']}
    partners_of_b = {'b1': ['a1'], 'b2': ['a2'], 'b3': ['a3']}
    negative = pd.DataFrame({
        'pair_key': ['a1__b2'],
        'cds_dna_hash_a': ['a1'],
        'cds_dna_hash_b': ['b2'],
        'label': [0],
        'pred_prob': [0.8],
        'pred_label': [1],
        'run': ['run0'],
        'is_false_positive': [True],
    })

    measured = measure_distances(
        negative, cache_a, cache_b, partners_of_a, partners_of_b)

    assert measured.loc[0, 'distance_slot_a'] == 3
    assert measured.loc[0, 'distance_slot_b'] == 3
    assert measured.loc[0, 'distance_min'] == 3
    assert measured.loc[0, 'distance_global'] == 2
    assert measured.loc[0, 'distance_gap'] == 1
