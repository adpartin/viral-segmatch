"""Build per-site pair features from two loaded caches through the baseline feature loader.

Run: python -m pytest tests/test_site_pair_features.py
"""
import numpy as np
import pandas as pd
import pytest

from src.models._pair_features import load_pair_features_for_baselines
from src.utils.site_utils import SiteCache


def make_cache(protein: str, function: str, codes: list, hashes: list) -> SiteCache:
    return SiteCache(protein=protein, unit='codon', codes=np.array(codes, dtype=np.uint8),
                     hash_to_row={h: row for row, h in enumerate(hashes)},
                     metadata={'function': function,
                               'code_map': {'AAA': 40, 'GAP': 97, 'AFTER_STOP': 98}})


@pytest.fixture
def caches():
    cache_a = make_cache('PB1', 'PB1 function', [[40, 97, 98], [97, 40, 98]], ['a1', 'a2'])
    cache_b = make_cache('HA', 'HA function', [[40, 40], [98, 97]], ['b1', 'b2'])
    return cache_a, cache_b


@pytest.fixture
def pairs():
    return pd.DataFrame({'func_a': ['PB1 function'] * 2, 'func_b': ['HA function'] * 2,
                         'cds_dna_hash_a': ['a2', 'a1'], 'cds_dna_hash_b': ['b1', 'b2'],
                         'label': [1, 0]})


def load(pairs, output_dir, **site):
    return load_pair_features_for_baselines(pairs, pairs, pairs, feature_source='site',
                                            feature_scaling='none', output_dir=output_dir, **site)


def test_slot_a_sites_come_before_slot_b_sites(tmp_path, caches, pairs):
    (X_train, y_train), _, _, scaler = load(pairs, tmp_path, site_caches=caches)
    assert X_train.tolist() == [[97, 40, 98, 40, 40], [40, 97, 98, 98, 97]]
    assert y_train.tolist() == [1, 0]
    assert scaler is None
    columns = pd.read_csv(tmp_path / 'site_feature_columns_codon_ordinal_both.csv')
    assert columns['protein'].tolist() == ['PB1'] * 3 + ['HA'] * 2
    assert columns['site'].tolist() == [1, 2, 3, 1, 2]


def test_requires_two_caches(tmp_path, pairs):
    with pytest.raises(ValueError, match='site_caches'):
        load(pairs, tmp_path)


def test_rejects_caches_in_the_wrong_slots(tmp_path, caches, pairs):
    with pytest.raises(ValueError, match='site_caches is in the wrong order'):
        load(pairs, tmp_path, site_caches=caches[::-1])
