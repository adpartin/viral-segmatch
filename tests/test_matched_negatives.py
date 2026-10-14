"""Unit tests for the matched balanced-usage negative sampler and the config it requires.

The sampler is in src/datasets/_negative_sampling.py; the profile check is in
`_validate_v2_config` in src/datasets/dataset_segment_pairs_v2.py. Together they are what makes a
negative belong to exactly one metadata group, which is what per-group scoring reads.

Run: python -m pytest tests/test_matched_negatives.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets._negative_regime_sampling import REGIME_NAMES
from src.datasets._negative_sampling import matched_balanced_usage_negatives
from src.datasets._pair_helpers import canonical_pair_key
from src.datasets.dataset_segment_pairs_v2 import _validate_v2_config
from src.utils import schema

PAIR_COLUMNS = schema.build_pair_columns()
HASH_COL = 'cds_dna_hash'
GROUP_SIZES = {
    ('Human', 'H3N2', 2022): 6,
    ('Human', 'H1N1', 2023): 3,
    ('Pig', 'H3N2', 2022): 2,
}
MATCHED_PROFILE = {
    'axes': ['host', 'hn_subtype', 'year'],
    'year_match': 'exact',
    'on_shortfall': 'error',
    'regime_targets': {name: 1.0 if name == 'host_subtype_year' else 0.0
                       for name in REGIME_NAMES},
}
SCHEMA_PAIR = ['RNA-dependent RNA polymerase PB2 subunit',
               'RNA-dependent RNA polymerase catalytic core PB1 subunit']


def make_positives(group_sizes=GROUP_SIZES):
    """Positives shaped like a Hopcroft-Karp selection: one per isolate, sequences used once.

    Args:
        group_sizes: metadata group tuple -> how many positives it holds.

    Returns:
        (positives in `PAIR_COLUMNS`, assembly_id -> metadata group).
    """
    rows, isolate_to_group = [], {}
    for group, size in group_sizes.items():
        host, subtype, year = group
        for index in range(size):
            assembly_id = f'{host}-{subtype}-{year}-{index}'
            isolate_to_group[assembly_id] = group
            row = {column: pd.NA for column in PAIR_COLUMNS}
            row.update({'assembly_id_a': assembly_id, 'assembly_id_b': assembly_id,
                        f'{HASH_COL}_a': f'a-{assembly_id}', f'{HASH_COL}_b': f'b-{assembly_id}',
                        'label': 1})
            row['pair_key'] = canonical_pair_key(row[f'{HASH_COL}_a'], row[f'{HASH_COL}_b'])
            rows.append(row)
    return pd.DataFrame(rows)[PAIR_COLUMNS], isolate_to_group


def draw(positives, isolate_to_group, cooccur=None, seen=None, split_name='train'):
    """Run the sampler on `positives`, blocking their own pair keys by default."""
    if cooccur is None:
        cooccur = set(positives['pair_key'])
    return matched_balanced_usage_negatives(
        positives, cooccur, isolate_to_group, seed=0, hash_col=HASH_COL,
        seen=set() if seen is None else seen, split_name=split_name)


def test_one_negative_per_positive_and_one_use_per_sequence():
    positives, isolate_to_group = make_positives()
    negatives, audit = draw(positives, isolate_to_group)

    assert len(negatives) == len(positives)
    assert list(negatives.columns) == PAIR_COLUMNS
    assert negatives[f'{HASH_COL}_a'].is_unique
    assert negatives[f'{HASH_COL}_b'].is_unique
    assert set(negatives[f'{HASH_COL}_a']) == set(positives[f'{HASH_COL}_a'])
    assert set(negatives[f'{HASH_COL}_b']) == set(positives[f'{HASH_COL}_b'])
    assert [row['positives'] for row in audit] == [row['negatives'] for row in audit]
    assert sum(row['positives'] for row in audit) == len(positives)


def test_both_endpoints_come_from_one_metadata_group():
    positives, isolate_to_group = make_positives()
    negatives, _ = draw(positives, isolate_to_group)

    group_a = negatives['assembly_id_a'].map(isolate_to_group)
    group_b = negatives['assembly_id_b'].map(isolate_to_group)
    assert (group_a == group_b).all()
    assert (negatives['assembly_id_a'] != negatives['assembly_id_b']).all()
    assert (negatives['label'] == 0).all()
    assert (negatives['neg_regime'] == 'host_subtype_year').all()
    assert (negatives['metadata_match_count'] == 3).all()


def test_observed_and_already_drawn_pairs_are_avoided():
    positives, isolate_to_group = make_positives({('Human', 'H3N2', 2022): 6,
                                                  ('Pig', 'H3N2', 2022): 4})
    cooccur = set(positives['pair_key'])
    # Two splits of the same groups, as a fold produces: different positives, one `seen` set.
    one = positives.iloc[::2].reset_index(drop=True)
    two = positives.iloc[1::2].reset_index(drop=True)
    seen = set()

    first, _ = draw(one, isolate_to_group, cooccur=cooccur, seen=seen)
    assert not set(first['pair_key']) & cooccur
    assert seen == set(first['pair_key'])

    second, _ = draw(two, isolate_to_group, cooccur=cooccur, seen=seen, split_name='val')
    assert not set(second['pair_key']) & set(first['pair_key'])
    assert not set(second['pair_key']) & cooccur
    assert seen == set(first['pair_key']) | set(second['pair_key'])


def test_group_of_one_positive_raises():
    positives, isolate_to_group = make_positives({('Pig', 'H3N2', 2022): 1,
                                                  ('Human', 'H3N2', 2022): 4})
    with pytest.raises(ValueError, match='no possible negative partner'):
        draw(positives, isolate_to_group)


def test_group_whose_cross_pairs_are_all_observed_raises():
    positives, isolate_to_group = make_positives({('Pig', 'H3N2', 2022): 2})
    blocked = set(positives['pair_key'])
    for left in positives[f'{HASH_COL}_a']:
        for right in positives[f'{HASH_COL}_b']:
            blocked.add(canonical_pair_key(left, right))
    with pytest.raises(ValueError, match='no assignment'):
        draw(positives, isolate_to_group, cooccur=blocked)


def test_repeated_sequence_in_a_slot_raises():
    positives, isolate_to_group = make_positives({('Pig', 'H3N2', 2022): 3})
    positives.loc[1, f'{HASH_COL}_a'] = positives.loc[0, f'{HASH_COL}_a']
    with pytest.raises(ValueError, match='hopcroft_karp'):
        draw(positives, isolate_to_group)


def test_positive_whose_sides_are_different_isolates_raises():
    positives, isolate_to_group = make_positives({('Pig', 'H3N2', 2022): 3})
    positives.loc[0, 'assembly_id_b'] = 'Pig-H3N2-2022-1'
    with pytest.raises(ValueError, match='different isolates'):
        draw(positives, isolate_to_group)


def balanced_config(profile=None, ratio=1.0):
    """A config with `negative_scope: balanced` and an optional negative_sampling profile."""
    dataset = {'schema_pair': SCHEMA_PAIR, 'neg_to_pos_ratio': ratio,
               'split_strategy': {'negative_scope': 'balanced'}}
    if profile is not None:
        dataset['negative_sampling'] = profile
    return OmegaConf.create({'dataset': dataset})


def test_balanced_accepts_the_matched_profile_and_no_profile():
    _validate_v2_config(balanced_config())
    _validate_v2_config(balanced_config(MATCHED_PROFILE))


@pytest.mark.parametrize('change, ratio, message', [
    ({'year_match': 'binned'}, 1.0, "year_match must be 'exact'"),
    ({'on_shortfall': 'redistribute'}, 1.0, "on_shortfall must be 'error'"),
    ({'regime_targets': {**MATCHED_PROFILE['regime_targets'],
                         'host_subtype_year': 0.5, 'none_match': 0.5}}, 1.0, 'regime_targets'),
    ({'axes': ['host', 'year']}, 1.0, 'axes must cover'),
    ({}, 3.0, 'neg_to_pos_ratio must be 1.0'),
])
def test_balanced_rejects_any_other_profile(change, ratio, message):
    with pytest.raises(ValueError, match=message):
        _validate_v2_config(balanced_config({**MATCHED_PROFILE, **change}, ratio=ratio))


def test_coverage_keeps_a_spread_profile():
    spread = dict(MATCHED_PROFILE)
    spread['year_match'] = 'binned'
    spread['on_shortfall'] = 'redistribute'
    spread['regime_targets'] = {name: 0.125 for name in REGIME_NAMES}
    config = OmegaConf.create({'dataset': {
        'schema_pair': SCHEMA_PAIR,
        'neg_to_pos_ratio': 3.0,
        'split_strategy': {'negative_scope': 'coverage'},
        'negative_sampling': spread,
    }})
    _validate_v2_config(config)
