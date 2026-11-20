"""Tests for dropping small metadata groups and stratifying the CV folds by group.

Covers `_drop_small_metadata_groups` and the `min_metadata_group_positives` path of
`generate_all_cv_folds_v2` in src/datasets/dataset_segment_pairs_v2.py, with matched
balanced-usage negatives, which need at least 2 positives per group in every split.

Run: python -m pytest tests/test_metadata_group_folds.py
"""
from collections import Counter

import pandas as pd
import pytest

from src.datasets._negative_regime_sampling import REGIME_NAMES
from src.datasets._pair_helpers import build_cooccurrence_set
from src.datasets.dataset_segment_pairs_v2 import (
    _drop_small_metadata_groups,
    generate_all_cv_folds_v2,
)

# Two groups above a minimum of 20 selected positives, two below it.
GROUP_SIZES = {('Human', 'H3N2', 2024): 30, ('Swine', 'H1N1', 2023): 22,
               ('Duck', 'H5N1', 2020): 5, ('Chicken', 'H5N1', 2016): 1}
MATCHED_QUOTAS = {name: 1.0 if name == 'host_subtype_year' else 0.0 for name in REGIME_NAMES}


def protein_frame() -> pd.DataFrame:
    """One isolate per positive, with unique sequences, so Hopcroft-Karp keeps every positive."""
    rows = []
    isolate = 0
    for (host, subtype, year), size in GROUP_SIZES.items():
        for _ in range(size):
            assembly_id = f'iso{isolate:03d}'
            for function in ('A', 'B'):
                seq_hash = f'{function.lower()}{isolate}'
                rows.append({'assembly_id': assembly_id, 'brc_fea_id': f'{assembly_id}_{function}',
                             'genbank_ctg_id': f'{assembly_id}_{function}_ctg',
                             'prot_seq': f'M{seq_hash}', 'ctg_dna_seq': f'ATG{seq_hash}',
                             'canonical_segment': function, 'function': function,
                             'prot_hash': seq_hash, 'ctg_dna_hash': f'ctg_{seq_hash}',
                             'cds_dna_hash': f'cds_{seq_hash}',
                             'host': host, 'hn_subtype': subtype, 'year': year})
            isolate += 1
    return pd.DataFrame(rows)


def build_folds(frame: pd.DataFrame, **overrides) -> list:
    cooccur_pairs, cooccur_stats = build_cooccurrence_set(frame, hash_col='cds_dna_hash')
    cooccur_stats['scope'] = 'corpus'
    kwargs = dict(df=frame, cooccur_pairs=cooccur_pairs, cooccur_stats=cooccur_stats, n_folds=5,
                  seed=7, neg_to_pos_ratio=1.0, val_ratio=0.1, schema_pair=('A', 'B'),
                  axes_for_flags=[], pair_key_alphabet='nt_cds', negative_scope='balanced',
                  axis_quotas=MATCHED_QUOTAS, sampling_axes=['host', 'hn_subtype', 'year'],
                  year_match='exact', on_shortfall='error',
                  positive_selection_method='hopcroft_karp', min_metadata_group_positives=20)
    kwargs.update(overrides)
    folds = list(generate_all_cv_folds_v2(**kwargs))
    return folds


def test_drop_keeps_whole_groups_at_or_above_the_minimum():
    frame = protein_frame()
    group_of = {row.assembly_id: (row.host, row.hn_subtype, row.year)
                for row in frame.drop_duplicates('assembly_id').itertuples()}
    selected = pd.DataFrame({'assembly_id_a': list(group_of), 'pair_key': list(group_of)})
    kept, kept_groups, record = _drop_small_metadata_groups(selected, group_of, 20)
    assert len(kept) == len(kept_groups) == 52
    assert set(kept_groups) == {str(('Human', 'H3N2', 2024)), str(('Swine', 'H1N1', 2023))}
    assert (record['groups_in'], record['groups_kept'], record['groups_dropped']) == (4, 2, 2)
    assert (record['positives_in'], record['positives_kept'], record['positives_dropped']) == (58, 52, 6)
    assert record['dropped_groups'] == [{'group': ['Chicken', 'H5N1', 2016], 'positives': 1},
                                        {'group': ['Duck', 'H5N1', 2020], 'positives': 5}]


def test_drop_raises_when_no_group_reaches_the_minimum():
    selected = pd.DataFrame({'assembly_id_a': ['i1', 'i2'], 'pair_key': ['k1', 'k2']})
    with pytest.raises(ValueError, match='no metadata group has 20'):
        _drop_small_metadata_groups(selected, {'i1': ('g',), 'i2': ('h',)}, 20)


def test_folds_hold_two_positives_per_kept_group_in_every_split():
    frame = protein_frame()
    group_of = {row.assembly_id: (row.host, row.hn_subtype, row.year)
                for row in frame.drop_duplicates('assembly_id').itertuples()}
    folds = build_folds(frame)
    assert len(folds) == 5
    record = folds[0]['metadata_group_drop']
    assert (record['groups_dropped'], record['positives_dropped']) == (2, 6)

    test_keys = []
    for fold in folds:
        for split in ('train_pairs', 'val_pairs', 'test_pairs'):
            pairs = fold[split]
            positives = pairs[pairs['label'] == 1]
            per_group = Counter(positives['assembly_id_a'].map(group_of))
            assert set(per_group) == {('Human', 'H3N2', 2024), ('Swine', 'H1N1', 2023)}
            assert min(per_group.values()) >= 2
            assert (pairs['label'] == 0).sum() == len(positives)
        test = fold['test_pairs']
        test_keys.extend(test.loc[test['label'] == 1, 'pair_key'])
    assert len(test_keys) == len(set(test_keys)) == 52


def test_without_the_minimum_a_one_positive_group_stops_the_build():
    with pytest.raises(ValueError, match='has 1 positive'):
        build_folds(protein_frame(), min_metadata_group_positives=None)


@pytest.mark.parametrize(('overrides', 'message'), [
    ({'positive_selection_method': 'all'}, 'needs an active positive_selection_method'),
    ({'axis_quotas': None}, 'needs axis_quotas'),
])
def test_minimum_needs_selection_and_metadata_groups(overrides, message):
    with pytest.raises(ValueError, match=message):
        build_folds(protein_frame(), **overrides)
