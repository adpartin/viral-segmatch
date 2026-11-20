"""Tests for dropping small metadata groups and stratifying the CV folds by group.

Covers `_drop_small_metadata_groups`, the `min_metadata_group_positives` path of
`generate_all_cv_folds_v2` and its rule in `_validate_v2_config`, all in
src/datasets/dataset_segment_pairs_v2.py, with matched balanced-usage negatives, which need at
least 2 positives per group in every split. Also covers `check_selected_test_positives` in
src/analysis/audit_pair_datasets.py.

Run: python -m pytest tests/test_metadata_group_folds.py
"""
import json
from collections import Counter

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.analysis.audit_pair_datasets import check_selected_test_positives
from src.datasets._negative_regime_sampling import REGIME_NAMES
from src.datasets._pair_helpers import build_cooccurrence_set
from src.datasets.dataset_segment_pairs_v2 import (
    _drop_small_metadata_groups,
    _validate_v2_config,
    generate_all_cv_folds_v2,
)

# Two groups above a minimum of 20 selected positives, two below it.
GROUP_SIZES = {('Human', 'H3N2', 2024): 30, ('Swine', 'H1N1', 2023): 22,
               ('Duck', 'H5N1', 2020): 5, ('Chicken', 'H5N1', 2016): 1}
AXES = ['host', 'hn_subtype', 'year']
MATCHED_QUOTAS = {name: 1.0 if name == 'host_subtype_year' else 0.0 for name in REGIME_NAMES}
MATCHED_PROFILE = {'axes': AXES, 'year_match': 'exact', 'on_shortfall': 'error',
                   'regime_targets': MATCHED_QUOTAS}


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
                  axis_quotas=MATCHED_QUOTAS, sampling_axes=AXES,
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
    kept, kept_groups, record = _drop_small_metadata_groups(selected, group_of, AXES, 20)
    assert len(kept) == len(kept_groups) == 52
    assert record['axes'] == AXES
    assert set(kept_groups) == {str(('Human', 'H3N2', 2024)), str(('Swine', 'H1N1', 2023))}
    assert (record['groups_in'], record['groups_kept'], record['groups_dropped']) == (4, 2, 2)
    assert (record['positives_in'], record['positives_kept'], record['positives_dropped']) == (58, 52, 6)
    assert record['dropped_groups'] == [{'group': ['Chicken', 'H5N1', 2016], 'positives': 1},
                                        {'group': ['Duck', 'H5N1', 2020], 'positives': 5}]


def test_drop_raises_when_no_group_reaches_the_minimum():
    selected = pd.DataFrame({'assembly_id_a': ['i1', 'i2'], 'pair_key': ['k1', 'k2']})
    with pytest.raises(ValueError, match='no metadata group has 20'):
        _drop_small_metadata_groups(selected, {'i1': ('g',), 'i2': ('h',)}, ['host'], 20)


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


def validator_config(minimum=20, method='hopcroft_karp', scope='balanced',
                     profile=MATCHED_PROFILE):
    """A cross-validation config with random splits, as active positive selection requires."""
    dataset = {'schema_pair': ['A', 'B'], 'neg_to_pos_ratio': 1.0, 'n_folds': 5,
               'split_strategy': {'mode': 'random', 'negative_scope': scope},
               'positive_pair_selection': {'method': method, 'ordering': 'pair_key'},
               'min_metadata_group_positives': minimum}
    if profile is not None:
        dataset['negative_sampling'] = profile
    return OmegaConf.create({'dataset': dataset})


def test_validator_accepts_the_minimum_with_selection_and_matched_negatives():
    _validate_v2_config(validator_config())
    _validate_v2_config(validator_config(minimum=None, method='all', profile=None))


@pytest.mark.parametrize(('changes', 'message'), [
    ({'minimum': 0}, 'positive integer'),
    ({'minimum': 20.5}, 'positive integer'),
    ({'minimum': True}, 'positive integer'),
    ({'method': 'all'}, "not 'all'"),
    ({'profile': None}, 'needs matched negatives'),
    ({'scope': 'coverage'}, 'needs matched negatives'),
])
def test_validator_rejects_the_minimum_without_its_requirements(changes, message):
    with pytest.raises(ValueError, match=message):
        _validate_v2_config(validator_config(**changes))


@pytest.fixture(scope='module')
def flagged_folds():
    """Folds whose pairs carry `<axis>_a` columns, which the audit reads groups from."""
    return build_folds(protein_frame(), axes_for_flags=AXES)


def write_run(run_dir, folds):
    """Write the files `check_selected_test_positives` reads, the way the builder names them."""
    folds[0]['positive_selection_manifest'].to_csv(run_dir / 'positive_pair_selection.csv',
                                                   index=False)
    (run_dir / 'metadata_groups_dropped.json').write_text(
        json.dumps(folds[0]['metadata_group_drop']))
    split_dirs = []
    for fold in folds:
        split_dir = f"fold_{fold['fold_id']}"
        (run_dir / split_dir).mkdir()
        fold['test_pairs'].to_csv(run_dir / split_dir / 'test_pairs.csv', index=False)
        split_dirs.append(split_dir)
    return split_dirs


def test_audit_passes_test_splits_of_the_kept_groups(tmp_path, flagged_folds):
    split_dirs = write_run(tmp_path, flagged_folds)
    config = OmegaConf.create({'dataset': {'min_metadata_group_positives': 20}})
    assert check_selected_test_positives(tmp_path, split_dirs, config) == []


def test_audit_reports_a_drop_record_without_the_minimum(tmp_path, flagged_folds):
    split_dirs = write_run(tmp_path, flagged_folds)
    config = OmegaConf.create({'dataset': {'min_metadata_group_positives': None}})
    failures = check_selected_test_positives(tmp_path, split_dirs, config)
    assert any('exists, but' in message for message in failures)
    assert any('52 positives, expected 58' in message for message in failures)


def test_audit_reports_a_positive_of_a_dropped_group_and_a_missing_one(tmp_path, flagged_folds):
    split_dirs = write_run(tmp_path, flagged_folds)
    test_path = tmp_path / split_dirs[0] / 'test_pairs.csv'
    test = pd.read_csv(test_path, keep_default_na=False, na_values=[''])
    first_positive = test.index[test['label'] == 1][0]
    test.loc[first_positive, ['host_a', 'hn_subtype_a', 'year_a']] = ['Duck', 'H5N1', 2020]
    second_positive = test.index[test['label'] == 1][1]
    test.drop(index=second_positive).to_csv(test_path, index=False)
    config = OmegaConf.create({'dataset': {'min_metadata_group_positives': 20}})
    failures = check_selected_test_positives(tmp_path, split_dirs, config)
    assert failures == ['1 test positives belong to a dropped metadata group',
                        'the test splits hold 51 positives, expected 52 '
                        '(58 selected minus 6 dropped)']
