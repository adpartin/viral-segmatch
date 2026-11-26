"""Tests for dropping small metadata groups and stratifying the CV folds by group.

Covers `_drop_small_metadata_groups`, the `min_metadata_group_positives` path of
`generate_all_cv_folds_v2` and its rule in `_validate_v2_config`, all in
src/datasets/dataset_segment_pairs_v2.py, with matched balanced-usage negatives, which need at
least 2 positives per group in every split. Also covers `check_selected_test_positives`,
`check_dropped_and_removed_groups_absent` and `check_matched_negatives` in
src/analysis/audit_pair_datasets.py.

Run: python -m pytest tests/test_metadata_group_folds.py
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.model_selection import KFold, train_test_split

from src.analysis.audit_pair_datasets import (
    check_dropped_and_removed_groups_absent,
    check_matched_negatives,
    check_selected_test_positives,
)
from src.datasets._negative_regime_sampling import REGIME_NAMES
from src.datasets._pair_helpers import build_cooccurrence_set, canonical_pair_key
from src.datasets._positive_pair_selection import select_positive_pairs
from src.datasets.dataset_segment_pairs_v2 import (
    _drop_small_metadata_groups,
    _validate_v2_config,
    create_positive_pairs_v2,
    generate_all_cv_folds_v2,
)
from src.utils import schema

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


def test_without_the_minimum_folds_are_kfold_over_the_selected_positives():
    """With the minimum off, each fold's test and validation positives are those of KFold over the
    selected positives followed by an unstratified validation split, for the same seed."""
    frame = protein_frame()
    folds = build_folds(frame, min_metadata_group_positives=None, axis_quotas=None)
    positives, _ = create_positive_pairs_v2(frame, schema_pair=('A', 'B'),
                                            pair_key_alphabet='nt_cds')
    hash_col_a, hash_col_b = schema.hash_col_ab('nt_cds')
    selected, _ = select_positive_pairs(positives, method='hopcroft_karp', hash_col_a=hash_col_a,
                                        hash_col_b=hash_col_b, ordering='pair_key')
    splits = KFold(n_splits=5, shuffle=True, random_state=7).split(np.arange(len(selected)))
    for fold_i, (fold, (trainval_idx, test_idx)) in enumerate(zip(folds, splits)):
        _, val = train_test_split(selected.iloc[trainval_idx], test_size=0.1 / (1.0 - 1.0 / 5),
                                  random_state=7 + fold_i)
        for split, expected in (('test_pairs', selected.iloc[test_idx]), ('val_pairs', val)):
            pairs = fold[split]
            assert set(pairs.loc[pairs['label'] == 1, 'pair_key']) == set(expected['pair_key'])


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


def unmatchable_group_cooccurrence(frame: pd.DataFrame, host: str) -> set:
    """The corpus pairs plus every cross pair inside one host's metadata group.

    With all of a group's cross pairs observed, no one-to-one assignment of its sequences is left,
    so the group cannot be matched in any split of any fold.

    Args:
        frame: the protein frame, from `protein_frame`.
        host: the host of the group to block, which `GROUP_SIZES` gives one group each.

    Returns:
        The canonical pair keys to pass as `cooccur_pairs`.
    """
    cooccur_pairs, _ = build_cooccurrence_set(frame, hash_col='cds_dna_hash')
    group = frame[frame['host'] == host]
    slot_a = group.loc[group['function'] == 'A', 'cds_dna_hash']
    slot_b = group.loc[group['function'] == 'B', 'cds_dna_hash']
    blocked = {canonical_pair_key(left, right) for left in slot_a for right in slot_b}
    return set(cooccur_pairs) | blocked


def test_an_unmatchable_group_stops_the_build_by_default():
    frame = protein_frame()
    with pytest.raises(ValueError, match='no assignment'):
        build_folds(frame, cooccur_pairs=unmatchable_group_cooccurrence(frame, 'Swine'))


def test_dropping_an_unmatchable_group_keeps_the_build_going_and_records_it():
    frame = protein_frame()
    folds = build_folds(frame, cooccur_pairs=unmatchable_group_cooccurrence(frame, 'Swine'),
                        on_infeasible_metadata_group='drop_group')
    assert len(folds) == 5

    record = folds[0]['metadata_group_infeasible']
    assert record['axes'] == AXES
    assert (record['groups_in'], record['groups_kept'], record['groups_removed']) == (2, 1, 1)
    assert (record['positives_in'], record['positives_kept'],
            record['positives_removed']) == (52, 30, 22)
    removed = record['removed_groups']
    assert len(removed) == 1
    assert removed[0]['group'] == ['Swine', 'H1N1', 2023]
    assert removed[0]['positives'] == 22
    # One entry per split of every fold, because the block leaves the group no assignment anywhere.
    assert removed[0]['infeasible_in'] == [f'fold_{fold}/{split}' for fold in range(5)
                                           for split in ('train', 'val', 'test')]

    group_of = {row.assembly_id: (row.host, row.hn_subtype, row.year)
                for row in frame.drop_duplicates('assembly_id').itertuples()}
    for fold in folds:
        for split in ('train_pairs', 'val_pairs', 'test_pairs'):
            pairs = fold[split]
            groups = set(pairs['assembly_id_a'].map(group_of)) | set(
                pairs['assembly_id_b'].map(group_of))
            assert groups == {('Human', 'H3N2', 2024)}


def test_removing_a_group_leaves_every_other_positive_in_its_fold_and_split():
    """The folds are partitioned before the removal, so the kept group's positives do not move."""
    frame = protein_frame()
    group_of = {row.assembly_id: (row.host, row.hn_subtype, row.year)
                for row in frame.drop_duplicates('assembly_id').itertuples()}
    feasible = build_folds(frame)
    removed = build_folds(frame, cooccur_pairs=unmatchable_group_cooccurrence(frame, 'Swine'),
                          on_infeasible_metadata_group='drop_group')

    for before, after in zip(feasible, removed):
        for split in ('train_pairs', 'val_pairs', 'test_pairs'):
            kept_keys = {}
            for name, fold in (('before', before), ('after', after)):
                pairs = fold[split]
                positives = pairs[pairs['label'] == 1]
                human = positives['assembly_id_a'].map(group_of) == ('Human', 'H3N2', 2024)
                kept_keys[name] = set(positives.loc[human, 'pair_key'])
            assert kept_keys['before'] == kept_keys['after']
            assert kept_keys['after']


def test_the_rule_changes_nothing_when_every_group_can_be_matched():
    """With nothing to remove, 'drop_group' yields the folds the default does, pair table for pair
    table, so rebuilding a dataset that never met an unmatchable group reproduces it."""
    frame = protein_frame()
    default = build_folds(frame)
    with_rule = build_folds(frame, on_infeasible_metadata_group='drop_group')

    assert with_rule[0]['metadata_group_infeasible']['groups_removed'] == 0
    for before, after in zip(default, with_rule):
        for split in ('train_pairs', 'val_pairs', 'test_pairs'):
            pd.testing.assert_frame_equal(before[split], after[split])


def test_dropping_a_group_needs_the_metadata_group_minimum():
    with pytest.raises(ValueError, match='needs min_metadata_group_positives'):
        build_folds(protein_frame(), min_metadata_group_positives=None,
                    on_infeasible_metadata_group='drop_group')


def test_an_unknown_infeasible_group_choice_is_rejected():
    with pytest.raises(ValueError, match='on_infeasible_metadata_group must be in'):
        build_folds(protein_frame(), on_infeasible_metadata_group='skip')


def validator_config(minimum=20, method='hopcroft_karp', scope='balanced',
                     profile=MATCHED_PROFILE, on_infeasible=None):
    """A cross-validation config with random splits, as active positive selection requires."""
    dataset = {'schema_pair': ['A', 'B'], 'neg_to_pos_ratio': 1.0, 'n_folds': 5,
               'split_strategy': {'mode': 'random', 'negative_scope': scope},
               'positive_pair_selection': {'method': method, 'ordering': 'pair_key'},
               'min_metadata_group_positives': minimum}
    if profile is not None:
        dataset['negative_sampling'] = profile
    if on_infeasible is not None:
        dataset['on_infeasible_metadata_group'] = on_infeasible
    return OmegaConf.create({'dataset': dataset})


def test_validator_accepts_the_minimum_with_selection_and_matched_negatives():
    _validate_v2_config(validator_config())
    _validate_v2_config(validator_config(minimum=None, method='all', profile=None))
    _validate_v2_config(validator_config(on_infeasible='drop_group'))
    _validate_v2_config(validator_config(minimum=None, method='all', profile=None,
                                         on_infeasible='error'))


@pytest.mark.parametrize(('changes', 'message'), [
    ({'on_infeasible': 'skip'}, 'on_infeasible_metadata_group must be in'),
    ({'on_infeasible': 'drop_group', 'minimum': None, 'method': 'all', 'profile': None},
     'needs dataset.min_metadata_group_positives'),
])
def test_validator_rejects_an_unusable_infeasible_group_choice(changes, message):
    with pytest.raises(ValueError, match=message):
        _validate_v2_config(validator_config(**changes))


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
    """Write the files the audit checks read, the way the builder names them."""
    folds[0]['positive_selection_manifest'].to_csv(run_dir / 'positive_pair_selection.csv',
                                                   index=False)
    (run_dir / 'metadata_groups_dropped.json').write_text(
        json.dumps(folds[0]['metadata_group_drop']))
    if folds[0].get('metadata_group_infeasible') is not None:
        (run_dir / 'metadata_groups_infeasible.json').write_text(
            json.dumps(folds[0]['metadata_group_infeasible']))
    split_dirs = []
    for fold in folds:
        split_dir = f"fold_{fold['fold_id']}"
        (run_dir / split_dir).mkdir()
        for split in ('train', 'val', 'test'):
            fold[f'{split}_pairs'].to_csv(run_dir / split_dir / f'{split}_pairs.csv', index=False)
        split_dirs.append(split_dir)
    return split_dirs


def test_audit_passes_test_splits_of_the_kept_groups(tmp_path, flagged_folds):
    split_dirs = write_run(tmp_path, flagged_folds)
    config = OmegaConf.create({'dataset': {'min_metadata_group_positives': 20}})
    assert check_selected_test_positives(tmp_path, split_dirs, config) == []


@pytest.fixture(scope='module')
def unmatchable_folds():
    """Folds built with one group removed because its matched negatives cannot be drawn."""
    frame = protein_frame()
    return build_folds(frame, axes_for_flags=AXES,
                       cooccur_pairs=unmatchable_group_cooccurrence(frame, 'Swine'),
                       on_infeasible_metadata_group='drop_group')


def infeasible_config(on_infeasible='drop_group'):
    return OmegaConf.create({'dataset': {'min_metadata_group_positives': 20,
                                         'on_infeasible_metadata_group': on_infeasible}})


def test_audit_passes_test_splits_when_a_group_was_removed(tmp_path, unmatchable_folds):
    split_dirs = write_run(tmp_path, unmatchable_folds)
    assert check_selected_test_positives(tmp_path, split_dirs, infeasible_config()) == []


def test_audit_reports_a_removal_record_the_config_does_not_ask_for(tmp_path, unmatchable_folds):
    split_dirs = write_run(tmp_path, unmatchable_folds)
    failures = check_selected_test_positives(tmp_path, split_dirs, infeasible_config('error'))
    assert failures == ["metadata_groups_infeasible.json exists, but "
                        "dataset.on_infeasible_metadata_group is 'error'",
                        'the test splits hold 1 kept metadata groups, expected 2 '
                        '(metadata_groups_dropped.json records 2 kept minus 0 removed as '
                        'unmatchable)',
                        'the test splits hold 30 positives, expected 52 '
                        '(58 selected minus 6 dropped minus 0 removed as unmatchable)']


def test_audit_reports_a_missing_removal_record(tmp_path, unmatchable_folds):
    split_dirs = write_run(tmp_path, unmatchable_folds)
    (tmp_path / 'metadata_groups_infeasible.json').unlink()
    failures = check_selected_test_positives(tmp_path, split_dirs, infeasible_config())
    assert failures[0] == ("metadata_groups_infeasible.json is absent, but "
                           "dataset.on_infeasible_metadata_group is 'drop_group'")


def test_audit_reports_a_test_positive_of_a_removed_group(tmp_path, unmatchable_folds):
    split_dirs = write_run(tmp_path, unmatchable_folds)
    test_path = tmp_path / split_dirs[0] / 'test_pairs.csv'
    test = pd.read_csv(test_path, keep_default_na=False, na_values=[''])
    first_positive = test.index[test['label'] == 1][0]
    test.loc[first_positive, ['host_a', 'hn_subtype_a', 'year_a']] = ['Swine', 'H1N1', 2023]
    test.to_csv(test_path, index=False)
    failures = check_selected_test_positives(tmp_path, split_dirs, infeasible_config())
    # Relabelling the positive also puts the removed group back among the test splits' groups, so
    # the kept-group count and that group's size are reported too.
    assert failures[0] == ('1 test positives belong to a metadata group '
                           'metadata_groups_infeasible.json says was removed')
    assert len(failures) == 3


def test_audit_reports_removal_totals_that_disagree_with_the_removed_groups(tmp_path,
                                                                           unmatchable_folds):
    split_dirs = write_run(tmp_path, unmatchable_folds)
    record = dict(unmatchable_folds[0]['metadata_group_infeasible'], removed_groups=[])
    (tmp_path / 'metadata_groups_infeasible.json').write_text(json.dumps(record))
    failures = check_selected_test_positives(tmp_path, split_dirs, infeasible_config())
    assert failures[0] == ('metadata_groups_infeasible.json totals (groups removed, positives '
                           'removed) are (1, 22), its list gives (0, 0)')


def test_audit_finds_no_row_of_a_dropped_or_removed_group(tmp_path, flagged_folds,
                                                          unmatchable_folds):
    for name, folds in (('flagged', flagged_folds), ('unmatchable', unmatchable_folds)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        split_dirs = write_run(run_dir, folds)
        assert check_dropped_and_removed_groups_absent(run_dir, split_dirs) == []


SWINE = ['Swine', 'H1N1', 2023]  # removed as unmatchable in `unmatchable_folds`
DUCK = ['Duck', 'H5N1', 2020]  # dropped for the minimum in both fixtures


@pytest.mark.parametrize('split', ['train', 'val', 'test'])
@pytest.mark.parametrize('label', [0, 1])
@pytest.mark.parametrize(('group', 'record'), [(SWINE, 'metadata_groups_infeasible.json'),
                                               (DUCK, 'metadata_groups_dropped.json')])
def test_audit_reports_a_row_of_a_dropped_or_removed_group(tmp_path, unmatchable_folds, split,
                                                           label, group, record):
    split_dirs = write_run(tmp_path, unmatchable_folds)
    path = tmp_path / split_dirs[1] / f'{split}_pairs.csv'
    pairs = pd.read_csv(path, keep_default_na=False, na_values=[''])
    row = pairs.index[pairs['label'] == label][0]
    for slot in ('a', 'b'):
        pairs.loc[row, [f'{axis}_{slot}' for axis in AXES]] = group
    pairs.to_csv(path, index=False)
    failures = check_dropped_and_removed_groups_absent(tmp_path, split_dirs)
    counts = '1 positives and 0 negatives' if label == 1 else '0 positives and 1 negatives'
    assert failures == [f'{split_dirs[1]}/{split}: {counts} have an endpoint in a group {record} '
                        f'lists']


def test_audit_reports_a_row_whose_slot_b_alone_is_in_a_removed_group(tmp_path,
                                                                      unmatchable_folds):
    split_dirs = write_run(tmp_path, unmatchable_folds)
    path = tmp_path / split_dirs[0] / 'val_pairs.csv'
    pairs = pd.read_csv(path, keep_default_na=False, na_values=[''])
    negative = pairs.index[pairs['label'] == 0][0]
    pairs.loc[negative, [f'{axis}_b' for axis in AXES]] = SWINE
    pairs.to_csv(path, index=False)
    failures = check_dropped_and_removed_groups_absent(tmp_path, split_dirs)
    assert failures == [f'{split_dirs[0]}/val: 0 positives and 1 negatives have an endpoint in a '
                        f'group metadata_groups_infeasible.json lists']


def test_audit_reports_group_records_with_different_axes(tmp_path, unmatchable_folds):
    split_dirs = write_run(tmp_path, unmatchable_folds)
    record = dict(unmatchable_folds[0]['metadata_group_infeasible'], axes=['host', 'year'])
    (tmp_path / 'metadata_groups_infeasible.json').write_text(json.dumps(record))
    failures = check_dropped_and_removed_groups_absent(tmp_path, split_dirs)
    assert len(failures) == 1
    assert failures[0].startswith('the metadata group records name different axes')


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
                        '(58 selected minus 6 dropped minus 0 removed as unmatchable)']


def test_audit_reports_drop_totals_that_disagree_with_the_dropped_groups(tmp_path, flagged_folds):
    split_dirs = write_run(tmp_path, flagged_folds)
    record = dict(flagged_folds[0]['metadata_group_drop'], dropped_groups=[])
    (tmp_path / 'metadata_groups_dropped.json').write_text(json.dumps(record))
    config = OmegaConf.create({'dataset': {'min_metadata_group_positives': 20}})
    failures = check_selected_test_positives(tmp_path, split_dirs, config)
    assert failures == ['metadata_groups_dropped.json totals (groups dropped, positives dropped, '
                        'positives kept) are (2, 6, 52), its list of dropped groups gives '
                        '(0, 0, 58)',
                        'the test splits hold 52 positives, expected 58 '
                        '(58 selected minus 0 dropped minus 0 removed as unmatchable)']


def test_audit_reports_a_dropped_group_listed_twice(tmp_path, flagged_folds):
    split_dirs = write_run(tmp_path, flagged_folds)
    record = flagged_folds[0]['metadata_group_drop']
    duck = [entry for entry in record['dropped_groups'] if entry['group'][0] == 'Duck']
    record = dict(record, dropped_groups=record['dropped_groups'] + duck)
    (tmp_path / 'metadata_groups_dropped.json').write_text(json.dumps(record))
    config = OmegaConf.create({'dataset': {'min_metadata_group_positives': 20}})
    failures = check_selected_test_positives(tmp_path, split_dirs, config)
    assert failures == ["metadata_groups_dropped.json lists these groups more than once: "
                        "[\"('Duck', 'H5N1', 2020)\"]",
                        'metadata_groups_dropped.json totals (groups dropped, positives dropped, '
                        'positives kept) are (2, 6, 52), its list of dropped groups gives '
                        '(3, 11, 47)',
                        'the test splits hold 52 positives, expected 47 '
                        '(58 selected minus 11 dropped minus 0 removed as unmatchable)']


def test_audit_passes_the_builders_matched_negatives(tmp_path, flagged_folds):
    split_dirs = write_run(tmp_path, flagged_folds)
    assert check_matched_negatives(tmp_path, split_dirs, AXES) == []


def test_audit_reports_a_negative_across_groups_and_a_missing_value(tmp_path, flagged_folds):
    split_dirs = write_run(tmp_path, flagged_folds)
    train_path = tmp_path / split_dirs[0] / 'train_pairs.csv'
    train = pd.read_csv(train_path, keep_default_na=False, na_values=[''])
    negatives = train.index[train['label'] == 0]
    train.loc[negatives[0], 'host_b'] = 'Duck'
    train.loc[negatives[1], 'year_b'] = None
    positive = train.index[train['label'] == 1][0]
    train.loc[positive, 'host_b'] = 'Duck'
    train.to_csv(train_path, index=False)
    failures = check_matched_negatives(tmp_path, split_dirs, AXES)
    assert failures == [f"{split_dirs[0]}/train: negatives whose endpoints differ, by axis: "
                        f"{{'host': 1, 'year': 1}}"]
