"""Tests for `filter_records_kept_by_retention_rule` in src/datasets/_pair_helpers.py.

HA has a gap-pattern selection written inside its length-filtered selection; NP's length-filtered
selection is its final one. Each HA protein row exercises one outcome: kept, incomplete CDS,
dropped CDS length, dropped gap pattern.

Run: python -m pytest tests/test_retention_rule_filter.py
"""
import json

import pandas as pd
import pytest

from src.datasets._pair_helpers import filter_records_kept_by_retention_rule
from src.utils.experiment_utils import file_sha256

SHORT = {'HA fn': 'HA', 'NP fn': 'NP'}
CDS_ROWS = [
    # assembly_id, function, cds_dna_hash, cds_length, is_complete_cds
    ('i1', 'HA fn', 'ha_kept', 6, True),
    ('i2', 'HA fn', 'ha_incomplete', 6, False),
    ('i3', 'HA fn', 'ha_short', 9, True),
    ('i4', 'HA fn', 'ha_rare_gaps', 6, True),
    ('i1', 'NP fn', 'np_kept', 6, True),
    ('i2', 'NP fn', 'np_incomplete', 6, False),
]


def write_selection(directory, keys, manifest):
    directory.mkdir(parents=True)
    pd.DataFrame(keys, columns=['assembly_id', 'function', 'cds_dna_hash']).to_parquet(
        directory / 'selected_cds.parquet', index=False)
    (directory / 'selection_manifest.json').write_text(json.dumps(manifest))


@pytest.fixture
def layout(tmp_path):
    cds_final = tmp_path / 'cds_dna_final.parquet'
    pd.DataFrame(CDS_ROWS, columns=['assembly_id', 'function', 'cds_dna_hash', 'cds_length',
                                    'is_complete_cds']).to_parquet(cds_final, index=False)

    ha_length = tmp_path / 'HA' / 'length_share_above_0.02'
    length_filter = {'dropped_lengths': [9]}
    write_selection(ha_length, [('i1', 'HA fn', 'ha_kept'), ('i4', 'HA fn', 'ha_rare_gaps')],
                    {'length_filter': length_filter})
    ha_final = ha_length / 'gap_pattern_share_above_0.02'
    write_selection(ha_final, [('i1', 'HA fn', 'ha_kept')], {
        'length_filter': length_filter,
        'gap_pattern_filter': {
            'parent_selected_cds_sha256': file_sha256(ha_length / 'selected_cds.parquet')}})
    np_final = tmp_path / 'NP' / 'length_share_above_0.02'
    write_selection(np_final, [('i1', 'NP fn', 'np_kept')], {'length_filter': {'dropped_lengths': []}})

    prot_df = pd.DataFrame({'assembly_id': ['i1', 'i2', 'i3', 'i4', 'i1', 'i2'],
                            'function': ['HA fn'] * 4 + ['NP fn'] * 2,
                            'prot_hash': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']})
    alignment_dirs = {'HA': ha_final, 'NP': np_final}
    return prot_df, cds_final, alignment_dirs


def test_keeps_final_records_and_counts_each_reason(layout):
    prot_df, cds_final, alignment_dirs = layout
    out, audit = filter_records_kept_by_retention_rule(prot_df, cds_final, alignment_dirs, SHORT)
    assert list(out.columns) == ['assembly_id', 'function', 'prot_hash']
    assert out['prot_hash'].tolist() == ['p1', 'p5']
    assert (audit['rows_in'], audit['rows_out'], audit['rows_dropped']) == (6, 2, 4)
    ha = audit['per_protein']['HA']
    assert (ha['rows_in'], ha['rows_kept'], ha['dropped_incomplete_cds'], ha['dropped_cds_length'],
            ha['dropped_gap_pattern']) == (4, 1, 1, 1, 1)
    np_counts = audit['per_protein']['NP']
    assert (np_counts['rows_kept'], np_counts['dropped_incomplete_cds']) == (1, 1)


def test_unexplained_drop_raises(layout):
    prot_df, cds_final, alignment_dirs = layout
    cds = pd.read_parquet(cds_final)
    cds.loc[len(cds)] = ['i5', 'HA fn', 'ha_unselected', 6, True]
    cds.to_parquet(cds_final, index=False)
    prot_df.loc[len(prot_df)] = ['i5', 'HA fn', 'p7']
    with pytest.raises(ValueError, match=r"'HA': \(1, \['i5'.*population differ"):
        filter_records_kept_by_retention_rule(prot_df, cds_final, alignment_dirs, SHORT)


def test_row_without_cds_raises(layout):
    prot_df, cds_final, alignment_dirs = layout
    prot_df.loc[len(prot_df)] = ['i9', 'HA fn', 'p9']
    with pytest.raises(ValueError, match='1 protein rows have no row in cds_dna_final'):
        filter_records_kept_by_retention_rule(prot_df, cds_final, alignment_dirs, SHORT)


def test_changed_parent_selection_raises(layout):
    prot_df, cds_final, alignment_dirs = layout
    parent = alignment_dirs['HA'].parent / 'selected_cds.parquet'
    pd.DataFrame([('i1', 'HA fn', 'ha_kept')],
                 columns=['assembly_id', 'function', 'cds_dna_hash']).to_parquet(parent, index=False)
    with pytest.raises(ValueError, match='was made from a parent'):
        filter_records_kept_by_retention_rule(prot_df, cds_final, alignment_dirs, SHORT)


def test_protein_without_alignment_dir_raises(layout):
    prot_df, cds_final, alignment_dirs = layout
    with pytest.raises(ValueError, match=r"no final alignment directory for \['NP'\]"):
        filter_records_kept_by_retention_rule(prot_df, cds_final, {'HA': alignment_dirs['HA']},
                                              SHORT)


def test_repeated_cds_rows_raise(layout):
    prot_df, cds_final, alignment_dirs = layout
    cds = pd.read_parquet(cds_final)
    pd.concat([cds, cds.iloc[[0]]]).to_parquet(cds_final, index=False)
    with pytest.raises(ValueError, match='1 repeated'):
        filter_records_kept_by_retention_rule(prot_df, cds_final, alignment_dirs, SHORT)
