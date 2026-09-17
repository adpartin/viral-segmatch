"""Tests for `src/analysis/summarize_pair_capacity.py`.

Covers:
  1. common_isolate_cohort keeps only isolates carrying every requested protein
  2. It rejects a protein that appears in no row, rather than returning an empty cohort
  3. isolate_overlap reports shared counts and Jaccard, and orders by descending Jaccard
  4. Disjoint matchings give Jaccard 0, identical ones give 1
  5. isolate_overlap marks the combinations whose two schema pairs share a protein
  6. sequence_reuse counts distinct partner sequences, and rejects an empty pair
  7. sort_by_segment orders on the two segment numbers rather than on the label text
  8. hk_matrix is symmetric, has an empty diagonal, and rejects a pair it has no row for
  9. segment_numbers maps each protein to its segment, and rejects one spanning two segments
 10. The column lists are what the CSVs promise

Run: python tests/test_summarize_pair_capacity.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.analysis.summarize_pair_capacity import (  # noqa: E402
    CAPACITY_COLUMNS,
    OVERLAP_COLUMNS,
    REUSE_COLUMNS,
    common_isolate_cohort,
    hk_matrix,
    isolate_overlap,
    segment_numbers,
    sequence_reuse,
    sort_by_segment,
)

SHORT = {'Hemagglutinin precursor': 'HA', 'Neuraminidase protein': 'NA',
         'Matrix protein 1': 'M1'}
HA, NA, M1 = 'Hemagglutinin precursor', 'Neuraminidase protein', 'Matrix protein 1'

CANONICAL = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']


SEGMENT_OF = {HA: 'S4', NA: 'S6', M1: 'S7'}


def _cds(pairs):
    return pd.DataFrame([{'assembly_id': iso, 'function': fn,
                          'canonical_segment': SEGMENT_OF[fn]} for iso, fn in pairs])


def test_cohort_keeps_only_isolates_carrying_every_protein():
    # i1 has all three; i2 is missing M1; i3 has only HA.
    cds = _cds([('i1', HA), ('i1', NA), ('i1', M1),
                ('i2', HA), ('i2', NA),
                ('i3', HA)])
    assert common_isolate_cohort(cds, ['HA', 'NA', 'M1'], SHORT) == {'i1'}
    assert common_isolate_cohort(cds, ['HA', 'NA'], SHORT) == {'i1', 'i2'}
    assert common_isolate_cohort(cds, ['HA'], SHORT) == {'i1', 'i2', 'i3'}


def test_cohort_rejects_a_protein_with_no_rows():
    # Silently returning an empty cohort would look like a data problem rather than a typo.
    cds = _cds([('i1', HA), ('i1', NA)])
    with pytest.raises(ValueError, match=r"no rows for \['M1'\]"):
        common_isolate_cohort(cds, ['HA', 'NA', 'M1'], SHORT)


def test_isolate_overlap_counts_and_orders():
    retained = {
        ('HA', 'NA'): {'i1', 'i2', 'i3', 'i4'},
        ('HA', 'M1'): {'i3', 'i4', 'i5'},       # shares 2 with HA-NA, union 5 -> 0.4
        ('NA', 'M1'): {'i9'},                   # shares nothing with either
    }
    table = isolate_overlap(retained)
    assert list(table.columns) == OVERLAP_COLUMNS
    assert len(table) == 3                                  # every unordered combination

    top = table.iloc[0]
    assert {top['pair A'], top['pair B']} == {'HA-NA', 'HA-M1'}
    assert top['shared'] == 2
    assert top['isolate jaccard'] == pytest.approx(0.4)
    assert top['isolates A'] == 3 and top['isolates B'] == 4  # sorted key order: HA-M1 first

    # Ordered by descending Jaccard, so the two disjoint rows come last.
    assert list(table['isolate jaccard'])[1:] == [0.0, 0.0]


def test_isolate_overlap_endpoints():
    same = {('HA', 'NA'): {'i1', 'i2'}, ('M1', 'NS1'): {'i1', 'i2'}}
    assert isolate_overlap(same).iloc[0]['isolate jaccard'] == pytest.approx(1.0)

    apart = {('HA', 'NA'): {'i1'}, ('M1', 'NS1'): {'i2'}}
    assert isolate_overlap(apart).iloc[0]['isolate jaccard'] == pytest.approx(0.0)


def test_isolate_overlap_marks_shared_protein():
    # Two schema pairs sharing a protein draw on the same sequences, so the flag separates them.
    retained = {('HA', 'NA'): {'i1'}, ('HA', 'M1'): {'i1'}, ('M1', 'NS1'): {'i1'}}
    table = isolate_overlap(retained).set_index(['pair A', 'pair B'])
    assert table.loc[('HA-M1', 'HA-NA'), 'shares protein']      # both carry HA
    assert table.loc[('HA-M1', 'M1-NS1'), 'shares protein']     # both carry M1
    assert not table.loc[('HA-NA', 'M1-NS1'), 'shares protein']


def _positives(hashes_a, hashes_b):
    return pd.DataFrame({'cds_dna_hash_a': hashes_a, 'cds_dna_hash_b': hashes_b})


def test_sequence_reuse_counts_distinct_partners():
    # x pairs with three distinct partners; y and z pair with one each.
    positives = _positives(['x', 'x', 'x', 'y', 'z'], ['p', 'q', 'r', 'p', 'q'])
    row = sequence_reuse(positives, 'cds_dna_hash_a', 'HA-NA', 'A', 'HA')
    assert list(row) == REUSE_COLUMNS
    assert row['pair'] == 'HA-NA' and row['slot'] == 'A' and row['protein'] == 'HA'
    assert row['positives'] == 5
    assert row['unique sequences'] == 3
    assert row['reuse mean'] == pytest.approx(5 / 3)
    assert row['reuse median'] == 1
    assert row['reuse p90'] == pytest.approx(2.6)   # linear interpolation over [1, 1, 3]
    assert row['reuse max'] == 3
    assert row['singleton share'] == pytest.approx(2 / 3)

    # The other slot is counted independently: p and q twice each, r once.
    other = sequence_reuse(positives, 'cds_dna_hash_b', 'HA-NA', 'B', 'NA')
    assert other['unique sequences'] == 3
    assert other['reuse max'] == 2
    assert other['singleton share'] == pytest.approx(1 / 3)


def test_sequence_reuse_rejects_an_empty_pair():
    # An empty frame would otherwise give a NaN maximum and fail inside int().
    with pytest.raises(ValueError, match='has no positives'):
        sequence_reuse(_positives([], []), 'cds_dna_hash_a', 'HA-NA', 'A', 'HA')


def test_sort_by_segment_orders_on_both_numbers():
    capacity = pd.DataFrame({'ID': [1, 2, 3, 4],
                             'Pair ID': ['7-8', '1-4', '4-6', '1-2'],
                             'HK matched': [440, 2042, 1698, 1404]})
    assert list(sort_by_segment(capacity)['Pair ID']) == ['1-2', '1-4', '4-6', '7-8']
    # `ID` keeps the matched-count rank rather than being renumbered.
    assert list(sort_by_segment(capacity)['ID']) == [4, 2, 3, 1]

    # Numeric, not lexicographic: '10-1' would sort before '2-1' as text.
    wide = pd.DataFrame({'Pair ID': ['10-1', '2-1']})
    assert list(sort_by_segment(wide)['Pair ID']) == ['2-1', '10-1']


def test_hk_matrix_is_symmetric_with_an_empty_diagonal():
    capacity = pd.DataFrame({'pair': ['HA-NA', 'HA-M1', 'NA-M1'],
                             'HK matched': [1698, 720, 657]})
    matrix = hk_matrix(capacity, ['NA', 'M1', 'HA'], CANONICAL)

    # Rows and columns come out in canonical order whatever order `proteins` was given in.
    assert list(matrix.index) == ['HA', 'NA', 'M1']
    assert list(matrix.columns) == ['HA', 'NA', 'M1']
    assert matrix.loc['HA', 'NA'] == 1698 and matrix.loc['NA', 'HA'] == 1698
    assert matrix.loc['NA', 'M1'] == 657 and matrix.loc['M1', 'NA'] == 657
    assert all(pd.isna(matrix.loc[protein, protein]) for protein in matrix.index)


def test_hk_matrix_rejects_a_pair_it_has_no_row_for():
    # A silently empty cell would read as a pair with no capacity rather than a missing row.
    capacity = pd.DataFrame({'pair': ['HA-NA'], 'HK matched': [1698]})
    with pytest.raises(KeyError):
        hk_matrix(capacity, ['HA', 'NA', 'M1'], CANONICAL)


def test_segment_numbers():
    cds = _cds([('i1', HA), ('i1', NA), ('i1', M1)])
    assert segment_numbers(cds, SHORT) == {'HA': 4, 'NA': 6, 'M1': 7}

    # A protein under two segment labels would silently give one pair two different Pair IDs.
    mixed = _cds([('i1', HA), ('i2', HA)])
    mixed.loc[1, 'canonical_segment'] = 'S5'
    with pytest.raises(ValueError, match='spans several segments'):
        segment_numbers(mixed, SHORT)


def test_column_lists():
    # `ID` is a rank over the sorted table; `Pair ID` is the segment pair, e.g. 1-4 for PB2-HA.
    assert CAPACITY_COLUMNS[:3] == ['ID', 'Pair ID', 'pair']
    for name in ('eligible isolates', 'positives', 'HK matched', 'HK share'):
        assert name in CAPACITY_COLUMNS
    for name in ('pair A', 'pair B', 'shares protein', 'shared', 'isolate jaccard'):
        assert name in OVERLAP_COLUMNS
    for name in ('pair', 'slot', 'protein', 'unique sequences', 'reuse mean', 'reuse max'):
        assert name in REUSE_COLUMNS


if __name__ == '__main__':
    tests = [
        test_cohort_keeps_only_isolates_carrying_every_protein,
        test_cohort_rejects_a_protein_with_no_rows,
        test_isolate_overlap_counts_and_orders,
        test_isolate_overlap_endpoints,
        test_isolate_overlap_marks_shared_protein,
        test_sequence_reuse_counts_distinct_partners,
        test_sequence_reuse_rejects_an_empty_pair,
        test_sort_by_segment_orders_on_both_numbers,
        test_hk_matrix_is_symmetric_with_an_empty_diagonal,
        test_hk_matrix_rejects_a_pair_it_has_no_row_for,
        test_segment_numbers,
        test_column_lists,
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
