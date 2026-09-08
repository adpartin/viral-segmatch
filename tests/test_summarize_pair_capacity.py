"""Tests for `src/analysis/summarize_pair_capacity.py`.

Covers:
  1. common_isolate_cohort keeps only isolates carrying every requested protein
  2. It rejects a protein that appears in no row, rather than returning an empty cohort
  3. isolate_overlap reports shared counts and Jaccard, and orders by descending Jaccard
  4. Disjoint matchings give Jaccard 0, identical ones give 1
  5. segment_numbers maps each protein to its segment, and rejects one spanning two segments
  6. The column lists are what the CSVs promise

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
    common_isolate_cohort,
    isolate_overlap,
    segment_numbers,
)

SHORT = {'Hemagglutinin precursor': 'HA', 'Neuraminidase protein': 'NA',
         'Matrix protein 1': 'M1'}
HA, NA, M1 = 'Hemagglutinin precursor', 'Neuraminidase protein', 'Matrix protein 1'


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
        'HA-NA': {'i1', 'i2', 'i3', 'i4'},
        'HA-M1': {'i3', 'i4', 'i5'},        # shares 2 with HA-NA, union 5 -> 0.4
        'NA-M1': {'i9'},                    # shares nothing with either
    }
    table = isolate_overlap(retained)
    assert list(table.columns) == OVERLAP_COLUMNS
    assert len(table) == 3                                  # every unordered combination

    top = table.iloc[0]
    assert {top['pair A'], top['pair B']} == {'HA-NA', 'HA-M1'}
    assert top['shared'] == 2
    assert top['isolate jaccard'] == pytest.approx(0.4)
    assert top['isolates A'] == 3 and top['isolates B'] == 4  # sorted label order: HA-M1 first

    # Ordered by descending Jaccard, so the two disjoint rows come last.
    assert list(table['isolate jaccard'])[1:] == [0.0, 0.0]


def test_isolate_overlap_endpoints():
    same = {'a': {'i1', 'i2'}, 'b': {'i1', 'i2'}}
    assert isolate_overlap(same).iloc[0]['isolate jaccard'] == pytest.approx(1.0)

    apart = {'a': {'i1'}, 'b': {'i2'}}
    assert isolate_overlap(apart).iloc[0]['isolate jaccard'] == pytest.approx(0.0)


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
    for name in ('positives', 'HK matched', 'HK share', 'cohort isolates'):
        assert name in CAPACITY_COLUMNS
    for name in ('pair A', 'pair B', 'shared', 'isolate jaccard'):
        assert name in OVERLAP_COLUMNS


if __name__ == '__main__':
    tests = [
        test_cohort_keeps_only_isolates_carrying_every_protein,
        test_cohort_rejects_a_protein_with_no_rows,
        test_isolate_overlap_counts_and_orders,
        test_isolate_overlap_endpoints,
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
