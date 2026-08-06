"""Regression test for Bug 2: compute_metadata_coverage per-seq null
undercount.

Synthetic case: a prot_hash with rows [None, "Human"] should NOT be counted
as null in the per-sequence view -- the protein has a known host (Human)
in at least one isolate.

Run: python tests/test_compute_metadata_coverage.py
"""
import sys
from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets.dataset_segment_pairs_v2 import compute_metadata_coverage


def test_per_seq_null_only_when_no_value_seen():
    df = pd.DataFrame({
        'assembly_id': ['ISO_A', 'ISO_B', 'ISO_C', 'ISO_D'],
        'prot_hash':    ['s1',    's1',    's2',    's3'],
        'host':        [None,    'Human', 'Human', None],
    })
    cov = compute_metadata_coverage(df, axes=['host'])

    # n_unique_seqs = 3 (s1, s2, s3)
    assert cov['host']['n_unique_seqs'] == 3, cov['host']
    # Only s3 has zero non-null values across its isolates
    assert cov['host']['n_unique_seqs_null'] == 1, cov['host']
    assert abs(cov['host']['pct_unique_null'] - 33.333) < 0.01, cov['host']
    # Per-row: 2 of 4 rows have null host
    assert cov['host']['n_rows_total'] == 4
    assert cov['host']['n_rows_null'] == 2
    assert abs(cov['host']['pct_null'] - 50.0) < 0.01


def test_pre_fix_undercount_case():
    # Specific shape that triggered the pre-fix bug: groupby first row was
    # null, later row had value. With .first() the seq was counted null;
    # with the dropna-first fix, it should be counted as having a value.
    df = pd.DataFrame({
        'assembly_id': ['ISO_A', 'ISO_B'],
        'prot_hash':    ['shared', 'shared'],
        'host':        [None, 'Human'],
    })
    cov = compute_metadata_coverage(df, axes=['host'])
    assert cov['host']['n_unique_seqs'] == 1
    assert cov['host']['n_unique_seqs_null'] == 0, (
        f'pre-fix bug: pre-dropna .first() returns None and counts the seq as null. '
        f'got {cov["host"]}'
    )


def test_axis_missing():
    df = pd.DataFrame({'assembly_id': ['A'], 'prot_hash': ['s1']})
    cov = compute_metadata_coverage(df, axes=['host'])
    assert cov['host'] == {'present': False}


def test_geo_location_clean_alias():
    df = pd.DataFrame({
        'assembly_id':        ['A'],
        'prot_hash':           ['s1'],
        'geo_location_clean': ['Wisconsin'],
    })
    cov = compute_metadata_coverage(df, axes=['geo_location'])
    assert cov['geo_location']['present'] is True
    assert cov['geo_location']['n_distinct_values'] == 1


if __name__ == '__main__':
    test_per_seq_null_only_when_no_value_seen()
    test_pre_fix_undercount_case()
    test_axis_missing()
    test_geo_location_clean_alias()
    print('Done. All tests passed.')
