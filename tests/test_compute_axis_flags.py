"""Regression test for Bug 1: compute_axis_flags must look up metadata by
assembly_id, not seq_hash.

Synthetic case: one protein sequence (one seq_hash) appears in two isolates
with different host metadata -- e.g., the same conserved HA captured from a
Pig and from a Human. With the buggy seq_hash-based lookup, side B's host
collapses to the "first" isolate's value, so any pair using the second
isolate gets the wrong host. With the assembly_id-based lookup, each side
returns its own isolate's value.

Run: python tests/test_compute_axis_flags.py
"""
import sys
from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets.dataset_segment_pairs_v2 import compute_axis_flags


def test_assembly_id_lookup_resolves_per_pair():
    df = pd.DataFrame({
        'assembly_id': ['ISO_A', 'ISO_B', 'ISO_C'],
        'seq_hash':    ['shared_seq', 'shared_seq', 'other_seq'],
        'function':    ['HA', 'HA', 'NA'],
        'host':        ['Pig', 'Human', 'Human'],
        'year':        [2018, 2024, 2024],
        'hn_subtype':  ['H1N1', 'H3N2', 'H3N2'],
    })

    pairs = pd.DataFrame({
        'assembly_id_a': ['ISO_A', 'ISO_B'],
        'assembly_id_b': ['ISO_C', 'ISO_C'],
        'seq_hash_a':    ['shared_seq', 'shared_seq'],
        'seq_hash_b':    ['other_seq', 'other_seq'],
        'label':         [0, 0],
    })

    out = compute_axis_flags(pairs, df, axes=['host', 'year', 'hn_subtype'])

    assert out.loc[0, 'host_a'] == 'Pig',   f"row 0 host_a should be Pig, got {out.loc[0,'host_a']!r}"
    assert out.loc[1, 'host_a'] == 'Human', f"row 1 host_a should be Human, got {out.loc[1,'host_a']!r}"
    assert out.loc[0, 'host_b'] == 'Human'
    assert out.loc[1, 'host_b'] == 'Human'

    assert bool(out.loc[0, 'same_host']) is False
    assert bool(out.loc[1, 'same_host']) is True

    assert out.loc[0, 'year_a'] == 2018
    assert out.loc[1, 'year_a'] == 2024
    assert bool(out.loc[0, 'same_year']) is False
    assert bool(out.loc[1, 'same_year']) is True


def test_null_metadata_yields_pd_na_same_axis():
    df = pd.DataFrame({
        'assembly_id': ['ISO_A', 'ISO_B'],
        'seq_hash':    ['s1', 's2'],
        'function':    ['HA', 'NA'],
        'host':        ['Human', None],
    })
    pairs = pd.DataFrame({
        'assembly_id_a': ['ISO_A'],
        'assembly_id_b': ['ISO_B'],
        'seq_hash_a':    ['s1'],
        'seq_hash_b':    ['s2'],
        'label':         [0],
    })
    out = compute_axis_flags(pairs, df, axes=['host'])
    assert out.loc[0, 'host_a'] == 'Human'
    assert pd.isna(out.loc[0, 'host_b'])
    assert pd.isna(out.loc[0, 'same_host']), "same_host with one null side should be pd.NA"


def test_geo_location_clean_alias():
    df = pd.DataFrame({
        'assembly_id': ['ISO_A', 'ISO_B'],
        'seq_hash':    ['s1', 's2'],
        'function':    ['HA', 'NA'],
        'geo_location_clean': ['Australia', 'Australia'],
    })
    pairs = pd.DataFrame({
        'assembly_id_a': ['ISO_A'],
        'assembly_id_b': ['ISO_B'],
        'seq_hash_a':    ['s1'],
        'seq_hash_b':    ['s2'],
        'label':         [0],
    })
    out = compute_axis_flags(pairs, df, axes=['geo_location'])
    assert out.loc[0, 'geo_location_a'] == 'Australia'
    assert out.loc[0, 'geo_location_b'] == 'Australia'
    assert bool(out.loc[0, 'same_geo_location']) is True


def test_empty_pairs_returns_empty():
    df = pd.DataFrame({'assembly_id': ['ISO_A'], 'seq_hash': ['s1'], 'host': ['Human']})
    pairs = pd.DataFrame({
        'assembly_id_a': [], 'assembly_id_b': [], 'seq_hash_a': [], 'seq_hash_b': [], 'label': [],
    })
    out = compute_axis_flags(pairs, df, axes=['host'])
    assert len(out) == 0


if __name__ == '__main__':
    test_assembly_id_lookup_resolves_per_pair()
    test_null_metadata_yields_pd_na_same_axis()
    test_geo_location_clean_alias()
    test_empty_pairs_returns_empty()
    print('Done. All tests passed.')
