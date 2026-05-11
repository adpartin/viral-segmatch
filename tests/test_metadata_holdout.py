"""Tests for metadata_holdout — generalized cross-population split.

Covers:
- filter_by_metadata: scalar / list / year_range value forms + failure modes.
- compute_metadata_holdout_isolates: happy path (carve from train + explicit
  val), determinism, multi-slot overlap, empty pool, schema typos.
- Dropped-isolates manifest schema + excluded_reason content.

Style follows tests/test_level1_neg_regimes.py: hand-built synthetic
DataFrames, one test function per behavior; parametrize only where the test
body is structurally identical (filter_by_metadata's happy-path table).

Plan: docs/plans/2026-05-11_metadata_holdout_plan.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make `src.datasets._pair_helpers` importable without installing the package.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.datasets._pair_helpers import (
    compute_metadata_holdout_isolates,
    drop_ambiguous_hn_subtype,
    filter_by_metadata,
)


# A small per-isolate world that exercises every axis. 12 isolates,
# every combination chosen so each test below has an unambiguous expected set.
#   A B C  -> Human / H3N2 / [2017,2018,2019,2020]
#   D E F  -> Pig   / H1N1 / [2022,2023,2024]
#   G      -> Human / H5N1 / 2024     (covers "third subtype")
#   H      -> Pig   / H3N2 / 2010     (covers "wrong host for both slots")
#   I      -> Human / H1N1 / 2015     (covers "wrong subtype for either slot")
#   J      -> Pig   / H3N2 / 2024     (drops on H1N1 filter even with right host+year)
#   K      -> Pig   / H1N1 / 2024     (test-slot match)
#   L      -> Human / H3N2 / 2017     (train-slot match)
def _synthetic_df() -> pd.DataFrame:
    rows = [
        ('A', 'H3N2', 'Human', 2018, 'US', 'Orig'),
        ('B', 'H3N2', 'Human', 2019, 'UK', 'S1'),
        ('C', 'H3N2', 'Human', 2020, 'US', 'S2'),
        ('D', 'H1N1', 'Pig',   2022, 'US', 'Orig'),
        ('E', 'H1N1', 'Pig',   2023, 'UK', 'S1'),
        ('F', 'H1N1', 'Pig',   2024, 'CN', 'S2'),
        ('G', 'H5N1', 'Human', 2024, 'CN', 'Orig'),
        ('H', 'H3N2', 'Pig',   2010, 'US', 'Orig'),
        ('I', 'H1N1', 'Human', 2015, 'UK', 'S1'),
        ('J', 'H3N2', 'Pig',   2024, 'CN', 'Orig'),
        ('K', 'H1N1', 'Pig',   2024, 'US', 'S2'),
        ('L', 'H3N2', 'Human', 2017, 'UK', 'Orig'),
    ]
    # Two protein rows per isolate to exercise the per-isolate groupby; the
    # second row's metadata MUST agree with the first (compute_axis_flags
    # uses .first()) -- we don't test inconsistent-metadata behaviour here.
    df = pd.DataFrame([
        {'assembly_id': aid, 'hn_subtype': hn, 'host': h, 'year': y,
         'geo_location_clean': geo, 'passage': pa, 'function': f'prot{i}',
         'file': f'{aid}.gto'}
        for (aid, hn, h, y, geo, pa) in rows
        for i in (1, 2)
    ])
    return df


# -- filter_by_metadata --------------------------------------------------------

@pytest.mark.parametrize(
    'kwargs, expected_assembly_ids',
    [
        # scalar host (backwards compat)
        ({'host': 'Human'}, {'A', 'B', 'C', 'G', 'I', 'L'}),
        # 1-element list -- set membership of {Human}
        ({'host': ['Human']}, {'A', 'B', 'C', 'G', 'I', 'L'}),
        # multi-element list
        ({'hn_subtype': ['H1N1', 'H5N1']}, {'D', 'E', 'F', 'G', 'I', 'K'}),
        # scalar year
        ({'year': 2020}, {'C'}),
        # year list (set membership, not range; deliberate)
        ({'year': [2020, 2024]}, {'C', 'F', 'G', 'J', 'K'}),
        # year_range inclusive
        ({'year_range': [2020, 2022]}, {'C', 'D'}),
        # combined: user's prompt example for train
        ({'host': 'Human', 'hn_subtype': 'H3N2', 'year_range': [1990, 2020]},
         {'A', 'B', 'C', 'L'}),
    ],
)
def test_filter_by_metadata_accepts_value_forms(kwargs, expected_assembly_ids):
    df = _synthetic_df()
    out = filter_by_metadata(df, **kwargs)
    assert set(out['assembly_id']) == expected_assembly_ids


def test_filter_by_metadata_rejects_year_and_year_range():
    df = _synthetic_df()
    with pytest.raises(ValueError, match='mutually exclusive'):
        filter_by_metadata(df, year=2020, year_range=[2018, 2022])


def test_filter_by_metadata_rejects_year_range_min_gt_max():
    df = _synthetic_df()
    with pytest.raises(ValueError, match='greater than'):
        filter_by_metadata(df, year_range=[2024, 2020])


def test_filter_by_metadata_rejects_year_range_wrong_length():
    df = _synthetic_df()
    with pytest.raises(ValueError, match='2-element'):
        filter_by_metadata(df, year_range=[2020])


def test_filter_by_metadata_rejects_year_range_non_int():
    df = _synthetic_df()
    with pytest.raises(ValueError, match='must be ints'):
        filter_by_metadata(df, year_range=[2020, 'x'])


def test_filter_by_metadata_rejects_empty_list():
    df = _synthetic_df()
    with pytest.raises(ValueError, match='cannot be empty'):
        filter_by_metadata(df, host=[])


# -- compute_metadata_holdout_isolates: happy path -----------------------------

def _user_example_cfg() -> dict:
    """The prompt's example: Human/H3N2/<=2020 train vs Pig/H1N1/>=2022 test."""
    return {
        'train': {'host': 'Human', 'hn_subtype': 'H3N2', 'year_range': [1990, 2020]},
        'test':  {'host': 'Pig',   'hn_subtype': 'H1N1', 'year_range': [2022, 9999]},
        'val':   None,
    }


def test_compute_metadata_holdout_isolates_basic():
    """Happy path: train pool 4 isolates (A,B,C,L), test pool 4 (D,E,F,K),
    dropped 4 (G,H,I,J). val carves 1 from train at val_ratio=0.25."""
    df = _synthetic_df()
    train_ids, val_ids, test_ids, dropped_df = compute_metadata_holdout_isolates(
        df, _user_example_cfg(), seed=42, val_ratio=0.25,
    )
    # train + val together cover the train-filter match set.
    assert set(train_ids) | set(val_ids) == {'A', 'B', 'C', 'L'}
    assert set(train_ids).isdisjoint(val_ids)
    assert len(val_ids) == 1                       # round(4 * 0.25) = 1
    assert len(train_ids) == 3
    # Test filter is independent of val carve.
    assert set(test_ids) == {'D', 'E', 'F', 'K'}
    # Dropped: G/H/I/J match neither train nor test filters.
    assert set(dropped_df['assembly_id']) == {'G', 'H', 'I', 'J'}


def test_metadata_holdout_carves_val_from_train_deterministically():
    """Same seed must produce the same (train, val) partition; different seed
    may produce a different val choice (we don't assert different, only that
    the API exposes the knob)."""
    df = _synthetic_df()
    cfg = _user_example_cfg()
    t1, v1, _, _ = compute_metadata_holdout_isolates(df, cfg, seed=42, val_ratio=0.25)
    t2, v2, _, _ = compute_metadata_holdout_isolates(df, cfg, seed=42, val_ratio=0.25)
    assert (t1, v1) == (t2, v2)


def test_metadata_holdout_uses_explicit_val_filter():
    """When val is an explicit filter dict, val comes from that filter, not
    from a carve off train."""
    df = _synthetic_df()
    cfg = {
        'train': {'host': 'Human', 'hn_subtype': 'H3N2', 'year_range': [1990, 2018]},
        'val':   {'host': 'Human', 'hn_subtype': 'H3N2', 'year': 2019},
        'test':  {'host': 'Pig',   'hn_subtype': 'H1N1', 'year_range': [2022, 9999]},
    }
    train_ids, val_ids, test_ids, _ = compute_metadata_holdout_isolates(
        df, cfg, seed=42, val_ratio=0.10,
    )
    # Note val_ratio is ignored on the explicit-val path (the entire B
    # isolate matches the val filter; no carve happens).
    assert set(train_ids) == {'A', 'L'}            # H3N2 Human in [1990, 2018]
    assert set(val_ids) == {'B'}                   # H3N2 Human year=2019
    assert set(test_ids) == {'D', 'E', 'F', 'K'}


# -- compute_metadata_holdout_isolates: failure modes --------------------------

def test_raises_when_train_pool_empty():
    df = _synthetic_df()
    cfg = {'train': {'host': 'Martian'}, 'test': {'host': 'Pig'}}
    with pytest.raises(ValueError, match='train pool is empty'):
        compute_metadata_holdout_isolates(df, cfg, seed=42, val_ratio=0.1)


def test_raises_when_test_pool_empty():
    df = _synthetic_df()
    cfg = {'train': {'host': 'Human'}, 'test': {'host': 'Alien'}}
    with pytest.raises(ValueError, match='test pool is empty'):
        compute_metadata_holdout_isolates(df, cfg, seed=42, val_ratio=0.1)


def test_raises_when_val_carve_rounds_to_zero():
    """A 1-isolate train pool with val_ratio=0.1 carves 0 -> hard error
    instructing the user to lower val_ratio or loosen the train filter."""
    df = _synthetic_df()
    cfg = {
        'train': {'host': 'Human', 'year': 2018},   # only isolate A matches
        'test':  {'host': 'Pig'},
    }
    with pytest.raises(ValueError, match='would carve 0'):
        compute_metadata_holdout_isolates(df, cfg, seed=42, val_ratio=0.1)


def test_raises_when_train_test_pools_overlap():
    """An isolate matching both train and test filters is leakage. The error
    must name the offending assembly_id(s)."""
    df = _synthetic_df()
    cfg = {
        'train': {'host': 'Human'},                # A,B,C,G,I,L
        'test':  {'hn_subtype': 'H3N2'},           # A,B,C,H,J,L
        'val':   None,                             # overlap: A,B,C,L
    }
    with pytest.raises(ValueError, match='overlapping'):
        compute_metadata_holdout_isolates(df, cfg, seed=42, val_ratio=0.1)


def test_raises_when_train_required_slot_missing():
    df = _synthetic_df()
    with pytest.raises(ValueError, match='train is required'):
        compute_metadata_holdout_isolates(
            df, {'test': {'host': 'Pig'}}, seed=42, val_ratio=0.1,
        )


def test_raises_when_test_required_slot_missing():
    df = _synthetic_df()
    with pytest.raises(ValueError, match='test is required'):
        compute_metadata_holdout_isolates(
            df, {'train': {'host': 'Human'}}, seed=42, val_ratio=0.1,
        )


def test_raises_when_unknown_axis_key():
    """Common typo: 'subtype' instead of 'hn_subtype'. Error names the supported
    axes so the user can fix without consulting the plan."""
    df = _synthetic_df()
    cfg = {
        'train': {'subtype': 'H3N2'},
        'test':  {'host': 'Pig'},
    }
    with pytest.raises(ValueError, match='unknown axis key'):
        compute_metadata_holdout_isolates(df, cfg, seed=42, val_ratio=0.1)


def test_raises_when_range_used_on_unordered_axis():
    """host_range / hn_subtype_range etc. are not supported. The error must
    tell the user to use a list for set membership instead."""
    df = _synthetic_df()
    cfg = {
        'train': {'host_range': ['Human', 'Pig']},
        'test':  {'host': 'Pig'},
    }
    with pytest.raises(ValueError, match='not an ordered axis'):
        compute_metadata_holdout_isolates(df, cfg, seed=42, val_ratio=0.1)


def test_raises_when_year_and_year_range_both_set_in_slot():
    df = _synthetic_df()
    cfg = {
        'train': {'host': 'Human', 'year': 2020, 'year_range': [1990, 2020]},
        'test':  {'host': 'Pig'},
    }
    with pytest.raises(ValueError, match='mutually exclusive'):
        compute_metadata_holdout_isolates(df, cfg, seed=42, val_ratio=0.1)


def test_raises_when_year_range_min_gt_max_in_slot():
    df = _synthetic_df()
    cfg = {
        'train': {'year_range': [2024, 2020]},
        'test':  {'host': 'Pig'},
    }
    with pytest.raises(ValueError, match='greater than'):
        compute_metadata_holdout_isolates(df, cfg, seed=42, val_ratio=0.1)


# -- dropped-isolates manifest -------------------------------------------------

def test_dropped_isolates_manifest_columns():
    """Dropped manifest must carry the identifier + metadata columns the user
    needs for forensic debugging (which isolates fell out, and why)."""
    df = _synthetic_df()
    _, _, _, dropped_df = compute_metadata_holdout_isolates(
        df, _user_example_cfg(), seed=42, val_ratio=0.25,
    )
    required = {
        'assembly_id', 'hn_subtype', 'host', 'year', 'geo_location_clean',
        'passage', 'matches_train', 'matches_val', 'matches_test',
        'excluded_reason', 'file',
    }
    missing = required - set(dropped_df.columns)
    assert not missing, f'manifest missing columns: {missing}'


# -- drop_ambiguous_hn_subtype --------------------------------------------------

def _df_with_subtypes(subtype_per_isolate: dict[str, str], n_rows_per_iso: int = 2
                     ) -> pd.DataFrame:
    """Make a synthetic prot_df where each isolate has n_rows_per_iso rows all
    carrying the same hn_subtype value (mirroring Stage-1 enrichment)."""
    rows = []
    for aid, sub in subtype_per_isolate.items():
        for i in range(n_rows_per_iso):
            rows.append({'assembly_id': aid, 'hn_subtype': sub,
                         'function': f'prot{i}'})
    return pd.DataFrame(rows)


def test_drop_ambig_subtype_typical_mix():
    """Mix of fully-specified, 'HN', 'H1N', and null. Only fully-specified survives."""
    df = _df_with_subtypes({
        'A': 'H3N2', 'B': 'H1N1', 'C': 'H5N1',  # keep
        'D': 'HN', 'E': 'H1N',                    # drop (ambiguous)
        'F': None,                                 # drop (null)
    })
    out, summary = drop_ambiguous_hn_subtype(df)
    assert set(out['assembly_id']) == {'A', 'B', 'C'}
    assert summary['n_isolates_total'] == 6
    assert summary['n_isolates_dropped'] == 3
    assert summary['n_isolates_kept'] == 3
    assert summary['n_rows_total'] == 12
    assert summary['n_rows_dropped'] == 6     # 3 dropped isolates x 2 rows each
    assert summary['n_rows_kept'] == 6
    # value_counts captures the per-value tally. Null gets coerced to '(null)'
    # for JSON serializability.
    assert summary['value_counts'] == {'HN': 1, 'H1N': 1, '(null)': 1}


def test_drop_ambig_subtype_no_op_when_all_clean():
    """All isolates fully-specified -> df returned unchanged, summary reports 0 drops."""
    df = _df_with_subtypes({'A': 'H3N2', 'B': 'H1N1'})
    out, summary = drop_ambiguous_hn_subtype(df)
    assert len(out) == len(df)
    assert summary['n_isolates_dropped'] == 0
    assert summary['n_rows_dropped'] == 0
    assert summary['value_counts'] == {}


def test_drop_ambig_subtype_missing_column_is_noop():
    """If hn_subtype column is absent (e.g. helper called before metadata
    enrichment), the helper returns the df unchanged rather than crashing."""
    df = pd.DataFrame({'assembly_id': ['A', 'B'], 'function': ['p1', 'p2']})
    out, summary = drop_ambiguous_hn_subtype(df)
    assert len(out) == len(df)
    assert summary['n_isolates_dropped'] == 0
    assert summary['value_counts'] == {}


def test_drop_ambig_subtype_regex_accepts_multi_digit_indices():
    """H11N9 / H10N7 / H16N3 are valid Flu A subtypes; multi-digit H/N indices
    must be kept (the regex is r'H\\d+N\\d+', not r'H\\dN\\d'). Conversely
    'H' alone or 'N' alone or 'h3n2' (lowercase) must be dropped."""
    df = _df_with_subtypes({
        'A': 'H11N9', 'B': 'H10N7', 'C': 'H16N3',   # keep (multi-digit)
        'D': 'h3n2',                                  # drop (lowercase)
        'E': 'H', 'F': 'N3', 'G': 'H1NX',            # drop (malformed)
    })
    out, _ = drop_ambiguous_hn_subtype(df)
    assert set(out['assembly_id']) == {'A', 'B', 'C'}


def test_dropped_isolates_manifest_excluded_reason_names_filters():
    """excluded_reason text must mention the per-slot filter descriptions so a
    reader can see which filter(s) the isolate failed to satisfy."""
    df = _synthetic_df()
    _, _, _, dropped_df = compute_metadata_holdout_isolates(
        df, _user_example_cfg(), seed=42, val_ratio=0.25,
    )
    assert len(dropped_df) > 0
    sample_reason = dropped_df['excluded_reason'].iloc[0]
    # Each slot's filter spec should be reflected in the reason text.
    for token in ('train=', 'val=', 'test=', 'matched no slot'):
        assert token in sample_reason, (token, sample_reason)
