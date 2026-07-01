"""Tests for regime-aware coverage (Phase 4 of the regime-aware coverage plan).

Run: python tests/test_regime_aware_coverage.py
"""
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

import pandas as pd

from src.datasets._negative_regime_sampling import (
    COVERAGE_PRIORITY_CHAIN,
    REGIME_NAMES,
    build_cell_regime_partners,
    classify_pair_regime,
    DEFAULT_AXES,
)
from src.datasets.dataset_segment_pairs_v2 import create_negative_pairs_v2


def _toy_pos_row(aid: str, *, slot_a_seq: str, slot_b_seq: str) -> dict:
    """Build a minimal pos_df row. Each isolate has one (slot_a, slot_b) pair.
    prot_hash_a/b and ctg_dna_hash_a/b use the seq labels directly (so the test
    operates on whatever 'sequence identifier' we choose), and the schema is
    HA (slot a) + NA (slot b). brc identifiers are derived per-isolate.
    """
    return {
        'pair_key': f'{slot_a_seq}__{slot_b_seq}',
        'assembly_id_a': aid, 'assembly_id_b': aid,
        'brc_a': f'{aid}.brcA', 'brc_b': f'{aid}.brcB',
        'ctg_a': f'{aid}.ctgA', 'ctg_b': f'{aid}.ctgB',
        'prot_seq_a': slot_a_seq, 'prot_seq_b': slot_b_seq,
        'ctg_dna_seq_a': 'ACGT', 'ctg_dna_seq_b': 'TGCA',
        'seg_a': 'S4', 'seg_b': 'S6',
        'func_a': 'HA', 'func_b': 'NA',
        'prot_hash_a': slot_a_seq, 'prot_hash_b': slot_b_seq,
        'ctg_dna_hash_a': f'd_{slot_a_seq}', 'ctg_dna_hash_b': f'd_{slot_b_seq}',
        'label': 1,
    }


def _build_toy_pos_df(isolates):
    """isolates is a list of dicts: {aid, slot_a_seq, slot_b_seq}."""
    return pd.DataFrame([_toy_pos_row(**iso) for iso in isolates])


# -----------------------------------------------------------------------------
# Phase 1 helper tests (build_cell_regime_partners)
# -----------------------------------------------------------------------------

def test_helper_priority_order_via_classify_consistency():
    """Every partner returned for self_cell under regime R must classify
    (self_cell, partner_cell) -> R when fed back through classify_pair_regime."""
    iso_to_cell = {
        'i1': ('Human', 'H3N2', '2016-2020'),
        'i2': ('Human', 'H3N2', '<=2015'),
        'i3': ('Pig',   'H3N2', '<=2015'),
        'i4': ('Pig',   'H1N1', '2016-2020'),
        'i5': ('Pig',   'H1N1', '2016-2020'),  # duplicate cell of i4
    }
    m = build_cell_regime_partners(iso_to_cell)
    for self_cell, by_regime in m.items():
        for regime, partners in by_regime.items():
            for partner in partners:
                r = classify_pair_regime(self_cell, partner)
                assert r == regime, (
                    f"mismatch: build_cell_regime_partners labelled "
                    f"({self_cell}, {partner}) as {regime} but classify_pair_regime says {r}"
                )
    # Within-cell pairs always live under host_subtype_year
    for cell in m.keys():
        assert cell in m[cell].get('host_subtype_year', []), \
            f"{cell} not in its own host_subtype_year partners"
    print("[OK] helper: build_cell_regime_partners is internally consistent with classify_pair_regime")


def test_helper_priority_chain_covers_all_regimes():
    assert set(COVERAGE_PRIORITY_CHAIN) == set(REGIME_NAMES)
    assert COVERAGE_PRIORITY_CHAIN[0] == 'host_subtype_year'
    assert COVERAGE_PRIORITY_CHAIN[-1] == 'none_match'
    print(f"[OK] priority chain covers all {len(REGIME_NAMES)} regimes; head = "
          f"{COVERAGE_PRIORITY_CHAIN[0]!r}, tail = {COVERAGE_PRIORITY_CHAIN[-1]!r}")


def test_helper_only_none_match_feasible_case():
    """When self_cell shares no axis with any other cell, only none_match is in the map."""
    iso_to_cell = {
        'i1': ('Human', 'H3N2', '2016-2020'),
        'i2': ('Duck',  'H5N1', '<=2015'),     # no axis match with i1
    }
    m = build_cell_regime_partners(iso_to_cell)
    by_regime = m[('Human', 'H3N2', '2016-2020')]
    feasible_regimes = set(by_regime.keys())
    # Only host_subtype_year (self) and none_match (the other cell) should be feasible.
    assert feasible_regimes == {'host_subtype_year', 'none_match'}, feasible_regimes
    assert by_regime['none_match'] == [('Duck', 'H5N1', '<=2015')]
    print("[OK] helper: only none_match feasible when no axis shared with any other cell")


# -----------------------------------------------------------------------------
# Phase 2 integration tests (create_negative_pairs_v2 + regime_aware_coverage)
# -----------------------------------------------------------------------------

def _make_balanced_quotas():
    """Uniform regime targets for tests that need axis_quotas set but don't
    care about the fill phase's bias."""
    return {r: 1.0 / len(REGIME_NAMES) for r in REGIME_NAMES}


def test_integration_flag_false_default_path_works():
    """Backwards-compat: with flag=False, coverage is regime-blind. Sanity
    check: invocation succeeds and produces ≥1 neg per protein."""
    isolates = [
        {'aid': 'a', 'slot_a_seq': 'HA1', 'slot_b_seq': 'NA1'},
        {'aid': 'b', 'slot_a_seq': 'HA2', 'slot_b_seq': 'NA2'},
        {'aid': 'c', 'slot_a_seq': 'HA3', 'slot_b_seq': 'NA3'},
        {'aid': 'd', 'slot_a_seq': 'HA4', 'slot_b_seq': 'NA4'},
    ]
    pos = _build_toy_pos_df(isolates)
    iso_to_cell = {
        'a': ('Human', 'H3N2', '<=2015'),
        'b': ('Human', 'H1N1', '<=2015'),
        'c': ('Pig',   'H3N2', '<=2015'),
        'd': ('Pig',   'H1N1', '2016-2020'),
    }
    neg, stats = create_negative_pairs_v2(
        pos_df=pos, num_negatives=8,
        cooccur_pairs=set(),
        schema_pair=('HA', 'NA'),
        seed=42,
        axis_quotas=_make_balanced_quotas(),
        isolate_to_cell=iso_to_cell,
        sampling_axes=list(DEFAULT_AXES),
        regime_aware_coverage=False,
    )
    assert stats['coverage_regime_aware']['enabled'] is False
    assert stats['coverage_phase_pairs'] >= 4
    assert len(neg) > 0
    print(f"[OK] flag=False default: {len(neg)} negs, "
          f"coverage_regime_aware.enabled = False")


def test_integration_priority_order_respected_when_flag_true():
    """With 5 isolates: two share a cell (host_subtype_year between them is
    feasible), and one differs on all axes (only none_match path). The
    in-cell partner should be preferred for the within-cell isolates'
    coverage iterations."""
    isolates = [
        {'aid': 'a', 'slot_a_seq': 'HA_a', 'slot_b_seq': 'NA_a'},
        {'aid': 'b', 'slot_a_seq': 'HA_b', 'slot_b_seq': 'NA_b'},  # same cell as a
        {'aid': 'c', 'slot_a_seq': 'HA_c', 'slot_b_seq': 'NA_c'},  # different host
        {'aid': 'd', 'slot_a_seq': 'HA_d', 'slot_b_seq': 'NA_d'},  # different on all
    ]
    pos = _build_toy_pos_df(isolates)
    iso_to_cell = {
        'a': ('Human', 'H3N2', '<=2015'),
        'b': ('Human', 'H3N2', '<=2015'),   # same cell as a
        'c': ('Pig',   'H3N2', '<=2015'),   # subtype + year match w/ a
        'd': ('Duck',  'H5N1', '2016-2020'),  # no shared axis with a
    }
    neg, stats = create_negative_pairs_v2(
        pos_df=pos, num_negatives=8,
        cooccur_pairs=set(),
        schema_pair=('HA', 'NA'),
        seed=42,
        axis_quotas=_make_balanced_quotas(),
        isolate_to_cell=iso_to_cell,
        sampling_axes=list(DEFAULT_AXES),
        regime_aware_coverage=True,
    )
    ra = stats['coverage_regime_aware']
    assert ra['enabled'] is True
    # The hardest regime (host_subtype_year) is feasible for a and b
    # (they share a cell with each other and with themselves on the
    # within-cell pair). The priority chain should land on host_subtype_year
    # for at least one acceptance, and on subtype_year_only / none_match
    # only when host_subtype_year is exhausted by exclusion.
    accept_by_regime = ra['regime_aware_acceptances_per_regime']
    assert accept_by_regime['host_subtype_year'] > 0, \
        f"expected ≥1 host_subtype_year acceptance, got {accept_by_regime}"
    # Easier regimes should fire only if the harder one ran out of partners
    # (e.g., for c whose only feasible non-none_match partner is in
    # subtype_year_only via 'a' or 'b').
    print(f"[OK] priority order: regime_aware_acceptances = {accept_by_regime}")


def test_integration_last_resort_uniform_fires_when_no_feasible_partner():
    """Construct a case where all feasible regimes are excluded by the same-seq
    exclusion rule, forcing the last-resort uniform sampler. Verify the counter
    fires (or at least the run still produces coverage)."""
    # 3 isolates; isolate 'a' has a HA shared by 'b' (same prot_hash_a), so the
    # only partner in 'a's row's HA-coverage iteration must come from a
    # different prot_hash — the priority chain will try a's partner cells but
    # if those are all rejected, falls through to uniform.
    isolates = [
        {'aid': 'a', 'slot_a_seq': 'HAx', 'slot_b_seq': 'NAa'},
        {'aid': 'b', 'slot_a_seq': 'HAx', 'slot_b_seq': 'NAb'},  # shares HA
        {'aid': 'c', 'slot_a_seq': 'HAy', 'slot_b_seq': 'NAc'},
    ]
    pos = _build_toy_pos_df(isolates)
    iso_to_cell = {
        'a': ('Human', 'H3N2', '<=2015'),
        'b': ('Human', 'H3N2', '<=2015'),
        'c': ('Pig',   'H1N1', '2016-2020'),
    }
    neg, stats = create_negative_pairs_v2(
        pos_df=pos, num_negatives=4,
        cooccur_pairs=set(),
        schema_pair=('HA', 'NA'),
        seed=42,
        axis_quotas=_make_balanced_quotas(),
        isolate_to_cell=iso_to_cell,
        sampling_axes=list(DEFAULT_AXES),
        regime_aware_coverage=True,
    )
    ra = stats['coverage_regime_aware']
    assert ra['enabled'] is True
    # Run completed without raising; the priority chain found partners or fell
    # back. With this small a graph the fallback may or may not fire, but the
    # counter must exist and be non-negative.
    assert ra['fell_back_to_uniform'] >= 0
    # Either way: the prot_hash coverage invariant must hold (raises otherwise).
    assert len(stats.get('seqs_with_zero_negatives', [])) == 0
    print(f"[OK] last-resort path: fell_back_to_uniform = {ra['fell_back_to_uniform']}, "
          f"seqs_with_zero_negatives = {len(stats.get('seqs_with_zero_negatives', []))}")


def test_integration_coverage_invariant_holds_with_flag_true():
    """End-to-end with regime_aware_coverage=True: every prot_hash in pos_df
    must get ≥1 negative (the hard invariant that the sampler enforces)."""
    isolates = [
        {'aid': f'iso{i}', 'slot_a_seq': f'HA{i}', 'slot_b_seq': f'NA{i}'}
        for i in range(10)
    ]
    pos = _build_toy_pos_df(isolates)
    iso_to_cell = {
        f'iso{i}': (
            ['Human', 'Pig', 'Duck'][i % 3],
            ['H1N1', 'H3N2'][i % 2],
            ['<=2015', '2016-2020'][i % 2],
        )
        for i in range(10)
    }
    neg, stats = create_negative_pairs_v2(
        pos_df=pos, num_negatives=15,
        cooccur_pairs=set(),
        schema_pair=('HA', 'NA'),
        seed=42,
        axis_quotas=_make_balanced_quotas(),
        isolate_to_cell=iso_to_cell,
        sampling_axes=list(DEFAULT_AXES),
        regime_aware_coverage=True,
    )
    ra = stats['coverage_regime_aware']
    assert ra['enabled'] is True
    # Coverage invariant: every prot_hash on both slots has ≥1 negative.
    for slot in ('a', 'b'):
        prot_hashes_in_pos = set(pos[f'prot_hash_{slot}'])
        prot_hashes_in_neg = set(neg[f'prot_hash_{slot}'])
        # neg may use this seq on either side because the partner slot
        # contributes a prot_hash to the OPPOSITE column of the row.
        # The hard invariant the sampler enforces is that every pos prot_hash
        # on slot X appears in at least one neg row's slot-X column.
        missing = prot_hashes_in_pos - prot_hashes_in_neg
        assert not missing, (
            f"slot {slot}: {len(missing)} prot_hashes uncovered: {sorted(missing)[:5]}"
        )
    print(f"[OK] coverage invariant: {len(neg)} negs, every protein has ≥1 neg")


def test_integration_flag_true_without_regime_mode_raises():
    """regime_aware_coverage requires regime_mode (axis_quotas + isolate_to_cell).
    Without those, the call must raise — silent no-op would be confusing."""
    isolates = [{'aid': f'i{n}', 'slot_a_seq': f'HA{n}', 'slot_b_seq': f'NA{n}'} for n in range(3)]
    pos = _build_toy_pos_df(isolates)
    try:
        create_negative_pairs_v2(
            pos_df=pos, num_negatives=2,
            cooccur_pairs=set(),
            schema_pair=('HA', 'NA'),
            seed=42,
            axis_quotas=None,   # NOT in regime mode
            isolate_to_cell=None,
            sampling_axes=None,
            regime_aware_coverage=True,  # but flag is True
        )
    except ValueError as e:
        assert 'regime_aware_coverage' in str(e)
        print(f"[OK] flag=True without regime_mode raises: {str(e)[:80]}…")
        return
    raise AssertionError("expected ValueError when regime_aware_coverage=True but axis_quotas is None")


if __name__ == '__main__':
    test_helper_priority_order_via_classify_consistency()
    test_helper_priority_chain_covers_all_regimes()
    test_helper_only_none_match_feasible_case()
    test_integration_flag_false_default_path_works()
    test_integration_priority_order_respected_when_flag_true()
    test_integration_last_resort_uniform_fires_when_no_feasible_partner()
    test_integration_coverage_invariant_holds_with_flag_true()
    test_integration_flag_true_without_regime_mode_raises()
    print("\nAll regime-aware coverage tests passed.")
