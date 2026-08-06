"""Unit tests for src/datasets/_negative_regime_sampling.py.

Run: python tests/test_negative_regime_sampling.py
"""
import sys
from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.datasets._negative_regime_sampling import (
    REGIME_NAMES,
    DEFAULT_AXES,
    DEFAULT_YEAR_BIN_EDGES,
    bin_year,
    build_isolate_cells,
    classify_pair_regime,
    compute_match_count,
    count_isolates_per_cell,
    count_available_per_regime,
    resolve_regime_targets,
)


def test_bin_year_default_edges():
    assert bin_year(2010) == '<=2015'
    assert bin_year(2015) == '<=2015'
    assert bin_year(2016) == '2016-2020'
    assert bin_year(2020) == '2016-2020'
    assert bin_year(2021) == '2021+'
    assert bin_year(2024) == '2021+'
    assert bin_year(None) is None
    assert bin_year(float('nan')) is None


def test_classify_pair_all_eight_regimes():
    # Each axis: same vs different across the pair to cover all 8 patterns.
    H1, H2 = 'Pig', 'Human'
    S1, S2 = 'H1N1', 'H3N2'
    Y1, Y2 = 2018, 2024
    cases = [
        ((H1, S1, Y1), (H2, S2, Y2), 'none_match'),
        ((H1, S1, Y1), (H1, S2, Y2), 'host_only'),
        ((H1, S1, Y1), (H2, S1, Y2), 'subtype_only'),
        ((H1, S1, Y1), (H2, S2, Y1), 'year_only'),
        ((H1, S1, Y1), (H1, S1, Y2), 'host_subtype_only'),
        ((H1, S1, Y1), (H1, S2, Y1), 'host_year_only'),
        ((H1, S1, Y1), (H2, S1, Y1), 'subtype_year_only'),
        ((H1, S1, Y1), (H1, S1, Y1), 'host_subtype_year'),
    ]
    for cell_a, cell_b, expected in cases:
        got = classify_pair_regime(cell_a, cell_b)
        assert got == expected, f'{cell_a} x {cell_b}: expected {expected}, got {got}'


def test_classify_null_treated_as_no_match():
    """As of 2026-05-11 the unknown_metadata_neg regime was retired. Pairs
    with null axes classify per the existing 8-regime mapping using the rule
    "null on either side of an axis is no-match on that axis."""
    # Left side null host, axes (host, subtype, year): subtype + year match
    # -> regime = subtype_year_only.
    assert classify_pair_regime((None, 'H3N2', 2024), ('Human', 'H3N2', 2024)) == 'subtype_year_only'
    # Right side null subtype: host + year match -> host_year_only.
    assert classify_pair_regime(('Human', 'H3N2', 2024), ('Human', None, 2024)) == 'host_year_only'
    # Null year on left: host + subtype match -> host_subtype_only.
    assert classify_pair_regime(('Human', 'H3N2', None), ('Human', 'H3N2', 2024)) == 'host_subtype_only'
    # Null on the SAME axis on both sides -> still treated as no-match.
    assert classify_pair_regime(('Human', 'H3N2', None), ('Human', 'H3N2', None)) == 'host_subtype_only'
    # All-null cells -> nothing matches -> none_match.
    assert classify_pair_regime((None, None, None), (None, None, None)) == 'none_match'


def test_compute_match_count():
    assert compute_match_count(('A', 'B', 'C'), ('A', 'B', 'C')) == 3
    assert compute_match_count(('A', 'B', 'C'), ('A', 'B', 'D')) == 2
    assert compute_match_count(('A', 'B', 'C'), ('A', 'X', 'D')) == 1
    assert compute_match_count(('A', 'B', 'C'), ('X', 'Y', 'Z')) == 0
    # Null axis no longer returns None; it just contributes 0 to the count.
    assert compute_match_count(('A', None, 'C'), ('A', 'B', 'C')) == 2
    assert compute_match_count((None, None, None), (None, None, None)) == 0


def test_build_isolate_cells_binned_year():
    df = pd.DataFrame({
        'assembly_id': ['A', 'B', 'C'],
        'host': ['Human', 'Pig', None],
        'hn_subtype': ['H3N2', 'H1N1', 'H3N2'],
        'year': [2024, 2018, 2014],
    })
    cells = build_isolate_cells(df, axes=DEFAULT_AXES, year_match='binned')
    assert cells['A'] == ('Human', 'H3N2', '2021+')
    assert cells['B'] == ('Pig', 'H1N1', '2016-2020')
    assert cells['C'] == (None, 'H3N2', '<=2015')


def test_build_isolate_cells_exact_year():
    df = pd.DataFrame({
        'assembly_id': ['A'],
        'host': ['Human'], 'hn_subtype': ['H3N2'], 'year': [2024],
    })
    cells = build_isolate_cells(df, axes=DEFAULT_AXES, year_match='exact')
    assert cells['A'] == ('Human', 'H3N2', 2024)


def test_count_available_per_regime_two_cells_same_subtype_year():
    # 3 isolates in cell1 (Human, H3N2, 2021+) and 2 in cell2 (Pig, H3N2, 2021+).
    # Within cell1: 3*2 = 6 host_subtype_year pairs (ordered).
    # Within cell2: 2*1 = 2 host_subtype_year pairs.
    # Across cells: 3*2 + 2*3 = 12 subtype_year_only pairs.
    cell1 = ('Human', 'H3N2', '2021+')
    cell2 = ('Pig',   'H3N2', '2021+')
    cell_counts = {cell1: 3, cell2: 2}
    avail = count_available_per_regime(cell_counts)
    assert avail['host_subtype_year'] == 6 + 2
    assert avail['subtype_year_only'] == 3*2 + 2*3
    assert avail['none_match'] == 0
    assert sum(avail.values()) == 5 * 4  # total ordered pairs i != j


def test_count_available_null_axis_routes_to_8_regimes():
    """As of 2026-05-11 there's no unknown_metadata_neg catch-all -- pairs
    involving a null axis classify into one of the 8 regimes via the
    null-is-no-match rule. With cells differing only on a null year axis,
    pairs that previously fell into unknown_metadata_neg now route to
    host_subtype_only (host + subtype match; null year mismatches)."""
    cell_known = ('Human', 'H3N2', '2021+')   # fully specified
    cell_unknown = ('Human', 'H3N2', None)     # null year
    cell_counts = {cell_known: 3, cell_unknown: 2}
    avail = count_available_per_regime(cell_counts)
    # Within-cell pairs of cell_known: 3*2 = 6 -> host_subtype_year (all match).
    # Within-cell pairs of cell_unknown: 2*1 = 2 -> host_subtype_only
    #   (host + subtype match; null year is no-match by rule).
    # Cross-cell: 3*2 + 2*3 = 12 -> host_subtype_only (year mismatches).
    assert avail['host_subtype_year'] == 6
    assert avail['host_subtype_only'] == 2 + 12
    assert sum(avail.values()) == 5 * 4


def test_count_isolates_per_cell():
    isolate_to_cell = {'A': ('X', 'Y'), 'B': ('X', 'Y'), 'C': ('Z', 'W')}
    counts = count_isolates_per_cell(isolate_to_cell)
    assert counts == {('X', 'Y'): 2, ('Z', 'W'): 1}


def test_resolve_regime_targets_round_correction():
    # 1000 negatives across the 8 regimes; 0.30 host_subtype_year, 0.20 none_match,
    # 0.10 each for host_only/subtype_only/year_only/host_subtype_only,
    # 0.05 each for host_year_only/subtype_year_only.
    targets = {
        'none_match': 0.20,
        'host_only': 0.10,
        'subtype_only': 0.10,
        'year_only': 0.10,
        'host_subtype_only': 0.10,
        'host_year_only': 0.05,
        'subtype_year_only': 0.05,
        'host_subtype_year': 0.30,
    }
    counts = resolve_regime_targets(targets, 1000)
    assert sum(counts.values()) == 1000
    # Spot-check exact values where rounding is unambiguous.
    assert counts['host_subtype_year'] == 300
    assert counts['none_match'] == 200


def test_resolve_regime_targets_drift_correction():
    # Targets sum to 1 but rounding gives 7 or 9 for 8 budget on equal split.
    targets = {r: 1.0 / 8 for r in REGIME_NAMES}
    counts = resolve_regime_targets(targets, 100)
    assert sum(counts.values()) == 100


if __name__ == '__main__':
    test_bin_year_default_edges()
    test_classify_pair_all_eight_regimes()
    test_classify_null_treated_as_no_match()
    test_compute_match_count()
    test_build_isolate_cells_binned_year()
    test_build_isolate_cells_exact_year()
    test_count_available_per_regime_two_cells_same_subtype_year()
    test_count_available_null_axis_routes_to_8_regimes()
    test_count_isolates_per_cell()
    test_resolve_regime_targets_round_correction()
    test_resolve_regime_targets_drift_correction()
    print('Done. All tests passed.')
