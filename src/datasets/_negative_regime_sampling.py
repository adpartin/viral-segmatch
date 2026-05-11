"""Metadata-aware negative-pair regime utilities.

Companion to `dataset_segment_pairs_v2.py`. Implements:

- 8 mutually-exclusive regimes over (host, hn_subtype, year_or_year_bin).
- Closed-form candidate-count via metadata-cell groupby (no enumeration).
- Pair-level regime classification + match_count.

Called from `create_negative_pairs_v2` once the `axis_quotas` parameter is
non-empty; otherwise the legacy regime-blind sampler runs unchanged.

Cell representation: a tuple in `(host, hn_subtype, year_or_year_bin)`
order matching the configured `axes` list. Null axis values become `None`
in the tuple. For classification, **null on either side of an axis counts
as no-match on that axis** (a null host is treated as "different from
everything," including another null host) -- the 8 regimes still cover
every cell pair; there is no separate "unknown" bucket. See
`docs/plans/2026-05-11_metadata_holdout_plan.md` for the 2026-05-11
removal of the legacy `unknown_metadata_neg` regime.

Determinism: this module is purely functional. It contains no RNG; any
sampling order is owned by the caller.
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


REGIME_NAMES = (
    'none_match',
    'host_only',
    'subtype_only',
    'year_only',
    'host_subtype_only',
    'host_year_only',
    'subtype_year_only',
    'host_subtype_year',
)

DEFAULT_AXES = ('host', 'hn_subtype', 'year')

DEFAULT_YEAR_BIN_EDGES = (
    (float('-inf'), 2015, '<=2015'),
    (2016,          2020, '2016-2020'),
    (2021, float('inf'),  '2021+'),
)


# Match tuple is (host_match, subtype_match, year_match).
# Order in the tuple follows DEFAULT_AXES; if the caller uses a different
# axis order, the classifier rebuilds the match tuple in DEFAULT_AXES order
# before lookup, so the table is invariant.
_MATCH_TUPLE_TO_REGIME = {
    (False, False, False): 'none_match',
    (True,  False, False): 'host_only',
    (False, True,  False): 'subtype_only',
    (False, False, True ): 'year_only',
    (True,  True,  False): 'host_subtype_only',
    (True,  False, True ): 'host_year_only',
    (False, True,  True ): 'subtype_year_only',
    (True,  True,  True ): 'host_subtype_year',
}


def bin_year(year_value, edges=DEFAULT_YEAR_BIN_EDGES) -> Optional[str]:
    """Map a year integer to its bin label. Returns None if year is None or
    falls outside every edge (which shouldn't happen with the default
    [-inf, +inf] edges). Accepts int, float, numpy-int, pandas Int64 NA."""
    if year_value is None or pd.isna(year_value):
        return None
    y = int(year_value)
    for lo, hi, label in edges:
        if lo <= y <= hi:
            return label
    return None


def build_isolate_cells(
    meta_df: pd.DataFrame,
    *,
    axes: Iterable[str] = DEFAULT_AXES,
    year_match: str = 'binned',
    year_bin_edges=DEFAULT_YEAR_BIN_EDGES,
    assembly_col: str = 'assembly_id',
    ) -> dict:
    """Map each assembly_id to a cell tuple.

    Args:
        meta_df: per-isolate metadata (one row per assembly_id).
        axes: ordered iterable of axis column names; cells are tuples in
            this order.
        year_match: 'binned' (use year_bin_edges) or 'exact' (use raw year).
        year_bin_edges: list of (lo_inclusive, hi_inclusive, label).
        assembly_col: name of the assembly_id column in meta_df.

    Returns:
        dict[assembly_id (str) -> cell (tuple)]
    """
    if year_match not in {'binned', 'exact'}:
        raise ValueError(f"year_match must be 'binned' or 'exact', got {year_match!r}")
    axes = list(axes)
    if assembly_col not in meta_df.columns:
        raise KeyError(f"meta_df missing column {assembly_col!r}")
    missing = [a for a in axes if a not in meta_df.columns]
    if missing:
        raise KeyError(f"meta_df missing axis columns: {missing}")

    out: dict = {}
    for row in meta_df.itertuples(index=False):
        aid = str(getattr(row, assembly_col))
        cell = []
        for axis in axes:
            v = getattr(row, axis)
            if v is None or pd.isna(v):
                cell.append(None)
            elif axis == 'year' and year_match == 'binned':
                cell.append(bin_year(v, year_bin_edges))
            else:
                cell.append(v)
        out[aid] = tuple(cell)
    return out


def _axis_match(a, b) -> bool:
    """True iff both sides are non-null and equal.

    Treats null on either side as no-match (including null == null). This
    is conservative: two isolates whose host is "unknown" are not claimed
    to share a host; they fall into whichever regime captures the other
    axes' agreement.
    """
    if a is None or b is None:
        return False
    if pd.isna(a) or pd.isna(b):
        return False
    return a == b


def classify_pair_regime(
    cell_a: tuple,
    cell_b: tuple,
    *,
    axes: Iterable[str] = DEFAULT_AXES,
    ) -> str:
    """Classify an ordered pair of isolate cells into one of eight regimes.

    Per-axis comparison uses `_axis_match`: null on either side of an axis
    counts as no-match on that axis. The resulting
    (host_match, subtype_match, year_match) tuple maps to one of the eight
    regimes via `_MATCH_TUPLE_TO_REGIME`. Pairs where every axis comparison
    is no-match (which includes any pair with all-null cells on both sides)
    end up in `none_match`.
    """
    axes = list(axes)
    matches = {}
    for ax_idx, axis in enumerate(axes):
        matches[axis] = _axis_match(cell_a[ax_idx], cell_b[ax_idx])
    key = (matches.get('host', False),
           matches.get('hn_subtype', False),
           matches.get('year', False))
    return _MATCH_TUPLE_TO_REGIME[key]


def compute_match_count(
    cell_a: tuple,
    cell_b: tuple,
    ) -> int:
    """Count axes (in the cell tuple order) where both sides are non-null
    and equal. Null on either side of an axis contributes 0 to the count
    (matches `_axis_match` semantics).
    """
    return sum(1 for a, b in zip(cell_a, cell_b) if _axis_match(a, b))


def count_isolates_per_cell(isolate_to_cell: dict) -> dict:
    """Invert isolate_to_cell into cell -> n_isolates."""
    counts: dict = {}
    for cell in isolate_to_cell.values():
        counts[cell] = counts.get(cell, 0) + 1
    return counts


def count_available_per_regime(
    cell_counts: dict,
    *,
    axes: Iterable[str] = DEFAULT_AXES,
    ) -> dict:
    """Closed-form available_count[regime] over ordered (cell_i, cell_j)
    pairs using cell sizes.

    Within-cell ordered pairs: n * (n-1).
    Cross-cell ordered pairs (c1, c2): n1 * n2.
    Each pair is classified into a regime and added to that regime's count.

    Cooccur and seq-exclusion rejections are uniform-in-expectation across
    regimes, so this is an upper bound (over-estimate by a known factor)
    -- correct for feasibility checks and manifests.
    """
    counts = {r: 0 for r in REGIME_NAMES}
    cells = list(cell_counts.keys())
    for c1 in cells:
        n1 = cell_counts[c1]
        for c2 in cells:
            n2 = cell_counts[c2]
            if c1 == c2:
                pairs = n1 * (n1 - 1)
            else:
                pairs = n1 * n2
            if pairs == 0:
                continue
            r = classify_pair_regime(c1, c2, axes=axes)
            counts[r] += pairs
    return counts


def resolve_regime_targets(
    regime_targets: dict,
    num_negatives: int,
    ) -> dict:
    """Convert per-regime target fractions to per-regime target counts.

    Targets must already be validated (sum to 1.0, all REGIME_NAMES present).
    Sum of rounded counts is forced to equal num_negatives by adjusting the
    largest target by +/-1 until the residual is zero. Adjustments are
    applied in deterministic order (sorted regime names) so the same input
    produces the same per-regime counts.
    """
    counts = {r: int(round(num_negatives * regime_targets[r])) for r in REGIME_NAMES}
    drift = num_negatives - sum(counts.values())
    if drift == 0:
        return counts
    direction = 1 if drift > 0 else -1
    for _ in range(abs(drift)):
        target_regime = max(
            (r for r in sorted(REGIME_NAMES) if regime_targets[r] > 0 or counts[r] > 0),
            key=lambda r: regime_targets[r],
        )
        counts[target_regime] += direction
    return counts
