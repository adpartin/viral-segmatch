"""Within-CC isolate pool + within-CC negative sampling for the maintained
CC-based CV builder (`dataset_pairs_cc.py`).

Moved here from `src/analysis/_cv_sampling.py` so the dataset generator can use
these without `src/datasets` importing `src/analysis` (the dependency rule
documented in `_megacc_cut.py`: analysis depends on datasets, never the
reverse). The analysis CV harness now imports these back from here.

What lives here (datasets-only deps — `_pair_helpers`,
`_negative_regime_sampling`):
  - `build_cc_isolate_pool` — the fold->isolate bridge: reconstructs, for the
    cluster structure the splitter produced, the isolates available inside each
    CC for within-CC negative sampling.
  - `sample_random_within_cc_negatives` / `sample_regime_negatives` — pair a
    self isolate's slot-a sequence with a partner isolate's slot-b sequence,
    BOTH drawn from the same CC, rejecting true co-occurrences and duplicates.

What does NOT live here: **atom assignment**. Callers build atoms with the
production primitives — `_split_helpers.attach_cluster_ids` +
`_pair_helpers.bipartite_components(col_a='cluster_id_a', col_b='cluster_id_b')`
— which are alphabet-clean (nt_cds keys on `cds_dna_hash`, NOT the analysis
`dna_hash`-means-CDS mislabel) and carry no `src/analysis` dependency. The 'cut'
(fragmentation) strategy reuses `_megacc_cut.apply_drop_budget_cut`.

Column convention: the sequence-hash columns are GENERIC (`hash_a` / `hash_b`),
holding whichever alphabet's per-slot sequence hash the membership carries
(aa: `seq_hash` / nt_cds: `cds_dna_hash` / nt_ctg: `dna_hash`). The caller renames
`hash_a`/`hash_b` to the alphabet-specific `_PAIR_COLUMNS` names when enriching
negatives. This keeps the samplers alphabet-agnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets._negative_regime_sampling import (  # noqa: E402
    DEFAULT_AXES,
    DEFAULT_YEAR_BIN_EDGES,
    REGIME_NAMES,
    build_cell_regime_partners,
    build_isolate_cells,
    classify_pair_regime,
    count_available_per_regime,
    count_isolates_per_cell,
    resolve_regime_targets,
)
from src.datasets._pair_helpers import canonical_pair_key  # noqa: E402
from src.utils.config_hydra import load_function_metadata  # noqa: E402
from src.utils.schema import SCHEMA as _SCHEMA  # noqa: E402

# Per-isolate membership table + hash column per alphabet, derived from the
# canonical schema registry (src/utils/schema.py) -- the single source of truth.
# cluster_memb_* has one row per (isolate, function) with the per-slot sequence
# hash, host/hn_subtype/year, and a cluster column per tXXX threshold.
_MEMB_DIR = PROJ / 'data/processed/flu/July_2025/cluster_membership'
_MEMB = {a: _MEMB_DIR / f'{s.memb_basename}.parquet' for a, s in _SCHEMA.items()}
_MEMB_HASH = {a: s.hash_col for a, s in _SCHEMA.items()}

# Regime name -> metadata match count (0..3); the regime fully determines it.
_REGIME_TO_MC = {'none_match': 0, 'host_only': 1, 'subtype_only': 1, 'year_only': 1,
                 'host_subtype_only': 2, 'host_year_only': 2, 'subtype_year_only': 2,
                 'host_subtype_year': 3}
_SHORT_TO_FULL: dict | None = None  # lazy {short -> full function name} from flu.yaml

_NEG_COLS = ['hash_a', 'hash_b', 'neg_regime', 'metadata_match_count']


def _short_to_full(short: str) -> str:
    """Map a protein short name (e.g. 'HA') to its full function name, lazily."""
    global _SHORT_TO_FULL
    if _SHORT_TO_FULL is None:
        _SHORT_TO_FULL = dict(load_function_metadata(PROJ / 'conf' / 'virus' / 'flu.yaml')
                              .short_to_function)
    return _SHORT_TO_FULL[short]


def build_cc_isolate_pool(
    cluster_to_atom: dict,
    cluster_to_cc: dict,
    slot_a: str,
    slot_b: str,
    alphabet: str,
    threshold: str,
    *,
    axes=DEFAULT_AXES,
    year_match: str = 'binned',
    year_bin_edges=DEFAULT_YEAR_BIN_EDGES,
    ) -> pd.DataFrame:
    """Per-isolate pool inside each CC, for within-CC negative sampling.

    The deduped pair universe loses isolate multiplicity (one row per unique
    pair_key), but regime negatives are an isolate-metadata property. This
    reconstructs, for the SAME cluster structure the splitter produced, the
    isolates available inside each CC.

    Args:
        cluster_to_atom / cluster_to_cc: {cluster_id (str) -> atom_id / cc_id},
            built by the caller from the production atoms — i.e. from
            `pos_df['cluster_id_a' / 'cluster_id_b']` paired with the
            `bipartite_components` component id. Cluster ids are globally unique
            across slots (e.g. `HA_123` vs `NA_45`), so no a:/b: prefix is needed.
        slot_a / slot_b: protein short names (e.g. 'HA', 'NA').
        alphabet: 'aa' | 'nt_cds' | 'nt_ctg' — selects `_MEMB` + `_MEMB_HASH`.
        threshold: tXXX cluster column in the membership table (e.g. 't099').

    Returns:
        iso[assembly_id, hash_a, hash_b, cc_id, atom_id, cell] — one row per
        isolate carrying BOTH slots; isolates whose slot-a cluster is absent from
        `cluster_to_atom` (cluster dropped from the routed universe) are dropped.
    """
    if alphabet not in _MEMB:
        raise ValueError(f"alphabet must be in {sorted(_MEMB)}; got {alphabet!r}")
    hash_col = _MEMB_HASH[alphabet]
    memb = pd.read_parquet(
        _MEMB[alphabet],
        columns=['assembly_id', 'function', hash_col, threshold, 'host', 'hn_subtype', 'year'])
    fa, fb = _short_to_full(slot_a), _short_to_full(slot_b)
    a = (memb[memb['function'] == fa]
         [['assembly_id', hash_col, threshold, 'host', 'hn_subtype', 'year']]
         .rename(columns={hash_col: 'hash_a', threshold: 'cluster_a'}))
    b = (memb[memb['function'] == fb][['assembly_id', hash_col, threshold]]
         .rename(columns={hash_col: 'hash_b', threshold: 'cluster_b'}))
    iso = a.merge(b, on='assembly_id', how='inner')
    iso['assembly_id'] = iso['assembly_id'].astype(str)

    # Cluster-disjointness guarantees an isolate's slot-a and slot-b clusters land
    # in the same atom, so the slot-a cluster alone determines atom_id / cc_id.
    iso['atom_id'] = iso['cluster_a'].astype(str).map(cluster_to_atom)
    iso['cc_id'] = iso['cluster_a'].astype(str).map(cluster_to_cc)
    iso = iso.dropna(subset=['atom_id']).copy()  # drop isolates whose cluster left the universe
    iso['atom_id'] = iso['atom_id'].astype(int)
    iso['cc_id'] = iso['cc_id'].astype(int)

    cells = build_isolate_cells(
        iso[['assembly_id', 'host', 'hn_subtype', 'year']].drop_duplicates('assembly_id'),
        axes=axes, year_match=year_match, year_bin_edges=year_bin_edges)
    iso['cell'] = iso['assembly_id'].map(cells)

    return (iso[['assembly_id', 'hash_a', 'hash_b', 'cc_id', 'atom_id', 'cell']]
            .reset_index(drop=True))


def sample_random_within_cc_negatives(
    cc_iso: pd.DataFrame,
    budget: int,
    cooccur: set,
    *,
    seed: int,
    axes=DEFAULT_AXES,
    ) -> pd.DataFrame:
    """Random (NOT regime-targeted) within-CC negatives; each labeled by its regime.

    Pairs a random self isolate's slot-a seq (`hash_a`) with a random partner's
    slot-b seq (`hash_b`) within one CC's isolate pool, rejects true pairs
    (`cooccur`) + duplicates, and tags the regime post-hoc via
    `classify_pair_regime`. Yields the CC's *natural* regime mix (whatever its
    metadata pairing produces). A single-isolate CC yields nothing.

    Returns columns [hash_a, hash_b, neg_regime, metadata_match_count].
    """
    if len(cc_iso) < 2 or budget <= 0:
        return pd.DataFrame(columns=_NEG_COLS)
    arr = cc_iso[['assembly_id', 'hash_a', 'hash_b', 'cell']].to_numpy()
    rng = np.random.RandomState(seed)
    rows, seen = [], set()
    placed, attempts, max_attempts = 0, 0, budget * 50 + 200
    while placed < budget and attempts < max_attempts:
        attempts += 1
        x = arr[rng.randint(len(arr))]
        y = arr[rng.randint(len(arr))]
        if x[0] == y[0]:
            continue  # same isolate
        ha, nb = x[1], y[2]  # self slot-a seq, partner slot-b seq
        pk = canonical_pair_key(ha, nb)
        if pk in cooccur or pk in seen:
            continue
        seen.add(pk)
        regime = classify_pair_regime(x[3], y[3], axes=axes)
        rows.append((ha, nb, regime, _REGIME_TO_MC[regime]))
        placed += 1
    return pd.DataFrame(rows, columns=_NEG_COLS)


def sample_regime_negatives(
    cc_iso: pd.DataFrame,
    regime_targets: dict,
    budget: int,
    cooccur: set,
    *,
    seed: int,
    axes=DEFAULT_AXES,
    ) -> pd.DataFrame:
    """Within-CC, regime-targeted negatives from one CC's isolate pool (best-effort).

    Pairs a self isolate's slot-a seq with a partner isolate's slot-b seq so that
    the (self_cell, partner_cell) metadata match yields the target regime; rejects
    true pairs (`cooccur`) and duplicates. Allocation: `resolve_regime_targets`
    then redistribute any per-regime shortfall (a regime unavailable in this CC)
    to regimes with spare availability, proportional to their target, up to
    `budget`. Availability is the closed-form `count_available_per_regime` (an
    upper bound — cooccur/dup rejections mean the achieved count can fall below
    the allocation), so the returned frame may have fewer than `budget` rows.

    Returns columns [hash_a, hash_b, neg_regime, metadata_match_count].
    """
    isolate_to_cell = dict(zip(cc_iso['assembly_id'], cc_iso['cell']))
    if len(isolate_to_cell) < 2 or budget <= 0:
        return pd.DataFrame(columns=_NEG_COLS)

    avail = count_available_per_regime(count_isolates_per_cell(isolate_to_cell), axes=axes)
    desired = resolve_regime_targets(regime_targets, budget)
    alloc = {r: min(desired.get(r, 0), avail.get(r, 0)) for r in REGIME_NAMES}

    # Redistribute the deficit to regimes with spare availability (preserve the
    # target shape as far as the CC's availability allows).
    deficit = budget - sum(alloc.values())
    while deficit > 0:
        spare = {r: avail.get(r, 0) - alloc[r]
                 for r in REGIME_NAMES if avail.get(r, 0) - alloc[r] > 0}
        if not spare:
            break
        wsum = sum(regime_targets.get(r, 0.0) for r in spare) or float(len(spare))
        added = 0
        for r in spare:
            share = (regime_targets.get(r, 0.0) or 1.0) / wsum
            take = min(spare[r], max(1, int(round(deficit * share))), deficit - added)
            alloc[r] += take
            added += take
            if added >= deficit:
                break
        if added == 0:
            break
        deficit -= added

    # cell -> array of (assembly_id, hash_a, hash_b) for O(1) random draws.
    cell_groups = {cell: g[['assembly_id', 'hash_a', 'hash_b']].to_numpy()
                   for cell, g in cc_iso.groupby('cell')}
    partners = build_cell_regime_partners(isolate_to_cell, axes=axes)

    rng = np.random.RandomState(seed)
    rows: list = []
    seen: set = set()
    for r in REGIME_NAMES:
        need = alloc[r]
        if need <= 0:
            continue
        cell_pairs = [(sc, pc) for sc, rmap in partners.items() for pc in rmap.get(r, [])]
        if not cell_pairs:
            continue
        placed, attempts, max_attempts = 0, 0, need * 50 + 200
        while placed < need and attempts < max_attempts:
            attempts += 1
            sc, pc = cell_pairs[rng.randint(len(cell_pairs))]
            sg, pg = cell_groups[sc], cell_groups[pc]
            x = sg[rng.randint(len(sg))]
            y = pg[rng.randint(len(pg))]
            if x[0] == y[0]:
                continue  # same isolate (within-cell self-pair)
            ha, nb = x[1], y[2]  # self slot-a seq, partner slot-b seq
            pk = canonical_pair_key(ha, nb)
            if pk in cooccur or pk in seen:
                continue  # reject true pair / duplicate negative
            seen.add(pk)
            rows.append((ha, nb, r, _REGIME_TO_MC[r]))
            placed += 1
    return pd.DataFrame(rows, columns=_NEG_COLS)
