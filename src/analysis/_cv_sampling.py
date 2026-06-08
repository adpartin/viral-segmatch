"""Cluster-disjoint CV sampling: atom assignment + per-unit positive/negative sampling.

The evolving sampling half of the cluster-disjoint CV experiment (orchestrated by
`cluster_disjoint_cv_experiment.py`). Three nested units on the cluster-level
bigraph (nodes = clusters, edges = co-occurrences):

  sequence pair   -- one positive example (a row of the pair universe)
    in a cluster pair  -- a unique (cluster_a, cluster_b); the SAMPLING unit
      in an atom       -- a connected (sub-)component; the GroupKFold SPLIT unit

`assign_atoms` tags every pair with `cluster_pair_id`, `cc_id`, and `atom_id`:
  - strategy='natural': atom = the bipartite CC (`atom_id == cc_id`). At low `t` one
    mega-CC dominates, so GroupKFold-by-atom needs per-CC capping to stay balanced.
  - strategy='cut': the mega-CC is fragmented by edge min-cut (bigraph_min_cut's
    `fragment_weighted` with `uniform_targets(k_folds)`) into atoms each <= ~1/k of
    the kept pairs; straddling pairs (endpoints in different atoms) are DROPPED.
    This lets per-cluster-pair sampling grow N at low `t` while the folds stay
    cluster-disjoint and balanced.

Cluster-disjointness invariant (asserted): every cluster belongs to exactly one
atom, so GroupKFold-by-`atom_id` never shares a cluster across folds, for any
sampling cap.

Selection happens WITHIN a unit, so the atom->fold assignment — hence
cluster-disjointness — is invariant to the sampling strategy. Metadata-aware
strategies (subtype-diverse positives; the 8 hard-negative regimes via
`_negative_regime_sampling`) plug into `sample_positives`/`make_negatives` later
without touching the fold structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.bigraph_properties import load_cluster_map  # noqa: E402 (membership-backed)
from src.analysis.bigraph_min_cut import fragment_weighted, uniform_targets  # noqa: E402
from src.datasets._pair_helpers import canonical_pair_key  # noqa: E402
from src.datasets._negative_regime_sampling import (  # noqa: E402
    REGIME_NAMES, DEFAULT_AXES, DEFAULT_YEAR_BIN_EDGES,
    build_isolate_cells, count_isolates_per_cell, count_available_per_regime,
    build_cell_regime_partners, resolve_regime_targets)
from src.utils.config_hydra import load_function_metadata  # noqa: E402

_ROOT = {'aa': PROJ / 'data/processed/flu/July_2025/clusters_aa',
         'nt_cds': PROJ / 'data/processed/flu/July_2025/clusters_nt_cds'}
# hash columns on the pair universe (load_pair_universe) per alphabet.
_HASH = {'aa': ('seq_hash_a', 'seq_hash_b'),
         'nt_cds': ('dna_hash_a', 'dna_hash_b')}
# per-isolate membership table per alphabet (assembly_id, function, seq_hash,
# host/hn_subtype/year, and a cluster column per tXXX) -- the protein-level
# (aa) / CDS-level (nt) source for the fold->isolate bridge.
_MEMB = {'aa': PROJ / 'data/processed/flu/July_2025/cluster_membership/cluster_memb_aa.parquet',
         'nt_cds': PROJ / 'data/processed/flu/July_2025/cluster_membership/cluster_memb_nt_cds.parquet'}
# regime name -> metadata match count (0..3); the regime fully determines it.
_REGIME_TO_MC = {'none_match': 0, 'host_only': 1, 'subtype_only': 1, 'year_only': 1,
                 'host_subtype_only': 2, 'host_year_only': 2, 'subtype_year_only': 2,
                 'host_subtype_year': 3}
_SHORT_TO_FULL: dict | None = None  # lazy {short -> full function name} from flu.yaml


def assign_atoms(
    universe: pd.DataFrame,
    slot_a: str,
    slot_b: str,
    alphabet: str,
    threshold: str,
    *,
    strategy: str = 'natural',
    k_folds: int = 5,
    cut_method: str = 'spectral',
    drift_pp: float = 0.05,
    seed: int = 1,
    return_audit: bool = False,
    ) -> pd.DataFrame:
    """Tag every pair with cluster_a/b, cluster_pair_id, cc_id, and atom_id.

    `cluster_pair_id` is the sampling unit (a unique (cluster_a, cluster_b));
    `atom_id` is the GroupKFold split unit. Under strategy='natural' atom == CC;
    under strategy='cut' the mega-CC is edge-min-cut into atoms <= ~1/k_folds and
    straddling pairs are dropped. Drops pairs whose either endpoint lacks a cluster
    at this threshold. Asserts no cluster spans two atoms. If `return_audit`, also
    returns a stats dict (n_natural, n_dropped, dropped_frac, n_kept, n_atoms,
    n_cluster_pairs, largest_atom_frac).
    """
    ha, hb = _HASH[alphabet]
    cmap_a = load_cluster_map(_ROOT[alphabet], slot_a, threshold)
    cmap_b = load_cluster_map(_ROOT[alphabet], slot_b, threshold)
    u = universe.copy()
    u['cluster_a'] = u[ha].map(cmap_a)
    u['cluster_b'] = u[hb].map(cmap_b)
    u = u.dropna(subset=['cluster_a', 'cluster_b']).reset_index(drop=True)

    u['node_a'] = 'a:' + u['cluster_a'].astype(str)
    u['node_b'] = 'b:' + u['cluster_b'].astype(str)
    u['cluster_pair_id'] = u['node_a'] + '|' + u['node_b']

    # Natural connected components (the leakage unit, pre-fragmentation).
    G = nx.Graph()
    G.add_edges_from(zip(u['node_a'], u['node_b']))
    node_cc = {n: i for i, c in enumerate(nx.connected_components(G)) for n in c}
    u['cc_id'] = u['node_a'].map(node_cc).to_numpy()

    n_natural = len(u)
    if strategy == 'natural':
        u['atom_id'] = u['cc_id']
        n_dropped = 0
    elif strategy == 'cut':
        u, n_dropped = _fragment_atoms(u, k_folds=k_folds, cut_method=cut_method,
                                       drift_pp=drift_pp, seed=seed)
    else:
        raise ValueError(f"strategy must be 'natural' or 'cut'; got {strategy!r}")

    _assert_cluster_disjoint(u)
    if not return_audit:
        return u
    sizes = u.groupby('atom_id').size()
    audit = {
        'strategy': strategy,
        'n_natural': int(n_natural),
        'n_dropped': int(n_dropped),
        'dropped_frac': round(n_dropped / n_natural, 4) if n_natural else 0.0,
        'n_kept': int(len(u)),
        'n_atoms': int(u['atom_id'].nunique()),
        'n_cluster_pairs': int(u['cluster_pair_id'].nunique()),
        'largest_atom_frac': round(float(sizes.max() / len(u)), 4) if len(u) else 0.0,
    }
    return u, audit


def _fragment_atoms(
    u: pd.DataFrame,
    *,
    k_folds: int,
    cut_method: str,
    drift_pp: float,
    seed: int,
    ) -> tuple[pd.DataFrame, int]:
    """Edge-min-cut the cluster bigraph to K-uniform atoms; drop straddling pairs.

    Builds the weighted simple graph (one edge per cluster pair, weight = #pairs),
    fragments it with `fragment_weighted(targets=uniform_targets(k_folds))`, then
    keeps only pairs whose two clusters share a final atom. The kept/dropped split
    must equal fragment_weighted's straddler count (asserted).
    """
    cp = u.groupby(['node_a', 'node_b']).size().reset_index(name='w')
    H = nx.Graph()
    for na, nb, w in zip(cp['node_a'], cp['node_b'], cp['w']):
        H.add_edge(na, nb, weight=int(w))
    cut_df, H_kept, _ = fragment_weighted(
        H, targets=uniform_targets(k_folds), method=cut_method,
        target_frac=1.0 / k_folds, drift_pp=drift_pp, seed=seed)

    node_atom = {n: i for i, c in enumerate(nx.connected_components(H_kept))
                 for n in c}
    atom_a = u['node_a'].map(node_atom)
    atom_b = u['node_b'].map(node_atom)
    keep = (atom_a == atom_b)
    n_dropped = int((~keep).sum())
    reported = int(cut_df.iloc[-1]['pairs_dropped'])
    assert n_dropped == reported, (
        f"straddler accounting mismatch: kept-side dropped {n_dropped} pairs but "
        f"fragment_weighted reported {reported}")
    u = u.loc[keep].copy()
    u['atom_id'] = atom_a[keep].astype(int).to_numpy()
    return u.reset_index(drop=True), reported


def _assert_cluster_disjoint(u: pd.DataFrame) -> None:
    """Every cluster (node) belongs to exactly one atom — the 2D-CD guarantee."""
    for side in ('node_a', 'node_b'):
        spans = u.groupby(side)['atom_id'].nunique()
        bad = spans[spans > 1]
        if len(bad):
            raise AssertionError(
                f"cluster-disjoint violation: {len(bad)} {side} cluster(s) span "
                f">1 atom (e.g. {bad.index[0]} in {int(bad.iloc[0])} atoms)")


def assign_cc(
    universe: pd.DataFrame,
    slot_a: str,
    slot_b: str,
    alphabet: str,
    threshold: str,
    ) -> pd.DataFrame:
    """Back-compat shim for `assign_atoms(strategy='natural')` (atom_id == cc_id)."""
    return assign_atoms(universe, slot_a, slot_b, alphabet, threshold,
                        strategy='natural')


def sample_positives(
    pairs: pd.DataFrame,
    max_per: int = 1,
    seed: int = 0,
    *,
    unit: str = 'cc',
    strategy: str = 'random',
    max_per_cc: int | None = None,
    ) -> pd.DataFrame:
    """Take up to `max_per` pairs per sampling `unit`.

    unit='cluster_pair': cap per (cluster_a, cluster_b) — the redundancy axis used
      by the fragmented fixed-N experiment. unit='cc': cap per connected component
      — the original per-CC m-sweep. `max_per_cc` is a deprecated alias for
      `max_per` (kept so pre-fragmentation callers run unchanged).

    `strategy='random'` for now; metadata-aware strategies (subtype-diverse,
    regime-matched) plug in here without changing the atom->fold structure.
    """
    if strategy != 'random':
        raise NotImplementedError(
            f"sample_positives strategy={strategy!r} not implemented; only 'random'. "
            f"Metadata-aware selection is a planned within-unit extension.")
    if max_per_cc is not None:
        max_per = max_per_cc
    col = {'cluster_pair': 'cluster_pair_id', 'cc': 'cc_id'}.get(unit)
    if col is None:
        raise ValueError(f"unit must be 'cluster_pair' or 'cc'; got {unit!r}")
    rng = np.random.RandomState(seed)

    def _take(g: pd.DataFrame) -> pd.DataFrame:
        return g if len(g) <= max_per else g.sample(n=max_per, random_state=rng)

    return (pairs.groupby(col, group_keys=False).apply(_take)
            .reset_index(drop=True))


def make_negatives(
    fold_pos: pd.DataFrame,
    alphabet: str,
    seed: int = 0,
    strategy: str = 'random'
    ) -> pd.DataFrame:
    """1:1 mismatch negatives within one fold (cluster-disjoint by construction).

    Pairs each fold positive's slot-a sequence with a shuffled slot-b sequence
    from the SAME fold, dropping accidental true pairs. Both endpoints stay inside
    the fold's atoms, so no cross-fold leakage (a fold may span several atoms; the
    within-fold shuffle keeps both endpoints in-fold). `strategy='regime'` (hard
    negatives via `_negative_regime_sampling`) is the planned extension.
    """
    if strategy != 'random':
        raise NotImplementedError(
            f"make_negatives strategy={strategy!r} not implemented; only 'random'.")
    ha, hb = _HASH[alphabet]
    rng = np.random.RandomState(seed)
    a = fold_pos[ha].to_numpy()
    b = fold_pos[hb].to_numpy()
    pos_keys = set(zip(a.tolist(), b.tolist()))
    perm = rng.permutation(len(b))
    na, nb = [], []
    for i in range(len(a)):
        j = perm[i]
        if a[i] == a[j] or (a[i], b[j]) in pos_keys:
            continue  # skip self-pair / accidental true positive
        na.append(a[i])
        nb.append(b[j])
    return pd.DataFrame({ha: na, hb: nb})


def _short_to_full(short: str) -> str:
    """Map a protein short name (e.g. 'HA') to its full function name, lazily."""
    global _SHORT_TO_FULL
    if _SHORT_TO_FULL is None:
        _SHORT_TO_FULL = dict(load_function_metadata(PROJ / 'conf' / 'virus' / 'flu.yaml')
                              .short_to_function)
    return _SHORT_TO_FULL[short]


def build_isolate_context(
    u: pd.DataFrame,
    universe: pd.DataFrame,
    slot_a: str,
    slot_b: str,
    alphabet: str,
    threshold: str,
    *,
    axes=DEFAULT_AXES,
    year_match: str = 'binned',
    year_bin_edges=DEFAULT_YEAR_BIN_EDGES,
    ) -> tuple[pd.DataFrame, set]:
    """Fold->isolate bridge: per-isolate table tagged with atom_id/cc_id + cell.

    Regime negatives are an isolate-metadata property, but the CV runs on the
    deduped pair universe. This reconstructs, for the SAME cluster structure
    `assign_atoms` produced (read off `u`'s node->atom map), the isolates
    available inside each CC for within-CC negative sampling.

    Source is the per-isolate membership table (`_MEMB[alphabet]`): one row per
    (isolate, function) with seq_hash, host/hn_subtype/year, and the tXXX cluster
    — so no metadata join, cluster reload, or seq_hash recompute. NATURAL-strategy
    only for now: under 'cut' an isolate whose positive edge straddled two atoms
    would map its slot-a/slot-b clusters to different atoms; that case is dropped
    here (atom taken from slot-a) and must be handled before running 'cut'.

    Returns:
        (iso, cooccur). `iso` has columns assembly_id, seq_hash_a, seq_hash_b,
        cc_id, atom_id, cell (one row per isolate carrying both slots). `cooccur`
        is the set of true canonical pair_keys (negatives must avoid it).
    """
    node_to_atom: dict = {}
    node_to_cc: dict = {}
    for col in ('node_a', 'node_b'):
        node_to_atom.update(dict(zip(u[col], u['atom_id'])))
        node_to_cc.update(dict(zip(u[col], u['cc_id'])))

    memb = pd.read_parquet(
        _MEMB[alphabet],
        columns=['assembly_id', 'function', 'seq_hash', threshold, 'host', 'hn_subtype', 'year'])
    fa, fb = _short_to_full(slot_a), _short_to_full(slot_b)
    a = (memb[memb['function'] == fa]
         [['assembly_id', 'seq_hash', threshold, 'host', 'hn_subtype', 'year']]
         .rename(columns={'seq_hash': 'seq_hash_a', threshold: 'cluster_a'}))
    b = (memb[memb['function'] == fb][['assembly_id', 'seq_hash', threshold]]
         .rename(columns={'seq_hash': 'seq_hash_b', threshold: 'cluster_b'}))
    iso = a.merge(b, on='assembly_id', how='inner')
    iso['assembly_id'] = iso['assembly_id'].astype(str)

    iso['node_a'] = 'a:' + iso['cluster_a'].astype(str)
    iso['atom_id'] = iso['node_a'].map(node_to_atom)
    iso['cc_id'] = iso['node_a'].map(node_to_cc)
    iso = iso.dropna(subset=['atom_id']).copy()  # drop isolates whose cluster was dropped in u
    iso['atom_id'] = iso['atom_id'].astype(int)
    iso['cc_id'] = iso['cc_id'].astype(int)

    cells = build_isolate_cells(
        iso[['assembly_id', 'host', 'hn_subtype', 'year']].drop_duplicates('assembly_id'),
        axes=axes, year_match=year_match, year_bin_edges=year_bin_edges)
    iso['cell'] = iso['assembly_id'].map(cells)

    cooccur = set(universe['pair_key'])
    return (iso[['assembly_id', 'seq_hash_a', 'seq_hash_b', 'cc_id', 'atom_id', 'cell']]
            .reset_index(drop=True), cooccur)


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
    the allocation), so the returned frame may have fewer than `budget` rows; a
    singleton / metadata-homogeneous CC returns few or none.

    Returns columns [seq_hash_a, seq_hash_b, neg_regime, metadata_match_count].
    """
    cols = ['seq_hash_a', 'seq_hash_b', 'neg_regime', 'metadata_match_count']
    isolate_to_cell = dict(zip(cc_iso['assembly_id'], cc_iso['cell']))
    if len(isolate_to_cell) < 2 or budget <= 0:
        return pd.DataFrame(columns=cols)

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

    # cell -> array of (assembly_id, seq_hash_a, seq_hash_b) for O(1) random draws.
    cell_groups = {cell: g[['assembly_id', 'seq_hash_a', 'seq_hash_b']].to_numpy()
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
    return pd.DataFrame(rows, columns=cols)
