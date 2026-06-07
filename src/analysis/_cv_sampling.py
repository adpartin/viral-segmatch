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

_ROOT = {'aa': PROJ / 'data/processed/flu/July_2025/clusters_aa',
         'nt_cds': PROJ / 'data/processed/flu/July_2025/clusters_nt_cds'}
# hash columns on the pair universe (load_pair_universe) per alphabet.
_HASH = {'aa': ('seq_hash_a', 'seq_hash_b'),
         'nt_cds': ('dna_hash_a', 'dna_hash_b')}


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
