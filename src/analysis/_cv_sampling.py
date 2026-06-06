"""Cluster-disjoint CV sampling: CC assignment + per-CC positive/negative sampling.

The evolving half of the cluster-disjoint CV experiment (orchestrated by
`cluster_disjoint_cv_experiment.py`). Designed so the sampling strategy can grow
from random one-per-CC to metadata-aware cap-k-per-CC WITHOUT touching the CV
structure: selection happens WITHIN each connected component, so the CC->fold
assignment — hence cluster-disjointness — is invariant to the strategy.

Key fact: representatives from distinct CCs share no cluster (else their cells
would sit in one CC), so cap-k-per-CC + GroupKFold-by-`cc_id` is cluster-disjoint
for any k (and k=1 degenerates to a plain KFold, since each group is a singleton).
No mega-CC cut is needed — capping bounds every group to <= k pairs.

Metadata hook: `assign_cc` carries the pair universe's hashes; a future
`strategy='metadata_*'` selects which <=k pairs to keep per CC using the
membership metadata, with the fold structure unchanged. Negatives mirror this via
`_negative_regime_sampling` (the 8 hard-negative regimes) — deferred.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.bigraph_properties import load_cluster_map  # noqa: E402 (membership-backed)

_ROOT = {'aa': PROJ / 'data/processed/flu/July_2025/clusters_aa',
         'nt_cds': PROJ / 'data/processed/flu/July_2025/clusters_nt_cds'}
# hash columns on the pair universe (load_pair_universe) per alphabet.
_HASH = {'aa': ('seq_hash_a', 'seq_hash_b'),
         'nt_cds': ('dna_hash_a', 'dna_hash_b')}


def assign_cc(universe: pd.DataFrame, slot_a: str, slot_b: str, alphabet: str,
              threshold: str) -> pd.DataFrame:
    """Tag every pair with cluster_a, cluster_b, and its connected-component id.

    Drops pairs whose either endpoint lacks a cluster at this threshold.
    """
    ha, hb = _HASH[alphabet]
    cmap_a = load_cluster_map(_ROOT[alphabet], slot_a, threshold)
    cmap_b = load_cluster_map(_ROOT[alphabet], slot_b, threshold)
    u = universe.copy()
    u['cluster_a'] = u[ha].map(cmap_a)
    u['cluster_b'] = u[hb].map(cmap_b)
    u = u.dropna(subset=['cluster_a', 'cluster_b']).reset_index(drop=True)

    node_a = ('a:' + u['cluster_a'].astype(str))
    node_b = ('b:' + u['cluster_b'].astype(str))
    G = nx.Graph()
    G.add_edges_from(zip(node_a, node_b))
    node_cc = {n: i for i, c in enumerate(nx.connected_components(G)) for n in c}
    u['cc_id'] = node_a.map(node_cc).to_numpy()
    return u


def sample_positives(pairs: pd.DataFrame, max_per_cc: int = 1, seed: int = 0,
                     strategy: str = 'random') -> pd.DataFrame:
    """Take up to `max_per_cc` pairs per CC. k=1 => one-per-CC (plain-KFold-ready).

    `strategy='random'` for now; metadata-aware strategies (e.g. subtype-diverse,
    regime-matched) plug in here without changing the CC->fold structure.
    """
    if strategy != 'random':
        raise NotImplementedError(
            f"sample_positives strategy={strategy!r} not implemented; only 'random'. "
            f"Metadata-aware selection is a planned within-CC extension.")
    rng = np.random.RandomState(seed)

    def _take(g: pd.DataFrame) -> pd.DataFrame:
        return g if len(g) <= max_per_cc else g.sample(n=max_per_cc, random_state=rng)

    return (pairs.groupby('cc_id', group_keys=False).apply(_take)
            .reset_index(drop=True))


def make_negatives(fold_pos: pd.DataFrame, alphabet: str, seed: int = 0,
                   strategy: str = 'random') -> pd.DataFrame:
    """1:1 mismatch negatives within one fold (cluster-disjoint by construction).

    Pairs each fold positive's slot-a sequence with a shuffled slot-b sequence
    from the SAME fold, dropping accidental true pairs. Both endpoints stay inside
    the fold's clusters, so no cross-fold leakage. `strategy='regime'` (hard
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
