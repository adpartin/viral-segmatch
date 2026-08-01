"""Gen-1 graph construction: map sequence hashes to clusters, then build the bigraph.

Retired helper, kept here because the four archived `bigraph_*` scripts call it. Nothing
in `src/analysis/` or `src/datasets/` uses it.

What it did, and why it is gone: the Gen-1 scripts started from a pair table identified by
sequence hashes, so they needed a lookup step -- "which cluster does this sequence belong
to?" -- before a graph could be built. Live analyses now read `pairs_with_cc.parquet`
(see `src/analysis/_cc_artifacts.py`), which already stores each pair's cluster on both
sides, so the lookup step is unnecessary and they call `_bigraph.build_pair_bigraph`
directly.

The Gen-1 path also fed on `cluster_pair_weight_topk.load_pair_universe`, whose default
dedup is aa-keyed: for nt_cds that collapses each protein pair onto one arbitrary CDS
representative, 58,826 HA-NA pairs against the production 78,764. Treat any nt_cds number
these scripts produced as aa-deduped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets._bigraph import build_pair_bigraph  # noqa: E402


def build_cluster_bigraph(
    pair_universe: pd.DataFrame,
    ha_cluster_map: dict,
    na_cluster_map: dict,
    alphabet: str,
) -> tuple[nx.Graph, int]:
    """Map the pair universe onto clusters, then build the cluster-level bigraph.

    The mapping half is what this adds; the graph itself comes from the one shared builder
    (`_bigraph.build_pair_bigraph`), so it is a weighted simple `nx.Graph` -- one node per
    cluster (slot-prefixed `a:`/`b:`), edge `weight` = positive pairs on that cluster pair.

    Args:
        pair_universe: from `load_pair_universe`; one row per unique canonical pair.
        ha_cluster_map: {hash -> cluster_id} for slot-a (HA).
        na_cluster_map: {hash -> cluster_id} for slot-b (NA).
        alphabet: 'aa' (uses prot_hash_{a,b}) or 'nt_cds' (uses cds_dna_hash_{a,b}).

    Returns:
        (H, n_unmapped). H is the weighted simple bigraph; n_unmapped is the number of
        pair-universe rows dropped because either endpoint lacked a cluster assignment.
    """
    if alphabet == 'aa':
        col_a, col_b = 'prot_hash_a', 'prot_hash_b'
    elif alphabet == 'nt_cds':
        col_a, col_b = 'cds_dna_hash_a', 'cds_dna_hash_b'
    else:
        raise ValueError(f"alphabet must be 'aa' or 'nt_cds', got {alphabet!r}")

    df = pair_universe.copy()
    df['_cluster_a'] = df[col_a].map(ha_cluster_map)
    df['_cluster_b'] = df[col_b].map(na_cluster_map)
    n_unmapped = int(df[['_cluster_a', '_cluster_b']].isna().any(axis=1).sum())
    df = df.dropna(subset=['_cluster_a', '_cluster_b']).reset_index(drop=True)

    H, _edge_rows = build_pair_bigraph(df, col_a='_cluster_a', col_b='_cluster_b')
    return H, n_unmapped
