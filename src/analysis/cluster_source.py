"""Membership-backed source for cluster maps (the per-(protein, t) lookup).

Centralizes the `{hash: cluster_id}` lookups the bigraph/cluster analysis scripts
need onto the records-weighted membership tables (built by
`src.preprocess.build_cluster_membership`):
    data/.../cluster_membership/cluster_memb_{aa,nt_cds}.parquet
One cached table read per alphabet replaces the per-(protein, threshold) cluster-
parquet reads scattered across `src/analysis/{cluster,bigraph}_*.py`.

`cluster_map_for_root` is a drop-in for the legacy
`load_cluster_map(clusters_root, protein, threshold)`: it infers the alphabet from
`clusters_root`, returns the membership-backed map when the table is present (and
`USE_MEMBERSHIP`), and otherwise falls back to reading the cluster parquet directly
with the alphabet's CORRECT key column (`prot_hash` for aa, `cds_dna_hash` for
nt_cds). The legacy helpers hardcoded `seq_hash`, which is why their nt path failed
on the live `clusters_nt_cds` layout; both the membership path and this fallback
fix that.

Equivalence (membership map == direct parquet map for every protein × threshold ×
alphabet) is asserted by `scripts/verify_membership_swap.py`. Set
`USE_MEMBERSHIP = False` (or env `SEGMATCH_NO_MEMBERSHIP=1`) to force the legacy
parquet path.

Not swapped: `load_pair_universe` stays on `cds_final` — a protein pair_key's
`dna_hash` representative (which isolate's CDS stands for the pair) is not uniquely
recoverable from the membership table, so swapping it could silently change nt
results. Cluster maps are the clean, provable swap.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.config_hydra import load_function_metadata  # noqa: E402
from src.utils.schema import SCHEMA as _SCHEMA  # noqa: E402

USE_MEMBERSHIP = os.environ.get('SEGMATCH_NO_MEMBERSHIP', '') != '1'

_MEMB_DIR = PROJ / 'data/processed/flu/July_2025/cluster_membership'
_MEMB_FILE = {'aa': _MEMB_DIR / 'cluster_memb_aa.parquet',
              'nt_cds': _MEMB_DIR / 'cluster_memb_nt_cds.parquet'}
# Per-alphabet membership/cluster key column from the schema registry (single
# source of truth). Was a hardcoded {'aa': 'seq_hash', ...} that drifted when the
# aa hash column was renamed seq_hash -> prot_hash; the registry prevents recurrence.
_KEY = {a: s.hash_col for a, s in _SCHEMA.items()}

_memb_cache: dict[str, pd.DataFrame] = {}
_short_to_full: Optional[dict] = None


def _short_to_full_map() -> dict:
    global _short_to_full
    if _short_to_full is None:
        _short_to_full = load_function_metadata(PROJ / 'conf/virus/flu.yaml').short_to_function
    return _short_to_full


def alphabet_from_root(clusters_root) -> str:
    """Infer 'aa' or 'nt_cds' from a clusters_* path name."""
    name = Path(clusters_root).name.lower()
    if 'nt' in name:
        return 'nt_cds'
    if 'aa' in name:
        return 'aa'
    raise ValueError(f"cannot infer alphabet from clusters_root {clusters_root!r}")


def membership_available(alphabet: str) -> bool:
    return _MEMB_FILE[alphabet].exists()


def load_membership(alphabet: str) -> pd.DataFrame:
    """Cached read of the per-alphabet membership table."""
    if alphabet not in _memb_cache:
        _memb_cache[alphabet] = pd.read_parquet(_MEMB_FILE[alphabet])
    return _memb_cache[alphabet]


def cluster_map(alphabet: str, protein_short: str, threshold_id: str) -> dict:
    """Membership-backed {hash: cluster_id} for one (protein, threshold).

    Each hash maps to exactly one cluster at a given threshold, so deduping the
    records-weighted table on the key recovers the same map the cluster parquet
    holds.
    """
    m = load_membership(alphabet)
    key = _KEY[alphabet]
    full = _short_to_full_map()[protein_short]
    sub = m.loc[m['function'] == full, [key, threshold_id]].dropna()
    sub = sub.drop_duplicates(subset=key)
    return dict(zip(sub[key].values, sub[threshold_id].values))


def _parquet_cluster_map(clusters_root, protein_short: str, threshold_id: str) -> dict:
    """Legacy fallback: read the cluster parquet directly with the right key col."""
    key = _KEY[alphabet_from_root(clusters_root)]
    pq = Path(clusters_root) / threshold_id / f'{protein_short}_cluster.parquet'
    if not pq.exists():
        return {}
    df = pd.read_parquet(pq, columns=[key, 'cluster_id'])
    return dict(zip(df[key].values, df['cluster_id'].values))


def cluster_map_for_root(clusters_root, protein_short: str, threshold_id: str) -> dict:
    """Drop-in for legacy load_cluster_map(clusters_root, protein, threshold)."""
    alphabet = alphabet_from_root(clusters_root)
    if USE_MEMBERSHIP and membership_available(alphabet):
        m = load_membership(alphabet)
        if threshold_id in m.columns and protein_short in _short_to_full_map():
            return cluster_map(alphabet, protein_short, threshold_id)
    return _parquet_cluster_map(clusters_root, protein_short, threshold_id)
