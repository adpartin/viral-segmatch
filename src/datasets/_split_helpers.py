"""Split-routing helpers that extend `_pair_helpers.py` for non-default modes.

Currently houses the cluster-disjoint routing helper (Experiment B in
`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`). The existing
`seq_disjoint_route_pos_df` in `_pair_helpers.py` is conceptually the
100%-identity special case of cluster-disjoint routing; this module
generalizes it to mmseqs2-defined clusters at lower identity thresholds.

Design parallels `seq_disjoint_route_pos_df`:
- Build a bipartite graph on (slot_a, slot_b) but with node ids = cluster ids
  (not seq_hashes). Every pair contributes one edge.
- Find connected components; each component is indivisible by construction.
- LPT-greedy bin-pack components into train/val/test by target ratios.
- Emit an audit dict mirroring `seq_disjoint_audit`'s schema.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from src.datasets._pair_helpers import bipartite_components


def load_cluster_lookup(cluster_path: Union[str, Path]) -> pd.DataFrame:
    """Load a (seq_hash -> cluster_id) lookup from a parquet file.

    Accepts the per-(function, threshold) parquet emitted by
    `protein_redundancy_per_function.py` or the per-threshold `combined_cluster.parquet`.
    Required columns: `seq_hash`, `cluster_id`. Optional: `function`, `function_short`,
    `threshold`, `cluster_rep` — left intact if present.

    Returns the DataFrame as-is; the caller is responsible for joining to pos_df.
    """
    cluster_path = Path(cluster_path)
    if not cluster_path.exists():
        raise FileNotFoundError(f"cluster lookup not found: {cluster_path}")
    df = pd.read_parquet(cluster_path)
    missing = {'seq_hash', 'cluster_id'} - set(df.columns)
    if missing:
        raise ValueError(
            f"cluster lookup at {cluster_path} is missing required columns: "
            f"{sorted(missing)}. Need at least seq_hash, cluster_id."
        )
    if df['seq_hash'].duplicated().any():
        n_dup = int(df['seq_hash'].duplicated().sum())
        raise ValueError(
            f"cluster lookup at {cluster_path} has {n_dup} duplicate seq_hash rows. "
            f"Each protein hash must map to exactly one cluster_id."
        )
    return df


def attach_cluster_ids(
    pos_df: pd.DataFrame,
    cluster_lookup: pd.DataFrame,
    pos_hash_col: str = 'seq_hash',
) -> tuple[pd.DataFrame, dict]:
    """Add `cluster_id_a` and `cluster_id_b` columns to pos_df via a join
    on the chosen hash column.

    For aa cluster_disjoint (default): join `pos_df.seq_hash_{a,b}` against
    `cluster_lookup.seq_hash`. For nt cluster_disjoint
    (Experiment B-nt), pass `pos_hash_col='cds_dna_hash'`; the join then
    consumes `pos_df.cds_dna_hash_{a,b}` (attached via
    `attach_cds_dna_hash_to_pos_df`). In both cases the cluster_lookup
    parquet itself keeps the legacy `seq_hash` column name — the values
    inside are aa md5 for `clusters/` and CDS-DNA md5 for `clusters_nt/`.

    Pairs whose pos-side hash is missing from the lookup are DROPPED.

    Returns:
        (pos_df_with_ids, audit_dict)
        pos_df_with_ids has cluster_id_a / cluster_id_b columns added.
        audit_dict has counts: n_input, n_kept, n_dropped_missing_a,
        n_dropped_missing_b, n_dropped_missing_both, and pos_hash_col.
    """
    pos_col_a = f'{pos_hash_col}_a'
    pos_col_b = f'{pos_hash_col}_b'
    if {pos_col_a, pos_col_b} - set(pos_df.columns):
        raise ValueError(
            f"attach_cluster_ids: pos_df must contain {pos_col_a} and "
            f"{pos_col_b} columns (pos_hash_col={pos_hash_col!r})."
        )
    if {'seq_hash', 'cluster_id'} - set(cluster_lookup.columns):
        raise ValueError(
            "attach_cluster_ids: cluster_lookup must contain seq_hash and cluster_id columns."
        )

    lookup = cluster_lookup[['seq_hash', 'cluster_id']].drop_duplicates(subset='seq_hash')

    out = pos_df.merge(
        lookup.rename(columns={'seq_hash': pos_col_a, 'cluster_id': 'cluster_id_a'}),
        on=pos_col_a, how='left',
    )
    out = out.merge(
        lookup.rename(columns={'seq_hash': pos_col_b, 'cluster_id': 'cluster_id_b'}),
        on=pos_col_b, how='left',
    )

    missing_a = out['cluster_id_a'].isna()
    missing_b = out['cluster_id_b'].isna()
    missing_both = missing_a & missing_b
    drop_mask = missing_a | missing_b

    audit = {
        'n_input': int(len(pos_df)),
        'n_kept': int((~drop_mask).sum()),
        'n_dropped_missing_a': int((missing_a & ~missing_both).sum()),
        'n_dropped_missing_b': int((missing_b & ~missing_both).sum()),
        'n_dropped_missing_both': int(missing_both.sum()),
        'pos_hash_col': pos_hash_col,
    }

    kept = out[~drop_mask].reset_index(drop=True)
    return kept, audit


def cluster_disjoint_route_pos_df(
    pos_df: pd.DataFrame,
    cluster_lookup: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    cluster_id_threshold: Optional[float] = None,
    cluster_lookup_path: Optional[str] = None,
    pos_hash_col: str = 'seq_hash',
    cluster_alphabet: str = 'aa',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Route `pos_df` rows into train/val/test such that no cluster spans splits.

    Algorithm (mirrors `seq_disjoint_route_pos_df`, with cluster_ids replacing
    seq_hashes as node ids):

      1. Attach cluster_id_a / cluster_id_b to pos_df via seq_hash join.
      2. Build the bipartite graph on (cluster_id_a, cluster_id_b).
      3. Compute connected components — each is indivisible (sharing a single
         cluster on either side forces both endpoints' pairs into the same
         split).
      4. LPT-greedy bin-pack components into {train, val, test} bins by target
         ratios; ties broken in train > val > test order for determinism.

    `seed` is reserved for future tie-break shuffling and is not consumed.

    Returns:
        (train_pos, val_pos, test_pos, audit) — DataFrames are row-disjoint
        partitions of the SUBSET of pos_df whose seq_hashes are covered by
        `cluster_lookup`. Rows with a missing-cluster join are dropped (counts
        recorded in audit['attach_audit']).
    """
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError(
            f"cluster_disjoint_route_pos_df: invalid ratios "
            f"train_ratio={train_ratio}, val_ratio={val_ratio}"
        )
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(
            f"cluster_disjoint_route_pos_df: train+val ratios sum to >1 "
            f"({train_ratio} + {val_ratio})"
        )

    pos_with_ids, attach_audit = attach_cluster_ids(
        pos_df, cluster_lookup, pos_hash_col=pos_hash_col,
    )

    if len(pos_with_ids) == 0:
        raise ValueError(
            "cluster_disjoint_route_pos_df: 0 rows survived the cluster join. "
            "Check that cluster_lookup covers the seq_hashes in pos_df."
        )

    component_id, cc_summary = bipartite_components(
        pos_with_ids, col_a='cluster_id_a', col_b='cluster_id_b',
    )

    sizes = component_id.value_counts().sort_index()
    sorted_comps = sorted(sizes.index, key=lambda c: (-int(sizes.loc[c]), int(c)))

    n_pairs = int(len(pos_with_ids))
    targets = {
        'train': train_ratio * n_pairs,
        'val':   val_ratio * n_pairs,
        'test':  test_ratio * n_pairs,
    }
    bin_count = {'train': 0, 'val': 0, 'test': 0}
    comp_to_split: dict = {}

    for c in sorted_comps:
        s = int(sizes.loc[c])
        order = ['train', 'val', 'test']
        deficits = {k: targets[k] - bin_count[k] for k in order}
        winner = max(order, key=lambda k: deficits[k])
        comp_to_split[c] = winner
        bin_count[winner] += s

    split_for_row = component_id.map(comp_to_split)

    train_pos = pos_with_ids[split_for_row == 'train'].reset_index(drop=True)
    val_pos   = pos_with_ids[split_for_row == 'val'].reset_index(drop=True)
    test_pos  = pos_with_ids[split_for_row == 'test'].reset_index(drop=True)

    achieved = {'train': len(train_pos), 'val': len(val_pos), 'test': len(test_pos)}

    # By-construction cluster_id overlap is 0; the audit records it as a sanity check.
    def _set(df: pd.DataFrame, col: str) -> set:
        return set(df[col].dropna()) if col in df.columns else set()
    cluster_overlaps: dict = {}
    for side in ('a', 'b'):
        col = f'cluster_id_{side}'
        sets = {sp: _set(d, col) for sp, d in
                (('train', train_pos), ('val', val_pos), ('test', test_pos))}
        cluster_overlaps[side] = {
            'train_val':  len(sets['train'] & sets['val']),
            'train_test': len(sets['train'] & sets['test']),
            'val_test':   len(sets['val'] & sets['test']),
        }

    # Secondary diagnostic: seq_hash and dna_hash overlaps. Under cluster-disjoint
    # routing at any threshold < 1.0, dna_hash and seq_hash overlap MAY be nonzero
    # because different sequences can land in the same cluster — but cluster_id
    # overlap is by construction 0.
    hash_overlaps: dict = {}
    for family in ('seq', 'dna'):
        family_overlaps: dict = {}
        for side in ('a', 'b'):
            col = f'{family}_hash_{side}'
            if col not in pos_with_ids.columns:
                continue
            sets = {sp: _set(d, col) for sp, d in
                    (('train', train_pos), ('val', val_pos), ('test', test_pos))}
            family_overlaps[side] = {
                'train_val':  len(sets['train'] & sets['val']),
                'train_test': len(sets['train'] & sets['test']),
                'val_test':   len(sets['val'] & sets['test']),
            }
        if family_overlaps:
            hash_overlaps[family] = family_overlaps

    target_pcts   = {k: 100.0 * v / n_pairs for k, v in targets.items()}
    achieved_pcts = {k: 100.0 * v / n_pairs for k, v in achieved.items()}

    audit = {
        'mode': 'cluster_disjoint',
        'algorithm': 'bipartite_cc_lpt_greedy_on_cluster_ids',
        'cluster_lookup_path': cluster_lookup_path,
        'cluster_id_threshold': cluster_id_threshold,
        'cluster_alphabet': cluster_alphabet,
        'pos_hash_col': pos_hash_col,
        'seed': int(seed),
        'attach_audit': attach_audit,
        'cc_summary': cc_summary,
        'targets_pairs': {k: int(round(v)) for k, v in targets.items()},
        'targets_pct':   {k: round(v, 4) for k, v in target_pcts.items()},
        'achieved_pairs': achieved,
        'achieved_pct':   {k: round(v, 4) for k, v in achieved_pcts.items()},
        'max_target_deviation_pct': round(
            max(abs(achieved_pcts[k] - target_pcts[k]) for k in achieved_pcts), 4
        ),
        'pairs_dropped_in_routing': 0,            # CC bin-packing never splits a component
        'pairs_dropped_in_cluster_join': attach_audit['n_input'] - attach_audit['n_kept'],
        'cluster_id_overlap': cluster_overlaps,   # by-construction zero (sanity check)
        'seq_hash_overlap': hash_overlaps.get('seq', {}),
        'dna_hash_overlap': hash_overlaps.get('dna', {}),
    }
    return train_pos, val_pos, test_pos, audit
