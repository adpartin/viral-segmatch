"""Mega-CC edge min-cut for the drop-budget 2D-CD router (operational home).

The diagnostics in `src/analysis/bipartite_*.py` explored this cut on the
analysis pair universe; this module is the operational version that runs inside
the splitter. It operates directly on the production `pos_with_ids`'s
`(cluster_id_a, cluster_id_b)` columns — no analysis-side `load_pair_universe`
— so the pair set and alphabet are exactly what the splitter routes (this also
sidesteps the analysis loader's protein-only dedup for `nt_cds`).

`apply_drop_budget_cut()` recursively bisects the largest connected component
(spectral/KL on the pair-weighted simple bigraph), dropping only the straddling
pairs of each cut, until the kept components LPT-bin-pack into the target ratios
within `drift_pp` — or it raises `DropBudgetExceeded` (with a menu) if recovery
would exceed `max_drop_frac`. Plug-in point: between `bipartite_components` and
`route_holdout` in `_split_helpers.cluster_disjoint_route_pos_df` (bilateral
holdout path). See docs/plans/2026-06-04_2d_cd_drop_budget_router_plan.md.

dependency note: src/datasets must not import src/analysis (analysis depends on
datasets). The bisection core is therefore duplicated here; a later cleanup can
have the analysis diagnostics import this module.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

# LPT 80/10/10 targets — must match the production bin-packer's intent
# (`_pair_helpers._lpt_bin_pack`); the feasibility gate mirrors splits.md §3.3.
_TARGETS = {'train': 0.80, 'val': 0.10, 'test': 0.10}
_BIN_ORDER = ['train', 'val', 'test']


class DropBudgetExceeded(RuntimeError):
    """Recovering 80/10/10 feasibility would drop more than `max_drop_frac`."""


def _lpt_max_drift(sizes, targets=_TARGETS, bin_order=_BIN_ORDER) -> float:
    """Max |achieved - target| over bins for LPT-greedy on these atom sizes.

    Mirrors `_pair_helpers._lpt_bin_pack`: largest atom first, to the bin with
    the biggest deficit (raw counts).
    """
    total = float(sum(sizes))
    if total <= 0:
        return 1.0
    caps = {b: targets[b] * total for b in bin_order}
    filled = {b: 0.0 for b in bin_order}
    for s in sorted(sizes, reverse=True):
        w = max(bin_order, key=lambda b: caps[b] - filled[b])
        filled[w] += s
    return max(abs(filled[b] / total - targets[b]) for b in bin_order)


def _bisect(H: nx.Graph, method: str, seed: int, kl_max_iter: int = 10) -> set:
    """Balanced/sparse bisection of connected graph H; returns node set A."""
    nodes = list(H.nodes())
    if len(nodes) <= 2:
        return {nodes[0]}
    if method == 'spectral':
        fv = nx.fiedler_vector(H, weight='weight', seed=seed)
        A = {n for n, v in zip(nodes, fv) if v < 0}
        if not A or len(A) == len(nodes):  # degenerate -> median split
            order = sorted(range(len(nodes)), key=lambda i: fv[i])
            A = {nodes[i] for i in order[:len(nodes) // 2]}
        return A
    if method == 'kl':
        A, _ = nx.algorithms.community.kernighan_lin_bisection(
            H, weight='weight', max_iter=kl_max_iter, seed=seed)
        return set(A)
    raise ValueError(f"cut_method must be 'spectral', 'kl', or 'none'; got {method!r}")


def _largest_cc(H: nx.Graph):
    return max(nx.connected_components(H),
               key=lambda c: H.subgraph(c).size(weight='weight'))


def apply_drop_budget_cut(
    pos_with_ids: pd.DataFrame,
    *,
    col_a: str = 'cluster_id_a',
    col_b: str = 'cluster_id_b',
    pair_key_col: str = 'pair_key',
    cut_method: str = 'spectral',
    target_frac: float = 0.80,
    drift_pp: float = 0.05,
    max_drop_frac: float = 0.20,
    seed: int = 1,
    max_cuts: int = 1000,
) -> tuple[pd.DataFrame, dict]:
    """Edge-min-cut the mega-CC; drop straddling pairs to reach LPT feasibility.

    Builds the pair-weighted simple bigraph from `pos_with_ids`'s cluster-id
    columns (nodes `a:<cluster_id_a>` / `b:<cluster_id_b>`, edge weight = number
    of pairs on that cluster-pair), then recursively bisects the largest
    component until the kept components LPT-bin-pack into the targets within
    `drift_pp`.

    Returns `(kept_pos, cut_audit)` — `kept_pos` is `pos_with_ids` minus the
    dropped (straddling) rows; the caller recomputes `component_id` on it.
    `cut_audit` carries the per-cut accounting + the dropped pair_keys.

    Raises `DropBudgetExceeded` (with a menu) if recovery needs > `max_drop_frac`.
    """
    n_total = int(len(pos_with_ids))
    if cut_method == 'none':
        return pos_with_ids, {'cut_method': 'none', 'pairs_dropped': 0,
                              'dropped_frac': 0.0, 'n_cuts': 0, 'per_cut': []}

    # Map each cluster-pair edge -> the row indices carrying it.
    ca = ('a:' + pos_with_ids[col_a].astype(str)).to_numpy()
    cb = ('b:' + pos_with_ids[col_b].astype(str)).to_numpy()
    idx = pos_with_ids.index.to_numpy()
    edge_rows: dict[tuple, list] = {}
    for u, v, i in zip(ca, cb, idx):
        edge_rows.setdefault((u, v), []).append(i)

    H = nx.Graph()
    for (u, v), rows in edge_rows.items():
        H.add_edge(u, v, weight=len(rows))

    dropped_edges: list[tuple] = []
    dropped = 0
    per_cut: list[dict] = []
    cut = 0

    while True:
        comps = list(nx.connected_components(H))
        sizes = [int(H.subgraph(c).size(weight='weight')) for c in comps]
        retained = n_total - dropped
        largest = max(sizes) if sizes else 0
        drift = _lpt_max_drift(sizes)
        per_cut.append({
            'cut': cut,
            'pairs_dropped': dropped,
            'dropped_frac': round(dropped / n_total, 6) if n_total else 0.0,
            'n_pieces': len(comps),
            'largest_frac_of_retained': round(largest / retained, 6) if retained else 0.0,
            'lpt_drift': round(drift, 6),
        })
        if drift <= drift_pp:
            break
        if (dropped / n_total) > max_drop_frac or cut >= max_cuts:
            raise DropBudgetExceeded(
                f"drop-budget 2D-CD: recovering 80/10/10 needs dropping "
                f">{max_drop_frac:.0%} of pairs (reached {dropped/n_total:.1%} after "
                f"{cut} cut(s); largest CC still {largest/retained:.1%} of retained). "
                f"Options (require an explicit config change):\n"
                f"  - raise cluster_id_threshold (looser cut, smaller mega-CC),\n"
                f"  - raise split_strategy.drop_budget.max_drop_frac to accept the loss,\n"
                f"  - or use single_slot 1D-CD for this pair (no pairs dropped)."
            )
        big = _largest_cc(H)
        sub = H.subgraph(big)
        A = _bisect(sub, cut_method, seed)
        cross = [(u, v) for u, v in sub.edges() if (u in A) != (v in A)]
        for (u, v) in cross:
            key = (u, v) if u.startswith('a:') else (v, u)
            dropped += H[u][v]['weight']
            dropped_edges.append(key)
        H.remove_edges_from(cross)
        cut += 1

    drop_idx = [i for key in dropped_edges for i in edge_rows[key]]
    kept_pos = pos_with_ids.drop(index=drop_idx)
    dropped_pair_keys = (
        pos_with_ids.loc[drop_idx, pair_key_col].tolist()
        if pair_key_col in pos_with_ids.columns else []
    )

    cut_audit = {
        'cut_method': cut_method,
        'seed': seed,
        'target_frac': target_frac,
        'drift_pp': drift_pp,
        'max_drop_frac': max_drop_frac,
        'n_cuts': cut,
        'pairs_dropped': dropped,
        'dropped_frac': round(dropped / n_total, 6) if n_total else 0.0,
        'largest_cc_frac_before': per_cut[0]['largest_frac_of_retained'],
        'largest_cc_frac_after': per_cut[-1]['largest_frac_of_retained'],
        'lpt_drift_after': per_cut[-1]['lpt_drift'],
        'n_atoms_after': per_cut[-1]['n_pieces'],
        'per_cut': per_cut,
        'dropped_pair_keys': dropped_pair_keys,
    }
    return kept_pos, cut_audit
