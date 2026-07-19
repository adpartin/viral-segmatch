"""Mega-CC edge min-cut for the drop-budget 2D-CD router (operational home).

The `src/analysis/bigraph_*.py` diagnostics explored this cut on the analysis
pair universe; this module is the operational version the splitter calls. It
works directly on the production `pos_with_ids` `(cluster_id_a, cluster_id_b)`
columns -- no analysis-side `load_pair_universe` -- so the pairs and alphabet
are exactly what the splitter routes (this also sidesteps the analysis loader's
protein-only dedup for nt_cds).

`apply_drop_budget_cut` builds the pair-weighted simple bigraph and recursively
bisects the largest connected component (spectral or KL), dropping only each
cut's straddling pairs, until the kept components LPT bin-pack into the target
ratios within `drift_pp` -- or it raises `DropBudgetExceeded` if that would need
dropping more than `max_drop_frac`. Plug-in point: `_split_helpers`
`cluster_disjoint_route_pos_df` (the bilateral 2D-CD holdout path). See
docs/plans/2026-06-04_2d_cd_drop_budget_router_plan.md.

The reusable pieces -- `build_pair_bigraph`, `fragment_largest_cc`,
`edges_to_row_index`, and `fragment_once` (a single cut, for P0/P1) -- are the
parts `apply_drop_budget_cut`'s budget loop is built from.

Dependency note: src/datasets must not import src/analysis (analysis depends on
datasets), so the bisection core is duplicated here; a later cleanup can have the
analysis diagnostics import this module.
"""
from __future__ import annotations

from typing import NamedTuple

import networkx as nx
import pandas as pd

# LPT 80/10/10 targets — must match the production bin-packer's intent
# (`_pair_helpers._lpt_bin_pack`); the feasibility gate mirrors splits.md §3.3.
_TARGETS = {'train': 0.80, 'val': 0.10, 'test': 0.10}
_BIN_ORDER = ['train', 'val', 'test']


class DropBudgetExceeded(RuntimeError):
    """Reaching an 80/10/10-feasible split would drop more than `max_drop_frac` of pairs."""


def _lpt_max_drift(sizes, targets=_TARGETS, bin_order=_BIN_ORDER) -> float:
    """Worst-split deviation from target after LPT bin-packing the atoms.

    Mirrors `_pair_helpers._lpt_bin_pack`: place each atom (largest first) into
    the split with the biggest remaining deficit, counting pairs.

    Args:
        sizes: per-atom pair counts (one connected component = one atom).
        targets: split name -> target fraction (default 80/10/10).
        bin_order: the splits, in tie-break order.

    Returns:
        The largest |achieved fraction - target fraction| over the splits
        (0.0 = an exact fit).
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
    """Bisect a connected simple bigraph into two node sets; return one side.

    'spectral' splits on the sign of the Fiedler vector (sparse, unbalanced);
    'kl' uses Kernighan-Lin (node-balanced). Both are seeded.

    Args:
        H: a connected simple bigraph (edge `weight` = number of pairs).
        method: 'spectral' or 'kl'.
        seed: RNG seed for the seeded bisection.
        kl_max_iter: Kernighan-Lin refinement passes (KL only).

    Returns:
        One side of the bisection as a set of nodes (the other side is the rest).
    """
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
    """The node set of the connected component carrying the most pairs
    (greatest total edge weight)."""
    return max(nx.connected_components(H),
               key=lambda c: H.subgraph(c).size(weight='weight'))


class CutStep(NamedTuple):
    """One edge min-cut of a connected component (from `fragment_largest_cc`).

    Removing `cross_edges` from the graph splits `cc_nodes` into the two sides
    `part_a` and (`cc_nodes - part_a`); the cost is the straddling pairs those
    edges carry.
    """
    cc_nodes: frozenset   # the connected component that was cut (the one with the most pairs)
    part_a: frozenset     # one side of the bisection (part_b = cc_nodes - part_a)
    cross_edges: list     # cluster pairs crossing the two sides (removed to realize the cut)
    dropped_pairs: int    # straddling pairs those edges carry (sum of their edge weights)


def build_pair_bigraph(
    pos_with_ids: pd.DataFrame, *,
    col_a: str = 'cluster_id_a',
    col_b: str = 'cluster_id_b',
) -> tuple[nx.Graph, dict]:
    """Build the pair-weighted simple bigraph for the edge min-cut.

    One node per cluster, slot-prefixed `a:` (slot A) / `b:` (slot B); one edge per
    cluster pair, with edge `weight` = the number of positive pairs (rows) on it.

    Args:
        pos_with_ids: positive-pair rows; `col_a`/`col_b` hold the slot-A/slot-B
            cluster ids, and the row index identifies each pair.
        col_a / col_b: slot-A / slot-B cluster-id column names.

    Returns:
        `(H, edge_rows)`: the simple bigraph `H`, and `edge_rows` mapping each
        `(a:, b:)` edge to the `pos_with_ids` row indices it carries (so a dropped
        edge maps back to its pair rows).
    """
    ca = ('a:' + pos_with_ids[col_a].astype(str)).to_numpy()
    cb = ('b:' + pos_with_ids[col_b].astype(str)).to_numpy()
    idx = pos_with_ids.index.to_numpy()
    edge_rows: dict[tuple, list] = {}
    for u, v, i in zip(ca, cb, idx):
        edge_rows.setdefault((u, v), []).append(i)
    H = nx.Graph()
    for (u, v), rows in edge_rows.items():
        H.add_edge(u, v, weight=len(rows))
    return H, edge_rows


def fragment_largest_cc(H: nx.Graph, *, method: str = 'spectral', seed: int = 1) -> CutStep:
    """One edge min-cut of `H`'s largest connected component (the one with the most pairs).

    `_bisect` assigns the component's clusters to two sides; the cluster pairs that
    cross the two sides are the cut. Does not mutate `H` -- the caller removes the
    returned `cross_edges`, so the same primitive serves a single cut
    (`fragment_once`) or the recursive budget loop (`apply_drop_budget_cut`).

    Args:
        H: the pair-weighted simple bigraph from `build_pair_bigraph`.
        method: bisection heuristic -- 'spectral' or 'kl'.
        seed: RNG seed for the seeded bisection.

    Returns:
        A `CutStep` with the component that was cut, one bisection side, the crossing
        cluster pairs, and the count of straddling pairs they carry.
    """
    big = _largest_cc(H)
    sub = H.subgraph(big)
    part_a = _bisect(sub, method, seed)
    cross = [(u, v) for u, v in sub.edges() if (u in part_a) != (v in part_a)]
    dropped = sum(sub[u][v]['weight'] for u, v in cross)
    return CutStep(frozenset(big), frozenset(part_a), cross, int(dropped))


def edges_to_row_index(cross_edges, edge_rows: dict) -> list:
    """Map crossing edges to the `pos_with_ids` row indices they carry (the dropped pairs).

    Args:
        cross_edges: cluster pairs to drop; endpoint order does not matter (each is
            canonicalized to its `(a:, b:)` key).
        edge_rows: the edge -> row-index map from `build_pair_bigraph`.

    Returns:
        The row indices to drop.
    """
    return [i for u, v in cross_edges
            for i in edge_rows[(u, v) if u.startswith('a:') else (v, u)]]


def fragment_once(
    pos_with_ids: pd.DataFrame, *,
    col_a: str = 'cluster_id_a',
    col_b: str = 'cluster_id_b',
    cut_method: str = 'spectral',
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, CutStep]:
    """Bisect the mega-CC once (no budget loop) and drop that cut's straddling pairs.

    Single-cut helper for P0/P1: build the bigraph, cut the largest connected
    component once, and drop its straddling pairs. The two fragments are
    `step.part_a` and (`step.cc_nodes - step.part_a`); each may itself be several
    connected components after the cut, and pairs outside the mega-CC are untouched.

    Args:
        pos_with_ids: positive-pair rows with `col_a`/`col_b` cluster ids.
        col_a / col_b: slot-A / slot-B cluster-id column names.
        cut_method: bisection heuristic -- 'spectral' or 'kl'.
        seed: RNG seed for the seeded bisection.

    Returns:
        `(kept_pos, dropped_pos, step)`: `pos_with_ids` minus the straddling pairs,
        just those straddling pairs, and the `CutStep`.
    """
    H, edge_rows = build_pair_bigraph(pos_with_ids, col_a=col_a, col_b=col_b)
    step = fragment_largest_cc(H, method=cut_method, seed=seed)
    drop_idx = edges_to_row_index(step.cross_edges, edge_rows)
    return pos_with_ids.drop(index=drop_idx), pos_with_ids.loc[drop_idx], step


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
    """Shrink the 2D mega-CC by edge min-cut so the kept components fit an 80/10/10 split.

    2D-CD routes each whole connected component (an atom) to one split, so a single very
    large component (the mega-CC) makes an 80/10/10 split impossible. This builds the
    pair-weighted simple bigraph and repeatedly bisects the largest component
    (`cut_method`), dropping each cut's straddling pairs, until the kept components LPT
    bin-pack into the 80/10/10 targets within `drift_pp` -- or it raises `DropBudgetExceeded`
    once dropping would exceed `max_drop_frac`. Clusters (the nodes) are never split, so the
    cost is counted in pairs.

    Args:
        pos_with_ids: positive-pair rows with `col_a`/`col_b` cluster ids (+ `pair_key_col`);
            the pairs being routed (its index identifies each pair).
        col_a / col_b: slot-a / slot-b cluster-id column names.
        pair_key_col: per-pair key column; dropped keys are recorded in the audit.
        cut_method: how to bisect the largest CC -- `'spectral'` (Fiedler vector),
            `'kl'` (Kernighan-Lin), or `'none'` (no cut; return `pos_with_ids` unchanged).
        target_frac: nominal train fraction (0.80); audit-only -- the LPT bin targets are
            the module `_TARGETS` (80/10/10).
        drift_pp: stop once the LPT pack's worst-bin deviation from target is <= this
            (a fraction; 0.05 = 5 percentage points).
        max_drop_frac: cap on the dropped-pair fraction; exceeding it raises
            `DropBudgetExceeded` instead of dropping more.
        seed: RNG seed for the seeded spectral / KL bisection.
        max_cuts: safety cap on the number of bisection iterations.

    Returns:
        `(kept_pos, cut_audit)`: `kept_pos` is `pos_with_ids` minus the dropped straddling
        pairs (the caller recomputes `component_id` on it); `cut_audit` holds the per-cut
        accounting and the dropped pair_keys.

    Raises:
        DropBudgetExceeded: if reaching 80/10/10 feasibility would need dropping more than
            `max_drop_frac` of pairs (the message lists the config knobs to relax).
    """
    n_total = int(len(pos_with_ids))
    if cut_method == 'none':
        return pos_with_ids, {'cut_method': 'none', 'pairs_dropped': 0,
                              'dropped_frac': 0.0, 'n_cuts': 0, 'per_cut': []}

    H, edge_rows = build_pair_bigraph(pos_with_ids, col_a=col_a, col_b=col_b)

    cross_edges: list[tuple] = []   # straddling edges dropped so far, in cut order
    dropped = 0                     # straddling pairs dropped so far (edge weight)
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
        step = fragment_largest_cc(H, method=cut_method, seed=seed)
        cross_edges.extend(step.cross_edges)
        dropped += step.dropped_pairs
        H.remove_edges_from(step.cross_edges)
        cut += 1

    drop_idx = edges_to_row_index(cross_edges, edge_rows)
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
