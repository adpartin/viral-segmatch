"""Mega-CC edge min-cut for the drop-budget 2D-CD router (operational home).

This module is edge-cut: shrink a mega-CC by dropping straddling pairs (cost in
pairs). The counterpart node-cut -- splitting an oversized single-side cluster
by dropping sequences -- is a separate, future concern, not handled here.

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

`apply_drop_budget_cut`'s budget loop is built from `build_pair_bigraph`,
`fragment_largest_cc`, and `edges_to_row_index`; `fragment_once` wraps those same
three into a single standalone cut (it is not used by the budget loop).

`fragment_until` is the routing-B sibling of `apply_drop_budget_cut`: the same loop
with a caller-supplied count stop (`stop_fn`, e.g. `stop_at_n_atoms`) instead of the
80/10/10 feasibility gate -- it grows the atom count for the GroupKFold CV builder.

Dependency note: src/datasets must not import src/analysis (analysis depends on
datasets), so the bisection core is duplicated here; a later cleanup can have the
analysis diagnostics import this module.
"""
from __future__ import annotations

from typing import NamedTuple

import networkx as nx
import pandas as pd
import scipy.linalg

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


def _bisect(H: nx.Graph, cut_method: str, seed: int, kl_max_iter: int = 10) -> set:
    """Bisect a connected simple bigraph into two node sets; return one side.

    'spectral' splits on the sign of the Fiedler vector (the eigenvector of the graph
    Laplacian's second-smallest eigenvalue); 'kl' uses Kernighan-Lin (node-balanced).

    Spectral determinism & cost:
        The Fiedler vector is obtained by a DIRECT dense eigensolve -- `nx.laplacian_matrix`
        on a canonical (sorted) node order, then `scipy.linalg.eigh` -- NOT by
        `nx.fiedler_vector`. networkx's default `tracemin_pcg` is an ITERATIVE solver that
        assembles its Laplacian in H's PYTHONHASHSEED-randomized node-iteration order, so its
        result is bit-reproducible WITHIN a process (hash seed fixed for the process) but varies
        ACROSS processes; empirically that made this fragmentation land on 123 vs 124 atoms with a
        different dropped set run-to-run. A dense eigensolve on a sorted-node Laplacian is
        byte-identical across processes (validated here) and at this scale also ~50x faster
        (measured: a full t095 `fragment_until` ~10.6s -> ~0.2s; the 440-node mega-CC solve
        ~3667ms -> ~15ms). Cost is the dense eigensolver's O(n^3) time / O(n^2) memory in the CC
        node count n -- a standard result, not benchmarked across sizes here; n (clusters per CC,
        <=440 at t095) is small, so it is negligible. `subset_by_index=[1, 1]` requests only the
        Fiedler eigenpair.

    Args:
        H: a connected simple bigraph -- in our use, one CC of the pair bigraph
            (edge `weight` = number of pairs).
        cut_method: 'spectral' or 'kl'.
        seed: RNG seed for the seeded KL bisection. Unused by 'spectral', which is now
            deterministic (the direct eigensolve takes no seed).
        kl_max_iter: Kernighan-Lin refinement passes (KL only).

    Returns:
        One side of the bisection as a set of nodes (the other side is the rest).
    """
    nodes = sorted(H.nodes())              # canonical order -> the eigensolve is process-reproducible
    if len(nodes) <= 2:
        return {nodes[0]}

    if cut_method == 'spectral':
        # fv = nx.fiedler_vector(H, weight='weight', seed=seed)   # replaced: nondeterministic across
        #   processes (iterative tracemin_pcg over hash-randomized node order) -- see docstring.
        # Laplacian in canonical node order; `.toarray()` is the dense n x n matrix (O(n^2) memory).
        L = nx.laplacian_matrix(H, nodelist=nodes, weight='weight').toarray().astype(float)
        # Direct symmetric eigensolve for the 2nd-smallest eigenpair only (index 1) = the Fiedler pair.
        _eigvals, eigvecs = scipy.linalg.eigh(L, subset_by_index=[1, 1])
        fv = eigvecs[:, 0]                 # the Fiedler vector, one column aligned to `nodes`
        # One side of the cut = the clusters on the Fiedler vector's negative lobe (the cut is
        # sign-invariant: the negative side and its complement drop the same straddling edges).
        A = {n for n, v in zip(nodes, fv) if v < 0}
        # Degenerate guard: if the sign split put every node on one side (A empty or all of `nodes`),
        # fall back to a balanced split at the median Fiedler value.
        if not A or len(A) == len(nodes):
            order = sorted(range(len(nodes)), key=lambda i: fv[i])   # node indices by ascending fv
            A = {nodes[i] for i in order[:len(nodes) // 2]}          # take the lower half
        return A

    if cut_method == 'kl':
        A, _ = nx.algorithms.community.kernighan_lin_bisection(
            H, weight='weight', max_iter=kl_max_iter, seed=seed
        )
        return set(A)

    raise ValueError(f"cut_method must be 'spectral' or 'kl'; got {cut_method!r}")


def _largest_cc(H: nx.Graph):
    """The node set of the connected component carrying the most pairs
    (greatest total edge weight)."""
    components = list(nx.connected_components(H))  # each CC is a set of nodes (clusters)
    # a CC's size = total edge weight inside it = the number of positive pairs it carries
    cc_sizes = [H.subgraph(c).size(weight='weight') for c in components]
    # argmax over cc_sizes: the CC with the most pairs (first index wins ties, so it is deterministic)
    best_i, best_size = 0, -1
    for i, size in enumerate(cc_sizes):
        if size > best_size:
            best_i, best_size = i, size
    return components[best_i]


class CutStep(NamedTuple):
    """One edge min-cut of a connected component (from `fragment_largest_cc`).

    Removing `cross_edges` from the graph splits `cc_nodes` into the two sides
    `part_a` and (`cc_nodes - part_a`); the cost is the straddling pairs those
    edges carry.
    """
    cc_nodes: frozenset   # the connected component that was cut (the one with the most pairs)
    part_a: frozenset     # one side of the bisection (part_b = cc_nodes - part_a)
    cross_edges: list     # cluster pairs crossing the two sides (removed to realize the cut)
    pairs_dropped: int    # straddling pairs those edges carry (sum of their edge weights)


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
            cluster ids, and the row index identifies each pair. The index MUST be
            unique -- it is the pair identity a cut is translated back through
            (asserted below).
        col_a / col_b: slot-A / slot-B cluster-id column names.

    Nodes are inserted in canonical (sorted) order, so the seeded bisection downstream is
    reproducible regardless of `pos_with_ids` row order (see the inline note below).

    Returns:
        `(H, edge_rows)`: the simple bigraph `H`, and `edge_rows` mapping each
        `(a:, b:)` edge to the `pos_with_ids` row indices it carries (so a dropped
        edge maps back to its pair rows).
    """
    # The row index IS the pair identity: `edges_to_row_index` hands these labels back and the
    # callers feed them to `.drop(index=...)` / `.loc[...]`, which would silently over-drop and
    # duplicate rows if a label repeated. Fail here rather than corrupt the split.
    assert pos_with_ids.index.is_unique, \
        'build_pair_bigraph: pos_with_ids.index must be unique (it identifies each pair).'

    # Phase 1 -- group rows by cluster pair. The 'a:'/'b:' prefixes are what make the graph
    # bipartite by construction: without them a slot-A and a slot-B cluster sharing an id string
    # would collapse into one node. Downstream (`edges_to_row_index`) reads the side off the prefix.
    slot_a_ids = ('a:' + pos_with_ids[col_a].astype(str)).to_numpy()  # slot-A node id per row ('a:'-prefixed)
    slot_b_ids = ('b:' + pos_with_ids[col_b].astype(str)).to_numpy()  # slot-B node id per row ('b:'-prefixed)
    row_idx = pos_with_ids.index.to_numpy()                           # the pos_with_ids row label per row
    edge_rows: dict[tuple, list] = {}
    for u, v, i in zip(slot_a_ids, slot_b_ids, row_idx):
        edge_rows.setdefault((u, v), []).append(i)  # group row labels by their (slot-A, slot-B) cluster pair

    # Phase 2 -- nodes. Taken from `edge_rows`, so a cluster is a node only if it appears in a pair.
    H = nx.Graph()
    # Nodes are the clusters on each side of the bigraph; each edge_rows key is one cluster pair
    # (slot-A node, slot-B node), slot-prefixed 'a:' / 'b:'.
    slot_a_nodes = {u for u, _ in edge_rows}   # slot-A clusters ('a:'-prefixed)
    slot_b_nodes = {v for _, v in edge_rows}   # slot-B clusters ('b:'-prefixed)
    all_nodes = sorted(slot_a_nodes | slot_b_nodes)
    # Build H in a canonical order -- nodes sorted, then edges sorted -- independent of pos row order.
    # nx.fiedler_vector's Laplacian assembly (node order AND sparse edge order) is order-sensitive at
    # a near-degenerate split, where a different order flips a boundary node -> a different (equally
    # valid) cut. Pinning both orders makes the edge min-cut reproducible across runs/machines
    # (PYTHONHASHSEED-independent); node order alone is not enough.
    H.add_nodes_from(all_nodes)
    # Phase 3 -- weighted edges. nx.Graph (not MultiGraph): parallel edges collapse onto one edge
    # carrying `weight`, so the weights sum to the row count -- the invariant that makes a cut's
    # cost countable in pairs.
    for (u, v) in sorted(edge_rows):
        rows = edge_rows[(u, v)]
        edge_weight = len(rows) # positive pairs on this (u, v) cluster pair; the edge min-cut weights by it
        H.add_edge(u, v, weight=edge_weight)
    return H, edge_rows


def fragment_largest_cc(H: nx.Graph, *, cut_method: str = 'spectral', seed: int = 1) -> CutStep:
    """One edge min-cut of `H`'s largest connected component (the one with the most pairs).

    `_bisect` assigns the component's clusters to two sides; the cluster pairs that
    cross the two sides are the cut. Does not mutate `H` -- the caller removes the
    returned `cross_edges`, so the same primitive serves a single cut
    (`fragment_once`) or the recursive budget loop (`apply_drop_budget_cut`).

    Args:
        H: the pair-weighted simple bigraph from `build_pair_bigraph`.
        cut_method: bisection heuristic -- 'spectral' or 'kl'.
        seed: RNG seed for the seeded bisection.

    Returns:
        A `CutStep` with the component that was cut, one bisection side, the crossing
        cluster pairs, and the count of straddling pairs they carry.
    """
    cc_nodes = _largest_cc(H)                          # node set (clusters) of the largest CC
    cc_subgraph = H.subgraph(cc_nodes)                 # that CC as an induced subgraph (its clusters + edges)
    part_a = _bisect(cc_subgraph, cut_method, seed)    # assign the CC's clusters to two sides
    # straddling edges: cluster pairs with exactly one endpoint in part_a ((u in A) != (v in A) is XOR),
    # i.e. the edges crossing the two sides -- these are the cut.
    cross_edges = [(u, v) for u, v in cc_subgraph.edges() if (u in part_a) != (v in part_a)]
    pairs_dropped = sum(cc_subgraph[u][v]['weight'] for u, v in cross_edges)  # positive pairs those edges carry
    return CutStep(frozenset(cc_nodes), frozenset(part_a), cross_edges, int(pairs_dropped))


def edges_to_row_index(cross_edges, edge_rows: dict) -> list:
    """Map crossing edges to the `pos_with_ids` row indices they carry (the dropped pairs).

    Args:
        cross_edges: cluster pairs to drop; endpoint order does not matter (each is
            canonicalized to its `(a:, b:)` key).
        edge_rows: the edge -> row-index map from `build_pair_bigraph`.

    Returns:
        The row indices to drop.
    """
    drop_idx = []
    for u, v in cross_edges:
        # edge_rows is keyed (a:, b:); a crossing edge may arrive as (b:, a:), so canonicalize
        key = (u, v) if u.startswith('a:') else (v, u)
        drop_idx.extend(edge_rows[key])  # every pos_with_ids row on this cluster pair
    return drop_idx


def fragment_once(
    pos_with_ids: pd.DataFrame, *,
    col_a: str = 'cluster_id_a',
    col_b: str = 'cluster_id_b',
    cut_method: str = 'spectral',
    seed: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, CutStep]:
    """Bisect the mega-CC once (no budget loop) and drop that cut's straddling pairs.

    A single standalone cut -- not used by `apply_drop_budget_cut`. Builds the
    bigraph, cuts the largest connected component once, and drops its straddling
    pairs. The two fragments are `step.part_a` and (`step.cc_nodes - step.part_a`);
    each may itself be several connected components after the cut, and pairs outside
    the mega-CC are untouched.

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
    step = fragment_largest_cc(H, cut_method=cut_method, seed=seed)
    drop_idx = edges_to_row_index(step.cross_edges, edge_rows)
    kept_pos = pos_with_ids.drop(index=drop_idx)   # pos_with_ids minus this cut's straddling pairs
    dropped_pos = pos_with_ids.loc[drop_idx]       # just those straddling pairs
    return kept_pos, dropped_pos, step


class FragmentState(NamedTuple):
    """State handed to a `fragment_until` stop predicate, evaluated before each cut."""
    n_atoms: int          # atoms so far: CCs with >=1 kept edge (== bipartite_components on kept rows)
    n_cuts: int           # cuts applied so far
    pairs_dropped: int    # straddling pairs dropped so far (sum of cut edge weights)
    n_total: int          # total pairs before any fragmentation


def stop_at_n_atoms(target_atoms: int):
    """A `fragment_until` `stop_fn`: stop once the graph holds >= `target_atoms` atoms.

    Routing-B's count stop -- grow the atom count to a target so the downstream GroupKFold
    CV builder (`dataset_pairs_cc.make_folds`) has enough independent atoms.
    """
    return lambda state: state.n_atoms >= target_atoms


def _live_atom_count(H: nx.Graph) -> int:
    """Number of connected components carrying >= 1 edge (i.e. >= 2 nodes).

    This is the atom count the builder actually routes. `bipartite_components` on the kept
    rows sees only clusters that appear in a kept pair, so a node stranded by a cut (all its
    edges dropped) is absent from it. Every kept edge joins an `a:` to a `b:` node, so a
    component with any kept edge has >= 2 nodes and a 1-node component is always a stranded
    node -- counting >= 2-node CCs therefore matches `bipartite_components(kept_rows)` exactly,
    whereas a raw `nx.number_connected_components` would also count the stranded singletons.
    """
    return sum(1 for c in nx.connected_components(H) if len(c) > 1)


def fragment_until(
    pos_with_ids: pd.DataFrame, *,
    col_a: str = 'cluster_id_a',
    col_b: str = 'cluster_id_b',
    cut_method: str = 'spectral',
    seed: int = 1,
    stop_fn,
    max_drop_frac: float = 1.0,
    max_cuts: int = 1000,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Repeatedly edge-min-cut the largest CC to grow the atom count, within a drop budget.

    Routing-B's L2 fragment-until loop -- the count-stop sibling of routing-A's
    `apply_drop_budget_cut`. Loops `fragment_largest_cc`, dropping each cut's straddling
    pairs, until `stop_fn(FragmentState)` is satisfied, the next cut would push the dropped
    fraction past `max_drop_frac`, or `max_cuts` is hit -- whichever comes first.
    `stop_at_n_atoms(target)` grows the atom count so the downstream GroupKFold CV builder
    has enough independent atoms. Clusters (nodes) are never split, so the cost is in pairs.

    `max_drop_frac` is the key guard: the edge-cut floor (a single dominant cluster's pair
    mass cannot be split) means a target atom count can be unreachable, and cutting past the
    floor sheds huge pair mass for almost no new atoms. The budget stops at the knee instead
    of shredding the graph -- e.g. on OOD nt_cds t095 a ~2% budget recovers ~120 atoms
    (natural 108), whereas an uncapped target of 200 drops ~100% (see the module docstring /
    glossary `edge min-cut`). It is a best-effort cap: the loop returns what it reached and
    never raises (unlike `apply_drop_budget_cut`, whose holdout infeasibility is a hard error).

    `stop_fn` is checked at the TOP of each iteration (before cutting), so an already-
    satisfied target does zero cuts; the budget is checked before each cut is applied, so the
    dropped fraction never exceeds `max_drop_frac`.

    Args:
        pos_with_ids: positive-pair rows with `col_a`/`col_b` cluster ids (its index
            identifies each pair).
        col_a / col_b: slot-A / slot-B cluster-id column names.
        cut_method: bisection heuristic -- 'spectral' or 'kl'.
        seed: RNG seed for the seeded bisection.
        stop_fn: predicate on a `FragmentState`; return True to STOP. Use
            `stop_at_n_atoms(target)` for the count stop.
        max_drop_frac: cap on the dropped-pair fraction; a cut that would exceed it is not
            applied and the loop stops (default 1.0 = no cap, i.e. `stop_fn`/`max_cuts` only).
        max_cuts: safety cap on the number of cuts.

    Returns:
        `(kept_pos, dropped_pos, audit)`: `pos_with_ids` minus the straddling pairs, just
        those straddling pairs, and an audit dict (`cut_method`, `seed`, `n_cuts`, `n_atoms`,
        `pairs_dropped`, `dropped_frac`, `max_drop_frac`, `stopped_reason`, `per_cut`).
        `stopped_reason` is one of 'stop_fn' | 'max_drop_frac' | 'max_cuts'.
    """
    n_total = int(len(pos_with_ids))
    H, edge_rows = build_pair_bigraph(pos_with_ids, col_a=col_a, col_b=col_b)

    cross_edges: list[tuple] = []   # straddling edges dropped so far, in cut order
    pairs_dropped = 0               # straddling pairs dropped so far (edge weight)
    per_cut: list[dict] = []
    cut = 0
    stopped_reason = 'max_cuts'

    while cut < max_cuts:
        n_atoms = _live_atom_count(H)
        state = FragmentState(n_atoms=n_atoms, n_cuts=cut, pairs_dropped=pairs_dropped, n_total=n_total)
        per_cut.append({
            'cut': cut,
            'n_atoms': n_atoms,
            'pairs_dropped': pairs_dropped,
            'dropped_frac': round(pairs_dropped / n_total, 6) if n_total else 0.0,
        })
        if stop_fn(state):
            stopped_reason = 'stop_fn'
            break
        step = fragment_largest_cc(H, cut_method=cut_method, seed=seed)
        if n_total and (pairs_dropped + step.pairs_dropped) / n_total > max_drop_frac:
            stopped_reason = 'max_drop_frac'   # applying this cut would break the budget -> stop
            break
        cross_edges.extend(step.cross_edges)
        pairs_dropped += step.pairs_dropped
        # drop the crossing edges -> the largest CC splits into >=2 CCs (the 2 bisection
        # sides; a side splits further if its internal links ran through the other side).
        H.remove_edges_from(step.cross_edges)
        cut += 1

    drop_idx = edges_to_row_index(cross_edges, edge_rows)
    kept_pos = pos_with_ids.drop(index=drop_idx)       # pos_with_ids minus the straddling pairs
    dropped_pos = pos_with_ids.loc[drop_idx]           # just those straddling pairs

    audit = {
        'cut_method': cut_method,
        'seed': seed,
        'n_cuts': cut,
        'n_atoms': _live_atom_count(H),
        'pairs_dropped': pairs_dropped,
        'dropped_frac': round(pairs_dropped / n_total, 6) if n_total else 0.0,
        'max_drop_frac': max_drop_frac,
        'stopped_reason': stopped_reason,
        'per_cut': per_cut,
    }
    return kept_pos, dropped_pos, audit


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

    Layer: routing-A's L2 fragment-until loop (stop = holdout 80/10/10 LPT-feasibility);
    `fragment_until` is the routing-B sibling (count stop). Both only shrink the graph --
    the actual routing (L3: `route_holdout` / `make_folds`) is the caller's -- so they live
    here with the cut primitives, not with the routers.

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
    pairs_dropped = 0               # straddling pairs dropped so far (edge weight)
    per_cut: list[dict] = []
    cut = 0

    while True:
        comps = list(nx.connected_components(H))
        cc_sizes = [int(H.subgraph(c).size(weight='weight')) for c in comps] # per-CC pair count
        retained = n_total - pairs_dropped
        largest = max(cc_sizes) if cc_sizes else 0
        drift = _lpt_max_drift(cc_sizes)
        per_cut.append({
            'cut': cut,
            'pairs_dropped': pairs_dropped,
            'dropped_frac': round(pairs_dropped / n_total, 6) if n_total else 0.0,
            'n_pieces': len(comps),  # raw CC count (may include a node stranded by a cut); cf. live n_atoms
            'largest_frac_of_retained': round(largest / retained, 6) if retained else 0.0,
            'lpt_drift': round(drift, 6),
        })
        if drift <= drift_pp:
            break
        if (pairs_dropped / n_total) > max_drop_frac or cut >= max_cuts:
            raise DropBudgetExceeded(
                f"drop-budget 2D-CD: recovering 80/10/10 needs dropping "
                f">{max_drop_frac:.0%} of pairs (reached {pairs_dropped/n_total:.1%} after "
                f"{cut} cut(s); largest CC still {largest/retained:.1%} of retained). "
                f"Options (require an explicit config change):\n"
                f"  - raise cluster_id_threshold (looser cut, smaller mega-CC),\n"
                f"  - raise split_strategy.drop_budget.max_drop_frac to accept the loss,\n"
                f"  - or use single_slot 1D-CD for this pair (no pairs dropped)."
            )
        step = fragment_largest_cc(H, cut_method=cut_method, seed=seed)
        cross_edges.extend(step.cross_edges)
        pairs_dropped += step.pairs_dropped
        # drop the crossing edges -> the largest CC splits into >=2 CCs (the 2 bisection
        # sides; a side splits further if its internal links ran through the other side).
        # The next loop's connected_components picks up the new pieces.
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
        'pairs_dropped': pairs_dropped,
        'dropped_frac': round(pairs_dropped / n_total, 6) if n_total else 0.0,
        'largest_cc_frac_before': per_cut[0]['largest_frac_of_retained'],
        'largest_cc_frac_after': per_cut[-1]['largest_frac_of_retained'],
        'lpt_drift_after': per_cut[-1]['lpt_drift'],
        'n_atoms_after': per_cut[-1]['n_pieces'],
        'per_cut': per_cut,
        'dropped_pair_keys': dropped_pair_keys,
    }
    return kept_pos, cut_audit
