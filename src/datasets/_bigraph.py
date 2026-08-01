"""The cluster-level bigraph: one builder, shared by every consumer.

A *bigraph* here is the co-occurrence graph of one schema pair: one node per mmseqs
cluster, slot-prefixed `a:` / `b:`, and one edge per cluster pair. It is built once, by
`build_pair_bigraph`, and consumed by all three routes that used to build their own:

  - `_pair_helpers.cluster_ccs`  -- components (the 2D-CD atom).
  - `_megacc_cut.*`              -- the edge min-cut loops.
  - `_cv_sampling`               -- the CV harness's atom assignment.
  - `src/analysis/bigraph_*`     -- the diagnostics (analysis -> datasets is allowed).

**Representation: a weighted simple `nx.Graph`, edge `weight` = positive pairs.** Not a
`nx.MultiGraph`. The two carry the same information -- verified equal on aa HA-NA t095 for
pair mass (`degree(weight='weight')` == multigraph `degree`), simple degree, CC partition,
per-CC pair count (`size(weight='weight')` == multigraph `number_of_edges`), bridges, and
cut nodes -- but the multigraph is strictly lossier to work with: `nx.bridges` and
`nx.articulation_points` need a simple graph, so a multigraph consumer pays an
`nx.Graph(...)` conversion at every component. Everything the multigraph offered is a stock
networkx `weight=` argument away on the simple graph.

Sizes are therefore ALWAYS weighted: a component's pair count is
`H.subgraph(c).size(weight='weight')`, never `number_of_edges()` (which counts cluster
pairs). `_megacc_cut._piece_pairs` is the shared spelling of that.

This module is a leaf -- it imports nothing from `src` -- so every layer can depend on it
without a cycle.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_pair_bigraph(
    pos_with_ids: pd.DataFrame, *,
    col_a: str = 'cluster_id_a',
    col_b: str = 'cluster_id_b',
    ) -> tuple[nx.Graph, dict]:
    """Build the pair-weighted simple bigraph.

    One node per cluster, slot-prefixed `a:` (slot A) / `b:` (slot B); one edge per
    cluster pair, with edge `weight` = the number of positive pairs (rows) on it.

    Args:
        pos_with_ids: positive-pair rows; `col_a`/`col_b` hold the slot-A/slot-B
            cluster ids, and the row index identifies each pair. The index MUST be
            unique -- it is the pair identity a cut is translated back through
            (asserted below).
        col_a / col_b: slot-A / slot-B cluster-id column names. Pass RAW cluster ids --
            this function adds the `a:`/`b:` prefixes itself, so handing it
            already-prefixed node ids double-prefixes them.

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

    # Phase 1 - group rows by cluster pair. The 'a:'/'b:' prefixes are what make the graph
    # two-sided by construction: without them a slot-A and a slot-B cluster sharing an id string
    # would collapse into one node. Downstream (`edges_to_row_index`) reads the side off the prefix.
    slot_a_ids = ('a:' + pos_with_ids[col_a].astype(str)).to_numpy()  # slot-A node id per row ('a:'-prefixed)
    slot_b_ids = ('b:' + pos_with_ids[col_b].astype(str)).to_numpy()  # slot-B node id per row ('b:'-prefixed)
    row_idx = pos_with_ids.index.to_numpy()                           # the pos_with_ids row label per row
    edge_rows: dict[tuple, list] = {}
    for u, v, i in zip(slot_a_ids, slot_b_ids, row_idx):
        edge_rows.setdefault((u, v), []).append(i)  # group row labels by their (slot-A, slot-B) cluster pair

    # Phase 2 - nodes. Taken from `edge_rows`, so a cluster is a node only if it appears in a pair.
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

    # Phase 3 - weighted edges. nx.Graph (not MultiGraph): parallel edges collapse onto one edge
    # carrying `weight`, so the weights sum to the row count -- the invariant that makes a cut's
    # cost countable in pairs.
    for (u, v) in sorted(edge_rows):
        rows = edge_rows[(u, v)]
        edge_weight = len(rows) # positive pairs on this (u, v) cluster pair; the edge min-cut weights by it
        H.add_edge(u, v, weight=edge_weight)
    return H, edge_rows


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


def ranked_ccs(H: nx.Graph) -> list:
    """Connected components of `H`, ordered canonically by `(-pair_count, min node id)`.

    Largest-first by pair mass, ties broken on the lowest slot-prefixed node id, so the
    ordering is independent of node insertion order and of the row order that built `H`.
    `nx.connected_components` alone yields insertion order, which makes any positional
    component index (and any tie-break that falls through to it) an accident of row order.

    This is the same ordering `_pair_helpers.cluster_ccs` labels `cc_id` with.
    """
    return sorted(nx.connected_components(H),
                  key=lambda comp: (-H.subgraph(comp).size(weight='weight'), min(comp)))
