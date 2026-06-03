# Glossary

Project-specific terminology (including graph theory). This is used across docs (e.g., `splits.md`, `clusters.md`, `leakage.md`) and analysis code.

When introducing a technical or project-specific term (graph property, routing mode, split concept, metric, alphabet, etc.) in code or docs, use the term from here; if it is not yet in the glossary, add it (or suggest adding it).

---

## General graph

- **Node** — An element in a graph. In our cluster-level bipartite graph for a schema pair: side A = left slot clusters, side B = right slot clusters (e.g., A = HA clusters, B = NA clusters in the HA-NA case).

- **Edge** — A connection between two nodes. In our cluster-level bipartite graph, each unique (HA_cluster, NA_cluster) co-occurrence is an edge.

- **Bipartite graph** — A graph with node set V = A ⊔ B (disjoint union of two sides) and edges only between A and B (never within A or within B).

- **Bipartite multigraph** — A bipartite graph that allows parallel edges between the same two nodes (multiple distinct cooccurrences of the same node pair contribute parallel edges). In our project: the natural representation of the cluster-level co-occurrence graph — one edge per row in the pair universe; parallel edges arise when multiple distinct sequence pairs map to the same `(cluster_a, cluster_b)` tuple. Number of edges = pair-universe size.

- **Simple bipartite graph** — A bipartite graph where at most one edge connects any pair of nodes (parallel edges collapsed). In our project: the simple projection of the bipartite multigraph. Number of edges = number of unique cluster pairs. **Convention**: connectivity properties (CCs, bridges, cut nodes) are IDENTICAL on the multigraph and its simple projection — we compute them on the simple projection. Size properties DIFFER: the multigraph counts pair-universe rows (pair-weighted view); the simple graph counts unique cluster pairs (cluster-pair-weighted view).

- **Connected component (CC)** — A maximal subgraph in which every pair of nodes is path-connected. In our HA-NA bipartite graph: a set of HA clusters and NA clusters reachable from one another via alternating HA↔NA edges. The largest CC at t095 = the "bipartite mega-component". CC identity is the same whether computed on the multigraph or its simple projection.

## Connectivity properties

- **Bridge** — An edge whose removal increases the number of CCs. λ(G) = 1 iff a bridge exists.

- **Edge cut** — A set of edges whose removal disconnects the graph. A bridge is an edge cut of size 1.

- **Edge connectivity λ(G)** — The minimum size of an edge cut for graph G. λ(G) = 1 means a bridge exists; higher λ means more edges must be removed to disconnect, i.e., the graph is harder to split.

- **Bipartite hub** — A node in one side whose degree is exceptionally high relative to the rest of its side, connected to a disproportionately large number of nodes on the opposite side. *(Examples to be added once you've fully understood the structure.)*

## Project-specific

- **Cluster (project)** — A group of sequences on one side of the bipartite graph; output of `mmseqs easy-linclust` at identity threshold t (within-side similarity graph).

- **Co-occurrence graph (project)** — The cluster-level bipartite graph our splitter operates on, for a given schema pair. Nodes = clusters (per slot). Edges = cooccurrences from the pair universe. Has both a multigraph view (one edge per pair-universe row, with parallel edges) and a simple view (one edge per cluster pair, parallel edges collapsed) — see Bipartite multigraph / Simple bipartite graph for the connectivity-vs-size convention.

- **Pair universe (project)** — The set of unique canonical positive pairs for a schema pair, after `canonical_pair_key(seq_hash_a, seq_hash_b)` dedup, derived from raw isolate cooccurrence in `cds_final.parquet`. One row per unique canonical protein pair. For HA-NA: 58,826 pairs. This is the splitter's INPUT and the multigraph edge set of the co-occurrence graph (one multigraph edge per pair-universe row).

- **Cluster pair (project)** — A unique `(cluster_a, cluster_b)` tuple after mapping each pair-universe row to its endpoints' clusters via the cluster parquet. One row per unique cluster-cluster cooccurrence. For HA-NA aa id095: 10,141 cluster pairs. This is the simple-graph edge set of the co-occurrence graph (one simple-graph edge per cluster pair).

- **Atom (project)** — The indivisible unit of a routing decision. Defined per routing mode: one pair (`random`), one unique sequence (`seq_disjoint`), one cluster (`cluster_disjoint single_slot`), or one bipartite CC (`cluster_disjoint` bilateral). Not a standard graph-theory term.

- **Cluster-disjoint partition** — A partition of atoms across splits such that no atom appears in more than one split. The atom type (CC for 2-D, cluster for 1-D) determines the disjointness regime. Three variants exist in this project: single_slot cluster_disjoint (1D-CD), bilateral cluster_disjoint (2D-CD), and test-only cluster-disjoint (2D-CD-test) — defined below.

- **single_slot cluster_disjoint (1D-CD)** — A 1-D cluster-disjoint partition where the cluster-disjoint constraint is enforced on ONE slot's clusters; the other slot is unconstrained. Atom = one slot's cluster. Code/config keys: `split_strategy.mode=cluster_disjoint` + `split_strategy.single_slot='a'|'b'`. Shorthand: `1D-CD` (use `1D-CD-a` / `1D-CD-b` when slot disambiguation matters).

- **bilateral cluster_disjoint (2D-CD)** — A 2-D cluster-disjoint partition where the cluster-disjoint constraint is enforced on BOTH slots; all three splits are pairwise cluster-disjoint. Atom = bipartite CC. Code/config key: `split_strategy.mode=cluster_disjoint` (default — no `single_slot` set). Shorthand: `2D-CD`.

- **Test-only cluster-disjoint** — A **2-D (bilateral)** cluster-disjoint partition where the cluster-disjoint constraint is enforced only between train and test (not val). Atom = bipartite CC. Val is sampled from train's CC scope (shares HA/NA clusters with train). Code/config key: `cluster_disjoint_test_only`. Shorthand: `2D-CD-test`.

---

## Notes (TODO: need to confirm)

1. **Multigraph aspect of edges**: at cluster level, the same (HA_cluster, NA_cluster) pair can arise from multiple distinct (HA_seq, NA_seq) sequence pairs. For CC computation these count as ONE edge (connectivity is unaffected by parallel edges); for size metrics (pair count per CC) each distinct sequence-pair counts separately. Should I add a one-liner "Multigraph" entry, or fold it into the Edge entry as a clarifying sentence? Currently it's not mentioned.

2. **Co-occurrence graph naming**: I picked this term, but it's not a perfect fit (it's not strictly about "isolate co-occurrence" — it's derived from the pair universe which is itself derived from co-occurrence). Alternative names: "pair graph", "cluster-pair graph", "HA-NA bipartite". Preference?

3. **Order**: I put Node first under General graph for the standard textbook ordering. If you'd rather lead with the project-anchored concept (e.g., open with "Cluster" → "Co-occurrence graph" → then standard graph terms), I'd reorder.
