# Glossary

Graph-theory and project-specific terminology used across the splits / clusters / leakage methods docs and across analysis code.

When introducing a graph property in code or docs, use the term from here; if it is not yet in the glossary, add it.

---

## General graph

- **Node** — An element in a graph. In our cluster-level bipartite graph for a schema pair: side A = left slot clusters, side B = right slot clusters (e.g., A = HA clusters, B = NA clusters in the HA-NA case).

- **Edge** — A connection between two nodes. In our cluster-level bipartite graph, each unique (HA_cluster, NA_cluster) co-occurrence is an edge.

- **Bipartite graph** — A graph with node set V = A ⊔ B (disjoint union of two sides) and edges only between A and B (never within A or within B).

- **Connected component (CC)** — A maximal subgraph in which every pair of nodes is path-connected. In our HA-NA bipartite graph: a set of HA clusters and NA clusters reachable from one another via alternating HA↔NA edges. The largest CC at t095 = the "bipartite mega-component".

## Connectivity properties

- **Bridge** — An edge whose removal increases the number of CCs. λ(G) = 1 iff a bridge exists.

- **Edge cut** — A set of edges whose removal disconnects the graph. A bridge is an edge cut of size 1.

- **Edge connectivity λ(G)** — The minimum size of an edge cut for graph G. λ(G) = 1 means a bridge exists; higher λ means more edges must be removed to disconnect, i.e., the graph is harder to split.

- **Bipartite hub** — A node in one side whose degree is exceptionally high relative to the rest of its side, connected to a disproportionately large number of nodes on the opposite side. *(Examples to be added once you've fully understood the structure.)*

## Project-specific

- **Cluster (project)** — A group of sequences on one side of the bipartite graph; output of `mmseqs easy-linclust` at identity threshold t (within-side similarity graph).

- **Co-occurrence graph (project)** — The bipartite graph our splitter operates on. Nodes are clusters (per slot); edges are (HA_cluster, NA_cluster) tuples derived from the pair universe (post-`pair_key` dedup).

- **Atom (project)** — The indivisible unit of a routing decision. Defined per routing mode: one pair (`random`), one unique sequence (`seq_disjoint`), one cluster (`cluster_disjoint single_slot`), or one bipartite CC (`cluster_disjoint` bilateral). Not a standard graph-theory term.

- **Cluster-disjoint partition** — A partition of atoms across splits such that no atom appears in more than one split. The atom type (CC for 2-D, cluster for 1-D) determines the disjointness regime.

- **Test-only cluster-disjoint** — A **2-D (bilateral)** cluster-disjoint partition where the cluster-disjoint constraint is enforced only between train and test (not val). Atom = bipartite CC. Val is sampled from train's CC scope (shares HA/NA clusters with train). Code/config key: `cluster_disjoint_test_only`. Tight abbreviation, tables only: `test-CD`.

---

## Notes (TODO: need to confirm)

1. **Multigraph aspect of edges**: at cluster level, the same (HA_cluster, NA_cluster) pair can arise from multiple distinct (HA_seq, NA_seq) sequence pairs. For CC computation these count as ONE edge (connectivity is unaffected by parallel edges); for size metrics (pair count per CC) each distinct sequence-pair counts separately. Should I add a one-liner "Multigraph" entry, or fold it into the Edge entry as a clarifying sentence? Currently it's not mentioned.

2. **Co-occurrence graph naming**: I picked this term, but it's not a perfect fit (it's not strictly about "isolate co-occurrence" — it's derived from the pair universe which is itself derived from co-occurrence). Alternative names: "pair graph", "cluster-pair graph", "HA-NA bipartite". Preference?

3. **Order**: I put Node first under General graph for the standard textbook ordering. If you'd rather lead with the project-anchored concept (e.g., open with "Cluster" → "Co-occurrence graph" → then standard graph terms), I'd reorder.
