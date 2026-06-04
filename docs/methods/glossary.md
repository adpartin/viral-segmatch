# Glossary

Project-specific terminology (including graph theory). This is used across docs (e.g., `splits.md`, `clusters.md`, `leakage.md`) and analysis code.

When introducing a technical or project-specific term (graph property, routing mode, split concept, metric, alphabet, etc.) in code or docs, use the term from here; if it is not yet in the glossary, add it (or suggest adding it).

---

## General graph

- **Node** — An element in a graph. In our cluster-level bigraph for a schema pair: side A = left slot clusters, side B = right slot clusters (e.g., A = HA clusters, B = NA clusters in the HA-NA case).

- **Edge** — A connection between two nodes. In our cluster-level bigraph, each unique (HA_cluster, NA_cluster) co-occurrence is an edge (sequences that co-occur in the same isolate).

- **Bigraph (bipartite graph)** — A graph with node set V = A ⊔ B (disjoint union of two sides) and edges only between A and B (never within A or within B).

- **Bipartite multigraph** — A bigraph that allows parallel edges between the same two nodes (multiple distinct co-occurrences of the same node pair contribute parallel edges). In our project: the natural representation of the cluster-level co-occurrence graph — one edge per row in the pair universe dataframe; parallel edges arise when multiple distinct sequence pairs map to the same `(cluster_a, cluster_b)` tuple. Number of edges in a multigraph = pair-universe size.

- **Simple bigraph** — A bigraph where at most one edge connects any pair of nodes (parallel edges collapsed). In our project: the **underlying simple graph** (a.k.a. simple projection) of the bipartite multigraph. Number of edges = number of unique cluster pairs. **Convention**: connectivity properties (CCs, bridges, cut nodes) are **IDENTICAL** on the multigraph and its simple projection — we compute them on the simple projection. Size properties **DIFFER**: the multigraph counts pair-universe rows (*pair-weighted* view); the simple graph counts unique cluster pairs (*cluster-pair-weighted* view).

- **Connected component (CC)** — A maximal subgraph in which every pair of nodes is path-connected. In our HA-NA bigraph: a set of HA clusters and NA clusters reachable from one another via alternating HA↔NA edges. The largest CC at t095 = the "bipartite mega-CC". CC identity is the same whether computed on the multigraph or its simple projection.

## Connectivity properties

- **Bridge** — An edge whose removal increases the number of CCs. λ(G) = 1 iff a bridge exists.

- **Edge cut** — A set of edges whose removal disconnects the graph. A bridge is an edge cut of size 1.

- **Edge connectivity λ(G)** — The minimum size of an edge cut for graph G. λ(G) = 1 means a bridge exists; higher λ means more edges must be removed to disconnect, i.e., the graph is harder to split.

- **Bipartite hub** — A node in one side whose degree is exceptionally high relative to the rest of its side, connected to a disproportionately large number of nodes on the opposite side. On Flu A the hubs are dominant-subtype clusters: HA-NA aa t095 — `NA_1880` (944 distinct HA-cluster partners, ~16% of the mega-CC's pairs); PB2-PB1 aa t095 — the conserved-collapse cluster `PB1_1122` (4,880 partners, ~79%).

## Mega-CC operations (recovering 2D-CD feasibility)

- **Mega-CC** — The single connected component that swallows most of the pair universe below t099 on Flu A (HA-NA aa: 49% of pairs at t100 → 98% at t095). Its pair-fraction sets 2D-CD feasibility: once it exceeds the train target size, bilateral routing cannot reach 80/10/10. Corpus-driven (fixed by which isolates carry which combos), not a clustering artifact.

- **Straddling pair** — A pair whose two cluster endpoints land in different splits; under a *drop-budget* it is DROPPED (a pair is kept only if both endpoints share a split). Same sense as DataSAIL's dropped interactions; the edge min-cut's cost is counted in straddling pairs.

- **Node-peel** — Shrinking the mega-CC by removing whole high-pair-mass nodes (clusters) — i.e. dropping all pairs on a cluster. Greedy node-peel is a loose upper bound on the drop cost: it discards a hub's entire pair mass without bisecting the dense core (HA-NA aa t095: 81% of pairs dropped).

- **Edge min-cut** — Fragmenting the mega-CC by removing edges (dropping straddling pairs) to bisect it into routable atoms. Two bisection heuristics used here: **KL (Kernighan–Lin) bisection** — node-balanced, via networkx; **spectral (Fiedler) bisection** — split on the sign of the Fiedler vector, unbalanced, finds sparse community boundaries. **METIS / KaHIP** are the external balanced-min-cut tools for the same job. Recursive bisection to an LPT-feasible partition is an upper bound on the true minimum drop (HA-NA aa t095: spectral reaches feasibility at 0.9% of pairs dropped).

## Project-specific

- **Schema-pair** — The ordered pair of protein functions `(slot A, slot B)` a dataset's positive pairs are drawn from — e.g., HA-NA, PB2-PB1. Determines the two sides of the bigraph. One of the 28 unordered major-protein pairs on Flu A (the 8 majors). Config: `virus.selected_functions`.

- **Cluster** — A group of sequences on one side of the bigraph; output of `mmseqs easy-linclust` at identity threshold `t` (within-side similarity graph).

- **Co-occurrence graph** — The cluster-level bigraph our splitter operates on, for a given schema pair. Nodes = clusters (per slot). Edges = co-occurrences from the pair universe. Has both a *bipartite multigraph* view (one edge per pair-universe row, with parallel edges) and a *simple bigraph* view (one edge per cluster pair, parallel edges collapsed) — see Bipartite multigraph / Simple bigraph for the connectivity-vs-size convention.

- **Pair universe** — The set of unique canonical positive pairs for a schema pair, deduped by `canonical_pair_key` under the chosen **pair_key alphabet** (`aa`: protein `seq_hash`; `nt_cds`: CDS-DNA `cds_dna_hash` — see `splits.md` §2.2), derived from isolate co-occurrence in `cds_final.parquet`. One row per unique canonical pair; it is the splitter's INPUT and the multigraph edge set of the co-occurrence graph. **Alphabet-specific**: the `aa` universe for HA-NA = 58,826 pairs; the `nt_cds` universe is larger (silent codon variants become distinct positives).

- **Cluster pair** — A unique `(cluster_a, cluster_b)` tuple after mapping each pair-universe row to its endpoints' clusters via the cluster parquet. One row per unique cluster-cluster co-occurrence. For HA-NA aa id095: 10,141 cluster pairs. This is the simple-graph edge set of the co-occurrence graph (one simple-graph edge per cluster pair).

- **Atom element (or Atom)** — The indivisible unit of a routing decision. Defined per routing mode: one pair (`random`), one unique sequence (`seq_disjoint`), one cluster (`cluster_disjoint single_slot`), or one bipartite CC (`cluster_disjoint` bilateral). Not a standard graph-theory term.

- **Cluster-disjoint partition** — A partition of atoms across splits such that no atom appears in more than one split. The atom type (CC for 2-D, cluster for 1-D) determines the disjointness regime. Three variants exist in this project: single_slot cluster_disjoint (*1D-CD*), bilateral cluster_disjoint (*2D-CD*), and test-only cluster-disjoint (*2D-CD-test*).

- **Single-slot cluster-disjoint (1D-CD)** — A 1-D cluster-disjoint partition where the cluster-disjoint constraint is enforced on ONE slot's clusters; the other slot is unconstrained. Atom = one slot's cluster. Code/config keys: `split_strategy.mode=cluster_disjoint` + `split_strategy.single_slot='a'|'b'`. Shorthand: `1D-CD` (use `1D-CD-a` / `1D-CD-b` when slot disambiguation matters).

- **Bilateral cluster-disjoint (2D-CD)** — A 2-D cluster-disjoint partition where the cluster-disjoint constraint is enforced on BOTH slots; all three splits are pairwise cluster-disjoint. Atom = bipartite CC. Code/config key: `split_strategy.mode=cluster_disjoint` (default — no `single_slot` set). Shorthand: `2D-CD`.

- **Test-only cluster-disjoint** — A **2-D (bilateral)** cluster-disjoint partition where the cluster-disjoint constraint is enforced only between train and test (not val). Atom = bipartite CC. Val is sampled from train's CC scope (shares HA/NA clusters with train). Code/config key: `cluster_disjoint_test_only`. Shorthand: `2D-CD-test`.

---

## Notes

1. **Multigraph aspect of edges**: at cluster level, the same (HA_cluster, NA_cluster) pair can arise from multiple distinct (HA_seq, NA_seq) sequence pairs. For CC computation these count as ONE edge (connectivity is unaffected by parallel edges); for size metrics (pair count per CC) each distinct sequence-pair counts separately.

2. **Co-occurrence graph naming**: I picked this term, but it's not a perfect fit (it's not strictly about "isolate co-occurrence" — it's derived from the pair universe which is itself derived from co-occurrence). Alternative names: "pair graph", "cluster-pair graph", "HA-NA bipartite". Preference?
