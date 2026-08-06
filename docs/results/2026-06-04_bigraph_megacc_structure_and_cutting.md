# Bigraph structure of the Flu A mega-CC, and cutting it for 2D-CD feasibility

**Date.** 2026-06-04.
**Scope.** HA-NA and PB2-PB1 schema pairs, `aa` alphabet, bilateral cluster-disjoint (**2D-CD**). One spectral cut realization (seed=1). Findings are on the canonical `aa` pair universe (HA-NA = 58,826 pairs; PB2-PB1 = 53,078).
**Chain.** `leakage.md` → `clusters.md` → `splits.md`; this doc extends `splits.md` §1.7. Terms: `glossary.md`.
**Verification tags.** **[V]** = measured this session. **[H]** = hypothesis / interpretation not fully verified. **[C]** = cited from prior work.

---

## 0. Summary

- **[V]** The Flu A HA-NA cluster-level bigraph collapses into a single **mega-CC** below t099 (49% of pairs at t100 → 98% at t095). This is **corpus-driven**: the threshold `t` controls cluster (node) granularity, but the edge set is fixed by which isolates carry which combos.
- **[V]** The mega-CC is **hub-driven**, not a uniform knot: on HA-NA the **NA (N-subtype) side dominates** (pair-mass Gini 0.95 at t095); it is dense subtype-communities hung on a skeleton of bridges (76% of cluster pairs are bridges).
- **[V]** To recover **2D-CD** feasibility below t099 you must drop pairs. **Node removal is the wrong operation** (HA-NA t095: 81% of pairs) — the efficient operation is an **edge min-cut** that drops only straddling pairs: HA-NA t095 reaches a feasible 80/10/10 at **0.9%** dropped (spectral) — ~90× cheaper than node-peel.
- **[V]** Cuttability is **mechanism-dependent**: where the obstacle is **antigenic community** structure (HA-NA) the cut is cheap; where it is a **conserved protein** collapsing to one mega-cluster (PB2-PB1 at low t) it is expensive (≥24% even with the better method).
- **[V]** The cheap cut is a **cross-subtype split**: the resulting atoms are subtype-organized (H3N2 99% / H1N1 93% pure; avian subtypes pooled), so it tests on subtypes the model did not train on.
- **[V/H]** The severed bridges are **2.3× enriched** for reassortant-type isolates (proxy below) — reassortment is a verified *contributing* factor to the thin inter-subtype boundaries, **[H]** not the sole mechanism.

---

## 1. The model and the central principle

For a schema pair, 2D-CD builds one **bigraph** (`glossary.md`): side A = slot-A clusters (HA), side B = slot-B clusters (NA), with one edge per pair in the pair universe. The routing atom under 2D-CD is a **connected component (CC)** — every pair in a CC must go to one split — so 2D-CD feasibility reduces to one number: the largest CC's share of pairs (it must fit the train target).

> **The central principle. `t` controls node granularity; biology controls the edge set. You hold the threshold; you do not hold the wiring.**

- **Node granularity** = how finely sequences are split into clusters (cluster count / sizes). It is set by `t` through the per-protein **residue-mismatch budget** `≈ (1−t)·L` (`clusters.md` §5); the resulting cluster counts per (protein, alphabet, `t`) are in `clusters.md` §6.1. Looser `t` → larger budget → fewer, coarser clusters → coarser nodes.
- **The edge set** (which A-cluster co-occurs with which B-cluster) is fixed by the corpus — which isolates carry which (HA, NA) combination — and `t` cannot change it.
- **[V] Consequence — node coarsening is monotone in connectivity.** Lowering `t` only merges clusters, and merging nodes can only merge components, never split them. So the largest-CC fraction can only *rise* as `t` drops; no threshold choice fragments a mega-CC once formed. (First-order; `easy-linclust`'s single-pass approximation adds small non-monotone noise at 1-pp steps — `clusters.md` §6.1.)

**[V/C] The mega-CC is corpus-driven, present even at maximum granularity.** At t100 — one cluster per unique sequence, no coarsening — HA-NA already has a single component holding **49%** of all pairs; lowering `t` grows it (t099 79.6%, t098 88.4%, t095 97.8% — feasibility table `splits.md` §1.7.2, cross-checked this session). Switching alphabet or tightening coverage does not help, because the bipartite *wiring* is set by Flu A's dominant HxNy × host × year cells.

---

## 2. Structure of the mega-CC [V]

Measured with `src/analysis/bigraph_properties.py` (per-node degree, pair-mass, bridges, cut nodes; pair-mass = multigraph degree = pairs dropped if the node is removed).

**HA-NA aa t095 — NA is the engine.** Largest CC = 57,526 pairs (97.8%), 8,710 nodes.

| side | top-1 cluster share | top-5 share | pair-mass Gini | max degree |
|---|---:|---:|---:|---:|
| HA | 13.8% | 40.1% | 0.85 | 311 |
| **NA** | **15.8%** | **46.2%** | **0.95** | **1,279** |

- Only **5.6%** of nodes (489) are cut nodes, but they carry the mass: the 236 NA cut nodes hold **90.6%** of the component's pairs, the 253 HA cut nodes **65.9%**. **76%** of the 10,141 unique cluster pairs are bridges, of which only ~8% touch a top-10 hub — i.e. a sparse hub skeleton with dense communities and peripheral spokes.
- The dominance is antigenic and present at every threshold: even at t100 one NA cluster already connects to **3,082** distinct HA clusters; NA pair-mass Gini rises 0.81 (t100) → 0.87 (t099) → 0.95 (t095). Concrete hubs: `NA_1880` (944 partners, 9,115 pairs), `HA_6938` (311, 7,926), `NA_687` (1,279 partners).

**PB2-PB1 aa — a different, symmetric mechanism.** Universe 53,078 pairs.
- **t099: symmetric** — neither slot dominates (PB2 Gini 0.73 / maxdeg 1,360; PB1 0.76 / 1,758). Unlike HA-NA, there is no antigenic hub side.
- **t095: a single conserved-collapse hub** — one PB1 cluster `PB1_1122` holds **78.8%** of the mega-CC's pairs (41,632 pairs, 4,880 partners), driven by PB1 collapsing to one cluster at t095 (`clusters.md` §6.3), *not* by antigenic structure.

**Reading:** the mega-CC is dense subtype/lineage communities connected by a thin bridge skeleton; *which* structure produces the hubs differs by pair (antigenic subtype for HA-NA; conserved-protein collapse for PB2-PB1).

---

## 3. Breaking the mega-CC for 2D-CD: methods and cost [V]

Below t099, 2D-CD is infeasible (largest CC > train target). Recovering it requires dropping pairs (a **drop-budget**, DataSAIL-S2 style). We evaluated two families; feasibility is gated by the real LPT 80/10/10 check (drift ≤ 5 pp, `splits.md` §1.3/§3.3).

**Methods.**
- **Node-peel** (`bigraph_hub_peel.py`) — greedily remove the heaviest cluster (node), dropping all its pairs.
- **Edge min-cut** (`bigraph_min_cut.py`) — recursively bisect the largest CC and drop only **straddling pairs**, via **KL (Kernighan–Lin, *not* Kullback–Leibler) balanced bisection** or **spectral (Fiedler) bisection** (networkx, in-env). METIS/KaHIP are the external balanced-min-cut tools for the same job. Recursive bisection to LPT-feasibility is an **upper bound** on the true minimum drop.

**Cost to recover 80/10/10 (% of the pair universe dropped):**

| schema pair | t | node-peel | KL edge-cut | spectral edge-cut |
|---|---|---:|---:|---:|
| HA-NA | t099 | — | 0% (largest 79.6%, already feasible) | 0% |
| HA-NA | t098 | — | 3.1% | — |
| HA-NA | **t095** | **81.4%** | 10.1% | **0.9%** |
| PB2-PB1 | t099 | 17.7% (1 hub ≈ audit's 9,371 pairs) | 0% (largest 81% within tolerance) | — |
| PB2-PB1 | **t095** | 93.9% | 60.0% | 23.5% |

**Findings.**
- **Node-peel is a loose upper bound.** On HA-NA t095 it drops 81% — removing the single biggest hub (`NA_1880`, 15.5% of pairs) moves the largest-CC fraction only 97.8% → 97.4%, because the dense core is robust to node removal.
- **The edge min-cut is the efficient lever**, and the balance constraint matters: spectral (unbalanced, finds sparse community boundaries) reaches feasibility on HA-NA t095 at **0.9%** (502 pairs, 7 cuts), ~10× better than KL's node-balanced 10.1% and ~90× better than node-peel. Cutting **7 pairs** alone splits a 21k-pair community off the mega — these are the *inter-community* bridges that `bigraph_properties` had already counted (76% bridges).
- **Cuttability is mechanism-dependent.** HA-NA's antigenic community structure cuts cheaply; PB2-PB1's single conserved-collapse mega-cluster does not (spectral still needs 23.5%, KL 60%) — there is no thin boundary to exploit when one node holds 79% of the pairs.

This makes **2D-CD recoverable below t099 for community-structured pairs at a small, quantified data cost** — the operation the prior `splits.md` feasibility ceiling treated as simply "broken."

---

## 4. What the cut is, biologically

Annotated with `bigraph_cut_subtype.py` (atom → `hn_subtype`) and `bigraph_reassort_check.py` (isolate-level), on the HA-NA t095 spectral cut (drop 502 pairs, 397 atoms).

**[V] The atoms are subtype-organized — so the cheap cut is a cross-subtype split.**

| atom | pairs | composition |
|---|---:|---|
| 0 | 27.7% | avian mix (H5N1, H3N8, H4N6, H5N2) |
| 1 | 25.7% | **H3N2 (99.3% pure)** |
| 2 | 25.0% | **H1N1 (92.8% pure)** |
| 3 | 11.4% | human reassortant mix (H3N2/H1N1/H1N2) |

Human seasonal lineages separate cleanly; avian subtypes pool into one densely-reassorting community (they share a reservoir). Because LPT routes the big atoms to train and the rest to val/test, **train and test land on different subtypes** — the strongest cluster-disjoint generalization test, but a test set skewed to the rare-subtype tail rather than representative circulating flu.

**[V/H] Reassortment is a verified contributing factor — not the whole story.** *We have no explicit reassortment label in the metadata.* We define a **reassortant pair** operationally: decompose the isolate's `hn_subtype` into its H-lineage and N-lineage, and call the pair reassortant when its NA is not the HA-lineage's corpus-modal N-partner **and** its HA is not the NA-lineage's modal H-partner (off both diagonals; e.g. H5N2, H1N2). Under this proxy:

- **[V]** dropped (bridge) pairs are **36.3%** reassortant vs **15.5%** for kept pairs — a **2.3× enrichment** — and the classic reassortant subtypes (H5N8, H5N6, H1N2, H6N8, H5N2) concentrate in the cut.
- **[H]** So reassortment thins the inter-subtype boundaries, but most severed bridges (~64%) are canonical-subtype boundary co-occurrences, so reassortment is *contributing*, not dominant.
- **Proxy caveat:** being corpus-relative, it misses reassortant-origin subtypes that became dominant (e.g. H7N9 counts as canonical here) — so 2.3× is a conservative lower bound. A phylogenetic (segment-tree) definition is out of scope.

---

## 5. Implications

- **[H] 2D-CD is the high-value target.** 1D-CD (single-slot) is the trivial relaxation; the contribution is making 2D-CD feasible at low `t` by cutting the mega-CC. This is general beyond flu segment-matching — any paired-entity task with a cluster-level bigraph and a giant-component obstacle (e.g. drug–target interaction, PPI). Demonstrating it across datasets would be a method contribution, not just an application.
- **[V→H] Cuttability as a pre-screen for which schema pairs are 2D-CD-tractable.** The bigraph analysis explains *why* some pairs (HA-NA) admit cheap 2D-CD and others (PB2-PB1 at low `t`) do not — a principled answer to "why weren't all 28 major-protein pairs trained under 2D-CD."
- **[H] Hubs → subtype connects structure to performance.** The hub-side = dominant subtype mapping is the likely explanation for the random-vs-CD performance gap (random splits leak via shared-subtype near-neighbors; CD/cross-subtype splits remove that shortcut). Confirming this requires the train-and-score experiment (below).

**[H] Relation to DataSAIL (preliminary; a real comparison needs a deep dive, out of scope here).** DataSAIL solves a cluster-then-ILP minimizing cross-fold similarity and drops straddling pairs (I2/S2). Two possible differentiators of the bigraph approach: (1) it *explains* the infeasibility structurally (giant component / community structure) and yields an interpretable cut that recovers the biology (subtype boundaries, reassortant bridges) rather than a black-box objective; (2) DataSAIL's ILP **collapsed to a partition-shape constant on this corpus** (`splits.md` §4.2, `2026-05-24_datasail_lpi_results.md`), where the bigraph min-cut produced a usable split. Whether this is a publishable *general* method — vs a Flu-A-specific recovery — is unproven and would require benchmarking against DataSAIL across multiple paired-entity datasets.

---

## 6. Caveats and scope

- **Proposed mode, not implemented.** The drop-budget edge-cut quantifies a routing mode the current Stage-3 router does not have — it never drops pairs (`splits.md` §1.3). A 2D-CD-with-drop-budget router must be built before this is usable end-to-end. Beyond LPT (biggest-CC→train, smallest→test), that router likely wants alternative atom-routing that distributes the post-cut CCs more evenly across train/val/test.
- **Generality untested.** Results are HA-NA and PB2-PB1, `aa` only, one spectral seed. The cut cost is a greedy upper bound, not the proven minimum (METIS/KaHIP or an exact balanced min-cut would tighten it).
- **Reassortment is a metadata-derived proxy**, not a labeled ground truth (§4).

---

## 7. Reproduction

Environment: `segmatch` conda env (pandas, networkx, scipy). Inputs on disk: `data/processed/flu/July_2025/{cds_final.parquet, clusters_aa/tXXX/<PROT>_cluster.parquet}`, `data/processed/flu/metadata_eda/flu_genomes_metadata_parsed.csv`.

```bash
# Structure: per-node degree, pair-mass, bridges, cut nodes, hub concentration
python -m src.analysis.bigraph_properties --schema_pair HA NA --alphabets aa \
    --thresholds t100 t099 t095 --out_dir results/flu/July_2025/runs/bigraph_properties/hub_confirm

# Node-peel (loose upper bound) and edge min-cut (efficient)
python -m src.analysis.bigraph_hub_peel  --schema_pair HA NA --alphabet aa --threshold t095 --strategy cut_node
python -m src.analysis.bigraph_min_cut   --schema_pair HA NA --alphabet aa --threshold t095 --method spectral
python -m src.analysis.bigraph_min_cut   --schema_pair PB2 PB1 --alphabet aa --threshold t095 --method spectral

# Biology: subtype composition of the cut + isolate-level reassortment proxy
python -m src.analysis.bigraph_cut_subtype        --schema_pair HA NA --alphabet aa --threshold t095 --method spectral
python -m src.analysis.bigraph_reassort_check --schema_pair HA NA --alphabet aa --threshold t095 --method spectral
```

Outputs under `results/flu/July_2025/runs/{bigraph_properties,bigraph_hub_peel,bigraph_min_cut}/`.

## 8. See also

- `docs/methods/glossary.md` — mega-CC, straddling pair, node-peel, edge min-cut, bipartite hub, pair universe.
- `docs/methods/clusters.md` §5–6 — residue-mismatch budget and cluster-count EDA (the granularity knob).
- `docs/methods/splits.md` §1.7 — bilateral feasibility table; §4 — DataSAIL / P&M cross-reference.
- `docs/results/2026-05-21_bicc_pair_drop_audit.md` — the prior by-hand drop estimates this work computes directly.
