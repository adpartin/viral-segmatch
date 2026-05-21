# Routing-geometry visualization — results (2026-05-20)

Visual comparison of how the train / val / test partition separates in
k-mer feature space as the Stage 3 split routing moves from `random` →
`seq_disjoint` → `cluster_disjoint`. Companion to
`docs/plans/2026-05-20_routing_geometry_viz_plan.md` (BACKLOG.md
**Methodology #2**).

Inspired by **DataSAIL Fig. 4** (Joeres et al., 2025), which compared
random vs S1-disjoint splits on a 2-input drug-protein task using
t-SNE on ECFP fingerprints. Our 2-input analog is HA × NA pairs;
single-entity 2-D embeddings use k-mer (nt, k=6) feature vectors
projected with TruncatedSVD + UMAP.

---

## Run inventory

All three runs share `flu_ha_na.yaml` as the parent config (full
unfiltered Flu A, neg:pos = 1.5, k-mer nt k=6). The only knob that
varies is `dataset.split_strategy.mode`.

| Routing | Bundle | Run dir |
|---|---|---|
| `random` | `flu_ha_na_random` | `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_random_20260520_210647/` |
| `seq_disjoint` | `flu_ha_na_seq_disjoint` | `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_seq_disjoint_20260520_211109/` |
| `cluster_disjoint @ aa id 0.99` | `flu_ha_na_cluster_id99` | `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_20260520_211534/` |

All three produce identical split sizes (Stage 3 is deterministic
under fixed `master_seed`):

| Split | Pairs | Positives | Negatives | neg:pos |
|---|---:|---:|---:|---:|
| Train | 116,775 | 46,710 | 70,065 | 1.50 |
| Val | 14,597 | 5,839 | 8,758 | 1.50 |
| Test | 14,597 | 5,839 | 8,758 | 1.50 |

---

## Quantitative finding: sequence-leakage counts

The sequence-level plot counts how many unique protein sequences
appear in more than one split — the strict leakage definition for
sequence-disjoint partitioning. Capped at 10,000 unique sequences per
run (5,000 per protein type), stratified to preserve the leakage
minority:

| Routing | In-multiple-splits | Interpretation |
|---|---:|---|
| `random` | **1,000 / 10,000** | Expected: random shuffling lets a sequence land in train + test. The 1000 figure is bounded by the stratified-sampling floor (`LEAKAGE_FLOOR=500` per protein type). |
| `seq_disjoint` | **0 / 10,000** | Zero by construction. Sequence-disjoint partitioning ensures each `seq_hash` belongs to exactly one split. |
| `cluster_disjoint` | **0 / 10,000** | Zero (subsumes seq_disjoint). Cluster-equivalent sequences are forced into the same split, which is a strictly stronger constraint. |

---

## Qualitative finding: 2-D geometry per routing

The headline figures are the **sequence UMAP** panels (closest analog
to DataSAIL Fig. 4) under `plots/kmer_sequence_umap.png` in each run
dir.

**`random`**:  train/val/test colors are fully interspersed in 2-D
space. The "In multiple splits" 4th-color category (gray) sits behind
the per-split layers, making the leakage signature visible in the
legend rather than as a distinct visual region (the leakage points
share neighborhoods with both their train AND test-side copies, by
construction).

**`seq_disjoint`**:  visually **very similar to random**. Each
sequence appears in only one split, but the 2-D positions of those
sequences are not pushed apart in feature space — sequences with
near-identical k-mer profiles can still land in different splits. The
*leakage count* changed (1000 → 0); the *spatial separation* did not.

**`cluster_disjoint`**:  clear regional separation visible. In the HA
sequence panel the upper-right neighbourhood is dominated by train
(teal), bands of test (red) sit on the opposite side, and val
(purple) is interspersed. Same pattern in NA. Cluster-equivalent
sequences are forced into the same split, which means a coherent
neighbourhood of similar sequences collapses to a single colour.

This matches the cluster-disjoint mechanism: bicc routing partitions
clusters, not feature-space distance directly. Sequences in the same
cluster occupy adjacent feature-space regions and now all get the
same split colour, which produces visible regional partitioning at
the cluster granularity.

---

## Caveat: cluster_disjoint separation is more modest than DataSAIL S1

The cluster_id99 separation **is real but not dramatic** compared to
DataSAIL Fig. 4 (S1 split puts test in a clearly distinct
embedding-space region). Two reasons:

1. **Different objectives**. DataSAIL's S1 split is the output of a
   constrained optimization that *actively maximizes* between-split
   distance subject to a size constraint. Bicc routing only
   *prohibits* within-cluster splits — it doesn't push splits apart
   globally. Two cluster boundaries can still sit adjacent in
   feature space.

2. **Threshold ceiling on Flu A**. id099 is the strictest
   cluster_disjoint threshold that is feasible on Flu A's HA / NA
   pair: at id100 the largest bipartite CC is 20% of pairs, at id099
   it's 80%, and at id098 it collapses to 94% (one cluster contains
   most pairs, so 80/10/10 routing is no longer feasible — see
   `docs/methods/clustering_overview.md` §9). At id099 most clusters
   are singletons or pairs, weakening the spatial-partition
   constraint.

This is honest depiction of what bicc routing achieves at the Flu A
feasibility ceiling. A potential next-step methodology is the
"boundary-sample drop scheme" (BACKLOG.md Methodology #1) which
would drop a small fraction of cluster-boundary sequences to unlock
stricter feasible thresholds and stronger separation.

---

## Negatives by regime (orthogonal analysis)

A separate `plot_kmer_pair_negatives_by_regime` panel asks whether
the 8 negative regimes (`none_match`, `host_only`, `subtype_only`,
`year_only`, plus the 4 pairwise / triple combinations) form
distinct clusters in feature space. Generated on
`dataset_flu_ha_na_regimes_ratio3_20260513_211559` (a regime-aware
run; the three routing runs above are regime-blind and have
`neg_regime` = NaN by inheritance from `flu_ha_na.yaml`). 4,000
sampled negatives (500 per regime × 8 regimes).

**Finding**: the 8 regimes are **largely intermixed** in k-mer space
on both SVD and UMAP. Some local micro-clusters appear to share one
dominant regime, but no large-scale regime-based partitioning of the
2-D space.

**Interpretation**: `neg_regime` is a **metadata-driven label**
(host / subtype / year co-occurrence axes), not a feature-space
structure. Two pairs with very different sequences can share a
regime if their metadata patterns coincide; two pairs with
near-identical sequences can sit in different regimes. The 8
categories carve the metadata axis, not the k-mer manifold. Local
micro-clustering on the UMAP reflects **protein-sequence similarity**
(similar HA / NA strains share a neighborhood) rather than
**regime similarity**.

Implication: the regime-aware-sampling lever and the
routing-geometry lever are operating on orthogonal axes — useful
to keep in mind when interpreting downstream model behaviour. The
model would not be able to "learn the regime label" from k-mer
features alone.

Output: `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_regimes_ratio3_20260513_211559/plots/kmer_pair_negatives_{svd,umap}.png`.

---

## Reproducibility

All three reducers are deterministic under `random_state=42`:
TruncatedSVD (randomized solver), UMAP (forces `n_jobs=1` with seed —
expected umap-learn behaviour), and the stratified sub-sampler.
Re-running Stage 3 on the same source data + bundle yields
bit-for-bit identical PNGs.

The four PNGs per run are produced by
`src.analysis.plot_kmer_routing_geometry.plot_kmer_routing_geometry`
(top-level wrapper) calling `plot_kmer_sequence_geometry` and
`plot_kmer_pair_geometry`. The negatives-by-regime panel is produced
by the standalone `plot_kmer_pair_negatives_by_regime`. Wired into
`visualize_dataset_stats.py`'s end-of-Stage-3 orchestrator so all
future Stage 3 runs auto-emit the new plots.

---

## Pointers

| Artifact | Path |
|---|---|
| Plan doc | `docs/plans/2026-05-20_routing_geometry_viz_plan.md` |
| Plotting module | `src/analysis/plot_kmer_routing_geometry.py` |
| Orchestrator wire-in | `src/analysis/visualize_dataset_stats.py` (`visualize_dataset_stats` function) |
| Routing-comparison bundles | `conf/bundles/flu_ha_na_{random,seq_disjoint,cluster_id99}.yaml` |
| Regime-aware run used for negatives plot | `dataset_flu_ha_na_regimes_ratio3_20260513_211559` |
| Clustering feasibility reference | `docs/methods/clustering_overview.md` §9 (HA/NA aa id099 = 80% largest CC) |
