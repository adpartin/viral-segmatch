# Leakage diagnostics: memorization vs biology — Plan

**Status: PROPOSED** (pending approval)
**Date:** 2026-05-07

## Context

`docs/results/2026-05-07_metadata_shortcut_negatives.md` quantified one
leakage mode (metadata-shortcut) on HA/NA mixed across two capacities
(h=[10] and h=[200]). It showed FP rate climbing 30×–50× along
match_count and concluded that more capacity *amplifies* the
shortcut. That writeup leaves several other leakage modes undiagnosed
and the central scientific question open: **is the model learning a
generalizable co-occurrence signal, or is high test AUC the result of
memorizing near-neighbors of training pairs?**

Aggregate metrics (AUC 0.978 → 0.990 across capacity) do not
distinguish these two stories. This plan lays out a focused set of
experiments that does.

## Dataset issues (that may lead to overly optimistic predictions)

| # | Canonical name | Synonyms | Description | Status |
|---|---|---|---|---|
| 1 | Same-pair **leakage** | pair-key leakage | Same `pair_key` in train and test. | ✅ ADDRESSED — v2 assertion + `forbidden_pair_keys` threading |
| 2 | Sequence-level label **imbalance** | slot label imbalance | A sequence appears only as positive (or only as negative) in train. | ✅ ADDRESSED — v2 coverage phase + per-sequence raise |
| 3 | Sequence-level **leakage** | Slot-level leakage | Same `seq_hash` / `dna_hash` appears in different pairs across splits. | ❌ NOT ADDRESSED — measured 11–16% on v2 |
| 4 | Cluster leakage | near-neighbor leakage | Test pair's joint feature vector is cosine-near a training pair's, even if no exact hash match. | ❌ NOT ADDRESSED — median nearest-train PB1 cos = 0.994 |
| 5 | Demographic shortcut leakage | metadata shortcut leakage | Model uses `same_host`, `same_subtype`, `same_year`, etc. as proxy for "same isolate." | ❌ NOT ADDRESSED — quantified 30×–50× FP-rate climb on match_count |

Experiments 3, 4, 5 from the action list below directly test these.
Experiments 1 and 2 are diagnostics; 6 is conceptual; 7 is escalation.

## Operational definition: "model learned biology"

The model has learned biology-relevant signal **insofar as** its
accuracy on tasks (a) and (b) below is meaningfully higher than a
1-NN lookup on the same splits:

(a) **Generalization to novel sequences** — pairs whose individual
sequences have cosine < 0.95 to all training sequences.
(b) **Generalization across populations** — pairs from a held-out
subtype, host, or year.

If MLP ≈ 1-NN on these tasks, the model is doing nothing more than
near-neighbor lookup. If MLP > 1-NN, real feature learning is
happening. This definition is testable and motivates Experiments
3 and 4.

To be written into `docs/methods/` as a short reference (Experiment 7).

---

## Experiments

Ordered by effort and dependency. Items 1–3 are quick and inform 4–5.

### Exp 1 — Cross-split overlap stats table

**Why.** We currently log `pair_key` overlap only (verified zero by
v2 invariant). We need explicit counts at the `seq_hash` and
`dna_hash` level too, stratified by side (a/b) and label (pos/neg),
so the sequence-level leakage (#3) is visible at a glance instead of
having to re-derive it from train/val/test pair tables.

**What.** Add `split_overlap_stats.csv` to Stage 3 output, alongside
`dataset_stats.json`. One row per `(split, label, axis, side)`
tuple. Columns: `n_unique`, `n_in_train`, `n_in_val`, `n_in_test`.

**How.** Function in `src/datasets/dataset_segment_pairs_v2.py`,
called from `split_dataset_v2` after the splits are finalized.
Reads from the train/val/test pair DataFrames already in memory.

**Effort.** ~50 lines, ~1 hour.

**Success.** A reader can answer "how many test sequences are also
in train?" by reading one CSV.

---

### Exp 2 — Stratified accuracy by nearest-train cosine

**Why.** Direct test of feature-near-neighbor leakage (#4). If
test-set accuracy correlates strongly with cosine-distance-to-nearest-
training-pair, the model is doing near-neighbor lookup, not
generalization.

**What.** Stratify the existing test predictions of an existing run
into deciles by `cosine(test_feature, nearest_train_feature)`. Plot
accuracy / FP rate / mean confidence per decile. New post-hoc output:
`stratified_by_train_distance.csv` + `.png`.

**How.** Compute joint k-mer feature for every train pair (cache as a
sparse matrix). For each test pair, compute cosine to every train
pair, take max. Stratify by max-cosine deciles. ~50 lines, runs in
under a minute on existing test sizes.

**Effort.** ~hour.

**Success.** A monotonic accuracy-vs-cosine plot decisively shows
whether near-neighbor lookup explains the headline accuracy.
Predicted: accuracy collapses on the lowest-cosine decile.

---

### Exp 3 — 1-NN baseline

**Why.** Operational test of the "model learned biology" definition.
If a parameter-free 1-NN classifier achieves AUC comparable to the
MLP, the MLP has not learned anything beyond what nearest-neighbor
matching captures.

**What.** Add `src/models/baseline_knn.py`. Train on the same
features used by the MLP (k-mer concat). Evaluate AUC, F1, MCC on
the same val/test splits. Write to a comparable `metrics.csv`.

**How.** sklearn `NearestNeighbors` on L2-normalized concat features.
Predict label of nearest train pair for each test pair. Output to a
sibling run dir
`models/flu/.../runs/baseline_knn_<bundle>_<ts>/`.

**Effort.** ~50 lines, ~1 hour.

**Success.** A side-by-side comparison table:

| split | metric | MLP | 1-NN | gap |
|---|---|---|---|---|

If gap is small (< 0.02 AUC), the MLP is operating mostly as a soft
1-NN. If gap is large, the MLP is genuinely learning a
generalizable representation.

---

### Exp 4 — Sequence-disjoint and strict-dedup splits

**Why.** Direct test of sequence-level leakage (#3) and feature
near-neighbor leakage (#4). The current isolate-level split allows
the same `dna_hash` to appear in train and test in different pairs.
Two stricter alternatives:

- **strict-dedup** (user proposal): drop duplicates on `dna_hash_a`,
  then on `dna_hash_b`. Each unique DNA appears at most once in the
  whole dataset.
- **seq-disjoint routing** (my proposal): partition unique
  `dna_hash`es into train/val/test sets; route each pair into the
  split where BOTH its DNAs sit. Drop split-mixed pairs.

**What.** Two new split modes in `split_dataset_v2`, selectable by
config (`split_mode: assembly | seq_disjoint | strict_dedup`).
Re-train `flu_ha_na` (h=[100]) under each mode and compare.

**How.**
1. Add the two split modes to `split_dataset_v2`. (~30 lines each.)
2. Re-build the HA/NA dataset under each mode.
3. Re-train. Compare aggregate metrics + match-count FP rate +
   nearest-train-cosine stratified accuracy.

**Effort.** Half a day implementation + 3 short training runs (~5
minutes each).

**Success.** Headline result for the project. Possible outcomes:
- **Both alternative splits crash.** Memorization was the dominant
  signal; current numbers are inflated.
- **Strict-dedup crashes, seq-disjoint holds.** The model learns
  within-cluster patterns but doesn't generalize across clusters.
- **Neither crashes.** Sequence-level leakage was overstated;
  metadata-shortcut (#5) is the dominant remaining concern.

---

### Exp 5 — Conservation analysis across representations

**Why.** Frames the leakage discussion biologically: how conserved
*are* the proteins we're working with, and is that conservation
preserved across the representations the model sees? Also tests
whether ESM-2 inflates similarity beyond what AA identity warrants.

**What.** A function-level conservation table + 2D embedding plot
per representation (raw AA, raw DNA, k-mer, ESM-2). New analysis
script `src/analysis/conservation_by_representation.py`. Output:
`docs/results/<date>_conservation_by_representation.md` plus
per-representation PCA/UMAP PNGs.

**How.**
1. Per function: sample 1000 random pairs of sequences.
2. Compute pairwise similarity in each of the four representations:
   AA identity, nucleotide identity, k-mer cosine, ESM-2 cosine.
3. Aggregate: within-type mean similarity, cross-type mean
   similarity, ratio.
4. PCA/UMAP per representation, colored by function-type.

**Effort.** Half a day.

**Success.** Concrete answer to "is PB1 more conserved than HA?" and
"does ESM-2 collapse function-types more than AA identity warrants?"
Initial counts (`uniq_prot/uniq_dna` per function on full dataset)
already confirm Jim's qualitative claim: HA = 41,896 unique proteins
vs PB1 = 31,226 across ~114K isolates each; M1 = 4,771 unique proteins
(most conserved). Exp 5 puts numbers and pictures on this.

---

### Exp 6 — Write `docs/methods/leakage_definitions.md`

**Why.** The "model learned biology" definition above needs to live
somewhere persistent so future runs / reviewers / collaborators read
the same definition. Also captures the leakage taxonomy in one
place.

**What.** Short markdown (~1 page) with the taxonomy table from this
plan plus the operational "biology learning" definition.

**Effort.** 30 minutes.

**Success.** A reader unfamiliar with the project can read this one
file and understand which leakage modes we've addressed, which we're
testing, and what the success criterion for "the model works" is.

---

### Exp 7 — mmseqs2 cluster-based splits (escalation)

**Why.** Sequence-level dedup (Exp 4) is strictly stricter than
isolate-level split but does NOT prevent near-neighbors with
synonymous codon variation (k-mer cosine ~0.99 across different DNA).
Cluster-based splits address this by partitioning at a similarity
threshold.

**What.** Use mmseqs2 to cluster proteins at e.g. 95% identity, then
split clusters across train/val/test. Each test pair has guaranteed
< 95% AA identity to any train pair. New option in
`split_dataset_v2`: `split_mode: cluster_disjoint`.

**How.**
1. Install mmseqs2 (one line, mamba/bioconda).
2. Export protein FASTA from `protein_final.csv`.
3. Run `mmseqs easy-cluster` at 95% identity, 80% coverage.
4. Parse cluster TSV, assign clusters to splits, route pairs.
5. Re-train and compare.

**Trigger condition.** Run only if Exp 4 (the cheaper sequence-level
dedup) leaves a story like "performance held — leakage less
problematic than feared." Then we want a stricter test to rule out
phylogenetic near-neighbors. If Exp 4 already crashes, Exp 7 is
unnecessary for the leakage story (but might still be useful for
phylo robustness).

**Effort.** ~1 day if no surprises (install + integrate).

---

## Execution order and dependencies

```
Exp 1 (overlap stats)      ─── independent diagnostic, run first
Exp 2 (cosine deciles)     ─── feeds Exp 4 interpretation
Exp 3 (1-NN baseline)      ─── feeds Exp 4 interpretation
Exp 6 (definitions doc)    ─── 30 min, write before Exp 4 runs
       │
       ▼
Exp 4 (seq-disjoint splits)─── HEADLINE EXPERIMENT
       │
       ▼  (only if Exp 4 results say so)
Exp 7 (mmseqs2 clusters)
       │
       ▼  (any time)
Exp 5 (conservation)       ─── biology framing, can run anywhere
```

Exp 1, 2, 3, 6 can run in parallel (~half day total). Exp 4 is the
load-bearing scientific test. Exp 5 and 7 are escalations / context.

## Out of scope

- Sequence-similarity controls beyond mmseqs2 clustering (e.g.
  phylogenetic distance trees) — defer until after Exp 7 results.
- Hard-negative mining / focal loss — these are *mitigations* once
  leakage is characterized, tracked separately in roadmap_v2.md
  Task 12.
- ESM-2 vs k-mer comparison under disjoint splits — easy follow-up
  once Exp 4 infrastructure is in place.

## See also

- `docs/results/2026-05-07_metadata_shortcut_negatives.md` — the
  finding that triggered this plan.
- `roadmap_v2.md` Task 12 — broader FP/FN diagnosis + mitigation
  thread this plan supports.
- `docs/post_hoc_analysis_design.md` — Level 1/2 stratified eval
  design that the new diagnostics fit alongside.
