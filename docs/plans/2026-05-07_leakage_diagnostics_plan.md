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

| # | Canonical name | Synonyms | Description | Assessed by | Status |
|---|---|---|---|---|---|
| 1 | Same-pair **leakage** | pair-key leakage | Same `pair_key` in train and test. | v2 `pair_key` overlap assertion | ✅ ADDRESSED — v2 assertion + `forbidden_pair_keys` threading |
| 2 | Sequence-level label **imbalance** | slot label imbalance | A sequence appears only as positive (or only as negative) in train. | v2 coverage assertion + `seqs_with_zero_negatives` raise | ✅ ADDRESSED — v2 coverage phase + per-sequence raise |
| 3 | Sequence-level **leakage** | Slot-level leakage | Same `seq_hash` / `dna_hash` appears in different pairs across splits. | Exp 1 (split overlap stats); Exp 4 (seq-disjoint / strict-dedup re-train) | ❌ NOT ADDRESSED — measured 11–16% on v2 |
| 4 | Cluster leakage | near-neighbor leakage | Test pair's joint feature vector is cosine-near a training pair's, even if no exact hash match. | Exp 2 (cosine deciles); Exp 3 (1-NN baseline); Exp 5 (mmseqs2 cluster splits) | ❌ NOT ADDRESSED — median nearest-train PB1 cos = 0.994 |
| 5 | Demographic shortcut leakage | metadata shortcut leakage | Model uses `same_host`, `same_subtype`, `same_year`, etc. as proxy for "same isolate." | Level 1 / Level 2 stratified eval; `analyze_negative_hardness` (match_count, match_pattern); Exp 2 cosine deciles | ❌ NOT ADDRESSED — quantified 30×–50× FP-rate climb on match_count |

Experiments 3 and 4 from the action list below directly test these.
Experiments 1 and 2 are diagnostics; 5 is escalation. The taxonomy
and the operational "biology learning" definition are now persisted
in `docs/methods/leakage_definitions.md`. The conservation analysis
sits in the **Background analyses** section as Anl 1 — it
characterizes the biological setting but doesn't itself test a
mode in the table.

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

This definition is also persisted in
`docs/methods/leakage_definitions.md` as the project-level
source-of-truth.

---

## Experiments

Ordered by effort and dependency. Items 1–3 are quick and inform 4–5.

### Exp 1 — Cross-split overlap stats table

**Why.** We currently log `pair_key` overlap only (verified zero by
v2 assertion). We need explicit counts at the `seq_hash` and
`dna_hash` level too, stratified by side (a/b) and label (pos/neg),
so the sequence-level leakage (#3) is visible at a glance instead of
having to re-derive it from train/val/test pair tables.

Full CSV from `dataset_flu_ha_na_20260508_122655` (HA/NA mixed, v2;
24 rows = 3 splits × 2 labels × 2 seq_types × 2 sides). The
`overlap_with_<own-split>` column trivially equals `n_unique` (the
sequences in this cell are by definition all in their own split);
likewise `pct_overlap_<own-split>` is trivially 100.0. Both are kept
in the schema so the column count is constant.

```
split label seq_type side  n_pairs  n_unique  overlap_with_train  overlap_with_val  overlap_with_test  pct_overlap_train  pct_overlap_val  pct_overlap_test
 test   neg dna_hash    a     7016      5331                 415                99               5331                7.8              1.9             100.0
 test   neg dna_hash    b     7016      5162                 659               160               5162               12.8              3.1             100.0
 test   neg seq_hash    a     7016      5072                1261               417               5072               24.9              8.2             100.0
 test   neg seq_hash    b     7016      4842                1596               577               4842               33.0             11.9             100.0
 test   pos dna_hash    a     5883      5776                 520               136               5776                9.0              2.4             100.0
 test   pos dna_hash    b     5883      5683                 818               213               5683               14.4              3.7             100.0
 test   pos seq_hash    a     5883      5072                1261               417               5072               24.9              8.2             100.0
 test   pos seq_hash    b     5883      4842                1596               577               4842               33.0             11.9             100.0
train   neg dna_hash    a    47859     37618               37618               417                403              100.0              1.1               1.1
train   neg dna_hash    b    47859     35260               35260               620                635              100.0              1.8               1.8
train   neg seq_hash    a    47859     34338               34338              1244               1261              100.0              3.6               3.7
train   neg seq_hash    b    47859     31010               31010              1524               1596              100.0              4.9               5.1
train   pos dna_hash    a    47060     43875               43875               563                520              100.0              1.3               1.2
train   pos dna_hash    b    47060     41799               41799               802                818              100.0              1.9               2.0
train   pos seq_hash    a    47060     34338               34338              1244               1261              100.0              3.6               3.7
train   pos seq_hash    b    47060     31010               31010              1524               1596              100.0              4.9               5.1
  val   neg dna_hash    a     7043      5310                 454              5310                100                8.5            100.0               1.9
  val   neg dna_hash    b     7043      5205                 676              5205                175               13.0            100.0               3.4
  val   neg seq_hash    a     7043      5057                1244              5057                417               24.6            100.0               8.2
  val   neg seq_hash    b     7043      4849                1524              4849                577               31.4            100.0              11.9
  val   pos dna_hash    a     5883      5775                 563              5775                136                9.7            100.0               2.4
  val   pos dna_hash    b     5883      5657                 802              5657                213               14.2            100.0               3.8
  val   pos seq_hash    a     5883      5057                1244              5057                417               24.6            100.0               8.2
  val   pos seq_hash    b     5883      4849                1524              4849                577               31.4            100.0              11.9
```

`seq_type` is the kind of identifier the row tracks: `seq_hash`
(protein hash) or `dna_hash` (nucleotide hash). `n_pairs` repeats 4×
per `(split, label)` (across the 4 seq_type/side rows) by design —
it's there so the redundancy ratio (e.g. `n_pairs=47,060` collapses
to `n_unique=34,338` for train pos seq_hash a) is visible inline.

Three patterns to read off this run (HA/NA mixed, v2):
- **seq_hash overlap is large**: ~25% of val/test HA sequences (side
  a) are also present in train (`pct_overlap_train` = 24.6 on val,
  24.9 on test). NA (side b) is even higher: ~31–33%.
- **dna_hash overlap is much smaller**: ~9% on val/test for side a
  (HA), ~13% on side b (NA). Synonymous codon variation between
  isolates breaks DNA-level matches even when the protein is the
  same.
- **Same-side, same-seq_type rows are identical between pos and
  neg** (e.g. train pos seq_hash a = train neg seq_hash a = 34,338
  unique). That's a v2 invariant: per-sequence label imbalance is
  prevented by the coverage phase, so every train sequence appears
  in BOTH pos and neg pairs on the same side.

**What.** Add `split_overlap_stats.csv` to Stage 3 output, alongside
`dataset_stats.json`. One row per `(split, label, seq_type, side)`
tuple. Columns: `n_pairs`, `n_unique`, three `overlap_with_*` counts,
three `pct_overlap_*` percentages.

**How.** Function in `src/datasets/dataset_segment_pairs_v2.py`,
called from `split_dataset_v2` after the splits are finalized.
Reads from the train/val/test pair DataFrames already in memory.

**Effort.** ~50 lines.

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

**Success.** A monotonic accuracy-vs-cosine plot decisively shows
whether near-neighbor lookup explains the headline accuracy.
Predicted: accuracy collapses on the lowest-cosine decile.

---

### Exp 3 — k-NN baseline (with k=1)

**Why.** Operational test of the "model learned biology" definition.
The k-NN family is the canonical non-parametric memorizer; with k=1
it predicts the label of the single closest training point in feature
space. If a 1-NN classifier achieves AUC comparable to the MLP, the
MLP has not learned anything beyond what nearest-neighbor matching
captures.

**What.** Add `src/models/baselines/knn.py`. Train on the same
features used by the MLP (k-mer concat). Evaluate AUC, F1, MCC on
the same val/test splits. Write to a comparable `metrics.csv`.

**How.** sklearn `NearestNeighbors` (k=1) on L2-normalized concat
features. Predict label of the single nearest train pair for each
test pair. Output to a sibling run dir
`models/flu/.../runs/baseline_knn_<bundle>_<ts>/`.

**Effort.** ~50 lines.

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

**Effort.** Implementation + 3 short training runs.

**Success.** Headline result for the project. Possible outcomes:
- **Both alternative splits crash.** Memorization was the dominant
  signal; current numbers are inflated.
- **Strict-dedup crashes, seq-disjoint holds.** The model learns
  within-cluster patterns but doesn't generalize across clusters.
- **Neither crashes.** Sequence-level leakage was overstated;
  metadata-shortcut (#5) is the dominant remaining concern.

---

### Exp 5 — mmseqs2 cluster-based splits (escalation)

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
phylogenetic near-neighbors. If Exp 4 already crashes, Exp 5 is
unnecessary for the leakage story (but might still be useful for
phylo robustness).

**Effort.** Install mmseqs2 + integrate into split dispatcher.

---

## Background analyses

These describe / characterize data; they don't test a hypothesis and
don't have a pass/fail criterion. They sit alongside the experiments
to inform interpretation. New entries go here under prefix `Anl N`.

### Anl 1 — Conservation analysis across representations

**Why.** Frames the leakage discussion biologically: how conserved
*are* the sequences we're working with, and is that conservation
preserved across the representations the model sees? Also tests
whether ESM-2 inflates similarity beyond what AA identity warrants.
Does NOT directly assess any mode in the taxonomy table — it
characterizes the biological setting in which cluster leakage (#4)
operates.

Naming note: this analysis uses **within-function / cross-function**
to avoid colliding with the **within-subtype / cross-subtype**
terminology of Level 1. "Function" matches the column in
`protein_final.csv` (HA, NA, PB1, PB2, PA, NP, M1, …).

**What.** A function-level conservation table + 2D embedding plot
per representation (raw AA, raw DNA, k-mer, ESM-2). New analysis
script `src/analysis/conservation_by_representation.py`. Output:
`docs/results/<date>_conservation_by_representation.md` plus
per-representation PCA/UMAP PNGs.

**How.**
1. Per function: sample 1000 random pairs of sequences.
2. Compute pairwise similarity in each of the four representations:
   AA identity, nucleotide identity, k-mer cosine, ESM-2 cosine.
3. Aggregate per representation:
   - **within-function mean similarity** = mean over pairs whose
     two sequences share a function (HA-HA, NA-NA, …). Higher means
     a tighter cluster for that function in this representation.
   - **cross-function mean similarity** = mean over pairs whose two
     sequences differ in function (HA-NA, HA-PB1, …). Acts as a
     "floor" for what unrelated sequences look like.
   - **ratio** = within-function / cross-function. >> 1 means the
     representation cleanly separates functions; ≈ 1 means it
     muddles them.
4. PCA/UMAP per representation, colored by function.

The first-page anchor is the `uniq_prot` / `uniq_dna` / `rows/prot`
/ `dna/prot` table I already computed (HA = 41,896 unique proteins
vs PB1 = 31,226 across ~114K isolates each; M1 = 4,771 — most
conserved). Anl 1 builds on that with the per-representation
similarity numbers and plots.

**Output.** Concrete answer to "is PB1 more conserved than HA?"
(within-function similarity per representation) and "does ESM-2
collapse functions more than AA identity warrants?" (compare
within/cross ratios across representations). No pass/fail — this is
descriptive context.

---

## Execution order and dependencies

```
Exp 1 (overlap stats)      ─── independent diagnostic, run first
Exp 2 (cosine deciles)     ─── feeds Exp 4 interpretation
Exp 3 (k-NN k=1 baseline)  ─── feeds Exp 4 interpretation
       │
       ▼
Exp 4 (seq-disjoint splits)─── HEADLINE EXPERIMENT
       │
       ▼  (only if Exp 4 results say so)
Exp 5 (mmseqs2 clusters)

Anl 1 (conservation)       ─── background, can run anywhere
```

Exp 1, 2, 3 can run in parallel (~half day total). Exp 4 is the
load-bearing scientific test. Exp 5 is escalation. Anl 1 sits
outside the experimental dependency chain — it's biology framing
that informs interpretation but doesn't gate other work. The
taxonomy + biology-learning definition are already landed in
`docs/methods/leakage_definitions.md`.

## Out of scope

- Sequence-similarity controls beyond mmseqs2 clustering (e.g.
  phylogenetic distance trees) — defer until after Exp 5 results.
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
