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

| # | Canonical name | Synonyms | Description | Assessed by | Addressed | Status |
|---|---|---|---|---|---|---|
| 1 | Same-pair **leakage** | pair-key leakage | Same `pair_key` in train and test. | v2 within-split + cross-split protein-pair dedup | ✅ within-split protein-pair dedup (v2 strict mode) + cross-split protein-pair dedup (`forbidden_pair_keys` threading) | ✅ Verified zero `pair_key` overlap across splits via v2 strict-mode assertion (enforced at construction time). |
| 2 | Sequence-level label **imbalance** | slot label imbalance | A sequence appears only as positive (or only as negative) within a split. | v2 coverage phase (per-`dna_hash` per slot) + protein-level safety raise; Exp 1 split overlap stats (per-seq_type pos vs neg `n_unique`); `n_dna_uncovered` in `dataset_stats.json` | ✅ Protein level (v2 coverage phase enforces ≥1 neg per `seq_hash` per slot). ✅ DNA level (v2 coverage phase enforces ≥1 per `dna_hash` per slot; tight bundles emit `n_dna_uncovered` in `dataset_stats.json` and a WARNING). | Protein level: 0 imbalance on every dataset built. DNA level (HA/NA mixed, post-implementation `dataset_flu_ha_na_20260508_171512`): pos `n_unique` == neg `n_unique` on every `dna_hash` row (43,875 / 43,875 slot a; 41,799 / 41,799 slot b); `n_dna_uncovered` = 0 across all splits. |
| 3 | Sequence-level **leakage** | slot-level leakage | Same `seq_hash` / `dna_hash` appears in different pairs across splits. | Exp 1 (split overlap stats); Exp 4 (seq-disjoint / strict-dedup re-train) | ❌ Not yet — Exp 4 will add `seq_disjoint` and `strict_dedup` split modes. | ❌ Confirmed present (Exp 1, HA/NA mixed): ~25% `seq_hash` overlap and ~10% `dna_hash` overlap between train and val/test on slot a. |
| 4 | Cluster leakage | near-neighbor leakage | Test pair's joint feature vector is cosine-near a training pair's, even if no exact hash match. | Exp 2 (1-NN baseline); Exp 3 (cosine deciles); Exp 4 (partial bound — handles exact-DNA case only); Exp 5 (mmseqs2 cluster splits) | ❌ Not yet — Exp 2/3/5 will quantify; mitigation depends on results. | ⚠️ Suggested but not formally measured. Anchor signal: median nearest-train PB1 cosine = 0.994. Awaiting Exp 2 (1-NN AUC) and Exp 3 (decile plot) for a formal verdict; Exp 5 for the strictest test. |
| 5 | Demographic shortcut leakage | metadata shortcut leakage | Model uses `same_host`, `same_subtype`, `same_year`, etc. as proxy for "same isolate." | Level 1 / Level 2 stratified eval; `analyze_negative_hardness` (match_count, match_pattern); Exp 3 cosine deciles | ❌ Not yet. Candidate mitigations: hard-negative mining (cheapest), adversarial / gradient-reversal training (DANN-style), Invariant Risk Minimization (IRM). Tracked in `roadmap_v2.md` Task 12. | ❌ Confirmed present: 30–50× FP-rate climb on match_count (HA/NA mixed, both h=[10] and h=[200]). See `docs/results/2026-05-07_metadata_shortcut_negatives.md`. |

Experiments 2 and 4 from the action list below directly test these.
Experiments 1 and 3 are diagnostics; 5 is escalation. The taxonomy
and the operational "biology learning" definition are now persisted
in `docs/methods/leakage_definitions.md`. The conservation analysis
sits in the **Background analyses** section as Anl 1 — it
characterizes the biological setting but doesn't itself test a
mode in the table.

## Operational definition: "model learned biology"

The model has learned biology-relevant signal **insofar as** its
accuracy on tasks (a) and (b) (see below) is meaningfully higher than a
1-NN lookup on the same splits:

(a) **Generalization to novel sequences** — pairs whose individual
sequences have cosine < 0.95 to all training sequences.
(b) **Generalization across populations** — pairs from a held-out
subtype, host, or year.

If MLP ≈ 1-NN on these tasks, the model is doing nothing more than
near-neighbor lookup. If MLP > 1-NN, real feature learning is
happening. This definition is testable and motivates Experiments
2 and 4.

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

Full CSV from `dataset_flu_ha_na_20260508_171512` (HA/NA mixed, v2;
12 rows per `seq_type` = 3 splits × 2 labels × 2 sides). The
`overlap_with_<own-split>` column trivially equals `n_unique` (the
sequences in this cell are by definition all in their own split);
likewise `pct_overlap_<own-split>` is trivially 100.0.

**dna_hash** (nucleotide hash):

```
side label split  n_pairs  n_unique  overlap_with_train  overlap_with_val  overlap_with_test  pct_overlap_train  pct_overlap_val  pct_overlap_test
   a  neg train    59598     43875               43875               563                520              100.0              1.3               1.2
   a  neg   val     7881      5775                 563              5775                136                9.7            100.0               2.4
   a  neg  test     7866      5776                 520               136               5776                9.0              2.4             100.0
   a  pos train    47060     43875               43875               563                520              100.0              1.3               1.2
   a  pos   val     5883      5775                 563              5775                136                9.7            100.0               2.4
   a  pos  test     5883      5776                 520               136               5776                9.0              2.4             100.0
   b  neg train    59598     41799               41799               802                818              100.0              1.9               2.0
   b  neg   val     7881      5657                 802              5657                213               14.2            100.0               3.8
   b  neg  test     7866      5683                 818               213               5683               14.4              3.7             100.0
   b  pos train    47060     41799               41799               802                818              100.0              1.9               2.0
   b  pos   val     5883      5657                 802              5657                213               14.2            100.0               3.8
   b  pos  test     5883      5683                 818               213               5683               14.4              3.7             100.0
```

**seq_hash** (protein hash):

```
side label split  n_pairs  n_unique  overlap_with_train  overlap_with_val  overlap_with_test  pct_overlap_train  pct_overlap_val  pct_overlap_test
   a  neg train    59598     34338               34338              1244               1261              100.0              3.6               3.7
   a  neg   val     7881      5057                1244              5057                417               24.6            100.0               8.2
   a  neg  test     7866      5072                1261               417               5072               24.9              8.2             100.0
   a  pos train    47060     34338               34338              1244               1261              100.0              3.6               3.7
   a  pos   val     5883      5057                1244              5057                417               24.6            100.0               8.2
   a  pos  test     5883      5072                1261               417               5072               24.9              8.2             100.0
   b  neg train    59598     31010               31010              1524               1596              100.0              4.9               5.1
   b  neg   val     7881      4849                1524              4849                577               31.4            100.0              11.9
   b  neg  test     7866      4842                1596               577               4842               33.0             11.9             100.0
   b  pos train    47060     31010               31010              1524               1596              100.0              4.9               5.1
   b  pos   val     5883      4849                1524              4849                577               31.4            100.0              11.9
   b  pos  test     5883      4842                1596               577               4842               33.0             11.9             100.0
```

Each train/val/test triplet for the same `(side, label)` is adjacent
within each table — cross-split overlap is read off as a 3-row column
scan.

Three patterns to read off this run (HA/NA mixed, v2):
- **seq_hash overlap is large**: ~25% of val/test HA sequences (side
  a) are also present in train (`pct_overlap_train` = 24.6 on val,
  24.9 on test). NA (side b) is even higher: ~31–33%.
- **dna_hash overlap is much smaller**: ~9% on val/test for side a
  (HA), ~13% on side b (NA). Synonymous codon variation.
- **Pos and neg `n_unique` match on every (seq_type, side) row.**
  That's mode #2 fully addressed at both protein and DNA levels:
  v2's coverage phase enforces ≥1 neg per `seq_hash` AND ≥1 neg
  per `dna_hash` per slot. Numbers: train pos `seq_hash` a (34,338)
  == train neg `seq_hash` a (34,338); train pos `dna_hash` a
  (43,875) == train neg `dna_hash` a (43,875). Same shape on slot
  b.

**What.** Add `split_overlap_stats.csv` to Stage 3 output, alongside
`dataset_stats.json`.

**How.** Function in `src/datasets/dataset_segment_pairs_v2.py`,
called from `split_dataset_v2` after the splits are finalized.
Reads from the train/val/test pair DataFrames already in memory.

**Effort.** ~50 lines.

**Success.** This can answer "how many test sequences are also
in train?"

---

### Exp 2 — k-NN baseline (with k=1)

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

### Exp 3 — Stratified accuracy by nearest-train cosine

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

### Exp 4 — Sequence-disjoint and strict-dedup splits

**Why.** Direct test of sequence-level leakage (#3); partial bound
on cluster leakage (#4). Both modes operate on **exact** `dna_hash`
identity — they prevent the same DNA from appearing in different
splits but say nothing about *near*-identical DNA (synonymous codon
neighbors with high k-mer cosine but different `dna_hash`). For the
near-neighbor case, see Exp 5 (mmseqs2 cluster splits).

The current isolate-level split allows the same `dna_hash` to
appear in train and test in different pairs. Two stricter
alternatives:

- **strict-dedup** (user proposal): drop duplicates on `dna_hash_a`,
  then on `dna_hash_b`. Each unique DNA appears at most once in the
  whole dataset.
- **seq-disjoint routing** (Claude's proposal): partition unique
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

**Success.** Headline result for sequence-level leakage. Possible
outcomes:
- **Both alternative splits crash.** Sequence-level memorization was
  the dominant inflation; current numbers are inflated.
- **Strict-dedup crashes, seq-disjoint holds.** The model learns
  within-cluster patterns but doesn't generalize across clusters.
- **Neither crashes.** Sequence-level leakage (#3) was overstated as
  the bottleneck. Cluster leakage (#4) and demographic shortcut (#5)
  remain candidates: Exp 5 (mmseqs2 cluster splits) tests the
  near-neighbor side; the existing metadata-shortcut writeup
  (`docs/results/2026-05-07_metadata_shortcut_negatives.md`) already
  quantifies #5.

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
Exp 2 (k-NN k=1 baseline)  ─── feeds Exp 4 interpretation
Exp 3 (cosine deciles)     ─── feeds Exp 4 interpretation
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
