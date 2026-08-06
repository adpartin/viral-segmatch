# Roadmap: Experiments and Publication Scope

**Original date:** 02/10/2026
**Last updated:** 2026-03-12 (aligned with group meeting decisions)

See `paper_outline_v2.md` for paper narrative, conceptual frameworks (negative pair taxonomy,
hierarchical analysis, UQ rationale), and section-level content.
This document covers: task definitions, implementation details, status, effort, HPC,
execution order, and biology reference material.

---

## Scope

> **[2026-03-12 meeting]** Jim: "scope to the flu... it would make our reviewers head explode."

- **In scope (Paper 1):** Influenza A only. All 8 segments. K-mers primary, ESM-2 as comparison.
- **Out of scope (Paper 1):** Bunyavirales, reads-based approaches, GenSLM (Paper 2).
- **Use cases:** Data remediation + wastewater surveillance. Deferred until model validation
  is solid. Wastewater data sourcing initiated (Carla → Rachel Paretsky; Jim has ~20K
  assembled mixed samples; SRA backup).

---

## ALCF HPC Systems

| System | Status | Specs | Best for |
|--------|--------|-------|----------|
| **Polaris** | Available (primary) | 560 nodes, 4× A100, 32-core AMD EPYC, 512 GB RAM | CV job arrays, large-scale training |
| **Aurora** | Launched Jan 2025 | 60K+ GPUs, exascale | GenSLM-scale (Paper 2 only) |

---

### Runtime Analysis (March 2026) — Informs HPC Walltime for Tasks 1 & 2

Runtime profiling runs on Lambda cluster (NVIDIA A100 GPUs). Both bundles use k-mer k=6,
slot_norm + concat, 10-fold CV. Forced to 100 epochs with patience=100 (no early stopping)
to measure full epoch-level timing. Prediction performance is secondary — these are not
final paper results (no early stopping, no hyperparameter tuning).

- **5K subset bundle:** `flu_schema_raw_kmer_k6_slot_norm_concat_cv10` (~5K isolates, ~8.5K test pairs/fold)
- **Full dataset bundle:** `flu_schema_raw_kmer_k6_slot_norm_concat_full_cv10` (~111K isolates, ~15K test pairs/fold)
- **Hardware:** Lambda cluster, 5× NVIDIA A100 GPUs (10 folds across 5 GPUs, 2 waves)

| Metric | 5K subset | Full dataset |
|--------|-----------|--------------|
| **Per epoch** | 4.74s mean (4.60–4.82s) | 97.1s mean (67–136s) |
| **Single fold (100 epochs)** | 9m 12s | 2h 43m 11s |
| **Total CV (10 folds, 5 GPUs)** | 19m 06s | 5h 47m 41s |
| **AUC-ROC** | 0.972 ± 0.004 | 0.993 ± 0.001 |
| **PR-AUC** | 0.935 ± 0.011 | 0.977 ± 0.002 |
| **F1** | 0.949 ± 0.007 | 0.973 ± 0.002 |
| **Brier** | 0.043 ± 0.005 | 0.017 ± 0.001 |

**Key takeaways:**
- Full dataset gives substantial improvement (AUC 0.972→0.993) with much tighter fold variance.
- ~20x data scale-up yields ~20x per-epoch time (4.7s→97s), as expected for MLP training.
- For HPC walltime: a single full-dataset fold at 100 epochs ≈ 2h 45m. With early stopping
  (typical convergence ~30–50 epochs), expect ~1–1.5h per fold. Set PBS walltime to 3h per
  fold for safety margin.

**Polaris (A100, ensemble-packed 4 folds/node) — measured Phase 3, 2026-04-08:**
After applying the optimizations from `speed_up.md` (batch_size=128,
`eval_train_metrics=false`, `infer_batch_size=8192`) **and** the Polaris-specific fixes
(`pin_memory=false`, TF GPU pre-allocation prevention; see `docs/hardware_notes.md`),
per-epoch wall-clock dropped to **~25 s** (data 3.0 s + compute 20.4 s + eval 1.0 s) and
per-fold runtime to **~44 min** at 100 epochs. This is comparable to a Lambda single-fold
run despite running 4 folds concurrently on the same node — perfect ensemble-packing
scaling once the cudaHostAlloc bottleneck was removed. All 28 pairs × 12 folds completed
in ~3.5 h wall-clock on 28 nodes, ~98 node-hours total.

---

## Experiments: High Priority (2026-03-12 meeting)

---

### 2. Subtype-balanced training and test sets — NOT IMPLEMENTED

> **[2026-03-12 meeting]** **TOP PRIORITY (Jim's #1).** Jim: "expanding your training set... to include a large and balanced subset. Like you want to balance your types and everything."

**Status clarification (2026-04):** The "scale up" portion of Task 2 is complete — Task 11 ran on the full ~111K-isolate dataset with strong results. What remains is the **balancing** portion, which was the substantive part of Jim's ask.

**Goal:** Construct subtype-balanced train, val, and test sets so that per-subtype test performance is a clean measurement of model capability, not confounded by training-frequency effects.

**Definitions (use consistently throughout the paper):**
- **subtype-balanced** — explicitly resampled so HN-subtype representation is equalized across subtypes (downsampling dominant subtypes, optionally upsampling rare ones).
- **subtype-filtered** — restricted to one subtype (e.g., H3N2-only).
- **natural / unfiltered** — sampled proportional to BV-BRC frequencies (dominated by H3N2, H1N1, H5N1).

**Implementation approaches:**

1. **Sub-sample from existing folds (faster pass).**
   Reuse the fold structure from the completed Task 11 runs. Add a post-dataset balancing filter that downsamples each fold's train/val/test to equalize HN-subtype representation at the isolate level (important: balance must operate at isolate level, not independently per pair — otherwise the 28 pair classifiers no longer share the same underlying sample). Script: `scripts/balance_folds_by_subtype.py`. Re-run 28-pair job on Polaris.
   - Pro: fast, no new dataset generation, preserves k-mer vs ESM-2 comparability.
   - Con: only supports downsampling; loses data. Upsampling rare subtypes requires Approach 2.

2. **Integrate balancing into dataset generation (canonical).**
   Add a `subtype_balancing` config option to `src/datasets/dataset_segment_pairs.py`. Generate a new set of 28-pair × 12-fold datasets with balanced splits. Re-run the 28-pair job on Polaris.
   - Pro: reusable; supports both down- and upsampling; principled; right place for pair-distribution ledger.
   - Con: requires new Polaris dataset-generation job.

**Recommended order:** (1) as a fast diagnostic — does balancing change results at all? If yes, (2) for the paper's canonical experiment. If no, stay with natural frequencies and report stratified eval only.

**Stratified evaluation requirements** (apply to both the natural and balanced runs — see NEW: Stratified evaluation task):
- Level 1 — pair-type regime (positive / within-subtype neg / cross-subtype neg)
- Level 2 — per-axis marginals: HN subtype, **H-type**, **N-type**, host, isolation year, region

Notes:
- Post-filter class balance audit **complete** — see
  `docs/audits/class_balance_audit.md`. Finding: post-filter val/test ratios
  are 0.20–0.51 (vs 1:1 target), driven by `pair_key` cross-split removal.
  Material for F1 interpretation; AUC-ROC conclusions unaffected. Optional
  future work: oversample negatives during generation (`neg_to_pos_ratio > 1`)
  and truncate post-filter to restore balance.
- Negative-pair RNG determinism audit **complete** — see
  `negative_pair_rng_fix_plan.md`. Finding: isolate-level splits are
  deterministic in (master_seed, fold_id) and identical across the 28 pair
  bundles. Negative-pair *identities* diverge across bundles (val/test overlap
  ~0%) due to a plumbed `seed` argument silently discarded in
  `create_negative_pairs`. Does not invalidate Task 11 results. Apply minimal
  fix before generating balanced-subtype datasets so future runs are reproducible.

**HPC fit:** Same as Task 11 — 28 pairs × 12 folds ensemble-packed on Polaris, ~3.5 h walltime.

---

### 11. All protein-pair combinations (8×8 heatmap) — COMPLETE (2026-04-08 k-mer / 2026-04-19 ESM-2 + H3N2 variants)

> **[2026-03-12 meeting]** **TOP PRIORITY.** Jim: "I basically want to see the table... for all eight segments."

**Goal:** 28 pairwise combinations (C(8,2)) of PB2, PB1, PA, HA, NP, NA, M1, NS1.
Results as 8×8 AUC/F1 heatmap.

See `paper_outline_v2.md` Section 3.7 for paper framing and biology context.

**Status:** **Complete on Polaris.** Four sweeps: k-mer vs ESM-2 × unfiltered vs H3N2.
All 28 pairs × 12 folds, full Flu A dataset, slot_norm + concat, 100 epochs, ensemble-packed
on Polaris. 336/336 folds for the three clean sweeps; the original k-mer unfilt sweep had
2 Polaris launcher races (pb1_ha/fold11, pb2_pa/fold6) re-run on Lambda.

**Results summary — 4 sweeps:**

| Sweep | Feature | Subset | AUC median [min, max] | F1 median | Manifest |
|---|---|---|---|---|---|
| val_unfilt | k-mer k=6 concat | full (~111K) | 0.9944 [0.9928, 0.9960] | 0.9712 | `allpairs_prod_val_unfilt_20260414_080617/` |
| val_unfilt_esm2 | ESM-2 concat | full (~111K) | 0.9757 [0.4971, 0.9874] | 0.8970 | `allpairs_prod_val_unfilt_esm2_20260416_182049/` |
| val_h3n2 | k-mer k=6 concat | H3N2-only | 0.9931 [0.9910, 0.9948] | 0.9659 | `allpairs_prod_val_h3n2_20260417_224216/` |
| val_h3n2_esm2 | ESM-2 concat | H3N2-only | 0.9486 [0.4981, 0.9751] | 0.7947 | `allpairs_prod_val_h3n2_esm2_20260419_012308/` |

Original (2026-04-08) k-mer unfilt sweep also archived at `allpairs_prod_20260408_063203/`.

**Key findings across the 4 sweeps:**
- **ESM-2 `concat` collapse is not H3N2-specific.** PB1/PA collapses on BOTH the full
  unfiltered dataset (AUC=0.4971) and the H3N2-filtered dataset (AUC=0.4981). The
  subspace-offset failure mode recurs on a conserved internal-gene pair, not just on
  HA/NA under H3N2 restriction as previously documented. K-mer concat never collapses
  on any pair in any sweep.
- **K-mer dominates ESM-2 across the full 28-pair heatmap.** Median AUC gap is 0.019
  (unfiltered) and 0.045 (H3N2). The gap widens on the homogeneous subset as expected —
  ESM-2's geometry is more fragile when the subtype axis is removed.
- **K-mer is nearly filter-invariant** (median 0.9944 → 0.9931 unfilt → H3N2); ESM-2 is
  not (0.9757 → 0.9486). Supports the hypothesis that k-mer features are
  interaction-agnostic because sparse frequency vectors don't carry ESM-2's
  protein-type-specific subspace offset.
- **FP/FN ratios are 2–3× higher with ESM-2 than k-mer** (unfilt 5.87 vs 2.36;
  H3N2 4.58 vs 3.06). Makes ESM-2 the natural target for the Task 12 FP/FN interventions.

**HPC findings:** Phase 3 only worked after diagnosing two interacting bugs unique to
multi-process ensemble packing on Polaris: (1) `pin_memory=True` + cudaHostAlloc
serialization across 4 concurrent fold processes (300× data-loading slowdown), and
(2) TensorFlow GPU pre-allocation via transitive HuggingFace import (OOM at
`model.to(device)`). Both fixed; see `polaris_plan.md` Phase 3 results and
`docs/hardware_notes.md` for the full diagnosis. **These lessons should be carried over
to any future Polaris scaling task** (Tasks 8, 9, GenSLM).

---

### NEW: Stratified evaluation by pair type and metadata — PARTIAL (Level 2 only)

> **[2026-03-12 meeting]** **Unconditional analytical deliverable.**

**Goal:** For every experiment, classify predictions by pair type (within-subtype vs
cross-subtype) and metadata (HN subtype, host, geography, year). Report per-category
metrics at three hierarchical levels.

See `paper_outline_v2.md` Section 3.5.1 for the full hierarchical analysis framework
(Level 1: pair-type regime; Level 2: per-category marginals with "other" bucket;
Level 3: diagnostic cross-tabulation heatmaps) and the metadata confounder discussion.

**Gap analysis (existing vs required):**

| Level | Requirement | Status |
|-------|-------------|--------|
| **1. Pair-type regime** | Per-category metrics for {positives, within-subtype negatives, cross-subtype negatives}. Requires tagging each *negative pair* by whether its two source isolates share a subtype. | **Missing.** The defining deliverable. Existing `error_analysis_by_hn_subtype.csv` uses a single `hn_subtype` column per row — works for positives (one isolate = one subtype) but cannot represent cross-subtype negatives (two subtypes). |
| **2. Per-category marginals** | For each metadata axis independently (subtype, host, region, year), report metrics per category with "other" pooling for small groups. | **Partial.** `analyze_stage4_train.py:analyze_errors_by_metadata()` produces `error_analysis_by_{host,hn_subtype,year}.csv`, `performance_metrics_by_metadata.png`, `error_rates_by_metadata.png`. Assumes single-subtype pairs only (same Level 1 gap). No region axis. No explicit "other" bucket. |
| **3. Cross-tab heatmap** | Full (subtype_a × subtype_b) FP-rate matrix as diagnostic. | **Missing.** |

**Implementation needed:**
- **Pair-type classifier** (unblocks Level 1 + fixes Level 2): for each prediction row,
  look up `(subtype_a, subtype_b, host_a, host_b, year_a, year_b, region_a, region_b)`
  from isolate metadata, tag each row: positive | within-subtype-neg | cross-subtype-neg
  (and same for the other axes). Requires joining `test_predicted.csv` with
  `isolate_metadata.csv`.
- **Level 1 reporting** — metrics per {positive, within-subtype-neg, cross-subtype-neg}.
- **Level 2 extension** — split negatives into within/cross in the existing marginal
  stratification; add region axis; explicit "other" bucket for small categories
  (~50–100 test pair minimum).
- **Level 3 heatmap** — (subtype_a × subtype_b) FP-rate matrix for diagnostics.
- **Pair-distribution ledger** output (Section 2.1.2 of `paper_outline_v2.md`).
- **Metric choice:** Report AUC-ROC as primary (class-prior invariant). Report F1
  alongside post-filter positive rate per pair. Consider also reporting PR-AUC or
  balanced F1 for cross-pair comparison. See `docs/audits/class_balance_audit.md`.

**Effort:** Low–medium. Extend `analyze_stage4_train.py:analyze_errors_by_metadata()`
(around line 547) rather than writing from scratch. Post-hoc on existing
`test_predicted.csv` + `isolate_metadata.csv`.

---

### NEW: Mixed-subtype discrimination test (Carla) — NOT IMPLEMENTED

> **[2026-03-12 meeting]** Carla: "demonstrate whether the model can discriminate... if we had a soup of different Hs and different Ns."

**Goal:** Test whether the model correctly matches segments in a mixed-subtype population
(wastewater scenario).

See `paper_outline_v2.md` Section 3.6 for test set design and training options.

**Implementation:** Multi-subtype dataset construction with pair-distribution ledger.
Shares infrastructure with Task 2 (subtype-balanced sampling).

**Effort:** Medium.

---

## Experiments: Required (not discussed but needed)

### 1. Cross-validation (N splits) — IMPLEMENTED

**Goal:** N=5 folds; report mean ± std.

**Status:** Implemented. `n_folds`/`fold_id` in dataset config, `scripts/run_cv_lambda.py`,
`scripts/aggregate_cv_results.py`. Bundle: `flu_schema_raw_slot_norm_unit_diff_cv5`.
Needs end-to-end run.

**HPC fit:** High. Job array on Polaris.

---

### 3. Temporal holdout (train 2021–2023, test 2024) — IMPLEMENTED

> **[2026-03-12 meeting]** Jim reframes as post-hoc stratification axis. Keep both: explicit temporal holdout + post-hoc year stratification.

**Status:** Implemented. Bundles: `flu_schema_raw_slot_norm_unit_diff_temporal` (ESM-2),
`flu_schema_raw_kmer_k6_slot_norm_unit_diff_temporal` (k-mer). Initial runs complete.
See `docs/plans/temporal_holdout_plan.md`.

**Caveat:** pair_key dedup artifact removed 42% of val/test pairs. Needs re-run with
dedup disabled for temporal mode.

**Preliminary results:** See `paper_outline_v2.md` Section 3.3.

**HPC fit:** Low. Single run.

---

### 7. Baseline validation experiments — NOT IMPLEMENTED

> **[2026-03-12 meeting]** Not discussed. Must-have for reviewer confidence.

**Goal:** Confirm model uses sequence-level features, not shortcuts.

**Proposed baselines:**
1. Embedding shuffle within protein
2. Mean embedding per protein
3. Swap-slot test
4. Random embeddings

**Status:** Plan in `docs/plans/baseline_validation_experiments_plan.md`. Not implemented.

**Implementation:** Config flags (`training.ablation.shuffle_embeddings` etc.) +
perturbation logic. ~50–100 lines in embedding loading code.

**Effort:** Medium.

---

### NEW: Uncertainty quantification (UQ) — NOT IMPLEMENTED — REQUIRED

**Goal:** Calibrated confidence estimates for every prediction.

See `paper_outline_v2.md` Section 4.2 for the full rationale (operational necessity +
complements stratified analysis by addressing the long tail of rare metadata categories).

**Recommended approach:** Deep ensemble (N=5 CV fold models) + calibration analysis
(reliability diagram + ECE). No additional training required.

**Implementation needed:**
- Ensemble inference script (load N fold models, run each, aggregate)
- Calibration analysis script (reliability diagram, ECE, temperature scaling if needed)
- Per-pair uncertainty output (mean, std, calibrated probability)

**Effort:** Low–medium. Last required experiment, after Sections 3.1–3.7 validation.

---

## Experiments: Conditional (triggered by results)

### 12. FP/FN diagnosis and intervention — NOT IMPLEMENTED

> **[2026-03-12 meeting]** **Conditional on stratified eval.** Jim: "if we see failure... we fix it. If not, we're good."

**Goal:** Understand and reduce high FP/FN ratio (model over-predicts co-occurrence).

See `paper_outline_v2.md` Sections 3.5.2–3.5.4 for the error characterization framework,
diagnostics, and the full list of data-centric and model-centric interventions.

**Key principle:** Data-centric and model-centric approaches are **parallel options**, not
sequential. The stratified analysis framework (`paper_outline_v2.md` Section 3.5.1) is the
constant — every intervention is evaluated against the same pair-type breakdown.

#### Diagnostics (no retraining)

Three analyses that guide whether to pursue data-centric or model-centric mitigation:

1. **Embedding distance distribution analysis.** Plot the distribution of pairwise distances
   (or cosine similarity) for: (a) positive pairs, (b) within-subtype negatives, (c)
   cross-subtype negatives. If within-subtype negatives heavily overlap with positives in
   embedding space, the problem is representational — the features don't separate them. If
   there's separation but the classifier draws the boundary wrong, the problem is the
   decision boundary (threshold or model).

2. **Predicted probability histogram.** Plot P(co-occurring) for TPs, FPs, FNs, TNs
   separately. If FPs have high confidence (P > 0.9), the model genuinely can't distinguish
   them — need better features or harder training signal. If FPs cluster near the threshold
   (P ≈ 0.5–0.6), threshold tuning or calibration may suffice.

3. **Pair-level metadata matrix.** For each negative pair, record (subtype_a, subtype_b,
   host_a, host_b, year_a, year_b). Compute FP rate per (subtype_a × subtype_b) cell.
   Confirms whether FPs concentrate in within-subtype negatives (hypothesis: they do,
   because cross-subtype negatives are trivially distinguishable).

**Concrete anchors from the 4 Task 11 sweeps (2026-04):**
- ESM-2 FP/FN ratio is 2–3× higher than k-mer on the same pairs (5.87 vs 2.36 unfilt;
  4.58 vs 3.06 H3N2). ESM-2 is the natural target for FP/FN mitigation experiments.
- The ESM-2 PB1/PA collapse (AUC≈0.50 in both unfilt and H3N2 sweeps) is a ready-made
  diagnostic case. Start embedding distance + probability histogram + metadata-matrix
  analyses there — the pair is already known to fail catastrophically with ESM-2 concat
  while succeeding with k-mer concat on the same data, isolating feature geometry as
  the cause.

**Effort:** Low. No retraining needed — uses existing predictions and metadata files.
Implement as an analysis script.

#### Data-centric approaches

Fix val/test; modify training data only.

**(a) Hard negative mining** (highest priority). Replace easy cross-subtype negatives with
within-subtype negatives (same subtype, same host, same year). Directly attacks the core
problem: force the model to distinguish pairs that are biologically similar but not
co-occurring. Modify `create_negative_pairs` in `dataset_segment_pairs.py` to constrain
negative sampling by metadata match.

**(b) Negative-to-positive ratio.** Currently 1:1. Increasing negatives (e.g., 3:1 or 5:1)
with emphasis on hard negatives gives the model more examples of the failure case. Simple
to implement — change pair generation.

**(c) Curriculum learning on negatives.** Start training with easy (cross-subtype) negatives,
progressively introduce harder (within-subtype) negatives. Avoids the model getting stuck
early on indistinguishable pairs. Requires epoch-aware data loading.

**(d) Subtype-balanced training.** If a subtype is underrepresented in training and the model
fails on that subtype's pairs, upsample those isolates (or downsample dominant subtypes) to
equalize subtype representation.

| Approach | Key idea | Effort |
|----------|----------|--------|
| Hard negative mining | Replace easy negatives with within-subtype | Low (~50 lines) |
| Controlled negative ratio | Within-subtype vs cross-subtype as tunable param | Low |
| Negative-to-positive ratio | Increase from 1:1 (e.g., 3:1) | Low |
| Subtype-balanced training | Equalize subtype representation | Low–medium |
| Curriculum learning | Easy → hard negatives over epochs | Medium |

#### Model-centric approaches

**(a) Focal loss.** Down-weights easy examples (confident correct predictions), up-weights
hard examples. Shifts training attention to the hard cases (within-subtype negatives the
model gets wrong). One-line change in loss function: replace BCEWithLogitsLoss with focal
loss variant.

**(b) Weighted BCE.** Increase the weight on FP errors (higher cost for predicting positive
when negative). Directly penalizes the observed failure mode. Simpler than focal loss but
less adaptive.

**(c) Contrastive learning (supervised contrastive or triplet loss).** Instead of BCE on
pair labels, explicitly optimize the representation space so co-occurring pairs are pulled
together and non-co-occurring pairs are pushed apart. Key difference from BCE: contrastive
loss directly shapes the *geometry* of the representation, not just the classifier's
decision boundary.

- **Pro:** Directly addresses representational overlap. If within-subtype negatives overlap
  with positives in feature space, contrastive learning pushes them apart.
- **Con:** Requires architectural rethinking. The current pipeline computes interaction
  features then classifies. Contrastive learning would operate on embeddings *before* the
  interaction, or on the interaction features directly. ESM-2 embeddings are frozen, so
  contrastive loss only shapes the MLP's learned representation, not the base embeddings.
- **When to try:** If diagnostics show that interaction features fundamentally can't separate
  hard negatives — meaning the problem is representational, not the classifier. This is the
  heavier intervention; save for when simpler approaches fall short.

**(d) Two-stage model.** First stage: coarse filter (reject easy cross-subtype negatives).
Second stage: fine discriminator trained only on hard within-subtype pairs. Separates the
easy and hard problems.

| Approach | Key idea | Effort |
|----------|----------|--------|
| XGBoost/LightGBM | Different inductive bias; may handle hard negatives better without data changes (dual role with Task 4b) | Medium |
| Focal loss | Down-weight easy, up-weight hard; one-line change | Low |
| Weighted BCE | Higher cost for FP errors | Low |
| Contrastive learning | Shape embedding geometry directly | High |
| Multi-task / custom loss | Jointly predict co-occurrence + subtype | High |

#### Recommended execution order

1. **Diagnostics** (embedding distances + probability histogram + metadata matrix). No
   retraining. Tells you whether the problem is representational or decisional.
2. **Hard negative mining** (data-centric). Highest expected impact, no model changes.
3. **Focal loss** (model-centric). One-line change. Complementary to hard negatives.
4. **Contrastive learning** (model-centric). Only if (1)–(3) reveal a fundamental
   representation problem that simpler approaches can't address.

**Status:** Not implemented. FP/FN CSV files exist from training runs.

---

## Experiments: Optional / Lower Priority

### 4b. K-mer + XGBoost/LightGBM — NOT STARTED

> Dual role: (a) optional baseline comparison, (b) model-centric intervention if
> stratified eval reveals pair-type failures (see Task 12).

**Status:** K-mer + MLP done. XGBoost/LightGBM needs new training script.

**K-mer + MLP results:**
- Mixed-subtype HA/NA: AUC 0.982 vs ESM-2 0.966–0.975
- H3N2-only HA/NA: AUC 0.988 (unit_diff) / 0.985 (concat) vs ESM-2 0.957 / 0.498
- Temporal holdout: AUC 0.941 vs ESM-2 0.891
- All-pairs heatmap (28 pairs × 12 folds, 100 epochs, concat):
  k-mer AUC 0.9944 (unfilt) / 0.9931 (H3N2) vs ESM-2 0.9757 / 0.9486.
  ESM-2 collapses fully on PB1/PA in BOTH sweeps (AUC≈0.50); k-mer never collapses.
- Key finding: k-mer concat is filter-invariant and pair-invariant. ESM-2 concat fails
  on specific protein pairs regardless of subtype filtering (the subspace-offset
  failure is geometry-specific, not H3N2-specific).

---

### 5. Accuracy vs genetic distance — NOT IMPLEMENTED

Needs clade/lineage metadata. Lower priority. **Effort:** Medium.

---

### 6. PB2/PB1 on H3N2 — OPTIONAL

Subsumed by Task 11. If 8×8 heatmap completed, this is redundant.
**Effort:** Low (one bundle).

---

### 8. Large test set evaluation — NOT IMPLEMENTED

Evaluate 5K-trained model on ~106K held-out isolates for tighter confidence intervals.
Needs inference script + dataset exclusion logic. **Effort:** Medium.

---

### 9. Learning curve — NOT IMPLEMENTED

Performance vs training set size (100 → 100K isolates). Needs LC launcher + fixed test
set + aggregation. Benefits from Task 8. **Effort:** Medium–high.

---

### 10. GenSLM embeddings — NOT IMPLEMENTED

Out of scope for Paper 1. Deferred to Paper 2. Needs new embedding pipeline.
**Effort:** High (potentially Aurora).

---

## Decision Tree and Execution Order

> **[2026-03-12 meeting]** Jim: "if we scale up and if the accuracies hold, then writing up the paper is pretty much a formality."

1. ~~Scale up to ~50–100K isolates (Task 2)~~ — done; Task 11 ran on full ~111K dataset.
2. ~~Run 8×8 all-segment heatmap with CV (Task 11)~~ — done; results in `allpairs_prod_20260408_063203/`.
   ESM-2 and H3N2-filter variants complete (4 sweeps total, Apr 2026). Feature-source
   comparison now available for stratified analysis.
3. **(NEXT)** Stratified error analysis by pair type and metadata (unconditional)
4. Carla's mixed-subtype discrimination test
5. If good across all pair types → no interventions needed
6. If failures → parallel data-centric + model-centric interventions (Task 12)
7. UQ: deep ensemble + calibration (required, last)
8. Use cases (deferred until above is solid)

**Codebase priorities:**
- Pair-distribution ledger output (unconditional)
- Stratified evaluation script (unconditional)
- 28-bundle job array for Task 11
- Subtype-balanced sampling (conditional lever)

---

## Summary Table

| Task | Status | Effort | Priority | Meeting |
|------|--------|--------|----------|---------|
| **2. Large dataset** | Partially supported | Scale + ledger | **High** | Jim's #1 |
| **11. All pairs (8×8)** | **Complete (2026-04-08 k-mer / 2026-04-19 ESM-2 + H3N2 variants)** | — | **High** | Jim's #2 |
| **NEW: Stratified eval** | Partial (Level 2 only) | Extend `analyze_errors_by_metadata` | **High (unconditional)** | Core deliverable |
| **NEW: Mixed-subtype** | Not started | Dataset + eval | **High** | Carla's test |
| 1. Cross-validation | Implemented | Run end-to-end | High | Default |
| 3. Temporal holdout | Implemented | Fix dedup, re-run | High | Reframed |
| 7. Baseline validation | Not implemented | Config + perturbation | High | Must-have |
| **NEW: UQ** | Not started | Ensemble + calibration | **Required (last)** | Elevated |
| 12. Interventions | Not started | Diagnostics + fixes | **Conditional** | Triggered by eval |
| 4b. XGBoost | Not started | New script | Optional / intervention | — |
| 5. Genetic distance | Not started | Needs metadata | Optional | — |
| 6. PB2/PB1 | Supported | One bundle | Optional (subsumed) | — |
| 8. Large test set | Not implemented | Inference + exclusion | Optional | — |
| 9. Learning curve | Not implemented | Launcher + plotting | Optional | — |
| 10. GenSLM | Not started | New pipeline | Optional (Paper 2) | — |

---

## Next Steps

**Done:**
- ~~Scale to full dataset (~111K isolates)~~ — validated on Lambda (k-mer concat, 10-fold CV). See Runtime Analysis.
- ~~Run CV end-to-end~~ — CV10 completed for: k-mer concat (5K + full), ESM-2 concat (5K), H3N2 variants (k-mer + ESM-2). Results in `models/flu/July_2025/cv_runs/`.
- ~~Run Task 11 (all 28 protein pairs)~~ — Complete on Polaris (12-fold CV, full dataset, 100 epochs). See Task 11 above and `polaris_plan.md` Phase 3 results.
- ~~Run Task 11 ESM-2 and H3N2-filter variants~~ — all 28 pairs × 12 folds complete for
  val_unfilt_esm2, val_h3n2, val_h3n2_esm2 (Apr 2026). Post-hoc artifacts generated for
  all 4 sweeps. Confirms ESM-2 concat collapse is not H3N2-specific (PB1/PA collapses
  on full dataset too).

**Remaining:**
1. **Implement pair-distribution ledger** for every dataset.
2. **Implement stratified evaluation script** (Levels 1–3).
3. **Run stratified analysis** — determine if model performs well or needs fixes.
4. **Implement Carla's mixed-subtype test.**
5. **If failures** → parallel interventions (Task 12).
6. **Fix pair_key dedup** for temporal holdout; re-run.
7. **Implement UQ** — ensemble inference + calibration. Last required step.
8. **Wastewater data sourcing** — Carla → Rachel Paretsky. Jim → assembled samples.
9. **Carry HPC lessons forward** — for any future Polaris scaling task (Tasks 8, 9, GenSLM),
   consult `docs/hardware_notes.md` first to avoid re-discovering the `pin_memory` and TF
   pre-allocation footguns.

---

## Appendix: Biology Context from 2026-03-12 Meeting

> Reference material for understanding the problem. Not paper prose.

### Terminology used in this paper

- **subtype-balanced** — the dataset has been explicitly resampled so that HN subtypes are approximately equally represented across splits (via downsampling dominant subtypes).
- **subtype-filtered** — the dataset has been restricted to a single HN subtype (e.g., H3N2-only).
- **natural** or **unfiltered** — the dataset reflects the actual (BV-BRC) frequencies without resampling; dominated by H3N2, H1N1, H5N1.

These three terms are not interchangeable. In particular, "H3N2-only" is a filter, not a form of balance, despite being trivially "balanced" in the degenerate single-category sense.

### H-type vs N-type vs HN subtype terminology

An H segment has an **H-type** (H1, H3, H5, etc.) and an N segment has an **N-type**
(N1, N2, etc.). The HN **subtype** (e.g., H3N2) is the combination assigned to the
*isolate*. Jim: "the H is the hemagglutinin protein on segment four... the N is
neuraminidase on segment six... you could have, in nature, an H3N1 or an H5N2... a
crisscross." Via reassortment, unusual combinations exist — they are rare subtypes, not
errors.

### Segment variability and immune selection

HA and NA are the most variable segments (immune targets). Internal segments (PB2, PB1,
PA, NP, M1, NS1) are more conserved. The 8×8 heatmap will show a gradient: HA/NA pairs
easiest to match, conserved pairs harder.

### Wastewater assembly challenges

Assembly against a reference can produce chimeric contigs when multiple genomes are
present. Upstream demixing (e.g., Freyja) may be needed before segment matching. Not
solved in this paper.

### Segment identification from metagenomics

Segment identity (H vs N vs PB2 etc.) is easy to determine from metagenomics (BLAST,
length). HN subtype requires knowing which segments co-occur — exactly what our model
predicts.

### Reads-based approach (out of scope)

Jim raised working from reads directly. Carla: "much, much tougher problem... get this
out first." Deferred to future work.
