# Roadmap: Experiments and Publication Scope (02/10/2026)

This document summarizes the experiments decided in the 02/10/2026 meeting, ALCF HPC options, design considerations, and a path toward publication.

---

## Applications

1. **Data remediation.** BV-BRC ingests genomic records from NCBI GenBank, which does not enforce metadata fields linking segments to the same viral isolate. Records may lack shared isolate IDs or have inconsistent naming. A segment-matching model can computationally re-link orphaned segments, improving the quality of large genomic databases.

2. **Wastewater surveillance.** Wastewater sequencing recovers mixed viral fragments from entire communities with no isolate-level metadata. A segment-matching model can help reconstruct which segments likely came from the same circulating strain, turning fragmented wastewater signal into actionable surveillance data. See [Wolfe et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412440/) for CDC influenza A monitoring in US wastewater during the 2024 avian flu outbreak.

3. **Reassortment detection and surveillance.** Segmented viruses like influenza swap genome segments between strains (reassortment), a key driver of pandemics. A model that learns which segment pairs naturally co-occur can flag unusual combinations that may indicate reassortment events, helping public health agencies prioritize emerging strains for closer monitoring.

---

## ALCF HPC Systems

| System | Status | Specs | Best for |
|--------|--------|-------|----------|
| **Polaris** | Available (primary) | 560 nodes, 4× NVIDIA A100 per node, 32-core AMD EPYC, 512 GB RAM/node, 34 PF peak | Cross-validation (many parallel jobs), large-scale training, GPU workloads |
| **Aurora** | Launched Jan 2025 | 60K+ GPUs, exascale-class | Extremely large training (full dataset), GenSLM-scale models |

**Recommendation:** Polaris is the most realistic for Tasks 1 and 2. It supports job arrays and batch scheduling (PBS/Slurm). Aurora is for exascale problems; our current workloads (5K–50K isolates, ESM-2 embeddings, MLP) fit well on Polaris. Apply for Aurora only if scaling to full BV-BRC Flu A (100K+ isolates) with heavy compute.

---

## Experiments (02/10/2026)

### 1. Cross-validation (N splits) — IMPLEMENTED

**Goal:** Replace single train/val/test with N splits; report mean ± std of metrics.

**Status:** Implemented. `n_folds`/`fold_id` in dataset config, `scripts/run_cv_lambda.py` launcher,
`scripts/aggregate_cv_results.py`. Bundle: `flu_schema_raw_slot_norm_unit_diff_cv5`.
See `.claude/memory.md` for full implementation details. Needs a full end-to-end run.

**HPC fit:** High. N independent training runs can run in parallel as a job array.

---

### 2. Train on much larger dataset (full or near-full) — SUPPORTED (not yet run)

**Goal:** Scale beyond 5K isolates to full Flu A (or a large subset).

**Status:** Supported by existing pipeline. Set `max_isolates_to_process: null` in a bundle.
Not yet tested at 100K+ scale. Stage 2 (ESM-2 embeddings) is the bottleneck — likely needs
HPC (Polaris) for full dataset. Stage 4 (MLP training) scales easily.

**HPC fit:** High. Polaris A100s handle ESM-2 well.

---

### 3. Generalize into the future (train 2021–2023, test 2024) — IMPLEMENTED

**Goal:** Temporal holdout to assess generalization to future seasons.

**Status:** Implemented. `year_train`/`year_test` config fields, `generate_temporal_split()`
in `dataset_segment_pairs.py`. Bundles: `flu_schema_raw_slot_norm_unit_diff_temporal` (ESM-2),
`flu_schema_raw_kmer_k6_slot_norm_unit_diff_temporal` (k-mer). Initial runs complete.
See `docs/plans/temporal_holdout_plan.md` for full analysis and results.

**Initial results (March 2026):**

| Metric | ESM-2 (unit_diff) | K-mer k=6 (unit_diff) |
|--------|-------------------|----------------------|
| AUC-ROC | 0.891 | **0.941** |
| F1 | 0.734 | **0.832** |
| Precision | 0.583 | **0.729** |
| Recall | 0.989 | 0.969 |

K-mers substantially outperform ESM-2 on temporal generalization. Both show AUC drop vs
random splits (~0.97), confirming genuine temporal difficulty from subtype distribution
shift (H5N1 24%→41% in 2024).

**Caveat:** Results are confounded by a pair_key dedup artifact that removed 42% of val/test
pairs and created 25/75 label imbalance. Needs re-run with dedup disabled for temporal mode.
See plan doc for details.

**HPC fit:** Low. Single dataset + training run; no HPC required.

---

### 4. K-mer baselines (XGBoost/LightGBM) — PARTIALLY IMPLEMENTED

**Goal:** Evaluate k-mer features with tree-based models as a baseline alongside the MLP.

**Status (March 2026):**
- K-mer + MLP baseline — implemented and tested (`feature_source=kmer` in training script).
  Results match or exceed ESM-2 on all tested configurations.
- K-mer + XGBoost/LightGBM — not started (new training script needed).

**Results (k-mer + MLP):**
- Mixed-subtype: k-mer AUC 0.982 vs ESM-2 AUC 0.966–0.975.
- H3N2-only: k-mer AUC 0.988 (unit_diff) / 0.985 (concat) vs ESM-2 AUC 0.957 (unit_diff) / 0.498 (concat).
- Temporal holdout: k-mer AUC 0.941 vs ESM-2 AUC 0.891.
- Key finding: **k-mer concat does NOT collapse on H3N2** — the ESM-2 concat failure is specific to
  ESM-2’s embedding geometry, not concatenation as an interaction. K-mer features are interaction-agnostic.

**Remaining work:** k-mer + XGBoost/LightGBM (new training script).

---

### 5. Accuracy vs genetic distance (clade) — Marcus’s suggestion — NOT IMPLEMENTED

**Goal:** Analyze whether accuracy degrades with increasing genetic distance (e.g., by clade).

**Status:** Not implemented. Needs clade/lineage metadata from BV-BRC. No code exists.

**Ideas:**
- Join predictions with phylogenetic/clade metadata (if available in BV-BRC or external sources).
- Stratify test set by genetic distance bins (e.g., within-clade vs between-clade pairs).
- Plot: accuracy (or F1) vs distance bin; or accuracy vs pairwise SNP distance.

**Effort:** Medium. Depends on metadata availability. Lower priority.

---

### 6. PB2/PB1 performance on H3N2-only — Jim’s suggestion — now OPTIONAL

**Goal:** When restricted to H3N2 sbutype, test whether conserved segments (PB2, PB1) produce better performance than (HA, NA). 

**Status:** Supported by existing pipeline. Just needs a new bundle YAML. No code changes.
Now that k-mers with unit_diff achieve high performance across configurations (including
temporal holdout), this experiment is lower priority — the main story is already strong
without additional protein combinations.

**Implementation:** New bundle, e.g. `flu_schema_raw_slot_norm_unit_diff_pb2_pb1_h3n2`:
- `virus.selected_functions`: PB2, PB1 (and optionally PA).
- `dataset.hn_subtype`: H3N2.
- Reuse existing schema-ordered pair logic.

**Effort:** Low. One new bundle.

---

## Experiments (Optional)

### 7. Baseline validation experiments — NOT IMPLEMENTED

**Goal:** Confirm the model learns from actual sequence-level features rather than exploiting
shortcuts (e.g., protein-identity pattern: "slot A = HA, slot B = NA → 1").

**Status:** Plan drafted in `docs/plans/baseline_validation_experiments_plan.md`. Not implemented.

**Proposed baselines (priority order):**
1. **Embedding shuffle within protein** — randomly permute embeddings across isolates within
   each protein group. Preserves protein identity but breaks isolate↔sequence link. If model
   still performs well, it’s using slot identity, not sequence content.
2. **Mean embedding per protein** — replace all HA embeddings with HA mean, all NA with NA mean.
   Stronger version of (1): removes all within-protein variation.
3. **Swap-slot test** — for positive pairs, swap slot A and slot B. Tests whether `unit_diff`
   direction matters (it should: A−B ≠ B−A).
4. **Random embeddings** — replace embeddings with random vectors from per-protein Gaussian.

**Implementation approach:** Add `training.ablation.shuffle_embeddings` (etc.) config flags.
Apply perturbation during training so we test whether the model can *learn* from corrupted
features, not just whether a trained model is robust. ~50-100 lines in embedding loading code.

**Effort:** Medium. Config flags + embedding perturbation logic + one bundle per ablation.

---

### 8. Large test set evaluation — NOT IMPLEMENTED

**Goal:** Evaluate the 5K-trained model on a much larger held-out test set to get tighter
confidence intervals on performance metrics.

**Approach:**
1. Train on the standard 5K-isolate dataset (0.8/0.1/0.1 split → ~4K train, 500 val, 500 test).
2. Take the remaining isolates (~111K − 5K = ~106K) that were never seen during training.
3. Create a large test-only dataset from those remaining isolates (positive pairs from
   co-occurring segments, negative pairs with co-occurrence blocking — same logic as Stage 3).
4. Run the trained model in inference mode on this large test set.

**Why this matters:**
- Current test sets (~500 pairs) have wide confidence intervals on metrics.
- A 100K+ test set gives near-exact performance estimates.
- Tests generalization beyond the 5K training population without retraining.

**Codebase changes required:**
1. **Inference script** — or add `--inference_only --model_dir` mode to the training script.
   Load trained model + optimal threshold, run on a provided dataset, save predictions + metrics.
2. **Stage 3 modification** — option to exclude a list of isolates (the 5K used in training)
   when creating the large test dataset. Could use `--exclude_isolates_file` pointing to
   `sampled_isolates.txt` from the training dataset run.
3. **Co-occurrence blocking** — the large test set must not contain contradictory pairs
   (same sequence pair labeled both positive and negative). Existing blocking logic handles this.

**Effort:** Medium. Inference mode + dataset exclusion logic.

---

### 9. Learning curve analysis — NOT IMPLEMENTED

**Goal:** Plot model performance on a fixed test set as training set size increases. Shows
how much data the model needs and whether performance is saturating or still improving.

**Approach** (following Partin et al., BMC Bioinformatics 2021, "Data partitioning" section):
1. Fix a held-out test set (e.g., 10% of isolates, or the large test set from Task 8).
2. Define a geometric sequence of training set sizes: e.g., 100, 250, 500, 1K, 2.5K, 5K,
   10K, 25K, 50K, 100K isolates.
3. For each size, sample that many isolates from the training pool, create pairs, train the
   model, and evaluate on the fixed test set.
4. Plot: test F1 / AUC vs training set size (log-scale x-axis). Optionally with error bars
   from multiple random samples per size.

**Why this matters:**
- Reveals whether 5K isolates is sufficient or whether scaling to 100K improves performance.
- If the curve plateaus early, the model has enough data; if it’s still rising, larger datasets
  help. This directly informs whether Task 2 (large dataset on HPC) is worthwhile.
- Standard analysis for ML papers; reviewers expect it.

**HPC fit:** High. Each training-set size is an independent run; sizes >10K need HPC for
Stage 2 (embeddings). Natural fit for PBS job arrays on Polaris.

**Codebase changes required:**
1. **LC launcher script** — similar to `run_cv_lambda.py`. Iterates over training set sizes,
   creates a dataset for each size (or subsamples from a pre-built large dataset), trains,
   evaluates on the fixed test set.
2. **Dataset subsampling** — `dataset_segment_pairs.py` already has `max_isolates_to_process`.
   Need to ensure the test set is fixed across all sizes (use `test_isolates_override`).
3. **Aggregation script** — collect metrics across sizes, plot LC curve.

**Effort:** Medium–high. Launcher + fixed-test-set logic + aggregation/plotting. Benefits from
Task 8 (large test set) being implemented first.

---

### 10. GenSLM embeddings — NOT IMPLEMENTED

**Goal:** Evaluate genome-level learned representations (GenSLM) as an alternative to
protein-level ESM-2 embeddings and k-mer features.

**Status:** Not started. Requires a new embedding pipeline (GenSLM tokenization + inference).
See `docs/genome_pipeline_design.md` for design context.

**Effort:** High. New embedding pipeline, potentially HPC (Aurora) for large-scale inference.

---

### 11. All protein-pair combinations — NOT IMPLEMENTED

**Goal:** Run experiments across all pairwise combinations of 8 major Flu A proteins (PB2,
PB1, PA, HA, NP, NA, M1, NS1 — one primary gene product per segment; M2 and NEP excluded
as alternative reading frame products). There are C(8,2) = 28 unique pairs. This reveals
which segment pairs are easiest/hardest to match and whether the model generalizes beyond
HA/NA.

**Status:** Not started. Supported by existing pipeline — each pair is a new bundle with
different `virus.selected_functions`. No code changes needed, only bundle generation and
a workflow to run all combinations.

**Implementation:**
- Generate 28 bundles programmatically (or a single parameterized bundle with overrides).
- HPC workflow on Polaris: PBS job array where each job trains one protein pair.
- Aggregate results into an 8×8 heatmap of AUC/F1 across all pairs.

**HPC fit:** High. 28 independent training runs; natural fit for a PBS job array on Polaris.

**Effort:** Low–medium. Bundle generation + job array script + results aggregation/plotting.

---

### 12. FP/FN ratio diagnosis and mitigation — NOT IMPLEMENTED

**Goal:** Understand and reduce the high FP/FN ratio (model over-predicts co-occurrence).
This is critical for the data remediation application, where a false link is worse than a
missed link. The approach combines diagnostics, data-centric fixes, and model-centric fixes,
applied in order of increasing complexity.

#### Diagnostics (before committing to a fix)

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

**Effort:** Low. No retraining needed — uses existing predictions and metadata files.
Implement as an analysis script.

#### Data-centric approaches

Modify what the model sees without changing architecture or loss. Fix the val and test
sets; modify only the training set composition, retrain, and evaluate on the same
held-out data.

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

**(d) Subtype-balanced training.** If H1N1 is underrepresented in training and the model
fails on H1N1 pairs, upsample H1N1 isolates (or downsample H3N2) to equalize subtype
representation.

**Effort:** Low–medium. Hard negative mining is the main code change (~50 lines in
negative pair generation). Others are config or data loader changes.

#### Model-centric approaches

Change how the model learns from the same data.

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
- **When to try:** If diagnostics (step 1) show that interaction features fundamentally
  can't separate hard negatives — meaning the problem is representational, not the
  classifier. This is the heavier intervention; save for when simpler approaches fall short.

**(d) Two-stage model.** First stage: coarse filter (reject easy cross-subtype negatives).
Second stage: fine discriminator trained only on hard within-subtype pairs. Separates the
easy and hard problems.

#### Recommended execution order

1. **Diagnostics** (embedding distances + probability histogram + metadata matrix). No
   retraining. Tells you whether the problem is representational or decisional.
2. **Hard negative mining** (data-centric). Highest expected impact, no model changes.
3. **Focal loss** (model-centric). One-line change. Complementary to hard negatives.
4. **Contrastive learning** (model-centric). Only if (1)–(3) reveal a fundamental
   representation problem that simpler approaches can't address.

**Status:** Not implemented. FP/FN detailed CSV files exist from training runs.

---

## Design: Publication Scope and Codebase Stability

**Your intent:** Run a subset of experiments soon, draft a paper outline, and use it to constrain scope and avoid endless “what if” experiments. Changes to the codebase should support these experiments without unnecessary churn.

**Recommendation:**

1. **Paper outline first.** Draft a 1–2 page outline with: (a) main question, (b) methods (protein embeddings + MLP, possibly k-mers), (c) experiments (baseline, cross-validation, temporal holdout, scale, genome baseline), (d) results structure. Circulate to the team before implementing everything.

2. **Prioritize for quick wins:**
   - **Task 3 (temporal holdout):** Small config change, strong story for “generalization to future.”
   - **Task 6 (PB2/PB1 + H3N2):** Trivial; one bundle.
   - **Task 1 (cross-validation):** Important for robust metrics; enables “mean ± std” in the paper.
   - **Task 2 (large dataset):** Good for “scale” narrative; run after CV is in place.

3. **Defer or phase:**
   - **Task 4 (genome):** Separate thread. Start with k-mers + LightGBM as a baseline; GenSLM later.
   - **Task 5 (genetic distance):** Nice-to-have analysis; add if metadata is available and time permits.

4. **Codebase changes that support all of the above:**
   - Dataset: `year_train` / `year_test` (or equivalent) for Task 3.
   - Dataset: `n_splits` or fold IDs for Task 1.
   - Workflow: job array or parsl/Snakemake for Tasks 1 and 2.
   - Keep existing bundles and pipeline stable; add new bundles and stages rather than rewriting core logic.

---

## Summary Table

| Task | Status | Remaining effort | Priority |
|------|--------|-----------------|----------|
| 1. Cross-validation | Implemented | Run end-to-end | High |
| 2. Large dataset | Supported | Scale testing on HPC | High |
| 3. Temporal holdout | Implemented | Fix pair_key dedup, re-run | High |
| 4. K-mer + MLP | Done | — | Done |
| 4b. K-mer + XGBoost | Not started | New training script | Medium |
| 5. Genetic distance | Not started | Needs metadata | Low |
| 6. PB2/PB1 + H3N2 | Supported | One new bundle YAML | Optional |
| 7. Baseline validation | Not implemented | Config flags + perturbation logic | Optional |
| 8. Large test set eval | Not implemented | Inference mode + dataset exclusion | Optional |
| 9. Learning curve | Not implemented | LC launcher + fixed test set + plotting | Optional |
| 10. GenSLM | Not started | New embedding pipeline | Optional |
| 11. All protein pairs | Not started | Bundle generation + HPC job array | Optional |
| 12. FP/FN mitigation | Not started | Diagnostics + hard negatives + focal loss | High |

---

## Next Steps

1. **Fix pair_key dedup for temporal holdout** — disable dedup for temporal mode, re-run for clean metrics.
2. Run Task 1 (cross-validation) end-to-end for robust mean ± std metrics.
3. Run Task 2 (large dataset) on Polaris after CV pipeline is validated.
4. Draft paper outline; constrain scope before adding more experiments.
