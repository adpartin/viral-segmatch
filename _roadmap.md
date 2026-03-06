# Roadmap: Experiments and Publication Scope (02/10/2026)

This document summarizes the experiments decided in the 02/10/2026 meeting, ALCF HPC options, design considerations, and a path toward publication.

---

## ALCF HPC Systems

| System | Status | Specs | Best for |
|--------|--------|-------|----------|
| **Polaris** | Available (primary) | 560 nodes, 4× NVIDIA A100 per node, 32-core AMD EPYC, 512 GB RAM/node, 34 PF peak | Cross-validation (many parallel jobs), large-scale training, GPU workloads |
| **Aurora** | Launched Jan 2025 | 60K+ GPUs, exascale-class | Extremely large training (full dataset), GenSLM-scale models |

**Recommendation:** Polaris is the most realistic for Tasks 1 and 2. It supports job arrays and batch scheduling (PBS/Slurm). Aurora is for exascale problems; our current workloads (5K–50K isolates, ESM-2 embeddings, MLP) fit well on Polaris. Apply for Aurora only if scaling to full BV-BRC Flu A (100K+ isolates) with heavy compute.

---

## Experiments

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

### 3. Generalize into the future (train 2021–2023, test 2024) — NOT IMPLEMENTED

**Goal:** Temporal holdout to assess generalization to future seasons.

**Status:** Not implemented. The current `year` config is a single-value **filter** (e.g.,
`year: 2024` keeps only 2024 isolates). What's needed is **split** semantics: assign isolates
to train/val vs test based on year, not filter them out.

**HPC fit:** Low. Single dataset + training run; no HPC required.

**What exists today:**
- `dataset.year` filter in `conf/dataset/default.yaml` — filters to a single year
- Metadata enrichment already adds `year` column via `metadata_enrichment.py`
- `split_dataset()` in `dataset_segment_pairs.py` supports `train/val/test_isolates_override`
  (added for CV) — this is the hook for temporal splits

**Codebase changes required:**

1. **Config schema** — add two new fields to `conf/dataset/default.yaml`:
   ```yaml
   dataset:
     year_train: null    # e.g., [2021, 2022, 2023] or "2021-2023"
     year_test: null     # e.g., [2024] or "2024"
   ```
   When both are null (default), fall back to current random isolate-level split.
   These are mutually exclusive with `dataset.year` (filter). If `year_train`/`year_test`
   are set, `dataset.year` should be null (or raise an error if both are set).

2. **Split logic in `dataset_segment_pairs.py`** — add a temporal split path:
   - After metadata enrichment (which adds the `year` column), partition isolates by year.
   - `train_isolates` = isolates with `year in year_train`
   - `test_isolates` = isolates with `year in year_test`
   - `val_isolates` = split off from `train_isolates` using `val_ratio`
   - Pass these to `split_dataset(..., train_isolates_override=..., test_isolates_override=...)`
     which already exists from the CV implementation.
   - The pair generation and co-occurrence blocking logic is unchanged.

3. **Bundle** — one new YAML, e.g. `flu_schema_raw_slot_norm_unit_diff_temporal.yaml`:
   ```yaml
   defaults:
     - flu_schema_raw_slot_norm_unit_diff
     - _self_
   dataset:
     year_train: [2021, 2022, 2023]
     year_test: [2024]
     max_isolates_to_process: null  # use all matching isolates
   ```

4. **Validation** — check that train and test year ranges don't overlap; warn if
   `year_test` isolates are very few (may produce small test sets).

**Effort:** Low–medium. The hard part (isolate-override splits) already exists from CV.
The main work is the config schema, the year-based partitioning logic (~30-50 lines),
and validation. No changes needed to Stage 4 (training) or shell scripts.

---

### 4. Genome as input: k-mers + LightGBM/XGBoost, then GenSLM — PARTIALLY IMPLEMENTED

**Goal:** Explore nucleotide-based features alongside protein embeddings.

**Status (March 2026):**
- ✅ K-mer + MLP baseline — `compute_kmer_features.py` (Stage 2b), `kmer_utils.py`,
  `feature_source=kmer` in training script. Tested on mixed-subtype and H3N2-only.
- ✅ `preprocess_flu.py` — unified protein + genome extraction.
- Not started: k-mer + XGBoost/LightGBM, GenSLM embeddings.

**Results:**
- Mixed-subtype: k-mer AUC 0.982 vs ESM-2 AUC 0.966–0.975.
- H3N2-only: k-mer AUC 0.988 (unit_diff) / 0.985 (concat) vs ESM-2 AUC 0.957 (unit_diff) / 0.498 (concat).
- Key finding: **k-mer concat does NOT collapse on H3N2** — the ESM-2 concat failure is specific to
  ESM-2's embedding geometry, not concatenation as an interaction. K-mer features are interaction-agnostic.

**Remaining work:** k-mer + XGBoost/LightGBM (new training script), GenSLM (new embedding pipeline).
See `docs/genome_pipeline_design.md` for full design.

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

### 6. PB2/PB1 performance on H3N2-only — Jim’s suggestion — SUPPORTED (not yet run)

**Goal:** Test whether conserved segments (PB2, PB1) perform better when restricted to H3N2.

**Status:** Supported by existing pipeline. Just needs a new bundle YAML. No code changes.

**Implementation:** New bundle, e.g. `flu_schema_raw_slot_norm_unit_diff_pb2_pb1_h3n2`:
- `virus.selected_functions`: PB2, PB1 (and optionally PA).
- `dataset.hn_subtype`: H3N2.
- Reuse existing schema-ordered pair logic.

**Effort:** Low. One new bundle.

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
| 1. Cross-validation | ✅ Implemented | Run end-to-end | High |
| 2. Large dataset | Supported | Scale testing on HPC | High |
| 3. Temporal holdout | **Not implemented** | Config + ~50 lines split logic | High |
| 4. K-mer + MLP | ✅ Implemented | — | Done |
| 4b. K-mer + XGBoost | Not started | New training script | Medium |
| 4c. GenSLM | Not started | New embedding pipeline | Low |
| 5. Genetic distance | Not started | Needs metadata | Low |
| 6. PB2/PB1 + H3N2 | Supported | One new bundle YAML | Low |

---

## Next Steps

1. **Implement Task 3 (temporal holdout)** — highest priority gap for publication.
2. Run Task 1 (cross-validation) end-to-end for robust mean ± std metrics.
3. Create Task 6 bundle (PB2/PB1 + H3N2) — trivial.
4. Run Task 2 (large dataset) on Polaris after CV pipeline is validated.
5. Baseline validation experiments (embedding shuffle etc.) — see `docs/plans/baseline_validation_experiments_plan.md`.
6. Draft paper outline; constrain scope before adding more experiments.
