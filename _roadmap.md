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

### 1. Cross-validation (N splits)

**Goal:** Replace single train/val/test with N splits; report mean ± std of metrics.

**HPC fit:** High. N independent training runs (e.g., N=5 or 10) can run in parallel as a job array. Each run: dataset creation → embeddings (if not cached) → training → evaluation.

**Implementation notes:**
- Add `n_splits` (or `cv_folds`) to dataset config; generate N stratified splits by isolate.
- Either: (a) N dataset run dirs + N training runs, or (b) one dataset with fold IDs, training script accepts `--fold N`.
- Aggregate script: read metrics from all fold dirs, compute mean/std, write `cv_summary.csv`.
- Workflow: parsl, Snakemake, or a simple PBS job array that loops over folds.

**Effort:** Medium. Requires dataset + training pipeline changes and an aggregation script.

---

### 2. Train on much larger dataset (full or near-full)

**Goal:** Scale beyond 5K isolates to full Flu A (or a large subset).

**HPC fit:** High. Embedding computation (ESM-2) and training both benefit from multi-GPU or multi-node. Polaris A100s handle ESM-2 well; training is modest.

**Implementation notes:**
- Set `max_isolates_to_process: null` (or a large cap) in dataset config.
- ESM-2 embedding stage: batch across GPUs; may need to chunk the protein list for memory.
- Dataset creation: same logic, larger pair counts. Consider downsampling negatives if full pairwise blows up.
- Training: current MLP scales; main bottleneck is embedding precompute and data loading.

**Effort:** Medium–high. Mostly scaling existing pipeline; watch disk I/O and memory.

---

### 3. Generalize into the future (train 2021–2023, test 2024)

**Goal:** Temporal holdout to assess generalization to future seasons.

**HPC fit:** Low. Single dataset + training run; no HPC required.

**Implementation notes:**
- Current `year` filter is a single value (e.g., `year: 2024`). Need **split** semantics: `year_train: [2021, 2022, 2023]` and `year_test: [2024]` (or `year_holdout: 2024`).
- Dataset logic: filter isolates by `year in train_years` for train/val, `year in test_years` for test. No isolate overlap across splits.
- Add to dataset config, e.g.:
  ```yaml
  dataset:
    year_train: [2021, 2022, 2023]  # or year_range_train: [2021, 2023]
    year_test: [2024]
  ```
- If not set, fall back to current random split by isolate.

**Effort:** Low–medium. Config schema change + dataset_segment_pairs split logic.

---

### 4. Genome as input: k-mers + LightGBM/XGBoost, then GenSLM

**Goal:** Explore nucleotide-based features alongside protein embeddings.

**HPC fit:** Medium. k-mer + tree models: moderate compute, can run on Polaris. GenSLM: GPU-heavy, may need Polaris or Aurora.

**Implementation notes:**
- **preprocess_bunya_dna.py** is the template. Create `preprocess_flu_dna.py` (or `preprocess_flu_genome.py`) for Flu A.
- **Data format for k-mers + XGBoost/LightGBM:**
  - Per-segment k-mer counts (or TF-IDF) → feature matrix.
  - Pair representation: concatenate k-mer vectors for (seg_a, seg_b), or use a siamese-style setup.
  - Output: same schema as protein pairs (assembly_id_a, assembly_id_b, label) with features instead of ESM-2 embeddings.
- **Data format for GenSLM:**
  - GenSLM expects nucleotide sequences; may need codon representation for coding regions.
  - Per-segment nucleotide (or codon) sequences; model produces embeddings.
  - Same pair structure: (emb_a, emb_b) → classifier.
- **Pipeline:** New stage(s): `preprocess_flu_genome` → `compute_kmers` (or `compute_genslm_embeddings`) → dataset/training. Can reuse dataset pair logic with a different feature source.

**Effort:** High. New preprocessing, new feature pipeline, new model code path. Start with k-mers + LightGBM as a simpler baseline.

---

### 5. Accuracy vs genetic distance (clade) — Marcus’s suggestion

**Goal:** Analyze whether accuracy degrades with increasing genetic distance (e.g., by clade).

**Ideas:**
- Join predictions with phylogenetic/clade metadata (if available in BV-BRC or external sources).
- Stratify test set by genetic distance bins (e.g., within-clade vs between-clade pairs).
- Plot: accuracy (or F1) vs distance bin; or accuracy vs pairwise SNP distance.
- Requires: genetic distance or clade assignments per isolate. Check BV-BRC for clade/lineage fields.

**Effort:** Medium. Depends on metadata availability. Lower priority.

---

### 6. PB2/PB1 performance on H3N2-only — Jim’s suggestion

**Goal:** Test whether conserved segments (PB2, PB1) perform better when restricted to H3N2.

**Implementation:** New bundle, e.g. `flu_schema_raw_slot_norm_unit_diff_pb2_pb1_h3n2`:
- `virus.selected_functions`: PB2, PB1 (and optionally PA).
- `dataset.hn_subtype`: H3N2.
- Reuse existing schema-ordered pair logic.

**Effort:** Low. One new bundle; existing pipeline supports it.

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

| Task | HPC | Effort | Priority | Notes |
|------|-----|--------|----------|-------|
| 1. Cross-validation | Yes (Polaris) | Medium | High | Job array, aggregation script |
| 2. Large dataset | Yes (Polaris) | Medium–high | High | Scale existing pipeline |
| 3. Temporal holdout | No | Low–medium | High | Config + split logic |
| 4. Genome (k-mers, GenSLM) | Medium | High | Medium | New pipeline; start with k-mers |
| 5. Accuracy vs genetic distance | No | Medium | Low | Needs clade/distance metadata |
| 6. PB2/PB1 + H3N2 | No | Low | Low | One new bundle |

---

## Next Steps

1. Draft paper outline (1–2 pages); get team alignment.
2. Implement Task 3 (temporal holdout) and Task 6 (PB2/PB1 H3N2 bundle).
3. Implement Task 1 (cross-validation) with Polaris job array.
4. Run Task 2 (large dataset) after CV pipeline is stable.
5. Start Task 4 with `preprocess_flu_genome.py` and k-mer + LightGBM baseline.
