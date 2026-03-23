# Code Cleanup Plan

**Status: ONGOING**

Non-urgent code quality improvements. Items here are not blocking any experiments
but would improve readability, consistency, and maintainability.

---

## 1. Harmonize Stage 2 featurization scripts

**Files**: `src/embeddings/compute_esm2_embeddings.py`, `src/embeddings/compute_kmer_features.py`

Both are Stage 2 featurization scripts but were written at different times and have
different code organization (section separators, CLI layout, logging style, path
resolution). Aligning their structure would make them easier to read side-by-side
and debug consistently.

---

## 2. Add per-code DNA ambiguity breakdown (match protein_utils pattern)

**Files**: `src/utils/dna_utils.py`

`protein_utils.analyze_protein_ambiguities()` provides a detailed per-residue-type
breakdown (X, B, Z, *, terminal vs internal stops, positions, etc.).
`dna_utils.summarize_dna_qc()` currently lumps all IUPAC ambiguity codes (N, R, Y,
S, W, K, M, B, D, H, V) into a single `ambig_count`/`ambig_frac`. We should consider
adding a per-code breakdown so downstream consumers can distinguish "mostly Ns" (general
sequencing uncertainty) from "many two-base ambiguities" (partial base calls).

See the IUPAC reference table in the `dna_utils.py` module docstring.

---

## 3. Slim down shell script wrappers

**Files**: `scripts/stage2_esm2.sh`, `scripts/stage2b_kmer.sh`

**DONE (stage3, stage4)**: `stage3_dataset.sh` and `stage4_train.sh` were rewritten to
match the lean `stage1_preprocess_flu.sh` pattern as part of the Stage 3/4 decoupling work.

**Remaining**: `stage2_esm2.sh` still has the verbose pattern (git provenance blocks,
`log()` helpers, elaborate headers/footers, registry integration). Should be refactored
to match the lean pattern used by stage1, stage2b, stage3, and stage4.

---

## 4. Consider renaming `src/embeddings/` directory

**Files**: `src/embeddings/`

The directory currently holds `compute_esm2_embeddings.py` and `compute_kmer_features.py`.
K-mer frequency vectors are not embeddings in the ML sense (they are hand-crafted features,
not learned representations). A name like `src/featurize/` or `src/features/` would better
reflect that this directory contains Stage 2 scripts that convert raw sequences into numerical
feature vectors — regardless of whether the method is a pretrained model (ESM-2) or a
counting procedure (k-mers). Would also need to update imports, CLAUDE.md, and shell scripts.

Similarly, `conf/embeddings/` (currently just `default.yaml` for ESM-2 settings) should be
renamed to `conf/featurize/` or `conf/features/` for consistency. Do both renames together.

**Note (March 2026):** K-mer scripts (`compute_kmer_features.py`, `kmer_utils.py`) are now
in active use with tested bundles, strengthening the case for this rename.

---

## 5. Revisit Stage 4 training script naming / structure

**Files**: `src/models/train_pair_classifier.py`

The script now supports multiple feature sources (ESM-2, k-mer) via `config.training.feature_source`,
so the name `train_pair_classifier.py` is misleading. Two options to discuss:

- **Rename** to something general like `train_pair_classifier.py` (the script already handles
  both feature types through the same MLP architecture).
- **Keep as-is** for MLP-based training and create a separate script for tree-based models
  (XGBoost/LightGBM) that would be used with large-k k-mers (k=10, 1M-dim features).

Decision depends on whether we want one training entry point or separate scripts per model family.

---

## 6. Remove 'raw' from Gen3 bundle file names

**Scope**: 156 occurrences across 35 files.

The `raw` token in Gen3 bundle names (e.g., `flu_schema_raw_slot_norm_unit_diff`) has no
functional meaning. It is not a config field and does not appear in any Python code logic.
It was introduced as a naming convention to distinguish Gen3 bundles from the Gen2
`flu_schema` base (which defaulted to `slot_transform: none`, `interaction: concat`), but
the distinction is already encoded by the `slot_transform` and `interaction` tokens that
follow it.

Removing `raw` would shorten names without losing information:
- `flu_schema_raw_slot_norm_unit_diff` → `flu_schema_slot_norm_unit_diff`
- `flu_schema_raw_kmer_k6_slot_norm_unit_diff` → `flu_schema_kmer_k6_slot_norm_unit_diff`

**Do NOT change**: `experiments/registry.yaml` (historical provenance),
`old_scripts/` (archived). Existing output directories (`data/datasets/`, `models/`)
will use old names — acceptable since results will be re-run.

### Full file inventory

**Bundle YAML files (20 files, ~20 occurrences):**
Rename files via `git mv`, update Hydra `defaults` chains in child bundles.
- `conf/bundles/flu_schema_raw_adapter.yaml`
- `conf/bundles/flu_schema_raw_shared.yaml`
- `conf/bundles/flu_schema_raw_slot.yaml`
- `conf/bundles/flu_schema_raw_none_unit_diff.yaml`
- `conf/bundles/flu_schema_raw_none_unit_diff_h3n2.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_concat.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_concat_2024.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_concat_h3n2.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_concat_human.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_concat_illinois.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_unit_diff.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_unit_diff_2024.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_unit_diff_cv5.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_unit_diff_h3n2.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_unit_diff_human.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_unit_diff_illinois.yaml`
- `conf/bundles/flu_schema_raw_slot_norm_unit_diff_temporal.yaml`
- `conf/bundles/flu_schema_raw_kmer_k6_slot_norm_concat.yaml`
- `conf/bundles/flu_schema_raw_kmer_k6_slot_norm_concat_h3n2.yaml`
- `conf/bundles/flu_schema_raw_kmer_k6_slot_norm_unit_diff.yaml`
- `conf/bundles/flu_schema_raw_kmer_k6_slot_norm_unit_diff_h3n2.yaml`
- `conf/bundles/flu_schema_raw_kmer_k6_slot_norm_unit_diff_temporal.yaml`

**Bundle docs (31 occurrences):**
- `conf/bundles/README.md` (30 refs — tables, examples)
- `conf/bundles/paper/README.md` (1 ref)

**Python code (21 occurrences — requires careful review):**
- `src/analysis/aggregate_experiment_results.py` (20 refs — likely hardcoded bundle
  names in result aggregation logic; not simple find-replace, needs manual review)
- `src/datasets/dataset_segment_pairs.py` (1 ref — likely a comment or example)

**Shell scripts (12 occurrences):**
- `scripts/stage3_dataset.sh` (1 ref — example in usage comment)
- `scripts/stage4_train.sh` (2 refs — example in usage comment)
- `scripts/run_cv_lambda.py` (6 refs — default bundle name, examples)
- `scripts/run_cv_polaris.pbs` (3 refs — bundle name, examples)

**Project docs (14 occurrences):**
- `CLAUDE.md` (3 refs)
- `.claude/memory.md` (7 refs)
- `roadmap_v1.md` (4 refs)
- `_ongoing_work.md` (6 refs)

**Other docs (48 occurrences):**
- `README.md` (16 refs)
- `docs/cv_run_guide.md` (16 refs)
- `docs/EXP_RESULTS_STATUS.md` (5 refs)
- `docs/PROJECT_REPORTS.md` (1 ref)
- `docs/plans/temporal_holdout_plan.md` (6 refs)
- `docs/plans/baseline_validation_experiments_plan.md` (3 refs)
- `docs/plans/decouple_dataset_training_plan.md` (4 refs)
- `docs/plans/code_cleanup_plan.md` (this file — self-references)

### Execution plan

1. `git mv` all 20+ bundle YAML files
2. Update Hydra `defaults` references in child bundles
3. Bulk find-replace `flu_schema_raw_` → `flu_schema_` in shell scripts and docs
4. **Manual review**: `aggregate_experiment_results.py` (hardcoded bundle names)
5. Verify Hydra config loading works: `python -c "from src.utils.config_hydra import ..."`
6. Commit on a dedicated branch (`refactor/remove-raw-from-bundles`)
