# viral-segmatch — Project Memory

This file is version-controlled in the repo (.claude/memory.md) so it is available on every machine.
Claude: read this at the start of every session. Update it when decisions change or new findings emerge.

---

## Project Summary
Flu A viral segment co-occurrence prediction. ESM-2 protein embeddings (frozen) + MLP binary classifier.
Primary virus: Influenza A. Bunya support exists but NOT actively maintained.

## Pipeline (4 stages)
- Stage 1: `src/preprocess/preprocess_flu_protein.py` → `data/processed/flu/{version}/protein_final.csv` (run once)
- Stage 2: `src/embeddings/compute_esm2_embeddings.py` → `data/embeddings/flu/{version}/master_esm2_embeddings.h5` (run once)
- Stage 3: `src/datasets/dataset_segment_pairs.py` → `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/` (per experiment)
- Stage 4: `src/models/train_pair_classifier.py` → `models/flu/{version}/runs/training_{bundle}_{ts}/` (per experiment)
- Shell wrappers: `scripts/stage2_esm2.sh`, `scripts/stage3_dataset.sh`, `scripts/stage4_train.sh`

## Config System
Hydra + bundle-per-experiment. `conf/bundles/{bundle}.yaml` = one file per named experiment.
Bundle naming: currently inconsistent across generations. Planned general signature (not yet enforced):
  `{virus}_{proteins}[_{n_isolates}][_{slot_transform}_{interaction}][_{data_filter}]`
  e.g. `flu_ha_na_5ks_slot_norm_unit_diff_h3n2` -- renaming existing bundles is a future task.
Config loader: `src/utils/config_hydra.py` via `hydra.compose(config_name="bundles/{name}")`.
No root config -- bundles are loaded directly. `src/utils/config.py` and `conf/config.yaml` deleted (legacy).

## Bundle Organization (see conf/bundles/README.md for full detail)
- Each bundle has `# STATUS: active|ablation|experimental|legacy|not maintained` header.
- Three generations: Gen1 (flu.yaml base), Gen2 (flu_schema.yaml base), Gen3 (flu_schema_raw_* -- current)
- Base bundles must stay flat (moving them breaks Hydra defaults chains in children)
- `conf/bundles/paper/` reserved for publication experiments
- Best model: `flu_schema_raw_slot_norm_unit_diff` (slot_norm + unit_diff, HA/NA)

## Key Findings
- ESM-2 `unit_diff` > `concat` on homogeneous data (H3N2-only): AUC 0.96 vs 0.50
- K-mer concat does NOT collapse on H3N2 (AUC 0.985) -- concat failure is ESM-2-specific, not interaction-specific
- K-mer dominates ESM-2 on H3N2: k-mer unit_diff AUC 0.988 vs ESM-2 unit_diff AUC 0.957; k-mers are interaction-agnostic
- K-mer (k=6, 4096-dim) matches or exceeds ESM-2 on mixed-subtype HA/NA (AUC 0.982 vs 0.966-0.975)
- LayerNorm (`slot_norm`) critical for ESM-2 on homogeneous subsets
- Delayed learning on H3N2 + unit_diff: increase patience to 40+
- High FP rate on filtered datasets (year/host/geo) -- likely population-level confounders
- **Temporal holdout**: K-mer AUC 0.941 vs ESM-2 AUC 0.891 (train 2021-2023, test 2024); k-mers generalize better across flu seasons

## Roadmap (02/10/2026 meeting + March 2026 updates) -- for publication
1. Cross-validation -- IMPLEMENTED, needs end-to-end run
2. Large dataset (full Flu A ~100K isolates, HPC) -- supported, not yet run
3. Temporal holdout -- IMPLEMENTED, needs dedup fix + re-run
4. K-mer + MLP -- DONE; k-mer + XGBoost/LightGBM still TODO
5. PB2/PB1 + H3N2 bundle -- optional (one new bundle)
11. All protein pairs (C(8,2)=28 pairs of 8 major proteins) -- NOT IMPLEMENTED, HPC
12. FP/FN ratio diagnosis + mitigation -- NOT IMPLEMENTED; see `roadmap_v1.md` Task 12
    - Diagnostics first (embedding distances, probability histograms, pair-level metadata matrix)
    - Data-centric: hard negative mining (highest priority), negative ratio, curriculum learning
    - Model-centric: focal loss, contrastive learning (if simpler approaches fail)

## Publication Strategy (March 2026)
- **Paper 1 (biology, primary):** Segment matching for data remediation + surveillance.
  Target: Bioinformatics / PLOS Comp Bio / Genome Biology. See `paper_outline_v1.md`.
- **Paper 2 (ML, follow-up):** ESM-2 concat collapse + GenSLM. Target: NeurIPS/ICML workshop.
- Paper outline: `paper_outline_v1.md` (v1), `paper_outline_v2.md` (v2, current)
- Applications: data remediation (BV-BRC), wastewater surveillance, reassortment detection (future)

## Directory Structure (post-cleanup Feb 2026)
- `.claude/` -- settings.json (permissions) + memory.md (this file)
- `eda/` -- exploratory scripts (bunya EDA moved here; NOT pipeline)
- `examples/` -- HuggingFace reference scripts (NOT pipeline)
- `old_scripts/` -- superseded scripts (NOT maintained)
- `flu_genomes_eda.py` stays in `src/preprocess/` -- generates flu_genomes_metadata_parsed.csv (pipeline input)

## Not Maintained
- `old_scripts/`, `src/preprocess/preprocess_bunya_protein.py`, `conf/bundles/bunya.yaml`

## Temporal Holdout (IMPLEMENTED — initial runs complete)
- Bundles: `flu_schema_raw_slot_norm_unit_diff_temporal` (ESM-2), `flu_schema_raw_kmer_k6_slot_norm_unit_diff_temporal` (k-mer)
- Train 2021-2023 (~20K isolates), val+test 2024 (~17K isolates)
- Notable subtype shift: H5N1 24%→41%, H3N2 40%→32% (2024 avian flu surge)
- **Pair-key dedup issue**: 42% of val/test pairs removed (positive pair_keys overlap with train due to same strains across years), creating 25/75 label imbalance in val/test. Needs fix before publication — likely disable dedup for temporal mode (approach A). See plan doc.
- **Initial results** (with dedup artifact, threshold=0.5):
  - ESM-2: AUC 0.891, F1 0.734, FP/FN=64.1
  - K-mer (k=6): AUC 0.941, F1 0.832, FP/FN=11.6
  - K-mers substantially outperform ESM-2 on temporal holdout; gap wider than on random splits
  - Both show AUC drop vs random splits (~0.97), confirming genuine temporal difficulty
- See `docs/plans/temporal_holdout_plan.md` for full analysis and results

## In Development
- Unified Flu preprocessing (`preprocess_flu.py`) -- see docs/genome_pipeline_design.md
- `src/utils/dna_utils.py` -- DNA QC utilities (summarize_dna_qc complete, clean_dna_sequences untested)

## HPC
- For 8-GPU dev cluster (no scheduler): Python subprocess launcher with CUDA_VISIBLE_DEVICES per fold.
- Polaris (ALCF), PBS job arrays. Do NOT use Hydra's submitit launcher (SLURM only).
- See `large_run_polaris.md` for Polaris-specific setup: env validation, PBS job script templates, queues, DDP patterns, conda on Lustre, common failure modes.

## Cross-validation (IMPLEMENTED — branch: feature/cross-validation)
### Output structure
- Stage 3 runs ONCE → `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/`
  - Nested: `fold_0/`, `fold_1/`, …, `fold_{N-1}/` each with train/val/test CSVs, stats, plots
  - Top-level `cv_info.json` with fold isolate assignments and seeds
- Stage 4 trains per fold → `models/flu/{version}/runs/training_{bundle}_fold{k}_{ts}/`
  - Each training dir has `test_predicted.csv`, `optimal_threshold.txt`
- After all folds: `cv_run_manifest.json` (dataset run dir), `cv_summary.csv/json`

### Config
- `conf/dataset/default.yaml`: `n_folds: null`, `fold_id: null` (null = single-split, backward compat)
- `conf/bundles/flu_schema_raw_slot_norm_unit_diff_cv5.yaml`: inherits base bundle, adds `n_folds: 5`

### Key implementation details
- `split_dataset()` gains `train/val/test_isolates_override` params (None = existing behavior)
- `generate_all_cv_folds()`: KFold on isolates; `val_frac = val_ratio / (1 - 1/n_folds)` for consistent val size
- Fold seed = `master_seed + fold_i` for reproducible but distinct negative sampling
- `--fold_id` added to training script (optional; appends `fold_{k}/` to `--dataset_dir`)
- `save_split_output()`: extracted helper used by both single-split and CV paths

### Launchers
- `scripts/run_cv_lambda.py`: subprocess.Popen per fold, CUDA_VISIBLE_DEVICES=gpu_k, saves manifest, calls aggregation
- `scripts/run_cv_polaris.pbs`: STAGE=dataset|train|aggregate; train uses PBS_ARRAY_INDEX=fold_id
- `scripts/aggregate_cv_results.py`: reads manifest or --training_dirs, computes mean±std, writes cv_summary.*

### Hydra limitation: no subdirectory bundles
CV bundle is flat (`conf/bundles/flu_schema_raw_slot_norm_unit_diff_cv5.yaml`), not in `paper/`.
Hydra's package resolution double-nests inherited configs from subdirs, breaking `get_virus_config_hydra`.
`conf/bundles/paper/` kept as directory but YAML bundles must stay flat. See README.md in bundles/.

### Next steps for CV
- Run dry run: `python scripts/run_cv_lambda.py --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 --dry_run`
- Run full CV: `python scripts/run_cv_lambda.py --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 --gpus 0 1 2 3 4`

## Stage 3/4 Decoupling (IMPLEMENTED — branch: feature/decouple-dataset-training)
- `stage4_train.sh` requires `--dataset_dir` explicitly; no bundle extraction from path
- `--allow_bundle_mismatch` flag removed (no longer needed)
- Training script saves `training_info.json` with full provenance (config_bundle, dataset_dir, HPs)
- Both shell scripts slimmed to ~60-100 lines matching the lean stage1/stage2b pattern
- Workflow: Stage 3 once → Stage 4 N times with different training bundles

## Task 11: All Protein-Pair Combinations (8x8 Heatmap) — NEXT UP
Goal: 28 pairwise combinations C(8,2) of 8 major proteins (PB2, PB1, PA, HA, NP, NA, M1, NS1).
Full run: 28 pairs × 10 CV folds × ~100 epochs × ~111K isolates (Polaris).
Test run first: subset of pairs, 1-2 folds, 5-10 epochs, 5K isolates.

### Relevant files
- Config: `conf/virus/flu.yaml` (8 proteins in `selected_functions`), `conf/bundles/flu_schema.yaml` (`pair_mode: schema_ordered` + `schema_pair`), `conf/bundles/flu_schema_raw_kmer_k6_slot_norm_concat_cv10.yaml` (k-mer CV template)
- Pipeline: `src/datasets/dataset_segment_pairs.py` (reads `schema_pair`), `src/models/train_pair_classifier.py`, `scripts/run_cv_lambda.py` (CV fold launcher)
- Analysis: `scripts/aggregate_cv_results.py`

### What needs to be built
1. **Bundle generator script** — produce 28 YAML files programmatically, each setting `schema_pair` to a different pair. Takes params for `n_folds`, `epochs`, `max_isolates_to_process` to dial between test/full runs.
2. **Outer launcher** — iterates over protein pairs, calling `run_cv_lambda.py` per pair (Lambda) or PBS job array per pair (Polaris).
3. **Cross-pair aggregation** — collect 28 CV summaries into 8×8 heatmap.

### Key constraint
Current pipeline: one bundle = one protein pair. `run_cv_lambda.py` handles the fold dimension. No outer loop over pairs exists yet. Hydra doesn't support CLI overrides for `schema_pair` — must flow through bundle files.

## What's Next
- **Task 11 (all 28 protein pairs)** -- build bundle generator + outer launcher (see section above)
- Fix pair_key dedup for temporal holdout -- re-run for clean metrics
- Run cross-validation end-to-end (see CV section above)
- FP/FN diagnostics (Task 12) -- understand error distribution before mitigation
- Quantify unlinked BV-BRC records (ask Jim) -- scopes the remediation demo
- Bundle naming cleanup -- deferred, future task

## User Preferences
- Concise responses, no emojis unless asked
- No unnecessary refactoring beyond what's asked
- Always ask before destructive operations (rm, git reset --hard, git push --force, etc.)
- CLAUDE.md is the authoritative project context; .claude/memory.md is the compact working memory
- Both files are in the repo -- update them when decisions change
- **One script per purpose**: follow the existing pattern in `src/analysis/` — propose a dedicated script with a clear name (e.g., `aggregate_cv_results.py`) rather than hedging between existing scripts. Commit to the obvious answer.
