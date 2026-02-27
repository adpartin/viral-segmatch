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
- Stage 4: `src/models/train_esm2_frozen_pair_classifier.py` → `models/flu/{version}/runs/training_{bundle}_{ts}/` (per experiment)
- Shell wrappers: `scripts/stage2_esm2.sh`, `scripts/stage3_dataset.sh`, `scripts/stage4_train.sh`

## Config System
Hydra + bundle-per-experiment. `conf/bundles/{bundle}.yaml` = one file per named experiment.
Bundle naming: currently inconsistent across generations. Planned general signature (not yet enforced):
  `{virus}_{proteins}[_{n_isolates}][_{pre_mlp_mode}_{interaction}][_{data_filter}]`
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
- `unit_diff` > `concat` on homogeneous data (H3N2-only): AUC 0.96 vs 0.50
- LayerNorm (`slot_norm`) critical for homogeneous subsets
- Delayed learning on H3N2 + unit_diff: increase patience to 40+
- High FP rate on filtered datasets (year/host/geo) -- likely population-level confounders

## Roadmap (02/10/2026 meeting) -- for publication
1. Cross-validation (fold_id/n_folds in dataset config + PBS job array on Polaris) -- IMPLEMENTED (branch: feature/cross-validation)
2. Genome features (k-mers + XGBoost; start from preprocess_bunya_dna.py -> preprocess_flu_dna.py)
3. Large dataset (full Flu A ~100K isolates, HPC)
4. Temporal holdout (year_train/year_test config fields)
5. PB2/PB1 + H3N2 bundle (trivial, one new bundle)

## Directory Structure (post-cleanup Feb 2026)
- `.claude/` -- settings.json (permissions) + memory.md (this file)
- `eda/` -- exploratory scripts (bunya EDA moved here; NOT pipeline)
- `examples/` -- HuggingFace reference scripts (NOT pipeline)
- `old_scripts/` -- superseded scripts (NOT maintained)
- `flu_genomes_eda.py` stays in `src/preprocess/` -- generates flu_genomes_metadata_parsed.csv (pipeline input)

## Not Maintained
- `old_scripts/`, `src/preprocess/preprocess_bunya_protein.py`, `conf/bundles/bunya.yaml`

## In Development
- `src/preprocess/preprocess_bunya_dna.py` -- template for preprocess_flu_dna.py
- `src/utils/dna_utils.py` -- DNA QC utilities
- Temporal holdout split logic (year_train/year_test)

## HPC
- For 8-GPU dev cluster (no scheduler): Python subprocess launcher with CUDA_VISIBLE_DEVICES per fold.
- Polaris (ALCF), PBS job arrays. Do NOT use Hydra's submitit launcher (SLURM only).

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

## What's Next
- Debug/test cross-validation end-to-end (see CV section above)
- Bundle naming cleanup (rename existing bundles to consistent signature) -- deferred, future task
- Use `git log --oneline -10` for current commit state; this file tracks decisions, not commits

## User Preferences
- Concise responses, no emojis unless asked
- No unnecessary refactoring beyond what's asked
- Always ask before destructive operations (rm, git reset --hard, git push --force, etc.)
- CLAUDE.md is the authoritative project context; .claude/memory.md is the compact working memory
- Both files are in the repo -- update them when decisions change
