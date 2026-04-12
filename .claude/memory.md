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
- See `polaris_plan.md` for Task 11 plan: phases 0-3, env setup, bundle design, queue strategy, scripts.
- See `speed_up.md` for training optimizations (67% speedup: batch_size=128, eval_train_metrics=false, pin_memory=true).
- See `docs/hardware_notes.md` for HW/SW interaction notes: `pin_memory` + cudaHostAlloc serialization under ensemble packing, `num_workers=0` rationale, TF GPU pre-allocation prevention, L3 cache vs working-set, and `[Extra]` background topics (CUDA async, AMP, CUDA_VISIBLE_DEVICES remapping, pinned-memory budget, ensemble packing vs PBS arrays).
- **Batch vs interactive env**: PBS batch mode doesn't source dotfiles. The fix is `#!/bin/bash -l` (login shell shebang) which loads the full default env (PrgEnv-nvidia, Cray PALS/mpiexec, libfabric, CUDA). No manual PATH or LD_LIBRARY_PATH needed — just conda + venv on top. Interactive mode uses `scripts/polaris_env.sh`.

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

## Task 11: All Protein-Pair Combinations (8x8 Heatmap) — IN PROGRESS
Goal: 28 pairwise combinations C(8,2) of 8 major proteins (PB2, PB1, PA, HA, NP, NA, M1, NS1).
Full run: 28 pairs × 12 CV folds × 100 epochs × ~111K isolates (Polaris).

### What's built
- **Master bundle**: `conf/bundles/flu_28_major_protein_pairs_master.yaml` — Phase 3 production settings (full dataset, 12-fold CV, batch_size=128, optimized training). Children override only `schema_pair`.
- **28 child bundles**: `conf/bundles/flu_28p_{protA}_{protB}.yaml` — generated by `scripts/generate_all_pairs_bundles.py`
- **Phase 0 launcher**: `scripts/run_allpairs_polaris_phase0.sh` — sequential single-GPU for validation
- **Prod launcher**: `scripts/run_allpairs_polaris_prod.sh` — multi-node PBS job (28 nodes, prod queue). Each node runs `run_cv_lambda.py` with 4 GPUs for 12-fold CV.
- **Full plan**: `polaris_plan.md` — phases 0-3, env setup, bundle design, HPC reference

### What needs to be built
1. **Cross-pair aggregation + heatmap** — collect 28 CV summaries into 8×8 AUC/F1 matrix for paper
2. **Resolved config snapshot** — added `save_config()` calls to Stage 3 and Stage 4 (saves `resolved_config.yaml` in output dir). Not yet tested in a run.

### Key constraints
- Master bundle is set to Phase 3 defaults (patience=100, effectively no early stopping). For Phases 0-2, temporarily edit the master.
- Data is on Polaris (copied from Lambda 2026-03-27): `protein_final.csv`, `kmer_k6_*`, ESM-2 embeddings.
- N=12 folds chosen for perfect GPU packing (12/4 = 3 waves, no idle GPU).

### Phase status (2026-04-02)
- **Phase 0** — COMPLETE (single pair, single GPU, sequential)
- **Phase 1** — COMPLETE (single pair, 12-fold CV, 4 GPUs)
- **Phase 2** — COMPLETE (Steps 1-6). Key results:
  - Steps 2-4: interactive `qsub -I` with SSH (Step 2) and mpiexec (Step 4)
  - Step 5: batch `qsub` with 2 pairs — 2/2 SUCCEEDED
  - Step 6: full 28-pair batch — 23/28 SUCCEEDED, 5 failed (CUDA OOM on specific nodes, transient)
  - Root cause of early env failures: `#!/bin/bash -l` needed for login shell in batch mode
- **Phase 3** — FAILED (walltime kill after 6h, 0/336 folds completed). Root cause diagnosed and fixed. Ready to re-test.
- **Branch**: `feature/polaris-mpiexec` (master as of latest commits)

### Phase 3 failure root cause + fixes (2026-04-02 → 2026-04-06)
**Problem**: Training ~15-34x slower per batch on Polaris vs Lambda (7.5 min/epoch vs 29s).
Three root causes identified:
1. **Memory explosion**: `KmerPairDataset` densified the ENTIRE 868K×4096 sparse matrix (14.2 GB) per fold. 4 concurrent folds = 56.9 GB → folds 1-11 OOM. Only fold 0 survived per pair.
2. **Cache thrashing**: 14.2 GB array >> L3 cache (~32-64 MB). Shuffled batch access = pure cache misses.
3. **Per-item overhead**: `self.pairs.iloc[idx]['label']` (pandas iloc) + `torch.tensor()` copy per item.

**Round 1 fixes** (in `src/models/train_pair_classifier.py`):
- Matrix subsetting: only densify rows used by this fold's pairs (~3.5 GB vs 14.2 GB). 4 folds fit: 14 GB vs 56.9 GB.
- Label pre-extraction: `self.labels = pairs['label'].values` — eliminates pandas iloc.
- Zero-copy tensors: `torch.from_numpy()` instead of `torch.tensor()` (safe with num_workers=0).
- tqdm `miniters=50` to reduce Lustre I/O from 1385 writes/epoch to ~28.
- `NUM_WORKERS` hard-coded to 0 (was configurable but num_workers>0 is 87% slower + incompatible with torch.from_numpy).
- Level 1 profiling added: per-epoch data_time/compute_time/eval_time in training_history.csv.

**Test run results (2026-04-06)**: Folds 0,1,3 ran 6 epochs before job died. Fold 2 OOM'd at model.to(device).
Profiling showed data loading is STILL 96-97% of epoch time (~460s data, ~4s compute, ~14s eval).
Matrix subsetting fixed memory but NOT speed. Two remaining issues:

**Round 2 diagnosis (2026-04-06)**:
1. **Fold 2 OOM**: TensorFlow installed in Polaris system conda env, loaded transitively by HuggingFace `transformers`
   (via esm2_utils.py). TF's default is to eagerly allocate all GPU memory. Added defensive env vars
   (`TF_CPP_MIN_LOG_LEVEL=3`, `TF_FORCE_GPU_ALLOW_GROWTH=true`) at top of script before any imports.
   Also added `torch.cuda.mem_get_info()` diagnostic before model.to(device) to capture GPU state.
2. **Data loading 460s/epoch**: Strongest hypothesis is `pin_memory=True` with 4 concurrent folds.
   pin_memory forces cudaHostAlloc (CUDA driver call that serializes across processes) for every batch.
   1385 batches/epoch × 4 folds = 5540 driver calls/epoch contending on same driver lock.
   The 18% speedup from pin_memory was benchmarked on Lambda (single fold) — likely counterproductive
   with 4 concurrent folds on Polaris. Added Level 2 diagnostic (micro-benchmark 10 batches with
   pin_memory=True vs False) to confirm before changing the config.

**Additional changes**:
- `run_cv_lambda.py`: `--skip_dataset` now auto-discovers latest dataset dir (no longer requires `--dataset_run_dir`).
- `src/analysis/analyze_training_profile.py`: post-hoc profiling aggregation (single run, CV, all-pairs modes).
- Cross-pair aggregation wired into prod launcher (`aggregate_allpairs_results.py`).

See `polaris_plan.md` for detailed step-by-step checklists per phase.

## What's Next (immediate)
**H3N2 all-pairs sweep** — rerun the 28 protein-pair × 12-fold CV with the
`dataset.hn_subtype=H3N2` filter applied via the new `--filter`/`--tag`
mechanism (no new bundles). Runbook: `docs/allpairs_filter_sweep_runbook.md`.

Plumbing added (committed on feature branch, not yet on master):
- `--override key=value` in `dataset_segment_pairs.py` and `train_pair_classifier.py`
- `--override` + `--tag` in `scripts/run_cv_lambda.py`
- `--filter` + `--filter-tag` in `scripts/run_allpairs_polaris_prod.sh`
- `--tag` in `src/analysis/aggregate_allpairs_results.py`

Workflow summary (see runbook for detail):
1. Stage 3 (serial loop over 28 bundles with `--skip_training --tag h3n2 --override dataset.hn_subtype=H3N2`)
2. Stage 4 — interactive `qsub -I` then `bash scripts/run_allpairs_polaris_prod.sh --filter dataset.hn_subtype=H3N2 --skip_dataset`
3. Aggregate with `--tag h3n2`

## What's Next (beyond Task 11)
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
