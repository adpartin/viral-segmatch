# Design Improvement Plan

**Status: ONGOING**

Non-urgent design improvements. Items here are not blocking any experiments
but would improve the overall architecture, usability, and robustness of the pipeline.

---

## 1. Relocate CV manifest from `data/` to `models/`

**Files**: `scripts/run_cv_lambda.py`, `scripts/aggregate_cv_results.py`

The CV run manifest (`cv_run_manifest.json`) and aggregated results (`cv_summary.csv`,
`cv_summary.json`) are currently saved in the dataset run directory under `data/datasets/`.
The manifest primarily references training run directories under `models/`, so `models/`
is the more logical home.

The current placement exists because the dataset run dir is the one directory shared across
all folds and known at launch time, before training starts. Training dirs have per-fold
timestamps (and re-runs create new ones), so there is no single training-side directory.

**Options**:

- **A. Create a CV-level directory under `models/`** (e.g.,
  `models/flu/July_2025/runs/cv_flu_schema_raw_slot_norm_concat_cv10_20260311/`) that holds
  the manifest, aggregated results, and optionally symlinks to per-fold training dirs.
  Cleaner semantics but adds complexity.
- **B. Keep in dataset dir** — it's a pointer file, and the dataset dir already serves as
  the CV run anchor. Pragmatic, no code changes needed.

**Update**: Item 2 below makes Option A the clear winner — the dataset dir cannot serve as
the results anchor when multiple experiments share the same folds.

---

## 2. Separate dataset folds from experiment results (shared folds across feature types)

**Files**: `scripts/run_cv_lambda.py`, `scripts/aggregate_cv_results.py`,
`src/analysis/visualize_cv_results.py`

### Problem

CV dataset folds are feature-agnostic: the fold CSVs contain isolate pair IDs, not
embeddings or k-mer vectors. Features are looked up at training time based on
`training.feature_source`. This means the same folds can (and should) be reused when
comparing different feature types (ESM-2 vs k-mer) or interaction methods on identical
data splits.

The current design conflates the dataset and the experiment by storing experiment-specific
outputs (`cv_run_manifest.json`, `cv_summary.csv`, `cv_summary.json`, plots) in the dataset
run directory. This creates two problems:

1. **Result overwrite**: Running a second experiment (e.g., k-mer) on the same folds
   overwrites the first experiment's (e.g., ESM-2) manifest and summary files.
2. **Unnecessary duplication**: Regenerating identical folds under a different bundle name
   wastes disk and obscures the fact that the splits are the same.

### Proposed design

Introduce a clear separation between *shared dataset folds* and *per-experiment CV runs*:

```
data/datasets/flu/July_2025/runs/
  dataset_flu_schema_cv10_20260311_142647/     # Shared folds (feature-agnostic)
    cv_info.json                                # Seed, n_folds, pair config
    fold_0/
      train_pairs.csv, val_pairs.csv, test_pairs.csv
    fold_1/
    ...
    fold_9/

models/flu/July_2025/cv_runs/
  cv_esm2_slot_norm_concat_cv10_20260311/      # ESM-2 experiment
    cv_run_manifest.json                        # Points to dataset dir + per-fold training dirs
    cv_summary.csv
    cv_summary.json
    cv_metrics_barplot.png
    cv_roc_curves.png
  cv_kmer_k6_slot_norm_concat_cv10_20260312/   # K-mer experiment (same folds)
    cv_run_manifest.json
    cv_summary.csv
    ...

models/flu/July_2025/runs/                     # Per-fold training dirs (unchanged)
  training_..._fold0_.../
  training_..._fold1_.../
```

### Key changes

1. **CV results directory**: `models/{virus}/{version}/cv_runs/cv_{bundle}_{timestamp}/`
   holds the manifest, aggregated metrics, and plots for one experiment.
2. **Dataset dir stays clean**: Only fold subdirectories and `cv_info.json` (seed,
   split config). No experiment-specific files.
3. **Manifest points both ways**: The manifest in `cv_runs/` references both the shared
   dataset dir and the per-fold training dirs under `models/.../runs/`.
4. **Launcher changes**:
   - `run_cv_lambda.py` creates the `cv_runs/` directory, writes the manifest there.
   - `--skip_dataset --dataset_run_dir` works as before for reusing folds.
   - Aggregation and visualization write to `cv_runs/`, not the dataset dir.
5. **Dataset bundle naming**: Since folds are shared, the dataset bundle name should
   reflect only the data config (virus, proteins, isolates, filters, n_folds), not the
   feature type. E.g., `flu_schema_cv10` rather than `flu_schema_raw_slot_norm_concat_cv10`.
   Feature-specific settings belong in the training bundle only.

### One bundle, not two

It may seem natural to split into "dataset bundles" and "training bundles," but this
would break the "one bundle = one reproducible experiment" convention and introduce
coordination overhead (which dataset bundle goes with which training bundle?).

The bundle stays unified. Different stages simply read different sections:
- Stage 3 reads `dataset.*` + `virus.*` (pair selection, splits, n_folds)
- Stage 4 reads `training.*` (feature_source, interaction, architecture)

Two bundles that differ only in `training.*` (e.g., ESM-2 concat vs k-mer concat) share
identical dataset config and can reuse the same folds. The dataset dir name should be
derived from the dataset-relevant config only, or folds can be reused explicitly via
`--skip_dataset --dataset_run_dir`.

### Multi-node scaling

This design scales to multi-node multi-GPU systems (PBS on Polaris, SLURM elsewhere)
without changes to the bundle/config system. The bundle is just a YAML passed as an
argument — what changes is only the launcher:

- **Single-node** (current): `run_cv_lambda.py` with subprocess pool, one fold per GPU.
- **Multi-node**: PBS/SLURM job array where each job receives
  `fold_id=$PBS_ARRAY_INDEX` and the bundle name. Each node writes to its own per-fold
  training dir independently.

The `cv_runs/` results directory helps here: per-fold training is embarrassingly parallel
with no shared state, and aggregation runs once after all jobs complete. The manifest
is written by the launcher (or a post-job aggregation script), not by individual
training jobs.

### Benefits

- **Fair comparison**: ESM-2 vs k-mer on identical splits with no risk of overwriting.
- **No duplication**: One fold set, many experiments.
- **Clean provenance**: Each experiment's results directory is self-contained with a
  manifest linking to the shared dataset and per-fold models.
- **Scale-ready**: No design changes needed for multi-node job arrays.

---
