# Design Improvement Plan

**Status: ONGOING**

Non-urgent design improvements. Items here are not blocking any experiments
but would improve the overall architecture, usability, and robustness of the pipeline.

---

## 1. Separate dataset folds from experiment results (shared folds across feature types)

**Status: IN PROGRESS**

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

Additionally, the manifest and aggregated results reference training run directories under
`models/`, so `models/` is the more logical home (previously tracked as a separate item,
now merged here).

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

### Re-running failed folds

When one or more folds fail during a CV run, the successful folds' training dirs
(under `models/.../runs/`) and the CV results dir (under `models/.../cv_runs/`) already
exist with a manifest referencing all folds. The failed folds have training dirs but
no `best_model.pt` or predictions.

Re-running creates new training dirs with a **new timestamp** (since `train_timestamp`
is set at launch time). Without special handling, this would create a new `cv_runs/`
directory containing only the re-run folds, making aggregation incomplete.

The `--cv_runs_dir` flag solves this by pointing the re-run to the **original** CV
results directory:

```bash
# Original run (folds 0 and 4 failed):
#   cv_runs/cv_bundle_20260312_100000/cv_run_manifest.json  ← has all 10 fold entries
#   runs/training_bundle_fold0_20260312_100000/             ← no best_model.pt (failed)
#   runs/training_bundle_fold4_20260312_100000/             ← no best_model.pt (failed)

# Re-run failed folds, merging into the original cv_runs dir:
python scripts/run_cv_lambda.py \
    --config_bundle <bundle> \
    --skip_dataset --dataset_run_dir <dataset_dir> \
    --folds 0 4 \
    --cv_runs_dir models/.../cv_runs/cv_bundle_20260312_100000

# Result:
#   cv_runs/cv_bundle_20260312_100000/cv_run_manifest.json  ← folds 0,4 updated to new run IDs
#   runs/training_bundle_fold0_20260312_103000/             ← new successful training
#   runs/training_bundle_fold4_20260312_103000/             ← new successful training
#   (old failed dirs remain on disk but are no longer referenced by the manifest)
```

The manifest merge logic: when `--cv_runs_dir` is provided and a manifest already exists
there, only the re-run fold entries are replaced. All other folds keep their original
training run IDs. Aggregation then runs over the full set of folds (original successes +
new re-runs).

### Benefits

- **Fair comparison**: ESM-2 vs k-mer on identical splits with no risk of overwriting.
- **No duplication**: One fold set, many experiments.
- **Clean provenance**: Each experiment's results directory is self-contained with a
  manifest linking to the shared dataset and per-fold models.
- **Scale-ready**: No design changes needed for multi-node job arrays.

---

## 2. Split CV launcher into separate dataset and training scripts

**Status: NOT STARTED**

**Files**: `scripts/run_cv_lambda.py`

### Problem

`run_cv_lambda.py` is a monolithic script that runs both Stage 3 (dataset/fold
generation) and Stage 4 (training across folds). Separating stages currently requires
flag gymnastics (`--skip_dataset --dataset_run_dir`). As the project moves toward HPC
(Polaris), Stage 3 and Stage 4 will routinely run on different machines (login node vs
compute nodes) and at different times.

### Proposed design

Split into three scripts:

1. **`scripts/run_cv_dataset.py`** — Stage 3 only. Creates fold directories from a
   config bundle. Outputs the dataset run dir path. Runs on any machine (no GPU needed).
2. **`scripts/run_cv_train.py`** — Stage 4 only. Takes a dataset run dir and config
   bundle, trains all (or a subset of) folds using a GPU pool. Writes manifest and
   triggers aggregation. This is the script that adapts per-platform:
   - Lambda: subprocess pool with `CUDA_VISIBLE_DEVICES` (current logic).
   - Polaris: PBS job array where each job trains one fold.
3. **`scripts/run_cv.py`** — Convenience wrapper that calls (1) then (2) sequentially.
   Equivalent to current `run_cv_lambda.py` for the common case.

### Benefits

- **Run stages independently** without flag gymnastics. Stage 3 can run alone for
  timing/validation (e.g., full-dataset prep before HPC submission).
- **Platform-specific training scripts** without duplicating dataset logic.
- **Reuse folds naturally**: run dataset script once, then multiple training scripts
  with different bundles (ESM-2, k-mer, etc.) pointing to the same folds.
- **Simpler testing**: each script has a single responsibility.

### Migration

- Extract Stage 3 logic from `run_cv_lambda.py` into `run_cv_dataset.py`.
- Extract Stage 4 + manifest + aggregation into `run_cv_train.py`.
- Rewrite `run_cv_lambda.py` (or rename to `run_cv.py`) as a thin wrapper.
- `--skip_dataset` and `--dataset_run_dir` flags move naturally to `run_cv_train.py`
  as the default interface (dataset dir is always explicit).

---

## 3. CV launcher calls shell wrappers instead of Python scripts directly

**Status: NOT STARTED**

**Files**: `scripts/run_cv_lambda.py`, `scripts/stage3_dataset.sh`, `scripts/stage4_train.sh`

**Related**: Item 2 (split CV launcher). If adopted, the split scripts (`run_cv_dataset.py`,
`run_cv_train.py`) would call the shell wrappers rather than Python scripts directly.

### Problem

`run_cv_lambda.py` calls `dataset_segment_pairs.py` and `train_pair_classifier.py` via
`subprocess.Popen`, bypassing the shell wrappers (`stage3_dataset.sh`, `stage4_train.sh`).
This means the CV launcher must re-implement logging behavior that the shell scripts already
provide:

1. **No real-time log files for Stage 3**: The shell scripts create a log file in
   `logs/{stage}/` immediately via `tee`, so output is tailable from the first line.
   The CV launcher pipes Stage 3 output through Python's TeeWriter, which writes to
   `logs/cv/` but not to `logs/datasets/`. The dataset output dir gets a copy only
   after Stage 3 finishes.
2. **Inconsistent logging**: Running Stage 3 standalone via `stage3_dataset.sh` produces
   a log in `logs/datasets/` + a copy in the output dir. Running via `run_cv_lambda.py`
   produces a log only in `logs/cv/`. Different entry points → different artifacts.
3. **Missing postprocessing**: `stage4_train.sh` runs `analyze_stage4_train.py` and
   `create_presentation_plots.py` after training. The CV launcher skips these for
   per-fold runs. This is intentional for CV (postprocessing per fold is wasteful), but
   it means single-fold re-runs also miss postprocessing.

### Proposed design

Have the CV launcher call the shell wrappers instead of Python scripts:

- **Stage 3**: Call `stage3_dataset.sh` with a `--run_id` override (new flag) so the
  launcher controls the run ID. The shell script handles `tee` logging, latest-symlink,
  and log copying to the output dir.
- **Stage 4**: Call `stage4_train.sh` per fold with `--skip_postprocessing` and
  `--cuda_name`. The shell script handles per-fold `tee` logging to `logs/training/`.

Both shell scripts would need a new `--run_id` (or `--run_output_subdir`) flag to accept
an externally generated run ID, since the launcher needs to know the exact output path.

### Pros

- **Immediate real-time log files**: Shell `tee` starts writing from the first byte.
  Users can `tail -f logs/datasets/...` or `tail -f logs/training/...` during the run.
- **DRY**: Logging, latest-symlink, and log-copy-to-output-dir logic lives in the shell
  scripts only. The launcher doesn't re-implement these.
- **Consistent artifacts**: Same log files and locations regardless of entry point
  (standalone shell script vs CV launcher).
- **Stage 4 postprocessing available**: For single-fold re-runs or serial mode, the
  shell script can run postprocessing. The `--skip_postprocessing` flag already exists.

### Cons

- **Redundant log copies**: Shell script writes to `logs/{stage}/`, launcher's TeeWriter
  writes to `logs/cv/`. Two copies of the same Stage 3 output on disk. Not harmful but
  not minimal.
- **Shell script modifications needed**: Both scripts need a `--run_id` /
  `--run_output_subdir` flag to accept externally generated run IDs. Currently they
  generate their own `RUN_ID` with their own timestamp.
- **Indirection**: Launcher → shell script → Python, instead of launcher → Python.
  Adds a layer for someone reading the code.

### Small vs large runs

- **Small runs (5K isolates, <30 min total)**: The logging gap is a minor annoyance.
  Stage 3 finishes quickly, so the missing real-time log barely matters. The main benefit
  is consistency — same artifacts regardless of entry point.
- **Large runs (full dataset, hours)**: The logging gap is a real problem. Stage 3 can
  take 1–2 hours for 100K+ isolates. Without a real-time log, there's no way to monitor
  progress or diagnose hangs without `tail -f` on the launcher log (which requires
  knowing the exact path in `logs/cv/`). Per-fold training on large datasets can take
  30+ min per fold; having per-fold logs in `logs/training/` (with latest-symlinks) is
  much more convenient than navigating to each training output dir.
- **HPC (Polaris)**: On HPC, PBS job scripts would call shell wrappers (or Python
  directly — PBS captures stdout to job logs). The shell wrapper approach aligns the
  local Lambda workflow with what HPC jobs would do, reducing the gap between dev and
  production runs.

### Implementation

1. Add `--run_output_subdir` flag to `stage3_dataset.sh` and `stage4_train.sh` (optional;
   if provided, overrides the internally generated `RUN_ID`).
2. Change `run_cv_lambda.py` to call shell scripts instead of Python.
3. For Stage 4, pass `--skip_postprocessing` in CV mode (parallel and serial).
4. Keep the launcher's TeeWriter for the overall CV log (`logs/cv/`), which captures
   launcher messages (fold dispatch, completion, aggregation) plus relayed subprocess
   output.

---

## 4. Stage 3 performance: pair generation dominates at full-dataset scale

**Status: NOT STARTED**

**Files**: `src/datasets/dataset_segment_pairs.py`

### Observation

Stage 3 takes ~1 min for 5K isolates but ~11 min for the full dataset (108K isolates,
schema_ordered HA/NA). The bottleneck is not CSV loading or metadata enrichment — it is
the pair generation logic, which runs per-fold and scales poorly with isolate count.

### Where the time goes

For each of the N CV folds (10 for CV10), `split_dataset()` performs:

1. **`create_positive_pairs()`** — iterates every isolate via `grp.itertuples()`,
   builds cross-product (schema mode) or `combinations()` (unordered mode) in a Python
   loop. Called 3x per fold (train/val/test subsets) = 30x total for 10-fold CV.

2. **`create_negative_pairs()`** — rejection-sampling loop: randomly sample two isolates,
   pick one protein from each, check against `cooccur_pairs` set + `seen_pairs` +
   `seen_seq_pairs`. Called 3x per fold = 30x total. On the full dataset, the cooccur
   set is much larger (~millions of pairs for 108K isolates), so more candidates are
   rejected per successful negative, increasing wall-clock time.

3. **`save_split_output()`** — writes CSVs containing full protein sequences (`seq_a`,
   `seq_b` columns) for every pair. With 108K isolates and 3:1 neg:pos ratio, each fold
   produces ~500K+ pair rows with long sequence strings. Written 10x for CV10.

4. **`build_cooccurrence_set()`** — iterates all isolates, computes all within-isolate
   seq_hash pairs via nested Python loop. Called once (shared across folds), but still
   O(isolates x proteins^2). For schema mode with 2 functions this is O(N) where N =
   isolates, but in unordered mode with 9 proteins it is O(N x 36).

### Potential solutions

#### A. Vectorize positive pair generation (high impact, moderate effort)

Replace the Python `for aid, grp in isolates` + `itertuples()` + `combinations()` loop
with vectorized pandas/numpy operations:

- **Schema mode**: `merge()` the HA subset with the NA subset on `assembly_id` (self-join).
  One vectorized merge replaces the per-isolate Python loop entirely.
- **Unordered mode**: Similar self-join with a `function_a != function_b` filter, plus
  canonicalization of pair orientation as a vectorized column operation.

Expected speedup: 5-20x for positive pair generation. The current implementation creates
a Python dict per pair and appends to a list — the merge approach creates the full
DataFrame in one operation.

#### B. Batch negative sampling (high impact, moderate effort)

Replace the one-at-a-time rejection loop with batch sampling:

1. Pre-sample K candidate pairs at once (vectorized `np.random.choice` on isolate IDs
   and protein indices within each isolate).
2. Compute `pair_key` for all K candidates in a vectorized pass.
3. Filter out blocked pairs (set intersection with `cooccur_pairs`), duplicates, and
   same-function violations in bulk.
4. Keep accepted pairs, repeat with a new batch until `num_negatives` is reached.

Expected speedup: 3-10x for negative pair generation. The current loop's per-iteration
cost is dominated by Python overhead (dict creation, set lookups, `random.choice`), not
by the actual filtering logic.

#### C. Drop full sequences from pair CSVs (high impact, easy)

The pair CSVs currently store `seq_a` and `seq_b` (full protein sequences, ~500-600 AA
each). These are not used by Stage 4 — training looks up embeddings by `seq_hash` or
`brc_fea_id` from the HDF5 cache. Dropping these two columns would:

- Shrink CSV file sizes by ~60-70% (sequences dominate row width).
- Reduce `save_split_output()` I/O by a similar factor (10 folds x 3 CSVs = 30 writes).
- Reduce `create_positive_pairs()` and `create_negative_pairs()` memory, since each pair
  dict currently stores two full sequence strings.

The sequences are still available in `protein_final.csv` and the embedding HDF5 for any
downstream analysis that needs them. If per-pair sequence access is needed, a lightweight
lookup by `seq_hash` or `brc_fea_id` suffices.

**Risk**: Some analysis scripts (e.g., `analyze_stage4_train.py` FP/FN inspection) may
read `seq_a`/`seq_b` from the pair CSVs. These would need to join on `seq_hash` instead.
Check all consumers before removing.

#### D. Vectorize `build_cooccurrence_set()` (moderate impact, easy)

Replace the nested Python loop with a self-merge:

```python
# Current: O(N x P^2) Python loop
for aid, grp in df.groupby('assembly_id'):
    seq_hashes = grp['seq_hash'].unique().tolist()
    for i in range(len(seq_hashes)):
        for j in range(i + 1, len(seq_hashes)):
            ...

# Proposed: vectorized self-join
pairs = df.merge(df, on='assembly_id', suffixes=('_a', '_b'))
pairs = pairs[pairs['seq_hash_a'] < pairs['seq_hash_b']]  # canonical + dedup
cooccur_pairs = set(pairs['seq_hash_a'] + '__' + pairs['seq_hash_b'])
```

For schema mode (2 proteins per isolate), the current loop is already O(N) and fast.
The main benefit is for unordered mode with more proteins, or if the function is ever
called on the full protein set (9 functions).

**Risk**: The self-merge approach creates a potentially large intermediate DataFrame
(~N x P^2 rows). For 108K isolates x 2 proteins, this is ~108K rows (manageable). For
108K x 9 proteins, ~108K x 36 = ~3.9M rows (still fine). Memory should not be an issue.

#### E. Parallelize fold generation (moderate impact, moderate effort)

`generate_all_cv_folds()` currently runs sequentially. Since folds are independent
(they share `cooccur_pairs` read-only), each fold's `split_dataset()` call could run
in a separate process via `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`.

Expected speedup: up to Nx on machines with enough cores (Lambda has 128 cores).
However, each fold's work is memory-intensive (full df copy per process), so practical
speedup may be 4-8x before memory becomes the bottleneck.

**Risk**: Increases peak memory by ~Nx (each process gets a copy of `df` and
`cooccur_pairs`). For the full dataset (~217K rows x ~20 columns + sequences), each
copy is ~1-2 GB. With 10 folds, this is 10-20 GB — feasible on Lambda (256 GB RAM)
but worth monitoring.

### Recommended priority

1. **C** (drop sequences from CSVs) — easiest win, reduces I/O immediately.
2. **A** (vectorize positives) — biggest computational win for the inner loop.
3. **B** (batch negatives) — second biggest computational win.
4. **D** (vectorize cooccur) — minor for schema mode, useful for unordered.
5. **E** (parallelize folds) — diminishing returns if A+B already make per-fold fast.

---
