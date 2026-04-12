# All-Pairs CV with Metadata Filters (Runbook)

How to run the 28 protein-pair × 12-fold CV on Polaris with a metadata filter
applied (e.g., H3N2-only), **without creating new bundles**.

The filter is passed as a Hydra-style dotlist override to the existing
`flu_28p_*` bundles. A short tag (auto-derived from the filter value) is
injected into every run directory so filtered runs don't collide with the
unfiltered baseline.

## How it works

- `--filter key=value` (CLI, repeatable) or `FILTERS_CSV="k=v,k2=v2"` (env var)
  → forwarded as Hydra overrides to Stage 3 (`dataset_segment_pairs.py`) and
  Stage 4 (`train_pair_classifier.py`) via
  `OmegaConf.merge(..., OmegaConf.from_dotlist(overrides))`.
- `--filter-tag <tag>` (CLI) or `FILTER_TAG=<tag>` (env var) → short suffix
  baked into every output directory name. If omitted, the launcher derives it
  by lowercasing the filter values and stripping non-alphanumerics
  (e.g., `dataset.hn_subtype=H3N2` → `h3n2`). Multiple filters are joined
  with `_` in sorted order.
- Tagged dir names:
  - `data/datasets/flu/July_2025/runs/dataset_flu_28p_{a}_{b}_{tag}_{ts}/`
  - `models/flu/July_2025/runs/training_flu_28p_{a}_{b}_{tag}_fold{k}_{ts}/`
  - `models/flu/July_2025/cv_runs/cv_flu_28p_{a}_{b}_{tag}_{ts}/`
  - `models/flu/July_2025/allpairs_prod_{tag}_{ts}/`
- `aggregate_allpairs_results.py --tag {tag}` restricts discovery to tagged
  cv dirs. Without `--tag`, tagged runs are excluded (so baseline aggregation
  is unaffected).

### Files that implement this

- `src/datasets/dataset_segment_pairs.py` — `--override` flag
- `src/models/train_pair_classifier.py` — `--override` flag
- `scripts/run_cv_lambda.py` — `--override`, `--tag` (propagated to both stages)
- `scripts/run_allpairs_polaris_prod.sh` — `--filter`, `--filter-tag`, `--dataset_manifest` (CLI);
  `FILTERS_CSV`, `FILTER_TAG`, `SKIP_DATASET`, `SKIP_TRAINING`, `DATASET_MANIFEST` (env vars for PBS batch)
- `src/analysis/aggregate_allpairs_results.py` — `--tag`
- `scripts/check_allpairs_status.py` — `--tag`, `--compare-to` (quick results check)

### CLI vs env var summary

All options work as CLI flags (interactive) and env vars (PBS batch).
PBS `qsub script.sh` cannot forward CLI arguments — only `-v` env vars survive
into the job. The env-var fallback makes the same script work for both modes.

| CLI flag               | Env var (`-v`)          | Notes                          |
|------------------------|------------------------|--------------------------------|
| `--filter key=value`   | `FILTERS_CSV="k=v,…"`  | Repeatable CLI, comma-sep env  |
| `--filter-tag tag`     | `FILTER_TAG="tag"`      | Auto-derived if omitted        |
| `--dataset_manifest f` | `DATASET_MANIFEST=f`    | Reuse datasets from a prior run|
| `--skip_dataset`       | `SKIP_DATASET=true`     |                                |
| `--skip_training`      | `SKIP_TRAINING=true`    | Stage 3 only (no GPU needed)   |
| `--skip_aggregate`     | `SKIP_AGGREGATE=true`   |                                |
| `--pairs "b1 b2"`      | `PAIRS="b1 b2"`         |                                |

---

## Step-by-step workflow (H3N2 example)

Replace `dataset.hn_subtype=H3N2` with any override. Add more comma-separated
overrides to `FILTERS_CSV` for stacking (e.g., `dataset.hn_subtype=H3N2,dataset.host=human`).

### Step 1 — Generate 28 filtered datasets (Stage 3, parallel, ~3 min)

Parallelize across 28 nodes via the launcher with `--skip_training` / `SKIP_TRAINING=true`.
Each node creates one pair's 12-fold dataset, then exits. GPUs sit idle but
the job is short (~3 min total for all 28).

**Batch submission:**
```bash
qsub -v FILTERS_CSV="dataset.hn_subtype=H3N2",SKIP_TRAINING=true \
     scripts/run_allpairs_polaris_prod.sh
```

**Interactive (debug-scaling, 1 node — runs pairs serially, ~60-90 min):**
```bash
qsub -I -l select=1:ncpus=64:ngpus=4 -l walltime=02:00:00 \
     -A IMPROVE_Aim1 -q debug-scaling -l filesystems=eagle
# on the compute node:
cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch
source scripts/polaris_env.sh
bash scripts/run_allpairs_polaris_prod.sh \
    --filter dataset.hn_subtype=H3N2 \
    --skip_training
```

Creates `data/datasets/flu/July_2025/runs/dataset_flu_28p_{a}_{b}_h3n2_{ts}/fold_0..fold_11/`.

### Step 2 — Train all 28 pairs (Stage 4, parallel, ~90 min)

Reuse the datasets from Step 1 via `--skip_dataset` / `SKIP_DATASET=true`.
Each node trains one pair (12 folds on 4 GPUs, 3 waves).

**Batch submission:**
```bash
qsub -v FILTERS_CSV="dataset.hn_subtype=H3N2",SKIP_DATASET=true \
     scripts/run_allpairs_polaris_prod.sh
```

**Interactive:**
```bash
qsub -I -l select=28:ncpus=64:ngpus=4 -l walltime=06:00:00 \
     -A IMPROVE_Aim1 -q prod -l filesystems=eagle
# on the head node:
cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch
source scripts/polaris_env.sh
bash scripts/run_allpairs_polaris_prod.sh \
    --filter dataset.hn_subtype=H3N2 \
    --skip_dataset
```

The filter must match Step 1 so the tag-scoped auto-discovery finds the right
datasets. The tag derives deterministically from the filter value, so passing
the same `--filter` / `FILTERS_CSV` is sufficient.

### Step 3 — Monitor / re-aggregate

The launcher runs cross-pair aggregation automatically at the end.

**Live progress** (from login node, while job is running):
```bash
python3 src/analysis/aggregate_allpairs_results.py --tag h3n2 --progress
```

**Re-aggregate** (after the fact):
```bash
python3 src/analysis/aggregate_allpairs_results.py \
    --tag h3n2 \
    --output_dir models/flu/July_2025/allpairs_prod_h3n2_<ts>
```

Outputs in the `allpairs_prod_h3n2_<ts>/` dir:
- `allpairs_summary.csv`  — 28 rows with mean +/- std per metric
- `allpairs_summary.json` — full structured summary
- `heatmap_auc_roc.{csv,png}`, `heatmap_f1_binary.{csv,png}`

---

## Dataset reuse across training configs

Stage 3 datasets are feature-agnostic (they store pair IDs, not features).
The same datasets can be reused with different training configs (e.g., k-mer
vs ESM-2) via the **dataset manifest**.

Stage 3 (`SKIP_TRAINING=true`) automatically writes `dataset_manifest.json`
to the allpairs manifest dir. This JSON maps each bundle to its dataset path.
Stage 4 reads it via `DATASET_MANIFEST` and passes `--dataset_run_dir` per
pair, bypassing glob-based auto-discovery entirely.

### Example: H3N2 with both k-mer and ESM-2

```bash
# Stage 3 (once): create H3N2 datasets
qsub -l walltime=00:30:00 \
     -v FILTERS_CSV="dataset.hn_subtype=H3N2",SKIP_TRAINING=true \
     scripts/run_allpairs_polaris_prod.sh
# → writes models/flu/July_2025/allpairs_prod_h3n2_<ts>/dataset_manifest.json

# Stage 4a: train with k-mer (default, auto-discovery works)
qsub -v FILTERS_CSV="dataset.hn_subtype=H3N2",SKIP_DATASET=true \
     scripts/run_allpairs_polaris_prod.sh

# Stage 4b: train with ESM-2 (reuse same datasets via manifest)
qsub -v FILTERS_CSV="training.feature_source=esm2",\
FILTER_TAG="h3n2_esm2",\
DATASET_MANIFEST="models/flu/July_2025/allpairs_prod_h3n2_<ts>/dataset_manifest.json",\
SKIP_DATASET=true \
     scripts/run_allpairs_polaris_prod.sh

# Check results
python3 scripts/check_allpairs_status.py --tag h3n2 --compare-to unfiltered
python3 scripts/check_allpairs_status.py --tag h3n2_esm2 --compare-to h3n2
```

Output dirs are fully separated by tag:
- k-mer: `cv_flu_28p_{a}_{b}_h3n2_*`, `allpairs_prod_h3n2_*`
- ESM-2: `cv_flu_28p_{a}_{b}_h3n2_esm2_*`, `allpairs_prod_h3n2_esm2_*`

Both use the exact same `dataset_flu_28p_{a}_{b}_h3n2_*` dataset dirs.

---

## Validation matrix (quick smoke test)

Before running full-epoch experiments with a new filter, run a short validation
matrix to verify that the scripts work end-to-end across filter combos and
feature sources. Each experiment uses 5 epochs / patience 5 to keep wall time
short (~15-20 min per Stage 4 job).

### 6-experiment grid

| # | Filter                     | Feature | Tag            | Datasets from |
|---|----------------------------|---------|----------------|---------------|
| 1 | None (unfiltered)          | k-mer   | `val_unfilt`   | Own Stage 3   |
| 2 | None (unfiltered)          | ESM-2   | `val_unfilt_esm2` | Exp 1 manifest |
| 3 | `dataset.hn_subtype=H3N2`  | k-mer   | `val_h3n2`     | Own Stage 3   |
| 4 | `dataset.hn_subtype=H3N2`  | ESM-2   | `val_h3n2_esm2`| Exp 3 manifest |
| 5 | `dataset.hn_subtype=H3N2,dataset.host=human` | k-mer | `val_h3n2_human` | Own Stage 3 |
| 6 | `dataset.hn_subtype=H3N2,dataset.host=human` | ESM-2 | `val_h3n2_human_esm2` | Exp 5 manifest |

### Submission order and commands

**Stage 3 — create datasets (3 jobs, can run in parallel):**

```bash
# Exp 1: unfiltered datasets
qsub -l walltime=00:30:00 \
     -v FILTER_TAG="val_unfilt",SKIP_TRAINING=true \
     scripts/run_allpairs_polaris_prod.sh

# Exp 3: H3N2 datasets
qsub -l walltime=00:30:00 \
     -v FILTERS_CSV="dataset.hn_subtype=H3N2",FILTER_TAG="val_h3n2",SKIP_TRAINING=true \
     scripts/run_allpairs_polaris_prod.sh

# Exp 5: H3N2 + human datasets
qsub -l walltime=00:30:00 \
     -v FILTERS_CSV="dataset.hn_subtype=H3N2|dataset.host=human",FILTER_TAG="val_h3n2_human",SKIP_TRAINING=true \
     scripts/run_allpairs_polaris_prod.sh
```

**Stage 4 — train (6 jobs, each depends on its Stage 3 completing):**

After each Stage 3 job completes, note the manifest path from the output
(`allpairs_prod_<tag>_<ts>/dataset_manifest.json`). Replace `<ts>` below
with the actual timestamp.

```bash
# Exp 1: unfiltered k-mer (auto-discovery, no manifest needed)
qsub -l walltime=01:00:00 \
     -v FILTERS_CSV="training.epochs=5|training.patience=5",FILTER_TAG="val_unfilt",SKIP_DATASET=true \
     scripts/run_allpairs_polaris_prod.sh

# Exp 2: unfiltered ESM-2 (reuse Exp 1 datasets via manifest)
qsub -l walltime=01:00:00 \
     -v FILTERS_CSV="training.feature_source=esm2|training.epochs=5|training.patience=5",FILTER_TAG="val_unfilt_esm2",DATASET_MANIFEST="models/flu/July_2025/allpairs_prod_val_unfilt_<ts>/dataset_manifest.json",SKIP_DATASET=true \
     scripts/run_allpairs_polaris_prod.sh

# Exp 3: H3N2 k-mer (auto-discovery)
qsub -l walltime=01:00:00 \
     -v FILTERS_CSV="dataset.hn_subtype=H3N2|training.epochs=5|training.patience=5",FILTER_TAG="val_h3n2",SKIP_DATASET=true \
     scripts/run_allpairs_polaris_prod.sh

# Exp 4: H3N2 ESM-2 (reuse Exp 3 datasets via manifest)
qsub -l walltime=01:00:00 \
     -v FILTERS_CSV="training.feature_source=esm2|training.epochs=5|training.patience=5",FILTER_TAG="val_h3n2_esm2",DATASET_MANIFEST="models/flu/July_2025/allpairs_prod_val_h3n2_<ts>/dataset_manifest.json",SKIP_DATASET=true \
     scripts/run_allpairs_polaris_prod.sh

# Exp 5: H3N2+human k-mer (auto-discovery)
qsub -l walltime=01:00:00 \
     -v FILTERS_CSV="dataset.hn_subtype=H3N2|dataset.host=human|training.epochs=5|training.patience=5",FILTER_TAG="val_h3n2_human",SKIP_DATASET=true \
     scripts/run_allpairs_polaris_prod.sh

# Exp 6: H3N2+human ESM-2 (reuse Exp 5 datasets via manifest)
qsub -l walltime=01:00:00 \
     -v FILTERS_CSV="training.feature_source=esm2|training.epochs=5|training.patience=5",FILTER_TAG="val_h3n2_human_esm2",DATASET_MANIFEST="models/flu/July_2025/allpairs_prod_val_h3n2_human_<ts>/dataset_manifest.json",SKIP_DATASET=true \
     scripts/run_allpairs_polaris_prod.sh
```

### Check results

```bash
# Individual tags
python3 scripts/check_allpairs_status.py --tag val_unfilt
python3 scripts/check_allpairs_status.py --tag val_unfilt_esm2 --compare-to val_unfilt
python3 scripts/check_allpairs_status.py --tag val_h3n2 --compare-to val_unfilt
python3 scripts/check_allpairs_status.py --tag val_h3n2_esm2 --compare-to val_h3n2
python3 scripts/check_allpairs_status.py --tag val_h3n2_human --compare-to val_h3n2
python3 scripts/check_allpairs_status.py --tag val_h3n2_human_esm2 --compare-to val_h3n2_human
```

### Notes

- `|` (pipe) is used as the separator in `FILTERS_CSV` to avoid PBS `-v` comma
  conflicts. The launcher auto-detects the separator.
- Stage 3 walltime is 30 min (generous; typically ~3 min on 28 nodes).
- Stage 4 walltime is 1 hour (5 epochs finishes in ~15-20 min on 28 nodes).
- The `val_` prefix in all tags keeps validation runs separate from production.
- `training.epochs=5` and `training.patience=5` are passed as config overrides
  via `FILTERS_CSV`, not as separate flags — the override mechanism works for
  any config key, not just dataset filters.

---

## Notes / caveats

- **Keep the filter symmetric** between Step 1 and Step 2 when using
  auto-discovery (no manifest). Both invocations must pass the same
  `FILTERS_CSV` / `--filter` so the tag matches. When using
  `DATASET_MANIFEST`, the filter can differ between Stage 3 and Stage 4
  (that's the whole point — e.g., H3N2 datasets with ESM-2 training).
- **Stage 3 filter passthrough**: `dataset_segment_pairs.py` loads the bundle,
  then applies CLI overrides. Filters like `dataset.hn_subtype` are honored
  by the standard filter logic downstream. Verify by checking the resolved
  config snapshot (`resolved_config.yaml`) in the dataset dir.
- **Baseline aggregation is not disturbed**:
  `aggregate_allpairs_results.py` without `--tag` excludes any bundle whose
  name has more than 4 underscore-separated tokens (i.e. anything with a
  tag suffix). Unfiltered 28-pair runs continue to aggregate normally.
- **Stacking filters**: `FILTERS_CSV="dataset.hn_subtype=H3N2,dataset.host=human"`
  auto-derives tag `h3n2_human`. No code changes needed.
- **Config override hook**: `--override` works for any key in the config tree,
  not just dataset filters. You could use the same mechanism to sweep
  `training.batch_size`, `training.lr`, etc., just give each sweep a
  different `--filter-tag`.
- **PBS queue note**: Step 1 (Stage 3 only) finishes in ~3 min so `debug` or
  `debug-scaling` queue is fine. Step 2 needs ~90 min so use `prod` queue.
  Both use the same 28-node allocation since the launcher currently requires
  1 node per pair. For Step 1, this wastes GPUs but saves wall time.
