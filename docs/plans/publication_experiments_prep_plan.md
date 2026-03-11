# Plan: Publication Experiments Preparation

**Status: IN PROGRESS**

Two tasks to address before running the publication experiments (P1 in `prompts.md`).

---

## Task 1: Results directory — use training run ID to prevent overwriting

### Problem

`data/` and `models/` use `runs/` + timestamp directories, but `results/` uses a flat
`results/{virus}/{data_version}/{config_bundle}/training_analysis/` structure. Re-running
analysis for the same bundle overwrites previous results.

### Solution

Tie each results directory to its source training run by reusing the training run ID.

**Current:**
```
results/flu/July_2025/flu_schema_raw_slot_norm_unit_diff/training_analysis/
```

**Proposed:**
```
results/flu/July_2025/runs/training_flu_schema_raw_slot_norm_unit_diff_20260310_143000/training_analysis/
```

The training run ID (e.g., `training_flu_schema_raw_slot_norm_unit_diff_20260310_143000`)
is already known at analysis time because the analysis scripts locate or accept `--model_dir`.
Extract the run directory name from the model path and use it as the results subdirectory.

### Files to modify

| File | Change |
|------|--------|
| `src/analysis/analyze_stage4_train.py` | Change results path construction (~line 1023) to use `results/{virus}/{data_version}/runs/{training_run_id}/training_analysis/` |
| `src/analysis/create_presentation_plots.py` | Same change (~line 454) |
| `src/analysis/aggregate_experiment_results.py` | Update `find_results_directory()` to search under `results/{virus}/{data_version}/runs/` for matching training run dirs |

### Backward compatibility

Old flat results under `results/{virus}/{data_version}/{bundle}/` become orphans.
No migration needed — they can be deleted manually after verifying the new runs work.

### Verification

1. Run `analyze_stage4_train.py` for an existing training run
2. Confirm output lands in `results/.../runs/training_{bundle}_{ts}/training_analysis/`
3. Run again — confirm it creates a new directory (or overwrites same ts), not the old flat path
4. Run `aggregate_experiment_results.py` — confirm it finds results under new paths

---

## Task 2: CV visualization with error bars

### Problem

`scripts/aggregate_cv_results.py` already computes per-fold metrics and writes
`cv_summary.csv` / `cv_summary.json` with mean +/- std. But there are no plots —
the publication experiments need bar charts with error bars (std across folds).

### Solution

Create `src/analysis/aggregate_cv_results.py` — a proper analysis script that reads
the existing `cv_summary.json` (or recomputes from per-fold predictions) and generates
publication-ready plots.

**Why a new file in `src/analysis/` instead of extending `scripts/aggregate_cv_results.py`?**
- `scripts/` is for pipeline orchestration (shell wrappers, launchers)
- `src/analysis/` is for analysis and visualization (all other analysis scripts live here)
- The existing `scripts/aggregate_cv_results.py` stays as-is for the pipeline; the new
  script adds visualization on top

### Inputs

Option A (preferred): Read `cv_summary.json` written by `scripts/aggregate_cv_results.py`.
Option B: Accept `--training_dirs` and recompute from `test_predicted.csv` files directly.

Support both: if `cv_summary.json` exists, use it; otherwise compute from scratch.

### Outputs

Save to `results/{virus}/{data_version}/runs/cv_{bundle}_{ts}/`:

1. **`cv_metrics_barplot.png`** — Bar chart of mean metrics (F1, AUC-ROC, Precision, Recall)
   with std error bars. One group per metric.
2. **`cv_fold_comparison.png`** — Per-fold line/dot plot showing metric variation across folds.
3. **`cv_summary_table.png`** — Formatted table image (mean +/- std) suitable for paper/slides.
4. **`cv_roc_curves.png`** — Overlaid ROC curves (one per fold) + mean ROC with shaded std band.
5. Updated `cv_summary.csv` / `cv_summary.json` (if recomputed).

### Interface

```bash
# From cv_summary.json
python src/analysis/aggregate_cv_results.py \
    --cv_summary data/datasets/flu/July_2025/runs/dataset_..._cv5_.../cv_summary.json

# From training dirs directly
python src/analysis/aggregate_cv_results.py \
    --training_dirs models/flu/July_2025/runs/training_..._fold0_... \
                    models/flu/July_2025/runs/training_..._fold1_... ...

# With config bundle (for path resolution)
python src/analysis/aggregate_cv_results.py \
    --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 \
    --manifest data/datasets/flu/.../cv_run_manifest.json
```

### Files to create/modify

| File | Action |
|------|--------|
| `src/analysis/aggregate_cv_results.py` | **Create** — CV visualization script |
| `CLAUDE.md` | Update Active Source Files to include it |

### Verification

1. Generate mock CV results (or use real per-fold predictions if available)
2. Run the script — confirm all 4 plot types are generated
3. Confirm error bars match std values from `cv_summary.json`
4. Confirm output directory follows the Task 1 convention (runs/ + identifier)
