# Cross-Validation Run Guide

How to run a full K-fold CV experiment on the Lambda 8-GPU cluster.

---

## Prerequisites

- Conda environment `viral-segmatch` is active.
- Stages 1 and 2 have already been run (preprocessed proteins and ESM-2 embeddings exist
  under `data/processed/flu/July_2025/` and `data/embeddings/flu/July_2025/`).
- Working directory is the project root.

---

## Step 1 — Dry run

Verify the command chain without touching disk:

```bash
python scripts/run_cv_lambda.py \
    --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 \
    --dry_run
```

Check the printed output:
- `n_folds` is 5 (read from the bundle config).
- GPU assignments: fold 0 → GPU 0, …, fold 4 → GPU 4.
- Three command groups are printed: Stage 3 (dataset), Stage 4 (training ×5), aggregation.

---

## Step 2 — Full CV run

Runs dataset generation, all 5 training folds in parallel, and aggregation end-to-end.
The script blocks until all folds finish.

```bash
python scripts/run_cv_lambda.py \
    --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 \
    --gpus 0 1 2 3 4
```

**Serial mode** (one fold at a time — useful for debugging):

```bash
python scripts/run_cv_lambda.py \
    --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 \
    --gpus 0 1 2 3 4 \
    --serial
```

**Re-use an existing dataset** (skip Stage 3 if dataset was already generated):

```bash
python scripts/run_cv_lambda.py \
    --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 \
    --gpus 0 1 2 3 4 \
    --skip_dataset \
    --dataset_run_dir data/datasets/flu/July_2025/runs/dataset_paper_flu_schema_raw_slot_norm_unit_diff_cv5_<TIMESTAMP>
```

**Skip aggregation** (run it manually later):

```bash
python scripts/run_cv_lambda.py \
    --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 \
    --gpus 0 1 2 3 4 \
    --skip_aggregate
```

---

## Step 3 — Manual aggregation (if needed)

If aggregation was skipped or failed, run it separately after all folds complete:

```bash
python scripts/aggregate_cv_results.py \
    --manifest data/datasets/flu/July_2025/runs/dataset_paper_flu_schema_raw_slot_norm_unit_diff_cv5_<TIMESTAMP>/cv_run_manifest.json
```

Or point directly at training directories (without a manifest):

```bash
python scripts/aggregate_cv_results.py \
    --training_dirs \
    models/flu/July_2025/runs/training_paper_flu_schema_raw_slot_norm_unit_diff_cv5_fold0_<TIMESTAMP> \
    models/flu/July_2025/runs/training_paper_flu_schema_raw_slot_norm_unit_diff_cv5_fold1_<TIMESTAMP> \
    models/flu/July_2025/runs/training_paper_flu_schema_raw_slot_norm_unit_diff_cv5_fold2_<TIMESTAMP> \
    models/flu/July_2025/runs/training_paper_flu_schema_raw_slot_norm_unit_diff_cv5_fold3_<TIMESTAMP> \
    models/flu/July_2025/runs/training_paper_flu_schema_raw_slot_norm_unit_diff_cv5_fold4_<TIMESTAMP>
```

---

## Output structure

All timestamps below are `YYYYMMDD_HHMMSS` strings set at launch time.

### Dataset splits (Stage 3)

Written once to:
```
data/datasets/flu/July_2025/runs/
  dataset_paper_flu_schema_raw_slot_norm_unit_diff_cv5_<TIMESTAMP>/
    cv_info.json                      # fold isolate assignments + seed
    cv_run_manifest.json              # maps fold_id → training run_id (written after training)
    cv_summary.csv                    # mean ± std metrics (written after aggregation)
    cv_summary.json                   # same, machine-readable
    fold_0/
      train_pairs.csv
      val_pairs.csv
      test_pairs.csv
      dataset_stats.json              # split sizes, class balance, metadata distributions
      isolate_metadata.csv            # one row per isolate (host, subtype, year, geo)
      duplicate_stats.json            # co-occurrence and rejection counts
      cooccurring_sequence_pairs.csv  # only written if co-occurring pairs exist
      plots/                          # PCA and distribution plots
    fold_1/ … fold_4/                 # same structure per fold
```

### Trained models (Stage 4)

One directory per fold:
```
models/flu/July_2025/runs/
  training_paper_flu_schema_raw_slot_norm_unit_diff_cv5_fold0_<TIMESTAMP>/
    test_predicted.csv        # per-sample predictions on test set (used by aggregation)
    optimal_threshold.txt     # classification threshold chosen on val set
    best_model.pt
    train_history.csv
    … other training outputs
  training_…_fold1_…/ … training_…_fold4_…/
```

### Aggregated CV results

Written back into the dataset run directory:
```
data/datasets/flu/July_2025/runs/dataset_…_cv5_<TIMESTAMP>/
  cv_summary.csv    # rows: fold_0 … fold_4, mean, std
                    # cols: f1_binary, f1_macro, auc_roc, precision, recall, brier
  cv_summary.json   # same content, machine-readable
```

The terminal prints the full dataset run directory path at the end of Stage 3.
Use that path for `--dataset_run_dir` if you ever need to re-run training or aggregation.

---

## Config reference

| Bundle | File |
|--------|------|
| `paper/flu_schema_raw_slot_norm_unit_diff_cv5` | `conf/bundles/paper/flu_schema_raw_slot_norm_unit_diff_cv5.yaml` |

Inherits from `flu_schema_raw_slot_norm_unit_diff` (slot_norm + unit_diff, HA→NA, no filter).
Adds `dataset.n_folds: 5`.
