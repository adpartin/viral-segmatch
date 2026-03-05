**Status: IMPLEMENTED**

# Plan: Decouple Stage 3 (Dataset) from Stage 4 (Training)

Decoupled Stage 3/4 so you can run Stage 3 once and Stage 4 N times with different
training bundles. See git branch `feature/decouple-dataset-training`.

## Changes Made

1. **`src/models/train_esm2_frozen_pair_classifier.py`**: Added `training_info.json`
   provenance file saved after training (config_bundle, dataset_dir, HPs, threshold, seed).

2. **`scripts/stage3_dataset.sh`**: Rewritten from 158 to ~60 lines matching the lean
   `stage1_preprocess_flu.sh` pattern. Removed git provenance block, log() helper,
   elaborate header/footer, experiment registry call.

3. **`scripts/stage4_train.sh`**: Rewritten from 388 to ~100 lines. Removed all bundle
   extraction from dataset_dir, `--allow_bundle_mismatch` flag, git provenance block,
   experiment registry. `--dataset_dir` is now required.

## Workflow

```bash
# Create dataset once
./scripts/stage3_dataset.sh flu_schema_raw_slot_norm_unit_diff

DSET=data/datasets/flu/July_2025/runs/dataset_flu_schema_raw_slot_norm_unit_diff_20260304_120000

# Train with different bundles against the same dataset
./scripts/stage4_train.sh flu_schema_raw_slot_norm_unit_diff --dataset_dir $DSET
./scripts/stage4_train.sh flu_schema_raw_concat              --dataset_dir $DSET
```
