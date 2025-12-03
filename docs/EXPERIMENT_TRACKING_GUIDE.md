# Experiment Tracking Guide

## Overview

This guide explains how to track and document all experiments in a robust, searchable way. The system automatically tracks experiments when you run scripts, and you can add manual notes for observations.

## Quick Start

### 1. Run Experiments (Auto-Tracked)

Just run your experiments normally - they're automatically tracked:

```bash
# Dataset creation - automatically registered
./scripts/stage3_dataset.sh flu_ha_na_5ks

# Training - automatically registered
./scripts/stage4_train.sh flu_ha_na_5ks --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS
```

### 2. View Recent Experiments

```bash
# List last 10 experiments
python src/utils/experiment_registry.py

# List experiments for specific config
python src/utils/experiment_registry.py --config_bundle flu_ha_na_5ks

# List only training experiments
python src/utils/experiment_registry.py --stage training

# List only failed experiments
python src/utils/experiment_registry.py --status failed
```

### 3. View Experiment Details

```bash
# Show full details for specific experiment
python src/utils/experiment_registry.py --experiment_id dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS
```

### 4. Add Manual Notes

Edit `experiments/registry.yaml` and add a `notes` field to any experiment:

```yaml
experiments:
  - experiment_id: dataset_flu_ha_na_5ks_20251202_140000
    date: 2025-12-02
    time: 14:00:00
    config_bundle: flu_ha_na_5ks
    stage: dataset
    # ... other fields ...
    notes: "Created dataset successfully. 5K isolates, HA-NA proteins. Ready for training."
```

## How It Works

### Automatic Tracking

When you run `stage3_dataset.sh` or `stage4_train.sh`, the script automatically:
1. Records the exact command run
2. Captures exit code (success/failure)
3. Links to log file
4. Links to output directory
5. Records git commit for reproducibility
6. Adds entry to `experiments/registry.yaml`

### Registry File Structure

The registry (`experiments/registry.yaml`) is a YAML file with all experiments:

```yaml
experiments:
  - experiment_id: dataset_flu_ha_na_5ks_20251202_140000
    date: 2025-12-02
    time: 14:00:00
    config_bundle: flu_ha_na_5ks
    stage: dataset
    command: python src/datasets/dataset_segment_pairs.py --config_bundle flu_ha_na_5ks
    exit_code: 0
    status: success
    git_commit: a1b2c3d
    log_file: logs/datasets/dataset_segment_pairs_flu_ha_na_5ks_20251202_140000.log
    output_dir: data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_20251202_140000
    notes: "Optional manual notes here"
```

## Common Workflows

### Track Your Experiment Plan

1. **Before running**: Note what you're testing
   ```bash
   # Edit registry.yaml and add a comment:
   # Phase 2: Overfitting test - testing if model can overfit 50 isolates
   ```

2. **Run experiment**:
   ```bash
   ./scripts/stage3_dataset.sh flu_overfit
   ./scripts/stage4_train.sh flu_overfit --cuda_name cuda:7 \
       --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_overfit_YYYYMMDD_HHMMSS
   ```

3. **After running**: Add observations
   ```yaml
   # Edit registry.yaml, find the experiment, add notes:
   notes: "Model overfitted successfully! Training F1=0.95, Val F1=0.65. Confirms capacity."
   ```

### Find Experiments

```bash
# What experiments did I run today?
python src/utils/experiment_registry.py --limit 20

# What training experiments failed?
python src/utils/experiment_registry.py --stage training --status failed

# What experiments used flu_ha_na_5ks config?
python src/utils/experiment_registry.py --config_bundle flu_ha_na_5ks
```

### Compare Experiments

```bash
# Get experiment IDs
python src/utils/experiment_registry.py --config_bundle flu_ha_na_5ks

# View details for comparison
python src/utils/experiment_registry.py --experiment_id <id1>
python src/utils/experiment_registry.py --experiment_id <id2>
```

## Best Practices

### 1. Add Notes Immediately After Running

Don't wait - add notes while observations are fresh:

```yaml
notes: |
  Learning test completed successfully.
  - Initialization loss: 0.693 ✅
  - Baseline F1: 0.60
  - Final F1: 0.75 ✅ (beats baseline)
  - Model learned! Ready for Phase 2.
```

### 2. Document Key Findings

```yaml
notes: |
  Overfitting test results:
  - Training F1: 0.95 (very high - overfitting confirmed)
  - Validation F1: 0.65 (lower - poor generalization)
  - Conclusion: Model has sufficient capacity ✅
  - Next: Test with regularization
```

### 3. Link Related Experiments

```yaml
notes: |
  Plateau analysis - extended training.
  Related experiments:
  - dataset_flu_pb2_pb1_pa_5ks_20251202_150000
  - training_flu_pb2_pb1_pa_5ks_20251202_150500
  Findings: Validation F1 plateaued at epoch 12.
```

### 4. Track Issues

```yaml
notes: |
  Failed due to CUDA out of memory.
  Solution: Reduced batch size from 32 to 16.
  Retry: training_flu_ha_na_5ks_20251202_160000
```

## Integration with Existing Metadata

This registry complements the existing metadata system:

- **Registry**: High-level tracking of all experiments (what, when, status)
- **Metadata files**: Detailed config and results per experiment (in output directories)
- **Logs**: Full execution logs (in `logs/` directory)

Use registry for:
- ✅ Quick overview of all experiments
- ✅ Finding experiments by date/config/stage
- ✅ Adding manual notes and observations
- ✅ Tracking experiment plan progress

Use metadata files for:
- ✅ Detailed configuration analysis
- ✅ Comparing experiment configurations
- ✅ Reproducing exact experiment setup

## Example: Complete Experiment Session

```bash
# 1. Run dataset creation
./scripts/stage3_dataset.sh flu_overfit

# 2. Check it was registered
python src/utils/experiment_registry.py --limit 1

# 3. Find dataset directory
DATASET_DIR=$(ls -td data/datasets/flu/July_2025/runs/dataset_flu_overfit_* | head -1)

# 4. Run training
./scripts/stage4_train.sh flu_overfit --cuda_name cuda:7 --dataset_dir "$DATASET_DIR"

# 5. Add notes about results
# Edit experiments/registry.yaml:
#   notes: "Overfitting confirmed! Training F1=0.95, Val F1=0.65. Model has capacity."

# 6. View all experiments for this config
python src/utils/experiment_registry.py --config_bundle flu_overfit
```

## Troubleshooting

### Registry not updating?

- Check that `experiments/registry.yaml` exists and is writable
- Check script output for registration errors (non-critical, won't fail script)

### Can't find experiment?

- Use `--limit` to increase results: `python src/utils/experiment_registry.py --limit 50`
- Check date filters if using `--date_from` / `--date_to`

### Want to manually add experiment?

Edit `experiments/registry.yaml` directly, following the format shown above.

## Advanced: Query Scripts

Create custom query scripts:

```bash
#!/bin/bash
# List all successful training experiments from last week
python src/utils/experiment_registry.py \
    --stage training \
    --status success \
    --limit 50
```

## Summary

✅ **Automatic**: Scripts register experiments automatically  
✅ **Searchable**: Query by config, stage, status, date  
✅ **Flexible**: Add manual notes anytime  
✅ **Robust**: YAML format, human-readable, git-friendly  
✅ **Complete**: Links to logs, outputs, git commits  

No more confusion about what experiments you ran and when!

---

## Related Documentation

### Technical Documentation (`docs/`)
- **Configuration Guide:** [CONFIGURATION_GUIDE.md](./CONFIGURATION_GUIDE.md) - Detailed configuration documentation
- **Seed System:** [SEED_SYSTEM.md](./SEED_SYSTEM.md) - Seed hierarchy and reproducibility
- **Experiment Results:** [EXPERIMENT_RESULTS_ANALYSIS.md](./EXPERIMENT_RESULTS_ANALYSIS.md) - Current experiment results

### User Guides (`documentation/`)
- **Quick Start:** [`../documentation/quick-start.md`](../documentation/quick-start.md) - Get started quickly
- **Pipeline Overview:** [`../documentation/pipeline-overview.md`](../documentation/pipeline-overview.md) - Understanding the pipeline
