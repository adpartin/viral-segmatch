# Experiment Tracking Guide

## Overview

This guide explains how to track and document all experiments in a robust, searchable way. The system automatically tracks experiments when you run scripts, and you can add manual notes for observations.

## Quick Start

### 1. Run Experiments (Auto-Tracked)

Just run your experiments normally - they're automatically tracked:

```bash
# Dataset creation - automatically registered
./scripts/run_dataset.sh flu_a_learning_test

# Training - automatically registered
./scripts/run_training.sh flu_a_learning_test --cuda_name cuda:0
```

### 2. View Recent Experiments

```bash
# List last 10 experiments
python src/utils/experiment_registry.py

# List experiments for specific config
python src/utils/experiment_registry.py --config_bundle flu_a_learning_test

# List only training experiments
python src/utils/experiment_registry.py --stage training

# List only failed experiments
python src/utils/experiment_registry.py --status failed
```

### 3. View Experiment Details

```bash
# Show full details for specific experiment
python src/utils/experiment_registry.py --experiment_id dataset_flu_a_learning_test_20251121_143022
```

### 4. Add Manual Notes

Edit `experiments/registry.yaml` and add a `notes` field to any experiment:

```yaml
experiments:
  - experiment_id: dataset_flu_a_learning_test_20251121_143022
    date: 2025-11-21
    time: 14:30:22
    config_bundle: flu_a_learning_test
    stage: dataset
    # ... other fields ...
    notes: "Created dataset successfully. 100 isolates, 3 proteins. Ready for training."
```

## How It Works

### Automatic Tracking

When you run `run_dataset.sh` or `run_training.sh`, the script automatically:
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
  - experiment_id: dataset_flu_a_learning_test_20251121_143022
    date: 2025-11-21
    time: 14:30:22
    config_bundle: flu_a_learning_test
    stage: dataset
    command: python src/datasets/dataset_segment_pairs.py --config_bundle flu_a_learning_test
    exit_code: 0
    status: success
    git_commit: a1b2c3d
    log_file: logs/datasets/dataset_segment_pairs_flu_a_learning_test_20251121_143022.log
    output_dir: data/datasets/flu_a/July_2025_seed_42_isolates_100_learning_test
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
   ./scripts/run_dataset.sh flu_a_overfit_test
   ./scripts/run_training.sh flu_a_overfit_test --cuda_name cuda:0 --skip_postprocessing
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

# What experiments used flu_a_learning_test config?
python src/utils/experiment_registry.py --config_bundle flu_a_learning_test
```

### Compare Experiments

```bash
# Get experiment IDs
python src/utils/experiment_registry.py --config_bundle flu_a_learning_test

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
  - dataset_flu_a_plateau_analysis_20251121_150000
  - training_flu_a_plateau_analysis_20251121_150500
  Findings: Validation F1 plateaued at epoch 12.
```

### 4. Track Issues

```yaml
notes: |
  Failed due to CUDA out of memory.
  Solution: Reduced batch size from 32 to 16.
  Retry: training_flu_a_overfit_test_20251121_160000
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
./scripts/run_dataset.sh flu_a_overfit_test

# 2. Check it was registered
python src/utils/experiment_registry.py --limit 1

# 3. Run training
./scripts/run_training.sh flu_a_overfit_test --cuda_name cuda:0 --skip_postprocessing

# 4. Add notes about results
# Edit experiments/registry.yaml:
#   notes: "Overfitting confirmed! Training F1=0.95, Val F1=0.65. Model has capacity."

# 5. View all experiments for this config
python src/utils/experiment_registry.py --config_bundle flu_a_overfit_test
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

