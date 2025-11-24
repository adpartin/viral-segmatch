# Experiment Tracking Guide

## Overview

This guide explains how to organize, track, and compare experiments when exploring different configurations (e.g., different protein selections, hyperparameters, data sampling strategies).

---

## **Problem: Too Many Configuration Dimensions**

When experimenting with:
- Different `selected_functions` (PB1+PB2 vs all core proteins)
- Different sample sizes (100 vs 2000 files)
- Different seeds
- Different hyperparameters

...encoding everything in directory names becomes unwieldy:
```
❌ July_2025_seed_42_2000files_pb1_pb2_batch16_lr0.001_...  (TOO LONG!)
```

---

## **Solution: Hybrid Approach**

### **1. Semantic directory names (short, meaningful)**
### **2. Automatic metadata tracking (complete config)**
### **3. Human-readable summaries**

---

## **Quick Start**

### **Step 1: Create experiment-specific bundle configs**

```yaml
# conf/bundles/flu_a_pb1_pb2.yaml
run_suffix: "_pb1_pb2"  # ← Short, semantic name
master_seed: 42
max_files_to_process: 2000

virus:
  selected_functions:
    - "RNA-dependent RNA polymerase PB2 subunit"
    - "RNA-dependent RNA polymerase catalytic core PB1 subunit"
```

**Creates directories:**
```
processed/flu_a/July_2025_pb1_pb2/
embeddings/flu_a/July_2025_pb1_pb2/
models/flu_a/July_2025_pb1_pb2/
```

---

### **Step 2: Run preprocessing**

```bash
python src/preprocess/preprocess_flu_protein.py \
    --config-name flu_a_pb1_pb2 \
    --virus_name flu_a
```

**Automatically generates:**
1. `experiment_metadata_preprocessing.json` - Full config + git info
2. `EXPERIMENT_SUMMARY_PREPROCESSING.txt` - Human-readable summary

---

### **Step 3: Review experiment metadata**

```bash
# Quick summary (human-readable)
cat processed/flu_a/July_2025_pb1_pb2/EXPERIMENT_SUMMARY_PREPROCESSING.txt
```

**Output:**
```
======================================================================
EXPERIMENT SUMMARY - PREPROCESSING
======================================================================
Generated: 2025-10-13 14:35:22
======================================================================

Virus: flu_a
Data version: July_2025
Run suffix: _pb1_pb2
Master seed: 42
Preprocessing seed: 42
Max files to process: 2000
Selected functions:
  - RNA-dependent RNA polymerase PB2 subunit
  - RNA-dependent RNA polymerase catalytic core PB1 subunit
Total proteins processed: 3500
Unique protein sequences: 2100
Unique files processed: 2000
Processing time: 2m 15s
```

```bash
# Full metadata (JSON with complete config)
cat processed/flu_a/July_2025_pb1_pb2/experiment_metadata_preprocessing.json
```

---

## **Example Experiments**

### **Experiment 1: PB1 + PB2 only**

**Config:** `conf/bundles/flu_a_pb1_pb2.yaml`

```yaml
run_suffix: "_pb1_pb2"
master_seed: 42
max_files_to_process: 2000

virus:
  selected_functions:
    - "RNA-dependent RNA polymerase PB2 subunit"
    - "RNA-dependent RNA polymerase catalytic core PB1 subunit"
```

**Run:**
```bash
python src/preprocess/preprocess_flu_protein.py \
    --config-name flu_a_pb1_pb2
```

**Directories:**
```
processed/flu_a/July_2025_pb1_pb2/
├── protein_final.csv
├── experiment_metadata_preprocessing.json  # ← Full config
├── EXPERIMENT_SUMMARY_PREPROCESSING.txt    # ← Quick reference
└── ... (other outputs)
```

---

### **Experiment 2: All 9 core proteins**

**Config:** `conf/bundles/flu_a_all_core.yaml`

```yaml
run_suffix: "_all_core"
master_seed: 42
max_files_to_process: 2000

virus:
  selected_functions: ${virus.core_functions}  # All 9 core proteins
```

**Run:**
```bash
python src/preprocess/preprocess_flu_protein.py \
    --config-name flu_a_all_core
```

**Directories:**
```
processed/flu_a/July_2025_all_core/
├── protein_final.csv
├── experiment_metadata_preprocessing.json
├── EXPERIMENT_SUMMARY_PREPROCESSING.txt
└── ...
```

---

### **Experiment 3: Quick test (100 files)**

```bash
# No need for separate config - use CLI overrides
python src/preprocess/preprocess_flu_protein.py \
    --config-name flu_a_pb1_pb2 \
    max_files_to_process=100 \
    run_suffix="_pb1_pb2_test"
```

**Creates:** `processed/flu_a/July_2025_pb1_pb2_test/`

---

## **Comparing Experiments**

### **Quick visual comparison:**

```bash
# List all experiments
ls -1 processed/flu_a/

# Output:
# July_2025_pb1_pb2/
# July_2025_all_core/
# July_2025_pb1_pb2_test/
```

### **Compare two experiments programmatically:**

```python
from pathlib import Path
from src.utils.experiment_utils import compare_experiments

diffs = compare_experiments(
    Path('processed/flu_a/July_2025_pb1_pb2'),
    Path('processed/flu_a/July_2025_all_core'),
    stage='preprocessing'
)

print(diffs['config_differences'])
# Shows: selected_functions, run_suffix, and any other differences
```

---

## **Directory Naming Best Practices**

### **✅ Good `run_suffix` names:**

```yaml
run_suffix: "_pb1_pb2"           # Experiment focus
run_suffix: "_all_core"          # Baseline
run_suffix: "_segment1"          # Specific segment
run_suffix: "_high_quality"      # Data filtering strategy
run_suffix: "_cv_fold1"          # Cross-validation
run_suffix: "_ablation_noM2"     # Ablation study
```

### **❌ Avoid:**

```yaml
run_suffix: "_exp1"              # Not descriptive
run_suffix: "_test"              # Too generic
run_suffix: "_20251013"          # Use timestamp instead
run_suffix: "_seed42_2000_pb1"   # Redundant (seed auto-added)
```

---

## **Metadata Files**

Each pipeline stage creates metadata files:

### **1. `experiment_metadata_{stage}.json`**

**Complete record with:**
- Full Hydra configuration (all parameters)
- Git commit, branch, and dirty status
- Timestamp
- Script name
- Custom metrics (e.g., proteins processed)

**Use for:**
- Perfect reproducibility
- Automated analysis
- Config diffs

### **2. `EXPERIMENT_SUMMARY_{STAGE}.txt`**

**Human-readable summary with:**
- Key configuration values
- Selected proteins
- Data statistics
- Processing time

**Use for:**
- Quick reference
- Lab notebook entries
- README documentation

---

## **Advanced: Experiment Matrix**

For systematic exploration:

```bash
# Create configs for different combinations
conf/bundles/
├── flu_a_pb1_pb2.yaml           # 2 proteins
├── flu_a_pb1_pb2_pa.yaml        # 3 proteins
├── flu_a_all_core.yaml          # 9 proteins
├── flu_a_pb1_pb2_500files.yaml  # Small sample
└── flu_a_pb1_pb2_5000files.yaml # Large sample
```

Then run systematically:
```bash
for config in pb1_pb2 pb1_pb2_pa all_core; do
    python src/preprocess/preprocess_flu_protein.py \
        --config-name flu_a_$config
done
```

---

## **Integration with External Tools**

### **MLflow (Optional)**

For advanced experiment tracking:

```python
import mlflow

mlflow.start_run(run_name="pb1_pb2_experiment")
mlflow.log_params(OmegaConf.to_container(config))
mlflow.log_artifact(output_dir / "experiment_metadata_preprocessing.json")
# ... train model ...
mlflow.log_metrics({"accuracy": 0.95})
mlflow.end_run()
```

### **Weights & Biases (Optional)**

```python
import wandb

wandb.init(
    project="viral-segmatch",
    name="pb1_pb2_experiment",
    config=OmegaConf.to_container(config)
)
# ... train model ...
wandb.log({"accuracy": 0.95})
```

---

## **File Organization**

Recommended structure:

```
data/
├── processed/
│   └── flu_a/
│       ├── July_2025_pb1_pb2/              # Experiment 1
│       │   ├── experiment_metadata_preprocessing.json
│       │   ├── EXPERIMENT_SUMMARY_PREPROCESSING.txt
│       │   └── protein_final.csv
│       ├── July_2025_all_core/             # Experiment 2
│       │   ├── experiment_metadata_preprocessing.json
│       │   ├── EXPERIMENT_SUMMARY_PREPROCESSING.txt
│       │   └── protein_final.csv
│       └── July_2025/                      # Full dataset baseline
│           └── ...
├── embeddings/
│   └── flu_a/
│       ├── July_2025_pb1_pb2/
│       │   ├── experiment_metadata_embeddings.json
│       │   ├── EXPERIMENT_SUMMARY_EMBEDDINGS.txt
│       │   └── esm2_embeddings.h5
│       └── July_2025_all_core/
│           └── ...
└── models/
    └── flu_a/
        ├── July_2025_pb1_pb2/
        │   ├── experiment_metadata_training.json
        │   ├── EXPERIMENT_SUMMARY_TRAINING.txt
        │   └── model_best.pt
        └── July_2025_all_core/
            └── ...
```

---

## **Tips**

1. **Use descriptive `run_suffix`** - Make it meaningful
2. **Review summaries before long runs** - Catch config errors early
3. **Keep configs in git** - Bundle files are your experiment log
4. **Document decisions** - Add comments to bundle configs
5. **Compare metadata files** - Understand what changed between experiments

---

## **See Also**

- `src/utils/experiment_utils.py` - Metadata utilities
- `conf/bundles/` - Example experiment configs
- `docs/SEED_SYSTEM.md` - Reproducibility via seeds


