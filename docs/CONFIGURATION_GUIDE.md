# Configuration System Guide

**Last Updated**: December 3, 2025  
**System Version**: 2.0 (Hydra-based with Dynamic Paths)  
**Status**: Production Ready ✅

---

## Overview

This guide explains the configuration system for creating and running experiments. Each experiment can have its own:
- Protein selection (specific proteins or all core proteins)
- Dataset sampling (full dataset or subset of isolates)
- Training hyperparameters
- Seeds (for reproducibility)

The system uses **Hydra-based config bundles** that compose reusable configuration groups into complete experiment definitions.

---

## Architecture: Hydra-Based Bundles

### Design Philosophy

- **Bundles** = Entry points that compose config groups
- **Virus configs** = Immutable biological facts
- **Training configs** = Reusable training recipes
- **Dynamic paths** = Auto-generated from sampling parameters
- **Flexible overrides** = Runtime composition via function parameters

### Directory Structure

#### Config Files
```
conf/
├── bundles/                    # 🎯 ENTRY POINTS - Experiment-level configs
│   ├── bunya.yaml             # Bunya baseline experiment
│   ├── flu.yaml               # Flu base config (PB2, PB1, PA)
│   ├── flu_ha_na_5ks.yaml     # Flu: HA and NA only (variable segments)
│   ├── flu_pb2_pb1_pa_5ks.yaml # Flu: Polymerase only (conserved segments)
│   ├── flu_overfit_5ks.yaml    # Flu: Overfitting capacity test (5K isolates)
│   └── flu_plateau_analysis.yaml # Flu: Plateau investigation
│
├── virus/                      # 🧬 BIOLOGICAL FACTS
│   ├── flu.yaml               # Note: virus_name is 'flu' (not 'flu_a')
│   └── bunya.yaml
│
├── paths/                      # 📁 PATH CONFIGS
│   ├── flu.yaml
│   └── bunya.yaml
│
├── embeddings/                 # 🧠 EMBEDDING CONFIGS
│   └── default.yaml           # ESM-2 embedding settings
│
├── dataset/                    # 📊 DATASET CONFIGS
│   └── default.yaml           # Dataset creation settings
│
└── training/                   # 🏋️ TRAINING RECIPES
    └── base.yaml              # Default training config
```

#### Output Directory Structure

**Key Principle:** Preprocessing and embeddings are **shared** per virus/data_version. Datasets and models are **experiment-specific**.

```
data/
├── processed/
│   ├── flu/July_2025/              # Shared preprocessing (no run_suffix)
│   └── bunya/April_2025/            # Shared preprocessing
│
├── embeddings/
│   ├── flu/July_2025/               # Shared embeddings (master cache)
│   └── bunya/April_2025/            # Shared embeddings
│
├── datasets/
│   ├── flu/July_2025/
│   │   └── runs/                    # All dataset experiments
│   │       ├── dataset_flu_ha_na_5ks_20251202_140000/
│   │       ├── dataset_flu_pb2_pb1_pa_5ks_20251202_140100/
│   │       └── dataset_flu_overfit_5ks_20251202_135158/
│   └── bunya/April_2025/
│       └── runs/
│           └── dataset_bunya_20251202_134457/
│
└── models/
    ├── flu/July_2025/
    │   └── runs/                    # All training experiments
    │       ├── training_flu_ha_na_5ks_20251202_141500/
    │       └── training_flu_pb2_pb1_pa_5ks_20251202_142000/
    └── bunya/April_2025/
        └── runs/
            └── training_bunya_20251202_134539/
```

**Important:** The config bundle name is **automatically included** in the directory name (e.g., `dataset_flu_ha_na_5ks_...`), making it easy to identify which experiment created each output.

---

## Dynamic Path Generation

### How It Works

The pipeline automatically generates directory names based on sampling parameters. However, **for dataset and training stages, the config bundle name is used instead of run_suffix**.

| Configuration | Generated Suffix | Example Path |
|--------------|------------------|--------------|
| Full dataset (max_isolates_to_process: null) | _(none)_ | `processed/flu/July_2025/` |
| Deterministic subset (seed: 42, max: 500) | `_seed_42_isolates_500` | `processed/flu/July_2025_seed_42_isolates_500/` |
| Random subset (seed: null, max: 100) | `_random_<timestamp>_isolates_100` | `processed/flu/July_2025_random_20251013_143522_isolates_100/` |
| Manual override (run_suffix: "_v2") | `_v2` | `processed/flu/July_2025_v2/` |

**Note:** For datasets and models, the config bundle name is automatically included in the directory name, so `run_suffix` should typically be `null`.

### Pipeline Path Flow

```
Stage 1: Preprocessing
  └── processed/{virus}/{version}{suffix}/

Stage 2: Embeddings  
  └── embeddings/{virus}/{version}{suffix}/

Stage 3: Datasets
  └── datasets/{virus}/{version}/runs/dataset_{bundle}_{timestamp}/

Stage 4: Training
  └── models/{virus}/{version}/runs/training_{bundle}_{timestamp}/
```

**Key Insight:** Preprocessing and embeddings use the same run identifier. Datasets and models use the config bundle name for identification.

### Output Locations

| Artifact Type | Location | Notes |
|---------------|----------|-------|
| Processed data | `data/processed/{virus}/{version}/` | Shared across experiments |
| Embeddings | `data/embeddings/{virus}/{version}/` | Shared across experiments |
| Datasets | `data/datasets/{virus}/{version}/runs/` | Experiment-specific |
| **Models** | `models/{virus}/{version}/runs/` | **Project root, not data/** |
| Logs | `logs/` | Project root |
| Results | `results/{virus}/{version}/{config_bundle}/` | Analysis outputs |

---

## Config Bundle Anatomy

Each bundle (`conf/bundles/`) defines a complete experiment:

```yaml
# Example: conf/bundles/flu_ha_na_5ks.yaml

defaults:
  - /virus: flu              # Inherit virus biological facts
  - /paths: flu              # Inherit path configurations
  - /embeddings: default     # Inherit embedding settings
  - /dataset: default        # Inherit dataset settings
  - /training: base          # Inherit training settings
  - _self_                   # This bundle's settings override above

# ====================================================================
# DIRECTORY NAMING
# ====================================================================
# Note: run_suffix should be null for dataset/training stages
# The config bundle name is automatically included in output directories
run_suffix: null

# ====================================================================
# SEED MANAGEMENT
# ====================================================================
master_seed: 42  # Reproducible experiment

process_seeds:
  preprocessing: null  # Uses master_seed (42)
  embeddings: null     # Uses master_seed (42)
  dataset: null        # Uses master_seed (42)
  training: null       # Uses master_seed (42)
  evaluation: null     # Uses master_seed (42)

# ====================================================================
# PROTEIN SELECTION
# ====================================================================
virus:
  selected_functions:
    - "Hemagglutinin precursor"  # HA (Segment 4)
    - "Neuraminidase protein"    # NA (Segment 6)

# ====================================================================
# DATASET SETTINGS
# ====================================================================
dataset:
  use_selected_only: true           # Use selected_functions only
  neg_to_pos_ratio: 1.0            # Ratio of negative to positive pairs
  allow_same_func_negatives: false # Disable same-function negatives
  max_isolates_to_process: 5000     # Sample 5000 isolates (null = full dataset)

# ====================================================================
# TRAINING SETTINGS
# ====================================================================
training:
  batch_size: 16
  learning_rate: 0.001
  patience: 15
  early_stopping_metric: 'f1'
  interaction: concat
```

---

## Training: Interaction and Pre-MLP

The model always receives raw embeddings `(emb_a, emb_b)` and computes interaction features internally.
Use `interaction` to specify which features to use; optionally apply pre-MLPs before interaction.

### Interaction spec

```yaml
training:
  interaction: concat            # concat | diff | prod | unit_diff | concat+unit_diff | etc.
```

Examples: `concat`, `diff`, `unit_diff`, `concat+unit_diff`, `concat+diff+prod`.

### Optional pre-MLP (before interaction)

```yaml
training:
  pre_mlp_mode: shared            # none | shared | slot_specific | shared_adapter | slot_norm
  pre_mlp_dims: [1280, 512, 256]  # required for shared/slot_specific/shared_adapter
  adapter_dims: [128]             # required for shared_adapter
  interaction: concat+diff+prod
```

---

## How to Run Experiments

### Full Pipeline (All 4 Stages)

```bash
# Bunya: Complete pipeline
./scripts/stage2_esm2.sh bunya --cuda_name cuda:7
./scripts/stage3_dataset.sh bunya
./scripts/stage4_train.sh bunya --cuda_name cuda:7 \
    --dataset_dir data/datasets/bunya/April_2025/runs/dataset_bunya_YYYYMMDD_HHMMSS
```

### Dataset + Training Only (Most Common)

For Flu experiments, preprocessing and embeddings are already done:

```bash
# Experiment 1: HA-NA (variable segments)
./scripts/stage3_dataset.sh flu_ha_na_5ks
./scripts/stage4_train.sh flu_ha_na_5ks --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS

# Experiment 2: PB2-PB1-PA (conserved segments)
./scripts/stage3_dataset.sh flu_pb2_pb1_pa_5ks
./scripts/stage4_train.sh flu_pb2_pb1_pa_5ks --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_pa_5ks_YYYYMMDD_HHMMSS
```

### Finding Dataset Directories

After running `stage3_dataset.sh`, find the created directory:

```bash
# List most recent dataset runs
ls -lt data/datasets/flu/July_2025/runs/ | head -5
ls -lt data/datasets/bunya/April_2025/runs/ | head -5
```

---

## Output Directory Naming

### Automatic Naming (Current System)

The system automatically creates directory names that include the config bundle:

```
Config Bundle: flu_ha_na_5ks
→ data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_20251202_140000/

Config Bundle: bunya
→ data/datasets/bunya/April_2025/runs/dataset_bunya_20251202_134457/

Config Bundle: flu_overfit_5ks
→ data/datasets/flu/July_2025/runs/dataset_flu_overfit_5ks_20251202_135158/
```

**Key Features:**
- Config bundle name is **always** in the directory name
- Timestamp ensures uniqueness
- All experiments for a virus/data_version are in one `runs/` directory
- Easy to identify which config created each output

### Why `run_suffix` Should Be `null`

The `run_suffix` parameter was previously used to create unique preprocessing directories. However:

1. **Preprocessing is now shared** - All experiments use the same preprocessing output
2. **Config bundle name provides identification** - The bundle name is automatically included in dataset/model directories
3. **Simpler structure** - No need to manually manage suffixes

**Recommendation:** Always set `run_suffix: null` in your config bundles.

---

## Creating a New Experiment

### Step 1: Create a Bundle Config

Create `conf/bundles/my_experiment.yaml`:

```yaml
defaults:
  - /virus: flu
  - /paths: flu
  - /embeddings: default
  - /dataset: default
  - /training: base
  - _self_

# ====================================================================
# DIRECTORY NAMING
# ====================================================================
run_suffix: null  # Always null - config bundle name is used instead

# ====================================================================
# SEED MANAGEMENT
# ====================================================================
master_seed: 42

process_seeds:
  preprocessing: null
  embeddings: null
  dataset: null
  training: null
  evaluation: null

# ====================================================================
# PROTEIN SELECTION
# ====================================================================
virus:
  selected_functions:
    - "Hemagglutinin precursor"
    - "Neuraminidase protein"

# ====================================================================
# DATASET SETTINGS
# ====================================================================
dataset:
  use_selected_only: true
  neg_to_pos_ratio: 1.0
  allow_same_func_negatives: false
  max_isolates_to_process: 1000  # null = full dataset

# ====================================================================
# TRAINING SETTINGS
# ====================================================================
training:
  batch_size: 16
  learning_rate: 0.001
  patience: 15
  early_stopping_metric: 'f1'
  interaction: concat
```

### Step 2: Run the Experiment

```bash
# Create dataset
./scripts/stage3_dataset.sh my_experiment

# Find the dataset directory
ls -lt data/datasets/flu/July_2025/runs/ | head -3

# Train model
./scripts/stage4_train.sh my_experiment --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_my_experiment_YYYYMMDD_HHMMSS
```

---

## Common Experiment Patterns

### 1. Segment-Specific Models (Conservation Hypothesis)

**Test:** Do variable segments (HA/NA) perform better than conserved segments (PB2/PB1/PA)?

```yaml
# conf/bundles/flu_ha_na_5ks.yaml
virus:
  selected_functions:
    - "Hemagglutinin precursor"  # Variable segment
    - "Neuraminidase protein"    # Variable segment

dataset:
  max_isolates_to_process: 5000
```

```yaml
# conf/bundles/flu_pb2_pb1_pa_5ks.yaml
virus:
  selected_functions:
    - "RNA-dependent RNA polymerase PB2 subunit"  # Conserved segment
    - "RNA-dependent RNA polymerase catalytic core PB1 subunit"  # Conserved segment
    - "RNA-dependent RNA polymerase PA subunit"  # Conserved segment

dataset:
  max_isolates_to_process: 5000
```

### 2. Overfitting Capacity Test

**Test:** Can the model overfit a tiny dataset? (Karpathy's capacity test)

```yaml
# conf/bundles/flu_overfit_5ks.yaml
dataset:
  max_isolates_to_process: 5000  # 5K isolates, but only 1% for training
  train_ratio: 0.01
  val_ratio: 0.5

training:
  epochs: 50
  patience: 50  # Disable early stopping
  dropout: 0.0  # No regularization
```

### 3. Full Dataset Baseline

```yaml
# conf/bundles/flu.yaml
dataset:
  max_isolates_to_process: null  # Full dataset
  use_selected_only: true

virus:
  selected_functions:
    - "RNA-dependent RNA polymerase PB2 subunit"
    - "RNA-dependent RNA polymerase catalytic core PB1 subunit"
    - "RNA-dependent RNA polymerase PA subunit"
```

---

## Configuration Parameters Reference

### Embeddings (`conf/embeddings/default.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_ckpt` | `facebook/esm2_t33_650M_UR50D` | ESM-2 model (1280D) |
| `esm2_max_residues` | 1022 | Max sequence length |
| `batch_size` | 64 | Batch size for embedding |
| `use_selected_only` | false | **Currently unused in script** |
| `pooling` | 'mean' | Embedding pooling method |
| `emb_storage_precision` | 'fp16' | Storage precision |

### Dataset (`conf/dataset/default.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_selected_only` | true | Use selected_functions only |
| `neg_to_pos_ratio` | 3.0 | Ratio of negative to positive pairs |
| `allow_same_func_negatives` | false | Allow same-function negatives |
| `max_same_func_ratio` | 0.5 | Max fraction of same-func negatives |

### Training (`conf/training/base.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.001 | Learning rate |
| `epochs` | 100 | Max training epochs |
| `patience` | 10 | Early stopping patience |
| `early_stopping_metric` | 'loss' | Metric for early stopping |

---

## Python API Usage

### Basic Usage

```python
from src.utils.config_hydra import get_virus_config_hydra

# Load config bundle
config = get_virus_config_hydra('bunya')

# Access flattened config
print(config.virus.data_version)      # 'April_2025'
print(config.training.batch_size)     # 16
print(config.virus.selected_functions)  # List of functions
```

### Override Training Config

```python
config = get_virus_config_hydra('flu', training_config='gpu8')
print(config.training.batch_size)  # 512 (from gpu8.yaml)
```

---

## Best Practices

1. **Use descriptive bundle names**: `flu_ha_na_5ks`, not `experiment_1`
2. **Always set `run_suffix: null`** - The config bundle name is automatically used
3. **Set `master_seed` for reproducible experiments**
4. **Document your experiment intent** in the bundle config comments
5. **Use consistent naming**: `flu_ha_na_5ks` clearly indicates "Flu, HA+NA, 5K isolates"
6. **Track your configs in git** to maintain full provenance
7. **Check output directories** - The bundle name should appear in the path

---

## Troubleshooting

### Problem: Can't find my dataset directory

**Solution:** The directory name includes the config bundle and timestamp:
```bash
# List all runs for a virus
ls -lt data/datasets/flu/July_2025/runs/

# Search for a specific bundle
ls -d data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_*
```

### Problem: Different bundle creates same directory

**Solution:** This shouldn't happen - each run gets a unique timestamp. If you see duplicates, check:
1. Are you using the same config bundle name?
2. Did you manually override `--output_dir`?

### Problem: Can't find my bundle

**Solution:** Bundle configs must be in `conf/bundles/` and referenced without the `.yaml` extension:
```bash
# Correct
./scripts/stage3_dataset.sh flu_ha_na_5ks

# Incorrect
./scripts/stage3_dataset.sh bundles/flu_ha_na_5ks.yaml
```

### Problem: Preprocessing file not found

**Solution:** Preprocessing is shared and should exist at:
```
data/processed/{virus}/{data_version}/protein_final.csv
```

The `run_suffix` does NOT affect preprocessing paths - they're always at the base path.

### Problem: Embeddings file not found

**Solution:** Embeddings are shared and should exist at:
```
data/embeddings/{virus}/{data_version}/master_esm2_embeddings.h5
```

If missing, run:
```bash
./scripts/stage2_esm2.sh {config_bundle} --cuda_name cuda:X
```

---

## Current Experiment Configs

| Bundle | Proteins | Isolates | Purpose |
|--------|----------|----------|---------|
| `bunya` | L, M, S | Full | Bunya baseline |
| `flu` | PB2, PB1, PA | Full | Flu base config |
| `flu_ha_na_5ks` | HA, NA | 5K | Variable segments (expect BETTER) |
| `flu_pb2_pb1_pa_5ks` | PB2, PB1, PA | 5K | Conserved segments (expect WORSE) |
| `flu_pb2_ha_na_5ks` | PB2, HA, NA | 5K | Mixed segments |
| `flu_overfit_5ks` | PB2, PB1, PA | 5K | Overfitting capacity test (1% train) |
| `flu_plateau_analysis` | PB2, PB1, PA | 1K | Plateau investigation |

---

## Technical Implementation

### Active Python Modules

- **`src/utils/config_hydra.py`** - Main config system
- **`src/utils/path_utils.py`** - Dynamic path generation
- **`src/utils/seed_utils.py`** - Hierarchical seed management

### Cleanup Notes

**Deleted (Obsolete):**
- `conf/config.yaml` - Old hierarchical config
- `conf/defaults/` - Wrong directory structure
- Various test/debug config files

---

## Related Documentation

### Technical Documentation (`docs/`)
- **`docs/SEED_SYSTEM.md`** - Comprehensive seed hierarchy guide
- **`docs/EXP_RESULTS_STATUS.md`** - Project status and research roadmap
- **`docs/EXPERIMENT_RESULTS_ANALYSIS.md`** - Detailed experiment results analysis
- **`docs/EXPERIMENT_TRACKING_GUIDE.md`** - How to track experiments

### User Guides (`documentation/`)
- **`documentation/quick-start.md`** - Get started quickly
- **`documentation/pipeline-overview.md`** - Understanding the 4-stage pipeline
- **`documentation/troubleshooting.md`** - Common issues and solutions

