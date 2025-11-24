# Experiment Configuration Guide

## Overview

This guide explains how to create and run different experiments using the config bundle system. Each experiment can have its own:
- Protein selection (specific proteins or all core proteins)
- Data sampling (full dataset or subset)
- Seeds (for reproducibility)
- Custom directory naming

## Directory Structure

```
conf/
├── bundles/                    # Experiment-level configs
│   ├── flu_a.yaml             # Default Flu A (all core + aux proteins)
│   ├── flu_a_pb1_pb2.yaml     # Custom: only PB1 and PB2 proteins
│   └── flu_a_all_core.yaml    # Custom: all 9 core proteins only
├── virus/                      # Virus biological facts
│   ├── flu_a.yaml
│   └── bunya.yaml
├── embeddings/                 # ESM-2 embedding settings
│   └── default.yaml
└── training/                   # Model training settings
    └── base.yaml
```

## Config Bundle Anatomy

Each bundle (`conf/bundles/`) defines a complete experiment:

```yaml
# Example: conf/bundles/flu_a_pb1_pb2.yaml

defaults:
  - /virus: flu_a              # Inherit virus biological facts
  - /embeddings: default       # Inherit embedding settings
  - /training: base            # Inherit training settings
  - _self_                     # This bundle's settings override above

# ==============================================================================
# EXPERIMENT IDENTIFICATION
# ==============================================================================
# Manual directory suffix - creates: data/processed/flu_a/July_2025_seed_42_GTOs_2000_pb1_pb2/
run_suffix: "_seed_42_GTOs_2000_pb1_pb2"

# ==============================================================================
# SEED MANAGEMENT
# ==============================================================================
master_seed: 42  # Reproducible experiment

process_seeds:
  preprocessing: null  # Uses master_seed (42)
  embeddings: null     # Uses master_seed (42)
  dataset: null        # Uses master_seed (42)
  training: null       # Uses master_seed (42)
  evaluation: null     # Uses master_seed (42)

# ==============================================================================
# DATA SAMPLING
# ==============================================================================
max_files_to_process: 2000  # Process 2000 GTO files (subset)

# ==============================================================================
# PROTEIN SELECTION
# ==============================================================================
virus:
  selected_functions:
    - "RNA-dependent RNA polymerase PB2 subunit"  # PB2
    - "RNA-dependent RNA polymerase catalytic core PB1 subunit"  # PB1

# ==============================================================================
# MODEL SETTINGS (Override defaults if needed)
# ==============================================================================
embeddings:
  batch_size: 16

training:
  batch_size: 128
```

## How to Run Experiments

### Method 1: Using Shell Scripts (Recommended)

```bash
# Run PB1+PB2 experiment
bash scripts/preprocess_flu_a_pb1_pb2.sh

# The script internally calls:
python src/preprocess/preprocess_flu_protein.py \
    --virus_name flu_a \
    --config_bundle flu_a_pb1_pb2
```

### Method 2: Direct Python Command

```bash
# Run with specific bundle
python src/preprocess/preprocess_flu_protein.py \
    --virus_name flu_a \
    --config_bundle flu_a_pb1_pb2

# Run with default bundle (uses flu_a.yaml)
python src/preprocess/preprocess_flu_protein.py \
    --virus_name flu_a
```

### Method 3: Command-Line Overrides (Advanced)

You can override any config parameter on the fly:

```bash
# Override run_suffix
python src/preprocess/preprocess_flu_protein.py \
    --virus_name flu_a \
    --config_bundle flu_a_pb1_pb2 \
    run_suffix="_test_experiment"

# Override max_files_to_process
python src/preprocess/preprocess_flu_protein.py \
    --virus_name flu_a \
    --config_bundle flu_a_pb1_pb2 \
    max_files_to_process=100

# Override selected_functions (complex override)
python src/preprocess/preprocess_flu_protein.py \
    --virus_name flu_a \
    'virus.selected_functions=["RNA-dependent RNA polymerase PB2 subunit"]'
```

## Output Directory Naming

### Automatic Naming (when `run_suffix` is NOT set in config)

The system auto-generates directory names based on sampling:

```
max_files_to_process=2000, master_seed=42
→ data/processed/flu_a/July_2025_seed_42_GTOs_2000/

max_files_to_process=100, master_seed=null
→ data/processed/flu_a/July_2025_random_20251014_153045_GTOs_100/

max_files_to_process=null (full dataset)
→ data/processed/flu_a/July_2025/
```

### Manual Naming (when `run_suffix` IS set in config)

You have full control over the directory name:

```yaml
# conf/bundles/flu_a_pb1_pb2.yaml
run_suffix: "_seed_42_GTOs_2000_pb1_pb2"
```

Creates:
```
data/processed/flu_a/July_2025_seed_42_GTOs_2000_pb1_pb2/
```

**Recommendation:** Use manual naming for experiments you'll reference frequently (like "pb1_pb2", "all_core") and auto-naming for quick tests.

## Creating a New Experiment

### Step 1: Create a Bundle Config

Create `conf/bundles/my_experiment.yaml`:

```yaml
defaults:
  - /virus: flu_a
  - /embeddings: default
  - /training: base
  - _self_

# Custom directory name
run_suffix: "_seed_42_GTOs_5000_ha_na"

# Reproducibility
master_seed: 42

process_seeds:
  preprocessing: null
  embeddings: null
  dataset: null
  training: null
  evaluation: null

# Data sampling
max_files_to_process: 5000

# Protein selection: Only HA and NA
virus:
  selected_functions:
    - "Hemagglutinin precursor"
    - "Neuraminidase"

# Model settings
embeddings:
  batch_size: 16

training:
  batch_size: 128
```

### Step 2: Create a Shell Script (Optional but Recommended)

Create `scripts/preprocess_my_experiment.sh`:

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG_BUNDLE="my_experiment"
VIRUS_NAME="flu_a"

python src/preprocess/preprocess_flu_protein.py \
    --virus_name "$VIRUS_NAME" \
    --config_bundle "$CONFIG_BUNDLE"
```

### Step 3: Run It

```bash
chmod +x scripts/preprocess_my_experiment.sh
bash scripts/preprocess_my_experiment.sh
```

## Common Experiment Patterns

### 1. Full Dataset, All Proteins (Default)

```yaml
# conf/bundles/flu_a_full.yaml
master_seed: 42
max_files_to_process: null  # Process ALL files
run_suffix: null            # Auto-generates: July_2025/

virus:
  selected_functions: ${virus.core_functions}  # All core proteins
```

### 2. Small Subset for Testing

```yaml
# conf/bundles/flu_a_test.yaml
master_seed: 42
max_files_to_process: 100   # Only 100 files
run_suffix: "_test"         # Manual name

virus:
  selected_functions: ${virus.core_functions}
```

### 3. Specific Proteins, Large Dataset

```yaml
# conf/bundles/flu_a_polymerase.yaml
master_seed: 42
max_files_to_process: 10000
run_suffix: "_seed_42_GTOs_10000_polymerase"

virus:
  selected_functions:
    - "RNA-dependent RNA polymerase PB2 subunit"
    - "RNA-dependent RNA polymerase catalytic core PB1 subunit"
    - "RNA-dependent RNA polymerase PA subunit"
```

### 4. Non-Reproducible (Random) Experiment

```yaml
# conf/bundles/flu_a_random.yaml
master_seed: null           # Random seeds everywhere
max_files_to_process: 1000
run_suffix: null            # Auto-generates with timestamp

virus:
  selected_functions: ${virus.core_functions}
```

Creates: `July_2025_random_20251014_153045_GTOs_1000/`

## Best Practices

1. **Use descriptive bundle names**: `flu_a_pb1_pb2`, not `experiment_1`
2. **Set run_suffix explicitly for important experiments** you'll reference later
3. **Use auto-generated suffixes for quick tests** (set `run_suffix: null`)
4. **Always set master_seed for reproducible experiments**
5. **Document your experiment intent** in the bundle config comments
6. **Create shell scripts** for experiments you'll run multiple times
7. **Track your configs in git** to maintain full provenance

## Troubleshooting

### Problem: Directory name doesn't match my expectation

**Solution:** Check if `run_suffix` is set in the bundle. If set, it overrides auto-generation.

### Problem: Different bundle creates same directory

**Solution:** Make sure each bundle has a unique `run_suffix` or unique sampling parameters (max_files, seed).

### Problem: Can't find my bundle

**Solution:** Bundle configs must be in `conf/bundles/` and referenced without the `.yaml` extension:
```bash
# Correct
--config_bundle flu_a_pb1_pb2

# Incorrect
--config_bundle bundles/flu_a_pb1_pb2.yaml
```

### Problem: Override not working

**Solution:** Use Hydra override syntax:
```bash
# Correct
max_files_to_process=100

# Incorrect
--max_files_to_process=100
```

For nested configs:
```bash
# Correct
virus.selected_functions='["PB2"]'

# Incorrect
selected_functions='["PB2"]'
```

