# Configuration System

**Note**: This is a brief overview. For comprehensive configuration documentation, see [`../docs/CONFIGURATION_GUIDE.md`](../docs/CONFIGURATION_GUIDE.md).

The viral-segmatch project uses Hydra for configuration management, providing a flexible and reproducible way to manage experiments.

## 🏗️ Quick Overview

### Bundle System
Configuration bundles are stored in `conf/bundles/` and define complete experiment setups:

```
conf/bundles/
├── bunya.yaml                 # Bunyavirus baseline
├── flu.yaml                   # Flu base config (PB2, PB1, PA)
├── flu_ha_na_5ks.yaml         # Flu: HA and NA only (variable segments)
├── flu_pb2_pb1_pa_5ks.yaml    # Flu: Polymerase only (conserved segments)
├── flu_pb2_ha_na_5ks.yaml     # Flu: PB2-HA-NA (mixed)
└── flu_overfit_5ks.yaml        # Flu: Overfitting capacity test (5K isolates)
```

### Current Configurations

| Bundle | Proteins | Isolates | Purpose |
|--------|----------|----------|---------|
| `bunya` | L, M, S | Full | Bunya baseline |
| `flu` | PB2, PB1, PA | Full | Flu base config |
| `flu_ha_na_5ks` | HA, NA | 5K | Variable segments (expect BETTER) |
| `flu_pb2_pb1_pa_5ks` | PB2, PB1, PA | 5K | Conserved segments (expect WORSE) |
| `flu_pb2_ha_na_5ks` | PB2, HA, NA | 5K | Mixed segments |
| `flu_overfit_5ks` | PB2, PB1, PA | 5K | Overfitting capacity test (1% train) |

## 📋 Basic Bundle Structure

```yaml
# Example: conf/bundles/flu_ha_na_5ks.yaml

defaults:
  - /virus: flu              # Inherit virus biological facts
  - /paths: flu              # Inherit path configurations
  - /embeddings: default     # Inherit embedding settings
  - /dataset: default        # Inherit dataset settings
  - /training: base          # Inherit training settings
  - _self_                   # This bundle's settings override above

run_suffix: null  # Always null - config bundle name is used instead

master_seed: 42

virus:
  selected_functions:
    - "Hemagglutinin precursor"  # HA (Segment 4)
    - "Neuraminidase protein"    # NA (Segment 6)

dataset:
  use_selected_only: true
  neg_to_pos_ratio: 1.0
  allow_same_func_negatives: false
  max_isolates_to_process: 5000  # null = full dataset

training:
  batch_size: 16
  learning_rate: 0.001
  patience: 15
  early_stopping_metric: 'f1'
```

## 🚀 Using Configurations

### Command Line Usage
```bash
# Stage 2: Embeddings
./scripts/stage2_esm2.sh flu --cuda_name cuda:7

# Stage 3: Dataset creation
./scripts/stage3_dataset.sh flu_ha_na_5ks

# Stage 4: Training
./scripts/stage4_train.sh flu_ha_na_5ks --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS
```

### Python API
```python
from src.utils.config_hydra import get_virus_config_hydra

# Load configuration
config = get_virus_config_hydra('flu_ha_na_5ks')

# Access parameters
print(f"Virus: {config.virus.virus_name}")
print(f"Max isolates: {config.max_isolates_to_process}")
```

## 📁 Path Structure

The system automatically generates paths with the config bundle name:

```
data/
├── processed/flu/July_2025/              # Shared preprocessing
├── embeddings/flu/July_2025/            # Shared embeddings (master cache)
├── datasets/flu/July_2025/
│   └── runs/                            # Experiment-specific
│       └── dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS/
└── models/flu/July_2025/
    └── runs/                            # Experiment-specific
        └── training_flu_ha_na_5ks_YYYYMMDD_HHMMSS/
```

**Key Principle**: Preprocessing and embeddings are **shared** per virus/data_version. Datasets and models are **experiment-specific** in `runs/` subdirectories.

## 🔧 Key Parameters

### Data Sampling
```yaml
max_isolates_to_process: 5000    # Number of isolates (null = full dataset)
master_seed: 42                   # Random seed for reproducibility
```

### Dataset Settings
```yaml
dataset:
  neg_to_pos_ratio: 1.0            # Ratio of negative to positive pairs
  allow_same_func_negatives: false # Disable same-function negatives
  use_selected_only: true          # Use selected_functions only
```

### Training Settings
```yaml
training:
  batch_size: 16
  learning_rate: 0.001
  patience: 15
  early_stopping_metric: 'f1'
  use_diff: false                  # Interaction features
  use_prod: false                  # Interaction features
```

## 📚 For More Information

- **[Full Configuration Guide](../docs/CONFIGURATION_GUIDE.md)** - Comprehensive configuration documentation
- **[Experiment Results](../docs/EXPERIMENT_RESULTS_ANALYSIS.md)** - Current experiment results and analysis
- **[Project Status](../docs/EXP_RESULTS_STATUS.md)** - Research status and roadmap
