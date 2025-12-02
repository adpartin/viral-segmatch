# Configuration System Status

**Last Updated**: December 1, 2025  
**System Version**: 2.0 (Hydra-based with Dynamic Paths)  
**Status**: Production Ready ✅

---

## 🎯 Current Architecture: Hydra-Based Bundles

### Design Philosophy
- **Bundles** = Entry points that compose config groups
- **Virus configs** = Immutable biological facts
- **Training configs** = Reusable training recipes
- **Dynamic paths** = Auto-generated from sampling parameters
- **Flexible overrides** = Runtime composition via function parameters

---

## ✅ Directory Structure

```
conf/
├── bundles/                    # 🎯 ENTRY POINTS
│   ├── bunya.yaml             # ✅ Bunya virus configuration
│   ├── flu_a.yaml             # ✅ Flu A base configuration
│   ├── flu_a_plateau_analysis.yaml  # Plateau investigation
│   └── flu_a_3p_1ks_*.yaml    # Various Flu A experiments
│
├── virus/                      # 🧬 BIOLOGICAL FACTS
│   ├── flu_a.yaml             # Flu A protein data
│   └── bunya.yaml             # Bunya protein data
│
├── training/                   # 🏋️ TRAINING RECIPES
│   └── base.yaml              # Default training config
│
├── embeddings/                 # 🧠 EMBEDDING CONFIGS
│   └── default.yaml           # ESM-2 embedding settings
│
├── dataset/                    # 📊 DATASET CONFIGS
│   └── default.yaml           # Dataset creation settings
│
└── paths/                      # 📁 PATH CONFIGS
    ├── bunya.yaml             # Bunya-specific paths
    └── flu_a.yaml             # Flu A-specific paths
```

---

## 🛤️ Dynamic Path Generation

### How It Works

The pipeline automatically generates directory names based on sampling parameters:

| Configuration | Generated Suffix | Example Path |
|--------------|------------------|--------------|
| Full dataset (max_isolates_to_process: null) | _(none)_ | `processed/flu_a/July_2025/` |
| Deterministic subset (seed: 42, max: 500) | `_seed_42_isolates_500` | `processed/flu_a/July_2025_seed_42_isolates_500/` |
| Random subset (seed: null, max: 100) | `_random_<timestamp>_isolates_100` | `processed/flu_a/July_2025_random_20251013_143522_isolates_100/` |
| Manual override (run_suffix: "_v2") | `_v2` | `processed/flu_a/July_2025_v2/` |

### Pipeline Path Flow

```
Stage 1: Preprocessing
  └── processed/{virus}/{version}{suffix}/

Stage 2: Embeddings  
  └── embeddings/{virus}/{version}{suffix}/

Stage 3: Datasets
  └── datasets/{virus}/{version}{suffix}/

Stage 4: Training
  └── models/{virus}/{version}{suffix}/runs/{run_id}/
```

**Key Insight:** All stages automatically use the same run identifier!

---

## 📦 Output Locations

| Artifact Type | Location | Notes |
|---------------|----------|-------|
| Processed data | `data/processed/{virus}/{version}/` | Shared across experiments |
| Embeddings | `data/embeddings/{virus}/{version}/` | Shared across experiments |
| Datasets | `data/datasets/{virus}/{version}/` | May vary by config |
| **Models** | `models/{virus}/{version}/` | **Project root, not data/** |
| Logs | `logs/` | Project root |
| Results | `results/{virus}/{version}/{config_bundle}/` | Analysis outputs |

---

## 🚀 Usage Examples

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
config = get_virus_config_hydra('flu_a', training_config='gpu8')
print(config.training.batch_size)  # 512 (from gpu8.yaml)
```

---

## 📋 Bundle Configuration Reference

### Key Bundle Settings

```yaml
# conf/bundles/bunya.yaml
defaults:
  - /virus: bunya
  - /paths: bunya
  - /embeddings: default
  - /dataset: default
  - /training: base
  - _self_

# Seed management
master_seed: 42
process_seeds:
  preprocessing: null  # Derive from master_seed
  embeddings: null
  dataset: null
  training: null

# Data sampling
max_isolates_to_process: null  # null = full dataset

# Directory naming
run_suffix: null  # null = auto-generate, or set manually

# Dataset settings
dataset:
  use_selected_only: true
  allow_same_func_negatives: true
  max_same_func_ratio: 0.5

# Training settings
training:
  use_diff: true
  use_prod: true
  patience: 15
  early_stopping_metric: 'f1'
```

---

## ⚙️ Configuration Parameters

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

## 🧹 Cleanup Notes

### Deleted (Obsolete)
- `conf/config.yaml` - Old hierarchical config
- `conf/defaults/` - Wrong directory structure
- Various test/debug config files

### Active Python Modules
- **`src/utils/config_hydra.py`** - Main config system
- **`src/utils/path_utils.py`** - Dynamic path generation
- **`src/utils/seed_utils.py`** - Hierarchical seed management

---

## 📚 Related Documentation

- `docs/SEED_SYSTEM.md` - Comprehensive seed hierarchy guide
- `docs/EXPERIMENT_CONFIGS.md` - How to create config bundles
- `docs/PLATEAU_DIAGNOSIS_PLAN.md` - Current investigation status
