# Configuration System Status (Updated: October 1, 2025)

## 🎯 Current Architecture: Hydra-Based Bundles

### **Design Philosophy:**
- **Bundles** = Entry points that compose config groups
- **Virus configs** = Immutable biological facts
- **Training configs** = Reusable training recipes
- **Flexible overrides** = Runtime composition via function parameters

---

## ✅ ACTIVE FILES (Current System)

### Directory Structure:
```
conf/
├── bundles/                    # 🎯 ENTRY POINTS
│   └── flu_a.yaml             # ✅ Main entry for Flu A (working!)
│
├── virus/                      # 🧬 BIOLOGICAL FACTS
│   ├── flu_a.yaml             # ✅ Complete with all protein data
│   └── bunya.yaml             # ✅ Created (needs full protein data)
│
├── training/                   # 🏋️ TRAINING RECIPES
│   ├── base.yaml              # ✅ Default training config
│   └── gpu8.yaml              # ✅ High-performance GPU config
│
├── embeddings/                 # 🧠 EMBEDDING CONFIGS
│   ├── default.yaml           # ✅ Standard embeddings
│   └── flu_a_large.yaml       # ✅ Large model embeddings
│
└── paths/                      # 📁 PATH CONFIGS
    └── default.yaml           # ✅ Standard data paths
```

### How It Works:
```python
# Load with defaults from bundle
from src.utils.config_hydra import get_virus_config_hydra
config = get_virus_config_hydra('flu_a')

# Override training config
config = get_virus_config_hydra('flu_a', training_config='gpu8')

# Override multiple configs
config = get_virus_config_hydra('flu_a', 
                                training_config='gpu8',
                                embeddings_config='flu_a_large')

# Access flattened config
print(config.virus.data_version)      # 'July_2025'
print(config.training.batch_size)     # 64
print(config.virus.core_functions)    # List of 8 functions
```

---

## 🗑️ DELETED FILES (Cleaned Up)

### Old Config Systems:
- ❌ `conf/config.yaml` - Old hierarchical config (deleted)
- ❌ `conf/config_generalized.yaml` - Old flat config (deleted)
- ❌ `conf/flu_a_config.yaml` - Old monolithic config (deleted)
- ❌ `conf/simple_config.yaml` - Old fallback config (deleted)
- ❌ `conf/working_config.yaml` - Test file (deleted)
- ❌ `conf/defaults/` - Wrong directory structure (deleted)

### Test/Debug Files:
- ❌ `conf/db/` - Hydra tutorial example (deleted)
- ❌ `conf/hydra_example.yaml` - Tutorial test (deleted)
- ❌ `conf/minimal_config.yaml` - Debug test (deleted)
- ❌ `conf/step1_config.yaml` - Debug test (deleted)
- ❌ `conf/step2_config.yaml` - Debug test (deleted)
- ❌ `conf/test_config.yaml` - Debug test (deleted)
- ❌ `conf/test_single_default.yaml` - Debug test (deleted)

### Old Training Duplicates:
- ❌ `conf/training/default.yaml` - Duplicate of base.yaml (deleted)
- ❌ `conf/training/flu_a.yaml` - Moved to bundles (deleted)
- ❌ `conf/training/fresh_default.yaml` - Test file (deleted)

### Old Python Modules:
- ❌ `src/utils/config_virus.py` - Old config system (deleted)
- ❌ `test_config_system.py` - Old tests (deleted)
- ❌ `test_virus_subset_processing.py` - Old tests (deleted)
- ❌ `test_flu_subset.py` - Old tests (deleted)
- ❌ `test_hydra_*.py` - Debug tests (deleted)

---

## 📦 Python Configuration Modules

### Active:
1. **`src/utils/config_hydra.py`** ✅ **NEW SYSTEM**
   - Hydra-based configuration
   - Used by: `preprocess_flu_protein.py`
   - Features:
     - Loads bundles (`bundles/flu_a.yaml`)
     - Flattens structure for backward compatibility
     - Supports runtime overrides for training, embeddings, paths
     - Clear documentation with examples

2. **`src/utils/config.py`** ⚠️ **LEGACY SYSTEM**
   - Old YAML-based configuration
   - Used by: 5 other scripts (not yet migrated)
     - `analyze_segment_classifier_results.py`
     - `analyze_esm2_embeddings.py`
     - `compute_esm2_embeddings.py`
     - `dataset_segment_pairs.py`
     - `protein_utils.py`
   - **Status:** Keep for now, migrate later

---

## 📝 Configuration File Contents

### `conf/bundles/flu_a.yaml`:
```yaml
defaults:
  - /virus: flu_a
  - /training: base
  - /paths: default
  - /embeddings: default
  - _self_

bundles:
  name: 'flu_a_bundle'
  description: 'Flu A preprocessing and training configuration'
```

### `conf/virus/flu_a.yaml`:
```yaml
virus_name: 'flu_a'
data_version: 'July_2025'
max_core_proteins: 8
max_files_to_process: 100
random_seed: 42

core_functions:
  - 'RNA-dependent RNA polymerase PB2 subunit'
  - 'RNA-dependent RNA polymerase catalytic core PB1 subunit'
  # ... (8 total)

aux_functions:
  - 'Nuclear export protein'
  - 'M2 ion channel'
  # ... (9 total)

segment_mapping:
  'RNA-dependent RNA polymerase PB2 subunit': '1'
  # ... (17 total mappings)

replicon_types:
  - 'Segment 1'
  # ... (8 total)
```

### `conf/training/base.yaml`:
```yaml
batch_size: 64
learning_rate: 0.0001
num_epochs: 10
optimizer: 'AdamW'
```

### `conf/training/gpu8.yaml`:
```yaml
batch_size: 512
learning_rate: 0.001
num_epochs: 20
optimizer: 'AdamW'
```

---

## 🚀 Usage Examples

### In `preprocess_flu_protein.py`:
```python
from src.utils.config_hydra import get_virus_config_hydra

# Load Flu A config with defaults
config = get_virus_config_hydra('flu_a', config_path=config_path)

# Access virus facts
DATA_VERSION = config.virus.data_version
MAX_FILES_TO_PROCESS = config.virus.max_files_to_process
RANDOM_SEED = config.virus.random_seed
core_functions = config.virus.core_functions
aux_functions = config.virus.aux_functions
```

### Override Training Config:
```python
# Override to use gpu8 training
config = get_virus_config_hydra('flu_a', 
                                training_config='gpu8',
                                config_path=config_path)

print(config.training.batch_size)  # 512 (from gpu8.yaml)
```

### Override Multiple Configs:
```python
# Full customization
config = get_virus_config_hydra('flu_a',
                                training_config='gpu8',
                                embeddings_config='flu_a_large',
                                config_path=config_path)
```

---

## 📋 Next Steps (Future Expansion)

### To Add When Needed:
1. **More bundles:**
   - `conf/bundles/bunya.yaml` (when processing Bunyavirales)
   - `conf/bundles/flu_b.yaml` (if adding Flu B)

2. **More config groups:**
   - `conf/dataset/` - Dataset-specific settings
   - `conf/preprocess/` - Preprocessing parameters
   - `conf/model/` - Model architectures

3. **MLOps integration:**
   - `conf/mlflow/` - MLflow tracking configs
   - `conf/wandb/` - Weights & Biases configs

4. **Environment-specific paths:**
   - `conf/paths/lambda_stor.yaml` - Lambda storage paths
   - `conf/paths/local.yaml` - Local development paths

---

## ✅ System Status: READY

**Current State:** ✅ **PRODUCTION READY**

- [x] Configuration system finalized
- [x] Hydra integration complete
- [x] Flu A config fully populated
- [x] Obsolete files cleaned up
- [x] Backward compatibility maintained
- [x] Documentation updated
- [x] Ready to run `preprocess_flu_protein.py`

**Flexibility:** Full runtime override support for:
- ✅ Training configs
- ✅ Embedding configs
- ✅ Path configs
- ✅ Future config groups (easy to add)

---

## 🎓 Key Concepts

### **Bundle Composition:**
Bundles use Hydra's `defaults` to compose multiple config groups:
```yaml
defaults:
  - /virus: flu_a        # Load virus/flu_a.yaml
  - /training: base      # Load training/base.yaml
  - /paths: default      # Load paths/default.yaml
  - /embeddings: default # Load embeddings/default.yaml
  - _self_              # Apply bundle-specific overrides last
```

### **Flattened Access:**
The `get_virus_config_hydra()` function flattens the bundle structure:
- **Internal:** `config.bundles.virus.data_version`
- **External:** `config.virus.data_version` (cleaner for scripts)

### **Runtime Overrides:**
Override any config group at runtime without modifying files:
```python
config = get_virus_config_hydra('flu_a', training_config='gpu8')
```

---

**Last Updated:** October 1, 2025  
**System Version:** 1.0 (Hydra-based)  
**Status:** Production Ready ✅
