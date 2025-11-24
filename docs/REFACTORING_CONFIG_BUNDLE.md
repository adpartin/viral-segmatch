# Refactoring: Unified `config_bundle` System

**Date**: 2025-10-14  
**Status**: ✅ Complete

---

## Overview

This document describes the complete refactoring that unified all pipeline scripts to use `--config_bundle` as the single source of truth for experiment configuration. The `virus_name` parameter is now **derived from the config** rather than passed as a CLI argument.

---

## Motivation

**Before this refactoring:**
- Scripts required multiple CLI arguments: `--virus_name`, `--data_version`, `--input_file`, `--output_dir`
- Shell scripts hardcoded paths based on `VIRUS_NAME` and `DATA_VERSION`
- Inconsistent parameter passing across preprocessing, embeddings, and training stages
- Risk of mismatched paths when parameters were not synchronized

**After this refactoring:**
- **One parameter to rule them all**: `--config_bundle`
- All configuration derived from bundle configs: `conf/bundles/{bundle_name}.yaml`
- Automatic path generation ensures consistency across pipeline stages
- Easy experiment switching: just change the bundle name

---

## Changes by Script

### 1. Preprocessing Scripts

#### **Python: `src/preprocess/preprocess_flu_protein.py`**

**Before:**
```python
parser.add_argument('--virus_name', type=str, default='flu_a')
args = parser.parse_args()
config = get_virus_config_hydra(args.virus_name, config_path=config_path)
```

**After:**
```python
parser.add_argument('--config_bundle', type=str, default=None)
parser.add_argument('--virus_name', type=str, default=None, help='[DEPRECATED]')
args = parser.parse_args()

# Backward compatibility
config_bundle = args.config_bundle if args.config_bundle else args.virus_name
if config_bundle is None:
    raise ValueError("Either --config_bundle or --virus_name must be provided")

config = get_virus_config_hydra(config_bundle, config_path=config_path)

# Extract virus_name from config (no longer from CLI)
VIRUS_NAME = config.virus.virus_name
```

**What gets derived from config:**
- `VIRUS_NAME` → `config.virus.virus_name`
- `DATA_VERSION` → `config.virus.data_version`
- `MAX_FILES_TO_PROCESS` → `config.max_files_to_process`
- `RANDOM_SEED` → `resolve_process_seed(config, 'preprocessing')`
- `selected_functions` → `config.virus.selected_functions`
- `RUN_SUFFIX` → `config.run_suffix` (or auto-generated)

#### **Shell Scripts:**
- `scripts/preprocess_flu_a_protein.sh` → uses `CONFIG_BUNDLE="flu_a"`
- `scripts/preprocess_bunya_protein.sh` → uses `CONFIG_BUNDLE="bunya"`
- `scripts/preprocess_flu_a_pb1_pb2.sh` → uses `CONFIG_BUNDLE="flu_a_pb1_pb2"`

**Before:**
```bash
VIRUS_NAME="flu_a"
python src/preprocess/preprocess_flu_protein.py --virus_name "$VIRUS_NAME"
```

**After:**
```bash
CONFIG_BUNDLE="flu_a"
python src/preprocess/preprocess_flu_protein.py --config_bundle "$CONFIG_BUNDLE"
```

---

### 2. Embeddings Scripts

#### **Python: `src/embeddings/compute_esm2_embeddings.py`**

**Before:**
```python
parser.add_argument('--virus_name', '-v', type=str, default=None)
args = parser.parse_args()
config = get_virus_config_hydra(args.virus_name, config_path=config_path)

# Hardcoded paths
input_base_dir = main_data_dir / 'processed' / args.virus_name / run_dir
output_base_dir = main_data_dir / 'embeddings' / args.virus_name / run_dir
```

**After:**
```python
parser.add_argument('--config_bundle', type=str, default=None)
parser.add_argument('--virus_name', '-v', type=str, default=None, help='[DEPRECATED]')
args = parser.parse_args()

# Backward compatibility
config_bundle = args.config_bundle if args.config_bundle else args.virus_name
if config_bundle is None:
    raise ValueError("Either --config_bundle or --virus_name must be provided")

config = get_virus_config_hydra(config_bundle, config_path=config_path)

# Extract virus_name from config
VIRUS_NAME = config.virus.virus_name

# Paths derived from config
input_base_dir = main_data_dir / 'processed' / VIRUS_NAME / run_dir
output_base_dir = main_data_dir / 'embeddings' / VIRUS_NAME / run_dir
```

**What gets derived from config:**
- `VIRUS_NAME` → `config.virus.virus_name`
- `DATA_VERSION` → `config.virus.data_version`
- `RUN_SUFFIX` → `config.run_suffix`
- `RANDOM_SEED` → `resolve_process_seed(config, 'embeddings')`
- `BATCH_SIZE` → `config.embeddings.batch_size`
- `selected_functions` → `config.virus.selected_functions` (for `--use_selected_only`)

#### **Shell Scripts:**
- `scripts/compute_esm2_embeddings_flu_a.sh` → uses `CONFIG_BUNDLE="flu_a"`
- `scripts/compute_esm2_embeddings_bunya.sh` → uses `CONFIG_BUNDLE="bunya"`

**Before:**
```bash
VIRUS_NAME="flu_a"
DATA_VERSION="July_2025"
INPUT_FILE="$PROJECT_ROOT/data/processed/${VIRUS_NAME}/${DATA_VERSION}/protein_final.csv"
OUTPUT_DIR="$PROJECT_ROOT/data/embeddings/${VIRUS_NAME}/${DATA_VERSION}"

python src/embeddings/compute_esm2_embeddings.py \
    --virus_name "$VIRUS_NAME" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --use_selected_only \
    --cuda_name "$CUDA_NAME"
```

**After:**
```bash
CONFIG_BUNDLE="flu_a"

python src/embeddings/compute_esm2_embeddings.py \
    --config_bundle "$CONFIG_BUNDLE" \
    --use_selected_only \
    --cuda_name "$CUDA_NAME"
```

**Benefits:**
- 🎯 **Simpler**: No need to specify paths manually
- ✅ **Consistent**: Paths automatically match preprocessing output
- 🔧 **Flexible**: Change bundle name to switch experiments
- 📊 **Traceable**: Config bundle name saved in logs and metadata

---

## Example Workflows

### **Workflow 1: Default Flu A (Full Dataset)**

```bash
# Step 1: Preprocess
bash scripts/preprocess_flu_a_protein.sh
# Uses: conf/bundles/flu_a.yaml
# Output: data/processed/flu_a/July_2025/

# Step 2: Compute embeddings
bash scripts/compute_esm2_embeddings_flu_a.sh
# Uses: conf/bundles/flu_a.yaml
# Input:  data/processed/flu_a/July_2025/protein_final.csv
# Output: data/embeddings/flu_a/July_2025/
```

---

### **Workflow 2: PB1+PB2 Subset Experiment**

```bash
# Step 1: Preprocess (using custom bundle)
bash scripts/preprocess_flu_a_pb1_pb2.sh
# Uses: conf/bundles/flu_a_pb1_pb2.yaml
# - master_seed: 42
# - max_files_to_process: 2000
# - selected_functions: [PB1, PB2]
# - run_suffix: "_seed_42_GTOs_2000_pb1_pb2"
# Output: data/processed/flu_a/July_2025_seed_42_GTOs_2000_pb1_pb2/

# Step 2: Compute embeddings (create similar script)
# scripts/compute_esm2_embeddings_flu_a_pb1_pb2.sh
CONFIG_BUNDLE="flu_a_pb1_pb2"
python src/embeddings/compute_esm2_embeddings.py \
    --config_bundle "$CONFIG_BUNDLE" \
    --use_selected_only \
    --cuda_name cuda:0
# Uses: conf/bundles/flu_a_pb1_pb2.yaml
# Input:  data/processed/flu_a/July_2025_seed_42_GTOs_2000_pb1_pb2/protein_final.csv
# Output: data/embeddings/flu_a/July_2025_seed_42_GTOs_2000_pb1_pb2/
```

---

### **Workflow 3: Manual Override (Advanced)**

You can still override specific paths if needed:

```bash
python src/embeddings/compute_esm2_embeddings.py \
    --config_bundle flu_a_pb1_pb2 \
    --input_file /custom/path/protein_final.csv \
    --output_dir /custom/output/dir \
    --cuda_name cuda:3
```

---

## Config Bundle Structure

### Example: `conf/bundles/flu_a_pb1_pb2.yaml`

```yaml
defaults:
  - /virus/flu_a
  - /embeddings/default
  - /training/default
  - /paths/default
  - _self_

# Experiment-level settings
master_seed: 42
max_files_to_process: 2000
run_suffix: "_seed_42_GTOs_2000_pb1_pb2"  # Manual override

# Process-specific seeds (null = use master_seed directly)
process_seeds:
  preprocessing: null
  embeddings: null
  training: null

# Override virus config: select only PB1 and PB2
virus:
  selected_functions:
    - PB1
    - PB2
```

---

## Backward Compatibility

The `--virus_name` argument is **still accepted** but marked as deprecated:

```bash
# Old style (still works)
python src/embeddings/compute_esm2_embeddings.py --virus_name flu_a

# New style (recommended)
python src/embeddings/compute_esm2_embeddings.py --config_bundle flu_a
```

**Implementation:**
```python
config_bundle = args.config_bundle if args.config_bundle else args.virus_name
if config_bundle is None:
    raise ValueError("Either --config_bundle or --virus_name must be provided")
```

---

## Files Modified

### Python Scripts (2 files)
1. ✅ `src/preprocess/preprocess_flu_protein.py`
   - Added `--config_bundle` parameter
   - Deprecated `--virus_name` parameter
   - Extract `VIRUS_NAME` from `config.virus.virus_name`
   - Updated experiment metadata to include `config_bundle`

2. ✅ `src/embeddings/compute_esm2_embeddings.py`
   - Added `--config_bundle` parameter
   - Deprecated `--virus_name` parameter
   - Extract `VIRUS_NAME` from `config.virus.virus_name`
   - Removed hardcoded path logic

### Shell Scripts (6 files)
1. ✅ `scripts/preprocess_flu_a_protein.sh`
2. ✅ `scripts/preprocess_bunya_protein.sh`
3. ✅ `scripts/preprocess_flu_a_pb1_pb2.sh`
4. ✅ `scripts/compute_esm2_embeddings_flu_a.sh`
5. ✅ `scripts/compute_esm2_embeddings_bunya.sh`
6. (To be created) `scripts/compute_esm2_embeddings_flu_a_pb1_pb2.sh`

**Common changes:**
- Replace `VIRUS_NAME=` with `CONFIG_BUNDLE=`
- Update command to use `--config_bundle` instead of `--virus_name`
- Remove hardcoded `INPUT_FILE` and `OUTPUT_DIR` variables
- Update log file names and symlinks

---

## Key Benefits

### 1. **Simplicity**
- **Before**: 3-5 parameters per script (`--virus_name`, `--data_version`, `--input_file`, `--output_dir`, etc.)
- **After**: 1 parameter (`--config_bundle`)

### 2. **Consistency**
- All pipeline stages automatically use matching paths
- No risk of path mismatches between preprocessing and embeddings
- Configuration is centralized in bundle files

### 3. **Flexibility**
- Easy to create new experiments: just create a new bundle config
- Switch experiments by changing one variable: `CONFIG_BUNDLE="..."`
- Override specific settings if needed (manual paths still supported)

### 4. **Traceability**
- Config bundle name logged in all output
- Experiment metadata includes `config_bundle` field
- Easy to reproduce experiments: `--config_bundle flu_a_pb1_pb2`

### 5. **Maintainability**
- Single source of truth for all configuration
- Easier to update and extend
- Less code duplication in shell scripts

---

## Migration Guide

### For Users

**Old way:**
```bash
python src/preprocess/preprocess_flu_protein.py --virus_name flu_a
python src/embeddings/compute_esm2_embeddings.py --virus_name flu_a --input_file ... --output_dir ...
```

**New way:**
```bash
python src/preprocess/preprocess_flu_protein.py --config_bundle flu_a
python src/embeddings/compute_esm2_embeddings.py --config_bundle flu_a
```

### For Shell Scripts

**Old way:**
```bash
VIRUS_NAME="flu_a"
DATA_VERSION="July_2025"
INPUT_FILE="$PROJECT_ROOT/data/processed/${VIRUS_NAME}/${DATA_VERSION}/protein_final.csv"
OUTPUT_DIR="$PROJECT_ROOT/data/embeddings/${VIRUS_NAME}/${DATA_VERSION}"

python ... --virus_name "$VIRUS_NAME" --input_file "$INPUT_FILE" --output_dir "$OUTPUT_DIR"
```

**New way:**
```bash
CONFIG_BUNDLE="flu_a"
python ... --config_bundle "$CONFIG_BUNDLE"
```

---

## Testing

### Verify Preprocessing
```bash
# Should work with new parameter
python src/preprocess/preprocess_flu_protein.py --config_bundle flu_a

# Should still work with old parameter (backward compatibility)
python src/preprocess/preprocess_flu_protein.py --virus_name flu_a
```

### Verify Embeddings
```bash
# Should work with new parameter
python src/embeddings/compute_esm2_embeddings.py --config_bundle flu_a --cuda_name cuda:0

# Should still work with old parameter (backward compatibility)
python src/embeddings/compute_esm2_embeddings.py --virus_name flu_a --cuda_name cuda:0
```

### Verify Path Consistency
```bash
# After preprocessing with flu_a_pb1_pb2 bundle:
# Output: data/processed/flu_a/July_2025_seed_42_GTOs_2000_pb1_pb2/

# Embeddings should automatically use the same path:
python src/embeddings/compute_esm2_embeddings.py --config_bundle flu_a_pb1_pb2 --cuda_name cuda:0
# Input:  data/processed/flu_a/July_2025_seed_42_GTOs_2000_pb1_pb2/protein_final.csv
# Output: data/embeddings/flu_a/July_2025_seed_42_GTOs_2000_pb1_pb2/
```

---

## Future Work

### Potential Improvements
1. **Remove backward compatibility** in a future version (deprecated `--virus_name`)
2. **Add validation** to ensure bundle configs are complete
3. **Create bundle templates** for common experiment patterns
4. **Auto-generate shell scripts** from bundle configs
5. **Add bundle discovery** command to list available bundles

### Additional Scripts to Update
- Training scripts (when implemented)
- Evaluation scripts (when implemented)
- Analysis scripts (when implemented)

---

## Summary

This refactoring establishes a **clean, consistent, and scalable** configuration system across the entire viral-segmatch pipeline. By using `config_bundle` as the single source of truth, we've:

✅ Simplified CLI interfaces  
✅ Eliminated hardcoded paths  
✅ Ensured consistency across pipeline stages  
✅ Improved experiment traceability  
✅ Made the codebase more maintainable  

**One parameter to rule them all!** 🎯

