# Dynamic Path Generation Implementation

## Summary

Implemented a dynamic path generation system that automatically creates directory names based on sampling parameters (max_files_to_process and seed). This eliminates the need for predefined subset directories and makes the pipeline more flexible and reproducible.

## Changes Made

### 1. New Utility File: `src/utils/path_utils.py`

**Purpose:** Centralized utilities for dynamic path and directory naming across the pipeline.

**Key Functions:**
- `generate_run_suffix()` - Auto-generates suffix from sampling parameters
- `resolve_run_suffix()` - Handles manual override vs auto-generation
- `build_preprocessing_paths()` - Creates standard directory structure

**Benefits:**
- Reusable across all virus preprocessing scripts
- Consistent naming convention throughout pipeline
- Easy to unit test in isolation

---

### 2. Updated: `src/preprocess/preprocess_flu_protein.py`

**Changes:**
1. Import new path utilities
2. Removed hardcoded subset directory preference logic
3. Added automatic run_suffix generation based on:
   - `max_files_to_process` (from config)
   - `RANDOM_SEED` (resolved via hierarchical seed system)
4. Uses `build_preprocessing_paths()` for consistent directory structure

**Before:**
```python
# Hardcoded logic with subset preference
if subset_data_dir.exists():
    raw_data_dir = subset_data_dir
    output_dir = main_data_dir / 'processed' / args.virus_name / f'{DATA_VERSION}_subset_5k'
else:
    raw_data_dir = base_data_dir
    output_dir = main_data_dir / 'processed' / args.virus_name / DATA_VERSION
```

**After:**
```python
# Dynamic path generation
RUN_SUFFIX = resolve_run_suffix(config, MAX_FILES_TO_PROCESS, RANDOM_SEED, auto_timestamp=True)
paths = build_preprocessing_paths(project_root, args.virus_name, DATA_VERSION, RUN_SUFFIX)
raw_data_dir = paths['raw_dir']
output_dir = paths['output_dir']
```

---

### 3. Updated: `conf/bundles/flu_a.yaml`

**Changes:**
1. Changed `master_seed` from `null` to `42` (deterministic by default)
2. Changed `max_files_to_process` from `100` to `null` (full dataset by default)
3. Changed `process_seeds.embeddings` from `42` to `null` (derive from master)
4. Added comprehensive documentation with examples
5. Organized into clear sections: SEED MANAGEMENT, DATA SAMPLING, DIRECTORY NAMING

**Key Improvements:**
- Clear documentation of directory naming behavior
- Examples showing expected output paths
- Better default values for production use

---

### 4. Updated: `conf/bundles/bunya.yaml`

**Changes:**
- Same improvements as flu_a.yaml
- Consistent structure and documentation
- Updated default seeds to match flu_a

---

### 5. Updated: `src/embeddings/compute_esm2_embeddings.py`

**Changes:**
- Added explicit backward compatibility comment for `run_suffix` usage
- Explains that downstream scripts read `run_suffix` from config

**Comment Added:**
```python
# BACKWARD COMPATIBILITY: Read run_suffix from config
# This maintains compatibility with the dynamic run_suffix system introduced in preprocessing.
# The preprocessing script auto-generates run_suffix from max_files_to_process and seed,
# and downstream scripts (embeddings, training) read it from config to use matching directories.
RUN_SUFFIX = config.run_suffix if config.run_suffix else ''
```

---

### 6. New Documentation: `docs/SEED_SYSTEM.md`

**Purpose:** Comprehensive guide to the hierarchical seed system and directory naming.

**Contents:**
- Overview of seed hierarchy (master → process → random)
- 5 detailed configuration examples with use cases
- Directory naming logic and examples
- Command-line override examples
- Implementation details
- Best practices (✅ Do / ❌ Don't)
- Troubleshooting guide
- Advanced usage patterns

**Benefits:**
- Complete reference for seed management
- Onboarding guide for new users
- Troubleshooting resource

---

## Directory Naming Examples

### Full Dataset (Production)
```yaml
master_seed: 42
max_files_to_process: null
```
→ `processed/flu_a/July_2025/`

### Deterministic Subset
```yaml
master_seed: 42
max_files_to_process: 500
```
→ `processed/flu_a/July_2025_seed_42_GTOs_500/`

### Random Subset
```yaml
master_seed: null
max_files_to_process: 100
```
→ `processed/flu_a/July_2025_random_20251013_143522_GTOs_100/`

### Manual Override
```yaml
master_seed: 42
max_files_to_process: 500
run_suffix: "_special_experiment"
```
→ `processed/flu_a/July_2025_special_experiment/`

---

## Pipeline Flow

```
1. Preprocessing
   - Reads: max_files_to_process, RANDOM_SEED from config
   - Generates: run_suffix (e.g., "_seed_42_GTOs_500")
   - Creates: processed/flu_a/July_2025_seed_42_GTOs_500/
   - Saves: run_suffix to config (for downstream use)

2. Embeddings
   - Reads: run_suffix from config (backward compatibility)
   - Reads from: processed/flu_a/July_2025_seed_42_GTOs_500/
   - Writes to: embeddings/flu_a/July_2025_seed_42_GTOs_500/

3. Training
   - Reads: run_suffix from config
   - Reads from: embeddings/flu_a/July_2025_seed_42_GTOs_500/
   - Writes to: models/flu_a/July_2025_seed_42_GTOs_500/
```

**Key Insight:** All stages automatically use the same run identifier!

---

## Benefits

### 1. Eliminates Data Duplication
- No need for `July_2025_subset_5k/` directories
- Single source of truth: `July_2025/` contains all raw data
- Saves storage space

### 2. Better Reproducibility
- Directory names document sampling strategy
- `seed_42_GTOs_500` explicitly shows how data was sampled
- Anyone can reproduce by looking at directory name

### 3. More Flexible Experimentation
- Change `max_files_to_process` without creating new directories
- Different seeds create different directories automatically
- Easy to run multiple experiments in parallel

### 4. Pipeline Consistency
- Same run identifier propagates through entire pipeline
- No manual path management needed
- Reduces configuration errors

### 5. Integrates with Existing System
- Works with hierarchical seed system
- Compatible with Hydra config overrides
- Backward compatible with manual `run_suffix`

---

## Migration Guide

### From Old System
```yaml
# OLD: Hardcoded subset preference
# Preprocessing looked for July_2025_subset_5k/ directory
```

### To New System
```yaml
# NEW: Dynamic path generation
master_seed: 42
max_files_to_process: 5000  # Sample size
# Creates: July_2025_seed_42_GTOs_5000/
```

### Steps:
1. ✅ No changes needed to existing experiments (backward compatible)
2. ✅ New experiments use auto-generated paths
3. ✅ Can delete subset directories (if not special/curated)
4. ✅ Use `max_files_to_process` for sampling instead

---

## Testing

### Test Cases

1. **Full dataset**
   ```bash
   python preprocess_flu_protein.py --virus_name flu_a
   # Expected: processed/flu_a/July_2025/
   ```

2. **Deterministic subset**
   ```bash
   python preprocess_flu_protein.py --virus_name flu_a \
       master_seed=42 max_files_to_process=500
   # Expected: processed/flu_a/July_2025_seed_42_GTOs_500/
   ```

3. **Random subset**
   ```bash
   python preprocess_flu_protein.py --virus_name flu_a \
       master_seed=null max_files_to_process=100
   # Expected: processed/flu_a/July_2025_random_<timestamp>_GTOs_100/
   ```

4. **Manual override**
   ```bash
   python preprocess_flu_protein.py --virus_name flu_a \
       run_suffix="_custom"
   # Expected: processed/flu_a/July_2025_custom/
   ```

5. **Pipeline propagation**
   ```bash
   # Preprocess
   python preprocess_flu_protein.py --virus_name flu_a \
       master_seed=42 max_files_to_process=500
   
   # Embeddings (should automatically use same directory)
   python compute_esm2_embeddings.py --virus_name flu_a
   # Should read from: processed/flu_a/July_2025_seed_42_GTOs_500/
   # Should write to: embeddings/flu_a/July_2025_seed_42_GTOs_500/
   ```

---

## Files Modified

1. ✅ `src/utils/path_utils.py` - **NEW** - Path generation utilities
2. ✅ `src/preprocess/preprocess_flu_protein.py` - Dynamic path generation
3. ✅ `conf/bundles/flu_a.yaml` - Updated defaults and documentation
4. ✅ `conf/bundles/bunya.yaml` - Updated defaults and documentation
5. ✅ `src/embeddings/compute_esm2_embeddings.py` - Backward compatibility comment
6. ✅ `docs/SEED_SYSTEM.md` - **NEW** - Comprehensive seed documentation
7. ✅ `docs/CHANGES_DYNAMIC_PATHS.md` - **NEW** - This summary document

---

## Next Steps

### Recommended Actions:

1. **Test the implementation:**
   ```bash
   # Test with small subset
   python preprocess_flu_protein.py --virus_name flu_a \
       master_seed=42 max_files_to_process=100
   ```

2. **Run full dataset:**
   ```bash
   # Production run
   python preprocess_flu_protein.py --virus_name flu_a \
       master_seed=42 max_files_to_process=null
   ```

3. **Verify embeddings pipeline:**
   ```bash
   # Should automatically use matching directory
   python compute_esm2_embeddings.py --virus_name flu_a --cuda_name cuda:6
   ```

4. **Delete old subset directories** (if not needed):
   ```bash
   # Only if these are just random subsets, not curated data
   rm -rf data/raw/Flu_A/July_2025_subset_5k/
   ```

5. **Update other preprocessing scripts** (e.g., Bunyavirales):
   - Apply same pattern to `preprocess_bunya_protein.py`
   - Use `path_utils` functions for consistency

---

## Questions?

See:
- `docs/SEED_SYSTEM.md` - Comprehensive seed system guide
- `src/utils/path_utils.py` - Implementation details
- `conf/bundles/flu_a.yaml` - Configuration examples

Or contact the development team!

