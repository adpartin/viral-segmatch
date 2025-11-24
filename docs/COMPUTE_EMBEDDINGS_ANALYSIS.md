# Comprehensive Analysis: `compute_esm2_embeddings.py`

## Overview
This script computes ESM-2 embeddings for protein sequences and stores them in a master cache system. It's the second stage in the pipeline (after preprocessing, before dataset creation and training).

## Pipeline Context

```
preprocess_flu_protein.py → compute_esm2_embeddings.py → dataset_segment_pairs.py → train_esm2_frozen_pair_classifier.py
     ↓                            ↓                              ↓                              ↓
protein_final.csv        master_esm2_embeddings.h5      train_pairs.csv              trained_model.pt
                         master_esm2_embeddings.parquet val_pairs.csv
                         sampled_isolates.txt            test_pairs.csv
                         failed_brc_fea_ids.csv
```

## Master Cache File Requirements

**⚠️ Important**: The master cache system consists of **two required files** that must be kept together:

1. **`master_esm2_embeddings.h5`** (HDF5 file)
   - Contains the actual embedding vectors in a single `emb` dataset
   - Uses row-based storage for efficient access
   - Stores embeddings in configurable precision (fp16/fp32)

2. **`master_esm2_embeddings.parquet`** (Parquet index file)
   - Maps `brc_fea_id` → row index in the HDF5 file
   - Required for efficient lookup during training
   - Created automatically by `compute_esm2_embeddings.py`

**Both files are created together** by `compute_esm2_embeddings.py` and **both are required** by:
- `train_esm2_frozen_pair_classifier.py` (for loading embeddings)
- `dataset_segment_pairs.py` (for validation)

**Location**: Both files are stored in the canonical embeddings directory:
```
data/embeddings/{virus}/{data_version}/master_esm2_embeddings.h5
data/embeddings/{virus}/{data_version}/master_esm2_embeddings.parquet
```

**Note**: The training script will raise a `ValueError` if the Parquet index file is missing, as it's essential for the `brc_fea_id` to row mapping.

---

## Line-by-Line Analysis

### **Lines 1-31: Imports and Setup**

```python
"""
Precompute ESM-2 embeddings for protein sequences.
Virus-agnostic version with command-line interface.
"""
```
**Purpose**: Docstring describing the script's role.

**Lines 5-13**: Standard imports
- `argparse`: CLI argument parsing
- `torch`: Preloaded early (heavy module, line 9 comment)
- `h5py`: HDF5 file I/O for master cache
- `numpy`, `pandas`: Data manipulation
- `psutil`: Memory monitoring

**Lines 15-18**: Project root setup
- Adds project root to `sys.path` to enable imports from `src/`
- Used by all pipeline scripts consistently

**Lines 20-26**: Utility imports
- `Timer`: Performance tracking
- `get_virus_config_hydra`: Config loading (Hydra-based)
- `resolve_process_seed`, `set_deterministic_seeds`: Reproducibility
- `resolve_run_suffix`, `build_embeddings_paths`: Path management
- `load_dataframe`: Unified CSV/Parquet loading
- `STANDARD_AMINO_ACIDS`: Validation constants
- `determine_device`: CUDA/CPU selection
- `compute_esm2_embeddings`: Core embedding computation function

**Lines 28-30**: Manual configs
- `ESM2_PROTEIN_SEQ_COL = 'esm2_ready_seq'`: Column name for cleaned sequences
- **Connection**: This column is created by `preprocess_flu_protein.py` via `prepare_sequences_for_esm2()`
- **TODO**: Consider moving to config.yaml (noted in comment)

**Line 32**: Global timer for total script execution time

**Lines 34-38**: `log_memory_usage()` helper
- **Purpose**: Monitor memory consumption at different stages
- **Usage**: Called before/after major operations (data loading, embedding computation)

---

### **Lines 40-67: CLI Argument Parsing**

**Purpose**: Define command-line interface for the script

**Line 42**: `--config_bundle` (required)
- **Example**: `flu_a`, `bunya`, `flu_a_3p_1ks`
- **Connection**: Used by all pipeline scripts for consistency
- **Downstream**: Same config bundle used in `dataset_segment_pairs.py` and `train_esm2_frozen_pair_classifier.py`

**Line 48**: `--cuda_name` (optional, default: `cuda:7`)
- **Purpose**: Specify GPU device for embedding computation
- **Connection**: Passed to `compute_esm2_embeddings()` → PyTorch model

**Lines 53-55**: `--input_file` (optional)
- **Purpose**: Override default input file path
- **Default**: Derived from `build_embeddings_paths()` → `data/processed/{virus}/{data_version}{run_suffix}/protein_final.csv`
- **Connection**: This is the output from `preprocess_flu_protein.py`

**Lines 58-60**: `--output_dir` (optional)
- **Purpose**: Override default output directory for metadata files
- **Default**: Derived from `build_embeddings_paths()` → `data/embeddings/{virus}/{data_version}{run_suffix}/`
- **Note**: Master cache is NOT saved here (it's in canonical location, see line 264)
- **Used for**: `sampled_isolates.txt` (run-specific metadata)

**Lines 63-65**: `--force-recompute` (flag)
- **Purpose**: Bypass cache and recompute all embeddings
- **Connection**: Passed to `compute_esm2_embeddings()` → skips cache checks

---

### **Lines 69-88: Config Loading and Extraction**

**Lines 70-75**: Load Hydra config
- **Purpose**: Centralized configuration management
- **Connection**: All scripts use same config system for consistency
- **Output**: `config` object with nested attributes (e.g., `config.virus.virus_name`)

**Lines 77-88**: Extract config values
- **VIRUS_NAME**: Used for path building (e.g., `flu_a`, `bunya`)
- **DATA_VERSION**: Data version tag (e.g., `July_2025`, `April_2025`)
- **RANDOM_SEED**: Process-specific seed for reproducibility
- **MAX_ISOLATES_TO_PROCESS**: Sampling parameter (can be `null` for full dataset)
- **MODEL_CKPT**: ESM-2 model checkpoint (e.g., `facebook/esm2_t33_650M_UR50D`)
- **ESM2_MAX_RESIDUES**: Max sequence length (default: 1022)
- **BATCH_SIZE**: Batch size for embedding computation
- **POOLING**: Pooling method (`mean`, `max`, `cls`, `attention`)
- **LAYER**: Layer selection (`last`, `second_last`, or int)
- **EMB_STORAGE_PRECISION**: Storage precision (`fp16`, `fp32`)

**Line 81**: `USE_SELECTED_ONLY` is commented out
- **Reason**: Embeddings are computed for ALL proteins (no filtering)
- **Connection**: Filtering happens in `dataset_segment_pairs.py` (line 527-541)

---

### **Lines 90-101: Run Suffix Resolution**

**Lines 95-101**: `resolve_run_suffix()`
- **Purpose**: Generate directory suffix based on sampling parameters
- **Logic**:
  - If `max_isolates_to_process: null` → empty suffix → canonical path
  - If `max_isolates_to_process: 1000` → suffix `_seed_42_isolates_1000` → unique path
- **Connection**: 
  - Used by `build_embeddings_paths()` for run-specific directories
  - Same logic in `dataset_segment_pairs.py` (line 449-454) and `train_esm2_frozen_pair_classifier.py` (line 523-528)
- **Critical**: Ensures all scripts use same directory structure

---

### **Lines 103-108: Seed Management**

**Lines 104-108**: Set deterministic seeds
- **Purpose**: Ensure reproducible embedding computation
- **Connection**: Same seed system used across all pipeline scripts
- **Note**: `cuda_deterministic=True` for GPU reproducibility (slower but deterministic)

---

### **Lines 110-133: Path Building**

**Lines 110-117**: Build canonical paths (no suffix)
- **Purpose**: Get master cache location (shared across all runs)
- **Result**: `data/embeddings/{virus}/{data_version}/` (no suffix)
- **Used for**: Master cache file, failed IDs file

**Lines 119-128**: Build run-specific paths (with suffix)
- **Purpose**: Get run-specific directory for metadata files
- **Result**: `data/embeddings/{virus}/{data_version}{run_suffix}/` (with suffix if sampling)
- **Used for**: `sampled_isolates.txt` (run-specific)

**Lines 130-133**: Apply CLI overrides
- **Purpose**: Allow manual path overrides via command line
- **Note**: `output_dir` is created here (for metadata files)

**Lines 135-142**: Print configuration summary
- **Purpose**: Log all key parameters for debugging/reproducibility

---

### **Lines 144-166: Data Loading and Initial Validation**

**Lines 145-153**: Load protein data
- **Input**: `protein_final.csv` from preprocessing stage
- **Function**: `load_dataframe()` handles CSV/Parquet automatically
- **Connection**: This file is created by `preprocess_flu_protein.py` (line 1144)

**Lines 155-159**: Validate required columns
- **Required**: `brc_fea_id`, `esm2_ready_seq`, `length`
- **Connection**: 
  - `brc_fea_id`: Used as key in parquet index (line 465 in `esm2_utils.py`)
  - `esm2_ready_seq`: Column created by preprocessing (cleaned sequences)
  - `length`: Used for validation (line 248)

**Lines 161-166**: Check for duplicate `brc_fea_id`
- **Purpose**: Prevent parquet index mapping issues
- **Connection**: Parquet index maps `brc_fea_id` → row index (line 153 in `train_esm2_frozen_pair_classifier.py`)
- **Note**: Duplicates would cause only last occurrence to be mapped

---

### **Lines 168-197: Isolate Sampling**

**Lines 170-197**: Conditional isolate sampling
- **Purpose**: Support subset processing for faster iteration
- **Logic**:
  - If `MAX_ISOLATES_TO_PROCESS` is set and < total isolates → sample
  - Otherwise → use all isolates
- **Sampling method**: `np.random.choice()` (deterministic if seed is set)

**Lines 185**: Filter DataFrame to sampled isolates
- **Purpose**: Only compute embeddings for sampled isolates
- **Note**: Master cache still stores all embeddings (deduplication by sequence)

**Lines 188-192**: Save `sampled_isolates.txt`
- **Purpose**: Track which isolates were used (for downstream scripts)
- **Location**: `output_dir` (run-specific directory)
- **Connection**: 
  - **Used by**: `dataset_segment_pairs.py` (line 508-519)
  - **Purpose**: Ensure dataset creation uses same isolates as embeddings
  - **Critical**: Without this, dataset script might use different isolates → missing embeddings

**Line 193-195**: Edge case handling
- If requested sampling > available isolates → use all (no sampling needed)

---

### **Lines 199-229: Protein Filtering (REMOVED)**

**Lines 199-223**: Commented out `USE_SELECTED_ONLY` filtering
- **Reason**: Embeddings should be computed for ALL proteins
- **Rationale**: 
  - Master cache is shared resource
  - Different dataset configs may need different protein subsets
  - Filtering happens downstream in `dataset_segment_pairs.py` (line 527-541)

**Lines 225-229**: Explanation comment
- **Key point**: Master cache contains ALL proteins
- **Connection**: `dataset_segment_pairs.py` filters by `selected_functions` when creating pairs

---

### **Lines 232-259: Data Validation and Deduplication**

**Lines 235-237**: Section header

**Lines 239-245**: Check for invalid sequences
- **Purpose**: Warn about non-standard amino acids
- **Note**: Sequences are NOT filtered (TODO comment suggests consideration)
- **Connection**: These sequences might fail during embedding computation

**Lines 247-251**: Identify truncated sequences
- **Purpose**: Warn about sequences longer than ESM-2 max length
- **Note**: Sequences are truncated during tokenization (not here)
- **Connection**: `max_length` parameter in `compute_esm2_embeddings()` handles truncation

**Lines 253-258**: Sequence deduplication
- **Purpose**: Remove duplicate sequences (keep unique `brc_fea_id` + sequence combinations)
- **Logic**: `drop_duplicates()` on `['brc_fea_id', ESM2_PROTEIN_SEQ_COL]`
- **Note**: This removes rows with identical `brc_fea_id` AND sequence
- **Important**: Different `brc_fea_id` with same sequence → both kept (biological duplicates)
- **Connection**: Master cache deduplicates by sequence hash (line 630 in `esm2_utils.py`)

**Line 259**: Memory logging after data loading

---

### **Lines 262-267: Master Cache Path Setup**

**Lines 262-267**: Master cache file path
- **Location**: `canonical_paths['output_dir'] / 'master_esm2_embeddings.h5'`
- **Result**: `data/embeddings/{virus}/{data_version}/master_esm2_embeddings.h5`
- **Purpose**: Shared cache across all runs (no suffix)
- **Connection**: 
  - **Used by**: `train_esm2_frozen_pair_classifier.py` (line 550, hardcoded but should match)
  - **Accessed via**: Parquet index for row-based lookup (line 150-153 in training script)

---

### **Lines 269-294: Embedding Computation**

**Lines 271-273**: Section header

**Lines 274-275**: Device setup
- **Purpose**: Determine CUDA/CPU device for model inference
- **Connection**: Passed to `compute_esm2_embeddings()` → PyTorch model

**Line 276**: Memory logging before computation

**Lines 278-291**: Call `compute_esm2_embeddings()`
- **Inputs**:
  - `sequences`: List of protein sequences (from `esm2_ready_seq` column)
  - `brc_fea_ids`: List of corresponding IDs
  - `embeddings_file`: Master cache path
  - `model_name`: ESM-2 checkpoint
  - `batch_size`: Processing batch size
  - `device`: CUDA/CPU
  - `max_length`: Max sequence length (+2 for CLS/SEP tokens)
  - `force_recompute`: Bypass cache flag
  - `use_parquet`: Enable parquet index creation
  - `pooling`, `layer`, `emb_storage_precision`: Embedding parameters

- **Returns**:
  - `embeddings`: Computed embeddings (numpy array) - **Note**: May be empty if all cached
  - `brc_fea_ids`: Successfully processed IDs
  - `failed_ids`: Failed IDs (e.g., OOM, invalid sequences)

- **Internal behavior** (`esm2_utils.py`):
  1. Generate cache keys: `seq_hash + model_sig` (line 630-631)
  2. Check cache: Skip if `cache_key` exists (line 635)
  3. Compute embeddings: Batch processing with ESM-2 model (line 699-742)
  4. Save to master cache: Batch save via `save_esm2_embeddings_batch()` (line 759)
  5. Update parquet index: Map `brc_fea_id` → row index (line 462-475)

**Lines 292-294**: Performance logging

---

### **Lines 296-317: Failed IDs Tracking**

**Lines 300-317**: Save failed IDs to master cache directory
- **Location**: `master_embeddings_file.parent / 'failed_brc_fea_ids.csv'`
- **Purpose**: Track sequences that failed to embed (shared across runs)
- **Logic**: Merge with existing failures (accumulate across runs)
- **Connection**: 
  - Failures are sequence-specific (consistent across runs)
  - Can be used to filter out problematic sequences in downstream scripts

---

### **Lines 320-341: Master Cache Validation**

**Lines 321-341**: Validate master cache contents
- **Purpose**: Verify cache was created/updated correctly
- **Checks**:
  - File exists
  - Contains `emb` dataset (embeddings array)
  - Contains `emb_keys` dataset (cache keys)
  - Metadata attributes present (model_name, pooling, layer, etc.)
- **Connection**: 
  - Same validation logic in `train_esm2_frozen_pair_classifier.py` (line 573-580)
  - Metadata validation ensures consistency across pipeline

---

## Key Connections to Other Scripts

### **Connection 1: Preprocessing → Embeddings**

**Input**: `protein_final.csv` from preprocessing
- **Location**: `data/processed/{virus}/{data_version}{run_suffix}/protein_final.csv`
- **Required columns**: `brc_fea_id`, `esm2_ready_seq`, `length`, `assembly_id`
- **Created by**: `preprocess_flu_protein.py` (line 1144)

**Output**: Master cache + metadata
- **Master cache**: `data/embeddings/{virus}/{data_version}/master_esm2_embeddings.h5`
- **Parquet index**: `data/embeddings/{virus}/{data_version}/master_esm2_embeddings.parquet`
- **Metadata**: `sampled_isolates.txt` (run-specific), `failed_brc_fea_ids.csv` (shared)

---

### **Connection 2: Embeddings → Dataset Creation**

**Used by**: `dataset_segment_pairs.py`

**Line 508-519**: Load `sampled_isolates.txt`
- **Purpose**: Ensure dataset uses same isolates as embeddings
- **Critical**: Without this, dataset might reference embeddings that don't exist
- **Location**: `embeddings_paths['output_dir'] / 'sampled_isolates.txt'`

**Line 527-541**: Filter by `selected_functions`
- **Purpose**: Create dataset with specific protein types
- **Note**: Embeddings were computed for ALL proteins, filtering happens here
- **Connection**: This is why embeddings script doesn't filter (line 225-229)

**Output**: Pair datasets (`train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv`)
- **Contains**: `brc_a`, `brc_b`, `label` columns
- **Connection**: These `brc_fea_id` values must exist in master cache

---

### **Connection 3: Embeddings → Training**

**Used by**: `train_esm2_frozen_pair_classifier.py`

**Line 550**: Master cache path (hardcoded)
- **Location**: `data/embeddings/{virus}/{data_version}/master_esm2_embeddings.h5`
- **Note**: Should match canonical path from embeddings script (line 264)

**Line 584**: Load parquet index
- **Purpose**: Build `brc_fea_id` → row index mapping
- **Connection**: Created by `save_esm2_embeddings_batch()` (line 462-475 in `esm2_utils.py`)

**Line 591-600**: Validate embedding availability
- **Purpose**: Ensure all required embeddings exist before training
- **Checks**: All `brc_a` and `brc_b` in pair datasets must have embeddings

**Line 603-608**: Create `SegmentPairDataset` instances
- **Purpose**: Load embeddings on-demand during training
- **Mechanism**: 
  - Load parquet index → build `id_to_row` mapping (line 120)
  - Open HDF5 file once (line 123)
  - Access embeddings via row indices: `h5['emb'][row_a, row_b]` (line 206)
- **Connection**: Row-based access is efficient (no string key lookups)

---

## Data Flow Summary

```
1. PREPROCESSING
   └─> protein_final.csv (all proteins, all isolates)

2. EMBEDDINGS COMPUTATION (this script)
   ├─> Load protein_final.csv
   ├─> Sample isolates (if MAX_ISOLATES_TO_PROCESS set)
   ├─> Compute embeddings for ALL proteins (no filtering)
   ├─> Save to master cache (deduplicated by sequence)
   ├─> Create parquet index (brc_fea_id → row mapping)
   └─> Output:
       ├─> master_esm2_embeddings.h5 (shared)
       ├─> master_esm2_embeddings.parquet (shared)
       ├─> sampled_isolates.txt (run-specific)
       └─> failed_brc_fea_ids.csv (shared)

3. DATASET CREATION
   ├─> Load protein_final.csv
   ├─> Load sampled_isolates.txt (filter to same isolates)
   ├─> Filter by selected_functions (if USE_SELECTED_ONLY)
   ├─> Create train/val/test pairs
   └─> Output: train_pairs.csv, val_pairs.csv, test_pairs.csv

4. TRAINING
   ├─> Load pair datasets
   ├─> Load master cache + parquet index
   ├─> Validate all required embeddings exist
   ├─> Create SegmentPairDataset (row-based access)
   └─> Train model
```

---

## Critical Design Decisions

### **1. Master Cache is Shared (No Suffix)**
- **Rationale**: Embeddings are deduplicated by sequence hash, so same sequence = same embedding
- **Benefit**: Different sampling runs share embeddings, avoiding recomputation
- **Location**: `data/embeddings/{virus}/{data_version}/master_esm2_embeddings.h5`

### **2. Metadata Files are Run-Specific (With Suffix)**
- **Rationale**: Sampling creates different isolate sets per run
- **Benefit**: Each run tracks its own sampled isolates
- **Location**: `data/embeddings/{virus}/{data_version}{run_suffix}/sampled_isolates.txt`

### **3. Embeddings Computed for ALL Proteins**
- **Rationale**: Master cache is shared resource, filtering happens downstream
- **Benefit**: Different dataset configs can filter as needed without recomputing
- **Connection**: Filtering in `dataset_segment_pairs.py` (line 527-541)

### **4. Sequence-Based Deduplication**
- **Rationale**: Same sequence = same embedding (biological duplicates)
- **Benefit**: Efficient storage, multiple `brc_fea_id` map to same row
- **Mechanism**: Cache key = `seq_hash + model_sig` (line 630-631 in `esm2_utils.py`)

### **5. Row-Based Access via Parquet Index**
- **Rationale**: Faster than string key lookups in HDF5
- **Benefit**: Efficient loading during training
- **Mechanism**: Parquet index maps `brc_fea_id` → row index, then `h5['emb'][row]`

---

## Potential Issues and TODOs

### **Line 233**: TODO comment about validation
- **Question**: Is sequence deduplication still required?
- **Answer**: Yes, but it's handled by master cache (sequence-based deduplication)

### **Line 242**: TODO about invalid sequences
- **Question**: Should invalid sequences be filtered?
- **Current**: Only warned, not filtered
- **Consideration**: They might fail during embedding computation anyway

### **Line 550 in training script**: Hardcoded master cache path
- **Issue**: Should use `canonical_paths` like embeddings script
- **Risk**: Path mismatch if config changes

### **Line 473-482 in dataset script**: Why build embeddings paths?
- **Purpose**: To find `sampled_isolates.txt` in run-specific directory
- **Note**: Comment says "TODO: why we need this?" - it's needed for isolate filtering

---

## Summary

This script is the **embedding computation stage** of the pipeline:
1. **Loads** preprocessed protein data
2. **Samples** isolates (optional, run-specific)
3. **Computes** embeddings for ALL proteins (no filtering)
4. **Saves** to master cache (shared, deduplicated by sequence)
5. **Creates** parquet index for efficient lookup
6. **Tracks** failures and sampled isolates

**Key outputs**:
- Master cache: Shared across all runs
- Parquet index: Fast `brc_fea_id` → row lookup
- Sampled isolates: Run-specific tracking
- Failed IDs: Shared failure tracking

**Downstream usage**:
- Dataset script: Uses sampled isolates + filters by function
- Training script: Loads embeddings via parquet index + row-based access

