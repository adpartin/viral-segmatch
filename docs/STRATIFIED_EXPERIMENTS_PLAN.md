# Stratified Experiments Implementation Plan

**Purpose**: Design and implement validation experiments to test for confounders (host, date, subtype) in model performance.

**Note on Terminology**: 
- **Confounders**: We use this term concisely to refer to variables (host, date, subtype) that could create spurious correlations. However, these may actually be:
  - **Biological confounders**: True biological associations (e.g., host-specific viral adaptations, subtype-specific characteristics)
  - **Batch effects**: Technical artifacts from data collection/processing (e.g., different sequencing machines, protocols, or personnel in 2020 vs 2000). Temporal patterns could reflect batch effects rather than biological evolution.
- Our experiments test for both: if performance drops with stratified splits, it could indicate either biological confounders or batch effects (or both). The distinction is less important than detecting that the model relies on these variables rather than true isolate signal.

**Status**: Planning phase

---

## Available Data Columns

### From `protein_final.csv` (Main protein data)
- **`assembly_id`**: Isolate identifier (e.g., "1316165.15", "000142e874")
- **`function`**: Protein function (e.g., "pb1", "pb2", "ha", "na")
- **`prot_seq`**: Protein amino acid sequence
- **`file`**: Source GTO file name
- **`brc_fea_id`**: BV-BRC feature ID
- **`genbank_ctg_id`**: GenBank contig ID (segment identifier)
- **`replicon_type`**: Segment type (S1, S2, etc.)

### From `Flu_Genomes.key` (Genome index)
- **`hash_id`**: Internal genome ID (primary key)
- **`virus_name`**: Full virus name
- **`hn_subtype`**: H/N subtype (e.g., "H1N1", "H3N2")
- **`seg_1` ... `seg_8`**: BV-BRC segment IDs (8 columns, typically 2 IDs per segment)
- **`host_inferred`**: Host inferred from virus name
- **`location`**: Location inferred from virus name
- **`year`**: Year inferred from virus name
- **`strain`**: Strain identifier

### From `Flu.first-seg.meta.tab` (Metadata)
- **`hash_id`**: Same as in Flu_Genomes.key (join key)
- **`virus_name`**: Full virus name
- **`hn_subtype`**: H/N subtype
- **`first_seg_id`**: BV-BRC ID of first segment
- **`host_common_name`**: Explicit host (e.g., "Human", "Pig", "Chicken")
- **`lab_host`**: Laboratory host/cell line
- **`passage`**: Passage history

### From Combined Metadata (via `flu_genomes_eda.py`)
- **`hash_id`**: Primary key
- **`host`**: Prefers `host_common_name`, falls back to `host_inferred`
- **`year`**: Numeric year (from virus name parsing)
- **`hn_subtype`**: H/N subtype
- **`location`**: Geographic location
- **`lab_host`**: Laboratory host (sparse)
- **`passage`**: Passage history (sparse)

---

## Dataset Size: 108,530 vs 111,797

**Key Finding**: The discrepancy is expected and normal.

- **111,797**: Total GTO files in raw data / isolates in metadata
- **108,530**: Unique `assembly_id` values in `protein_final.csv` (isolates that made it through preprocessing)
- **3,267 isolates (2.9%) filtered** during preprocessing
- **Reasons**: Quality filters, missing proteins, failed ESM-2 preparation
- **Conclusion**: Stratified experiments will work with the 108,530 isolates that made it through preprocessing.

## Mapping Strategy: `assembly_id` → `hash_id`

**Key Finding**: `assembly_id` directly matches `hash_id`!

- `assembly_id` is extracted from GTO file name (e.g., `0000003012.gto` → `assembly_id = "0000003012"`)
- GTO file name format is `{hash_id}.gto`
- Therefore: `assembly_id == hash_id`

**Verification Results** (from `src/analysis/verify_metadata_mapping.py`):
- ✅ **100% coverage**: All 108,530 `assembly_id` values can be mapped to `hash_id`
- ✅ **One-to-one mapping**: No conflicts (each `hash_id` maps to unique `assembly_id`)

**Implementation**:
```python
# Simple direct mapping
protein_df['hash_id'] = protein_df['assembly_id']

# Join with metadata
metadata = load_flu_metadata()  # hash_id, host, year, hn_subtype
protein_df = protein_df.merge(
    metadata[['hash_id', 'host', 'year', 'hn_subtype']],
    on='hash_id',
    how='left'
)
```

---

## Experiment Types

### Experiment Type 1: Within-Stratum Experiments
**Objective**: Test model performance when train/val/test splits are created from isolates within a single stratum (specific {year, host, subtype} combination).

**Method**:
1. Filter isolates to a specific {year, host, subtype} combination (e.g., `{year=2020, host=Human, subtype=H3N2}`)
2. Split filtered isolates into train/val/test (maintaining isolate-level grouping)
3. Train and evaluate model
4. Repeat for different {year, host, subtype} combinations

**Purpose**: Assess if model can learn from a homogeneous stratum and generalize within that stratum.

### Experiment Type 2: Cross-Confounder Experiments
**Objective**: Test model generalization across confounders by training on one stratum and testing on a different one.

**Method**:
1. Filter isolates for training stratum (e.g., `{year=2020, host=Human, subtype=H3N2}`)
2. Filter isolates for test stratum (e.g., `{year=2021, host=Human, subtype=H3N2}`)
3. Split training isolates into train/val (test uses separate stratum)
4. Train on training stratum, evaluate on test stratum
5. Repeat for different train/test combinations

**Variations**:
- **Temporal**: Train on year X, test on year Y (same host, subtype)
- **Host shift**: Train on host X, test on host Y (same year, subtype)
- **Subtype shift**: Train on subtype X, test on subtype Y (same year, host)

**Purpose**: Assess if model relies on confounders (host/year/subtype) rather than true isolate signal.

---

## Implementation Approach: Incremental, Modular

**Recommended Order**:
1. **Phase 1**: Metadata enrichment utility (foundation, low risk)
2. **Phase 2**: Filtering and splitting functions (core functionality)
3. **Phase 3**: Experiment runner with config support (orchestration)

**Key Design Decisions**:
- Create `split_dataset_stratified_v2()` as wrapper (preserves backward compatibility)
- Add filtering function to select isolates by {year, host, subtype}
- Support both within-stratum (filter then split) and cross-confounder (separate train/test filters)
- Enrich metadata in main script, not in splitting function
- Use `StratifiedGroupKFold` from sklearn for within-stratum splits (maintains isolate-level grouping)
- Handle missing metadata: exclude isolates with missing metadata initially (can relax later)

#### Step 1: Metadata Enrichment Utility
**File**: `src/utils/metadata_enrichment.py`

```python
def enrich_protein_data_with_metadata(protein_df, metadata_files):
    """Enrich protein_df with metadata (host, year, hn_subtype)."""
    # Simple: assembly_id == hash_id
    protein_df['hash_id'] = protein_df['assembly_id']
    # Merge with metadata, handle missing gracefully
    return enriched_df
```

#### Step 2: Filtering and Splitting Functions
**File**: `src/datasets/dataset_segment_pairs.py`

**Filtering function**:
```python
def filter_isolates_by_metadata(
    df: pd.DataFrame,
    host: Optional[str] = None,
    year: Optional[int] = None,
    hn_subtype: Optional[str] = None
) -> pd.DataFrame:
    """Filter isolates by metadata values. None = no filter."""
    # Filter logic
```

**Splitting function**:
Create `split_dataset_stratified_v2()` as wrapper around existing `split_dataset_v2()`:
- Add parameter: `stratify_by: Optional[List[str]] = None`
- If `None`: use existing logic (backward compatible)
- If provided: use `StratifiedGroupKFold` for isolate splitting

```python
from sklearn.model_selection import StratifiedGroupKFold

def split_dataset_stratified_v2(..., stratify_by: Optional[List[str]] = None):
    if stratify_by is None:
        return split_dataset_v2(...)  # Existing logic
    
    # Create stratum label (combination of stratify_by columns)
    isolate_metadata['stratum'] = isolate_metadata[stratify_by].apply(
        lambda x: '_'.join(x.astype(str)), axis=1
    )
    # Use StratifiedGroupKFold for 3-way split
    # ...
```

#### Step 3: Experiment Runner
**File**: `src/experiments/run_stratified_experiment.py`

Orchestration: load config → enrich metadata → filter isolates → split → train → save results.
Supports both experiment types via config.

---

## Configuration Structure

### Dataset Config Extension
Add to `conf/dataset/default.yaml` or create `conf/dataset/stratified.yaml`:

```yaml
# Stratified experiment configuration
stratified:
  enabled: false  # Set to true to enable stratified experiments
  
  # Experiment type: "within_stratum" or "cross_confounder"
  experiment_type: "within_stratum"
  
  # For within-stratum experiments: filter to single stratum, then split
  within_stratum:
    filter:
      host: "Human"      # null = no filter
      year: 2020         # null = no filter
      hn_subtype: "H3N2" # null = no filter
    stratify_by: null    # null = random split, or ["host", "year", "hn_subtype"]
  
  # For cross-confounder experiments: separate train and test filters
  cross_confounder:
    train_filter:
      host: "Human"
      year: 2020
      hn_subtype: "H3N2"
    test_filter:
      host: "Human"
      year: 2021
      hn_subtype: "H3N2"
    # val_filter: null  # null = split from train_filter isolates
```

### Example Config Bundles

**Within-stratum experiment** (`conf/bundles/flu_stratified_human_h3n2_2020.yaml`):
```yaml
defaults:
  - /virus: flu
  - /paths: flu
  - /embeddings: default
  - /dataset: stratified  # Use stratified dataset config
  - /training: base
  - _self_

dataset:
  stratified:
    enabled: true
    experiment_type: "within_stratum"
    within_stratum:
      filter:
        host: "Human"
        year: 2020
        hn_subtype: "H3N2"
      stratify_by: null  # Random split within this stratum
```

**Cross-confounder experiment** (`conf/bundles/flu_cross_year_2020_to_2021.yaml`):
```yaml
defaults:
  - /virus: flu
  - /paths: flu
  - /embeddings: default
  - /dataset: stratified
  - /training: base
  - _self_

dataset:
  stratified:
    enabled: true
    experiment_type: "cross_confounder"
    cross_confounder:
      train_filter:
        host: "Human"
        year: 2020
        hn_subtype: "H3N2"
      test_filter:
        host: "Human"
        year: 2021
        hn_subtype: "H3N2"
```

**Rationale for config structure**:
- Uses existing Hydra bundle system (consistent with current codebase)
- Clear separation between experiment types
- Easy to create new experiments by copying and modifying bundle files
- Supports both single experiments and batch processing (can loop over configs)

---

## Code Design Support for Proposed Experiments

**Question**: Does the proposed code design support the two experiment types?

**Answer**: Yes, with the following additions:

### Support for Experiment Type 1 (Within-Stratum)
✅ **Supported** by:
- `filter_isolates_by_metadata()` function (filters to specific {year, host, subtype})
- `split_dataset_stratified_v2()` function (splits filtered isolates into train/val/test)
- Config structure: `dataset.stratified.within_stratum.filter` defines the stratum

**Flow**: Filter → Split → Train → Evaluate

### Support for Experiment Type 2 (Cross-Confounder)
✅ **Supported** by:
- `filter_isolates_by_metadata()` function (applied separately to train and test)
- Config structure: `dataset.stratified.cross_confounder.train_filter` and `test_filter`
- Experiment runner handles separate train/test filtering

**Flow**: Filter train isolates → Split train into train/val → Filter test isolates → Train → Evaluate on test

### Configuration Location
✅ **Config structure defined in**: `conf/dataset/stratified.yaml` (or extension of `default.yaml`)
✅ **Experiment-specific configs in**: `conf/bundles/flu_*.yaml` (following existing pattern)

**Rationale**: 
- Uses existing Hydra bundle system (consistent with current codebase)
- Experiment parameters (filters) are defined in config, not hardcoded
- Easy to create new experiments by copying/modifying bundle files
- Supports batch processing (can loop over multiple config files)

---

## Critical Concerns

1. **Data Quality**: Missing metadata (host, year, subtype) for some isolates
   - **Action**: Quantify completeness first. If >95% complete, exclude missing (conservative). If <95%, consider "Unknown" category.

2. **Stratum Sizes**: Rare host+year+subtype combinations may be too small for valid splits
   - **Action**: Set minimum stratum size (e.g., 10 isolates), report excluded strata, consider collapsing rare categories.

3. **Backward Compatibility**: Don't break existing experiments
   - **Action**: Keep `split_dataset_v2()` unchanged, new function is opt-in (only if `stratify_by` provided).

4. **Performance Baseline**: Need to compare stratified vs. non-stratified performance
   - **Action**: Run baseline first (current non-stratified split), then stratified with same config, compare F1/AUC/accuracy.

---

## Implementation Steps

1. ✅ **Verify mapping** - COMPLETED
   - Verified: `assembly_id` directly equals `hash_id` (100% coverage)

2. **Create metadata enrichment utility** (`src/utils/metadata_enrichment.py`)
   - Simple: `protein_df['hash_id'] = protein_df['assembly_id']` then merge
   - Handle missing metadata gracefully (report, don't fail)

3. **Extend dataset creation** (`src/datasets/dataset_segment_pairs.py`)
   - Add `filter_isolates_by_metadata()` function (host, year, hn_subtype filters)
   - Create `split_dataset_stratified_v2()` as wrapper
   - Add `stratify_by` parameter (backward compatible)
   - Support both within-stratum and cross-confounder modes

4. **Create dataset config** (`conf/dataset/stratified.yaml`)
   - Define configuration structure for both experiment types
   - Integrate with existing Hydra bundle system

5. **Create experiment runner** (`src/experiments/run_stratified_experiment.py`)
   - Load config → enrich metadata → filter → split → train → save
   - Support both experiment types via config
   - Can extend to batch experiments later

6. **Create example config bundles**
   - `conf/bundles/flu_stratified_human_h3n2_2020.yaml` (within-stratum example)
   - `conf/bundles/flu_cross_year_2020_to_2021.yaml` (cross-confounder example)

---

## Next Steps

1. ✅ Verify `assembly_id` → `hash_id` mapping - COMPLETED
2. ⏳ Implement metadata enrichment function
3. ⏳ Implement filtering and stratified splitting functions
4. ⏳ Create dataset config structure (`conf/dataset/stratified.yaml`)
5. ⏳ Create experiment runner
6. ⏳ Create example config bundles
7. ⏳ Run baseline (non-stratified) for comparison
8. ⏳ Run first within-stratum experiment (e.g., Human-H3N2-2020)
9. ⏳ Run first cross-confounder experiment (e.g., train on 2020, test on 2021)
10. ⏳ Compare performance and iterate

