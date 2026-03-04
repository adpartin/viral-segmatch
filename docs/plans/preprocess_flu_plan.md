# Plan: `preprocess_flu.py` — Unified Flu A Preprocessing

**Status: IMPLEMENTED**

## Context

Currently `preprocess_flu_protein.py` handles protein-only extraction from GTO files. We want `preprocess_flu.py` that parses each GTO file **once** and extracts both protein and genome data, producing `protein_final.csv` (unchanged) and `genome_final.csv` (new). The design is documented in `docs/genome_pipeline_design.md`.

`preprocess_flu_protein.py` stays as-is for backward compatibility. The new script replaces it for future runs.

---

## Implementation

### 1. Add `replicon_to_segment` to `conf/virus/flu.yaml`

Simple mapping for genome segment assignment (proteins use the more complex `conditional_segment_mappings`):

```yaml
replicon_to_segment:
  'Segment 1': 'S1'
  ...
  'Segment 8': 'S8'
```

### 2. Create `src/preprocess/preprocess_flu.py`

**Structure:**

```
Imports (same as preprocess_flu_protein.py + dna_utils)
Constants (SEQ_COL_NAME, DNA_SEQ_COL_NAME)

# --- Combined GTO extraction ---
get_data_from_gto(path) -> (protein_df, genome_df)
    - Opens GTO JSON once
    - Shared metadata extraction (assembly_id, quality, taxonomy)
    - Builds segment_map from contigs (shared)
    - Extracts protein data from features (same logic as current)
    - Extracts genome data from contigs
    - Returns both DataFrames

aggregate_data_from_gto_files(gto_dir, max_files, random_seed)
    -> (protein_df, genome_df)

# --- Protein pipeline functions (copied from preprocess_flu_protein.py) ---
validate_protein_counts()
analyze_protein_counts_per_file()
analyze_intra_file_function_duplicates()
assign_segment_using_core_proteins()
assign_segment_using_aux_proteins()
assign_segments()
apply_basic_filters()
handle_duplicates()
analyze_sequence_duplicates_for_pair_classification()

# --- Genome pipeline functions (new, simple) ---
assign_genome_segments(genome_df, replicon_to_segment) -> genome_df
    - Maps replicon_type -> canonical_segment directly

apply_genome_basic_filters(genome_df, output_dir) -> genome_df
    - Drop unassigned canonical_segment
    - Drop Poor quality
    - Drop unassigned replicon_type
    - Drop missing dna_seq

handle_genome_duplicates(genome_df, output_dir) -> genome_df
    - Dedup on (dna_seq, assembly_id), keep first
    - Log and save any duplicates found

# --- Module-level pipeline code ---
1. argparse (--config_bundle, --force-reprocess)
2. Config loading
3. Path resolution
4. GTO aggregation with caching:
   - protein_agg_from_GTOs.parquet (same name, backward compat)
   - genome_agg_from_GTOs.parquet (new)
   - Both must exist to use cache; else reprocess
5. PROTEIN PIPELINE (identical to preprocess_flu_protein.py):
   - EDA, assign_segments, apply_basic_filters, handle_duplicates
   - analyze_protein_ambiguities, prepare_sequences_for_esm2
   - Save protein_final.csv + .parquet
6. GENOME PIPELINE:
   - assign_genome_segments
   - apply_genome_basic_filters
   - DNA QC via summarize_dna_qc(genome_df, seq_col='dna_seq')
   - handle_genome_duplicates
   - Save genome_final.csv + .parquet
7. Save experiment metadata (protein + genome stats)
```

### 3. Key decisions

- **genome_final.csv schema**: `assembly_id, file, quality, genbank_ctg_id, replicon_type, canonical_segment, dna_seq, length, gc_content, ambig_count, ambig_frac`
- **Column name**: `dna_seq` (not `dna` as in bunya script -- matches design doc)
- **Protein functions use module-level globals** (config, core_functions, output_dir, etc.) -- copied verbatim from preprocess_flu_protein.py to preserve identical behavior
- **No shell wrapper** -- per existing convention ("preprocessing is run directly")
- **No --skip-protein/--skip-genome flags** -- keep it simple, always runs both

### 4. Files created/modified

| File | Action |
|------|--------|
| `src/preprocess/preprocess_flu.py` | **Created** -- unified script |
| `conf/virus/flu.yaml` | **Edited** -- added `replicon_to_segment` mapping |

### 5. Verification

1. Run `preprocess_flu.py --config_bundle flu` on existing data
2. Compare `protein_final.csv` output against existing one from `preprocess_flu_protein.py` -- must be identical
3. Verify `genome_final.csv`: 8 segments per complete assembly, canonical_segment in {S1..S8}, no missing dna_seq, gc_content in [0,1]
4. Verify join key `(assembly_id, genbank_ctg_id)` matches between protein and genome outputs
