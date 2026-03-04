# Genome Pipeline Design

Design decisions for adding genome (DNA/RNA) support alongside the existing protein pipeline.
Agreed during feature/genome-preprocessing implementation (March 2026).

---

## Motivation

The current pipeline is protein-only: GTO files -> protein extraction -> ESM-2 embeddings ->
segment pair classification (MLP). We want to add genome-based features for comparison and
eventual combination with protein embeddings.

Planned models (in order of implementation):
1. **ESM-2 (frozen) + MLP** — current baseline (protein-only, already implemented)
2. **K-mers + MLP** — genome-only, k-mer count vectors as input features
3. **K-mers + XGBoost/LightGBM** — genome-only, tree-based classifier
4. **GenSLM embeddings + MLP** — genome-only, language model embeddings (future)
5. **ESM-2 + k-mers combined** — multi-modal, protein + genome features in one model (future)

---

## Key Structural Facts (from GTO files)

- **1 GTO file = 1 isolate (assembly_id)**
- **contigs[] = DNA sequences**: one per genomic segment. Flu A has 8 segments.
  Each contig has: `id` (GenBank accession), `replicon_type` ("Segment 1"–"Segment 8"), `dna`.
- **features[] = proteins**: multiple per segment (e.g., Segment 7 -> M1 + M2).
  Each feature links to a contig via `location[0][0]` = `genbank_ctg_id`.

**Cardinality**: 1 DNA per segment per isolate, but multiple proteins per segment.
Join key between protein and genome data: `(assembly_id, genbank_ctg_id)`.

---

## Decision: Unified Preprocessing Script

### Problem

`preprocess_flu_protein.py` and `preprocess_bunya_dna.py` (now `preprocess_bunya_genome.py`)
each iterate over GTO files independently, duplicating metadata extraction logic
(assembly_id, quality, taxonomy, segment mapping).

### Solution

Create **`preprocess_flu.py`** — a single script that:
1. Parses each GTO file once (shared metadata extraction)
2. Extracts proteins -> runs protein-specific QC -> saves `protein_final.csv`
3. Extracts genomes -> runs DNA-specific QC -> saves `genome_final.csv`

This replaces `preprocess_flu_protein.py` as Stage 1.

### Why separate output files (not a single merged DataFrame)

- **Cardinality mismatch**: protein_final.csv has multiple rows per segment per isolate
  (e.g., M1 and M2 on Segment 7). genome_final.csv has exactly one row per segment.
  Merging would duplicate DNA sequences across protein rows, bloating the file.
- **Downstream independence**: protein data feeds ESM-2 embeddings (Stage 2a).
  Genome data feeds k-mer features (Stage 2b). Different consumers, different schemas.
- **Join when needed**: at Stage 3 (dataset_segment_pairs.py), both can be loaded
  and joined on `(assembly_id, genbank_ctg_id)` if a combined model is needed.

### Output schemas

**protein_final.csv** (unchanged from current):
```
assembly_id, file, quality, function, prot_seq, esm2_ready_seq,
genbank_ctg_id, replicon_type, canonical_segment, length,
brc_fea_id, [ambiguity columns], [metadata columns]
```

**genome_final.csv** (new):
```
assembly_id, file, quality, genbank_ctg_id, replicon_type,
canonical_segment, dna_seq, length, gc_content, ambig_count, ambig_frac
```

---

## Decision: K-mers as a Separate Stage 2b

### Problem

K-mer computation is a featurization step, not a preprocessing step.
The choice of `k` (e.g., k=3, k=4, k=6) and representation (counts, TF-IDF, normalized)
are hyperparameters that may vary across experiments.

### Solution

Create **`compute_kmer_features.py`** as Stage 2b, parallel to ESM-2 embeddings (Stage 2a):

```
Stage 1:  preprocess_flu.py             -> protein_final.csv + genome_final.csv
Stage 2a: compute_esm2_embeddings.py    -> master_esm2_embeddings.h5  (protein)
Stage 2b: compute_kmer_features.py      -> kmer_features.h5 or .npz   (genome)
Stage 3:  dataset_segment_pairs.py      -> train/val/test_pairs.csv
Stage 4:  train_*.py                    -> model + predictions
```

Stage 2b would:
- Read `genome_final.csv`
- Compute k-mer count vectors for each DNA sequence
- Save as HDF5 or sparse matrix (keyed by assembly_id + segment)
- Config controls: `k` value, normalization, vocabulary

### Why not compute k-mers inside dataset_segment_pairs.py?

- K-mer computation is potentially expensive for large datasets (100K+ isolates)
- Caching avoids recomputation across experiments (same genome, different splits/models)
- Matches the ESM-2 pattern: compute once, load many times

---

## Decision: dataset_segment_pairs.py Changes (Future)

When genome features are ready, Stage 3 will need to support loading both feature sources.
Controlled by config:

```yaml
dataset:
  feature_sources:
    - esm2           # load from master_esm2_embeddings.h5
    - kmer           # load from kmer_features.h5
```

For protein-only experiments: `feature_sources: [esm2]` (current behavior, default).
For genome-only experiments: `feature_sources: [kmer]`.
For combined: `feature_sources: [esm2, kmer]`.

The pair generation logic (positive/negative sampling, isolate splitting, co-occurrence
blocking) is feature-agnostic — it operates on assembly_id and function metadata.
Only the feature loading and concatenation changes.

---

## QC: General vs Sequence-Specific

| QC step | Applies to | Module |
|---------|-----------|--------|
| Assembly deduplication | Both | gto_utils.py |
| Genome quality filter | Both | preprocess_flu.py |
| Single-file enforcement | Both | gto_utils.py |
| Replicon type validation | Both | preprocess_flu.py |
| Amino acid ambiguity (B/Z/J/X/U/O) | Protein only | protein_utils.py |
| Internal/terminal stop codons | Protein only | protein_utils.py |
| ESM-2 sequence preparation | Protein only | protein_utils.py |
| IUPAC ambiguity codes (N/R/Y/W/S/K) | DNA only | dna_utils.py |
| GC content analysis | DNA only | dna_utils.py |
| Sequence length validation | Both (different thresholds) | respective utils |

---

## Rename Log

- `preprocess_bunya_dna.py` -> `preprocess_bunya_genome.py` (March 2026)
  Reason: "genome" is more accurate than "dna" (covers both DNA and RNA viruses).
  File remains NOT actively maintained; kept as reference for Bunya genome extraction.

---

## Implementation Order

1. Rename `preprocess_bunya_dna.py` -> `preprocess_bunya_genome.py` (done)
2. Create `preprocess_flu.py` (unified protein + genome extraction)
3. Create `compute_kmer_features.py` (Stage 2b)
4. Update `dataset_segment_pairs.py` to support `feature_sources` config
5. Create k-mer + MLP training script (or extend existing trainer)
6. Create k-mer + XGBoost/LightGBM training script
