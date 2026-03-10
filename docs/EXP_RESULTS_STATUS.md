# Experimental Results & Research Status

**Purpose**: This document tracks project status, provides research context (biological and technical background), outlines the research roadmap, and summarizes experimental results. For detailed analysis of experiment results see [`EXPERIMENT_RESULTS_ANALYSIS.md`](EXPERIMENT_RESULTS_ANALYSIS.md).

**Last updated**: March 2026
**Status**: Gen3 interaction/normalization ablations complete; k-mer feature source validated
**Primary Findings**: Conservation hypothesis confirmed (Gen1); ESM-2 concat collapse is embedding-geometry-specific (Gen3); k-mer features dominate on homogeneous data

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Status & TODO](#2-current-status--todo)
3. [Experimental Results](#3-experimental-results)
4. [Biological Background](#4-biological-background)
5. [Technical Background](#5-technical-background)
6. [Validation Experiments: Testing for Batch Effects and Confounders](#6-validation-experiments-testing-for-batch-effects-and-confounders)
7. [Research Roadmap](#7-research-roadmap)
8. [Next Steps](#8-next-steps)

---

## 1. Executive Summary

### Gen1: Conservation Hypothesis Confirmed (Dec 2025) ✅

Performance directly correlates with protein conservation levels:
- **Variable segments (HA-NA)**: accuracy=0.923, **F1=0.916**, AUC=0.953
- **Mixed segments (PB2-HA-NA)**: accuracy=0.854, **F1=0.855**, AUC=0.920
- **Conserved segments (PB2-PB1-PA)**: accuracy=0.719, **F1=0.753**, AUC=0.750

### Gen3: Interaction, Normalization, and Feature Source (Mar 2026)

Key advances since Gen1:
1. **`unit_diff` + `slot_norm`** is the best ESM-2 configuration. `concat` fails on homogeneous data (H3N2-only: AUC 0.498); `unit_diff` succeeds (AUC 0.957).
2. **K-mer (k=6) features outperform ESM-2** on both mixed-subtype (AUC 0.982 vs 0.966) and H3N2-only (AUC 0.988 vs 0.957) data.
3. **K-mer `concat` does NOT collapse on H3N2** (AUC 0.985). The ESM-2 concat failure is specific to its embedding geometry (protein-type subspace offset), not concatenation itself.
4. **K-mer features are interaction-agnostic**: `unit_diff` and `concat` perform equally well.
5. **Stage 3/4 decoupled**: create a dataset once, train with different bundles against it.
6. **Cross-validation** support implemented (`n_folds`/`fold_id`).

**Current best**: `flu_schema_raw_slot_norm_unit_diff` (ESM-2) or `flu_schema_raw_kmer_k6_slot_norm_unit_diff` (k-mer, experimental).

---

## 2. Current Status & TODO

### Completed Tasks ✅

| Task | Status | Notes |
|------|--------|-------|
| Train on Bunya | ✅ Complete | Val F1: 0.91, Test AUC: 0.956, Test F1: 0.914 |
| Analyze Bunya training results | ✅ Complete | 100% same-func negative accuracy |
| Re-run Bunya with allow_same_func_negatives=false | ✅ Complete | Retains very good performance |
| Segment-specific models for Flu A | ✅ Complete | HA-NA: 0.916 F1, PB2-PB1-PA: 0.753 F1 |
| Conservation hypothesis testing | ✅ Complete | Confirmed: 0.204 accuracy gap, 0.163 F1 gap |

### Completed Since Dec 2025

| Task | Status | Notes |
|------|--------|-------|
| Interaction features (`unit_diff`, `concat`, `diff`, `prod`) | ✅ Complete | `unit_diff` + `slot_norm` is best for ESM-2 |
| K-mer genome features | ✅ Complete | K-mer (k=6) baseline; outperforms ESM-2 |
| Stage 3/4 decoupling | ✅ Complete | Dataset once, train many |
| Cross-validation support | ✅ Complete | `n_folds`/`fold_id` in dataset config |
| H3N2 k-mer vs ESM-2 comparison | ✅ Complete | K-mer concat does NOT collapse; ESM-2 concat does |

### Pending Tasks

| Task | Priority | Notes |
|------|----------|-------|
| Baseline validation experiments | P1 | Embedding shuffle, mean embedding, swap-slot (see `docs/plans/baseline_validation_experiments_plan.md`) |
| Cross-validation full run | P1 | Run N-fold CV for publication metrics |
| Temporal holdout (train 2021-2023, test 2024) | P1 | Needs `year_train`/`year_test` config fields |
| Large dataset (full Flu A ~100K isolates) | P2 | HPC required (Polaris) |
| K-mer + XGBoost/LightGBM | P2 | Tree-based classifier comparison |
| Run embedding similarity analysis | P3 | Quantify embedding overlap |
| Contrastive fine-tuning of ESM-2 | P4 | If current approach insufficient |

---

## 3. Experimental Results

*For detailed analysis of each experiment, including error analysis, and segment pair performance, see [`EXPERIMENT_RESULTS_ANALYSIS.md`](EXPERIMENT_RESULTS_ANALYSIS.md).*

### Performance Summary

| Rank | Experiment | Accuracy | **F1 Score** | AUC | Segment Type | Conservation |
|------|------------|----------|--------------|-----|--------------|--------------|
| 1 | **Flu HA-NA (5ks)** | **0.923** | **0.916** | **0.953** | Variable | 70-90% |
| 2 | **Bunya (all)** | **0.910** | **0.911** | **0.927** | All segments | Low (different virus) |
| 3 | **Flu PB2-HA-NA (5ks)** | **0.854** | **0.855** | **0.920** | Mixed | 95% + 70-90% |
| 4 | Flu Overfit (5ks) | 0.716 | **0.750** | 0.750 | Conserved | 94-98% (1% train, 5K isolates) |
| 5 | **Flu PB2-PB1-PA (5ks)** | **0.719** | **0.753** | **0.750** | Conserved | 94-98% |

---

## 4. Biological Background

### Influenza Protein Conservation

Studies show influenza internal proteins are highly conserved ([PMC3036627](https://pmc.ncbi.nlm.nih.gov/articles/PMC3036627/)). For human strains, conservation ranges from 94-98% (most conserved). The paper's "Inter- and Intra- host strains conservation and variability analysis" section reports PB1 at 98.1% (highest). Other segments fall within the 94-98% range, with specific percentages not provided in the main text:

| Segment | Protein | Conservation (Human Strains) | Expected Signal |
|---------|---------|------------------------------|-----------------|
| 2 | PB1 | **98.1%** (highest) | Very Low |
| 1 | PB2 | 94-98% | Low |
| 3 | PA | 94-98% | Low |
| 5 | NP | 94-98% | Low |
| 4 | HA | Variable (immune pressure) | Medium-High |
| 6 | NA | Variable (immune pressure) | Medium |
| 7 | M1/M2 | 94-98% | Low |
| 8 | NS1/NEP | 94-98% | Low-Medium |

**Key insight**: Internal proteins (polymerase complex) are highly conserved; surface proteins (HA/NA) have more variation due to antigenic drift.

### Why HA-NA Performs Better

1. **Antigenic Drift**: HA and NA are under immune pressure, leading to:
   - Higher sequence diversity
   - More isolate-specific variation
   - Better embedding separability

### Why PB2-PB1-PA Performs Worse

1. **High Conservation**: 94-98% sequence identity means:
   - Most positions are identical across isolates
   - Limited isolate-specific signal
   - ESM-2 embeddings cluster by function, not isolate

---

## 5. Technical Background

### Why ESM-2 Struggles with This Task

1. **ESM-2 was trained on masked language modeling (MLM)** — predicting amino acids from context. This captures:
   - Protein structure and function
   - Evolutionary relationships
   - Amino acid properties

2. **ESM-2 was NOT trained to distinguish isolate origin**. Two PB2 proteins from different isolates produce nearly identical embeddings because they're functionally equivalent (despite being from different isolates).

3. **The information-theoretic limit**: If sequences are >98% identical, and ESM-2 generalizes over evolutionary variation, the embeddings will be nearly indistinguishable. No downstream model can recover signal that isn't in the representation.

*Reference: [ESM-2 paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)*

### Data Leakage Prevention (Status)

| Issue | Status | Notes |
|-------|--------|-------|
| Contradictory labels (duplicate sequences) | ✅ Addressed | Co-occurrence blocking implemented (see Technical Details below) |
| Data leakage (same pairs in train/test) | ✅ Addressed | Pair-key validation added |
| Same-function negative bias | ✅ Addressed | `allow_same_func_negatives: false` |
| Near-identical non-duplicate sequences | ❌ Not addressed | May be the core issue |
| Low effective diversity | ❓ Unknown | Need to quantify |

**Co-occurrence Blocking (Technical Detail):**
Highly conserved viral sequences can appear in multiple isolates. Without blocking, the same sequence pair (seq_a, seq_b) could be labeled as POSITIVE (same isolate) AND NEGATIVE (different isolates), creating contradictory training signals. The solution blocks negative pairs where the sequence pair co-occurs in ANY isolate (see `build_cooccurrence_set()` in `src/datasets/dataset_segment_pairs.py`).

### Embedding Similarity Analysis

**Key Diagnostic**: `src/analysis/analyze_embedding_similarity.py`

**The key question we need to answer:**
```
If cosine_similarity(positive_pairs) ≈ cosine_similarity(negative_pairs)
Then → ESM-2 embeddings fundamentally cannot solve this task
```

**Metrics Computed**:
| Metric | What It Measures | Interpretation |
|--------|-----------------|----------------|
| Cosine similarity distributions | Pos vs neg pair similarity | Overlap = hard to distinguish |
| Effect size (Cohen's d) | Standardized group difference | d < 0.5 = weak signal |
| Histogram overlap | Distribution separability | >80% = very weak signal |
| Intra-function similarity | Same-function protein similarity | >0.95 = high conservation |
| t-SNE visualization | Cluster structure | Function clusters vs isolate mixing |

**What High Overlap Means (And Doesn't Mean)**:

- **What it tells us**: In the RAW embedding space, positive and negative pairs look similar.
- **What it does NOT tell us**: That the information isn't recoverable through:
  1. **Non-linear transformations**: MLPs can learn complex decision boundaries
  2. **Interaction features**: `|emb_a - emb_b|` and `emb_a * emb_b` can highlight differences
  3. **Attention mechanisms**: Focusing on discriminative dimensions
  4. **Fine-tuned embeddings**: Contrastive learning can reshape the space

**Analogy**: Raw pixel values of cat vs dog images overlap heavily, yet CNNs distinguish them perfectly.

**Additional Diagnostic Analyses to Consider**:
- **Non-linear separability test**: Train a small MLP on embedding pairs to test if signal exists but isn't linear
- **Dimension analysis**: Identify which of the 1280 dimensions carry isolate signal
- **Residual analysis**: Regress out function-related variance and measure remaining isolate signal
- **Segment-stratified analysis**: Analyze HA/NA pairs separately from PB1/PB2 pairs

---

## 6. Validation Experiments: Testing for Confounders

**Purpose**: Design experiments to ensure models learn true isolate-specific signal rather than spurious correlations with metadata variables (confounders).

### Overview

Current models show strong performance, but we need to validate that they're learning the intended signal (isolate origin) rather than confounders such as:
- **Host-specific patterns**: Models might learn "Human isolates look similar" rather than true isolate signal
- **Temporal patterns**: Models might learn "2020 isolates look similar" due to temporal clustering
- **Subtype patterns**: Models might learn "H3N2 isolates look similar" due to subtype-specific characteristics

**Terminology Note**: 
- **Confounders**: Variables (host, date, subtype) that are associated with both the input (protein sequences) and could affect the outcome (isolate matching), creating spurious correlations. We use "confounders" as a concise term, though some may actually be:
  - **Biological confounders**: True biological associations (e.g., host-specific viral adaptations, subtype-specific characteristics)
  - **Batch effects**: Technical artifacts from data collection/processing (e.g., different sequencing machines, protocols, or personnel in 2020 vs 2000). Temporal patterns could reflect batch effects rather than biological evolution.
- Our experiments test for both types: if performance drops with stratified splits, it could indicate either biological confounders or batch effects (or both).

### Planned Validation Experiments

#### Experiment 1: Stratified Dataset Splitting (High Priority)

**Objective**: Test if performance is maintained when train/val/test splits are stratified across key biological variables.

**Stratification dimensions**:
- **Host organisms**: Ensure all splits have similar host distributions (Human, Pig, Chicken, etc.)
- **Collection date/year**: Temporal stratification to prevent temporal leakage
- **HA/NA subtype**: Ensure H1N1, H3N2, H5N1, etc. are distributed across splits

**Hypothesis**: 
- If performance **remains stable** → Model learns true isolate signal
- If performance **drops significantly** → Model relies on confounders (host/date/subtype patterns)

**Implementation**: 
- Use metadata from `Flu_Genomes.key` and `Flu.first-seg.meta.tab` (see `src/preprocess/flu_genomes_eda.py`)
- Modify `split_dataset_v2()` in `src/datasets/dataset_segment_pairs.py` to use stratified splitting
- Use scikit-learn's `StratifiedGroupKFold` or similar to maintain isolate-level grouping while stratifying

**Status**: 
- ✅ Metadata EDA completed
- ✅ Mapping verified: `assembly_id` → `hash_id` mapping confirmed (100% coverage, see `src/analysis/verify_metadata_mapping.py`)
- 📋 Detailed implementation plan in [`STRATIFIED_EXPERIMENTS_PLAN.md`](./STRATIFIED_EXPERIMENTS_PLAN.md)
- ⏳ Implementation pending

#### Experiment 2: Host-Held-Out Validation (Future)

**Objective**: Train on isolates from certain hosts, test on completely different hosts.

**Design**: 
- Train: Human + Pig isolates
- Test: Chicken + Duck isolates (or other host combinations)

**Hypothesis**: If performance drops dramatically, suggests host-specific confounders.

#### Experiment 3: Temporal Held-Out Validation (Future)

**Objective**: Train on older isolates, test on recent isolates (or vice versa).

**Design**:
- Train: 1990-2010 isolates
- Test: 2020-2025 isolates

**Hypothesis**: If performance drops, suggests temporal confounders or evolution effects.

#### Experiment 4: Subtype Held-Out Validation (Future)

**Objective**: Train on certain subtypes, test on others.

**Design**:
- Train: H1N1 + H3N2
- Test: H5N1 + H9N2

**Hypothesis**: If performance drops, suggests subtype-specific confounders.

### Interpretation Framework

For each validation experiment:
1. **Baseline performance**: Current model performance with random isolate-level splits
2. **Stratified/held-out performance**: Model performance with stratified/held-out splits
3. **Performance delta**: Difference between baseline and stratified performance
4. **Interpretation**:
   - **Small delta (<5% F1)**: Model learns true isolate signal ✅
   - **Medium delta (5-15% F1)**: Some confounder influence, but core signal present ⚠️
   - **Large delta (>15% F1)**: Significant confounder reliance ❌

### Related Work

These validation experiments are similar to:
- **Domain generalization** tests in ML
- **Temporal validation** in time-series models
- **Stratified cross-validation** in clinical studies
- **Batch effect correction** in genomics

---

## 7. Research Roadmap

### Approaches Ranked by Expected Impact

| Rank | Approach | Expected Impact | Effort | Status |
|------|----------|-----------------|--------|--------|
| 1 | **Segment-specific models** | High | Low | ✅ Done (Gen1) |
| 2 | **Interaction features** | High | Low | ✅ Done (Gen3) — `unit_diff` + `slot_norm` is best for ESM-2 |
| 3 | **K-mer genome features** | High | Medium | ✅ Done — k-mer (k=6) outperforms ESM-2 |
| 4 | **Cross-validation** | High | Medium | ✅ Implemented, needs full run |
| 5 | **Contrastive fine-tuning of ESM-2** | Medium-High | Medium | Not started |
| 6 | **Genome foundation models (GenSLM)** | Medium-High | High | Not started |
| 7 | **Custom viral foundation model** | Highest (if works) | Very High | Not started |

### Detailed Strategies

#### Level 1: Optimize Current Framework (Quick Wins)

**A. Interaction Features** ✅ **COMPLETED (Gen3)**
- Tested: `concat`, `diff`, `unit_diff`, `prod` (see `_ongoing_work.md` for formulas)
- **Result**: `unit_diff` + `slot_norm` is best for ESM-2. `concat` fails on homogeneous data (H3N2).
- Config: `training.interaction: unit_diff`, `training.slot_transform: slot_norm`

**B. Segment-Specific Models** ✅ **COMPLETED (Gen1)**
- **Result**: HA-NA model achieves 0.923 accuracy, 0.916 F1 (vs ~0.72 accuracy, ~0.75 F1 for conserved segments)
- **Conclusion**: Confirms conservation as the primary limiting factor

**C. K-mer Genome Features** ✅ **COMPLETED**
- K-mer (k=6, 4096-dim) outperforms ESM-2 on both mixed-subtype and H3N2-only data
- K-mer concat does NOT collapse on H3N2 (AUC 0.985 vs ESM-2 AUC 0.498)
- Config: `training.feature_source: kmer`, `conf/kmer/default.yaml`

#### Level 2: Fine-Tune Protein Foundation Models

**Contrastive Fine-Tuning (Most Promising)**:
```
Objective: Make embeddings from SAME isolate closer,
           embeddings from DIFFERENT isolates farther

Positive pairs: Proteins from same isolate
Negative pairs: Proteins from different isolates
Loss: InfoNCE / NT-Xent / Triplet loss
```

This directly reshapes the embedding space for our task.

**Multi-Task Fine-Tuning**:
- Task 1: Masked language modeling (preserve protein understanding)
- Task 2: Isolate discrimination (learn isolate-specific features)

#### Level 3: Genome Foundation Models

**Why Nucleotide-Level Models Might Help**:

1. **Synonymous mutations**: Same amino acid, different codon — invisible to protein models but may carry strain signal
2. **UTR regions**: Non-coding regions captured by genome models
3. **Codon usage bias**: Strain-specific patterns

**Models to Explore**:

| Model | Parameters | Strengths |
|-------|------------|----------|
| GenSLM | 25B | Trained on microbial genomes, captures viral-scale patterns |
| Evo2 | 7B | Long-range dependencies, diverse training data |
| HyenaDNA | - | Handles very long sequences (full segments) |
| DNABERT-2 | - | Efficient, multi-species, good baseline |

**Experimental Design**:
1. Embed each segment's NUCLEOTIDE sequence (not amino acid)
2. Create segment-pair representations
3. Train classifier on genome embeddings
4. Compare: protein vs genome vs combined embeddings

#### Level 4: Custom Architecture (If All Else Fails)

**Viral-Specific Foundation Model**:
- Train from scratch on all sequenced viral genomes
- Objective: Predict segment co-occurrence
- Architecture: Cross-segment attention

This would directly learn the signal we care about but requires significant compute resources.

---

## Files Referenced

### Results Files
- See [`EXPERIMENT_RESULTS_ANALYSIS.md`](EXPERIMENT_RESULTS_ANALYSIS.md) Section "Files Referenced" for complete list
- Learning curves: `models/{virus}/{data_version}/runs/training_*/learning_curves.png`

### Key Scripts
- `src/analysis/analyze_stage2_embeddings.py` - Embedding space visualization
- `src/analysis/analyze_stage3_datasets.py` - Dataset statistics
- `src/analysis/analyze_stage4_train.py` - Training results analysis
- `src/analysis/analyze_embedding_similarity.py` - Critical diagnostic for embedding overlap
- `src/datasets/dataset_segment_pairs.py` - Dataset creation with duplicate handling and co-occurrence blocking
- `src/embeddings/compute_esm2_embeddings.py` - Master embeddings computation
- `scripts/stage2_esm2.sh` - Embedding computation (uses master cache: `data/embeddings/{virus}/{data_version}/master_esm2_embeddings.h5`)
- `scripts/stage3_dataset.sh` - Dataset creation
- `scripts/stage4_train.sh` - Model training

### Configuration Files

Gen1 (conservation hypothesis):
- `conf/bundles/flu_ha_na_5ks.yaml` (no longer exists — superseded by Gen3 bundles)
- `conf/bundles/flu_pb2_pb1_pa_5ks.yaml`, `conf/bundles/flu_pb2_ha_na_5ks.yaml`

Gen3 (current):
- `conf/bundles/flu_schema_raw_slot_norm_unit_diff.yaml` — best ESM-2
- `conf/bundles/flu_schema_raw_slot_norm_concat.yaml` — concat comparison
- `conf/bundles/flu_schema_raw_kmer_k6_slot_norm_unit_diff.yaml` — k-mer baseline
- `conf/bundles/flu_schema_raw_kmer_k6_slot_norm_concat.yaml` — k-mer + concat
- See `conf/bundles/README.md` for the full list and inheritance conventions

---

## Related Documentation

### Technical Documentation (`docs/`)
- **Detailed Experiment Analysis**: See [EXPERIMENT_RESULTS_ANALYSIS.md](./EXPERIMENT_RESULTS_ANALYSIS.md) for comprehensive results analysis, metrics, and statistical comparisons
- **Configuration Guide**: See [CONFIGURATION_GUIDE.md](./CONFIGURATION_GUIDE.md) for detailed configuration documentation

### User Guides (`documentation/`)
- **Quick Start**: See [`../documentation/quick-start.md`](../documentation/quick-start.md) to get started quickly
- **Pipeline Overview**: See [`../documentation/pipeline-overview.md`](../documentation/pipeline-overview.md) for pipeline understanding
- **Troubleshooting**: See [`../documentation/troubleshooting.md`](../documentation/troubleshooting.md) for common issues

