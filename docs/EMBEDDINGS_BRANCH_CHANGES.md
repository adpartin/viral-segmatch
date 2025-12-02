# Embeddings Branch: Major Changes Summary

**Date:** December 2, 2025  
**Branch:** `embeddings`

This document summarizes the major changes made in the `embeddings` branch, including the new master embeddings system, data leakage prevention in dataset creation, and improvements to the training pipeline.

---

## Table of Contents
1. [Master Embeddings System](#1-master-embeddings-system)
2. [Data Leakage Prevention](#2-data-leakage-prevention)
3. [Training Pipeline Improvements](#3-training-pipeline-improvements)
4. [Configuration System](#4-configuration-system)
5. [Analysis Scripts Consolidation](#5-analysis-scripts-consolidation)
6. [Shell Script Standardization](#6-shell-script-standardization)
7. [ESM-2: Understanding Its Limitations](#7-esm-2-understanding-its-limitations-for-this-task)
8. [Embedding Similarity Analysis: Interpretation Guide](#8-embedding-similarity-analysis-interpretation-guide)
9. [Research Roadmap: Improving Isolate Prediction](#9-research-roadmap-improving-isolate-prediction)
10. [Biological Constraints: What's Actually Solvable?](#10-biological-constraints-whats-actually-solvable)

---

## 1. Master Embeddings System

### Problem Solved
Previously, embeddings were computed on-demand or stored in per-run directories, leading to:
- Redundant computation of the same embeddings
- Inconsistent embedding versions across experiments
- Long training startup times due to embedding computation

### New Architecture

**Master Cache Format:**
```
data/embeddings/{virus}/{data_version}/
├── master_esm2_embeddings.h5      # HDF5 with 'emb' dataset: (N, 1280)
└── master_esm2_embeddings.parquet # Index mapping: brc_fea_id → row
```

**Key Files:**
- `src/embeddings/compute_esm2_embeddings.py` - Virus-agnostic embedding computation script
- `scripts/stage2_esm2.sh` - Consolidated shell script for running embedding computation

**Features:**
- **One-time computation**: Compute embeddings once per virus/data_version, reuse everywhere
- **Efficient lookup**: O(1) access to any protein's embedding via parquet index
- **Metadata tracking**: Model name, computation timestamp, and parameters stored in HDF5 attrs
- **Memory-efficient**: Uses row-based indexing instead of string key lookup
- **Cache detection**: Automatically skips computation if master cache exists (use `--force-recompute` to override)

**Usage:**
```bash
# Compute embeddings for a virus
./scripts/stage2_esm2.sh bunya --cuda_name cuda:7
./scripts/stage2_esm2.sh flu_a --cuda_name cuda:0

# Force recomputation
./scripts/stage2_esm2.sh bunya --cuda_name cuda:7 --force_recompute
```

---

## 2. Data Leakage Prevention

### Problem Solved
The original dataset creation had potential data leakage issues:

#### 2.1 Co-occurrence Leakage

**The Problem:** A pair (seq_a, seq_b) could appear as POSITIVE (same isolate) AND NEGATIVE (different isolates), creating contradictory labels.

**Origin of the Problem:**
Viral protein sequences are often highly conserved, meaning the same amino acid sequence can appear in multiple isolates:

```
Isolate A: [PB1_seq_X, PB2_seq_Y, PA_seq_Z]
Isolate B: [PB1_seq_X, PB2_seq_W, PA_seq_Z]  ← Same PB1 and PA sequences!
```

When creating pairs:
- **Positive**: (PB1_seq_X from A, PA_seq_Z from A) → label=1 (same isolate)
- **Negative**: (PB1_seq_X from B, PA_seq_Z from A) → label=0 (different isolates)

But **the sequences are identical**! The model sees:
- Feature vector `f(PB1_seq_X, PA_seq_Z)` → label=1
- Feature vector `f(PB1_seq_X, PA_seq_Z)` → label=0

This is a **contradictory training signal** that prevents learning.

**Solution:** Block negative pairs where the sequence pair co-occurs in ANY isolate (see `build_cooccurrence_set()`).

#### 2.2 Same-Function Negatives and Conservation Bias

**The Problem:** Using same-function proteins as negatives (e.g., PB1 from isolate A vs PB1 from isolate B) can create training bias due to protein conservation.

**Scientific Background - Influenza Protein Conservation:**
Studies show that influenza internal proteins are highly conserved:
- **PB1 segment**: 98.1% conservation in human strains (highest among all segments)
- **PB2, NP, PA segments**: ~94-97% conservation
- **HA/NA segments**: Lower conservation due to immune pressure

*Reference: [PMC3036627](https://pmc.ncbi.nlm.nih.gov/articles/PMC3036627/) - Conservation analysis across human, avian, and swine strains*

**The Mechanism:**
1. PB1 is ~98% conserved across flu strains
2. PB1_isolate_A ≈ PB1_isolate_B in sequence space
3. ESM-2(PB1_A) ≈ ESM-2(PB1_B) in embedding space
4. If we label (PB1_A, HA_X) as negative when from different isolates...
5. ...the model learns: "when one embedding looks like PB1, predict negative"
6. This biases AGAINST correct classification of true positives involving PB1

**Can Different Sequences Map to Identical Embeddings?**
Yes. ESM-2 embeddings have:
- Finite precision (1280 dims × 32-bit floats)
- Evolutionary training objective (captures function/structure, not origin)
- Conservation pressure compression (similar sequences → similar embeddings)

If two sequences map to indistinguishable embeddings, **no downstream model can tell them apart** — this is an information-theoretic limit.

**Configuration:** Set `allow_same_func_negatives: false` to avoid this bias.

### New Dataset Creation Features

**Key File:** `src/datasets/dataset_segment_pairs.py`

**1. Co-occurrence Blocking:**
```python
def build_cooccurrence_set(df: pd.DataFrame) -> tuple[set, dict]:
    """Build a set of all sequence pairs that co-occur in any isolate.
    
    These pairs could be labeled as positive (same isolate), so they should NOT
    be used as negative pairs to avoid contradictory labels.
    """
```

**2. Same-Function Negative Control:**
```yaml
# In config bundle (e.g., conf/bundles/bunya.yaml)
dataset:
  allow_same_func_negatives: false  # Disable same-function negatives
  max_same_func_ratio: 0.5          # If enabled, limit to 50% of negatives
```

**3. Rejection Statistics:**
The script now tracks and reports why negative pairs were rejected:
- `blocked_cooccur`: Sequences co-occur in some isolate (would create contradictory labels)
- `duplicate_brc`: Same BRC pair already seen
- `duplicate_seq`: Same sequence pair already seen
- `same_func_limit`: Same-function limit reached

**4. Duplicate Statistics Output:**
Saves `duplicate_stats.json` with co-occurrence analysis:
```json
{
  "total_cooccur_pairs": 12345,
  "pairs_in_multiple_isolates": 5678,
  "max_isolates_per_pair": 42
}
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `allow_same_func_negatives` | Allow negative pairs with same protein function | `false` |
| `max_same_func_ratio` | Max fraction of same-function negatives (if allowed) | `0.5` |
| `neg_to_pos_ratio` | Ratio of negative to positive pairs | `1.0` |
| `hard_partition_isolates` | Keep all proteins from an isolate in same split | `true` |

---

## 3. Training Pipeline Improvements

### Key File: `src/models/train_esm2_frozen_pair_classifier.py`

**1. Interaction Features:**
The model now supports optional interaction features beyond simple concatenation:
```python
class SegmentPairDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        embeddings_file: str,
        use_diff: bool = False,   # Include |emb_a - emb_b|
        use_prod: bool = False,   # Include emb_a * emb_b
        ...
    )
```

**Configuration:**
```yaml
training:
  use_diff: false   # Add absolute difference feature
  use_prod: false   # Add element-wise product feature
```

**2. Threshold Optimization:**
Multiple F-beta score options for threshold selection:
```yaml
training:
  threshold_metric: 'f1'  # Options: 'f1', 'f0.5', 'f2', null (use 0.5)
```

**3. Early Stopping:**
```yaml
training:
  patience: 15
  early_stopping_metric: 'f1'  # Options: 'loss', 'f1', 'auc', null
```

**4. Shared Embedding Cache:**
The `SegmentPairDataset` class uses a shared cache across train/val/test splits:
```python
class SegmentPairDataset(Dataset):
    _shared_embedding_cache: dict[str, np.ndarray] = {}  # Class-level cache
```

**5. Learning Verification:**
Integrated utilities to verify the model is actually learning:
```python
from src.utils.learning_verification_utils import (
    check_initialization_loss,
    compute_baseline_metrics,
    plot_learning_curves
)
```

---

## 4. Configuration System

### Hydra-Based Config Bundles
All pipeline scripts now use a unified Hydra configuration system.

**Bundle Structure:**
```
conf/bundles/
├── bunya.yaml
├── flu_a.yaml
├── flu_a_3p_1ks_no_same_func.yaml
├── flu_a_3p_1ks_same_func.yaml
├── flu_a_overfit_test.yaml
└── flu_a_plateau_analysis.yaml
```

**Bundle Composition:**
```yaml
# conf/bundles/bunya.yaml
defaults:
  - /virus: bunya
  - /paths: bunya
  - /embeddings: default
  - /dataset: default
  - /training: base
  - _self_

# Overrides
dataset:
  allow_same_func_negatives: false
  neg_to_pos_ratio: 1.0

training:
  use_diff: false
  use_prod: false
```

### Seed Management
Hierarchical seed system for reproducibility:
```yaml
master_seed: 42  # Master seed for all processes

process_seeds:
  preprocessing: null  # Derive from master_seed
  embeddings: null
  dataset: null
  training: null
  evaluation: null
```

### Path Utilities
Dynamic path generation based on configuration:
```python
# src/utils/path_utils.py
def build_embeddings_paths(project_root, virus_name, data_version, config):
    """Build paths for embedding files."""
    
def build_dataset_paths(project_root, virus_name, data_version, config):
    """Build paths for dataset files."""
    
def build_training_paths(project_root, virus_name, data_version, config):
    """Build paths for training outputs."""
```

---

## 5. Analysis Scripts Consolidation

### New Structure
All analysis scripts consolidated into `src/analysis/`:

| Script | Stage | Purpose |
|--------|-------|---------|
| `analyze_stage1_preprocess.py` | 1 | Preprocessing quality analysis |
| `analyze_stage2_embeddings.py` | 2 | Embedding space visualization (PCA, UMAP) |
| `analyze_stage3_datasets.py` | 3 | Dataset statistics and pair analysis |
| `analyze_stage4_train.py` | 4 | Training results analysis |
| `create_presentation_plots.py` | 4 | Publication-ready plots |
| `analyze_embedding_similarity.py` | Diagnostic | Pos/neg pair similarity distributions |

### Standardized Interface
All scripts now use consistent CLI arguments:
```bash
python src/analysis/analyze_stage2_embeddings.py \
    --virus bunya \
    --data_version April_2025

python src/analysis/analyze_stage3_datasets.py \
    --virus bunya \
    --data_version April_2025 \
    --config_bundle bunya
```

### Output Structure
```
results/{virus}/{data_version}/
├── preprocess_analysis/      # Stage 1 outputs
├── embeddings_analysis/      # Stage 2 outputs (shared across bundles)
└── {config_bundle}/
    ├── dataset_analysis/     # Stage 3 outputs
    └── training_analysis/    # Stage 4 outputs
```

---

## 6. Shell Script Standardization

### Consolidated Pipeline Scripts
```
scripts/
├── stage2_esm2.sh      # Embedding computation
├── stage3_dataset.sh   # Dataset creation
└── stage4_train.sh     # Model training
```

### Common Features
All stage scripts include:
- **Git provenance**: Commit hash, branch, dirty status
- **Structured logging**: Timestamped logs saved to `logs/{stage}/`
- **Experiment registry**: Automatic registration of runs
- **Symlinked latest log**: `{stage}_{bundle}_latest.log`

### Usage Examples
```bash
# Full pipeline for Bunya
./scripts/stage2_esm2.sh bunya --cuda_name cuda:7
./scripts/stage3_dataset.sh bunya
./scripts/stage4_train.sh bunya --cuda_name cuda:7 \
    --dataset_dir data/datasets/bunya/April_2025/runs/dataset_bunya_20251201_182905

# Full pipeline for Flu A
./scripts/stage2_esm2.sh flu_a --cuda_name cuda:0
./scripts/stage3_dataset.sh flu_a_3p_1ks_no_same_func
./scripts/stage4_train.sh flu_a_3p_1ks_no_same_func --cuda_name cuda:0
```

---

## 7. ESM-2: Understanding Its Limitations for This Task

### Critical Insight: ESM-2 Was Not Trained for Isolate Discrimination

**Training Objective:** ESM-2 uses masked language modeling (MLM) — predicting masked amino acids from context. This teaches the model to capture:
- Protein structure and function
- Evolutionary relationships
- Amino acid properties and motifs

**What ESM-2 Does NOT Capture:**
- Isolate/strain identity
- Temporal information (when a sequence was sampled)
- Host species origin
- Geographic origin

**Implication:** Proteins with the same function will have similar embeddings regardless of which isolate they came from. This is by design — ESM-2 generalizes across evolution — but it works against our isolate-discrimination task.

*Reference: [ESM-2 paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)*

---

## 8. Embedding Similarity Analysis: Interpretation Guide

### Key Diagnostic Script
`src/analysis/analyze_embedding_similarity.py` - Measures whether embeddings contain isolate-distinguishing signal.

**Metrics Computed:**
| Metric | What It Measures | Interpretation |
|--------|-----------------|----------------|
| Cosine similarity distributions | Pos vs neg pair similarity | Overlap = hard to distinguish |
| Effect size (Cohen's d) | Standardized group difference | d < 0.5 = weak signal |
| Histogram overlap | Distribution separability | >80% = very weak signal |
| Intra-function similarity | Same-function protein similarity | >0.95 = high conservation |
| t-SNE visualization | Cluster structure | Function clusters vs isolate mixing |

### What High Overlap Means (And Doesn't Mean)

**What it tells us:**
In the RAW embedding space, positive and negative pairs look similar.

**What it does NOT tell us:**
That the information isn't recoverable through:
1. **Non-linear transformations**: MLPs can learn complex decision boundaries
2. **Interaction features**: `|emb_a - emb_b|` and `emb_a * emb_b` can highlight differences
3. **Attention mechanisms**: Focusing on discriminative dimensions
4. **Fine-tuned embeddings**: Contrastive learning can reshape the space

**Analogy:** Raw pixel values of cat vs dog images overlap heavily, yet CNNs distinguish them perfectly.

### Additional Analyses to Consider

The current diagnostic could be extended with:

1. **Non-linear Separability Test**
   - Train a small MLP on the embedding pairs
   - If it achieves >80% accuracy, signal exists but isn't linear
   - Implementation: Add `--test_mlp_separability` flag

2. **Attention/Dimension Analysis**
   - Compute per-dimension discriminative power (e.g., mutual information)
   - Identify which of the 1280 dimensions carry isolate signal
   - May reveal that signal is concentrated in few dimensions

3. **Residual Analysis**
   - Regress out function-related variance from embeddings
   - Measure remaining signal after function is accounted for
   - Tests: "After removing what ESM-2 captures, is there residual isolate info?"

4. **Segment-Stratified Analysis**
   - Analyze HA/NA pairs separately from PB1/PB2 pairs
   - Conservation varies by segment; signal may be stronger for variable segments

---

## 9. Research Roadmap: Improving Isolate Prediction

### Approaches Ranked by Expected Impact

| Rank | Approach | Expected Impact | Effort | Rationale |
|------|----------|-----------------|--------|-----------|
| 1 | **Segment-specific models** | High | Low | Variable segments (HA/NA) should have more signal |
| 2 | **Contrastive fine-tuning of ESM-2** | High | Medium | Directly optimizes for isolate discrimination |
| 3 | **Interaction features** | Medium | Low | `use_diff=true, use_prod=true` captures pairwise relationships |
| 4 | **Genome foundation models** | Medium-High | High | Nucleotide-level signal (synonymous mutations) |
| 5 | **Cross-attention architecture** | Medium | Medium | Learns to focus on discriminative regions |
| 6 | **Multi-task learning** | Medium | Medium | Predict isolate + function jointly |
| 7 | **Custom viral foundation model** | Highest (if works) | Very High | Domain-specific, but requires massive compute |

### Detailed Strategies

#### Level 1: Optimize Current Framework (Quick Wins)

**A. Feature Engineering**
```yaml
training:
  use_diff: true   # |emb_a - emb_b| highlights differences
  use_prod: true   # emb_a * emb_b captures interactions
```

**B. Segment-Specific Models**
Train separate models for different segment pairs:
- Model 1: HA-NA pairs (high variability, expect good performance)
- Model 2: PB1-PB2-PA pairs (high conservation, expect lower performance)
- Model 3: Mixed pairs

This tests the **conservation hypothesis**: if HA-NA performs well but polymerase pairs don't, conservation is the limiting factor.

**C. Loss Function**
Switch from binary cross-entropy to contrastive loss:
```python
# InfoNCE loss: pull same-isolate pairs together, push different apart
loss = -log(exp(sim(a,b)/τ) / Σ exp(sim(a,neg)/τ))
```

#### Level 2: Fine-Tune Protein Foundation Models

**Contrastive Fine-Tuning (Most Promising):**
```
Objective: Make embeddings from SAME isolate closer,
           embeddings from DIFFERENT isolates farther

Positive pairs: Proteins from same isolate
Negative pairs: Proteins from different isolates
Loss: InfoNCE / NT-Xent / Triplet loss
```

This directly reshapes the embedding space for our task.

**Multi-Task Fine-Tuning:**
- Task 1: Masked language modeling (preserve protein understanding)
- Task 2: Isolate discrimination (learn isolate-specific features)

#### Level 3: Genome Foundation Models

**Why Nucleotide-Level Models Might Help:**

1. **Synonymous mutations**: Same amino acid, different codon — invisible to protein models but may carry strain signal
2. **UTR regions**: Non-coding regions captured by genome models
3. **Codon usage bias**: Strain-specific patterns

**Models to Explore:**

| Model | Parameters | Strengths |
|-------|------------|-----------|
| GenSLM | 25B | Trained on microbial genomes, captures viral-scale patterns |
| Evo2 | 7B | Long-range dependencies, diverse training data |
| HyenaDNA | - | Handles very long sequences (full segments) |
| DNABERT-2 | - | Efficient, multi-species, good baseline |

**Experimental Design:**
1. Embed each segment's NUCLEOTIDE sequence (not amino acid)
2. Create segment-pair representations
3. Train classifier on genome embeddings
4. Compare: protein vs genome vs combined embeddings

#### Level 4: Custom Architecture (If All Else Fails)

**Viral-Specific Foundation Model:**
- Train from scratch on all sequenced viral genomes
- Objective: Predict segment co-occurrence
- Architecture: Cross-segment attention

This would directly learn the signal we care about but requires significant compute resources.

---

## 10. Biological Constraints: What's Actually Solvable?

### The Conservation Reality

**Influenza A Protein Conservation by Segment:**

| Segment | Protein | Conservation (Human) | Expected Signal |
|---------|---------|---------------------|-----------------|
| 1 | PB2 | ~95% | Low |
| 2 | PB1 | ~98% | Very Low |
| 3 | PA | ~94% | Low |
| 4 | HA | ~70-85% | Medium-High |
| 5 | NP | ~95% | Low |
| 6 | NA | ~80-90% | Medium |
| 7 | M1/M2 | ~95% | Low |
| 8 | NS1/NEP | ~93% | Low-Medium |

*Note: HA/NA vary more due to immune pressure (antigenic drift)*

### Realistic Performance Expectations

| Pair Type | Expected Performance | Rationale |
|-----------|---------------------|-----------|
| HA-NA pairs | High (~85-95%) | Most variable segments |
| HA-PB pairs | Medium (~75-85%) | One variable, one conserved |
| PB1-PB2 pairs | Lower (~65-75%) | Both highly conserved |
| All pairs | Moderate (~75-80%) | Conservation limits overall |

### The Segment-Specific Hypothesis

**Core Idea:** Train and evaluate separate models for different segment combinations to:
1. Identify which segment pairs are predictable
2. Understand the conservation-performance relationship
3. Guide future modeling decisions

**Experiments:**
```
Experiment 1: HA-centric pairs (HA-NA, HA-NP, HA-M)
Experiment 2: Polymerase pairs (PB1-PB2, PB1-PA, PB2-PA)
Experiment 3: All pairs (current approach)

Compare: accuracy, F1, AUC across experiments
```

If HA-centric models significantly outperform polymerase models, this confirms conservation as the primary limiting factor.

---

## Key Diagnostics Added

### Embedding Similarity Analysis
`src/analysis/analyze_embedding_similarity.py` - Critical diagnostic for understanding if embeddings can distinguish same-isolate vs different-isolate pairs.

**Metrics computed:**
- Cosine similarity distributions for positive vs negative pairs
- Effect size (Cohen's d) between distributions
- Histogram overlap percentage
- Intra-function vs inter-function similarity (tests if ESM-2 clusters by function)
- t-SNE visualization colored by function and isolate

**Purpose:** Determine if the model plateau is due to:
1. High protein conservation (similar embeddings across isolates)
2. ESM-2 clustering by function, not isolate
3. Overlapping positive/negative distributions

---

## Migration Notes

### For Existing Experiments
1. **Embeddings**: Re-run `stage2_esm2.sh` to generate master cache format
2. **Datasets**: Re-run `stage3_dataset.sh` with appropriate `allow_same_func_negatives` setting
3. **Training**: Update to use master cache format (automatic if embeddings updated)

### Breaking Changes
- Old embedding format (per-protein HDF5 keys) is no longer supported in training
- Dataset creation now requires config bundle with `allow_same_func_negatives` setting
- Analysis scripts require `--virus` and `--data_version` arguments

---

## Files Changed Summary

### New Files
- `src/embeddings/compute_esm2_embeddings.py`
- `scripts/stage2_esm2.sh`
- `scripts/stage3_dataset.sh` (renamed from `run_dataset.sh`)
- `scripts/stage4_train.sh` (renamed from `run_training.sh`)
- `src/analysis/analyze_embedding_similarity.py`
- `src/utils/path_utils.py`
- `src/utils/experiment_registry.py`
- `src/utils/learning_verification_utils.py`

### Modified Files
- `src/datasets/dataset_segment_pairs.py` - Data leakage prevention
- `src/models/train_esm2_frozen_pair_classifier.py` - Master cache support, interaction features
- `conf/bundles/*.yaml` - New config options
- `src/analysis/*.py` - Standardized interfaces

### Deleted Files
- `src/eda/generate_plots.py` (redundant)
- `src/postprocess/postprocess_pair_classifier.py` (empty TODO)
- Legacy shell scripts replaced by stage scripts

