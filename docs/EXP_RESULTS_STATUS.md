# Experimental Results & Research Status

**Purpose**: This document tracks project status, provides research context (biological and technical background), outlines the research roadmap, and summarizes experimental results. For detailed analysis of experiment results see [`EXPERIMENT_RESULTS_ANALYSIS.md`](EXPERIMENT_RESULTS_ANALYSIS.md).

**Date**: December 3, 2025  
**Status**: Conservation Hypothesis Confirmed ✅  
**Primary Finding**: Performance directly correlates with protein conservation levels

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Status & TODO](#2-current-status--todo)
3. [Experimental Results](#3-experimental-results)
4. [Biological Background](#4-biological-background)
5. [Technical Background](#5-technical-background)
6. [Research Roadmap](#6-research-roadmap)
7. [Next Steps](#7-next-steps)

---

## 1. Executive Summary

### Key Finding: **Conservation Hypothesis Confirmed** ✅

Performance directly correlates with protein conservation levels:
- **Variable segments (HA-NA)**: accuracy=0.923, **F1=0.916**, AUC=0.953
- **Mixed segments (PB2-HA-NA)**: accuracy=0.854, **F1=0.855**, AUC=0.920
- **Conserved segments (PB2-PB1-PA)**: accuracy=0.719, **F1=0.753**, AUC=0.750

### Main Conclusions

1. **Variable segments (HA-NA) achieve accuracy=0.923, F1=0.916** - Actually outperforms Bunya (accuracy=0.910, F1=0.911) with 5K isolates
2. **Conserved segments (PB2-PB1-PA) achieve accuracy=0.719, F1=0.753** - Biological limitation, but scores suggest some recoverable signal
3. **Segment-specific models might be the path forward** - Use HA-NA model for variable segments (F1=0.916)
4. **Pipeline is sound** - Both Bunya and Flu HA-NA prove the approach works well for variable segments

**Recommendation**: Deploy segment-specific models with realistic performance expectations.

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

### Pending Tasks

| Task | Priority | Notes |
|------|----------|-------|
| Run embedding similarity on Bunya and Flu A | P1 | Quantify embedding overlap |
| Compare Bunya vs Flu A similarity distributions | P1 | Quantify conservation impact on embedding separability |
| Try use_diff=True, use_prod=True | P2 | May improve conserved segment performance |
| Contrastive fine-tuning of ESM-2 | P3 | If current approach insufficient |
| Explore genome foundation models | P4 | GenSLM, Evo2 for nucleotide signal |

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

## 6. Research Roadmap

### Approaches Ranked by Expected Impact

| Rank | Approach | Expected Impact | Effort | Rationale |
|------|----------|-----------------|--------|-----------|
| 1 | **Segment-specific models** | High | Low | ✅ **COMPLETED** - HA-NA: 0.916 F1, PB2-PB1-PA: 0.753 F1 |
| 2 | **Contrastive fine-tuning of ESM-2** | High | Medium | Directly optimizes for isolate discrimination |
| 3 | **Interaction features** | Medium | Low | `use_diff=true, use_prod=true` captures pairwise relationships |
| 4 | **Genome foundation models** | Medium-High | High | Nucleotide-level signal (synonymous mutations) |
| 5 | **Cross-attention architecture** | Medium | Medium | Learns to focus on discriminative regions |
| 6 | **Multi-task learning** | Medium | Medium | Predict isolate + function jointly |
| 7 | **Custom viral foundation model** | Highest (if works) | Very High | Domain-specific, but requires massive compute |

### Detailed Strategies

#### Level 1: Optimize Current Framework (Quick Wins)

**A. Interaction Features** (Medium Priority)
```yaml
training:
  use_diff: true   # |emb_a - emb_b| highlights differences
  use_prod: true   # emb_a * emb_b captures interactions
```
- **Status**: Not yet tested on Flu experiments
- **Hypothesis**: May improve conserved segment performance by highlighting differences
- **Note**: Bunya already uses `use_diff=True, use_prod=True` and achieves good results

**B. Segment-Specific Models** ✅ **COMPLETED**
- **Result**: HA-NA model achieves 0.923 accuracy, 0.916 F1 (vs ~0.72 accuracy, ~0.75 F1 for conserved segments)
- **Conclusion**: Confirms conservation as the primary limiting factor
- **Recommendation**: Deploy HA-NA model for variable segments, accept lower performance for conserved segments

**C. Loss Function**
Switch from binary cross-entropy to contrastive loss:
```python
# InfoNCE loss: pull same-isolate pairs together, push different apart
loss = -log(exp(sim(a,b)/τ) / Σ exp(sim(a,neg)/τ))
```

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
- `conf/bundles/bunya.yaml`
- `conf/bundles/flu_ha_na_5ks.yaml`
- `conf/bundles/flu_pb2_pb1_pa_5ks.yaml`
- `conf/bundles/flu_pb2_ha_na_5ks.yaml`
- `conf/bundles/flu_overfit_5ks.yaml`

---

## Related Documentation

### Technical Documentation (`docs/`)
- **Detailed Experiment Analysis**: See [EXPERIMENT_RESULTS_ANALYSIS.md](./EXPERIMENT_RESULTS_ANALYSIS.md) for comprehensive results analysis, metrics, and statistical comparisons
- **Configuration Guide**: See [CONFIGURATION_GUIDE.md](./CONFIGURATION_GUIDE.md) for detailed configuration documentation

### User Guides (`documentation/`)
- **Quick Start**: See [`../documentation/quick-start.md`](../documentation/quick-start.md) to get started quickly
- **Pipeline Overview**: See [`../documentation/pipeline-overview.md`](../documentation/pipeline-overview.md) for pipeline understanding
- **Troubleshooting**: See [`../documentation/troubleshooting.md`](../documentation/troubleshooting.md) for common issues

