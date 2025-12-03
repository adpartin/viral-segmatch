# Experimental Results & Research Status

**Purpose**: This document tracks project status, provides research context (biological and technical background), outlines the research roadmap, and summarizes experimental results. For detailed analysis of experiment results, metrics, and statistical comparisons, see [`EXPERIMENT_RESULTS_ANALYSIS.md`](EXPERIMENT_RESULTS_ANALYSIS.md).

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
- **Variable segments (HA-NA)**: 92.3% accuracy, **91.6% F1**, 0.953 AUC
- **Mixed segments (PB2-HA-NA)**: 85.4% accuracy, **85.5% F1**, 0.920 AUC  
- **Conserved segments (PB2-PB1-PA)**: 71.9% accuracy, **75.3% F1**, 0.750 AUC

### Main Conclusions

1. ✅ **Variable segments (HA-NA) achieve 92.3% accuracy, 91.6% F1** - Actually outperforms Bunya (91.0% accuracy, 91.1% F1) with 5K isolates
2. ⚠️ **Conserved segments (PB2-PB1-PA) achieve 71.9% accuracy, 75.3% F1** - Biological limitation, but F1 suggests some recoverable signal
3. ✅ **Segment-specific models are the path forward** - Use HA-NA model for variable segments (91.6% F1)
4. ✅ **Pipeline is sound** - Both Bunya and Flu HA-NA prove the approach works excellently for variable segments

**Recommendation**: Deploy segment-specific models with realistic performance expectations:
- **Variable segments (HA-NA)**: Expect ~92% accuracy, ~92% F1 (with 5K+ isolates)
- **Conserved segments (PB2-PB1-PA)**: Expect ~72% accuracy, ~75% F1

---

## 2. Current Status & TODO

### Completed Tasks ✅

| Task | Status | Notes |
|------|--------|-------|
| Train model on Bunya | ✅ Complete | Val F1: 0.75→0.91, Test AUC: 0.956, Test F1: 91.4% |
| Analyze Bunya training results | ✅ Complete | 100% same-func negative accuracy |
| Run embedding similarity on Bunya | ✅ Complete | Results in embedding_similarity/ |
| Re-run Bunya with allow_same_func_negatives=false | ✅ Complete | Still achieves 91.2% accuracy, 0.940 AUC |
| Document embeddings branch changes | ✅ Complete | Technical details integrated into this document |
| **Segment-specific models for Flu A** | ✅ **Complete** | **HA-NA: 91.6% F1, PB2-PB1-PA: 75.3% F1** |
| **Conservation hypothesis testing** | ✅ **Complete** | **Confirmed: 20.4% accuracy gap, 16.3% F1 gap** |

### Pending Tasks

| Task | Priority | Notes |
|------|----------|-------|
| Run embedding similarity on Flu A | P1 | Critical diagnostic to quantify embedding overlap |
| Compare Bunya vs Flu A similarity distributions | P1 | Quantify conservation impact on embedding separability |
| Try use_diff=True, use_prod=True on Flu A | P2 | May improve conserved segment performance |
| Contrastive fine-tuning of ESM-2 | P3 | If current approach insufficient |
| Explore genome foundation models | P4 | GenSLM, Evo2 for nucleotide signal |

---

## 3. Experimental Results

*For detailed analysis of each experiment, including error analysis, segment pair performance, and statistical comparisons, see [`EXPERIMENT_RESULTS_ANALYSIS.md`](EXPERIMENT_RESULTS_ANALYSIS.md).*

### Performance Summary

| Rank | Experiment | Accuracy | **F1 Score** | AUC | Segment Type | Conservation |
|------|------------|----------|--------------|-----|--------------|--------------|
| 1 | **Flu HA-NA (5ks)** | **92.3%** | **91.6%** | **0.953** | Variable | 70-90% |
| 2 | **Bunya (all)** | **91.0%** | **91.1%** | **0.927** | All segments | Low (different virus) |
| 3 | **Flu PB2-HA-NA (5ks)** | **85.4%** | **85.5%** | **0.920** | Mixed | 95% + 70-90% |
| 4 | Flu Overfit | 79.3% | **82.4%** | 0.771 | Conserved | 94-98% (small dataset) |
| 5 | **Flu PB2-PB1-PA (5ks)** | **71.9%** | **75.3%** | **0.750** | Conserved | 94-98% |

### Key Findings

1. **Conservation-Performance Correlation** ✅
   - Variable segments (HA-NA): **92.3%** accuracy, **91.6% F1** — Excellent performance
   - Conserved segments (PB2-PB1-PA): **71.9%** accuracy, **75.3% F1** — Biological limitation
   - **20.4 percentage point accuracy difference** (16.3% F1 difference) confirms conservation limits performance

2. **Flu HA-NA Outperforms Bunya** — With 5K isolates, variable segments achieve better performance than the baseline

3. **Segment-Specific Models Are Valuable** — HA-NA model achieves 91.6% F1 vs ~75% F1 for conserved segments

*See [`EXPERIMENT_RESULTS_ANALYSIS.md`](EXPERIMENT_RESULTS_ANALYSIS.md) for detailed metrics, error analysis, biological interpretation, and statistical summaries.*

---

## 4. Biological Background

### Influenza Protein Conservation

Studies show influenza internal proteins are highly conserved ([PMC3036627](https://pmc.ncbi.nlm.nih.gov/articles/PMC3036627/)):

| Segment | Protein | Conservation (Human Strains) | Expected Signal |
|---------|---------|------------------------------|-----------------|
| 2 | PB1 | **98.1%** (highest) | Very Low |
| 1 | PB2 | ~95% | Low |
| 3 | PA | ~94% | Low |
| 5 | NP | ~95% | Low |
| 4 | HA | 70-85% (immune pressure) | Medium-High |
| 6 | NA | 80-90% (immune pressure) | Medium |
| 7 | M1/M2 | ~95% | Low |
| 8 | NS1/NEP | ~93% | Low-Medium |

**Key insight**: Internal proteins (polymerase complex) are highly conserved; surface proteins (HA/NA) have more variation due to antigenic drift.

### Why HA-NA Performs Better

1. **Antigenic Drift**: HA and NA are under immune pressure, leading to:
   - Higher sequence diversity (70-90% identity vs 94-98%)
   - More isolate-specific variation
   - Better embedding separability

2. **ESM-2 Captures Variation**: Variable segments have enough differences for ESM-2 to distinguish isolates

### Why PB2-PB1-PA Performs Worse

1. **High Conservation**: 94-98% sequence identity means:
   - Most positions are identical across isolates
   - Limited isolate-specific signal
   - ESM-2 embeddings cluster by function, not isolate

2. **Information-Theoretic Limit**: If sequences are >98% identical, embeddings will be nearly indistinguishable regardless of model architecture

3. **PB2-PB1 Worst**: Both segments are highly conserved, creating the hardest discrimination task

### Realistic Performance Expectations

| Pair Type | Expected Performance | Rationale |
|-----------|---------------------|-----------|
| HA-NA pairs | High (~85-95%) | Most variable segments |
| HA-PB pairs | Medium (~75-85%) | One variable, one conserved |
| PB1-PB2 pairs | Lower (~65-75%) | Both highly conserved |
| All pairs | Moderate (~75-80%) | Conservation limits overall |

---

## 5. Technical Background

### Why ESM-2 Struggles with This Task

1. **ESM-2 was trained on masked language modeling (MLM)** — predicting amino acids from context. This captures:
   - Protein structure and function
   - Evolutionary relationships
   - Amino acid properties

2. **ESM-2 was NOT trained to distinguish isolate origin**. Two HA proteins from different isolates produce nearly identical embeddings because they're functionally equivalent.

3. **The information-theoretic limit**: If sequences are >98% identical, and ESM-2 generalizes over evolutionary variation, the embeddings will be nearly indistinguishable. No downstream model can recover signal that isn't in the representation.

**In simpler terms**: If I hand you two PB1 protein embeddings and ask "are these from the same isolate?" — if all PB1 proteins are 98% identical and map to essentially the same embedding, the task is impossible regardless of model architecture.

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
| 1 | **Segment-specific models** | High | Low | ✅ **COMPLETED** - HA-NA: 86.5% F1, PB2-PB1-PA: 77.1% F1 |
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
- **Result**: HA-NA model achieves 86.3% accuracy, 86.5% F1 (vs ~73% accuracy, ~77% F1 for all segments)
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

## 7. Next Steps

*For detailed recommended next steps based on experiment results, see [`EXPERIMENT_RESULTS_ANALYSIS.md`](EXPERIMENT_RESULTS_ANALYSIS.md) Section "Implications for Future Work".*

### Immediate Actions

1. **Run embedding similarity analysis on Flu A**
   - Quantify cosine similarity distributions for positive vs negative pairs
   - Compare with Bunya results to confirm conservation impact
   - See `src/analysis/analyze_embedding_similarity.py`

2. **Test interaction features** (`use_diff=True, use_prod=True`)
   - May improve conserved segment performance
   - See `conf/bundles/flu_pb2_pb1_pa_5ks.yaml` for example config

3. **Segment-specific model deployment**
   - Deploy HA-NA model for variable segments (92.3% accuracy, 91.6% F1)
   - Accept lower performance for conserved segments (71.9% accuracy, 75.3% F1)

### Success Criteria

- **Short-term**: ✅ Achieved for Bunya (Val F1 improves beyond 0.74, Test F1: 91.4%)
- **Medium-term**: Quantify the theoretical ceiling for Flu A based on embedding similarity
- **Long-term**: Establish robust pipeline for viral segment matching (or document biological limitations for Flu A)

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
- `conf/bundles/flu_overfit.yaml`

---

## Related Documentation

### Technical Documentation (`docs/`)
- **Detailed Experiment Analysis**: See [EXPERIMENT_RESULTS_ANALYSIS.md](./EXPERIMENT_RESULTS_ANALYSIS.md) for comprehensive results analysis, metrics, and statistical comparisons
- **Configuration Guide**: See [CONFIGURATION_GUIDE.md](./CONFIGURATION_GUIDE.md) for detailed configuration documentation

### User Guides (`documentation/`)
- **Quick Start**: See [`../documentation/quick-start.md`](../documentation/quick-start.md) to get started quickly
- **Pipeline Overview**: See [`../documentation/pipeline-overview.md`](../documentation/pipeline-overview.md) for pipeline understanding
- **Troubleshooting**: See [`../documentation/troubleshooting.md`](../documentation/troubleshooting.md) for common issues

