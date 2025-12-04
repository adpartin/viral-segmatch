# Experiment Results Analysis

**Purpose**: This document provides a detailed scientific analysis of completed experiments, focusing on performance metrics and biological interpretation of results. For project status, research roadmap, and background context, see [`EXP_RESULTS_STATUS.md`](EXP_RESULTS_STATUS.md).

**Date**: December 3, 2025  
**Experiments Analyzed**: 5 config bundles across Bunya and Flu A (5K isolates for Flu experiments)

---

## Executive Summary

This analysis compares model performance across different segment combinations to test the **conservation hypothesis**: that with highly conserved proteins (PB1, PB2, PA) it's harder to predict the isolate of origin than with variable proteins (HA, NA).

### Key Finding: **Conservation Hypothesis Confirmed** ✅

Performance directly correlates with protein conservation levels:
- **Variable segments (HA-NA)**: 0.923 accuracy, **0.916 F1**, 0.953 AUC
- **Mixed segments (PB2-HA-NA)**: 0.854 accuracy, **0.855 F1**, 0.920 AUC  
- **Conserved segments (PB2-PB1-PA)**: 0.719 accuracy, **0.753 F1**, 0.750 AUC

---

## Detailed Results

### 1. Bunya (Baseline - Core protein only)

**Config**: `bunya.yaml`  
**Segments**: L (polymerase), M (glycoprotein), S (nucleocapsid)  
**Dataset**: Full dataset, no same-function negatives

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.910** |
| **F1 Score** | **0.911** |
| **AUC-ROC** | **0.927** |
| **Avg Precision** | **0.882** |

**Segment Pair Performance**:

| Segment Pair | Accuracy | F1 Score | AUC | Notes |
|--------------|----------|----------|-----|-------|
| M-S | 0.928 | **0.925** | 0.933 | Best performance |
| L-M | 0.910 | **0.918** | 0.926 | |
| L-S | 0.894 | **0.892** | 0.913 | |

**Conclusion**: ✅ **Pipeline works well** - Bunya demonstrates strong performance across all three segment pairs.

---

### 2. Flu HA-NA (Variable Segments)

**Config**: `flu_ha_na_5ks.yaml`  
**Segments**: HA (Segment 4), NA (Segment 6)  
**Conservation**: 70-90% (variable due to immune pressure)  
**Dataset**: 5K isolates, no same-function negatives

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.923** |
| **F1 Score** | **0.916** |
| **AUC-ROC** | **0.953** |
| **Avg Precision** | **0.899** |

**Segment Pair Performance**:

| Segment Pair | Accuracy | F1 Score | AUC | Notes |
|--------------|----------|----------|-----|-------|
| S4-S6 (HA-NA) | 0.923 | **0.916** | 0.953 | |

**Error Analysis**:
- False Positives: 60 (avg confidence: 0.809)
- False Negatives: 7 (avg confidence: 0.294)

**Conclusion**: ✅ **Strong performance** - Variable segments show good separability.

---

### 3. Flu PB2-PB1-PA (Highly Conserved Polymerase)

**Config**: `flu_pb2_pb1_pa_5ks.yaml`  
**Segments**: PB2 (Segment 1, ~95%), PB1 (Segment 2, **98.1%**), PA (Segment 3, ~94%)  
**Conservation**: Highest conservation (94-98%)  
**Dataset**: 5K isolates, no same-function negatives

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.719** |
| **F1 Score** | **0.753** |
| **AUC-ROC** | **0.750** |
| **Avg Precision** | **0.598** |

**Error Analysis**:
- False Positives: 698 (avg confidence: 0.704)
- False Negatives: 0 (avg confidence: N/A)

**Segment Pair Performance**:

| Segment Pair | Accuracy | F1 Score | AUC | Notes |
|--------------|----------|----------|-----|-------|
| S1-S3 (PB2-PA) | 0.735 | **0.769** | 0.737 | Best performance |
| S1-S2 (PB2-PB1) | 0.719 | **0.751** | 0.761 | |
| S2-S3 (PB1-PA) | 0.704 | **0.739** | 0.755 | |

**Conclusion**: ⚠️ **Lower performance** - Highly conserved polymerase segments show reduced separability, with PB2-PB1 (both ~95-98% conserved) performing worst.

---

### 4. Flu PB2-HA-NA (Mixed: Conserved + Variable)

**Config**: `flu_pb2_ha_na_5ks.yaml`  
**Segments**: PB2 (Segment 1, ~95% conserved), HA (Segment 4, 70-85% variable), NA (Segment 6, 80-90% variable)  
**Dataset**: 5K isolates, no same-function negatives

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.854** |
| **F1 Score** | **0.855** |
| **AUC-ROC** | **0.920** |
| **Avg Precision** | **0.842** |

**Error Analysis**:
- False Positives: 366 (avg confidence: 0.757)
- False Negatives: 9 (avg confidence: 0.284)

**Segment Pair Performance**:

| Segment Pair | Accuracy | F1 Score | AUC | Notes |
|--------------|----------|----------|-----|-------|
| S4-S6 (HA-NA) | 0.900 | **0.894** | 0.937 | Variable-variable pair (best) |
| S1-S6 (PB2-NA) | 0.842 | **0.846** | 0.901 | Conserved-variable pair |
| S1-S4 (PB2-HA) | 0.820 | **0.828** | 0.913 | Conserved-variable pair |

**Conclusion**: ✅ **Good performance** - Mixing one conserved segment (PB2) with two variable segments (HA, NA) achieves strong performance, better than conserved-only but slightly below variable-only.

---

### 5. Flu Overfit Test (5K Isolates)

**Config**: `flu_overfit_5ks.yaml`  
**Segments**: PB2, PB1, PA (polymerase)  
**Dataset**: **5K isolates**, 1% train, 50% val  
**Purpose**: Test model capacity with small training set (overfitting capacity test)

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.716** |
| **F1 Score** | **0.750** |
| **AUC-ROC** | **0.750** |
| **Avg Precision** | **0.602** |

**Segment Pair Performance**:

| Segment Pair | Accuracy | F1 Score | AUC | Notes |
|--------------|----------|----------|-----|-------|
| S2-S3 (PB1-PA) | 0.729 | **0.760** | 0.769 | Best performance |
| S1-S3 (PB2-PA) | 0.714 | **0.750** | 0.764 | |
| S1-S2 (PB2-PB1) | 0.705 | **0.740** | 0.757 | |

**Conclusion**: ⚠️ **Small training set limits performance** - With only 1% training data (50 isolates), model achieves 0.716 accuracy, 0.750 F1, similar to full PB2-PB1-PA experiment (0.719 accuracy, 0.753 F1). Does this suggest the model is not overfitting but rather limited by the conserved nature of these segments? 

---

## Comparative Analysis

### Performance Ranking by Segment Type

| Rank | Experiment | Accuracy | **F1 Score** | AUC | Segment Type | Conservation |
|------|------------|----------|--------------|-----|--------------|--------------|
| 1 | **Flu HA-NA (5ks)** | **0.923** | **0.916** | **0.953** | Variable | 70-90% |
| 2 | **Bunya (all)** | **0.910** | **0.911** | **0.927** | All segments | Low (different virus) |
| 3 | **Flu PB2-HA-NA (5ks)** | **0.854** | **0.855** | **0.920** | Mixed | 95% + 70-90% |
| 4 | Flu Overfit (5ks) | 0.716 | **0.750** | 0.750 | Conserved | 94-98% (1% train, 5K isolates) |
| 5 | **Flu PB2-PB1-PA (5ks)** | **0.719** | **0.753** | **0.750** | Conserved | 94-98% |

### Key Insights

TODO

---

## Biological Interpretation (TODO: these are mostly hypotheses and needs to confirmed)

*For detailed biological background on protein conservation and ESM-2 limitations, see [`EXP_RESULTS_STATUS.md`](EXP_RESULTS_STATUS.md) Section 4 (Biological Background) and Section 5 (Technical Background).*

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

---

## Implications for Future Work

### ✅ Confirmed Hypotheses

1. **Pipeline works** - This has been confirmed with Bunya and Flu A
2. **Conservation limits performance** - Variable segments (HA-NA) outperform conserved (PB2-PB1-PA)
3. **Segment-specific models should be considered** - instead of all-segment model

### 🎯 Recommended Next Steps

1. **Interaction Features**
   - Test `use_diff=True, use_prod=True`
   - May improve performance by highlighting differences

2. **Contrastive Fine-Tuning**
   - Fine-tune ESM-2 specifically for isolate discrimination
   - Could improve conserved segment embeddings

3. **Genome Foundation Models**
   - Nucleotide-level models may capture isolate-specific signals

---

## Files Referenced

- `results/bunya/April_2025/bunya/training_analysis/metrics.csv`
- `results/flu/July_2025/flu_ha_na_5ks/training_analysis/metrics.csv`
- `results/flu/July_2025/flu_pb2_ha_na_5ks/training_analysis/metrics.csv`
- `results/flu/July_2025/flu_pb2_pb1_pa_5ks/training_analysis/metrics.csv`
- `results/flu/July_2025/flu_overfit_5ks/training_analysis/metrics.csv`
- Segment performance: `results/{virus}/{data_version}/{config_bundle}/training_analysis/segment_performance.csv` (contains F1 scores, AUC, accuracy per segment pair)
- Confusion matrices: `results/{virus}/{data_version}/{config_bundle}/training_analysis/confusion_matrix.csv`
- Learning curves: `models/{virus}/{data_version}/runs/training_*/learning_curves.png`
- Performance plots: `results/{virus}/{data_version}/{config_bundle}/training_analysis/performance_summary.png` (visualizes segment-level F1 scores)

---

## Conclusion

The experiments **definitively confirm the conservation hypothesis**:

1. **Variable segments (HA-NA) achieve 0.923 accuracy, 0.916 F1** - Actually outperforms Bunya (accuracy=0.910, F1=0.911) with 5K isolates
2. **Conserved segments (PB2-PB1-PA) achieve 0.719 accuracy, 0.753 F1** - Biological limitation, but F1 suggests some recoverable signal
3. **Segment-specific models might be the path forward** - Use HA-NA model for variable segments (F1=0.916)
4. **Pipeline is sound** - Both Bunya and Flu HA-NA prove the approach works well for variable segments

---

## Related Documentation

### Technical Documentation (`docs/`)
- **Project Status**: See [EXP_RESULTS_STATUS.md](./EXP_RESULTS_STATUS.md) for research status and roadmap
- **Configuration Guide**: See [CONFIGURATION_GUIDE.md](./CONFIGURATION_GUIDE.md) for detailed configuration documentation

### User Guides (`documentation/`)
- **Quick Start**: See [`../documentation/quick-start.md`](../documentation/quick-start.md) to get started quickly
- **Results Analysis**: See [`../documentation/analysis/results-analysis.md`](../documentation/analysis/results-analysis.md) for user guide on interpreting results

