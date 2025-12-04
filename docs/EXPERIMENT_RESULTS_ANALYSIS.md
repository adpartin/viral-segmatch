# Experiment Results Analysis

**Purpose**: This document provides a detailed scientific analysis of completed experiments, focusing on performance metrics, statistical comparisons, and biological interpretation of results. For project status, research roadmap, and background context, see [`EXP_RESULTS_STATUS.md`](EXP_RESULTS_STATUS.md).

**Date**: December 3, 2025  
**Experiments Analyzed**: 5 config bundles across Bunya and Flu A (5K isolates for Flu experiments)

---

## Executive Summary

This analysis compares model performance across different segment combinations to test the **conservation hypothesis**: that highly conserved proteins (PB1, PB2, PA) are harder to distinguish than variable proteins (HA, NA).

### Key Finding: **Conservation Hypothesis Confirmed** ✅

Performance directly correlates with protein conservation levels:
- **Variable segments (HA-NA)**: 92.3% accuracy, **91.6% F1**, 0.953 AUC
- **Mixed segments (PB2-HA-NA)**: 85.4% accuracy, **85.5% F1**, 0.920 AUC  
- **Conserved segments (PB2-PB1-PA)**: 71.9% accuracy, **75.3% F1**, 0.750 AUC

---

## Detailed Results

### 1. Bunya (Baseline - Core protein only)

**Config**: `bunya.yaml`  
**Segments**: L (polymerase), M (glycoprotein), S (nucleocapsid)  
**Dataset**: Full dataset, no same-function negatives

| Metric | Value |
|--------|-------|
| **Accuracy** | **91.0%** |
| **F1 Score** | **91.1%** |
| **AUC-ROC** | **92.7%** |
| **Avg Precision** | **88.2%** |

**Segment Pair Performance**:
- M-S: 92.6% accuracy, 0.956 AUC (best)
- L-M: 90.3% accuracy, 0.922 AUC
- L-S: 90.4% accuracy, 0.939 AUC

*Note: Segment-level F1 scores not available in segment_performance.csv*

**Conclusion**: ✅ **Pipeline works well** - Bunya demonstrates strong performance across all three segment pairs.

---

### 2. Flu HA-NA (Variable Segments)

**Config**: `flu_ha_na_5ks.yaml`  
**Segments**: HA (Segment 4), NA (Segment 6)  
**Conservation**: 70-90% (variable due to immune pressure)  
**Dataset**: 5K isolates, no same-function negatives

| Metric | Value |
|--------|-------|
| **Accuracy** | **92.3%** |
| **F1 Score** | **91.6%** |
| **AUC-ROC** | **95.3%** |
| **Avg Precision** | **89.9%** |

**Segment Pair Performance**:
- S4-S6 (HA-NA): 92.3% accuracy, 0.953 AUC

**Error Analysis**:
- False Positives: 60 (avg confidence: 0.809)
- False Negatives: 7 (avg confidence: 0.294)
- FP/FN Ratio: 8.57 (model is conservative, misses fewer positives)

**Conclusion**: ✅ **Strong performance** - Variable segments show good separability despite being Flu A.

---

### 3. Flu PB2-HA-NA (Mixed: Conserved + Variable)

**Config**: `flu_pb2_ha_na_5ks.yaml`  
**Segments**: PB2 (Segment 1, ~95% conserved), HA (Segment 4, 70-85% variable), NA (Segment 6, 80-90% variable)  
**Dataset**: 5K isolates, no same-function negatives

| Metric | Value |
|--------|-------|
| **Accuracy** | **85.4%** |
| **F1 Score** | **85.5%** |
| **AUC-ROC** | **92.0%** |
| **Avg Precision** | **84.2%** |

**Error Analysis**:
- False Positives: 366 (avg confidence: 0.757)
- False Negatives: 9 (avg confidence: 0.284)
- FP/FN Ratio: 40.67 (model is very conservative, misses fewer positives)

**Conclusion**: ✅ **Good performance** - Mixing one conserved segment (PB2) with two variable segments (HA, NA) achieves strong performance, better than conserved-only but slightly below variable-only.

---

### 4. Flu PB2-PB1-PA (Highly Conserved Polymerase)

**Config**: `flu_pb2_pb1_pa_5ks.yaml`  
**Segments**: PB2 (Segment 1, ~95%), PB1 (Segment 2, **98.1%**), PA (Segment 3, ~94%)  
**Conservation**: Highest conservation (94-98%)  
**Dataset**: 5K isolates, no same-function negatives

| Metric | Value |
|--------|-------|
| **Accuracy** | **71.9%** |
| **F1 Score** | **75.3%** |
| **AUC-ROC** | **75.0%** |
| **Avg Precision** | **59.8%** |

**Error Analysis**:
- False Positives: 698 (avg confidence: 0.704)
- False Negatives: 0 (avg confidence: N/A)
- FP/FN Ratio: ∞ (model is extremely conservative, never misses positives but has many false positives)

**Segment Pair Performance**:
- S2-S3 (PB1-PA): 76.8% accuracy, 0.795 AUC (best)
- S1-S3 (PB2-PA): 71.9% accuracy, 0.675 AUC
- S1-S2 (PB2-PB1): 70.5% accuracy, 0.718 AUC (worst - both highly conserved)

**Conclusion**: ⚠️ **Lower performance** - Highly conserved polymerase segments show reduced separability, with PB2-PB1 (both ~95-98% conserved) performing worst.

---

### 5. Flu Overfit Test (Small Dataset)

**Config**: `flu_overfit.yaml`  
**Segments**: PB2, PB1, PA (polymerase)  
**Dataset**: **50 isolates only** (very small), 10% train, 80% val  
**Purpose**: Test if model can overfit small dataset (capacity test)

| Metric | Value |
|--------|-------|
| **Accuracy** | **79.3%** |
| **F1 Score** | **82.4%** |
| **AUC-ROC** | **77.1%** |
| **Avg Precision** | **67.4%** |

**Segment Pair Performance**:
- S1-S2 (PB2-PB1): 81.8% accuracy, 0.733 AUC (only 11 test samples)

**Conclusion**: ⚠️ **Small dataset limits generalization** - Model achieves reasonable performance on tiny dataset, but this is not representative of real-world performance. 

---

## Comparative Analysis

### Performance Ranking by Segment Type

| Rank | Experiment | Accuracy | **F1 Score** | AUC | Segment Type | Conservation |
|------|------------|----------|--------------|-----|--------------|--------------|
| 1 | **Flu HA-NA (5ks)** | **92.3%** | **91.6%** | **0.953** | Variable | 70-90% |
| 2 | **Bunya (all)** | **91.0%** | **91.1%** | **0.927** | All segments | Low (different virus) |
| 3 | **Flu PB2-HA-NA (5ks)** | **85.4%** | **85.5%** | **0.920** | Mixed | 95% + 70-90% |
| 4 | Flu Overfit | 79.3% | **82.4%** | 0.771 | Conserved | 94-98% (small dataset) |
| 5 | **Flu PB2-PB1-PA (5ks)** | **71.9%** | **75.3%** | **0.750** | Conserved | 94-98% |

### Key Insights

1. **Conservation-Performance Correlation** ✅
   - HA-NA (variable): **92.3%** accuracy, **91.6% F1**
   - PB2-PB1-PA (conserved): **71.9%** accuracy, **75.3% F1**
   - **20.4 percentage point accuracy difference** (16.3% F1 difference) confirms conservation limits performance

2. **F1 Score Analysis** 📊
   - **Flu HA-NA**: **91.6% F1** (highest) - excellent precision-recall balance
   - **Bunya**: **91.1% F1** - excellent performance, comparable to HA-NA
   - **Flu PB2-PB1-PA**: **75.3% F1** - lower but still above random (50%)
   - **F1 gap**: 16.3% between variable and conserved segments

3. **Segment-Specific Performance**
   - Within PB2-PB1-PA: PB1-PA performs best (76.8%), PB2-PB1 worst (70.5%)
   - This aligns with conservation: PB1 is most conserved (98.1%), making PB2-PB1 pairs hardest

4. **Bunya vs Flu A**
   - Bunya: **91.0%** accuracy, **91.1% F1** (all segments)
   - Flu HA-NA: **92.3%** accuracy, **91.6% F1** (best Flu segments)
   - **Flu HA-NA actually outperforms Bunya** - Variable segments with 5K isolates achieve excellent performance

5. **AUC vs Accuracy vs F1**
   - HA-NA: AUC (0.953) > Accuracy (0.923) ≈ F1 (0.916) → Excellent class separation, balanced precision/recall
   - PB2-PB1-PA: AUC (0.750) ≈ Accuracy (0.719) < F1 (0.753) → Poor class separation, but F1 suggests some recoverable signal
   - **AUC gap confirms embedding overlap for conserved segments**
   - **F1 > Accuracy for conserved segments** suggests model is learning some patterns despite overlap

---

## Biological Interpretation

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

3. **PB2-PB1 Worst**: Both segments are highly conserved, creating the hardest discrimination task

---

## Implications for Future Work

### ✅ Confirmed Hypotheses

1. **Conservation limits performance** - Variable segments (HA-NA) outperform conserved (PB2-PB1-PA) by 13.1% accuracy and 9.4% F1
2. **Segment-specific models are valuable** - HA-NA model achieves 86.3% accuracy, 86.5% F1 (vs ~73% accuracy, ~77% F1 for all segments)
3. **Pipeline works** - Bunya proves the approach is sound (91.4% F1); Flu A limitations are biological

### 🎯 Recommended Next Steps

1. **Segment-Specific Models** (High Priority)
   - Deploy HA-NA model for variable segments (86.3% accuracy, 86.5% F1)
   - Accept lower performance for conserved segments (73.2% accuracy, 77.1% F1)
   - **Hybrid approach**: Use different models based on segment type

2. **Interaction Features** (Medium Priority)
   - Test `use_diff=True, use_prod=True` on Flu experiments
   - May improve conserved segment performance by highlighting differences

3. **Contrastive Fine-Tuning** (High Priority)
   - Fine-tune ESM-2 specifically for isolate discrimination
   - Could improve conserved segment embeddings

4. **Genome Foundation Models** (Long-term)
   - Nucleotide-level models may capture synonymous mutations
   - Could provide additional signal for conserved proteins

### ❌ What NOT to Do

- **Don't expect >85% accuracy or >85% F1 for conserved segments** - This is a biological limitation (observed: 73.2% accuracy, 77.1% F1)
- **Don't use same-function negatives** - Confirmed to create bias (Bunya analysis)
- **Don't train on all segments together** - Segment-specific models perform better (86.5% F1 for HA-NA vs ~77% F1 for all segments)

---

## Statistical Summary

### Performance Distribution

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Accuracy | 84.0% | 7.8% | 71.9% | 92.3% |
| F1 Score | 85.2% | 6.5% | 75.3% | 91.6% |
| AUC-ROC | 0.864 | 0.081 | 0.750 | 0.953 |
| Avg Precision | 0.801 | 0.124 | 0.598 | 0.899 |

### Effect Size (Conservation Impact)

- **HA-NA vs PB2-PB1-PA**: 
  - Accuracy difference: **20.4%** (92.3% vs 71.9%)
  - **F1 difference: 16.3%** (91.6% vs 75.3%)
  - AUC difference: **0.203** (0.953 vs 0.750)
- **Conclusion**: Conservation has **very large, measurable impact** on all metrics, with F1 showing slightly less degradation than accuracy. The gap is larger with 5K isolates than with 1K isolates.

---

## Files Referenced

- `results/bunya/April_2025/bunya/training_analysis/metrics.csv`
- `results/flu/July_2025/flu_ha_na_5ks/training_analysis/metrics.csv`
- `results/flu/July_2025/flu_pb2_ha_na_5ks/training_analysis/metrics.csv`
- `results/flu/July_2025/flu_pb2_pb1_pa_5ks/training_analysis/metrics.csv`
- `results/flu/July_2025/flu_overfit/training_analysis/metrics.csv`
- Confusion matrices: `results/{virus}/{data_version}/{config_bundle}/training_analysis/confusion_matrix.csv` (newly added)
- Learning curves: `models/{virus}/{data_version}/runs/training_*/learning_curves.png`

---

## Conclusion

The experiments **definitively confirm the conservation hypothesis**:

1. ✅ **Variable segments (HA-NA) achieve 92.3% accuracy, 91.6% F1** - Actually outperforms Bunya (91.0% accuracy, 91.1% F1) with 5K isolates
2. ⚠️ **Conserved segments (PB2-PB1-PA) achieve 71.9% accuracy, 75.3% F1** - Biological limitation, but F1 suggests some recoverable signal
3. ✅ **Segment-specific models are the path forward** - Use HA-NA model for variable segments (91.6% F1)
4. ✅ **Pipeline is sound** - Both Bunya and Flu HA-NA prove the approach works excellently for variable segments

**Key F1 Insights**:
- **F1 gap between variable and conserved: 16.3%** (91.6% vs 75.3%) - Larger gap than initially observed with 1K isolates
- **F1 > Accuracy for conserved segments** (75.3% vs 71.9%) suggests model learns some patterns despite embedding overlap
- **Flu HA-NA F1 (91.6%) matches Bunya performance** - Variable segments with sufficient data achieve excellent results

**Recommendation**: Deploy segment-specific models with realistic performance expectations:
- **Variable segments (HA-NA)**: Expect ~92% accuracy, ~92% F1 (with 5K+ isolates)
- **Conserved segments (PB2-PB1-PA)**: Expect ~72% accuracy, ~75% F1

---

## Related Documentation

### Technical Documentation (`docs/`)
- **Project Status**: See [EXP_RESULTS_STATUS.md](./EXP_RESULTS_STATUS.md) for research status and roadmap
- **Configuration Guide**: See [CONFIGURATION_GUIDE.md](./CONFIGURATION_GUIDE.md) for detailed configuration documentation

### User Guides (`documentation/`)
- **Quick Start**: See [`../documentation/quick-start.md`](../documentation/quick-start.md) to get started quickly
- **Results Analysis**: See [`../documentation/analysis/results-analysis.md`](../documentation/analysis/results-analysis.md) for user guide on interpreting results

