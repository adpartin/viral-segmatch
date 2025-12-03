# Model Improvement Recommendations

**Expert Analysis of `train_esm2_frozen_pair_classifier.py`**

**Status**: Many recommendations have been implemented. This document serves as a historical record and reference for future improvements.

**Note**: For current training implementation details, see the source code in `src/models/train_esm2_frozen_pair_classifier.py`. For experiment results, see [`../docs/EXPERIMENT_RESULTS_ANALYSIS.md`](../docs/EXPERIMENT_RESULTS_ANALYSIS.md).

---

## ✅ Implemented Improvements

### 1. **Learning Rate Scheduling** ✅
- **Status**: Implemented
- **Implementation**: Uses `create_lr_scheduler()` from `src/utils/torch_utils.py`
- **Supported Types**: ReduceLROnPlateau, CosineAnnealingLR, StepLR
- **Configuration**: `use_lr_scheduler`, `lr_scheduler`, `lr_scheduler_patience`, etc.

### 2. **Threshold Optimization** ✅
- **Status**: Implemented
- **Implementation**: `find_optimal_threshold_pr()` function
- **Supported Metrics**: F1, F0.5, F2
- **Configuration**: `threshold_metric` in training config

### 3. **Early Stopping on F1** ✅
- **Status**: Implemented
- **Implementation**: Configurable `early_stopping_metric` (loss, f1, auc)
- **Configuration**: `early_stopping_metric: 'f1'`

### 4. **Interaction Features (use_diff, use_prod)** ✅
- **Status**: Implemented
- **Implementation**: Element-wise difference and product features
- **Configuration**: `use_diff: true`, `use_prod: true` in training config
- **Note**: Currently set to `false` in most configs; can be enabled for experiments

### 5. **Training History Logging** ✅
- **Status**: Implemented
- **Implementation**: Saves `training_history.csv` with metrics per epoch
- **Includes**: Loss, F1, AUC, learning rate per epoch

### 6. **Separate Learning Curve Plots** ✅
- **Status**: Implemented
- **Implementation**: Creates `loss.png` and `f1.png` in addition to `learning_curves.png`

---

## 🐛 Critical Issues (Historical)

### 1. **AUC Calculation Bug** ✅ FIXED
**Issue**: Validation AUC was calculated using binary predictions instead of probabilities.

**Status**: Fixed in current implementation - uses probabilities correctly.

### 2. **Early Stopping Mismatch** ✅ FIXED
**Issue**: Model selection used validation loss, but F1 score was the primary metric.

**Status**: Fixed - now uses configurable `early_stopping_metric` (can be 'f1', 'loss', 'auc').

### 3. **Fixed Threshold for F1 Score** ✅ FIXED
**Issue**: F1 score used hardcoded threshold of 0.5.

**Status**: Fixed - now uses `find_optimal_threshold_pr()` to optimize threshold.

---

## 🚀 Remaining Improvements (Ranked by Difficulty)

### ⚡ **EASY** (1-2 hours each)

#### 1. **Add Gradient Clipping**
- **Difficulty**: ⭐ Easy (15 minutes)
- **Impact**: Medium (training stability)
- **Status**: Not yet implemented
- **Implementation**:
  ```python
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()
  ```
- **Expected Benefit**: More stable training, prevents exploding gradients

---

### ⚙️ **MEDIUM** (3-8 hours each)

#### 2. **Add Class Weighting to Loss Function**
- **Difficulty**: ⭐⭐ Medium (1 hour)
- **Impact**: Medium (if class imbalance exists)
- **Status**: Not yet implemented
- **Implementation**:
  ```python
  pos_weight = torch.tensor(len(negatives) / len(positives))
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  ```
- **Expected Benefit**: Better handling of class imbalance

#### 3. **Cache Embeddings in Memory**
- **Difficulty**: ⭐⭐ Medium (3 hours)
- **Impact**: Medium (significantly faster training)
- **Status**: Partially implemented (shared cache across splits)
- **Note**: Current implementation uses class-level cache; could be improved

---

### 🔧 **MEDIUM-HARD** (1-2 days)

#### 4. **Add Batch Normalization**
- **Difficulty**: ⭐⭐⭐ Medium-Hard (4 hours)
- **Impact**: Medium-High (can improve convergence)
- **Status**: Not yet implemented
- **Implementation**: Add `nn.BatchNorm1d` after each linear layer
- **Expected Benefit**: Faster convergence, potentially 1-2% F1 improvement

#### 5. **Implement Attention Mechanism for Pair Comparison**
- **Difficulty**: ⭐⭐⭐ Medium-Hard (1 day)
- **Impact**: High (captures complex interactions)
- **Status**: Not yet implemented
- **Implementation**: Cross-attention between protein embeddings
- **Expected Benefit**: 2-5% F1 improvement by learning pairwise interactions

#### 6. **Add Label Smoothing**
- **Difficulty**: ⭐⭐⭐ Medium-Hard (2 hours)
- **Impact**: Medium (regularization)
- **Status**: Not yet implemented
- **Implementation**: Modify loss to use soft labels (0.9 instead of 1.0)
- **Expected Benefit**: Better generalization, prevents overconfidence

---

### 🏗️ **HARD** (2-5 days)

#### 7. **Implement Siamese Network Architecture**
- **Difficulty**: ⭐⭐⭐⭐ Hard (3-5 days)
- **Impact**: **Very High** (biologically more appropriate)
- **Status**: Not yet implemented
- **Implementation**: Learn distance/similarity metric between embeddings
- **Why**: More appropriate for pairwise comparison tasks
- **Expected Benefit**: 5-10% F1 improvement potential

#### 8. **Add Contrastive Loss**
- **Difficulty**: ⭐⭐⭐⭐ Hard (2-3 days)
- **Impact**: High (better representation learning)
- **Status**: Not yet implemented
- **Implementation**: Replace BCE loss with contrastive loss
- **Expected Benefit**: 3-7% F1 improvement

---

## 📊 Current Performance Baseline

Based on latest experiments (December 2025):

- **Flu HA-NA (5K isolates)**: 92.3% accuracy, **91.6% F1**, 0.953 AUC
- **Flu PB2-PB1-PA (5K isolates)**: 71.9% accuracy, **75.3% F1**, 0.750 AUC
- **Bunya (full dataset)**: 91.0% accuracy, **91.1% F1**, 0.927 AUC

*For detailed results, see [`../docs/EXPERIMENT_RESULTS_ANALYSIS.md`](../docs/EXPERIMENT_RESULTS_ANALYSIS.md)*

---

## 💡 Key Insights

1. **Element-wise operations** (difference, product) are implemented but not enabled by default
2. **Early stopping on F1** is now configurable and working
3. **Threshold optimization** provides immediate gains without retraining
4. **Architecture changes** (Siamese, attention) have highest potential but require more effort
5. **Conservation hypothesis confirmed**: Variable segments (HA-NA) significantly outperform conserved segments (PB2-PB1-PA)

---

## 📝 Notes

- All improvements are additive - you can implement multiple changes together
- Test each change incrementally to measure individual impact
- Consider biological constraints when implementing domain-specific improvements
- Monitor for overfitting when adding capacity (attention, larger networks)

---

## 🔬 Experimental Protocol

For each improvement:
1. Run baseline model (current implementation)
2. Implement one change
3. Train with same hyperparameters
4. Compare F1, AUC, and loss
5. Document results

This allows you to measure the impact of each change independently.

---

**Last Updated**: December 3, 2025  
**Status**: Many recommendations implemented; document serves as reference for future work

**Related Documentation**:
- **[Experiment Results](../docs/EXPERIMENT_RESULTS_ANALYSIS.md)** - Current experiment results
- **[Project Status](../docs/EXP_RESULTS_STATUS.md)** - Research status and roadmap
