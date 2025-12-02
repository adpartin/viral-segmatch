# Learning Verification Guide

## Purpose

This document describes the learning verification approach used to ensure the model actually learns from data, following [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/) for training neural networks.

## Why This Matters

Neural network training can fail silently - everything may be syntactically correct, but the model might not be learning anything useful. Common issues include:
- Data preprocessing bugs (e.g., flipped labels)
- Incorrect loss computation
- Model initialization problems
- Data leakage or other logical errors

## Verification Strategy

### 1. Small Dataset Test (`flu_a_learning_test.yaml`)

**Configuration**: 100 isolates, 10 epochs, patience=3

**Rationale**: 
- If the model learns, it should **overfit** a tiny dataset quickly
- This is a classic sanity check: if you can't overfit a small dataset, something is fundamentally wrong
- Very few epochs help detect if the model is just producing random predictions

**Important Note**: 
- The model **might learn well** from 100 isolates - this is actually a good sign!
- The goal is to verify learning, not necessarily to see poor generalization
- If the model performs well on both train and validation with 100 isolates, that's fine - it just means the task is learnable with limited data

**What to Look For**:
- Training loss should decrease from epoch 1
- Validation F1 should improve (even if slightly) before early stopping
- Model should beat baseline metrics (random classifier, majority class)

**Overfitting Pattern** (if dataset is too small):
- With many epochs and a very small training set, you may see:
  - **Training loss drops very low** (model memorizes training data)
  - **Validation loss stays high** (model doesn't generalize)
  - This is the classic overfitting pattern and confirms the model is learning (just not generalizing)

### 2. Initialization Loss Check

**Karpathy's Recommendation**: Verify loss at initialization

**Expected Behavior**:
- For balanced binary classification: initial loss ≈ -log(0.5) ≈ 0.693
- If loss is way off, indicates data imbalance or model setup issues

**Implementation**: `check_initialization_loss()` function runs before training starts

### 3. Baseline Comparison

**Baselines Computed**:
- **Random classifier**: Predicts randomly (50% accuracy for balanced data)
- **Majority class classifier**: Always predicts the most common class

**Success Criteria**: Model should beat both baselines. If it doesn't, the model isn't learning.

**Implementation**: `compute_baseline_metrics()` function

### 4. Learning Curves Visualization

**What's Plotted**:
- **Train/Val Loss**: Should see training loss decrease; validation loss may increase if overfitting
- **Validation F1**: Should improve over epochs (indicates learning)
- **Validation AUC**: Should improve over epochs

**Implementation**: `plot_learning_curves()` function saves `learning_curves.png` to output directory

**Plot Location**: Plots are saved per **epoch** (not per step/batch) for clarity

## Usage

### Run Learning Verification Test

```bash
# 1. Create dataset with 100 isolates
python src/datasets/dataset_segment_pairs.py --config_bundle flu_a_learning_test

# 2. Train with learning verification checks
python src/models/train_esm2_frozen_pair_classifier.py \
    --config_bundle flu_a_learning_test \
    --cuda_name cuda:0
```

### Expected Output

The training script will print:

1. **Initialization Loss Check**:
   ```
   ============================================================
   INITIALIZATION LOSS CHECK (Karpathy-style)
   ============================================================
   Initial loss: 0.6931
   Expected loss (balanced binary): ~0.6931
   ✅ Initialization loss looks reasonable
   ============================================================
   ```

2. **Baseline Metrics**:
   ```
   ============================================================
   BASELINE METRICS (for comparison)
   ============================================================
   Random classifier F1: 0.4500
   Majority class (1) F1: 0.6000
   Majority class accuracy: 0.6500
   Class balance: 650 positive, 350 negative (65.00%)
   ============================================================
   ```

3. **Training Progress** (per epoch):
   ```
   Epoch 1: Train Loss: 0.6931, Val Loss: 0.6920, Val F1: 0.5000, Val AUC: 0.5000 [F1: 0.5000]
   Epoch 2: Train Loss: 0.6500, Val Loss: 0.6800, Val F1: 0.5500, Val AUC: 0.6000 [F1: 0.5500]
   ...
   ```

4. **Learning Summary** (at end):
   ```
   ============================================================
   LEARNING SUMMARY
   ============================================================
   Initial train loss: 0.6931
   Final train loss: 0.4500
   Initial val F1: 0.5000
   Best val F1: 0.7500
   Baseline (majority class) F1: 0.6000
   ✅ Model learned! (F1 > baseline)
   ============================================================
   ```

5. **Learning Curves Plot**: Saved to `output_dir/learning_curves.png`

## Interpreting Results

### ✅ Good Signs (Model is Learning)

- Initialization loss is reasonable (~0.693 for balanced data)
- Training loss decreases consistently
- Validation F1 improves over epochs
- Model beats baseline metrics
- Learning curves show clear improvement

**Note**: Even if validation loss stays high while training loss drops low, this indicates learning (overfitting), which is better than no learning at all.

### ⚠️ Warning Signs (Model May Not Be Learning)

- Initialization loss is way off expected value
- Training loss doesn't decrease (or increases)
- Validation F1 stays flat or decreases
- Model doesn't beat majority class baseline
- Learning curves show no improvement

### 🔴 Red Flags (Something is Wrong)

- Validation F1 < 0.5 (worse than random)
- Training loss increases dramatically
- All predictions are the same class
- Model performance identical to random baseline

### 📊 Overfitting Pattern (Expected with Small Dataset)

When training with a very small dataset (e.g., 100 isolates) and many epochs:
- **Training loss**: Drops very low (model memorizes training examples)
- **Validation loss**: Stays high or increases (poor generalization)
- **Training F1**: High (model fits training data well)
- **Validation F1**: Lower than training F1 (model doesn't generalize)

**This is actually a good sign** - it confirms the model is learning! The solution is to:
1. Use more training data
2. Add regularization (dropout, weight decay)
3. Reduce model complexity
4. Use early stopping (already implemented)

## Next Steps After Verification

Once learning is verified on the small dataset:

1. **Scale up gradually**: Increase dataset size (100 → 500 → 1000 → full)
2. **Monitor for overfitting**: As dataset grows, validation metrics should improve
3. **Regularize if needed**: If overfitting occurs, add regularization (dropout, weight decay)
4. **Hyperparameter tuning**: Once learning is confirmed, tune learning rate, architecture, etc.

## References

- [Karpathy's Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
- Key principle: "Build from simple to complex and at every step validate with experiments"

