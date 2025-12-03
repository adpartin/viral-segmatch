# Results Analysis

Understanding and interpreting the output from the viral-segmatch pipeline.

**Note**: For detailed experiment results and analysis, see [`../../docs/EXPERIMENT_RESULTS_ANALYSIS.md`](../../docs/EXPERIMENT_RESULTS_ANALYSIS.md).

## 📊 Analysis Scripts

### 1. Comprehensive Analysis
**Script**: `src/analysis/analyze_stage4_train.py`

**Purpose**: Detailed ML analysis with extensive metrics and domain-specific insights.

**Usage**:
```bash
# Basic analysis (automatically finds latest training run)
python src/analysis/analyze_stage4_train.py --config_bundle flu_ha_na_5ks

# With explicit model directory
python src/analysis/analyze_stage4_train.py \
  --config_bundle flu_ha_na_5ks \
  --model_dir models/flu/July_2025/runs/training_flu_ha_na_5ks_YYYYMMDD_HHMMSS
```

**Outputs**:
- `confusion_matrix.png` - Classification confusion matrix
- `roc_curve.png` - ROC curve with AUC
- `precision_recall_curve.png` - Precision-recall curve
- `learning_curves.png` - Training history plots
- `loss.png` - Loss curves
- `f1.png` - F1 score curves
- `metrics.csv` - Summary metrics
- `confusion_matrix.csv` - Confusion matrix data
- `confusion_matrix_summary.csv` - TP/TN/FP/FN summary
- `error_analysis_summary.csv` - Error statistics

### 2. Presentation Plots
**Script**: `src/analysis/create_presentation_plots.py`

**Purpose**: Clean, publication-ready visualizations for presentations.

**Usage**:
```bash
# Generate presentation plots
python src/analysis/create_presentation_plots.py --config_bundle flu_ha_na_5ks
```

**Outputs**:
- `performance_summary.png` - 4-panel performance overview
- `biological_insights.png` - Biological interpretation plots
- `learning_curves.png` - Training history (if available)

## 📈 Key Metrics

### Classification Performance
- **Accuracy**: Overall correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **Average Precision**: Area under precision-recall curve

### Error Analysis
- **False Positives (FP)**: Incorrectly predicted as same isolate
- **False Negatives (FN)**: Incorrectly predicted as different isolate
- **FP/FN Ratio**: Balance between error types
- **Confidence Analysis**: Model certainty in predictions

### Domain-Specific Metrics
- **Segment-wise Performance**: Performance by protein segment pair
- **Function Analysis**: Performance by protein function
- **Same-function Negatives**: Hardest cases to classify

## 🔍 Understanding Results

### 1. Confusion Matrix
```
                Predicted
Actual    Negative  Positive
Negative     TN       FP
Positive     FN       TP
```

**Interpretation**:
- **TN (True Negatives)**: Correctly identified different isolates
- **TP (True Positives)**: Correctly identified same isolates
- **FP (False Positives)**: Incorrectly predicted same isolate
- **FN (False Negatives)**: Incorrectly predicted different isolate

### 2. ROC Curve
- **X-axis**: False Positive Rate (1 - Specificity)
- **Y-axis**: True Positive Rate (Sensitivity)
- **AUC**: Area under curve (0.5 = random, 1.0 = perfect)

### 3. Precision-Recall Curve
- **X-axis**: Recall (Sensitivity)
- **Y-axis**: Precision (Positive Predictive Value)
- **AP**: Average Precision (area under curve)

### 4. Learning Curves
- **Training Loss**: Model performance on training data
- **Validation Loss**: Model performance on validation data
- **F1 Score**: F1 score over epochs
- **Learning Rate**: Learning rate schedule

## 📊 Current Performance Results

### Key Finding: Conservation Hypothesis Confirmed ✅

Performance directly correlates with protein conservation levels:
- **Variable segments (HA-NA)**: 92.3% accuracy, **91.6% F1**, 0.953 AUC
- **Mixed segments (PB2-HA-NA)**: 85.4% accuracy, **85.5% F1**, 0.920 AUC  
- **Conserved segments (PB2-PB1-PA)**: 71.9% accuracy, **75.3% F1**, 0.750 AUC

*For detailed analysis, see [`../../docs/EXPERIMENT_RESULTS_ANALYSIS.md`](../../docs/EXPERIMENT_RESULTS_ANALYSIS.md)*

## 📁 Output File Structure

```
results/
└── {virus}/
    └── {data_version}/
        └── {config_bundle}/
            ├── training_analysis/          # Comprehensive analysis
            │   ├── confusion_matrix.png
            │   ├── roc_curve.png
            │   ├── precision_recall_curve.png
            │   ├── learning_curves.png
            │   ├── loss.png
            │   ├── f1.png
            │   ├── metrics.csv
            │   ├── confusion_matrix.csv
            │   └── error_analysis_summary.csv
            └── presentation_plots/         # Publication-ready plots
                ├── performance_summary.png
                └── biological_insights.png
```

## 🔧 Customizing Analysis

### 1. Modify Analysis Parameters
```python
# In analyze_stage4_train.py
def main(config_bundle: str,
         model_dir: str = None,
         create_performance_plots: bool = True,  # Toggle plots
         show_confusion_labels: bool = True      # Toggle labels
    ):
```

### 2. Add Custom Metrics
```python
# Add to compute_basic_metrics function
def compute_basic_metrics(y_true, y_pred, y_prob):
    # ... existing code ...
    
    # Add custom metric
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_roc': auc,
        'avg_precision': avg_precision,
        'balanced_accuracy': balanced_accuracy  # New metric
    }
```

## 📊 Interpreting Results

### Good Performance Indicators
- **AUC > 0.8**: Good discrimination ability
- **F1 > 0.7**: Good balance of precision and recall
- **Accuracy > 0.8**: High overall correctness
- **Low FP/FN ratio**: Balanced error types

### Red Flags
- **AUC < 0.6**: Poor discrimination (worse than random)
- **F1 < 0.5**: Poor precision-recall balance
- **Very high FP**: Too many false positives
- **Very high FN**: Too many false negatives

### Domain-specific Considerations
- **Same-function negatives**: Should be hardest cases
- **Cross-function pairs**: Should show different performance patterns
- **Function-specific errors**: May indicate data quality issues

## 🔍 Debugging Analysis

### 1. Check Input Data
```python
# Verify test predictions file
import pandas as pd
from pathlib import Path

model_dir = Path('models/flu/July_2025/runs/training_flu_ha_na_5ks_YYYYMMDD_HHMMSS')
df = pd.read_csv(model_dir / 'test_predicted.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution: {df['label'].value_counts()}")
```

### 2. Validate Predictions
```python
# Check prediction ranges
print(f"Prediction probabilities: {df['pred_prob'].min():.3f} - {df['pred_prob'].max():.3f}")
print(f"Prediction labels: {df['pred_label'].unique()}")
```

### 3. Verify Results Directory
```python
# Check output directory
from pathlib import Path
results_dir = Path('results/flu/July_2025/flu_ha_na_5ks/training_analysis')
print(f"Results directory exists: {results_dir.exists()}")
print(f"Files in results: {list(results_dir.glob('*.png'))}")
```

## 📝 Best Practices

### 1. Document Analysis
- Save analysis parameters
- Document any custom modifications
- Keep analysis logs

### 2. Compare Results
- Compare across different experiments
- Track performance over time
- Document improvements

### 3. Validate Results
- Check for data quality issues
- Verify biological plausibility
- Cross-reference with domain knowledge

## 📚 Related Documentation

- **[Detailed Experiment Analysis](../../docs/EXPERIMENT_RESULTS_ANALYSIS.md)** - Comprehensive results analysis
- **[Project Status](../../docs/EXP_RESULTS_STATUS.md)** - Research status and roadmap
- **[Presentation Plots](presentation-plots.md)** - Creating publication-ready plots
