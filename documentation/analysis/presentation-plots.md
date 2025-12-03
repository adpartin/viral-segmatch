# Presentation Plots

Creating publication-ready visualizations for the viral-segmatch project.

## 🎨 Overview

The `create_presentation_plots.py` script generates clean, professional visualizations specifically designed for presentations, publications, and stakeholder reports.

## 📊 Plot Types

### 1. Performance Summary Plot
**File**: `performance_summary.png`

**Layout**: 2x2 grid showing key performance metrics

**Panels**:
- **Top Left**: Overall metrics bar chart (Accuracy, F1, AUC, Avg Precision) - reads from `metrics.csv`
- **Top Right**: Performance by segment pair (cross-function pairs only)
- **Bottom Left**: Pair classification by type (positive, same-function negative, different-function negative)
- **Bottom Right**: Prediction confidence distribution

**Usage**:
```bash
python src/analysis/create_presentation_plots.py --config_bundle flu_ha_na_5ks
```

### 2. Biological Insights Plot
**File**: `biological_insights.png`

**Layout**: Dynamic (1x1 or 1x2) based on data availability

**Panels**:
- **Left**: Classification errors by segment pair
- **Right**: Same-function negative performance by protein type (if same-function negatives exist)

**Key Features**:
- Focuses on cross-function pairs
- Shows error patterns
- Highlights protein-specific performance
- Dynamically adjusts subplots based on data

### 3. Learning Curves Plot
**File**: `learning_curves.png` (if training history available)

**Layout**: Combined training history

**Panels**:
- Training and validation loss
- Training and validation F1 score
- Learning rate schedule

**Note**: Also creates separate `loss.png` and `f1.png` files.

## 🎯 Design Principles

### Visual Style
- **Clean aesthetics**: Minimal clutter, professional appearance
- **Consistent colors**: Harmonious color palette
- **Clear typography**: Readable fonts and sizes
- **Proper spacing**: Well-balanced layouts

### Data Presentation
- **Value labels**: Exact values on bars and points
- **Error bars**: Confidence intervals where appropriate
- **Trend lines**: Clear relationships between variables
- **Annotations**: Key insights highlighted

### Accessibility
- **High contrast**: Clear distinction between elements
- **Large fonts**: Readable from presentation distance
- **Color-blind friendly**: Alternative to color-only encoding
- **Clear legends**: Self-explanatory labels

## 🔧 Customization

### 1. Modify Plot Styling
```python
# In create_presentation_plots.py
def create_performance_summary_plot():
    # Change color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Custom colors
    
    # Modify figure size
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))  # Larger
    
    # Custom styling
    ax1.set_title('Custom Performance Metrics', fontsize=16, fontweight='bold')
```

### 2. Add Custom Metrics
```python
# Add new metrics to performance summary
def create_performance_summary_plot():
    # ... existing code ...
    
    # Add custom metric
    metrics = ['Accuracy', 'F1 Score', 'AUC-ROC', 'Avg Precision', 'Balanced Accuracy']
    values = [0.932, 0.876, 0.955, 0.819, 0.901]  # Add new value
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD']  # Add new color
```

## 📈 Plot Interpretation

### Performance Summary
- **Overall Metrics**: Quick performance overview (from `metrics.csv`)
- **Segment Pairs**: Which protein combinations work best
- **Pair Classification**: Positive vs same-function vs different-function negatives
- **Confidence Distribution**: Model certainty patterns

### Biological Insights
- **Error Patterns**: Which segment pairs cause most errors
- **Function Performance**: How well each protein type is classified
- **Biological Relevance**: Insights into protein function relationships

### Learning Curves
- **Training History**: Model learning over epochs
- **Loss Curves**: Training and validation loss
- **F1 Curves**: F1 score progression
- **Learning Rate**: LR schedule over training

## 🎨 Styling Options

### Color Palettes
```python
# Professional palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Scientific palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Custom palette
colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60']
```

### Figure Sizes
```python
# Standard presentation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Large presentation
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Publication quality
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
```

### Resolution Settings
```python
# High resolution for publications
plt.savefig(results_dir / 'performance_summary.png', 
           dpi=300, bbox_inches='tight')

# Standard resolution for presentations
plt.savefig(results_dir / 'performance_summary.png', 
           dpi=150, bbox_inches='tight')
```

## 🔍 Troubleshooting Plots

### Common Issues

**1. Empty Plots**
```python
# Check data availability
print(f"Data shape: {df.shape}")
print(f"Segment pairs: {df['seg_pair'].unique()}")
print(f"Functions: {df['func_a'].unique()}")
```

**2. Missing Values**
```python
# Handle missing data
df = df.dropna(subset=['f1_score'])
df = df[df['count'] > 5]  # Filter low-count pairs
```

**3. Plotting Errors**
```python
# Check data types
print(df.dtypes)
print(df['f1_score'].min(), df['f1_score'].max())

# Handle edge cases
if len(df) == 0:
    print("No data to plot")
    return
```

### Debugging Tips

**1. Verify Input Data**
```python
# Check metrics file
import pandas as pd
from pathlib import Path

results_dir = Path('results/flu/July_2025/flu_ha_na_5ks/training_analysis')
metrics = pd.read_csv(results_dir / 'metrics.csv')
print(f"Metrics: {metrics.columns.tolist()}")
```

**2. Test Individual Plots**
```python
# Test each plot function separately
create_performance_summary_plot()
create_biological_insights_plot()
```

**3. Check File Permissions**
```python
# Verify output directory
print(f"Results directory: {results_dir}")
print(f"Directory exists: {results_dir.exists()}")
print(f"Directory writable: {os.access(results_dir, os.W_OK)}")
```

## 📝 Best Practices

### 1. Consistent Styling
- Use same color palette across all plots
- Maintain consistent font sizes
- Keep consistent figure dimensions

### 2. Clear Annotations
- Add value labels to bars and points
- Include sample sizes where relevant
- Highlight key insights

### 3. Professional Quality
- High resolution for publications
- Clean, uncluttered layouts
- Clear, descriptive titles

### 4. Documentation
- Save plot parameters
- Document any customizations
- Keep plot generation scripts

## 📚 Related Documentation

- **[Results Analysis](results-analysis.md)** - Understanding analysis outputs
- **[Detailed Experiment Analysis](../../docs/EXPERIMENT_RESULTS_ANALYSIS.md)** - Comprehensive results
- **[Project Status](../../docs/EXP_RESULTS_STATUS.md)** - Research status
