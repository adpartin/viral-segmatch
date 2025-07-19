"""
Create presentation-ready plots for the segment classifier results.

This script generates additional visualizations specifically designed for
presentation slides, focusing on the most important insights.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Config
VIRUS_NAME = 'bunya'
DATA_VERSION = 'April_2025'
TASK_NAME = 'segment_pair_classifier'

# Define paths
models_dir = project_root / 'models' / VIRUS_NAME / DATA_VERSION / TASK_NAME
results_dir = project_root / 'results' / VIRUS_NAME / DATA_VERSION / TASK_NAME

# Load data
test_results_file = models_dir / 'test_predicted.csv'
df = pd.read_csv(test_results_file)

# Set up plotting style for presentations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')


def create_performance_summary_plot():
    """Create a summary plot showing key metrics."""
    # breakpoint()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Overall metrics bar chart
    metrics = ['Accuracy', 'F1 Score', 'AUC-ROC', 'Avg Precision']
    values = [0.932, 0.876, 0.955, 0.819]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Segment pair performance (cross-function pairs only)
    segment_perf = pd.read_csv(results_dir / 'segment_performance.csv')
    cross_func = segment_perf[segment_perf['seg_pair'].str.contains('-')]
    cross_func = cross_func.dropna(subset=['f1_score'])

    bars = ax2.bar(cross_func['seg_pair'], cross_func['f1_score'], 
                   color='#4ECDC4', alpha=0.8)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Performance by Segment Pair', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)

    # Add value labels
    for bar, value in zip(bars, cross_func['f1_score']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 3. Same-function vs Different-function negative pairs
    same_func_neg = df[(df['label'] == 0) & (df['func_a'] == df['func_b'])]
    diff_func_neg = df[(df['label'] == 0) & (df['func_a'] != df['func_b'])]

    same_func_acc = (same_func_neg['label'] == same_func_neg['pred_label']).mean()
    diff_func_acc = (diff_func_neg['label'] == diff_func_neg['pred_label']).mean()
    
    categories = ['Same-Function\nNegatives', 'Different-Function\nNegatives']
    accuracies = [same_func_acc, diff_func_acc]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax3.bar(categories, accuracies, color=colors, alpha=0.8)
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Negative Pair Classification', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Prediction confidence distribution
    df['confidence'] = np.abs(df['pred_prob'] - 0.5)
    conf_bins = ['Very Low\n(0.4-0.6)', 'Low\n(0.3-0.4, 0.6-0.7)', 
                 'Medium\n(0.2-0.3, 0.7-0.8)', 'High\n(0.0-0.2, 0.8-1.0)']
    
    # Count predictions in each confidence bin
    very_low = len(df[df['confidence'] < 0.1])
    low = len(df[(df['confidence'] >= 0.1) & (df['confidence'] < 0.2)])
    medium = len(df[(df['confidence'] >= 0.2) & (df['confidence'] < 0.3)])
    high = len(df[df['confidence'] >= 0.3])
    
    counts = [very_low, low, medium, high]
    colors = ['#FF6B6B', '#FFE66D', '#4ECDC4', '#45B7D1']
    
    bars = ax4.bar(conf_bins, counts, color=colors, alpha=0.8)
    ax4.set_ylabel('Number of Predictions')
    ax4.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, value in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_biological_insights_plot():
    """Create plots highlighting biological insights."""
    # breakpoint()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Error analysis by segment pair
    df['seg_pair'] = df.apply(
        lambda row: f"{row['seg_a']}-{row['seg_b']}" 
        if row['seg_a'] <= row['seg_b'] 
        else f"{row['seg_b']}-{row['seg_a']}", axis=1
    )
    
    errors = df[df['label'] != df['pred_label']]
    error_counts = errors['seg_pair'].value_counts()
    
    # Only show cross-function pairs
    cross_func_errors = error_counts[error_counts.index.str.contains('-')]
    
    if len(cross_func_errors) > 0:
        bars = ax1.bar(cross_func_errors.index, cross_func_errors.values, 
                       color='#E74C3C', alpha=0.8)
        ax1.set_ylabel('Number of Errors')
        ax1.set_title('Classification Errors by Segment Pair', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, cross_func_errors.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Function-specific performance (for same-function negatives)
    same_func_neg = df[(df['label'] == 0) & (df['func_a'] == df['func_b'])]
    
    if len(same_func_neg) > 0:
        func_performance = []
        for func in same_func_neg['func_a'].unique():
            subset = same_func_neg[same_func_neg['func_a'] == func]
            if len(subset) > 5:
                acc = (subset['label'] == subset['pred_label']).mean()
                func_performance.append({
                    'function': func.split()[-1],  # Get last word for shorter labels
                    'accuracy': acc,
                    'count': len(subset)
                })
        
        if func_performance:
            func_df = pd.DataFrame(func_performance)
            bars = ax2.bar(func_df['function'], func_df['accuracy'], 
                          color='#3498DB', alpha=0.8)
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Same-Function Negative Performance\nby Protein Type', 
                         fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 1)
            
            # Add value labels
            for bar, value, count in zip(bars, func_df['accuracy'], func_df['count']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}\n(n={count})', ha='center', va='bottom', 
                        fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'biological_insights.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_model_calibration_plot():
    """Create a plot showing model calibration and confidence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Calibration plot (reliability diagram)
    y_true = df['label'].values
    y_prob = df['pred_prob'].values
    
    # Create bins for predicted probabilities
    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Select predictions in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
            bin_sizes.append(in_bin.sum())
    
    # Plot calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax1.plot(confidences, accuracies, 'bo-', linewidth=2, markersize=8, 
             label='Model Calibration')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Model Calibration (Reliability Diagram)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction probability histogram by true label
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    
    ax2.hist(neg_probs, bins=20, alpha=0.7, label='Negative (Different Isolate)', 
             color='#E74C3C', density=True)
    ax2.hist(pos_probs, bins=20, alpha=0.7, label='Positive (Same Isolate)', 
             color='#3498DB', density=True)
    ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, 
                label='Decision Threshold')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'model_calibration.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Generate all presentation plots."""    
    print('1. Performance summary plot.')
    create_performance_summary_plot()
    
    print('2. Biological insights plot.')
    create_biological_insights_plot()
    
    print('3. Model calibration plot.')
    create_model_calibration_plot()
    
    print(f'All plots saved to: {results_dir}')

if __name__ == '__main__':
    main() 