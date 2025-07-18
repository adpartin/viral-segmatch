"""
Analyze segment pair classifier results.

This script analyzes the test results from the trained ESM-2 frozen pair classifier,
providing both standard ML metrics and domain-specific insights for viral protein
segment classification.

Questions:
What's worse FP or FN?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, f1_score, roc_auc_score, average_precision_score
)

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
results_dir.mkdir(parents=True, exist_ok=True)

# Load test results
test_results_file = models_dir / 'test_predicted.csv'
if not test_results_file.exists():
    raise FileNotFoundError(f'Test results file not found: {test_results_file}')

df = pd.read_csv(test_results_file)
print(f'Loaded {len(df)} test predictions')

# Extract labels and predictions
y_true = df['label'].values
y_pred = df['pred_label'].values  
y_prob = df['pred_prob'].values

# Set up plotting style
plt.style.use('default')
sns.set_palette('husl')
fig_size = (10, 8)


def compute_basic_metrics(y_true, y_pred, y_prob):
    """Compute and display basic classification metrics."""
    print('\n' + '='*50)
    print('BASIC CLASSIFICATION METRICS')
    print('='*50)
    
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)
    accuracy = (y_true == y_pred).mean()
    avg_precision = average_precision_score(y_true, y_prob)
    
    print(f'Accuracy: {accuracy:.3f}')
    print(f'F1 Score: {f1:.3f}')
    print(f'AUC-ROC:  {auc:.3f}')
    print(f'Average Precision: {avg_precision:.3f}')
    
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred,
            target_names=['Negative (Different Isolate)',
            'Positive (Same Isolate)'])
    )
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_roc': auc,
        'avg_precision': avg_precision
    }


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix with percentages and counts.
    TODO: text fontsize
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    # Create annotations with both counts and percentages
    total = cm.sum()
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = count / total * 100
            row.append(f'{count}\n({percent:.1f}%)')
        annotations.append(row)
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=['Negative (Different Isolate)', 'Positive (Same Isolate)'],
                yticklabels=['Negative (Different Isolate)', 'Positive (Same Isolate)'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix: Segment Pair Classifier')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Segment Pair Classifier')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                label=f'Random (AP = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve: Segment Pair Classifier')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_distribution(y_true, y_prob, save_path=None):
    """Plot distribution of prediction probabilities by true label."""
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by true label
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    
    plt.hist(neg_probs, bins=50, alpha=0.7, label='Negative (Different Isolate)', 
             color='red', density=True)
    plt.hist(pos_probs, bins=50, alpha=0.7, label='Positive (Same Isolate)', 
             color='blue', density=True)
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_segment_performance(df):
    """Analyze performance by segment combinations."""
    print('\n' + '='*50)
    print('SEGMENT-WISE PERFORMANCE ANALYSIS')
    print('='*50)

    # Create segment pair labels
    # This creates a unique string for each segment pair (e.g., makes sure we
    # don't have S-M and M-S as different segment pairs).
    df['seg_pair'] = df.apply(
        lambda row: f'{row["seg_a"]}-{row["seg_b"]}' 
        if row['seg_a'] <= row['seg_b'] 
        else f'{row["seg_b"]}-{row["seg_a"]}', axis=1
    )

    # Performance by segment pair
    # breakpoint()
    seg_stats = []
    for seg_pair in df['seg_pair'].unique():
        subset = df[df['seg_pair'] == seg_pair]
        if len(subset) > 10:  # Only analyze pairs with sufficient data
            # Check if both classes are present for F1 and AUC
            unique_labels = subset['label'].unique()
            if len(unique_labels) > 1:
                f1 = f1_score(subset['label'], subset['pred_label'], average='binary')
                auc = roc_auc_score(subset['label'], subset['pred_prob'])
            else:
                f1 = np.nan   # F1 undefined for single class
                auc = np.nan  # AUC undefined for single class

            acc = (subset['label'] == subset['pred_label']).mean()

            seg_stats.append({
                'seg_pair': seg_pair,
                'count': len(subset),
                'f1_score': f1,
                'auc': auc,
                'accuracy': acc,
                'pos_rate': subset['label'].mean()
            })

    seg_df = pd.DataFrame(seg_stats).sort_values('f1_score', ascending=False)
    print('\nPerformance by Segment Pair:')
    print(seg_df.round(3))

    return seg_df


def analyze_same_function_pairs(df):
    """Analyze performance on same-function negative pairs."""
    print('\n' + '='*50)
    print('SAME-FUNCTION NEGATIVE PAIR ANALYSIS')
    print('='*50)

    # Same-function negative pairs (these should be hardest to classify)
    # breakpoint()
    same_func_neg = df[(df['label'] == 0) & (df['func_a'] == df['func_b'])]
    diff_func_neg = df[(df['label'] == 0) & (df['func_a'] != df['func_b'])] # TODO: should this be in a separate function?

    print(f'Same-function negative pairs: {len(same_func_neg)}')
    print(f'Different-function negative pairs: {len(diff_func_neg)}')

    if len(same_func_neg) > 0:
        same_func_acc = (same_func_neg['label'] == same_func_neg['pred_label']).mean()
        # Check if both classes are present for F1 score
        unique_labels = same_func_neg['label'].unique()
        if len(unique_labels) > 1:
            same_func_f1 = f1_score(same_func_neg['label'], same_func_neg['pred_label'], average='binary')
            print(f'Same-function negative F1: {same_func_f1:.3f}')
        else:
            print(f'Same-function negative F1: N/A (single class)')
        print(f'Same-function negative accuracy: {same_func_acc:.3f}')

        # Analyze by function type
        func_stats = []
        for func in same_func_neg['func_a'].unique():
            subset = same_func_neg[same_func_neg['func_a'] == func]
            if len(subset) > 5:
                acc = (subset['label'] == subset['pred_label']).mean()
                avg_prob = subset['pred_prob'].mean()
                func_stats.append({
                    'function': func,
                    'count': len(subset),
                    'accuracy': acc,
                    'avg_prob': avg_prob,
                    'errors': len(subset) - (subset['label'] == subset['pred_label']).sum()
                })

        if func_stats:
            func_df = pd.DataFrame(func_stats).sort_values('accuracy', ascending=False)
            print('\nSame-function negative performance by protein function:')
            print(func_df.round(3))

    # TODO: should this be in a separate function?
    if len(diff_func_neg) > 0:
        diff_func_acc = (diff_func_neg['label'] == diff_func_neg['pred_label']).mean()
        print(f'Different-function negative accuracy: {diff_func_acc:.3f}')
    return None


def analyze_prediction_confidence(df):
    """Analyze model confidence and identify uncertain predictions."""
    print('\n' + '='*50)
    print('PREDICTION CONFIDENCE ANALYSIS')
    print('='*50)

    # Define confidence bins
    # breakpoint()
    df['confidence'] = np.abs(df['pred_prob'] - 0.5)
    df['confidence_bin'] = pd.cut(
        df['confidence'], bins=[0, 0.1, 0.2, 0.3, 0.5], 
        labels=['Very Low (0.4-0.6)', 'Low (0.3-0.4, 0.6-0.7)', 
                'Medium (0.2-0.3, 0.7-0.8)', 'High (0.0-0.2, 0.8-1.0)']
    )

    # Accuracy by confidence bin
    conf_stats = df.groupby('confidence_bin').agg({
        'label': 'count',
        'pred_label': lambda x: (df.loc[x.index, 'label'] == x).mean()
    }).round(3)
    conf_stats.columns = ['count', 'accuracy']

    print('Accuracy by prediction confidence:')
    print(conf_stats)

    # Most uncertain predictions
    uncertain = df[df['confidence'] < 0.1].copy()
    if len(uncertain) > 0:
        print(f'\nMost uncertain predictions (prob 0.4-0.6): {len(uncertain)}')
        print('Sample of uncertain predictions:')
        cols = ['assembly_id_a', 'assembly_id_b', 'seg_a', 'seg_b',
                'func_a', 'func_b', 'label', 'pred_prob']
        print(uncertain[cols].head(10))
    return None


def main():
    """Main analysis function."""
    print('Analyze Segment Pair Classifier Results.')
    print('='*50)
    
    # Basic metrics
    metrics = compute_basic_metrics(y_true, y_pred, y_prob)
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, results_dir / 'confusion_matrix.png')
    plot_roc_curve(y_true, y_prob, results_dir / 'roc_curve.png')
    plot_precision_recall_curve(y_true, y_prob, results_dir / 'precision_recall_curve.png')
    plot_prediction_distribution(y_true, y_prob, results_dir / 'prediction_distribution.png')
    
    # Domain-specific analyses
    # breakpoint()
    segment_df = analyze_segment_performance(df)
    analyze_same_function_pairs(df)
    analyze_prediction_confidence(df)

    # Save summary metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / 'metrics.csv', index=False)

    # Save segment performance
    if 'segment_df' in locals():
        segment_df.to_csv(results_dir / 'segment_performance.csv', index=False)
    
    print(f'\nAnalysis complete! Results saved to: {results_dir}')

if __name__ == '__main__':
    main() 