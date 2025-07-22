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

from src.utils.config import VIRUS_NAME, DATA_VERSION, TASK_NAME
from src.utils.plot_config import apply_default_style, SEGMENT_COLORS, SEGMENT_ORDER, map_protein_name

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
apply_default_style()
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


def plot_confusion_matrix(y_true, y_pred, save_path=None, show_labels=True):
    """Plot confusion matrix with optional TP, TN, FP, FN labels and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Save current matplotlib settings and override grid
    import matplotlib as mpl
    original_rc = mpl.rcParams.copy()
    mpl.rcParams.update({
        'axes.grid': False,
        'grid.alpha': 0,
        'axes.axisbelow': False
    })
    
    # Create figure with explicit styling to disable all lines
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Disable all grid lines and styling
    ax.grid(False)
    ax.set_axisbelow(False)
    
    # Create annotations with counts, percentages, and optional quadrant labels
    total = cm.sum()
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = count / total * 100
            
            if show_labels:
                # Determine quadrant label
                if i == 0 and j == 0:  # True Negative
                    label = 'TN'
                elif i == 0 and j == 1:  # False Positive
                    label = 'FP'
                elif i == 1 and j == 0:  # False Negative
                    label = 'FN'
                else:  # True Positive
                    label = 'TP'
                
                # Vertical alignment with line breaks
                row.append(f'{label}\n{count}\n({percent:.1f}%)')
            else:
                # Just count and percentage
                row.append(f'{count}\n({percent:.1f}%)')
        annotations.append(row)
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=['Negative (Different Isolate)', 'Positive (Same Isolate)'],
                yticklabels=['Negative (Different Isolate)', 'Positive (Same Isolate)'],
                cbar_kws={'label': 'Count'}, linewidths=0, ax=ax)
    
    # Explicitly disable grid on the heatmap
    ax.grid(False)
    
    plt.title('Confusion Matrix: Segment Pair Classifier', fontweight='bold', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Restore original settings
    mpl.rcParams.update(original_rc)
    
    return fig


def plot_confusion_matrix_simple(y_true, y_pred, save_path=None):
    """Simplest possible confusion matrix plot to test for lines."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Save current matplotlib settings
    import matplotlib as mpl
    original_rc = mpl.rcParams.copy()
    
    # Override the global grid setting that's causing lines
    mpl.rcParams.update({
        'axes.grid': False,
        'grid.alpha': 0,
        'axes.axisbelow': False
    })
    
    # Create figure with explicit styling to disable all lines
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Disable all grid lines and styling
    ax.grid(False)
    ax.set_axisbelow(False)
    
    # Most basic heatmap with minimal settings
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                linewidths=0, cbar=False, ax=ax)
    
    # Explicitly disable grid on the heatmap
    ax.grid(False)
    
    plt.title('Simple Confusion Matrix Test')
    plt.tight_layout()
    
    if save_path:
        test_path = save_path.parent / 'confusion_matrix_simple_test.png'
        plt.savefig(test_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Restore original settings
    mpl.rcParams.update(original_rc)
    
    return fig


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
    plt.title('ROC Curve: Segment Pair Classifier', fontweight='bold', fontsize=14)
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
    plt.title('Precision-Recall Curve: Segment Pair Classifier', fontweight='bold', fontsize=14)
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
    plt.title('Distribution of Prediction Probabilities', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_fp_fn_errors(df, y_true, y_pred, y_prob):
    """Detailed analysis of False Positives and False Negatives.
    
    This function generates the "False Positives by Protein Function" plot
    and other FP/FN analysis visualizations.
    """
    print('\n' + '='*60)
    print('FALSE POSITIVE & FALSE NEGATIVE ANALYSIS')
    print('='*60)
    
    # Create segment pair labels first
    df['seg_pair'] = df.apply(
        lambda row: f'{row["seg_a"]}-{row["seg_b"]}' 
        if row['seg_a'] <= row['seg_b'] 
        else f'{row["seg_b"]}-{row["seg_a"]}', axis=1
    )
    
    # Add prediction columns to dataframe for easier analysis
    df_analysis = df.copy()
    df_analysis['y_true'] = y_true
    df_analysis['y_pred'] = y_pred
    df_analysis['y_prob'] = y_prob
    
    # Identify FP and FN
    fp_mask = (df_analysis['y_true'] == 0) & (df_analysis['y_pred'] == 1)  # Predicted positive, actually negative
    fn_mask = (df_analysis['y_true'] == 1) & (df_analysis['y_pred'] == 0)  # Predicted negative, actually positive
    
    fp_df = df_analysis[fp_mask].copy()
    fn_df = df_analysis[fn_mask].copy()
    
    print(f'False Positives (FP): {len(fp_df)}')
    print(f'False Negatives (FN): {len(fn_df)}')
    print(f'FP/FN ratio: {len(fp_df)/len(fn_df):.2f}')
    
    # Confidence analysis
    print(f'\nConfidence Analysis:')
    print(f'FP average confidence: {fp_df["y_prob"].mean():.3f}')
    print(f'FN average confidence: {fn_df["y_prob"].mean():.3f}')
    print(f'FP confidence std: {fp_df["y_prob"].std():.3f}')
    print(f'FN confidence std: {fn_df["y_prob"].std():.3f}')
    
    # Create FP/FN analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confidence distribution comparison
    axes[0, 0].hist(fp_df['y_prob'], bins=20, alpha=0.7, label=f'False Positives (n={len(fp_df)})', 
                    color='red', density=False)
    axes[0, 0].hist(fn_df['y_prob'], bins=20, alpha=0.7, label=f'False Negatives (n={len(fn_df)})', 
                    color='blue', density=False)
    axes[0, 0].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    axes[0, 0].set_xlabel('Prediction Probability')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Confidence Distribution: FP and FN', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error rate by segment pair
    # Calculate error rates by segment pair
    seg_error_stats = []
    for seg_pair in df_analysis['seg_pair'].unique():
        subset = df_analysis[df_analysis['seg_pair'] == seg_pair]
        if len(subset) >= 5:  # Only analyze pairs with sufficient data
            # Create masks for this subset
            subset_fp_mask = (subset['y_true'] == 0) & (subset['y_pred'] == 1)
            subset_fn_mask = (subset['y_true'] == 1) & (subset['y_pred'] == 0)
            
            fp_count = subset_fp_mask.sum()
            fn_count = subset_fn_mask.sum()
            total_errors = fp_count + fn_count
            error_rate = total_errors / len(subset)
            
            seg_error_stats.append({
                'seg_pair': seg_pair,
                'total_samples': len(subset),
                'fp_count': fp_count,
                'fn_count': fn_count,
                'total_errors': total_errors,
                'error_rate': error_rate,
                'fp_rate': fp_count / len(subset),
                'fn_rate': fn_count / len(subset)
            })
    
    if seg_error_stats:
        seg_error_df = pd.DataFrame(seg_error_stats).sort_values('error_rate', ascending=False)
        
        # Plot top 10 segment pairs by error rate
        top_errors = seg_error_df.head(10)
        x_pos = np.arange(len(top_errors))
        
        fp_bars = axes[0, 1].bar(x_pos, top_errors['fp_rate'], alpha=0.7, label='FP Rate', color='red')
        fn_bars = axes[0, 1].bar(x_pos, top_errors['fn_rate'], alpha=0.7, label='FN Rate', color='blue', 
                                bottom=top_errors['fp_rate'])
        axes[0, 1].set_xlabel('Segment Pair')
        axes[0, 1].set_ylabel('Error Rate')
        axes[0, 1].set_title('Error Rates by Segment Pair (Top 10)', fontweight='bold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(top_errors['seg_pair'], rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (fp_bar, fn_bar, fp_rate, fn_rate) in enumerate(zip(fp_bars, fn_bars, top_errors['fp_rate'], top_errors['fn_rate'])):
            # Add FP rate label
            if fp_rate > 0:
                axes[0, 1].text(fp_bar.get_x() + fp_bar.get_width()/2., fp_rate/2, 
                               f'{fp_rate:.2f}', ha='center', va='center', 
                               fontweight='bold', color='white', fontsize=8)
            
            # Add FN rate label
            if fn_rate > 0:
                axes[0, 1].text(fn_bar.get_x() + fn_bar.get_width()/2., fp_rate + fn_rate/2, 
                               f'{fn_rate:.2f}', ha='center', va='center', 
                               fontweight='bold', color='white', fontsize=8)
    
    # 3. Function analysis for errors
    # Analyze FP by function (considering both func_a and func_b)
    if len(fp_df) > 0:
        print(f"\nDebug: FP function analysis")
        print(f"Total FP count: {len(fp_df)}")
        
        # Create a combined list of all functions involved in FPs
        all_fp_functions = []
        for _, row in fp_df.iterrows():
            all_fp_functions.append(row['func_a'])
            all_fp_functions.append(row['func_b'])
        
        print(f"Total function occurrences in FPs: {len(all_fp_functions)}")
        print(f"Unique functions in FPs: {set(all_fp_functions)}")
        
        # Count function occurrences
        from collections import Counter
        func_counts = Counter(all_fp_functions)
        print(f"Function counts in FPs:")
        for func, count in func_counts.most_common():
            print(f"  {func}: {count}")
        
        # Create function statistics
        fp_func_stats = []
        for func, count in func_counts.most_common():
            # Calculate average confidence for this function
            func_confidences = []
            for _, row in fp_df.iterrows():
                if row['func_a'] == func or row['func_b'] == func:
                    func_confidences.append(row['y_prob'])
            
            fp_func_stats.append({
                'function': func,
                'count': count,
                'avg_confidence': np.mean(func_confidences),
                'std_confidence': np.std(func_confidences)
            })
        
        fp_func_df = pd.DataFrame(fp_func_stats)
        print(f"\nFP function stats (combined func_a and func_b):")
        print(fp_func_df.round(3))
        print(f"Number of unique functions with FPs: {len(fp_func_df)}")
        
        # Plot FP by function
        if len(fp_func_df) > 0:
            bars = axes[1, 0].bar(range(len(fp_func_df)), fp_func_df['count'], 
                                 color='#6C5CE7', alpha=0.7)  # Calmer purple color
            axes[1, 0].set_xlabel('Protein Function')
            axes[1, 0].set_ylabel('False Positive Count')
            axes[1, 0].set_title('False Positives by Protein Function', fontweight='bold')
            axes[1, 0].set_xticks(range(len(fp_func_df)))
            
            # Use protein name abbreviations
            func_labels = [map_protein_name(func) for func in fp_func_df['function']]
            axes[1, 0].set_xticklabels(func_labels, rotation=45, ha='right')
            
            # Add count labels in the middle of bars
            for bar, count in zip(bars, fp_func_df['count']):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height/2, 
                               f'{count}', ha='center', va='center', 
                               fontweight='bold', color='white', fontsize=10)
            
            axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Confidence vs error type
    all_errors = pd.concat([fp_df, fn_df])
    all_errors['error_type'] = ['FP'] * len(fp_df) + ['FN'] * len(fn_df)
    
    axes[1, 1].boxplot([fp_df['y_prob'], fn_df['y_prob']], 
                      labels=['False Positives', 'False Negatives'])
    axes[1, 1].set_ylabel('Prediction Probability')
    axes[1, 1].set_title('Confidence Distribution: FP and FN', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'fp_fn_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed error statistics
    error_summary = {
        'fp_count': len(fp_df),
        'fn_count': len(fn_df),
        'fp_fn_ratio': len(fp_df) / len(fn_df) if len(fn_df) > 0 else float('inf'),
        'fp_avg_confidence': fp_df['y_prob'].mean() if len(fp_df) > 0 else 0,
        'fn_avg_confidence': fn_df['y_prob'].mean() if len(fn_df) > 0 else 0,
        'fp_confidence_std': fp_df['y_prob'].std() if len(fp_df) > 0 else 0,
        'fn_confidence_std': fn_df['y_prob'].std() if len(fn_df) > 0 else 0
    }
    
    error_summary_df = pd.DataFrame([error_summary])
    error_summary_df.to_csv(results_dir / 'error_analysis_summary.csv', index=False)
    
    # Save detailed FP and FN data
    if len(fp_df) > 0:
        fp_df.to_csv(results_dir / 'false_positives_detailed.csv', index=False)
    if len(fn_df) > 0:
        fn_df.to_csv(results_dir / 'false_negatives_detailed.csv', index=False)
    
    print(f'\n✅ FP/FN analysis complete!')
    print(f'  - FP/FN ratio: {error_summary["fp_fn_ratio"]:.2f}')
    print(f'  - FP avg confidence: {error_summary["fp_avg_confidence"]:.3f}')
    print(f'  - FN avg confidence: {error_summary["fn_avg_confidence"]:.3f}')
    
    return error_summary


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
    same_func_neg = df[(df['label'] == 0) & (df['func_a'] == df['func_b'])]
    diff_func_neg = df[(df['label'] == 0) & (df['func_a'] != df['func_b'])]

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


def main(create_performance_plots=True, show_confusion_labels=True):
    """Main analysis function."""
    print('Analyze Segment Pair Classifier Results.')
    print('='*50)
    
    # Basic metrics
    metrics = compute_basic_metrics(y_true, y_pred, y_prob)
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, results_dir / 'confusion_matrix.png', show_labels=show_confusion_labels)
    
    # Test simple confusion matrix to check for lines
    plot_confusion_matrix_simple(y_true, y_pred, results_dir / 'confusion_matrix.png')
    
    # Optional performance plots
    if create_performance_plots:
        plot_roc_curve(y_true, y_prob, results_dir / 'roc_curve.png')
        plot_precision_recall_curve(y_true, y_prob, results_dir / 'precision_recall_curve.png')
        plot_prediction_distribution(y_true, y_prob, results_dir / 'prediction_distribution.png')
        print("✅ Performance plots created (roc_curve.png, precision_recall_curve.png, prediction_distribution.png)")
    else:
        print("⏭️  Skipping performance plots (create_performance_plots=False)")
    
    # Enhanced FP/FN analysis
    error_summary = analyze_fp_fn_errors(df, y_true, y_pred, y_prob)
    
    # Domain-specific analyses
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
    
    # Key insights summary
    print('\n' + '='*60)
    print('KEY INSIGHTS SUMMARY')
    print('='*60)
    print(f'• Overall accuracy: {metrics["accuracy"]:.3f}')
    print(f'• F1 score: {metrics["f1_score"]:.3f}')
    print(f'• AUC-ROC: {metrics["auc_roc"]:.3f}')
    print(f'• Average precision: {metrics["avg_precision"]:.3f}')
    print(f'• False Positives: {error_summary["fp_count"]}')
    print(f'• False Negatives: {error_summary["fn_count"]}')
    print(f'• FP/FN ratio: {error_summary["fp_fn_ratio"]:.2f}')
    print(f'• FP avg confidence: {error_summary["fp_avg_confidence"]:.3f}')
    print(f'• FN avg confidence: {error_summary["fn_avg_confidence"]:.3f}')

if __name__ == '__main__':
    # Set to False to skip performance plots (roc, precision-recall, prediction distribution)
    # Set to False to hide confusion matrix labels (TP, TN, FP, FN)
    main(create_performance_plots=True, show_confusion_labels=True)