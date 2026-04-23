"""
Analyze Stage 4: Training results from the segment pair classifier.

This script analyzes the test results from the trained ESM-2 frozen pair classifier,
providing both standard ML metrics and domain-specific insights for viral protein
segment classification.

Output directory: <training_run_dir>/post_hoc/ — colocated with the run's
test_predicted.csv and best_model.pt, but kept in a subfolder so analysis
artifacts are easy to distinguish from training outputs.

Generated plots:
- confusion_matrix.png: Confusion matrix with TP/TN/FP/FN labels
- roc_curve.png: ROC curve with AUC
- precision_recall_curve.png: Precision-recall curve with AP
- prediction_distribution.png: Distribution of prediction probabilities by label
- fp_fn_analysis.png: False positive/negative analysis by segment and function

Usage:
    python src/analysis/analyze_stage4_train.py --config_bundle bunya
    python src/analysis/analyze_stage4_train.py --config_bundle bunya --model_dir ./models/bunya/April_2025
    python src/analysis/analyze_stage4_train.py --config_bundle flu_a_3p_1ks
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, roc_auc_score, average_precision_score
)

# Regex used to decide whether an hn_subtype string is parseable. Anything that
# does not match lands in the "unknown" bucket for Level 1/2 stratification.
# See docs/post_hoc_analysis_design.md ("Subtype parsing") for the empirical
# distribution that motivates this rule (99.07% parse cleanly on val_unfilt;
# the ~1% unknown is dominated by the untypable "HN" value).
SUBTYPE_RE = re.compile(r'^H\d+N\d+$')

# Year bins used for Level 2 year-axis stratification. Three bins keep strata
# dense while separating the pre-2016 tail, the 2016-2020 pre-pandemic window,
# and the 2021+ recent era. See docs/post_hoc_analysis_design.md ("Axis: year_bin").
YEAR_BIN_EDGES = [(-np.inf, 2015, '<=2015'), (2016, 2020, '2016-2020'), (2021, np.inf, '2021+')]

# Minimum samples to keep a stratum as its own row in Level 2 tables. Smaller
# strata are collapsed into "other" per axis to avoid noisy per-fold metrics.
LEVEL2_MIN_SAMPLES = 20

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config_hydra import get_virus_config_hydra
from src.utils.path_utils import build_training_paths
from src.utils.plot_config import apply_default_style, SEGMENT_COLORS, SEGMENT_ORDER, map_protein_name


def compute_basic_metrics(y_true, y_pred, y_prob):
    """Compute and display basic classification metrics.

    All metrics are computed on the test set (from test_predicted.csv).
    """
    print('\n' + '='*50)
    print('BASIC CLASSIFICATION METRICS')
    print('='*50)
    
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_prob)
    accuracy = (y_true == y_pred).mean()
    avg_precision = average_precision_score(y_true, y_prob)
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    # BCE loss on the test set (matches training criterion)
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    bce_loss = float(-np.mean(
        y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped)
    ))
    
    print(f'Accuracy: {accuracy:.3f}')
    print(f'F1 Score (binary): {f1:.3f}')
    print(f'F1 Score (macro): {f1_macro:.3f}')
    print(f'AUC-ROC:  {roc_auc:.3f}')
    print(f'Average Precision: {avg_precision:.3f}')
    print(f'Brier Score: {brier_score:.4f}')
    print(f'BCE Loss: {bce_loss:.4f}')
    
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred,
            target_names=['Negative (Different Isolate)',
            'Positive (Same Isolate)'])
    )
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'f1_macro': f1_macro,
        'auc_roc': roc_auc,
        'avg_precision': avg_precision,
        'brier_score': brier_score,
        'loss': bce_loss
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
    """Plot distribution of prediction probabilities by true label.

    Colors, xlabel, and title are kept consistent with the right panel of
    ``create_model_calibration_plot`` in ``create_presentation_plots.py``.
    """
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by true label
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    
    plt.hist(neg_probs, bins=30, alpha=0.7, label='Negative (Different Isolate)', 
             color='#E74C3C', density=True)
    plt.hist(pos_probs, bins=30, alpha=0.7, label='Positive (Same Isolate)', 
             color='#3498DB', density=True)
    
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Decision Threshold')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Probabilities', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_fp_fn_errors(df, y_true, y_pred, y_prob, results_dir: Path):
    """Detailed analysis of False Positives and False Negatives.
    
    This function generates the "False Positives by Protein Function" plot
    and other FP/FN analysis visualizations.
    
    Args:
        df: DataFrame with predictions and metadata
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        results_dir: Directory to save analysis outputs
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
    if len(fn_df) > 0:
        print(f'FP/FN ratio: {len(fp_df)/len(fn_df):.2f}')
    else:
        print(f'FP/FN ratio: ∞ (no false negatives)')
    
    # Confidence analysis
    print(f'\nConfidence Analysis:')
    if len(fp_df) > 0:
        print(f'FP average confidence: {fp_df["y_prob"].mean():.3f}')
        print(f'FP confidence std: {fp_df["y_prob"].std():.3f}')
    else:
        print(f'FP average confidence: N/A (no false positives)')
        print(f'FP confidence std: N/A (no false positives)')
    
    if len(fn_df) > 0:
        print(f'FN average confidence: {fn_df["y_prob"].mean():.3f}')
        print(f'FN confidence std: {fn_df["y_prob"].std():.3f}')
    else:
        print(f'FN average confidence: N/A (no false negatives)')
        print(f'FN confidence std: N/A (no false negatives)')
    
    # Create FP/FN analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confidence distribution comparison
    if len(fp_df) > 0:
        axes[0, 0].hist(fp_df['y_prob'], bins=20, alpha=0.7, label=f'False Positives (n={len(fp_df)})', 
                        color='red', density=False)
    if len(fn_df) > 0:
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
    if len(fp_df) > 0 and len(fn_df) > 0:
        all_errors = pd.concat([fp_df, fn_df])
        all_errors['error_type'] = ['FP'] * len(fp_df) + ['FN'] * len(fn_df)
        
        axes[1, 1].boxplot([fp_df['y_prob'], fn_df['y_prob']], 
                          labels=['False Positives', 'False Negatives'])
        axes[1, 1].set_ylabel('Prediction Probability')
        axes[1, 1].set_title('Confidence Distribution: FP and FN', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    elif len(fp_df) > 0:
        axes[1, 1].boxplot([fp_df['y_prob']], labels=['False Positives'])
        axes[1, 1].set_ylabel('Prediction Probability')
        axes[1, 1].set_title('Confidence Distribution: FP Only', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    elif len(fn_df) > 0:
        axes[1, 1].boxplot([fn_df['y_prob']], labels=['False Negatives'])
        axes[1, 1].set_ylabel('Prediction Probability')
        axes[1, 1].set_title('Confidence Distribution: FN Only', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No errors found', ha='center', va='center', 
                        transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('Confidence Distribution: No Errors', fontweight='bold')
    
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
    
    print(f'\nDone. FP/FN analysis complete.')
    print(f'  - FP/FN ratio: {error_summary["fp_fn_ratio"]:.2f}')
    print(f'  - FP avg confidence: {error_summary["fp_avg_confidence"]:.3f}')
    print(f'  - FN avg confidence: {error_summary["fn_avg_confidence"]:.3f}')
    
    return error_summary


# ============================================================================
# Stratified evaluation helpers (Level 1 / Level 2)
#
# Design reference: docs/post_hoc_analysis_design.md
#
# Two-sided metadata enrichment:
#   The isolate_metadata table is joined twice onto each test pair (once per
#   assembly_id_{a,b}) to produce per-side columns host_{a,b}, hn_subtype_{a,b}_raw,
#   year_{a,b}. The original analyze_errors_by_metadata joined only on
#   assembly_id_a, which silently dropped side-B metadata and was incorrect
#   for cross-stratum negative pairs (e.g., a H1N1+H3N2 cross-subtype negative
#   would be counted as H1N1 only). Positive pairs have assembly_id_a ==
#   assembly_id_b by construction, so per-side columns collapse for positives.
#
# Subtype parsing:
#   `^H\d+N\d+$` matches ~99% of isolates; the ~1% unknown (mostly literal "HN")
#   is tracked as an explicit bucket rather than dropped.
#
# Year binning:
#   Three fixed bins (<=2015, 2016-2020, 2021+) keep per-bin counts dense for
#   per-fold analysis. Adjust YEAR_BIN_EDGES (module constant) if finer
#   granularity is needed.
# ============================================================================


def _parse_subtype(s) -> str:
    """Map a raw hn_subtype string to a canonical H<d>N<d> form or 'unknown'.

    Any value not matching SUBTYPE_RE (NaN, 'HN', 'H1N', etc.) returns 'unknown'.
    """
    if pd.isna(s):
        return 'unknown'
    s = str(s)
    return s if SUBTYPE_RE.match(s) else 'unknown'


def _year_to_bin(y) -> str:
    """Map a year (float/int) to a coarse bin label. NaN returns 'unknown'."""
    if pd.isna(y):
        return 'unknown'
    y = float(y)
    for lo, hi, label in YEAR_BIN_EDGES:
        if lo <= y <= hi:
            return label
    return 'unknown'


def _enrich_pairs_with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add two-sided metadata columns to a pair-level dataframe.

    For each pair we need metadata for BOTH assembly ids. We load the flu
    metadata table once, then merge it twice: once on assembly_id_a, once on
    assembly_id_b. Output columns:
      host_a, host_b, hn_subtype_a_raw, hn_subtype_b_raw, year_a, year_b
      subtype_a, subtype_b       (parsed with 'unknown' fallback)
      year_bin_a, year_bin_b

    Returns a COPY; raises on missing metadata file (caller catches).
    """
    from src.utils.metadata_enrichment import load_flu_metadata

    meta = load_flu_metadata(project_root=project_root)
    # hash_id == assembly_id in this pipeline; keep just the columns we stratify on.
    meta = meta[['hash_id', 'host', 'hn_subtype', 'year']].rename(columns={'hash_id': 'assembly_id'})
    meta['assembly_id'] = meta['assembly_id'].astype(str)

    out = df.copy()
    out['assembly_id_a'] = out['assembly_id_a'].astype(str)
    out['assembly_id_b'] = out['assembly_id_b'].astype(str)

    # Side A
    meta_a = meta.rename(columns={
        'assembly_id': 'assembly_id_a',
        'host': 'host_a',
        'hn_subtype': 'hn_subtype_a_raw',
        'year': 'year_a',
    })
    out = out.merge(meta_a, on='assembly_id_a', how='left')

    # Side B
    meta_b = meta.rename(columns={
        'assembly_id': 'assembly_id_b',
        'host': 'host_b',
        'hn_subtype': 'hn_subtype_b_raw',
        'year': 'year_b',
    })
    out = out.merge(meta_b, on='assembly_id_b', how='left')

    # Parse subtype and year_bin on each side. Using .apply is fine here: the
    # test set is ~20K rows max and these helpers keep NaN handling readable.
    out['subtype_a'] = out['hn_subtype_a_raw'].apply(_parse_subtype)
    out['subtype_b'] = out['hn_subtype_b_raw'].apply(_parse_subtype)
    out['year_bin_a'] = out['year_a'].apply(_year_to_bin)
    out['year_bin_b'] = out['year_b'].apply(_year_to_bin)
    return out


def _compute_stratum_metrics(subset: pd.DataFrame) -> dict:
    """Compute the full metric row for one stratum subset.

    Metrics that require both classes (F1, AUC-ROC, AUC-PR, precision, recall)
    are returned as NaN when the subset contains a single class rather than
    silently coerced to 0.0. Uses `label` / `pred_label` / `pred_prob` from
    the enriched dataframe; `pred_label` is already computed with the run's
    val-optimal threshold upstream (see docs/post_hoc_analysis_design.md,
    "Threshold handling"), so no re-thresholding is done here.
    """
    y_true = subset['label'].values
    y_pred = subset['pred_label'].values
    y_prob = subset['pred_prob'].values
    n = len(subset)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    accuracy = (tp + tn) / n if n > 0 else np.nan
    fp_rate = fp / n if n > 0 else np.nan
    fn_rate = fn / n if n > 0 else np.nan
    error_rate = (fp + fn) / n if n > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    # True-positive rate (sensitivity) = recall; only defined when stratum has
    # any positives. True-negative rate (specificity) = TN/(TN+FP); only
    # defined when stratum has any negatives. For single-class strata (e.g.,
    # Level 1 negative-only regimes), the inapplicable side is NaN.
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    if precision and recall and not (np.isnan(precision) or np.isnan(recall)):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = np.nan

    # AUC metrics are ill-defined when the stratum has only one class.
    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
    else:
        auc_roc = np.nan
        auc_pr = np.nan

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    return {
        'n_samples': n,
        'n_pos': int(pos_mask.sum()),
        'n_neg': int(neg_mask.sum()),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'tpr': tpr,
        'tnr': tnr,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'error_rate': error_rate,
        'fp_avg_confidence': float(y_prob[neg_mask].mean()) if neg_mask.any() else np.nan,
        'fn_avg_confidence': float(y_prob[pos_mask].mean()) if pos_mask.any() else np.nan,
    }


def analyze_level1_pair_regime(df_enriched: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Level 1 stratified evaluation: classify each test pair into a mutually
    exclusive pair regime, then report per-regime metrics.

    Regimes (see docs/post_hoc_analysis_design.md "Level 1"):
      - positive: label == 1 (same isolate on both sides by construction).
      - within_subtype_neg: label == 0, both sides parse to H<d>N<d>, and the
        two subtypes match. "Hard" negatives: model cannot use subtype as a
        shortcut.
      - cross_subtype_neg: label == 0, both sides parse, and subtypes differ.
        "Easy" negatives: subtype difference is a direct cue.
      - unknown_neg: label == 0 and at least one side has an unparseable
        subtype. Small residual bucket reported separately to keep it from
        contaminating the two informative negative regimes.

    A large gap (cross_subtype_neg accuracy ~ 1.0 but within_subtype_neg
    accuracy much lower) is evidence the model is using subtype as a shortcut.
    """
    print('\n' + '=' * 60)
    print('LEVEL 1 STRATIFIED EVAL: pair-regime')
    print('=' * 60)

    def _regime(row):
        if row['label'] == 1:
            return 'positive'
        sa, sb = row['subtype_a'], row['subtype_b']
        if sa == 'unknown' or sb == 'unknown':
            return 'unknown_neg'
        return 'within_subtype_neg' if sa == sb else 'cross_subtype_neg'

    df_enriched = df_enriched.copy()
    df_enriched['pair_regime'] = df_enriched.apply(_regime, axis=1)

    # Fixed ordering; keeps downstream cross-fold aggregation scripts simple.
    regimes = ['positive', 'within_subtype_neg', 'cross_subtype_neg', 'unknown_neg']
    rows = []
    for regime in regimes:
        sub = df_enriched[df_enriched['pair_regime'] == regime]
        if len(sub) == 0:
            rows.append({'pair_regime': regime, 'n_samples': 0})
            continue
        m = _compute_stratum_metrics(sub)
        m['pair_regime'] = regime
        rows.append(m)

    stats_df = pd.DataFrame(rows)
    cols = ['pair_regime'] + [c for c in stats_df.columns if c != 'pair_regime']
    stats_df = stats_df[cols]

    output_file = results_dir / 'level1_pair_regime.csv'
    stats_df.to_csv(output_file, index=False)
    print(f"Saved Level 1 metrics to: {output_file}")
    print(stats_df.round(3).to_string(index=False))

    _plot_level1_pair_regime(stats_df, results_dir)
    return stats_df


def _plot_level1_pair_regime(stats_df: pd.DataFrame, results_dir: Path) -> None:
    """Bar plot of TPR / TNR / F1 per pair regime with value labels.

    Per-regime definability (single-class strata make most "standard" metrics
    undefined — see docs/post_hoc_analysis_design.md "Level 1"):
      - positive (all label=1): TPR defined (= recall = fraction of positive
        pairs correctly classified). TNR undefined (no negatives). F1 defined
        (precision = 1 because no FPs possible in this stratum).
      - within_subtype_neg / cross_subtype_neg / unknown_neg (all label=0):
        TNR defined (= fraction of negative pairs correctly classified). TPR
        and F1 undefined (no positives).

    TPR + TNR together give a unified "correctly-classified fraction" story
    across both regime types. The asymmetric visibility (TPR appears only on
    `positive`, TNR only on negatives) is intentional — it makes the shortcut
    signal (TNR gap between within_subtype_neg and cross_subtype_neg) easy
    to read off. Value labels render above each defined bar; undefined bars
    are drawn as thin grey placeholders labeled "N/A" so the reader can tell
    "undefined" apart from "zero".
    """
    plot_df = stats_df[stats_df['n_samples'] > 0].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(plot_df))
    width = 0.25
    specs = [
        (-width, 'tpr',      'TPR (sensitivity)',  'seagreen'),
        (0.0,    'tnr',      'TNR (specificity)',  'steelblue'),
        ( width, 'f1_score', 'F1',                 'purple'),
    ]
    for offset, col, label, color in specs:
        values = plot_df[col].values
        for xi, v in zip(x, values):
            if pd.isna(v):
                # Thin placeholder bar + "N/A" label so undefined is visibly
                # different from a true zero.
                ax.bar(xi + offset, 0.02, width, color='lightgrey',
                       edgecolor='grey', alpha=0.5,
                       label=label if xi == x[0] and not any(plot_df[col].notna()) else None)
                ax.text(xi + offset, 0.03, 'N/A', ha='center', va='bottom',
                        fontsize=8, color='grey')
            else:
                ax.bar(xi + offset, v, width, color=color, edgecolor='black',
                       alpha=0.85,
                       label=label if xi == next(i for i, vv in enumerate(values) if not pd.isna(vv)) + 0 and True else None)
                ax.text(xi + offset, v + 0.01, f'{v:.3f}', ha='center',
                        va='bottom', fontsize=8)
    # Clean legend: build it manually because the per-bar label trick above
    # can end up repeating entries.
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor='black', label=lbl)
                      for _, _, lbl, c in specs]
    ax.legend(handles=legend_handles, loc='lower right')
    # n_samples annotation above each regime.
    for xi, n in zip(x, plot_df['n_samples']):
        ax.text(xi, 1.08, f'n={n:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['pair_regime'], rotation=20, ha='right')
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel('Score')
    ax.set_title('Level 1: TPR / TNR / F1 by pair regime')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = results_dir / 'level1_pair_regime.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Level 1 plot to: {out}")


def analyze_level2_by_axis(df_enriched: pd.DataFrame, axis: str, results_dir: Path) -> pd.DataFrame:
    """Level 2 per-axis marginal stratification.

    axis is one of {'host', 'subtype', 'year_bin'} and expects columns
    {axis}_a and {axis}_b on df_enriched.

    A pair is counted in stratum value V only if BOTH sides share that value
    (host_a == host_b == V, etc.). Pairs where sides disagree go to a single
    'mixed' stratum; pairs where either side is NaN/unknown go to 'unknown'.
    This avoids double-counting and keeps per-stratum metrics interpretable
    (the model is evaluated on a homogeneous within-stratum population plus
    an explicit mixed/unknown residual).

    Rare strata with fewer than LEVEL2_MIN_SAMPLES pairs are collapsed into
    a single 'other' stratum to prevent very noisy per-fold metrics.
    """
    print('\n' + '=' * 60)
    print(f'LEVEL 2 STRATIFIED EVAL: {axis}')
    print('=' * 60)

    col_a, col_b = f'{axis}_a', f'{axis}_b'
    if col_a not in df_enriched.columns or col_b not in df_enriched.columns:
        print(f"WARNING: columns {col_a}/{col_b} not available; skipping axis {axis}.")
        return pd.DataFrame()

    df = df_enriched.copy()
    va = df[col_a].astype(str)
    vb = df[col_b].astype(str)
    unknown_mask = (va == 'unknown') | (vb == 'unknown') | (va == 'nan') | (vb == 'nan')
    same_mask = (~unknown_mask) & (va == vb)

    # Keep only values with enough same-side pairs as dedicated strata; rarer
    # values fold into 'other' to avoid per-fold noise.
    value_counts = va[same_mask].value_counts()
    kept_values = set(value_counts[value_counts >= LEVEL2_MIN_SAMPLES].index)

    def _stratum(row):
        a, b = str(row[col_a]), str(row[col_b])
        if a in ('unknown', 'nan') or b in ('unknown', 'nan'):
            return 'unknown'
        if a != b:
            return 'mixed'
        return a if a in kept_values else 'other'

    df['stratum'] = df.apply(_stratum, axis=1)

    rows = []
    for value, sub in df.groupby('stratum', dropna=False):
        if len(sub) == 0:
            continue
        m = _compute_stratum_metrics(sub)
        m['stratum'] = value
        rows.append(m)

    stats_df = pd.DataFrame(rows).sort_values('n_samples', ascending=False).reset_index(drop=True)
    cols = ['stratum'] + [c for c in stats_df.columns if c != 'stratum']
    stats_df = stats_df[cols]

    output_file = results_dir / f'level2_by_{axis}.csv'
    stats_df.to_csv(output_file, index=False)
    print(f"Saved Level 2 ({axis}) metrics to: {output_file}")
    print(stats_df.round(3).to_string(index=False))

    _plot_level2_axis(stats_df, axis, results_dir)
    return stats_df


def _plot_level2_axis(stats_df: pd.DataFrame, axis: str, results_dir: Path) -> None:
    """Bar plot of F1 / AUC-ROC / AUC-PR per stratum for one axis.

    AUC-PR is preferred over Accuracy for the Level 2 plots because
    per-stratum subsets can be heavily class-imbalanced (e.g., a subtype
    where nearly all pairs are negative), and AUC-PR summarizes the
    precision-recall tradeoff without being dominated by the majority class.
    All three metrics match the set used in performance_metrics_by_metadata.png
    so the two plot families are directly comparable.

    Strata with n_samples < LEVEL2_MIN_SAMPLES are collapsed into 'other'
    upstream. The plot additionally surfaces 'mixed' (sides disagree) and
    'unknown' (NaN/malformed side) buckets IF they appear in the data; the
    title lists only the residual buckets actually present.
    """
    plot_df = stats_df[stats_df['n_samples'] > 0].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(plot_df) + 4), 5))
    x = np.arange(len(plot_df))
    width = 0.25
    # Render each metric per-stratum so undefined (NaN) values can be shown
    # as thin grey "N/A" placeholders instead of missing bars. See
    # docs/post_hoc_analysis_design.md "Reading the plots" — two different
    # reasons a bar can be missing (single-class stratum → AUC NaN;
    # all-negative stratum → F1 also NaN).
    specs = [
        (-width, 'f1_score', 'F1',      'purple'),
        (0.0,    'auc_roc',  'AUC-ROC', 'teal'),
        ( width, 'auc_pr',   'AUC-PR',  'orange'),
    ]
    for offset, col, _label, color in specs:
        for xi, v in zip(x, plot_df[col].values):
            if pd.isna(v):
                ax.bar(xi + offset, 0.02, width, color='lightgrey',
                       edgecolor='grey', alpha=0.5)
                ax.text(xi + offset, 0.03, 'N/A', ha='center', va='bottom',
                        fontsize=7, color='grey')
            else:
                ax.bar(xi + offset, v, width, color=color,
                       edgecolor='black', alpha=0.85)
    # Legend built manually because per-bar drawing above would produce
    # duplicate/missing legend entries.
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor='black', label=lbl)
                      for _, _, lbl, c in specs]
    for xi, n in zip(x, plot_df['n_samples']):
        ax.text(xi, 1.02, f'n={n:,}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['stratum'], rotation=45, ha='right')
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel('Score')
    residuals_present = [b for b in ('mixed', 'unknown', 'other')
                         if (plot_df['stratum'] == b).any()]
    residual_note = f"; residuals in {{{', '.join(residuals_present)}}}" if residuals_present else ''
    ax.set_title(f'Level 2: metrics by {axis} (both sides matching{residual_note})')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(handles=legend_handles, loc='lower right')
    plt.tight_layout()
    out = results_dir / f'level2_by_{axis}.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Level 2 ({axis}) plot to: {out}")


def write_stratified_eval_summary(level1_df: pd.DataFrame, level2: dict, results_dir: Path) -> None:
    """Dump a short human-readable summary.md combining Level 1 and Level 2 tables.

    Intended for a reviewer to glance at per-run before running any
    cross-sweep aggregation. CSVs are the source of truth; this summary is
    for convenience only.
    """
    out = results_dir / 'stratified_eval_summary.md'
    lines = ['# Stratified evaluation summary', '', '## Level 1: pair regime', '']
    lines.append(level1_df.round(3).to_markdown(index=False))
    lines.append('')
    for axis_label, l2_df in level2.items():
        if l2_df is None or l2_df.empty:
            continue
        lines.append(f'## Level 2: {axis_label}')
        lines.append('')
        lines.append(l2_df.round(3).to_markdown(index=False))
        lines.append('')
    out.write_text('\n'.join(lines))
    print(f"Saved stratified eval summary to: {out}")


# ============================================================================
# Legacy error-analysis entry point (preserved filenames, corrected semantics)
# ============================================================================


def analyze_errors_by_metadata(df, y_true, y_pred, y_prob, results_dir: Path):
    """Stratified error analysis by host, hn_subtype, and year.

    This is the historical entry point. It preserves the output filenames
    `error_analysis_by_{host,hn_subtype,year}.csv` and their plots, and in
    addition emits the Level 1 (pair-regime) and Level 2 (per-axis with
    mixed/unknown buckets) tables and plots defined in
    docs/post_hoc_analysis_design.md.

    Semantic change vs the pre-2026-04 implementation:
      1. Metadata is merged on BOTH sides of each pair (assembly_id_a AND
         assembly_id_b), then a pair is counted in stratum V only when both
         sides share V. The old code joined on side A only, silently dropping
         side-B metadata for cross-stratum negatives.
      2. The 'year' axis in the legacy CSV uses year bins (see YEAR_BIN_EDGES)
         instead of raw year values, to keep per-fold strata dense.

    Args:
        df: DataFrame with predictions (must have assembly_id_a, assembly_id_b,
            label, pred_label, pred_prob).
        y_true, y_pred, y_prob: accepted for backwards-compatible signature;
            the function reads columns directly from df.
        results_dir: Where to write CSVs and PNGs.

    Returns:
        Dict keyed by 'host' / 'hn_subtype' / 'year' with per-stratum DataFrames.
    """
    print('\n' + '=' * 60)
    print('ERROR ANALYSIS BY METADATA (two-sided)')
    print('=' * 60)

    try:
        df_enriched = _enrich_pairs_with_metadata(df)
    except Exception as e:
        # Without metadata we cannot stratify at all; the rest of the pipeline
        # keeps running with basic metrics only.
        print(f"WARNING: Could not enrich with metadata: {e}")
        print("   Skipping metadata-stratified error analysis (Level 1/2 also skipped).")
        return {}

    # Legacy-schema CSVs: one row per value where BOTH sides agree on that
    # value. Pairs with mismatched sides (mixed) and unknown-side pairs are
    # simply excluded here, which keeps the legacy CSV schema unchanged.
    # Level 2 (below) reports mixed/unknown as explicit strata instead.
    legacy_axes = {
        'host': ('host_a', 'host_b'),
        'hn_subtype': ('subtype_a', 'subtype_b'),
        'year': ('year_bin_a', 'year_bin_b'),
    }
    all_stratified_stats = {}
    for axis_name, (col_a, col_b) in legacy_axes.items():
        sub = df_enriched.copy()
        va, vb = sub[col_a].astype(str), sub[col_b].astype(str)
        same_mask = (va == vb) & (va != 'unknown') & (va != 'nan')
        sub = sub.loc[same_mask].copy()
        sub['_stratum'] = va[same_mask]

        rows = []
        for value, group in sub.groupby('_stratum'):
            if len(group) < 5:
                continue
            m = _compute_stratum_metrics(group)
            m[axis_name] = value
            rows.append(m)
        if not rows:
            print(f"   No sufficient data for {axis_name} analysis (both-sides matching).")
            continue

        stats_df = pd.DataFrame(rows).sort_values('error_rate', ascending=False, na_position='last')
        cols = [axis_name] + [c for c in stats_df.columns if c != axis_name]
        stats_df = stats_df[cols]
        all_stratified_stats[axis_name] = stats_df
        out = results_dir / f'error_analysis_by_{axis_name}.csv'
        stats_df.to_csv(out, index=False)
        print(f"   Saved stratified metrics to: {out}")
        print(f"   Analyzed {len(stats_df)} unique {axis_name} values (both-sides matching)")

    if all_stratified_stats:
        _plot_errors_by_metadata(all_stratified_stats, results_dir)

    # Level 1 / Level 2 (new outputs) — emitted from the same enriched df.
    level1_df = analyze_level1_pair_regime(df_enriched, results_dir)
    level2 = {}
    for axis in ('host', 'subtype', 'year_bin'):
        level2[axis] = analyze_level2_by_axis(df_enriched, axis, results_dir)
    write_stratified_eval_summary(level1_df, level2, results_dir)

    return all_stratified_stats


def _plot_errors_by_metadata(stratified_stats: dict, results_dir: Path):
    """
    Create plots showing error rates and performance metrics by metadata.
    
    Creates two separate figures:
    1. Error rates (FP rate, FN rate) by metadata
    2. Performance metrics (F1 Score, AUC-ROC, AUC-PR) by metadata
    
    Args:
        stratified_stats: Dictionary with keys 'host', 'hn_subtype', 'year', each containing a DataFrame
        results_dir: Directory to save plots
    """
    metadata_vars = ['host', 'hn_subtype', 'year']
    var_labels = {'host': 'Host', 'hn_subtype': 'HN Subtype', 'year': 'Year'}
    
    # Get available variables
    available_vars = [v for v in metadata_vars if v in stratified_stats]
    n_vars = len(available_vars)
    if n_vars == 0:
        return
    
    # ========================================================================
    # Figure 1: Error Rates (FP Rate, FN Rate)
    # ========================================================================
    fig1, axes1 = plt.subplots(n_vars, 1, figsize=(14, 5 * n_vars))
    if n_vars == 1:
        axes1 = [axes1]
    
    for plot_idx, var in enumerate(available_vars):
        stats_df = stratified_stats[var]
        
        # Filter to top N by sample size (for readability)
        top_n = min(15, len(stats_df))
        stats_df_plot = stats_df.nlargest(top_n, 'n_samples')
        
        ax = axes1[plot_idx]
        x_pos = np.arange(len(stats_df_plot))
        width = 0.35
        
        # FP Rate bars (color-only, no hatch; see commit message for rationale)
        ax.bar(x_pos - width/2, stats_df_plot['fp_rate'], width,
               label='FP Rate', color='red', alpha=0.8, edgecolor='black', linewidth=1.5)

        # FN Rate bars
        ax.bar(x_pos + width/2, stats_df_plot['fn_rate'], width,
               label='FN Rate', color='blue', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        var_label = var_labels[var]
        ax.set_xlabel(var_label, fontsize=12)
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_title(f'Error Rates by {var_label} (Top {top_n} by sample size)', 
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in stats_df_plot[var]], rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        max_rate = max(stats_df_plot[['fp_rate', 'fn_rate']].max())
        ax.set_ylim(0, max_rate * 1.15)
    
    plt.tight_layout()
    output_path1 = results_dir / 'error_rates_by_metadata.png'
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved error rates plot to: {output_path1}")
    
    # ========================================================================
    # Figure 2: Performance Metrics (F1 Score, AUC-ROC, AUC-PR)
    # ========================================================================
    fig2, axes2 = plt.subplots(n_vars, 1, figsize=(14, 5 * n_vars))
    if n_vars == 1:
        axes2 = [axes2]
    
    for plot_idx, var in enumerate(available_vars):
        stats_df = stratified_stats[var]
        
        # Filter to top N by sample size (for readability)
        top_n = min(15, len(stats_df))
        stats_df_plot = stats_df.nlargest(top_n, 'n_samples')
        
        ax = axes2[plot_idx]
        x_pos = np.arange(len(stats_df_plot))
        width = 0.25
        
        # F1 Score bars (color-only, no hatch)
        ax.bar(x_pos - width, stats_df_plot['f1_score'], width,
               label='F1 Score', color='purple', alpha=0.8, edgecolor='black', linewidth=1.5)

        # AUC-ROC bars
        ax.bar(x_pos, stats_df_plot['auc_roc'], width,
               label='AUC-ROC', color='teal', alpha=0.8, edgecolor='black', linewidth=1.5)

        # AUC-PR bars
        ax.bar(x_pos + width, stats_df_plot['auc_pr'], width,
               label='AUC-PR', color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        var_label = var_labels[var]
        ax.set_xlabel(var_label, fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Performance Metrics by {var_label} (Top {top_n} by sample size)', 
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in stats_df_plot[var]], rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    output_path2 = results_dir / 'performance_metrics_by_metadata.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance metrics plot to: {output_path2}")


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

    # Performance by segment pair (same metrics as metrics.csv for consistency)
    seg_stats = []
    for seg_pair in df['seg_pair'].unique():
        subset = df[df['seg_pair'] == seg_pair]
        if len(subset) > 10:  # Only analyze pairs with sufficient data
            # Check if both classes are present for F1, AUC, AP, Brier
            unique_labels = subset['label'].unique()
            if len(unique_labels) > 1:
                f1 = f1_score(subset['label'], subset['pred_label'], average='binary')
                auc = roc_auc_score(subset['label'], subset['pred_prob'])
                avg_precision = average_precision_score(subset['label'], subset['pred_prob'])
                brier_score = float(np.mean((subset['pred_prob'].values - subset['label'].values) ** 2))
            else:
                f1 = np.nan
                auc = np.nan
                avg_precision = np.nan
                brier_score = np.nan

            acc = (subset['label'] == subset['pred_label']).mean()

            seg_stats.append({
                'seg_pair': seg_pair,
                'count': len(subset),
                'accuracy': acc,
                'f1_score': f1,
                'auc_roc': auc,
                'avg_precision': avg_precision,
                'brier_score': brier_score,
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


def main(config_bundle: str,
         model_dir: Optional[Path] = None,
         create_performance_plots: bool = True,
         show_confusion_labels: bool = True
    ) -> None:
    """Main analysis function."""
    print(f"\n{'='*50}")
    print('Analyze Segment Pair Classifier Results v2.')
    print('='*50)

    # Load config
    config_path = str(project_root / 'conf')
    config = get_virus_config_hydra(config_bundle, config_path=config_path)
    # print_config_summary(config)
    print(f"Using config bundle: {config_bundle}")
    print(f"Virus: {config.virus.virus_name}")
    print(f"Data version: {config.virus.data_version}")

    # Determine model directory
    if model_dir:
        # Use explicitly provided model directory
        models_dir = Path(model_dir)
        print(f"Using provided model directory: {models_dir}")
    else:
        # Use standard Hydra path (base directory)
        paths = build_training_paths(
            project_root=project_root,
            virus_name=config.virus.virus_name,
            data_version=config.virus.data_version,
            # run_suffix=config.run_suffix if hasattr(config, 'run_suffix') and config.run_suffix else "",
            run_suffix="",  # Not used - kept for backward compatibility
            config=config
        )
        base_models_dir = paths['output_dir']
        
        # Look for most recent training run in runs/ subdirectory
        runs_dir = base_models_dir / 'runs'
        if runs_dir.exists():
            # Find most recent training run matching this config bundle
            training_runs = sorted(
                [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(f'training_{config_bundle}_')],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if training_runs:
                models_dir = training_runs[0]
                print(f"Using most recent training run: {models_dir}")
            else:
                # Fallback: use most recent training run regardless of bundle
                all_runs = sorted(
                    [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('training_')],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                if all_runs:
                    models_dir = all_runs[0]
                    print(f"WARNING: No run found for bundle '{config_bundle}', using most recent run: {models_dir}")
                else:
                    models_dir = base_models_dir
                    print(f"WARNING: No training runs found, using base directory: {models_dir}")
        else:
            models_dir = base_models_dir
            print(f"Using base models directory: {models_dir}")
    
    # Write analysis artifacts under the training run dir, colocated with
    # test_predicted.csv and best_model.pt. Self-contained per-run artifacts.
    results_dir = models_dir / 'post_hoc'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load test results
    test_results_file = models_dir / 'test_predicted.csv'
    if not test_results_file.exists():
        raise FileNotFoundError(
            f'ERROR: Test results file not found: {test_results_file}\n'
            f'   Model directory: {models_dir}\n'
            f'   Hint: Use --model_dir to specify the exact training run directory'
        )

    df = pd.read_csv(test_results_file)
    print(f'Loaded {len(df)} test predictions')

    # Extract labels and predictions
    y_true = df['label'].values
    y_pred = df['pred_label'].values  
    y_prob = df['pred_prob'].values

    # Set up plotting style
    apply_default_style()
    
    # Make variables globally available for the analysis functions
    globals()['df'] = df
    globals()['y_true'] = y_true
    globals()['y_pred'] = y_pred
    globals()['y_prob'] = y_prob
    globals()['results_dir'] = results_dir
    
    # Basic metrics
    metrics = compute_basic_metrics(y_true, y_pred, y_prob)
    
    # Generate plots
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        y_true, y_pred,
        results_dir / 'confusion_matrix.png',
        show_labels=show_confusion_labels
    )
    
    # Test simple confusion matrix to check for lines
    plot_confusion_matrix_simple(y_true, y_pred, results_dir / 'confusion_matrix.png')
    
    # Save confusion matrix to CSV
    cm_df = pd.DataFrame(
        cm,
        index=['True Negative', 'True Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )
    # Add row/column labels for clarity
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    cm_df.to_csv(results_dir / 'confusion_matrix.csv')
    print(f"Confusion matrix saved to: {results_dir / 'confusion_matrix.csv'}")
    
    # Also save in TP/TN/FP/FN format
    tn, fp, fn, tp = cm.ravel()
    cm_summary = pd.DataFrame({
        'metric': ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)'],
        'count': [tn, fp, fn, tp],
        'percentage': [tn/cm.sum()*100, fp/cm.sum()*100, fn/cm.sum()*100, tp/cm.sum()*100]
    })
    cm_summary.to_csv(results_dir / 'confusion_matrix_summary.csv', index=False)
    print(f"Confusion matrix summary saved to: {results_dir / 'confusion_matrix_summary.csv'}")
    
    # Optional performance plots
    if create_performance_plots:
        plot_roc_curve(y_true, y_prob, results_dir / 'roc_curve.png')
        plot_precision_recall_curve(y_true, y_prob, results_dir / 'precision_recall_curve.png')
        plot_prediction_distribution(y_true, y_prob, results_dir / 'prediction_distribution.png')
        print("Performance plots created (roc_curve.png, precision_recall_curve.png, prediction_distribution.png)")
    else:
        print("Skipping performance plots (create_performance_plots=False)")
    
    # Enhanced FP/FN analysis
    error_summary = analyze_fp_fn_errors(df, y_true, y_pred, y_prob, results_dir)
    
    # Metadata-stratified error analysis
    metadata_error_stats = analyze_errors_by_metadata(df, y_true, y_pred, y_prob, results_dir)
    
    # Domain-specific analyses
    segment_df = analyze_segment_performance(df)
    analyze_same_function_pairs(df)
    analyze_prediction_confidence(df)

    # Save summary metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / 'metrics.csv', index=False)

    # Save segment metrics
    if 'segment_df' in locals():
        segment_df.to_csv(results_dir / 'segment_metrics.csv', index=False)
    
    print(f'\nAnalysis complete! Results saved to: {results_dir}')

    # Key insights summary
    print(f"\n{'='*60}")
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
    parser = argparse.ArgumentParser(description='Analyze Stage 4: Training results from segment pair classifier')
    parser.add_argument(
        '--config_bundle',
        type=str, required=True,
        help='Configuration bundle name (e.g., flu_a_3p_1ks, bunya)'
    )
    parser.add_argument(
        '--model_dir',
        type=str, default=None,
        help='Explicit model directory path (overrides config-based path)'
    )
    parser.add_argument(
        '--create_performance_plots',
        action='store_true', default=True,
        help='Create performance plots (roc, precision-recall, prediction distribution)'
    )
    parser.add_argument(
        '--show_confusion_labels',
        action='store_true', default=True,
        help='Show confusion matrix labels (TP, TN, FP, FN)'
    )
    args = parser.parse_args()
    
    main(
        config_bundle=args.config_bundle,
        model_dir=args.model_dir,
        create_performance_plots=args.create_performance_plots,
        show_confusion_labels=args.show_confusion_labels
    )
