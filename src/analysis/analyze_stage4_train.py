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
    precision_recall_curve, f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, matthews_corrcoef,
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
from src.utils.plot_config import apply_default_style
from src.analysis.plot_calibration_curve import plot_calibration_curve


def compute_basic_metrics(y_true, y_pred, y_prob):
    """Compute and display basic classification metrics.

    All metrics are computed on the test set (from test_predicted.csv).
    """
    print('\n' + '='*50)
    print('BASIC CLASSIFICATION METRICS')
    print('='*50)
    
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    accuracy = (y_true == y_pred).mean()
    auc_pr = average_precision_score(y_true, y_prob)
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
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'MCC: {mcc:.3f}')
    print(f'AUC-ROC: {auc_roc:.3f}')
    print(f'AUC-PR: {auc_pr:.3f}')
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
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'brier_score': brier_score,
        'loss': bce_loss
    }


def analyze_metrics_summary(metrics: dict, results_dir,
                            save_name: str = 'metrics.png') -> None:
    """Bar chart of headline binary classification metrics.

    Five bars: F1 (binary), F1 (macro), AUC-ROC, AUC-PR, MCC. Drops
    Accuracy (uninformative on class-imbalanced binary) and threshold-
    independent ranking metrics that overlap with AUC.

    Reads from a metrics dict (the return value of
    ``compute_basic_metrics``) -- the caller is responsible for
    computing it on the appropriate predictions (test_predicted.csv or
    test_predicted_swapped.csv).

    MCC lives in [-1, 1] but is non-negative for any reasonable model.
    The y-axis is fixed at [0, 1.05] for visual parity with the [0,1]
    metrics; a negative MCC bar would clip at the axis -- the printed
    value-label still shows the true value.
    """
    metric_names = ['F1 (binary)', 'F1 (macro)', 'AUC-ROC', 'AUC-PR', 'MCC']
    values = [
        metrics['f1_score'],
        metrics['f1_macro'],
        metrics['auc_roc'],
        metrics['auc_pr'],
        metrics['mcc'],
    ]
    colors = ['#2E86AB', '#A23B72', '#6A4C93', '#F18F01', '#C73E1D']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(metric_names, values, color=colors, alpha=0.85, edgecolor='black')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics', fontsize=13, fontweight='bold')
    for bar, value in zip(bars, values):
        # Place the label just above the visible top of the bar; clip at 0 so
        # negative values still get a label at axis level rather than off-canvas.
        h = max(bar.get_height(), 0.0)
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01,
                f'{value:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = results_dir / save_name
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics summary plot to: {out}")


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


def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC-ROC = {auc_roc:.3f})')
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
    auc_pr = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC-PR = {auc_pr:.3f})')

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                label=f'Random (AUC-PR = {baseline:.3f})')
    
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


def plot_prediction_distribution(y_true, y_prob, save_path=None, ax=None):
    """Plot distribution of prediction probabilities by true label.

    If `ax` is provided, render onto it without creating a figure or saving;
    the caller owns layout and saving. Used as a sub-panel by
    `create_model_calibration_plot` in `create_presentation_plots.py`.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]

    ax.hist(neg_probs, bins=30, alpha=0.7, label='Negative (Different Isolate)',
            color='#E74C3C', density=True)
    ax.hist(pos_probs, bins=30, alpha=0.7, label='Positive (Same Isolate)',
            color='#3498DB', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.8,
               label='Decision Threshold')
    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Prediction Probabilities',
                 fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def _axis_columns(axis: str) -> tuple[str, str]:
    """Return the (col_a, col_b) pair for a logical axis name.

    Axes follow Level 1/2 conventions: 'subtype' uses the parsed canonical
    form (subtype_a/_b after enrichment), 'year_bin' uses the binned form,
    'host' / 'geo_location' / 'passage' use the v2-attached raw columns.
    """
    if axis == 'subtype':
        return 'subtype_a', 'subtype_b'
    if axis == 'year_bin':
        return 'year_bin_a', 'year_bin_b'
    return f'{axis}_a', f'{axis}_b'


def _within_cross_masks(df_enriched: pd.DataFrame, axis: str) -> tuple[pd.Series, pd.Series]:
    """Return (within_mask, cross_mask) for a given axis on enriched pairs.

    Pairs are excluded from BOTH masks when either side is missing or
    'unknown' on that axis -- we only ask within/cross questions when both
    sides actually have a parseable value.
    """
    col_a, col_b = _axis_columns(axis)
    a = df_enriched[col_a].astype('object')
    b = df_enriched[col_b].astype('object')
    known = a.notna() & b.notna() & (a != 'unknown') & (b != 'unknown')
    within = known & (a == b)
    cross = known & (a != b)
    return within, cross


def plot_neg_prob_by_axis(df_enriched: pd.DataFrame, results_dir: Path,
                          axes: tuple = ('subtype', 'host', 'year_bin')) -> None:
    """Pred-prob distributions of negatives, split into within-axis vs cross-axis.

    For each axis, plot two overlaid histograms over `pred_prob`:
      - Within-axis: negative pairs where both sides share the axis value
        (e.g., both H3N2). These are the "hard" negatives.
      - Cross-axis: negative pairs where sides differ (e.g., H3N2 vs H1N1).
        The "easy" negatives.

    A clear right-shift of the within-axis distribution past 0.5 is direct
    visual evidence the model uses that axis as a shortcut: same-axis pairs
    look more like positives.

    Saves `neg_prob_distribution_by_axis.png`. No CSV output -- this is a
    visual companion to Level 1/2 numerical tables.
    """
    print('\n' + '=' * 60)
    print('NEGATIVE PRED-PROB DISTRIBUTION BY AXIS')
    print('=' * 60)

    neg = df_enriched[df_enriched['label'] == 0]
    if len(neg) == 0:
        print('  No negative pairs in test set; skipping.')
        return

    fig, axes_arr = plt.subplots(1, len(axes), figsize=(5.5 * len(axes), 4.5),
                                  sharey=True)
    if len(axes) == 1:
        axes_arr = [axes_arr]

    for ax_obj, axis in zip(axes_arr, axes):
        within, cross = _within_cross_masks(neg, axis)
        within_probs = neg.loc[within, 'pred_prob'].values
        cross_probs = neg.loc[cross, 'pred_prob'].values
        n_within = len(within_probs)
        n_cross = len(cross_probs)

        # FP rates per stratum (FP = predicted positive among true negatives).
        fp_within = (within_probs > 0.5).mean() if n_within > 0 else float('nan')
        fp_cross = (cross_probs > 0.5).mean() if n_cross > 0 else float('nan')

        if n_cross > 0:
            ax_obj.hist(cross_probs, bins=30, alpha=0.65,
                        label=f'cross-{axis} (n={n_cross:,}, FP={fp_cross:.1%})',
                        color='steelblue', density=True, range=(0, 1))
        if n_within > 0:
            ax_obj.hist(within_probs, bins=30, alpha=0.65,
                        label=f'within-{axis} (n={n_within:,}, FP={fp_within:.1%})',
                        color='crimson', density=True, range=(0, 1))
        ax_obj.axvline(0.5, color='black', linestyle='--', alpha=0.7,
                       label='Threshold 0.5')
        ax_obj.set_xlabel('pred_prob')
        ax_obj.set_title(f'Axis: {axis}', fontweight='bold')
        ax_obj.legend(loc='upper center', fontsize=8)
        ax_obj.grid(True, alpha=0.3)
        ax_obj.set_xlim(0, 1)
        print(f'  {axis:14s} within FP={fp_within:.3f} (n={n_within:,})  '
              f'cross FP={fp_cross:.3f} (n={n_cross:,})')

    axes_arr[0].set_ylabel('Density')
    fig.suptitle('Negative pairs: pred_prob distribution within vs cross category',
                 fontweight='bold')
    plt.tight_layout()
    out = results_dir / 'neg_prob_distribution_by_axis.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved to: {out}')


def analyze_fp_fn_errors(df, y_true, y_pred, y_prob, results_dir: Path):
    """Detailed analysis of False Positives and False Negatives.

    Saves `fp_fn_analysis.png` (two confidence panels) plus
    `error_analysis_summary.csv`, `false_positives_detailed.csv`, and
    `false_negatives_detailed.csv`.

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

    # Add prediction columns for easier analysis
    df_analysis = df.copy()
    df_analysis['y_true'] = y_true
    df_analysis['y_pred'] = y_pred
    df_analysis['y_prob'] = y_prob

    # Identify FP and FN
    fp_mask = (df_analysis['y_true'] == 0) & (df_analysis['y_pred'] == 1)
    fn_mask = (df_analysis['y_true'] == 1) & (df_analysis['y_pred'] == 0)

    fp_df = df_analysis[fp_mask].copy()
    fn_df = df_analysis[fn_mask].copy()

    print(f'False Positives (FP): {len(fp_df)}')
    print(f'False Negatives (FN): {len(fn_df)}')
    if len(fn_df) > 0:
        print(f'FP/FN ratio: {len(fp_df)/len(fn_df):.2f}')
    else:
        print(f'FP/FN ratio: inf (no false negatives)')

    # Confidence analysis
    print(f'\nConfidence Analysis:')
    if len(fp_df) > 0:
        print(f'FP average confidence: {fp_df["y_prob"].mean():.3f}')
        print(f'FP confidence std:     {fp_df["y_prob"].std():.3f}')
    else:
        print('FP average confidence: N/A (no false positives)')
    if len(fn_df) > 0:
        print(f'FN average confidence: {fn_df["y_prob"].mean():.3f}')
        print(f'FN confidence std:     {fn_df["y_prob"].std():.3f}')
    else:
        print('FN average confidence: N/A (no false negatives)')

    # Two confidence views: histogram (left) and boxplot (right). Earlier
    # 2x2 grid included segment-pair and protein-function bar charts that
    # are degenerate in v2 (one schema_pair => one bar each); removed.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Confidence distribution histogram
    if len(fp_df) > 0:
        axes[0].hist(fp_df['y_prob'], bins=20, alpha=0.7,
                     label=f'False Positives (n={len(fp_df)})',
                     color='red', density=False)
    if len(fn_df) > 0:
        axes[0].hist(fn_df['y_prob'], bins=20, alpha=0.7,
                     label=f'False Negatives (n={len(fn_df)})',
                     color='blue', density=False)
    axes[0].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    axes[0].set_xlabel('Prediction Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution: FP and FN', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Confidence boxplot
    if len(fp_df) > 0 and len(fn_df) > 0:
        axes[1].boxplot([fp_df['y_prob'], fn_df['y_prob']],
                        labels=['False Positives', 'False Negatives'])
        axes[1].set_title('Confidence Distribution: FP and FN', fontweight='bold')
    elif len(fp_df) > 0:
        axes[1].boxplot([fp_df['y_prob']], labels=['False Positives'])
        axes[1].set_title('Confidence Distribution: FP Only', fontweight='bold')
    elif len(fn_df) > 0:
        axes[1].boxplot([fn_df['y_prob']], labels=['False Negatives'])
        axes[1].set_title('Confidence Distribution: FN Only', fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No errors found', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Confidence Distribution: No Errors', fontweight='bold')
    axes[1].set_ylabel('Prediction Probability')
    axes[1].grid(True, alpha=0.3)

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


_ENRICH_BASE_COLS = ['host_a', 'host_b', 'hn_subtype_a', 'hn_subtype_b',
                     'year_a', 'year_b']


def _enrich_pairs_with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-side parsed metadata columns to a pair-level dataframe.

    Output columns added (or refreshed) on a copy of `df`:
      subtype_a, subtype_b   -- canonical H<d>N<d> form, 'unknown' fallback
      year_bin_a, year_bin_b -- coarse bin label (see YEAR_BIN_EDGES)

    v2-attached pair DataFrames already carry host_{a,b}, hn_subtype_{a,b},
    year_{a,b} (and same_* flags). When all the needed base columns are present
    we derive subtype/year_bin in place. Otherwise (older v1 outputs) we fall
    back to merging the isolate_metadata table, which raises if missing.
    """
    out = df.copy()
    have_all_base = all(c in out.columns for c in _ENRICH_BASE_COLS)

    if not have_all_base:
        from src.utils.metadata_enrichment import load_flu_metadata

        meta = load_flu_metadata(project_root=project_root)[
            ['assembly_id', 'host', 'hn_subtype', 'year']]
        out['assembly_id_a'] = out['assembly_id_a'].astype(str)
        out['assembly_id_b'] = out['assembly_id_b'].astype(str)
        for side in ('a', 'b'):
            renamed = meta.rename(columns={
                'assembly_id': f'assembly_id_{side}',
                'host': f'host_{side}',
                'hn_subtype': f'hn_subtype_{side}',
                'year': f'year_{side}',
            })
            out = out.merge(renamed, on=f'assembly_id_{side}', how='left')

    # Vectorized derivations from the base columns -- 2x str.match (subtype)
    # and 2x cut (year bin); no per-row Python.
    for side in ('a', 'b'):
        raw = out[f'hn_subtype_{side}'].astype('string')
        canonical = raw.where(raw.str.match(SUBTYPE_RE, na=False), other='unknown')
        out[f'subtype_{side}'] = canonical.fillna('unknown').astype(str)

        years = pd.to_numeric(out[f'year_{side}'], errors='coerce')
        bins = pd.cut(years,
                      bins=[-np.inf, 2015, 2020, np.inf],
                      labels=['<=2015', '2016-2020', '2021+'])
        out[f'year_bin_{side}'] = bins.astype(object).where(bins.notna(), 'unknown')

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


# ----------------------------------------------------------------------------
# Level 1: per-regime stratified evaluation over the 9-regime taxonomy from
# the v2 metadata-aware negative sampler. Replaces the prior 4-bucket
# (positive / within_subtype_neg / cross_subtype_neg / unknown_neg) view.
#
# See:
#   docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md
#   src/datasets/_negative_regime_sampling.py
# ----------------------------------------------------------------------------

# Display order for the per-regime view. Positive first, then negatives by
# ascending hardness (none-match easiest, all-three-match hardest).
_LEVEL1_REGIME_ORDER = (
    'positive',
    'none_match',
    'host_only',
    'subtype_only',
    'year_only',
    'host_subtype_only',
    'host_year_only',
    'subtype_year_only',
    'host_subtype_year',
)


def _derive_neg_regime(df: pd.DataFrame) -> pd.Series:
    """Per-row regime label, derived from per-side raw metadata.

    Matches the v2 sampler's classification logic
    (`src.datasets._negative_regime_sampling.classify_pair_regime`):
    per-axis comparison treats null on either side as no-match (including
    null == null) -- the resulting (host_match, subtype_match,
    year_bin_match) tuple maps to one of the 8 regimes via
    `_MATCH_TUPLE_TO_REGIME`. `bin_year` is reused so binning matches the
    sampler exactly.

    Used as a fallback when `neg_regime` is missing or null on a row
    (legacy datasets pre-regime-aware sampling). The 2026-05-11 removal of
    the `unknown_metadata_neg` regime simplified the mapping: pairs with
    null axes simply land in whichever of the 8 regimes captures the
    non-null axes that do match.
    """
    from src.datasets._negative_regime_sampling import bin_year, _MATCH_TUPLE_TO_REGIME

    def _col(name):
        if name in df.columns:
            return df[name]
        return pd.Series(pd.NA, index=df.index, dtype='object')

    host_a, host_b = _col('host_a'), _col('host_b')
    sub_a, sub_b = _col('hn_subtype_a'), _col('hn_subtype_b')
    year_a_raw, year_b_raw = _col('year_a'), _col('year_b')
    year_a = year_a_raw.apply(lambda y: bin_year(y) if pd.notna(y) else None)
    year_b = year_b_raw.apply(lambda y: bin_year(y) if pd.notna(y) else None)

    # Per-axis match: True iff both sides are non-null AND equal. pandas '=='
    # already returns False on NaN==NaN and on NaN==value, so we only need to
    # explicitly suppress matches when either side is NA.
    host_match = (host_a == host_b) & host_a.notna() & host_b.notna()
    sub_match = (sub_a == sub_b) & sub_a.notna() & sub_b.notna()
    year_match = (year_a == year_b) & year_a.notna() & year_b.notna()

    # Default everything to 'none_match' then refine; the loop below covers
    # all 8 (h, s, y) bool tuples so every row gets an assignment.
    regime = pd.Series('none_match', index=df.index, dtype='object')
    for (h, s, y), name in _MATCH_TUPLE_TO_REGIME.items():
        mask = (host_match == h) & (sub_match == s) & (year_match == y)
        regime[mask] = name
    return regime


def _resolve_neg_regime_column(df: pd.DataFrame) -> pd.Series:
    """Per-row regime label for the level1 plots.

    Positive rows -> 'positive'. Negative rows -> the v2-written `neg_regime`
    column when present and non-null on the row; otherwise the derived value
    from `_derive_neg_regime`. The two paths agree on regime-aware datasets;
    the fallback is only exercised on legacy datasets (no `neg_regime`
    column) or on rows that the sampler classified as `unknown_metadata_neg`
    and serialized as null.
    """
    out = pd.Series('positive', index=df.index, dtype='object')
    neg_mask = (df['label'] == 0)
    if not neg_mask.any():
        return out

    if 'neg_regime' in df.columns:
        sampler_vals = df.loc[neg_mask, 'neg_regime']
        has_val = sampler_vals.notna()
        out.loc[sampler_vals[has_val].index] = sampler_vals[has_val].astype(str)
        derive_idx = sampler_vals[~has_val].index
    else:
        derive_idx = df.index[neg_mask]

    if len(derive_idx) > 0:
        out.loc[derive_idx] = _derive_neg_regime(df.loc[derive_idx])
    return out


def _resolve_match_count_column(df: pd.DataFrame) -> pd.Series:
    """Per-row metadata-match count (0..3) for negatives, pd.NA for positives.

    Prefers the v2-written `metadata_match_count` column when present;
    derives from the resolved regime (number of axes that match) otherwise.
    """
    out = pd.Series(pd.NA, index=df.index, dtype='Int64')
    neg_mask = (df['label'] == 0)
    if not neg_mask.any():
        return out

    if 'metadata_match_count' in df.columns:
        sampler_vals = pd.to_numeric(df.loc[neg_mask, 'metadata_match_count'],
                                     errors='coerce').astype('Int64')
        has_val = sampler_vals.notna()
        out.loc[sampler_vals[has_val].index] = sampler_vals[has_val]
        derive_idx = sampler_vals[~has_val].index
    else:
        derive_idx = df.index[neg_mask]

    if len(derive_idx) > 0:
        regimes = _derive_neg_regime(df.loc[derive_idx])
        regime_to_count = {
            'none_match': 0,
            'host_only': 1, 'subtype_only': 1, 'year_only': 1,
            'host_subtype_only': 2, 'host_year_only': 2, 'subtype_year_only': 2,
            'host_subtype_year': 3,
        }
        for idx, r in regimes.items():
            out.loc[idx] = regime_to_count[r]
    return out


def analyze_level1_neg_regimes(df_enriched: pd.DataFrame,
                               results_dir: Path) -> pd.DataFrame:
    """Level 1: per-regime TPR (positive) / TNR (negatives) over the
    8-regime taxonomy from the v2 metadata-aware negative sampler.

    Reads `neg_regime` from the predictions df when present; falls back to
    deriving from host_a/_b, hn_subtype_a/_b, year_a/_b when not.
    """
    print('\n' + '=' * 60)
    print('LEVEL 1: per-regime TPR / TNR (8-regime taxonomy)')
    print('=' * 60)

    df = df_enriched.copy()
    df['regime'] = _resolve_neg_regime_column(df)

    rows = []
    for regime in _LEVEL1_REGIME_ORDER:
        sub = df[df['regime'] == regime]
        if len(sub) == 0:
            rows.append({'regime': regime, 'n_samples': 0})
            continue
        m = _compute_stratum_metrics(sub)
        m['regime'] = regime
        rows.append(m)

    stats_df = pd.DataFrame(rows)
    cols = ['regime'] + [c for c in stats_df.columns if c != 'regime']
    stats_df = stats_df[cols]

    output_file = results_dir / 'level1_neg_regimes.csv'
    stats_df.to_csv(output_file, index=False)
    print(f"Saved Level 1 (neg regimes) metrics to: {output_file}")
    print(stats_df.round(3).to_string(index=False))

    _plot_level1_neg_regimes(stats_df, results_dir)
    return stats_df


def _plot_level1_neg_regimes(stats_df: pd.DataFrame, results_dir: Path) -> None:
    """Bar plot of per-regime TPR (positive) / TNR (negatives).

    Positive bar: seagreen (matches the existing TPR color elsewhere).
    Negative bars: crimson (matches error_by_match_count.png).
    """
    plot_df = stats_df[stats_df['n_samples'] > 0].copy().reset_index(drop=True)
    if plot_df.empty:
        return

    pos_color = 'seagreen'
    neg_color = 'crimson'

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(plot_df))
    width = 0.5
    for xi, row in zip(x, plot_df.itertuples(index=False)):
        if row.regime == 'positive':
            v, color = row.tpr, pos_color
        else:
            v, color = row.tnr, neg_color
        if pd.isna(v):
            continue
        ax.bar(xi, v, width, color=color, edgecolor='black', alpha=0.85)
        ax.text(xi, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=pos_color, edgecolor='black', label='TPR (positive)'),
        Patch(facecolor=neg_color, edgecolor='black', label='TNR (negative regimes)'),
    ], loc='lower right')

    for xi, n in zip(x, plot_df['n_samples']):
        ax.text(xi, 1.08, f'n={n:,}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['regime'], rotation=25, ha='right')
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel('Score')
    ax.set_title('Level 1: per-regime TPR / TNR')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = results_dir / 'level1_neg_regimes.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Level 1 plot to: {out}")


def analyze_level1_neg_regimes_agg(df_enriched: pd.DataFrame,
                                   results_dir: Path) -> pd.DataFrame:
    """Level 1: aggregated TPR (positive) + TNR by metadata-match count.

    Compact 4-bar decomposition that collapses the 8 metadata-defined
    regimes into match_count = 0/1/2/3. Reads `metadata_match_count` from
    the predictions df when present; falls back to deriving from per-side
    metadata.
    """
    print('\n' + '=' * 60)
    print('LEVEL 1: aggregated TPR / TNR by metadata-match count')
    print('=' * 60)

    df = df_enriched.copy()
    df['_match_count'] = _resolve_match_count_column(df)

    rows = []
    pos_sub = df[df['label'] == 1]
    if len(pos_sub) > 0:
        m = _compute_stratum_metrics(pos_sub)
        m['bucket'] = 'positive'
        rows.append(m)
    else:
        rows.append({'bucket': 'positive', 'n_samples': 0})

    for c in (0, 1, 2, 3):
        sub = df[(df['label'] == 0) & (df['_match_count'] == c)]
        if len(sub) == 0:
            rows.append({'bucket': f'match_count_{c}', 'n_samples': 0})
            continue
        m = _compute_stratum_metrics(sub)
        m['bucket'] = f'match_count_{c}'
        rows.append(m)

    stats_df = pd.DataFrame(rows)
    cols = ['bucket'] + [c for c in stats_df.columns if c != 'bucket']
    stats_df = stats_df[cols]

    output_file = results_dir / 'level1_neg_regimes_agg.csv'
    stats_df.to_csv(output_file, index=False)
    print(f"Saved Level 1 (aggregated) metrics to: {output_file}")
    print(stats_df.round(3).to_string(index=False))

    _plot_level1_neg_regimes_agg(stats_df, results_dir)
    return stats_df


def _plot_level1_neg_regimes_agg(stats_df: pd.DataFrame,
                                 results_dir: Path) -> None:
    """Bar plot of TPR (positive) + TNR (4 match-count buckets, optional unknown).

    Positive bar: seagreen, matches level1_neg_regimes.png (same baseline
    across both plots).
    Negative bars: indigo, deliberately distinct from level1_neg_regimes.png's
    crimson so it's obvious at-a-glance that this is a different
    decomposition. Alternatives considered: 'darkred' / 'firebrick' (same
    hue family, deeper), wine '#722f37' (near-burgundy), 'mediumpurple'
    (lighter purple).
    """
    plot_df = stats_df[stats_df['n_samples'] > 0].copy().reset_index(drop=True)
    if plot_df.empty:
        return

    pos_color = 'seagreen'
    neg_color = 'indigo'

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(plot_df))
    width = 0.5
    for xi, row in zip(x, plot_df.itertuples(index=False)):
        if row.bucket == 'positive':
            v, color = row.tpr, pos_color
        else:
            v, color = row.tnr, neg_color
        if pd.isna(v):
            continue
        ax.bar(xi, v, width, color=color, edgecolor='black', alpha=0.85)
        ax.text(xi, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=pos_color, edgecolor='black', label='TPR (positive)'),
        Patch(facecolor=neg_color, edgecolor='black', label='TNR (negative buckets)'),
    ], loc='lower right')

    for xi, n in zip(x, plot_df['n_samples']):
        ax.text(xi, 1.08, f'n={n:,}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['bucket'], rotation=25, ha='right')
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel('Score')
    ax.set_title('Level 1: aggregated TPR / TNR by metadata-match count')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = results_dir / 'level1_neg_regimes_agg.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Level 1 (aggregated) plot to: {out}")


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


def write_stratified_eval_summary(level1_regimes_df: pd.DataFrame,
                                  level1_agg_df: pd.DataFrame,
                                  level2: dict,
                                  results_dir: Path) -> None:
    """Dump a short human-readable summary.md combining Level 1 and Level 2 tables.

    Intended for a reviewer to glance at per-run before running any
    cross-sweep aggregation. CSVs are the source of truth; this summary is
    for convenience only.
    """
    out = results_dir / 'stratified_eval_summary.md'
    lines = ['# Stratified evaluation summary', '',
             '## Level 1: per-regime TPR / TNR (9-regime taxonomy)', '']
    lines.append(level1_regimes_df.round(3).to_markdown(index=False))
    lines.append('')
    lines.append('## Level 1: aggregated TPR / TNR by metadata-match count')
    lines.append('')
    lines.append(level1_agg_df.round(3).to_markdown(index=False))
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


def analyze_negative_hardness(
    df_enriched: pd.DataFrame,
    results_dir: Path,
    axes: tuple = ('subtype', 'host', 'year_bin', 'geo_location', 'passage'),
) -> tuple:
    """Decompose negative-pair errors by how many metadata axes match.

    For each negative pair, count how many of the given axes have BOTH
    sides equal AND non-missing. That count (0..len(axes)) is the
    "hardness" -- a negative where every axis matches looks identical to
    a positive on metadata, so the model has no metadata shortcut.

    Two outputs:
      - error_by_match_count.csv -- one row per match_count (0..N),
        aggregate FP rate, sample sizes, mean predicted probability.
        Headline: monotonic FP-rate climb with match_count = model uses
        metadata correlations as shortcuts.
      - error_by_match_pattern.csv -- one row per unique pattern (e.g.
        'host,subtype'), same metrics. Tells you WHICH axes drive the
        hardness, since match_count alone collapses different axis
        combinations into the same bucket.

    Plus error_by_match_count.png: bar chart of FP rate vs match_count.

    Returns (count_df, pattern_df).
    """
    print('\n' + '=' * 60)
    print('NEGATIVE HARDNESS (match-count + match-pattern)')
    print('=' * 60)

    neg = df_enriched[df_enriched['label'] == 0].copy()
    if len(neg) == 0:
        print('  No negative pairs in test set; skipping.')
        return pd.DataFrame(), pd.DataFrame()

    # For each axis, compute boolean same-flag (both sides present, both
    # non-unknown, equal). Vectorized -- no per-row Python.
    match_flags = pd.DataFrame(index=neg.index)
    valid_axes = []
    for axis in axes:
        col_a, col_b = _axis_columns(axis)
        if col_a not in neg.columns or col_b not in neg.columns:
            print(f'  Axis "{axis}" not available (missing {col_a}/{col_b}); skipping.')
            continue
        a = neg[col_a].astype('object')
        b = neg[col_b].astype('object')
        known = a.notna() & b.notna() & (a != 'unknown') & (b != 'unknown')
        match_flags[axis] = (known & (a == b)).astype(bool)
        valid_axes.append(axis)

    if not valid_axes:
        print('  No usable axes; skipping.')
        return pd.DataFrame(), pd.DataFrame()

    neg['match_count'] = match_flags[valid_axes].sum(axis=1).astype(int)
    # match_pattern: sorted comma-joined list of matching axes ('' = no matches).
    pattern_arrays = match_flags[valid_axes].values
    axis_labels = np.array(valid_axes)
    neg['match_pattern'] = [
        ','.join(axis_labels[row].tolist()) if row.any() else 'none'
        for row in pattern_arrays
    ]

    fp = (neg['pred_label'] == 1).astype(int)
    neg = neg.assign(_fp=fp)

    def _summarize(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        n_fp = int(group['_fp'].sum())
        return pd.Series({
            'n_neg': n,
            'n_fp': n_fp,
            'fp_rate': n_fp / n if n > 0 else float('nan'),
            'mean_pred_prob': float(group['pred_prob'].mean()) if n > 0 else float('nan'),
            'fp_avg_pred_prob': float(group.loc[group['_fp'] == 1, 'pred_prob'].mean())
                                 if n_fp > 0 else float('nan'),
        })

    # match-count table: one row per count from 0..len(valid_axes)
    count_rows = []
    for c in range(len(valid_axes) + 1):
        sub = neg[neg['match_count'] == c]
        row = _summarize(sub).to_dict()
        row['match_count'] = c
        count_rows.append(row)
    count_df = pd.DataFrame(count_rows)
    count_df = count_df[['match_count', 'n_neg', 'n_fp', 'fp_rate',
                         'mean_pred_prob', 'fp_avg_pred_prob']]
    count_path = results_dir / 'error_by_match_count.csv'
    count_df.to_csv(count_path, index=False)
    print(f'  Saved match-count table: {count_path}')
    print(count_df.round(4).to_string(index=False))

    # match-pattern table: one row per unique pattern, sorted by count then
    # by frequency descending so the headline patterns rise to the top.
    patt_rows = []
    for pattern, sub in neg.groupby('match_pattern'):
        row = _summarize(sub).to_dict()
        row['match_pattern'] = pattern
        row['match_count'] = int(sub['match_count'].iloc[0])
        patt_rows.append(row)
    pattern_df = pd.DataFrame(patt_rows)
    pattern_df = pattern_df[['match_pattern', 'match_count', 'n_neg', 'n_fp',
                             'fp_rate', 'mean_pred_prob', 'fp_avg_pred_prob']]
    pattern_df = pattern_df.sort_values(
        ['match_count', 'n_neg'], ascending=[True, False]
    ).reset_index(drop=True)
    pattern_path = results_dir / 'error_by_match_pattern.csv'
    pattern_df.to_csv(pattern_path, index=False)
    print(f'\n  Saved match-pattern table: {pattern_path}')
    print(pattern_df.round(4).to_string(index=False))

    # Plot: FP rate by match_count, with sample-size annotations.
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = count_df['n_neg'] > 0
    bars = ax.bar(count_df.loc[valid, 'match_count'],
                  count_df.loc[valid, 'fp_rate'],
                  color='crimson', edgecolor='black', alpha=0.8)
    for xi, n, rate in zip(count_df.loc[valid, 'match_count'],
                            count_df.loc[valid, 'n_neg'],
                            count_df.loc[valid, 'fp_rate']):
        ax.text(xi, rate + 0.005, f'{rate:.1%}\n(n={n:,})',
                ha='center', va='bottom', fontsize=9)
    ax.set_xlabel(f'match_count (axes that match between sides; max = {len(valid_axes)})')
    ax.set_ylabel('FP rate')
    ax.set_title('Negative-hardness: FP rate by metadata match-count')
    ax.set_xticks(range(len(valid_axes) + 1))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = results_dir / 'error_by_match_count.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved plot: {plot_path}')

    return count_df, pattern_df


def analyze_errors_by_metadata(df, y_true, y_pred, y_prob, results_dir: Path):
    """Stratified error analysis by host, hn_subtype, and year.

    Emits the Level 1 (pair-regime, 8 buckets + positive) and Level 2 (per-axis
    with mixed/unknown/other residual buckets) tables and plots defined in
    `docs/post_hoc_analysis_design.md`, plus the negative-hardness writeup
    and the prediction-probability-by-axis plot.

    Two-sided semantics: metadata is merged on BOTH sides of each pair
    (`assembly_id_a` AND `assembly_id_b`); strata are defined by both-sides
    agreement. The `year` axis uses bins (see `YEAR_BIN_EDGES`) instead of
    raw integers to keep per-fold strata dense.

    Removed 2026-05-11: the legacy
    `error_analysis_by_{host,hn_subtype,year}.csv` emission. Those CSVs were
    a strict subset of `level2_by_{host,subtype,year_bin}.csv` (same
    both-sides-matching schema, fewer metrics, no residual buckets); they
    fed only the now-deleted `_plot_errors_by_metadata`.

    Args:
        df: DataFrame with predictions (must have `assembly_id_a`,
            `assembly_id_b`, `label`, `pred_label`, `pred_prob`).
        y_true, y_pred, y_prob: accepted for backwards-compatible signature;
            the function reads columns directly from `df`.
        results_dir: Where to write CSVs and PNGs.
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

    # Removed 2026-05-11: legacy `error_analysis_by_{host,hn_subtype,year}.csv`
    # emission. Those CSVs were only consumed by the now-deleted
    # `_plot_errors_by_metadata` and used both-sides-matching semantics that
    # `analyze_level2_by_axis` already covers (with AUC-ROC / AUC-PR and the
    # explicit mixed / unknown / other residual buckets).

    # Level 1 / Level 2 (new outputs) — emitted from the same enriched df.
    # Two Level 1 views: per-regime (8 buckets) and aggregated (by match-count).
    level1_regimes_df = analyze_level1_neg_regimes(df_enriched, results_dir)
    level1_agg_df = analyze_level1_neg_regimes_agg(df_enriched, results_dir)
    level2 = {}
    for axis in ('host', 'subtype', 'year_bin'):
        level2[axis] = analyze_level2_by_axis(df_enriched, axis, results_dir)
    write_stratified_eval_summary(level1_regimes_df, level1_agg_df, level2, results_dir)

    # Visual companion to Level 1: pred-prob distribution within vs cross axis.
    plot_neg_prob_by_axis(df_enriched, results_dir)

    # Negative-hardness: how many metadata axes match per negative pair?
    analyze_negative_hardness(df_enriched, results_dir)


# Removed 2026-05-11: `_plot_errors_by_metadata` (emitted
# `error_rates_by_metadata.png` and `performance_metrics_by_metadata.png`).
# Single-side metadata stratification is strictly subsumed by the
# `level2_by_{host,subtype,year_bin}.png` trio (both-sides-matching + explicit
# mixed/unknown/other residual buckets) for performance metrics, and by
# `error_by_match_count.{csv,png}` + the per-FP/FN drill-down CSVs for error
# direction.


def analyze_segment_performance(df):
    """Analyze performance by segment combinations.

    **NOTE: Degenerate for v2** -- v2 schema_pair fixes one (func_left,
    func_right), so the test set has a single seg_pair. Output is one row
    that duplicates `metrics.csv`. Kept for v1 / future cross-protein runs.
    """
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
            # Check if both classes are present for F1, AUC-ROC, AUC-PR, Brier
            unique_labels = subset['label'].unique()
            if len(unique_labels) > 1:
                f1 = f1_score(subset['label'], subset['pred_label'], average='binary')
                auc_roc = roc_auc_score(subset['label'], subset['pred_prob'])
                auc_pr = average_precision_score(subset['label'], subset['pred_prob'])
                brier_score = float(np.mean((subset['pred_prob'].values - subset['label'].values) ** 2))
            else:
                f1 = np.nan
                auc_roc = np.nan
                auc_pr = np.nan
                brier_score = np.nan

            acc = (subset['label'] == subset['pred_label']).mean()

            seg_stats.append({
                'seg_pair': seg_pair,
                'count': len(subset),
                'accuracy': acc,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'brier_score': brier_score,
                'pos_rate': subset['label'].mean()
            })

    seg_df = pd.DataFrame(seg_stats).sort_values('f1_score', ascending=False)
    print('\nPerformance by Segment Pair:')
    print(seg_df.round(3))

    return seg_df


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
        plot_calibration_curve(y_true, y_prob, save_path=results_dir / 'calibration_curve.png')
        print("Performance plots created (roc_curve.png, precision_recall_curve.png, "
              "prediction_distribution.png, calibration_curve.png)")
    else:
        print("Skipping performance plots (create_performance_plots=False)")
    
    # Enhanced FP/FN analysis
    error_summary = analyze_fp_fn_errors(df, y_true, y_pred, y_prob, results_dir)
    
    # Metadata-stratified error analysis (level1 + level2 + summary)
    analyze_errors_by_metadata(df, y_true, y_pred, y_prob, results_dir)

    # Domain-specific analyses. analyze_segment_performance is degenerate
    # under v2 (schema_ordered fixes a single (func_left, func_right), so the
    # test set has one seg_pair and segment_metrics.csv would just restate
    # metrics.csv). The function is retained in the module for v1 / future
    # cross-protein runs but is not invoked from the default v2 pipeline.
    analyze_prediction_confidence(df)

    # Save summary metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / 'metrics.csv', index=False)

    # Headline metrics bar chart -- replaces the old presentation-plots
    # path (see deletion of src/analysis/create_presentation_plots.py).
    analyze_metrics_summary(metrics, results_dir)

    # Swap-test variant: only when the trainer also wrote
    # test_predicted_swapped.csv (eval_swapped_test=True). Recompute basic
    # metrics on the swapped predictions and emit a parallel bar chart so
    # asymmetry vs the canonical orientation is visible at a glance.
    swap_csv = models_dir / 'test_predicted_swapped.csv'
    if swap_csv.exists():
        swap_df = pd.read_csv(swap_csv)
        swap_metrics = compute_basic_metrics(
            swap_df['label'].values,
            swap_df['pred_label'].values,
            swap_df['pred_prob'].values,
        )
        analyze_metrics_summary(swap_metrics, results_dir,
                                save_name='metrics_swapped.png')

    print(f'\nAnalysis complete! Results saved to: {results_dir}')

    # Key insights summary
    print(f"\n{'='*60}")
    print('KEY INSIGHTS SUMMARY')
    print('='*60)
    print(f'• Overall accuracy: {metrics["accuracy"]:.3f}')
    print(f'• F1 score: {metrics["f1_score"]:.3f}')
    print(f'• AUC-ROC: {metrics["auc_roc"]:.3f}')
    print(f'• AUC-PR: {metrics["auc_pr"]:.3f}')
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
