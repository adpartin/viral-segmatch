"""
Create presentation-ready plots for the segment classifier results.

This script generates additional visualizations specifically designed for
presentation slides, focusing on the most important insights.

Output directory: results/{virus_name}/{data_version}/{config_bundle}/training_analysis/
(Shares output with analyze_stage4_train.py)

Usage:
    python src/analysis/create_presentation_plots.py --config_bundle bunya
    python src/analysis/create_presentation_plots.py --config_bundle flu_a_3p_1ks --model_dir ./models/flu_a/July_2025...
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config_hydra import get_virus_config_hydra
from src.utils.path_utils import build_training_paths
from src.utils.plot_config import apply_default_style

# Set up plotting style for presentations (shared project defaults)
apply_default_style()
sns.set_palette('Set2')


def _read_metrics_values(df: pd.DataFrame, results_dir: Path):
    """Read metrics from metrics.csv (or compute from df as fallback).

    Returns (metric_names, values, loss) where loss may be None for older
    metrics.csv files that don't contain it.
    """
    metrics_csv = results_dir / 'metrics.csv'
    loss = None
    if metrics_csv.exists():
        metrics_df = pd.read_csv(metrics_csv)
        if 'f1_macro' in metrics_df.columns:
            f1_macro = metrics_df.iloc[0]['f1_macro']
        else:
            f1_macro = f1_score(df['label'], df['pred_label'], average='macro')
            print(f"   Note: f1_macro not in metrics.csv, computed from test set: {f1_macro:.4f}")
        values = [
            metrics_df.iloc[0]['accuracy'],
            metrics_df.iloc[0]['f1_score'],
            f1_macro,
            metrics_df.iloc[0]['auc_roc'],
            metrics_df.iloc[0]['avg_precision'],
        ]
        if 'loss' in metrics_df.columns:
            loss = metrics_df.iloc[0]['loss']
    else:
        values = [
            accuracy_score(df['label'], df['pred_label']),
            f1_score(df['label'], df['pred_label'], average='binary', pos_label=1),
            f1_score(df['label'], df['pred_label'], average='macro'),
            roc_auc_score(df['label'], df['pred_prob']),
            average_precision_score(df['label'], df['pred_prob']),
        ]
    metric_names = ['Accuracy', 'F1 Score', 'F1 Macro', 'AUC-ROC', 'AUC-PR']
    return metric_names, values, loss


def _plot_metrics_bar(ax, metric_names, values, loss=None):
    """Plot a metrics bar chart on *ax* with optional loss annotation."""
    colors = ['#2E86AB', '#A23B72', '#6A4C93', '#F18F01', '#C73E1D']
    bars = ax.bar(metric_names, values, color=colors, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    if loss is not None:
        ax.text(0.98, 0.02, f'Loss: {loss:.4f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=10, fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))


def create_metrics_summary_plot(
    df: pd.DataFrame,
    results_dir: Path,
    pred_csv_name: str = 'test_predicted.csv',
    save_name: str = 'metrics_summary.png',
) -> None:
    """Create a standalone metrics bar chart (single figure).

    If *pred_csv_name* is 'test_predicted_swapped.csv', metrics are recomputed
    from that file so the plot reflects swapped-input predictions.
    """
    # Determine metrics source
    if pred_csv_name == 'test_predicted.csv':
        metric_names, values, loss = _read_metrics_values(df, results_dir)
    else:
        # Recompute metrics from the provided df (e.g. swapped predictions)
        y_true = df['label'].values
        y_pred = df['pred_label'].values
        y_prob = df['pred_prob'].values
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        loss = float(-np.mean(
            y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped)
        ))
        values = [
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='binary', pos_label=1),
            f1_score(y_true, y_pred, average='macro'),
            roc_auc_score(y_true, y_prob),
            average_precision_score(y_true, y_prob),
        ]
        metric_names = ['Accuracy', 'F1 Score', 'F1 Macro', 'AUC-ROC', 'AUC-PR']

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_metrics_bar(ax, metric_names, values, loss)
    plt.tight_layout()
    plt.savefig(results_dir / save_name, dpi=300, bbox_inches='tight')
    plt.close()


def create_results_misc_plot(df: pd.DataFrame, results_dir: Path) -> None:
    """Create results_misc.png -- a 2x2 grid with:

    Top-left  : Classification errors by segment pair (from biological_insights)
    Top-right : Performance by segment pair
    Bottom-left: Pair classification by type
    Bottom-right: Prediction confidence distribution
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # ---- 1. Error analysis by segment pair (moved from biological_insights) ----
    df_copy = df.copy()
    df_copy['seg_pair'] = df_copy.apply(
        lambda row: f"{row['seg_a']}-{row['seg_b']}"
        if row['seg_a'] <= row['seg_b']
        else f"{row['seg_b']}-{row['seg_a']}", axis=1
    )
    errors = df_copy[df_copy['label'] != df_copy['pred_label']]
    error_counts = errors['seg_pair'].value_counts()
    cross_func_errors = error_counts[error_counts.index.str.contains('-')]

    if len(cross_func_errors) > 0:
        bars = ax1.bar(cross_func_errors.index, cross_func_errors.values,
                       color='#E74C3C', alpha=0.8)
        ax1.set_ylabel('Number of Errors')
        ax1.set_title('Classification Errors by Segment Pair', fontsize=14, fontweight='bold')
        for bar, value in zip(bars, cross_func_errors.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{value}', ha='center', va='bottom', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No cross-function errors',
                 ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Classification Errors by Segment Pair', fontsize=14, fontweight='bold')

    # ---- 2. Segment pair performance (cross-function pairs only) ----
    seg_metrics_path = results_dir / 'segment_metrics.csv'
    if seg_metrics_path.exists():
        segment_perf = pd.read_csv(seg_metrics_path)
        cross_func = segment_perf[segment_perf['seg_pair'].str.contains('-')]
        cross_func = cross_func.dropna(subset=['f1_score'])
        if len(cross_func) > 0:
            bars = ax2.bar(cross_func['seg_pair'], cross_func['f1_score'],
                           color='#4ECDC4', alpha=0.8)
            ax2.set_ylabel('F1 Score')
            ax2.set_title('Performance by Segment Pair', fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 1)
            for bar, value in zip(bars, cross_func['f1_score']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No cross-function pairs',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Performance by Segment Pair', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'segment_metrics.csv not found',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Performance by Segment Pair', fontsize=14, fontweight='bold')

    # ---- 3. Pair classification by type ----
    same_func_neg = df[(df['label'] == 0) & (df['func_a'] == df['func_b'])]
    diff_func_neg = df[(df['label'] == 0) & (df['func_a'] != df['func_b'])]
    positive_pairs = df[df['label'] == 1]

    same_func_acc = (same_func_neg['label'] == same_func_neg['pred_label']).mean() if len(same_func_neg) > 0 else None
    diff_func_acc = (diff_func_neg['label'] == diff_func_neg['pred_label']).mean() if len(diff_func_neg) > 0 else None
    positive_acc = (positive_pairs['label'] == positive_pairs['pred_label']).mean() if len(positive_pairs) > 0 else None

    categories = []
    accuracies = []
    colors_list = []
    if positive_acc is not None:
        categories.append('Positive\nPairs')
        accuracies.append(positive_acc)
        colors_list.append('#27AE60')
    if same_func_acc is not None:
        categories.append('Same-Function\nNegatives')
        accuracies.append(same_func_acc)
        colors_list.append('#FF6B6B')
    if diff_func_acc is not None:
        categories.append('Different-Function\nNegatives')
        accuracies.append(diff_func_acc)
        colors_list.append('#E74C3C')

    if len(categories) > 0:
        bars = ax3.bar(categories, accuracies, color=colors_list, alpha=0.8)
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Pair Classification by Type', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1)
        for bar, value in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No pair data available',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Pair Classification by Type', fontsize=14, fontweight='bold')

    # ---- 4. Prediction confidence distribution ----
    confidence = np.abs(df['pred_prob'] - 0.5)
    conf_bins = ['Very Low\n(0.4-0.6)', 'Low\n(0.3-0.4, 0.6-0.7)',
                 'Medium\n(0.2-0.3, 0.7-0.8)', 'High\n(0.0-0.2, 0.8-1.0)']
    very_low = int((confidence < 0.1).sum())
    low = int(((confidence >= 0.1) & (confidence < 0.2)).sum())
    medium = int(((confidence >= 0.2) & (confidence < 0.3)).sum())
    high = int((confidence >= 0.3).sum())
    counts = [very_low, low, medium, high]
    colors = ['#FF6B6B', '#FFE66D', '#4ECDC4', '#45B7D1']

    bars = ax4.bar(conf_bins, counts, color=colors, alpha=0.8)
    ax4.set_ylabel('Number of Predictions')
    ax4.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    for bar, value in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{value}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / 'results_misc.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_error_analysis_plot(df: pd.DataFrame, results_dir: Path) -> None:
    """Create plots highlighting error analysis by segment pair and function.

    Previously named ``create_biological_insights_plot``. Saves
    ``biological_insights.png`` (kept for backward compatibility).
    """
    # Check if there are same-function negatives
    same_func_neg = df[(df['label'] == 0) & (df['func_a'] == df['func_b'])]
    has_same_func_neg = len(same_func_neg) > 0
    
    # Create subplots conditionally
    if has_same_func_neg:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))
    
    # 1. Error analysis by segment pair
    df['seg_pair'] = df.apply(
        lambda row: f"{row['seg_a']}-{row['seg_b']}" 
        if row['seg_a'] <= row['seg_b'] 
        else f"{row['seg_b']}-{row['seg_a']}", axis=1
    )

    errors = df[df['label'] != df['pred_label']]
    error_counts = errors['seg_pair'].value_counts()

    # Only show cross-function pairs
    cross_func_errors = error_counts[error_counts.index.str.contains('-')] # TODO: in what case there will be '-'?

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
    
    # 2. Function-specific performance (for same-function negatives) - only if they exist
    if has_same_func_neg:
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


def create_model_calibration_plot(df: pd.DataFrame, results_dir: Path) -> None:
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
    # Colors, xlabel, and title kept consistent with plot_prediction_distribution()
    # in analyze_stage4_train.py
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    
    ax2.hist(neg_probs, bins=30, alpha=0.7, label='Negative (Different Isolate)', 
             color='#E74C3C', density=True)
    ax2.hist(pos_probs, bins=30, alpha=0.7, label='Positive (Same Isolate)', 
             color='#3498DB', density=True)
    ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, 
                label='Decision Threshold')
    ax2.set_xlabel('Prediction Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Prediction Probabilities', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'model_calibration.png', dpi=300, bbox_inches='tight')
    plt.show()


def main(config_bundle: str, model_dir: Optional[Path] = None) -> None:
    """Generate all presentation plots."""
    print(f"\n{'='*50}")
    print('Create Presentation Plots for Segment Classifier Results')
    print('='*50)
    
    # Load config
    config_path = str(project_root / 'conf')
    config = get_virus_config_hydra(config_bundle, config_path=config_path)
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
                    print(f"⚠️  No run found for bundle '{config_bundle}', using most recent run: {models_dir}")
                else:
                    models_dir = base_models_dir
                    print(f"⚠️  No training runs found, using base directory: {models_dir}")
        else:
            models_dir = base_models_dir
            print(f"Using base models directory: {models_dir}")
    
    # Construct results directory (config-specific)
    # Format: results/{virus_name}/{data_version}/{config_bundle}/training_analysis/
    results_dir = project_root / 'results' / config.virus.virus_name / config.virus.data_version / config_bundle / 'training_analysis'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load test results
    test_results_file = models_dir / 'test_predicted.csv'
    if not test_results_file.exists():
        raise FileNotFoundError(
            f'❌ Test results file not found: {test_results_file}\n'
            f'   Model directory: {models_dir}\n'
            f'   Hint: Use --model_dir to specify the exact training run directory'
        )

    df = pd.read_csv(test_results_file)
    print(f'✅ Loaded {len(df)} test predictions')

    print('1. Metrics summary plot (metrics_summary.png).')
    create_metrics_summary_plot(df, results_dir)

    print('2. Results misc plot (results_misc.png).')
    create_results_misc_plot(df, results_dir)

    # NOTE: create_error_analysis_plot() (formerly create_biological_insights_plot)
    # is no longer called separately; its content is included in results_misc.png.

    print('3. Model calibration plot (model_calibration.png).')
    create_model_calibration_plot(df, results_dir)

    # Swapped-input metrics summary (if swapped predictions exist)
    swapped_file = models_dir / 'test_predicted_swapped.csv'
    if swapped_file.exists():
        df_swapped = pd.read_csv(swapped_file)
        print(f'4. Metrics summary (swapped) plot (metrics_summary_swapped.png).')
        create_metrics_summary_plot(
            df_swapped, results_dir,
            pred_csv_name='test_predicted_swapped.csv',
            save_name='metrics_summary_swapped.png',
        )
    else:
        print('4. Skipping metrics_summary_swapped.png (test_predicted_swapped.csv not found).')

    print(f'All plots saved to: {results_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create presentation plots for segment classifier results')
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
    args = parser.parse_args()
    
    main(
        config_bundle=args.config_bundle,
        model_dir=args.model_dir
    )
