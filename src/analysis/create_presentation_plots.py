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


def create_performance_summary_plot(df: pd.DataFrame, results_dir: Path) -> None:
    """Create a summary plot showing key metrics."""
    # breakpoint()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Overall metrics bar chart - read from metrics.csv
    # Note: All metrics are computed on the TEST set (from test_predicted.csv)
    metrics_csv = results_dir / 'metrics.csv'
    if metrics_csv.exists():
        metrics_df = pd.read_csv(metrics_csv)
        # Try to get f1_macro, fallback to computing if not available
        if 'f1_macro' in metrics_df.columns:
            f1_macro = metrics_df.iloc[0]['f1_macro']
        else:
            # Compute from test set if not in CSV (for backward compatibility)
            f1_macro = f1_score(df['label'], df['pred_label'], average='macro')
            print(f"   Note: f1_macro not in metrics.csv, computed from test set: {f1_macro:.4f}")
        values = [
            metrics_df.iloc[0]['accuracy'],
            metrics_df.iloc[0]['f1_score'],
            f1_macro,
            metrics_df.iloc[0]['auc_roc'],
            metrics_df.iloc[0]['avg_precision']
        ]
    else:
        # Fallback: compute from dataframe if metrics.csv doesn't exist
        values = [
            accuracy_score(df['label'], df['pred_label']),
            f1_score(df['label'], df['pred_label'], average='binary', pos_label=1),
            f1_score(df['label'], df['pred_label'], average='macro'),
            roc_auc_score(df['label'], df['pred_prob']),
            average_precision_score(df['label'], df['pred_prob'])
        ]
    
    metrics = ['Accuracy', 'F1 Score', 'F1 Macro', 'AUC-ROC', 'AUC-PR']
    colors = ['#2E86AB', '#A23B72', '#6A4C93', '#F18F01', '#C73E1D']

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

    # 3. Pair classification by type (positive, same-function negative, different-function negative)
    same_func_neg = df[(df['label'] == 0) & (df['func_a'] == df['func_b'])]
    diff_func_neg = df[(df['label'] == 0) & (df['func_a'] != df['func_b'])]
    positive_pairs = df[df['label'] == 1]

    same_func_acc = (same_func_neg['label'] == same_func_neg['pred_label']).mean() if len(same_func_neg) > 0 else None
    diff_func_acc = (diff_func_neg['label'] == diff_func_neg['pred_label']).mean() if len(diff_func_neg) > 0 else None
    positive_acc = (positive_pairs['label'] == positive_pairs['pred_label']).mean() if len(positive_pairs) > 0 else None
    
    # Build categories and accuracies dynamically
    categories = []
    accuracies = []
    colors_list = []
    
    if positive_acc is not None:
        categories.append('Positive\nPairs')
        accuracies.append(positive_acc)
        colors_list.append('#27AE60')  # Green for positive
    
    if same_func_acc is not None:
        categories.append('Same-Function\nNegatives')
        accuracies.append(same_func_acc)
        colors_list.append('#FF6B6B')  # Light red/coral for same-function
    
    if diff_func_acc is not None:
        categories.append('Different-Function\nNegatives')
        accuracies.append(diff_func_acc)
        colors_list.append('#E74C3C')  # Darker red for different-function
    
    if len(categories) > 0:
        bars = ax3.bar(categories, accuracies, color=colors_list, alpha=0.8)
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Pair Classification by Type', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No pair data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Pair Classification by Type', fontsize=14, fontweight='bold')
    
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


def create_biological_insights_plot(df: pd.DataFrame, results_dir: Path) -> None:
    """Create plots highlighting biological insights."""
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
    
    print('1. Performance summary plot.')
    create_performance_summary_plot(df, results_dir)

    print('2. Biological insights plot.')
    create_biological_insights_plot(df, results_dir)

    print('3. Model calibration plot.')
    create_model_calibration_plot(df, results_dir)

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
