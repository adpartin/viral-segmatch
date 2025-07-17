"""
Analyze dataset statistics for the segment pair classifier.

This script analyzes the train/val/test datasets created in Stage 2,
providing insights into dataset composition, balance, and distribution.

# Datasets:
# Histogram of segmant pairs in each of the train/val/test sets, grouped by positive and negative groups
# Histogram of unique isolates in each set
# How many of the same-function pairs actually have same protein sequences?
# Can we somehow color-code by genus-level?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import plot configuration
from src.utils.plot_config import (
    DATASET_COLORS_LIST, SAMPLE_PATTERNS, get_dataset_color, 
    map_protein_name, apply_default_style
)

# Config
VIRUS_NAME = 'bunya'
DATA_VERSION = 'April_2025'
TASK_NAME = 'segment_pair_classifier'

# Define paths
dataset_dir = project_root / 'data' / 'datasets' / VIRUS_NAME / DATA_VERSION / TASK_NAME
results_dir = project_root / 'results' / VIRUS_NAME / DATA_VERSION / 'dataset_analysis'
results_dir.mkdir(parents=True, exist_ok=True)

print(f"Dataset directory: {dataset_dir}")
print(f"Results directory: {results_dir}")

# Load datasets
print("\nLoading datasets...")
train_df = pd.read_csv(dataset_dir / 'train_pairs.csv')
val_df = pd.read_csv(dataset_dir / 'val_pairs.csv')
test_df = pd.read_csv(dataset_dir / 'test_pairs.csv')

print(f'Train pairs: {len(train_df)}')
print(f'Val pairs: {len(val_df)}')
print(f'Test pairs: {len(test_df)}')

# Set up plotting style
apply_default_style()
sns.set_palette('Set2')


def analyze_basic_statistics():
    """Analyze and display basic dataset statistics."""
    print('\n' + '='*60)
    print('BASIC DATASET STATISTICS')
    print('='*60)
    
    datasets = {'Train': train_df, 'Val': val_df, 'Test': test_df}
    
    stats = []
    for name, df in datasets.items():
        total_pairs = len(df)
        positive_pairs = (df['label'] == 1).sum()
        negative_pairs = (df['label'] == 0).sum()
        pos_rate = positive_pairs / total_pairs
        
        # Count unique isolates
        unique_isolates = set(df['assembly_id_a']).union(set(df['assembly_id_b']))
        
        # Same-function negative pairs
        same_func_neg = df[(df['label'] == 0) & (df['func_a'] == df['func_b'])]
        same_func_neg_count = len(same_func_neg)
        same_func_neg_rate = same_func_neg_count / negative_pairs if negative_pairs > 0 else 0
        
        stats.append({
            'Dataset': name,
            'Total Pairs': total_pairs,
            'Positive Pairs': positive_pairs,
            'Negative Pairs': negative_pairs,
            'Positive Rate': f"{pos_rate:.3f}",
            'Unique Isolates': len(unique_isolates),
            'Same-Func Neg': same_func_neg_count,
            'Same-Func Neg Rate': f"{same_func_neg_rate:.3f}"
        })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    
    # Save to CSV
    stats_df.to_csv(results_dir / 'dataset_basic_statistics.csv', index=False)
    
    return stats_df


def create_segment_pair_histograms():
    """Create histograms of segment pairs by dataset and label."""
    print("\nCreate segment pair histograms.")
    
    # Add segment pair column to each dataset
    for df in [train_df, val_df, test_df]:
        df['seg_pair'] = df.apply(lambda row: f"{row['seg_a']}-{row['seg_b']}" 
                                     if row['seg_a'] <= row['seg_b'] 
                                     else f"{row['seg_b']}-{row['seg_a']}", axis=1)
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    datasets = [('Train', train_df), ('Val', val_df), ('Test', test_df)]
    
    # Plot positive pairs (top row)
    for idx, (name, df) in enumerate(datasets):
        ax = axes[0, idx]
        pos_pairs = df[df['label'] == 1]
        if len(pos_pairs) > 0:
            segment_counts = pos_pairs['seg_pair'].value_counts()
            bars = ax.bar(segment_counts.index, segment_counts.values, 
                         color=get_dataset_color(name), alpha=0.8,
                         hatch=SAMPLE_PATTERNS['positive'])
            ax.set_title(f'{name} - Positive Pairs (Same Isolate)', fontweight='bold')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            
            # Add count and percentage labels
            total_pos = segment_counts.sum()
            for bar, count in zip(bars, segment_counts.values):
                height = bar.get_height()
                pct = count / total_pos * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{int(height)}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot negative pairs (bottom row)
    for idx, (name, df) in enumerate(datasets):
        ax = axes[1, idx]
        neg_pairs = df[df['label'] == 0]
        if len(neg_pairs) > 0:
            segment_counts = neg_pairs['seg_pair'].value_counts()
            bars = ax.bar(segment_counts.index, segment_counts.values, 
                         color=get_dataset_color(name), alpha=0.8,
                         hatch=SAMPLE_PATTERNS['negative'])
            ax.set_title(f'{name} - Negative Pairs (Different Isolates)', fontweight='bold')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            
            # Add count and percentage labels
            total_neg = segment_counts.sum()
            for bar, count in zip(bars, segment_counts.values):
                height = bar.get_height()
                pct = count / total_neg * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{int(height)}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'segment_pair_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_isolate_distribution_plots():
    """Create plots showing isolate distributions across datasets."""
    print("\nCreate isolate distribution plots.")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Number of unique isolates per dataset
    datasets = ['Train', 'Val', 'Test']
    isolate_counts = []
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        unique_isolates = set(df['assembly_id_a']).union(set(df['assembly_id_b']))
        isolate_counts.append(len(unique_isolates))
    
    bars = ax1.bar(datasets, isolate_counts, color=DATASET_COLORS_LIST, alpha=0.8)
    ax1.set_title('Unique Isolates per Dataset', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Number of Unique Isolates')
    
    # Add count and percentage labels
    total_isolates = sum(isolate_counts)
    for bar, count in zip(bars, isolate_counts):
        height = bar.get_height()
        pct = count / total_isolates * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # 2. Dataset size distribution
    pair_counts = [len(train_df), len(val_df), len(test_df)]
    bars = ax2.bar(datasets, pair_counts, color=DATASET_COLORS_LIST, alpha=0.8)
    ax2.set_title('Total Pairs per Dataset', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Number of Pairs')
    
    # Add count and percentage labels
    total_pairs = sum(pair_counts)
    for bar, count in zip(bars, pair_counts):
        height = bar.get_height()
        pct = count / total_pairs * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # 3. Positive vs Negative pairs distribution
    pos_counts = [(df['label'] == 1).sum() for df in [train_df, val_df, test_df]]
    neg_counts = [(df['label'] == 0).sum() for df in [train_df, val_df, test_df]]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars_pos = ax3.bar(x - width/2, pos_counts, width, 
             label='Positive (+)', color=DATASET_COLORS_LIST, alpha=0.8,
             hatch=SAMPLE_PATTERNS['positive'])
    bars_neg = ax3.bar(x + width/2, neg_counts, width, 
             label='Negative (-)', color=DATASET_COLORS_LIST, alpha=0.8,
             hatch=SAMPLE_PATTERNS['negative'])
    
    ax3.set_title('Positive vs Negative Pairs', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Number of Pairs')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets)
    ax3.legend()
    
    # Add count and percentage labels for grouped bars
    total_per_dataset = [pos + neg for pos, neg in zip(pos_counts, neg_counts)]
    for i, (bar_pos, bar_neg, pos_count, neg_count, total_count) in enumerate(zip(bars_pos, bars_neg, pos_counts, neg_counts, total_per_dataset)):
        # Positive bars
        height_pos = bar_pos.get_height()
        pct_pos = pos_count / total_count * 100
        ax3.text(bar_pos.get_x() + bar_pos.get_width()/2., height_pos + 10,
                f'{pos_count}\n({pct_pos:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Negative bars
        height_neg = bar_neg.get_height()
        pct_neg = neg_count / total_count * 100
        ax3.text(bar_neg.get_x() + bar_neg.get_width()/2., height_neg + 10,
                f'{neg_count}\n({pct_neg:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. Same-function negative pairs analysis
    same_func_neg_counts = []
    total_neg_counts = []
    
    for df in [train_df, val_df, test_df]:
        neg_pairs = df[df['label'] == 0]
        same_func_neg = neg_pairs[neg_pairs['func_a'] == neg_pairs['func_b']]
        same_func_neg_counts.append(len(same_func_neg))
        total_neg_counts.append(len(neg_pairs))
    
    same_func_rates = [sf/total if total > 0 else 0 
                      for sf, total in zip(same_func_neg_counts, total_neg_counts)]
    
    bars = ax4.bar(datasets, same_func_rates, color=DATASET_COLORS_LIST, alpha=0.8)
    ax4.set_title('Same-Function Negative Pair Rate', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Proportion of Negative Pairs')
    ax4.set_ylim(0, 1)
    
    # Add count and rate labels
    for bar, rate, count, total in zip(bars, same_func_rates, same_func_neg_counts, total_neg_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{count}/{total}\n({rate:.3f})', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'isolate_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_function_distribution():
    """Analyze protein function distributions across datasets."""
    print("\nAnalyze function distributions.")
    
    # Combine all datasets with dataset labels - ensure train, val, test order
    all_data = []
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        df_copy = df.copy()
        df_copy['dataset'] = name
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Function pair analysis using protein name mapping
    combined_df['func_pair'] = combined_df.apply(
        lambda row: f"{map_protein_name(row['func_a'])}-{map_protein_name(row['func_b'])}" 
        if map_protein_name(row['func_a']) <= map_protein_name(row['func_b'])
        else f"{map_protein_name(row['func_b'])}-{map_protein_name(row['func_a'])}", axis=1
    )
    
    # Define alternative colors that don't conflict with train/val/test (blue/orange/green)
    # Using purple, red, and teal instead
    FUNCTION_COLORS = ['#9B59B6', '#E74C3C', '#17A2B8']  # Purple, Red, Teal
    
    # Create function pair distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Positive pairs by function
    pos_pairs = combined_df[combined_df['label'] == 1]
    func_counts_pos = pos_pairs.groupby(['dataset', 'func_pair']).size().unstack(fill_value=0)
    
    # Ensure datasets are in correct order
    func_counts_pos = func_counts_pos.reindex(['Train', 'Val', 'Test'])
    
    func_counts_pos.plot(kind='bar', ax=ax1, alpha=0.8, color=FUNCTION_COLORS)
    ax1.set_title('Positive Pairs by Function Combination', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Dataset')
    ax1.legend(title='Function Pair', bbox_to_anchor=(1.05, 1), loc='best')
    ax1.tick_params(axis='x', rotation=0)
    
    # Negative pairs by function (same-function only)
    neg_same_func = combined_df[(combined_df['label'] == 0) & 
                               (combined_df['func_a'] == combined_df['func_b'])]
    
    if len(neg_same_func) > 0:
        func_counts_neg = neg_same_func.groupby(['dataset', 'func_pair']).size().unstack(fill_value=0)
        # Ensure datasets are in correct order
        func_counts_neg = func_counts_neg.reindex(['Train', 'Val', 'Test'])
        
        func_counts_neg.plot(kind='bar', ax=ax2, alpha=0.8, color=FUNCTION_COLORS)
        ax2.set_title('Same-Function Negative Pairs', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Dataset')
        ax2.legend(title='Function', bbox_to_anchor=(1.05, 1), loc='best')
        ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'function_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return combined_df


def create_dataset_summary_table():
    """Create a comprehensive summary table."""
    print("\nCreate dataset summary table.")
    
    summary_data = []
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        # Basic counts
        total_pairs = len(df)
        pos_pairs = (df['label'] == 1).sum()
        neg_pairs = (df['label'] == 0).sum()
        
        # Isolate information
        isolates_a = set(df['assembly_id_a'])
        isolates_b = set(df['assembly_id_b'])
        unique_isolates = isolates_a.union(isolates_b)
        
        # Segment pair analysis
        df['seg_pair'] = df.apply(lambda row: f"{row['seg_a']}-{row['seg_b']}" 
                                     if row['seg_a'] <= row['seg_b'] 
                                     else f"{row['seg_b']}-{row['seg_a']}", axis=1)
        
        unique_seg_pairs = df['seg_pair'].nunique()
        
        # Same-function analysis
        same_func_neg = df[(df['label'] == 0) & (df['func_a'] == df['func_b'])]
        
        summary_data.append({
            'Dataset': name,
            'Total_Pairs': total_pairs,
            'Positive_Pairs': pos_pairs,
            'Negative_Pairs': neg_pairs,
            'Positive_Rate': f"{pos_pairs/total_pairs:.3f}",
            'Unique_Isolates': len(unique_isolates),
            'Unique_Segment_Pairs': unique_seg_pairs,
            'Same_Func_Negatives': len(same_func_neg),
            'Same_Func_Neg_Rate': f"{len(same_func_neg)/neg_pairs:.3f}" if neg_pairs > 0 else "0"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\nDataset Summary:")
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(results_dir / 'dataset_comprehensive_summary.csv', index=False)
    
    return summary_df


def main():
    """Run all analyses."""
    print('Analyzing Dataset Statistics for Stage 2')
    print('='*60)
    
    # Basic statistics
    stats_df = analyze_basic_statistics()
    
    # Create visualizations
    create_segment_pair_histograms()
    create_isolate_distribution_plots()
    combined_df = analyze_function_distribution()
    
    # Summary table
    summary_df = create_dataset_summary_table()
    
    print(f'\nAnalysis complete! All results saved to: {results_dir}')
    
    # Display key insights
    print('\n' + '='*60)
    print('KEY INSIGHTS')
    print('='*60)
    
    total_pairs = len(train_df) + len(val_df) + len(test_df)
    total_isolates = len(set(train_df['assembly_id_a']).union(
        set(train_df['assembly_id_b']),
        set(val_df['assembly_id_a']),
        set(val_df['assembly_id_b']),
        set(test_df['assembly_id_a']),
        set(test_df['assembly_id_b'])
    ))
    
    print(f'• Total protein pairs: {total_pairs:,}')
    print(f'• Total unique isolates: {total_isolates}')
    print(f'• Dataset split: {len(train_df)/total_pairs:.1%} train, {len(val_df)/total_pairs:.1%} val, {len(test_df)/total_pairs:.1%} test')
    print(f"• Positive pair rate: {(train_df['label']==1).mean():.1%} (consistent across all sets)")
    
    # Same-function negative rates
    train_sf_rate = len(train_df[(train_df['label']==0) & (train_df['func_a']==train_df['func_b'])]) / (train_df['label']==0).sum()
    print(f'• Same-function negative rate: ~{train_sf_rate:.1%} (prevents task from being too easy)')

if __name__ == '__main__':
    main() 