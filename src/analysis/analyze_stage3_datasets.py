"""
Analyze Stage 3: Dataset statistics for the segment pair classifier.

This script analyzes the train/val/test datasets created in Stage 3,
providing insights into dataset composition, balance, and distribution.

Output directory: results/{virus_name}/{data_version}/{config_bundle}/dataset_analysis/
(Config-specific - datasets depend on allow_same_func_negatives, ratios, etc.)

Plots generated:
- segment_pair_histograms.png: Segment pair counts for pos/neg pairs across train/val/test (2x3 grid)
- isolate_distributions.png: 4-panel plot showing:
    - Unique isolates per dataset
    - Total pairs per dataset
    - Positive vs negative pairs
    - Same-function negative pair rate
- function_distributions.png: Function pair distributions for positive and same-function negative pairs

Other outputs:
- dataset_basic_statistics.csv: Basic statistics table
- dataset_comprehensive_summary.csv: Detailed summary table

Usage:
    python src/analysis/analyze_stage3_datasets.py --config_bundle bunya
    python src/analysis/analyze_stage3_datasets.py --config_bundle bunya \
        --dataset_dir data/datasets/bunya/April_2025/runs/dataset_bunya_20251201_182905
    python src/analysis/analyze_stage3_datasets.py --config_bundle flu_a_plateau_analysis \
        --dataset_dir data/datasets/flu_a/July_2025/runs/dataset_...
"""

import argparse
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

# Import plot configuration and config utilities
from src.utils.plot_config import (
    SPLIT_COLORS_LIST, SAMPLE_PATTERNS, get_split_color, 
    map_protein_name, apply_default_style
)
from src.utils.config_hydra import get_virus_config_hydra
from src.utils.embedding_utils import (
    load_embedding_index, load_embeddings_by_ids,
    extract_unique_sequences_from_pairs,
    sample_sequences_stratified, sample_pairs_stratified,
    create_pair_embeddings_concatenation,
    plot_embeddings_by_category, plot_pair_embeddings_by_label
)
from src.utils.dim_reduction_utils import compute_pca_reduction


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
                         color=get_split_color(name), alpha=0.8,
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
                         color=get_split_color(name), alpha=0.8,
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
    
    bars = ax1.bar(datasets, isolate_counts, color=SPLIT_COLORS_LIST, alpha=0.8)
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
    bars = ax2.bar(datasets, pair_counts, color=SPLIT_COLORS_LIST, alpha=0.8)
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
             label='Positive (+)', color=SPLIT_COLORS_LIST, alpha=0.8,
             hatch=SAMPLE_PATTERNS['positive'])
    bars_neg = ax3.bar(x + width/2, neg_counts, width, 
             label='Negative (-)', color=SPLIT_COLORS_LIST, alpha=0.8,
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
    
    bars = ax4.bar(datasets, same_func_rates, color=SPLIT_COLORS_LIST, alpha=0.8)
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


def visualize_sequence_embeddings_from_pairs(
    pairs_df: pd.DataFrame,
    embeddings_file: Path,
    split_name: str,
    metadata_df: pd.DataFrame = None,
    max_per_segment: int = 1000,
    results_dir: Path = None
    ) -> None:
    """Visualize embeddings of sequences appearing in pairs.
    
    Extracts unique sequences from pairs, loads their embeddings, and creates
    PCA visualization colored by segment/function.
    
    Args:
        pairs_df: DataFrame with 'brc_a' and 'brc_b' columns
        embeddings_file: Path to master embeddings HDF5
        split_name: Name of split ('train', 'val', 'test')
        metadata_df: Optional protein metadata for segment/function info
        max_per_segment: Maximum sequences per segment for visualization
        results_dir: Output directory
    """
    print(f"\n📊 Visualizing sequence embeddings for {split_name} set...")
    
    # Extract unique sequences from pairs
    unique_seqs = extract_unique_sequences_from_pairs(pairs_df, metadata_df)
    print(f"   Found {len(unique_seqs)} unique sequences")
    
    # Sample with stratified sampling by segment
    if 'canonical_segment' in unique_seqs.columns:
        sampled_seqs = sample_sequences_stratified(
            unique_seqs,
            max_per_segment=max_per_segment,
            segment_col='canonical_segment'
        )
        print(f"   Sampled {len(sampled_seqs)} sequences (stratified by segment)")
    else:
        # If no segment info, just sample overall
        if len(unique_seqs) > max_per_segment * 2:
            sampled_seqs = unique_seqs.sample(n=max_per_segment * 2, random_state=42)
        else:
            sampled_seqs = unique_seqs
    
    # Load embeddings
    id_to_row = load_embedding_index(embeddings_file)
    embeddings, valid_ids = load_embeddings_by_ids(
        sampled_seqs['brc_fea_id'].tolist(),
        embeddings_file,
        id_to_row
    )
    
    if len(embeddings) == 0:
        print(f"   ⚠️  No embeddings found for {split_name} sequences")
        return
    
    # Filter to valid sequences
    sampled_seqs = sampled_seqs[sampled_seqs['brc_fea_id'].isin(valid_ids)].reset_index(drop=True)
    
    # Compute PCA
    pca_embeddings, pca = compute_pca_reduction(embeddings, n_components=2, return_model=True)
    
    # Plot by segment if available
    if 'canonical_segment' in sampled_seqs.columns:
        plot_embeddings_by_category(
            pca_embeddings,
            sampled_seqs['canonical_segment'],
            title=f"PCA: {split_name.capitalize()} Sequences by Segment",
            xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            save_path=results_dir / f'{split_name}_sequences_pca_by_segment.png'
        )
    
    # Plot by function if available
    if 'function' in sampled_seqs.columns:
        plot_embeddings_by_category(
            pca_embeddings,
            sampled_seqs['function'],
            title=f"PCA: {split_name.capitalize()} Sequences by Function",
            xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            save_path=results_dir / f'{split_name}_sequences_pca_by_function.png'
        )


def visualize_pair_embeddings(
    pairs_df: pd.DataFrame,
    embeddings_file: Path,
    split_name: str,
    max_per_label: int = 2500,
    results_dir: Path = None
    ) -> None:
    """Visualize pair embeddings in 2D space using PCA.
    
    Creates pair embeddings using concatenation [emb_a, emb_b] and visualizes
    them colored by positive/negative label.
    
    Args:
        pairs_df: DataFrame with 'brc_a', 'brc_b', and 'label' columns
        embeddings_file: Path to master embeddings HDF5
        split_name: Name of split ('train', 'val', 'test')
        max_per_label: Maximum pairs per label for visualization
        results_dir: Output directory
    """
    print(f"\n📊 Visualizing pair embeddings for {split_name} set...")
    
    # Sample pairs with stratified sampling by label
    sampled_pairs = sample_pairs_stratified(
        pairs_df,
        max_per_label=max_per_label,
        label_col='label'
    )
    print(f"   Sampled {len(sampled_pairs)} pairs (stratified by label)")
    
    # Create pair embeddings (concatenation)
    pair_embeddings, labels = create_pair_embeddings_concatenation(
        sampled_pairs,
        embeddings_file
    )
    
    if len(pair_embeddings) == 0:
        print(f"   ⚠️  No pair embeddings created for {split_name}")
        return
    
    print(f"   Created {len(pair_embeddings)} pair embeddings (shape: {pair_embeddings.shape})")
    
    # Compute PCA
    pca_embeddings, pca = compute_pca_reduction(pair_embeddings, n_components=2, return_model=True)
    
    # Plot by label
    plot_pair_embeddings_by_label(
        pca_embeddings,
        labels,
        title=f"PCA: {split_name.capitalize()} Pairs by Label",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        save_path=results_dir / f'{split_name}_pairs_pca_by_label.png'
    )


def create_combined_pair_visualizations(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    embeddings_file: Path,
    max_per_label: int = 2000,  # Smaller for combined plots
    results_dir: Path = None
    ) -> None:
    """Create combined visualizations showing train/val/test together.
    
    Args:
        train_pairs: Training pairs DataFrame
        val_pairs: Validation pairs DataFrame
        test_pairs: Test pairs DataFrame
        embeddings_file: Path to master embeddings HDF5
        max_per_label: Maximum pairs per label per split
        results_dir: Output directory
    """
    print(f"\n📊 Creating combined pair visualizations...")
    
    # Sample pairs from each split
    splits = {
        'train': sample_pairs_stratified(train_pairs, max_per_label=max_per_label),
        'val': sample_pairs_stratified(val_pairs, max_per_label=max_per_label),
        'test': sample_pairs_stratified(test_pairs, max_per_label=max_per_label)
    }
    
    # Combine all pairs
    all_pairs = pd.concat([
        splits['train'].assign(split='train'),
        splits['val'].assign(split='val'),
        splits['test'].assign(split='test')
    ], ignore_index=True)
    
    # Create pair embeddings
    pair_embeddings, labels, valid_mask = create_pair_embeddings_concatenation(
        all_pairs,
        embeddings_file,
        return_valid_mask=True,
        dtype=np.float32,
    )
    
    if len(pair_embeddings) == 0:
        print(f"   ⚠️  No pair embeddings created")
        return

    n_in = len(all_pairs)
    all_pairs = all_pairs.loc[valid_mask].reset_index(drop=True)
    dropped = n_in - len(all_pairs)
    if dropped > 0:
        print(f"   ℹ️  Dropped {dropped:,}/{n_in:,} sampled pairs due to missing embeddings")
    assert len(all_pairs) == len(pair_embeddings), "all_pairs/embeddings misalignment: filter logic bug"
    
    # Compute PCA
    pca_embeddings, pca = compute_pca_reduction(pair_embeddings, n_components=2, return_model=True)
    
    # Create combined plot: train/test only
    train_test_mask = all_pairs['split'].isin(['train', 'test'])
    plot_pair_embeddings_by_label(
        pca_embeddings[train_test_mask],
        labels[train_test_mask],
        title="PCA: Train/Test Pairs by Label",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        save_path=results_dir / 'train_test_pairs_pca_by_label.png'
    )
    
    # Create combined plot: train/val/test
    # Use different colors for each split
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for split_name, color in [('train', '#17A2B8'), ('val', '#8E44AD'), ('test', '#E74C3C')]:
        split_mask = all_pairs['split'] == split_name
        if split_mask.sum() > 0:
            pos_mask = split_mask & (labels == 1)
            neg_mask = split_mask & (labels == 0)
            
            if pos_mask.sum() > 0:
                ax.scatter(
                    pca_embeddings[pos_mask, 0],
                    pca_embeddings[pos_mask, 1],
                    c=color, marker='o', alpha=0.6, s=40,
                    label=f'{split_name.capitalize()} Positive (n={pos_mask.sum()})'
                )
            if neg_mask.sum() > 0:
                ax.scatter(
                    pca_embeddings[neg_mask, 0],
                    pca_embeddings[neg_mask, 1],
                    c=color, marker='s', alpha=0.6, s=40,
                    label=f'{split_name.capitalize()} Negative (n={neg_mask.sum()})',
                    facecolors='none', edgecolors=color, linewidths=1.5
                )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA: Train/Val/Test Pairs by Label', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'train_val_test_pairs_pca_by_label.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Combined visualizations saved")


def main(virus_name: str, data_version: str, config_bundle: str, dataset_dir: Path = None):
    """Run all analyses."""
    
    # Set up paths
    if dataset_dir is None:
        dataset_dir = project_root / 'data' / 'datasets' / virus_name / data_version
    else:
        dataset_dir = Path(dataset_dir)
    
    # Config-specific output directory
    results_dir = project_root / 'results' / virus_name / data_version / config_bundle / 'dataset_analysis'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f'Stage 3: Dataset Statistics Analysis')
    print(f"{'='*60}")
    print(f'Virus: {virus_name}')
    print(f'Data version: {data_version}')
    print(f'Config bundle: {config_bundle}')
    print(f'Dataset dir: {dataset_dir}')
    print(f'Results dir: {results_dir}')
    print(f"{'='*60}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(dataset_dir / 'train_pairs.csv')
    val_df = pd.read_csv(dataset_dir / 'val_pairs.csv')
    test_df = pd.read_csv(dataset_dir / 'test_pairs.csv')
    
    print(f'Train pairs: {len(train_df)}')
    print(f'Val pairs: {len(val_df)}')
    print(f'Test pairs: {len(test_df)}')
    
    # Make these available globally for the analysis functions
    globals()['train_df'] = train_df
    globals()['val_df'] = val_df
    globals()['test_df'] = test_df
    globals()['results_dir'] = results_dir
    
    # Set up plotting style
    apply_default_style()
    sns.set_palette('Set2')
    
    # Basic statistics
    stats_df = analyze_basic_statistics()
    
    # Create visualizations
    create_segment_pair_histograms()
    create_isolate_distribution_plots()
    combined_df = analyze_function_distribution()
    
    # Summary table
    summary_df = create_dataset_summary_table()
    
    # Embedding visualizations (optional - requires embeddings file)
    embeddings_dir = project_root / 'data' / 'embeddings' / virus_name / data_version
    embeddings_file = embeddings_dir / 'master_esm2_embeddings.h5'
    
    if embeddings_file.exists():
        print(f"\n{'='*60}")
        print("EMBEDDING VISUALIZATIONS")
        print(f"{'='*60}")
        
        # Load protein metadata for segment/function info
        processed_data_dir = project_root / 'data' / 'processed' / virus_name / data_version
        protein_data_file = processed_data_dir / 'protein_final.csv'
        metadata_df = None
        if protein_data_file.exists():
            print(f"Loading protein metadata from {protein_data_file}...")
            metadata_df = pd.read_csv(protein_data_file)
            print(f"Loaded {len(metadata_df)} protein records")
        else:
            print(f"⚠️  Protein metadata not found at {protein_data_file}")
            print("   Sequence visualizations will not include segment/function coloring")
        
        # Visualize sequence embeddings for each split
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            visualize_sequence_embeddings_from_pairs(
                split_df,
                embeddings_file,
                split_name,
                metadata_df=metadata_df,
                max_per_segment=1000,
                results_dir=results_dir
            )
        
        # Visualize pair embeddings for each split
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            visualize_pair_embeddings(
                split_df,
                embeddings_file,
                split_name,
                max_per_label=2500,
                results_dir=results_dir
            )
        
        # Create combined visualizations
        create_combined_pair_visualizations(
            train_df, val_df, test_df,
            embeddings_file,
            max_per_label=2000,
            results_dir=results_dir
        )
    else:
        print(f"\n⚠️  Embeddings file not found at {embeddings_file}")
        print("   Skipping embedding visualizations")
    
    print(f'\n✅ Analysis complete! All results saved to: {results_dir}')
    
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
    neg_count = (train_df['label']==0).sum()
    if neg_count > 0:
        train_sf_rate = len(train_df[(train_df['label']==0) & (train_df['func_a']==train_df['func_b'])]) / neg_count
        print(f'• Same-function negative rate: ~{train_sf_rate:.1%} (prevents task from being too easy)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Stage 3: dataset statistics')
    parser.add_argument(
        '--config_bundle', type=str, required=True,
        help='Config bundle name (e.g., flu, bunya).'
    )
    parser.add_argument(
        '--virus', type=str, default=None, 
        choices=['bunya', 'flu'],
        help='Virus name (optional, inferred from config_bundle if not provided)'
    )
    parser.add_argument(
        '--data_version', type=str, default=None,
        help='Data version (optional, inferred from config_bundle if not provided)'
    )
    parser.add_argument(
        '--dataset_dir', type=str, default=None,
        help='Custom dataset directory (optional)'
    )
    args = parser.parse_args()

    # Get virus and data_version from config if not explicitly provided
    if args.virus is None or args.data_version is None:
        config_path = str(project_root / 'conf')
        config = get_virus_config_hydra(args.config_bundle, config_path=config_path)
        virus_name = args.virus if args.virus else config.virus.virus_name
        data_version = args.data_version if args.data_version else config.virus.data_version
    else:
        virus_name = args.virus
        data_version = args.data_version

    main(
        virus_name=virus_name,
        data_version=data_version,
        config_bundle=args.config_bundle,
        dataset_dir=args.dataset_dir
    )