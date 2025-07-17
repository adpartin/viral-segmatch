"""
Analyze preprocessing results (Stage 1) for Bunyavirales protein data.

This script includes analysis of the data preprocessing pipeline including:
- Data quality metrics
- Segment assignment statistics
- Filtering impact analysis
- Duplicate handling results
- Sequence quality assessment
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.protein_utils import analyze_protein_ambiguities, summarize_ambiguities

# Configuration
VIRUS_NAME = 'bunya'
DATA_VERSION = 'April_2025'
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)

# Global color scheme for segments (consistent across all visualizations)
SEGMENT_COLORS = {
    'S': '#1f77b4',  # Blue
    'M': '#ff7f0e',  # Orange
    'L': '#2ca02c'   # Green
}

# Standard segment order for consistent visualization
SEGMENT_ORDER = ['S', 'M', 'L']

# Define paths
main_data_dir = project_root / 'data'
processed_data_dir = main_data_dir / 'processed' / VIRUS_NAME / DATA_VERSION
results_dir = project_root / 'results' / VIRUS_NAME / DATA_VERSION / 'preprocess_analysis'
results_dir.mkdir(parents=True, exist_ok=True)

print(f'Processed data directory: {processed_data_dir}')
print(f'Results directory: {results_dir}')

# Define core and auxiliary functions
if DATA_VERSION == 'April_2025':
    core_functions = [
        'RNA-dependent RNA polymerase',
        'Pre-glycoprotein polyprotein GP complex',
        'Nucleocapsid protein'
    ]
    aux_functions = [
        'Bunyavirales mature nonstructural membrane protein (NSm)',
        'Bunyavirales small nonstructural protein NSs',
        'Phenuiviridae mature nonstructural 78-kD protein',
    ]
elif DATA_VERSION == 'Feb_2025':
    core_functions = [
        'RNA-dependent RNA polymerase (L protein)',
        'Pre-glycoprotein polyprotein GP complex (GPC protein)',
        'Nucleocapsid protein (N protein)'
    ]
    aux_functions = [
        'Bunyavirales mature nonstructural membrane protein (NSm protein)',
        'Bunyavirales small nonstructural protein (NSs protein)',
        'Phenuiviridae mature nonstructural 78-kD protein',
        'Small Nonstructural Protein NSs (NSs Protein)',
    ]
else:
    raise ValueError(f'Unknown data_version: {DATA_VERSION}.')


def load_preprocessing_data():
    """Load all preprocessing stage data files."""
    data_files = {
        'initial': 'protein_agg_from_GTOs.csv',
        'after_segment_assignment': 'protein_assigned_segments.csv',
        'after_basic_filtering': 'protein_filtered_basic.csv',
        'final': 'protein_final.csv'
    }

    # breakpoint()
    data = {}
    for stage, fname in data_files.items():
        fpath = processed_data_dir / fname
        if fpath.exists():
            data[stage] = pd.read_csv(fpath)
            print(f'Loaded {stage}: {data[stage].shape}')
        else:
            print(f'Warning: {fname} not found')
    
    return data


def analyze_data_count_flow_metrics(data):
    """Analyze how data flows through the preprocessing pipeline."""
    metrics = {}

    # breakpoint()
    for stage, df in data.items():
        if df is not None:
            metrics[stage] = {
                'total_records': int(len(df)),
                'unique_proteins': df['prot_seq'].nunique() if 'prot_seq' in df.columns else 0,
                'unique_assemblies': df['assembly_id'].nunique() if 'assembly_id' in df.columns else 0,
                'unique_files': df['file'].nunique() if 'file' in df.columns else 0,
                'core_proteins': len(df[df['function'].isin(core_functions)]) if 'function' in df.columns else 0,
                'aux_proteins': len(df[df['function'].isin(aux_functions)]) if 'function' in df.columns else 0,
                'assigned_segments': df['canonical_segment'].notna().sum() if 'canonical_segment' in df.columns else 0,
                'mean_seq_length': df['prot_seq'].str.len().mean() if 'prot_seq' in df.columns else 0,
                'median_seq_length': df['prot_seq'].str.len().median() if 'prot_seq' in df.columns else 0,
            }

    int_columns = ['total_records', 'unique_proteins', 'unique_assemblies',
        'unique_files', 'core_proteins', 'aux_proteins']
    float_columns = ['assigned_segments', 'mean_seq_length', 'median_seq_length']
    mm = pd.DataFrame(metrics).T
    mm[int_columns] = mm[int_columns].astype(int)
    mm[float_columns] = mm[float_columns].astype(float)
    mm[float_columns] = mm[float_columns].round(2)
    mm.to_csv(results_dir / 'preprocessing_flow_metrics.csv', sep=',', index=True)

    return mm


def create_data_count_flow_visualization(metrics):
    """Create visualization showing data count flow through preprocessing steps."""
    # breakpoint()
    metrics = metrics.T
    stages = list(metrics.keys())
    total_records = [int(metrics[stage]['total_records']) for stage in stages]
    unique_proteins = [int(metrics[stage]['unique_proteins']) for stage in stages]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Total records flow
    ax1.plot(stages, total_records, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_title('Total Records Through Preprocessing Pipeline', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Records', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (stage, count) in enumerate(zip(stages, total_records)):
        if i > 0:
            pct_change = ((count - total_records[0]) / total_records[0]) * 100
            ax1.annotate(f'{pct_change:+.1f}%', 
                        xy=(i, count), xytext=(0, 20), 
                        textcoords='offset points', ha='center',
                        fontsize=10, color='red')
    
    # Unique proteins flow
    ax2.plot(stages, unique_proteins, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_title('Unique Proteins Through Preprocessing Pipeline', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Unique Proteins', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (stage, count) in enumerate(zip(stages, unique_proteins)):
        if i > 0:
            pct_change = ((count - unique_proteins[0]) / unique_proteins[0]) * 100
            ax2.annotate(f'{pct_change:+.1f}%', 
                        xy=(i, count), xytext=(0, 20), 
                        textcoords='offset points', ha='center',
                        fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'data_count_flow_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    return fig


def analyze_segment_assignment(data):
    """Analyze the segment assignment process."""
    if 'final' not in data:
        return None
    
    # breakpoint()
    df = data['final']
    
    # Overall segment assignment statistics
    segment_stats = df['canonical_segment'].value_counts()
    assignment_rate = df['canonical_segment'].notna().mean()
    
    # Core protein segment assignment
    core_df = df[df['function'].isin(core_functions)]
    core_segment_stats = core_df['canonical_segment'].value_counts()
    core_assignment_rate = core_df['canonical_segment'].notna().mean()
    
    # Auxiliary protein segment assignment
    aux_df = df[df['function'].isin(aux_functions)]
    aux_segment_stats = aux_df['canonical_segment'].value_counts()
    aux_assignment_rate = aux_df['canonical_segment'].notna().mean()
    
    return {
        'overall_stats': segment_stats,
        'assignment_rate': assignment_rate,
        'core_stats': core_segment_stats,
        'core_assignment_rate': core_assignment_rate,
        'aux_stats': aux_segment_stats,
        'aux_assignment_rate': aux_assignment_rate
    }


def create_segment_assignment_visualization(data, separate_figures=False):
    """Create comprehensive segment assignment visualizations.
    
    Args:
        data: Dictionary containing data at different stages
        separate_figures: If True, create separate figures for each plot
    """
    if 'final' not in data:
        return None
    
    # breakpoint()
    df = data['final']
    
    if separate_figures:
        return _create_separate_segment_figures(df)
    else:
        return _create_combined_segment_figure(df)


def _create_combined_segment_figure(df):
    """Create combined segment assignment figure."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall segment distribution
    segment_counts = df['canonical_segment'].value_counts()
    # Reorder to standard segment order
    segment_counts = segment_counts.reindex(SEGMENT_ORDER)
    
    ax1 = axes[0, 0]
    bars = ax1.bar(segment_counts.index, segment_counts.values, 
                   color=[SEGMENT_COLORS[seg] for seg in SEGMENT_ORDER if seg in segment_counts.index])
    ax1.set_title('Overall Segment Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Proteins', fontsize=14)  # Larger Y-axis label
    ax1.set_xlabel('Segment', fontsize=14)  # Larger X-axis label
    ax1.tick_params(axis='x', rotation=0, labelsize=12)  # Larger X-axis tick labels
    ax1.tick_params(axis='y', labelsize=12)  # Larger Y-axis tick labels
    
    # Add count and percentage labels inside bars
    total = segment_counts.sum()
    for i, (bar, count) in enumerate(zip(bars, segment_counts.values)):
        height = bar.get_height()
        pct = count / total * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height/2, 
                f'{count}\n({pct:.1f}%)', 
                ha='center', va='center', fontweight='bold', color='white', fontsize=12)  # Larger font for labels inside bars
    
    # 2. Core protein segment assignment
    core_df = df[df['function'].isin(core_functions)]
    core_segment_func = pd.crosstab(core_df['function'], core_df['canonical_segment'])
    ax2 = axes[0, 1]
    core_bars = core_segment_func.plot(kind='bar', ax=ax2, stacked=True)
    ax2.set_title('Core Protein Segment Assignment', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Proteins')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add count and percentage labels for core proteins
    for i, (func, row) in enumerate(core_segment_func.iterrows()):
        cumulative = 0
        total_func = row.sum()
        for j, (segment, count) in enumerate(row.items()):
            if count > 0:
                pct = count / total_func * 100
                ax2.text(i, cumulative + count/2, f'{count}\n({pct:.1f}%)', 
                        ha='center', va='center', fontweight='bold', fontsize=9)
                cumulative += count
    
    # 3. Auxiliary protein segment assignment
    aux_df = df[df['function'].isin(aux_functions)]
    if not aux_df.empty:
        aux_segment_func = pd.crosstab(aux_df['function'], aux_df['canonical_segment'])
        ax3 = axes[1, 0]
        aux_segment_func.plot(kind='bar', ax=ax3, stacked=True)
        ax3.set_title('Auxiliary Protein Segment Assignment', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Proteins')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Assignment rate by replicon type
    replicon_assignment = df.groupby('replicon_type').agg({
        'canonical_segment': lambda x: x.notna().mean()
    }).sort_values('canonical_segment', ascending=False)
    
    ax4 = axes[1, 1]
    replicon_assignment.plot(kind='bar', ax=ax4, color='coral')
    ax4.set_title('Assignment Rate by Replicon Type', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Assignment Rate')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 1)
    
    # Add percentage labels for replicon assignment
    for i, v in enumerate(replicon_assignment['canonical_segment'].values):
        ax4.text(i, v + 0.02, f'{v*100:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'segment_assignment_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    return fig


def _create_separate_segment_figures(df):
    """Create separate figures for each segment assignment plot."""
    figures = []
    
    # Figure 1: Overall segment distribution
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    segment_counts = df['canonical_segment'].value_counts()
    # Reorder to standard segment order
    segment_counts = segment_counts.reindex(SEGMENT_ORDER)
    
    bars = ax1.bar(segment_counts.index, segment_counts.values, 
                   color=[SEGMENT_COLORS[seg] for seg in SEGMENT_ORDER if seg in segment_counts.index])
    ax1.set_title('Overall Segment Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Proteins', fontsize=14)  # Larger Y-axis label
    ax1.set_xlabel('Segment', fontsize=14)  # Larger X-axis label
    ax1.tick_params(axis='x', rotation=0, labelsize=12)  # Larger X-axis tick labels
    ax1.tick_params(axis='y', labelsize=12)  # Larger Y-axis tick labels
    
    # Add count and percentage labels inside bars
    total = segment_counts.sum()
    for i, (bar, count) in enumerate(zip(bars, segment_counts.values)):
        height = bar.get_height()
        pct = count / total * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height/2, 
                f'{count}\n({pct:.1f}%)', 
                ha='center', va='center', fontweight='bold', color='white', fontsize=12)  # Larger font for labels inside bars
    
    plt.tight_layout()
    plt.savefig(results_dir / 'segment_distribution.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    figures.append(fig1)
    
    # Figure 2: Core protein segment assignment
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    core_df = df[df['function'].isin(core_functions)]
    core_segment_func = pd.crosstab(core_df['function'], core_df['canonical_segment'])
    core_segment_func.plot(kind='bar', ax=ax2, stacked=True)
    ax2.set_title('Core Protein Segment Assignment', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Proteins')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add count and percentage labels for core proteins
    for i, (func, row) in enumerate(core_segment_func.iterrows()):
        cumulative = 0
        total_func = row.sum()
        for j, (segment, count) in enumerate(row.items()):
            if count > 0:
                pct = count / total_func * 100
                ax2.text(i, cumulative + count/2, f'{count}\n({pct:.1f}%)', 
                        ha='center', va='center', fontweight='bold', fontsize=9)
                cumulative += count
    
    plt.tight_layout()
    plt.savefig(results_dir / 'core_protein_assignment.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    figures.append(fig2)
    
    # Figure 3: Auxiliary protein segment assignment
    aux_df = df[df['function'].isin(aux_functions)]
    if not aux_df.empty:
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        aux_segment_func = pd.crosstab(aux_df['function'], aux_df['canonical_segment'])
        aux_segment_func.plot(kind='bar', ax=ax3, stacked=True)
        ax3.set_title('Auxiliary Protein Segment Assignment', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Proteins')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'auxiliary_protein_assignment.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        figures.append(fig3)
    
    # Figure 4: Assignment rate by replicon type
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
    replicon_assignment = df.groupby('replicon_type').agg({
        'canonical_segment': lambda x: x.notna().mean()
    }).sort_values('canonical_segment', ascending=False)
    
    replicon_assignment.plot(kind='bar', ax=ax4, color='coral')
    ax4.set_title('Assignment Rate by Replicon Type', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Assignment Rate')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 1)
    
    # Add percentage labels
    for i, v in enumerate(replicon_assignment['canonical_segment'].values):
        ax4.text(i, v + 0.02, f'{v*100:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'replicon_assignment_rate.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    figures.append(fig4)
    
    return figures


def analyze_sequence_quality(data):
    """Analyze protein sequence quality metrics."""
    if 'final' not in data:
        return None
    
    # breakpoint()
    df = data['final']
    
    # Sequence length distribution
    seq_lengths = df['prot_seq'].str.len()
    
    # Analyze ambiguities
    df_with_ambig = analyze_protein_ambiguities(df)
    ambig_summary = summarize_ambiguities(df_with_ambig)
    
    # Quality metrics by segment
    segment_quality = df.groupby('canonical_segment').agg({
        'prot_seq': lambda x: x.str.len().mean(),
        'length': 'mean'
    }).round(2)
    
    return {
        'seq_lengths': seq_lengths,
        'ambig_summary': ambig_summary,
        'segment_quality': segment_quality,
        'length_stats': seq_lengths.describe()
    }


def create_sequence_quality_visualization(data, separate_figures=False):
    """Create sequence quality visualizations.
    
    Args:
        data: Dictionary containing data at different stages
        separate_figures: If True, create separate figures for each plot
    """
    if 'final' not in data:
        return None
    
    # breakpoint()
    df = data['final']
    
    if separate_figures:
        return _create_separate_quality_figures(df)
    else:
        return _create_combined_quality_figure(df)


def _create_combined_quality_figure(df):
    """Create combined sequence quality figure."""
    seq_lengths = df['prot_seq'].str.len()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sequence length distribution
    ax1 = axes[0, 0]
    seq_lengths.hist(bins=50, ax=ax1, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Protein Sequence Length Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sequence Length (amino acids)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(seq_lengths.mean(), color='red', linestyle='--', 
                label=f'Mean: {seq_lengths.mean():.0f}')
    ax1.axvline(seq_lengths.median(), color='orange', linestyle='--', 
                label=f'Median: {seq_lengths.median():.0f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sequence length by segment
    ax2 = axes[0, 1]
    segment_lengths = []
    segment_labels = []
    for segment in SEGMENT_ORDER:
        seg_data = df[df['canonical_segment'] == segment]['prot_seq'].str.len()
        if not seg_data.empty:
            segment_lengths.append(seg_data)
            segment_labels.append(f'{segment}\n(n={len(seg_data)})')
    
    if segment_lengths:
        ax2.boxplot(segment_lengths, tick_labels=segment_labels)
        ax2.set_title('Sequence Length by Segment', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Sequence Length (amino acids)')
        ax2.grid(True, alpha=0.3)
    
    # 3. Function distribution
    ax3 = axes[1, 0]
    func_counts = df['function'].value_counts().head(10)
    func_counts.plot(kind='barh', ax=ax3)
    ax3.set_title('Top 10 Protein Functions', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Number of Proteins')
    
    # 4. Quality distribution by segment
    ax4 = axes[1, 1]
    quality_counts = df.groupby(['canonical_segment', 'quality']).size().unstack(fill_value=0)
    quality_counts.plot(kind='bar', ax=ax4, stacked=True)
    ax4.set_title('Quality Distribution by Segment', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Proteins')
    ax4.tick_params(axis='x', rotation=0)
    ax4.legend(title='Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'sequence_quality_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    return fig


def _create_separate_quality_figures(df):
    """Create separate figures for each sequence quality plot."""
    figures = []
    seq_lengths = df['prot_seq'].str.len()
    
    # Figure 1: Sequence length distribution by segment
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create histogram for each segment
    segment_data = []
    segment_labels = []
    segment_colors_used = []
    for segment in SEGMENT_ORDER:
        seg_lengths = df[df['canonical_segment'] == segment]['prot_seq'].str.len()
        if not seg_lengths.empty:
            segment_data.append(seg_lengths)
            segment_labels.append(f'{segment} (n={len(seg_lengths)})')
            segment_colors_used.append(SEGMENT_COLORS[segment])
    
    if segment_data:
        # Create overlapping histograms
        ax1.hist(segment_data, bins=50, alpha=0.7, 
                color=segment_colors_used,
                label=segment_labels, edgecolor='black', linewidth=0.5)
        
        # Add overall statistics lines
        ax1.axvline(seq_lengths.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Overall Mean: {seq_lengths.mean():.0f}')
        ax1.axvline(seq_lengths.median(), color='darkred', linestyle=':', linewidth=2,
                    label=f'Overall Median: {seq_lengths.median():.0f}')
    else:
        # Fallback to original if no segment data
        seq_lengths.hist(bins=50, ax=ax1, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(seq_lengths.mean(), color='red', linestyle='--', 
                    label=f'Mean: {seq_lengths.mean():.0f}')
        ax1.axvline(seq_lengths.median(), color='orange', linestyle='--', 
                    label=f'Median: {seq_lengths.median():.0f}')
    
    ax1.set_title('Protein Sequence Length Distribution by Segment', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sequence Length (amino acids)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'sequence_length_distribution.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    figures.append(fig1)
    
    # Figure 2: Sequence length by segment
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    segment_lengths = []
    segment_labels = []
    for segment in SEGMENT_ORDER:
        seg_data = df[df['canonical_segment'] == segment]['prot_seq'].str.len()
        if not seg_data.empty:
            segment_lengths.append(seg_data)
            segment_labels.append(f'{segment}\n(n={len(seg_data)})')
    
    if segment_lengths:
        ax2.boxplot(segment_lengths, tick_labels=segment_labels)
        ax2.set_title('Sequence Length by Segment', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Sequence Length (amino acids)', fontsize=12)
        ax2.set_xlabel('Segment', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'sequence_length_by_segment.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        figures.append(fig2)
    
    # Figure 3: Function distribution
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    func_counts = df['function'].value_counts().head(10)
    func_counts.plot(kind='barh', ax=ax3, color='lightcoral')
    ax3.set_title('Top 10 Protein Functions', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Number of Proteins', fontsize=12)
    ax3.set_ylabel('Function', fontsize=12)
    
    # Add count labels on bars
    for i, v in enumerate(func_counts.values):
        ax3.text(v + max(func_counts.values) * 0.01, i, str(v), 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'function_distribution.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    figures.append(fig3)
    
    # Figure 4: Quality distribution by segment
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
    quality_counts = df.groupby(['canonical_segment', 'quality']).size().unstack(fill_value=0)
    quality_counts.plot(kind='bar', ax=ax4, stacked=True)
    ax4.set_title('Quality Distribution by Segment', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Proteins', fontsize=12)
    ax4.set_xlabel('Segment', fontsize=12)
    ax4.tick_params(axis='x', rotation=0)
    ax4.legend(title='Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'quality_distribution_by_segment.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    figures.append(fig4)
    
    return figures


def analyze_duplicate_handling(data):
    """Analyze the duplicate handling process."""
    if 'final' not in data:
        return None
    
    # breakpoint()
    df = data['final']
    
    # Find duplicates in final data
    seq_duplicates = df[df.duplicated(subset=['prot_seq'], keep=False)]
    
    if seq_duplicates.empty:
        return {
            'total_duplicates': 0,
            'unique_sequences': 0,
            'duplicate_stats': None
        }
    
    # Analyze duplicate patterns
    dup_stats = seq_duplicates.groupby('prot_seq').agg({
        'file': 'nunique',
        'assembly_id': 'nunique',
        'function': lambda x: list(set(x)),
        'canonical_segment': lambda x: list(set(x.dropna())),
        'assembly_prefix': lambda x: list(set(x))
    }).sort_values('file', ascending=False)
    
    return {
        'total_duplicates': len(seq_duplicates),
        'unique_sequences': len(dup_stats),
        'duplicate_stats': dup_stats,
        'files_per_duplicate': dup_stats['file'].describe()
    }


def create_summary_statistics(data, metrics):
    """Create comprehensive summary statistics."""
    # breakpoint()
    metrics = metrics.T
    summary = {
        'pipeline_overview': {
            'total_stages': len(data),
            'initial_records': metrics['initial']['total_records'] if 'initial' in metrics else 0,
            'final_records': metrics['final']['total_records'] if 'final' in metrics else 0,
            'records_retained': None,
            'retention_rate': None
        },
        'segment_assignment': {},
        'quality_metrics': {},
        'duplicate_handling': {}
    }
    
    if 'initial' in metrics and 'final' in metrics:
        initial_count = metrics['initial']['total_records']
        final_count = metrics['final']['total_records']
        summary['pipeline_overview']['records_retained'] = final_count
        summary['pipeline_overview']['retention_rate'] = final_count / initial_count if initial_count > 0 else 0
    
    # Segment assignment summary
    if 'final' in data:
        df = data['final']
        segment_counts = df['canonical_segment'].value_counts()
        summary['segment_assignment'] = {
            'total_assigned': df['canonical_segment'].notna().sum(),
            'assignment_rate': df['canonical_segment'].notna().mean(),
            'segment_distribution': segment_counts.to_dict()
        }
    
    return summary


def generate_comprehensive_report(data, metrics, summary):
    """Generate a comprehensive text report."""
    report = []
    report.append("=" * 80)
    report.append("STAGE 1: PREPROCESSING ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Data Version: {DATA_VERSION}")
    report.append("")
    # breakpoint()
    
    # Pipeline Overview
    report.append("PIPELINE OVERVIEW")
    report.append("-" * 40)
    if 'pipeline_overview' in summary:
        overview = summary['pipeline_overview']
        report.append(f"Initial Records: {overview['initial_records']:,}")
        report.append(f"Final Records: {overview['final_records']:,}")
        if overview['retention_rate'] is not None:
            report.append(f"Retention Rate: {overview['retention_rate']:.2%}")
    report.append("")
    
    # Stage-by-stage metrics
    report.append("STAGE-BY-STAGE METRICS")
    report.append("-" * 40)
    metrics = metrics.T
    for stage, stage_metrics in metrics.items():
        report.append(f"{stage.upper()}:")
        report.append(f"  Total Records: {stage_metrics['total_records']:,}")
        report.append(f"  Unique Proteins: {stage_metrics['unique_proteins']:,}")
        report.append(f"  Unique Assemblies: {stage_metrics['unique_assemblies']:,}")
        report.append(f"  Core Proteins: {stage_metrics['core_proteins']:,}")
        report.append(f"  Aux Proteins: {stage_metrics['aux_proteins']:,}")
        if stage_metrics['assigned_segments'] > 0:
            report.append(f"  Assigned Segments: {stage_metrics['assigned_segments']:,}")
        report.append(f"  Mean Seq Length: {stage_metrics['mean_seq_length']:.1f}")
        report.append("")
    
    # Segment Assignment
    if 'segment_assignment' in summary:
        report.append("SEGMENT ASSIGNMENT")
        report.append("-" * 40)
        seg_assign = summary['segment_assignment']
        report.append(f"Total Assigned: {seg_assign['total_assigned']:,}")
        report.append(f"Assignment Rate: {seg_assign['assignment_rate']:.2%}")
        report.append("Segment Distribution:")
        for segment, count in seg_assign['segment_distribution'].items():
            pct = count / seg_assign['total_assigned'] * 100
            report.append(f"  {segment}: {count:,} ({pct:.1f}%)")
        report.append("")
    
    # Data Quality
    report.append("DATA QUALITY SUMMARY")
    report.append("-" * 40)
    if 'final' in data:
        df = data['final']
        report.append(f"Total Unique Sequences: {df['prot_seq'].nunique():,}")
        report.append(f"Mean Sequence Length: {df['prot_seq'].str.len().mean():.1f}")
        report.append(f"Median Sequence Length: {df['prot_seq'].str.len().median():.1f}")
        
        quality_dist = df['quality'].value_counts()
        report.append("Quality Distribution:")
        for quality, count in quality_dist.items():
            pct = count / len(df) * 100
            report.append(f"  {quality}: {count:,} ({pct:.1f}%)")
    
    report.append("")
    report.append("=" * 80)
    
    # Save report
    with open(results_dir / 'preprocessing_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)


def main():
    """Main analysis function."""

    print('Stage 1: Preprocessing Analysis')
    print('=' * 50)

    # Load data
    print('\n1. Load preprocessing data.')
    data = load_preprocessing_data()

    # Analyze data flow
    print('\n2. Analyze data count flow metrics.')
    metrics = analyze_data_count_flow_metrics(data)

    # Create data count flow visualization
    print('\n3. Create data count flow visualization.')
    create_data_count_flow_visualization(metrics)
    
    # Analyze segment assignment
    print('\n4. Analyze segment assignment.')
    segment_analysis = analyze_segment_assignment(data)
    create_segment_assignment_visualization(data, separate_figures=True)
    
    # Analyze sequence quality
    print('\n5. Analyze sequence quality.')
    quality_analysis = analyze_sequence_quality(data)
    create_sequence_quality_visualization(data, separate_figures=True)
    
    # Analyze duplicate handling
    print('\n6. Analyze duplicate handling.')
    duplicate_analysis = analyze_duplicate_handling(data)
    
    # Create summary statistics
    print('\n7. Create summary statistics.')
    summary = create_summary_statistics(data, metrics)
    
    # Save detailed results
    print('\n8. Save detailed results.')
    results = {
        'metrics': metrics,
        'segment_analysis': segment_analysis,
        'quality_analysis': quality_analysis,
        'duplicate_analysis': duplicate_analysis,
        'summary': summary
    }
    
    # Save metrics as JSON
    with open(results_dir / 'preprocessing_metrics.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            return obj
        
        # Deep convert the results
        import json
        json_str = json.dumps(results, default=convert_numpy, indent=2)
        f.write(json_str)
    
    # Generate comprehensive report
    print('\n9. Generate comprehensive report.')
    report = generate_comprehensive_report(data, metrics, summary)
    
    print(f'\nAnalysis complete! Results saved in: {results_dir}')
    print(f'Key outputs:')
    print(f'  - Data flow visualization: data_flow_analysis.png')
    print(f'  - Segment assignment analysis: segment_assignment_analysis.png')
    print(f'  - Sequence quality analysis: sequence_quality_analysis.png')
    print(f'  - Comprehensive report: preprocessing_analysis_report.txt')
    print(f'  - Detailed metrics: preprocessing_metrics.json')
    
    return results

if __name__ == '__main__':
    results = main()