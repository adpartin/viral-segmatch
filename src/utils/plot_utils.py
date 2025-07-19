"""
Shared plotting utilities for viral-segmatch project.
"""

import matplotlib.pyplot as plt
import pandas as pd
from .plot_config import SEGMENT_COLORS, SEGMENT_ORDER


def plot_sequence_length_distribution(df, seq_column='prot_seq', segment_column='canonical_segment', 
                                     title='Sequence Length Distribution by Segment',
                                     show_esm2_limit=False, esm2_max_residues=None,
                                     save_path=None, show_plot=True, figsize=(10, 6)):
    """
    Create a standardized sequence length distribution plot by segment.
    
    Args:
        df: DataFrame containing protein data
        seq_column: Column name containing protein sequences
        segment_column: Column name containing segment assignments
        title: Plot title
        show_esm2_limit: Whether to show ESM-2 sequence limit line
        esm2_max_residues: ESM-2 maximum residues limit
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        figsize: Figure size tuple
    
    Returns:
        fig: matplotlib figure object
    """
    # Calculate sequence lengths
    seq_lengths = df[seq_column].str.len()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot histogram for each segment in standard order [S, M, L]
    available_segments = set(df[segment_column].dropna().unique())
    segment_data = []
    segment_labels = []
    segment_colors_used = []
    
    for segment in SEGMENT_ORDER:  # Use standard order S, M, L
        if segment in available_segments:
            seg_lengths = df[df[segment_column] == segment][seq_column].str.len()
            if not seg_lengths.empty:
                segment_data.append(seg_lengths)
                segment_labels.append(f'{segment} (n={len(seg_lengths)})')
                segment_colors_used.append(SEGMENT_COLORS[segment])
    
    # Create overlapping histograms
    if segment_data:
        ax.hist(segment_data, bins=50, alpha=0.7, 
               color=segment_colors_used, label=segment_labels, 
               edgecolor='black', linewidth=0.5)
    
    # Add ESM-2 limit line if requested
    if show_esm2_limit and esm2_max_residues is not None:
        ax.axvline(x=esm2_max_residues, color='red', linestyle='--', linewidth=2,
                  label=f'ESM-2 limit ({esm2_max_residues})')
    
    # Styling
    ax.set_xlabel('Sequence Length (amino acids)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig 