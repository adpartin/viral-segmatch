"""
Shared plotting utilities for viral-segmatch project.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_config import SEGMENT_COLORS, SEGMENT_ORDER, apply_default_style


def setup_plot_style(
    use_seaborn_palette: bool = True,
    palette: str = 'Set2') -> None:
    """Apply a consistent plot style across scripts.

    This is a thin wrapper around `plot_config.apply_default_style()` with optional seaborn palette.
    It's safe to call multiple times.
    """
    apply_default_style()
    if use_seaborn_palette:
        try:
            import seaborn as sns
            sns.set_palette(palette)
        except Exception:
            # seaborn is optional for many workflows; don't hard-fail here
            pass


def savefig(
    path: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = 'tight',
    facecolor: Optional[str] = 'white',
    close: bool = True,
    ) -> Path:
    """Save the current matplotlib figure with standardized defaults."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
    if close:
        plt.close()
    return p


def savefig_to_dirs(
    filename: str,
    output_dirs: Sequence[Union[str, Path]],
    dpi: int = 300,
    bbox_inches: str = 'tight',
    facecolor: Optional[str] = 'white',
    close: bool = True,
    ) -> list[Path]:
    """Save the current figure to multiple directories (same filename)."""
    saved: list[Path] = []
    for d in output_dirs:
        out_path = Path(d) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
        saved.append(out_path)
    if close:
        plt.close()
    return saved


def plot_sequence_length_distribution(
    df, seq_column='prot_seq',
    segment_column='canonical_segment', 
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


def size_barplot(
    sizes: Union[pd.Series, Sequence[int]],
    *,
    top_n: int,
    out_png: Union[str, Path],
    title: str,
    xlabel: str,
    ylabel: str,
    xticklabels: Optional[Sequence] = None,
    bar_color: str = '#4c72b0',
    dpi: int = 180,
    ) -> None:
    """Top-N ranked-size barplot shared by the 1D cluster and 2D CC size figures.

    Draws the `top_n` largest values as bars (rank-ordered, largest first), each
    labeled with its raw count and its % of the total (summed over ALL `sizes`, not
    just the drawn top-N). The caller supplies the text that distinguishes the two
    views: `title`, `xlabel`, `ylabel`, `xticklabels`.

    Args:
        sizes: bar heights in DESCENDING order (Series or sequence); the full set
            sets the % denominator and the total, only the first `top_n` are drawn.
        top_n: number of leading values to draw.
        out_png: output PNG path (parent dirs created).
        title: figure title (caller-composed).
        xlabel: x-axis label.
        ylabel: y-axis label.
        xticklabels: labels for the drawn bars (default: the Series index).
        bar_color: single bar fill color.
        dpi: raster resolution.
    """
    if not isinstance(sizes, pd.Series):
        sizes = pd.Series(list(sizes))
    total = float(sizes.sum())
    top = sizes.head(top_n)
    heights = top.to_numpy()
    pcts = heights / total * 100.0 if total else np.zeros(len(heights))
    labels = list(xticklabels) if xticklabels is not None else list(top.index)
    labels = [str(x) for x in labels[:len(top)]]

    fig, ax = plt.subplots(figsize=(max(9.0, len(top) * 0.55), 5.6))
    xs = np.arange(len(top))
    ax.bar(xs, heights, color=bar_color, edgecolor='black', linewidth=0.5)
    for x, c, p in zip(xs, heights, pcts):
        ax.annotate(f'{int(c):,}\n{p:.1f}%', xy=(x, c), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7, color='#222')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, heights.max() * 1.18 if len(heights) else 1.0)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
