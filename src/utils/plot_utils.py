"""
Shared plotting utilities for viral-segmatch project.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

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


def stacked_composition_barplot(
    comp: pd.DataFrame,
    *,
    item_col: str,
    category_col: str,
    value_col: str,
    item_order: Sequence,
    out_png: Union[str, Path],
    title: str,
    xlabel: str,
    ylabel: str,
    top_k: int = 4,
    item_labels: Optional[Sequence] = None,
    normalize: bool = False,
    label_min_frac: float = 0.06,
    dpi: int = 180,
    ) -> None:
    """One bar per item (in `item_order`), stacked by that item's top-`top_k` categories +
    a gray 'Others' block, colored by within-bar rank -- so a color marks rank, not a fixed
    category (each bar's blue is *that* item's largest category). Exposes concentration: a
    hub-dominated item is one solid block, a spread item is many thin segments.

    With `normalize=True` every bar is scaled to 1.0 so items of very different total size stay
    comparable (needed when one mega-CC dwarfs the tail); the y-axis then reads as share-of-item.
    Each top-`top_k` segment at least `label_min_frac` of its bar is labeled in place with its
    category id (white on the colored blocks, dark on the gray 'Others'); the dominant category's
    share of the bar is annotated above it.

    `comp` is long-form (`item_col`, `category_col`, `value_col`); it is filtered per item and
    sorted by `value_col` desc. Generic -- shared by the CC cluster-composition figures and
    (later) the per-CC metadata figures; nothing here is alphabet- or field-specific.
    """
    palette = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(max(9.0, len(item_order) * 0.6), 5.8))
    xs = np.arange(len(item_order))
    for x, item in zip(xs, item_order):
        g = comp[comp[item_col] == item].sort_values(value_col, ascending=False)
        vals = g[value_col].to_numpy(dtype=float)
        cats = g[category_col].astype(str).tolist()
        total = float(vals.sum())
        if total <= 0:
            continue
        bottom = 0.0
        for r in range(min(top_k, len(vals))):
            frac = vals[r] / total
            h = frac if normalize else vals[r]
            ax.bar(x, h, bottom=bottom, color=palette(r % 10), edgecolor='white', linewidth=0.4)
            if frac >= label_min_frac:
                ax.text(x, bottom + h / 2.0, cats[r], ha='center', va='center',
                        fontsize=6, color='white')
            bottom += h
        other_raw = float(vals[top_k:].sum()) if len(vals) > top_k else 0.0
        if other_raw > 0:
            other_frac = other_raw / total
            other_h = other_frac if normalize else other_raw
            ax.bar(x, other_h, bottom=bottom, color='#d9d9d9', edgecolor='white', linewidth=0.4)
            if other_frac >= label_min_frac:
                ax.text(x, bottom + other_h / 2.0, 'Others', ha='center', va='center',
                        fontsize=6, color='#222')
            bottom += other_h
        ax.annotate(f'{100.0 * vals[0] / total:.0f}%', xy=(x, bottom), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom', fontsize=6.5, color='#222')
    labels = list(item_labels) if item_labels is not None else list(item_order)
    labels = [str(v) for v in labels]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if normalize:
        ax.set_ylim(0, 1.10)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def select_categories_with_others(
    labels: Sequence,
    *,
    min_share: float = 0.01,
    cap: int = 12,
    palette='tab20',
    ) -> dict:
    """Split per-point categories into 'colored distinctly' vs a single 'Others'.

    A category is colored distinctly if its share of `labels` is >= `min_share`, keeping at
    most `cap` of them (largest first); every other point folds into 'Others'. Colors are
    assigned by size rank from `palette`. One shared definition of the 'top categories +
    gray Others' rule, so `umap_scatter` (and any other categorical scatter) agree.

    Returns a dict:
      'selected': list of (category, color_rgba, count, share), largest first (len <= cap);
      'is_selected': bool ndarray aligned to `labels` (True where a point's category is
                     colored distinctly);
      'others_count' / 'others_share': the folded remainder.
    """
    labels = np.asarray(labels)
    n = int(len(labels))
    total = float(n) if n else 1.0
    vc = pd.Series(labels).value_counts()  # largest first
    chosen = [c for c in vc.index if vc[c] / total >= min_share][:cap]
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        colors = [cmap(i % cmap.N) for i in range(max(1, len(chosen)))]
    else:
        colors = list(palette)  # explicit list of color specs
    selected = [(c, colors[i % len(colors)], int(vc[c]), int(vc[c]) / total)
                for i, c in enumerate(chosen)]
    is_selected = np.isin(labels, chosen)
    others_count = n - int(is_selected.sum())
    return {
        'selected': selected,
        'is_selected': is_selected,
        'others_count': others_count,
        'others_share': others_count / total,
    }


def umap_scatter(
    X,
    categories: Sequence,
    *,
    out_png: Union[str, Path],
    title: str,
    min_share: float = 0.01,
    cap: int = 12,
    palette='tab20',
    metric: str = 'cosine',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    others_color: str = '#4d4d4d',
    point_size: float = 12.0,
    other_size: float = 6.0,
    legend_title: Optional[str] = None,
    category_labeler=None,
    others_labeler=None,
    title_fontsize: int = 10,
    dpi: int = 200,
    ) -> dict:
    """Generic 2-D category scatter shared by the ESM-2 cluster UMAP and the CC / k-mer UMAPs.

    Reduces `X` (N, D) to 2-D with UMAP unless it is already 2-D (D == 2, used as coordinates),
    colors the categories with share >= `min_share` (<= `cap`, largest first) each distinctly,
    folds the rest into one dark-gray 'Others', adds a legend ('<cat> <share>%', overridable via
    `category_labeler(cat, count, share)` / `others_labeler(count, share)`) and title, and
    saves. `categories` is the length-N per-point label (cluster_id, fragment, ...). The
    representation-specific step (ESM-2 vs k-mer, any pre-SVD/PCA) is the caller's; this function
    is representation-agnostic. Extracted from `plot_clusters.plot_cluster_umap`.

    Returns {'n_points', 'n_categories', 'n_selected', 'others_share'}.
    """
    from .dim_reduction_utils import compute_umap_reduction  # lazy: pulls in umap/numba

    X = np.asarray(X)
    categories = np.asarray(categories)
    xy = X if (X.ndim == 2 and X.shape[1] == 2) else compute_umap_reduction(
        X, n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=seed)[0]

    sel = select_categories_with_others(categories, min_share=min_share, cap=cap, palette=palette)

    def _cat_label(cat, cnt, share):
        return f'{cat} {share:.0%}'

    def _oth_label(cnt, share):
        return f'Others {share:.1%} (n={cnt:,})'

    cat_label = category_labeler or _cat_label
    oth_label = others_labeler or _oth_label

    fig, ax = plt.subplots(figsize=(9, 8))
    other = ~sel['is_selected']
    if other.any():
        ax.scatter(xy[other, 0], xy[other, 1], s=other_size, c=others_color, linewidths=0,
                   rasterized=True, label=oth_label(sel['others_count'], sel['others_share']))
    for cat, color, cnt, share in sel['selected']:
        m = categories == cat
        ax.scatter(xy[m, 0], xy[m, 1], s=point_size, color=color, linewidths=0,
                   rasterized=True, label=cat_label(cat, cnt, share))
    ax.legend(loc='best', fontsize=7, framealpha=0.9, title=legend_title)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title(title, fontsize=title_fontsize)
    savefig(out_png, dpi=dpi)
    return {'n_points': int(len(categories)),
            'n_categories': int(pd.Series(categories).nunique()),
            'n_selected': len(sel['selected']), 'others_share': sel['others_share']}
