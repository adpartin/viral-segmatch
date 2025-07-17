"""
Global configuration for colors and labels used in plots across the viral-segmatch project.

This module provides consistent color schemes and naming conventions for:
- Train/validation/test dataset colors
- Positive/negative sample styling
- Protein function name mappings
- Segment colors (if needed)
"""

# =============================================================================
# TRAIN/VAL/TEST COLORS
# =============================================================================
# Muted, professional colors that work well for presentations
DATASET_COLORS = {
    'train': '#5B9BD5',    # Muted blue
    'val': '#ff7f0e',      # Brighter orange (matches analyze_preprocessing_results.py)
    'test': '#70AD47',     # Muted green
}

# Alternative access as list (for backwards compatibility)
DATASET_COLORS_LIST = [DATASET_COLORS['train'], DATASET_COLORS['val'], DATASET_COLORS['test']]

# =============================================================================
# POSITIVE/NEGATIVE SAMPLE STYLING
# =============================================================================
# Using shape + fill patterns to distinguish pos/neg while preserving dataset colors
SAMPLE_STYLES = {
    'positive': {
        'marker': 'o',          # Circle
        'fillstyle': 'full',    # Solid fill
        'alpha': 0.8,
        'edgecolor': 'black',
        'linewidth': 0.5,
        'label_suffix': '(+)',
    },
    'negative': {
        'marker': 's',          # Square
        'fillstyle': 'none',    # Empty/outline only
        'alpha': 0.8,
        'edgecolor': 'black',
        'linewidth': 1.0,
        'label_suffix': '(-)',
    }
}

# For bar plots and other non-scatter visualizations
SAMPLE_PATTERNS = {
    'positive': '',           # Solid fill
    'negative': '///',        # Diagonal hatching
}

# =============================================================================
# PROTEIN FUNCTION NAME MAPPINGS
# =============================================================================
PROTEIN_NAME_MAPPING = {
    'RNA-dependent RNA polymerase': 'RdRp protein',
    'Pre-glycoprotein polyprotein GP complex': 'GPC protein',
    'Nucleocapsid protein': 'N protein',
    # Also handle the longer versions with suffixes
    'RNA-dependent RNA polymerase (L protein)': 'RdRp protein',
    'Pre-glycoprotein polyprotein GP complex (GPC protein)': 'GPC protein',
    'Nucleocapsid protein (N protein)': 'N protein',
}

# =============================================================================
# SEGMENT COLORS (for consistency with existing code)
# =============================================================================
SEGMENT_COLORS = {
    'S': '#1f77b4',  # Blue
    'M': '#ff7f0e',  # Orange
    'L': '#2ca02c'   # Green
}

SEGMENT_ORDER = ['S', 'M', 'L']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_dataset_color(dataset_name):
    """Get color for a dataset (train/val/test)."""
    return DATASET_COLORS.get(dataset_name.lower(), '#808080')  # Gray fallback

def get_sample_style(sample_type, dataset_name):
    """Get complete styling for a sample type in a dataset."""
    style = SAMPLE_STYLES[sample_type].copy()
    style['color'] = get_dataset_color(dataset_name)
    return style

def map_protein_name(original_name):
    """Map long protein names to shorter versions."""
    return PROTEIN_NAME_MAPPING.get(original_name, original_name)

def get_plot_label(dataset_name, sample_type=None):
    """Generate consistent plot labels."""
    label = dataset_name.capitalize()
    if sample_type:
        suffix = SAMPLE_STYLES[sample_type]['label_suffix']
        label += f' {suffix}'
    return label

# =============================================================================
# MATPLOTLIB STYLING
# =============================================================================

def apply_default_style():
    """Apply default matplotlib styling for consistent plots."""
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
    }) 