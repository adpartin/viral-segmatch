"""
Global configuration for colors and labels used in plots across the viral-segmatch project.

This module provides consistent color schemes and naming conventions for:
- Train/validation/test dataset colors
- Positive/negative sample styling
- Protein function name mappings
- Segment colors (if needed)
"""

# =============================================================================
# TRAIN/VAL/TEST (SPLIT) COLORS
# =============================================================================
# Distinctive colors that don't conflict with segment colors (blue/orange/green)
SPLIT_COLORS = {
    'train': '#17A2B8',  # Teal
    'val': '#8E44AD',    # Purple
    'test': '#E74C3C',   # Red
}

# Alternative access as list (for backwards compatibility / bar plots)
SPLIT_COLORS_LIST = [SPLIT_COLORS['train'], SPLIT_COLORS['val'], SPLIT_COLORS['test']]

# Markers for train/val/test in scatter plots (e.g., embedding overlap plots)
SPLIT_MARKERS = {
    'train': 'o',
    'val': '^',
    'test': 's',
}


# =============================================================================
# POSITIVE/NEGATIVE LABEL STYLING
# =============================================================================
# This is designed to be used together with split coloring (train/val/test):
# - Positive: filled marker (facecolor = split color), no edge
# - Negative: hollow marker (facecolor = none), colored edge = split color
LABEL_SCATTER_STYLES = {
    1: {  # positive
        'facecolors': 'auto',
        'edgecolors': 'none',
        'linewidths': 0.0,
        'alpha': 0.65,
        'label': 'Positive (+)',
    },
    0: {  # negative
        'facecolors': 'none',
        'edgecolors': 'auto',
        'linewidths': 0.9,
        'alpha': 0.75,
        'label': 'Negative (-)',
    },
}

# For bar plots and other non-scatter visualizations
SAMPLE_PATTERNS = {
    'positive': '',      # Solid fill
    'negative': '///',   # Diagonal hatching
}

# =============================================================================
# PROTEIN FUNCTION NAME MAPPINGS
# =============================================================================
# TODO. I think this PROTEIN_NAME_MAPPING is bunya specific.
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
# TODO. I think these SEGMENT_COLORS and SEGMENT_ORDER are bunya specific.
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
    return SPLIT_COLORS.get(dataset_name.lower(), '#808080')  # Gray fallback


def get_split_color(split_name: str) -> str:
    """Get color for a split (train/val/test)."""
    return SPLIT_COLORS.get(str(split_name).lower(), '#808080')

def map_protein_name(original_name):
    """Map long protein names to shorter versions."""
    return PROTEIN_NAME_MAPPING.get(original_name, original_name)

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
