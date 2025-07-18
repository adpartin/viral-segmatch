"""
Configuration module for viral-segmatch project.

This module loads configuration from YAML and provides centralized parameters
to ensure consistency across different pipeline stages.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to conf/config.yaml relative to project root
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / 'conf' / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# Load configuration
_config = load_config()

# =============================================================================
# CONVENIENCE ACCESSORS (for backward compatibility)
# =============================================================================

# Project settings
VIRUS_NAME = _config['project']['virus_name']
DATA_VERSION = _config['project']['data_version']
TASK_NAME = _config['project']['task_name']
SEED = _config['project']['seed']

# Core protein settings
USE_CORE_PROTEINS_ONLY = _config['proteins']['use_core_proteins_only']
CORE_FUNCTIONS = _config['proteins']['core_functions']

# Dataset settings
TRAIN_RATIO = _config['dataset']['train_ratio']
VAL_RATIO = _config['dataset']['val_ratio']
NEG_TO_POS_RATIO = _config['dataset']['neg_to_pos_ratio']
ALLOW_SAME_FUNC_NEGATIVES = _config['dataset']['allow_same_func_negatives']
MAX_SAME_FUNC_RATIO = _config['dataset']['max_same_func_ratio']

# Embedding settings
ESM2_MAX_RESIDUES = _config['embeddings']['max_residues']
ESM2_PROTEIN_SEQ_COL = _config['embeddings']['protein_seq_col']

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def print_config_summary():
    """Print a summary of current configuration."""
    print("=" * 60)
    print("VIRAL-SEGMATCH CONFIGURATION")
    print("=" * 60)
    print(f"Project: {VIRUS_NAME} {DATA_VERSION}")
    print(f"Task: {TASK_NAME}")
    print(f"Use core proteins only: {USE_CORE_PROTEINS_ONLY}")
    if USE_CORE_PROTEINS_ONLY:
        print("Core functions:")
        for func in CORE_FUNCTIONS:
            print(f"  - {func}")
    print(f"Dataset split: {TRAIN_RATIO:.1%} train, {VAL_RATIO:.1%} val, {1-TRAIN_RATIO-VAL_RATIO:.1%} test")
    print(f"Negative to positive ratio: {NEG_TO_POS_RATIO}")
    print(f"Random seed: {SEED}")
    print("=" * 60)

# =============================================================================
# CONFIGURATION ACCESS
# =============================================================================

def get_config() -> Dict[str, Any]:
    """Get the full configuration dictionary."""
    return _config

def get_config_value(key_path: str, default=None):
    """Get a configuration value using dot notation (e.g., 'project.virus_name')."""
    keys = key_path.split('.')
    value = _config
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default 