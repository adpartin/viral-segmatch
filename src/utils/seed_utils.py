"""
Seed management utilities for reproducible experiments.

This module provides hierarchical seed management where a master seed
generates deterministic sub-seeds for different processes (preprocessing,
embeddings, dataset creation, training, etc.).

The seed system is designed to work with Hydra configuration management:
- Master seed is defined in bundle configs (e.g., conf/bundles/flu_a.yaml)
- Process-specific seeds are derived from master seed
- Command-line overrides can override any seed via Hydra's override system
- Custom override support allows process-specific seed overrides

Example usage:
    # In bundle config (conf/bundles/flu_a.yaml):
    master_seed: 42
    process_seeds:
        preprocessing: null  # Use derived seed
        embeddings: null      # Use derived seed
        dataset: null        # Use derived seed
        training: null       # Use derived seed

    # Command-line overrides:
    python script.py experiment.master_seed=123
    python script.py experiment.process_seeds.embeddings=456
    python script.py --override_seed embeddings=456

    # In code:
    seed = get_process_seed(config.experiment.master_seed, 'embeddings')
    set_deterministic_seeds(seed)
"""

import hashlib
import random
from typing import Optional

import numpy as np
import torch


def get_process_seed(master_seed: int, process_name: str) -> int:
    """
    Generate a deterministic seed for a specific process from the master seed.
    
    This function uses a hash-based approach to ensure that:
    1. Same master seed + process name always produces same result
    2. Different processes get different seeds
    3. Seeds are uniformly distributed across the 32-bit integer space
    
    Args:
        master_seed: The master seed that controls all randomness
        process_name: Name of the process (e.g., 'preprocessing', 'embeddings')
        
    Returns:
        A deterministic 32-bit integer seed for the process
        
    Example:
        >>> get_process_seed(42, 'embeddings')
        1234567890  # Always the same for master_seed=42, process='embeddings'
        >>> get_process_seed(42, 'preprocessing') 
        9876543210  # Different but deterministic for same master_seed
    """
    # Create a deterministic hash from master seed and process name
    seed_string = f"{master_seed}_{process_name}"
    seed_hash = hashlib.md5(seed_string.encode()).hexdigest()
    
    # Convert to 32-bit integer
    return int(seed_hash[:8], 16) % (2**32)


def get_all_process_seeds(master_seed: int, process_names: list[str]) -> dict[str, int]:
    """
    Generate seeds for multiple processes from the master seed.
    
    Args:
        master_seed: The master seed that controls all randomness
        process_names: List of process names
        
    Returns:
        Dictionary mapping process names to their seeds
        
    Example:
        >>> get_all_process_seeds(42, ['preprocessing', 'embeddings'])
        {'preprocessing': 9876543210, 'embeddings': 1234567890}
    """
    return {name: get_process_seed(master_seed, name) for name in process_names}


def set_deterministic_seeds(seed: int, cuda_deterministic: bool = True) -> None:
    """
    Set all random seeds for deterministic behavior.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA deterministic behavior (optional)
    
    Args:
        seed: The seed value to use
        cuda_deterministic: Whether to enable CUDA deterministic behavior
                           (slower but more reproducible)
    
    Note:
        CUDA deterministic behavior can significantly slow down training
        but ensures perfect reproducibility. Use False for faster training
        when perfect reproducibility is not required.
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set CUDA deterministic behavior
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def parse_override_seed(override_string: str) -> tuple[str, int]:
    """
    Parse a custom override seed string.
    
    This function supports the custom override format used in command-line
    arguments: --override_seed process_name=seed_value
    
    Args:
        override_string: String in format "process_name=seed_value"
        
    Returns:
        Tuple of (process_name, seed_value)
        
    Raises:
        ValueError: If the format is invalid
        
    Example:
        >>> parse_override_seed("embeddings=456")
        ('embeddings', 456)
        >>> parse_override_seed("preprocessing=123")
        ('preprocessing', 123)
    """
    try:
        process_name, seed_str = override_string.split('=')
        return process_name.strip(), int(seed_str.strip())
    except ValueError as e:
        raise ValueError(f"Invalid override seed format: {override_string}. Expected 'process_name=seed_value'") from e


def apply_seed_overrides(
    config_seeds: dict[str, Optional[int]],
    override_seeds: list[str]
    ) -> dict[str, int]:
    """
    Apply custom seed overrides to configuration seeds.
    
    This function takes the seeds from the Hydra config and applies
    any custom overrides specified via command-line arguments.
    
    Args:
        config_seeds: Dictionary of process seeds from config (may contain None values)
        override_seeds: List of override strings in format "process_name=seed_value"
        
    Returns:
        Dictionary with all seeds resolved (no None values)
        
    Example:
        >>> config_seeds = {'preprocessing': None, 'embeddings': None}
        >>> override_seeds = ['embeddings=456']
        >>> apply_seed_overrides(config_seeds, override_seeds)
        {'preprocessing': 9876543210, 'embeddings': 456}
    """
    # Parse override seeds
    overrides = {}
    for override_str in override_seeds:
        process_name, seed_value = parse_override_seed(override_str)
        overrides[process_name] = seed_value
    
    # Apply overrides
    result = {}
    for process_name, seed_value in config_seeds.items():
        if process_name in overrides:
            # Use override value
            result[process_name] = overrides[process_name]
        elif seed_value is not None:
            # Use config value
            result[process_name] = seed_value
        else:
            # This shouldn't happen if config is properly structured
            raise ValueError(f"No seed value found for process '{process_name}'")
    
    return result


# Standard process names used across the project
STANDARD_PROCESSES = [
    'preprocessing',
    'embeddings', 
    'dataset',
    'training',
    'evaluation'
]


def print_seed_summary(master_seed: int, process_seeds: dict[str, int]) -> None:
    """
    Print a summary of all seeds being used.
    
    Args:
        master_seed: The master seed
        process_seeds: Dictionary of process-specific seeds
    """
    print("=" * 60)
    print("SEED CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Master seed: {master_seed}")
    print()
    print("Process-specific seeds:")
    for process_name, seed in process_seeds.items():
        print(f"  {process_name:15}: {seed}")
    print("=" * 60)


def resolve_process_seed(config, process_name: str) -> Optional[int]:
    """
    Resolve the seed for a specific process from the config.
    
    Logic:
    1. If process seed is set, use it
    2. If process seed is not set, but master seed is set, derive from master seed
    3. If neither is set, return None (truly random)
    
    Args:
        config: Hydra config object with master_seed and process_seeds
        process_name: Name of the process (e.g., 'preprocessing', 'embeddings', 'dataset', 'training', 'evaluation')
        
    Returns:
        Resolved seed value or None for truly random behavior
    """
    process_seed = config.process_seeds.get(process_name)

    if process_seed is not None:
        # If process seed is set, use it
        print(f'Using explicit {process_name} seed from config: {process_seed}')
        return process_seed
    elif config.master_seed is not None:
        # If process seed is not set, use master seed directly
        # NOTE: All processes will use the same seed value. This is simple and intuitive,
        # but be aware that if multiple processes use the same random number generator
        # (e.g., numpy.random or torch.Generator) within the same script/session, they
        # may produce correlated random sequences. For most pipelines where processes
        # are isolated (different scripts/stages), this is not a concern.
        # Alternative approach: derive unique seeds per process using hash function
        # (see get_process_seed() function) to ensure statistical independence.
        print(f'Using master_seed for {process_name}: {config.master_seed}')
        return config.master_seed
    else:
        # If neither is set, use None (truly random)
        print(f'No seed set for {process_name} - using truly random behavior')
        return None
