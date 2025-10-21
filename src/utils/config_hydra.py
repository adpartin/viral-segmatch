"""
Hydra-compatible configuration module for viral-segmatch project.

This module provides a Hydra-based configuration system that allows for
hierarchical configuration management with easy switching between different
virus configurations and training parameters.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Any, Optional
from pprint import pprint

# =============================================================================
# HYDRA CONFIGURATION LOADING
# =============================================================================

def load_hydra_config(
    config_name: str,
    config_path: str = "conf",
    overrides: Optional[list] = None
    ) -> DictConfig:
    """Load configuration using Hydra.
    
    Args:
        config_name: Name of config file (e.g., "bundles/flu_a", "training/base")
        config_path: Path to config directory (default: "conf")
        overrides: List of overrides (e.g., ['bundles.training=training/gpu8'])
    
    Returns:
        DictConfig: Hydra configuration object
    """
    # breakpoint()
    # Clear any existing Hydra initialization
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # Use simple config approach that works
    abs_config_path = str(Path(config_path).resolve())
    hydra.initialize_config_dir(config_dir=abs_config_path, version_base=None)
    
    if overrides is None:
        overrides = []
    
    config = hydra.compose(config_name=config_name, overrides=overrides)
    return config


def get_virus_config_hydra(
    config_bundle: str,
    training_config: Optional[str] = None,
    embeddings_config: Optional[str] = None,
    paths_config: Optional[str] = None,
    config_path: Optional[str] = None
    ) -> DictConfig:
    """Get virus-specific configuration using Hydra bundles.
    
    This function loads a virus bundle and optionally overrides any config group.
    
    Args:
        config_bundle: Name of config bundle (e.g., 'flu_a', 'flu_a_3p_5ks', 'bunya')
        training_config: Optional training override (e.g., 'training/gpu8', 'gpu8')
                        If None, uses the training config defined in the bundle
        embeddings_config: Optional embeddings override (e.g., 'embeddings/flu_a_large', 'flu_a_large')
                          If None, uses the embeddings config defined in the bundle
        paths_config: Optional paths override (e.g., 'paths/custom', 'custom')
                     If None, uses the paths config defined in the bundle
        config_path: Path to config directory (uses project root/conf if None)
    
    Returns:
        DictConfig: Complete configuration with flattened access:
                   - config.virus.* (virus biological facts)
                   - config.training.* (training parameters)
                   - config.paths.* (file paths)
                   - config.embeddings.* (embedding settings)
    
    Examples:
        # Use bundle defaults
        config = get_virus_config_hydra('flu_a')
        
        # Override training only
        config = get_virus_config_hydra('flu_a', training_config='gpu8')
        
        # Override multiple configs
        config = get_virus_config_hydra('flu_a', 
                                       training_config='gpu8',
                                       embeddings_config='flu_a_large')
        
        # Full override
        config = get_virus_config_hydra('flu_a',
                                       training_config='training/gpu8',
                                       embeddings_config='embeddings/flu_a_large',
                                       paths_config='paths/default')
    """
    # Load the bundle for this config
    config_name = f"bundles/{config_bundle}"

    # Build overrides for any provided config groups
    overrides = []

    if paths_config is not None:
        # Normalize paths config path
        if not paths_config.startswith('paths/'):
            paths_config = f"paths/{paths_config}"
        overrides.append(f"bundles.paths={paths_config}")

    if embeddings_config is not None:
        # Normalize embeddings config path
        if not embeddings_config.startswith('embeddings/'):
            embeddings_config = f"embeddings/{embeddings_config}"
        overrides.append(f"bundles.embeddings={embeddings_config}")

    if training_config is not None:
        # Normalize training config path
        if not training_config.startswith('training/'):
            training_config = f"training/{training_config}"
        overrides.append(f"bundles.training={training_config}")

    full_config = load_hydra_config(
        config_name=config_name,
        config_path=config_path,
        overrides=overrides
    )
    # DictConfig (from OmegaConf) allows dot notation for hierarchical access.
    # E.g., config.bundles.virus.core_functions instead of config['bundles']['virus']['core_functions']
    # breakpoint()
    # print(full_config.keys()) --> ['bundles']
    # print(full_config.bundles.keys()) --> ['virus', 'training', 'paths', 'embeddings']
    # print(full_config.bundles.virus.keys())
    # print(full_config.bundles.training.keys())

    # Flatten the structure for backward compatibility
    # Instead of config.bundles.virus.*, scripts can use config.virus.*
    # Also include bundle-level settings at the top level
    config_groups = {'virus', 'paths', 'embeddings', 'training'}
    
    # Extract virus name from loaded config
    virus_name = full_config.bundles.virus.virus_name if hasattr(full_config.bundles.virus, 'virus_name') else 'unknown'
    
    flattened = DictConfig({
        'bundle_name': config_bundle,  # Store the bundle name (e.g., "flu_a_3p_5ks")
        'config_path': config_path,  # Store the config path for reference
        'virus_name': virus_name,  # Extract virus name from loaded config
        'virus': full_config.bundles.virus,
        'paths': full_config.bundles.paths if 'paths' in full_config.bundles else {},
        'embeddings': full_config.bundles.embeddings if 'embeddings' in full_config.bundles else {},
        'training': full_config.bundles.training if 'training' in full_config.bundles else {},
    })
    
    # Add all other bundle-level keys dynamically (experiment-level settings)
    # This allows adding new bundle-level params without modifying this function
    for key, value in full_config.bundles.items():
        if key not in config_groups and key not in flattened:
            flattened[key] = value

    # breakpoint()
    # print(flattened.keys()) --> ['virus', 'paths', 'embeddings', 'training', 'master_seed', 'max_files_to_process', 'process_seeds', 'run_suffix', ...]

    return flattened

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def print_config_summary(config: DictConfig):
    """Print a summary of the configuration."""
    print("=" * 60)
    print("HYDRA CONFIG SUMMARY")
    print("=" * 60)
    if hasattr(config, 'bundle_name'):
        print(f"Config bundle: {config.bundle_name}")
    if hasattr(config, 'config_path') and config.config_path:
        print(f"Config path: {config.config_path}")

    # Virus section
    if hasattr(config, 'virus') and config.virus:
        if hasattr(config.virus, 'virus_name'):
            print(f"Virus: {config.virus.virus_name}")
            print(f"Data version: {config.virus.data_version}")
            max_files = getattr(config, 'max_files_to_process', None)
            print(f"Max files: {max_files if max_files is not None else 'All (full dataset)'}")
        else:
            print(f"Virus: {config.virus.get('virus_name', 'Unknown')}")
            print(f"Data version: {config.virus.get('data_version', 'Unknown')}")
    else:
        print("Virus: Not configured")

    # Paths section
    if hasattr(config, 'paths') and config.paths:
        print("Paths:")
        pprint(config.paths)
    else:
        print("Paths: Not configured")

    # Embeddings section
    if hasattr(config, 'embeddings') and config.embeddings:
        print(f"Embeddings:")
        pprint(config.embeddings)
    else:
        print("Embeddings: Not configured")

    # Training section
    if hasattr(config, 'training') and config.training:
        print(f"Training:")
        pprint(config.training)
    else:
        print("Training: Not configured")

    print("=" * 60); print("")


def save_config(config: DictConfig, output_path: str):
    """Save configuration to file."""
    OmegaConf.save(config, output_path)


def get_core_function_segment_mapping(config: DictConfig) -> list[dict[str, str]]:
    """Get core function to segment mapping for the current virus config."""
    mappings = []
    segment_mapping = config.virus.segment_mapping
    core_functions = config.virus.core_functions

    for function, segment in segment_mapping.items():
        if function in core_functions:
            # Find the replicon type for this segment
            replicon_type = _get_replicon_type_for_segment(config.virus.virus_name, segment)
            if replicon_type:
                mappings.append({
                    'function': function,
                    'replicon_type': replicon_type,
                    'core_segment': segment
                })
    return mappings


def get_aux_function_segment_mapping(config: DictConfig) -> list[dict[str, str]]:
    """Get auxiliary function to segment mapping for the current virus config."""
    mappings = []
    segment_mapping = config.virus.segment_mapping
    aux_functions = config.virus.aux_functions

    for function, segment in segment_mapping.items():
        if function in aux_functions:
            # Find the replicon type for this segment
            replicon_type = _get_replicon_type_for_segment(config.virus.virus_name, segment)
            if replicon_type:
                mappings.append({
                    'function': function,
                    'replicon_type': replicon_type,
                    'aux_segment': segment
                })
    return mappings


def _get_replicon_type_for_segment(virus_name: str, segment: str) -> Optional[str]:
    """Get the replicon type for a given segment."""
    if virus_name == 'bunya':
        if segment == 'L':
            return 'Large RNA Segment'
        elif segment == 'M':
            return 'Medium RNA Segment'
        elif segment == 'S':
            return 'Small RNA Segment'
    elif virus_name == 'flu_a':
        return f'Segment {segment}'
    return None

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Load Flu A configuration with exploration training
    config = get_virus_config_hydra('flu_a', training_config='flu_a_exploration')
    print_config_summary(config)

    # Example: Load Bunya configuration with optimized training
    config = get_virus_config_hydra('bunya', training_config='bunya_optimized')
    print_config_summary(config)
