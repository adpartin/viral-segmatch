"""Utilities for dynamic path and directory naming across the pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_run_suffix(
    max_files: Optional[int] = None,
    seed: Optional[int] = None,
    timestamp: bool = True
    ) -> str:
    """Generate run suffix for directory naming based on sampling parameters.
    
    The run suffix encodes the sampling strategy used for data processing:
    - Full dataset: no suffix (empty string)
    - Deterministic sampling: "_seed_{seed}_GTOs_{max_files}"
    - Random sampling: "_random_{timestamp}_GTOs_{max_files}"
    
    This ensures that different sampling strategies produce unique directory names
    and that the directory name documents how the data was sampled.
    
    Args:
        max_files: Maximum number of files to process. None means full dataset.
        seed: Random seed used for sampling. None means random (non-deterministic).
        timestamp: If True, add timestamp for random runs to prevent collisions.
                  Only used when seed is None and max_files is not None.
    
    Returns:
        Run suffix string to append to data version directory name.
        Examples:
            - "" (full dataset, no sampling)
            - "_seed_42_GTOs_500" (deterministic sample)
            - "_random_20251013_143522_GTOs_100" (random sample with timestamp)
            - "_random_GTOs_100" (random sample without timestamp)
    
    Examples:
        >>> generate_run_suffix(max_files=None, seed=42)
        ""
        
        >>> generate_run_suffix(max_files=500, seed=42)
        "_seed_42_GTOs_500"
        
        >>> generate_run_suffix(max_files=100, seed=None, timestamp=False)
        "_random_GTOs_100"
    """
    if max_files is None:
        return ""  # Full dataset, no suffix
    
    if seed is not None:
        return f"_seed_{seed}_GTOs_{max_files}"
    else:
        # Random sampling - optionally add timestamp to prevent collisions
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"_random_{ts}_GTOs_{max_files}"
        else:
            return f"_random_GTOs_{max_files}"


def resolve_run_suffix(
    config,
    max_files: Optional[int] = None,
    seed: Optional[int] = None,
    auto_timestamp: bool = True
    ) -> str:
    """Resolve run suffix with precedence: manual config > auto-generated.
    
    This function implements a two-level precedence system:
    1. Manual override from config (highest priority)
    2. Auto-generation from sampling parameters (fallback)
    
    Args:
        config: Hydra config object
        max_files: Number of files to process (None = full dataset)
        seed: Random seed for sampling (None = random)
        auto_timestamp: Add timestamp for random runs when auto-generating
    
    Returns:
        Resolved run suffix string
    
    Examples:
        With manual override:
            config.run_suffix = "_seed_42_GTOs_2000_pb1_pb2"
            → Returns: "_seed_42_GTOs_2000_pb1_pb2"
        
        Auto-generated:
            max_files=500, seed=42
            → Returns: "_seed_42_GTOs_500"
        
        Full dataset:
            max_files=None, seed=42
            → Returns: ""
    """
    # Check for manual override in config
    if hasattr(config, 'run_suffix') and config.run_suffix:
        print(f"📁 Using manual run_suffix from config: {config.run_suffix}")
        return config.run_suffix
    
    # Auto-generate from sampling parameters
    suffix = generate_run_suffix(max_files, seed, timestamp=auto_timestamp)
    if suffix:
        print(f"📁 Auto-generated run_suffix: {suffix}")
    else:
        print(f"📁 Processing full dataset (no run_suffix)")
    
    return suffix


def build_preprocessing_paths(
    project_root: Path,
    virus_name: str,
    data_version: str,
    run_suffix: str = "",
    config: Optional[object] = None
    ) -> dict[str, Path]:
    """Build standard preprocessing paths for a virus.
    
    # TODO: It should be documented in a README the assumed directory structure.
    This function creates a consistent directory structure across the pipeline:
    - Raw data: data/raw/{virus_raw_name}/{data_version}/
    - Processed: data/processed/{virus_name}/{data_version}{run_suffix}/

    Args:
        project_root: Project root directory
        virus_name: Virus name (e.g., 'flu_a', 'bunya')
        data_version: Data version (e.g., 'July_2025', 'April_2025')
        run_suffix: Optional run suffix (e.g., '_seed_42_GTOs_500')

    Returns:
        Dictionary with keys: 'raw_dir', 'output_dir'

    Example:
        >>> paths = build_preprocessing_paths(
        ...     project_root=Path('/path/to/project'),
        ...     virus_name='flu_a',
        ...     data_version='July_2025',
        ...     run_suffix='_seed_42_GTOs_500'
        ... )
        >>> paths['raw_dir']
        PosixPath('/path/to/project/data/raw/Flu_A/July_2025')
        >>> paths['output_dir']
        PosixPath('/path/to/project/data/processed/flu_a/July_2025_seed_42_GTOs_500')
    """
    # Use config paths if available, otherwise fallback to hardcoded
    if config and hasattr(config, 'paths') and hasattr(config.paths, 'data_dir'):
        main_data_dir = project_root / config.paths.data_dir
    else:
        main_data_dir = project_root / 'data'

    # Map virus_name to raw data directory name
    # Raw data directories use different naming conventions (legacy)
    virus_raw_map = {
        'flu_a': 'Flu_A',
        'bunya': 'Bunyavirales'
    }
    raw_virus_name = virus_raw_map.get(virus_name, virus_name.capitalize())

    raw_dir = main_data_dir / 'raw' / raw_virus_name / data_version
    output_dir = main_data_dir / 'processed' / virus_name / f'{data_version}{run_suffix}'
    
    return {
        'raw_dir': raw_dir,
        'output_dir': output_dir
    }


def build_embeddings_paths(
    project_root: Path,
    virus_name: str,
    data_version: str,
    run_suffix: str = "",
    config: Optional[object] = None
    ) -> dict[str, Path]:
    """Build standard embeddings paths for a virus.

    This function creates a consistent directory structure for embeddings:
    - Input: data/processed/{virus_name}/{data_version}{run_suffix}/protein_final.csv
    - Output: data/embeddings/{virus_name}/{data_version}{run_suffix}/

    Args:
        project_root: Project root directory
        virus_name: Virus name (e.g., 'flu_a', 'bunya')
        data_version: Data version (e.g., 'July_2025', 'April_2025')
        run_suffix: Optional run suffix (e.g., '_seed_42_GTOs_500')

    Returns:
        Dictionary with keys: 'input_file', 'output_dir'

    Example:
        >>> paths = build_embeddings_paths(
        ...     project_root=Path('/path/to/project'),
        ...     virus_name='flu_a',
        ...     data_version='July_2025',
        ...     run_suffix='_seed_42_GTOs_500'
        ... )
        >>> paths['input_file']
        PosixPath('/path/to/project/data/processed/flu_a/July_2025_seed_42_GTOs_500/protein_final.csv')
        >>> paths['output_dir']
        PosixPath('/path/to/project/data/embeddings/flu_a/July_2025_seed_42_GTOs_500')
    """
    # Use config paths if available, otherwise fallback to hardcoded
    if config and hasattr(config, 'paths') and hasattr(config.paths, 'data_dir'):
        main_data_dir = project_root / config.paths.data_dir
    else:
        main_data_dir = project_root / 'data'
    run_dir = f'{data_version}{run_suffix}'

    # Input: read from processed data
    input_base_dir = main_data_dir / 'processed' / virus_name / run_dir
    input_file = input_base_dir / 'protein_final.csv'

    # Output: write to embeddings directory (matching the run_dir)
    output_dir = main_data_dir / 'embeddings' / virus_name / run_dir
    
    return {
        'input_file': input_file,
        'output_dir': output_dir
    }


def build_dataset_paths(
    project_root: Path,
    virus_name: str,
    data_version: str,
    task_name: str,
    run_suffix: str = "",
    config: Optional[object] = None
    ) -> dict[str, Path]:
    """Build standard dataset paths for a virus.

    This function creates a consistent directory structure for datasets:
    - Input: data/processed/{virus_name}/{data_version}{run_suffix}/protein_final.csv
    - Output: data/datasets/{virus_name}/{data_version}{run_suffix}/{task_name}/

    Args:
        project_root: Project root directory
        virus_name: Virus name (e.g., 'flu_a', 'bunya')
        data_version: Data version (e.g., 'July_2025', 'April_2025')
        task_name: Task name (e.g., 'segment_pair_classifier')
        run_suffix: Optional run suffix (e.g., '_seed_42_GTOs_500')

    Returns:
        Dictionary with keys: 'input_file', 'output_dir'
    """
    # Use config paths if available, otherwise fallback to hardcoded
    if config and hasattr(config, 'paths') and hasattr(config.paths, 'data_dir'):
        main_data_dir = project_root / config.paths.data_dir
    else:
        main_data_dir = project_root / 'data'
    run_dir = f'{data_version}{run_suffix}'
    
    # Input: processed data
    input_file = main_data_dir / 'processed' / virus_name / run_dir / 'protein_final.csv'
    
    # Output: datasets
    output_dir = main_data_dir / 'datasets' / virus_name / run_dir / task_name
    
    return {
        'input_file': input_file,
        'output_dir': output_dir
    }


def build_training_paths(
    project_root: Path,
    virus_name: str,
    data_version: str,
    task_name: str,
    run_suffix: str = "",
    config: Optional[object] = None
    ) -> dict[str, Path]:
    """Build standard training paths for a virus.

    This function creates a consistent directory structure for training:
    - Dataset input: data/datasets/{virus_name}/{data_version}{run_suffix}/{task_name}/
    - Embeddings input: data/embeddings/{virus_name}/{data_version}{run_suffix}/esm2_embeddings.h5
    - Model output: models/{virus_name}/{data_version}{run_suffix}/{task_name}/

    Args:
        project_root: Project root directory
        virus_name: Virus name (e.g., 'flu_a', 'bunya')
        data_version: Data version (e.g., 'July_2025', 'April_2025')
        task_name: Task name (e.g., 'segment_pair_classifier')
        run_suffix: Optional run suffix (e.g., '_seed_42_GTOs_500')

    Returns:
        Dictionary with keys: 'dataset_dir', 'embeddings_file', 'output_dir'
    """
    # Use config paths if available, otherwise fallback to hardcoded
    if config and hasattr(config, 'paths') and hasattr(config.paths, 'data_dir'):
        main_data_dir = project_root / config.paths.data_dir
    else:
        main_data_dir = project_root / 'data'
    run_dir = f'{data_version}{run_suffix}'
    
    # Dataset input directory
    dataset_dir = main_data_dir / 'datasets' / virus_name / run_dir / task_name
    
    # Embeddings input file
    embeddings_file = main_data_dir / 'embeddings' / virus_name / run_dir / 'esm2_embeddings.h5'
    
    # Model output directory
    output_dir = project_root / 'models' / virus_name / run_dir / task_name
    
    return {
        'dataset_dir': dataset_dir,
        'embeddings_file': embeddings_file,
        'output_dir': output_dir
    }
