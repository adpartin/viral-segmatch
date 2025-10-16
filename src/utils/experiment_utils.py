"""Utilities for experiment tracking and metadata management."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from omegaconf import DictConfig, OmegaConf


def get_git_info() -> Dict[str, str]:
    """Get current git repository information.
    
    Returns:
        Dictionary with git commit, branch, and status information
    """
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        commit_short = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        # Check if there are uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        is_dirty = len(status) > 0
        
        return {
            'commit': commit,
            'commit_short': commit_short,
            'branch': branch,
            'is_dirty': is_dirty,
            'status': 'dirty' if is_dirty else 'clean'
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            'commit': 'unknown',
            'commit_short': 'unknown',
            'branch': 'unknown',
            'is_dirty': None,
            'status': 'unknown'
        }


def save_experiment_metadata(
    output_dir: Path,
    config: DictConfig,
    stage: str,
    script_name: str,
    additional_info: Optional[Dict[str, Any]] = None
    ) -> Path:
    """Save experiment metadata to JSON file for reproducibility.
    
    This creates a comprehensive record of the experiment configuration,
    including Hydra config, git information, timestamps, and custom metadata.
    
    Args:
        output_dir: Directory where experiment outputs are saved
        config: Hydra configuration object
        stage: Pipeline stage (e.g., 'preprocessing', 'embeddings', 'training')
        script_name: Name of the script being run
        additional_info: Optional dictionary with custom metadata
    
    Returns:
        Path to the saved metadata file
    
    Example:
        >>> save_experiment_metadata(
        ...     output_dir=Path('processed/flu_a/July_2025_pb1_pb2'),
        ...     config=config,
        ...     stage='preprocessing',
        ...     script_name='preprocess_flu_protein.py',
        ...     additional_info={'proteins_processed': 1500}
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build metadata
    metadata = {
        'experiment': {
            'stage': stage,
            'script': script_name,
            'timestamp': datetime.now().isoformat(),
            'output_dir': str(output_dir)
        },
        'git': get_git_info(),
        'config': OmegaConf.to_container(config, resolve=True),
    }
    
    # Add custom metadata if provided
    if additional_info:
        metadata['additional_info'] = additional_info
    
    # Save to JSON
    metadata_file = output_dir / f'experiment_metadata_{stage}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"💾 Saved experiment metadata to: {metadata_file}")
    return metadata_file


def save_experiment_summary(
    output_dir: Path,
    stage: str,
    summary: Dict[str, Any]
    ) -> Path:
    """Save a human-readable experiment summary.
    
    This creates a simple text file with key experiment information
    that's easy to read without parsing JSON.
    
    Args:
        output_dir: Directory where experiment outputs are saved
        stage: Pipeline stage
        summary: Dictionary with key-value pairs to save
    
    Returns:
        Path to the saved summary file
    
    Example:
        >>> save_experiment_summary(
        ...     output_dir=Path('processed/flu_a/July_2025_pb1_pb2'),
        ...     stage='preprocessing',
        ...     summary={
        ...         'Virus': 'flu_a',
        ...         'Selected functions': 'PB1, PB2',
        ...         'Total samples': 1500,
        ...         'Master seed': 42
        ...     }
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_dir / f'EXPERIMENT_SUMMARY_{stage.upper()}.txt'
    
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"EXPERIMENT SUMMARY - {stage.upper()}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        for key, value in summary.items():
            # Format lists nicely
            if isinstance(value, (list, tuple)):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"📋 Saved experiment summary to: {summary_file}")
    return summary_file


def load_experiment_metadata(metadata_file: Path) -> Dict[str, Any]:
    """Load experiment metadata from JSON file.
    
    Args:
        metadata_file: Path to metadata JSON file
    
    Returns:
        Dictionary with experiment metadata
    """
    with open(metadata_file, 'r') as f:
        return json.load(f)


def compare_experiments(
    experiment1_dir: Path,
    experiment2_dir: Path,
    stage: str = 'preprocessing'
    ) -> Dict[str, Any]:
    """Compare configurations between two experiments.
    
    Args:
        experiment1_dir: Directory of first experiment
        experiment2_dir: Directory of second experiment
        stage: Pipeline stage to compare
    
    Returns:
        Dictionary highlighting differences
    
    Example:
        >>> diffs = compare_experiments(
        ...     Path('processed/flu_a/July_2025_pb1_pb2'),
        ...     Path('processed/flu_a/July_2025_all_core'),
        ...     stage='preprocessing'
        ... )
        >>> print(diffs['config_differences'])
    """
    meta1 = load_experiment_metadata(
        experiment1_dir / f'experiment_metadata_{stage}.json'
    )
    meta2 = load_experiment_metadata(
        experiment2_dir / f'experiment_metadata_{stage}.json'
    )
    
    # Find differences in configs
    config1 = meta1['config']
    config2 = meta2['config']
    
    differences = {}
    all_keys = set(str(k) for k in config1.keys()) | set(str(k) for k in config2.keys())
    
    for key in all_keys:
        val1 = config1.get(key)
        val2 = config2.get(key)
        if val1 != val2:
            differences[key] = {
                'experiment1': val1,
                'experiment2': val2
            }
    
    return {
        'experiment1': {
            'dir': str(experiment1_dir),
            'timestamp': meta1['experiment']['timestamp'],
            'git_commit': meta1['git']['commit_short']
        },
        'experiment2': {
            'dir': str(experiment2_dir),
            'timestamp': meta2['experiment']['timestamp'],
            'git_commit': meta2['git']['commit_short']
        },
        'config_differences': differences
    }
