"""Centralized experiment registry for tracking all experiments."""

import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess


def get_registry_path(project_root: Path) -> Path:
    """Get path to experiment registry file."""
    return project_root / 'experiments' / 'registry.yaml'


def load_registry(project_root: Path) -> Dict[str, Any]:
    """Load experiment registry."""
    registry_path = get_registry_path(project_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not registry_path.exists():
        return {'experiments': []}
    
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f) or {'experiments': []}


def save_registry(project_root: Path, registry: Dict[str, Any]) -> None:
    """Save experiment registry."""
    registry_path = get_registry_path(project_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(registry_path, 'w') as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


def get_git_commit_short(project_root: Path) -> str:
    """Get short git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def register_experiment(
    project_root: Path,
    config_bundle: str,
    stage: str,
    command: str,
    exit_code: int,
    log_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    notes: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Register an experiment in the central registry.
    
    Args:
        project_root: Project root directory
        config_bundle: Config bundle name
        stage: Pipeline stage (dataset, training, embeddings, preprocessing)
        command: Exact command that was run
        exit_code: Exit code (0 = success, non-zero = failure)
        log_file: Path to log file (relative to project_root)
        output_dir: Path to output directory (relative to project_root)
        notes: Optional manual notes
        additional_metadata: Optional additional metadata dict
    
    Returns:
        Experiment ID (auto-generated)
    """
    registry = load_registry(project_root)
    
    # Generate experiment ID
    now = datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M%S')
    experiment_id = f"{stage}_{config_bundle}_{date_str}_{time_str}"
    
    # Build experiment entry
    experiment = {
        'experiment_id': experiment_id,
        'date': now.strftime('%Y-%m-%d'),
        'time': now.strftime('%H:%M:%S'),
        'config_bundle': config_bundle,
        'stage': stage,
        'command': command,
        'exit_code': exit_code,
        'status': 'success' if exit_code == 0 else 'failed',
        'git_commit': get_git_commit_short(project_root)
    }
    
    if log_file:
        experiment['log_file'] = log_file
    
    if output_dir:
        experiment['output_dir'] = output_dir
    
    if notes:
        experiment['notes'] = notes
    
    if additional_metadata:
        experiment['metadata'] = additional_metadata
    
    # Add to registry
    registry['experiments'].append(experiment)
    
    # Save registry
    save_registry(project_root, registry)
    
    return experiment_id


def query_experiments(
    project_root: Path,
    config_bundle: Optional[str] = None,
    stage: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Query experiments from registry.
    
    Args:
        project_root: Project root directory
        config_bundle: Filter by config bundle
        stage: Filter by stage
        status: Filter by status ('success' or 'failed')
        date_from: Filter from date (YYYY-MM-DD)
        date_to: Filter to date (YYYY-MM-DD)
    
    Returns:
        List of matching experiments
    """
    registry = load_registry(project_root)
    experiments = registry.get('experiments', [])
    
    # Apply filters
    filtered = []
    for exp in experiments:
        if config_bundle and exp.get('config_bundle') != config_bundle:
            continue
        if stage and exp.get('stage') != stage:
            continue
        if status and exp.get('status') != status:
            continue
        if date_from and exp.get('date', '') < date_from:
            continue
        if date_to and exp.get('date', '') > date_to:
            continue
        filtered.append(exp)
    
    return filtered


def print_experiment_summary(experiment: Dict[str, Any]) -> None:
    """Print a formatted summary of an experiment."""
    print(f"\n{'='*70}")
    print(f"Experiment: {experiment.get('experiment_id', 'unknown')}")
    print(f"{'='*70}")
    print(f"Date:       {experiment.get('date', 'unknown')} {experiment.get('time', 'unknown')}")
    print(f"Config:     {experiment.get('config_bundle', 'unknown')}")
    print(f"Stage:      {experiment.get('stage', 'unknown')}")
    print(f"Status:     {experiment.get('status', 'unknown')} (exit code: {experiment.get('exit_code', 'unknown')})")
    print(f"Git commit: {experiment.get('git_commit', 'unknown')}")
    
    if 'log_file' in experiment:
        print(f"Log file:   {experiment['log_file']}")
    
    if 'output_dir' in experiment:
        print(f"Output dir: {experiment['output_dir']}")
    
    if 'notes' in experiment:
        print(f"\nNotes:")
        print(f"  {experiment['notes']}")
    
    print(f"\nCommand:")
    print(f"  {experiment.get('command', 'unknown')}")
    print(f"{'='*70}\n")


def list_experiments(
    project_root: Path,
    config_bundle: Optional[str] = None,
    stage: Optional[str] = None,
    limit: int = 10
) -> None:
    """List recent experiments."""
    experiments = query_experiments(project_root, config_bundle=config_bundle, stage=stage)
    
    # Sort by date/time (most recent first)
    experiments.sort(key=lambda x: f"{x.get('date', '')} {x.get('time', '')}", reverse=True)
    
    # Limit results
    experiments = experiments[:limit]
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"\nFound {len(experiments)} experiment(s):\n")
    
    for exp in experiments:
        status_icon = "✅" if exp.get('status') == 'success' else "❌"
        print(f"{status_icon} {exp.get('date', 'unknown')} {exp.get('time', 'unknown')} | "
              f"{exp.get('stage', 'unknown'):12} | "
              f"{exp.get('config_bundle', 'unknown'):30} | "
              f"{exp.get('status', 'unknown')}")


if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Query or register experiments')
    parser.add_argument('--register', action='store_true', help='Register a new experiment')
    parser.add_argument('--config_bundle', type=str, help='Config bundle name (for register) or filter (for query)')
    parser.add_argument('--stage', type=str, help='Stage name (for register) or filter (for query)')
    parser.add_argument('--command', type=str, help='Command that was run (for register)')
    parser.add_argument('--exit_code', type=int, help='Exit code (for register)')
    parser.add_argument('--log_file', type=str, help='Log file path relative to project root (for register)')
    parser.add_argument('--output_dir', type=str, help='Output directory path relative to project root (for register)')
    parser.add_argument('--notes', type=str, help='Optional notes (for register)')
    parser.add_argument('--status', type=str, choices=['success', 'failed'], help='Filter by status (for query)')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of results (for query)')
    parser.add_argument('--experiment_id', type=str, help='Show details for specific experiment (for query)')
    
    args = parser.parse_args()
    
    # Get project root (assume script is run from project root)
    project_root = Path.cwd()
    
    if args.register:
        # Register new experiment
        if not all([args.config_bundle, args.stage, args.command, args.exit_code is not None]):
            print("Error: --register requires --config_bundle, --stage, --command, and --exit_code", file=sys.stderr)
            sys.exit(1)
        
        experiment_id = register_experiment(
            project_root=project_root,
            config_bundle=args.config_bundle,
            stage=args.stage,
            command=args.command,
            exit_code=args.exit_code,
            log_file=args.log_file,
            output_dir=args.output_dir,
            notes=args.notes
        )
        print(f"Registered experiment: {experiment_id}")
    elif args.experiment_id:
        # Show specific experiment
        experiments = query_experiments(project_root)
        matching = [e for e in experiments if e.get('experiment_id') == args.experiment_id]
        if matching:
            print_experiment_summary(matching[0])
        else:
            print(f"Experiment {args.experiment_id} not found.")
    else:
        # List experiments
        list_experiments(
            project_root,
            config_bundle=args.config_bundle,
            stage=args.stage,
            limit=args.limit
        )

