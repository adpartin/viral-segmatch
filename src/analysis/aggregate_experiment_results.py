"""
Aggregate experiment results from dataset_stats.json and metrics.csv files.

This script reads:
1. dataset_stats.json files from each experiment run (for dataset statistics)
2. metrics.csv files from training results (for test set performance)

It creates a summary table showing dataset statistics (isolates, pairs, prevalence) and
test set performance metrics (F1, AUC-ROC, AUC-PR) for each experiment.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional


def find_run_directory(base_dir: Path, bundle_name: str) -> Optional[Path]:
    """
    Find the run directory matching a bundle name.
    
    Args:
        base_dir: Base directory containing runs/
        bundle_name: Bundle name (e.g., 'flu_2024', 'flu_human_h3n2_2024')
    
    Returns:
        Path to the run directory, or None if not found
    """
    runs_dir = base_dir / 'runs'
    if not runs_dir.exists():
        return None
    
    # Look for directories starting with 'dataset_{bundle_name}_'
    # Important: Must match exactly - the next part should be a timestamp (YYYYMMDD_HHMMSS)
    # This prevents 'flu_human' from matching 'flu_human_2024' or 'flu_human_h3n2_2024'
    prefix = f'dataset_{bundle_name}_'
    matching_dirs = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith(prefix):
            # Check that after the prefix, we have a timestamp pattern (starts with digits)
            # This ensures we don't match longer bundle names like 'flu_human_2024'
            remaining = d.name[len(prefix):]
            if remaining and remaining[0].isdigit():  # Timestamp starts with year (YYYY)
                matching_dirs.append(d)
    
    if not matching_dirs:
        return None
    
    # If multiple matches, use the most recent one (by name, which includes timestamp)
    return sorted(matching_dirs, key=lambda x: x.name)[-1]


def load_dataset_stats(run_dir: Path) -> Optional[dict]:
    """Load dataset_stats.json from a run directory."""
    stats_file = run_dir / 'dataset_stats.json'
    if not stats_file.exists():
        return None
    
    with open(stats_file, 'r') as f:
        return json.load(f)


def find_results_directory(results_base_dir: Path, bundle_name: str) -> Optional[Path]:
    """
    Find the results directory for a bundle.
    
    Args:
        results_base_dir: Base results directory (e.g., results/flu/July_2025)
        bundle_name: Bundle name (e.g., 'flu_2024', 'flu_human_h3n2_2024')
    
    Returns:
        Path to the results directory, or None if not found
    """
    results_dir = results_base_dir / bundle_name
    if results_dir.exists() and results_dir.is_dir():
        return results_dir
    return None


def load_test_metrics(results_dir: Path) -> Optional[dict[str, float]]:
    """
    Load test set performance metrics from results directory.
    
    Args:
        results_dir: Results directory for a bundle
    
    Returns:
        Dictionary with metrics (f1_score, accuracy, auc_roc, etc.) or None
    """
    metrics_file = results_dir / 'training_analysis' / 'metrics.csv'
    if not metrics_file.exists():
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        if len(df) == 0:
            return None
        # Return first row as dictionary
        return df.iloc[0].to_dict()
    except Exception as e:
        print(f"⚠️  Warning: Error loading metrics from {metrics_file}: {e}")
        return None


def get_bundle_filters(bundle_name: str) -> dict[str, Optional[str]]:
    """
    Get the filters applied in each bundle based on the bundle name.
    
    This is a simple mapping - in a more robust version, we could read
    the actual YAML config files.
    """
    filters = {
        'flu_2024': {'host': None, 'year': '2024', 'hn_subtype': None},
        'flu_h3n2': {'host': None, 'year': None, 'hn_subtype': 'H3N2'},
        'flu_human': {'host': 'Human', 'year': None, 'hn_subtype': None},
        'flu_human_2024': {'host': 'Human', 'year': '2024', 'hn_subtype': None},
        'flu_human_h3n2_2024': {'host': 'Human', 'year': '2024', 'hn_subtype': 'H3N2'},
        'flu_ha_na_5ks': {'host': None, 'year': None, 'hn_subtype': None},
    }
    return filters.get(bundle_name, {'host': None, 'year': None, 'hn_subtype': None})


def format_filters(filters: dict[str, Optional[str]]) -> str:
    """Format filters as a readable string."""
    parts = []
    if filters['host']:
        parts.append(f"host={filters['host']}")
    if filters['year']:
        parts.append(f"year={filters['year']}")
    if filters['hn_subtype']:
        parts.append(f"subtype={filters['hn_subtype']}")
    return ', '.join(parts) if parts else 'None'


def aggregate_experiment_results(
    base_dir: Path,
    results_base_dir: Path,
    bundle_names: list[str]
    ) -> pd.DataFrame:
    """
    Aggregate dataset statistics and test performance from all experiment bundles.
    
    Args:
        base_dir: Base directory containing runs/ (e.g., data/datasets/flu/July_2025)
        results_base_dir: Base results directory (e.g., results/flu/July_2025)
        bundle_names: List of bundle names to process
    
    Returns:
        DataFrame with columns: Bundle, Filters, Total Isolates, Total Pairs, 
        Train, Val, Test, Train Prev, Val Prev, Test Prev,
        F1, F1 Macro, AUC-ROC, AUC-PR
    """
    results = []
    
    for bundle_name in bundle_names:
        run_dir = find_run_directory(base_dir, bundle_name)
        if run_dir is None:
            print(f"⚠️  Warning: No run directory found for bundle '{bundle_name}'")
            results.append({
                'Bundle': bundle_name,
                'Filters': 'N/A',
                'Total Isolates': 'N/A',
                'Total Pairs': 'N/A',
                'Train': 'N/A',
                'Val': 'N/A',
                'Test': 'N/A',
                'Train Prev': 'N/A',
                'Val Prev': 'N/A',
                'Test Prev': 'N/A',
                'F1': 'N/A',
                'F1 Macro': 'N/A',
                'AUC-ROC': 'N/A',
                'AUC-PR': 'N/A',
            })
            continue
        
        stats = load_dataset_stats(run_dir)
        if stats is None:
            print(f"⚠️  Warning: No dataset_stats.json found in {run_dir}")
            results.append({
                'Bundle': bundle_name,
                'Filters': 'N/A',
                'Total Isolates': 'N/A',
                'Total Pairs': 'N/A',
                'Train': 'N/A',
                'Val': 'N/A',
                'Test': 'N/A',
                'Train Prev': 'N/A',
                'Val Prev': 'N/A',
                'Test Prev': 'N/A',
                'F1': 'N/A',
                'F1 Macro': 'N/A',
                'AUC-ROC': 'N/A',
                'AUC-PR': 'N/A',
            })
            continue
        
        filters = get_bundle_filters(bundle_name)
        filters_str = format_filters(filters)
        
        total_isolates = stats['total']['isolates']
        total_pairs = stats['total']['pairs']
        
        # Get pair counts (not isolate counts) for each split
        train_pairs = stats['split_sizes']['train']['pairs']
        val_pairs = stats['split_sizes']['val']['pairs']
        test_pairs = stats['split_sizes']['test']['pairs']
        
        # Compute prevalence (positive class prevalence) for each split
        # Prevalence: proportion of positive pairs in the dataset
        def format_prevalence(positive_ratio):
            """Format positive ratio as percentage string."""
            if positive_ratio is not None:
                return f"{positive_ratio:.1%}"
            return 'N/A'
        
        train_prev = format_prevalence(stats['split_sizes']['train'].get('positive_ratio', None))
        val_prev = format_prevalence(stats['split_sizes']['val'].get('positive_ratio', None))
        test_prev = format_prevalence(stats['split_sizes']['test'].get('positive_ratio', None))
        
        # Load test performance metrics
        results_dir = find_results_directory(results_base_dir, bundle_name)
        metrics = load_test_metrics(results_dir) if results_dir else None
        
        result = {
            'Bundle': bundle_name,
            'Filters': filters_str,
            'Total Isolates': total_isolates,
            'Total Pairs': total_pairs,
            'Train': train_pairs,
            'Val': val_pairs,
            'Test': test_pairs,
            'Train Prev': train_prev,
            'Val Prev': val_prev,
            'Test Prev': test_prev,
        }
        
        if metrics:
            result['F1'] = round(metrics.get('f1_score', 0), 4)
            result['F1 Macro'] = round(metrics.get('f1_macro', 0), 4)
            result['AUC-ROC'] = round(metrics.get('auc_roc', 0), 4)
            result['AUC-PR'] = round(metrics.get('avg_precision', 0), 4)
        else:
            result['F1'] = 'N/A'
            result['F1 Macro'] = 'N/A'
            result['AUC-ROC'] = 'N/A'
            result['AUC-PR'] = 'N/A'
        
        results.append(result)
    
    return pd.DataFrame(results)


def main():
    """Main function to generate the table."""
    # Project root
    project_root = Path(__file__).resolve().parents[2]
    
    # Base directory for datasets
    base_dir = project_root / 'data' / 'datasets' / 'flu' / 'July_2025'
    
    # Base directory for results
    results_base_dir = project_root / 'results' / 'flu' / 'July_2025'
    
    # Bundle names to process
    bundle_names = [
        'flu_2024',
        'flu_h3n2',
        'flu_human',
        'flu_human_2024',
        'flu_human_h3n2_2024',
        'flu_ha_na_5ks',
    ]
    
    # Aggregate results
    print(f"📊 Aggregating experiment results from: {base_dir}")
    print(f"📊 Loading test performance from: {results_base_dir}")
    print()
    
    df = aggregate_experiment_results(base_dir, results_base_dir, bundle_names)
    
    # Sort by F1 Macro in descending order
    # Convert 'F1 Macro' to numeric, treating 'N/A' as NaN for sorting
    df['F1 Macro_sort'] = pd.to_numeric(df['F1 Macro'], errors='coerce')
    df = df.sort_values('F1 Macro_sort', ascending=False, na_position='last')
    df = df.drop('F1 Macro_sort', axis=1)  # Remove temporary sorting column

    # Display table
    print("=" * 120)
    print("Experiment Results Summary")
    print("(Sorted by F1 Macro, descending)")
    print("=" * 120)
    print()
    print("Note: Train/Val/Test show pair counts (not isolate counts)")
    print("      Prev = positive class prevalence (proportion of positive pairs) in each split")
    print()
    print(df.to_string(index=False))
    print()

    # Save to CSV
    # Save in results directory since this aggregates both dataset stats and training metrics
    output_file = results_base_dir / 'experiment_results.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"💾 Saved table to: {output_file}")
    
    # Also save as markdown (if tabulate is available)
    try:
        output_md = results_base_dir / 'experiment_results.md'
        with open(output_md, 'w') as f:
            f.write("# Experiment Results Summary\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")
        print(f"💾 Saved markdown table to: {output_md}")
    except (ImportError, AttributeError) as e:
        print(f"⚠️  Skipping markdown export (tabulate not available): {e}")


if __name__ == '__main__':
    main()

