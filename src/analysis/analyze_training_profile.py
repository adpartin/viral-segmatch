#!/usr/bin/env python3
"""
Post-hoc analysis of Level 1 training profiling data.

Reads training_history.csv files (which include per-epoch data_time_sec,
compute_time_sec, eval_time_sec columns) and produces a profiling summary
showing where wall-clock time is spent.

Supports three modes:
  1. Single run: analyze one training_history.csv
  2. CV run: aggregate across all folds in a cv_runs directory
  3. All-pairs: aggregate across all 28 protein-pair CV experiments

Usage
-----
  # Single training run:
  python src/analysis/analyze_training_profile.py \
      models/flu/July_2025/runs/training_flu_28p_ha_na_fold0_20260401_200426/training_history.csv

  # All folds in a CV run:
  python src/analysis/analyze_training_profile.py \
      --cv_dir models/flu/July_2025/cv_runs/cv_flu_28p_ha_na_20260401_200426/

  # All 28 pairs (auto-discover latest):
  python src/analysis/analyze_training_profile.py --all_pairs

  # Save summary to CSV:
  python src/analysis/analyze_training_profile.py --all_pairs --output summary.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Timing columns added by Level 1 profiling
TIMING_COLS = ['epoch_time_sec', 'data_time_sec', 'compute_time_sec', 'eval_time_sec']


def load_history(csv_path: Path) -> pd.DataFrame:
    """Load a training_history.csv, returning None if missing or no timing data."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # Check if timing columns exist (runs before profiling was added won't have them)
    if 'data_time_sec' not in df.columns:
        return None
    return df


def summarize_single_run(df: pd.DataFrame, label: str = "") -> dict:
    """Compute profiling summary for a single training run."""
    n_epochs = len(df)
    if n_epochs == 0:
        return None

    total_time = df['epoch_time_sec'].sum()
    total_data = df['data_time_sec'].sum()
    total_compute = df['compute_time_sec'].sum()
    total_eval = df['eval_time_sec'].sum()
    total_other = total_time - total_data - total_compute - total_eval

    return {
        'label': label,
        'n_epochs': n_epochs,
        'total_time_sec': round(total_time, 1),
        'mean_epoch_sec': round(df['epoch_time_sec'].mean(), 2),
        'mean_data_sec': round(df['data_time_sec'].mean(), 2),
        'mean_compute_sec': round(df['compute_time_sec'].mean(), 2),
        'mean_eval_sec': round(df['eval_time_sec'].mean(), 2),
        'pct_data': round(100 * total_data / total_time, 1) if total_time > 0 else 0,
        'pct_compute': round(100 * total_compute / total_time, 1) if total_time > 0 else 0,
        'pct_eval': round(100 * total_eval / total_time, 1) if total_time > 0 else 0,
        'pct_other': round(100 * total_other / total_time, 1) if total_time > 0 else 0,
        # First vs last epoch (detect warmup effects)
        'epoch1_time_sec': round(df['epoch_time_sec'].iloc[0], 2),
        'epoch1_data_sec': round(df['data_time_sec'].iloc[0], 2),
        'last_epoch_time_sec': round(df['epoch_time_sec'].iloc[-1], 2),
        'last_epoch_data_sec': round(df['data_time_sec'].iloc[-1], 2),
    }


def print_single_summary(s: dict) -> None:
    """Print a formatted profiling summary for one run."""
    print(f"\n{'='*70}")
    if s['label']:
        print(f"Profiling summary: {s['label']}")
    else:
        print("Profiling summary")
    print(f"{'='*70}")
    print(f"  Epochs:          {s['n_epochs']}")
    print(f"  Total wall time: {s['total_time_sec']:.1f}s ({s['total_time_sec']/60:.1f} min)")
    print(f"  Mean epoch:      {s['mean_epoch_sec']:.2f}s")
    print(f"")
    print(f"  Time breakdown (mean per epoch):")
    print(f"    Data loading:  {s['mean_data_sec']:>7.2f}s  ({s['pct_data']:>5.1f}%)")
    print(f"    GPU compute:   {s['mean_compute_sec']:>7.2f}s  ({s['pct_compute']:>5.1f}%)")
    print(f"    Evaluation:    {s['mean_eval_sec']:>7.2f}s  ({s['pct_eval']:>5.1f}%)")
    print(f"    Other:         {'':>7}   ({s['pct_other']:>5.1f}%)")
    print(f"")
    # Bottleneck identification
    bottleneck = max(
        ('Data loading', s['pct_data']),
        ('GPU compute', s['pct_compute']),
        ('Evaluation', s['pct_eval']),
        key=lambda x: x[1]
    )
    print(f"  Bottleneck: {bottleneck[0]} ({bottleneck[1]:.1f}% of epoch time)")
    if s['epoch1_time_sec'] > 2 * s['last_epoch_time_sec']:
        print(f"  WARNING: Epoch 1 ({s['epoch1_time_sec']:.1f}s) was {s['epoch1_time_sec']/s['last_epoch_time_sec']:.1f}x "
              f"slower than last epoch ({s['last_epoch_time_sec']:.1f}s) — likely warmup or memory contention")
    print(f"{'='*70}")


def discover_fold_histories(cv_dir: Path) -> list:
    """Find training_history.csv files for all folds in a CV run."""
    # CV runs store fold results in models/.../runs/training_{bundle}_fold{k}_{ts}/
    # The cv_dir has a cv_summary.json that references the fold dirs.
    import json
    summary_file = cv_dir / 'cv_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        fold_dirs = summary.get('fold_dirs', [])
        results = []
        for fold_dir in fold_dirs:
            csv_path = Path(fold_dir) / 'training_history.csv'
            results.append(csv_path)
        return results

    # Fallback: glob for training_history.csv in sibling directories
    return []


def discover_allpairs_cv_dirs(base_dir: Path = None) -> dict:
    """Find all cv_flu_28p_* directories, returning {bundle: cv_dir}."""
    import re
    if base_dir is None:
        base_dir = PROJECT_ROOT / 'models' / 'flu' / 'July_2025' / 'cv_runs'
    if not base_dir.exists():
        return {}

    dirs = {}
    for d in sorted(base_dir.glob('cv_flu_28p_*')):
        if not d.is_dir():
            continue
        ts_match = re.search(r'_(\d{8}_\d{6})$', d.name)
        if not ts_match:
            continue
        ts = ts_match.group(1)
        bundle = d.name[3:-(len(ts) + 1)]
        dirs.setdefault(bundle, []).append(d)

    # Take latest per bundle
    return {b: sorted(paths)[-1] for b, paths in dirs.items()}


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Level 1 training profiling data (timing breakdown)')
    parser.add_argument('csv_path', nargs='?', default=None,
                        help='Path to a single training_history.csv')
    parser.add_argument('--cv_dir', type=str, default=None,
                        help='CV run directory (aggregates across folds)')
    parser.add_argument('--all_pairs', action='store_true',
                        help='Aggregate across all 28 protein-pair CV experiments')
    parser.add_argument('--cv_base_dir', type=str, default=None,
                        help='Base dir for --all_pairs (default: models/flu/July_2025/cv_runs/)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save summary table to CSV')
    args = parser.parse_args()

    summaries = []

    if args.csv_path:
        # Single run mode
        csv_path = Path(args.csv_path)
        df = load_history(csv_path)
        if df is None:
            print(f"ERROR: No profiling data in {csv_path}")
            print("Training history CSV must contain data_time_sec, compute_time_sec, eval_time_sec columns.")
            print("These columns are added by Level 1 profiling (runs before this feature won't have them).")
            sys.exit(1)
        s = summarize_single_run(df, label=str(csv_path.parent.name))
        if s:
            summaries.append(s)
            print_single_summary(s)

    elif args.cv_dir:
        # CV run mode: aggregate across folds
        cv_dir = Path(args.cv_dir)
        fold_csvs = discover_fold_histories(cv_dir)
        if not fold_csvs:
            print(f"ERROR: No fold histories found in {cv_dir}")
            sys.exit(1)

        all_dfs = []
        for csv_path in fold_csvs:
            df = load_history(csv_path)
            if df is not None:
                fold_label = csv_path.parent.name
                s = summarize_single_run(df, label=fold_label)
                if s:
                    summaries.append(s)
                    all_dfs.append(df)

        if summaries:
            # Print per-fold table
            print(f"\n{'Fold':<50} {'Epochs':>7} {'Mean(s)':>8} {'Data%':>6} {'Compute%':>9} {'Eval%':>6}")
            print('-' * 90)
            for s in summaries:
                print(f"{s['label']:<50} {s['n_epochs']:>7} {s['mean_epoch_sec']:>8.2f} "
                      f"{s['pct_data']:>6.1f} {s['pct_compute']:>9.1f} {s['pct_eval']:>6.1f}")

            # Aggregate stats
            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                agg = summarize_single_run(combined, label=f"AGGREGATE ({cv_dir.name})")
                if agg:
                    print_single_summary(agg)
                    summaries.append(agg)

    elif args.all_pairs:
        # All-pairs mode
        base_dir = Path(args.cv_base_dir) if args.cv_base_dir else None
        cv_dirs = discover_allpairs_cv_dirs(base_dir)
        if not cv_dirs:
            print("ERROR: No cv_flu_28p_* directories found")
            sys.exit(1)

        print(f"Found {len(cv_dirs)} protein-pair CV directories\n")
        pair_summaries = []

        for bundle, cv_dir in sorted(cv_dirs.items()):
            fold_csvs = discover_fold_histories(cv_dir)
            fold_dfs = []
            for csv_path in fold_csvs:
                df = load_history(csv_path)
                if df is not None:
                    fold_dfs.append(df)

            if fold_dfs:
                combined = pd.concat(fold_dfs, ignore_index=True)
                s = summarize_single_run(combined, label=bundle)
                if s:
                    s['n_folds'] = len(fold_dfs)
                    pair_summaries.append(s)
                    summaries.append(s)

        if pair_summaries:
            print(f"{'Pair':<30} {'Folds':>5} {'Epochs':>7} {'Mean(s)':>8} {'Data%':>6} {'Compute%':>9} {'Eval%':>6} {'Bottleneck':<15}")
            print('-' * 100)
            for s in pair_summaries:
                bottleneck = max(
                    ('Data', s['pct_data']),
                    ('Compute', s['pct_compute']),
                    ('Eval', s['pct_eval']),
                    key=lambda x: x[1]
                )[0]
                print(f"{s['label']:<30} {s.get('n_folds', '?'):>5} {s['n_epochs']:>7} "
                      f"{s['mean_epoch_sec']:>8.2f} {s['pct_data']:>6.1f} {s['pct_compute']:>9.1f} "
                      f"{s['pct_eval']:>6.1f} {bottleneck:<15}")

            # Grand aggregate
            print(f"\n{'='*70}")
            means = {
                'mean_epoch': np.mean([s['mean_epoch_sec'] for s in pair_summaries]),
                'mean_data_pct': np.mean([s['pct_data'] for s in pair_summaries]),
                'mean_compute_pct': np.mean([s['pct_compute'] for s in pair_summaries]),
                'mean_eval_pct': np.mean([s['pct_eval'] for s in pair_summaries]),
            }
            print(f"Grand average across {len(pair_summaries)} pairs:")
            print(f"  Mean epoch:  {means['mean_epoch']:.2f}s")
            print(f"  Data:        {means['mean_data_pct']:.1f}%")
            print(f"  Compute:     {means['mean_compute_pct']:.1f}%")
            print(f"  Eval:        {means['mean_eval_pct']:.1f}%")
            print(f"{'='*70}")
    else:
        parser.print_help()
        sys.exit(1)

    # Save summary CSV
    if args.output and summaries:
        out_df = pd.DataFrame(summaries)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved profiling summary to: {out_path}")


if __name__ == '__main__':
    main()
