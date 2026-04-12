#!/usr/bin/env python3
"""
Quick status check for all-pairs CV runs.

Reads status.json and cv_summary.json from cv_runs/ directories, prints a
one-command summary: completion status, per-pair metrics, comparison to
baseline (unfiltered), and timing.

Usage
-----
  # Check latest unfiltered run:
  python scripts/check_allpairs_status.py

  # Check H3N2-filtered run:
  python scripts/check_allpairs_status.py --tag h3n2

  # Compare two tags side by side:
  python scripts/check_allpairs_status.py --tag h3n2 --compare-to unfiltered

  # Check a specific allpairs_prod dir:
  python scripts/check_allpairs_status.py --allpairs-dir models/flu/July_2025/allpairs_prod_h3n2_20260409_200047
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CV_RUNS_BASE = PROJECT_ROOT / "models" / "flu" / "July_2025" / "cv_runs"

PROTEINS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M1", "NS1"]
PROTEIN_SHORT = {p.lower(): p for p in PROTEINS}

METRICS_DISPLAY = [
    ("auc_roc",    "AUC-ROC",   ".4f"),
    ("f1_binary",  "F1",        ".4f"),
    ("precision",  "Precision",  ".4f"),
    ("recall",     "Recall",    ".4f"),
    ("brier",      "Brier",     ".4f"),
    ("fp_fn_ratio","FP/FN",     ".2f"),
]


def discover_cv_dirs(base_dir: Path, tag: str = None):
    """Find cv_flu_28p_* dirs, return {pair_label: path}."""
    results = {}
    for d in sorted(base_dir.glob("cv_flu_28p_*")):
        if not d.is_dir():
            continue
        name = d.name
        ts_match = re.search(r"_(\d{8}_\d{6})$", name)
        if not ts_match:
            continue
        ts = ts_match.group(1)
        bundle = name[3:-(len(ts) + 1)]  # strip cv_ and _timestamp
        tokens = bundle.split("_")

        if tag:
            if not bundle.endswith(f"_{tag}"):
                continue
        else:
            if len(tokens) > 4:
                continue

        # Extract pair from bundle: flu_28p_{a}_{b}[_{tag}]
        m = re.match(r"flu_28p_([a-z0-9]+)_([a-z0-9]+)", bundle)
        if not m:
            continue
        a = PROTEIN_SHORT.get(m.group(1), m.group(1))
        b = PROTEIN_SHORT.get(m.group(2), m.group(2))
        pair_label = f"{a}/{b}"

        # Keep latest per pair
        if pair_label not in results or d.name > results[pair_label].name:
            results[pair_label] = d

    return results


def load_pair_data(cv_dir: Path):
    """Load status and summary from a cv_run dir."""
    data = {"status": "UNKNOWN", "n_done": 0, "n_failed": 0, "n_folds": 0}

    status_file = cv_dir / "status.json"
    if status_file.exists():
        s = json.load(open(status_file))
        data["status"] = s.get("status", "UNKNOWN")
        data["n_done"] = s.get("n_done", 0)
        data["n_failed"] = s.get("n_failed", 0)
        data["n_folds"] = s.get("n_folds", 0)

    summary_file = cv_dir / "cv_summary.json"
    if summary_file.exists():
        summary = json.load(open(summary_file))
        data["mean"] = summary.get("mean", {})
        data["std"] = summary.get("std", {})
        data["n_folds"] = summary.get("n_folds", data["n_folds"])
        if data["status"] == "UNKNOWN":
            data["status"] = "SUCCEEDED"

    runtime_file = cv_dir / "runtime.json"
    if runtime_file.exists():
        rt = json.load(open(runtime_file))
        data["runtime_min"] = rt.get("hours", 0) * 60 + rt.get("minutes", 0) + rt.get("seconds", 0) / 60

    return data


def print_summary(tag, cv_dirs, compare_data=None):
    """Print formatted summary."""
    tag_label = tag if tag else "unfiltered"
    print(f"\n{'=' * 90}")
    print(f"All-Pairs CV Status: {tag_label}")
    print(f"{'=' * 90}")

    # Completion
    pair_data = {}
    for pair, d in sorted(cv_dirs.items()):
        pair_data[pair] = load_pair_data(d)

    n_total = len(pair_data)
    n_succeeded = sum(1 for d in pair_data.values() if d["status"] == "SUCCEEDED")
    n_failed = sum(1 for d in pair_data.values() if d["status"] == "FAILED")
    n_running = sum(1 for d in pair_data.values() if d["status"] == "RUNNING")
    n_other = n_total - n_succeeded - n_failed - n_running

    print(f"\nCompletion: {n_succeeded}/{n_total} succeeded", end="")
    if n_failed:
        print(f", {n_failed} failed", end="")
    if n_running:
        print(f", {n_running} running", end="")
    if n_other:
        print(f", {n_other} unknown", end="")
    print()

    if n_failed > 0:
        failed_pairs = [p for p, d in pair_data.items() if d["status"] == "FAILED"]
        print(f"  Failed: {', '.join(failed_pairs)}")

    if n_running > 0:
        running_pairs = [p for p, d in pair_data.items() if d["status"] == "RUNNING"]
        for p in running_pairs:
            d = pair_data[p]
            print(f"  Running: {p} ({d['n_done']}/{d['n_folds']} folds done)")

    # Metrics table (only for succeeded pairs)
    succeeded = {p: d for p, d in pair_data.items() if "mean" in d}
    if not succeeded:
        print("\nNo completed pairs with metrics.")
        return

    # Header
    hdr = f"{'Pair':<12}"
    for key, label, _ in METRICS_DISPLAY:
        hdr += f" {label:>14}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    # Rows sorted by AUC descending
    sorted_pairs = sorted(succeeded.keys(),
                          key=lambda p: succeeded[p]["mean"].get("auc_roc", 0),
                          reverse=True)
    for pair in sorted_pairs:
        d = succeeded[pair]
        row = f"{pair:<12}"
        for key, _, fmt in METRICS_DISPLAY:
            m = d["mean"].get(key)
            s = d["std"].get(key)
            if m is not None and s is not None:
                row += f" {m:{fmt}}+/-{s:{fmt}}"
            elif m is not None:
                row += f" {m:{fmt}}{'':>9}"
            else:
                row += f" {'N/A':>14}"
        print(row)

    # Aggregate stats
    aucs = [d["mean"]["auc_roc"] for d in succeeded.values() if "auc_roc" in d["mean"]]
    f1s = [d["mean"]["f1_binary"] for d in succeeded.values() if "f1_binary" in d["mean"]]
    briers = [d["mean"]["brier"] for d in succeeded.values() if "brier" in d["mean"]]

    print(f"\n{'Across ' + str(len(succeeded)) + ' pairs':}")
    print(f"  AUC-ROC:  {min(aucs):.4f} - {max(aucs):.4f}  (median {np.median(aucs):.4f})")
    print(f"  F1:       {min(f1s):.4f} - {max(f1s):.4f}  (median {np.median(f1s):.4f})")
    print(f"  Brier:    {min(briers):.4f} - {max(briers):.4f}  (median {np.median(briers):.4f})")

    # Comparison
    if compare_data:
        c_aucs = [d["mean"]["auc_roc"] for d in compare_data.values() if "mean" in d and "auc_roc" in d["mean"]]
        if c_aucs:
            print(f"\n  vs. baseline ({len(c_aucs)} pairs):")
            print(f"    Baseline AUC median: {np.median(c_aucs):.4f}")
            print(f"    Current  AUC median: {np.median(aucs):.4f}")
            print(f"    Delta:               {np.median(aucs) - np.median(c_aucs):+.4f}")

    # Timing
    runtimes = [d["runtime_min"] for d in succeeded.values() if "runtime_min" in d]
    if runtimes:
        print(f"\nTiming (wall clock per pair, training):")
        print(f"  Min: {min(runtimes):.1f} min  Max: {max(runtimes):.1f} min  Median: {np.median(runtimes):.1f} min")

    print(f"{'=' * 90}")


def main():
    p = argparse.ArgumentParser(description="Check all-pairs CV run status and metrics")
    p.add_argument("--tag", type=str, default=None,
                   help="Filter tag (e.g., 'h3n2'). Omit for unfiltered baseline.")
    p.add_argument("--compare-to", type=str, default=None,
                   help="Tag to compare against (e.g., 'unfiltered' or omit for no comparison). "
                        "Use 'unfiltered' to compare against the untagged baseline.")
    p.add_argument("--cv-base-dir", type=str, default=None,
                   help=f"Base dir for cv_runs (default: {CV_RUNS_BASE})")
    p.add_argument("--allpairs-dir", type=str, default=None,
                   help="Specific allpairs_prod_* dir (reads manifest for pair list)")
    args = p.parse_args()

    base = Path(args.cv_base_dir) if args.cv_base_dir else CV_RUNS_BASE
    if not base.exists():
        print(f"ERROR: cv_runs base not found: {base}")
        sys.exit(1)

    cv_dirs = discover_cv_dirs(base, tag=args.tag)
    if not cv_dirs:
        tag_label = args.tag or "unfiltered"
        print(f"No cv_flu_28p_* dirs found for tag '{tag_label}' in {base}")
        sys.exit(1)

    # Load comparison data if requested
    compare_data = None
    if args.compare_to:
        compare_tag = None if args.compare_to == "unfiltered" else args.compare_to
        compare_dirs = discover_cv_dirs(base, tag=compare_tag)
        if compare_dirs:
            compare_data = {p: load_pair_data(d) for p, d in compare_dirs.items()}

    print_summary(args.tag, cv_dirs, compare_data)


if __name__ == "__main__":
    main()
