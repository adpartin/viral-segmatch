#!/usr/bin/env python3
"""
Compare CV results across multiple runs in a side-by-side table.

Usage:
  # Auto-discover all cv_runs under models/:
  python src/analysis/compare_cv_runs.py

  # Specific cv_summary.json files:
  python src/analysis/compare_cv_runs.py \
      models/.../cv_runs/cv_esm2_.../cv_summary.json \
      models/.../cv_runs/cv_kmer_.../cv_summary.json

  # Save to CSV:
  python src/analysis/compare_cv_runs.py --csv results/cv_comparison.csv
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Short display labels and sort keys: map bundle substrings to readable names.
# Order matters — first match wins.
# Sort key: (data_group, feature_type) so same-data runs are adjacent.
#   data_group: 0=no filter, 1=H3N2, ...
#   feature_type: 0=ESM-2, 1=K-mer, ...
LABEL_RULES = [
    ("kmer_k6_slot_norm_concat_h3n2",    "K-mer k=6, H3N2",              (1, 1)),
    ("kmer_k6_slot_norm_concat",         "K-mer k=6, no filter",         (0, 1)),
    ("kmer_k6_slot_norm_unit_diff_h3n2", "K-mer k=6 unit_diff, H3N2",   (1, 1)),
    ("kmer_k6_slot_norm_unit_diff",      "K-mer k=6 unit_diff, no filter", (0, 1)),
    ("slot_norm_concat_h3n2",            "ESM-2 concat, H3N2",           (1, 0)),
    ("slot_norm_concat",                 "ESM-2 concat, no filter",      (0, 0)),
    ("slot_norm_unit_diff_h3n2",         "ESM-2 unit_diff, H3N2",        (1, 0)),
    ("slot_norm_unit_diff",              "ESM-2 unit_diff, no filter",   (0, 0)),
]

METRICS = ["f1_binary", "f1_macro", "auc_roc", "auc_pr", "precision", "recall", "brier"]
METRIC_LABELS = {
    "f1_binary":  "F1",
    "f1_macro":   "F1 (macro)",
    "auc_roc":    "AUC-ROC",
    "auc_pr":     "AUC-PR",
    "precision":  "Precision",
    "recall":     "Recall",
    "brier":      "Brier",
}


def label_from_bundle(bundle: str) -> tuple[str, tuple]:
    """Derive a short label and sort key from the config bundle name."""
    for substr, label, sort_key in LABEL_RULES:
        if substr in bundle:
            return label, sort_key
    return bundle, (99, 99)


def discover_cv_summaries() -> list[Path]:
    """Find all cv_summary.json files under models/*/cv_runs/."""
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("**/cv_runs/*/cv_summary.json"))


def load_run(summary_path: Path) -> dict:
    """Load a cv_summary.json and extract label, means, stds."""
    with open(summary_path) as f:
        data = json.load(f)
    # Try to get bundle name from manifest in same directory
    manifest_path = summary_path.parent / "cv_run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        bundle = manifest.get("config_bundle", summary_path.parent.name)
    else:
        bundle = summary_path.parent.name
    label, sort_key = label_from_bundle(bundle)
    return {
        "bundle": bundle,
        "label": label,
        "sort_key": sort_key,
        "mean": data["mean"],
        "std": data["std"],
        "n_folds": data["n_folds"],
        "path": str(summary_path.parent),
    }


def print_table(runs: list[dict], metrics: list[str]) -> None:
    """Print metric-per-row, run-per-column table."""
    labels = [r["label"] for r in runs]
    col_width = max(max(len(l) for l in labels), 16) + 2
    metric_label_width = max(len(v) for v in METRIC_LABELS.values()) + 2

    # Header
    header = " " * metric_label_width
    for label in labels:
        header += label.center(col_width)
    print(header)
    print("-" * len(header))

    # Rows: one per metric
    for m in metrics:
        ml = METRIC_LABELS.get(m, m)
        row = ml.ljust(metric_label_width)
        for r in runs:
            mean_val = r["mean"].get(m)
            std_val = r["std"].get(m)
            if mean_val is not None and std_val is not None:
                cell = f"{mean_val:.4f} +/- {std_val:.4f}"
            elif mean_val is not None:
                cell = f"{mean_val:.4f}"
            else:
                cell = "N/A"
            row += cell.center(col_width)
        print(row)

    # Footer: n_folds
    print("-" * len(header))
    row = "N folds".ljust(metric_label_width)
    for r in runs:
        row += str(r["n_folds"]).center(col_width)
    print(row)


def save_csv(runs: list[dict], metrics: list[str], csv_path: Path) -> None:
    """Save comparison table as CSV."""
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + [r["label"] for r in runs])
        for m in metrics:
            row = [METRIC_LABELS.get(m, m)]
            for r in runs:
                mean_val = r["mean"].get(m)
                std_val = r["std"].get(m)
                if mean_val is not None and std_val is not None:
                    row.append(f"{mean_val:.4f} +/- {std_val:.4f}")
                else:
                    row.append("")
            writer.writerow(row)
        writer.writerow(["n_folds"] + [str(r["n_folds"]) for r in runs])
    print(f"\nSaved CSV to: {csv_path}")


def main():
    p = argparse.ArgumentParser(description="Compare CV results across runs")
    p.add_argument("summaries", nargs="*",
                   help="Paths to cv_summary.json files (default: auto-discover)")
    p.add_argument("--csv", type=str, default=None,
                   help="Save comparison table to CSV file")
    p.add_argument("--metrics", nargs="+", default=["f1_binary", "auc_roc", "auc_pr"],
                   help="Metrics to compare (default: f1_binary auc_roc)")
    args = p.parse_args()

    if args.summaries:
        summary_paths = [Path(s) for s in args.summaries]
    else:
        summary_paths = discover_cv_summaries()

    if not summary_paths:
        print("No cv_summary.json files found.")
        sys.exit(1)

    runs = sorted([load_run(p) for p in summary_paths], key=lambda r: r["sort_key"])
    print(f"\nComparing {len(runs)} CV runs:\n")
    print_table(runs, args.metrics)

    if args.csv:
        save_csv(runs, args.metrics, Path(args.csv))


if __name__ == "__main__":
    main()
