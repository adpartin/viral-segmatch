#!/usr/bin/env python3
"""
Aggregate results from all 28 protein-pair CV experiments into an 8x8 heatmap.

Scans cv_runs/ directories matching a given allpairs manifest (or auto-discovers
the latest cv_flu_28p_* dirs), collects cv_summary.json from each, and produces:
  1. A cross-pair summary CSV (28 rows, one per pair)
  2. 8x8 heatmaps for key metrics (AUC, F1, etc.)
  3. A live progress table (when run during an active job)

Usage
-----
  # Auto-discover latest cv_flu_28p_* dirs:
  python src/analysis/aggregate_allpairs_results.py

  # From a specific allpairs manifest:
  python src/analysis/aggregate_allpairs_results.py \
      --manifest models/flu/July_2025/allpairs_prod_20260401_184726/manifest.txt

  # Live progress check (shows status.json from each pair):
  python src/analysis/aggregate_allpairs_results.py --progress

  # Specify output directory:
  python src/analysis/aggregate_allpairs_results.py --output_dir results/allpairs/
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Canonical protein order (segment order S1-S8)
PROTEINS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M1", "NS1"]
PROTEIN_SHORT_TO_FULL = {p.lower(): p for p in PROTEINS}

# Key metrics to include in cross-pair summary and heatmaps
METRICS = ["auc_roc", "f1_binary", "precision", "recall", "brier",
           "tp", "fp", "tn", "fn", "fp_fn_ratio"]
HEATMAP_METRICS = ["auc_roc", "f1_binary"]  # Generate heatmap plots for these


def parse_pair_name(bundle_name: str) -> Optional[Tuple[str, str]]:
    """Extract (protA, protB) from a bundle name like 'flu_28p_ha_na'."""
    m = re.match(r"flu_28p_([a-z0-9]+)_([a-z0-9]+)", bundle_name)
    if m:
        return m.group(1), m.group(2)
    return None


def discover_cv_dirs(base_dir: Path, timestamp_filter: Optional[str] = None,
                     tag_filter: Optional[str] = None) -> Dict[str, Path]:
    """Find cv_flu_28p_* directories, returning {bundle_name: cv_dir}.

    If timestamp_filter is given, only match dirs with that timestamp suffix.
    If tag_filter is given, only match dirs whose bundle name ends with ``_{tag}``.
    When tag_filter is None, dirs whose bundle has more than 4 underscore-separated
    tokens are excluded — this prevents tagged runs from mixing into untagged
    aggregation (and vice versa).
    Otherwise, take the latest dir per bundle.
    """
    pattern = "cv_flu_28p_*"
    dirs = {}  # type: Dict[str, List[Path]]
    for d in sorted(base_dir.glob(pattern)):
        if not d.is_dir():
            continue
        # Extract bundle name: cv_{bundle}_{timestamp} -> bundle
        name = d.name
        # Find the timestamp suffix (last 15 chars: YYYYMMDD_HHMMSS)
        ts_match = re.search(r"_(\d{8}_\d{6})$", name)
        if not ts_match:
            continue
        ts = ts_match.group(1)
        if timestamp_filter and ts != timestamp_filter:
            continue
        bundle = name[3:-(len(ts) + 1)]  # strip "cv_" prefix and "_timestamp" suffix
        if tag_filter:
            # Require bundle to end with _{tag}; skip untagged runs
            if not bundle.endswith(f"_{tag_filter}"):
                continue
        else:
            # Exclude tagged runs so baseline aggregation is not polluted.
            # Untagged bundles have exactly 4 tokens: flu_28p_{a}_{b}.
            if len(bundle.split("_")) > 4:
                continue
        dirs.setdefault(bundle, []).append(d)

    # Take latest per bundle (sorted alphabetically = chronologically)
    result = {}
    for bundle, paths in dirs.items():
        result[bundle] = sorted(paths)[-1]
    return result


def load_cv_summary(cv_dir: Path) -> Optional[dict]:
    """Load cv_summary.json from a CV results directory."""
    summary_file = cv_dir / "cv_summary.json"
    if not summary_file.exists():
        return None
    with open(summary_file) as f:
        return json.load(f)


def load_status(cv_dir: Path) -> Optional[dict]:
    """Load status.json (live progress) from a CV results directory."""
    status_file = cv_dir / "status.json"
    if not status_file.exists():
        return None
    with open(status_file) as f:
        return json.load(f)


def parse_manifest(manifest_path: Path) -> List[Tuple[str, str]]:
    """Parse allpairs manifest.txt → [(bundle, status), ...]."""
    results = []
    with open(manifest_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                bundle, _node, status = parts[0], parts[1], parts[2]
                results.append((bundle, status))
    return results


def check_post_hoc_coverage(cv_dir: Path) -> Tuple[int, int, List[str]]:
    """Count how many training dirs for this CV run have post_hoc/ artifacts.

    Reads cv_run_manifest.json to enumerate training_run_ids, then resolves each
    to `models/{virus}/{version}/runs/{training_run_id}/` and checks whether
    `post_hoc/` exists inside. Returns (n_with, n_total, missing_fold_ids).
    """
    manifest_path = cv_dir / "cv_run_manifest.json"
    if not manifest_path.exists():
        return 0, 0, []
    with open(manifest_path) as f:
        manifest = json.load(f)
    training_run_ids = manifest.get("training_run_ids", {})
    # cv_dir layout: models/{virus}/{version}/cv_runs/cv_{bundle}_{ts}/
    # training dirs: models/{virus}/{version}/runs/{training_run_id}/
    runs_base = cv_dir.parent.parent / "runs"
    n_total = len(training_run_ids)
    n_with = 0
    missing = []
    for fold_id, run_id in sorted(training_run_ids.items(), key=lambda kv: int(kv[0])):
        post_hoc_dir = runs_base / run_id / "post_hoc"
        if post_hoc_dir.is_dir():
            n_with += 1
        else:
            missing.append(fold_id)
    return n_with, n_total, missing


def _import_data_libs():
    """Import numpy/pandas lazily (not needed for --progress mode)."""
    import numpy as np
    import pandas as pd
    return np, pd


def build_cross_pair_table(cv_dirs: Dict[str, Path]):
    """Build a summary table with mean metrics for each protein pair."""
    np, pd = _import_data_libs()
    rows = []
    for bundle, cv_dir in sorted(cv_dirs.items()):
        pair = parse_pair_name(bundle)
        if pair is None:
            continue
        prot_a, prot_b = pair

        summary = load_cv_summary(cv_dir)
        if summary is None:
            rows.append({
                "bundle": bundle,
                "prot_a": PROTEIN_SHORT_TO_FULL.get(prot_a, prot_a),
                "prot_b": PROTEIN_SHORT_TO_FULL.get(prot_b, prot_b),
                "status": "NO_SUMMARY",
            })
            continue

        row = {
            "bundle": bundle,
            "prot_a": PROTEIN_SHORT_TO_FULL.get(prot_a, prot_a),
            "prot_b": PROTEIN_SHORT_TO_FULL.get(prot_b, prot_b),
            "status": "COMPLETE",
            "n_folds": summary.get("n_folds", None),
        }
        mean = summary.get("mean", {})
        std = summary.get("std", {})
        for metric in METRICS:
            row[f"{metric}_mean"] = mean.get(metric, None)
            row[f"{metric}_std"] = std.get(metric, None)

        # Post-hoc coverage: how many folds have analyze_stage4_train.py artifacts
        n_with, n_total, missing = check_post_hoc_coverage(cv_dir)
        row["post_hoc_n_with"] = n_with
        row["post_hoc_n_total"] = n_total
        row["post_hoc_missing_folds"] = ",".join(missing) if missing else ""
        rows.append(row)

    return pd.DataFrame(rows)


def build_heatmap_matrix(table, metric: str):
    """Build an 8x8 symmetric matrix from the cross-pair table."""
    np, pd = _import_data_libs()
    col = f"{metric}_mean"
    matrix = pd.DataFrame(np.nan, index=PROTEINS, columns=PROTEINS)

    for _, row in table.iterrows():
        if row.get("status") != "COMPLETE":
            continue
        a, b = row["prot_a"], row["prot_b"]
        val = row.get(col, None)
        if val is not None and a in PROTEINS and b in PROTEINS:
            matrix.loc[a, b] = val
            matrix.loc[b, a] = val  # Symmetric

    return matrix


def plot_heatmap(matrix, metric: str, output_path: Path) -> None:
    """Generate and save a heatmap plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(f"WARNING: matplotlib/seaborn not available, skipping heatmap plot for {metric}")
        return

    metric_labels = {
        "auc_roc": "AUC-ROC",
        "f1_binary": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
        "brier": "Brier Score",
    }
    label = metric_labels.get(metric, metric)

    fig, ax = plt.subplots(figsize=(9, 7.5))

    # Use appropriate colormap (lower is better for Brier)
    if metric == "brier":
        cmap = "YlOrRd"
        vmin, vmax = 0.0, 0.2
    else:
        cmap = "YlGnBu"
        vmin, vmax = 0.7, 1.0

    mask = matrix.isna()
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap=cmap,
        vmin=vmin, vmax=vmax, mask=mask,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"label": label},
    )
    ax.set_title(f"Protein Pair {label} (12-fold CV, mean)", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Diagonal label
    for i in range(len(PROTEINS)):
        ax.text(i + 0.5, i + 0.5, "--", ha="center", va="center",
                fontsize=10, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap: {output_path}")


def plot_barplot(table, metric: str, output_path: Path) -> None:
    """Generate a horizontal bar plot with error bars for all 28 pairs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"WARNING: matplotlib not available, skipping bar plot for {metric}")
        return

    metric_labels = {
        "auc_roc": "AUC-ROC",
        "f1_binary": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
    }
    label = metric_labels.get(metric, metric)

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # Filter to complete pairs with valid metrics
    valid = table.dropna(subset=[mean_col, std_col]).copy()
    if valid.empty:
        print(f"WARNING: no valid data for bar plot ({metric})")
        return

    # Build pair label (e.g., "HA/M1") and sort alphabetically
    valid["pair_label"] = valid["prot_a"] + "/" + valid["prot_b"]
    valid = valid.sort_values("pair_label", ascending=True)

    means = valid[mean_col].values
    stds = valid[std_col].values
    labels = valid["pair_label"].values

    fig, ax = plt.subplots(figsize=(8, 10))
    y_pos = range(len(labels))

    bars = ax.barh(y_pos, means, xerr=stds, height=0.7,
                   color="#4c91c9", edgecolor="white", capsize=3)

    # Annotate each bar with the mean value (3 decimal places).
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.002, i, f"{mean:.3f}",
                va="center", ha="left", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(label)
    ax.set_title(f"Protein Pair {label} (12-fold CV, mean ± std)", fontsize=13)
    ax.set_xlim(left=0.0, right=1.05)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bar plot: {output_path}")


def show_progress(cv_dirs: Dict[str, Path]) -> None:
    """Print live progress table from status.json files."""
    print(f"\n{'Bundle':<30} {'Status':<12} {'Done':>6} {'Failed':>8} {'Running':>9} {'Updated':<20}")
    print("-" * 95)

    n_succeeded = 0
    n_failed = 0
    n_running = 0
    n_unknown = 0

    for bundle in sorted(cv_dirs.keys()):
        cv_dir = cv_dirs[bundle]
        status = load_status(cv_dir)
        if status is None:
            # Fall back to checking cv_summary.json
            summary = load_cv_summary(cv_dir)
            if summary is not None:
                print(f"{bundle:<30} {'SUCCEEDED':<12} {'?':>6} {'?':>8} {'0':>9} {'(no status.json)':<20}")
                n_succeeded += 1
            else:
                print(f"{bundle:<30} {'UNKNOWN':<12} {'?':>6} {'?':>8} {'?':>9} {'(no status.json)':<20}")
                n_unknown += 1
            continue

        s = status["status"]
        print(f"{bundle:<30} {s:<12} {status['n_done']:>6} {status['n_failed']:>8} "
              f"{status['n_running']:>9} {status.get('updated_at', '?'):<20}")

        if s == "SUCCEEDED":
            n_succeeded += 1
        elif s == "FAILED":
            n_failed += 1
        elif s == "RUNNING":
            n_running += 1
        else:
            n_unknown += 1

    print("-" * 95)
    total = len(cv_dirs)
    print(f"Total: {total} pairs  |  Succeeded: {n_succeeded}  Running: {n_running}  "
          f"Failed: {n_failed}  Unknown: {n_unknown}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all protein-pair CV results into cross-pair summary and heatmaps")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to allpairs manifest.txt (optional; auto-discovers if not given)")
    parser.add_argument("--cv_base_dir", type=str, default=None,
                        help="Base directory containing cv_flu_28p_* dirs "
                             "(default: models/flu/July_2025/cv_runs/)")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Filter cv dirs by timestamp (e.g., 20260401_184726)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for summary CSV and heatmaps "
                             "(default: same as manifest dir, or models/flu/July_2025/allpairs_summary/)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Restrict discovery to cv dirs whose bundle ends with _{tag} "
                             "(e.g., 'h3n2'). Without --tag, tagged runs are excluded.")
    parser.add_argument("--progress", action="store_true",
                        help="Show live progress table from status.json files and exit")
    args = parser.parse_args()

    # Resolve cv_base_dir
    if args.cv_base_dir:
        cv_base_dir = Path(args.cv_base_dir)
    else:
        cv_base_dir = PROJECT_ROOT / "models" / "flu" / "July_2025" / "cv_runs"

    if not cv_base_dir.exists():
        print(f"ERROR: CV base directory not found: {cv_base_dir}")
        sys.exit(1)

    # Discover CV dirs
    cv_dirs = discover_cv_dirs(cv_base_dir, timestamp_filter=args.timestamp,
                                tag_filter=args.tag)
    if not cv_dirs:
        print(f"No cv_flu_28p_* directories found in {cv_base_dir}")
        sys.exit(1)

    print(f"Found {len(cv_dirs)} protein-pair CV directories in {cv_base_dir}")

    # Progress mode: show live status and exit
    if args.progress:
        show_progress(cv_dirs)
        return

    # Build cross-pair table
    table = build_cross_pair_table(cv_dirs)
    complete = table[table["status"] == "COMPLETE"]
    print(f"Complete: {len(complete)}/{len(table)} pairs have cv_summary.json")

    if len(complete) == 0:
        print("No completed pairs found. Use --progress to check live status.")
        sys.exit(1)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.manifest:
        output_dir = Path(args.manifest).parent
    else:
        output_dir = PROJECT_ROOT / "models" / "flu" / "July_2025" / "allpairs_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cross-pair summary CSV
    csv_path = output_dir / "allpairs_summary.csv"
    table.to_csv(csv_path, index=False)
    print(f"Saved cross-pair summary: {csv_path}")

    # Post-hoc coverage report: flag pairs with missing post_hoc/ dirs so
    # silent analyze_stage4_train.py failures (in-train call) are visible.
    # To backfill missing dirs: bash scripts/run_allpairs_post_hoc.sh <TAG>
    if "post_hoc_n_total" in table.columns and table["post_hoc_n_total"].sum() > 0:
        total_with = int(table["post_hoc_n_with"].sum())
        total_all = int(table["post_hoc_n_total"].sum())
        incomplete = table[
            (table["post_hoc_n_total"] > 0)
            & (table["post_hoc_n_with"] < table["post_hoc_n_total"])
        ]
        print(f"\nPost-hoc coverage: {total_with}/{total_all} training dirs have post_hoc/ artifacts")
        if len(incomplete) > 0:
            print(f"WARNING: {len(incomplete)} pair(s) have incomplete post-hoc coverage:")
            for _, r in incomplete.iterrows():
                print(f"    {r['bundle']:40s}  {r['post_hoc_n_with']}/{r['post_hoc_n_total']}  "
                      f"missing folds: [{r['post_hoc_missing_folds']}]")
            print(f"  Backfill with: bash scripts/run_allpairs_post_hoc.sh <TAG>")

    # Save fp_fn sub-table (pairs ranked by FP/FN ratio — quick health check)
    if "fp_fn_ratio_mean" in table.columns:
        fp_fn_cols = [c for c in ["bundle", "prot_a", "prot_b", "status",
                                  "fp_fn_ratio_mean", "fp_fn_ratio_std"]
                      if c in table.columns]
        fp_fn_path = output_dir / "allpairs_summary_fp_fn.csv"
        table[fp_fn_cols].sort_values("fp_fn_ratio_mean", ascending=False,
                                      na_position="last").to_csv(fp_fn_path, index=False)
        print(f"Saved FP/FN ratio sub-table: {fp_fn_path}")

    # Print summary table
    print(f"\n{'Pair':<20} {'AUC':>12} {'F1':>12} {'Prec':>12} {'Recall':>12} {'Brier':>12}")
    print("-" * 80)
    for _, row in complete.sort_values("auc_roc_mean", ascending=False).iterrows():
        pair_label = f"{row['prot_a']}/{row['prot_b']}"
        auc = f"{row['auc_roc_mean']:.3f}±{row['auc_roc_std']:.3f}" if row['auc_roc_mean'] else "N/A"
        f1 = f"{row['f1_binary_mean']:.3f}±{row['f1_binary_std']:.3f}" if row['f1_binary_mean'] else "N/A"
        prec = f"{row['precision_mean']:.3f}±{row['precision_std']:.3f}" if row['precision_mean'] else "N/A"
        rec = f"{row['recall_mean']:.3f}±{row['recall_std']:.3f}" if row['recall_mean'] else "N/A"
        brier = f"{row['brier_mean']:.3f}±{row['brier_std']:.3f}" if row['brier_mean'] else "N/A"
        print(f"{pair_label:<20} {auc:>12} {f1:>12} {prec:>12} {rec:>12} {brier:>12}")

    # Generate heatmaps
    for metric in HEATMAP_METRICS:
        matrix = build_heatmap_matrix(table, metric)
        # Save matrix CSV
        matrix_csv = output_dir / f"heatmap_{metric}.csv"
        matrix.to_csv(matrix_csv)
        print(f"Saved heatmap matrix: {matrix_csv}")

        # Plot heatmap
        plot_path = output_dir / f"heatmap_{metric}.png"
        plot_heatmap(matrix, metric, plot_path)

        # Plot bar chart
        bar_path = output_dir / f"barplot_{metric}.png"
        plot_barplot(complete, metric, bar_path)

    # Save full results as JSON (for programmatic access)
    json_path = output_dir / "allpairs_summary.json"
    summary_json = {
        "n_pairs_found": len(cv_dirs),
        "n_pairs_complete": len(complete),
        "pairs": table.to_dict(orient="records"),
    }
    # Add heatmap matrices
    for metric in HEATMAP_METRICS:
        matrix = build_heatmap_matrix(table, metric)
        summary_json[f"heatmap_{metric}"] = matrix.where(matrix.notna(), None).to_dict()

    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    print(f"Saved full results: {json_path}")

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
