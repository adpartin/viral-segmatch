#!/usr/bin/env python3
"""
Aggregate per-fold test metrics from a CV run into mean ± std summary.

Input
-----
Reads cv_run_manifest.json (written by run_cv_lambda.py) to locate training
run directories. For each fold, loads test_predicted.csv and computes:
  F1 (binary), F1 (macro), AUC-ROC, Precision, Recall, Brier score.

Output
------
Writes to the dataset run directory (next to cv_run_manifest.json):
  cv_summary.csv   — per-fold metrics + mean + std rows
  cv_summary.json  — same data in JSON (for programmatic use)

Usage
-----
  python scripts/aggregate_cv_results.py \\
      --manifest data/datasets/flu/July_2025/runs/dataset_..._cv5_.../cv_run_manifest.json

  # Pass training dirs directly (without manifest):
  python scripts/aggregate_cv_results.py \\
      --training_dirs \\
      models/flu/July_2025/runs/training_..._fold0_... \\
      models/flu/July_2025/runs/training_..._fold1_... \\
      ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def compute_metrics(pred_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Compute test metrics from a test_predicted.csv DataFrame."""
    y_true  = pred_df["label"].astype(int).values
    y_probs = pred_df["pred_prob"].astype(float).values
    y_pred  = (y_probs >= threshold).astype(int)

    return {
        "f1_binary":   float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "f1_macro":    float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
        "auc_roc":     float(roc_auc_score(y_true, y_probs)),
        "precision":   float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "brier":       float(np.mean((y_probs - y_true) ** 2)),
        "n_test":      int(len(y_true)),
        "n_positive":  int(y_true.sum()),
        "threshold":   float(threshold),
    }


def find_training_dirs_from_manifest(manifest_path: Path) -> tuple[list[Path], Path]:
    """Read manifest and return (list of training run dirs, output_dir)."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    dataset_run_dir = Path(manifest["dataset_run_dir"])
    training_run_ids = manifest.get("training_run_ids", {})

    # training_run_ids: {fold_id_str: run_id_str}
    # Run dirs are under models/<virus>/<version>/runs/<run_id>
    from src.utils.config_hydra import get_virus_config_hydra
    config_bundle = manifest["config_bundle"]
    cfg = get_virus_config_hydra(config_bundle, config_path=str(PROJECT_ROOT / "conf"))
    virus_name   = cfg.virus.virus_name
    data_version = cfg.virus.data_version
    models_base  = PROJECT_ROOT / "models" / virus_name / data_version / "runs"

    training_dirs = []
    for fold_id in sorted(training_run_ids.keys(), key=int):
        run_id = training_run_ids[str(fold_id)]
        tdir = models_base / run_id
        training_dirs.append(tdir)

    return training_dirs, dataset_run_dir


def aggregate(training_dirs: list[Path], output_dir: Path) -> None:
    """Load per-fold test_predicted.csv files, compute metrics, write summary."""
    rows = []
    for fold_i, tdir in enumerate(training_dirs):
        pred_file = tdir / "test_predicted.csv"
        if not pred_file.exists():
            print(f"⚠️  test_predicted.csv not found in fold {fold_i}: {tdir}")
            continue

        # Read optimal threshold if saved
        th_file = tdir / "optimal_threshold.txt"
        threshold = 0.5
        if th_file.exists():
            try:
                threshold = float(th_file.read_text().splitlines()[0].strip())
            except Exception:
                pass

        pred_df = pd.read_csv(pred_file)
        metrics = compute_metrics(pred_df, threshold=threshold)
        metrics["fold_id"] = fold_i
        metrics["training_dir"] = str(tdir)
        rows.append(metrics)
        print(f"  fold {fold_i}: F1={metrics['f1_binary']:.4f}  AUC={metrics['auc_roc']:.4f}  "
              f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
              f"Brier={metrics['brier']:.4f}  n={metrics['n_test']}")

    if not rows:
        print("❌ No per-fold results found. Nothing to aggregate.")
        return

    metric_cols = ["f1_binary", "f1_macro", "auc_roc", "precision", "recall", "brier"]
    per_fold_df = pd.DataFrame(rows)

    # Mean and std rows
    means = {c: per_fold_df[c].mean() for c in metric_cols}
    stds  = {c: per_fold_df[c].std(ddof=1) for c in metric_cols}
    means.update({"fold_id": "mean", "n_test": int(per_fold_df["n_test"].mean()), "threshold": None, "training_dir": ""})
    stds.update({"fold_id": "std",  "n_test": int(per_fold_df["n_test"].std()), "threshold": None, "training_dir": ""})

    summary_df = pd.concat([per_fold_df, pd.DataFrame([means, stds])], ignore_index=True)

    csv_path  = output_dir / "cv_summary.csv"
    json_path = output_dir / "cv_summary.json"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved cv_summary.csv → {csv_path}")

    # Read seeds from cv_info.json if it exists in the output dir
    master_seed = None
    fold_seeds = None
    cv_info_path = output_dir / "cv_info.json"
    if cv_info_path.exists():
        try:
            with open(cv_info_path) as f:
                cv_info = json.load(f)
                master_seed = cv_info.get("master_seed")
                fold_seeds = cv_info.get("fold_seeds")
        except Exception:
            pass

    summary_json = {
        "n_folds":     len(rows),
        "master_seed": master_seed,
        "fold_seeds":  fold_seeds,
        "per_fold":    [{k: v for k, v in r.items() if k not in ("training_dir",)} for r in rows],
        "mean":        means,
        "std":         stds,
    }
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved cv_summary.json → {json_path}")

    print(f"\n{'='*50}")
    print("CV Summary (mean ± std)")
    print(f"{'='*50}")
    for c in metric_cols:
        print(f"  {c:12s}: {means[c]:.4f} ± {stds[c]:.4f}")
    print(f"{'='*50}")


def main():
    p = argparse.ArgumentParser(description="Aggregate CV fold metrics")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest",      type=str, help="Path to cv_run_manifest.json")
    g.add_argument("--training_dirs", type=str, nargs="+",
                   help="Explicit list of training run directories (one per fold, in fold order)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write cv_summary.*  (default: parent of manifest / first training dir)")
    args = p.parse_args()

    if args.manifest:
        manifest_path   = Path(args.manifest)
        training_dirs, dataset_run_dir = find_training_dirs_from_manifest(manifest_path)
        output_dir = Path(args.output_dir) if args.output_dir else dataset_run_dir
    else:
        training_dirs = [Path(d) for d in args.training_dirs]
        output_dir    = Path(args.output_dir) if args.output_dir else training_dirs[0].parent

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nAggregating {len(training_dirs)} CV folds → {output_dir}")
    aggregate(training_dirs, output_dir)


if __name__ == "__main__":
    main()
