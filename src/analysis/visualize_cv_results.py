"""
Visualize cross-validation results with error bars and per-fold comparisons.

Reads cv_summary.json (written by scripts/aggregate_cv_results.py) or recomputes
from per-fold test_predicted.csv files. Generates publication-ready plots:
  - cv_metrics_barplot.png: mean metrics with std error bars
  - cv_fold_comparison.png: per-fold metric dot plot
  - cv_roc_curves.png: overlaid per-fold ROC curves with mean + std band

Usage:
  # From cv_summary.json (preferred — assumes scripts/aggregate_cv_results.py already ran)
  python src/analysis/visualize_cv_results.py \\
      --cv_summary data/datasets/flu/.../cv_summary.json

  # From training dirs directly
  python src/analysis/visualize_cv_results.py \\
      --training_dirs models/flu/.../training_..._fold0_... \\
                      models/flu/.../training_..._fold1_... ...

  # From manifest
  python src/analysis/visualize_cv_results.py \\
      --manifest data/datasets/flu/.../cv_run_manifest.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot_config import apply_default_style


# ---------------------------------------------------------------------------
# Metric computation (mirrors scripts/aggregate_cv_results.py)
# ---------------------------------------------------------------------------

def compute_fold_metrics(pred_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Compute metrics from a single fold's test_predicted.csv."""
    y_true = pred_df["label"].astype(int).values
    y_probs = pred_df["pred_prob"].astype(float).values
    y_pred = (y_probs >= threshold).astype(int)

    return {
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_probs)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "brier": float(np.mean((y_probs - y_true) ** 2)),
        "n_test": int(len(y_true)),
        "threshold": float(threshold),
    }


def load_fold_predictions(training_dir: Path) -> tuple[pd.DataFrame, float]:
    """Load test_predicted.csv and optimal threshold from a training dir."""
    pred_file = training_dir / "test_predicted.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"test_predicted.csv not found in {training_dir}")

    threshold = 0.5
    th_file = training_dir / "optimal_threshold.txt"
    if th_file.exists():
        try:
            threshold = float(th_file.read_text().splitlines()[0].strip())
        except Exception:
            pass

    return pd.read_csv(pred_file), threshold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_cv_summary(cv_summary_path: Path) -> tuple[list[dict], dict, dict]:
    """Load per-fold metrics, mean, std from cv_summary.json."""
    with open(cv_summary_path) as f:
        data = json.load(f)
    return data["per_fold"], data["mean"], data["std"]


def load_from_training_dirs(training_dirs: list[Path]) -> tuple[list[dict], dict, dict]:
    """Compute per-fold metrics from training directories."""
    rows = []
    for fold_i, tdir in enumerate(training_dirs):
        pred_df, threshold = load_fold_predictions(tdir)
        metrics = compute_fold_metrics(pred_df, threshold=threshold)
        metrics["fold_id"] = fold_i
        rows.append(metrics)
        print(f"  fold {fold_i}: F1={metrics['f1_binary']:.4f}  AUC={metrics['auc_roc']:.4f}")

    metric_cols = ["f1_binary", "f1_macro", "auc_roc", "precision", "recall", "brier"]
    per_fold_df = pd.DataFrame(rows)
    means = {c: float(per_fold_df[c].mean()) for c in metric_cols}
    stds = {c: float(per_fold_df[c].std(ddof=1)) for c in metric_cols}

    return rows, means, stds


def load_from_manifest(manifest_path: Path) -> list[Path]:
    """Read manifest and return training directories."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    from src.utils.config_hydra import get_virus_config_hydra
    config_bundle = manifest["config_bundle"]
    cfg = get_virus_config_hydra(config_bundle, config_path=str(PROJECT_ROOT / "conf"))
    models_base = PROJECT_ROOT / "models" / cfg.virus.virus_name / cfg.virus.data_version / "runs"

    training_run_ids = manifest.get("training_run_ids", {})
    return [models_base / training_run_ids[str(k)] for k in sorted(training_run_ids.keys(), key=int)]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

DISPLAY_METRICS = ["f1_binary", "f1_macro", "auc_roc", "precision", "recall"]
METRIC_LABELS = {
    "f1_binary": "F1 (binary)",
    "f1_macro": "F1 (macro)",
    "auc_roc": "AUC-ROC",
    "precision": "Precision",
    "recall": "Recall",
    "brier": "Brier Score",
}


def plot_metrics_barplot(per_fold: list[dict], means: dict, stds: dict, output_path: Path):
    """Bar chart of mean metrics with std error bars."""
    apply_default_style()

    metrics = DISPLAY_METRICS
    labels = [METRIC_LABELS[m] for m in metrics]
    values = [means[m] for m in metrics]
    errors = [stds[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    bars = ax.bar(x, values, yerr=errors, capsize=5, color='#3498DB', edgecolor='white',
                  error_kw={'linewidth': 1.5, 'color': '#2C3E50'})

    # Add value labels on bars
    for bar, val, err in zip(bars, values, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    n_folds = len(per_fold)
    ax.set_title(f'Cross-Validation Metrics (mean +/- std, {n_folds} folds)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path.name}")


def plot_fold_comparison(per_fold: list[dict], output_path: Path):
    """Per-fold dot plot showing metric variation across folds."""
    apply_default_style()

    metrics = DISPLAY_METRICS
    n_folds = len(per_fold)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.7 / n_folds
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_folds, 3)))

    for fold_i, fold_data in enumerate(per_fold):
        offsets = x + (fold_i - n_folds / 2 + 0.5) * width
        values = [fold_data[m] for m in metrics]
        ax.scatter(offsets, values, color=colors[fold_i], s=50, zorder=3,
                   label=f'Fold {fold_data.get("fold_id", fold_i)}', edgecolors='white', linewidths=0.5)

    # Add mean line
    means = [np.mean([f[m] for f in per_fold]) for m in metrics]
    ax.plot(x, means, 'k_', markersize=15, markeredgewidth=2, zorder=4, label='Mean')

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Fold Metric Comparison ({n_folds} folds)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path.name}")


def plot_roc_curves(training_dirs: list[Path], output_path: Path):
    """Overlaid per-fold ROC curves with mean ROC and std band."""
    apply_default_style()

    fig, ax = plt.subplots(figsize=(7, 7))
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(training_dirs), 3)))

    for fold_i, tdir in enumerate(training_dirs):
        pred_df, _ = load_fold_predictions(tdir)
        y_true = pred_df["label"].astype(int).values
        y_probs = pred_df["pred_prob"].astype(float).values

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc_val = roc_auc_score(y_true, y_probs)
        aucs.append(auc_val)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax.plot(fpr, tpr, color=colors[fold_i], alpha=0.4, linewidth=1,
                label=f'Fold {fold_i} (AUC={auc_val:.3f})')

    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1)
    ax.plot(mean_fpr, mean_tpr, color='#2C3E50', linewidth=2.5,
            label=f'Mean (AUC={mean_auc:.3f} +/- {std_auc:.3f})')

    # Std band
    std_tpr = np.std(tprs, axis=0, ddof=1)
    ax.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    color='#2C3E50', alpha=0.15, label='+/- 1 std')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves ({len(training_dirs)} folds)')
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Visualize CV results with error bars")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cv_summary", type=str, help="Path to cv_summary.json")
    g.add_argument("--training_dirs", type=str, nargs="+",
                   help="Training run directories (one per fold, in fold order)")
    g.add_argument("--manifest", type=str, help="Path to cv_run_manifest.json")

    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to save plots (default: alongside cv_summary or first training dir)")
    args = p.parse_args()

    # Resolve training dirs and metrics
    training_dirs = None

    if args.cv_summary:
        cv_summary_path = Path(args.cv_summary)
        per_fold, means, stds = load_from_cv_summary(cv_summary_path)
        output_dir = Path(args.output_dir) if args.output_dir else cv_summary_path.parent
        # Try to find training dirs for ROC curves
        manifest_path = cv_summary_path.parent / "cv_run_manifest.json"
        if manifest_path.exists():
            training_dirs = load_from_manifest(manifest_path)

    elif args.manifest:
        manifest_path = Path(args.manifest)
        training_dirs = load_from_manifest(manifest_path)
        per_fold, means, stds = load_from_training_dirs(training_dirs)
        output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent

    else:
        training_dirs = [Path(d) for d in args.training_dirs]
        per_fold, means, stds = load_from_training_dirs(training_dirs)
        output_dir = Path(args.output_dir) if args.output_dir else training_dirs[0].parent

    output_dir.mkdir(parents=True, exist_ok=True)
    n_folds = len(per_fold)
    print(f"\nVisualizing {n_folds}-fold CV results -> {output_dir}")

    # Print summary
    metric_cols = ["f1_binary", "f1_macro", "auc_roc", "precision", "recall", "brier"]
    print(f"\n{'='*50}")
    print("CV Summary (mean +/- std)")
    print(f"{'='*50}")
    for c in metric_cols:
        print(f"  {METRIC_LABELS.get(c, c):12s}: {means[c]:.4f} +/- {stds[c]:.4f}")
    print(f"{'='*50}\n")

    # Generate plots
    plot_metrics_barplot(per_fold, means, stds, output_dir / "cv_metrics_barplot.png")
    plot_fold_comparison(per_fold, output_dir / "cv_fold_comparison.png")

    if training_dirs and len(training_dirs) == n_folds:
        plot_roc_curves(training_dirs, output_dir / "cv_roc_curves.png")
    else:
        print("WARNING: Training dirs not available, skipping ROC curves plot")

    print(f"\nDone. All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
