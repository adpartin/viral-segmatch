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

Run-manifest mode
-----------------
With `--training_run_manifest` and/or `--redraw_run_manifest`, the report covers exactly the
directories those manifests record instead of the latest CV directory per bundle, and any failed
check stops the report before anything is written:

- From a training run: every fold of every pair must hold all required fit outputs, and its test
  metrics, recomputed from `test_predicted.csv`, must match `metrics_summary.json`. The folds of
  a pair must agree on `n_features`, which becomes the `Features` column. Eight columns of the
  pair-capacity table (`src/analysis/summarize_pair_capacity.py`) are joined from the capacity
  CSV the dataset manifest records, by population and pair. Outputs: `allpairs_summary.csv` (numeric), `allpairs_display.csv`
  (`mean ± SD` text), and the lower-triangle F1 macro matrix colored by the mean over folds with
  the SD over folds on a second line.
- From a redraw run: one row per pair and fold, with the mean and SD of test F1 macro across
  draws, the median per-pair probability and raw-score SD across draws for each test class, and
  the two negative-pair overlap counts. The lower-triangle matrix shows, per pair, the mean over
  folds of the within-fold SD of F1 macro across draws.
- `--final` also requires all 28 pairs, and every fit at the datasets' commit with a clean
  worktree.

  python src/analysis/aggregate_allpairs_results.py \
      --training_run_manifest models/flu/July_2025/training_runs/<name>/training_run_manifest.json \
      --redraw_run_manifest models/flu/July_2025/training_runs/<name>/redraw_run_manifest.json \
      --output_dir results/flu/July_2025/<report> --final
"""

import argparse
import json
import math
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
METRICS = ["auc_roc", "auc_pr", "f1_binary", "f1_macro", "precision", "recall", "brier",
           "tp", "fp", "tn", "fn", "fp_fn_ratio"]
HEATMAP_METRICS = ["auc_roc", "f1_binary", "f1_macro"]  # Generate heatmap plots for these


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


def plot_heatmap(matrix, metric: str, output_path: Path, n_folds: Optional[int] = None) -> None:
    """Generate and save a heatmap plot.

    Args:
      matrix: the symmetric protein-by-protein matrix of mean values.
      metric: metric name, e.g. `f1_macro`.
      output_path: where to write the PNG.
      n_folds: folds the runs used, named in the title. Omitted when the runs disagree.

    Returns:
      None. Writes `output_path`.
    """
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
        "auc_pr": "AUC-PR",
        "f1_binary": "F1 binary",
        "f1_macro": "F1 macro",
        "precision": "Precision",
        "recall": "Recall",
        "brier": "Brier Score",
    }
    label = metric_labels.get(metric, metric)

    # Same figure size as the capacity matrix in summarize_pair_capacity.py, so the two
    # 8x8 matrices sit side by side at one scale.
    fig, ax = plt.subplots(figsize=(6.5, 5.3))

    # Use appropriate colormap (lower is better for Brier)
    if metric == "brier":
        cmap = "YlOrRd"
        vmin, vmax = 0.0, 0.2
    else:
        cmap = "YlGnBu"
        # A fixed floor of 0.7 renders every pair below it in one colour. Start at the lowest
        # cell instead, rounded down to the next 0.05, so the weakest pairs stay distinguishable.
        lowest = float(matrix.min().min())
        vmin, vmax = min(0.7, math.floor(lowest * 20) / 20), 1.0

    mask = matrix.isna()
    sns.heatmap(
        matrix, annot=True, fmt=".3f", annot_kws={"size": 8}, cmap=cmap,
        vmin=vmin, vmax=vmax, mask=mask,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"label": label},
    )
    fold_text = f"{n_folds}-fold CV" if n_folds else "CV"
    ax.set_title(f"Protein Pair {label} ({fold_text}, mean)", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Diagonal label
    for i in range(len(PROTEINS)):
        ax.text(i + 0.5, i + 0.5, "--", ha="center", va="center",
                fontsize=9, color="gray")

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
    folds = valid['n_folds'].dropna().unique() if 'n_folds' in valid else []
    fold_text = f'{int(folds[0])}-fold CV' if len(folds) == 1 else 'CV'
    ax.set_title(f"Protein Pair {label} ({fold_text}, mean ± std)", fontsize=13)
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


# ── Manifest mode ──────────────────────────────────────────────────────────────────────────────
# Reports built from the directories a training run manifest or a redraw run manifest records,
# rather than from the latest CV directory per bundle. Every check below fails the report.

# The pair-capacity table columns (src/analysis/summarize_pair_capacity.py) that the training-run
# table carries.
CAPACITY_TABLE_COLUMNS = ['Pair ID', 'Schema pair', 'Eligible isolates', 'Unique positives',
                          'Unique slot-A', 'Unique slot-B', 'HK selected', 'HK share']
# Test metrics the trainer writes to metrics_summary.json, all recomputed from test_predicted.csv.
FIT_TEST_METRICS = ['f1', 'f1_macro', 'precision', 'recall', 'auc_roc']
DISPLAY_METRICS = {'f1_macro': 'F1 macro', 'auc_roc': 'AUC-ROC', 'precision': 'Precision',
                   'recall': 'Recall'}
N_PAIRS = 28


def recomputed_test_metrics(model_dir: Path) -> Tuple[dict, List[str]]:
    """A fit's test metrics recomputed from its predictions, and any disagreement with the saved ones.

    Args:
      model_dir: a finished fit's run directory.

    Returns:
      The recomputed metrics (`FIT_TEST_METRICS`) and one message per metric that differs from
      `metrics_summary.json` by more than 1e-9.
    """
    np, pd = _import_data_libs()
    from src.models._pair_metrics import compute_pair_metrics
    info = json.loads((model_dir / "training_info.json").read_text())
    pred = pd.read_csv(model_dir / "test_predicted.csv", usecols=["label", "pred_prob"],
                       keep_default_na=False, na_values=[""])
    metrics, _ = compute_pair_metrics(pred["label"].to_numpy(), pred["pred_prob"].to_numpy(),
                                      float(info["optimal_threshold"]), pred,
                                      print_classification_report=False)
    saved = json.loads((model_dir / "metrics_summary.json").read_text())["test"]
    failures = [f"{model_dir.name}: {key} recomputed {metrics[key]:.12f}, saved {saved[key]:.12f}"
                for key in FIT_TEST_METRICS if not np.isclose(metrics[key], saved[key], rtol=0,
                                                               atol=1e-9)]
    recomputed = {key: float(metrics[key]) for key in FIT_TEST_METRICS}
    return recomputed, failures


def final_fit_failures(model_dir: Path, commit: str) -> List[str]:
    """Reasons a fit may not enter a final report: another commit, or a dirty worktree.

    Args:
      model_dir: a finished fit's run directory.
      commit: the commit the datasets were built at.

    Returns:
      One message per problem; empty when the fit recorded `commit` and a clean worktree.
    """
    code = json.loads((model_dir / "training_info.json").read_text()).get("code", {})
    failures = []
    if code.get("commit") != commit:
        failures.append(f"{model_dir.name}: fitted at commit {code.get('commit')}, datasets at {commit}")
    if code.get("is_dirty") is not False:
        failures.append(f"{model_dir.name}: code.is_dirty={code.get('is_dirty')}")
    return failures


def lower_triangle_matrix(values: Dict[str, float]):
    """Place one value per schema pair in the strictly lower triangle of the protein grid.

    Args:
      values: pair label (`{slot A}-{slot B}`, e.g. `PB2-HA`) -> value.

    Returns:
      A `PROTEINS` x `PROTEINS` frame: each pair's value sits in the row of whichever of its two
      proteins comes later in segment order and the column of the earlier one. Every other cell
      is NaN.

    Raises:
      ValueError: a label names a protein outside `PROTEINS`, or two labels name the same
          unordered pair.
    """
    np, pd = _import_data_libs()
    matrix = pd.DataFrame(np.nan, index=PROTEINS, columns=PROTEINS)
    for label, value in values.items():
        a, b = label.split("-")
        if a not in PROTEINS or b not in PROTEINS or a == b:
            raise ValueError(f"lower_triangle_matrix: {label} is not a pair of two of {PROTEINS}")
        row, col = sorted((a, b), key=PROTEINS.index, reverse=True)
        if not np.isnan(matrix.loc[row, col]):
            raise ValueError(f"lower_triangle_matrix: two labels name the pair {row}-{col}")
        matrix.loc[row, col] = value
    return matrix


def plot_lower_triangle(mean, sd, title: str, cbar_label: str, output_path: Path) -> None:
    """Draw a lower-triangle pair matrix colored by `mean`, with the SD on a second line.

    Args:
      mean: from `lower_triangle_matrix`; colors the cells.
      sd: a matrix of the same layout written under each mean as `± SD`, or None for one line.
      title: figure title.
      cbar_label: color bar label.
      output_path: where to write the PNG.

    Returns:
      None. Writes `output_path`.
    """
    np, _ = _import_data_libs()
    from src.utils.plot_utils import annotated_heatmap
    n = len(PROTEINS)
    upper = ~np.tri(n, n, k=-1, dtype=bool)  # the diagonal and everything above it
    text = [["" for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            value = mean.iat[i, j]
            if np.isnan(value):
                continue
            text[i][j] = f"{value:.3f}" if sd is None else f"{value:.3f}\n± {sd.iat[i, j]:.3f}"
    finite = mean.to_numpy()[~np.isnan(mean.to_numpy())]
    annotated_heatmap(mean.to_numpy(), row_labels=PROTEINS, col_labels=PROTEINS,
                      out_png=output_path, title=title, cbar_label=cbar_label, cmap="YlGnBu",
                      vmin=float(finite.min()) if finite.size else None,
                      vmax=float(finite.max()) if finite.size else None,
                      cell_text=text, blank_mask=upper, figsize=(7.5, 6.2), cell_fontsize=7,
                      row_fontsize=10)
    print(f"Saved matrix figure: {output_path}")


def _load_run_manifests(run_manifest_path: Path) -> Tuple[dict, dict]:
    """A run manifest and the dataset manifest it names, checked to be the one it recorded.

    Args:
      run_manifest_path: a `training_run_manifest.json` or `redraw_run_manifest.json`.

    Returns:
      The run manifest and the dataset manifest.

    Raises:
      ValueError: the dataset manifest changed since the run started.
    """
    from src.utils.dataset_manifest import load_dataset_manifest
    run_manifest = json.loads(Path(run_manifest_path).read_text())
    dataset_manifest = load_dataset_manifest(Path(run_manifest["dataset_manifest"]["path"]))
    if dataset_manifest["manifest_sha256"] != run_manifest["dataset_manifest"]["sha256"]:
        raise ValueError(f"{dataset_manifest['manifest_path']} changed since "
                         f"{Path(run_manifest_path).name} was written")
    return run_manifest, dataset_manifest


def _pair_labels(dataset_manifest: dict) -> Dict[str, Tuple[str, str]]:
    """Each manifest pair's label and population, read from its dataset's resolved config.

    Args:
      dataset_manifest: from `load_dataset_manifest`.

    Returns:
      Pair token -> (pair label, population label).
    """
    from omegaconf import OmegaConf

    from src.analysis.audit_pair_datasets import run_pair_and_population
    labels = {}
    for entry in dataset_manifest["datasets"].values():
        config = OmegaConf.load(Path(entry["dataset_dir"]) / "resolved_config.yaml")
        labels[entry["pair"]] = run_pair_and_population(config)
    return labels


def training_run_table(run_manifest_path: Path, final: bool):
    """The training-run table: one row per schema pair, with its capacity columns.

    Args:
      run_manifest_path: the run's `training_run_manifest.json`.
      final: require all 28 pairs and, for every fit, the datasets' commit and a clean worktree.

    Returns:
      The table (numeric; one `{metric}_mean` and `{metric}_sd_folds` per test metric), the
      failure messages, and the fold count.
    """
    np, pd = _import_data_libs()
    from src.models.baseline_fits import REQUIRED_FIT_OUTPUTS
    run_manifest, dataset_manifest = _load_run_manifests(run_manifest_path)
    commit = dataset_manifest["code"]["commit"]
    n_folds = dataset_manifest["n_folds"]
    labels = _pair_labels(dataset_manifest)
    capacity_path = Path(dataset_manifest["capacity_csv"]["path"])
    capacity = pd.read_csv(capacity_path, keep_default_na=False, na_values=[""])

    failures, rows = [], []
    if final and len(run_manifest["pairs"]) != N_PAIRS:
        failures.append(f"a final report needs {N_PAIRS} pairs; the run has "
                        f"{len(run_manifest['pairs'])}")
    for pair, plan in sorted(run_manifest["pairs"].items()):
        label, population = labels[pair]
        fold_metrics, widths = [], set()
        if len(plan["folds"]) != n_folds:
            failures.append(f"{label}: {len(plan['folds'])} folds, the datasets have {n_folds}")
        for fold in plan["folds"].values():
            model_dir = Path(fold["model_dir"])
            missing = [name for name in REQUIRED_FIT_OUTPUTS if not (model_dir / name).exists()]
            if missing:
                failures.append(f"{model_dir.name}: missing {missing}")
                continue
            metrics, metric_failures = recomputed_test_metrics(model_dir)
            failures.extend(metric_failures)
            fold_metrics.append(metrics)
            widths.add(json.loads((model_dir / "training_info.json").read_text()).get("n_features"))
            if final:
                failures.extend(final_fit_failures(model_dir, commit))
        if len(widths) != 1 or None in widths:
            failures.append(f"{label}: folds record n_features {sorted(widths, key=str)}")

        matches = capacity[(capacity["population"] == population)
                           & (capacity["Schema pair"] == label)]
        if len(matches) != 1:
            failures.append(f"{label}: the capacity table has {len(matches)} rows for "
                            f"{population}, expected exactly 1")
            continue
        row = {column: matches.iloc[0][column] for column in CAPACITY_TABLE_COLUMNS}
        row.update({"population": population, "Features": widths.pop() if len(widths) == 1 else None,
                    "n_folds": len(fold_metrics), "bundle": plan["bundle"], "cv_dir": plan["cv_dir"]})
        per_fold = pd.DataFrame(fold_metrics)
        for metric in FIT_TEST_METRICS:
            row[f"{metric}_mean"] = per_fold[metric].mean() if len(per_fold) else np.nan
            row[f"{metric}_sd_folds"] = per_fold[metric].std(ddof=1) if len(per_fold) > 1 else np.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
        table.insert(0, "Rank", range(1, len(table) + 1))
    return table, failures, n_folds


def redraw_run_table(run_manifest_path: Path, final: bool):
    """The redraw-run table: one row per schema pair and fold.

    Args:
      run_manifest_path: the run's `redraw_run_manifest.json`.
      final: require all 28 pairs and, for every fit, the datasets' commit and a clean worktree.

    Returns:
      The table (numeric; SD columns named `_sd_draws`, since only the training negatives vary
      between the fits of one row) and the failure messages.
    """
    _, pd = _import_data_libs()
    from src.models.baseline_fits import REQUIRED_FIT_OUTPUTS
    run_manifest, dataset_manifest = _load_run_manifests(run_manifest_path)
    commit = dataset_manifest["code"]["commit"]
    labels = _pair_labels(dataset_manifest)
    seeds = run_manifest["seeds"]

    failures, rows = [], []
    if final and len(run_manifest["pairs"]) != N_PAIRS:
        failures.append(f"a final report needs {N_PAIRS} pairs; the run has "
                        f"{len(run_manifest['pairs'])}")
    for pair, plan in sorted(run_manifest["pairs"].items()):
        label, _ = labels[pair]
        analysis_dir = Path(plan["analysis_dir"])
        metrics_path = analysis_dir / "test_metrics_by_seed.csv"
        sd_path = analysis_dir / "per_pair_sd_by_class.csv"
        if not (metrics_path.exists() and sd_path.exists()):
            failures.append(f"{label}: no analysis outputs in {analysis_dir}")
            continue
        by_seed = pd.read_csv(metrics_path)
        sd_table = pd.read_csv(sd_path, keep_default_na=False, na_values=[""])
        for fold_id, fold in sorted(plan["folds"].items(), key=lambda kv: int(kv[0])):
            for paths in fold["seeds"].values():
                model_dir = Path(paths["model_dir"])
                missing = [name for name in REQUIRED_FIT_OUTPUTS if not (model_dir / name).exists()]
                if missing:
                    failures.append(f"{model_dir.name}: missing {missing}")
                elif final:
                    failures.extend(final_fit_failures(model_dir, commit))
            draws = by_seed[by_seed["fold"] == int(fold_id)]
            if sorted(draws["seed"]) != sorted(seeds):
                failures.append(f"{label} fold {fold_id}: analysis covers seeds "
                                f"{sorted(draws['seed'])}, the run has {sorted(seeds)}")
                continue
            summary = json.loads((Path(fold["redraw_dir"]) / "redraw_summary.json").read_text())
            row = {"Schema pair": label, "fold": int(fold_id), "n_draws": len(draws),
                   "f1_macro_mean_draws": draws["f1_macro"].mean(),
                   "f1_macro_sd_draws": draws["f1_macro"].std(ddof=1)}
            fold_sd = sd_table[sd_table["fold"] == int(fold_id)]
            for score in ("prob", "logit"):
                for name, short in (("positive", "pos"), ("negative", "neg")):
                    cell = fold_sd[(fold_sd["score"] == score) & (fold_sd["class"] == name)]
                    row[f"median_{score}_sd_draws_{short}"] = (float(cell["median_sd"].iloc[0])
                                                              if len(cell) == 1 else None)
            for key in ("max_negative_pairs_shared_by_two_draws", "negative_pairs_in_multiple_draws"):
                if key not in summary:
                    failures.append(f"{label} fold {fold_id}: redraw_summary.json has no {key}")
                row[key] = summary.get(key)
            rows.append(row)
    table = pd.DataFrame(rows)
    return table, failures


def write_training_report(table, n_folds: int, output_dir: Path) -> None:
    """Write the training-run table, its display version, and the F1 macro pair matrix.

    Args:
      table: from `training_run_table`.
      n_folds: the datasets' fold count, named in the figure title.
      output_dir: where the files go.

    Returns:
      None. Writes `allpairs_summary.csv`, `allpairs_display.csv`, `matrix_f1_macro_mean.csv`,
      `matrix_f1_macro_sd_folds.csv` and `matrix_f1_macro.png`.
    """
    _, pd = _import_data_libs()
    table.to_csv(output_dir / "allpairs_summary.csv", index=False)
    display = table[["Rank", "Schema pair", "HK selected", "Features"]].copy()
    for metric, name in DISPLAY_METRICS.items():
        display[name] = [f"{m:.3f} ± {s:.3f}" for m, s in
                         zip(table[f"{metric}_mean"], table[f"{metric}_sd_folds"])]
    display.to_csv(output_dir / "allpairs_display.csv", index=False)

    labels = table["Schema pair"]
    mean = lower_triangle_matrix(dict(zip(labels, table["f1_macro_mean"])))
    sd = lower_triangle_matrix(dict(zip(labels, table["f1_macro_sd_folds"])))
    mean.to_csv(output_dir / "matrix_f1_macro_mean.csv")
    sd.to_csv(output_dir / "matrix_f1_macro_sd_folds.csv")
    plot_lower_triangle(mean, sd, f"Test F1 macro, mean ± SD across {n_folds} folds",
                        "F1 macro (mean across folds)", output_dir / "matrix_f1_macro.png")
    print(f"Saved the training-run tables and matrix in {output_dir}")


def write_redraw_report(table, output_dir: Path) -> None:
    """Write the redraw-run table and the matrix of the mean within-fold SD of F1 macro across draws.

    Args:
      table: from `redraw_run_table`.
      output_dir: where the files go.

    Returns:
      None. Writes `redraw_by_pair_fold.csv`, `matrix_f1_macro_sd_draws.csv` and
      `matrix_f1_macro_sd_draws.png`.
    """
    table.to_csv(output_dir / "redraw_by_pair_fold.csv", index=False)
    per_pair = table.groupby("Schema pair")["f1_macro_sd_draws"].mean()
    matrix = lower_triangle_matrix(per_pair.to_dict())
    matrix.to_csv(output_dir / "matrix_f1_macro_sd_draws.csv")
    plot_lower_triangle(matrix, None,
                        "Test F1 macro: mean across folds of the within-fold SD across draws",
                        "SD of F1 macro across draws (mean across folds)",
                        output_dir / "matrix_f1_macro_sd_draws.png")
    print(f"Saved the redraw-run table and matrix in {output_dir}")


def run_manifest_mode(args) -> None:
    """Build the reports a training run manifest and/or a redraw run manifest name.

    Args:
      args: the parsed command line, with `training_run_manifest` and/or `redraw_run_manifest`.

    Returns:
      None. Exits non-zero, writing nothing, when any check fails.
    """
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        raise SystemExit("ERROR: --output_dir is required with a run manifest")

    failures, training, redraw = [], None, None
    if args.training_run_manifest:
        table, table_failures, n_folds = training_run_table(Path(args.training_run_manifest),
                                                            args.final)
        failures.extend(table_failures)
        training = (table, n_folds)
    if args.redraw_run_manifest:
        table, table_failures = redraw_run_table(Path(args.redraw_run_manifest), args.final)
        failures.extend(table_failures)
        redraw = table
    if failures:
        print("\n".join(f"ERROR: {message}" for message in failures))
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    if training is not None:
        write_training_report(*training, output_dir)
    if redraw is not None:
        write_redraw_report(redraw, output_dir)
    print(f"\nDone. All outputs in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all protein-pair CV results into cross-pair summary and heatmaps")
    parser.add_argument("--training_run_manifest", type=str, default=None,
                        help="training_run_manifest.json from run_allpairs_baselines.py: report "
                             "exactly the CV directories it lists")
    parser.add_argument("--redraw_run_manifest", type=str, default=None,
                        help="redraw_run_manifest.json from run_allpairs_balanced_redraws.py: "
                             "report the redraw run it records")
    parser.add_argument("--final", action="store_true",
                        help="with a run manifest: require all 28 pairs, and every fit at the "
                             "datasets' commit with a clean worktree")
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

    if args.training_run_manifest or args.redraw_run_manifest:
        run_manifest_mode(args)
        return
    if args.final:
        parser.error("--final applies only with a run manifest")

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
            print("  Backfill with: bash scripts/run_allpairs_post_hoc.sh <TAG>")

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

    # Fold count for the figure titles. Runs that disagree leave it unnamed rather than
    # labelling every pair with one pair's number.
    fold_counts = set(complete["n_folds"].dropna().astype(int)) if "n_folds" in complete else set()
    n_folds = fold_counts.pop() if len(fold_counts) == 1 else None

    # Generate heatmaps. Each matrix is built once and reused for the CSV, the figure and the JSON.
    matrices = {metric: build_heatmap_matrix(table, metric) for metric in HEATMAP_METRICS}
    for metric, matrix in matrices.items():
        # Save matrix CSV
        matrix_csv = output_dir / f"heatmap_{metric}.csv"
        matrix.to_csv(matrix_csv)
        print(f"Saved heatmap matrix: {matrix_csv}")

        # Plot heatmap
        plot_path = output_dir / f"heatmap_{metric}.png"
        plot_heatmap(matrix, metric, plot_path, n_folds=n_folds)

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
    for metric, matrix in matrices.items():
        summary_json[f"heatmap_{metric}"] = matrix.where(matrix.notna(), None).to_dict()

    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    print(f"Saved full results: {json_path}")

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
