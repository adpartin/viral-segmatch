"""Exp 3 — cosine-decile stratification for cluster-leakage diagnosis.

Plan reference: ``docs/plans/2026-05-07_leakage_diagnostics_plan.md``,
Exp 3 ("Stratified accuracy by nearest-train cosine"). Operates
post-hoc on existing trained runs — does NOT rebuild splits or
retrain. Cheap diagnostic that pairs naturally with Exp 2 (the 1-NN
baseline) to characterise mode #4 (cluster leakage) in the leakage
taxonomy at ``docs/methods/leakage.md``.

What it measures
----------------
For each test pair, compute the maximum cosine similarity to any
train pair in the same joint-feature space the classifier consumed
(k-mer concat by default). Stratify test pairs into equal-frequency
buckets (deciles) over that max-cosine value. For each provided run
directory's ``test_predicted.csv``, compute test metrics (AUC, F1,
accuracy, FP rate) per bucket.

The bucket-0 stratum is the *farthest-from-train* slice of the
existing test set — the closest analog the dataset gives us to
"out-of-cluster" pairs without re-building splits.

Reading
-------
- **AUC flat across buckets, near headline.** Both runs find
  near-neighbors at every cosine level → cluster leakage (#4) is
  saturated even at the natural test distribution's tail. Exp A
  (cosine-controlled split construction) is needed to push beyond
  the dataset's own cosine floor.
- **AUC drops smoothly as bucket index decreases (low-cosine =
  hard).** The decile curve directly shows the cosine threshold at
  which performance breaks.
- **MLP AUC > 1-NN AUC in low-cosine buckets.** Positive evidence of
  biology learning beyond memorisation: the MLP succeeds on
  far-from-train pairs where 1-NN cannot. Triggers Exp A and Exp 5
  to confirm at split-construction time.
- **MLP and 1-NN drop together in low-cosine buckets.** Soft-1-NN
  story confirmed. The MLP captures only what cosine-NN already
  captures — no biology learning visible at this dataset/feature
  combination.

Why max-cosine (not k-NN distances per pos/neg)
-----------------------------------------------
For Exp 3 we want to characterise the *test pair's distance to
train as a whole*, not the label-specific margin. The cluster-leakage
question is "does the test pair have ANY near-neighbor in train" —
which is independent of the neighbor's label. A test pair that
matches a near-cousin train pair will inherit its label regardless
of whether that cousin is positive or negative; what matters is
whether the cousin exists at all.

How fast
--------
Implementation uses chunked dense matrix multiplication on
L2-normalised features (BLAS-3). For ~14K test × ~106K train ×
~8K-dim k-mer concat features, total runtime on a typical multi-core
node is 2–5 minutes — substantially faster than sklearn
``NearestNeighbors`` for this shape (BLAS-2 distance loops).

Inputs / outputs
----------------
- ``--dataset_dir``: Stage 3 directory. Provides
  ``train_pairs.csv``, ``test_pairs.csv``; the k-mer matrix is
  resolved from ``virus.data_version`` via ``build_embeddings_paths``.
- ``--run_dirs``: one or more run directories that contain
  ``test_predicted.csv`` row-aligned with ``test_pairs.csv``. Both
  MLP runs (``training_*``) and baseline runs
  (``baseline_*``) are supported — the harness contract for
  ``test_predicted.csv`` is identical.
- ``--run_labels``: optional friendly labels (default: dir
  basenames).
- ``--output_dir``: where to write the per-test-pair cosine, the
  per-run-per-bucket metrics, the JSON summary, and the AUC-vs-decile
  PNG.

Outputs
-------
``<output_dir>/``:
    test_max_cosine.csv         # one row per test pair: idx, max_cosine, bucket
    cosine_deciles.csv          # long-format: one row per (run, bucket)
    cosine_deciles.png          # AUC vs bucket, one line per run
    cosine_deciles_summary.json # config + bucket edges + per-run aggregates

Usage
-----
    python src/analysis/exp3_cosine_deciles.py \\
        --config_bundle flu_ha_na \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260508_171512 \\
        --run_dirs models/flu/July_2025/runs/training_flu_ha_na_20260508_214253 \\
                   models/flu/July_2025/runs/baseline_knn_flu_ha_na_20260508_211301 \\
        --run_labels MLP 1-NN \\
        --output_dir results/flu/July_2025/runs/exp3_cosine_deciles_ha_na_20260508
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: no DISPLAY needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

# Make src/ imports work when invoked directly as a script.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models._pair_features import load_pair_features_for_baselines
from src.utils.config_hydra import get_virus_config_hydra
from src.utils.path_utils import build_embeddings_paths


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Returns float32 array of unit-norm
    rows. Pairs with a zero row (all-zero k-mer vector) get NaN; this
    shouldn't happen for valid features but is left to surface as NaN
    rather than silently divide by zero.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (X / norms).astype(np.float32, copy=False)


def compute_max_train_cosine(
    X_train: np.ndarray,
    X_test: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """For each test pair, return the max cosine similarity to any
    train pair.

    Implementation: L2-normalise both matrices once, then chunk the
    test set and compute ``chunk @ X_train_n.T`` (a dense
    matrix-matrix product per chunk). ``max(axis=1)`` gives the per-
    test-row maximum without materialising the full
    ``n_test × n_train`` similarity matrix in memory.

    Parameters
    ----------
    X_train, X_test : np.ndarray, shape (n_train, d) / (n_test, d)
        Pair features (e.g., k-mer concat). Dense.
    chunk_size : int
        Test rows per matmul. ~512 keeps each chunk's similarity
        slice under ~250MB for ~110K train pairs (float32) — fits
        comfortably in cache and avoids fragmentation.

    Returns
    -------
    np.ndarray, shape (n_test,), float32 in [-1, 1].
    """
    Xn_train = _l2_normalize(X_train)
    Xn_test = _l2_normalize(X_test)
    n_test = Xn_test.shape[0]
    out = np.empty(n_test, dtype=np.float32)
    for i in range(0, n_test, chunk_size):
        sim = Xn_test[i : i + chunk_size] @ Xn_train.T  # (chunk, n_train)
        out[i : i + chunk_size] = sim.max(axis=1)
    return out


def stratify_by_cosine(
    max_cos: np.ndarray,
    n_buckets: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratify into ``n_buckets`` equal-frequency buckets.

    Returns
    -------
    bucket_id : np.ndarray, shape (n_test,), int
        Bucket index in [0, n_buckets-1]. Bucket 0 = lowest cosine
        (farthest from train), n_buckets-1 = highest cosine (most
        memorisable).
    edges : np.ndarray, shape (n_buckets+1,)
        Quantile edges over ``max_cos``.
    """
    edges = np.quantile(max_cos, np.linspace(0, 1, n_buckets + 1))
    # nudge the rightmost edge up so the global max lands inside the last bucket
    edges = edges.copy()
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bucket = np.digitize(max_cos, edges, right=False) - 1
    bucket = np.clip(bucket, 0, n_buckets - 1)
    return bucket, edges


def per_bucket_metrics(
    pred_df: pd.DataFrame,
    bucket: np.ndarray,
    n_buckets: int,
) -> pd.DataFrame:
    """Compute classification metrics for each bucket.

    Returns one row per bucket with ``n``, ``n_pos``, ``frac_pos``,
    ``auc_roc``, ``f1``, ``accuracy``, ``fp_rate``, ``mean_pred_prob``.
    AUC-ROC is NaN on single-class buckets (sklearn would raise).
    """
    rows = []
    y = pred_df["label"].to_numpy()
    p = pred_df["pred_prob"].to_numpy()
    yp = pred_df["pred_label"].to_numpy()
    for k in range(n_buckets):
        mask = bucket == k
        n = int(mask.sum())
        if n == 0:
            continue
        y_k = y[mask]
        p_k = p[mask]
        yp_k = yp[mask]
        try:
            auc_roc = float(roc_auc_score(y_k, p_k))
        except ValueError:
            auc_roc = float("nan")
        n_pos = int(y_k.sum())
        n_neg = n - n_pos
        rows.append({
            "bucket": k,
            "n": n,
            "n_pos": n_pos,
            "frac_pos": float(y_k.mean()),
            "auc_roc": auc_roc,
            "f1": float(f1_score(y_k, yp_k, zero_division=0)),
            "accuracy": float((yp_k == y_k).mean()),
            "fp_rate": float((yp_k[y_k == 0] == 1).mean()) if n_neg > 0 else float("nan"),
            "mean_pred_prob": float(p_k.mean()),
        })
    return pd.DataFrame(rows)


def plot_auc_vs_bucket(
    long_df: pd.DataFrame,
    edges: np.ndarray,
    out_png: Path,
) -> None:
    """One line per run: AUC-ROC across buckets. Bucket centres are
    annotated with the cosine-edge mid-point so readers can decode
    the x-axis to a similarity range."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for run, sub in long_df.groupby("run", sort=False):
        ax.plot(sub["bucket"], sub["auc_roc"], marker="o", label=run)
    centres = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(range(len(centres)))
    ax.set_xticklabels([f"{i}\n[{edges[i]:.3f},\n{edges[i+1]:.3f})"
                        for i in range(len(centres))],
                       fontsize=8)
    ax.set_xlabel("Cosine bucket  (0 = farthest from train, 9 = closest)")
    ax.set_ylabel("Test AUC-ROC")
    ax.set_title("AUC-ROC vs max-train-cosine bucket\n"
                 "(bucket-0 is the closest analog to out-of-cluster pairs in this test set)")
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, ls="--", c="grey", alpha=0.5, label="chance")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 3 — cosine-decile stratification of test predictions.",
    )
    parser.add_argument("--config_bundle", default="flu_ha_na",
                        help="Hydra bundle name (drives data_version + kmer.k).")
    parser.add_argument("--dataset_dir", type=Path, required=True,
                        help="Stage 3 directory with train_pairs.csv + test_pairs.csv.")
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True,
                        help="Run directories with test_predicted.csv (MLP and/or baselines).")
    parser.add_argument("--run_labels", type=str, nargs="+", default=None,
                        help="Friendly labels per run (default: directory basenames).")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Where to write CSV / PNG / JSON outputs.")
    parser.add_argument("--n_buckets", type=int, default=10,
                        help="Number of equal-frequency cosine buckets (default 10).")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Test-row chunk size for dense matmul (default 512).")
    args = parser.parse_args()

    if args.run_labels and len(args.run_labels) != len(args.run_dirs):
        raise SystemExit("--run_labels must have the same length as --run_dirs.")
    labels = args.run_labels or [d.name for d in args.run_dirs]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve config + k-mer dir ─────────────────────────────────────────
    config = get_virus_config_hydra(args.config_bundle, config_path=str(project_root / "conf"))
    kmer_k = int(config.kmer.k) if hasattr(config, "kmer") and config.get("kmer") is not None else 6
    kmer_dir = build_embeddings_paths(
        project_root=project_root,
        virus_name=config.virus.virus_name,
        data_version=config.virus.data_version,
        run_suffix="",
        config=config,
    )["output_dir"]

    # ── Load pair CSVs + features ──────────────────────────────────────────
    print(f"Loading pair CSVs from {args.dataset_dir}")
    train = pd.read_csv(args.dataset_dir / "train_pairs.csv", engine="python")
    val = pd.read_csv(args.dataset_dir / "val_pairs.csv", engine="python")
    test = pd.read_csv(args.dataset_dir / "test_pairs.csv", engine="python")
    print(f"  train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")

    print(f"Loading k-mer pair features (k={kmer_k}) from {kmer_dir}")
    (X_train, _), (_, _), (X_test, _), _ = load_pair_features_for_baselines(
        train, val, test,
        feature_source="kmer",
        feature_scaling="none",
        kmer_dir=kmer_dir,
        kmer_k=kmer_k,
        output_dir=args.output_dir,  # also where the (unused) scaler would go
    )
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}  dtype: {X_train.dtype}")

    # ── Compute max-train-cosine per test pair ─────────────────────────────
    print(f"\nComputing max-train-cosine for {len(X_test):,} test pairs "
          f"(chunk_size={args.chunk_size})...")
    max_cos = compute_max_train_cosine(X_train, X_test, chunk_size=args.chunk_size)
    print(f"  cosine range: [{max_cos.min():.4f}, {max_cos.max():.4f}]  "
          f"median: {np.median(max_cos):.4f}  mean: {max_cos.mean():.4f}")

    bucket, edges = stratify_by_cosine(max_cos, n_buckets=args.n_buckets)
    print(f"  bucket edges: {[f'{e:.4f}' for e in edges]}")

    pd.DataFrame({
        "test_index": np.arange(len(max_cos)),
        "max_cosine": max_cos,
        "bucket": bucket,
    }).to_csv(args.output_dir / "test_max_cosine.csv", index=False)

    # ── Per-run per-bucket metrics ─────────────────────────────────────────
    long_rows = []
    per_run_overall = {}
    for label, run_dir in zip(labels, args.run_dirs):
        pred_csv = run_dir / "test_predicted.csv"
        pred_df = pd.read_csv(pred_csv, engine="python")
        if len(pred_df) != len(test):
            raise SystemExit(
                f"{label}: test_predicted.csv has {len(pred_df):,} rows "
                f"but test_pairs.csv has {len(test):,}. "
                "Predictions must be row-aligned with test_pairs.csv."
            )
        df = per_bucket_metrics(pred_df, bucket, args.n_buckets)
        df["run"] = label
        long_rows.append(df)

        # headline AUC-ROC for the summary
        try:
            overall_auc_roc = float(roc_auc_score(pred_df["label"], pred_df["pred_prob"]))
        except ValueError:
            overall_auc_roc = float("nan")
        per_run_overall[label] = {
            "run_dir": str(run_dir),
            "overall_auc_roc": overall_auc_roc,
            "bucket0_auc_roc": float(df.loc[df.bucket == 0, "auc_roc"].iloc[0]) if (df.bucket == 0).any() else None,
        }

        print(f"\n{label}:")
        print(df[["bucket", "n", "n_pos", "auc_roc", "f1", "fp_rate"]].to_string(index=False))

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df.to_csv(args.output_dir / "cosine_deciles.csv", index=False)

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_auc_vs_bucket(long_df, edges, args.output_dir / "cosine_deciles.png")

    # ── Summary JSON ───────────────────────────────────────────────────────
    summary = {
        "config_bundle": args.config_bundle,
        "dataset_dir": str(args.dataset_dir),
        "kmer_k": kmer_k,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_buckets": args.n_buckets,
        "bucket_edges": edges.tolist(),
        "cosine_stats": {
            "min": float(max_cos.min()),
            "max": float(max_cos.max()),
            "mean": float(max_cos.mean()),
            "median": float(np.median(max_cos)),
            "p10": float(np.quantile(max_cos, 0.10)),
            "p90": float(np.quantile(max_cos, 0.90)),
        },
        "per_run": per_run_overall,
    }
    with open(args.output_dir / "cosine_deciles_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote outputs to: {args.output_dir}")
    print(f"  test_max_cosine.csv          ({len(max_cos):,} rows)")
    print(f"  cosine_deciles.csv           ({len(long_df):,} rows)")
    print(f"  cosine_deciles.png")
    print(f"  cosine_deciles_summary.json")


if __name__ == "__main__":
    main()
