"""Stratified diagnostic analysis of pair-classifier predictions.

Purpose
-------
Given a ``test_predicted.csv`` from ANY pair-classifier run (MLP, 1-NN
baseline, logistic, etc.) plus the Stage 3 dataset directory the run
consumed, this script produces three diagnostic stratifications that
inform the leakage-diagnostics plan
(``docs/plans/2026-05-07_leakage_diagnostics_plan.md``):

1. **Hash overlap with train** (mode #3 / mode #4 proxy).
   Per test pair, count how many of its slots have a ``seq_hash`` /
   ``dna_hash`` that also appears in the train split's same-slot
   column. Strata: 0 (both slots disjoint from train), 1 (one slot
   shared), 2 (both slots shared). Computed separately for protein
   hashes and DNA hashes.

   Reading: if the classifier's AUC is *invariant* across these
   strata, exact-hash leakage (mode #3) is NOT the dominant inflation
   — the model is finding *near-neighbors at the feature level* even
   for hash-disjoint pairs (mode #4: cluster leakage).

   This stratum-on-existing-splits analysis is a strong proxy for the
   Plan Exp 4 split-construction experiment (DNA-disjoint splits).
   The classifier's behavior on the ``dna_overlap=0`` subset
   approximates what a full Exp 4 re-train would show, because the
   1-NN's "memorization" works via k-mer cosine, not hash equality.

2. **Metadata match_count on negatives** (mode #5: demographic
   shortcut). For each negative pair, count how many of
   ``{same_hn_subtype, same_host, same_year, same_geo_location,
   same_passage}`` are True. Report FP rate and mean predicted
   probability per match_count bucket. Replicates the diagnostic in
   ``docs/results/2026-05-07_metadata_shortcut_negatives.md`` for any
   classifier — useful to confirm the shortcut lives in the FEATURES
   (not the model) when a non-parametric baseline reproduces the
   pattern.

3. **Score distribution by true label**. mean / median / std / range
   of ``pred_prob`` per true label. A wide bimodal separation
   indicates clear class signal; tight clustering near the decision
   boundary (e.g., all probs in [0.43, 0.57]) means the classifier
   rides a razor-thin margin even if AUC is high — a hallmark of
   1-NN-style cluster lookup in dense feature space.

Inputs / outputs
----------------
Input: ``--run_dir`` containing ``test_predicted.csv``,
``--dataset_dir`` containing ``train_pairs.csv``.
The pred CSV must have columns: ``label``, ``pred_label``,
``pred_prob``, ``seq_hash_a``, ``seq_hash_b``, ``dna_hash_a``,
``dna_hash_b``, plus the ``same_*`` metadata columns for stratum 2.
These are emitted by both ``train_pair_classifier.py`` (MLP) and
``train_pair_baselines.py`` (sklearn baselines).

Outputs (under ``<run_dir>/post_hoc_stratified/`` by default):
    stratified_hash_overlap.csv      # one row per (hash_type, overlap_count)
    stratified_match_count.csv       # one row per match_count bucket (negatives)
    stratified_score_distribution.csv  # one row per true label
    stratified_summary.json          # all three above as a single JSON

The CSVs are intentionally small (5–10 rows each) — they are designed
to be re-read directly into other scripts or pasted into result
writeups, not browsed in a viewer.

Why these three stratifications
-------------------------------
The leakage taxonomy in ``docs/methods/leakage_definitions.md`` lists
five canonical modes. This script directly addresses three of them in
post-hoc form (no re-training required):

| Mode | Stratification used |
|------|---------------------|
| #3 sequence-level leakage | hash overlap (seq_hash, dna_hash) |
| #4 cluster leakage         | hash overlap (proxy: invariance across strata implies cluster-level memorization) |
| #5 demographic shortcut    | match_count on negatives |

Modes #1 (same-pair leakage) and #2 (label imbalance) are addressed
at construction time by ``dataset_segment_pairs_v2.py`` — they don't
need post-hoc stratification.

Usage
-----
    python src/analysis/analyze_predictions_stratified.py \\
        --run_dir models/flu/July_2025/runs/baseline_knn_flu_ha_na_20260508_211301 \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260508_171512

To compare two runs (e.g., MLP vs 1-NN), invoke twice with different
``--run_dir`` arguments and diff the resulting CSVs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Metadata fields used to compute per-pair "match_count" (mode #5).
# Each is a binary same_* flag emitted by Stage 3 (``dataset_segment_pairs_v2.py``).
# Higher match_count == more demographic agreement between the two
# pair members → easier to confuse with a positive even when the pair
# is artificially mixed.
MATCH_FIELDS = [
    "same_hn_subtype",
    "same_host",
    "same_year",
    "same_geo_location",
    "same_passage",
]


def _safe_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC that returns NaN when only one label is present.

    sklearn raises ``ValueError`` on a single-class y_true, which is
    common in tiny strata. Returning NaN keeps the per-stratum table
    interpretable without losing the rest of the rows.
    """
    if len(np.unique(y_true)) < 2 or len(y_true) < 5:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _classification_metrics(df: pd.DataFrame) -> dict:
    """Standard binary metrics for a stratum.

    Returns ``{n, n_pos, frac_pos, auc_roc, f1, precision, recall,
    accuracy}``. ``f1`` / ``precision`` / ``recall`` use
    ``zero_division=0`` so that all-one-class strata don't error out
    — they return 0.0 instead, which is the correct conservative
    estimate when the prediction can't be evaluated on the absent
    class.
    """
    n = len(df)
    if n == 0:
        return {"n": 0}
    y_true = df.label.to_numpy()
    y_score = df.pred_prob.to_numpy()
    y_pred = df.pred_label.to_numpy()
    return {
        "n": int(n),
        "n_pos": int(y_true.sum()),
        "frac_pos": float(y_true.mean()),
        "auc_roc": _safe_auc_roc(y_true, y_score),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_pred == y_true).mean()),
    }


def stratify_by_hash_overlap(
    test: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Stratify test predictions by hash-overlap-with-train (modes #3/#4).

    Parameters
    ----------
    test : pd.DataFrame
        Test predictions with ``label``, ``pred_label``, ``pred_prob``,
        ``seq_hash_{a,b}``, ``dna_hash_{a,b}`` columns.
    train : pd.DataFrame
        Train pairs DataFrame from Stage 3, used only to build the
        per-slot hash-set memberships.

    Returns
    -------
    pd.DataFrame
        One row per (hash_type, overlap_count). overlap_count is in
        {0, 1, 2} representing how many of the pair's two slots have
        a hash also present in train's same-slot column.

    Why this works as a mode #3/#4 diagnostic
    -----------------------------------------
    - Mode #3 (sequence-level leakage): if metric on overlap=2 >>
      metric on overlap=0, exact-hash overlap is inflating the
      headline number.
    - Mode #4 (cluster leakage): if metric on overlap=0 is already
      near the headline number, the classifier is finding
      *near-neighbors at the feature level* even when no hash matches
      — i.e., cluster leakage, not exact-hash leakage. For a 1-NN
      baseline on dense k-mer features, this almost always indicates
      mode #4 dominates.

    Note on protein vs DNA framing
    ------------------------------
    For k-mer-based models, the substantively-important hash is
    ``dna_hash`` (k-mer features are derived from DNA, so two pairs
    with the same ``dna_hash`` have byte-identical features). For
    ESM-2-based models, ``seq_hash`` (protein) is the right framing.
    Both are computed here so the same script applies to both
    feature sources.
    """
    train_hashes = {
        "seq_hash": (set(train.seq_hash_a), set(train.seq_hash_b)),
        "dna_hash": (set(train.dna_hash_a), set(train.dna_hash_b)),
    }

    rows = []
    for hash_type, (set_a, set_b) in train_hashes.items():
        in_a = test[f"{hash_type}_a"].isin(set_a)
        in_b = test[f"{hash_type}_b"].isin(set_b)
        overlap = in_a.astype(int) + in_b.astype(int)
        for k in (0, 1, 2):
            sub = test[overlap == k]
            metrics = _classification_metrics(sub)
            rows.append({"hash_type": hash_type, "overlap_count": k, **metrics})
    return pd.DataFrame(rows)


def stratify_by_match_count(test: pd.DataFrame) -> pd.DataFrame:
    """Stratify negatives by metadata match_count (mode #5).

    Parameters
    ----------
    test : pd.DataFrame
        Test predictions with ``label``, ``pred_label``, ``pred_prob``,
        and the ``same_*`` columns listed in ``MATCH_FIELDS``.

    Returns
    -------
    pd.DataFrame
        One row per match_count bucket (0..len(MATCH_FIELDS)). Reports
        FP rate (fraction of negatives the classifier predicted as
        positive), mean predicted probability, and counts.

    Reading
    -------
    A monotonic FP-rate climb from match_count=0 to
    match_count=max means the classifier is using metadata
    correlations as a shortcut to predict "positive" (same isolate).
    Documented in
    ``docs/results/2026-05-07_metadata_shortcut_negatives.md`` for
    the MLP. Reproducing the same pattern in a non-parametric 1-NN
    is direct evidence the shortcut is encoded in the FEATURES (not
    the model) — k-mer / ESM-2 features carry subtype/host/year
    signal regardless of architecture.
    """
    neg = test[test.label == 0].copy()
    available_fields = [c for c in MATCH_FIELDS if c in neg.columns]
    if not available_fields:
        # Some older pipelines may not emit same_* columns; degrade
        # gracefully rather than crash so the other strata still land.
        return pd.DataFrame(columns=["match_count", "n", "fp_rate", "mean_pred_prob"])
    neg["match_count"] = neg[available_fields].sum(axis=1)

    rows = []
    for mc in sorted(neg.match_count.unique()):
        sub = neg[neg.match_count == mc]
        rows.append({
            "match_count": int(mc),
            "n": int(len(sub)),
            "fp_rate": float((sub.pred_label == 1).mean()),
            "mean_pred_prob": float(sub.pred_prob.mean()),
            "median_pred_prob": float(sub.pred_prob.median()),
        })
    return pd.DataFrame(rows)


def score_distribution(test: pd.DataFrame) -> pd.DataFrame:
    """Score distribution of ``pred_prob`` by true label.

    Parameters
    ----------
    test : pd.DataFrame
        Test predictions with ``label`` and ``pred_prob`` columns.

    Returns
    -------
    pd.DataFrame
        One row per label class with mean / median / std / min / max
        and key quantiles of ``pred_prob``.

    Reading
    -------
    A bimodal distribution with negatives clustered near 0 and
    positives near 1 indicates clean class separation. Tight
    clustering near 0.5 (e.g., all probs in [0.43, 0.57]) means the
    classifier rides a razor-thin decision margin — a hallmark of
    1-NN-style cluster lookup in dense feature space, where every
    test point has BOTH a near-positive AND a near-negative neighbor
    at almost equal cosine distance, but the SIGN of the tiny tilt
    is reliable.
    """
    rows = []
    for label, sub in test.groupby("label"):
        p = sub.pred_prob
        rows.append({
            "label": int(label),
            "n": int(len(sub)),
            "mean": float(p.mean()),
            "median": float(p.median()),
            "std": float(p.std()),
            "min": float(p.min()),
            "max": float(p.max()),
            "q10": float(p.quantile(0.10)),
            "q90": float(p.quantile(0.90)),
        })
    return pd.DataFrame(rows)


def overall_metrics(test: pd.DataFrame) -> dict:
    """Headline metrics on the full test set (no stratification).

    Returns ``{n, n_pos, frac_pos, auc, f1, precision, recall,
    accuracy}`` — same schema as the per-stratum metrics, so they
    can be diffed cell-by-cell.
    """
    return _classification_metrics(test)


def run_analysis(
    run_dir: Path,
    dataset_dir: Path,
    output_dir: Optional[Path] = None,
    ) -> dict:
    """Top-level entry point: load CSVs, run all three stratifications,
    write outputs, return the summary dict.

    Wired so that callers (a notebook, another script, or the CLI in
    ``main()``) can either trigger the side-effecting writes by
    passing ``output_dir`` or just consume the in-memory summary.
    """
    test_csv = run_dir / "test_predicted.csv"
    train_csv = dataset_dir / "train_pairs.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test predictions: {test_csv}")
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train pairs: {train_csv}")

    print(f"Loading test predictions: {test_csv}")
    # ``engine='python'`` matches the trainer/baseline harness — avoids
    # rare segfaults on the C parser with certain seq characters.
    test = pd.read_csv(test_csv, engine="python")
    print(f"  test rows: {len(test):,}")
    print(f"Loading train pairs:      {train_csv}")
    train = pd.read_csv(train_csv, engine="python")
    print(f"  train rows: {len(train):,}")

    print("\n=== Overall test metrics ===")
    overall = overall_metrics(test)
    print(f"  n={overall['n']:,}  pos_frac={overall['frac_pos']:.3f}  "
          f"AUC-ROC={overall['auc_roc']:.4f}  F1={overall['f1']:.4f}  "
          f"Acc={overall['accuracy']:.4f}")

    print("\n=== 1) Hash-overlap stratification (modes #3 / #4) ===")
    hash_df = stratify_by_hash_overlap(test, train)
    print(hash_df.to_string(index=False))

    print("\n=== 2) Metadata match_count stratification (negatives, mode #5) ===")
    mc_df = stratify_by_match_count(test)
    print(mc_df.to_string(index=False))

    print("\n=== 3) Score distribution by true label ===")
    sd_df = score_distribution(test)
    print(sd_df.to_string(index=False))

    summary = {
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "overall": overall,
        "hash_overlap": hash_df.to_dict(orient="records"),
        "match_count": mc_df.to_dict(orient="records"),
        "score_distribution": sd_df.to_dict(orient="records"),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        hash_df.to_csv(output_dir / "stratified_hash_overlap.csv", index=False)
        mc_df.to_csv(output_dir / "stratified_match_count.csv", index=False)
        sd_df.to_csv(output_dir / "stratified_score_distribution.csv", index=False)
        with open(output_dir / "stratified_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote outputs to: {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified diagnostic analysis of pair-classifier predictions."
    )
    parser.add_argument(
        "--run_dir", type=Path, required=True,
        help="Run directory containing test_predicted.csv (MLP or baseline).",
    )
    parser.add_argument(
        "--dataset_dir", type=Path, required=True,
        help="Stage 3 dataset directory containing train_pairs.csv.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help=("Where to write the per-stratum CSVs and summary JSON. "
              "Default: <run_dir>/post_hoc_stratified/."),
    )
    args = parser.parse_args()

    out = args.output_dir or (args.run_dir / "post_hoc_stratified")
    run_analysis(args.run_dir, args.dataset_dir, out)


if __name__ == "__main__":
    sys.exit(main())
