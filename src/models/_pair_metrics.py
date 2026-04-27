"""Pair-classification metric helpers shared by the MLP trainer and baselines.

This module owns the model-agnostic prediction/metric/IO helpers that used to
live inside ``train_pair_classifier.py``. The MLP trainer keeps the torch
inference loop; everything downstream of "we have probabilities and labels"
lives here so baselines can reuse the same logic.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold_pr(y_true, y_probs, metric='f1'):
    """
    Find optimal threshold using Precision-Recall curve.
    TODO: allow an option to generate a plot of the PR curve, showing the optimal threshold and the best score.

    This method is preferred for imbalanced datasets and when optimizing F1.
    It's faster than grid search and directly optimizes F1 score.

    Args:
        y_true: True binary labels (array-like)
        y_probs: Predicted probabilities (array-like)
        metric: Metric to optimize ('f1', 'f0.5', 'f2')
            - 'f1': Maximize F1 score (harmonic mean of precision and recall)
            - 'f0.5': Emphasize precision more (F0.5 = (1+0.5²) * P*R / (0.5²*P + R))
            - 'f2': Emphasize recall more (F2 = (1+2²) * P*R / (2²*P + R))

    Returns:
        optimal_threshold: Threshold that maximizes the specified metric
        best_score: Best score achieved at optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    # Handle edge case: no thresholds (all predictions same class)
    if len(thresholds) == 0:
        return 0.5, 0.0

    # Calculate F-beta scores
    if metric == 'f1':
        # F1 = 2 * (precision * recall) / (precision + recall)
        # Add small epsilon to avoid division by zero
        f_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    elif metric == 'f0.5':
        # F0.5 emphasizes precision: beta = 0.5
        beta_sq = 0.5 ** 2
        f_scores = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-10)
    elif metric == 'f2':
        # F2 emphasizes recall: beta = 2
        beta_sq = 2 ** 2
        f_scores = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-10)
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from 'f1', 'f0.5', 'f2'")

    # Find best F-score (excluding last point which has threshold=None)
    # The last point in precision_recall_curve has threshold=None and corresponds
    # to the case where all predictions are positive
    optimal_idx = np.argmax(f_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    best_score = f_scores[optimal_idx]

    return optimal_threshold, best_score


def compute_pair_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    pairs_df: pd.DataFrame,
    *,
    logits: Optional[np.ndarray] = None,
    print_classification_report: bool = True,
    ) -> tuple[dict, pd.DataFrame]:
    """Compute binary classification metrics and build a per-pair predictions df.

    Mirrors the metric block that used to live inline in
    ``evaluate_on_split``. Model-agnostic: feed it probabilities and labels.

    Args:
        y_true: True binary labels.
        y_probs: Predicted probabilities for the positive class.
        threshold: Classification threshold applied to ``y_probs``.
        pairs_df: Original pair-level DataFrame; columns are preserved and
            pred_label/pred_prob/pred_logit are appended.
        logits: Raw logits from a torch model. ``None`` for sklearn-style
            baselines that don't expose logits, in which case ``pred_logit``
            is omitted from ``res_df``.
        print_classification_report: If True, print sklearn's
            ``classification_report``. Caller-controlled because some loops
            (per-epoch eval) don't want the noise.

    Returns:
        metrics: dict with keys f1, f1_macro, precision, recall, auc.
        res_df:  copy of ``pairs_df`` with prediction columns appended.
    """
    pred_labels = (y_probs > threshold).astype(np.float32)

    # Compute metrics for positive class (label=1, same isolate)
    # Explicit parameters: average='binary', pos_label=1
    f1 = f1_score(y_true, pred_labels, average='binary', pos_label=1)
    f1_macro = f1_score(y_true, pred_labels, average='macro')
    precision = precision_score(y_true, pred_labels, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(y_true, pred_labels, average='binary', pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        # Degenerate predictions (all same class) break roc_auc_score; fall
        # back to chance.
        auc = 0.5

    if print_classification_report:
        print('\nClassification Report:')
        print(classification_report(y_true, pred_labels, target_names=['Negative', 'Positive']))

    res_df = pairs_df.copy()
    res_df['pred_label'] = pred_labels
    res_df['pred_prob'] = y_probs
    if logits is not None:
        res_df['pred_logit'] = logits

    metrics = {
        'f1': f1,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'auc': auc,
    }
    return metrics, res_df


def swap_pairs_df_columns(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of pairs_df with every *_a/*_b column pair swapped.

    This is used for the "swap test" diagnostic: evaluate a trained directed model
    (e.g., features=[emb_a, emb_b]) on swapped inputs (b,a) with labels unchanged.
    """
    swapped = pairs_df.copy()
    cols = list(swapped.columns)
    for col_a in cols:
        if not col_a.endswith("_a"):
            continue
        col_b = col_a[:-2] + "_b"
        if col_b in swapped.columns:
            tmp = swapped[col_a].copy()
            swapped[col_a] = swapped[col_b]
            swapped[col_b] = tmp
    return swapped
