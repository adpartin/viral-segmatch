"""Generalized k-NN voting baseline for pair classification.

Smoothed local-neighborhood baseline: for each test pair, gather its k
nearest train pairs in cosine-distance space and predict by vote
(uniform or distance-weighted). Distinct from
``baselines/knn1_margin.py`` which is the dedicated k=1 cosine-margin
*leakage diagnostic* -- this one is the generic non-parametric
classifier you would expect a referee to compare against.

Implementation: thin wrapper over
``sklearn.neighbors.KNeighborsClassifier(metric='cosine')``. The
estimator's ``predict_proba`` returns vote fractions (or
distance-weighted fractions when ``weights='distance'``), giving a
continuous score for AUC and the threshold-tuning step.

Bundle overrides under ``config.baseline_knn_vote.*`` (defaults shown):

| key              | default   | notes                                          |
|------------------|-----------|------------------------------------------------|
| k                | 5         | odd k avoids hard-vote ties on binary labels.  |
| weights          | distance  | 'uniform' or 'distance'. distance gives a      |
|                  |           | finer probability gradient near the boundary.  |
| algorithm        | brute     | brute = BLAS matmul, fastest for ~8K-dim       |
|                  |           | k-mer concat.                                  |
| n_jobs           | -1        | passed to sklearn KNeighborsClassifier         |
| feature_scaling  | none      | cosine is scale-invariant; non-negative k-mer  |
|                  |           | counts shouldn't be standardized.              |

Notes on training-set self-match:
The brute-force cosine search includes the query when fitting and
predicting on identical X, so train-set ``predict_proba`` reflects a
k-NN that may include the query itself. This is the standard sklearn
behavior; it does not affect val/test predictions (which come from
disjoint splits in v2 by construction). If you ever need leave-one-out
scoring for a "correct" train metric, see ``knn1_margin.py``'s
``LOO_EPS`` pattern.
"""
from typing import Optional

import numpy as np


def name() -> str:
    return "knn_vote"


def feature_scaling_default() -> str:
    return "none"


def get_estimator(config, *, random_state: Optional[int] = None):
    from sklearn.neighbors import KNeighborsClassifier

    cfg = getattr(config, 'baseline_knn_vote', None)
    cfg = dict(cfg) if cfg is not None else {}

    k = int(cfg.get('k', 5))
    if k < 1:
        raise ValueError(f"baseline_knn_vote.k must be >= 1, got {k}")
    if k % 2 == 0:
        # Even k can produce hard-vote ties on binary tasks; sklearn
        # breaks them by class index (predicts class 0), which biases
        # accuracy. Warn rather than error so the user can opt-in.
        print(
            f"WARNING: baseline_knn_vote.k={k} is even. Hard-vote ties "
            "are broken toward class 0 by sklearn; prefer odd k. "
            "(weights='distance' mostly avoids ties anyway.)"
        )

    return KNeighborsClassifier(
        n_neighbors=k,
        weights=str(cfg.get('weights', 'distance')),
        metric='cosine',
        algorithm=str(cfg.get('algorithm', 'brute')),
        n_jobs=int(cfg.get('n_jobs', -1)),
    )
