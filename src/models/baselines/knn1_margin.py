"""1-nearest-neighbor baseline for pair classification (Plan Exp 2).

Operational test of the "model learned biology" criterion in
``docs/methods/leakage.md``: if the MLP's metrics on the
same dataset and the same features are not meaningfully better than a
1-NN classifier (e.g., < 0.02 AUC gap), the MLP is doing soft
near-neighbor lookup, not generalization. Plug-in baseline for the
``train_pair_baselines.py`` harness — same Stage 3 dataset, same
feature loader, same pair-level metrics as the other baselines.

Hard prediction: label of the single nearest train pair (k=1) under
cosine distance.

Continuous score (for AUC): a cosine *margin*,
``score = cosine(test, nearest_train_positive)
       - cosine(test, nearest_train_negative)``,
affinely mapped from [-2, 2] to [0, 1] for ``predict_proba``. This
preserves a continuous ranking (sklearn's
``KNeighborsClassifier.predict_proba`` would return only {0, 1} at
k=1 and degenerate the AUC). The hard 1-NN decision and the
``predict_proba >= 0.5`` decision coincide, so any val-driven
threshold optimization in the harness still snaps back to the 1-NN
boundary at the default 0.5. AUC is rank-invariant under monotonic
transforms, so the affine mapping does not change AUC values.

Train predictions use leave-one-out automatically: when the nearest
same-label point has cosine distance below ``LOO_EPS`` (i.e., it is
the query itself), the second-nearest is used. v2 pair_key dedup
guarantees that distinct pairs have distinct k-mer feature vectors
(distinct ``(seq_hash_a, seq_hash_b)`` → at least one differing
protein → differing k-mer composition), so the eps test only fires on
genuine self-matches.

Bundle overrides under ``config.baseline_knn1_margin.*`` (defaults shown):

| key             | default | notes                                           |
|-----------------|---------|-------------------------------------------------|
| n_jobs          | -1      | passed to sklearn NearestNeighbors              |
| algorithm       | brute   | best for ~8K-dim k-mer concat (BLAS matmul)     |
| feature_scaling | none    | cosine is scale-invariant; StandardScaler would |
|                 |         | destroy non-negativity of k-mer counts          |

The companion ``knn_vote`` baseline (``baselines/knn_vote.py``) wraps
sklearn's standard KNeighborsClassifier with configurable k and weighting
-- use that one for "smoothed local-neighborhood baseline" comparisons;
this file is the dedicated leakage diagnostic.
"""
from typing import Optional

import numpy as np


LOO_EPS = 1e-9  # cosine distance below this counts as a self-match


def name() -> str:
    return "knn1_margin"


def feature_scaling_default() -> str:
    return "none"


class KNN1Margin:
    """k=1 cosine-NN classifier with cosine-margin scoring.

    sklearn-compatible ``fit`` / ``predict`` / ``predict_proba``;
    populates ``classes_`` and ``n_features_in_`` on ``fit`` for parity
    with sklearn estimators (some downstream tooling reads them).
    """

    def __init__(self, *, n_jobs: int = -1, algorithm: str = 'brute'):
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def fit(self, X, y):
        from sklearn.neighbors import NearestNeighbors

        y = np.asarray(y).astype(int)
        pos_mask = (y == 1)
        n_pos = int(pos_mask.sum())
        n_neg = int((~pos_mask).sum())
        if n_pos < 2 or n_neg < 2:
            raise ValueError(
                f"KNN1Margin needs >= 2 of each label for LOO scoring; "
                f"got n_pos={n_pos}, n_neg={n_neg}."
            )

        kw = dict(metric='cosine', algorithm=self.algorithm, n_jobs=self.n_jobs)
        # n_neighbors=2 enables on-the-fly leave-one-out for train queries.
        self.nn_pos_ = NearestNeighbors(n_neighbors=2, **kw).fit(X[pos_mask])
        self.nn_neg_ = NearestNeighbors(n_neighbors=2, **kw).fit(X[~pos_mask])
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.n_pos_train_ = n_pos
        self.n_neg_train_ = n_neg
        return self

    def _distances_pos_neg(self, X):
        """Cosine distance to the nearest positive / negative train point.

        Auto-applies leave-one-out when the 1st-nearest distance is
        below ``LOO_EPS`` (self-match): use the 2nd-nearest in that
        index instead.
        """
        d_pos, _ = self.nn_pos_.kneighbors(X, n_neighbors=2)
        d_neg, _ = self.nn_neg_.kneighbors(X, n_neighbors=2)
        d_pos_eff = np.where(d_pos[:, 0] < LOO_EPS, d_pos[:, 1], d_pos[:, 0])
        d_neg_eff = np.where(d_neg[:, 0] < LOO_EPS, d_neg[:, 1], d_neg[:, 0])
        return d_pos_eff, d_neg_eff

    def predict(self, X):
        d_pos, d_neg = self._distances_pos_neg(X)
        return (d_pos < d_neg).astype(int)

    def predict_proba(self, X):
        d_pos, d_neg = self._distances_pos_neg(X)
        sim_pos = 1.0 - d_pos          # cosine similarity in [-1, 1]
        sim_neg = 1.0 - d_neg
        margin = sim_pos - sim_neg     # in [-2, 2]
        prob_pos = np.clip((margin + 2.0) / 4.0, 0.0, 1.0).astype(np.float64)
        return np.column_stack([1.0 - prob_pos, prob_pos])


def get_estimator(config, *, random_state: Optional[int] = None) -> KNN1Margin:
    cfg = getattr(config, 'baseline_knn1_margin', None)
    cfg = dict(cfg) if cfg is not None else {}
    return KNN1Margin(
        n_jobs=int(cfg.get('n_jobs', -1)),
        algorithm=str(cfg.get('algorithm', 'brute')),
    )
