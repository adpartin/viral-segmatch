"""1-nearest-neighbor baseline for pair classification (Plan Exp 2).

The lookup floor that ``docs/methods/leakage.md`` compares a trained model against. It reads a
model that scores no better than a 1-NN classifier, on the same dataset and the same features, as
doing near-neighbor lookup rather than generalizing, and sets the bar at a gap under 0.02 AUC.
Treat that as a descriptive comparison, not a test of mechanism: similar scores say the two rank
these rows similarly, and establish nothing about what the trained model computes. The bar itself
was set for an MLP against a cosine 1-NN in one feature space, so it does not carry over to a
different metric unexamined. Plug-in baseline for the ``train_pair_baselines.py`` harness -- same
Stage 3 dataset, same feature loader, same pair-level metrics as the other baselines.

Hard prediction: label of the single nearest train pair (k=1) under the
configured distance.

Continuous score (for AUC): a *margin*,
``score = (distance to the nearest train negative)
        - (distance to the nearest train positive)``,
affinely mapped to ``predict_proba`` by ``(margin + 2) / 4``. This
preserves a continuous ranking (sklearn's
``KNeighborsClassifier.predict_proba`` would return only {0, 1} at
k=1 and degenerate the AUC). The hard 1-NN decision and the
``predict_proba > 0.5`` decision coincide, so any val-driven
threshold optimization in the harness still snaps back to the 1-NN
boundary at the default 0.5. AUC is rank-invariant under monotonic
transforms, so the affine mapping does not change AUC values.

Two metrics are supported, and they take different decision paths.

- ``cosine`` (default) suits the k-mer and ESM-2 feature spaces. The margin is
  ``(1 - d_pos) - (1 - d_neg)``, which lies in [-2, 2], so probabilities span [0, 1].
- ``hamming`` suits per-site features, whose ordinal codes are labels rather than magnitudes.
  Hamming distance only tests whether two codes are equal, so it reads them the way
  ``train_pair_baselines.py`` already reads them when it declares every ordinal site column
  categorical. sklearn returns Hamming as a fraction that drifts off the ``k / n_features``
  lattice, and the positive and negative neighbors come from two separately fitted searches, so
  two equal site counts can arrive as unequal floats. Every ``hamming`` comparison therefore runs
  on the integer count ``round(distance * n_features_in_)``. A count tie then gives a margin of
  exactly 0, hence a probability of exactly 0.5, which ``_pair_metrics.py``'s strict
  ``y_probs > threshold`` assigns to class 0. The margin lies in [-1, 1], so probabilities span
  [0.25, 0.75]; AUC is unaffected, being rank-invariant.

Leave-one-out is applied by training row index, never by distance. A query that *is* a training
row must not match itself, but another training row at distance 0 is a legitimate neighbor, and a
validation or test row at distance 0 is an exact train match -- the thing a leakage audit exists to
surface. Callers pass ``train_rows`` only when scoring the training split; ``predict`` and
``predict_proba`` exclude nothing by default.

Bundle overrides under ``config.baseline_knn1_margin.*`` (defaults shown):

| key             | default | notes                                           |
|-----------------|---------|-------------------------------------------------|
| metric          | cosine  | 'cosine' | 'hamming' (per-site ordinal codes)   |
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

METRICS = ('cosine', 'hamming')


def name() -> str:
    return "knn1_margin"


def feature_scaling_default() -> str:
    return "none"


class KNN1Margin:
    """k=1 nearest-neighbor classifier with distance-margin scoring.

    sklearn-compatible ``fit`` / ``predict`` / ``predict_proba``;
    populates ``classes_`` and ``n_features_in_`` on ``fit`` for parity
    with sklearn estimators (some downstream tooling reads them).
    """

    def __init__(self, *, n_jobs: int = -1, algorithm: str = 'brute', metric: str = 'cosine'):
        if metric not in METRICS:
            raise ValueError(f"KNN1Margin metric must be one of {list(METRICS)}; got {metric!r}.")
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.metric = metric

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

        kw = dict(metric=self.metric, algorithm=self.algorithm, n_jobs=self.n_jobs)
        # n_neighbors=2 so a training-row query can drop its own match and still have a neighbor.
        self.nn_pos_ = NearestNeighbors(n_neighbors=2, **kw).fit(X[pos_mask])
        self.nn_neg_ = NearestNeighbors(n_neighbors=2, **kw).fit(X[~pos_mask])
        # Subset row -> training row. Each search is fitted on one class's rows, so its returned
        # indices mean nothing until mapped back through these.
        self.pos_rows_ = np.flatnonzero(pos_mask)
        self.neg_rows_ = np.flatnonzero(~pos_mask)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.n_pos_train_ = n_pos
        self.n_neg_train_ = n_neg
        return self

    def _counts(self, distances):
        """Integer count of disagreeing features behind a Hamming fraction.

        Args:
          distances: Hamming distances, each a fraction of `n_features_in_`.

        Returns:
          The same values as integer counts, rounded onto the exact lattice.
        """
        counts = np.rint(distances * self.n_features_in_)
        return counts

    def _distances_pos_neg(self, X, train_rows=None):
        """Distance to the nearest positive and the nearest negative training row.

        Args:
          X: query rows.
          train_rows: for each query row, its own index into the training matrix, when the queries
              ARE training rows. That row is then skipped so a query cannot match itself. Pass None
              for validation and test, where every match is real and must be kept.

        Returns:
          Two arrays, the distance to the nearest positive and to the nearest negative, one entry
          per query row.
        """
        d_pos, i_pos = self.nn_pos_.kneighbors(X, n_neighbors=2)
        d_neg, i_neg = self.nn_neg_.kneighbors(X, n_neighbors=2)
        if train_rows is None:
            return d_pos[:, 0], d_neg[:, 0]

        # A training row sits in exactly one of the two searches, so at most one side can be a
        # self-match. Identity is decided on the row index, not on the distance: a different
        # training row at distance 0 stays eligible.
        own = np.asarray(train_rows)
        self_pos = self.pos_rows_[i_pos[:, 0]] == own
        self_neg = self.neg_rows_[i_neg[:, 0]] == own
        d_pos_eff = np.where(self_pos, d_pos[:, 1], d_pos[:, 0])
        d_neg_eff = np.where(self_neg, d_neg[:, 1], d_neg[:, 0])
        return d_pos_eff, d_neg_eff

    def margin(self, X, train_rows=None):
        """Signed score, positive when the nearest training positive is the closer of the two.

        Under `hamming` the two distances are compared as integer site counts, so a tie scores
        exactly 0. Under `cosine` the score is the similarity difference, unchanged.

        Args:
          X: query rows.
          train_rows: see `_distances_pos_neg`.

        Returns:
          One score per query row, in [-1, 1] under `hamming` and [-2, 2] under `cosine`.
        """
        d_pos, d_neg = self._distances_pos_neg(X, train_rows)
        if self.metric == 'hamming':
            k_pos, k_neg = self._counts(d_pos), self._counts(d_neg)
            scores = (k_neg - k_pos) / self.n_features_in_
            return scores
        sim_pos = 1.0 - d_pos          # cosine similarity in [-1, 1]
        sim_neg = 1.0 - d_neg
        scores = sim_pos - sim_neg     # in [-2, 2]
        return scores

    def predict(self, X, train_rows=None):
        d_pos, d_neg = self._distances_pos_neg(X, train_rows)
        if self.metric == 'hamming':
            k_pos, k_neg = self._counts(d_pos), self._counts(d_neg)
            labels = (k_pos < k_neg).astype(int)  # 0 on a tie and on a nearer negative
            return labels
        labels = (d_pos < d_neg).astype(int)
        return labels

    def predict_proba(self, X, train_rows=None):
        scores = self.margin(X, train_rows)
        prob_pos = np.clip((scores + 2.0) / 4.0, 0.0, 1.0).astype(np.float64)
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def co_nearest(self, X, X_train, y_train, train_rows=None):
        """Every training row tied at the minimum distance from each query row.

        Needs the training matrix rather than a stored copy, so the fitted estimator stays small
        on disk. Computes the full query-by-train distance block, which is why it is called once
        per split for an audit rather than on the prediction path.

        Args:
          X: query rows.
          X_train: the matrix `fit` was given, in the same row order.
          y_train: its labels, used for the per-class counts.
          train_rows: see `_distances_pos_neg`.

        Returns:
          minimum: the minimum distance per query row, as an integer site count under `hamming`
              and as the metric's own value under `cosine`.
          members: one array of training row indices per query row, holding every row at that
              minimum.
          n_pos: how many of each `members` array are positive.
          n_neg: how many are negative.
        """
        from sklearn.metrics import pairwise_distances

        block = pairwise_distances(X, X_train, metric=self.metric, n_jobs=self.n_jobs)
        if self.metric == 'hamming':
            block = self._counts(block)  # integer counts, so a tie is exact rather than near
        if train_rows is not None:
            block[np.arange(len(block)), np.asarray(train_rows)] = np.inf
        minimum = block.min(axis=1)
        members = [np.flatnonzero(row == row_min) for row, row_min in zip(block, minimum)]

        labels = np.asarray(y_train).astype(int)
        n_pos = np.array([int(labels[m].sum()) for m in members])
        n_neg = np.array([len(m) for m in members]) - n_pos
        return minimum, members, n_pos, n_neg


def get_estimator(config, *, random_state: Optional[int] = None) -> KNN1Margin:
    cfg = getattr(config, 'baseline_knn1_margin', None)
    cfg = dict(cfg) if cfg is not None else {}
    return KNN1Margin(
        n_jobs=int(cfg.get('n_jobs', -1)),
        algorithm=str(cfg.get('algorithm', 'brute')),
        metric=str(cfg.get('metric', 'cosine')),
    )
