"""Unit tests for the 1-NN baseline's leave-one-out and tie handling.

Both behaviours are invisible on the production data. On `exp3_28p_codon_pb2_ha/fold_0` no
validation or test row sits at Hamming distance 0 from train, and no training row has a distance-0
neighbour in the opposite class, so a run there exercises neither the self-match path nor the
cross-class duplicate it must not hide. These fixtures put identical vectors where the real data
has none.

Leave-one-out is decided by training row index, not by distance. The fixture is built so that the
distance rule the module used to apply -- skip any neighbour closer than an epsilon -- gives a
different answer from the row-index rule on three of the tests below, which is what makes them
worth running.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.models.baselines.knn1_margin import KNN1Margin  # noqa: E402

# Row 1 duplicates row 0 within the positive class. Row 5 duplicates it across classes. Row 2 is a
# unique positive, so a test copy of it has exactly one distance-0 train match and nothing behind
# it -- the arrangement that separates the two leave-one-out rules.
X_TRAIN = np.array([
    [0, 0, 0, 0],   # 0  positive
    [0, 0, 0, 0],   # 1  positive, identical to row 0
    [1, 1, 0, 0],   # 2  positive, unique
    [1, 1, 1, 1],   # 3  negative
    [1, 1, 1, 0],   # 4  negative
    [0, 0, 0, 0],   # 5  negative, identical to rows 0 and 1
], dtype=np.float32)
Y_TRAIN = np.array([1, 1, 1, 0, 0, 0])
TRAIN_ROWS = np.arange(len(X_TRAIN))
LOO_EPS = 1e-9   # the epsilon the module used to compare distances against


def _fitted(metric: str = 'hamming') -> KNN1Margin:
    return KNN1Margin(metric=metric, n_jobs=1).fit(X_TRAIN, Y_TRAIN)


def _distance_rule(model, X):
    """What the retired distance-based rule would have returned, for contrast."""
    d_pos, _ = model.nn_pos_.kneighbors(X, n_neighbors=2)
    d_neg, _ = model.nn_neg_.kneighbors(X, n_neighbors=2)
    return (np.where(d_pos[:, 0] < LOO_EPS, d_pos[:, 1], d_pos[:, 0]),
            np.where(d_neg[:, 0] < LOO_EPS, d_neg[:, 1], d_neg[:, 0]))


def test_loo_keeps_a_duplicate_training_row():
    """Row 0's nearest positive, excluding itself, is row 1 at distance 0."""
    model = _fitted()
    d_pos, _ = model._distances_pos_neg(X_TRAIN, train_rows=TRAIN_ROWS)
    assert d_pos[0] == pytest.approx(0.0), 'the duplicate positive was skipped along with the self'


def test_loo_does_not_hide_a_cross_class_duplicate():
    """Row 0 also has a distance-0 negative, row 5, which the audit must see.

    The opposite-class search can never hold the query itself, so nothing there is a self-match and
    nothing there may be skipped. The distance rule skipped it and reported a far neighbour.
    """
    model = _fitted()
    _, d_neg = model._distances_pos_neg(X_TRAIN, train_rows=TRAIN_ROWS)
    assert d_neg[0] == pytest.approx(0.0), 'the cross-class duplicate was skipped'
    assert _distance_rule(model, X_TRAIN)[1][0] > 0.0, 'fixture must separate the two rules'

    # Equal counts on both sides, so the margin is exactly 0 and the tie rule assigns class 0.
    assert model.margin(X_TRAIN, train_rows=TRAIN_ROWS)[0] == 0.0
    assert model.predict(X_TRAIN, train_rows=TRAIN_ROWS)[0] == 0
    assert model.predict_proba(X_TRAIN, train_rows=TRAIN_ROWS)[0, 1] == 0.5


def test_an_exact_test_match_is_never_excluded():
    """A test row identical to training row 2 reports distance 0.

    This is the case a leakage audit exists to surface, so scoring without `train_rows` must keep
    it. Row 2 is unique, so the distance rule skipped its only match and reported 0.5.
    """
    model = _fitted()
    X_test = X_TRAIN[[2]].copy()
    d_pos, _ = model._distances_pos_neg(X_test)
    assert d_pos[0] == pytest.approx(0.0)
    assert _distance_rule(model, X_test)[0][0] > 0.0, 'fixture must separate the two rules'


def test_co_nearest_enumerates_every_tied_row_and_its_classes():
    """The co-nearest set holds all three distance-0 rows for a test copy of row 0.

    Two are positive and one negative, so the set straddles both classes -- which is what makes a
    representative's label differ from the prediction.
    """
    model = _fitted()
    minimum, members, n_pos, n_neg = model.co_nearest(X_TRAIN[[0]].copy(), X_TRAIN, Y_TRAIN)
    assert minimum[0] == 0
    assert sorted(members[0]) == [0, 1, 5]
    assert (n_pos[0], n_neg[0]) == (2, 1)


def test_co_nearest_skips_only_the_query_row_on_the_train_split():
    """Scoring train row 0, its own index is excluded and its two duplicates are not."""
    model = _fitted()
    _, members, _, _ = model.co_nearest(X_TRAIN, X_TRAIN, Y_TRAIN, train_rows=TRAIN_ROWS)
    assert sorted(members[0]) == [1, 5]


def test_hamming_decides_on_counts_not_floats():
    """Equal mismatch counts tie even when sklearn's fractions differ in the last bits.

    The two distances come from separately fitted searches, so the same count can arrive as unequal
    floats. Rounding to counts is what makes the tie exact.
    """
    model = _fitted()
    counts = model._counts(np.array([2.0 / 4 + 1e-12, 2.0 / 4 - 1e-12]))
    assert counts.tolist() == [2.0, 2.0]


def test_cosine_still_scores_and_an_unknown_metric_raises():
    """Cosine remains usable, and a metric the class cannot honour fails at construction."""
    model = _fitted(metric='cosine')
    assert model.predict(X_TRAIN[[3]]).shape == (1,)
    with pytest.raises(ValueError, match='metric must be one of'):
        KNN1Margin(metric='euclidean')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
