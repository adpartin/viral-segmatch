"""Logistic regression baseline for pair classification.

Reads optional bundle overrides from ``config.baseline_logistic.*``:
- ``C``: inverse regularization strength (default 1.0).
- ``penalty``: 'l2' (default), 'l1', 'elasticnet', or 'none'.
- ``solver``: default 'lbfgs'. Use 'saga' for l1/elasticnet.
- ``max_iter``: default 1000.
- ``class_weight``: e.g., 'balanced'. Default None (val imbalance is intentional).
- ``n_jobs``: default -1 (all cores).
- ``feature_scaling``: 'standard' (default) or 'none'. LR coefficients are
  scale-sensitive; the default fits a StandardScaler on the train split.
"""
from typing import Optional

from sklearn.linear_model import LogisticRegression


def name() -> str:
    return "logistic"


def feature_scaling_default() -> str:
    return "standard"


def get_estimator(config, *, random_state: Optional[int] = None) -> LogisticRegression:
    cfg = getattr(config, 'baseline_logistic', None)
    cfg = dict(cfg) if cfg is not None else {}

    return LogisticRegression(
        C=float(cfg.get('C', 1.0)),
        penalty=cfg.get('penalty', 'l2'),
        solver=cfg.get('solver', 'lbfgs'),
        max_iter=int(cfg.get('max_iter', 1000)),
        class_weight=cfg.get('class_weight', None),
        n_jobs=int(cfg.get('n_jobs', -1)),
        random_state=random_state,
    )
