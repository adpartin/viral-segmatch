"""LightGBM baseline for pair classification.

Tree-based gradient boosting; insensitive to feature scale, so the default
``feature_scaling`` is ``'none'``. The module is named ``lgbm`` (not
``lightgbm``) so it doesn't shadow the installed pip package on
``import lightgbm``.

Bundle overrides under ``config.baseline_lgbm.*`` (defaults shown).
LightGBM uses two parallel naming schemes — the sklearn-style names below
and the native LightGBM aliases. We use the sklearn names; aliases are
listed for cross-reference if you read LightGBM docs / params.

| sklearn name           | LightGBM alias        | default | notes                                  |
|------------------------|-----------------------|---------|----------------------------------------|
| n_estimators           | num_iterations        | 1000    | cap; early stopping usually halts much earlier. |
| learning_rate          | eta                   | 0.05    |                                        |
| num_leaves             | num_leaf              | 31      | max leaves per tree.                   |
| min_child_samples      | min_data_in_leaf      | 20      | min samples per leaf; main             |
|                        |                       |         | overfitting regularizer for leaf-wise growth. |
| colsample_bytree       | feature_fraction      | 0.25    | feature subsample per tree;            |
|                        |                       |         | cheap regularizer for 8192-dim concat. |
| reg_alpha              | lambda_l1             | 0.1     |                                        |
| reg_lambda             | lambda_l2             | 0.1     |                                        |
| n_jobs                 | num_threads           | -1      |                                        |
| (fit kwarg) early_stopping_rounds | (callback) | 50      | val metric plateau window.             |

Early stopping uses the val split's AUC as the eval metric (matches the
MLP path's val-driven model selection).
"""
from typing import Optional

# LightGBM is imported lazily inside callables so the registry can list
# this baseline even when the package isn't installed in the env.


def name() -> str:
    return "lgbm"


def feature_scaling_default() -> str:
    return "none"


def get_estimator(config, *, random_state: Optional[int] = None):
    from lightgbm import LGBMClassifier
    cfg = getattr(config, 'baseline_lgbm', None)
    cfg = dict(cfg) if cfg is not None else {}
    return LGBMClassifier(
        n_estimators=int(cfg.get('n_estimators', 1000)),
        learning_rate=float(cfg.get('learning_rate', 0.05)),
        num_leaves=int(cfg.get('num_leaves', 31)),
        min_child_samples=int(cfg.get('min_child_samples', 20)),
        colsample_bytree=float(cfg.get('colsample_bytree', 0.25)),
        reg_alpha=float(cfg.get('reg_alpha', 0.1)),
        reg_lambda=float(cfg.get('reg_lambda', 0.1)),
        objective='binary',
        n_jobs=int(cfg.get('n_jobs', -1)),
        random_state=random_state,
        verbose=-1,
    )


def fit(estimator, X_train, y_train, *, X_val=None, y_val=None, config=None) -> None:
    """Fit LightGBM with val-driven early stopping.

    Reads ``early_stopping_rounds`` from ``config.baseline_lgbm.early_stopping_rounds``
    (default 50). Uses AUC on the val split as the eval metric.
    Falls back to plain ``fit(X_train, y_train)`` when no val set is given.
    """
    import lightgbm as lgb
    cfg = getattr(config, 'baseline_lgbm', None) if config is not None else None
    cfg = dict(cfg) if cfg is not None else {}
    es_rounds = int(cfg.get('early_stopping_rounds', 50))

    if X_val is not None and y_val is not None:
        estimator.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=es_rounds, verbose=True)],
        )
    else:
        estimator.fit(X_train, y_train)
