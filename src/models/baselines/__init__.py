"""Sklearn-style baselines for viral pair classification.

Each baseline module exposes:
- ``name() -> str``: short identifier used in registry + run dir prefix.
- ``get_estimator(config, *, random_state) -> sklearn-like estimator``:
  build a fresh, unfitted estimator. Reads optional per-baseline config
  from the bundle (e.g., ``config.baseline_logistic.*``).
- ``feature_scaling_default() -> str`` (optional): baseline's preferred
  default for feature_scaling when the bundle doesn't override.

The registry lives in ``train_pair_baselines.py``.
"""
