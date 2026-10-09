"""Read an experiment spec and check configs against it.

An experiment spec is a YAML file, written before any dataset is built, that states the settings
one experiment must have (see "Experiment spec" in docs/methods/glossary.md). It has two sections:

- `overrides`: an ordered list of `key=value` dotlist entries applied on top of every bundle the
  experiment uses. The builder and the trainer receive them as `--override`.
- `expect`: dotted config key -> required value, checked against each dataset's
  `resolved_config.yaml` and each planned fit config.

A key that appears in both sections must have the same value in both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.experiment_utils import file_sha256

SPEC_SECTIONS = ('overrides', 'expect')
_ABSENT = object()


def load_experiment_spec(path: Path) -> dict:
    """Read and validate an experiment spec.

    Args:
      path: the spec's YAML file.

    Returns:
      `path` (absolute, as a string), `sha256` of the file, `overrides` (list of `key=value`
      strings, possibly empty) and `expect` (dotted key -> value).

    Raises:
      ValueError: the file has a section other than `overrides` and `expect`, an override is not a
          `key=value` string, `expect` is not a mapping, or a key appears in both sections with
          different values.
    """
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = sorted(set(raw) - set(SPEC_SECTIONS))
    if unknown:
        raise ValueError(f"load_experiment_spec: {path.name} has unknown sections {unknown}; "
                         f"an experiment spec holds only {list(SPEC_SECTIONS)}.")

    overrides = raw.get('overrides') or []
    bad = [entry for entry in overrides if not isinstance(entry, str) or '=' not in entry]
    if bad:
        raise ValueError(f"load_experiment_spec: {path.name} overrides must be 'key=value' "
                         f"strings; got {bad}.")
    expect = raw.get('expect') or {}
    if not isinstance(expect, dict):
        raise ValueError(f"load_experiment_spec: {path.name} expect must map dotted config keys "
                         f"to values; got {type(expect).__name__}.")

    # Parse the overrides the way the builder and trainer will, so a value written as text in the
    # list compares against the typed value in `expect`.
    parsed = OmegaConf.from_dotlist(list(overrides))
    conflicts = []
    for entry in overrides:
        key = entry.split('=', 1)[0]
        if key in expect:
            applied = plain_value(OmegaConf.select(parsed, key))
            if applied != expect[key]:
                conflicts.append(f'{key}: overrides set {applied!r}, expect says {expect[key]!r}')
    if conflicts:
        raise ValueError(f"load_experiment_spec: {path.name} disagrees with itself: {conflicts}.")

    spec = {'path': str(path), 'sha256': file_sha256(path),
            'overrides': list(overrides), 'expect': dict(expect)}
    return spec


def plain_value(value):
    """Convert an OmegaConf list or mapping to plain Python, leaving other values unchanged.

    Args:
      value: a value selected from a config.

    Returns:
      A list or dict for an OmegaConf container, else `value` itself.
    """
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    return value


def expected_setting_failures(config, expect: dict) -> list:
    """Messages for every `expect` key whose value in `config` differs or is absent.

    Args:
      config: a composed config or a loaded `resolved_config.yaml`.
      expect: dotted key -> required value, from an experiment spec.

    Returns:
      One message per mismatched or absent key; empty when every key matches.
    """
    failures = []
    for key, required in expect.items():
        found = OmegaConf.select(config, key, default=_ABSENT)
        if found is _ABSENT:
            failures.append(f'{key}: absent from the config, expected {required!r}')
            continue
        value = plain_value(found)
        if value != required:
            failures.append(f'{key}: config has {value!r}, expected {required!r}')
    return failures


def expected_n_folds(spec: dict) -> Optional[int]:
    """The fold count an experiment spec requires.

    Args:
      spec: a spec from `load_experiment_spec`.

    Returns:
      The required `dataset.n_folds`; None when the spec requires a single split.

    Raises:
      ValueError: `expect` does not state `dataset.n_folds`.
    """
    if 'dataset.n_folds' not in spec['expect']:
        raise ValueError(f"expected_n_folds: {Path(spec['path']).name} does not state "
                         f"dataset.n_folds under expect, so the fold count cannot be checked.")
    n_folds = spec['expect']['dataset.n_folds']
    return n_folds


def split_dir_names(n_folds: Optional[int]) -> list:
    """The split directories a dataset with this fold count holds, relative to its run directory.

    Args:
      n_folds: the dataset's `dataset.n_folds`; None or 1 for a single split.

    Returns:
      `['fold_0', ..., 'fold_{n_folds - 1}']` for cross-validation, `['.']` for a single split,
      whose files sit at the run directory's root.
    """
    if n_folds is None or n_folds == 1:
        return ['.']
    names = [f'fold_{i}' for i in range(n_folds)]
    return names
