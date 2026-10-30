"""Write and check the dataset manifest of a set of pair datasets built together.

`scripts/run_allpairs_datasets.py` builds one dataset per schema pair under one dataset prefix and
writes `{prefix}_manifest.json` next to them. The manifest records what the audit, the training
runner and the balanced-redraw coordinator need to confirm that they read the datasets that were
built, with the settings they were built with:

- the experiment spec (path and checksum), the capacity CSV, the override list, the fold count and
  the Git commit of the build;
- per dataset: its pair, its split directories, and checksums of its `resolved_config.yaml` and of
  each split's pair files;
- checksums of the site-feature cache files the datasets' fits will read.

The manifest is written once and never changed afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.utils.config_hydra import get_function_short_name_map
from src.utils.experiment_spec import expected_n_folds, split_dir_names
from src.utils.experiment_utils import file_sha256
from src.utils.path_utils import build_embeddings_paths
from src.utils.site_utils import site_cache_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The files each split directory holds that a fit reads, or that decide which pairs are negatives.
SPLIT_FILES = ('train_pairs.csv', 'val_pairs.csv', 'test_pairs.csv',
               'cooccurring_sequence_pairs.csv')


def read_split_dirs(dataset_dir: Path) -> list:
    """The split directories a built dataset holds, relative to its run directory.

    Args:
      dataset_dir: a dataset run directory.

    Returns:
      `cv_info.json`'s `fold_dirs` for a cross-validation dataset; `['.']` for a single split,
      whose files sit at the run directory's root.

    Raises:
      ValueError: the directory has neither `cv_info.json` nor a root `train_pairs.csv`.
    """
    cv_info_path = dataset_dir / 'cv_info.json'
    if cv_info_path.exists():
        split_dirs = list(json.loads(cv_info_path.read_text())['fold_dirs'])
        return split_dirs
    if (dataset_dir / 'train_pairs.csv').exists():
        return ['.']
    raise ValueError(f"read_split_dirs: {dataset_dir} has neither cv_info.json nor "
                     f"train_pairs.csv; was the build interrupted?")


def split_file_key(split_dir: str, name: str) -> str:
    """The manifest key of one split file: its path relative to the dataset run directory.

    Args:
      split_dir: `fold_k` or `.`.
      name: a file name from `SPLIT_FILES`.

    Returns:
      For example `fold_0/train_pairs.csv`, or `train_pairs.csv` for a single split.
    """
    key = (Path(split_dir) / name).as_posix()
    return key


def dataset_file_checksums(dataset_dir: Path, split_dirs: list) -> dict:
    """Checksums of a dataset's `resolved_config.yaml` and of every split's `SPLIT_FILES`.

    Args:
      dataset_dir: a dataset run directory.
      split_dirs: its split directories, from `read_split_dirs`.

    Returns:
      Path relative to `dataset_dir` -> SHA-256.
    """
    keys = ['resolved_config.yaml']
    keys += [split_file_key(split_dir, name) for split_dir in split_dirs for name in SPLIT_FILES]
    checksums = {key: file_sha256(dataset_dir / key) for key in keys}
    return checksums


def site_cache_files(config) -> list:
    """The site-feature cache files a dataset's fits read, or none when they use other features.

    Args:
      config: the dataset's composed or resolved config.

    Returns:
      Absolute paths of the `.npz`, `_index.parquet` and `_metadata.json` files of both slots'
      proteins at the config's `site.unit`; empty unless `training.feature_source` is `site`.
    """
    if str(config.training.get('feature_source')) != 'site':
        return []
    cache_dir = build_embeddings_paths(
        project_root=PROJECT_ROOT, virus_name=config.virus.virus_name,
        data_version=config.virus.data_version, run_suffix='', config=config)['output_dir']
    short_of = get_function_short_name_map(config)
    pins = {str(protein): int(pin['nt']) for protein, pin in dict(config.virus.cds_length).items()}
    paths = []
    for function in config.dataset.schema_pair:
        protein = short_of[function]
        cache = site_cache_paths(cache_dir, str(config.site.unit), protein, pins[protein])
        paths.extend(path.resolve() for path in cache.values())
    return paths


def pair_cache_checksums(manifest: dict, config) -> dict:
    """The manifest's checksums of the cache files one pair's fits read.

    Args:
      manifest: from `load_dataset_manifest`.
      config: the pair's planned or resolved config.

    Returns:
      Absolute cache path -> SHA-256, for the two proteins of the config's schema pair.

    Raises:
      ValueError: the manifest does not record one of those files.
    """
    paths = [str(path) for path in site_cache_files(config)]
    unrecorded = [path for path in paths if path not in manifest['cache_files']]
    if unrecorded:
        raise ValueError(f'pair_cache_checksums: the dataset manifest records no checksum for '
                         f'{unrecorded}')
    checksums = {path: manifest['cache_files'][path] for path in paths}
    return checksums


def fit_input_checksums(entry: dict, split_dir: str, cache_checksums: dict) -> dict:
    """The input checksums a fit of one split of a manifest dataset must record.

    Args:
      entry: the dataset's entry in the manifest's `datasets` section.
      split_dir: one of its split directories.
      cache_checksums: from `pair_cache_checksums`.

    Returns:
      Absolute path -> SHA-256 for the split's `SPLIT_FILES`, the dataset's `resolved_config.yaml`
      and the cache files: the form `train_pair_baselines.input_checksums` records.
    """
    dataset_dir = Path(entry['dataset_dir'])
    keys = ['resolved_config.yaml'] + [split_file_key(split_dir, name) for name in SPLIT_FILES]
    expected = {str((dataset_dir / key).resolve()): entry['files'][key] for key in keys}
    expected.update(cache_checksums)
    return expected


def changed_files(recorded: dict, base: Optional[Path] = None) -> list:
    """Messages for recorded files that are now missing or have a different checksum.

    Args:
      recorded: path -> SHA-256, as a manifest stores them.
      base: directory the recorded paths are relative to; None when they are absolute.

    Returns:
      One message per missing or changed file; empty when every file matches.
    """
    problems = []
    for key, checksum in recorded.items():
        path = Path(base) / key if base is not None else Path(key)
        if not path.exists():
            problems.append(f'{path}: missing')
        elif file_sha256(path) != checksum:
            problems.append(f'{path}: changed since the manifest was written')
    return problems


def load_dataset_manifest(path: Path) -> dict:
    """Read a dataset manifest written by `scripts/run_allpairs_datasets.py`.

    Args:
      path: the `{prefix}_manifest.json` file.

    Returns:
      The manifest, with its own path and checksum added under `manifest_path` and
      `manifest_sha256`.

    Raises:
      ValueError: the manifest has no `datasets` section, so it was written before manifests
          recorded split directories and checksums.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if 'datasets' not in manifest:
        raise ValueError(f"load_dataset_manifest: {path.name} has no 'datasets' section; it was "
                         f"written before the manifest recorded split directories and checksums.")
    manifest['manifest_path'] = str(path)
    manifest['manifest_sha256'] = file_sha256(path)
    return manifest


def fold_count_failures(manifest: dict, spec: dict) -> list:
    """Messages for every disagreement between the spec's fold count and the manifest's datasets.

    Args:
      manifest: from `load_dataset_manifest`.
      spec: from `src.utils.experiment_spec.load_experiment_spec`.

    Returns:
      One message per disagreement; empty when the manifest's fold count equals the spec's and
      every dataset holds exactly the split directories that count implies.
    """
    required = expected_n_folds(spec)
    failures = []
    if manifest.get('n_folds') != required:
        failures.append(f"the manifest's datasets were built with n_folds={manifest.get('n_folds')}, "
                        f"the experiment spec requires {required}")
    expected_dirs = split_dir_names(required)
    for bundle, entry in manifest['datasets'].items():
        if entry['split_dirs'] != expected_dirs:
            failures.append(f"{bundle}: split directories {entry['split_dirs']}, "
                            f"expected {expected_dirs}")
    return failures


def pair_token(pair_label: str) -> str:
    """The pair token used in bundle and directory names, from a pair label.

    Args:
      pair_label: `{slot A}-{slot B}` short names, e.g. `PB2-HA`.

    Returns:
      The lower-case token, e.g. `pb2_ha`.
    """
    token = pair_label.lower().replace('-', '_')
    return token
