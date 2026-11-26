#!/usr/bin/env python3
"""Write a dataset manifest for the metadata-group datasets extracted from one manifest's datasets.

`scripts/extract_metadata_group_dataset.py` extracts one metadata group from one pair dataset at a
time and writes `extraction_info.json` beside it. `scripts/run_allpairs_baselines.py` trains from
a dataset manifest. This script lists the extracted datasets of every pair in a source manifest
in a manifest of the same form, so the runner can train them.

How?

- The extracted dataset of a source dataset is the directory `<source dataset_dir>_<group
  suffix>`, where the suffix joins the group's values in lower case, e.g. `human_h3n2_2024`.
- Each extraction must name that source dataset, the requested group and the source's split
  directories. Every file the source manifest records for that dataset must have the checksum
  the extraction recorded when it read the file, so the extraction read the files the source
  manifest describes.
- The site-feature cache files must still have the checksums the source manifest records. The
  extracted datasets' fits read the same caches.
- The new manifest copies the source manifest's experiment spec, overrides, bundle prefix and fold
  count, and records the current commit as its own. The runner requires that commit at training
  time. The source manifest's path and checksum, and each extraction's commit, are recorded too.
- Any failed check stops the script before anything is written, and an existing output manifest
  stops it as well.

CLI:
    python scripts/write_extraction_manifest.py \
        --source_manifest data/datasets/flu/July_2025/runs/<prefix>_manifest.json \
        --group Human H3N2 2024

Outputs:
    data/datasets/flu/July_2025/runs/<source prefix><group suffix>_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.metadata_group_report import GROUP_AXES  # noqa: E402
from src.utils.dataset_manifest import (  # noqa: E402
    changed_files,
    dataset_file_checksums,
    load_dataset_manifest,
)
from src.utils.experiment_utils import get_git_info  # noqa: E402

# Keys of the source manifest copied unchanged into the new one.
COPIED_MANIFEST_KEYS = ('experiment_spec', 'overrides', 'bundle_prefix', 'n_folds', 'cache_files')


def group_suffix(group: dict) -> str:
    """The directory-name suffix of an extracted metadata group.

    Args:
      group: axis name -> value, for the axes in `GROUP_AXES`.

    Returns:
      The values in `GROUP_AXES` order, lower case, joined by `_`, e.g. `human_h3n2_2024`.
    """
    suffix = '_'.join(str(group[axis]).lower() for axis in GROUP_AXES)
    return suffix


def extraction_failures(source_entry: dict, extracted_dir: Path, group: dict) -> list:
    """Messages for every way an extracted dataset fails to match its source manifest entry.

    Args:
      source_entry: the source dataset's entry in the source manifest's `datasets` section.
      extracted_dir: the directory the extraction of that dataset was written to.
      group: axis name -> value as strings, the group that was extracted.

    Returns:
      One message per failed check; empty when the extraction names the source dataset, the
      group and the split directories, and recorded the checksums the source manifest records.
    """
    info_path = extracted_dir / 'extraction_info.json'
    if not info_path.exists():
        return [f'{info_path}: missing']
    info = json.loads(info_path.read_text())
    failures = []
    source_dir = Path(source_entry['dataset_dir']).resolve()
    if Path(info['source_dataset']).resolve() != source_dir:
        failures.append(f"{extracted_dir.name}: extracted from {info['source_dataset']}, "
                        f"not {source_dir}")
    if info['group'] != group:
        failures.append(f"{extracted_dir.name}: extracted group {info['group']}, not {group}")
    if info['split_dirs'] != source_entry['split_dirs']:
        failures.append(f"{extracted_dir.name}: split directories {info['split_dirs']}, the "
                        f"source has {source_entry['split_dirs']}")
    # The extraction recorded each source file it read; those must be the files the manifest records.
    differing = sorted(key for key, checksum in source_entry['files'].items()
                       if info['source_sha256'].get(key) != checksum)
    if differing:
        failures.append(f'{extracted_dir.name}: the extraction read {len(differing)} source files '
                        f'with checksums other than the source manifest records, e.g. '
                        f'{differing[:3]}')
    return failures


def extraction_manifest(source_manifest: dict, group: dict) -> tuple[dict, list]:
    """The manifest of the extracted datasets of every dataset in a source manifest.

    Args:
      source_manifest: from `load_dataset_manifest`.
      group: axis name -> value as strings, the group that was extracted.

    Returns:
      The manifest and the failure messages. The manifest is complete only when the list is empty.
    """
    suffix = group_suffix(group)
    failures = changed_files(source_manifest['cache_files'])
    datasets = {}
    for bundle, source_entry in source_manifest['datasets'].items():
        extracted_dir = Path(f"{source_entry['dataset_dir']}_{suffix}")
        if not extracted_dir.is_dir():
            failures.append(f'{extracted_dir}: missing')
            continue
        entry_failures = extraction_failures(source_entry, extracted_dir, group)
        failures.extend(entry_failures)
        if entry_failures:
            continue
        info = json.loads((extracted_dir / 'extraction_info.json').read_text())
        datasets[bundle] = {
            'pair': source_entry['pair'],
            'dataset_dir': str(extracted_dir),
            'n_folds': source_entry['n_folds'],
            'split_dirs': source_entry['split_dirs'],
            'files': dataset_file_checksums(extracted_dir, source_entry['split_dirs']),
            'extraction_commit': info['code']['commit'],
        }

    manifest = {
        'dataset_prefix': f"{source_manifest['dataset_prefix']}{suffix}_",
        'source_manifest': {'path': source_manifest['manifest_path'],
                            'sha256': source_manifest['manifest_sha256']},
        'group': group,
        **{key: source_manifest[key] for key in COPIED_MANIFEST_KEYS},
        'code': get_git_info(),
        'datasets': datasets,
    }
    return manifest, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source_manifest', type=Path, required=True,
                        help='the {prefix}_manifest.json the datasets were extracted from')
    parser.add_argument('--group', nargs=len(GROUP_AXES), required=True,
                        metavar=tuple(axis.upper() for axis in GROUP_AXES),
                        help=f'the extracted metadata group, given as {" ".join(GROUP_AXES)}')
    args = parser.parse_args()

    source_manifest = load_dataset_manifest(args.source_manifest)
    group = {axis: str(value) for axis, value in zip(GROUP_AXES, args.group)}
    manifest, failures = extraction_manifest(source_manifest, group)
    if failures:
        print('\n'.join(f'ERROR: {message}' for message in failures))
        sys.exit(1)

    output_path = (Path(source_manifest['manifest_path']).parent
                   / f"{manifest['dataset_prefix'].rstrip('_')}_manifest.json")
    if output_path.exists():
        print(f'ERROR: {output_path} already exists')
        sys.exit(1)
    output_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Done. {len(manifest['datasets'])} extracted datasets at commit "
          f"{manifest['code']['commit_short']} ({manifest['code']['status']}): {output_path}")


if __name__ == '__main__':
    main()
