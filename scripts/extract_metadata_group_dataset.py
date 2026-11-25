#!/usr/bin/env python3
"""Extract one metadata group from a built pair dataset, keeping every split assignment.

A dataset built by `scripts/run_allpairs_datasets.py` selects its positives and assigns its folds
once per schema pair. Restricting such a dataset to a single metadata group gives a narrower
dataset whose pairs, labels and split assignments all come from that one build, so a model fitted
on it and the model fitted on the whole population can be scored on identical test pairs. See
`docs/plans/2026-10-08_paper_plan.md`.

How?

- The split directories come from the source dataset's `cv_info.json`, through `read_split_dirs`,
  so the extraction covers every fold the build wrote.
- A row belongs to the group when its slot-A host, HN subtype and year match the requested values.
  `check_axes_match` runs first and raises if any negative's two endpoints disagree on an axis,
  which is what allows one endpoint to name the row's group.
- Each split is written as `train_pairs.csv`, `val_pairs.csv` and `test_pairs.csv`, and
  `cooccurring_sequence_pairs.csv` is copied unchanged because it is the corpus-wide blocking set a
  negative redraw must avoid. Those four files are `SPLIT_FILES`, the ones a fit reads or that
  decide which pairs are negatives.
- `resolved_config.yaml` and `cv_info.json` are copied unchanged. In aligned mode a fit requires
  the first and checks its alignment directories against the ones the fit is loading, and
  `aggregate_cv_results.py` reads the master seed from the second.
- The source dataset's statistics files, such as `dataset_stats.json` and
  `split_overlap_stats.csv`, are not copied. Their counts describe the whole population and would
  be wrong beside the extracted pairs.
- The pair tables are read as text, because a filter that writes its rows back must not change any
  value. See `read_pair_table_for_rewrite`.
- Every written file is read back and compared with the rows it came from, across every column, so
  no row can change in the round trip.
- An existing output directory, a missing source file, or a group absent from a split stops the
  script before anything is written. The checksums of every source file read or copied are
  recorded, with this script's own checksum and the current commit.

CLI:
    python scripts/extract_metadata_group_dataset.py \
        --source_dataset data/datasets/flu/July_2025/runs/<run>_pb1_ha \
        --group Human H3N2 2024 \
        --output_dataset data/datasets/flu/July_2025/runs/<run>_pb1_ha_human_h3n2_2024

Outputs:
    <output_dataset>/{split}/train_pairs.csv, val_pairs.csv, test_pairs.csv  the group's rows
    <output_dataset>/{split}/cooccurring_sequence_pairs.csv                  copied unchanged
    <output_dataset>/resolved_config.yaml, cv_info.json                      copied unchanged
    <output_dataset>/extraction_info.json   source, group, counts, checksums and commit
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.metadata_group_report import (  # noqa: E402
    GROUP_AXES,
    check_axes_match,
    group_values,
)
from src.utils.dataset_manifest import (  # noqa: E402
    SPLIT_FILES,
    read_pair_table_for_rewrite,
    read_split_dirs,
)
from src.utils.experiment_utils import file_sha256, get_git_info  # noqa: E402

# The pair table of each split, by the name the counts and the printout use.
PAIR_FILES = {'train': 'train_pairs.csv', 'val': 'val_pairs.csv', 'test': 'test_pairs.csv'}
# The rest of `SPLIT_FILES`: read by a redraw rather than filtered, so they are copied as they are.
COPIED_SPLIT_FILES = tuple(name for name in SPLIT_FILES if name not in PAIR_FILES.values())
COPIED_DATASET_FILES = ('resolved_config.yaml', 'cv_info.json')


def group_rows(pairs: pd.DataFrame, group: dict, source: str) -> pd.DataFrame:
    """The rows of one pair table that belong to one metadata group.

    Args:
        pairs: a pair table carrying `label` and `<axis>_a` / `<axis>_b` for `GROUP_AXES`.
        group: axis name -> required value, as strings.
        source: the file the table came from, used in the error messages.

    Returns:
        The rows whose slot-A endpoint matches every axis, in the order they appear in `pairs`.

    Raises:
        ValueError: a column is missing, a negative's two endpoints disagree on an axis, or no row
            belongs to `group`.
    """
    check_axes_match(pairs, source)
    values = group_values(pairs)
    keep = pd.Series(True, index=pairs.index)
    for axis, required in group.items():
        keep &= values[axis] == required
    if not keep.any():
        present = sorted({tuple(row) for row in values.itertuples(index=False)})
        raise ValueError(
            f"group_rows: {source} holds no pair of the group {group}. It holds "
            f"{len(present)} groups, the first of them {present[:5]}.")
    selected = pairs[keep]
    return selected


def write_and_verify(rows: pd.DataFrame, output_path: Path) -> dict:
    """Write one split's extracted rows, then read them back and compare them with `rows`.

    Args:
        rows: the rows to write, as `group_rows` returned them.
        output_path: the CSV to write, whose directory already exists.

    Returns:
        `n_pos` and `n_neg`, the written rows of each label.

    Raises:
        AssertionError: the file does not read back with the same values, in the same order.
    """
    rows.to_csv(output_path, index=False)
    written = read_pair_table_for_rewrite(output_path)
    expected = rows.reset_index(drop=True)
    pd.testing.assert_frame_equal(written, expected, check_dtype=False, obj=str(output_path))
    counts = {'n_pos': int((rows['label'] == 1).sum()), 'n_neg': int((rows['label'] == 0).sum())}
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source_dataset', type=Path, required=True,
                        help='a dataset run directory built by run_allpairs_datasets.py')
    parser.add_argument('--group', nargs=len(GROUP_AXES), required=True,
                        metavar=tuple(axis.upper() for axis in GROUP_AXES),
                        help=f'the metadata group to keep, given as {" ".join(GROUP_AXES)}')
    parser.add_argument('--output_dataset', type=Path, required=True,
                        help='the dataset run directory to write; it must not exist')
    args = parser.parse_args()

    if args.output_dataset.exists():
        parser.error(f'Output already exists: {args.output_dataset}')
    group = {axis: str(value) for axis, value in zip(GROUP_AXES, args.group)}

    split_dirs = read_split_dirs(args.source_dataset)
    sources = [args.source_dataset / split_dir / name
               for split_dir in split_dirs for name in SPLIT_FILES]
    sources += [args.source_dataset / name for name in COPIED_DATASET_FILES]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        parser.error(f'Source dataset is missing {missing}')

    # Filter and check every split before writing anything, so a rejected source leaves no output.
    extracted = {}
    for split_dir in split_dirs:
        extracted[split_dir] = {}
        for split, name in PAIR_FILES.items():
            source_path = args.source_dataset / split_dir / name
            extracted[split_dir][split] = group_rows(read_pair_table_for_rewrite(source_path), group,
                                                     str(source_path))

    args.output_dataset.mkdir(parents=True)
    counts = {}
    for split_dir, tables in extracted.items():
        output_split = args.output_dataset / split_dir
        output_split.mkdir(parents=True, exist_ok=True)
        counts[split_dir] = {}
        for split, rows in tables.items():
            counts[split_dir][split] = write_and_verify(rows, output_split / PAIR_FILES[split])
        for name in COPIED_SPLIT_FILES:
            shutil.copy2(args.source_dataset / split_dir / name, output_split / name)
        written = counts[split_dir]
        print(f'Done. {split_dir}: ' + ', '.join(
            f'{split} {written[split]["n_pos"] + written[split]["n_neg"]:,} '
            f'({written[split]["n_pos"]:,}+/{written[split]["n_neg"]:,}-)'
            for split in PAIR_FILES))

    for name in COPIED_DATASET_FILES:
        shutil.copy2(args.source_dataset / name, args.output_dataset / name)

    info = {
        'source_dataset': str(args.source_dataset.resolve()),
        'group': group,
        'split_dirs': list(split_dirs),
        'counts': counts,
        'copied_unchanged': {'per_split': list(COPIED_SPLIT_FILES),
                             'dataset': list(COPIED_DATASET_FILES)},
        'source_sha256': {str(path.relative_to(args.source_dataset)): file_sha256(path)
                          for path in sources},
        # `get_git_info` ignores untracked files, so the commit alone does not identify this
        # script while it is untracked. Its checksum does.
        'script_sha256': file_sha256(Path(__file__).resolve()),
        'code': get_git_info(),
    }
    info_path = args.output_dataset / 'extraction_info.json'
    info_path.write_text(json.dumps(info, indent=2))
    print(f'Done. Extracted {group} from {args.source_dataset} to {args.output_dataset}')


if __name__ == '__main__':
    main()
