"""Random-split control arm for an existing k-fold dataset: same rows, different partition.

Takes a built dataset (`fold_k/{train,val,test}_pairs.csv`) and, for each fold independently,
pools that fold's three splits, shuffles the rows, and re-cuts them at the SAME three sizes.
The result is a size-matched in-distribution control: every arm-pair holds exactly the same
rows, so a score difference between the arms is attributable to the partition alone.

Aimed at the 2D-CD builder's output (`dataset_pairs_cc.py`), where the source partition keeps
whole atoms together and this one does not, but it assumes nothing beyond the fold layout.

Three properties make the re-cut safe, all asserted per fold rather than trusted:
  - Within one fold, train/val/test partition the rows with no repeated `pair_key`, so a
    permutation cannot put the same pair in two splits.
  - The output split sizes equal the input's, so the arms differ only in fold membership.
  - Each fold is read back from disk after writing and checked against its source, because a
    write truncated by a full filesystem reads as a valid shorter file.

Negatives are MOVED, not redrawn: the source drew them per split under its own partition
(`negative_scope: within_fold`), so the control inherits negatives shaped by the source's
split. That is what makes the row sets identical, and it means the arm answers "is the source
partition harder on this data", not "what would a random-split pipeline produce".

Each fold is re-cut independently, so the folds' test sets may overlap -- this is a set of k
paired controls, not a k-fold cross-validation.

Rows are read with `dtype=str`: contig ids such as `11320.652550` parse as floats and lose
their trailing zero on write, which would corrupt `ctg_a`/`ctg_b` (the nt_ctg k-mer join keys).
Reading every column as text makes the copy byte-exact.

CLI:
    python src/datasets/build_random_arm.py \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_cc_nt_cds_cm0_t099 \\
        --out_dir     data/datasets/flu/July_2025/runs/dataset_cc_nt_cds_cm0_t099_random \\
        [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.experiment_utils import get_git_info  # noqa: E402

# Written in this order by every fold writer in the repo; also the order rows are re-cut in.
_SPLITS = ('train', 'val', 'test')


def read_fold(fold_dir: Path) -> dict:
    """Read one fold's three split CSVs as text.

    Args:
        fold_dir: a `fold_k` directory holding `{train,val,test}_pairs.csv`.

    Returns:
        {split_name: DataFrame}, every column dtype `str`.

    Raises:
        SystemExit: if a split CSV is missing.
        ValueError: if the three splits do not share the same columns.
    """
    splits = {}
    for name in _SPLITS:
        path = fold_dir / f'{name}_pairs.csv'
        if not path.exists():
            raise SystemExit(f"ERROR: no {name} split at {path}.")
        splits[name] = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[],
                                   low_memory=False)

    columns = {name: tuple(df.columns) for name, df in splits.items()}
    if len(set(columns.values())) != 1:
        raise ValueError(f"splits in {fold_dir} disagree on columns: {columns}")
    return splits


def repartition_fold(splits: dict, seed: int) -> dict:
    """Shuffle one fold's pooled rows and re-cut them at the input split sizes.

    Args:
        splits: {split_name: DataFrame} for one fold, all sharing columns.
        seed: seeds the row shuffle.

    Returns:
        {split_name: DataFrame} over the same rows, with the same per-split sizes.

    Raises:
        ValueError: if a `pair_key` repeats across the fold's splits, which would let one
            permutation place the same pair in two splits.
    """
    sizes = {name: len(df) for name, df in splits.items()}
    pooled = pd.concat([splits[name] for name in _SPLITS], ignore_index=True)

    n_unique = pooled['pair_key'].nunique()
    if n_unique != len(pooled):
        raise ValueError(f"fold pools {len(pooled):,} rows but only {n_unique:,} distinct "
                         f"pair_key values; a random re-cut would split a repeated pair.")

    shuffled = pooled.sample(frac=1, random_state=np.random.RandomState(seed))
    out, start = {}, 0
    for name in _SPLITS:
        out[name] = shuffled.iloc[start:start + sizes[name]].reset_index(drop=True)
        start += sizes[name]
    return out


def write_fold(fold_dir: Path, splits: dict) -> None:
    """Write one fold's splits plus its `dataset_stats.json`.

    Args:
        fold_dir: destination `fold_k` directory; created if absent.
        splits: {split_name: DataFrame} to write.

    Returns:
        None.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(fold_dir / f'{name}_pairs.csv', index=False)
    stats = {f'{name}_pairs': int(len(df)) for name, df in splits.items()}
    stats.update({f'{name}_pos': int((df['label'] == '1').sum()) for name, df in splits.items()})
    (fold_dir / 'dataset_stats.json').write_text(json.dumps(stats, indent=2))


def check_arm_matches_source(source: dict, arm: dict) -> None:
    """Verify the re-cut fold holds the source fold's rows at the source's split sizes.

    This is the experiment's premise -- the arms must differ only in which split each row
    lands in -- so it is checked rather than assumed.

    Args:
        source: {split_name: DataFrame} as read from the source fold.
        arm: {split_name: DataFrame} produced by `repartition_fold`.

    Returns:
        None.

    Raises:
        ValueError: if any split size changed, or if the two folds cover different rows.
    """
    for name in _SPLITS:
        if len(arm[name]) != len(source[name]):
            raise ValueError(f"{name}: re-cut has {len(arm[name]):,} rows, "
                             f"source has {len(source[name]):,}.")

    source_keys = set(pd.concat([source[n]['pair_key'] for n in _SPLITS], ignore_index=True))
    arm_keys = set(pd.concat([arm[n]['pair_key'] for n in _SPLITS], ignore_index=True))
    if source_keys != arm_keys:
        raise ValueError(f"re-cut covers different rows than the source: "
                         f"{len(source_keys - arm_keys):,} missing, "
                         f"{len(arm_keys - source_keys):,} added.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset_dir', type=Path, required=True,
                   help='Source dataset directory holding fold_k/ subdirectories.')
    p.add_argument('--out_dir', type=Path, required=True,
                   help='Destination for the random arm; mirrors the source fold layout.')
    p.add_argument('--seed', type=int, default=42,
                   help='Base seed for the row shuffle; fold k uses seed + k (default 42).')
    args = p.parse_args()

    fold_dirs = sorted(args.dataset_dir.glob('fold_*'))
    if not fold_dirs:
        raise SystemExit(f"ERROR: no fold_* directories under {args.dataset_dir}.")

    print(f'=== build_random_arm: {len(fold_dirs)} folds from {args.dataset_dir} ===')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_sizes = {}
    for k, fold_dir in enumerate(fold_dirs):
        source = read_fold(fold_dir)
        arm = repartition_fold(source, args.seed + k)
        check_arm_matches_source(source, arm)
        write_fold(args.out_dir / fold_dir.name, arm)
        # Read back rather than trust the write: a CSV truncated by a full filesystem is still a
        # readable CSV, just a shorter one, and would pass every in-memory check above.
        check_arm_matches_source(source, read_fold(args.out_dir / fold_dir.name))
        fold_sizes[fold_dir.name] = {name: int(len(arm[name])) for name in _SPLITS}
        print(f"  {fold_dir.name}: " + ' '.join(f'{n}={len(arm[n]):,}' for n in _SPLITS))

    arm_info = {'source_dataset_dir': str(args.dataset_dir), 'seed': args.seed,
                'partition': 'random rows, per fold, at the source split sizes',
                'fold_dirs': [d.name for d in fold_dirs], 'fold_sizes': fold_sizes,
                'code': get_git_info()}
    (args.out_dir / 'arm_info.json').write_text(json.dumps(arm_info, indent=2))
    print(f'Done. -> {args.out_dir}')


if __name__ == '__main__':
    main()
