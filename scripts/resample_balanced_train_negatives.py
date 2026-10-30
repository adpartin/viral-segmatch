"""Redraw balanced training negatives while freezing a fold's other rows.

The redraw uses the sampler the source dataset was built with, which it reads from that dataset's
`resolved_config.yaml` (`balanced_sampler_name`): `balanced_usage_negatives`, or
`matched_balanced_usage_negatives` when the config carries a `dataset.negative_sampling` profile.
On a matched dataset the metadata groups are rebuilt from the frozen positives, so every redrawn
negative pairs two sequences of one group, as the dataset's own negatives do.

Example:
    python scripts/resample_balanced_train_negatives.py \
        --source-fold data/datasets/flu/July_2025/runs/exp2_.../fold_0 \
        --output-root data/datasets/flu/July_2025/runs/resampled_..._fold0 \
        --seeds 42 43 44 45 46 47 48 49 50 51

Each seed directory is accepted by train_pair_baselines.py as --dataset_dir.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets._negative_regime_sampling import DEFAULT_AXES, build_isolate_cells
from src.datasets._negative_sampling import (
    balanced_usage_negatives,
    matched_balanced_usage_negatives,
)
from src.datasets.dataset_segment_pairs_v2 import balanced_sampler_name, compute_axis_flags

_SIDE_FIELDS = {
    'assembly_id': 'assembly_id', 'brc': 'brc_fea_id',
    'ctg': 'genbank_ctg_id', 'prot_seq': 'prot_seq',
    'ctg_dna_seq': 'ctg_dna_seq', 'seg': 'canonical_segment',
    'func': 'function', 'prot_hash': 'prot_hash',
    'ctg_dna_hash': 'ctg_dna_hash', 'cds_dna_hash': 'cds_dna_hash',
}
_AXES = ('host', 'year', 'hn_subtype', 'geo_location', 'passage')


def _front_from_positives(pos: pd.DataFrame) -> pd.DataFrame:
    """Build sampler lookup rows from the fixed positive pair table.

    Args:
        pos: Positive rows from one frozen training split.

    Returns:
        One source row per pair side, with sampler field names.
    """
    sides = []
    for suffix in ('a', 'b'):
        sides.append(pd.DataFrame({dst: pos[f'{src}_{suffix}']
                                   for src, dst in _SIDE_FIELDS.items()}))
    return pd.concat(sides, ignore_index=True)


def _axis_lookup(pos: pd.DataFrame) -> pd.DataFrame:
    """Build the per-isolate metadata table that `compute_axis_flags` reads.

    Args:
        pos: Positive rows from one frozen training split.

    Returns:
        One row per pair side, with `assembly_id` and one column per metadata axis.
    """
    sides = []
    for suffix in ('a', 'b'):
        sides.append(pd.DataFrame({col: pos[f'{col}_{suffix}']
                                   for col in ('assembly_id', *_AXES)}))
    return pd.concat(sides, ignore_index=True)


def _dataset_config(source_fold: Path):
    """Load the resolved config of the dataset a frozen fold belongs to.

    Args:
        source_fold: the fold directory passed as `--source-fold`.

    Returns:
        The loaded `resolved_config.yaml`, taken from the fold directory when the dataset holds a
        single split and from its parent when the fold is one of several.

    Raises:
        FileNotFoundError: neither directory holds one.
    """
    for directory in (source_fold, source_fold.parent):
        path = directory / 'resolved_config.yaml'
        if path.exists():
            config = OmegaConf.load(path)
            return config
    raise FileNotFoundError(f'No resolved_config.yaml in {source_fold} or {source_fold.parent}; '
                            f'--source-fold must name a fold of a built dataset')


def _matched_isolate_groups(config, pos: pd.DataFrame) -> dict | None:
    """The metadata groups a matched redraw must keep its negatives inside.

    Args:
        config: the source dataset's resolved config.
        pos: positive rows from the frozen training split, which carry each isolate's metadata.

    Returns:
        assembly_id -> metadata group tuple when the source dataset drew matched negatives, None
        when it drew plain balanced-usage negatives.

    Raises:
        ValueError: the dataset's negatives came from neither sampler, or were matched on binned
            years, which this redraw does not reproduce.
    """
    sampler = balanced_sampler_name(config)
    if sampler is None:
        scope = OmegaConf.select(config, 'dataset.split_strategy.negative_scope')
        raise ValueError(f'The source dataset was built with '
                         f'dataset.split_strategy.negative_scope={scope!r}; this redraw replaces '
                         f"balanced-usage negatives, so it needs 'balanced'")
    if sampler == 'balanced_usage_negatives':
        return None

    year_match = OmegaConf.select(config, 'dataset.negative_sampling.year_match')
    if year_match != 'exact':
        raise ValueError(f'The source dataset matched its negatives with '
                         f'year_match={year_match!r}; this redraw rebuilds the metadata groups '
                         f"from the saved positives and does so for 'exact' years only")
    axes = OmegaConf.select(config, 'dataset.negative_sampling.axes')
    axes = list(axes) if axes else list(DEFAULT_AXES)
    isolates = _axis_lookup(pos).drop_duplicates('assembly_id')
    groups = build_isolate_cells(isolates, axes=axes, year_match=year_match)
    return groups


def _redraw(train: pd.DataFrame, cooccur: set[str], fixed_keys: set[str], seed: int,
            isolate_to_group: dict | None) -> tuple[pd.DataFrame, set[str]]:
    """Replace training-negative partners and audit the resulting pairs.

    Args:
        train: Frozen training split with one positive and negative use per sequence.
        cooccur: Observed pair keys that negatives must avoid.
        fixed_keys: Validation and test negative pair keys that must remain distinct.
        seed: Sampler seed.
        isolate_to_group: assembly_id -> metadata group, from `_matched_isolate_groups`, to draw
            matched negatives; None to draw plain balanced-usage negatives.

    Returns:
        Training rows in their original order and the new negative pair keys.
    """
    pos = train.loc[train.label == 1]
    old_neg = train.loc[train.label == 0]
    n = len(pos)
    if n != len(old_neg):
        raise ValueError(f'Expected 1:1 class balance; got {n} positives and {len(old_neg)} negatives')
    a_col, b_col = 'cds_dna_hash_a', 'cds_dna_hash_b'
    for col in (a_col, b_col):
        if pos[col].nunique() != n or old_neg[col].nunique() != n:
            raise ValueError(f'Expected one use per sequence in {col}')
        if set(pos[col]) != set(old_neg[col]):
            raise ValueError(f'Positive and negative sequence sets differ in {col}')

    if isolate_to_group is None:
        sampled = balanced_usage_negatives(
            pos, cooccur, _front_from_positives(pos),
            (str(pos.func_a.iloc[0]), str(pos.func_b.iloc[0])),
            neg_to_pos_ratio=1.0, seed=seed, hash_col='cds_dna_hash',
            seen=set(fixed_keys),
        )
    else:
        sampled, _ = matched_balanced_usage_negatives(
            pos, cooccur, isolate_to_group, seed=seed, hash_col='cds_dna_hash',
            seen=set(fixed_keys), split_name='train',
        )
    keys = set(sampled.pair_key)
    if (len(sampled) != n or len(keys) != n or keys & (cooccur | fixed_keys)
            or set(sampled[a_col]) != set(pos[a_col])
            or set(sampled[b_col]) != set(pos[b_col])
            or sampled[a_col].nunique() != n or sampled[b_col].nunique() != n):
        raise ValueError(f'Seed {seed}: redraw failed the one-use or pair-exclusion audit')
    if isolate_to_group is not None:
        group_a = sampled['assembly_id_a'].astype(str).map(isolate_to_group)
        group_b = sampled['assembly_id_b'].astype(str).map(isolate_to_group)
        crossed = int((group_a != group_b).sum())
        if crossed:
            raise ValueError(f'Seed {seed}: {crossed:,} redrawn negatives join two metadata '
                             f'groups, which a matched draw never does')

    # Put each new negative in the row of the old negative with the same slot-A sequence.
    new_neg = sampled.set_index(a_col, drop=False).loc[old_neg[a_col]]
    new_neg.index = old_neg.index
    new_neg = compute_axis_flags(new_neg, _axis_lookup(pos))

    out = pd.concat([pos, new_neg]).sort_index()[train.columns]
    pd.testing.assert_frame_equal(out.loc[train.label == 1], pos,
                                  check_dtype=False)
    if set(out.loc[train.label == 0, 'pair_key']) != keys:
        raise AssertionError('Exported negative pairs differ from the sampler')
    return out, keys


def draw_overlap_counts(draws: list) -> dict:
    """Count how the negative pairs of one fold's draws overlap.

    The sampler keeps each negative pair unique within a draw; nothing stops a pair from being
    drawn again under another seed, so these counts measure how often that happens.

    Args:
        draws: one set of negative pair keys per draw.

    Returns:
        `max_negative_pairs_shared_by_two_draws`: the largest intersection between any two draws
        (None with fewer than two draws), and `negative_pairs_in_multiple_draws`: how many
        distinct negative pair keys appear in at least two draws.
    """
    shared = [len(a & b) for i, a in enumerate(draws) for b in draws[i + 1:]]
    uses = Counter(key for draw in draws for key in draw)
    counts = {
        'max_negative_pairs_shared_by_two_draws': max(shared) if shared else None,
        'negative_pairs_in_multiple_draws': sum(1 for n in uses.values() if n >= 2),
    }
    return counts


def main() -> None:
    """Write audited training redraws and a pair-overlap summary.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-fold', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Seeds must be distinct')
    if args.output_root.exists():
        parser.error(f'Output already exists: {args.output_root}')

    source = args.source_fold
    train = pd.read_csv(source / 'train_pairs.csv', engine='python',
                        keep_default_na=False, na_values=[''])
    cooccur = set(pd.read_csv(source / 'cooccurring_sequence_pairs.csv', usecols=['pair_key'],
                              keep_default_na=False, na_values=['']).pair_key)
    fixed_keys = set()
    for split in ('val', 'test'):
        pairs = pd.read_csv(source / f'{split}_pairs.csv', usecols=['pair_key', 'label'],
                            keep_default_na=False, na_values=[''])
        fixed_keys.update(pairs.loc[pairs.label == 0, 'pair_key'])

    config = _dataset_config(source)
    sampler = balanced_sampler_name(config)
    isolate_to_group = _matched_isolate_groups(config, train.loc[train.label == 1])

    # Complete every audit before writing any variant.
    variants = {}
    for seed in args.seeds:
        variants[seed] = _redraw(train, cooccur, fixed_keys, seed, isolate_to_group)
        print(f'Done. seed {seed}: {len(variants[seed][1]):,} training negatives from {sampler}')

    args.output_root.mkdir(parents=True)
    for seed, (pairs, _) in variants.items():
        directory = args.output_root / f'seed_{seed}'
        directory.mkdir()
        pairs.to_csv(directory / 'train_pairs.csv', index=False)
        for name in ('val_pairs.csv', 'test_pairs.csv', 'cooccurring_sequence_pairs.csv'):
            shutil.copy2(source / name, directory / name)

    sets = [value[1] for value in variants.values()]
    comparisons = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    summary = {
        'source_fold': str(source), 'seeds': args.seeds,
        'negative_sampler': sampler,
        'train_positive_count': int((train.label == 1).sum()),
        'train_negative_count': int((train.label == 0).sum()),
        'distinct_negative_pairs_across_seeds': len(set.union(*sets)),
        'pairwise_negative_jaccard_min': min(comparisons) if comparisons else None,
        'pairwise_negative_jaccard_median': float(pd.Series(comparisons).median()) if comparisons else None,
        'pairwise_negative_jaccard_max': max(comparisons) if comparisons else None,
        **draw_overlap_counts(sets),
    }
    (args.output_root / 'redraw_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Done.', json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
