"""Measure DataSAIL's L(π) leakage metric on a given train/val/test partition.

L(π) ("L of pi") is the leakage measure from the DataSAIL paper
(Joeres et al. 2025, Eq. 20). It quantifies how much within-cluster
similarity gets "broken" across split boundaries — lower is better,
zero means perfect cluster-disjointness.

This script runs DataSAIL's `eval_split` function in
**measurement-only mode**: given an existing per-pair partition
(produced by segmatch's bicc cluster_disjoint, seq_disjoint, etc.),
it derives a per-slot entity-to-split assignment and asks DataSAIL
to score the partition by L(π). DataSAIL internally re-clusters the
entities via mmseqs2 (`similarity='mmseqs'` → `mmseqs easy-cluster`),
then sums cross-split cluster-pair weights.

**S1 (single-input) only.** This script measures L(π) per slot
independently (one number for HA, one for NA), not a joint pair-level
S2 L(π). S2 requires a custom paired-input implementation and is
out of scope for v1.

**Important**: this script imports `datasail`, which is only
installed in the dedicated `datasail` conda env. Run with:

    conda run -n datasail python src/analysis/datasail_lpi_measure.py \\
        --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_<bundle>_<TS> \\
        --routing_label cluster_disjoint_id099 \\
        --out_csv results/flu/July_2025/runs/datasail_lpi/<TS>_<routing>.csv

See `docs/plans/2026-05-22_split_separation_metrics_plan.md` Step 1
for the broader plan; `src/analysis/datasail_bakeoff.py` is the
sibling script that runs DataSAIL in solving mode (Phase 0 bake-off).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from datasail.eval import eval_split
except ImportError as e:
    sys.stderr.write(
        f"ERROR: cannot import datasail ({e}). This script must run in the "
        f"'datasail' conda env. Try:\n"
        f"    conda run -n datasail python {sys.argv[0]} ...\n"
    )
    sys.exit(1)


def load_positives(dataset_dir: Path) -> pd.DataFrame:
    """Concatenate train/val/test pair CSVs, return positives only.

    Adds an `orig_split` column ('train' / 'val' / 'test') derived from
    the source CSV. This is the per-pair partition we'll measure L(π)
    against.
    """
    frames = []
    for split in ('train_pairs.csv', 'val_pairs.csv', 'test_pairs.csv'):
        p = dataset_dir / split
        if not p.exists():
            raise FileNotFoundError(f"Expected {p}; not found.")
        df = pd.read_csv(p, low_memory=False)
        df['orig_split'] = split.replace('_pairs.csv', '')
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    pos = full[full['label'] == 1].reset_index(drop=True)
    return pos


def derive_per_slot_inputs(pos: pd.DataFrame, slot: str
                            ) -> tuple[dict, dict, list]:
    """Build {seq_hash: seq} data dict and {seq_hash: split} assignment dict
    for one slot ('a' or 'b').

    Returns (data, split_assignment, ambiguous_hashes). The ambiguous list
    contains seq_hashes that appeared in multiple splits (which violates
    eval_split's one-split-per-entity contract). Under cluster_disjoint and
    seq_disjoint this should be empty.
    """
    hash_col = f'seq_hash_{slot}'
    seq_col = f'seq_{slot}'

    data = dict(zip(pos[hash_col], pos[seq_col]))

    grouped = pos.groupby(hash_col)['orig_split'].agg(lambda s: set(s))
    ambiguous = [h for h, splits in grouped.items() if len(splits) > 1]
    split_assignment = {h: list(splits)[0] for h, splits in grouped.items()}

    return data, split_assignment, ambiguous


def measure_one_slot(pos: pd.DataFrame, slot: str, similarity: str,
                      ) -> dict:
    """Run eval_split for one slot. Returns dict with the three L(π) outputs
    plus diagnostic counts.
    """
    data, split_assignment, ambiguous = derive_per_slot_inputs(pos, slot)
    if ambiguous:
        sys.stderr.write(
            f"WARNING: slot_{slot} has {len(ambiguous)} seq_hashes spanning "
            f"multiple splits — eval_split's one-split-per-entity contract "
            f"is violated. First-seen split is used; results may be noisy.\n"
        )

    t0 = time.time()
    ratio, absolute, total = eval_split(
        datatype='P',
        data=data,
        weights=None,
        similarity=similarity,
        distance=None,
        dist_conv=None,
        split_assignment=split_assignment,
    )
    wall = time.time() - t0

    return {
        f'slot': slot,
        f'n_unique_entities': len(data),
        f'n_ambiguous_split_entities': len(ambiguous),
        f'lpi_ratio': float(ratio),
        f'lpi_absolute': float(absolute),
        f'lpi_total': float(total),
        f'wall_seconds': round(wall, 2),
    }


def main():
    p = argparse.ArgumentParser(
        description='Measure DataSAIL L(π) on an existing partition (S1, per slot).',
    )
    p.add_argument('--dataset_dir', required=True, type=Path,
                   help='Stage 3 dataset directory (with train/val/test_pairs.csv).')
    p.add_argument('--routing_label', required=True, type=str,
                   help='Label describing the partition for the output row, '
                        'e.g. "cluster_disjoint_aa_id099", "seq_disjoint".')
    p.add_argument('--similarity', default='mmseqs',
                   choices=['mmseqs', 'mmseqspp'],
                   help='DataSAIL similarity method. "mmseqs" (default) uses '
                        'easy-cluster + binary 1-similarity between cluster '
                        'reps (cheap, coarse). "mmseqspp" uses align+convertalis '
                        'fident scores (continuous, slower).')
    p.add_argument('--n_isolates', type=int, default=None,
                   help='Optional subsample for sanity runs. Omit for full set.')
    p.add_argument('--seed', type=int, default=0,
                   help='Subsample seed (only used if --n_isolates is set).')
    p.add_argument('--out_csv', required=True, type=Path,
                   help='Output CSV path. One row per slot (HA, NA).')
    args = p.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading positives from {args.dataset_dir} ...")
    pos = load_positives(args.dataset_dir)
    n_all = len(pos)
    print(f"  Loaded {n_all:,} positives.")

    if args.n_isolates is not None:
        # Subsample by assembly_id — keep all pairs whose assembly_id_a is in
        # the sampled set (under cluster_disjoint on Flu A, assembly_id_a ==
        # assembly_id_b for HA/NA pairs).
        isolates = pos['assembly_id_a'].drop_duplicates()
        sampled = isolates.sample(
            n=min(args.n_isolates, len(isolates)),
            random_state=args.seed,
        )
        pos = pos[pos['assembly_id_a'].isin(set(sampled))].reset_index(drop=True)
        print(f"  Subsampled to {len(pos):,} positives from "
              f"{args.n_isolates} isolates (seed={args.seed}).")

    print(f"Running eval_split with similarity='{args.similarity}' ...")
    rows = []
    for slot in ('a', 'b'):
        print(f"  [slot_{slot}] ...")
        result = measure_one_slot(pos, slot, similarity=args.similarity)
        result['routing_label'] = args.routing_label
        result['dataset_dir'] = str(args.dataset_dir)
        result['similarity_method'] = args.similarity
        result['n_used_positives'] = len(pos)
        result['n_input_positives'] = n_all
        rows.append(result)
        print(f"    n_unique_entities={result['n_unique_entities']:,}  "
              f"lpi_ratio={result['lpi_ratio']:.6f}  "
              f"wall={result['wall_seconds']:.1f}s")

    out_df = pd.DataFrame(rows)
    # Sensible column ordering for the output CSV.
    col_order = [
        'routing_label', 'slot', 'similarity_method',
        'lpi_ratio', 'lpi_absolute', 'lpi_total',
        'n_unique_entities', 'n_ambiguous_split_entities',
        'n_used_positives', 'n_input_positives',
        'wall_seconds', 'dataset_dir',
    ]
    out_df = out_df[col_order]
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")
    print(out_df.to_string(index=False))
    print('\nDone.')


if __name__ == '__main__':
    main()
