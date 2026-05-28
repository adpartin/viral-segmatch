#!/usr/bin/env python3
"""
DNA-level coverage feasibility for v2 negative sampling.

Status (2026-05-13)
-------------------
Companion to `docs/results/2026-05-08_dna_coverage_feasibility.md`,
which documents the verdict. The feasibility verdict it produced
motivated extending v2's coverage phase to iterate `(slot, dna_hash)`
tuples (mode #2 leakage fix at DNA level — landed 2026-05-08; see
`dataset_segment_pairs_v2.py` and `docs/methods/leakage.md`
mode #2). The script is preserved as a reproducibility hook for that
result doc.

**Note on hard-coded inputs:** the `DATASET_DIR` constant near the top
of the file points at `dataset_flu_ha_na_20260508_125547`, which no
longer exists in `data/datasets/`. Update the constant to point at a
current build (e.g. a `dataset_flu_ha_na_*` run from
`data/datasets/flu/July_2025/runs/`) before re-running.

Question
--------
v2's coverage phase guarantees: for every protein `seq_hash` H on each slot,
at least one positive AND at least one negative pair carries H. This is a
PROTEIN-level invariant; mode #2 (sequence-level label imbalance) is
addressed at the protein level.

Mode #2 is NOT addressed at the DNA level. A `seq_hash` H may be encoded by
K different `dna_hash`es across isolates (synonymous codon variation). v2's
coverage gives H one negative; the K-1 other DNAs may have no negative
counterexample. For models that consume DNA-derived features (k-mers), this
is a per-feature label imbalance that the protein-level coverage doesn't
prevent.

Could v2's coverage phase be extended to dna_hash? Pair_keys are at the
seq_hash level (canonical(seq_hash_a, seq_hash_b)), so to give K DNAs of
the same H their own distinct neg pair_keys, we need K distinct partner
seq_hashes on the OPPOSITE slot, all not in cooccur with H. This script
measures whether that's feasible per split and across splits.

Outputs
-------
- /tmp/dna_coverage_feasibility/per_seq.csv
    one row per (split, slot, seq_hash) with K, partner_pool, ratio.
- /tmp/dna_coverage_feasibility/cross_split.csv
    one row per (slot, seq_hash) with cross-split demand vs supply.
- stdout: summary tables and the verdict.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path(
    'data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260508_125547'
)
OUT_DIR = Path('/tmp/dna_coverage_feasibility')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ('train', 'val', 'test')
SLOT_OPPOSITE = {'a': 'b', 'b': 'a'}


def _load_pair_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=['seq_hash_a', 'seq_hash_b', 'dna_hash_a', 'dna_hash_b', 'label'],
    )


def _build_cooccur_lookup(cooccur_csv: Path) -> dict:
    """`cooccurring_sequence_pairs.csv` has `pair_key, num_isolates` columns.
    `pair_key` is the canonical 'h1__h2' string (sorted). Return
    cooccur_by_seq[h] = set of seq_hash partners that h has co-occurred with.
    """
    df = pd.read_csv(cooccur_csv)
    by_seq: dict = defaultdict(set)
    for pk in df['pair_key'].astype(str):
        h1, h2 = pk.split('__', 1)
        by_seq[h1].add(h2)
        by_seq[h2].add(h1)
    return by_seq


def main() -> None:
    print(f'Loading dataset: {DATASET_DIR}')
    splits = {name: _load_pair_table(DATASET_DIR / f'{name}_pairs.csv') for name in SPLITS}
    for name, df in splits.items():
        print(f'  {name}: {len(df):,} pairs ({(df["label"] == 1).sum():,} pos / '
              f'{(df["label"] == 0).sum():,} neg)')

    print('Loading cooccurring_sequence_pairs ...')
    cooccur_by_seq = _build_cooccur_lookup(
        DATASET_DIR / 'cooccurring_sequence_pairs.csv')
    print(f'  unique seq_hashes appearing in cooccur set: {len(cooccur_by_seq):,}')

    # K(split, slot, seq_hash) = number of distinct dna_hashes encoding seq_hash
    # on POSITIVE pairs at this slot in this split. Pos is the relevant side
    # for mode #2: every positive-side DNA needs a negative counterexample at
    # the same DNA. (Neg DNAs are a strict subset of pos DNAs in v2 train, so
    # no extra demand from neg.)
    K_data: dict = {}
    seq_universe: dict = {}
    for split_name, df in splits.items():
        pos = df[df['label'] == 1]
        for slot in ('a', 'b'):
            grouped = pos.groupby(f'seq_hash_{slot}')[f'dna_hash_{slot}'].nunique()
            K_data[(split_name, slot)] = grouped.to_dict()
            seq_universe[(split_name, slot)] = set(grouped.index)
            print(f'  K[{split_name},{slot}]: '
                  f'{len(grouped):,} seq_hashes, '
                  f'K range [{grouped.min()}, {grouped.max()}], '
                  f'mean={grouped.mean():.2f}')

    # Per-split feasibility: for each (split, slot, seq_hash), compute
    # partner_pool (size of the OPPOSITE-slot universe in this split, minus
    # cooccur partners) and ratio K/partner_pool.
    rows: list = []
    for (split_name, slot), K_map in K_data.items():
        opposite = SLOT_OPPOSITE[slot]
        partner_universe = seq_universe[(split_name, opposite)]
        for h, K in K_map.items():
            cooccur_partners = cooccur_by_seq.get(h, set())
            blocked = cooccur_partners & partner_universe
            partner_pool = len(partner_universe) - len(blocked)
            ratio = K / partner_pool if partner_pool > 0 else float('inf')
            rows.append({
                'split': split_name,
                'slot': slot,
                'seq_hash': h,
                'K': K,
                'partner_universe': len(partner_universe),
                'cooccur_blocked': len(blocked),
                'partner_pool': partner_pool,
                'ratio': ratio,
            })
    feas = pd.DataFrame(rows).sort_values('ratio', ascending=False)
    feas.to_csv(OUT_DIR / 'per_seq.csv', index=False)
    print(f'\nSaved {OUT_DIR / "per_seq.csv"} ({len(feas):,} rows)')

    print('\n=== Per-split feasibility summary ===')
    summary = feas.groupby(['split', 'slot']).agg(
        n_seq=('seq_hash', 'count'),
        K_max=('K', 'max'),
        K_mean=('K', 'mean'),
        K_p99=('K', lambda x: float(np.percentile(x, 99))),
        partner_pool_min=('partner_pool', 'min'),
        partner_pool_median=('partner_pool', 'median'),
        ratio_max=('ratio', 'max'),
        ratio_p99=('ratio', lambda x: float(np.percentile(x, 99))),
        n_infeasible=('ratio', lambda x: int((x > 1).sum())),
    )
    print(summary.to_string())

    # Cross-split demand vs supply: total K across all splits vs
    # full-universe partner pool minus cooccur blocks.
    full_partner = {
        slot: set().union(*(seq_universe[(s, slot)] for s in SPLITS))
        for slot in ('a', 'b')
    }
    cross_rows: list = []
    for slot in ('a', 'b'):
        opposite = SLOT_OPPOSITE[slot]
        partner_full = full_partner[opposite]
        demand_per_h: dict = defaultdict(int)
        for split_name in SPLITS:
            for h, K in K_data[(split_name, slot)].items():
                demand_per_h[h] += K
        for h, demand in demand_per_h.items():
            cooccur_partners = cooccur_by_seq.get(h, set())
            blocked = cooccur_partners & partner_full
            supply = len(partner_full) - len(blocked)
            cross_rows.append({
                'slot': slot,
                'seq_hash': h,
                'demand': demand,
                'supply': supply,
                'demand_supply_ratio': demand / supply if supply > 0 else float('inf'),
            })
    cross = pd.DataFrame(cross_rows).sort_values(
        'demand_supply_ratio', ascending=False)
    cross.to_csv(OUT_DIR / 'cross_split.csv', index=False)
    print(f'\nSaved {OUT_DIR / "cross_split.csv"} ({len(cross):,} rows)')

    print('\n=== Cross-split demand vs supply summary ===')
    cross_summary = cross.groupby('slot').agg(
        n_seq=('seq_hash', 'count'),
        demand_max=('demand', 'max'),
        demand_p99=('demand', lambda x: float(np.percentile(x, 99))),
        supply_min=('supply', 'min'),
        ratio_max=('demand_supply_ratio', 'max'),
        ratio_p99=('demand_supply_ratio', lambda x: float(np.percentile(x, 99))),
        n_infeasible=('demand_supply_ratio', lambda x: int((x > 1).sum())),
    )
    print(cross_summary.to_string())

    # Verdict
    print('\n=== Verdict ===')
    n_per_split_infeasible = int((feas['ratio'] > 1).sum())
    n_cross_infeasible = int((cross['demand_supply_ratio'] > 1).sum())
    print(f'  Per-split infeasible (K > partner_pool): {n_per_split_infeasible} '
          f'of {len(feas):,} (split, slot, seq_hash) rows')
    print(f'  Cross-split infeasible (demand > supply): {n_cross_infeasible} '
          f'of {len(cross):,} (slot, seq_hash) rows')
    if n_per_split_infeasible == 0 and n_cross_infeasible == 0:
        print('  >>> DNA-level coverage is feasible. <<<')
    else:
        print('  >>> Some seq_hashes cannot be DNA-covered. See per_seq.csv / cross_split.csv. <<<')

    print('\n=== Top 10 worst per-split ratios ===')
    print(feas.head(10).to_string(index=False))
    print('\n=== Top 10 worst cross-split ratios ===')
    print(cross.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
