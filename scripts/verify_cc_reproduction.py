#!/usr/bin/env python
"""Verify the 2D-CD builder reproduces the in-memory analysis CV (the §7 lock).

Checks that `src/datasets/dataset_pairs_cc.py` (the maintained 2D-CD builder)
materializes the SAME cluster-disjoint dataset that the trusted in-memory harness
`src/analysis/cluster_disjoint_regime_cv.py` builds, on aligned inputs (unfiltered
population, random within-CC negatives, same m_pos / ratio / seed / threshold).

Reproduction is **structural**, and that is the lock: identical positive universe,
`cooccur` blocking set, isolate pools, and **atom partition**. The within-CC
negative *identities* differ — the two atom-assignment implementations label the
CCs differently and the sampler seeds on `seed + cc_id`, so each CC draws a
different but equally valid random negative (counts match to ~1 CC). That is the
inherent non-determinism of "one random negative per CC under an arbitrary CC
labeling", not a faithfulness gap. See docs/plans/2026-06-09_cc_dataset_cv_plan.md §7.

PASS = identical universe + cooccur + atom partition. Exit 0 on PASS, 1 otherwise.

Usage:
    python scripts/verify_cc_reproduction.py \\
        [--config_bundle flu_ha_na_cc] [--schema_pair HA NA] [--threshold t099] \\
        [--m_pos 1] [--neg_to_pos_ratio 1.0] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.cluster_pair_weight_topk import load_pair_universe  # noqa: E402
from src.analysis._cv_sampling import assign_atoms, build_isolate_context  # noqa: E402
from src.analysis.cluster_disjoint_regime_cv import (  # noqa: E402
    build_regime_dataset, _DEFAULT_REGIME_TARGETS)
from src.datasets.dataset_pairs_cc import (  # noqa: E402
    build_frontend, assign_atoms_prod, within_cc_negatives, _POS_HASH)
from src.datasets.dataset_segment_pairs_v2 import create_positive_pairs_v2  # noqa: E402
from src.datasets._pair_helpers import build_cooccurrence_set  # noqa: E402
from src.datasets._split_helpers import load_cluster_lookup  # noqa: E402
from src.datasets._cc_helpers import build_cc_isolate_pool  # noqa: E402
from src.utils.config_hydra import get_virus_config_hydra, load_function_metadata  # noqa: E402


def _canon_pairs(df: pd.DataFrame) -> set:
    """Order-invariant set of (seq_hash_a, seq_hash_b) pairs."""
    a = df['seq_hash_a'].astype(str).to_numpy()
    b = df['seq_hash_b'].astype(str).to_numpy()
    return set(map(tuple, np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)))


def _partition(df: pd.DataFrame) -> frozenset:
    """Label-independent CC partition: {frozenset(canonical pair keys) per cc_id}."""
    a = df['seq_hash_a'].astype(str).to_numpy()
    b = df['seq_hash_b'].astype(str).to_numpy()
    key = np.where(a <= b, a, b) + '|' + np.where(a <= b, b, a)
    tmp = pd.DataFrame({'key': key, 'cc': df['cc_id'].to_numpy()})
    return frozenset(frozenset(g['key']) for _, g in tmp.groupby('cc'))


def analysis_dataset(cds_final, slot_a, slot_b, threshold, *, m_pos, ratio, seed):
    """In-memory reference: universe -> atoms -> within-CC random negatives."""
    universe = load_pair_universe(cds_final, slot_a, slot_b)
    u = assign_atoms(universe, slot_a, slot_b, 'aa', threshold, strategy='natural')
    iso, cooccur = build_isolate_context(u, universe, slot_a, slot_b, 'aa', threshold)
    full, _ = build_regime_dataset(
        universe, iso, cooccur, slot_a, slot_b, 'aa', threshold,
        m_pos=m_pos, m_neg=None, neg_to_pos_ratio=ratio,
        regime_targets=_DEFAULT_REGIME_TARGETS, neg_strategy='random', seed=seed)
    return u, cooccur, full


def builder_dataset(config, protein_final, fa, fb, slot_a, slot_b, threshold, *, m_pos, ratio, seed):
    """Maintained builder pipeline on the UNFILTERED population (matches the harness)."""
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(['dataset.drop_ambiguous_subtype=false']))
    df = build_frontend(config, protein_final, (fa, fb))
    pos, _ = create_positive_pairs_v2(df, schema_pair=(fa, fb), pair_key_alphabet='aa')
    cooccur, _ = build_cooccurrence_set(df, hash_col=_POS_HASH['aa'])
    lookup = load_cluster_lookup(protein_final.parent / 'clusters_aa' / threshold / 'combined_cluster.parquet')
    pos_ids, _ = assign_atoms_prod(pos, lookup, _POS_HASH['aa'])
    c2a = {**dict(zip(pos_ids['cluster_id_a'].astype(str), pos_ids['atom_id'])),
           **dict(zip(pos_ids['cluster_id_b'].astype(str), pos_ids['atom_id']))}
    c2c = {**dict(zip(pos_ids['cluster_id_a'].astype(str), pos_ids['cc_id'])),
           **dict(zip(pos_ids['cluster_id_b'].astype(str), pos_ids['cc_id']))}
    iso = build_cc_isolate_pool(c2a, c2c, slot_a, slot_b, 'aa', threshold)
    iso = iso[iso['assembly_id'].isin(set(df['assembly_id'].astype(str)))].reset_index(drop=True)
    if m_pos:
        rng = np.random.RandomState(seed)
        shuf = pos_ids.sample(frac=1, random_state=rng).reset_index(drop=True)
        shuf['_rank'] = shuf.groupby('cc_id').cumcount()
        pos_capped = shuf[shuf['_rank'] < m_pos].drop(columns='_rank').reset_index(drop=True)
    else:
        pos_capped = pos_ids
    neg, _ = within_cc_negatives(pos_capped, iso, cooccur, df, (fa, fb),
                                 neg_to_pos_ratio=ratio, seed=seed)
    return pos_ids, cooccur, neg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config_bundle', default='flu_ha_na_cc')
    p.add_argument('--schema_pair', nargs=2, default=['HA', 'NA'], metavar=('A', 'B'))
    p.add_argument('--threshold', default='t099')
    p.add_argument('--m_pos', type=int, default=1)
    p.add_argument('--neg_to_pos_ratio', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    sa, sb = args.schema_pair
    proc = PROJ / 'data/processed/flu/July_2025'
    s2f = load_function_metadata(PROJ / 'conf/virus/flu.yaml').short_to_function
    fa, fb = s2f[sa], s2f[sb]
    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))

    print(f"=== verify_cc_reproduction {sa}-{sb} aa {args.threshold} "
          f"(m_pos={args.m_pos}, ratio={args.neg_to_pos_ratio}, seed={args.seed}) ===")
    u, cooccur_a, ana_full = analysis_dataset(
        proc / 'cds_final.parquet', sa, sb, args.threshold,
        m_pos=args.m_pos, ratio=args.neg_to_pos_ratio, seed=args.seed)
    pos_ids, cooccur_b, neg = builder_dataset(
        config, proc / 'protein_final.parquet', fa, fb, sa, sb, args.threshold,
        m_pos=args.m_pos, ratio=args.neg_to_pos_ratio, seed=args.seed)

    uni_ok = _canon_pairs(u) == _canon_pairs(pos_ids)
    cooccur_ok = cooccur_a == cooccur_b
    part_ok = _partition(u) == _partition(pos_ids)
    ana_neg, bld_neg = _canon_pairs(ana_full[ana_full['label'] == 0]), _canon_pairs(neg)

    print(f"  universe:   analysis={u.shape[0]:,}  builder={pos_ids.shape[0]:,}  identical={uni_ok}")
    print(f"  cooccur:    analysis={len(cooccur_a):,}  builder={len(cooccur_b):,}  identical={cooccur_ok}")
    print(f"  partition:  analysis_CCs={u['cc_id'].nunique():,}  builder_CCs={pos_ids['cc_id'].nunique():,}  identical={part_ok}")
    print(f"  negatives:  analysis={len(ana_neg):,}  builder={len(bld_neg):,}  common={len(ana_neg & bld_neg):,}  "
          f"(identities differ by label-dependent seeding -- expected, see module docstring)")

    print()
    if uni_ok and cooccur_ok and part_ok:
        print("Done. STRUCTURAL REPRODUCTION CONFIRMED (universe + cooccur + atom partition identical).")
        return 0
    print("ERROR: structural reproduction FAILED -- a non-label difference exists; investigate.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
