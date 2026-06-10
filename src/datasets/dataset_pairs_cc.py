"""Stage 3 (CC builder): materialized cluster-disjoint K-fold pair datasets.

Maintained companion to `dataset_segment_pairs_v2.py` for the CC-based CV track.
Where v2's `cluster_disjoint` mode does single-slot k-fold / bilateral holdout with
cross-isolate negatives, this builder does what the CC analysis established:

  - **bilateral connected-component GroupKFold** — atoms = bipartite CCs on
    `(cluster_id_a, cluster_id_b)` (production `attach_cluster_ids` +
    `bipartite_components`); whole CCs stay in one fold.
  - **within-CC negatives** — every negative drawn from the same CC as its
    positives (`_cc_helpers.sample_random_within_cc_negatives`), so train/test
    negatives are cluster-disjoint by construction.

Output is drop-in Stage-4 datasets: `fold_k/{train,val,test}_pairs.csv` carrying
the v2 `_PAIR_COLUMNS` schema, one dir per fold.

Scaffold scope (aa, Phase 1): argparse CLI, no metadata filter, slim writer
(CSV + a small dataset_stats.json — not the full v2 saver). Hydra/`--config_bundle`
wiring, nt_cds/nt_ctg, regime negatives, and `n_repeats>1` are later phases.
See `docs/plans/2026-06-09_cc_dataset_cv_plan.md`.

CLI:
    python src/datasets/dataset_pairs_cc.py \\
        [--schema_pair HA NA] [--alphabet aa] [--threshold t099] \\
        [--k_folds 5] [--n_repeats 1] [--m_pos 100] [--neg_to_pos_ratio 1.0] \\
        [--val_ratio 0.1] [--drop_single_isolate_ccs] [--seed 0] --out_dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets.dataset_segment_pairs_v2 import create_positive_pairs_v2, _PAIR_COLUMNS  # noqa: E402
from src.datasets._pair_helpers import (  # noqa: E402
    attach_dna_to_prot_df, build_cooccurrence_set, bipartite_components)
from src.datasets._split_helpers import attach_cluster_ids, load_cluster_lookup  # noqa: E402
from src.datasets._cc_helpers import build_cc_isolate_pool, sample_random_within_cc_negatives  # noqa: E402
from src.utils.path_utils import load_dataframe  # noqa: E402
from src.utils.config_hydra import load_function_metadata  # noqa: E402

_PROC = PROJ / 'data/processed/flu/July_2025'
_CLUSTERS = {'aa': _PROC / 'clusters_aa', 'nt_cds': _PROC / 'clusters_nt_cds',
             'nt_ctg': _PROC / 'clusters_nt_ctg'}
# pos-side hash used to join clusters AND as the cooccurrence/cluster key.
# aa=protein, nt_cds=CDS, nt_ctg=contig — all md5 of the respective sequence.
_POS_HASH = {'aa': 'seq_hash', 'nt_cds': 'cds_dna_hash', 'nt_ctg': 'dna_hash'}

# Per-protein source columns copied into each pair side (a/b) of _PAIR_COLUMNS.
_SIDE_SRC = ['assembly_id', 'brc_fea_id', 'genbank_ctg_id', 'prot_seq', 'dna_seq',
             'canonical_segment', 'function', 'seq_hash', 'dna_hash']
_SIDE_RENAME = {'assembly_id': 'assembly_id', 'brc_fea_id': 'brc', 'genbank_ctg_id': 'ctg',
                'prot_seq': 'seq', 'dna_seq': 'dna_seq', 'canonical_segment': 'seg',
                'function': 'func', 'seq_hash': 'seq_hash', 'dna_hash': 'dna_hash'}


def load_frontend(protein_path: Path, schema_pair_full: tuple) -> pd.DataFrame:
    """Protein-level frame narrowed to schema_pair, with seq_hash + dna columns.

    Reuses the v2 front-end primitives (load + `attach_dna_to_prot_df`). No
    metadata enrich/filter in the scaffold — unfiltered full corpus.
    """
    df = load_dataframe(protein_path)
    df = df[df['function'].isin(schema_pair_full)].reset_index(drop=True)
    if 'seq_hash' not in df.columns:
        df['seq_hash'] = df['prot_seq'].map(lambda s: hashlib.md5(str(s).encode()).hexdigest())
    df = attach_dna_to_prot_df(df, protein_path)  # adds dna_seq, dna_hash
    return df


def assign_atoms_prod(pos: pd.DataFrame, cluster_lookup: pd.DataFrame, pos_hash_col: str):
    """Attach cluster ids + bilateral bipartite-CC atom_id/cc_id (production path).

    Atoms = natural connected components on (cluster_id_a, cluster_id_b). Returns
    (pos_with_ids, cc_summary). atom_id == cc_id (no fragmentation in Phase 1).
    """
    pos_ids, attach_audit = attach_cluster_ids(pos, cluster_lookup, pos_hash_col=pos_hash_col)
    component_id, cc_summary = bipartite_components(
        pos_ids, col_a='cluster_id_a', col_b='cluster_id_b')
    pos_ids = pos_ids.copy()
    pos_ids['cc_id'] = component_id.to_numpy()
    pos_ids['atom_id'] = pos_ids['cc_id']
    cc_summary['n_dropped_cluster_join'] = attach_audit['n_input'] - attach_audit['n_kept']
    return pos_ids, cc_summary


def _side_rep(df: pd.DataFrame, func: str, suffix: str) -> pd.DataFrame:
    """{one row per seq_hash} of per-side fields for `func`, renamed to *_<suffix>.

    First occurrence per seq_hash is the representative (matches v2's keep='first').
    """
    cols = list(_SIDE_SRC) + (['cds_dna_hash'] if 'cds_dna_hash' in df.columns else [])
    rep = df[df['function'] == func][cols].drop_duplicates('seq_hash', keep='first').copy()
    ren = {k: f'{v}_{suffix}' for k, v in _SIDE_RENAME.items()}
    if 'cds_dna_hash' in df.columns:
        ren['cds_dna_hash'] = f'cds_dna_hash_{suffix}'
    return rep.rename(columns=ren)


def within_cc_negatives(pos_ids: pd.DataFrame, iso: pd.DataFrame, cooccur: set,
                        df: pd.DataFrame, schema_pair_full: tuple, *,
                        neg_to_pos_ratio: float, drop_single_isolate_ccs: bool,
                        seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-CC within-CC random negatives, enriched to `_PAIR_COLUMNS`.

    budget per CC = round(neg_to_pos_ratio * n_pos_in_cc). CCs with a single
    isolate (no within-CC negative possible) are dropped when
    `drop_single_isolate_ccs`. Returns (neg_pairs[_PAIR_COLUMNS + atom_id/cc_id],
    cc_log).
    """
    fa, fb = schema_pair_full
    iso_by_cc = {cc: g for cc, g in iso.groupby('cc_id')}
    cc_to_atom = dict(zip(pos_ids['cc_id'], pos_ids['atom_id']))
    pos_per_cc = pos_ids.groupby('cc_id').size()

    raw, log_rows = [], []
    for cc, n_pos in pos_per_cc.items():
        cc_iso = iso_by_cc.get(cc)
        n_iso = 0 if cc_iso is None else int(cc_iso['assembly_id'].nunique())
        budget = int(round(neg_to_pos_ratio * n_pos))
        row = {'cc_id': int(cc), 'n_pos': int(n_pos), 'n_isolates': n_iso, 'budget': budget}
        if cc_iso is None or n_iso < 2 or budget <= 0:
            row['n_neg'] = 0
            row['dropped_single_isolate'] = bool(drop_single_isolate_ccs and n_iso < 2)
            log_rows.append(row)
            continue
        neg = sample_random_within_cc_negatives(cc_iso, budget, cooccur, seed=seed + int(cc))
        if len(neg):
            neg = neg.assign(cc_id=int(cc), atom_id=cc_to_atom[cc])
            raw.append(neg)
        row['n_neg'] = int(len(neg))
        row['dropped_single_isolate'] = False
        log_rows.append(row)

    cc_log = pd.DataFrame(log_rows)
    if not raw:
        return pd.DataFrame(columns=list(_PAIR_COLUMNS) + ['cc_id', 'atom_id']), cc_log

    neg_all = pd.concat(raw, ignore_index=True)  # hash_a, hash_b, neg_regime, metadata_match_count, cc_id, atom_id
    ra, rb = _side_rep(df, fa, 'a'), _side_rep(df, fb, 'b')
    out = (neg_all.rename(columns={'hash_a': 'seq_hash_a', 'hash_b': 'seq_hash_b'})
           .merge(ra, on='seq_hash_a', how='left')
           .merge(rb, on='seq_hash_b', how='left'))
    miss = out[['assembly_id_a', 'assembly_id_b']].isna().any(axis=1)
    if miss.any():
        print(f"WARNING: dropping {int(miss.sum()):,} negatives whose sequence is "
              f"absent from the (filtered) protein frame.")
        out = out[~miss].reset_index(drop=True)
    a = out['seq_hash_a'].astype(str).to_numpy()
    b = out['seq_hash_b'].astype(str).to_numpy()
    out['pair_key'] = np.where(a <= b, a, b) + '__' + np.where(a <= b, b, a)
    out['label'] = 0
    for c in ('cds_dna_hash_a', 'cds_dna_hash_b'):
        if c not in out.columns:
            out[c] = pd.NA
    keep = list(_PAIR_COLUMNS) + ['cc_id', 'atom_id']
    return out[keep].reset_index(drop=True), cc_log


def make_folds(full: pd.DataFrame, k_folds: int, val_ratio: float, seed: int):
    """GroupKFold(by atom_id) -> per fold (train, val, test). val carved group-aware
    from the non-test atoms (whole atoms never split across train/val)."""
    gkf = GroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    groups = full['atom_id'].to_numpy()
    n_total = len(full)
    folds = []
    for tv_idx, te_idx in gkf.split(full, groups=groups):
        test = full.iloc[te_idx]
        tv = full.iloc[tv_idx]
        # group-aware val carve: take whole atoms until ~val_ratio of the WHOLE set.
        rng = np.random.RandomState(seed)
        atoms = tv['atom_id'].drop_duplicates().to_numpy()
        rng.shuffle(atoms)
        sizes = tv.groupby('atom_id').size()
        target = val_ratio * n_total
        val_atoms, acc = set(), 0
        for a in atoms:
            if acc >= target:
                break
            val_atoms.add(a)
            acc += int(sizes[a])
        val = tv[tv['atom_id'].isin(val_atoms)]
        train = tv[~tv['atom_id'].isin(val_atoms)]
        folds.append((train.reset_index(drop=True), val.reset_index(drop=True),
                      test.reset_index(drop=True)))
    return folds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--protein_final', default=str(_PROC / 'protein_final.parquet'))
    p.add_argument('--schema_pair', nargs=2, default=['HA', 'NA'], metavar=('A', 'B'))
    p.add_argument('--alphabet', default='aa', choices=['aa'])  # nt_cds/nt_ctg: later phases
    p.add_argument('--threshold', default='t099')
    p.add_argument('--k_folds', type=int, default=5)
    p.add_argument('--n_repeats', type=int, default=1)
    p.add_argument('--m_pos', type=int, default=100, help='cap positives per CC (fold balance).')
    p.add_argument('--neg_to_pos_ratio', type=float, default=1.0)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--drop_single_isolate_ccs', action='store_true', default=True)
    p.add_argument('--keep_single_isolate_ccs', dest='drop_single_isolate_ccs', action='store_false')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', type=Path, required=True)
    args = p.parse_args()
    if args.n_repeats != 1:
        raise NotImplementedError("n_repeats>1 is a placeholder; only n_repeats=1 is wired.")
    t0 = time.time()

    sa, sb = args.schema_pair
    s2f = load_function_metadata(PROJ / 'conf/virus/flu.yaml').short_to_function
    fa, fb = s2f[sa], s2f[sb]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== dataset_pairs_cc {sa}-{sb} {args.alphabet} {args.threshold} (k={args.k_folds}) ===")

    df = load_frontend(Path(args.protein_final), (fa, fb))
    pos, _ = create_positive_pairs_v2(df, schema_pair=(fa, fb), pair_key_alphabet='aa')
    cooccur, _ = build_cooccurrence_set(df, hash_col=_POS_HASH[args.alphabet])
    lookup = load_cluster_lookup(_CLUSTERS[args.alphabet] / args.threshold / 'combined_cluster.parquet')
    pos_ids, cc_summary = assign_atoms_prod(pos, lookup, _POS_HASH[args.alphabet])
    print(f"  positives: {len(pos):,} -> {len(pos_ids):,} after cluster join; "
          f"{cc_summary['n_components']:,} CCs; largest {cc_summary['largest_component_pairs']:,} pairs")

    # Atom maps + isolate pool from the FULL (uncapped) atom assignment, so the
    # within-CC negative pool covers every cluster of each CC even when positives
    # are capped below.
    c2a = {**dict(zip(pos_ids['cluster_id_a'].astype(str), pos_ids['atom_id'])),
           **dict(zip(pos_ids['cluster_id_b'].astype(str), pos_ids['atom_id']))}
    c2c = {**dict(zip(pos_ids['cluster_id_a'].astype(str), pos_ids['cc_id'])),
           **dict(zip(pos_ids['cluster_id_b'].astype(str), pos_ids['cc_id']))}
    iso = build_cc_isolate_pool(c2a, c2c, sa, sb, args.alphabet, args.threshold)

    if args.m_pos:
        # Cap m_pos positives per CC: shuffle (seeded), rank within CC, keep first m.
        # Deterministic, keeps all columns, and dodges the groupby.apply grouping-column
        # deprecation. Same per-CC count as a per-group random sample.
        rng = np.random.RandomState(args.seed)
        shuf = pos_ids.sample(frac=1, random_state=rng).reset_index(drop=True)
        shuf['_rank'] = shuf.groupby('cc_id').cumcount()
        pos_ids = shuf[shuf['_rank'] < args.m_pos].drop(columns='_rank').reset_index(drop=True)
        print(f"  capped positives per CC at m_pos={args.m_pos}: {len(pos_ids):,} kept")

    neg, cc_log = within_cc_negatives(
        pos_ids, iso, cooccur, df, (fa, fb), neg_to_pos_ratio=args.neg_to_pos_ratio,
        drop_single_isolate_ccs=args.drop_single_isolate_ccs, seed=args.seed)
    n_dropped_cc = int(cc_log['dropped_single_isolate'].sum()) if len(cc_log) else 0
    print(f"  negatives: {len(neg):,} (within-CC); {n_dropped_cc:,} single-isolate CCs dropped")

    # Drop positives belonging to dropped (single-isolate) CCs so every kept CC is balanceable.
    if args.drop_single_isolate_ccs and n_dropped_cc:
        dropped_ccs = set(cc_log.loc[cc_log['dropped_single_isolate'], 'cc_id'])
        pos_ids = pos_ids[~pos_ids['cc_id'].isin(dropped_ccs)].reset_index(drop=True)

    pos_full = pos_ids.copy()
    pos_full['neg_regime'] = pd.NA
    pos_full['metadata_match_count'] = pd.NA
    keep = list(_PAIR_COLUMNS) + ['cc_id', 'atom_id']
    full = pd.concat([pos_full[keep], neg[keep]], ignore_index=True)
    print(f"  full set: {len(full):,} pairs ({int((full.label == 1).sum()):,} pos / "
          f"{int((full.label == 0).sum()):,} neg) across {full['atom_id'].nunique():,} atoms")

    cv_info = {'k_folds': args.k_folds, 'n_repeats': args.n_repeats, 'seed': args.seed,
               'schema_pair': [sa, sb], 'alphabet': args.alphabet, 'threshold': args.threshold,
               'm_pos': args.m_pos, 'neg_to_pos_ratio': args.neg_to_pos_ratio,
               'drop_single_isolate_ccs': args.drop_single_isolate_ccs,
               'fold_dirs': [f'fold_{k}' for k in range(args.k_folds)]}
    (out_dir / 'cv_info.json').write_text(json.dumps(cv_info, indent=2))
    cc_log.to_csv(out_dir / 'cc_sampling_log.csv', index=False)

    folds = make_folds(full, args.k_folds, args.val_ratio, args.seed)
    for k, (train, val, test) in enumerate(folds):
        fdir = out_dir / f'fold_{k}'
        fdir.mkdir(parents=True, exist_ok=True)
        for name, split in [('train', train), ('val', val), ('test', test)]:
            split[list(_PAIR_COLUMNS)].to_csv(fdir / f'{name}_pairs.csv', index=False)
        stats = {f'{n}_pairs': int(len(s)) for n, s in
                 [('train', train), ('val', val), ('test', test)]}
        stats.update({f'{n}_pos': int((s.label == 1).sum()) for n, s in
                      [('train', train), ('val', val), ('test', test)]})
        (fdir / 'dataset_stats.json').write_text(json.dumps(stats, indent=2))
        print(f"  fold_{k}: train={len(train):,} val={len(val):,} test={len(test):,}")

    print(f"\nDone in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == '__main__':
    main()
