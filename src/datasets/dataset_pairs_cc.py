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

Phase-1 scope (aa): Hydra/`--config_bundle` driven, reuses v2's protein-level
front-end (load/enrich/filter), within-CC random negatives, slim writer (CSV +
a small dataset_stats.json — not the full v2 saver). nt_cds/nt_ctg, regime
negatives, subtype balancing / max_isolates, full saver, and `n_repeats>1` are
later phases. See `docs/plans/2026-06-09_cc_dataset_cv_plan.md`.

CLI:
    python src/datasets/dataset_pairs_cc.py \\
        --config_bundle flu_ha_na_cc --out_dir <dir> \\
        [--override dataset.n_folds=5 ...] [--protein_final <path>]
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
from omegaconf import OmegaConf, ListConfig
from sklearn.model_selection import GroupKFold

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets.dataset_segment_pairs_v2 import create_positive_pairs_v2, _PAIR_COLUMNS  # noqa: E402
from src.datasets._pair_helpers import (  # noqa: E402
    attach_dna_to_prot_df, build_cooccurrence_set, bipartite_components,
    drop_ambiguous_hn_subtype, filter_by_metadata)
from src.datasets._split_helpers import attach_cluster_ids, load_cluster_lookup  # noqa: E402
from src.datasets._cc_helpers import build_cc_isolate_pool, sample_random_within_cc_negatives  # noqa: E402
from src.utils.path_utils import load_dataframe  # noqa: E402
from src.utils.config_hydra import (  # noqa: E402
    get_virus_config_hydra, load_function_metadata, print_config_summary, save_config)
from src.utils.metadata_enrichment import enrich_prot_data_with_metadata  # noqa: E402
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds  # noqa: E402

# pos-side hash used to join clusters AND as the cooccurrence/cluster key.
# aa=protein, nt_cds=CDS, nt_ctg=contig — all md5 of the respective sequence.
_POS_HASH = {'aa': 'seq_hash', 'nt_cds': 'cds_dna_hash', 'nt_ctg': 'dna_hash'}

# Per-protein source columns copied into each pair side (a/b) of _PAIR_COLUMNS.
_SIDE_SRC = ['assembly_id', 'brc_fea_id', 'genbank_ctg_id', 'prot_seq', 'dna_seq',
             'canonical_segment', 'function', 'seq_hash', 'dna_hash']
_SIDE_RENAME = {'assembly_id': 'assembly_id', 'brc_fea_id': 'brc', 'genbank_ctg_id': 'ctg',
                'prot_seq': 'seq', 'dna_seq': 'dna_seq', 'canonical_segment': 'seg',
                'function': 'func', 'seq_hash': 'seq_hash', 'dna_hash': 'dna_hash'}


def build_frontend(config, input_file: Path, schema_pair_full: tuple) -> pd.DataFrame:
    """v2's protein-level front-end, narrowed to schema_pair (calls v2's helpers).

    Mirrors the v2 CLI (`dataset_segment_pairs.py`): load -> attach_dna -> enrich
    -> drop_ambiguous_subtype -> filter_by_metadata -> selected_functions/schema
    narrow -> seq_hash. Calls the same helpers (no v2 refactor) so the population
    matches v2 for the same bundle. subtype balancing and max_isolates sampling
    are NOT yet wired here — if a bundle sets them they raise rather than silently
    no-op (deferred Phase-1 scope).
    """
    ss = getattr(config.dataset, 'subtype_selection', None)
    if ss is not None and str(getattr(ss, 'mode', 'natural')) == 'balanced':
        raise NotImplementedError(
            "dataset.subtype_selection.mode=balanced is not yet wired into the 2D-CD builder.")
    if getattr(config.dataset, 'max_isolates_to_process', None):
        raise NotImplementedError(
            "dataset.max_isolates_to_process is not yet wired into the 2D-CD builder.")

    df = load_dataframe(input_file)
    df = attach_dna_to_prot_df(df, input_file)                 # dna_seq, dna_hash
    df = enrich_prot_data_with_metadata(df, project_root=PROJ)  # host, year, hn_subtype, ...
    if bool(getattr(config.dataset, 'drop_ambiguous_subtype', True)):
        df, _ = drop_ambiguous_hn_subtype(df)

    def _coerce(v):
        return list(v) if isinstance(v, ListConfig) else v
    df = filter_by_metadata(
        df,
        hn_subtype=_coerce(getattr(config.dataset, 'hn_subtype', None)),
        host=_coerce(getattr(config.dataset, 'host', None)),
        year=_coerce(getattr(config.dataset, 'year', None)),
        year_range=_coerce(getattr(config.dataset, 'year_range', None)),
        geo_location=_coerce(getattr(config.dataset, 'geo_location', None)),
        passage=_coerce(getattr(config.dataset, 'passage', None)),
    )

    df = df[df['function'].isin(list(config.virus.selected_functions))].reset_index(drop=True)
    df = df[df['function'].isin(schema_pair_full)].reset_index(drop=True)
    if 'seq_hash' not in df.columns:
        df['seq_hash'] = df['prot_seq'].map(lambda s: hashlib.md5(str(s).encode()).hexdigest())
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
                        neg_to_pos_ratio: float, drop_singleton_ccs: bool,
                        seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-CC within-CC random negatives, enriched to `_PAIR_COLUMNS`.

    budget per CC = round(neg_to_pos_ratio * n_pos_in_cc). Singleton /
    negative-infeasible CCs (n_neg == 0 — one unique seq pair, so no within-CC
    negative can be drawn) are dropped when `drop_singleton_ccs`. Returns
    (neg_pairs[_PAIR_COLUMNS + atom_id/cc_id], cc_log).
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
        neg = (sample_random_within_cc_negatives(cc_iso, budget, cooccur, seed=seed + int(cc))
               if (cc_iso is not None and budget > 0) else None)
        n_neg = 0 if neg is None else int(len(neg))
        # Singleton / negative-infeasible CC: no within-CC negative could be drawn
        # (one unique seq pair, or every recombination reconstructs a positive).
        row['n_neg'] = n_neg
        row['dropped_singleton'] = bool(drop_singleton_ccs and n_neg == 0)
        if n_neg and not row['dropped_singleton']:
            raw.append(neg.assign(cc_id=int(cc), atom_id=cc_to_atom[cc]))
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
    p.add_argument('--config_bundle', required=True,
                   help='Hydra bundle; must set dataset.split_strategy.mode=cluster_disjoint_cc.')
    p.add_argument('--override', nargs='+', default=None,
                   help='Hydra-style dotlist overrides (e.g., dataset.n_folds=5).')
    p.add_argument('--protein_final', default=None,
                   help='Override protein_final path (default: alongside cluster_id_path).')
    p.add_argument('--out_dir', type=Path, required=True)
    args = p.parse_args()
    t0 = time.time()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    print_config_summary(config)
    ds, ss = config.dataset, config.dataset.split_strategy

    # --- this builder owns exactly one mode ---
    mode = OmegaConf.select(config, 'dataset.split_strategy.mode')
    if mode != 'cluster_disjoint_cc':
        raise ValueError(
            f"dataset_pairs_cc requires dataset.split_strategy.mode='cluster_disjoint_cc'; "
            f"got {mode!r}. (Other modes are built by dataset_segment_pairs_v2.)")

    # --- knobs (Phase 1: aa only; regime negatives + n_repeats>1 are placeholders) ---
    alphabet = str(getattr(ss, 'cluster_alphabet', 'aa') or 'aa')
    if alphabet != 'aa':
        raise NotImplementedError(f"Phase 1 is aa-only; got cluster_alphabet={alphabet!r}.")
    pair_key_alphabet = str(getattr(ss, 'pair_key_alphabet', 'aa') or 'aa')
    if pair_key_alphabet != 'aa':
        raise NotImplementedError(f"Phase 1 is aa pair_key only; got {pair_key_alphabet!r}.")
    if getattr(ds, 'negative_sampling', None) is not None:
        raise NotImplementedError(
            "Regime-targeted within-CC negatives (dataset.negative_sampling) are a placeholder; "
            "leave it null for random within-CC negatives in Phase 1.")
    n_repeats = int(getattr(ds, 'n_repeats', 1) or 1)
    if n_repeats != 1:
        raise NotImplementedError("n_repeats>1 is a placeholder; only n_repeats=1 is wired.")

    k_folds = getattr(ds, 'n_folds', None)
    if not k_folds or int(k_folds) < 2:
        raise ValueError(f"dataset.n_folds must be an int >= 2 for the 2D-CD CV builder; got {k_folds!r}.")
    k_folds = int(k_folds)
    neg_to_pos_ratio = float(ds.neg_to_pos_ratio)
    val_ratio = float(ds.val_ratio)
    drop_singleton_ccs = bool(getattr(ss, 'drop_singleton_ccs', True))
    m_pos = getattr(ss, 'm_pos_per_cc', 1)
    m_pos = None if m_pos is None else int(m_pos)
    seed = resolve_process_seed(config, 'datasets')
    if seed is None:
        raise ValueError("Could not resolve a master seed (resolve_process_seed returned None).")
    set_deterministic_seeds(seed, cuda_deterministic=False)

    cluster_id_path = getattr(ss, 'cluster_id_path', None)
    if cluster_id_path is None:
        raise ValueError("dataset.split_strategy.cluster_id_path must be set for cluster_disjoint_cc.")
    cluster_id_path = Path(str(cluster_id_path))
    if not cluster_id_path.is_absolute():
        cluster_id_path = PROJ / cluster_id_path
    threshold = cluster_id_path.parent.name  # the tXXX dir, e.g. 't099'

    # --- schema pair: full names from the bundle, canonicalized to protein_order ---
    meta = load_function_metadata(PROJ / 'conf' / 'virus' / 'flu.yaml')
    schema_raw = [str(x) for x in ds.schema_pair]
    if len(schema_raw) != 2 or schema_raw[0] == schema_raw[1]:
        raise ValueError(f"dataset.schema_pair must be two distinct functions; got {schema_raw!r}.")
    order = list(config.virus.protein_order)
    fa, fb = (schema_raw if order.index(schema_raw[0]) <= order.index(schema_raw[1])
              else [schema_raw[1], schema_raw[0]])
    sa, sb = meta.function_to_short[fa], meta.function_to_short[fb]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(out_dir / 'resolved_config.yaml'))
    print(f"=== dataset_pairs_cc {sa}-{sb} {alphabet} {threshold} "
          f"(k={k_folds}, ratio={neg_to_pos_ratio}, m_pos_per_cc={m_pos}, "
          f"drop_singleton_ccs={drop_singleton_ccs}, seed={seed}) ===")

    input_file = (Path(args.protein_final) if args.protein_final
                  else cluster_id_path.parents[2] / 'protein_final.parquet')

    df = build_frontend(config, input_file, (fa, fb))
    print(f"  front-end: {len(df):,} protein rows / {df['assembly_id'].nunique():,} isolates ({sa}+{sb})")

    pos, _ = create_positive_pairs_v2(df, schema_pair=(fa, fb), pair_key_alphabet=pair_key_alphabet)
    cooccur, _ = build_cooccurrence_set(df, hash_col=_POS_HASH[alphabet])
    lookup = load_cluster_lookup(cluster_id_path)
    pos_ids, cc_summary = assign_atoms_prod(pos, lookup, _POS_HASH[alphabet])
    print(f"  positives: {len(pos):,} -> {len(pos_ids):,} after cluster join; "
          f"{cc_summary['n_components']:,} CCs; largest {cc_summary['largest_component_pairs']:,} pairs")

    # Atom maps + isolate pool from the FULL (uncapped) atom assignment, so the
    # within-CC negative pool covers every cluster of each CC even when positives
    # are capped below.
    c2a = {**dict(zip(pos_ids['cluster_id_a'].astype(str), pos_ids['atom_id'])),
           **dict(zip(pos_ids['cluster_id_b'].astype(str), pos_ids['atom_id']))}
    c2c = {**dict(zip(pos_ids['cluster_id_a'].astype(str), pos_ids['cc_id'])),
           **dict(zip(pos_ids['cluster_id_b'].astype(str), pos_ids['cc_id']))}
    iso = build_cc_isolate_pool(c2a, c2c, sa, sb, alphabet, threshold)
    # The membership pool is corpus-level; restrict it to the front-end-filtered
    # population so within-CC negatives are drawn only from isolates present in df
    # (otherwise a negative can reference a sequence the filters dropped, then get
    # discarded at enrich -> positive-only atoms). Clusters stay corpus-level/stable;
    # only the negative population follows the experiment's df.
    n_pool0 = len(iso)
    iso = iso[iso['assembly_id'].isin(set(df['assembly_id'].astype(str)))].reset_index(drop=True)
    if len(iso) != n_pool0:
        print(f"  negative pool restricted to df isolates: {n_pool0:,} -> {len(iso):,} pool rows")

    if m_pos:
        # Cap m_pos positives per CC: shuffle (seeded), rank within CC, keep first m.
        # Deterministic, keeps all columns, and dodges the groupby.apply grouping-column
        # deprecation. Same per-CC count as a per-group random sample.
        rng = np.random.RandomState(seed)
        shuf = pos_ids.sample(frac=1, random_state=rng).reset_index(drop=True)
        shuf['_rank'] = shuf.groupby('cc_id').cumcount()
        pos_ids = shuf[shuf['_rank'] < m_pos].drop(columns='_rank').reset_index(drop=True)
        print(f"  capped positives per CC at m_pos_per_cc={m_pos}: {len(pos_ids):,} kept")

    neg, cc_log = within_cc_negatives(
        pos_ids, iso, cooccur, df, (fa, fb), neg_to_pos_ratio=neg_to_pos_ratio,
        drop_singleton_ccs=drop_singleton_ccs, seed=seed)
    n_dropped_cc = int(cc_log['dropped_singleton'].sum()) if len(cc_log) else 0
    print(f"  negatives: {len(neg):,} (within-CC); {n_dropped_cc:,} singleton CCs dropped")

    # Drop positives of dropped (singleton) CCs so every kept CC is class-balanceable.
    if drop_singleton_ccs and n_dropped_cc:
        dropped_ccs = set(cc_log.loc[cc_log['dropped_singleton'], 'cc_id'])
        pos_ids = pos_ids[~pos_ids['cc_id'].isin(dropped_ccs)].reset_index(drop=True)

    pos_full = pos_ids.copy()
    pos_full['neg_regime'] = pd.NA
    pos_full['metadata_match_count'] = pd.NA
    keep = list(_PAIR_COLUMNS) + ['cc_id', 'atom_id']
    full = pd.concat([pos_full[keep], neg[keep]], ignore_index=True)
    print(f"  full set: {len(full):,} pairs ({int((full.label == 1).sum()):,} pos / "
          f"{int((full.label == 0).sum()):,} neg) across {full['atom_id'].nunique():,} atoms")

    cv_info = {'k_folds': k_folds, 'n_repeats': n_repeats, 'seed': seed,
               'config_bundle': args.config_bundle, 'schema_pair': [sa, sb],
               'alphabet': alphabet, 'threshold': threshold, 'cluster_id_path': str(cluster_id_path),
               'm_pos_per_cc': m_pos, 'neg_to_pos_ratio': neg_to_pos_ratio,
               'pair_key_alphabet': pair_key_alphabet, 'drop_singleton_ccs': drop_singleton_ccs,
               'fold_dirs': [f'fold_{k}' for k in range(k_folds)]}
    (out_dir / 'cv_info.json').write_text(json.dumps(cv_info, indent=2))
    cc_log.to_csv(out_dir / 'cc_sampling_log.csv', index=False)

    folds = make_folds(full, k_folds, val_ratio, seed)
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
