"""Stage 3 (CC builder): cluster-disjoint K-fold pair datasets.

Maintained companion to `dataset_segment_pairs_v2.py` for the CC-based CV track.
Where v2's `cluster_disjoint` does single-slot k-fold (a/b) or bilateral holdout
(one 80/10/10) with cross-isolate negatives, this builder does what the CC analysis
established:

  - **2D connected-component (CC) GroupKFold** — atoms = CCs on
    `(cluster_id_a, cluster_id_b)` (production `attach_cluster_ids` +
    `cluster_ccs`); whole CCs stay in one fold.
  - **within-CC negatives** — every negative drawn from the same CC as its
    positives (`_cc_helpers.sample_random_within_cc_negatives`), so train/test
    negatives are cluster-disjoint by construction.

Output is drop-in Stage-4 datasets: `fold_k/{train,val,test}_pairs.csv` carrying
the v2 `_PAIR_COLUMNS` schema, one dir per fold.

Hydra/`--config_bundle` driven, reusing v2's protein-level front-end
(load/enrich/filter). Supports aa / nt_cds / nt_ctg, within-CC and within-fold random
negatives, and a slim writer (CSV + a small dataset_stats.json). Not wired (these
raise rather than silently no-op): regime-targeted negatives, subtype balancing /
max_isolates, the full v2 saver, and `n_repeats>1`. See
`docs/plans/2026-06-09_cc_dataset_cv_plan.md`.

One block near the bottom (banner: "OOD-vs-random paired CV") is experiment
scaffolding rather than production routing -- it serves a single bundle and is
reachable only through `negative_scope: within_cc`.

CLI:
    python src/datasets/dataset_pairs_cc.py \\
        --config_bundle flu_ha_na_cc_aa --out_dir <dir> \\
        [--override dataset.n_folds=5 ...] [--protein_final <path>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import ListConfig, OmegaConf
from sklearn.model_selection import GroupKFold

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.datasets._cc_helpers import build_cc_isolate_pool, sample_random_within_cc_negatives  # noqa: E402
from src.datasets._megacc_cut import fragment_until, stop_at_n_atoms  # noqa: E402
from src.datasets._pair_helpers import (  # noqa: E402
    attach_cds_dna_hash_to_prot_df,
    attach_ctg_dna_to_prot_df,
    build_cooccurrence_set,
    canonical_pair_key,
    cluster_ccs,
    drop_ambiguous_hn_subtype,
    filter_by_metadata,
)
from src.datasets._split_helpers import attach_cluster_ids, load_cluster_lookup  # noqa: E402
from src.datasets.dataset_segment_pairs_v2 import _PAIR_COLUMNS, create_positive_pairs_v2  # noqa: E402
from src.utils import schema  # noqa: E402
from src.utils.config_hydra import (  # noqa: E402
    get_virus_config_hydra,
    load_function_metadata,
    print_config_summary,
    save_config,
)
from src.utils.metadata_enrichment import enrich_prot_data_with_metadata  # noqa: E402
from src.utils.path_utils import load_dataframe  # noqa: E402
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds  # noqa: E402

# pos-side hash used to join clusters AND as the cooccurrence/cluster key.
# aa=protein, nt_cds=CDS, nt_ctg=contig — all md5 of the respective sequence.
# Single source of truth: the schema registry.
_POS_HASH = {a: s.hash_col for a, s in schema.SCHEMA.items()} # "a" for alphabet, "s" for schema

# Per-protein source columns copied into each pair side (a/b) of _PAIR_COLUMNS.
_SIDE_SRC = ['assembly_id', 'brc_fea_id', 'genbank_ctg_id', 'prot_seq', 'ctg_dna_seq',
             'canonical_segment', 'function', 'prot_hash', 'ctg_dna_hash']
_SIDE_RENAME = {'assembly_id': 'assembly_id', 'brc_fea_id': 'brc', 'genbank_ctg_id': 'ctg',
                'prot_seq': 'prot_seq', 'ctg_dna_seq': 'ctg_dna_seq', 'canonical_segment': 'seg',
                'function': 'func', 'prot_hash': 'prot_hash', 'ctg_dna_hash': 'ctg_dna_hash'}


def build_frontend(
    config,
    input_file: Path,  # protein_final.parquet
    schema_pair_full: tuple,
    cds_final_path: Path = None) -> pd.DataFrame:
    """v2's protein-level front-end, narrowed to schema_pair (same helpers as v2,
    so the population matches v2 for the same bundle).

    Pipeline (mirrors the v2 CLI `dataset_segment_pairs.py`):
        load protein_final.parquet -> attach_dna (ctg_dna_seq/hash) -> enrich ->
        drop_ambiguous_subtype (if enabled) -> filter_by_metadata ->
        selected_functions/schema narrow -> prot_hash (if missing).
    subtype balancing and max_isolates are NOT wired (if a bundle sets them raise error).

    `cds_final_path` (nt_cds only): attach `cds_dna_hash` from that cds_dna_final AFTER
    the narrow, so create_positive_pairs_v2(pair_key_alphabet='nt_cds') can key on it.
    None for aa/nt_ctg (cds_dna_hash_a/b stay empty).

    protein_final is loaded for EVERY alphabet; per-slot hashes are attached here:
        - prot_hash     -> always (computed if missing)         [aa key]      (protein)
        - ctg_dna_hash  -> always, via attach_ctg_dna_to_prot_df    [nt_ctg key]  (contig DNA)
        - cds_dna_hash  -> only when cds_final_path set         [nt_cds key]  (CDS DNA)
    The cluster_alphabet (set by the caller) later selects which of these keys
    pair_key / dedup / cluster-join.
    """
    subsel = getattr(config.dataset, 'subtype_selection', None)
    if subsel is not None and str(getattr(subsel, 'mode', 'natural')) == 'balanced':
        raise NotImplementedError(
            "dataset.subtype_selection.mode=balanced is not yet wired into the 2D-CD builder.")
    if getattr(config.dataset, 'max_isolates_to_process', None):
        raise NotImplementedError(
            "dataset.max_isolates_to_process is not yet wired into the 2D-CD builder.")

    # Load protein_final.parquet and attach ctg_dna_{seq,hash} via ctg_dna_final.parquet
    df = load_dataframe(input_file) # protein_final.parquet
    df = attach_ctg_dna_to_prot_df(df, input_file) # attach ctg_dna_{seq,hash} to protein df
    df = enrich_prot_data_with_metadata(df, project_root=PROJ) # host, year, hn_subtype, ...
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
    if 'prot_hash' not in df.columns:
        df['prot_hash'] = df['prot_seq'].map(lambda s: hashlib.md5(str(s).encode()).hexdigest())
    if cds_final_path is not None:
        df = attach_cds_dna_hash_to_prot_df(df, cds_final_path) # cds_dna_hash (nt_cds)
    return df


def assign_atoms_prod(
    pos: pd.DataFrame,
    cluster_lookup:
    pd.DataFrame,
    pos_hash_col: str, *,
    edge_cut: dict | None = None):
    """Attach cluster ids + the 2D routing atom (`cc_id`/`atom_id`) to the positive pairs
    (the dataset-builder path, vs the CV harness's `_cv_sampling.assign_atoms`, which
    resolves clusters through the membership table rather than a cluster-parquet lookup).

    Atom = one CC on (cluster_id_a, cluster_id_b), so `atom_id == cc_id`. Two modes:
    - natural (`edge_cut` None/disabled): one atom per whole CC.
    - edge-cut: `_megacc_cut.fragment_until` bisects the mega-CC, dropping straddling pairs to
      grow the atom count within a drop budget. `cc_id`/`atom_id` become the post-cut fragment;
      the pre-cut CC is kept on `natural_cc_id` (analysis-only -- the fold CSVs re-select
      `_PAIR_COLUMNS`, which drops it).

    `edge_cut` (when enabled): `{cut_method, target_atoms, max_drop_frac, seed}`. Returns
    `(pos_with_ids, cc_summary)`; the edge-cut run adds `cc_summary['edge_cut']` (the cut audit).
    See docs/plans/2026-07-17_2d_cc_edge_cut_fragmentation_plan.md.
    """
    pos_ids, attach_audit = attach_cluster_ids(pos, cluster_lookup, pos_hash_col=pos_hash_col)
    pos_ids = pos_ids.copy()

    # Natural CC atoms on (cluster_id_a, cluster_id_b).
    cc_id, cc_summary = cluster_ccs(pos_ids, col_a='cluster_id_a', col_b='cluster_id_b')
    pos_ids['cc_id'] = cc_id.to_numpy()

    if edge_cut and edge_cut.get('enabled'):
        # Grow the atom count: bisect the mega-CC and drop straddling pairs within a drop budget.
        # cc_id/atom_id become the post-cut fragment; natural_cc_id keeps the pre-cut CC (analysis).
        pos_ids['natural_cc_id'] = pos_ids['cc_id']
        natural_cc_sizes = pos_ids.groupby('natural_cc_id').size()   # pre-cut natural CC sizes (ALL pairs)
        kept_pos, _dropped_pos, cut_audit = fragment_until(
            pos_ids, col_a='cluster_id_a', col_b='cluster_id_b',
            cut_method=edge_cut['cut_method'], seed=edge_cut['seed'],
            stop_fn=stop_at_n_atoms(edge_cut['target_atoms']),
            max_drop_frac=edge_cut['max_drop_frac']
        )
        pos_ids = kept_pos.reset_index(drop=True)
        # Re-derive atoms on the fragmented (kept) pairs -- each fragment is a CC == atom.
        cc_id, cc_summary = cluster_ccs(pos_ids, col_a='cluster_id_a', col_b='cluster_id_b')
        pos_ids['cc_id'] = cc_id.to_numpy()
        cc_summary['edge_cut'] = cut_audit  # full fragment_until audit (cut_method/seed/max_drop_frac/per_cut)
        # Faithful before/after for the 2D CC-size barplots: natural CCs on ALL pre-cut pairs vs
        # fragments on the kept pairs (the dropped straddling pairs show up as the difference).
        cc_summary['cc_sizes'] = {'cc_pair_sizes.csv': natural_cc_sizes,
                                  'cc_pair_sizes_post_edge_cut.csv': pos_ids.groupby('cc_id').size()}
        print(f"  edge_cut ({edge_cut['cut_method']}, target {edge_cut['target_atoms']} atoms): dropped "
              f"{cut_audit['pairs_dropped']:,} straddling pairs ({cut_audit['dropped_frac']:.1%}); "
              f"atoms -> {cut_audit['n_atoms']:,} [{cut_audit['stopped_reason']}]")

    pos_ids['atom_id'] = pos_ids['cc_id']
    cc_summary['n_dropped_cluster_join'] = attach_audit['n_input'] - attach_audit['n_kept']
    return pos_ids, cc_summary


def _side_rep(df: pd.DataFrame, func: str, suffix: str, key_col: str = 'prot_hash') -> pd.DataFrame:
    """{one row per `key_col`} of per-side fields for `func`, renamed to *_<suffix>.

    `key_col` is the alphabet's per-slot hash column the negative enrichment joins
    on (aa: prot_hash, nt_ctg: ctg_dna_hash, nt_cds: cds_dna_hash). First occurrence
    per key is the representative (matches v2's keep='first'). Within one function a
    DNA hash maps to exactly one protein, so the rep's other columns are unambiguous.
    """
    cols = list(_SIDE_SRC) + (['cds_dna_hash'] if 'cds_dna_hash' in df.columns else [])
    rep = df[df['function'] == func][cols].drop_duplicates(key_col, keep='first').copy()
    ren = {k: f'{v}_{suffix}' for k, v in _SIDE_RENAME.items()}
    if 'cds_dna_hash' in df.columns:
        ren['cds_dna_hash'] = f'cds_dna_hash_{suffix}'
    return rep.rename(columns=ren)


def compute_negative_infeasible_ccs(
    pos_ids: pd.DataFrame, cooccur: set,
    hash_a_col: str = 'prot_hash_a',
    hash_b_col: str = 'prot_hash_b') -> set:
    """CCs from which no within-CC negative can be drawn — the drop set for
    `drop_negative_infeasible_ccs`, computed structurally on the positives.

    A CC is negative-infeasible iff every recombination of its distinct slot-A x
    slot-B sequences is a co-occurrence (in `cooccur`): then every candidate
    within-CC negative reconstructs a true pair. The singleton CC (one unique
    pair_key -> one distinct-a x one distinct-b) is the base case; the superset
    also holds dense CCs where every cross pairing co-occurs.

    Seed-independent, and mirrors `sample_random_within_cc_negatives`' feasibility
    exactly: any combo (a, b) not in `cooccur` is drawable from two *distinct*
    isolates, because a same-isolate combo would itself be that isolate's positive
    (hence in `cooccur`). Early-exits per CC on the first drawable negative, so
    feasible CCs (incl. the mega-CC) cost O(1) in practice.
    """
    infeasible = set()
    for cc, g in pos_ids.groupby('cc_id'):
        a_vals = g[hash_a_col].astype(str).unique()
        b_vals = g[hash_b_col].astype(str).unique()
        feasible = any(canonical_pair_key(a, b) not in cooccur
                       for a in a_vals for b in b_vals)
        if not feasible:
            infeasible.add(int(cc))
    return infeasible


def within_cc_negatives(pos_ids: pd.DataFrame, iso: pd.DataFrame, cooccur: set,
                        df: pd.DataFrame, schema_pair_full: tuple, *,
                        neg_to_pos_ratio: float, seed: int,
                        hash_col: str = 'prot_hash') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw random negatives inside each CC, from that CC's isolate pool, enriched to
    `_PAIR_COLUMNS`.

    Each CC's budget is `round(neg_to_pos_ratio * n_pos_in_cc)`, so negatives are proportional to
    positives per CC and every negative inherits its CC's `atom_id`. Negative-infeasible CCs are
    expected to be dropped upstream when `drop_negative_infeasible_ccs` (see
    `compute_negative_infeasible_ccs`); a CC that still yields none simply contributes none.

    Args:
        pos_ids: positives carrying `cc_id` and `atom_id`.
        iso: per-CC isolate pool from `build_cc_isolate_pool`, restricted to the front-end population.
        cooccur: canonical pair_keys of all observed positives (rejection set).
        df: front-end protein frame, used to enrich bare hashes.
        schema_pair_full: (slot-a function, slot-b function), full names.
        neg_to_pos_ratio: per-CC budget as a multiple of that CC's positives.
        seed: base seed; CC `c` samples with `seed + c`.
        hash_col: the alphabet's per-slot hash column (aa: `prot_hash`).

    Returns:
        (negatives in `_PAIR_COLUMNS` + `cc_id`/`atom_id`,
         cc_log with columns cc_id, n_pos, n_isolates, budget, n_neg).
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
        row['n_neg'] = n_neg
        if n_neg:
            raw.append(neg.assign(cc_id=int(cc), atom_id=cc_to_atom[cc]))
        log_rows.append(row)

    cc_log = pd.DataFrame(log_rows)
    if not raw:
        return pd.DataFrame(columns=list(_PAIR_COLUMNS) + ['cc_id', 'atom_id']), cc_log

    neg_all = pd.concat(raw, ignore_index=True)  # hash_a, hash_b, neg_regime, metadata_match_count, cc_id, atom_id
    # The sampler's hash_a/hash_b carry the alphabet's per-slot hash (aa: prot_hash,
    # nt_ctg: ctg_dna_hash); key the enrichment merge AND pair_key on it so negatives
    # match the positives' pair_key_alphabet (not a hardcoded protein pair_key).
    ha_col, hb_col = f'{hash_col}_a', f'{hash_col}_b'
    ra, rb = _side_rep(df, fa, 'a', hash_col), _side_rep(df, fb, 'b', hash_col)
    out = (neg_all.rename(columns={'hash_a': ha_col, 'hash_b': hb_col})
           .merge(ra, on=ha_col, how='left')
           .merge(rb, on=hb_col, how='left'))
    miss = out[['assembly_id_a', 'assembly_id_b']].isna().any(axis=1)
    if miss.any():
        print(f"WARNING: dropping {int(miss.sum()):,} negatives whose sequence is "
              f"absent from the (filtered) protein frame.")
        out = out[~miss].reset_index(drop=True)
    a = out[ha_col].astype(str).to_numpy()
    b = out[hb_col].astype(str).to_numpy()
    out['pair_key'] = np.where(a <= b, a, b) + '__' + np.where(a <= b, b, a)
    out['label'] = 0
    for c in ('cds_dna_hash_a', 'cds_dna_hash_b'):
        if c not in out.columns:
            out[c] = pd.NA
    keep = list(_PAIR_COLUMNS) + ['cc_id', 'atom_id']
    return out[keep].reset_index(drop=True), cc_log


def _carve_val_atoms(tv: pd.DataFrame, val_ratio: float, n_total: int, seed: int):
    """Carve val out of one fold's non-test rows by taking WHOLE atoms, so no atom straddles
    train/val.

    Atoms are shuffled with `seed` and accumulated into val until val reaches `val_ratio` of
    `n_total` -- the whole set, not just `tv` -- which keeps the val fraction comparable across folds.

    Args:
        tv: one fold's non-test rows; must carry an `atom_id` column.
        val_ratio: val size target, as a fraction of `n_total`.
        n_total: row count of the whole set the folds were built from.
        seed: seeds the atom shuffle.

    Returns:
        (train, val) -- row-disjoint frames partitioning `tv`, original index preserved.
    """
    rng = np.random.RandomState(seed)
    atoms = tv['atom_id'].drop_duplicates().to_numpy()
    rng.shuffle(atoms)
    sizes = tv.groupby('atom_id').size()

    # Take whole atoms in shuffled order until val reaches its target row count.
    target = val_ratio * n_total
    val_atoms, acc = set(), 0
    for a in atoms:
        if acc >= target:
            break
        val_atoms.add(a)
        acc += int(sizes[a])

    is_val = tv['atom_id'].isin(val_atoms)
    return tv[~is_val], tv[is_val]


def groupkfold_by_atom(pairs: pd.DataFrame, k_folds: int, val_ratio: float, seed: int) -> list:
    """Partition `pairs` into k folds by GroupKFold on `atom_id`, carving val group-aware.

    Whole atoms stay in one split everywhere: GroupKFold keeps an atom out of every fold but one,
    and `_carve_val_atoms` moves whole atoms into val. Both negative scopes route through this.
    Under within_cc, `_partition_full` passes positives + pre-built negatives, which already carry
    their CC's `atom_id` and so travel with it; under within_fold, `make_folds_within_fold` passes
    positives only and draws each split's negatives afterwards.

    Args:
        pairs: rows to partition; must carry an `atom_id` column.
        k_folds: number of folds (K).
        val_ratio: val size target, as a fraction of `len(pairs)`.
        seed: seeds the val atom shuffle. It does not affect the test-fold assignment, which
            `GroupKFold(shuffle=False)` derives deterministically from the atom sizes.

    Returns:
        list of k (train, val, test) frames, index reset; together they partition `pairs`.
    """
    # shuffle=False keeps GroupKFold's size balancing -- largest atom to the lightest fold, so
    # every fold receives one of the k largest atoms. shuffle=True instead cuts the atoms into
    # equal-COUNT chunks, which unbalances the folds badly when atom sizes are skewed.
    gkf = GroupKFold(n_splits=k_folds, shuffle=False)
    groups = pairs['atom_id'].to_numpy()
    n_total = len(pairs)
    folds = []
    for train_val_idx, test_idx in gkf.split(pairs, groups=groups):
        test = pairs.iloc[test_idx]
        train_val = pairs.iloc[train_val_idx]
        train, val = _carve_val_atoms(train_val, val_ratio, n_total, seed)
        train, val, test = train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
        folds.append((train, val, test))
    return folds


def within_fold_negatives(
    split_pos: pd.DataFrame,
    cooccur: set,
    df: pd.DataFrame,
    schema_pair_full: tuple, *,
    neg_to_pos_ratio: float,
    seed: int,
    hash_col: str = 'prot_hash') -> pd.DataFrame:
    """Draw within-fold negatives for ONE split: a random positive's slot-a sequence paired with
    another positive's slot-b sequence, both taken from THIS split's positives.

    Rejects true co-occurrences and duplicates. CC membership is not consulted, so a negative may
    fall within one CC or across CCs; either way both endpoints stay in-split, so the fold remains
    cluster-disjoint. Unlike a within-CC negative this does NOT remove the cluster shortcut.

    Callers pass one split at a time, and the `seen` dedup set is per call -- yet the same negative
    cannot appear in two splits of a fold, because every row carrying a given hash shares one
    `atom_id` (a hash joins to one cluster, a cluster sits in one CC, and the edge cut puts each
    cluster node in one fragment), so the splits' positive hash sets are disjoint.

    Args:
        split_pos: this split's positive rows.
        cooccur: canonical pair_keys of all observed positives; a draw hitting one is rejected.
        df: front-end protein frame, used to enrich bare hashes to `_PAIR_COLUMNS`.
        schema_pair_full: (slot-a function, slot-b function), full names.
        neg_to_pos_ratio: budget = round(ratio * len(split_pos)).
        seed: seeds the reject sampler.
        hash_col: the alphabet's per-slot hash column (aa: `prot_hash`).

    Returns:
        negatives in `_PAIR_COLUMNS`, index reset; empty frame if none could be drawn.
    """
    fa, fb = schema_pair_full
    ha_col, hb_col = f'{hash_col}_a', f'{hash_col}_b'  # alphabet's per-slot hash (aa: prot_hash)
    a = split_pos[ha_col].astype(str).to_numpy()
    b = split_pos[hb_col].astype(str).to_numpy()
    budget = int(round(neg_to_pos_ratio * len(split_pos))) # num negatives to sample
    if len(a) < 2 or budget <= 0:
        return pd.DataFrame(columns=list(_PAIR_COLUMNS))

    rng = np.random.RandomState(seed)
    na, nb, seen = [], [], set()  # neg slot-a, neg slot-b, seen neg pair_keys
    placed, attempts, max_attempts = 0, 0, budget * 50 + 200 # reject-sampling ceiling: ~50 attempts + 200 floor for tiny budgets
    while placed < budget and attempts < max_attempts:
        attempts += 1
        ha, nbh = a[rng.randint(len(a))], b[rng.randint(len(b))]
        pk = canonical_pair_key(ha, nbh) # canonical pair_key --> used to reject sampled negatives that match existing positives
        if pk in cooccur or pk in seen:
            continue  # reject sampled positives and negative duplicates
        seen.add(pk)
        na.append(ha)
        nb.append(nbh)
        placed += 1
    if not na:
        return pd.DataFrame(columns=list(_PAIR_COLUMNS))
    out = pd.DataFrame({ha_col: na, hb_col: nb})

    ra = _side_rep(df, fa, 'a', hash_col) # side-a lookup (one row per hash) to enrich the bare hash_a negatives
    rb = _side_rep(df, fb, 'b', hash_col) # side-b lookup (one row per hash) to enrich the bare hash_b negatives
    out = out.merge(ra, on=ha_col, how='left').merge(rb, on=hb_col, how='left')
    aa = out[ha_col].astype(str).to_numpy()
    bb = out[hb_col].astype(str).to_numpy()
    out['pair_key'] = np.where(aa <= bb, aa, bb) + '__' + np.where(aa <= bb, bb, aa)

    out['label'] = 0  # neg label
    out['neg_regime'] = pd.NA  # placeholder for regime-targeted negatives (not wired)
    out['metadata_match_count'] = pd.NA  # TODO: a placeholder?

    # Assign pd.NA to cds_dna_hash_a/b since ... [TODO]
    for c in ('cds_dna_hash_a', 'cds_dna_hash_b'):
        if c not in out.columns:
            out[c] = pd.NA

    return out[list(_PAIR_COLUMNS)].reset_index(drop=True)


def make_folds_within_fold(
    pos_full: pd.DataFrame,
    k_folds: int,
    val_ratio: float,
    seed: int, *,
    neg_to_pos_ratio: float,
    cooccur: set,
    df: pd.DataFrame,
    schema_pair_full: tuple,
    hash_col: str = 'prot_hash'):
    """Fold-maker for the within_fold scope: GroupKFold the POSITIVES by atom, then draw each
    split's negatives from its own positives.

    Negatives must come after the split, since `within_fold_negatives` samples from the split's
    positives. Both endpoints stay in-split, so folds remain cluster-disjoint. Routing is
    `groupkfold_by_atom`.

    Args:
        pos_full: positive rows only, carrying `atom_id`.
        k_folds: number of folds (K).
        val_ratio: val size target, as a fraction of `len(pos_full)`.
        seed: seeds the routing; each split's negatives use `seed + fold*100 + split`.
        neg_to_pos_ratio: negative budget per split, as a multiple of that split's positives.
        cooccur: canonical pair_keys of all observed positives (rejection set).
        df: front-end protein frame, used to enrich negatives.
        schema_pair_full: (slot-a function, slot-b function), full names.
        hash_col: the alphabet's per-slot hash column.

    Returns:
        list of k (train, val, test) frames in `_PAIR_COLUMNS`, each positives + its own negatives.
    """
    cols = list(_PAIR_COLUMNS)
    pos_folds = groupkfold_by_atom(pos_full, k_folds, val_ratio, seed)

    folds = []
    for fold_id, pos_splits in enumerate(pos_folds):
        splits = []
        for split_id, split_pos in enumerate(pos_splits):  # (train, val, test)
            neg = within_fold_negatives(
                split_pos, cooccur, df, schema_pair_full,
                neg_to_pos_ratio=neg_to_pos_ratio,
                seed=seed + fold_id * 100 + split_id, hash_col=hash_col
            )
            splits.append(pd.concat([split_pos[cols], neg[cols]], ignore_index=True).reset_index(drop=True))
        folds.append(tuple(splits))

    return folds


@dataclass(frozen=True)
class CCSpec:
    """Resolved knobs for one CC build (produced by `_resolve_spec`)."""
    config_bundle: str
    alphabet: str
    pair_key_alphabet: str
    k_folds: int
    n_repeats: int
    neg_to_pos_ratio: float
    val_ratio: float
    negative_scope: str
    drop_negative_infeasible_ccs: bool
    m_pos: int | None  # None = no cap (keep all pairs per CC); else cap rows-per-CC
    max_atoms: int | None  # None = no cap; caps #atoms for size-controlled sweeps
    edge_cut: dict | None  # None = disabled; else {cut_method, target_atoms, max_drop_frac, seed} for fragment_until
    seed: int
    cluster_id_path: Path
    threshold: str
    fa: str # full function name ('a' side of the pair)
    fb: str # full function name ('b' side of the pair)
    sa: str # short function name ('a' side of the pair)
    sb: str # short function name ('b' side of the pair)
    membership_path: Path | None  # within_cc isolate-pool membership override (None = _MEMB set-cover)
    fold_assignment: str  # 'groupkfold' (default) | 'leave_cc_out' (each of the k largest CCs = one test fold)
    tail_ccs_to_train: bool  # leave_cc_out: keep the non-test (tail) CCs in train (both arms)
    paired_random: bool  # leave_cc_out: also emit a size-matched random arm reusing the same `full`


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config_bundle', required=True,
                   help='Hydra bundle; must set dataset.split_strategy.mode=cluster_disjoint_cc.')
    p.add_argument('--override', nargs='+', default=None,
                   help='Hydra-style dotlist overrides (e.g., dataset.n_folds=5).')
    p.add_argument('--protein_final', default=None,
                   help='Override protein_final path (default: alongside cluster_id_path).')
    p.add_argument('--out_dir', type=Path, required=True)
    return p.parse_args()


def _resolve_schema_pair(config, ds) -> tuple:
    """(fa, fb, sa, sb): the two schema_pair functions canonicalized to protein_order + short names."""
    meta = load_function_metadata(PROJ / 'conf' / 'virus' / 'flu.yaml')
    schema_raw = [str(x) for x in ds.schema_pair]
    if len(schema_raw) != 2 or schema_raw[0] == schema_raw[1]:
        raise ValueError(f"dataset.schema_pair must be two distinct functions; got {schema_raw!r}.")
    order = list(config.virus.protein_order)
    fa, fb = (schema_raw if order.index(schema_raw[0]) <= order.index(schema_raw[1])
              else [schema_raw[1], schema_raw[0]])
    sa, sb = meta.function_to_short[fa], meta.function_to_short[fb]
    return fa, fb, sa, sb


def _resolve_spec(args, config) -> CCSpec:
    """Validate the bundle for cluster_disjoint_cc + resolve all knobs into a CCSpec.
    No code-level defaults: every knob must be set in conf/... — a missing/invalid key raises.
    """
    ds, ss = config.dataset, config.dataset.split_strategy

    # this builder supports only the cluster_disjoint_cc mode
    mode = OmegaConf.select(config, 'dataset.split_strategy.mode')
    if mode != 'cluster_disjoint_cc':
        raise ValueError(
            f"dataset_pairs_cc requires dataset.split_strategy.mode='cluster_disjoint_cc'; got {mode!r}.")

    # nt_cds attaches cds_dna_hash in build_frontend, so positives carry populated
    # cds_dna_hash_{a,b} for the cluster join (pair_key_alphabet == cluster_alphabet).
    _ENABLED_ALPHABETS = ('aa', 'nt_cds', 'nt_ctg')
    if 'cluster_alphabet' not in ss:
        raise ValueError("dataset.split_strategy.cluster_alphabet must be set for cluster_disjoint_cc.")
    alphabet = str(ss.cluster_alphabet)
    if alphabet not in _ENABLED_ALPHABETS:
        raise NotImplementedError(
            f"cluster_alphabet={alphabet!r} is not a known molecule axis for the 2D-CD builder "
            f"(allowed: {list(_ENABLED_ALPHABETS)}).")

    # Single-axis builder: pair_key is always the cluster alphabet
    pair_key_alphabet = alphabet

    # Placeholder (regime-targeted negatives)
    if getattr(ds, 'negative_sampling', None) is not None:
        raise NotImplementedError(
            "It's a placeholder for regime-targeted within-CC negatives (dataset.negative_sampling)")

    # Placeholder (repeated CV)
    n_repeats = int(getattr(ds, 'n_repeats', None) or 1)
    if n_repeats != 1:
        raise NotImplementedError("It's a placeholder for repeated CV; currently n_repeats=1 is wired.")

    k_folds = ds.n_folds   # the >= 2 check rejects a single holdout split
    if not k_folds or int(k_folds) < 2:
        raise ValueError(f"dataset.n_folds must be an int >= 2 for the 2D-CD CV builder; got {k_folds!r}.")

    drop_negative_infeasible_ccs = ss.drop_negative_infeasible_ccs  # raises if absent
    if not isinstance(drop_negative_infeasible_ccs, bool):
        raise ValueError(f"dataset.split_strategy.drop_negative_infeasible_ccs must be a bool; "
                         f"got {drop_negative_infeasible_ccs!r}.")

    negative_scope = ss.negative_scope  # raises if absent
    if negative_scope not in ('within_cc', 'within_fold'):
        raise ValueError(f"dataset.split_strategy.negative_scope must be 'within_cc' or "
                         f"'within_fold'; got {negative_scope!r}.")

    m_pos = ss.m_pos_per_cc  # raises if absent (a positive int, or null = no cap = keep all pairs per CC)
    if m_pos is not None and (not isinstance(m_pos, int) or m_pos < 1):
        raise ValueError(f"dataset.split_strategy.m_pos_per_cc must be a positive int or null; got {m_pos!r}.")

    # Optional atom-count cap for size-controlled sweeps (default null = no cap).
    _max_atoms = OmegaConf.select(config, 'dataset.split_strategy.max_atoms')
    max_atoms = int(_max_atoms) if _max_atoms is not None else None
    if max_atoms is not None and max_atoms < 1:
        raise ValueError(f"dataset.split_strategy.max_atoms must be a positive int or null; got {max_atoms!r}.")

    seed = resolve_process_seed(config, 'datasets')
    if seed is None:
        raise ValueError("Could not resolve a master seed (resolve_process_seed returned None).")

    # Optional edge-cut fragmentation: grow the atom count by bisecting the mega-CC.
    # Default off (existing bundles unaffected). See _megacc_cut.fragment_until.
    edge_cut = None
    if bool(OmegaConf.select(config, 'dataset.split_strategy.edge_cut.enabled', default=False)):
        ec_target = OmegaConf.select(config, 'dataset.split_strategy.edge_cut.target_atoms')
        if ec_target is None or int(ec_target) < 1:
            raise ValueError("edge_cut.enabled=true requires split_strategy.edge_cut.target_atoms (positive int).")
        ec_method = str(OmegaConf.select(config, 'dataset.split_strategy.edge_cut.cut_method') or 'spectral')
        if ec_method not in ('spectral', 'kl'):
            raise ValueError(f"split_strategy.edge_cut.cut_method must be 'spectral' or 'kl'; got {ec_method!r}.")
        ec_max_drop_frac = OmegaConf.select(config, 'dataset.split_strategy.edge_cut.max_drop_frac')
        edge_cut = {'enabled': True, 'cut_method': ec_method, 'target_atoms': int(ec_target),
                    'max_drop_frac': float(ec_max_drop_frac) if ec_max_drop_frac is not None else 1.0, 'seed': seed}

    if 'cluster_id_path' not in ss:
        raise ValueError("dataset.split_strategy.cluster_id_path must be set for cluster_disjoint_cc.")
    cluster_id_path = Path(str(ss.cluster_id_path))
    if not cluster_id_path.is_absolute():
        cluster_id_path = PROJ / cluster_id_path
    threshold = cluster_id_path.parent.name  # the tXXX dir, e.g. 't099'

    # OOD-vs-random paired-CV knobs (all default to the existing single-arm groupkfold behavior).
    _memb = OmegaConf.select(config, 'dataset.split_strategy.membership_path')
    membership_path = None
    if _memb is not None:
        membership_path = Path(str(_memb))
        if not membership_path.is_absolute():
            membership_path = PROJ / membership_path
    fold_assignment = str(OmegaConf.select(
        config, 'dataset.split_strategy.fold_assignment', default='groupkfold'))
    if fold_assignment not in ('groupkfold', 'leave_cc_out'):
        raise ValueError("dataset.split_strategy.fold_assignment must be 'groupkfold' or "
                         f"'leave_cc_out'; got {fold_assignment!r}.")
    tail_ccs_to_train = bool(OmegaConf.select(
        config, 'dataset.split_strategy.tail_ccs_to_train', default=True))
    paired_random = bool(OmegaConf.select(
        config, 'dataset.split_strategy.paired_random', default=False))
    if fold_assignment == 'leave_cc_out' and negative_scope != 'within_cc':
        raise ValueError("fold_assignment='leave_cc_out' requires negative_scope='within_cc' (the "
                         f"reusable within-CC negative pool); got negative_scope={negative_scope!r}.")

    fa, fb, sa, sb = _resolve_schema_pair(config, ds)
    return CCSpec(
        config_bundle=args.config_bundle, alphabet=alphabet, pair_key_alphabet=pair_key_alphabet,
        k_folds=int(k_folds), n_repeats=n_repeats, neg_to_pos_ratio=float(ds.neg_to_pos_ratio),
        val_ratio=float(ds.val_ratio), negative_scope=negative_scope,
        drop_negative_infeasible_ccs=drop_negative_infeasible_ccs, m_pos=m_pos, max_atoms=max_atoms,
        edge_cut=edge_cut, seed=seed,
        cluster_id_path=cluster_id_path, threshold=threshold, fa=fa, fb=fb, sa=sa, sb=sb,
        membership_path=membership_path, fold_assignment=fold_assignment,
        tail_ccs_to_train=tail_ccs_to_train, paired_random=paired_random)


def _subsample_atoms(pos_ids: pd.DataFrame, max_atoms, seed: int) -> pd.DataFrame:
    """Cap the atom count at `max_atoms` by keeping a seeded random subset of atom_ids
    (all rows of each kept atom). No-op when max_atoms is None or already within budget.

    Keys on atom_id (the final routing unit), which after edge-cut fragmentation equals the
    post-cut fragment's cc_id (see assign_atoms_prod / _megacc_cut.fragment_until).
    """
    if max_atoms is None:
        return pos_ids
    atoms = pos_ids['atom_id'].unique()
    if len(atoms) <= max_atoms:
        return pos_ids
    rng = np.random.RandomState(seed)
    keep = set(rng.choice(atoms, size=int(max_atoms), replace=False))
    return pos_ids[pos_ids['atom_id'].isin(keep)].reset_index(drop=True)


def _build_positives(config, spec: CCSpec, args):
    """Front-end -> positive pairs -> cooccurrence set -> CC atoms -> drop negative-infeasible CCs.

    Regardless of the specified alphabet (aa, nt_cds, nt_ctg), we load protein_final (it carries prot_hash).
    build_frontend ATTACHES the DNA hashes from sibling files: ctg_dna_hash from ctg_dna_final (always),
    cds_dna_hash from cds_dna_final (only if nt_cds is set as alphabet).
    The alphabet only picks which hash keys pair_key/dedup/cluster-join.
    We pass the path to build_frontend(), not a preloaded df, because build_frontend() does the load and
    input_file.parent locates cds_dna_final.

    Returns (df, pos_ids, cooccur, cc_sizes); pos_ids carries cluster_id_a/b + cc_id + atom_id
    (+ natural_cc_id under edge-cut). cc_sizes is the pre/post-cut CC-size dict for the barplots
    (None without a cut).
    """
    # Path to protein_final.parquet
    input_file = (Path(args.protein_final) if args.protein_final
                  else spec.cluster_id_path.parents[2] / 'protein_final.parquet')

    # nt_cds only: path to cds_dna_final.parquet, which build_frontend uses to attach cds_dna_hash.
    # Default: beside protein_final. split_strategy.cds_final_path can override it (no bundle sets it today).
    cds_final_path = None
    if spec.alphabet == 'nt_cds':
        _cds = OmegaConf.select(config, 'dataset.split_strategy.cds_final_path')
        cds_final_path = Path(str(_cds)) if _cds else input_file.parent / 'cds_dna_final.parquet'
        if not cds_final_path.is_absolute():
            cds_final_path = PROJ / cds_final_path

    df = build_frontend(config, input_file, (spec.fa, spec.fb), cds_final_path=cds_final_path)
    print(f"  front-end: {len(df):,} protein rows / {df['assembly_id'].nunique():,} isolates ({spec.sa}+{spec.sb})")

    pos, _ = create_positive_pairs_v2(df, schema_pair=(spec.fa, spec.fb), pair_key_alphabet=spec.pair_key_alphabet)
    cooccur, _ = build_cooccurrence_set(df, hash_col=_POS_HASH[spec.alphabet])
    lookup = load_cluster_lookup(spec.cluster_id_path) # load cluster_id_{a,b} lookup df
    pos_ids, cc_summary = assign_atoms_prod(pos, lookup, _POS_HASH[spec.alphabet], edge_cut=spec.edge_cut)
    print(f"  positives: {len(pos):,} -> {len(pos_ids):,} after cluster join; "
          f"{cc_summary['n_atoms']:,} CCs; largest {cc_summary['largest_atom_pairs']:,} pairs")

    # Computed on the UNCAPPED positives (before the m_pos cap) so capping can't make a CC
    # look infeasible. See compute_negative_infeasible_ccs for the definition.
    neg_infeasible_ccs = compute_negative_infeasible_ccs(pos_ids, cooccur,
        hash_a_col=f'{_POS_HASH[spec.alphabet]}_a', hash_b_col=f'{_POS_HASH[spec.alphabet]}_b'
    )

    # Unified drop (both scopes): remove negative-infeasible CCs up front so every kept CC is
    # class-balanceable. When disabled, they survive as positives-only atoms (within_fold can
    # still give them cross-CC negatives; within_cc leaves them with positives only).
    if spec.drop_negative_infeasible_ccs and neg_infeasible_ccs:
        n0_cc = pos_ids['cc_id'].nunique()
        pos_ids = pos_ids[~pos_ids['cc_id'].isin(neg_infeasible_ccs)].reset_index(drop=True)
        print(f"  drop_negative_infeasible_ccs: dropped {len(neg_infeasible_ccs):,} CCs "
              f"({n0_cc:,} -> {pos_ids['cc_id'].nunique():,} kept).")

    # Optional size-control: cap #atoms AFTER the drop (so kept atoms stay feasible).
    # No-op unless dataset.split_strategy.max_atoms is set. See _subsample_atoms.
    if spec.max_atoms is not None:
        n0_atoms = pos_ids['atom_id'].nunique()
        pos_ids = _subsample_atoms(pos_ids, spec.max_atoms, spec.seed)
        print(f"  max_atoms={spec.max_atoms}: atoms {n0_atoms:,} -> {pos_ids['atom_id'].nunique():,}")

    return df, pos_ids, cooccur, cc_summary.get('cc_sizes')  # cc_sizes: pre/post-cut CC-size Series, or None


# =============================================================================
# OOD-vs-random paired CV -- experiment scaffolding, NOT the production path.
#
# Everything down to the closing banner serves one bundle,
# `flu_ha_na_cc_nt_cds_ood_ood_vs_random`: leave-one-atom-out folds against a
# size-matched random control, both partitioning the SAME rows so the split is
# the only difference between the arms. Reached only through
# `negative_scope: within_cc`, of which this block is the sole consumer.
# Design: docs/plans/2026-07-21_ood_vs_random_split_plan.md
# =============================================================================


def _carve_val_pairs(pairs: pd.DataFrame, val_ratio: float, seed: int):
    """Carve val out of one fold's non-test rows at ROW level, so atoms may straddle train/val.

    The row-level twin of `_carve_val_atoms` (defined above with the production routing), used by
    the two arms below, where only the test fold is held out and val is deliberately
    in-distribution.

    Args:
        pairs: one fold's non-test rows.
        val_ratio: val size target, as a fraction of `len(pairs)`.
        seed: seeds the row shuffle.

    Returns:
        (train, val) -- row-disjoint frames partitioning `pairs`, index reset.
    """
    shuf = pairs.sample(frac=1, random_state=np.random.RandomState(seed))
    n_val = int(round(val_ratio * len(shuf)))
    val, train = shuf.iloc[:n_val], shuf.iloc[n_val:]
    return train.reset_index(drop=True), val.reset_index(drop=True)


def pick_largest_atoms(full: pd.DataFrame, n: int) -> list:
    """The `n` atom_ids carrying the most POSITIVE pairs, largest first.

    Negatives are excluded because the within-CC budget is proportional to each CC's positive
    count, so counting them would rescale every atom by the same factor.

    Args:
        full: the pos+neg pool after any edge-cut fragmentation; carries `atom_id` and `label`.
        n: how many atom_ids to return.

    Returns:
        list of up to `n` atom_ids, most positive pairs first.
    """
    return full[full['label'] == 1].groupby('atom_id').size().nlargest(n).index.tolist()


def make_folds_leave_cc_out(full: pd.DataFrame, test_atom_ids, val_ratio: float, seed: int):
    """Leave-one-atom-out folds: each atom in `test_atom_ids` is the sole test fold once.

    Train is every other row of `full` (the remaining test atoms plus whatever tail `full` carries);
    val is a row-level carve of train, so atoms straddle train/val here by design -- only test is
    held out.

    Args:
        full: the pos+neg pool, carrying `atom_id`.
        test_atom_ids: atoms to rotate through the test slot, one fold each.
        val_ratio: val size target, as a fraction of the non-test rows.
        seed: base seed; fold i uses `seed + i`.

    Returns:
        list of len(test_atom_ids) (train, val, test) frames.
    """
    folds = []
    for i, atom in enumerate(test_atom_ids):
        is_test = full['atom_id'] == atom
        train, val = _carve_val_pairs(full[~is_test], val_ratio, seed + i)
        folds.append((train, val, full[is_test].reset_index(drop=True)))
    return folds


def make_folds_random(main_atom_pairs: pd.DataFrame, tail_atom_pairs: pd.DataFrame,
                      val_ratio: float, seed: int, *, per_fold_sizes):
    """Size-matched random control for the leave-one-atom-out arm: partitions ROWS, not atoms.

    Shuffles `main_atom_pairs` once and cuts it into consecutive test folds of `per_fold_sizes`, so
    each row is tested exactly once and the per-fold test sizes match the OOD arm over the same
    rows -- only the partition differs. Atoms straddle splits here, which is what makes this the
    in-distribution control.

    Args:
        main_atom_pairs: rows of the test atoms (the material the OOD arm tests on).
        tail_atom_pairs: rows appended to every train; may be empty.
        val_ratio: val size target, as a fraction of the non-test rows.
        seed: base seed; fold i uses `seed + i`.
        per_fold_sizes: test row count per fold; must sum to len(main_atom_pairs).

    Returns:
        list of len(per_fold_sizes) (train, val, test) frames.
    """
    shuf = main_atom_pairs.sample(frac=1, random_state=np.random.RandomState(seed)).reset_index(drop=True)
    if sum(per_fold_sizes) != len(shuf):
        raise ValueError(f"per_fold_sizes (sum {sum(per_fold_sizes)}) must sum to "
                         f"len(main_atom_pairs)={len(shuf)}; got {list(per_fold_sizes)}.")
    folds, start = [], 0
    for i, n_test in enumerate(per_fold_sizes):
        test = shuf.iloc[start:start + n_test]
        # Non-test = everything outside this fold's slice, plus the tail atoms.
        rest = pd.concat([shuf.iloc[:start], shuf.iloc[start + n_test:]], ignore_index=True)
        non_test = pd.concat([rest, tail_atom_pairs], ignore_index=True)
        train, val = _carve_val_pairs(non_test, val_ratio, seed + i)
        folds.append((train, val, test.reset_index(drop=True)))
        start += n_test
    return folds

def _partition_full(full: pd.DataFrame, spec: CCSpec) -> dict:
    """Partition the fixed pos+neg `full` into fold arms, per `spec.fold_assignment`.

    `groupkfold` gives one unnamed arm. `leave_cc_out` gives the `ood` arm -- each of the k largest
    atoms is the sole test fold once, with the remaining (tail) atoms in train unless
    `tail_ccs_to_train=false` -- plus, when `paired_random`, a size-matched `random` arm built from
    the SAME rows.

    Args:
        full: positives + within-CC negatives, carrying `atom_id`.
        spec: resolved build knobs.

    Returns:
        {arm_name: [(train, val, test), ...]}; '' names the single unnamed arm.
    """
    if spec.fold_assignment == 'groupkfold':
        # Each within-CC negative carries its CC's atom_id, so it travels with the atom and the
        # folds stay cluster-disjoint even though positives and negatives are routed together.
        folds = groupkfold_by_atom(full, spec.k_folds, spec.val_ratio, spec.seed)
        return {'': folds}

    # leave_cc_out: the k largest atoms rotate as the sole test fold.
    test_atom_ids = pick_largest_atoms(full, spec.k_folds)
    is_main = full['atom_id'].isin(test_atom_ids)
    main_atom_pairs, tail_atom_pairs = full[is_main], full[~is_main]
    ood_full = full if spec.tail_ccs_to_train else main_atom_pairs
    print(f"  leave_cc_out: {len(test_atom_ids)} test atoms {test_atom_ids} | "
          f"{len(main_atom_pairs):,} main + {len(tail_atom_pairs):,} tail pairs | "
          f"tail_ccs_to_train={spec.tail_ccs_to_train}")

    arms = {'ood': make_folds_leave_cc_out(ood_full, test_atom_ids, spec.val_ratio, spec.seed)}
    if spec.paired_random:
        # Match the OOD arm's per-fold test sizes so the two arms differ only in the partition.
        train_tail = tail_atom_pairs if spec.tail_ccs_to_train else full.iloc[0:0]
        per_fold_sizes = [len(test) for _, _, test in arms['ood']]
        arms['random'] = make_folds_random(
            main_atom_pairs, train_tail, spec.val_ratio, spec.seed,
            per_fold_sizes=per_fold_sizes
        )
    return arms


# === end OOD-vs-random paired CV ===


def _make_folds_for_scope(spec: CCSpec, df, pos_ids, cooccur, out_dir: Path) -> dict:
    """Draw negatives for the configured `negative_scope`, then partition into fold arms.

    within_cc: build the uncapped isolate pool, cap positives per CC, draw within-CC negatives,
        concat to one fixed `full` (writes cc_sampling_log.csv), then hand to `_partition_full`
        (groupkfold or leave_cc_out).
    within_fold: cap positives, then `make_folds_within_fold` -- one arm, negatives per split.

    Args:
        spec: resolved build knobs.
        df: front-end protein frame.
        pos_ids: positives carrying `cluster_id_a/b`, `cc_id`, `atom_id`.
        cooccur: canonical pair_keys of all observed positives (rejection set).
        out_dir: where the within_cc sampling log is written.

    Returns:
        {arm_name: [(train, val, test), ...]}; '' names the single unnamed arm.
    """
    # Isolate pool: within_cc only (within_fold draws negatives from each split's own positives,
    # no pool). Built from the FULL (uncapped) atom assignment so the pool covers every cluster
    # of each CC even when positives are capped.
    iso = None
    if spec.negative_scope == 'within_cc':
        c2a = {**dict(zip(pos_ids['cluster_id_a'].astype(str), pos_ids['atom_id'])),
               **dict(zip(pos_ids['cluster_id_b'].astype(str), pos_ids['atom_id']))}
        c2c = {**dict(zip(pos_ids['cluster_id_a'].astype(str), pos_ids['cc_id'])),
               **dict(zip(pos_ids['cluster_id_b'].astype(str), pos_ids['cc_id']))}
        iso = build_cc_isolate_pool(c2a, c2c, spec.sa, spec.sb, spec.alphabet, spec.threshold,
                                    membership_path=spec.membership_path)
        # The membership pool is corpus-level; restrict it to the front-end-filtered population so
        # within-CC negatives are drawn only from isolates present in df (otherwise a negative can
        # reference a sequence the filters dropped, then get discarded at enrich -> positive-only
        # atoms). Clusters stay corpus-level/stable; only the negative population follows df.
        n_pool0 = len(iso)
        iso = iso[iso['assembly_id'].isin(set(df['assembly_id'].astype(str)))].reset_index(drop=True)
        if len(iso) != n_pool0:
            print(f"  negative pool restricted to df isolates: {n_pool0:,} -> {len(iso):,} pool rows")

    if spec.m_pos:
        # Cap m_pos positives per CC: shuffle (seeded), rank within CC, keep first m. Deterministic,
        # keeps all columns, and dodges the groupby.apply grouping-column deprecation.
        rng = np.random.RandomState(spec.seed)
        shuf = pos_ids.sample(frac=1, random_state=rng).reset_index(drop=True)
        shuf['_rank'] = shuf.groupby('cc_id').cumcount()  # 0-based rank within each CC, in the shuffled (random) order
        pos_ids = shuf[shuf['_rank'] < spec.m_pos].drop(columns='_rank').reset_index(drop=True)  # keep m_pos random positives per CC/atom (m=1 -> one row per atom)
        print(f"  capped positives per CC at m_pos_per_cc={spec.m_pos}: {len(pos_ids):,} kept")

    if spec.negative_scope == 'within_cc':
        neg, cc_log = within_cc_negatives(
            pos_ids, iso, cooccur, df, (spec.fa, spec.fb), neg_to_pos_ratio=spec.neg_to_pos_ratio,
            seed=spec.seed, hash_col=_POS_HASH[spec.alphabet])
        # Defensive: with dropping enabled every kept CC is structurally feasible, so a CC with
        # budget>0 yet 0 sampled negatives means the random sampler under-filled; drop its positives
        # to keep folds balanced and warn. (With dropping disabled, 0-negative CCs are the
        # intentionally-kept infeasible ones -> left as positives-only.)
        if spec.drop_negative_infeasible_ccs and len(cc_log):
            undersampled = set(cc_log.loc[(cc_log['budget'] > 0) & (cc_log['n_neg'] == 0), 'cc_id'])
            if undersampled:
                print(f"WARNING: {len(undersampled):,} feasible CC(s) yielded 0 sampled negatives "
                      f"(sampler under-filled); dropping their positives.")
                pos_ids = pos_ids[~pos_ids['cc_id'].isin(undersampled)].reset_index(drop=True)
        print(f"  negatives: {len(neg):,} (within-CC)")
        pos_full = pos_ids.copy()
        pos_full['neg_regime'] = pd.NA
        pos_full['metadata_match_count'] = pd.NA
        keep = list(_PAIR_COLUMNS) + ['cc_id', 'atom_id']
        full = pd.concat([pos_full[keep], neg[keep]], ignore_index=True)
        print(f"  full set: {len(full):,} pairs ({int((full.label == 1).sum()):,} pos / "
              f"{int((full.label == 0).sum()):,} neg) across {full['atom_id'].nunique():,} atoms")
        cc_log.to_csv(out_dir / 'cc_sampling_log.csv', index=False)
        return _partition_full(full, spec)

    # within_fold: split positives by atom, then draw each split's negatives from its own positives.
    pos_full = pos_ids.copy()
    pos_full['neg_regime'] = pd.NA            # negative-only field (which negative regime); NA on positive rows
    pos_full['metadata_match_count'] = pd.NA  # negative-only field (pos<->neg metadata overlap); NA on positive rows
    print(f"  positives: {len(pos_full):,} across {pos_full['atom_id'].nunique():,} atoms; "
          f"within-fold negatives generated per split")

    # One unnamed arm: `fold_assignment` is not consulted here because `_resolve_spec` rejects
    # leave_cc_out under within_fold, leaving groupkfold as the only reachable value.
    folds = make_folds_within_fold(
        pos_full, spec.k_folds, spec.val_ratio, spec.seed, neg_to_pos_ratio=spec.neg_to_pos_ratio,
        cooccur=cooccur, df=df, schema_pair_full=(spec.fa, spec.fb), hash_col=_POS_HASH[spec.alphabet])
    return {'': folds}


def _write_output(out_dir: Path, folds, spec: CCSpec) -> None:
    """Write cv_info.json + per-fold {train,val,test}_pairs.csv + dataset_stats.json."""
    out_dir.mkdir(parents=True, exist_ok=True)  # arm subdir (out_dir/{ood,random}) may not exist yet
    cv_info = {'k_folds': spec.k_folds, 'n_repeats': spec.n_repeats, 'seed': spec.seed,
               'config_bundle': spec.config_bundle, 'schema_pair': [spec.sa, spec.sb],
               'alphabet': spec.alphabet, 'threshold': spec.threshold,
               'cluster_id_path': str(spec.cluster_id_path),
               'm_pos_per_cc': spec.m_pos, 'max_atoms': spec.max_atoms, 'edge_cut': spec.edge_cut,
               'neg_to_pos_ratio': spec.neg_to_pos_ratio,
               'pair_key_alphabet': spec.pair_key_alphabet, 'negative_scope': spec.negative_scope,
               'drop_negative_infeasible_ccs': spec.drop_negative_infeasible_ccs,
               'fold_assignment': spec.fold_assignment,
               'fold_dirs': [f'fold_{k}' for k in range(spec.k_folds)]}
    (out_dir / 'cv_info.json').write_text(json.dumps(cv_info, indent=2))
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


def _write_cc_sizes(out_dir: Path, sizes: pd.Series, filename: str) -> None:
    """Write `filename` (cc_id, n_pairs) from a per-unit size Series, largest first.
    Header is always (cc_id, n_pairs) so src/analysis/plot_cc_sizes.py reads either file."""
    cc = sizes.sort_values(ascending=False).rename('n_pairs')
    cc.index.name = 'cc_id'
    cc = cc.reset_index()
    cc.to_csv(out_dir / filename, index=False)
    print(f"  wrote {filename} ({len(cc):,} units, {int(cc['n_pairs'].sum()):,} pairs)")


def _write_cc_pair_sizes(out_dir: Path, pos_ids: pd.DataFrame, cc_sizes: dict | None = None) -> None:
    """Emit the 2D CC-size barplot input(s). Edge-cut runs pass `cc_sizes` = {filename: size_series}
    with the pre-cut natural sizes (cc_pair_sizes.csv, ALL pairs) and the post-cut fragment sizes
    (cc_pair_sizes_post_edge_cut.csv) -- the faithful before/after over the full fragmentation
    universe. Without a cut, one file grouped on the final atoms. plot_cc_sizes.py reads them."""
    if cc_sizes:
        for filename, sizes in cc_sizes.items():
            _write_cc_sizes(out_dir, sizes, filename)
    else:
        _write_cc_sizes(out_dir, pos_ids.groupby('cc_id').size(), 'cc_pair_sizes.csv')


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    config = get_virus_config_hydra(args.config_bundle, config_path=str(PROJ / 'conf'))
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    print_config_summary(config)

    spec = _resolve_spec(args, config)
    set_deterministic_seeds(spec.seed, cuda_deterministic=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(out_dir / 'resolved_config.yaml'))
    print(f"=== dataset_pairs_cc {spec.sa}-{spec.sb} {spec.alphabet} {spec.threshold} "
          f"(k={spec.k_folds}, ratio={spec.neg_to_pos_ratio}, m_pos_per_cc={spec.m_pos}, "
          f"drop_negative_infeasible_ccs={spec.drop_negative_infeasible_ccs}, seed={spec.seed}) ===")

    df, pos_ids, cooccur, cc_sizes = _build_positives(config, spec, args)
    _write_cc_pair_sizes(out_dir, pos_ids, cc_sizes)
    arms = _make_folds_for_scope(spec, df, pos_ids, cooccur, out_dir)
    for arm, folds in arms.items():
        arm_dir = out_dir / arm if arm else out_dir
        if arm:
            print(f"--- arm: {arm} -> {arm_dir} ---")
        _write_output(arm_dir, folds, spec)
    print(f"\nDone in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == '__main__':
    main()
