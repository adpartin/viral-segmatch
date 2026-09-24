"""Negative-pair samplers shared by the two dataset builders.

`within_fold_negatives` is selected by `split_strategy.negative_scope=within_fold`, and both
builders reach it: the 2D-CD builder (`dataset_pairs_cc.py`) and the v2 builder
(`dataset_segment_pairs_v2.py`) for its random, seq_disjoint and single-slot cluster_disjoint
modes. It lived in the 2D-CD builder, which made the v2 builder import one of its peers and
forced that import to be function-local to break a cycle. Nothing here imports either builder,
so both can import it plainly.

The other two samplers stay where their own routing lives: `within_cc_negatives` needs the CC
structure and belongs to the 2D-CD builder, and the coverage-first sampler belongs to v2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets._pair_helpers import _side_rep, canonical_pair_key
from src.utils import schema

# The pair-table schema, from the registry rather than from a builder, so this module depends on
# neither of them.
_PAIR_COLUMNS = schema.build_pair_columns()


def within_fold_negatives(
    split_pos: pd.DataFrame,
    cooccur: set,
    df: pd.DataFrame,
    schema_pair_full: tuple, *,
    neg_to_pos_ratio: float,
    seed: int,
    hash_col: str = 'prot_hash',
    seen: set | None = None) -> pd.DataFrame:
    """Draw within-fold negatives for ONE split: a random positive's slot-a sequence paired with
    another positive's slot-b sequence, both taken from THIS split's positives.

    Rejects true co-occurrences and duplicates. CC membership is not consulted, so a negative may
    fall within one CC or across CCs; either way both endpoints stay in-split, so the fold remains
    cluster-disjoint. Unlike a within-CC negative this does NOT remove the cluster shortcut.

    Pass ONE `seen` set across a fold's three splits, as `make_folds_within_fold` does, or a pair
    can be drawn twice: train and val share atoms, hence sequences, so their draw pools overlap.
    A test fold's atoms are held out of both, so it cannot collide with either.

    Args:
        split_pos: this split's positive rows.
        cooccur: canonical pair_keys of all observed positives; a draw hitting one is rejected.
        df: front-end protein frame, used to enrich bare hashes to `_PAIR_COLUMNS`.
        schema_pair_full: (slot-a function, slot-b function), full names.
        neg_to_pos_ratio: budget = round(ratio * len(split_pos)).
        seed: seeds the reject sampler.
        hash_col: the alphabet's per-slot hash column (aa: `prot_hash`).
        seen: pair_keys already drawn, extended in place. Defaults to a fresh set, which dedups
            within this call only.

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
    na, nb = [], []               # neg slot-a, neg slot-b
    if seen is None:
        seen = set()              # drawn neg pair_keys; caller-supplied to span a fold's splits
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
