"""Negative-pair samplers shared by the two dataset builders.

Two samplers live here, each selected by `split_strategy.negative_scope`. `within_fold_negatives`
(`within_fold`) draws both slots uniformly from the split's own positives. `balanced_usage_negatives`
(`balanced`) draws from each slot's least-used sequences first, so one sequence is not reused far
more often than another. Both builders reach both samplers: the 2D-CD builder
(`dataset_pairs_cc.py`) and the v2 builder (`dataset_segment_pairs_v2.py`) for its random,
seq_disjoint and single-slot cluster_disjoint modes. `within_fold_negatives` lived in the 2D-CD
builder, which made the v2 builder import one of its peers and forced that import to be
function-local to break a cycle. Nothing here imports either builder, so both can import this
module plainly.

The other two samplers stay where their own routing lives: `within_cc_negatives` needs the CC
structure and belongs to the 2D-CD builder, and the coverage-first sampler belongs to v2.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.datasets._pair_helpers import _side_rep, canonical_pair_key
from src.utils import schema

# The pair-table schema, from the registry rather than from a builder, so this module depends on
# neither of them.
_PAIR_COLUMNS = schema.build_pair_columns()
_MAX_DENSE_REPAIR_SIZE = 2000  # 2000^2 int32 costs take 16 MB; larger assignments grow quickly.
_MAX_DIRECT_REPAIR_CHECKS = 100_000


def _enrich_negatives(
    out: pd.DataFrame,
    df: pd.DataFrame,
    schema_pair_full: tuple,
    ha_col: str,
    hb_col: str,
    hash_col: str) -> pd.DataFrame:
    """Turn a frame of bare per-slot hashes into pair rows in `_PAIR_COLUMNS`.

    Args:
        out: one row per drawn negative, carrying only `ha_col` and `hb_col`.
        df: front-end protein frame, the source of the per-slot metadata columns.
        schema_pair_full: (slot-a function, slot-b function), full names.
        ha_col: slot-a hash column, e.g. `prot_hash_a`.
        hb_col: slot-b hash column, e.g. `prot_hash_b`.
        hash_col: the alphabet's per-slot hash column, e.g. `prot_hash`.

    Returns:
        the same rows in `_PAIR_COLUMNS`, labelled 0, index reset.
    """
    fa, fb = schema_pair_full
    ra = _side_rep(df, fa, 'a', hash_col)  # side-a lookup (one row per hash) to enrich the bare hash_a negatives
    rb = _side_rep(df, fb, 'b', hash_col)  # side-b lookup (one row per hash) to enrich the bare hash_b negatives
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

    Both slots are drawn uniformly with replacement, so how often a sequence is reused is left to
    chance. `balanced_usage_negatives` is the sampler that controls it.

    Pass ONE `seen` set across a fold's three splits, as `make_folds_then_negatives` does, or a
    pair can be drawn twice: train and val share atoms, hence sequences, so their draw pools
    overlap. A test fold's atoms are held out of both, so it cannot collide with either.

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
    return _enrich_negatives(out, df, schema_pair_full, ha_col, hb_col, hash_col)


def _shuffled(items, rng: np.random.RandomState) -> list:
    """A shuffled copy of `items`, leaving the input untouched.

    Args:
        items: the sequences to copy.
        rng: the sampler's random state.

    Returns:
        a new list holding the same elements in random order.
    """
    out = list(items)
    rng.shuffle(out)
    return out


def _open_partner(ha: str, candidates: list, cooccur: set, seen: set, start: int) -> int | None:
    """Index of the first candidate that pairs with `ha` without reproducing a positive or
    repeating a drawn negative, scanning cyclically from `start`.

    Args:
        ha: the slot-a hash a partner is wanted for.
        candidates: slot-b hashes to scan.
        cooccur: canonical pair_keys of all observed positives.
        seen: pair_keys already drawn.
        start: index to begin the scan at; the scan wraps around the end of `candidates`.

    Returns:
        the index into `candidates`, or None if every candidate is blocked.
    """
    n = len(candidates)
    for step in range(n):
        idx = (start + step) % n
        pk = canonical_pair_key(ha, candidates[idx])
        if pk not in cooccur and pk not in seen:
            return idx
    return None


def _repair_exact_one_use(
    na: list, nb: list, uniq_a: list, uniq_b: list, cooccur: set, seen: set,
) -> tuple[list, list]:
    """Restore one negative use per endpoint while preserving greedy pairs where possible.

    Only called when the draw budget equals both unique per-slot sequence counts. Direct swaps run
    first; a bounded minimum-cost assignment handles the remaining imbalance. Observed and
    previously drawn pairs are blocked.
    """
    n = len(na)
    if len(set(na)) == n and len(set(nb)) == n:
        return na, nb

    original_a, original_b = na.copy(), nb.copy()
    checks = 0
    # A single blocked last partner usually needs only one earlier pair changed.
    for slot, pool in ((na, uniq_a), (nb, uniq_b)):
        counts = Counter(slot)
        overused_rows = [i for i, value in enumerate(slot) if counts[value] > 1]
        for unused in pool:
            if counts[unused] or checks >= _MAX_DIRECT_REPAIR_CHECKS:
                continue
            for i in overused_rows:
                if counts[slot[i]] <= 1:
                    continue
                checks += 1
                if checks > _MAX_DIRECT_REPAIR_CHECKS:
                    break
                new_a = unused if slot is na else na[i]
                new_b = unused if slot is nb else nb[i]
                new_key = canonical_pair_key(new_a, new_b)
                if new_key in cooccur or new_key in seen:
                    continue
                old_key = canonical_pair_key(na[i], nb[i])
                old_value = slot[i]
                slot[i] = unused
                counts[old_value] -= 1
                counts[unused] += 1
                seen.remove(old_key)
                seen.add(new_key)
                break

    if len(set(na)) == n and len(set(nb)) == n:
        changed = sum(a != old_a or b != old_b for a, b, old_a, old_b
                      in zip(na, nb, original_a, original_b))
        print(f"Done. balanced_usage_negatives reassigned {changed} negative pairs "
              "to use each sequence once.")
        return na, nb

    if n > _MAX_DENSE_REPAIR_SIZE:
        print(f"WARNING: balanced_usage_negatives skipped dense one-use repair for {n:,} "
              f"pairs (limit {_MAX_DENSE_REPAIR_SIZE:,}); valid negatives retain uneven usage.")
        return na, nb

    from scipy.optimize import linear_sum_assignment

    old_keys = {canonical_pair_key(a, b) for a, b in zip(na, nb)}
    blocked = cooccur | (seen - old_keys)
    a_order = list(dict.fromkeys(na))
    used_a = set(a_order)
    a_order.extend(a for a in uniq_a if a not in used_a)
    b_order = uniq_b
    a_idx = {a: i for i, a in enumerate(a_order)}
    b_idx = {b: i for i, b in enumerate(b_order)}

    # Any all-allowed assignment costs at most n, so n+1 marks a forbidden pair.
    forbidden = n + 1
    cost = np.ones((n, n), dtype=np.int32)
    for i, a in enumerate(a_order):
        for j, b in enumerate(b_order):
            if canonical_pair_key(a, b) in blocked:
                cost[i, j] = forbidden
    for a, b in zip(na, nb):
        cost[a_idx[a], b_idx[b]] = 0

    rows, cols = linear_sum_assignment(cost)
    new_a = [a_order[i] for i in rows]
    new_b = [b_order[j] for j in cols]
    new_keys = {canonical_pair_key(a, b) for a, b in zip(new_a, new_b)}
    if (np.any(cost[rows, cols] == forbidden) or len(new_keys) != n):
        print("WARNING: balanced_usage_negatives could not give every sequence exactly one "
              "negative use without an observed or duplicate pair.")
        return na, nb

    seen.difference_update(old_keys)
    seen.update(new_keys)
    print(f"Done. balanced_usage_negatives reassigned "
          f"{sum(a != old_a or b != old_b for a, b, old_a, old_b in zip(new_a, new_b, original_a, original_b))} "
          "negative pairs to use each sequence once.")
    return new_a, new_b


def balanced_usage_negatives(
    split_pos: pd.DataFrame,
    cooccur: set,
    df: pd.DataFrame,
    schema_pair_full: tuple, *,
    neg_to_pos_ratio: float,
    seed: int,
    hash_col: str = 'prot_hash',
    seen: set | None = None) -> pd.DataFrame:
    """Draw negatives for ONE split, spreading the draws evenly over that split's sequences: a
    slot-a sequence paired with a slot-b sequence, both taken from THIS split's positives.

    Rejects true co-occurrences and duplicates, exactly as `within_fold_negatives` does. What
    differs is which sequences are offered. `within_fold_negatives` draws both slots uniformly with
    replacement, so on a Hopcroft-Karp positive set, where each sequence occurs once per slot,
    chance alone decides that some sequences land in several negatives and others in none. Here each
    slot keeps its sequences in two bags, the least-used and the rest, and a draw takes only from
    the least-used bag. Normally a sequence reaches k+1 uses only once every sequence in its slot
    has k, so within one slot the use counts differ by at most one.

    A blocked least-used partner can force a greedy fallback to reuse a sequence. When the draw
    budget equals both unique per-slot sequence counts, an assignment step repairs this
    whenever one use per sequence is feasible under the pair exclusions.

    Balance holds at any budget. A budget smaller than a slot's sequence count leaves some of that
    slot's sequences unused, and uses each of the rest once. The two slots refill separately, so
    when they hold different numbers of sequences, the smaller slot's sequences are used more often
    on average.

    Pass ONE `seen` set across a fold's three splits, as `make_folds_then_negatives` does, for the
    reason given in `within_fold_negatives`.

    Args:
        split_pos: this split's positive rows.
        cooccur: canonical pair_keys of all observed positives; a draw hitting one is rejected.
        df: front-end protein frame, used to enrich bare hashes to `_PAIR_COLUMNS`.
        schema_pair_full: (slot-a function, slot-b function), full names.
        neg_to_pos_ratio: budget = round(ratio * len(split_pos)).
        seed: seeds the bag order and the draws.
        hash_col: the alphabet's per-slot hash column (aa: `prot_hash`).
        seen: pair_keys already drawn, extended in place. Defaults to a fresh set, which dedups
            within this call only.

    Returns:
        negatives in `_PAIR_COLUMNS`, index reset; empty frame if none could be drawn.
    """
    ha_col, hb_col = f'{hash_col}_a', f'{hash_col}_b'  # alphabet's per-slot hash (aa: prot_hash)
    uniq_a = sorted(set(split_pos[ha_col].astype(str)))  # sorted so the bag order depends on the seed alone
    uniq_b = sorted(set(split_pos[hb_col].astype(str)))
    budget = int(round(neg_to_pos_ratio * len(split_pos)))  # num negatives to sample
    if not uniq_a or not uniq_b or budget <= 0:
        return pd.DataFrame(columns=list(_PAIR_COLUMNS))

    rng = np.random.RandomState(seed)
    if seen is None:
        seen = set()              # drawn neg pair_keys; caller-supplied to span a fold's splits
    open_a, drawn_a = _shuffled(uniq_a, rng), []   # least-used slot-a sequences, and those used this pass
    open_b, drawn_b = _shuffled(uniq_b, rng), []

    na, nb = [], []               # neg slot-a, neg slot-b
    placed, attempts, max_attempts = 0, 0, budget * 50 + 200  # same ceiling as within_fold_negatives
    while placed < budget and attempts < max_attempts:
        attempts += 1
        if not open_a:            # every slot-a sequence has been offered; start the next pass
            open_a, drawn_a = _shuffled(drawn_a, rng), []
        if not open_b:
            open_b, drawn_b = _shuffled(drawn_b, rng), []

        ia = rng.randint(len(open_a))
        ha = open_a[ia]
        ib = _open_partner(ha, open_b, cooccur, seen, rng.randint(len(open_b)))
        if ib is not None:
            hb = open_b.pop(ib)
            drawn_b.append(hb)
        else:
            # Nothing in the least-used slot-b bag pairs with `ha`. Take a sequence already drawn
            # this pass rather than leave `ha` unusable; its count then runs two ahead.
            start = rng.randint(len(drawn_b)) if drawn_b else 0
            iw = _open_partner(ha, drawn_b, cooccur, seen, start)
            if iw is None:
                # `ha` pairs with nothing in either bag. Retire it unused so the next draw picks a
                # different sequence; it returns when the slot-a bag refills.
                drawn_a.append(open_a.pop(ia))
                continue
            hb = drawn_b[iw]

        seen.add(canonical_pair_key(ha, hb))
        drawn_a.append(open_a.pop(ia))
        na.append(ha)
        nb.append(hb)
        placed += 1

    if placed < budget:
        print(f"WARNING: balanced_usage_negatives placed {placed:,} of {budget:,} negatives in "
              f"{attempts:,} attempts.")
    if placed == budget == len(uniq_a) == len(uniq_b):
        na, nb = _repair_exact_one_use(na, nb, uniq_a, uniq_b, cooccur, seen)
    if not na:
        return pd.DataFrame(columns=list(_PAIR_COLUMNS))

    out = pd.DataFrame({ha_col: na, hb_col: nb})
    return _enrich_negatives(out, df, schema_pair_full, ha_col, hb_col, hash_col)
