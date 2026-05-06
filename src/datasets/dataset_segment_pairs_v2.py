"""Pair-builder v2: schema-mode-only, with coverage-first negative sampling,
within-split positive deduplication, exposure tracking, and metadata-axis
annotations.

This module is a parallel implementation to `dataset_segment_pairs.py` (v1).
v1 stays callable; v2 is opt-in via `dataset.pair_builder_version: v2`. See
`docs/plans/done/design_dataset_gen_v2.md` for the full design and decisions.

Primary updates over v1
-----------------------
1. Coverage-first negative sampling: every sequence in positives is guaranteed
   at least one negative pair (modulo sequences for which no valid partner can
   be found). v1 produced negatives by pure random sampling, which left some
   positive-pair sequences without any negative coverage.
2. Within-split positive dedup: same `pair_key` from two different isolates
   that both land in the same split is dropped (keep first). v1 only caught
   cross-split duplicates.
3. Per-sequence exposure table: `sequence_exposure.csv` per split, with
   pos/neg counts per slot and an exposure label (`dual`/`pos_only`/`neg_only`).
4. Metadata-axis annotations: `<axis>_a`, `<axis>_b`, `same_<axis>` columns
   attached to every pair for axes host/year/hn_subtype/geo_location/passage.
5. `metadata_coverage.json` documents null counts per axis at sequence and
   unique-sequence level. Written once per run by the CLI dispatch (not here).

Hard-coded constraints (vs v1)
------------------------------
v2 supports only schema-ordered pairs (`pair_mode = "schema_ordered"`,
`schema_pair` required). Several v1-configurable behaviors are hard-coded
because they are vestigial or auto-disabled in schema mode:

- `allow_same_func_negatives = False`, `max_same_func_ratio = 0.5` (vestigial):
  same-function negatives are impossible in schema mode by construction.
- `canonicalize_pair_orientation_enabled = False`: schema mode defines slot
  orientation by function; hash-based canonicalization conflicts with that.
- `hard_partition_isolates = True`: train/val/test isolates are always disjoint.
- `drop_within_split_pos_duplicates = True`: always runs.

The corresponding parameters are dropped from v2 function signatures. v2
config validation rejects any setting that contradicts these. v1
(`dataset_segment_pairs.py`) retains the configurable versions for users who
need other modes.

Duplicate handling (canonical reference for the three cases)
------------------------------------------------------------
1. **Within-split positive duplicates**: same `pair_key` produced by two
   different isolates in the same split. v2 drops these unconditionally (keep
   first) inside `create_positive_pairs_v2`. This is new in v2.
2. **Within-split negative duplicates**: same `(brc_a, brc_b)` or `pair_key`
   resampled during negative generation. v2 rejects via `seen_pairs` and
   `seen_seq_pairs` sets, same as v1.
3. **Cross-split duplicates**: same `pair_key` appearing in train AND val/test
   (e.g., because two isolates with the identical sequence pair landed in
   different splits). v2 detects this via the post-split `pair_key` overlap
   check inherited from v1, removing overlapping rows from val/test (keeping
   train).
"""

import ipdb
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project root: src/datasets/dataset_segment_pairs_v2.py -> root
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))

from src.datasets._pair_helpers import (
    canonical_pair_key,
    _validate_schema_pair,
    build_cooccurrence_set,
    get_metadata_distributions,
)

# Pair columns (kept identical to v1's positive/negative output) so downstream
# code that reads pair CSVs continues to work. v2 may append extra columns
# (axis flags); the core columns are unchanged.
_PAIR_COLUMNS = [
    'pair_key',
    'assembly_id_a', 'assembly_id_b',
    'brc_a', 'brc_b',
    'ctg_a', 'ctg_b',
    'seq_a', 'seq_b', # TODO. Consider renaming to `aa_seq_a`/`aa_seq_b` or `prot_seq_a`/`prot_seq_b`
    'dna_seq_a', 'dna_seq_b',
    'seg_a', 'seg_b',
    'func_a', 'func_b',
    'seq_hash_a', 'seq_hash_b',
    'dna_hash_a', 'dna_hash_b',
    'label',
]

_DEFAULT_AXES = ("host", "year", "hn_subtype", "geo_location", "passage")


# ---------------------------------------------------------------------------
# 4.1 create_positive_pairs_v2
# ---------------------------------------------------------------------------
def create_positive_pairs_v2(
    df: pd.DataFrame,
    schema_pair: Tuple[str, str],
    # seed: int = 42,
    ) -> tuple[pd.DataFrame, dict]:
    """Generate within-isolate positive pairs in schema-ordered mode, then drop
    duplicate `pair_key` rows (within the input).

    Vectorized via merge-on-`assembly_id`. Slot A is `func_left`, slot B is
    `func_right` by construction (no post-hoc orientation needed). The previous
    Python double-loop implementation is recoverable from git history (see
    branch v2-pos-global-split).

    v2 hard-codes pair_mode='schema_ordered', drop_within_split_pos_duplicates=True,
    canonicalize_pair_orientation_enabled=False. See dataset_segment_pairs.py (v1)
    for the configurable versions. v2 config validation rejects any setting that
    contradicts these.

    Args:
        df: Protein-level DataFrame already filtered to the relevant isolates.
        schema_pair: (func_left, func_right). func_left occupies slot A in every
            output pair; func_right occupies slot B. Must satisfy
            func_left != func_right.

    Returns:
        (pos_df, dedup_stats) where pos_df has the v1 positive-pair columns
        (plus the standard label=1) and dedup_stats reports
        n_pos_before_dedup, n_pos_after_dedup, n_pos_duplicates_dropped, and
        a capped sample of duplicate_isolate_pairs for log readability.
    """
    func_left, func_right = _validate_schema_pair(schema_pair, "create_positive_pairs_v2")

    empty_stats = {
        'n_pos_before_dedup': 0,
        'n_pos_after_dedup': 0,
        'n_pos_duplicates_dropped': 0,
        'duplicate_isolate_pairs': [],
    }
    if len(df) == 0:
        return pd.DataFrame(columns=_PAIR_COLUMNS), empty_stats

    # All req_cols must be present; missing columns indicate an upstream
    # regression (Stage 1 + attach_dna_to_prot_df should produce all nine).
    # Padding with None would let k-mer / DNA features silently degrade.
    req_cols = ['assembly_id', 'brc_fea_id', 'genbank_ctg_id', 'prot_seq',
                'dna_seq', 'canonical_segment', 'function', 'seq_hash', 'dna_hash']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"create_positive_pairs_v2: df missing required columns: {missing}")

    def _side(side_func: str, suffix: str) -> pd.DataFrame:
        out = df.loc[df['function'] == side_func, req_cols].copy()
        return out.rename(columns={
            'brc_fea_id': f'brc_{suffix}',
            'genbank_ctg_id': f'ctg_{suffix}',
            'prot_seq': f'seq_{suffix}',
            'dna_seq': f'dna_seq_{suffix}',
            'canonical_segment': f'seg_{suffix}',
            'function': f'func_{suffix}',
            'seq_hash': f'seq_hash_{suffix}',
            'dna_hash': f'dna_hash_{suffix}',
        })

    left = _side(func_left, 'a')
    right = _side(func_right, 'b')
    pos_df = left.merge(right, on='assembly_id', how='inner')

    # Warn about isolates that have <2 proteins (one or both schema functions missing).
    isolates_in_df = set(df['assembly_id'].unique())
    isolates_with_pairs = set(pos_df['assembly_id'].unique()) if len(pos_df) > 0 else set()
    isolates_with_few_proteins = isolates_in_df - isolates_with_pairs
    if isolates_with_few_proteins:
        print(f'Warning: {len(isolates_with_few_proteins)} isolates have <2 proteins '
              f'(missing func_left and/or func_right). '
              f'These rows are excluded from positive-pair generation.')

    if len(pos_df) == 0:
        return pd.DataFrame(columns=_PAIR_COLUMNS), empty_stats

    pos_df['assembly_id_a'] = pos_df['assembly_id']
    pos_df['assembly_id_b'] = pos_df['assembly_id']
    pos_df['label'] = 1

    # Create canonical pair_key in a vectorized way: lexicographically sorted seq_hash pair joined by '__'.
    # Matches canonical_pair_key('a','b') = '__'.join(sorted([a,b])) in _pair_helpers.py.
    h_a = pos_df['seq_hash_a'].astype(str).values
    h_b = pos_df['seq_hash_b'].astype(str).values
    a_first = h_a <= h_b
    lo = np.where(a_first, h_a, h_b)
    hi = np.where(a_first, h_b, h_a)
    pos_df['pair_key'] = lo + '__' + hi

    # Sanity check: no duplicate brc_fea_id must exist within an isolate.
    # brc_fea_id is unique per isolate (Stage 1 invariant); same brc on both sides
    # of a pair would mean a duplicated row leaked through preprocessing.
    if (pos_df['brc_a'] == pos_df['brc_b']).any():
        bad = pos_df.loc[pos_df['brc_a'] == pos_df['brc_b'], 'assembly_id'].head(5).tolist()
        raise AssertionError(
            f"Pair creation: duplicate brc_fea_id within isolate(s) {bad} "
            f"(must be unique per isolate; check Stage 1)."
        )

    pos_df = pos_df.loc[:, _PAIR_COLUMNS]

    # TODO. Below we drop dups based on pair_key computed on protein seq hashes.
    # We also want to understand the duplicate situation on the DNA side, as well
    # as on the k-mer side.

    # Within-input pair_key dedup. v2 hard-codes drop_within_split_pos_duplicates=True.
    # In the global-pos flow (split_dataset_v2), this runs once on the full pos_df and
    # gives every pair_key a single representative isolate.
    n_before = len(pos_df)
    is_dup = pos_df.duplicated(subset=['pair_key'], keep='first') # TODO. keep='first' is arbitrary; we can later decide to keep isolate by some criteria
    dup_rows = pos_df.loc[is_dup, ['assembly_id_a', 'assembly_id_b', 'pair_key']]
    duplicate_isolate_pairs = [
        (str(r.assembly_id_a), str(r.assembly_id_b), str(r.pair_key))
        for r in dup_rows.head(100).itertuples()
    ]
    pos_df = pos_df.loc[~is_dup].reset_index(drop=True)
    n_after = len(pos_df)
    n_dropped = n_before - n_after

    if n_dropped > 0:
        print(f"Positive pair dedup: dropped {n_dropped:,} duplicate pair_key rows; "
              f"{n_before:,} -> {n_after:,}.")

    dedup_stats = {
        'n_pos_before_dedup': int(n_before),
        'n_pos_after_dedup': int(n_after),
        'n_pos_duplicates_dropped': int(n_dropped),
        'duplicate_isolate_pairs': duplicate_isolate_pairs,
    }
    return pos_df, dedup_stats


# ---------------------------------------------------------------------------
# 4.2 create_negative_pairs_v2
# ---------------------------------------------------------------------------
def create_negative_pairs_v2(
    df: pd.DataFrame,
    pos_pairs: pd.DataFrame,
    num_negatives: int,
    isolate_ids: list,
    cooccur_pairs: set,
    schema_pair: Tuple[str, str],
    seed: int = 42,
    max_attempts_per_seq: int = 50,
    max_attempts_multiplier: int = 100,
    axis_quotas: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
    """Generate negative pairs with a coverage-first two-phase sampler.

    v2 hard-codes pair_mode='schema_ordered', allow_same_func_negatives=False
    (impossible in schema mode), canonicalize_pair_orientation_enabled=False.
    See dataset_segment_pairs.py (v1) for the configurable versions. v2 config
    validation rejects any setting that contradicts these.

    Two phases:
      1. Coverage phase: iterate every seq_hash appearing in `pos_pairs` (in
         sorted order) and try up to `max_attempts_per_seq` to find a valid
         negative partner from a different isolate. This guarantees every
         positive-pair sequence appears in at least one negative (modulo
         `seqs_with_zero_negatives`).
      2. Fill phase: if the coverage phase produced fewer than `num_negatives`
         pairs, top up using v1's random-sampling logic (sample two isolates,
         one func_left from each, one func_right from the other) until
         `num_negatives` is reached or `total_attempts` exhausts the budget.

    Forward note: a future iteration may replace this with a multi-pass
    round-robin sampler that bounds `max(neg_count) - min(neg_count) <= 1`
    across all sequences. Not implemented now -- the simpler "every seq has
    >=1 negative" guarantee is sufficient for the immediate evaluation needs.

    Coverage floor (interaction with `num_negatives`): the minimum number of
    negatives this function will produce is
    `max(|unique_slot_A_seqs|, |unique_slot_B_seqs|)` over `pos_pairs`. If
    `num_negatives` falls below that, the coverage phase produces ~floor pairs
    anyway and the fill phase becomes a no-op; `coverage_overrode_ratio` is
    set to True and a warning is logged. See spec §4.2 "Coverage floor".

    Returns:
        (neg_df, rejection_stats). rejection_stats includes the v1 schema-mode
        keys (blocked_cooccur, duplicate_brc, duplicate_seq, missing_left_func,
        missing_right_func, total_attempts) and v2-specific keys
        (requested_negatives, min_required_for_coverage, coverage_phase_pairs,
        fill_phase_pairs, achieved_negatives, coverage_overrode_ratio,
        coverage_skipped, seqs_with_zero_negatives).
    """
    # ipdb.set_trace(context=10)
    func_left, func_right = _validate_schema_pair(schema_pair, "create_negative_pairs_v2")
    if axis_quotas is not None and len(axis_quotas) > 0:
        raise NotImplementedError("axis_quotas not yet supported; pass None")

    rng = random.Random(seed) # noqa: F841
    t_setup = time.time()
    print(f"[diag] create_negative_pairs_v2: setup start "
          f"(|df|={len(df):,}, |pos_pairs|={len(pos_pairs):,}, "
          f"|isolate_ids|={len(isolate_ids):,})", flush=True)

    # Build per-isolate function buckets exactly as v1's schema-mode branch does.
    df_subset = df[df['assembly_id'].isin(isolate_ids)].reset_index(drop=True)
    isolate_groups = {aid: list(grp.itertuples()) for aid, grp in df_subset.groupby('assembly_id')}
    isolate_func_groups = {
        aid: {
            func_left: [r for r in rows if r.function == func_left],
            func_right: [r for r in rows if r.function == func_right],
        }
        for aid, rows in isolate_groups.items()
    }
    print(f"[diag] create_negative_pairs_v2: isolate_func_groups built in "
          f"{time.time()-t_setup:.2f}s ({len(isolate_func_groups):,} isolates).", flush=True)

    # Coverage targets: seq_hashes appearing in slot A (func_left) and slot B
    # (func_right) of pos_pairs. In schema mode the slot is determined by
    # function, so the two sets have disjoint keys (a func_left seq only
    # appears in target_seqs_left).
    if len(pos_pairs) > 0:
        target_seqs_left = set(pos_pairs.loc[:, 'seq_hash_a'].tolist())
        target_seqs_right = set(pos_pairs.loc[:, 'seq_hash_b'].tolist())
    else:
        target_seqs_left = set()
        target_seqs_right = set()

    neg_count_a: dict = {s: 0 for s in target_seqs_left}
    neg_count_b: dict = {s: 0 for s in target_seqs_right}

    # seq_hash -> set of assembly_ids it appears in (across the FULL df, not
    # the split-restricted subset, because biological duplicates outside the
    # split still count as "same sequence").
    t_s2i = time.time()
    seq_to_isolates: dict = {}
    for r in df.itertuples():
        s = r.seq_hash
        a = r.assembly_id
        if s in seq_to_isolates:
            seq_to_isolates[s].add(a)
        else:
            seq_to_isolates[s] = {a}
    print(f"[diag] create_negative_pairs_v2: seq_to_isolates built in "
          f"{time.time()-t_s2i:.2f}s ({len(seq_to_isolates):,} unique seq_hashes).", flush=True)

    # Coverage floor: max(|unique_slot_A|, |unique_slot_B|). Each negative pair
    # contributes one A-coverage and one B-coverage simultaneously, so the
    # floor is governed by whichever side has more unique sequences.
    min_required_for_coverage = max(len(target_seqs_left), len(target_seqs_right))
    requested_negatives = int(num_negatives)
    coverage_overrode_ratio = min_required_for_coverage > requested_negatives
    if coverage_overrode_ratio and min_required_for_coverage > 0:
        print(
            f"WARNING: coverage floor overrides num_negatives. "
            f"requested_negatives={requested_negatives:,} but "
            f"min_required_for_coverage={min_required_for_coverage:,}. "
            f"Coverage takes precedence; fill phase will be a no-op."
        )

    seen_pairs: set = set()
    seen_seq_pairs: set = set()
    rejection_stats: dict = {
        'blocked_cooccur': 0,
        'duplicate_brc': 0,
        'duplicate_seq': 0,
        'missing_left_func': 0,
        'missing_right_func': 0,
        'total_attempts': 0,
        'coverage_skipped': [],
    }
    neg_pairs: list[dict] = []

    def _build_pair_dct(row_a, row_b) -> dict:
        return {
            'pair_key': canonical_pair_key(row_a.seq_hash, row_b.seq_hash),
            'assembly_id_a': row_a.assembly_id, 'assembly_id_b': row_b.assembly_id,
            'brc_a': row_a.brc_fea_id, 'brc_b': row_b.brc_fea_id,
            'ctg_a': getattr(row_a, 'genbank_ctg_id', None),
            'ctg_b': getattr(row_b, 'genbank_ctg_id', None),
            'seq_a': row_a.prot_seq, 'seq_b': row_b.prot_seq,
            'dna_seq_a': getattr(row_a, 'dna_seq', None),
            'dna_seq_b': getattr(row_b, 'dna_seq', None),
            'seg_a': row_a.canonical_segment, 'seg_b': row_b.canonical_segment,
            'func_a': row_a.function, 'func_b': row_b.function,
            'seq_hash_a': row_a.seq_hash, 'seq_hash_b': row_b.seq_hash,
            'dna_hash_a': getattr(row_a, 'dna_hash', None),
            'dna_hash_b': getattr(row_b, 'dna_hash', None),
            'label': 0,
        }

    def _try_accept(row_a, row_b) -> bool:
        """Apply the same rejection checks as v1 schema mode. Returns True if
        the pair was accepted (and side-effects: appended to neg_pairs, seen_*
        sets updated, neg_count_a/b incremented)."""
        brc_pair_key = tuple(sorted([row_a.brc_fea_id, row_b.brc_fea_id]))
        if brc_pair_key in seen_pairs:
            rejection_stats['duplicate_brc'] += 1
            return False
        seq_pair_key = canonical_pair_key(row_a.seq_hash, row_b.seq_hash)
        if seq_pair_key in cooccur_pairs:
            rejection_stats['blocked_cooccur'] += 1
            return False
        if seq_pair_key in seen_seq_pairs:
            rejection_stats['duplicate_seq'] += 1
            return False
        # Schema orientation is correct by construction (we picked func_left
        # for slot A and func_right for slot B); no orient_pair_by_schema swap
        # is needed.
        neg_pairs.append(_build_pair_dct(row_a, row_b))
        seen_pairs.add(brc_pair_key)
        seen_seq_pairs.add(seq_pair_key)
        if row_a.seq_hash in neg_count_a:
            neg_count_a[row_a.seq_hash] += 1
        if row_b.seq_hash in neg_count_b:
            neg_count_b[row_b.seq_hash] += 1
        return True

    # ---- Coverage phase --------------------------------------------------
    isolate_ids_list = sorted(isolate_ids)
    isolate_ids_set = set(isolate_ids_list)

    # seq_hash -> function lookup (used to determine which slot a target seq
    # belongs to). Hard error if any seq_hash maps to >1 function -- v2 slot
    # semantics depend on this invariant. See spec §4.3 / §4.4.
    seq_to_func = _build_seq_to_func(df)

    # Pre-compute seq_hash -> first row in df_subset. We need a concrete row
    # for the "self" representative when constructing each negative pair (the
    # pair dict includes per-row metadata: brc_fea_id, prot_seq, segment,
    # etc.). Doing `df_subset[df_subset['seq_hash'] == s]` per target seq is
    # O(|df_subset|) -> 50K target seqs x 700K rows = 35B ops on a real flu
    # train split. Pre-build the dict once.
    seq_hash_to_row: dict = {}
    for row in df_subset.itertuples():
        seq_hash_to_row.setdefault(row.seq_hash, row)

    target_union = sorted(target_seqs_left | target_seqs_right)
    n_targets = len(target_union)
    print(f"\nCoverage phase: {n_targets:,} target seqs to cover "
          f"(min_required_for_coverage={min_required_for_coverage:,}).", flush=True)
    progress_step = max(n_targets // 20, 1)  # ~20 progress prints
    coverage_phase_pairs_start = 0  # always 0; coverage phase fills neg_pairs from empty
    for i, s in enumerate(target_union):
        if i > 0 and i % progress_step == 0:
            print(f"  coverage phase: {i:,}/{n_targets:,} target seqs "
                  f"({100.0*i/n_targets:.1f}%); accepted={len(neg_pairs):,}, "
                  f"attempts={rejection_stats['total_attempts']:,}", flush=True)
        s_func = seq_to_func.get(s)
        if s_func == func_left:
            if neg_count_a.get(s, 0) > 0:
                continue
            partner_bucket = func_right
            s_in_left = True
        elif s_func == func_right:
            if neg_count_b.get(s, 0) > 0:
                continue
            partner_bucket = func_left
            s_in_left = False
        else:
            # Shouldn't happen: pos_pairs sourced from df, and seq_to_func is
            # built from df. Skip defensively.
            rejection_stats['coverage_skipped'].append((s, 'unknown_function'))
            continue

        excluded = seq_to_isolates.get(s, set())
        # Restrict `excluded` to this split's isolate set, so the early-bail
        # check ("`s` is in every isolate of this split") is accurate. O(|excluded|).
        excluded_in_split = excluded & isolate_ids_set
        if len(excluded_in_split) >= len(isolate_ids_list):
            # No isolate in this split lacks `s` -> no possible negative partner.
            rejection_stats['coverage_skipped'].append((s, 'no_other_isolate'))
            continue

        s_row = seq_hash_to_row.get(s)
        if s_row is None:
            # s appears in pos_pairs but not in df_subset (the split-restricted
            # data). Should not happen if pos_pairs were generated from a
            # subset of df_subset. Defensive skip.
            rejection_stats['coverage_skipped'].append((s, 'no_row_for_seq'))
            continue

        accepted = False
        for _attempt in range(max_attempts_per_seq):
            rejection_stats['total_attempts'] += 1
            # Sample-and-reject: |excluded_in_split| is typically ~1 vs |isolate_ids|=86K,
            # so the rejection rate is negligible. This avoids rebuilding a filtered
            # candidate list O(N) per attempt.
            other_aid = rng.choice(isolate_ids_list)
            if other_aid in excluded_in_split:
                continue
            partner_rows = isolate_func_groups.get(other_aid, {}).get(partner_bucket, [])
            if not partner_rows:
                if partner_bucket == func_right:
                    rejection_stats['missing_right_func'] += 1
                else:
                    rejection_stats['missing_left_func'] += 1
                continue
            partner_row = rng.choice(partner_rows)

            if s_in_left:
                row_a, row_b = s_row, partner_row
            else:
                row_a, row_b = partner_row, s_row

            if _try_accept(row_a, row_b):
                accepted = True
                break

        if not accepted:
            # Cap to 100 entries for log readability; further skips are dropped.
            if len(rejection_stats['coverage_skipped']) < 100:
                rejection_stats['coverage_skipped'].append((s, 'attempts_exhausted'))

    coverage_phase_pairs = len(neg_pairs) - coverage_phase_pairs_start
    print(f"Coverage phase complete: {coverage_phase_pairs:,} negatives produced "
          f"({rejection_stats['total_attempts']:,} attempts).", flush=True)

    # ---- Fill phase ------------------------------------------------------
    fill_phase_target = max(requested_negatives, 0)
    max_total_attempts = rejection_stats['total_attempts'] + max(
        fill_phase_target * max_attempts_multiplier, 1
    )
    if len(neg_pairs) < fill_phase_target:
        print(f"\nFill phase: targeting {fill_phase_target:,} total negatives "
              f"({fill_phase_target - len(neg_pairs):,} more to add).", flush=True)

    while (
        len(neg_pairs) < fill_phase_target
        and rejection_stats['total_attempts'] < max_total_attempts
    ):
        rejection_stats['total_attempts'] += 1
        if len(isolate_ids_list) < 2:
            break
        aid1, aid2 = rng.sample(isolate_ids_list, 2)
        left_candidates = isolate_func_groups[aid1][func_left]
        if not left_candidates:
            rejection_stats['missing_left_func'] += 1
            continue
        right_candidates = isolate_func_groups[aid2][func_right]
        if not right_candidates:
            rejection_stats['missing_right_func'] += 1
            continue
        row_a = rng.choice(left_candidates)
        row_b = rng.choice(right_candidates)
        _try_accept(row_a, row_b)

    fill_phase_pairs = len(neg_pairs) - coverage_phase_pairs

    if fill_phase_target > 0 and len(neg_pairs) < fill_phase_target and not coverage_overrode_ratio:
        print(
            f"WARNING: only generated {len(neg_pairs)}/{fill_phase_target} negative pairs "
            f"after {rejection_stats['total_attempts']} attempts. May indicate high "
            f"sequence overlap across isolates."
        )

    seqs_with_zero_negatives: list = []
    for s, c in neg_count_a.items():
        if c == 0 and len(seqs_with_zero_negatives) < 100:
            seqs_with_zero_negatives.append(s)
    for s, c in neg_count_b.items():
        if c == 0 and len(seqs_with_zero_negatives) < 100:
            seqs_with_zero_negatives.append(s)

    rejection_stats['requested_negatives'] = int(requested_negatives)
    rejection_stats['min_required_for_coverage'] = int(min_required_for_coverage)
    rejection_stats['coverage_phase_pairs'] = int(coverage_phase_pairs)
    rejection_stats['fill_phase_pairs'] = int(fill_phase_pairs)
    rejection_stats['achieved_negatives'] = int(len(neg_pairs))
    rejection_stats['coverage_overrode_ratio'] = bool(coverage_overrode_ratio)
    rejection_stats['seqs_with_zero_negatives'] = seqs_with_zero_negatives

    neg_df = pd.DataFrame(neg_pairs, columns=_PAIR_COLUMNS) if neg_pairs else pd.DataFrame(columns=_PAIR_COLUMNS)
    return neg_df, rejection_stats


def _build_seq_to_func(df: pd.DataFrame) -> dict:
    """Build seq_hash -> function map. Hard error if any seq_hash maps to >1
    distinct function -- v2 slot semantics depend on this invariant."""
    funcs_per_hash = df.groupby('seq_hash')['function'].nunique()
    bad = funcs_per_hash[funcs_per_hash > 1]
    if len(bad) > 0:
        offenders = bad.head(5).index.tolist()
        raise ValueError(
            f"seq_hash maps to >1 distinct function for {len(bad)} hash(es): "
            f"{offenders} (showing up to 5). v2 assumes seq_hash -> function is "
            f"many-to-one; this indicates a Stage 1 (preprocessing) data-quality bug."
        )
    return df.groupby('seq_hash')['function'].first().to_dict()


# ---------------------------------------------------------------------------
# 4.3 compute_exposure_stats
# ---------------------------------------------------------------------------
def compute_exposure_stats(
    pos_pairs: pd.DataFrame,
    neg_pairs: pd.DataFrame,
    df: pd.DataFrame,
    ) -> pd.DataFrame:
    """Per-sequence exposure table for one split.

    pair_mode is not a parameter -- v2 is schema-mode only.

    Returns a DataFrame with one row per seq_hash that appears in pos_pairs OR
    neg_pairs. Columns: seq_hash, function, assembly_ids (list), n_pos_slot_a,
    n_pos_slot_b, n_pos_total, n_neg_slot_a, n_neg_slot_b, n_neg_total,
    exposure (`dual` / `pos_only` / `neg_only`).

    Slot semantics are fixed: slot A = func_left, slot B = func_right. So a
    func_left seq has n_*_slot_b == 0 by construction (literal 0, not null).
    """
    seq_to_func = _build_seq_to_func(df)

    # Build seq_hash -> sorted list of assembly_ids from df (full data, not
    # just the split). Biological duplicates may span isolates outside the
    # split; we report all assembly_ids for transparency.
    seq_to_assemblies: dict = {}
    for r in df.itertuples():
        s = r.seq_hash
        a = r.assembly_id
        if s in seq_to_assemblies:
            seq_to_assemblies[s].add(a)
        else:
            seq_to_assemblies[s] = {a}

    # Counts per slot
    counters: dict = {}

    def _bump(s: str, slot: str, label: int):
        d = counters.setdefault(s, {
            'n_pos_slot_a': 0, 'n_pos_slot_b': 0,
            'n_neg_slot_a': 0, 'n_neg_slot_b': 0,
        })
        if label == 1:
            d['n_pos_slot_a' if slot == 'a' else 'n_pos_slot_b'] += 1
        else:
            d['n_neg_slot_a' if slot == 'a' else 'n_neg_slot_b'] += 1

    for df_pairs, label in [(pos_pairs, 1), (neg_pairs, 0)]:
        if len(df_pairs) == 0:
            continue
        for r in df_pairs[['seq_hash_a', 'seq_hash_b']].itertuples(index=False):
            _bump(r.seq_hash_a, 'a', label)
            _bump(r.seq_hash_b, 'b', label)

    rows: list[dict] = []
    for s, c in counters.items():
        n_pos_total = c['n_pos_slot_a'] + c['n_pos_slot_b']
        n_neg_total = c['n_neg_slot_a'] + c['n_neg_slot_b']
        if n_pos_total == 0 and n_neg_total == 0:
            continue
        if n_pos_total > 0 and n_neg_total > 0:
            exposure = 'dual'
        elif n_pos_total > 0:
            exposure = 'pos_only'
        else:
            exposure = 'neg_only'
        rows.append({
            'seq_hash': s,
            'function': seq_to_func.get(s),
            'assembly_ids': sorted(seq_to_assemblies.get(s, [])),
            'n_pos_slot_a': c['n_pos_slot_a'],
            'n_pos_slot_b': c['n_pos_slot_b'],
            'n_pos_total': n_pos_total,
            'n_neg_slot_a': c['n_neg_slot_a'],
            'n_neg_slot_b': c['n_neg_slot_b'],
            'n_neg_total': n_neg_total,
            'exposure': exposure,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4.4 compute_axis_flags
# ---------------------------------------------------------------------------
def compute_axis_flags(
    pairs: pd.DataFrame,
    df: pd.DataFrame,
    axes: list = _DEFAULT_AXES,
    ) -> pd.DataFrame:
    """Attach <axis>_a, <axis>_b, same_<axis> columns to `pairs` for each axis
    that exists as a column in `df`. Accepts `geo_location` aliased to
    `geo_location_clean`.

    same_<axis> is a nullable boolean: True if both values are non-null and
    equal; False if both non-null and different; pd.NA if either is null.
    Two unknowns are not "same".

    Implementation: build a `seq_hash -> metadata` lookup via groupby +
    first(). Warns if any seq_hash maps to multiple distinct values for an
    axis (upstream metadata inconsistency). The function column has a
    stricter policy -- a seq_hash mapping to >1 distinct function is a hard
    error (raised by `_build_seq_to_func`, called elsewhere in v2).

    Returns a copy of `pairs` with the new columns appended.
    """
    out = pairs.copy()
    if len(out) == 0:
        return out

    # Per-seq metadata: take first non-null per axis. Warn on conflicts.
    df_for_lookup = df[['seq_hash'] + [c for c in df.columns if c != 'seq_hash']].copy()

    for axis in axes:
        col = axis
        if col not in df_for_lookup.columns:
            # Try the geo_location_clean alias for the geo_location axis name.
            if axis == 'geo_location' and 'geo_location_clean' in df_for_lookup.columns:
                col = 'geo_location_clean'
            else:
                # Axis missing: skip (matches v2 spec §5 "axis missing -> warn,
                # drop, continue"). Warn once.
                print(f"WARNING: axis {axis!r} not present in df; skipping axis flags.")
                continue

        # seq_hash -> first-non-null value lookup
        per_hash = df_for_lookup.groupby('seq_hash')[col].agg(
            lambda s: s.dropna().iloc[0] if s.notna().any() else None
        )

        # Conflict check: log warning if any seq_hash has >1 distinct value
        # (across non-null observations).
        conflict_count = (
            df_for_lookup.dropna(subset=[col])
            .groupby('seq_hash')[col]
            .nunique()
            .gt(1)
            .sum()
        )
        if conflict_count > 0:
            print(
                f"WARNING: {conflict_count} seq_hash(es) have multiple distinct values "
                f"for axis {axis!r} across their source rows -- using first non-null. "
                f"This indicates upstream metadata inconsistency."
            )

        a_vals = out['seq_hash_a'].map(per_hash)
        b_vals = out['seq_hash_b'].map(per_hash)
        out[f'{axis}_a'] = a_vals
        out[f'{axis}_b'] = b_vals

        # same_<axis>: nullable boolean. True/False only when both non-null;
        # pd.NA otherwise. Use a Python-level loop via mask logic (concise).
        both_present = a_vals.notna() & b_vals.notna()
        equal = a_vals == b_vals
        same_col = pd.Series([pd.NA] * len(out), dtype='object')
        same_col[both_present & equal] = True
        same_col[both_present & ~equal] = False
        # Coerce to nullable boolean for clean serialization.
        out[f'same_{axis}'] = pd.array(same_col.tolist(), dtype='boolean')

    return out


# ---------------------------------------------------------------------------
# 4.5 compute_metadata_coverage
# ---------------------------------------------------------------------------
def compute_metadata_coverage(
    df: pd.DataFrame,
    axes: list = _DEFAULT_AXES,
    ) -> dict:
    """Summarize how well each metadata axis is populated in `df`.

    Purpose
    -------
    Stage-3 QC: before pairs are built, report how complete each metadata
    axis (host, year, hn_subtype, geo_location, passage) is in the filtered
    protein dataframe, and what its dominant values are. The output is
    written to `dataset_stats.json` so downstream readers can flag bundles
    where, e.g., 40% of rows have null `host` and a hard "same_host" flag
    would silently drop most pairs.

    Two coverage views per axis
    ---------------------------
      - Per-row:  n_rows_total / n_rows_null / pct_null
        Counts every protein row. Reflects *isolate-level* coverage:
        a sequence shared across 100 isolates contributes 100 rows.
      - Per-sequence: n_unique_seqs / n_unique_seqs_null / pct_unique_null
        Collapses by `seq_hash` via `.first()` and counts unique sequences
        with a non-null value. Reflects *sequence-level* coverage, which
        is what matters for embedding-keyed lookups.

    Caveats on per-sequence stats
    -----------------------------
    NOTE/TODO!
    `.first()` assumes one value per `seq_hash`. This is well-defined for
    HA/NA on the H/N subtype components, but for internal-protein bundles
    (PB2/PB1/PA/NP/M1/NS1) a single sequence routinely appears in isolates
    of different `hn_subtype`, `host`, or `year` (reassortment, host
    sharing, multi-year persistence). For those axes, `.first()` returns
    an arbitrary representative; the stat is computed but should be read
    as "coverage of the per-sequence representative value", not as
    "every isolate carrying this sequence has this value".

    Other reported fields
    ---------------------
      - n_distinct_values: number of distinct non-null values in the column.
      - top_5_values: [(value, count), ...] for the 5 most common values.

    Special cases
    -------------
      - `geo_location` falls back to `geo_location_clean` if the raw column
        is missing.
      - If neither the axis nor a known alias is present, the entry is
        `{'present': False}`.
    """
    # breakpoint()
    out: dict = {}
    for axis in axes:
        col = axis
        if col not in df.columns:
            if axis == 'geo_location' and 'geo_location_clean' in df.columns:
                col = 'geo_location_clean'
            else:
                out[axis] = {'present': False}
                continue

        n_rows_total = len(df)
        n_rows_null = int(df[col].isna().sum())
        pct_null = (n_rows_null / n_rows_total * 100.0) if n_rows_total > 0 else 0.0

        # seq_hash is protein sequence identity; counting unique seq_hashes captures the number of unique sequences, which is more meaningful for coverage than counting rows (many rows may share the same sequence).
        # n_unique_seqs_null counts how many unique sequences have null for this axis;
        # pct_unique_null is the percentage of unique sequences with null for this axis.
        per_seq = df.groupby('seq_hash')[col].first()
        n_unique_seqs = int(len(per_seq))
        n_unique_seqs_null = int(per_seq.isna().sum())
        pct_unique_null = (n_unique_seqs_null / n_unique_seqs * 100.0) if n_unique_seqs > 0 else 0.0

        non_null = df[col].dropna()
        n_distinct_values = int(non_null.nunique())
        top_counts = non_null.value_counts().head(5)
        top_5_values = [(_jsonable(k), int(v)) for k, v in top_counts.items()]

        out[axis] = {
            'present': True,
            'n_rows_total': int(n_rows_total),
            'n_rows_null': n_rows_null,
            'pct_null': round(float(pct_null), 3),
            'n_unique_seqs': n_unique_seqs,
            'n_unique_seqs_null': n_unique_seqs_null,
            'pct_unique_null': round(float(pct_unique_null), 3),
            'n_distinct_values': n_distinct_values,
            'top_5_values': top_5_values,
        }
    return out


def _jsonable(v):
    """Coerce numpy / pandas scalars to native Python types for JSON serialization."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return str(v) if not isinstance(v, (int, float, str, bool)) else v


# ---------------------------------------------------------------------------
# 4.6 split_dataset_v2
# ---------------------------------------------------------------------------
def split_dataset_v2(
    df: pd.DataFrame,
    schema_pair: Tuple[str, str],
    neg_to_pos_ratio: float = 3.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_attempts_per_seq: int = 50,
    max_attempts_multiplier: int = 100,
    axes_for_flags: list = _DEFAULT_AXES,
    axis_quotas: Optional[dict] = None,
    train_isolates_override: Optional[list] = None,
    val_isolates_override: Optional[list] = None,
    test_isolates_override: Optional[list] = None,
    cooccur_pairs: Optional[set] = None,
    cooccur_stats: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """Build train/val/test splits with global pair_key dedup, a plain
    row-level shuffle on the deduped pos_df (safe under v2's strict
    one-pair-per-isolate invariant), coverage-first negative sampling, axis
    flag annotations, and exposure tables.

    Flow:
      1. Build the global positive table with one call to
         `create_positive_pairs_v2(df, ...)`. This dedups pair_keys across the
         full df and assigns each unique pair_key a single representative
         assembly_id (the lexicographically smallest one).
      2. Strict invariant: assert `pos_df['assembly_id_a'].is_unique`. With
         schema_ordered + one func_left + one func_right per isolate, every
         isolate yields exactly one pair_key. A violation means upstream data
         has multi-row-per-function isolates and v2 cannot proceed.
      3. Choose train/val/test: either from the override lists (CV path)
         or via a plain `train_test_split` on pos_df rows (auto path).
         Because the invariant holds, row-level split == isolate-level split,
         and train/val/test are pair_key- AND isolate-disjoint by construction.
      4. Negatives are generated per split via the coverage-first sampler
         (unchanged from prior v2).

    v2 hard-codes pair_mode='schema_ordered', allow_same_func_negatives=False,
    canonicalize_pair_orientation_enabled=False, hard_partition_isolates=True,
    drop_within_split_pos_duplicates=True. See dataset_segment_pairs.py (v1)
    for the configurable versions. v2 config validation rejects any setting
    that contradicts these.

    Returns:
        (train_pairs, val_pairs, test_pairs, duplicate_stats, exposure_tables)
        where exposure_tables is `{'train': df, 'val': df, 'test': df}`.
        duplicate_stats includes `pos_dedup_global` (global pair_key dedup
        before/after/dropped) and `coverage_stats` (per split).
    """
    func_left, func_right = _validate_schema_pair(schema_pair, "split_dataset_v2")
    if axis_quotas is not None and len(axis_quotas) > 0:
        raise NotImplementedError("axis_quotas not yet supported; pass None")

    # Build co-occurrence (or accept pre-computed for CV reuse).
    # cooccur_pairs: set of canonical pair-key strings, each of the form
    # "<seq_hash>__<seq_hash>" with the two hashes sorted lexicographically.
    # One entry per sequence pair that appears together in at least one isolate.
    # Used to block candidate negatives that would actually be positives somewhere.
    if cooccur_pairs is None:
        print("\nsplit_dataset_v2: Build co-occurrence set (sequences that appear together in any isolate)...")
        cooccur_pairs, cooccur_stats = build_cooccurrence_set(df)
        print(f"   Total co-occurring sequence pairs: {cooccur_stats['total_cooccur_pairs']:,}")
        print(f"   Pairs appearing in multiple isolates: {cooccur_stats['pairs_in_multiple_isolates']:,}")
        print(f"   Max isolates for a single pair: {cooccur_stats['max_isolates_per_pair']}")
    elif cooccur_stats is None:
        raise ValueError("cooccur_stats must be provided together with cooccur_pairs")

    # Build the global positive pair table once, deduped to one row per unique
    # pair_key with a deterministic representative assembly_id. The split is then
    # a partition of this table -- train/val/test are pair_key-disjoint by
    # construction, so no post-hoc cross-split overlap removal is needed.
    print("\nsplit_dataset_v2: Build global positive pairs (one call across full df)...", flush=True)
    pos_df, pos_dedup_stats = create_positive_pairs_v2(df, schema_pair=schema_pair)
    if len(pos_df) == 0:
        raise ValueError(
            f"No positive pairs generated from df. Verify schema_pair={schema_pair} "
            f"matches the function values present in df."
        )

    # Each isolate must produce exactly one pair_key after global dedup.
    # Holds when schema_ordered uses one func_left + one func_right per
    # isolate (the common Flu A case). A violation means upstream data has
    # multi-row-per-function isolates and v2's plain-split flow cannot
    # proceed.
    if not pos_df['assembly_id_a'].is_unique:
        counts = pos_df['assembly_id_a'].value_counts()
        multi = counts[counts > 1]
        raise ValueError(
            f"v2 strict mode: {len(multi)} isolate(s) produce >1 pair_key after "
            f"global dedup. Schema-ordered v2 requires exactly one pair per isolate."
        )
    ipdb.set_trace(context=10)  # debug breakpoint preserved from earlier session

    # Decide train/val/test membership.
    if train_isolates_override is not None:
        # TODO/NOTE: haven't tested the CV with v2 yet!
        # CV / external-control path: caller has already partitioned isolates.
        # Filter the global pos_df by membership; pair_keys whose representative
        # falls in val/test_isolates_override land in those splits, so disjointness
        # is preserved as long as the override lists are themselves disjoint.
        train_isolates = list(train_isolates_override)
        val_isolates = list(val_isolates_override) if val_isolates_override is not None else []
        test_isolates = list(test_isolates_override) if test_isolates_override is not None else []
        print(f"split_dataset_v2: using isolate overrides "
              f"(train={len(train_isolates):,}, val={len(val_isolates):,}, "
              f"test={len(test_isolates):,})", flush=True)

        train_pos = pos_df[pos_df['assembly_id_a'].isin(set(train_isolates))].reset_index(drop=True)
        val_pos = pos_df[pos_df['assembly_id_a'].isin(set(val_isolates))].reset_index(drop=True)
        test_pos = pos_df[pos_df['assembly_id_a'].isin(set(test_isolates))].reset_index(drop=True)
    else:
        # Plain shuffle-split on the deduped pos_df. Safe because the is_unique assertion above
        # guarantees each row is its own isolate, so row-level split == isolate-level split.
        print("\nsplit_dataset_v2: Plain shuffle-split on pos_df rows...", flush=True)
        test_size_outer = 1.0 - train_ratio - val_ratio
        trainval_pos, test_pos = train_test_split(pos_df, test_size=test_size_outer, random_state=seed)
        test_size_inner = val_ratio / (train_ratio + val_ratio)
        train_pos, val_pos = train_test_split(trainval_pos, test_size=test_size_inner, random_state=seed)

        train_pos = train_pos.reset_index(drop=True)
        val_pos = val_pos.reset_index(drop=True)
        test_pos = test_pos.reset_index(drop=True)

        # is_unique invariant => one row == one isolate, so .tolist() is the
        # unique-isolate list directly (no .unique() needed).
        train_isolates = sorted(train_pos['assembly_id_a'].tolist())
        val_isolates = sorted(val_pos['assembly_id_a'].tolist())
        test_isolates = sorted(test_pos['assembly_id_a'].tolist())
        print(f"split_dataset_v2: shuffle-split completed "
              f"(train={len(train_isolates):,} pairs, "
              f"val={len(val_isolates):,}, test={len(test_isolates):,})", flush=True)

    # Generate negatives with coverage-first sampler.
    print("\nCreate negative pairs (coverage-first, schema mode)...", flush=True)
    _t = time.time()
    train_neg, train_reject_stats = create_negative_pairs_v2(
        df,
        pos_pairs=train_pos,
        num_negatives=int(len(train_pos) * neg_to_pos_ratio),
        isolate_ids=train_isolates,
        cooccur_pairs=cooccur_pairs,
        schema_pair=schema_pair,
        seed=seed,
        max_attempts_per_seq=max_attempts_per_seq,
        max_attempts_multiplier=max_attempts_multiplier,
        axis_quotas=axis_quotas,
    )
    print(f"[diag] split_dataset_v2: create_negative_pairs_v2 train in {time.time()-_t:.2f}s "
          f"({len(train_neg):,} pairs, {train_reject_stats.get('total_attempts', 0):,} attempts; "
          f"coverage={train_reject_stats['coverage_phase_pairs']:,}, "
          f"fill={train_reject_stats['fill_phase_pairs']:,})", flush=True)

    _t = time.time()
    val_neg, val_reject_stats = create_negative_pairs_v2(
        df,
        pos_pairs=val_pos,
        num_negatives=int(len(val_pos) * neg_to_pos_ratio),
        isolate_ids=val_isolates,
        cooccur_pairs=cooccur_pairs,
        schema_pair=schema_pair,
        seed=seed,
        max_attempts_per_seq=max_attempts_per_seq,
        max_attempts_multiplier=max_attempts_multiplier,
        axis_quotas=axis_quotas,
    )
    print(f"[diag] split_dataset_v2: create_negative_pairs_v2 val in {time.time()-_t:.2f}s "
          f"({len(val_neg):,} pairs)", flush=True)

    _t = time.time()
    test_neg, test_reject_stats = create_negative_pairs_v2(
        df,
        pos_pairs=test_pos,
        num_negatives=int(len(test_pos) * neg_to_pos_ratio),
        isolate_ids=test_isolates,
        cooccur_pairs=cooccur_pairs,
        schema_pair=schema_pair,
        seed=seed,
        max_attempts_per_seq=max_attempts_per_seq,
        max_attempts_multiplier=max_attempts_multiplier,
        axis_quotas=axis_quotas,
    )
    print(f"[diag] split_dataset_v2: create_negative_pairs_v2 test in {time.time()-_t:.2f}s "
          f"({len(test_neg):,} pairs)", flush=True)

    # Combine pos+neg per split, then attach axis flags, then compute exposure stats.
    train_pairs = pd.concat([train_pos, train_neg], ignore_index=True)
    val_pairs = pd.concat([val_pos, val_neg], ignore_index=True)
    test_pairs = pd.concat([test_pos, test_neg], ignore_index=True)

    # Schema-mode sanity assertions: every pair has func_a==func_left and func_b==func_right.
    for name, pairs_df in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        if len(pairs_df) == 0:
            continue
        bad = pairs_df[(pairs_df["func_a"] != func_left) | (pairs_df["func_b"] != func_right)]
        if not bad.empty:
            raise ValueError(
                f"Schema constraint violated in {name} split: expected func_a=={func_left!r} "
                f"and func_b=={func_right!r}, found {len(bad)} violating rows."
            )

    # Axis flag annotations.
    print("\nAttaching metadata-axis flags to pairs...")
    _t = time.time()
    train_pairs = compute_axis_flags(train_pairs, df, axes=axes_for_flags)
    val_pairs = compute_axis_flags(val_pairs, df, axes=axes_for_flags)
    test_pairs = compute_axis_flags(test_pairs, df, axes=axes_for_flags)
    print(f"[diag] split_dataset_v2: compute_axis_flags in {time.time()-_t:.2f}s", flush=True)

    # Exposure tables (computed on the pre-overlap-removal pos/neg DataFrames so
    # they reflect the actual rows that went into pair_key dedup).
    _t = time.time()
    train_exp = compute_exposure_stats(train_pos, train_neg, df)
    val_exp = compute_exposure_stats(val_pos, val_neg, df)
    test_exp = compute_exposure_stats(test_pos, test_neg, df)
    print(f"[diag] split_dataset_v2: compute_exposure_stats in {time.time()-_t:.2f}s", flush=True)

    # Cross-split pair_key overlap check (same logic as v1: keep in train,
    # remove from val/test).
    print("\nValidating pair_key partitioning...")
    _t = time.time()
    train_pair_keys = set(train_pairs['pair_key'])
    val_pair_keys = set(val_pairs['pair_key'])
    test_pair_keys = set(test_pairs['pair_key'])
    val_pairs_before = len(val_pairs)
    test_pairs_before = len(test_pairs)
    train_val_key_overlap = train_pair_keys & val_pair_keys
    train_test_key_overlap = train_pair_keys & test_pair_keys
    val_test_key_overlap = val_pair_keys & test_pair_keys
    total_key_overlap = len(train_val_key_overlap) + len(train_test_key_overlap) + len(val_test_key_overlap)

    if total_key_overlap > 0:
        train_val_pct = (len(train_val_key_overlap) / val_pairs_before * 100) if val_pairs_before > 0 else 0
        train_test_pct = (len(train_test_key_overlap) / test_pairs_before * 100) if test_pairs_before > 0 else 0
        val_test_pct = (len(val_test_key_overlap) / test_pairs_before * 100) if test_pairs_before > 0 else 0
        print(f"   WARNING: Found {total_key_overlap} overlapping pair_keys across splits!")
        print(f"      Train-Val:  {len(train_val_key_overlap)} ({train_val_pct:.2f}% of val)")
        print(f"      Train-Test: {len(train_test_key_overlap)} ({train_test_pct:.2f}% of test)")
        print(f"      Val-Test:   {len(val_test_key_overlap)} ({val_test_pct:.2f}% of test)")
        val_pairs = val_pairs[~val_pairs['pair_key'].isin(train_val_key_overlap | val_test_key_overlap)]
        test_pairs = test_pairs[~test_pairs['pair_key'].isin(train_test_key_overlap | val_test_key_overlap)]
        val_removed = val_pairs_before - len(val_pairs)
        test_removed = test_pairs_before - len(test_pairs)
        val_removed_pct = (val_removed / val_pairs_before * 100) if val_pairs_before > 0 else 0
        test_removed_pct = (test_removed / test_pairs_before * 100) if test_pairs_before > 0 else 0
        print(f"   After removal: Train={len(train_pairs)}, "
              f"Val={len(val_pairs)} (removed {val_removed}, {val_removed_pct:.2f}%), "
              f"Test={len(test_pairs)} (removed {test_removed}, {test_removed_pct:.2f}%)")
    else:
        print(f"   No pair_key overlap detected. Train, Val, Test are mutually exclusive on pair_key.")
    print(f"[diag] split_dataset_v2: pair_key validation in {time.time()-_t:.2f}s", flush=True)

    # Validate non-empty splits (preserved from v1).
    if len(train_pairs) == 0 or len(val_pairs) == 0 or len(test_pairs) == 0:
        raise ValueError('One or more sets are empty.')

    # Check isolate overlap in pairs (preserved from v1; useful even with
    # hard_partition_isolates hard-coded to True).
    train_set = set(train_pairs['assembly_id_a']).union(set(train_pairs['assembly_id_b']))
    val_set = set(val_pairs['assembly_id_a']).union(set(val_pairs['assembly_id_b']))
    test_set = set(test_pairs['assembly_id_a']).union(set(test_pairs['assembly_id_b']))
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"Error: Overlap detected in isolate sets!")
        if train_val_overlap:
            print(f"Train-Val overlap: {train_val_overlap}")
        if train_test_overlap:
            print(f"Train-Test overlap: {train_test_overlap}")
        if val_test_overlap:
            print(f"Val-Test overlap: {val_test_overlap}")
        raise ValueError("Isolate overlap detected in pair sets.")

    duplicate_stats = {
        'cooccur_stats': cooccur_stats,
        'train_reject_stats': train_reject_stats,
        'val_reject_stats': val_reject_stats,
        'test_reject_stats': test_reject_stats,
        'pair_key_overlaps': {
            'train_val': {
                'count': len(train_val_key_overlap),
                'pct_of_val': (len(train_val_key_overlap) / val_pairs_before * 100) if val_pairs_before > 0 else 0,
            },
            'train_test': {
                'count': len(train_test_key_overlap),
                'pct_of_test': (len(train_test_key_overlap) / test_pairs_before * 100) if test_pairs_before > 0 else 0,
            },
            'val_test': {
                'count': len(val_test_key_overlap),
                'pct_of_test': (len(val_test_key_overlap) / test_pairs_before * 100) if test_pairs_before > 0 else 0,
            },
            'total_overlap': total_key_overlap,
            'val_pairs_before_removal': val_pairs_before,
            'test_pairs_before_removal': test_pairs_before,
            'val_pairs_after_removal': len(val_pairs),
            'test_pairs_after_removal': len(test_pairs),
            'val_removed_pct': ((val_pairs_before - len(val_pairs)) / val_pairs_before * 100) if val_pairs_before > 0 else 0,
            'test_removed_pct': ((test_pairs_before - len(test_pairs)) / test_pairs_before * 100) if test_pairs_before > 0 else 0,
        },
        # Global pair_key dedup stats (one pass over the full df). In the
        # global-pos flow there is no within-split dedup — every pair_key has
        # one home isolate and lands in exactly one split — so per-split dedup
        # numbers would be degenerate.
        'pos_dedup_global': pos_dedup_stats,
        'coverage_stats': {
            'train': _coverage_substats(train_reject_stats),
            'val': _coverage_substats(val_reject_stats),
            'test': _coverage_substats(test_reject_stats),
        },
    }
    exposure_tables = {'train': train_exp, 'val': val_exp, 'test': test_exp}
    return train_pairs, val_pairs, test_pairs, duplicate_stats, exposure_tables


def _coverage_substats(reject_stats: dict) -> dict:
    """Pluck the coverage-relevant sub-keys out of rejection_stats for cleaner
    duplicate_stats serialization."""
    return {
        'requested_negatives': reject_stats.get('requested_negatives', 0),
        'min_required_for_coverage': reject_stats.get('min_required_for_coverage', 0),
        'achieved_negatives': reject_stats.get('achieved_negatives', 0),
        'coverage_phase_pairs': reject_stats.get('coverage_phase_pairs', 0),
        'fill_phase_pairs': reject_stats.get('fill_phase_pairs', 0),
        'coverage_overrode_ratio': reject_stats.get('coverage_overrode_ratio', False),
        'seqs_with_zero_negatives': reject_stats.get('seqs_with_zero_negatives', []),
        'coverage_skipped': reject_stats.get('coverage_skipped', []),
    }


# ---------------------------------------------------------------------------
# 4.7 generate_all_cv_folds_v2
# ---------------------------------------------------------------------------
def generate_all_cv_folds_v2(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
    neg_to_pos_ratio: float,
    val_ratio: float,
    schema_pair: Tuple[str, str],
    max_attempts_per_seq: int = 50,
    max_attempts_multiplier: int = 100,
    axes_for_flags: list = _DEFAULT_AXES,
    axis_quotas: Optional[dict] = None,
    ) -> Iterator[dict]:
    """Generate all N CV fold splits, yielding each as a dict containing
    fold_id, train_pairs, val_pairs, test_pairs, duplicate_stats, and
    exposure_tables.

    v2 hard-codes pair_mode='schema_ordered', allow_same_func_negatives=False,
    canonicalize_pair_orientation_enabled=False, hard_partition_isolates=True.
    See dataset_segment_pairs.py (v1) for the configurable versions.

    cooccur_pairs is built once at the dataset level and reused across folds
    (matches v1's CV behavior).
    """
    from sklearn.model_selection import KFold

    print("\nBuilding co-occurrence set (sequences that appear together in any isolate)...")
    cooccur_pairs, cooccur_stats = build_cooccurrence_set(df)
    print(f"   Total co-occurring sequence pairs: {cooccur_stats['total_cooccur_pairs']:,}")
    print(f"   Pairs appearing in multiple isolates: {cooccur_stats['pairs_in_multiple_isolates']:,}")
    print(f"   Max isolates for a single pair: {cooccur_stats['max_isolates_per_pair']}")

    unique_isolates = np.array(sorted(df['assembly_id'].unique()))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    val_frac = val_ratio / (1.0 - 1.0 / n_folds)
    val_frac = min(val_frac, 0.5)

    for fold_i, (trainval_idx, test_idx) in enumerate(kf.split(unique_isolates)):
        print(f"\n{'='*60}")
        print(f"CV (v2): generating fold {fold_i + 1}/{n_folds}  "
              f"(test={len(test_idx)}, trainval={len(trainval_idx)} isolates)")
        print(f"{'='*60}")

        trainval_ids = unique_isolates[trainval_idx].tolist()
        test_ids = unique_isolates[test_idx].tolist()

        if len(trainval_ids) < 2:
            train_ids, val_ids = trainval_ids, []
        else:
            train_ids, val_ids = train_test_split(
                trainval_ids, test_size=val_frac, random_state=seed + fold_i
            )

        train_pairs, val_pairs, test_pairs, dup_stats, exposure_tables = split_dataset_v2(
            df=df,
            schema_pair=schema_pair,
            neg_to_pos_ratio=neg_to_pos_ratio,
            train_ratio=0.8,
            val_ratio=val_ratio,
            seed=seed + fold_i,
            max_attempts_per_seq=max_attempts_per_seq,
            max_attempts_multiplier=max_attempts_multiplier,
            axes_for_flags=axes_for_flags,
            axis_quotas=axis_quotas,
            train_isolates_override=train_ids,
            val_isolates_override=val_ids,
            test_isolates_override=test_ids,
            cooccur_pairs=cooccur_pairs,
            cooccur_stats=cooccur_stats,
        )
        yield {
            'fold_id': fold_i,
            'train_pairs': train_pairs,
            'val_pairs': val_pairs,
            'test_pairs': test_pairs,
            'duplicate_stats': dup_stats,
            'exposure_tables': exposure_tables,
        }


# ---------------------------------------------------------------------------
# 4.8 generate_temporal_split_v2 (stub)
# ---------------------------------------------------------------------------
def generate_temporal_split_v2(*args, **kwargs):
    raise NotImplementedError(
        "Temporal split is not yet supported in pair_builder_version=v2. "
        "Use pair_builder_version=v1 for temporal experiments, or contribute v2 support."
    )


# ---------------------------------------------------------------------------
# 4.9 save_split_output_v2
# ---------------------------------------------------------------------------
def save_split_output_v2(
    output_dir: Path,
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    duplicate_stats: dict,
    exposure_tables: dict,
    df: pd.DataFrame,
    config_bundle: str,
    schema_pair,
    filters_applied: dict,
    axes_for_flags: list = _DEFAULT_AXES,
    generate_visualizations: bool = True,
    skip_esm_pca_plots: bool = False,
    skip_kmer_pca_plots: bool = False,
    ) -> None:
    """Save v2 output files for one split (or one CV fold).

    Same v1 outputs (pair CSVs/parquets, isolate_metadata.csv, dataset_stats.json,
    duplicate_stats.json, cooccurring_sequence_pairs.csv, optional plots) plus:
      1. sequence_exposure.csv from exposure_tables.
      2. dataset_stats.json extended with pair_builder_version, exposure_summary,
         axis_flag_summary, pos_dedup, coverage sections.

    NOTE: metadata_coverage.json is NOT written here. Per spec §4.9/§4.10,
    that artifact has per-run scope and is written once by the CLI dispatch
    before split / CV branching.
    """
    breakpoint()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pair CSVs + parquets
    _t = time.time()
    print(f"[diag] save_v2: write pair CSVs start "
          f"(train={len(train_pairs):,}, val={len(val_pairs):,}, test={len(test_pairs):,})",
          flush=True)
    train_pairs.to_csv(output_dir / 'train_pairs.csv', index=False)
    val_pairs.to_csv(output_dir / 'val_pairs.csv', index=False)
    test_pairs.to_csv(output_dir / 'test_pairs.csv', index=False)
    print(f"[diag] save_v2: pair CSVs done in {time.time()-_t:.2f}s", flush=True)

    _t = time.time()
    print(f"[diag] save_v2: write pair parquets start", flush=True)
    train_pairs.to_parquet(output_dir / 'train_pairs.parquet', compression='zstd', index=False)
    val_pairs.to_parquet(output_dir / 'val_pairs.parquet', compression='zstd', index=False)
    test_pairs.to_parquet(output_dir / 'test_pairs.parquet', compression='zstd', index=False)
    print(f"[diag] save_v2: pair parquets done in {time.time()-_t:.2f}s", flush=True)

    # Per-split sequence_exposure.csv
    for split_name, exp_df in exposure_tables.items():
        if exp_df is None or len(exp_df) == 0:
            continue
        out = exp_df.copy()
        # assembly_ids -> ';'-joined string for CSV roundtrip
        out['assembly_ids'] = out['assembly_ids'].apply(lambda lst: ';'.join(sorted(lst)) if lst else '')
        out.to_csv(output_dir / f'sequence_exposure_{split_name}.csv', index=False)
    print(f"Saved per-split sequence_exposure_*.csv to: {output_dir}")

    # Per-split isolate sets (used for stats)
    train_iso = set(train_pairs['assembly_id_a']).union(set(train_pairs['assembly_id_b']))
    val_iso = set(val_pairs['assembly_id_a']).union(set(val_pairs['assembly_id_b']))
    test_iso = set(test_pairs['assembly_id_a']).union(set(test_pairs['assembly_id_b']))

    total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
    dataset_stats = {
        'pair_builder_version': 'v2',
        'split_sizes': {
            'train': {
                'pairs': len(train_pairs), 'isolates': len(train_iso),
                'positive_pairs': int(train_pairs['label'].sum()),
                'negative_pairs': int((train_pairs['label'] == 0).sum()),
                'positive_ratio': round(float(train_pairs['label'].mean()), 3),
            },
            'val': {
                'pairs': len(val_pairs), 'isolates': len(val_iso),
                'positive_pairs': int(val_pairs['label'].sum()),
                'negative_pairs': int((val_pairs['label'] == 0).sum()),
                'positive_ratio': round(float(val_pairs['label'].mean()), 3),
            },
            'test': {
                'pairs': len(test_pairs), 'isolates': len(test_iso),
                'positive_pairs': int(test_pairs['label'].sum()),
                'negative_pairs': int((test_pairs['label'] == 0).sum()),
                'positive_ratio': round(float(test_pairs['label'].mean()), 3),
            },
        },
        'total': {
            'pairs': total_pairs,
            'isolates': len(df['assembly_id'].unique()),
        },
        'metadata_distributions': {
            'train': get_metadata_distributions(df, train_iso),
            'val':   get_metadata_distributions(df, val_iso),
            'test':  get_metadata_distributions(df, test_iso),
        },
        'co_occurrence_blocking': {
            'total_cooccur_pairs':        duplicate_stats['cooccur_stats']['total_cooccur_pairs'],
            'pairs_in_multiple_isolates': duplicate_stats['cooccur_stats']['pairs_in_multiple_isolates'],
            'max_isolates_per_pair':      duplicate_stats['cooccur_stats']['max_isolates_per_pair'],
            'train_blocked': duplicate_stats['train_reject_stats']['blocked_cooccur'],
            'val_blocked':   duplicate_stats['val_reject_stats']['blocked_cooccur'],
            'test_blocked':  duplicate_stats['test_reject_stats']['blocked_cooccur'],
            'total_blocked': (duplicate_stats['train_reject_stats']['blocked_cooccur'] +
                              duplicate_stats['val_reject_stats']['blocked_cooccur'] +
                              duplicate_stats['test_reject_stats']['blocked_cooccur']),
        },
        'filters_applied': filters_applied,
        'exposure_summary': _exposure_summary(exposure_tables),
        'axis_flag_summary': {
            'train': _axis_flag_summary(train_pairs, axes_for_flags),
            'val':   _axis_flag_summary(val_pairs, axes_for_flags),
            'test':  _axis_flag_summary(test_pairs, axes_for_flags),
        },
        'pos_dedup': {
            'n_before': duplicate_stats['pos_dedup_global']['n_pos_before_dedup'],
            'n_after': duplicate_stats['pos_dedup_global']['n_pos_after_dedup'],
            'n_dropped': duplicate_stats['pos_dedup_global']['n_pos_duplicates_dropped'],
        },
        'coverage': {
            split: {
                'requested_negatives': cs['requested_negatives'],
                'min_required_for_coverage': cs['min_required_for_coverage'],
                'achieved_negatives': cs['achieved_negatives'],
                'coverage_phase_pairs': cs['coverage_phase_pairs'],
                'fill_phase_pairs': cs['fill_phase_pairs'],
                'coverage_overrode_ratio': cs['coverage_overrode_ratio'],
                'n_seqs_with_zero_negatives': len(cs['seqs_with_zero_negatives']),
            }
            for split, cs in duplicate_stats['coverage_stats'].items()
        },
    }

    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(dataset_stats, f, indent=2, default=_jsonable)
    print(f"Saved dataset stats to: {output_dir / 'dataset_stats.json'}")

    # Per-isolate metadata
    isolate_metadata_cols = ['assembly_id', 'host', 'hn_subtype', 'year']
    if 'geo_location_clean' in df.columns:
        isolate_metadata_cols.append('geo_location_clean')
    if 'passage' in df.columns:
        isolate_metadata_cols.append('passage')
    try:
        isolate_meta = df.groupby('assembly_id')[isolate_metadata_cols].first().reset_index(drop=True)
        isolate_meta.to_csv(output_dir / 'isolate_metadata.csv', index=False)
        print(f"Saved isolate metadata to: {output_dir / 'isolate_metadata.csv'}")
    except Exception as e:
        print(f"WARNING: failed to save isolate_metadata.csv ({type(e).__name__}: {e})")

    # Duplicate/rejection stats
    cooccur_stats = duplicate_stats['cooccur_stats']
    cooccur_stats_summary = {k: v for k, v in cooccur_stats.items() if k != 'isolate_pair_counts'}
    duplicate_summary = {
        'cooccurrence': cooccur_stats_summary,
        'negative_pair_rejections': {
            'train': dict(duplicate_stats['train_reject_stats']),
            'val':   dict(duplicate_stats['val_reject_stats']),
            'test':  dict(duplicate_stats['test_reject_stats']),
        },
        'pair_key_overlaps': duplicate_stats['pair_key_overlaps'],
        'pos_dedup': duplicate_stats['pos_dedup_global'],
        'coverage': duplicate_stats['coverage_stats'],
    }
    with open(output_dir / 'duplicate_stats.json', 'w') as f:
        json.dump(duplicate_summary, f, indent=2, default=_jsonable)
    print(f"Saved duplicate stats to: {output_dir / 'duplicate_stats.json'}")

    # Co-occurring pairs list (same as v1)
    if cooccur_stats['total_cooccur_pairs'] > 0:
        isolate_pair_counts = cooccur_stats.get('isolate_pair_counts', {})
        if isolate_pair_counts:
            cooccur_df = pd.DataFrame([
                {'pair_key': k, 'num_isolates': v} for k, v in isolate_pair_counts.items()
            ]).sort_values('num_isolates', ascending=False)
            cooccur_df.to_csv(output_dir / 'cooccurring_sequence_pairs.csv', index=False)
            print(f"Saved {len(cooccur_df)} co-occurring sequence pairs to: cooccurring_sequence_pairs.csv")

    # Visualization plots (best-effort; v1 visualizer takes the same dataset_stats.json)
    if generate_visualizations:
        try:
            print(f'\nGenerating dataset visualization plots...')
            from src.analysis.visualize_dataset_stats import visualize_dataset_stats
            visualize_dataset_stats(
                dataset_stats_path=output_dir / 'dataset_stats.json',
                bundle_name=config_bundle,
                output_dir_dataset=output_dir,
                skip_esm_pca_plots=skip_esm_pca_plots,
                skip_kmer_pca_plots=skip_kmer_pca_plots,
            )
        except Exception as e:
            print(f"WARNING: Failed to generate visualizations: {e}")


def _exposure_summary(exposure_tables: dict) -> dict:
    """Summarize per-split exposure tables for dataset_stats.json."""
    out = {}
    for split, exp_df in exposure_tables.items():
        if exp_df is None or len(exp_df) == 0:
            out[split] = {'dual': 0, 'pos_only': 0, 'neg_only': 0, 'n_unique_seqs': 0}
            continue
        counts = exp_df['exposure'].value_counts()
        out[split] = {
            'dual': int(counts.get('dual', 0)),
            'pos_only': int(counts.get('pos_only', 0)),
            'neg_only': int(counts.get('neg_only', 0)),
            'n_unique_seqs': int(len(exp_df)),
        }
    return out


def _axis_flag_summary(pairs: pd.DataFrame, axes: list) -> dict:
    """Summarize same/diff/null counts per axis for one split."""
    out = {}
    if len(pairs) == 0:
        return {axis: {'same': 0, 'diff': 0, 'null': 0} for axis in axes}
    for axis in axes:
        col = f'same_{axis}'
        if col not in pairs.columns:
            out[axis] = {'present': False}
            continue
        s = pairs[col]
        same_n = int((s == True).sum())  # noqa: E712 -- nullable boolean comparison
        diff_n = int((s == False).sum())  # noqa: E712
        null_n = int(s.isna().sum())
        out[axis] = {'same': same_n, 'diff': diff_n, 'null': null_n}
    return out


# ---------------------------------------------------------------------------
# 5. v2 config validation
# ---------------------------------------------------------------------------
def _validate_v2_config(config) -> None:
    """Validate that the loaded config is compatible with v2's hard-coded
    constraints. Raises ValueError / NotImplementedError with messages that
    name the offending key, the value found, and what v2 requires.

    Reads the raw config (via OmegaConf.select) so that "absent" and
    "explicitly set to a forbidden value" are distinguishable in error
    messages. Runtime extraction further down the dispatch may use getattr
    with v2-appropriate defaults; the two patterns are deliberately separate.
    """
    from omegaconf import OmegaConf

    # Required: schema_pair
    schema_pair = OmegaConf.select(config, "dataset.schema_pair")
    if schema_pair is None:
        raise ValueError(
            "v2 requires dataset.schema_pair to be set; got null/absent. "
            "Set it to a [func_left, func_right] list with func_left != func_right."
        )
    schema_list = list(schema_pair)
    if len(schema_list) != 2:
        raise ValueError(
            f"v2 requires dataset.schema_pair to be length 2; got {schema_list!r}."
        )
    if schema_list[0] == schema_list[1]:
        raise ValueError(
            f"v2 requires dataset.schema_pair to contain two different functions; "
            f"got {schema_list!r}."
        )

    # Hard-coded values that must not contradict
    pair_mode = OmegaConf.select(config, "dataset.pair_mode")
    if pair_mode is not None and pair_mode != "schema_ordered":
        raise ValueError(
            f"v2 requires dataset.pair_mode='schema_ordered' (or absent); "
            f"got {pair_mode!r}."
        )

    allow_same = OmegaConf.select(config, "dataset.allow_same_func_negatives")
    if allow_same is not None and allow_same is not False:
        raise ValueError(
            f"v2 requires dataset.allow_same_func_negatives=false (or absent); "
            f"got {allow_same!r}."
        )

    canon = OmegaConf.select(config, "dataset.canonicalize_pair_orientation")
    if canon is not None and canon is not False:
        raise ValueError(
            f"v2 requires dataset.canonicalize_pair_orientation=false (or absent); "
            f"got {canon!r}."
        )

    hard_part = OmegaConf.select(config, "dataset.hard_partition_isolates")
    if hard_part is not None and hard_part is not True:
        raise ValueError(
            f"v2 requires dataset.hard_partition_isolates=true (or absent); "
            f"got {hard_part!r}."
        )

    # axis_quotas placeholder
    axis_quotas = OmegaConf.select(config, "dataset.axis_quotas")
    if axis_quotas is not None and len(axis_quotas) > 0:
        raise NotImplementedError(
            f"v2 does not yet implement dataset.axis_quotas; got {axis_quotas!r}. "
            "Set it to null or omit."
        )

    # Temporal not yet supported
    year_train = OmegaConf.select(config, "dataset.year_train")
    if year_train is not None:
        raise ValueError(
            "v2 does not yet support temporal split (dataset.year_train). "
            "Use pair_builder_version=v1 for temporal experiments, or contribute v2 support."
        )
