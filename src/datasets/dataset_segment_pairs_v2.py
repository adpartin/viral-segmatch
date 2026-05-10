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

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', 7)

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
    pos_df: pd.DataFrame,
    num_negatives: int,
    cooccur_pairs: set,
    schema_pair: Tuple[str, str],
    seed: int = 42,
    max_attempts_per_seq: int = 50,
    max_attempts_multiplier: int = 100,
    axis_quotas: Optional[dict] = None,
    forbidden_pair_keys: Optional[set] = None,
    ) -> tuple[pd.DataFrame, dict]:
    """Generate negative pairs with a coverage-first two-phase sampler.

    Operates entirely on `pos_df` (the within-split positive table from
    `create_positive_pairs_v2` + the split partition). Under v2's strict
    invariant (`pos_df['assembly_id_a'].is_unique`), each row is one
    isolate's full schema-pair record: slot-A fields in `_a` columns and
    slot-B fields in `_b` columns. Both the "self" side of a coverage pair
    and the "partner" side are sourced from these rows -- no original `df`
    is needed.

    Two phases:
      1. Coverage phase: iterate every (slot, dna_hash) in `pos_df` (sorted)
         and try up to `max_attempts_per_seq` random partner isolates until
         at least one accepted negative covers that DNA. The DNA-level
         coverage is **best-effort**: per the apriori feasibility analysis
         (docs/results/2026-05-08_dna_coverage_feasibility_sweep.md), a few
         DNA variants in tightly-filtered bundles cannot be covered when the
         dominant protein has more DNA encodings than the partner-protein
         universe can supply distinct neg pair_keys for. Uncovered DNAs are
         logged in rejection_stats['dna_hashes_with_zero_negatives'] but do
         NOT raise. The seq_hash-level guarantee (every seq_hash gets >=1
         neg) is enforced as a hard raise at the end -- DNA-level coverage
         subsumes seq_hash-level coverage in the typical case, so this raise
         only fires if every DNA encoding a seq_hash failed.
      2. Fill phase: if coverage produced fewer than `num_negatives`, top up
         by sampling two distinct isolates and pairing the first's slot-A
         row with the second's slot-B row, until quota is met or the attempt
         budget exhausts.

    Both phases route candidates through `_try_accept`, which short-circuits
    on cooccur/duplicate-brc/duplicate-seq/forbidden_pair_keys before any
    counter or list grows. Rejected candidates do not count toward either
    phase's quota.

    Coverage floor (interaction with `num_negatives`): the minimum number of
    negatives this function will produce is
    `max(|unique_slot_A_dnas|, |unique_slot_B_dnas|)` over `pos_df` (DNA
    level, not seq_hash level). If `num_negatives` falls below that, the
    coverage phase produces ~floor pairs anyway and the fill phase becomes
    a no-op; `coverage_overrode_ratio` is set to True and a warning is
    logged.

    forbidden_pair_keys: optional set of canonical pair_keys that must not
    appear in the output. Used by `split_dataset_v2` to thread already-
    generated negatives from previous splits through subsequent calls,
    making cross-split neg-neg collisions impossible by construction. If
    None or empty, behaves identically to the prior independent-per-split
    behavior. Rejections via this set are counted under
    `cross_split_collision` in rejection_stats.

    Returns:
        (neg_df, rejection_stats). rejection_stats includes blocked_cooccur,
        duplicate_brc, duplicate_seq, cross_split_collision, total_attempts,
        coverage_skipped, and v2 keys: requested_negatives,
        min_required_for_coverage, coverage_phase_pairs, fill_phase_pairs,
        achieved_negatives, coverage_overrode_ratio, seqs_with_zero_negatives,
        dna_hashes_with_zero_negatives, n_dna_uncovered.
    """
    # Slot identity is carried by the _a/_b columns of pos_df, so we no longer
    # need func_left / func_right inside this function. The call is kept for
    # its side effect: rejecting a malformed schema_pair early.
    _validate_schema_pair(schema_pair, "create_negative_pairs_v2")
    if axis_quotas is not None and len(axis_quotas) > 0:
        raise NotImplementedError("axis_quotas not yet supported; pass None")
    if len(pos_df) == 0:
        raise ValueError("create_negative_pairs_v2: pos_df is empty; nothing to generate negatives for.")

    rng = random.Random(seed)
    t_setup = time.time()
    print(f"[diag] create_negative_pairs_v2: setup start (|pos_df|={len(pos_df):,})", flush=True)

    # Coverage targets: seq_hashes appearing in slot A and slot B of pos_df.
    # In schema mode each seq_hash has exactly one function, so
    # target_seqs_left and target_seqs_right are disjoint.
    target_seqs_left = set(pos_df['seq_hash_a'].tolist())
    target_seqs_right = set(pos_df['seq_hash_b'].tolist())
    shared = target_seqs_left & target_seqs_right
    if shared:
        offenders = list(shared)[:5]
        raise ValueError(
            f"create_negative_pairs_v2: {len(shared)} seq_hash(es) appear in both "
            f"slot A and slot B (e.g., {offenders})."
        )

    neg_count_a: dict = {s: 0 for s in target_seqs_left}
    neg_count_b: dict = {s: 0 for s in target_seqs_right}

    # DNA-level coverage targets (option C from
    # docs/results/2026-05-08_dna_coverage_feasibility_sweep.md): per-slot
    # unique dna_hash sets and per-(slot, dna) neg counts. The coverage
    # phase iterates these instead of seq_hashes so that every DNA variant
    # of every protein gets at least one negative counterexample where
    # feasible. Uncovered DNAs are logged, not raised (best-effort).
    target_dnas_left = set(pos_df['dna_hash_a'].tolist())
    target_dnas_right = set(pos_df['dna_hash_b'].tolist())
    neg_count_dna_a: dict = {d: 0 for d in target_dnas_left}
    neg_count_dna_b: dict = {d: 0 for d in target_dnas_right}

    # Single pass over pos_df builds:
    #   isolate_to_row     - aid -> the row for that isolate (it's 1:1)
    #   seq_hash_to_row    - seq -> first row in which it appears (used as the
    #                        "self" representative in the coverage phase) --> NOTE. What if a seq appears in multiple isolates? We could pick one at random, but that would add complexity and non-determinism. Instead we just take the first, which is deterministic but arbitrary. This means some isolates get favored as coverage partners over others; we can later analyze the distribution of coverage partners per isolate to see if this is a problem.
    #   dna_to_row_a       - dna -> first row in which it appears as slot-A.
    #   dna_to_row_b       - dna -> first row in which it appears as slot-B.
    #                        Used as the "self" representative in DNA-level
    #                        coverage (analogous to seq_hash_to_row, but
    #                        per-slot to give us a row whose slot-X dna_hash
    #                        is the target DNA we want to cover).
    #   seq_to_isolates    - seq -> set of in-split isolates that contain it.
    #                        Used to exclude same-seq isolates when picking a
    #                        partner. Built from pos_df only.
    isolate_to_row: dict = {}
    seq_hash_to_row: dict = {}
    dna_to_row_a: dict = {}
    dna_to_row_b: dict = {}
    seq_to_isolates: dict = {}
    for row in pos_df.itertuples(index=False):
        aid = row.assembly_id_a
        isolate_to_row[aid] = row
        seq_hash_to_row.setdefault(row.seq_hash_a, row)
        seq_hash_to_row.setdefault(row.seq_hash_b, row)
        dna_to_row_a.setdefault(row.dna_hash_a, row)
        dna_to_row_b.setdefault(row.dna_hash_b, row)
        seq_to_isolates.setdefault(row.seq_hash_a, set()).add(aid)
        seq_to_isolates.setdefault(row.seq_hash_b, set()).add(aid)
    isolate_ids_list = sorted(isolate_to_row.keys())
    print(f"[diag] create_negative_pairs_v2: setup done in "
          f"{time.time()-t_setup:.2f}s ({len(isolate_ids_list):,} isolates, "
          f"{len(seq_to_isolates):,} unique seq_hashes; "
          f"{len(target_dnas_left):,} slot-A DNAs, "
          f"{len(target_dnas_right):,} slot-B DNAs).", flush=True)

    # Coverage floor: max(|unique_slot_A_dnas|, |unique_slot_B_dnas|). Each
    # accepted negative contributes one A-DNA-coverage and one B-DNA-coverage
    # simultaneously, so the floor is governed by whichever side has more
    # unique DNAs. This is strictly higher than the previous seq_hash floor
    # by exactly the synonymous-codon redundancy of the dataset.
    min_required_for_coverage = max(len(target_dnas_left), len(target_dnas_right))
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
    # forbidden_pair_keys is the set of canonical pair_keys already produced
    # in earlier (cross-split) calls. Treat None as empty-set so the check
    # is a uniform set membership test below.
    _forbidden = forbidden_pair_keys if forbidden_pair_keys is not None else set()
    rejection_stats: dict = {
        'blocked_cooccur': 0,
        'duplicate_brc': 0,
        'duplicate_seq': 0,
        'cross_split_collision': 0,
        'total_attempts': 0,
        'coverage_skipped': [],
    }
    neg_pairs: list[dict] = []

    def _build_neg_pair(a_src, b_src) -> dict:
        """Build a negative pair dict using slot-A fields from a_src and
        slot-B fields from b_src. Both args are pos_df itertuples rows."""
        return {
            'pair_key': canonical_pair_key(a_src.seq_hash_a, b_src.seq_hash_b),
            'assembly_id_a': a_src.assembly_id_a,
            'assembly_id_b': b_src.assembly_id_a,  # b_src's home isolate
            'brc_a': a_src.brc_a, 'brc_b': b_src.brc_b,
            'ctg_a': a_src.ctg_a, 'ctg_b': b_src.ctg_b,
            'seq_a': a_src.seq_a, 'seq_b': b_src.seq_b,
            'dna_seq_a': a_src.dna_seq_a, 'dna_seq_b': b_src.dna_seq_b,
            'seg_a': a_src.seg_a, 'seg_b': b_src.seg_b,
            'func_a': a_src.func_a, 'func_b': b_src.func_b,
            'seq_hash_a': a_src.seq_hash_a, 'seq_hash_b': b_src.seq_hash_b,
            'dna_hash_a': a_src.dna_hash_a, 'dna_hash_b': b_src.dna_hash_b,
            'label': 0,
        }

    def _try_accept(a_src, b_src) -> bool:
        """Apply rejection checks. Returns True iff the pair was accepted (and
        side-effects: appended to neg_pairs, seen_* sets updated, neg_count_a/b
        incremented). Rejection short-circuits before any side effect, so a
        rejected candidate never counts toward a phase quota."""
        brc_pair_key = tuple(sorted([a_src.brc_a, b_src.brc_b]))
        if brc_pair_key in seen_pairs:
            rejection_stats['duplicate_brc'] += 1
            return False
        seq_pair_key = canonical_pair_key(a_src.seq_hash_a, b_src.seq_hash_b)
        if seq_pair_key in cooccur_pairs:
            rejection_stats['blocked_cooccur'] += 1
            return False
        if seq_pair_key in seen_seq_pairs:
            rejection_stats['duplicate_seq'] += 1
            return False
        if seq_pair_key in _forbidden:
            # Already produced as a negative in an earlier (cross-split) call.
            # Reject to keep splits' pair_key sets disjoint by construction.
            rejection_stats['cross_split_collision'] += 1
            return False
        # Schema orientation is correct by construction (slot A = func_left,
        # slot B = func_right); no orientation swap needed.
        neg_pairs.append(_build_neg_pair(a_src, b_src))
        seen_pairs.add(brc_pair_key)
        seen_seq_pairs.add(seq_pair_key)
        if a_src.seq_hash_a in neg_count_a:
            neg_count_a[a_src.seq_hash_a] += 1
        if b_src.seq_hash_b in neg_count_b:
            neg_count_b[b_src.seq_hash_b] += 1
        if a_src.dna_hash_a in neg_count_dna_a:
            neg_count_dna_a[a_src.dna_hash_a] += 1
        if b_src.dna_hash_b in neg_count_dna_b:
            neg_count_dna_b[b_src.dna_hash_b] += 1
        return True

    # ---- Coverage phase --------------------------------------------------
    # Iterate every (slot, dna_hash) target. Each accepted neg covers one
    # slot-A DNA and one slot-B DNA simultaneously; iterating both slots
    # exposes any DNA that hasn't been covered by a side-effect of an
    # earlier iteration. The order is (slot a DNAs sorted) then (slot b
    # DNAs sorted) so the iteration is deterministic.
    target_dna_iter = (
        [(d, 'left')  for d in sorted(target_dnas_left)] +
        [(d, 'right') for d in sorted(target_dnas_right)]
    )
    n_targets = len(target_dna_iter)
    print(f"\nCoverage phase: {n_targets:,} target DNAs to cover "
          f"(min_required_for_coverage={min_required_for_coverage:,}).", flush=True)
    progress_step = max(n_targets // 20, 1)  # ~20 progress prints
    for i, (d, slot) in enumerate(target_dna_iter):
        if i > 0 and i % progress_step == 0:
            print(f"  coverage phase: {i:,}/{n_targets:,} target DNAs "
                  f"({100.0*i/n_targets:.1f}%); accepted={len(neg_pairs):,}, "
                  f"attempts={rejection_stats['total_attempts']:,}", flush=True)

        if slot == 'left':
            if neg_count_dna_a.get(d, 0) > 0:
                continue
            s_row = dna_to_row_a[d]
            s_in_left = True
            self_seq = s_row.seq_hash_a
        else:
            if neg_count_dna_b.get(d, 0) > 0:
                continue
            s_row = dna_to_row_b[d]
            s_in_left = False
            self_seq = s_row.seq_hash_b

        # Excluded partners: any isolate that shares the self-row's seq_hash.
        # Same-seq cross-pairs degenerate (would build a near-pos in feature
        # space). Exclusion is by seq_hash because that's the unit of
        # cooccur and pair_key identity; DNA-level membership is implicit
        # (every isolate with this DNA has this seq_hash).
        excluded = seq_to_isolates.get(self_seq, set())
        if len(excluded) >= len(isolate_ids_list):
            rejection_stats['coverage_skipped'].append((d, slot, 'no_other_isolate'))
            continue

        accepted = False
        for _attempt in range(max_attempts_per_seq):
            rejection_stats['total_attempts'] += 1
            other_aid = rng.choice(isolate_ids_list)
            if other_aid in excluded:
                continue
            other_row = isolate_to_row[other_aid]

            if s_in_left:
                a_src, b_src = s_row, other_row
            else:
                a_src, b_src = other_row, s_row

            if _try_accept(a_src, b_src):
                accepted = True
                break

        if not accepted:
            # Cap to 100 entries for log readability; further skips are dropped.
            if len(rejection_stats['coverage_skipped']) < 100:
                rejection_stats['coverage_skipped'].append((d, slot, 'attempts_exhausted'))

    coverage_phase_pairs = len(neg_pairs)
    print(f"Coverage phase complete: {coverage_phase_pairs:,} negatives produced "
          f"({rejection_stats['total_attempts']:,} attempts).", flush=True)

    # ---- Fill phase ------------------------------------------------------
    # ipdb.set_trace(context=10)
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
        _try_accept(isolate_to_row[aid1], isolate_to_row[aid2])

    fill_phase_pairs = len(neg_pairs) - coverage_phase_pairs

    if fill_phase_target > 0 and len(neg_pairs) < fill_phase_target and not coverage_overrode_ratio:
        print(
            f"WARNING: only generated {len(neg_pairs)}/{fill_phase_target} negative pairs "
            f"after {rejection_stats['total_attempts']} attempts. May indicate high "
            f"sequence overlap across isolates."
        )

    seqs_with_zero_negatives = [s for s, c in neg_count_a.items() if c == 0]
    seqs_with_zero_negatives.extend(s for s, c in neg_count_b.items() if c == 0)
    dna_uncovered_a = [d for d, c in neg_count_dna_a.items() if c == 0]
    dna_uncovered_b = [d for d, c in neg_count_dna_b.items() if c == 0]
    n_dna_uncovered = len(dna_uncovered_a) + len(dna_uncovered_b)

    rejection_stats['requested_negatives'] = int(requested_negatives)
    rejection_stats['min_required_for_coverage'] = int(min_required_for_coverage)
    rejection_stats['coverage_phase_pairs'] = int(coverage_phase_pairs)
    rejection_stats['fill_phase_pairs'] = int(fill_phase_pairs)
    rejection_stats['achieved_negatives'] = int(len(neg_pairs))
    rejection_stats['coverage_overrode_ratio'] = bool(coverage_overrode_ratio)
    rejection_stats['seqs_with_zero_negatives'] = seqs_with_zero_negatives
    rejection_stats['n_dna_uncovered'] = int(n_dna_uncovered)
    rejection_stats['n_dna_uncovered_a'] = int(len(dna_uncovered_a))
    rejection_stats['n_dna_uncovered_b'] = int(len(dna_uncovered_b))
    # Cap full lists at 100 each for stats compactness; full audit lives in
    # rejection_stats['coverage_skipped'] which records (dna_hash, slot, reason).
    rejection_stats['dna_uncovered_a_sample'] = dna_uncovered_a[:100]
    rejection_stats['dna_uncovered_b_sample'] = dna_uncovered_b[:100]

    # DNA-level coverage is best-effort (per
    # docs/results/2026-05-08_dna_coverage_feasibility_sweep.md): some tight
    # bundles physically can't cover every DNA because the dominant protein
    # has more DNA encodings than the partner-protein universe can supply
    # distinct neg pair_keys for. WARN only; do not raise.
    if n_dna_uncovered > 0:
        print(
            f"WARNING: DNA-level coverage is partial -- "
            f"{n_dna_uncovered:,} DNAs received zero negative pairs "
            f"({len(dna_uncovered_a):,} on slot a, {len(dna_uncovered_b):,} on slot b). "
            f"This is expected on tight bundles where partner-protein supply runs out; "
            f"see rejection_stats['n_dna_uncovered'] and "
            f"docs/results/2026-05-08_dna_coverage_feasibility_sweep.md."
        )

    # Hard coverage check at the SEQ_HASH level: every seq_hash in pos_df
    # (slot A or slot B) must appear in at least one negative pair. The
    # DNA-level loop above will satisfy this in the typical case (covering
    # any one DNA of a seq_hash also covers that seq_hash), so this raise
    # is a safety net for the rare case where every DNA encoding a single
    # seq_hash failed.
    if seqs_with_zero_negatives:
        raise ValueError(
            f"create_negative_pairs_v2: protein-level coverage guarantee "
            f"violated -- {len(seqs_with_zero_negatives)} seq_hash(es) have "
            f"zero negative pairs (first 5: {seqs_with_zero_negatives[:5]}). "
            f"Every DNA encoding these seq_hashes failed; inspect "
            f"rejection_stats['coverage_skipped']. This is rarer than DNA-level "
            f"misses; if it fires, the bundle is likely too tight to be usable."
        )

    neg_df = pd.DataFrame(neg_pairs, columns=_PAIR_COLUMNS) if neg_pairs else pd.DataFrame(columns=_PAIR_COLUMNS)

    """
    ipdb.set_trace(context=10)
    a_counts = neg_df['seq_hash_a'].value_counts()  # Series: seq_hash → count
    b_counts = neg_df['seq_hash_b'].value_counts()

    print(a_counts.describe())   # min/25%/50%/75%/max for slot A
    print(b_counts.describe())
    print(f"slot A: {(a_counts == 0).sum()} uncovered")  # should be 0 after our hard check
    print(f"slot B: {(b_counts == 0).sum()} uncovered")

    # Tidy two-column report per seq_hash and slot:
    prev = pd.concat([
        a_counts.rename('n_as_slot_a'),
        b_counts.rename('n_as_slot_b'),
    ], axis=1).fillna(0).astype(int)
    prev['total'] = prev.sum(axis=1)
    prev.sort_values('total', ascending=False).head(20)  # most-used seqs

    # For coverage vs. positives sanity (every pos seq should be in the index):
    pos_a = set(pos_df['seq_hash_a'])
    pos_b = set(pos_df['seq_hash_b'])
    print('slot-A pos seqs missing from neg:', len(pos_a - set(a_counts.index)))
    print('slot-B pos seqs missing from neg:', len(pos_b - set(b_counts.index)))
    """
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

    Returns a DataFrame with one row per seq_hash that appears in pos_pairs OR
    neg_pairs. Columns: seq_hash, function, assembly_ids (list), n_pos_slot_a,
    n_pos_slot_b, n_pos_total, n_neg_slot_a, n_neg_slot_b, n_neg_total,
    exposure (`dual` / `pos_only` / `neg_only`).

    Slot semantics are fixed: slot A = func_left, slot B = func_right. So a
    func_left seq has n_*_slot_b == 0 by construction (literal 0, not null).
    """
    seq_to_func = _build_seq_to_func(df)

    # seq_hash -> sorted list of assembly_ids that contain it (full data, not
    # just this split -- biological duplicates outside the split are reported
    # for transparency). Sort once globally before groupby so each group's
    # list comes out sorted without a per-group lambda.
    seq_to_assemblies = (
        df[['seq_hash', 'assembly_id']]
          .drop_duplicates()
          .sort_values(['seq_hash', 'assembly_id'])
          .groupby('seq_hash')['assembly_id']
          .agg(list)
          .to_dict()
    )

    # Per-slot, per-label counts via value_counts (vectorized; no Python loop).
    # Each Series is indexed by seq_hash; concat aligns on the union of keys.
    def _slot_counts(pairs_df: pd.DataFrame, slot: str, label: str) -> pd.Series:
        name = f'n_{label}_slot_{slot}'
        if len(pairs_df) == 0:
            return pd.Series(dtype=int, name=name)
        return pairs_df[f'seq_hash_{slot}'].value_counts().rename(name)

    counts = pd.concat(
        [
            _slot_counts(pos_pairs, 'a', 'pos'),
            _slot_counts(pos_pairs, 'b', 'pos'),
            _slot_counts(neg_pairs, 'a', 'neg'),
            _slot_counts(neg_pairs, 'b', 'neg'),
        ],
        axis=1,
    ).fillna(0).astype(int)

    out_cols = [
        'seq_hash', 'function', 'assembly_ids',
        'n_pos_slot_a', 'n_pos_slot_b', 'n_pos_total',
        'n_neg_slot_a', 'n_neg_slot_b', 'n_neg_total',
        'exposure',
    ]
    if len(counts) == 0:
        return pd.DataFrame(columns=out_cols)

    counts.index.name = 'seq_hash'
    counts['n_pos_total'] = counts['n_pos_slot_a'] + counts['n_pos_slot_b']
    counts['n_neg_total'] = counts['n_neg_slot_a'] + counts['n_neg_slot_b']

    has_pos = counts['n_pos_total'] > 0
    has_neg = counts['n_neg_total'] > 0
    exposure = pd.Series('neg_only', index=counts.index, dtype='object')
    exposure[has_pos & has_neg] = 'dual'
    exposure[has_pos & ~has_neg] = 'pos_only'
    counts['exposure'] = exposure

    # Defensive: a seq_hash present in pairs but absent from df gets None /
    # [] (matches the old code's `.get(s)` / `.get(s, [])` semantics). Under
    # v2 strict mode this can't happen, but preserve the contract.
    counts['function'] = [seq_to_func.get(s) for s in counts.index]
    counts['assembly_ids'] = [seq_to_assemblies.get(s, []) for s in counts.index]

    return counts.reset_index()[out_cols]


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

    Lookup keys on `assembly_id`, not `seq_hash`. seq_hash collapses identical
    amino-acid sequences across isolates with legitimately different metadata
    (cross-host transmission, conserved sequences across years, etc.), so
    keying on it can return a different isolate's value than the pair's actual
    source isolate. assembly_id is the unique-per-isolate key that matches
    what the pair actually carries; metadata enrichment guarantees one value
    per assembly per axis. Quantified disagreement before the fix:
    `eda/probe_metadata_lookup.py`.

    Returns a copy of `pairs` with the new columns appended.
    """
    out = pairs.copy()
    if len(out) == 0:
        return out

    for axis in axes:
        col = axis
        if col not in df.columns:
            if axis == 'geo_location' and 'geo_location_clean' in df.columns:
                col = 'geo_location_clean'
            else:
                print(f"WARNING: axis {axis!r} not present in df; skipping axis flags.")
                continue

        per_axis = (
            df[['assembly_id', col]]
            .drop_duplicates('assembly_id')
            .set_index('assembly_id')[col]
        )

        a_vals = out['assembly_id_a'].map(per_axis)
        b_vals = out['assembly_id_b'].map(per_axis)
        out[f'{axis}_a'] = a_vals
        out[f'{axis}_b'] = b_vals

        both_present = a_vals.notna() & b_vals.notna()
        same = pd.Series(pd.NA, index=out.index, dtype='boolean')
        same.loc[both_present] = (a_vals == b_vals).loc[both_present]
        out[f'same_{axis}'] = same

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
        n_unique_seqs_null counts seq_hashes for which NO isolate carries
        any non-null value -- it's the embedding-keyed-lookup floor (an
        embedding referenced by such a seq has no metadata anywhere). For
        seq_hashes that span multiple isolates with different metadata
        (reassortment, multi-year persistence, etc.), per-sequence here
        does NOT mean "every isolate carrying this seq has this value";
        for that, look at pair-level <axis>_a / <axis>_b columns built by
        compute_axis_flags (which keys on assembly_id).

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

        # n_unique_seqs_null = seq_hashes with NO non-null value across any
        # isolate. Drop nulls first so seqs whose first row happens to be null
        # but which have a valid value elsewhere are not undercounted.
        n_unique_seqs = int(df['seq_hash'].nunique())
        seqs_with_value = df.dropna(subset=[col])['seq_hash'].nunique()
        n_unique_seqs_null = n_unique_seqs - int(seqs_with_value)
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

    NOTE/TODO: cross-split negative-negative pair_key collisions.
    Positives are pair_key-disjoint across splits by construction (global
    dedup gives each pair_key one home isolate). Negatives, however, are
    sampled independently per split, so two splits can in principle produce
    the same pair_key. The current code handles this with a post-hoc
    overlap cleanup that drops the duplicates from val/test -- which can
    silently void the per-seq coverage guarantee enforced inside
    `create_negative_pairs_v2`. In practice no collisions have been
    observed on Flu A so this is left as-is; if collisions show up, switch
    to threading a `forbidden_pair_keys` set through the val/test calls so
    overlap is impossible by construction.

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

    # Build the global positive pair dataframe once, deduped to one row per unique
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
    # `forbidden_so_far` accumulates pair_keys produced in earlier splits and
    # is threaded into subsequent calls. This makes cross-split neg-neg
    # collisions impossible by construction (rather than relying on a
    # post-hoc cleanup that silently voids the per-seq coverage guarantee).
    # Order matters: train first, val next (forbidden = train), test last
    # (forbidden = train | val).
    print("\nCreate negative pairs (coverage-first, schema mode)...", flush=True)
    forbidden_so_far: set = set()
    train_neg, train_reject_stats = create_negative_pairs_v2(
        pos_df=train_pos,
        num_negatives=int(len(train_pos) * neg_to_pos_ratio),
        cooccur_pairs=cooccur_pairs,
        schema_pair=schema_pair,
        seed=seed,
        max_attempts_per_seq=max_attempts_per_seq,
        max_attempts_multiplier=max_attempts_multiplier,
        axis_quotas=axis_quotas,
        forbidden_pair_keys=forbidden_so_far,
    )
    print(f"split_dataset_v2: train negatives created: "
          f"({len(train_neg):,} pairs, {train_reject_stats.get('total_attempts', 0):,} attempts; "
          f"coverage={train_reject_stats['coverage_phase_pairs']:,}, "
          f"fill={train_reject_stats['fill_phase_pairs']:,})", flush=True)
    forbidden_so_far |= set(train_neg['pair_key'])

    val_neg, val_reject_stats = create_negative_pairs_v2(
        pos_df=val_pos,
        num_negatives=int(len(val_pos) * neg_to_pos_ratio),
        cooccur_pairs=cooccur_pairs,
        schema_pair=schema_pair,
        seed=seed,
        max_attempts_per_seq=max_attempts_per_seq,
        max_attempts_multiplier=max_attempts_multiplier,
        axis_quotas=axis_quotas,
        forbidden_pair_keys=forbidden_so_far,
    )
    forbidden_so_far |= set(val_neg['pair_key'])

    test_neg, test_reject_stats = create_negative_pairs_v2(
        pos_df=test_pos,
        num_negatives=int(len(test_pos) * neg_to_pos_ratio),
        cooccur_pairs=cooccur_pairs,
        schema_pair=schema_pair,
        seed=seed,
        max_attempts_per_seq=max_attempts_per_seq,
        max_attempts_multiplier=max_attempts_multiplier,
        axis_quotas=axis_quotas,
        forbidden_pair_keys=forbidden_so_far,
    )

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
    train_pairs = compute_axis_flags(train_pairs, df, axes=axes_for_flags)
    val_pairs = compute_axis_flags(val_pairs, df, axes=axes_for_flags)
    test_pairs = compute_axis_flags(test_pairs, df, axes=axes_for_flags)

    # Exposure tables (computed on the pre-overlap-removal pos/neg DataFrames so
    # they reflect the actual rows that went into pair_key dedup).
    # ipdb.set_trace(context=10)
    print(f"Compute exposure stats", flush=True)
    train_exp = compute_exposure_stats(train_pos, train_neg, df)
    val_exp = compute_exposure_stats(val_pos, val_neg, df)
    test_exp = compute_exposure_stats(test_pos, test_neg, df)
    # jj = test_exp[['seq_hash', 'function', 'n_pos_total', 'n_neg_total', 'exposure']]
    # jj['n_pos_total'] - jj['n_neg_total']

    # Cross-split pair_key overlap check.
    # Under v2's strict regime (global pos dedup + cooccur_pairs blocking pos
    # vs neg + forbidden_pair_keys threading across split-level neg calls),
    # train/val/test pair_keys are disjoint by construction. Any overlap here
    # is a serious leakage condition: identical (seq_a, seq_b) negatives in
    # two splits give the model the same (features, label) example in both
    # training AND evaluation, silently inflating metrics. Treat as fatal.
    print("\nValidating pair_key partitioning...")
    train_pair_keys = set(train_pairs['pair_key'])
    val_pair_keys = set(val_pairs['pair_key'])
    test_pair_keys = set(test_pairs['pair_key'])
    train_val_key_overlap = train_pair_keys & val_pair_keys
    train_test_key_overlap = train_pair_keys & test_pair_keys
    val_test_key_overlap = val_pair_keys & test_pair_keys
    total_key_overlap = (
        len(train_val_key_overlap)
        + len(train_test_key_overlap)
        + len(val_test_key_overlap)
    )
    if total_key_overlap > 0:
        raise ValueError(
            f"Cross-split pair_key collision detected (LEAKAGE): "
            f"train-val={len(train_val_key_overlap)}, "
            f"train-test={len(train_test_key_overlap)}, "
            f"val-test={len(val_test_key_overlap)}. "
            f"This means identical (seq_a, seq_b) negatives appear in two "
            f"splits, so the model would be evaluated on rows it trained on. "
            f"forbidden_pair_keys threading across the per-split "
            f"create_negative_pairs_v2 calls should make this impossible; "
            f"if this fires, check that the running `forbidden_so_far` set "
            f"is being passed and updated correctly between calls."
        )
    print("   No pair_key overlap. Train, Val, Test mutually exclusive on pair_key.")

    # Validate non-empty splits (preserved from v1).
    if len(train_pairs) == 0 or len(val_pairs) == 0 or len(test_pairs) == 0:
        raise ValueError('One or more sets are empty.')

    # Defense-in-depth: under v2 strict mode (assembly_id_a.is_unique +
    # row-level train_test_split + per-split negative sampling) the splits
    # are isolate-disjoint by construction. A violation here means a deep
    # regression in the split logic -- assert as a tripwire, do not handle.
    train_set = set(train_pairs['assembly_id_a']) | set(train_pairs['assembly_id_b'])
    val_set = set(val_pairs['assembly_id_a']) | set(val_pairs['assembly_id_b'])
    test_set = set(test_pairs['assembly_id_a']) | set(test_pairs['assembly_id_b'])
    assert not (train_set & val_set or train_set & test_set or val_set & test_set), \
        "v2 invariant violated: isolate overlap across train/val/test"

    duplicate_stats = {
        'cooccur_stats': cooccur_stats,
        'train_reject_stats': train_reject_stats,
        'val_reject_stats': val_reject_stats,
        'test_reject_stats': test_reject_stats,
        # pair_key_overlaps: schema preserved for the existing JSON consumer
        # (downstream `duplicate_stats.json`). Under v2 strict mode the
        # cross-split overlap assertion above guarantees these are all zero;
        # any non-zero would have raised. The *_before_removal / *_after_removal
        # fields are vestigial from the v1 cleanup model -- kept for backward
        # compat, populated as before == after (no removal happens any more).
        'pair_key_overlaps': {
            'train_val':  {'count': 0, 'pct_of_val': 0},
            'train_test': {'count': 0, 'pct_of_test': 0},
            'val_test':   {'count': 0, 'pct_of_test': 0},
            'total_overlap': 0,
            'val_pairs_before_removal': len(val_pairs),
            'test_pairs_before_removal': len(test_pairs),
            'val_pairs_after_removal': len(val_pairs),
            'test_pairs_after_removal': len(test_pairs),
            'val_removed_pct': 0,
            'test_removed_pct': 0,
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
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pair CSVs + parquets
    _t = time.time()
    print(f"save_v2: write pair CSVs start "
          f"(train={len(train_pairs):,}, val={len(val_pairs):,}, test={len(test_pairs):,})",
          flush=True)
    # NOTE: do not name the loop variable `df` -- it would shadow the outer
    # `df` parameter (the protein-level dataframe) and leave it pointing at
    # test_pairs after the loop, breaking later code that does df['assembly_id'].
    for split_name, split_df in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        _t_split = time.time()
        split_df.to_csv(output_dir / f'{split_name}_pairs.csv', index=False)
        print(f"save_v2: wrote {split_name}_pairs.csv (n={len(split_df):,}) "
              f"in {time.time()-_t_split:.2f}s", flush=True)
    print(f"save_v2: pair CSVs done in {time.time()-_t:.2f}s", flush=True)

    _t = time.time()
    print(f"save_v2: write pair parquets start", flush=True)
    for split_name, split_df in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        _t_split = time.time()
        split_df.to_parquet(output_dir / f'{split_name}_pairs.parquet',
                            compression='zstd', index=False)
        print(f"save_v2: wrote {split_name}_pairs.parquet (n={len(split_df):,}) "
              f"in {time.time()-_t_split:.2f}s", flush=True)
    print(f"save_v2: pair parquets done in {time.time()-_t:.2f}s", flush=True)

    # Cross-split overlap stats (Exp 1 of the leakage diagnostics plan).
    # Surfaces seq_hash / dna_hash overlap across splits that pair_key
    # disjointness does not prevent.
    emit_split_overlap_stats(train_pairs, val_pairs, test_pairs, output_dir)

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


def emit_split_overlap_stats(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    output_dir: Path,
    ) -> pd.DataFrame:
    """Emit `split_overlap_stats.csv`: per `(split, label, seq_type, side)`
    row, the count of unique values and how many also appear in each of the
    other splits, plus percentages.

    Pair-key disjointness across splits is already enforced by v2 invariants;
    this surfaces the *sequence-level* overlap (`seq_hash` and `dna_hash`)
    that pair-key disjointness does NOT prevent. A reader can answer "how
    many test sequences are also in train?" by reading one CSV instead of
    re-deriving it from train/val/test pair tables.

    Schema (one row per (split, label, seq_type, side)):
        split               : 'train' | 'val' | 'test'
        label               : 'pos' | 'neg'
        seq_type            : 'seq_hash' | 'dna_hash'  (identifier tracked)
        side                : 'a' | 'b'
        n_pairs             : count of pairs in this (split, label) (repeats
                              4x per (split, label) -- denormalized for
                              readability of the redundancy ratio
                              n_pairs vs n_unique).
        n_unique            : unique values in this cell.
        overlap_with_train  : count of values from this cell that also
                              appear anywhere in train (any label, same
                              seq_type, same side). Trivially equals
                              n_unique on train rows; kept for column
                              regularity.
        overlap_with_val    : same for val.
        overlap_with_test   : same for test.
        pct_overlap_train   : 100 * overlap_with_train / n_unique, rounded
                              to 1 decimal. Trivially 100.0 on train rows.
        pct_overlap_val     : same for val.
        pct_overlap_test    : same for test.

    See plan: docs/plans/2026-05-07_leakage_diagnostics_plan.md (Exp 1).
    """
    splits = {'train': train_pairs, 'val': val_pairs, 'test': test_pairs}

    # Per-(split, label) total pair counts (used to populate n_pairs).
    n_pairs_by_split_label = {
        (split_name, label_name): int(((pairs_df['label'] == label_value)).sum())
        for split_name, pairs_df in splits.items()
        for label_value, label_name in [(1, 'pos'), (0, 'neg')]
    }

    # Pre-compute per-split per-(seq_type, side) value sets, pooled across
    # labels. Keyed (split_name, seq_type, side) -> set. Used for cross-split
    # overlap.
    pooled: dict = {}
    # Per-(split, label, seq_type, side) sets used for the n_unique row and
    # for the same-split overlap (which equals n_unique).
    per_cell: dict = {}

    for split_name, pairs_df in splits.items():
        for seq_type in ('seq_hash', 'dna_hash'):
            for side in ('a', 'b'):
                col = f'{seq_type}_{side}'
                if col not in pairs_df.columns:
                    continue
                pooled[(split_name, seq_type, side)] = set(pairs_df[col].dropna())
                for label_value, label_name in [(1, 'pos'), (0, 'neg')]:
                    sub = pairs_df[pairs_df['label'] == label_value]
                    per_cell[(split_name, label_name, seq_type, side)] = set(sub[col].dropna())

    rows = []
    for (split_name, label_name, seq_type, side), values in per_cell.items():
        n_unique = len(values)
        # Compute overlap counts first; percentages are derived after.
        overlap_counts = {}
        for other in ('train', 'val', 'test'):
            if other == split_name:
                overlap_counts[other] = n_unique
            else:
                other_pool = pooled.get((other, seq_type, side), set())
                overlap_counts[other] = len(values & other_pool)

        row = {
            'seq_type': seq_type,
            'side': side,
            'label': label_name,
            'split': split_name,
            'n_pairs': n_pairs_by_split_label[(split_name, label_name)],
            'n_unique': n_unique,
            'overlap_with_train': overlap_counts['train'],
            'overlap_with_val': overlap_counts['val'],
            'overlap_with_test': overlap_counts['test'],
        }
        # Percentages last (mirrors user-requested column order).
        for other in ('train', 'val', 'test'):
            row[f'pct_overlap_{other}'] = (
                round(100.0 * overlap_counts[other] / n_unique, 1)
                if n_unique > 0 else 0.0
            )
        rows.append(row)

    # Sort so train/val/test rows for the same (seq_type, side, label) sit
    # adjacent -- cross-split comparison becomes a single column scan rather
    # than jumping across the table. Categorical on `split` enforces the
    # canonical pipeline order (train -> val -> test), not alphabetical
    # (which would give test -> train -> val).
    out = pd.DataFrame(rows)
    out['split'] = pd.Categorical(out['split'],
                                   categories=['train', 'val', 'test'],
                                   ordered=True)
    out = out.sort_values(
        ['seq_type', 'side', 'label', 'split']
    ).reset_index(drop=True)
    out['split'] = out['split'].astype(str)

    csv_path = output_dir / 'split_overlap_stats.csv'
    out.to_csv(csv_path, index=False)
    print(f"Saved split overlap stats to: {csv_path}")
    return out


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
