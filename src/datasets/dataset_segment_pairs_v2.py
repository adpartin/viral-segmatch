"""Pair-builder v2: schema-mode-only, with coverage-first negative sampling,
within-split positive deduplication, exposure tracking, and metadata-axis
annotations.

Lives behind the `dataset_segment_pairs.py` CLI, which dispatches here for every build
(v1 was retired 2026-06-03; `pair_builder_version` must be `v2`). See
`docs/plans/done/2026-05-11_design_dataset_gen_v2.md` for the full design and decisions.

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
from src.utils import schema

# Pair columns (kept identical to v1's positive/negative output) so downstream
# code that reads pair CSVs continues to work. v2 may append extra columns
# (axis flags); the core columns are unchanged. Built from the schema registry
# (src/utils/schema.py) so the per-alphabet column names live in one place:
# prot_seq/ctg_dna_seq (sequences), prot_hash/ctg_dna_hash/cds_dna_hash (hashes).
# cds_dna_hash_{a,b} are populated only when pair_key_alphabet='nt_cds' (the
# orchestrator pre-attaches cds_dna_hash to prot_df, _side renames to *_{a,b});
# otherwise pd.NA. neg_regime / metadata_match_count are populated only on
# regime-sampled negatives (pd.NA elsewhere).
_PAIR_COLUMNS = schema.build_pair_columns()

_DEFAULT_AXES = ("host", "year", "hn_subtype", "geo_location", "passage")

# Regime-aware coverage: per-regime attempt budget before falling through to
# the next regime in COVERAGE_PRIORITY_CHAIN. With 8 regimes total, the
# worst case is 8 * 10 = 80 attempts per (slot, ctg_dna_hash); the last-resort
# uniform sampler adds another `max_attempts_per_seq` (default 50). In
# practice acceptance comes long before any cap.
# Plan: docs/plans/2026-05-14_regime_aware_coverage_plan.md (decision #6).
_REGIME_AWARE_COVERAGE_BUDGET = 10


# ---------------------------------------------------------------------------
# 4.1 create_positive_pairs_v2
# ---------------------------------------------------------------------------
def create_positive_pairs_v2(
    df: pd.DataFrame,
    schema_pair: Tuple[str, str],
    pair_key_alphabet: str = 'aa',
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
            For pair_key_alphabet='aa' must contain `prot_hash`. For
            pair_key_alphabet='nt_cds' must additionally contain
            `cds_dna_hash` (caller pre-attaches via merge against cds_final).
        schema_pair: (func_left, func_right). func_left occupies slot A in every
            output pair; func_right occupies slot B. Must satisfy
            func_left != func_right.
        pair_key_alphabet: 'aa' (default; protein-level dedup via
            prot_hash) or 'nt_cds' (CDS-DNA-level dedup via cds_dna_hash —
            inflates the unique pair_key universe by +35-57 % on Flu A
            HA-NA / PB2-PB1, per docs/plans/2026-06-02_pair_key_alphabet_plan.md
            § 4). The pair_key column is constructed on the selected hash
            family; same-protein-different-CDS pairs that collapse under
            'aa' are distinct under 'nt_cds'.

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
                'ctg_dna_seq', 'canonical_segment', 'function', 'prot_hash', 'ctg_dna_hash']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"create_positive_pairs_v2: df missing required columns: {missing}")

    # Include cds_dna_hash in the per-side flow when present in df (orchestrator
    # attaches it when pair_key_alphabet='nt_cds'; absent for aa flow).
    has_cds = 'cds_dna_hash' in df.columns
    side_cols = list(req_cols) + (['cds_dna_hash'] if has_cds else [])

    def _side(side_func: str, suffix: str) -> pd.DataFrame:
        out = df.loc[df['function'] == side_func, side_cols].copy()
        rename_map = {
            'brc_fea_id': f'brc_{suffix}',
            'genbank_ctg_id': f'ctg_{suffix}',
            'prot_seq': f'prot_seq_{suffix}',
            'ctg_dna_seq': f'ctg_dna_seq_{suffix}',
            'canonical_segment': f'seg_{suffix}',
            'function': f'func_{suffix}',
            'prot_hash': f'prot_hash_{suffix}',
            'ctg_dna_hash': f'ctg_dna_hash_{suffix}',
        }
        if has_cds:
            rename_map['cds_dna_hash'] = f'cds_dna_hash_{suffix}'
        return out.rename(columns=rename_map)

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

    # Create canonical pair_key in a vectorized way: lexicographically sorted
    # hash pair joined by '__'. Hash family is alphabet-specific:
    #   pair_key_alphabet='aa'     -> prot_hash_{a,b} (protein)
    #   pair_key_alphabet='nt_cds' -> cds_dna_hash_{a,b} (CDS DNA)
    #   pair_key_alphabet='nt_ctg' -> ctg_dna_hash_{a,b} (contig DNA)
    # Matches canonical_pair_key('a','b') = '__'.join(sorted([a,b])) in _pair_helpers.py.
    if pair_key_alphabet == 'aa':
        hash_col_a, hash_col_b = 'prot_hash_a', 'prot_hash_b'
    elif pair_key_alphabet == 'nt_cds':
        hash_col_a, hash_col_b = 'cds_dna_hash_a', 'cds_dna_hash_b'
    elif pair_key_alphabet == 'nt_ctg':
        hash_col_a, hash_col_b = 'ctg_dna_hash_a', 'ctg_dna_hash_b'
    else:
        raise ValueError(
            f"create_positive_pairs_v2: pair_key_alphabet must be 'aa', 'nt_cds', "
            f"or 'nt_ctg', got {pair_key_alphabet!r}"
        )
    if hash_col_a not in pos_df.columns or hash_col_b not in pos_df.columns:
        raise ValueError(
            f"create_positive_pairs_v2: pos_df missing {hash_col_a!r}/{hash_col_b!r} "
            f"for pair_key_alphabet={pair_key_alphabet!r}. For 'nt_cds', the "
            f"caller must attach cds_dna_hash to df before calling (merge against "
            f"cds_final on (assembly_id, function))."
        )
    h_a = pos_df[hash_col_a].astype(str).values
    h_b = pos_df[hash_col_b].astype(str).values
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

    pos_df = pos_df.reindex(columns=_PAIR_COLUMNS)

    # TODO. Below we drop dups based on pair_key computed on protein seq hashes.
    # We also want to understand the duplicate situation on the DNA side, as well
    # as on the k-mer side.

    # Within-input pair_key dedup. v2 hard-codes drop_within_split_pos_duplicates=True.
    # In the global-pos flow (split_dataset_v2), this runs once on the full pos_df and
    # gives every pair_key a single representative isolate.
    n_before = len(pos_df)
    # `keep='first'` is arbitrary on host: the surviving isolate per pair_key is
    # whichever assembly_id comes first lexicographically. If we ever want to
    # bias the representative toward common hosts (e.g., Human > Pig > Chicken
    # > Duck > ... > unknown) without dropping diversity, the clean fix is:
    # pre-sort `pos_df` by a host_priority column here, then keep='first' will
    # pick the highest-priority host that carries each pair. The alternative
    # ("Human-only experiment") is to filter with `dataset.host: ['Human']`
    # in the bundle — that drops the other hosts entirely instead of just
    # changing which isolate represents each pair. The two are orthogonal:
    # the filter narrows the corpus and shrinks the regime taxonomy to the
    # 4 host-match regimes; this priority knob preserves full diversity and
    # only affects representative metadata. Not yet implemented because the
    # downstream impact (regime classification uses the representative's cell)
    # has not been measured.
    is_dup = pos_df.duplicated(subset=['pair_key'], keep='first')
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
    isolate_to_cell: Optional[dict] = None,
    sampling_axes: Optional[list] = None,
    on_shortfall: str = 'redistribute',
    regime_aware_coverage: bool = False,
    pair_key_alphabet: str = 'aa',
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
      1. Coverage phase: iterate every (slot, ctg_dna_hash) in `pos_df` (sorted)
         and try up to `max_attempts_per_seq` random partner isolates until
         at least one accepted negative covers that DNA. The DNA-level
         coverage is **best-effort**: per the apriori feasibility analysis
         (docs/results/2026-05-08_dna_coverage_feasibility_sweep.md), a few
         DNA variants in tightly-filtered bundles cannot be covered when the
         dominant protein has more DNA encodings than the partner-protein
         universe can supply distinct neg pair_keys for. Uncovered DNAs are
         logged in rejection_stats['ctg_dna_hashes_with_zero_negatives'] but do
         NOT raise. The prot_hash-level guarantee (every prot_hash gets >=1
         neg) is enforced as a hard raise at the end -- DNA-level coverage
         subsumes prot_hash-level coverage in the typical case, so this raise
         only fires if every DNA encoding a prot_hash failed.
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
    level, not prot_hash level). If `num_negatives` falls below that, the
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

    Regime-aware mode (axis_quotas + isolate_to_cell + sampling_axes
    populated): the fill phase aims for the per-regime target counts
    derived from `axis_quotas` (regime -> target fraction). on_shortfall
    controls behavior when a regime can't be filled to target. See
    `docs/plans/2026-05-09_metadata_aware_negatives_plan.md`.

    `regime_aware_coverage` (default False) — when True (and regime mode is
    active), the COVERAGE phase walks `COVERAGE_PRIORITY_CHAIN` and prefers
    partners from the hardest feasible regime for each (slot, ctg_dna_hash).
    The per-regime attempt budget is 10; if all 8 regimes are exhausted
    without acceptance, the existing uniform-random sampler runs as a last
    resort (mode #2 coverage guarantee). When False (the historical default),
    coverage stays regime-blind. See
    `docs/plans/2026-05-14_regime_aware_coverage_plan.md`.

    Returns:
        (neg_df, rejection_stats). rejection_stats includes blocked_cooccur,
        duplicate_brc, duplicate_seq, cross_split_collision, total_attempts,
        coverage_skipped, and v2 keys: requested_negatives,
        min_required_for_coverage, coverage_phase_pairs, fill_phase_pairs,
        achieved_negatives, coverage_overrode_ratio, seqs_with_zero_negatives,
        ctg_dna_hashes_with_zero_negatives, n_dna_uncovered. In regime-aware
        mode it also includes regime_manifest (per-regime
        target/available/coverage_placed/fill_placed/achieved/shortfall_reason).
    """
    _validate_schema_pair(schema_pair, "create_negative_pairs_v2")
    # Alphabet-specific pair_key construction (Phase 2 of clustering cleanup
    # plan). aa uses prot_hash_{a,b}, nt_cds uses cds_dna_hash_{a,b}. Must
    # match the alphabet of the cooccur_pairs and forbidden_pair_keys sets;
    # a mismatch silently passes contradictory pairs.
    if pair_key_alphabet == 'aa':
        _PK_HASH_A, _PK_HASH_B = 'prot_hash_a', 'prot_hash_b'
    elif pair_key_alphabet == 'nt_cds':
        _PK_HASH_A, _PK_HASH_B = 'cds_dna_hash_a', 'cds_dna_hash_b'
        if 'cds_dna_hash_a' not in pos_df.columns or 'cds_dna_hash_b' not in pos_df.columns:
            raise ValueError(
                "create_negative_pairs_v2: pair_key_alphabet='nt_cds' requires "
                "pos_df to have cds_dna_hash_a / cds_dna_hash_b columns. "
                "Caller (orchestrator / split_dataset_v2) must attach cds_dna_hash "
                "to prot_df via attach_cds_dna_hash_to_prot_df before pair construction."
            )
    elif pair_key_alphabet == 'nt_ctg':
        _PK_HASH_A, _PK_HASH_B = 'ctg_dna_hash_a', 'ctg_dna_hash_b'
        if 'ctg_dna_hash_a' not in pos_df.columns or 'ctg_dna_hash_b' not in pos_df.columns:
            raise ValueError(
                "create_negative_pairs_v2: pair_key_alphabet='nt_ctg' requires "
                "pos_df to have ctg_dna_hash_a / ctg_dna_hash_b columns "
                "(attach_dna_to_prot_df populates these at Stage 3 load)."
            )
    else:
        raise ValueError(
            f"create_negative_pairs_v2: pair_key_alphabet must be 'aa', 'nt_cds', "
            f"or 'nt_ctg', got {pair_key_alphabet!r}"
        )
    regime_mode = (axis_quotas is not None) and (len(axis_quotas) > 0)
    if regime_mode:
        if isolate_to_cell is None or sampling_axes is None:
            raise ValueError(
                "create_negative_pairs_v2: axis_quotas requires isolate_to_cell "
                "and sampling_axes; caller must build cells once and pass them in."
            )
        if on_shortfall not in {'redistribute', 'warn_only', 'error'}:
            raise ValueError(
                f"on_shortfall must be in {{'redistribute', 'warn_only', 'error'}}, "
                f"got {on_shortfall!r}"
            )
        from src.datasets._negative_regime_sampling import (
            REGIME_NAMES,
            COVERAGE_PRIORITY_CHAIN,
            build_cell_regime_partners,
            classify_pair_regime,
            compute_match_count,
            count_isolates_per_cell,
            count_available_per_regime,
            resolve_regime_targets,
        )
        missing_regimes = set(REGIME_NAMES) - set(axis_quotas.keys())
        if missing_regimes:
            raise ValueError(
                f"axis_quotas missing regimes: {sorted(missing_regimes)}. "
                f"Targets dict must include all 9 regime names."
            )
        target_sum = sum(axis_quotas.values())
        if abs(target_sum - 1.0) > 1e-6:
            raise ValueError(
                f"axis_quotas fractions must sum to 1.0; got {target_sum}."
            )
        for r, v in axis_quotas.items():
            if v < 0 or pd.isna(v):
                raise ValueError(f"axis_quotas[{r!r}] must be >= 0; got {v!r}")
    elif regime_aware_coverage:
        # Flag is only meaningful in regime mode (needs isolate_to_cell +
        # sampling_axes to build the partner map). Fail loud rather than
        # silently no-op.
        raise ValueError(
            "create_negative_pairs_v2: regime_aware_coverage=True requires "
            "regime mode (axis_quotas + isolate_to_cell + sampling_axes). "
            "Either provide axis_quotas or set regime_aware_coverage=False."
        )

    if len(pos_df) == 0:
        raise ValueError("create_negative_pairs_v2: pos_df is empty; nothing to generate negatives for.")

    rng = random.Random(seed)
    t_setup = time.time()
    print(f"[diag] create_negative_pairs_v2: setup start (|pos_df|={len(pos_df):,})", flush=True)

    # Coverage targets: prot_hashes appearing in slot A and slot B of pos_df.
    # In schema mode each prot_hash has exactly one function, so
    # target_seqs_left and target_seqs_right are disjoint.
    target_seqs_left = set(pos_df['prot_hash_a'].tolist())
    target_seqs_right = set(pos_df['prot_hash_b'].tolist())
    shared = target_seqs_left & target_seqs_right
    if shared:
        offenders = list(shared)[:5]
        raise ValueError(
            f"create_negative_pairs_v2: {len(shared)} prot_hash(es) appear in both "
            f"slot A and slot B (e.g., {offenders})."
        )

    neg_count_a: dict = {s: 0 for s in target_seqs_left}
    neg_count_b: dict = {s: 0 for s in target_seqs_right}

    # DNA-level coverage targets (option C from
    # docs/results/2026-05-08_dna_coverage_feasibility_sweep.md): per-slot
    # unique ctg_dna_hash sets and per-(slot, dna) neg counts. The coverage
    # phase iterates these instead of prot_hashes so that every DNA variant
    # of every protein gets at least one negative counterexample where
    # feasible. Uncovered DNAs are logged, not raised (best-effort).
    target_dnas_left = set(pos_df['ctg_dna_hash_a'].tolist())
    target_dnas_right = set(pos_df['ctg_dna_hash_b'].tolist())
    neg_count_dna_a: dict = {d: 0 for d in target_dnas_left}
    neg_count_dna_b: dict = {d: 0 for d in target_dnas_right}

    # Single pass over pos_df builds:
    #   isolate_to_row     - aid -> the row for that isolate (it's 1:1)
    #   prot_hash_to_row    - seq -> first row in which it appears (used as the
    #                        "self" representative in the coverage phase) --> NOTE. What if a seq appears in multiple isolates? We could pick one at random, but that would add complexity and non-determinism. Instead we just take the first, which is deterministic but arbitrary. This means some isolates get favored as coverage partners over others; we can later analyze the distribution of coverage partners per isolate to see if this is a problem.
    #   dna_to_row_a       - dna -> first row in which it appears as slot-A.
    #   dna_to_row_b       - dna -> first row in which it appears as slot-B.
    #                        Used as the "self" representative in DNA-level
    #                        coverage (analogous to prot_hash_to_row, but
    #                        per-slot to give us a row whose slot-X ctg_dna_hash
    #                        is the target DNA we want to cover).
    #   seq_to_isolates    - seq -> set of in-split isolates that contain it.
    #                        Used to exclude same-seq isolates when picking a
    #                        partner. Built from pos_df only.
    isolate_to_row: dict = {}
    prot_hash_to_row: dict = {}
    dna_to_row_a: dict = {}
    dna_to_row_b: dict = {}
    seq_to_isolates: dict = {}
    for row in pos_df.itertuples(index=False):
        aid = row.assembly_id_a
        isolate_to_row[aid] = row
        prot_hash_to_row.setdefault(row.prot_hash_a, row)
        prot_hash_to_row.setdefault(row.prot_hash_b, row)
        dna_to_row_a.setdefault(row.ctg_dna_hash_a, row)
        dna_to_row_b.setdefault(row.ctg_dna_hash_b, row)
        seq_to_isolates.setdefault(row.prot_hash_a, set()).add(aid)
        seq_to_isolates.setdefault(row.prot_hash_b, set()).add(aid)
    isolate_ids_list = sorted(isolate_to_row.keys())
    cell_to_isolates: dict = {}
    cell_partner_map: dict = {}
    if regime_mode and isolate_to_cell is not None:
        isolate_to_cell = {
            aid: cell for aid, cell in isolate_to_cell.items() if aid in isolate_to_row
        }
        # Build cell -> [isolate_ids] map once; used by both regime-aware
        # coverage (Phase 2 of the regime-aware-coverage plan) and the
        # existing regime-aware fill phase below.
        for aid, cell in isolate_to_cell.items():
            cell_to_isolates.setdefault(cell, []).append(aid)
        for cell in cell_to_isolates:
            cell_to_isolates[cell].sort()
        if regime_aware_coverage:
            # {self_cell -> {regime -> [partner_cells]}} (deterministic order).
            cell_partner_map = build_cell_regime_partners(
                isolate_to_cell, axes=sampling_axes,
            )
    print(f"[diag] create_negative_pairs_v2: setup done in "
          f"{time.time()-t_setup:.2f}s ({len(isolate_ids_list):,} isolates, "
          f"{len(seq_to_isolates):,} unique prot_hashes; "
          f"{len(target_dnas_left):,} slot-A DNAs, "
          f"{len(target_dnas_right):,} slot-B DNAs).", flush=True)

    # Coverage floor: max(|unique_slot_A_dnas|, |unique_slot_B_dnas|). Each
    # accepted negative contributes one A-DNA-coverage and one B-DNA-coverage
    # simultaneously, so the floor is governed by whichever side has more
    # unique DNAs. This is strictly higher than the previous prot_hash floor
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
    # Regime-aware coverage diagnostics (per
    # docs/plans/2026-05-14_regime_aware_coverage_plan.md Phase 3).
    # Populated only when regime_aware_coverage=True; surfaced under
    # rejection_stats['coverage_regime_aware'] at the end.
    cov_aware_attempts: dict = {r: 0 for r in REGIME_NAMES} if regime_aware_coverage else {}
    cov_aware_acceptances: dict = {r: 0 for r in REGIME_NAMES} if regime_aware_coverage else {}
    cov_aware_fell_back_to_uniform = 0
    neg_pairs: list[dict] = []
    # Regime mode bookkeeping. regime_counts is incremented by _try_accept and
    # is the source-of-truth for the manifest's coverage_placed and fill_placed.
    regime_counts: dict = {}
    if regime_mode:
        regime_counts = {r: 0 for r in REGIME_NAMES}

    def _classify_for_aids(a_aid, b_aid) -> tuple:
        """(neg_regime, metadata_match_count) for a candidate isolate pair.
        Returns (None, None) when not in regime_mode; the pair CSV columns
        end up as pd.NA via DataFrame construction."""
        if not regime_mode:
            return None, None
        cell_a = isolate_to_cell.get(a_aid)
        cell_b = isolate_to_cell.get(b_aid)
        # build_isolate_cells produces a cell for every assembly_id in df, so
        # a None lookup here means the assembly_id isn't in df -- which means
        # the sampler is operating on stale state.
        if cell_a is None or cell_b is None:
            raise RuntimeError(
                f"_classify_for_aids: missing cell for assembly_id "
                f"a={a_aid!r} (cell={cell_a!r}), b={b_aid!r} (cell={cell_b!r}). "
                f"isolate_to_cell should cover every isolate in df; this "
                f"indicates a stale cell map."
            )
        return (
            classify_pair_regime(cell_a, cell_b, axes=sampling_axes),
            compute_match_count(cell_a, cell_b),
        )

    def _build_neg_pair(a_src, b_src, regime, mc) -> dict:
        # cds_dna_hash columns may or may not be present on a_src/b_src
        # (present iff orchestrator pre-attached, i.e. pair_key_alphabet='nt_cds').
        cds_a = getattr(a_src, 'cds_dna_hash_a', pd.NA)
        cds_b = getattr(b_src, 'cds_dna_hash_b', pd.NA)
        return {
            'pair_key': canonical_pair_key(getattr(a_src, _PK_HASH_A), getattr(b_src, _PK_HASH_B)),
            'assembly_id_a': a_src.assembly_id_a,
            'assembly_id_b': b_src.assembly_id_a,
            'brc_a': a_src.brc_a, 'brc_b': b_src.brc_b,
            'ctg_a': a_src.ctg_a, 'ctg_b': b_src.ctg_b,
            'prot_seq_a': a_src.prot_seq_a, 'prot_seq_b': b_src.prot_seq_b,
            'ctg_dna_seq_a': a_src.ctg_dna_seq_a, 'ctg_dna_seq_b': b_src.ctg_dna_seq_b,
            'seg_a': a_src.seg_a, 'seg_b': b_src.seg_b,
            'func_a': a_src.func_a, 'func_b': b_src.func_b,
            'prot_hash_a': a_src.prot_hash_a, 'prot_hash_b': b_src.prot_hash_b,
            'ctg_dna_hash_a': a_src.ctg_dna_hash_a, 'ctg_dna_hash_b': b_src.ctg_dna_hash_b,
            'cds_dna_hash_a': cds_a, 'cds_dna_hash_b': cds_b,
            'label': 0,
            'neg_regime': regime if regime is not None else pd.NA,
            'metadata_match_count': mc if mc is not None else pd.NA,
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
        # Pair key on the alphabet-dispatched hash family (see top of function).
        seq_pair_key = canonical_pair_key(getattr(a_src, _PK_HASH_A), getattr(b_src, _PK_HASH_B))
        if seq_pair_key in cooccur_pairs:
            rejection_stats['blocked_cooccur'] += 1
            return False
        if seq_pair_key in seen_seq_pairs:
            rejection_stats['duplicate_seq'] += 1
            return False
        if seq_pair_key in _forbidden:
            rejection_stats['cross_split_collision'] += 1
            return False
        regime, mc = _classify_for_aids(a_src.assembly_id_a, b_src.assembly_id_a)
        neg_pairs.append(_build_neg_pair(a_src, b_src, regime, mc))
        seen_pairs.add(brc_pair_key)
        seen_seq_pairs.add(seq_pair_key)
        if a_src.prot_hash_a in neg_count_a:
            neg_count_a[a_src.prot_hash_a] += 1
        if b_src.prot_hash_b in neg_count_b:
            neg_count_b[b_src.prot_hash_b] += 1
        if a_src.ctg_dna_hash_a in neg_count_dna_a:
            neg_count_dna_a[a_src.ctg_dna_hash_a] += 1
        if b_src.ctg_dna_hash_b in neg_count_dna_b:
            neg_count_dna_b[b_src.ctg_dna_hash_b] += 1
        if regime_mode and regime is not None:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        return True

    # ---- Coverage phase --------------------------------------------------
    # Iterate every (slot, ctg_dna_hash) target. Each accepted neg covers one
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
            self_seq = s_row.prot_hash_a
        else:
            if neg_count_dna_b.get(d, 0) > 0:
                continue
            s_row = dna_to_row_b[d]
            s_in_left = False
            self_seq = s_row.prot_hash_b

        # Excluded partners: any isolate that shares the self-row's prot_hash.
        # Same-seq cross-pairs degenerate (would build a near-pos in feature
        # space). Exclusion is by prot_hash because that's the unit of
        # cooccur and pair_key identity; DNA-level membership is implicit
        # (every isolate with this DNA has this prot_hash).
        excluded = seq_to_isolates.get(self_seq, set())
        if len(excluded) >= len(isolate_ids_list):
            rejection_stats['coverage_skipped'].append((d, slot, 'no_other_isolate'))
            continue

        accepted = False

        # Regime-aware coverage: walk COVERAGE_PRIORITY_CHAIN, drawing partner
        # cells weighted by cell size; per-regime attempt budget = 10. Fall
        # through to the uniform sampler below when the chain is exhausted
        # without acceptance.
        # Plan: docs/plans/2026-05-14_regime_aware_coverage_plan.md Phase 2.
        if regime_aware_coverage:
            self_aid = s_row.assembly_id_a
            self_cell = isolate_to_cell.get(self_aid)
            if self_cell is None:
                raise RuntimeError(
                    f"regime_aware_coverage: missing cell for self isolate "
                    f"{self_aid!r}. isolate_to_cell should cover every isolate "
                    f"in pos_df."
                )
            partner_map = cell_partner_map.get(self_cell, {})
            for regime in COVERAGE_PRIORITY_CHAIN:
                partner_cells = partner_map.get(regime, [])
                if not partner_cells:
                    continue
                weights = [cell_to_isolates.get(c, []) and len(cell_to_isolates[c]) or 0
                           for c in partner_cells]
                if sum(weights) == 0:
                    continue
                cov_aware_attempts[regime] += 1
                for _attempt in range(_REGIME_AWARE_COVERAGE_BUDGET):
                    rejection_stats['total_attempts'] += 1
                    cell_p = rng.choices(partner_cells, weights=weights, k=1)[0]
                    pool = cell_to_isolates.get(cell_p, [])
                    if not pool:
                        continue
                    other_aid = rng.choice(pool)
                    if other_aid in excluded:
                        continue
                    other_row = isolate_to_row[other_aid]
                    if s_in_left:
                        a_src, b_src = s_row, other_row
                    else:
                        a_src, b_src = other_row, s_row
                    if _try_accept(a_src, b_src):
                        accepted = True
                        cov_aware_acceptances[regime] += 1
                        break
                if accepted:
                    break
            if not accepted:
                cov_aware_fell_back_to_uniform += 1

        if not accepted:
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
    coverage_regime_counts = dict(regime_counts) if regime_mode else None
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

    if regime_mode:
        # Per-regime targets and per-regime samplers.
        regime_target_count = resolve_regime_targets(axis_quotas, fill_phase_target)
        cell_counts = count_isolates_per_cell(isolate_to_cell)
        regime_available = count_available_per_regime(cell_counts, axes=sampling_axes)

        cells_sorted = sorted(cell_counts.keys(), key=lambda c: tuple(str(v) for v in c))
        cell_pairs_by_regime: dict = {r: ([], []) for r in REGIME_NAMES}
        for c1 in cells_sorted:
            n1 = cell_counts[c1]
            for c2 in cells_sorted:
                n2 = cell_counts[c2]
                w = (n1 * (n1 - 1)) if c1 == c2 else (n1 * n2)
                if w == 0:
                    continue
                r = classify_pair_regime(c1, c2, axes=sampling_axes)
                cell_pairs_by_regime[r][0].append((c1, c2))
                cell_pairs_by_regime[r][1].append(w)

        # cell_to_isolates was built once in setup (regime_mode branch) and is
        # reused here by the fill phase's _sample_pair_in_regime helper.

        def _sample_pair_in_regime(regime: str):
            cells, weights = cell_pairs_by_regime.get(regime, ([], []))
            if not cells:
                return None
            cell_pair = rng.choices(cells, weights=weights, k=1)[0]
            c1, c2 = cell_pair
            a_pool = cell_to_isolates.get(c1, [])
            b_pool = cell_to_isolates.get(c2, [])
            if not a_pool or not b_pool:
                return None
            aid_a = rng.choice(a_pool)
            if c1 == c2:
                if len(b_pool) < 2:
                    return None
                aid_b = aid_a
                while aid_b == aid_a:
                    aid_b = rng.choice(b_pool)
            else:
                aid_b = rng.choice(b_pool)
            return aid_a, aid_b

        # Greedy per-regime fill in deterministic order.
        for regime in REGIME_NAMES:
            target = regime_target_count.get(regime, 0)
            already = regime_counts.get(regime, 0)
            residual = max(target - already, 0)
            if residual == 0:
                continue
            attempts_for_regime = 0
            attempt_budget = residual * max_attempts_multiplier
            while (
                regime_counts.get(regime, 0) < target
                and len(neg_pairs) < fill_phase_target
                and attempts_for_regime < attempt_budget
            ):
                rejection_stats['total_attempts'] += 1
                attempts_for_regime += 1
                sampled = _sample_pair_in_regime(regime)
                if sampled is None:
                    break
                aid_a, aid_b = sampled
                _try_accept(isolate_to_row[aid_a], isolate_to_row[aid_b])

        # Optional redistribute pass: any remaining budget is filled regime-blind
        # but only accepting pairs whose regime is still under (overall) target.
        if on_shortfall == 'redistribute' and len(neg_pairs) < fill_phase_target:
            redistribute_attempts = 0
            redistribute_budget = (fill_phase_target - len(neg_pairs)) * max_attempts_multiplier
            while (
                len(neg_pairs) < fill_phase_target
                and redistribute_attempts < redistribute_budget
                and rejection_stats['total_attempts'] < max_total_attempts
            ):
                rejection_stats['total_attempts'] += 1
                redistribute_attempts += 1
                if len(isolate_ids_list) < 2:
                    break
                aid1, aid2 = rng.sample(isolate_ids_list, 2)
                a_aid, b_aid = aid1, aid2
                regime, _ = _classify_for_aids(a_aid, b_aid)
                if regime is None:
                    continue
                # Skip regimes that are already at or above target.
                if regime_counts.get(regime, 0) >= regime_target_count.get(regime, 0):
                    continue
                _try_accept(isolate_to_row[a_aid], isolate_to_row[b_aid])

        if on_shortfall == 'error' and len(neg_pairs) < fill_phase_target:
            shortfall_summary = {
                r: {
                    'target': regime_target_count.get(r, 0),
                    'achieved': regime_counts.get(r, 0),
                    'available': regime_available.get(r, 0),
                }
                for r in REGIME_NAMES
                if regime_counts.get(r, 0) < regime_target_count.get(r, 0)
            }
            raise RuntimeError(
                f"on_shortfall='error': could not fill all regime targets. "
                f"Achieved {len(neg_pairs):,}/{fill_phase_target:,} total. "
                f"Shortfalls: {shortfall_summary}"
            )
    else:
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

    if regime_aware_coverage:
        # Per docs/plans/2026-05-14_regime_aware_coverage_plan.md Phase 3.
        # Diagnostics for whether the priority chain was effective. Surfaced
        # in dataset_stats.json via the existing coverage_stats path.
        rejection_stats['coverage_regime_aware'] = {
            'enabled': True,
            'per_regime_budget': _REGIME_AWARE_COVERAGE_BUDGET,
            'priority_chain': list(COVERAGE_PRIORITY_CHAIN),
            'regime_aware_attempts_per_regime': {r: int(c) for r, c in cov_aware_attempts.items()},
            'regime_aware_acceptances_per_regime': {r: int(c) for r, c in cov_aware_acceptances.items()},
            'fell_back_to_uniform': int(cov_aware_fell_back_to_uniform),
            'n_target_dnas': int(n_targets),
        }
    else:
        rejection_stats['coverage_regime_aware'] = {'enabled': False}

    if regime_mode:
        regime_manifest = []
        for r in REGIME_NAMES:
            achieved = int(regime_counts.get(r, 0))
            target = int(regime_target_count.get(r, 0))
            cov_placed = int(coverage_regime_counts.get(r, 0)) if coverage_regime_counts else 0
            fill_placed = achieved - cov_placed
            available = int(regime_available.get(r, 0))
            shortfall = max(target - achieved, 0)
            if shortfall == 0:
                reason = None
            elif available <= cov_placed:
                reason = 'supply_exhausted'
            elif on_shortfall == 'redistribute':
                reason = 'supply_exhausted_after_redistribute'
            else:
                reason = f'attempt_budget_exceeded_or_{on_shortfall}'
            regime_manifest.append({
                'regime': r,
                'target': target,
                'available': available,
                'coverage_placed': cov_placed,
                'fill_placed': fill_placed,
                'achieved': achieved,
                'shortfall_reason': reason,
            })
        rejection_stats['regime_manifest'] = regime_manifest
        rejection_stats['regime_config'] = {
            'sampling_axes': list(sampling_axes),
            'on_shortfall': on_shortfall,
            'regime_targets': dict(axis_quotas),
        }
    # Cap full lists at 100 each for stats compactness; full audit lives in
    # rejection_stats['coverage_skipped'] which records (ctg_dna_hash, slot, reason).
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

    # Hard coverage check at the SEQ_HASH level: every prot_hash in pos_df
    # (slot A or slot B) must appear in at least one negative pair. The
    # DNA-level loop above will satisfy this in the typical case (covering
    # any one DNA of a prot_hash also covers that prot_hash), so this raise
    # is a safety net for the rare case where every DNA encoding a single
    # prot_hash failed.
    if seqs_with_zero_negatives:
        raise ValueError(
            f"create_negative_pairs_v2: protein-level coverage guarantee "
            f"violated -- {len(seqs_with_zero_negatives)} prot_hash(es) have "
            f"zero negative pairs (first 5: {seqs_with_zero_negatives[:5]}). "
            f"Every DNA encoding these prot_hashes failed; inspect "
            f"rejection_stats['coverage_skipped']. This is rarer than DNA-level "
            f"misses; if it fires, the bundle is likely too tight to be usable."
        )

    neg_df = pd.DataFrame(neg_pairs, columns=_PAIR_COLUMNS) if neg_pairs else pd.DataFrame(columns=_PAIR_COLUMNS)

    """
    ipdb.set_trace(context=10)
    a_counts = neg_df['prot_hash_a'].value_counts()  # Series: prot_hash → count
    b_counts = neg_df['prot_hash_b'].value_counts()

    print(a_counts.describe())   # min/25%/50%/75%/max for slot A
    print(b_counts.describe())
    print(f"slot A: {(a_counts == 0).sum()} uncovered")  # should be 0 after our hard check
    print(f"slot B: {(b_counts == 0).sum()} uncovered")

    # Tidy two-column report per prot_hash and slot:
    prev = pd.concat([
        a_counts.rename('n_as_slot_a'),
        b_counts.rename('n_as_slot_b'),
    ], axis=1).fillna(0).astype(int)
    prev['total'] = prev.sum(axis=1)
    prev.sort_values('total', ascending=False).head(20)  # most-used seqs

    # For coverage vs. positives sanity (every pos seq should be in the index):
    pos_a = set(pos_df['prot_hash_a'])
    pos_b = set(pos_df['prot_hash_b'])
    print('slot-A pos seqs missing from neg:', len(pos_a - set(a_counts.index)))
    print('slot-B pos seqs missing from neg:', len(pos_b - set(b_counts.index)))
    """
    return neg_df, rejection_stats


def _build_seq_to_func(df: pd.DataFrame) -> dict:
    """Build prot_hash -> function map. Hard error if any prot_hash maps to >1
    distinct function -- v2 slot semantics depend on this invariant."""
    funcs_per_hash = df.groupby('prot_hash')['function'].nunique()
    bad = funcs_per_hash[funcs_per_hash > 1]
    if len(bad) > 0:
        offenders = bad.head(5).index.tolist()
        raise ValueError(
            f"prot_hash maps to >1 distinct function for {len(bad)} hash(es): "
            f"{offenders} (showing up to 5). v2 assumes prot_hash -> function is "
            f"many-to-one; this indicates a Stage 1 (preprocessing) data-quality bug."
        )
    return df.groupby('prot_hash')['function'].first().to_dict()


# ---------------------------------------------------------------------------
# 4.3 compute_exposure_stats
# ---------------------------------------------------------------------------
def compute_exposure_stats(
    pos_pairs: pd.DataFrame,
    neg_pairs: pd.DataFrame,
    df: pd.DataFrame,
    ) -> pd.DataFrame:
    """Per-sequence exposure table for one split.

    Returns a DataFrame with one row per prot_hash that appears in pos_pairs OR
    neg_pairs. Columns: prot_hash, function, assembly_ids (list), n_pos_slot_a,
    n_pos_slot_b, n_pos_total, n_neg_slot_a, n_neg_slot_b, n_neg_total,
    exposure (`dual` / `pos_only` / `neg_only`).

    Slot semantics are fixed: slot A = func_left, slot B = func_right. So a
    func_left seq has n_*_slot_b == 0 by construction (literal 0, not null).
    """
    seq_to_func = _build_seq_to_func(df)

    # prot_hash -> sorted list of assembly_ids that contain it (full data, not
    # just this split -- biological duplicates outside the split are reported
    # for transparency). Sort once globally before groupby so each group's
    # list comes out sorted without a per-group lambda.
    seq_to_assemblies = (
        df[['prot_hash', 'assembly_id']]
          .drop_duplicates()
          .sort_values(['prot_hash', 'assembly_id'])
          .groupby('prot_hash')['assembly_id']
          .agg(list)
          .to_dict()
    )

    # Per-slot, per-label counts via value_counts (vectorized; no Python loop).
    # Each Series is indexed by prot_hash; concat aligns on the union of keys.
    def _slot_counts(pairs_df: pd.DataFrame, slot: str, label: str) -> pd.Series:
        name = f'n_{label}_slot_{slot}'
        if len(pairs_df) == 0:
            return pd.Series(dtype=int, name=name)
        return pairs_df[f'prot_hash_{slot}'].value_counts().rename(name)

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
        'prot_hash', 'function', 'assembly_ids',
        'n_pos_slot_a', 'n_pos_slot_b', 'n_pos_total',
        'n_neg_slot_a', 'n_neg_slot_b', 'n_neg_total',
        'exposure',
    ]
    if len(counts) == 0:
        return pd.DataFrame(columns=out_cols)

    counts.index.name = 'prot_hash'
    counts['n_pos_total'] = counts['n_pos_slot_a'] + counts['n_pos_slot_b']
    counts['n_neg_total'] = counts['n_neg_slot_a'] + counts['n_neg_slot_b']

    has_pos = counts['n_pos_total'] > 0
    has_neg = counts['n_neg_total'] > 0
    exposure = pd.Series('neg_only', index=counts.index, dtype='object')
    exposure[has_pos & has_neg] = 'dual'
    exposure[has_pos & ~has_neg] = 'pos_only'
    counts['exposure'] = exposure

    # Defensive: a prot_hash present in pairs but absent from df gets None /
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

    Lookup keys on `assembly_id`, not `prot_hash`. prot_hash collapses identical
    amino-acid sequences across isolates with legitimately different metadata
    (cross-host transmission, conserved sequences across years, etc.), so
    keying on it can return a different isolate's value than the pair's actual
    source isolate. assembly_id is the unique-per-isolate key that matches
    what the pair actually carries; metadata enrichment guarantees one value
    per assembly per axis. (The disagreement that motivated this fix was
    quantified by `eda/probe_metadata_lookup.py`, deleted 2026-05-13;
    recoverable via git history.)

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
        n_unique_seqs_null counts prot_hashes for which NO isolate carries
        any non-null value -- it's the embedding-keyed-lookup floor (an
        embedding referenced by such a seq has no metadata anywhere). For
        prot_hashes that span multiple isolates with different metadata
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

        # n_unique_seqs_null = prot_hashes with NO non-null value across any
        # isolate. Drop nulls first so seqs whose first row happens to be null
        # but which have a valid value elsewhere are not undercounted.
        n_unique_seqs = int(df['prot_hash'].nunique())
        seqs_with_value = df.dropna(subset=[col])['prot_hash'].nunique()
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
    sampling_axes: Optional[list] = None,
    year_match: str = 'binned',
    year_bin_edges: Optional[list] = None,
    on_shortfall: str = 'redistribute',
    regime_aware_coverage: bool = False,
    train_isolates_override: Optional[list] = None,
    val_isolates_override: Optional[list] = None,
    test_isolates_override: Optional[list] = None,
    cooccur_pairs: Optional[set] = None,
    cooccur_stats: Optional[dict] = None,
    split_strategy_mode: str = 'random',
    split_strategy_hash_key: str = 'seq',
    cluster_id_path: Optional[str] = None,
    cluster_id_threshold: Optional[float] = None,
    cluster_alphabet: str = 'aa',
    cds_final_path: Optional[str] = None,
    single_slot: Optional[str] = None,
    routed_pos_override: Optional[dict] = None,
    pair_key_alphabet: str = 'aa',
    drop_budget: Optional[dict] = None,
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

    isolate_to_cell: Optional[dict] = None
    resolved_sampling_axes: Optional[list] = None
    if axis_quotas is not None and len(axis_quotas) > 0:
        from src.datasets._negative_regime_sampling import (
            DEFAULT_AXES as _NEG_DEFAULT_AXES,
            DEFAULT_YEAR_BIN_EDGES as _NEG_DEFAULT_YEAR_BIN_EDGES,
            build_isolate_cells,
        )
        resolved_sampling_axes = list(sampling_axes) if sampling_axes else list(_NEG_DEFAULT_AXES)
        meta_per_iso = (
            df[['assembly_id'] + resolved_sampling_axes]
            .drop_duplicates('assembly_id')
            .copy()
        )
        edges_for_build = year_bin_edges if year_bin_edges is not None else _NEG_DEFAULT_YEAR_BIN_EDGES
        isolate_to_cell = build_isolate_cells(
            meta_per_iso,
            axes=resolved_sampling_axes,
            year_match=year_match,
            year_bin_edges=edges_for_build,
        )
        print(f'\nsplit_dataset_v2: regime-aware mode enabled. axes={resolved_sampling_axes}, '
              f'year_match={year_match}, on_shortfall={on_shortfall}, '
              f'cells={len(set(isolate_to_cell.values())):,}.')

    # Build co-occurrence (or accept pre-computed for CV reuse).
    # cooccur_pairs: set of canonical pair-key strings, each of the form
    # "<prot_hash>__<prot_hash>" with the two hashes sorted lexicographically.
    # One entry per sequence pair that appears together in at least one isolate.
    # Used to block candidate negatives that would actually be positives somewhere.
    # Cooccur set must use the same alphabet as pair_key (mismatched alphabets
    # would block / allow the wrong pairs as negatives). For pair_key_alphabet=
    # 'nt_cds' the df must have cds_dna_hash attached (orchestrator's
    # responsibility — see attach_cds_dna_hash_to_prot_df).
    _cooccur_hash_col = schema.hash_col(pair_key_alphabet)  # aa->prot_hash, nt_cds->cds_dna_hash, nt_ctg->ctg_dna_hash
    if cooccur_pairs is None:
        print(f"\nsplit_dataset_v2: Build co-occurrence set on {_cooccur_hash_col} "
              f"(pair_key_alphabet={pair_key_alphabet!r})...")
        cooccur_pairs, cooccur_stats = build_cooccurrence_set(df, hash_col=_cooccur_hash_col)
        print(f"   Total co-occurring pairs: {cooccur_stats['total_cooccur_pairs']:,}")
        print(f"   Pairs appearing in multiple isolates: {cooccur_stats['pairs_in_multiple_isolates']:,}")
        print(f"   Max isolates for a single pair: {cooccur_stats['max_isolates_per_pair']}")
    elif cooccur_stats is None:
        raise ValueError("cooccur_stats must be provided together with cooccur_pairs")
    elif cooccur_stats.get('hash_col') != _cooccur_hash_col:
        raise ValueError(
            f"split_dataset_v2: pre-computed cooccur_stats hash_col="
            f"{cooccur_stats.get('hash_col')!r} but pair_key_alphabet="
            f"{pair_key_alphabet!r} requires {_cooccur_hash_col!r}."
        )

    # Build the global positive pair dataframe once, deduped to one row per unique
    # pair_key with a deterministic representative assembly_id. The split is then
    # a partition of this table -- train/val/test are pair_key-disjoint by
    # construction, so no post-hoc cross-split overlap removal is needed.
    print("\nsplit_dataset_v2: Build global positive pairs (one call across full df)...", flush=True)
    pos_df, pos_dedup_stats = create_positive_pairs_v2(
        df, schema_pair=schema_pair, pair_key_alphabet=pair_key_alphabet,
    )
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
    # Audit dicts from non-random routing modes (None for the other paths). Threaded
    # through duplicate_stats and emitted as {seq,cluster}_disjoint_audit.json by the
    # saver. See docs/plans/2026-05-10_seq_disjoint_routing_plan.md and
    # docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md.
    seq_disjoint_audit: Optional[dict] = None
    cluster_disjoint_audit: Optional[dict] = None
    if routed_pos_override is not None:
        # k-fold path: pre-routed positives provided by the generator
        # (`generate_all_cluster_disjoint_cv_folds_v2`). Skip the routing
        # dispatch entirely; use the override's train_pos / val_pos /
        # test_pos / cluster_disjoint_audit / pos_dedup_stats. pos_df was
        # built above (wasted work for the override case; acceptable cost
        # ~few seconds per fold on Flu A — avoids restructuring the build
        # block). See Phase 3 of
        # docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md.
        if split_strategy_mode != 'cluster_disjoint':
            raise ValueError(
                f"split_dataset_v2: routed_pos_override requires "
                f"split_strategy_mode='cluster_disjoint'; got {split_strategy_mode!r}."
            )
        train_pos = routed_pos_override['train_pos']
        val_pos = routed_pos_override['val_pos']
        test_pos = routed_pos_override['test_pos']
        cluster_disjoint_audit = routed_pos_override['cluster_disjoint_audit']
        pos_dedup_stats = routed_pos_override['pos_dedup_stats']
        print(f"split_dataset_v2: using routed_pos_override (k-fold fold_id="
              f"{routed_pos_override.get('fold_id', '?')}); "
              f"train={len(train_pos):,} val={len(val_pos):,} test={len(test_pos):,}",
              flush=True)
    elif train_isolates_override is not None:
        if split_strategy_mode != 'random':
            raise NotImplementedError(
                f"split_dataset_v2: split_strategy_mode={split_strategy_mode!r} is not "
                f"compatible with the CV-override path (train_isolates_override is set). "
                f"Use mode='random' for CV; seq_disjoint / cluster_disjoint are "
                f"single-split-only for now."
            )
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
    elif split_strategy_mode == 'seq_disjoint':
        # Bipartite-component routing: each connected component (on the
        # hash family selected by split_strategy_hash_key — prot_hash by
        # default for protein-level partitioning, or ctg_dna_hash for the
        # looser nucleotide-level partition) is indivisible; whole
        # components are LPT-greedy bin-packed into train/val/test. Drops
        # zero pairs by construction; cross-split hash overlap on the
        # chosen family is impossible by construction (verified in audit).
        # See docs/plans/2026-05-10_seq_disjoint_routing_plan.md.
        from src.datasets._pair_helpers import seq_disjoint_route_pos_df
        print(f"\nsplit_dataset_v2: seq_disjoint routing on pos_df rows "
              f"(bipartite CC + LPT-greedy bin-pack, hash_key={split_strategy_hash_key!r})...",
              flush=True)
        train_pos, val_pos, test_pos, seq_disjoint_audit = seq_disjoint_route_pos_df(
            pos_df, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
            hash_key=split_strategy_hash_key,
        )
        ach = seq_disjoint_audit['achieved_pct']
        print(f"split_dataset_v2: seq_disjoint routing completed "
              f"(train={len(train_pos):,} pairs [{ach['train']:.2f}%], "
              f"val={len(val_pos):,} [{ach['val']:.2f}%], "
              f"test={len(test_pos):,} [{ach['test']:.2f}%], "
              f"n_components={seq_disjoint_audit['cc_summary']['n_components']:,}, "
              f"largest_component={seq_disjoint_audit['cc_summary']['largest_component_pairs']} pairs)",
              flush=True)
    elif split_strategy_mode == 'cluster_disjoint':
        # mmseqs2-cluster-disjoint routing: each connected component on the
        # bipartite (cluster_id_a, cluster_id_b) graph is indivisible. Routes
        # pairs into the split where BOTH sequences' clusters land; pairs whose
        # prot_hashes aren't covered by the cluster lookup are dropped.
        # See docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md.
        # cluster_alphabet='aa' (default) clusters protein sequences;
        # 'nt_cds' clusters CDS DNA (joins on cds_dna_hash); 'nt_ctg' clusters
        # contig DNA (joins on ctg_dna_hash). The pos-side hash is attached below.
        from src.datasets._split_helpers import (
            cluster_disjoint_route_pos_df, load_cluster_lookup,
        )
        if cluster_id_path is None:
            raise ValueError(
                "split_dataset_v2: cluster_id_path is required when "
                "split_strategy_mode='cluster_disjoint'."
            )
        if cluster_alphabet not in {'aa', 'nt_cds', 'nt_ctg'}:
            raise ValueError(
                f"split_dataset_v2: cluster_alphabet must be 'aa', 'nt_cds', or "
                f"'nt_ctg', got {cluster_alphabet!r}"
            )
        if cluster_alphabet == 'nt_cds':
            if cds_final_path is None:
                raise ValueError(
                    "split_dataset_v2: cluster_alphabet='nt_cds' requires "
                    "cds_final_path (the cds_dna_final.parquet path)."
                )
            # cds_dna_hash_{a,b} are already populated if the orchestrator
            # pre-attached cds_dna_hash to df (pair_key_alphabet='nt_cds' path).
            # build_pair_columns always EMITS these columns, but they are empty
            # (all-NaN) when pair_key != nt_cds -- so gate on POPULATED, not
            # present: otherwise nt_cds clustering + a non-nt_cds pair_key (e.g.
            # the historical protein-pair_key + nt_cds-cluster config) leaves them
            # empty and the cluster join crashes on a float64-vs-string merge.
            # Drop the empty placeholders and attach.
            _populated = (
                'cds_dna_hash_a' in pos_df.columns and 'cds_dna_hash_b' in pos_df.columns
                and not pos_df['cds_dna_hash_a'].isna().all()
                and not pos_df['cds_dna_hash_b'].isna().all()
            )
            if not _populated:
                pos_df = pos_df.drop(columns=[c for c in ('cds_dna_hash_a', 'cds_dna_hash_b')
                                              if c in pos_df.columns])
                from src.datasets._pair_helpers import attach_cds_dna_hash_to_pos_df
                pos_df = attach_cds_dna_hash_to_pos_df(
                    pos_df, cds_final_path, schema_pair=schema_pair,
                )
        # pos-side cluster-join hash from the registry: aa->prot_hash,
        # nt_cds->cds_dna_hash, nt_ctg->ctg_dna_hash. nt_ctg's ctg_dna_hash_{a,b}
        # are already populated on pos_df (only nt_cds needs the attach above).
        pos_hash_col = schema.hash_col(cluster_alphabet)
        if single_slot is not None and single_slot not in ('a', 'b'):
            raise ValueError(
                f"split_dataset_v2: single_slot must be None, 'a', or 'b'; got {single_slot!r}"
            )
        routing_label = ('bipartite CC + LPT-greedy bin-pack on cluster_ids'
                         if single_slot is None
                         else f'single-slot ({single_slot}) + LPT-greedy bin-pack on cluster_ids')
        print(f"\nsplit_dataset_v2: cluster_disjoint routing "
              f"({routing_label}; "
              f"cluster_alphabet={cluster_alphabet}, "
              f"cluster_id_path={cluster_id_path}, "
              f"cluster_id_threshold={cluster_id_threshold})...", flush=True)
        cluster_lookup = load_cluster_lookup(cluster_id_path)
        train_pos, val_pos, test_pos, cluster_disjoint_audit = cluster_disjoint_route_pos_df(
            pos_df, cluster_lookup,
            train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
            cluster_id_threshold=cluster_id_threshold,
            cluster_lookup_path=str(cluster_id_path),
            pos_hash_col=pos_hash_col,
            cluster_alphabet=cluster_alphabet,
            single_slot=single_slot,
            drop_budget=drop_budget,
        )
        ach = cluster_disjoint_audit['achieved_pct']
        att = cluster_disjoint_audit['attach_audit']
        print(f"split_dataset_v2: cluster_disjoint routing completed "
              f"(train={len(train_pos):,} pairs [{ach['train']:.2f}%], "
              f"val={len(val_pos):,} [{ach['val']:.2f}%], "
              f"test={len(test_pos):,} [{ach['test']:.2f}%], "
              f"n_components={cluster_disjoint_audit['cc_summary']['n_components']:,}, "
              f"largest_component={cluster_disjoint_audit['cc_summary']['largest_component_pairs']} pairs, "
              f"pairs_dropped_in_cluster_join={att['n_input']-att['n_kept']})", flush=True)
    elif split_strategy_mode == 'random':
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

        print(f"split_dataset_v2: shuffle-split completed "
              f"(train={len(train_pos):,} pairs, "
              f"val={len(val_pos):,}, test={len(test_pos):,})", flush=True)
    else:
        raise ValueError(
            f"split_dataset_v2: unknown split_strategy_mode={split_strategy_mode!r}; "
            f"expected 'random', 'seq_disjoint', or 'cluster_disjoint'."
        )

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
        isolate_to_cell=isolate_to_cell,
        sampling_axes=resolved_sampling_axes,
        on_shortfall=on_shortfall,
        regime_aware_coverage=regime_aware_coverage,
        pair_key_alphabet=pair_key_alphabet,
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
        isolate_to_cell=isolate_to_cell,
        sampling_axes=resolved_sampling_axes,
        on_shortfall=on_shortfall,
        regime_aware_coverage=regime_aware_coverage,
        pair_key_alphabet=pair_key_alphabet,
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
        isolate_to_cell=isolate_to_cell,
        sampling_axes=resolved_sampling_axes,
        on_shortfall=on_shortfall,
        regime_aware_coverage=regime_aware_coverage,
        pair_key_alphabet=pair_key_alphabet,
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
    # jj = test_exp[['prot_hash', 'function', 'n_pos_total', 'n_neg_total', 'exposure']]
    # jj['n_pos_total'] - jj['n_neg_total']

    # Cross-split pair_key overlap check.
    # Under v2's strict regime (global pos dedup + cooccur_pairs blocking pos
    # vs neg + forbidden_pair_keys threading across split-level neg calls),
    # train/val/test pair_keys are disjoint by construction. Any overlap here
    # is a serious leakage condition: identical (prot_seq_a, prot_seq_b) negatives in
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
            f"This means identical (prot_seq_a, prot_seq_b) negatives appear in two "
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
    if isolate_to_cell is not None:
        duplicate_stats['regime_manifest'] = {
            'config': train_reject_stats.get('regime_config'),
            'splits': {
                'train': train_reject_stats.get('regime_manifest'),
                'val':   val_reject_stats.get('regime_manifest'),
                'test':  test_reject_stats.get('regime_manifest'),
            },
        }
    if seq_disjoint_audit is not None:
        # Re-audit the FINAL DataFrames (positives + negatives + axis flags +
        # any post-hoc cleanup), not just the routed positives. Under v2's
        # strict invariants this should still be all-zero for the chosen
        # hash family -- the negatives sampler picks partners from per-split
        # pos_df rows, so it cannot introduce a hash from another split's
        # pool. Verifying here makes the audit JSON a regression check on
        # the whole pipeline, not just the routing step.
        #
        # Both prot_hash and ctg_dna_hash counters are emitted regardless of the
        # chosen hash_key: the one matching hash_key is the by-construction
        # guarantee (hard-failed by the saver); the other is a diagnostic.
        # Under hash_key='seq', ctg_dna_hash is also 0 (protein equality
        # implies DNA grouping). Under hash_key='dna', prot_hash is usually
        # non-zero (synonymous variants land in different splits).
        # 'seq'/'dna' track the hash_key knob (the saver indexes this audit key by
        # it); the column each reads comes from the schema registry.
        for family, _alphabet in (('seq', 'aa'), ('dna', 'nt_ctg')):
            _hcol = schema.hash_col(_alphabet)
            for side in ('a', 'b'):
                col = f'{_hcol}_{side}'
                if col not in train_pairs.columns:
                    continue
                t = set(train_pairs[col].dropna())
                v = set(val_pairs[col].dropna())
                te = set(test_pairs[col].dropna())
                seq_disjoint_audit[f'{family}_hash_overlap_full_pairs_' + side] = {
                    'train_val':  len(t & v),
                    'train_test': len(t & te),
                    'val_test':   len(v & te),
                }
        duplicate_stats['seq_disjoint_audit'] = seq_disjoint_audit
    if cluster_disjoint_audit is not None:
        # Re-audit the FINAL DataFrames at the cluster-id and sequence-hash level.
        #
        # By construction cluster_id_{a,b} overlap across splits should be zero
        # on whichever slot(s) the routing constrains (negatives are sampled
        # from per-split positives, so they can't pull cluster_ids from outside
        # their split's positive pool — same invariant as seq_disjoint).
        # prot_hash / ctg_dna_hash overlap MAY be non-zero at thresholds < 1.0 on
        # the constrained side (different sequences in the same cluster can
        # land in different splits' positives → their hashes can co-occur in
        # negatives via partner sampling).
        #
        # For constrained-slot routing (single_slot='a' or 'b'), the
        # UNCONSTRAINED side is not bound by either invariant: prot_hash leakage
        # on the unconstrained side is the empirical residual leakage the
        # routing leaves on the table (see splits.md § 1.8).
        #
        # Fields are symmetric across slots a / b and across cluster_id /
        # prot_hash / ctg_dna_hash. The routing-mode-specific interpretation lives
        # in the prose docs, not in the field names.
        # 'seq'/'dna' are the two hash families (protein / nucleotide), kept
        # stable in the audit JSON; the column each reads comes from the registry.
        _leak_col = {'cluster_id': 'cluster_id',
                     'seq_hash': schema.hash_col('aa'),      # prot_hash
                     'dna_hash': schema.hash_col('nt_ctg')}  # ctg_dna_hash
        for family in ('cluster_id', 'seq_hash', 'dna_hash'):
            _base = _leak_col[family]
            for side in ('a', 'b'):
                col = f'{_base}_{side}'
                if col not in train_pairs.columns:
                    continue
                t  = set(train_pairs[col].dropna())
                v  = set(val_pairs[col].dropna())
                te = set(test_pairs[col].dropna())
                union = t | v | te
                n_unique = len(union)
                if n_unique == 0:
                    continue
                # Count unique values appearing in 2+ splits.
                in_multi = sum(
                    1 for x in union
                    if int(x in t) + int(x in v) + int(x in te) >= 2
                )
                cluster_disjoint_audit[f'{family}_leakage_{side}'] = {
                    'n_unique':              n_unique,
                    'n_in_multiple_splits':  in_multi,
                    'pct_leakage':           round(100.0 * in_multi / n_unique, 2),
                    'split_pair_overlap': {
                        'train_val':  len(t & v),
                        'train_test': len(t & te),
                        'val_test':   len(v & te),
                    },
                }
        duplicate_stats['cluster_disjoint_audit'] = cluster_disjoint_audit
        # One-line summary: cross-split leakage on each slot, for each hash family.
        # In constrained-slot routing, the unconstrained side's prot_hash leakage
        # is the residual lookup signal (see splits.md § 1.8).
        for side in ('a', 'b'):
            parts = []
            for family in ('cluster_id', 'seq_hash', 'dna_hash'):
                key = f'{family}_leakage_{side}'
                if key in cluster_disjoint_audit:
                    blk = cluster_disjoint_audit[key]
                    parts.append(
                        f"{family} {blk['n_in_multiple_splits']:,}/{blk['n_unique']:,} "
                        f"({blk['pct_leakage']:.1f}%)"
                    )
            if parts:
                print(f"  slot_{side} cross-split leakage: " + "  |  ".join(parts))
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
        # Regime-aware coverage diagnostics (always present; 'enabled' tells
        # consumers whether the priority chain was active).
        # See docs/plans/2026-05-14_regime_aware_coverage_plan.md Phase 3.
        'coverage_regime_aware': reject_stats.get('coverage_regime_aware', {'enabled': False}),
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
    sampling_axes: Optional[list] = None,
    year_match: str = 'binned',
    year_bin_edges: Optional[list] = None,
    on_shortfall: str = 'redistribute',
    regime_aware_coverage: bool = False,
    pair_key_alphabet: str = 'aa',
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

    _cooccur_hash_col = schema.hash_col(pair_key_alphabet)  # aa->prot_hash, nt_cds->cds_dna_hash, nt_ctg->ctg_dna_hash
    print(f"\nBuilding co-occurrence set on {_cooccur_hash_col} "
          f"(pair_key_alphabet={pair_key_alphabet!r})...")
    cooccur_pairs, cooccur_stats = build_cooccurrence_set(df, hash_col=_cooccur_hash_col)
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
            sampling_axes=sampling_axes,
            year_match=year_match,
            year_bin_edges=year_bin_edges,
            on_shortfall=on_shortfall,
        regime_aware_coverage=regime_aware_coverage,
            train_isolates_override=train_ids,
            val_isolates_override=val_ids,
            test_isolates_override=test_ids,
            cooccur_pairs=cooccur_pairs,
            cooccur_stats=cooccur_stats,
            pair_key_alphabet=pair_key_alphabet,
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
# 4.8 generate_all_cluster_disjoint_cv_folds_v2 (Phase 3 of the k-fold plan)
# ---------------------------------------------------------------------------
def generate_all_cluster_disjoint_cv_folds_v2(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
    neg_to_pos_ratio: float,
    val_ratio: float,
    schema_pair: Tuple[str, str],
    single_slot: str,
    cluster_id_path: str,
    cluster_id_threshold: Optional[float],
    cluster_alphabet: str = 'aa',
    cds_final_path: Optional[str] = None,
    max_acceptable_drift_pp: float = 0.05,
    min_test_frac: float = 0.05,
    max_attempts_per_seq: int = 50,
    max_attempts_multiplier: int = 100,
    axes_for_flags: list = _DEFAULT_AXES,
    axis_quotas: Optional[dict] = None,
    sampling_axes: Optional[list] = None,
    year_match: str = 'binned',
    year_bin_edges: Optional[list] = None,
    on_shortfall: str = 'redistribute',
    regime_aware_coverage: bool = False,
    pair_key_alphabet: str = 'aa',
    ) -> Iterator[dict]:
    """Generate k-fold splits for single-slot cluster_disjoint routing.

    Differs from generate_all_cv_folds_v2 (random-mode CV):
    1. Routes via cluster_disjoint_route_pos_df(n_folds=k) on the global
       pos_df, NOT sklearn KFold on isolates. Atom ordering is pinned per
       D1 of the k-fold plan; GroupKFold partition is bit-reproducible.
    2. Per-fold negative sampling uses an independent forbidden_pair_keys
       set inside each split_dataset_v2 call (within-fold threading
       across the 3 splits; reset between folds for fold independence
       per D3 of the plan).
    3. Collect-all D3 evaluation: all folds routed first, then either
       yield all folds (if all pass) or raise the D4 menu listing every
       failing fold and its drift_pp / test_frac values.

    Yields fold_data dicts matching generate_all_cv_folds_v2's shape:
        {'fold_id', 'train_pairs', 'val_pairs', 'test_pairs',
         'duplicate_stats', 'exposure_tables'}.

    See `docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md` Phase 3.
    """
    from src.datasets._split_helpers import (
        cluster_disjoint_route_pos_df, load_cluster_lookup,
    )
    from src.datasets._pair_helpers import attach_cds_dna_hash_to_pos_df

    # ----- Build co-occurrence set once -----
    _cooccur_hash_col = schema.hash_col(pair_key_alphabet)  # aa->prot_hash, nt_cds->cds_dna_hash, nt_ctg->ctg_dna_hash
    print(f"\nBuilding co-occurrence set on {_cooccur_hash_col} "
          f"(pair_key_alphabet={pair_key_alphabet!r})...")
    cooccur_pairs, cooccur_stats = build_cooccurrence_set(df, hash_col=_cooccur_hash_col)
    print(f"   Total co-occurring sequence pairs: {cooccur_stats['total_cooccur_pairs']:,}")

    # ----- Build global pos_df once -----
    print("\nBuilding global positive pairs (one call for all folds)...")
    pos_df, pos_dedup_stats = create_positive_pairs_v2(
        df, schema_pair=schema_pair, pair_key_alphabet=pair_key_alphabet,
    )
    if len(pos_df) == 0:
        raise ValueError(
            f"No positive pairs generated from df. Verify schema_pair={schema_pair}."
        )
    if not pos_df['assembly_id_a'].is_unique:
        raise ValueError(
            "v2 strict mode: assembly_id_a not unique in pos_df. "
            "Schema-ordered v2 requires exactly one pair per isolate."
        )
    print(f"   Global pos_df: {len(pos_df):,} unique pair_keys after dedup.")

    # ----- Attach nt_cds CDS-DNA hash if nt_cds-alphabet routing -----
    pos_hash_col = 'prot_hash'
    if cluster_alphabet == 'nt_cds':
        if cds_final_path is None:
            raise ValueError(
                "cluster_alphabet='nt_cds' requires cds_final_path "
                "(cds_final.parquet path from Stage 1.5)."
            )
        # Gate on POPULATED, not present (see split_dataset_v2 above):
        # build_pair_columns emits empty cds_dna_hash_{a,b} when pair_key != nt_cds,
        # so attaching only on absence would leave them empty and crash the cluster
        # join on a float64-vs-string merge. Drop the empty placeholders and attach.
        _populated = (
            'cds_dna_hash_a' in pos_df.columns and 'cds_dna_hash_b' in pos_df.columns
            and not pos_df['cds_dna_hash_a'].isna().all()
            and not pos_df['cds_dna_hash_b'].isna().all()
        )
        if not _populated:
            pos_df = pos_df.drop(columns=[c for c in ('cds_dna_hash_a', 'cds_dna_hash_b')
                                          if c in pos_df.columns])
            pos_df = attach_cds_dna_hash_to_pos_df(
                pos_df, Path(cds_final_path), schema_pair=schema_pair,
            )
        pos_hash_col = 'cds_dna_hash'

    # ----- Load cluster lookup -----
    print(f"\nLoading cluster lookup from {cluster_id_path}...")
    cluster_lookup = load_cluster_lookup(cluster_id_path)

    # ----- Route into k folds (single call to cluster_disjoint_route_pos_df) -----
    print(f"\nRouting via cluster_disjoint_route_pos_df(n_folds={n_folds}, "
          f"single_slot={single_slot!r}, max_acceptable_drift_pp="
          f"{max_acceptable_drift_pp}, min_test_frac={min_test_frac})...")
    folds = cluster_disjoint_route_pos_df(
        pos_df, cluster_lookup,
        train_ratio=0.8, val_ratio=val_ratio, seed=seed,
        cluster_id_threshold=cluster_id_threshold,
        cluster_lookup_path=str(cluster_id_path),
        pos_hash_col=pos_hash_col,
        cluster_alphabet=cluster_alphabet,
        single_slot=single_slot,
        n_folds=n_folds,
        max_acceptable_drift_pp=max_acceptable_drift_pp,
        min_test_frac=min_test_frac,
    )

    # ----- D4 collect-all check: raise with menu if any fold fails D3 -----
    failing = []
    for fold_id, (_, _, _, audit) in enumerate(folds):
        fc = audit['feasibility_check']
        if not fc['all_pass']:
            failing.append(fold_id)
    if failing:
        msg = [
            "cluster_disjoint k-fold D3 collect-all failure (no folds written).",
            f"  Configuration: schema_pair={schema_pair}, "
            f"alphabet={cluster_alphabet}, single_slot={single_slot!r}, "
            f"threshold={cluster_id_threshold}, n_folds={n_folds}.",
            f"  Failing folds: {failing}. Per-fold D3 status:",
        ]
        for fold_id, (_, _, _, audit) in enumerate(folds):
            fc = audit['feasibility_check']
            status = 'PASS' if fc['all_pass'] else 'FAIL'
            drift = fc['max_acceptable_drift_pp']['achieved']
            tf = fc['min_test_frac']['achieved']
            reasons = []
            if not fc['max_acceptable_drift_pp']['pass']:
                reasons.append(
                    f"drift_pp={drift:.4f} > {fc['max_acceptable_drift_pp']['threshold']}"
                )
            if not fc['min_test_frac']['pass']:
                reasons.append(
                    f"test_frac={tf:.4f} < {fc['min_test_frac']['threshold']}"
                )
            tag = '' if not reasons else '  <-- ' + ', '.join(reasons)
            msg.append(
                f"    fold {fold_id}: {status}  "
                f"drift_pp={drift:.4f}, test_frac={tf:.4f}{tag}"
            )
        msg += [
            "  Options (require explicit config change):",
            f"    - reduce dataset.n_folds (try <= {n_folds - 1})",
            "    - relax to a looser dataset.split_strategy.cluster_id_threshold",
            "    - increase dataset.split_strategy.feasibility.max_acceptable_drift_pp",
            "    - decrease dataset.split_strategy.feasibility.min_test_frac",
            "      (last two loosen feasibility; produce noisier per-fold metrics)",
            "  See D4 of docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md.",
        ]
        raise NotImplementedError('\n'.join(msg))

    print(f"\nAll {n_folds} folds passed D3 feasibility. "
          f"Proceeding with per-fold negative generation...")

    # ----- Build cross-fold kfold_summary (Phase 4 of the k-fold plan) -----
    # Constant across all folds since it summarizes the whole partition.
    # Injected into each fold's duplicate_stats before yielding; the saver
    # surfaces it in that fold's dataset_stats.json (mirrors the existing
    # slot_leakage_summary pattern). composition_mode is reserved-null per
    # OoS #7 forward-compat.
    _first_audit = folds[0][3]
    kfold_summary = {
        'k':                              int(n_folds),
        'max_feasible_k_strict':          int(_first_audit['max_feasible_k_strict']),
        'max_feasible_k_at_build_drift':  int(_first_audit['max_feasible_k_at_build_drift']),
        'build_drift_pp':                 float(max_acceptable_drift_pp),
        'all_folds_pass':                 True,  # collect-all check above guarantees this
        'single_slot':                    single_slot,
        'cluster_alphabet':               cluster_alphabet,
        'atom_ordering_key':              '(-size, cluster_id)',
        'composition_mode':               None,
        'per_fold': [
            {
                'fold_id':   int(audit['fold_id']),
                'drift_pp':  float(audit['feasibility_check']['max_acceptable_drift_pp']['achieved']),
                'test_frac': float(audit['feasibility_check']['min_test_frac']['achieved']),
                'pass':      bool(audit['feasibility_check']['all_pass']),
            }
            for _, _, _, audit in folds
        ],
    }

    # ----- Per-fold: generate negatives via split_dataset_v2 with override -----
    for fold_id, (train_pos, val_pos, test_pos, fold_audit) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"CV (cluster_disjoint k={n_folds}): fold {fold_id + 1}/{n_folds}  "
              f"(pos: train={len(train_pos):,} val={len(val_pos):,} test={len(test_pos):,})")
        print(f"{'='*60}")

        # Per-fold seed for negative sampling. Within each split_dataset_v2
        # call, forbidden_pair_keys is threaded across train/val/test (mode #1
        # intra-model leakage prevention). Across folds, the set is reset
        # because each split_dataset_v2 call has its own `forbidden_so_far`.
        fold_seed = seed + fold_id

        train_pairs, val_pairs, test_pairs, dup_stats, exposure_tables = split_dataset_v2(
            df=df,
            schema_pair=schema_pair,
            neg_to_pos_ratio=neg_to_pos_ratio,
            train_ratio=0.8,
            val_ratio=val_ratio,
            seed=fold_seed,
            max_attempts_per_seq=max_attempts_per_seq,
            max_attempts_multiplier=max_attempts_multiplier,
            axes_for_flags=axes_for_flags,
            axis_quotas=axis_quotas,
            sampling_axes=sampling_axes,
            year_match=year_match,
            year_bin_edges=year_bin_edges,
            on_shortfall=on_shortfall,
            regime_aware_coverage=regime_aware_coverage,
            cooccur_pairs=cooccur_pairs,
            cooccur_stats=cooccur_stats,
            split_strategy_mode='cluster_disjoint',
            cluster_id_path=cluster_id_path,
            cluster_id_threshold=cluster_id_threshold,
            cluster_alphabet=cluster_alphabet,
            cds_final_path=cds_final_path,
            single_slot=single_slot,
            routed_pos_override={
                'fold_id':                fold_id,
                'train_pos':              train_pos,
                'val_pos':                val_pos,
                'test_pos':               test_pos,
                'cluster_disjoint_audit': fold_audit,
                'pos_dedup_stats':        pos_dedup_stats,
            },
            pair_key_alphabet=pair_key_alphabet,
        )
        # Inject the cross-fold kfold_summary into this fold's duplicate_stats
        # so the saver can surface it in this fold's dataset_stats.json.
        dup_stats['kfold_summary'] = kfold_summary
        yield {
            'fold_id': fold_id,
            'train_pairs': train_pairs,
            'val_pairs': val_pairs,
            'test_pairs': test_pairs,
            'duplicate_stats': dup_stats,
            'exposure_tables': exposure_tables,
        }


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
    holdout_cfg: Optional[dict] = None,
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
    # Surfaces prot_hash / ctg_dna_hash overlap across splits that pair_key
    # disjointness does not prevent.
    emit_split_overlap_stats(train_pairs, val_pairs, test_pairs, output_dir)

    # cluster_disjoint mode emits a routing audit. Hard-fail on cluster_id
    # cross-split overlap (zero by construction). prot_hash / ctg_dna_hash overlaps
    # at the full-pairs level are diagnostic only — at thresholds < 1.0,
    # different proteins in the same cluster can land in different splits'
    # negatives via partner sampling.
    # Under single-slot routing, only the constrained side is by-construction
    # zero; the other side may legitimately have non-zero overlap. The check
    # then runs only on the constrained side.
    cd_audit = duplicate_stats.get('cluster_disjoint_audit')
    if cd_audit is not None:
        with open(output_dir / 'cluster_disjoint_audit.json', 'w') as f:
            json.dump(cd_audit, f, indent=2, default=str)
        print(f"Saved cluster_disjoint_audit.json to: {output_dir}")
        single_slot = cd_audit.get('single_slot')
        sides_to_check = (single_slot,) if single_slot is not None else ('a', 'b')
        for side in sides_to_check:
            for pair_label, audit_key in (
                ('positives-only', 'cluster_id_overlap'),
                ('full-pairs',     f'cluster_id_overlap_full_pairs_{side}'),
            ):
                if pair_label == 'positives-only':
                    overlaps = cd_audit.get(audit_key, {}).get(side, {})
                else:
                    overlaps = cd_audit.get(audit_key, {})
                bad = {k: v for k, v in overlaps.items() if v != 0}
                if bad:
                    raise RuntimeError(
                        f"cluster_disjoint routing audit FAILED ({pair_label}, "
                        f"side={side}, single_slot={single_slot}): non-zero "
                        f"cross-split cluster_id overlap {bad}. This means a "
                        f"cluster spans splits despite "
                        f"split_strategy.mode=cluster_disjoint. Check the routing helper "
                        f"(src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df) "
                        f"and the negatives sampler (must source partners only from "
                        f"per-split pos_df rows)."
                    )

    # seq_disjoint mode emits a routing audit. Hard-fail if the routing
    # claimed any cross-split DNA overlap (it should be zero by construction;
    # any non-zero is a bug in the routing or a regression in the negatives
    # sampler that pulled a DNA from another split's pool).
    sd_audit = duplicate_stats.get('seq_disjoint_audit')
    if sd_audit is not None:
        with open(output_dir / 'seq_disjoint_audit.json', 'w') as f:
            json.dump(sd_audit, f, indent=2, default=str)
        print(f"Saved seq_disjoint_audit.json to: {output_dir}")
        # Hard-fail only on the hash family that the routing was built on
        # (split_strategy.hash_key — 'seq' by default for protein-level
        # disjointness; 'dna' for nucleotide-level). The OTHER family is
        # diagnostic only: its non-zero counters are expected under
        # hash_key='dna' (synonymous variants of the same protein in
        # different splits) and should be zero under hash_key='seq'.
        active_key = sd_audit.get('hash_key', 'seq')  # 'seq' default; back-compat for older audits
        positives_scope_key = f'{active_key}_hash_overlap'
        for side in ('a', 'b'):
            for pair_label, scope_key in (
                ('positives-only', positives_scope_key),
                ('full-pairs', f'{active_key}_hash_overlap_full_pairs_' + side),
            ):
                if scope_key == positives_scope_key:
                    overlaps = sd_audit[positives_scope_key].get(side, {})
                else:
                    overlaps = sd_audit.get(scope_key, {})
                bad = {k: v for k, v in overlaps.items() if v != 0}
                if bad:
                    raise RuntimeError(
                        f"seq_disjoint routing audit FAILED ({pair_label}, "
                        f"hash_key={active_key!r}, side={side}): non-zero "
                        f"cross-split {active_key}_hash overlap {bad}. "
                        f"This means cross-split sequence-level leakage is present despite "
                        f"split_strategy.mode=seq_disjoint. Check the routing helper "
                        f"(src/datasets/_pair_helpers.py::seq_disjoint_route_pos_df) "
                        f"and the negatives sampler (must source partners only from "
                        f"per-split pos_df rows)."
                    )

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
        # metadata_holdout summary; present only when the holdout dispatch
        # was used. See docs/plans/2026-05-11_metadata_holdout_plan.md.
        **(
            {'metadata_holdout': duplicate_stats['metadata_holdout']}
            if 'metadata_holdout' in duplicate_stats else {}
        ),
        # Per-slot cross-split leakage headline. Present only under
        # cluster_disjoint routing. Symmetric across slots a/b and across
        # hash families (cluster_id / prot_hash / ctg_dna_hash). Full per-pair
        # overlap counts live in cluster_disjoint_audit.json. See
        # docs/methods/splits.md § 1.8.
        **(
            {'slot_leakage_summary': {
                f'slot_{side}': {
                    family: {
                        'pct_leakage':          duplicate_stats['cluster_disjoint_audit'][f'{family}_leakage_{side}']['pct_leakage'],
                        'n_in_multiple_splits': duplicate_stats['cluster_disjoint_audit'][f'{family}_leakage_{side}']['n_in_multiple_splits'],
                        'n_unique':             duplicate_stats['cluster_disjoint_audit'][f'{family}_leakage_{side}']['n_unique'],
                    }
                    for family in ('cluster_id', 'seq_hash', 'dna_hash')
                    if f'{family}_leakage_{side}' in duplicate_stats['cluster_disjoint_audit']
                }
                for side in ('a', 'b')
                if any(
                    f'{family}_leakage_{side}' in duplicate_stats['cluster_disjoint_audit']
                    for family in ('cluster_id', 'seq_hash', 'dna_hash')
                )
            }}
            if 'cluster_disjoint_audit' in duplicate_stats else {}
        ),
        # Cross-fold kfold_summary headline (Phase 4 of the k-fold plan).
        # Present only under cluster_disjoint k-fold mode (n_folds >= 2);
        # injected by generate_all_cluster_disjoint_cv_folds_v2 into
        # duplicate_stats. Mirrors slot_leakage_summary's conditional-include
        # pattern. Schema documented in the plan's Phase 4 section.
        **(
            {'kfold_summary': duplicate_stats['kfold_summary']}
            if 'kfold_summary' in duplicate_stats else {}
        ),
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

    # Inject pair_share + isolate_share into split_sizes so downstream
    # consumers (plots, banner) don't have to recompute denominators.
    from src.datasets._pair_helpers import (
        compute_split_shares,
        format_split_summary_banner,
    )
    compute_split_shares(dataset_stats['split_sizes'])

    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(dataset_stats, f, indent=2, default=_jsonable)
    print(f"Saved dataset stats to: {output_dir / 'dataset_stats.json'}")

    # End-of-stage summary banner. holdout_cfg already comes in as a plain
    # dict (or None) from the caller — the OmegaConf → dict conversion lives
    # at the dispatch site in dataset_segment_pairs.py.
    print('\nFinal split sizes: '
          + format_split_summary_banner(dataset_stats['split_sizes'], holdout_cfg))

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

    if 'regime_manifest' in duplicate_stats and duplicate_stats['regime_manifest'] is not None:
        manifest = duplicate_stats['regime_manifest']
        with open(output_dir / 'negative_regime_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2, default=_jsonable)
        rows = []
        for split_name in ('train', 'val', 'test'):
            split_rows = manifest.get('splits', {}).get(split_name) or []
            for row in split_rows:
                rows.append({'split': split_name, **row})
        if rows:
            pd.DataFrame(rows).to_csv(
                output_dir / 'negative_regime_manifest.csv', index=False
            )
        print(f"Saved regime manifest: {output_dir / 'negative_regime_manifest.json'}")

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
    this surfaces the *sequence-level* overlap (`prot_hash` and `ctg_dna_hash`)
    that pair-key disjointness does NOT prevent. A reader can answer "how
    many test sequences are also in train?" by reading one CSV instead of
    re-deriving it from train/val/test pair tables.

    Schema (one row per (split, label, seq_type, side)):
        split               : 'train' | 'val' | 'test'
        label               : 'pos' | 'neg'
        seq_type            : 'seq_hash' | 'dna_hash'  (hash family; reads the
                              molecule-level prot_hash / ctg_dna_hash columns)
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
        # seq_type labels the hash family ('seq_hash'/'dna_hash', stable in the
        # CSV); the column it reads is the molecule-level hash from the registry.
        for seq_type, _alphabet in (('seq_hash', 'aa'), ('dna_hash', 'nt_ctg')):
            _hcol = schema.hash_col(_alphabet)
            for side in ('a', 'b'):
                col = f'{_hcol}_{side}'
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

    # cluster_disjoint_cc is the 2D-CD builder's mode (src/datasets/dataset_pairs_cc.py),
    # not a v2 mode. Reject loudly so a CC bundle isn't silently misrouted through v2.
    sd_mode = OmegaConf.select(config, "dataset.split_strategy.mode")
    if sd_mode == 'cluster_disjoint_cc':
        raise ValueError(
            "dataset.split_strategy.mode='cluster_disjoint_cc' is the 2D-CD builder's mode "
            "(src/datasets/dataset_pairs_cc.py), not a v2 mode. Run it with "
            "`python src/datasets/dataset_pairs_cc.py --config_bundle <bundle>`, "
            "not the v2 Stage-3 CLI."
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

    # negative_sampling: optional regime-aware mode. Validate the dict shape
    # if it's set; the actual sampler is in src/datasets/_negative_regime_sampling.py.
    neg_sampling = OmegaConf.select(config, "dataset.negative_sampling")
    if neg_sampling is not None:
        from src.datasets._negative_regime_sampling import REGIME_NAMES, DEFAULT_AXES
        regime_targets = OmegaConf.select(config, "dataset.negative_sampling.regime_targets")
        if regime_targets is None:
            raise ValueError(
                "dataset.negative_sampling is set but regime_targets is null. "
                "Provide a complete dict over the 8 regime names "
                f"({sorted(REGIME_NAMES)}) summing to 1.0."
            )
        rt_keys = set(regime_targets.keys()) if hasattr(regime_targets, 'keys') else set(dict(regime_targets).keys())
        # Legacy key removal: unknown_metadata_neg was retired 2026-05-11
        # alongside the dataset.drop_ambiguous_subtype default. Reject loudly
        # so old bundles surface the migration message rather than silently
        # mis-allocating budget to a nonexistent regime.
        if 'unknown_metadata_neg' in rt_keys:
            raise ValueError(
                "dataset.negative_sampling.regime_targets contains "
                "'unknown_metadata_neg' which is no longer supported "
                "(retired 2026-05-11). Remove the key; its budget "
                "redistributes to the 8 remaining regimes via on_shortfall."
            )
        missing = set(REGIME_NAMES) - rt_keys
        if missing:
            raise ValueError(
                f"dataset.negative_sampling.regime_targets missing regimes: {sorted(missing)}."
            )
        rt_values = list(dict(regime_targets).values())
        s = sum(float(v) for v in rt_values)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(
                f"dataset.negative_sampling.regime_targets must sum to 1.0; got {s}."
            )
        if any((v is None) or (float(v) < 0) for v in rt_values):
            raise ValueError(
                f"dataset.negative_sampling.regime_targets values must be >= 0; "
                f"got {dict(regime_targets)}."
            )
        ym = OmegaConf.select(config, "dataset.negative_sampling.year_match")
        if ym is not None and ym not in {'binned', 'exact'}:
            raise ValueError(f"year_match must be 'binned' or 'exact'; got {ym!r}.")
        os = OmegaConf.select(config, "dataset.negative_sampling.on_shortfall")
        if os is not None and os not in {'redistribute', 'warn_only', 'error'}:
            raise ValueError(
                f"on_shortfall must be in {{'redistribute', 'warn_only', 'error'}}; "
                f"got {os!r}."
            )
        axes = OmegaConf.select(config, "dataset.negative_sampling.axes")
        if axes is not None:
            extras = set(axes) - set(DEFAULT_AXES)
            if extras:
                raise ValueError(
                    f"dataset.negative_sampling.axes must be a subset of "
                    f"{sorted(DEFAULT_AXES)}; got extras {sorted(extras)}."
                )
        # Regime-aware coverage flag — must be bool. Default off.
        rac = OmegaConf.select(config, "dataset.negative_sampling.regime_aware_coverage")
        if rac is not None and not isinstance(rac, bool):
            raise ValueError(
                f"dataset.negative_sampling.regime_aware_coverage must be a bool "
                f"(or absent — defaults to False); got {rac!r} of type {type(rac).__name__}. "
                f"See docs/plans/2026-05-14_regime_aware_coverage_plan.md."
            )

    # legacy axis_quotas key (was a NotImplementedError placeholder); keep
    # rejection so old bundles error explicitly and migrate to negative_sampling.
    axis_quotas = OmegaConf.select(config, "dataset.axis_quotas")
    if axis_quotas is not None and len(axis_quotas) > 0:
        raise ValueError(
            f"dataset.axis_quotas is no longer the entry point for regime-aware "
            f"sampling; use dataset.negative_sampling.regime_targets instead. "
            f"See docs/plans/2026-05-09_metadata_aware_negatives_plan.md."
        )

    # Removed: dataset.year_train / dataset.year_test. The legacy temporal-
    # holdout mechanism was retired 2026-05-11 in favor of the general
    # metadata_holdout path (year-axis holdout is its degenerate case). If a
    # bundle still references these keys, Hydra raises at config-load time
    # because they no longer exist in conf/dataset/default.yaml. See
    # docs/plans/2026-05-11_metadata_holdout_plan.md.

    # split_strategy.mode dispatch.
    split_mode = OmegaConf.select(config, "dataset.split_strategy.mode")
    if split_mode is not None and split_mode not in {'random', 'seq_disjoint', 'cluster_disjoint'}:
        raise ValueError(
            f"v2 requires dataset.split_strategy.mode in "
            f"{{'random', 'seq_disjoint', 'cluster_disjoint'}} (or absent); "
            f"got {split_mode!r}. See "
            f"docs/plans/2026-05-10_seq_disjoint_routing_plan.md and "
            f"docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md."
        )
    # split_strategy.hash_key: 'seq' (protein, default — stricter) or 'dna'
    # (nucleotide — appropriate when training on DNA k-mer features).
    # Only meaningful for mode='seq_disjoint'; cluster_disjoint always operates
    # on protein-level clusters from mmseqs2.
    split_hash_key = OmegaConf.select(config, "dataset.split_strategy.hash_key")
    if split_hash_key is not None and split_hash_key not in {'seq', 'dna'}:
        raise ValueError(
            f"v2 requires dataset.split_strategy.hash_key in {{'seq', 'dna'}} "
            f"(or absent — defaults to 'seq'); got {split_hash_key!r}."
        )
    # cluster_disjoint mode requires a cluster_id_path pointing at the mmseqs2 lookup parquet.
    if split_mode == 'cluster_disjoint':
        cluster_id_path = OmegaConf.select(config, "dataset.split_strategy.cluster_id_path")
        if cluster_id_path is None:
            raise ValueError(
                "dataset.split_strategy.mode='cluster_disjoint' requires "
                "dataset.split_strategy.cluster_id_path to point at a (<hash>, cluster_id) "
                "parquet (typically data/processed/flu/<version>/clusters_aa/t<NN>/combined_cluster.parquet "
                "or clusters_nt_cds/t<NN>/combined_cluster.parquet; <hash> is alphabet-specific "
                "per Phase 2). See docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md."
            )
        cluster_id_threshold = OmegaConf.select(config, "dataset.split_strategy.cluster_id_threshold")
        if cluster_id_threshold is not None:
            try:
                float(cluster_id_threshold)
            except (TypeError, ValueError):
                raise ValueError(
                    f"dataset.split_strategy.cluster_id_threshold must be a float "
                    f"(or absent); got {cluster_id_threshold!r}."
                )
        cluster_alphabet = OmegaConf.select(config, "dataset.split_strategy.cluster_alphabet")
        if cluster_alphabet is not None and cluster_alphabet not in {'aa', 'nt_cds', 'nt_ctg'}:
            raise ValueError(
                f"dataset.split_strategy.cluster_alphabet must be 'aa', 'nt_cds', or "
                f"'nt_ctg' (or absent — defaults to 'aa'); got {cluster_alphabet!r}."
            )
        # pair_key_alphabet config validation (Phase 2 of clustering cleanup plan)
        pair_key_alphabet = OmegaConf.select(config, "dataset.split_strategy.pair_key_alphabet")
        if pair_key_alphabet is not None and pair_key_alphabet not in {'aa', 'nt_cds', 'nt_ctg'}:
            raise ValueError(
                f"dataset.split_strategy.pair_key_alphabet must be 'aa', 'nt_cds', or 'nt_ctg' "
                f"(or absent — defaults to cluster_alphabet for cluster_disjoint mode, "
                f"else 'aa'); got {pair_key_alphabet!r}."
            )
    n_folds = OmegaConf.select(config, "dataset.n_folds")
    if split_mode == 'seq_disjoint' and n_folds is not None and int(n_folds) > 1:
        raise NotImplementedError(
            f"dataset.split_strategy.mode='seq_disjoint' is not yet compatible with "
            f"dataset.n_folds={n_folds} (CV mode). seq_disjoint is single-split-only "
            f"for now; see docs/plans/2026-05-10_seq_disjoint_routing_plan.md "
            f"'Out of scope' section."
        )
    if split_mode == 'cluster_disjoint' and n_folds is not None and int(n_folds) > 1:
        # k-fold for cluster_disjoint is supported ONLY when single_slot is set.
        # Bilateral cluster_disjoint k-fold is out of scope (OoS #5 of the
        # k-fold variance plan).
        ss = OmegaConf.select(config, "dataset.split_strategy.single_slot")
        if ss is None:
            raise NotImplementedError(
                f"dataset.split_strategy.mode='cluster_disjoint' with "
                f"dataset.n_folds={n_folds} (CV mode) requires "
                f"split_strategy.single_slot in {{'a','b'}}. Bilateral "
                f"cluster_disjoint k-fold is not currently supported. See "
                f"OoS #5 of docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md."
            )

    # metadata_holdout: structural validation only (deeper validation happens
    # inside compute_metadata_holdout_isolates after df is loaded). Reject
    # combinations with CV mode and with split_strategy.mode != 'random'.
    holdout = OmegaConf.select(config, "dataset.metadata_holdout")
    if holdout is not None:
        if 'train' not in holdout or holdout.get('train') is None:
            raise ValueError(
                "dataset.metadata_holdout.train is required (filter dict). See "
                "docs/plans/2026-05-11_metadata_holdout_plan.md."
            )
        if 'test' not in holdout or holdout.get('test') is None:
            raise ValueError(
                "dataset.metadata_holdout.test is required (filter dict). See "
                "docs/plans/2026-05-11_metadata_holdout_plan.md."
            )
        if n_folds is not None and int(n_folds) > 1:
            raise NotImplementedError(
                f"dataset.metadata_holdout is not yet compatible with "
                f"dataset.n_folds={n_folds} (CV mode). metadata_holdout is "
                f"single-split-only for now."
            )
        if split_mode is not None and split_mode != 'random':
            raise NotImplementedError(
                f"dataset.metadata_holdout is mutually exclusive with "
                f"dataset.split_strategy.mode={split_mode!r}. Set "
                f"split_strategy.mode to 'random' (the default) or omit it. "
                f"(seq_disjoint / cluster_disjoint produce their own train/val/test "
                f"routing and cannot be combined with the metadata-holdout dispatch.)"
            )
