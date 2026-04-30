# `dataset_segment_pairs_v2.py` — Design Spec

## 0. Reading order

1. Section 1 (Context) — what v1 does and why v2 exists.
2. Section 2 (Goals / Non-goals) — scope boundary.
3. Section 9 (Decisions log) — choices already made; do not re-litigate.
4. Section 4 (Function-by-function spec) — implementation surface.
5. Sections 5–8 — config, outputs, tests, edge cases.

Before writing code: read v1 (`dataset_segment_pairs.py`) and the sibling utils (`src/utils/path_utils.py`, `src/utils/metadata_enrichment.py`, `src/utils/config_hydra.py`, `src/utils/seed_utils.py`) to confirm the interfaces v2 imports from. The spec assumes these are unchanged.


---


## 1. Context

`dataset_segment_pairs.py` (v1) builds train/val/test pair datasets for the segment-matching task. Each pair is two protein sequences, labeled 1 if they co-occur in the same isolate (`assembly_id`) and 0 otherwise. Positives are generated combinatorially within each isolate (cross-function pairs); negatives are sampled randomly across isolates with a co-occurrence block to prevent contradictory labels.

v1 has the following gaps:
- No guarantee that every sequence in positives also appears in at least one negative. Some sequences end up only in positives, which means evaluation can't distinguish "model learned the pairing" from "model memorized embeddings of frequently-seen sequences".
- No per-sequence exposure tracking or summary stats.
- Random negative sampling with rejection is inefficient on dense datasets (high `cooccur_pairs` block rate).
- Within-split positive duplicates (same `pair_key` from two different isolates that both land in the same split) are not deduplicated — the cross-split overlap check catches between-split overlap but not within-split.
- No metadata-axis annotations on negatives, so distinguishing between hard-negative vs easy-negative requires pulling metadata. This can be done during dataset creation or post-hoc after training.

v2 addresses these gaps in a parallel implementation (new file, new functions) so v1 stays callable and runs are gated by a config flag.


---


## 2. Goals and non-goals

### Goals

1. **Within-split positive deduplication.** After generating positives per split, drop duplicate `pair_key` rows (keep first), gated by a config flag (default on).
2. **Coverage-first negative sampling.** Replace v1's pure-random negative sampling with a multi-pass round-robin scheme that guarantees every positive-pair sequence appears in at least one negative, then continues round-robin until `num_negatives` is reached.
3. **Per-sequence exposure tracking.** Produce a per-sequence table (`sequence_exposure.csv`) with positive/negative counts per slot and an exposure label (`dual` / `pos_only` / `neg_only`).
4. **Hard/easy axis annotations.** Attach metadata-axis columns (`<axis>_a`, `<axis>_b`, `same_<axis>`) to every pair, for axes `host`, `year`, `hn_subtype`, `geo_location`, `passage`. Axis-quota sampling itself is NOT implemented in this iteration — leave a placeholder parameter that is currently a no-op.
5. **Metadata coverage reporting.** Produce `metadata_coverage.json` documenting null counts per axis at sequence and unique-sequence level. Fail loudly if any axis configured for axis-quota sampling has nulls (currently irrelevant since axis quotas are not implemented, but the validation hook should exist).
6. **Single-split + CV support.** Both modes work end-to-end with v2.
7. **Temporal placeholder.** A function stub that raises `NotImplementedError`. CLI dispatch errors clearly if the user combines `pair_builder_version=v2` with temporal config.
8. **Backward compatibility.** v1 path remains unchanged. v2 is opt-in via `dataset.pair_builder_version: v2`.

### Non-goals (explicitly out of scope)

- Computing model performance metrics (AUC, F1, etc.). v2 produces the *data* that enables sliced evaluation; the downstream training script computes metrics. v2 attaches `exposure` and `same_<axis>` columns to pairs, and writes per-seq exposure tables — that is the boundary.
- Axis-quota sampling logic. The parameter exists in the function signature; it must be `None` or empty for now; passing a populated dict raises `NotImplementedError` with a clear message.
- Temporal split support beyond a stub.
- Modifying any upstream preprocessing (`preprocess_flu.py`, metadata enrichment, etc.).
- Modifying v1 functions. v2 should import shared helpers (`canonical_pair_key`, `build_cooccurrence_set`, `attach_dna_to_prot_df`, `select_balanced_isolate_pool`, `filter_by_metadata`, `get_metadata_distributions`, `compute_isolate_pair_counts`) from v1 if needed; these are pure helpers and v2 must not duplicate them.

### Hard-coded constraints (v2 simplifications)

To keep the v2 code base readable, several v1-configurable behaviors are **hard-coded** in v2. The corresponding parameters are dropped from v2 function signatures, and v2 config validation rejects any setting that contradicts them. Each constraint must be documented in a code comment at its enforcement site, with an explicit reference to v1 (`dataset_segment_pairs.py`) where the parameter still exists for users who need other modes.

The constraints:

1. **`pair_mode = "schema_ordered"`.** v2 supports schema-ordered pairs only. The unordered branch (random orientation, hash-based canonicalization) is removed. v1 still supports unordered mode via `dataset.pair_mode: unordered`.
2. **`schema_pair` required.** Must be a non-null `(func_left, func_right)` tuple with `func_left != func_right`. v2 functions that previously accepted `schema_pair: Optional[Tuple[str, str]] = None` now require it as a positional/required argument.
3. **`allow_same_func_negatives = False` and `max_same_func_ratio = 0.5` (vestigial).** Same-function negatives are impossible in schema mode by construction (since `func_left != func_right`), so these parameters and all related counting/quota logic are removed. `same_func_count` is no longer tracked or returned. v1 retains both parameters for unordered mode.
4. **`canonicalize_pair_orientation_enabled = False`.** Schema mode defines slot orientation by function; hash-based canonicalization is incompatible and was already auto-disabled in v1 schema mode. v2 removes the parameter and the `canonicalize_pair_orientation` helper call. `pair_key` (used for dedup) is still hash-canonicalized — that is independent of slot orientation. v1 retains the parameter for unordered mode.
5. **`hard_partition_isolates = True`.** Train/val/test isolates are always disjoint in v2. The unassigned-isolate fallback logic and overlap-detection check are kept (still useful), but the parameter is gone. v1 retains the parameter.
6. **`drop_within_split_pos_duplicates = True`.** Within-split positive duplicates are always dropped (keep first). v2 has no flag for this; v1's parameter (added in v1 only as a discussion artifact, not actually present) does not exist either — the dedup behavior itself is new in v2. A comment should be added where `drop_within_split_pos_duplicates` is implemented.

When v2 code enforces one of these, the comment template is:

```python
# v2 hard-codes <constraint>. See dataset_segment_pairs.py (v1) for the
# configurable version. v2 config validation rejects any setting that
# contradicts this.
```

This way a future reader of v2 immediately knows where the configurable version lives and that the simplification is intentional, not an oversight. Top docstring for v2 should document the primary updates as compared to v1.


---


## 3. File layout

- New file: `dataset_segment_pairs_v2.py` (sibling of v1, same directory).
- v2 imports shared helpers from v1 by name. If v1's name is not importable as-is (e.g., script-level code runs at import time), refactor only the minimum necessary in v1 to make those helpers importable: wrap the script-level code in `if __name__ == "__main__":` or move helpers to a new `_pair_helpers.py` and have both v1 and v2 import from it. Prefer the latter if v1 has script-level code that can't run at import time.
- The CLI entry point (the script that argparses `--config_bundle`) stays in v1 and dispatches to v1 or v2 builders based on `dataset.pair_builder_version`.


---


## 4. Function-by-function spec

### 4.1 `create_positive_pairs_v2`

**Signature:**

```python
def create_positive_pairs_v2(
    df: pd.DataFrame,
    schema_pair: Tuple[str, str],   # required: (func_left, func_right)
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
```

Note: `pair_mode`, `canonicalize_pair_orientation_enabled`, and `drop_within_split_pos_duplicates` from v1 are **not parameters in v2** — see Section 2 "Hard-coded constraints". v2 always runs in schema-ordered mode with within-split pos dedup on. The dropped-parameter comment template applies at the top of this function.

**Behavior:**

1. Validate `schema_pair` (non-null, length 2, `func_left != func_right`).
2. Run combinatorial schema-ordered positive generation: for each isolate, take the cross-product of `func_left` rows × `func_right` rows, oriented as left → right (slot A = func_left, slot B = func_right). This is v1's schema-ordered branch with the unordered branch removed.
3. After generation, deduplicate on `pair_key` keeping `first`. Always — there is no flag. Track:
   - `n_pos_before_dedup`
   - `n_pos_after_dedup`
   - `n_pos_duplicates_dropped`
   - `duplicate_isolate_pairs`: list of (assembly_id_a, assembly_id_b, pair_key) for each dropped row, capped at first 100 for log readability.
4. Return `(pos_df, dedup_stats)` where `dedup_stats` is the dict above.

**Edge cases:**
- `df` empty → return empty DataFrame with the same columns as v1's positive output, plus an empty `dedup_stats`.
- No qualifying isolates (no isolate has both `func_left` and `func_right`) → empty DataFrame, no error.


### 4.2 `create_negative_pairs_v2`

**Signature:**

```python
def create_negative_pairs_v2(
    df: pd.DataFrame,
    pos_pairs: pd.DataFrame,
    num_negatives: int,
    isolate_ids: list[str],
    cooccur_pairs: set,
    schema_pair: Tuple[str, str],   # required: (func_left, func_right)
    seed: int = 42,
    max_attempts_per_seq: int = 50,
    max_attempts_multiplier: int = 100,
    axis_quotas: Optional[dict] = None,  # placeholder; must be None
) -> tuple[pd.DataFrame, dict]:
```

Note on dropped parameters (see Section 2 "Hard-coded constraints"):
- `allow_same_func_negatives`, `max_same_func_ratio` — same-function negatives are impossible in schema mode (`func_left != func_right`), so these are not parameters and same-function counting/quota logic is removed. Return tuple no longer includes `same_func_count`.
- `pair_mode` — schema mode only.
- `canonicalize_pair_orientation_enabled` — already disabled in v1 schema mode; not a parameter in v2.
- `schema_pair` is required, not optional.

The dropped-parameter comment template applies at the top of this function.

**Behavior:**

The sampler runs in **two phases**: a coverage phase that guarantees every positive-pair sequence gets at least one negative, then a fill phase that tops up to `num_negatives` using v1's random-sampling logic.

The coverage requirement: **every sequence appearing in `pos_pairs` must appear in at least one negative pair (modulo sequences for which no valid partner can be found).** A stricter balance guarantee (max imbalance ≤ 1 across all sequences via multi-pass round-robin) is explicitly out of scope for this iteration — leave it as a future enhancement noted in a code comment, but do not implement it now.

1. **Initialization:**
   - If `axis_quotas` is not `None` and not empty: raise `NotImplementedError("axis_quotas not yet supported; pass None")`.
   - Validate `schema_pair` (non-null, length 2, `func_left != func_right`).
   - Build `isolate_groups` and `isolate_func_groups` exactly as v1's schema-mode branch does. The unordered-mode setup (single `isolate_groups` only) is removed.
   - Build two target sets:
     - `target_seqs_left`: `seq_hash` values appearing in slot A of `pos_pairs` (these have `function == func_left`).
     - `target_seqs_right`: `seq_hash` values appearing in slot B of `pos_pairs` (these have `function == func_right`).
   - Track `neg_count_a[seq_hash]` for slot-A coverage and `neg_count_b[seq_hash]` for slot-B coverage. (In schema mode the slot is determined by function, so these dicts have disjoint keys: a `func_left` seq only appears in `neg_count_a`, a `func_right` seq only in `neg_count_b`.)
   - Build `seq_to_isolates: dict[seq_hash → set[assembly_id]]` from `df`. A sequence may appear in multiple isolates (biological duplicates). Coverage is keyed by `seq_hash`, not by `(seq_hash, assembly_id)`.
   - Initialize `seen_pairs`, `seen_seq_pairs`, `rejection_stats`.

2. **Coverage phase:**
   - Iterate `target_seqs_left ∪ target_seqs_right` in sorted order (for determinism). For each sequence `s`:
     - Determine its slot from its function: `func_left` → fill slot A coverage (`neg_count_a[s] == 0`); `func_right` → fill slot B (`neg_count_b[s] == 0`). Skip if already covered.
     - Try up to `max_attempts_per_seq` to find a valid partner:
       - Pick an isolate that does NOT contain `s` (use `seq_to_isolates[s]` to exclude). Sample uniformly from the remaining `isolate_ids`.
       - If `s` is `func_left`: pick a partner from the chosen isolate's `func_right` bucket. The pair is oriented (`s` in slot A, partner in slot B). If that bucket is empty, count as `missing_right_func` and retry.
       - If `s` is `func_right`: pick a partner from the chosen isolate's `func_left` bucket. The pair is oriented (partner in slot A, `s` in slot B). If empty, count as `missing_left_func` and retry.
       - Apply rejection checks: `seen_pairs` (brc), `cooccur_pairs` (block contradictory), `seen_seq_pairs` (dedup). Schema orientation is correct by construction (no `orient_pair_by_schema` swap needed since we pick from the correct bucket).
     - If a valid partner is found: append the pair, update counters, update `seen_*` sets.
     - If `max_attempts_per_seq` is exhausted: log to `rejection_stats['coverage_skipped']` (list of `(seq_hash, reason)` capped at 100) and move on.
   - Stop the coverage phase when every sequence in the target set either has ≥1 negative in its slot or hit the attempt cap.

3. **Fill phase:**
   - If `len(neg_pairs) < num_negatives`, continue sampling using the schema-mode random-sampling logic from v1 (sample two isolates, sample `func_left` from one and `func_right` from the other, apply all rejection checks) until either `len(neg_pairs) == num_negatives` or `total_attempts >= num_negatives * max_attempts_multiplier`.
   - This phase does NOT enforce balance — some sequences may end up with 1 negative, others with more. Accepted for this iteration.
   - If `len(neg_pairs) >= num_negatives` already (because coverage phase exceeded the target — see "Coverage floor" below), the fill phase is a no-op.

4. **Return:**
   - DataFrame of negatives (same columns as v1).
   - `rejection_stats` dict including:
     - v1 keys still applicable in schema mode (`blocked_cooccur`, `duplicate_brc`, `duplicate_seq`, `missing_left_func`, `missing_right_func`, `total_attempts`).
     - **Removed:** `same_func_limit` (impossible in schema mode).
     - New keys:
       - `requested_negatives`: int — `num_negatives` as passed in (`int(len(pos_pairs) * neg_to_pos_ratio)`)
       - `min_required_for_coverage`: int — the coverage floor (see below)
       - `coverage_phase_pairs`: int — count of negatives produced during coverage phase
       - `fill_phase_pairs`: int — count produced during fill phase
       - `achieved_negatives`: int — `coverage_phase_pairs + fill_phase_pairs` (== len of returned df)
       - `coverage_overrode_ratio`: bool — True if `min_required_for_coverage > requested_negatives`
       - `coverage_skipped`: list of (seq_hash, reason), capped at 100
       - `seqs_with_zero_negatives`: list of seq_hash from `target_seqs_left ∪ target_seqs_right` that ended up with 0 negatives in their relevant slot, capped at 100

**Coverage guarantee:** every sequence in `target_seqs_left ∪ target_seqs_right` appears in ≥1 negative in its appropriate slot, except sequences listed in `seqs_with_zero_negatives` (which hit the attempt cap — these should be rare and are surfaced for inspection).

**Coverage floor (interaction with `num_negatives`):** Coverage takes precedence over the `num_negatives` target. The minimum number of negatives the function will produce is:
 
```
min_required_for_coverage = max(|unique_seqs_in_slot_A_of_pos_pairs|,
                                |unique_seqs_in_slot_B_of_pos_pairs|)
```
 
Each negative pair contributes coverage to one slot-A and one slot-B sequence simultaneously, so the floor is governed by whichever side has more unique sequences. This is generally below `len(pos_pairs)` because the same sequence often appears in multiple positive pairs (e.g., the same HA `seq_hash` appearing in many isolates, paired with different NA sequences in each).
 
The actual output count is:
```
achieved_negatives ≈ max(num_negatives, min_required_for_coverage)
```
 
with two qualifications: (i) `coverage_phase_pairs` may fall short of `min_required_for_coverage` if some sequences hit `max_attempts_per_seq` (those land in `seqs_with_zero_negatives`); (ii) the fill phase may fall short of its target if rejection rates are high (`total_attempts` cap reached).
 
**Behavior cases:**
- `num_negatives ≥ min_required_for_coverage`: coverage phase produces ≈ `min_required_for_coverage` pairs, fill phase produces the remainder, total ≈ `num_negatives`. `coverage_overrode_ratio = False`.
- `num_negatives < min_required_for_coverage`: coverage phase produces ≈ `min_required_for_coverage` pairs (exceeds target), fill phase is a no-op, total ≈ `min_required_for_coverage`. `coverage_overrode_ratio = True`. The function logs a clear warning naming both numbers and explaining that the user-requested ratio was overridden by the coverage requirement.

**Implication for `neg_to_pos_ratio`:** users can set `neg_to_pos_ratio` below the implicit floor (e.g., 0.5 when the floor sits at 0.83), but the floor wins. The effective minimum ratio is `min_required_for_coverage / len(pos_pairs)`, which v2 computes per split rather than asking the user to know it in advance.

**Forward note (do not implement now):** A future iteration may replace the two-phase scheme with a multi-pass round-robin sampler that bounds `max(neg_count) - min(neg_count) ≤ 1`. Add a brief comment in the function pointing to this possibility.


### 4.3 `compute_exposure_stats`

**Signature:**

```python
def compute_exposure_stats(
    pos_pairs: pd.DataFrame,
    neg_pairs: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
```

`pair_mode` is not a parameter — v2 is schema-mode only. See Section 2 "Hard-coded constraints".

**Behavior:**

Returns a per-sequence DataFrame with columns:
- `seq_hash`
- `function`
- `assembly_ids` (list, since biological duplicates exist)
- `n_pos_slot_a`, `n_pos_slot_b`, `n_pos_total`
- `n_neg_slot_a`, `n_neg_slot_b`, `n_neg_total`
- `exposure`: one of `dual`, `pos_only`, `neg_only`

Slot semantics are fixed: slot A = `func_left`, slot B = `func_right`. So a sequence with `function == func_left` will have `n_*_slot_b == 0` by construction (use literal 0, not null — these are counts, "never appeared in slot B" is a true zero). Conversely a `func_right` sequence will have `n_*_slot_a == 0`.

The exposure label:
- `dual` if `n_pos_total > 0` and `n_neg_total > 0`
- `pos_only` if `n_pos_total > 0` and `n_neg_total == 0`
- `neg_only` if `n_pos_total == 0` and `n_neg_total > 0`

A sequence with both 0 should not appear in this table (it's not in any pair).

**`seq_hash → function` consistency (defensive check).** When materializing the per-sequence table from `df`, build a `seq_hash → function` map via `df.groupby('seq_hash')['function'].unique()`. If any seq_hash maps to more than one distinct function, raise `ValueError` (or at minimum log a loud `WARNING:` and use `first()`) naming the offending seq_hashes. v2 assumes `seq_hash → function` is many-to-one — if it ever isn't, that's a Stage 1 (preprocessing) data-quality bug, not something v2 should silently paper over.


### 4.4 `compute_axis_flags`

**Signature:**

```python
def compute_axis_flags(
    pairs: pd.DataFrame,
    df: pd.DataFrame,
    axes: list[str] = ("host", "year", "hn_subtype", "geo_location", "passage"),
) -> pd.DataFrame:
```

**Behavior:**

For each axis in `axes` that exists as a column in `df` (also accept `geo_location_clean` aliased as `geo_location`), look up the value for each `seq_hash_a` and `seq_hash_b` in `pairs` and add three columns:
- `<axis>_a`: value from sequence A's source row (use first match if biological duplicate spans multiple isolates with conflicting metadata — log a warning if conflicts exist)
- `<axis>_b`: value from sequence B's source row
- `same_<axis>`: nullable boolean. `True` if both values are non-null and equal; `False` if both non-null and different; `pd.NA` if either is null.

Implementation note: build `seq_hash → metadata` lookup from `df` once (groupby `seq_hash` + `first()` per axis), then map twice. Watch for and log any seq_hash that has multiple distinct values for an axis across its source rows — this indicates upstream metadata inconsistency.

The same defensive policy from §4.3 applies to the `function` column, but escalated: a seq_hash mapping to >1 distinct `function` is treated as a hard error (raise `ValueError`), not a warning. Axis values like `host`/`year` can plausibly drift across biological duplicates (annotation noise); `function` is an upstream invariant that v2's slot semantics depend on, so violating it must fail loudly.

Returns the input `pairs` with the new columns appended (do not mutate the caller's df; return a copy or work on a copy internally).


### 4.5 `compute_metadata_coverage`

**Signature:**

```python
def compute_metadata_coverage(
    df: pd.DataFrame,
    axes: list[str] = ("host", "year", "hn_subtype", "geo_location", "passage"),
) -> dict:
```

**Behavior:**

For each axis, report:
- `n_rows_total`, `n_rows_null`, `pct_null`
- `n_unique_seqs`, `n_unique_seqs_null`, `pct_unique_null`
- `n_distinct_values` (excluding null)
- `top_5_values`: list of `(value, count)` for sanity

Returned as a dict suitable for `json.dump`. If an axis is not present in `df`, mark as `{"present": False}`.


### 4.6 `split_dataset_v2`

**Signature:**

```python
def split_dataset_v2(
    df: pd.DataFrame,
    schema_pair: Tuple[str, str],            # required
    neg_to_pos_ratio: float = 3.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_attempts_per_seq: int = 50,
    max_attempts_multiplier: int = 100,
    axes_for_flags: list[str] = ("host", "year", "hn_subtype", "geo_location", "passage"),
    axis_quotas: Optional[dict] = None,      # placeholder; must be None
    train_isolates_override: Optional[list] = None,
    val_isolates_override: Optional[list] = None,
    test_isolates_override: Optional[list] = None,
    cooccur_pairs: Optional[set] = None,
    cooccur_stats: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
```

Note on dropped parameters (see Section 2 "Hard-coded constraints"):
- `allow_same_func_negatives`, `max_same_func_ratio` — same-function negatives impossible in schema mode.
- `hard_partition_isolates` — always True in v2; isolate-disjointness logic and overlap detection are kept.
- `canonicalize_pair_orientation_enabled` — schema mode defines orientation.
- `pair_mode` — schema mode only.
- `drop_within_split_pos_duplicates` — always True in v2; dedup runs unconditionally.

The dropped-parameter comment template applies at the top of this function.

**Behavior:**

Same overall flow as v1's `split_dataset`, schema-mode-only branch, plus the v2 additions:

1. Build co-occurrence set (or accept pre-computed via `cooccur_pairs` / `cooccur_stats` for CV reuse).
2. Stratified isolate split using `compute_isolate_pair_counts` (or use overrides if provided).
3. For each split, generate positives via `create_positive_pairs_v2` (returns dedup stats).
4. For each split, generate negatives via `create_negative_pairs_v2` passing the split's positive DataFrame as `pos_pairs` (so coverage targets are split-local).
5. After concatenating pos + neg per split: call `compute_axis_flags(split_pairs, df, axes_for_flags)` to attach `<axis>_a`, `<axis>_b`, `same_<axis>` columns.
6. After axis flags: call `compute_exposure_stats` per split.
7. Run the cross-split `pair_key` overlap check the same way v1 does, and apply the same removal logic.
8. Schema-mode sanity assertions (every pair has `func_a == func_left`, `func_b == func_right`) — kept from v1.
9. Return tuple: `(train_pairs, val_pairs, test_pairs, duplicate_stats, exposure_tables)` where `exposure_tables` is `{'train': df, 'val': df, 'test': df}`.

`duplicate_stats` extends v1's dict with:
- `pos_dedup_stats`: per-split dedup stats from positives
- `coverage_stats`: per-split `{coverage_phase_pairs, fill_phase_pairs, seqs_with_zero_negatives, coverage_skipped}`


### 4.7 `generate_all_cv_folds_v2`

Same generator pattern as v1's `generate_all_cv_folds` but calls `split_dataset_v2`. Yielded fold dicts include `exposure_tables`.

Parameters dropped from v2 (see Section 2): `allow_same_func_negatives`, `max_same_func_ratio`, `hard_partition_isolates`, `canonicalize_pair_orientation_enabled`, `pair_mode`. `schema_pair` is required.

The dropped-parameter comment template applies at the top of this function.


### 4.8 `generate_temporal_split_v2`

```python
def generate_temporal_split_v2(*args, **kwargs):
    raise NotImplementedError(
        "Temporal split is not yet supported in pair_builder_version=v2. "
        "Use pair_builder_version=v1 for temporal experiments, or contribute v2 support."
    )
```


### 4.9 `save_split_output_v2`

Same as v1's `save_split_output` plus:

1. Write `sequence_exposure.csv` from the exposure table.
2. Extend `dataset_stats.json` with new sections — see section 6.
3. Add a `pair_builder_version: "v2"` field to `dataset_stats.json`.

Note: `metadata_coverage.json` is **not** written here. Its scope is per-run, not per-split (the source `df` is identical across splits and across CV folds), so it is computed and written once by the CLI dispatch (see §4.10) before the split / CV branch. Putting it in the per-split saver would either duplicate the file across folds or require a `write_metadata_coverage: bool` flag that pushes CV-vs-single-split branching into the wrong layer.


### 4.10 CLI dispatch

In the main script (kept in v1 file or extracted to a small launcher), after config load:

```python
PAIR_BUILDER_VERSION = getattr(config.dataset, "pair_builder_version", "v1")

if PAIR_BUILDER_VERSION == "v2":
    _validate_v2_config(config)  # see §5; uses raw config access, not getattr
    if YEAR_TRAIN is not None:
        raise ValueError(
            "pair_builder_version=v2 does not yet support temporal split "
            "(year_train/year_test). Use v1 or remove temporal config."
        )
    from dataset_segment_pairs_v2 import (
        split_dataset_v2,
        generate_all_cv_folds_v2,
        save_split_output_v2,
        compute_metadata_coverage,
    )

    # Per-run artifact: written once before split / CV branching.
    # df is already finalized at this point (filtering + isolate sampling done).
    coverage = compute_metadata_coverage(df, axes=AXES_FOR_FLAGS)
    with open(run_output_dir / "metadata_coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    # Then dispatch to v2 builders for single-split / CV.
elif PAIR_BUILDER_VERSION == "v1":
    # existing v1 dispatch
else:
    raise ValueError(f"Unknown pair_builder_version: {PAIR_BUILDER_VERSION!r}")
```

Note: the launcher `scripts/run_cv_lambda.py` does not need changes. Its contract with Stage 3 is "spawn the script once, expect `fold_*/` subdirs in the run output dir" — `metadata_coverage.json` simply lands in the parent of those fold dirs.


---


## 5. Config additions

Under `dataset.*` (probably `conf/dataset/default.yaml`):

```yaml
pair_builder_version: v1                    # "v1" (default) | "v2"
max_attempts_per_seq: 50                    # v2-only; coverage-phase attempt cap
axes_for_flags:                             # v2-only; axes annotated on pairs
  - host
  - year
  - hn_subtype
  - geo_location
  - passage
axis_quotas: null                           # v2-only; placeholder, must be null
```

Note: `drop_within_split_pos_duplicates` is **not** a config key in v2 (always True; see Section 2 "Hard-coded constraints").

### v2 config validation

When `pair_builder_version == "v2"`, validate before dispatching to v2 builders. **Each validation failure must raise with a message that names the offending key, the value found, and the value v2 requires** (so a user reading the error can fix their config without consulting this spec).

Required:
- `dataset.schema_pair` — must be present, length 2, and `func_left != func_right`. Raise `ValueError` otherwise.

Hard-coded values that must not contradict (raise `ValueError` if set to anything else; absent or matching → OK):
- `dataset.pair_mode` — must be `"schema_ordered"` or absent.
- `dataset.allow_same_func_negatives` — must be `false` or absent.
- `dataset.canonicalize_pair_orientation` — must be `false` or absent.
- `dataset.hard_partition_isolates` — must be `true` or absent.

Other validations:
- `dataset.axis_quotas` non-null/non-empty → raise `NotImplementedError` (placeholder; not yet implemented).
- `dataset.year_train` set → raise `ValueError` (temporal not yet supported in v2).
- `dataset.axes_for_flags` axis missing from `df` → warn, drop from list, continue.

Validation lives in a single function (e.g., `_validate_v2_config(config)`) called from the CLI dispatch before importing the v2 module. Keep it loud and explicit — these errors are the user's primary feedback for the v2 contract.

**Validation reads the raw config, not Python-resolved values.** Use `OmegaConf.select(config, "dataset.pair_mode")` (returns `None` for absent keys) rather than `getattr(config.dataset, "pair_mode", "schema_ordered")` (collapses absent and explicit-but-default into the same string). This lets error messages distinguish "you set `pair_mode=unordered`" from "key was absent (defaulted)" — the former is a violation, the latter is fine. Runtime extraction further down the dispatch may keep the `getattr(..., default=<v2-appropriate>)` pattern; by that point the validator has already rejected anything forbidden, so defaults are safe. Do not unify the two patterns.


---


## 6. Output schema

### Pair CSVs (`train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv`)

All v1 columns, plus, for each axis in `axes_for_flags`:
- `<axis>_a`
- `<axis>_b`
- `same_<axis>` (nullable boolean)

### `sequence_exposure.csv` (per split, written to split's output dir)

Columns: `seq_hash`, `function`, `assembly_ids` (semicolon-joined string for CSV roundtrip), `n_pos_slot_a`, `n_pos_slot_b`, `n_pos_total`, `n_neg_slot_a`, `n_neg_slot_b`, `n_neg_total`, `exposure`.

### `metadata_coverage.json`

Written once at the run output directory by the CLI dispatch (§4.10), before any split / CV branching. The file lives at the run root in both single-split and CV modes — in CV mode this is the parent of `fold_0/ … fold_{N-1}/`. The source `df` is identical across splits and across folds, so per-fold copies would be redundant.

Schema: dict from axis name to coverage stats (see 4.5).

### `dataset_stats.json` (extended)

Add these top-level sections:

```json
{
  "pair_builder_version": "v2",
  "exposure_summary": {
    "train": {"dual": N, "pos_only": N, "neg_only": N, "n_unique_seqs": N},
    "val": {...},
    "test": {...}
  },
  "axis_flag_summary": {
    "train": {
      "host":       {"same": N, "diff": N, "null": N},
      "year":       {...},
      "hn_subtype": {...},
      "geo_location": {...},
      "passage":    {...}
    },
    "val": {...},
    "test": {...}
  },
  "pos_dedup": {
    "train": {"n_before": N, "n_after": N, "n_dropped": N},
    "val": {...},
    "test": {...}
  },
  "coverage": {
    "train": {
      "requested_negatives": N,
      "min_required_for_coverage": N,
      "achieved_negatives": N,
      "coverage_phase_pairs": N,
      "fill_phase_pairs": N,
      "coverage_overrode_ratio": false,
      "n_seqs_with_zero_negatives": N
    },
    "val": {...},
    "test": {...}
  }
}
```

All v1 sections (`split_sizes`, `metadata_distributions`, `co_occurrence_blocking`, `filters_applied`) remain.


---


## 7. Test / verification checklist

After implementing, run:

1. **v1 regression.** Run the existing pipeline with `pair_builder_version=v1` (or unset) on a known config (`flu_ha_na.yaml`) — outputs must be byte-identical to current v1 outputs (or differ only in fields v1 doesn't write).

2. **v2 single-split smoke test.** Run on the existing flu config with `pair_builder_version=v2`, single-split mode. Verify:
   - All v1 output files present.
   - `sequence_exposure.csv` exists per split.
   - `metadata_coverage.json` exists at run root.
   - `dataset_stats.json` contains all new sections.
   - For every split, every `seq_hash` appearing in `train_pos`/`val_pos`/`test_pos` has `n_neg_total >= 1` in that split's exposure CSV (the coverage guarantee — modulo `seqs_with_zero_negatives` which should be empty or have a logged reason).

3. **v2 CV smoke test.** Run with `n_folds=3`, `pair_builder_version=v2`. Verify each `fold_*/` directory has all v2 outputs; `metadata_coverage.json` is at the parent (not per fold).

4. **v2 temporal rejection.** Run with `year_train` set and `pair_builder_version=v2`. Must error clearly with the message specified in section 4.10.

5. **Within-split positive dedup.** Build a synthetic small dataset where two isolates contain the same protein pair (e.g., same HA seq + same NA seq in two different isolates). Run v2. Verify `pos_dedup.train.n_dropped >= 1` in `dataset_stats.json` and that `train_pairs.csv` has no duplicate `pair_key` rows.

6. **Axis flag correctness.** On a small slice of real data, manually verify a handful of pairs: pick a positive where both seqs come from the same H3N2 human isolate → expect `same_hn_subtype=True`, `same_host=True`. Pick a negative across H3N2 and H1N1 → expect `same_hn_subtype=False`.

7. **Exposure correctness.** On the same slice, verify a sequence that appears in both pos and neg has `exposure=dual`; one that appears only in pos has `exposure=pos_only`. Confirm `neg_only` is empty (expected under current upstream filtering).

8. **`axis_quotas` rejection.** Pass a non-empty `axis_quotas` dict via override. Must raise `NotImplementedError` with the message in section 4.2.

9. **`max_attempts_per_seq` exhaustion.** Construct a tiny dataset where one sequence has no possible valid negative partner (e.g., all its compatible partners are blocked by `cooccur_pairs`). Verify it appears in `seqs_with_zero_negatives` and `coverage_skipped`, and that the run completes without crashing.

10. **Hard-coded constraint validation.** Run with `pair_builder_version=v2` and each of the following overrides one at a time. Each must raise `ValueError` with a message naming the offending key:
    - `dataset.pair_mode=unordered`
    - `dataset.allow_same_func_negatives=true`
    - `dataset.canonicalize_pair_orientation=true`
    - `dataset.hard_partition_isolates=false`
    - `dataset.schema_pair=null` (or omitted)

11. **Hard-coded constraint pass-through.** Run with `pair_builder_version=v2` and each of the following set to its required value (or absent). Run must proceed normally:
    - `dataset.pair_mode=schema_ordered` (or absent)
    - `dataset.allow_same_func_negatives=false` (or absent)
    - `dataset.canonicalize_pair_orientation=false` (or absent)
    - `dataset.hard_partition_isolates=true` (or absent)

12. **Coverage floor override.** Run with `dataset.neg_to_pos_ratio` set deliberately low (e.g., `0.1`) so that `num_negatives < min_required_for_coverage`. Verify in `dataset_stats.json`:
    - `coverage.train.coverage_overrode_ratio == true`
    - `coverage.train.achieved_negatives ≥ coverage.train.min_required_for_coverage`
    - `coverage.train.achieved_negatives > coverage.train.requested_negatives`
    - `coverage.train.fill_phase_pairs == 0` (fill is a no-op when coverage already exceeds target)
    - The run logged a clear warning naming both `requested_negatives` and `min_required_for_coverage`.

13. **Coverage floor non-override.** Run with `dataset.neg_to_pos_ratio` set high enough (e.g., `3.0`) so that `num_negatives ≥ min_required_for_coverage`. Verify:
    - `coverage.train.coverage_overrode_ratio == false`
    - `coverage.train.achieved_negatives ≈ coverage.train.requested_negatives` (within fill-phase rejection slack)
    - `coverage.train.fill_phase_pairs > 0`

---


## 8. Edge cases and decisions to confirm during implementation

- **Biological duplicate sequences with conflicting metadata.** A sequence appearing in two isolates with different `host` values is upstream data quality issue. v2 must detect and log this in `compute_axis_flags` (warn, use first match) but not crash.
- **`same_<axis>` when both values are null.** Encode as `pd.NA`, not `True`. Two unknowns are not "same".
- **Slot-aware coverage.** A `func_left` sequence needs ≥1 negative in slot A; a `func_right` sequence needs ≥1 negative in slot B. Track `neg_count_a` and `neg_count_b` separately; do not conflate.
- **CV `cooccur_pairs` reuse.** v1's `generate_all_cv_folds` builds `cooccur_pairs` once and passes to all folds. v2 must do the same — coverage targets are per-fold but `cooccur_pairs` is dataset-level.
- **Determinism.** Every random draw must come from a `random.Random(seed)` instance owned by the function (no reads from the process-global random stream). v1 already established this; v2 must preserve it. CV folds use `seed + fold_i` per fold, same as v1.
- **`num_negatives` as target with coverage floor.** Section 4.2 documents the floor-override rule: actual output may exceed `num_negatives` if the coverage floor is higher, and may fall short of either if rejection rates are high. Logging must report `requested_negatives`, `min_required_for_coverage`, and `achieved_negatives`. Downstream code that assumes `len(neg_pairs) == int(len(pos_pairs) * neg_to_pos_ratio)` exactly will break — search for this assumption in the codebase before declaring done.
- **Empty splits.** v1 raises if any split is empty. v2 preserves this.


---


## 9. Decisions log (already settled — do not re-litigate)

These were decided in the design discussion (with Claude.ai) that produced this spec. Record here so they don't get rediscussed.

1. **New file, not modify in place.** `dataset_segment_pairs_v2.py` as sibling to v1.
2. **Config flag gates dispatch.** `dataset.pair_builder_version: v1 | v2`, default `v1`.
3. **Exposure naming.** `dual` / `pos_only` / `neg_only`. `neg_only` is empty under current pipeline (single-protein isolates dropped upstream + combinatorial positives) but field exists for forward compatibility.
4. **Axis encoding.** Three columns per axis: `<axis>_a`, `<axis>_b`, `same_<axis>`. Boolean is nullable. Don't collapse to value-or-N/A.
5. **Coverage algorithm.** Two-phase: (a) coverage phase guarantees every `target_seq` gets ≥1 negative, (b) fill phase tops up to `num_negatives` using v1's random-sampling logic. Stricter balance via multi-pass round-robin (`max - min ≤ 1`) is explicitly deferred to a future iteration; mention in a code comment but do not implement.
6. **Coverage keyed by `seq_hash`.** Not by `(seq_hash, assembly_id)`. Biological duplicates are one coverage unit.
7. **Slot-aware coverage.** Track `neg_count_a` and `neg_count_b` separately. In schema mode (the only mode in v2) the slot is determined by function, so a given seq_hash appears in exactly one of the two dicts.
8. **Within-split pos dedup is unconditional in v2.** Always runs; no flag. (See decision 16 below.)
9. **Axis quotas are placeholder only.** Parameter exists; non-empty value raises `NotImplementedError`.
10. **Temporal is placeholder.** Stub raises `NotImplementedError`. CLI dispatch errors clearly.
11. **Eval metrics out of scope.** v2 produces `exposure` and `same_<axis>` columns + per-seq tables. Computing AUC-by-slice is downstream training/eval script's job.
12. **Single-split + CV in scope; temporal stub only.**
13. **Shared helpers imported from v1** (or factored to `_pair_helpers.py` if v1 has script-level code blocking import). Do not duplicate `canonical_pair_key`, `build_cooccurrence_set`, `attach_dna_to_prot_df`, etc.
14. **`metadata_coverage.json` written once per run**, not per fold.
15. **Three duplicate-handling cases must be documented in the v2 module docstring**: within-split positive duplicates (handled by unconditional dedup), within-split negative duplicates (handled by `seen_*` sets), cross-split duplicates (handled by `pair_key` overlap check). The "Duplicate handling" section of the docstring is the canonical reference.
16. **Several v1-configurable behaviors are hard-coded in v2.** `pair_mode = "schema_ordered"`, `allow_same_func_negatives = False`, `max_same_func_ratio = 0.5` (vestigial), `canonicalize_pair_orientation_enabled = False`, `hard_partition_isolates = True`, `drop_within_split_pos_duplicates = True`. The corresponding parameters are dropped from v2 function signatures and v2 config validation rejects contradicting values. Each enforcement site has a comment referring back to v1 (`dataset_segment_pairs.py`) where the configurable version still lives. Full list and rationale in Section 2 "Hard-coded constraints (v2 simplifications)".
17. **Coverage floor overrides `num_negatives`.** `min_required_for_coverage = max(|unique_seqs_in_slot_A|, |unique_seqs_in_slot_B|)` over `pos_pairs`. When `int(len(pos_pairs) * neg_to_pos_ratio) < min_required_for_coverage`, the coverage phase produces ≈ `min_required_for_coverage` pairs anyway and the fill phase becomes a no-op. The override is logged as a warning and surfaced in `dataset_stats.json` via `coverage_overrode_ratio: bool` plus `requested_negatives` and `min_required_for_coverage` numeric fields. Coverage takes precedence; the user-supplied ratio acts as a floor for fill, not a ceiling on output. Adding a `coverage_required: bool` opt-out flag is deliberately deferred — users wanting v1's no-coverage sampler can use `pair_builder_version=v1`. Detailed semantics in Section 4.2 "Coverage floor".
18. **`metadata_coverage.json` is a per-run artifact, written by the CLI dispatch.** It is computed once after `df` is finalized and before split / CV branching, and saved to the run output dir (the parent of `fold_*/` in CV mode). `save_split_output_v2` does not write it. This keeps CV-vs-single-split branching out of the per-split saver and avoids redundant per-fold copies. Detail in §4.9 / §4.10 / §6.
19. **v2 config validation reads raw config; runtime extraction may use defaults.** `_validate_v2_config` uses `OmegaConf.select` (returns `None` for absent keys) so "absent" and "explicitly set to a forbidden value" are distinguishable in error messages. Once validation passes, the runtime extraction code may use `getattr(config.dataset, key, <v2-appropriate default>)` as usual. The two patterns are deliberately separate; do not unify. Detail in §5.
20. **`seq_hash → function` is a hard invariant in v2.** The seq_hash → function lookup raises `ValueError` if any seq_hash maps to >1 distinct function (treated as a Stage 1 preprocessing bug, not something v2 silently papers over). This is stronger than the warn-and-use-first policy applied to other axis values (host/year/etc.), because v2's slot semantics depend on function being well-defined per sequence. Detail in §4.3 / §4.4.
