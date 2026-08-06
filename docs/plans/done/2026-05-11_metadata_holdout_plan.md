# metadata_holdout — generalized cross-population split — Plan

**Status: IMPLEMENTED** (2026-05-11)
**Date:** 2026-05-11
**Branch:** feature/metadata-holdout

---

## Context

Stage 3 needs to support cross-population generalization experiments
where train and test live in different regions of `(host, hn_subtype,
year, geo_location, passage)` space. Example user query:

```
train = (host=Human,  hn_subtype=H3N2, year ∈ [1990, 2020])
test  = (host=Pig,    hn_subtype=H1N1, year ≥ 2022)
```

The v2 builder already exposes `train_isolates_override` /
`val_isolates_override` / `test_isolates_override` parameters on
`split_dataset_v2` (the CV path uses them). What's missing is a
config-level recipe that computes the three isolate-id lists from
metadata filters and routes them through that hook.

The existing `year_train`/`year_test` mechanism (v1 only — v2 stub
just rejects) covers the year-axis-only special case. Now that no
temporal bundles, datasets, or runs exist, we delete it outright
rather than maintain two mechanisms.

Related layers (do NOT collapse with this one):
- `filter_by_metadata`: **universe restriction** — narrows the input
  df before any split happens. Composes with `metadata_holdout` (e.g.,
  filter to `host=Human`, then split by `hn_subtype` within that).
- `dataset.split_strategy.mode={random, seq_disjoint}`: **graph- or
  shuffle-based split** within the universe. Mutually exclusive with
  `metadata_holdout` for now.

---

## Design

### Two functions in `_pair_helpers.py`

1. **`filter_by_metadata(df, **filter_kwargs)`** (existing — extended).
   Universe-restriction utility. Continues to take a single filter
   spec, returns a single filtered df. Accepts scalars, lists, and
   ranges per axis (see schema below). Used today by the
   single-subtype / single-host bundles; the extension is
   backwards-compatible.

2. **`compute_metadata_holdout_isolates(df, holdout_cfg, seed, val_ratio) -> (train_ids, val_ids, test_ids, dropped_df)`** (new).
   Calls `filter_by_metadata` per slot to build the three isolate
   sets, carves val from train if `val: null`, detects multi-slot
   matches, and produces the dropped-isolates manifest.

Two responsibilities, two functions. `filter_by_metadata` knows nothing
about train/val/test; `compute_metadata_holdout_isolates` knows nothing
about how to apply a single filter.

### Filter value schema (extended `filter_by_metadata`)

| Form | Semantics | Applies to |
|---|---|---|
| `host: "Human"` (scalar) | exact match | host, hn_subtype, geo_location, passage, year |
| `host: ["Human", "Pig"]` (list) | **set membership** — any length, including 1-element `["Human"]` | host, hn_subtype, geo_location, passage, year |
| `year_range: [2020, 2024]` (inclusive range) | min ≤ year ≤ max | **year only** |

Rules:
- Lists are **always set membership**, regardless of length. No
  length-based magic (`[2020, 2024]` is `{2020, 2024}`, not `[2020..2024]`).
- Ranges live only under their own key (`year_range`). No `host_range`
  etc. (host/subtype/geo/passage aren't ordered axes).
- `year` and `year_range` cannot both be set in the same filter dict.
  Validator rejects.

### Config block

```yaml
dataset:
  metadata_holdout:
    train:
      host: Human
      hn_subtype: H3N2
      year_range: [1990, 2020]
    test:
      host: Pig
      hn_subtype: H1N1
      year_range: [2022, 9999]
    val: null   # null -> carve val_ratio off train pool randomly
    # or: val: {host: Human, hn_subtype: H3N2, year: 2021}
```

- `train` and `test` are required (non-null).
- `val: null` → carve `val_ratio` fraction (from `dataset.val_ratio`,
  default 0.1) off the train pool by random partition.
- `val: {filter}` → independent filter; computed like train/test.
- Every isolate-axis key on every slot is optional. Omitted axis = no
  constraint on that axis.

### Algorithm (`compute_metadata_holdout_isolates`)

1. **Validate `holdout_cfg`** (called by the validator earlier — see
   "Failure modes"). Reject unknown axis keys; reject `year` + `year_range`
   simultaneously; reject `year_range[0] > year_range[1]`; reject
   `train: null` or `test: null`.
2. **Apply train filter, test filter** to `df` → two sets of
   `assembly_id`s.
3. **Multi-slot overlap check.** Any isolate matching both train and
   test filters → collect into `overlaps_train_test`. If non-empty:
   raise `ValueError` listing the first 20 offending
   `(assembly_id, host, hn_subtype, year)` rows and which slots they
   matched. Same check for train↔val and val↔test if `val` is an
   explicit filter dict.
4. **Val carving** (only if `val: null`):
   - Partition train pool by `train_test_split(train_pool,
     test_size=val_ratio, random_state=seed)`.
   - Val ends up as a random subset of train's metadata distribution
     (the default behavior matches the temporal-holdout use case where
     val is in-distribution with train for early stopping).
5. **Coverage feasibility tripwire.** For each of (train, val, test):
   if the pool is empty, raise. Per-`seq_hash` coverage failure is
   detected later by `create_negative_pairs_v2` (already a hard raise).
6. **Build dropped-isolates manifest.** For every isolate in `df` not
   in (train ∪ val ∪ test):
   - Collect `assembly_id, file, host, hn_subtype, year, geo_location,
     passage, matches_train, matches_val, matches_test, excluded_reason`.
   - `excluded_reason` is a short text: e.g.,
     `"host=Pig does not match train.host=Human, test.host=Pig but
     year=2020 does not match test.year_range=[2022, 9999]"`.
     Generated mechanically by walking each slot's filter dict.
7. **Return** `(sorted(train_ids), sorted(val_ids), sorted(test_ids),
   dropped_df)`.

### CLI dispatch wiring (in `dataset_segment_pairs.py`)

After `df` is finalized and before `split_dataset_v2`:

```python
HOLDOUT_CFG = getattr(config.dataset, "metadata_holdout", None)
if HOLDOUT_CFG is not None:
    if SPLIT_STRATEGY_MODE != "random":
        raise ValueError(...)
    if N_FOLDS is not None and N_FOLDS > 1:
        raise ValueError(...)
    train_ids, val_ids, test_ids, dropped_df = compute_metadata_holdout_isolates(
        df, HOLDOUT_CFG, seed=RANDOM_SEED, val_ratio=VAL_RATIO,
    )
    dropped_df.to_csv(output_dir / "metadata_holdout_dropped.csv", index=False)
    # ... pass train_ids/val_ids/test_ids as overrides to split_dataset_v2
```

The override path already exists (CV uses it); `split_dataset_v2`'s
`train_isolates_override` branch handles routing positives + negatives
correctly.

### `dataset_stats.json` extension

Add a top-level `metadata_holdout` section when the holdout path is
used:

```json
"metadata_holdout": {
  "config": { ... <holdout_cfg, normalized> ... },
  "n_train_isolates": N,
  "n_val_isolates": N,
  "n_test_isolates": N,
  "n_dropped": N,
  "val_source": "carved_from_train" | "explicit_filter"
}
```

---

## Failure modes (validator + runtime)

Caught at v2 config-validation time (`_validate_v2_config`):

| Trigger | Error |
|---|---|
| `metadata_holdout.train` is null | ValueError |
| `metadata_holdout.test` is null | ValueError |
| Unknown axis key in any slot filter (e.g., `subtype: H3N2` instead of `hn_subtype`) | ValueError listing supported axes |
| `year` and `year_range` both set in the same slot filter | ValueError |
| `year_range[0] > year_range[1]` | ValueError |
| `year_range` malformed (not a 2-element list of ints) | ValueError |
| `host_range` / `hn_subtype_range` / etc. (range key on unordered axis) | ValueError |
| `metadata_holdout` set AND `n_folds > 1` (CV mode) | ValueError, mutually exclusive |
| `metadata_holdout` set AND `dataset.split_strategy.mode != 'random'` | ValueError, mutually exclusive |
| `year_train` or `year_test` key present anywhere | ValueError — these keys are removed (see scope item 1) |

Caught at compute time (`compute_metadata_holdout_isolates`):

| Trigger | Error |
|---|---|
| Train, val, or test pool is empty after filtering | ValueError naming the empty slot and its filter |
| Any isolate matches > 1 slot | ValueError listing first 20 conflicting `assembly_id`s + their slots |
| Val carve produces an empty pool (e.g., train < 1/val_ratio isolates) | ValueError |

Caught downstream (existing v2 mechanism, unchanged):

| Trigger | Error |
|---|---|
| Per-`seq_hash` coverage failure inside `create_negative_pairs_v2` for any split | ValueError (existing hard raise) |

---

## Files touched

### New

- `docs/plans/2026-05-11_metadata_holdout_plan.md` — this file.
- `tests/test_metadata_holdout.py` — see "Test plan" below.

### Modified

- `src/datasets/_pair_helpers.py` — extend `filter_by_metadata` (range
  + list support); add `compute_metadata_holdout_isolates`.
- `src/datasets/dataset_segment_pairs.py` — wire `metadata_holdout`
  into the v2 CLI dispatch; **delete** v1 `generate_temporal_split`
  and all `year_train` / `year_test` config extraction.
- `src/datasets/dataset_segment_pairs_v2.py` — extend `_validate_v2_config`
  with the rules in the table above; **delete** the existing
  `year_train` rejection block (the key won't exist after cleanup);
  add the metadata_holdout validation.
- `conf/dataset/default.yaml` — add commented `metadata_holdout: null`
  example block; **delete** `year_train: null` / `year_test: null`.

### Doc cleanup

- `docs/plans/done/2026-05-11_design_dataset_gen_v2.md` — done; the
  year_train rejection reference has been updated to point at
  `metadata_holdout`.
- (`docs/EXP_RESULTS_STATUS.md` was deleted during the 2026-05-12
  docs prune; no further cleanup needed there.)

### Deleted

- `docs/plans/done/temporal_holdout_plan.md` — deleted (user
  instruction; the temporal-holdout work it described is superseded by
  this plan).

---

## Test plan (`tests/test_metadata_holdout.py`)

Follow `tests/test_level1_neg_regimes.py`'s pattern: hand-built
synthetic DataFrames, one test function per behavior, parametrize only
where the test body is genuinely identical across cases.

### Happy path — `filter_by_metadata` extension (parametrized)

`test_filter_by_metadata_accepts_*` — one parametrized table with ~6
rows covering: scalar host; list host; scalar year; year list (set
membership including 1-element); year_range inclusive; combined
multi-axis filter. Each row asserts the resulting `assembly_id` set.

### Happy path — `compute_metadata_holdout_isolates`

- `test_compute_metadata_holdout_isolates_basic` — 3-slot explicit
  filters; verifies returned (train, val, test) lists are correct,
  sorted, disjoint, and that `dropped_df` accounts for the rest.
- `test_metadata_holdout_carves_val_from_train_when_val_null` —
  `val: null` carves `val_ratio` off train; check fraction and
  determinism (re-running with same seed yields identical val_ids).
- `test_metadata_holdout_uses_explicit_val_filter` — `val` is an
  independent filter; val comes from that filter, not from train carve.

### Failure modes — one function each

- `test_raises_when_train_pool_empty`
- `test_raises_when_val_pool_empty` (both for empty filter and for
  too-small carve)
- `test_raises_when_test_pool_empty`
- `test_raises_when_year_and_year_range_both_set`
- `test_raises_when_year_range_min_gt_max`
- `test_raises_when_year_range_wrong_length`
- `test_raises_when_unknown_axis_key`
- `test_raises_when_host_range_used` (range on non-ordered axis)
- `test_raises_when_train_test_pools_overlap` — isolate matches both
  train and test filters; error message lists offending assembly_ids
- `test_raises_when_combined_with_n_folds_cv`
- `test_raises_when_combined_with_seq_disjoint_split_strategy`
- `test_raises_when_year_train_legacy_key_present` — leftover from old
  bundles after key deletion; clear "key removed; use metadata_holdout"
  message.

### Manifest

- `test_dropped_isolates_manifest_columns` — schema check on
  `metadata_holdout_dropped.csv` columns.
- `test_dropped_isolates_manifest_excluded_reason` — verifies the
  `excluded_reason` text names the specific axis(es) that mismatched.

Total ~15 tests, all under 50 lines each.

---

## Out of scope

- Composition with `seq_disjoint` (mutually exclusive for v1 of this
  feature). A future iteration could allow `seq_disjoint`-style routing
  *within each metadata-defined pool* (so the test pool is internally
  seq-disjoint from itself; trivially is already, by metadata
  construction) — but cross-pool seq-disjointness only matters once
  the pools share metadata, which by design they don't here.
- Composition with CV (also mutually exclusive). CV bake-in for
  cross-population holdouts would need a per-fold sampling strategy
  over the metadata space — not in scope.
- ESM-2-feature path. Same as seq_disjoint: one bundle change away
  once metadata_holdout is in.
- Auto-translation of removed `year_train`/`year_test` keys into
  `metadata_holdout` (per decision Q1: delete entirely, don't migrate).

---

## Success criteria

1. `year_train` / `year_test` removed everywhere; running a bundle
   that still references them errors at Hydra/validate time with a
   clear message ("key removed; use `dataset.metadata_holdout`").
2. New bundle `flu_ha_na_metadata_holdout.yaml` (or similar) builds
   end-to-end with the example config in the "Config block" section;
   `metadata_holdout_dropped.csv` is non-empty and well-formed;
   `dataset_stats.json` carries the `metadata_holdout` section;
   train/val/test isolate sets are non-empty, sorted, and disjoint.
3. All ~15 tests in `tests/test_metadata_holdout.py` pass.
4. Existing bundles (random-split, seq_disjoint, regime-aware) build
   unchanged — `metadata_holdout` defaults to `None`, no behavior
   change.
5. v1 path completely free of temporal-holdout code; `git grep
   year_train\|year_test` returns hits only in
   `docs/plans/done/*` historical archives (if any), this plan file,
   and tests that assert the removed-key error message.
