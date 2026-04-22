# Plan: Fix Negative-Pair RNG Determinism in Dataset Generation

**Status: DRAFT** — diagnosis complete, fix proposed, not yet implemented

## Motivation

All 28 paper-experiment bundles (`flu_28p_*`) inherit from
`flu_28_major_protein_pairs_master.yaml` and share `master_seed: 42`. Only
`dataset.schema_pair` differs between them. The expectation going into Task 11 was
that the isolate-level train/val/test splits would be identical across the 28
configurations, with only the feature-extraction step differing per pair.

This plan asks a sharper question: does that determinism extend to the specific
*negative pairs* sampled — i.e., for `ha_na/fold_0` and `pb2_pb1/fold_0`, do we get
the same set of `(isolate_a_id, isolate_b_id)` pairs labeled negative, or do they
diverge? The answer matters because it affects how cleanly results across the 28
pairs can be compared, and because seemingly small RNG divergences hint at deeper
reproducibility gaps.

## TL;DR

- Isolate-level splits **are** identical across the 28 bundles at the same fold_id
  (pure function of `(master_seed, fold_id)`).
- Negative-pair identities **are not** — train overlap is ~85%, val/test overlap is
  ~0%.
- Root cause: `create_negative_pairs` uses the module-level `random` state, and the
  `seed` argument passed from `generate_all_cv_folds` (`seed + fold_i`, commented
  "fold-specific seed for negative sampling") is silently discarded. Rejection
  patterns in schema_ordered mode depend on `schema_pair`, so the global RNG stream
  desynchronizes within fold 0 and stays desynchronized across all subsequent folds.
- Minimal fix: instantiate a local `random.Random(seed)` inside
  `create_negative_pairs` and use it for all `sample` / `choice` calls. Makes
  negative sampling a pure function of `(master_seed, fold_id, df, isolate_ids,
  cooccur_pairs, schema_pair)` and removes cross-fold / cross-bundle state
  pollution.
- Caveat: the minimal fix does **not** make `(aid_a, aid_b)` sets identical across
  pair bundles. That would require decoupling isolate-pair sampling from
  function-availability rejection — a larger change (see "Optional: cross-bundle
  identity" below).

## Seeding path (code inspection)

**Global seed setup** — `src/datasets/dataset_segment_pairs.py:1548,1600`:

- `RANDOM_SEED = resolve_process_seed(config, 'datasets')` resolves to
  `master_seed` (42 in all 28 paper bundles; process-specific seeds are null).
- `set_deterministic_seeds(RANDOM_SEED)` is called **once** at script startup.
  This seeds `random`, `numpy.random`, and `torch` globally. No further reseeding
  occurs anywhere in the pair-generation path.

**Per-fold isolate split** — `generate_all_cv_folds` (line 1352):

- `unique_isolates = np.array(sorted(df['assembly_id'].unique()))` — sorted for
  determinism, and identical across bundles because all 28 children inherit the
  same `virus.selected_functions` (all 8 proteins) from the master bundle. Only
  `dataset.schema_pair` differs, and it does not filter `df` at this stage.
- `KFold(n_splits=12, shuffle=True, random_state=seed)` uses its own
  `numpy.random.RandomState`; isolate→fold mapping is a pure function of
  `(seed, n_folds)`.
- `train_test_split(trainval_ids, test_size=val_frac, random_state=seed + fold_i)`
  also uses its own RandomState; train/val assignment is a pure function of
  `(seed, fold_id)`.
- **Isolate partition is deterministic in `(master_seed, fold_id)` and independent
  of `schema_pair`.**

**Negative-pair generation** — `create_negative_pairs` (line 334):

- The docstring (line 373) and a comment (line 382) explicitly state that the
  `seed` argument is ignored: *"Seeding is handled upstream by
  set_deterministic_seeds() in the CLI section."*
- Inside the sampling loop (lines 454, 466–470):
  - `random.sample(isolate_ids, 2)`
  - `random.choice(left_candidates)` / `random.choice(right_candidates)` (schema
    mode)
  - `random.choice(isolate_groups[aid1])` / `random.choice(isolate_groups[aid2])`
    (unordered mode)

  All use the module-level `random` state.
- `generate_all_cv_folds` (line 1420) passes `seed=seed+fold_i` through
  `split_dataset`'s `neg_kwargs` (line 853). The call-site comment reads
  *"fold-specific seed for negative sampling"* — but the value is never applied.

**Positive-pair generation** — `create_positive_pairs` (line 217):

- Purely combinatorial (`itertools.combinations` within each isolate group). No
  random calls. Fully deterministic.

## Empirical verification

Compared `flu_28p_ha_na` vs `flu_28p_pb2_pb1` at `fold_0`,
`master_seed=42`, sweep timestamp `val_unfilt_20260413_151650`:

| Quantity | ha_na | pb2_pb1 | Match? |
|---|---|---|---|
| Reconstructed isolate partition (train/val/test) | 88,632 / 10,853 / 9,045 | 88,632 / 10,853 / 9,045 | ✓ identical |
| Train assembly-ids observed in pair rows | 88,632 (100% of assigned) | 88,632 (100%) | ✓ |
| Val assembly-ids observed | 9,793 (1,060 missing HA or NA) | 9,457 (1,396 missing PB2 or PB1) | — differ by availability |
| Train negative `(aid_a, aid_b)` set overlap | — | — | **85.44%** |
| Val negative `(aid_a, aid_b)` set overlap | — | — | **0.02%** |
| Test negative `(aid_a, aid_b)` set overlap | — | — | **0.03%** |

The isolate partition reconstructed by replaying `KFold` + `train_test_split` with
`master_seed=42` matches the partition observed in both datasets' pair files
exactly. The post-filter isolate sets in val/test are smaller than the assigned
set because isolates lacking one of the two schema functions produce no pairs —
the drop count differs per bundle (1,060 vs 1,396) because different isolates
lack HA vs lack PB2, which is the mechanism that eventually desyncs the RNG.

Negative `(aid_a, aid_b)` overlap is high in train (~85%) because train carries
177K negatives over ~88K isolates — the pair space is near-saturated, so both
bundles land on largely the same pairs by coincidence. In val/test (~10K and ~8K
negatives respectively) overlap collapses to essentially random-coincidence level
(~0.02%).

## Why the negative pairs diverge

Two compounding sources, both in `create_negative_pairs`:

- **(A) No local RNG.** At the moment fold-0's negative-sampling loop begins, the
  module-level Python `random` state happens to be identical across bundles
  because only `numpy.random` RNGs fire between `set_deterministic_seeds` and this
  point. So the first call to `random.sample(isolate_ids, 2)` returns the same
  pair across bundles.
- **(B) Rejection patterns are schema-specific.** In `schema_ordered` mode, the
  loop checks `isolate_func_groups[aid1][func_left]` and
  `isolate_func_groups[aid2][func_right]`. If either is empty, the loop hits
  `continue` (`missing_left_func` / `missing_right_func`, lines 459–464) **after**
  one `random.sample` draw but **before** any `random.choice`. The set of isolates
  lacking HA differs from those lacking PB2, so the count of rejected-before-choice
  iterations differs per bundle. This permanently desyncs the Python-random stream
  from iteration 2 onward. Desync compounds across folds 1–11 because the script
  never reseeds.

The same mechanism applies to `duplicate_brc` / `duplicate_seq` /
`blocked_cooccur` rejections — they consume random draws before deciding to skip,
and `cooccur_pairs` is identical across bundles so those classes contribute less,
but the `missing_*_func` class alone is enough to desync.

## Proposed fix (minimal)

Localize the RNG inside `create_negative_pairs`. This honors the `seed` argument
that `generate_all_cv_folds` already computes as `master_seed + fold_i`, removes
cross-fold state pollution, and removes dependence on what Python-random consumers
ran earlier in the process.

```diff
--- a/src/datasets/dataset_segment_pairs.py
+++ b/src/datasets/dataset_segment_pairs.py
@@ -379,8 +379,13 @@ def create_negative_pairs(
         - Number of same-function negative pairs
         - Dict with rejection statistics
     """
-    # Seeding is handled upstream by set_deterministic_seeds() in the CLI section.
+    # Per-fold deterministic RNG. Isolates this function from the process-global
+    # random stream so (a) state from prior folds / prior bundles doesn't leak in,
+    # and (b) the `seed` argument passed by generate_all_cv_folds (seed+fold_i)
+    # actually takes effect. Negative sampling is now a pure function of
+    # (master_seed, fold_id, df, isolate_ids, cooccur_pairs, schema_pair).
+    rng = random.Random(seed)
     neg_pairs = []
     seen_pairs = set()  # Track unique pairs by brc_fea_id
@@ -451,15 +456,15 @@ def create_negative_pairs(
     while len(neg_pairs) < num_negatives and attempts < max_attempts:
         attempts += 1
-        aid1, aid2 = random.sample(isolate_ids, 2)
+        aid1, aid2 = rng.sample(isolate_ids, 2)
         if pair_mode == "schema_ordered":
             left_candidates = isolate_func_groups[aid1][func_left] if isolate_func_groups is not None else []
             if not left_candidates:
                 rejection_stats['missing_left_func'] += 1
                 continue
             right_candidates = isolate_func_groups[aid2][func_right] if isolate_func_groups is not None else []
             if not right_candidates:
                 rejection_stats['missing_right_func'] += 1
                 continue
-            row_a = random.choice(left_candidates)
-            row_b = random.choice(right_candidates)
+            row_a = rng.choice(left_candidates)
+            row_b = rng.choice(right_candidates)
         else:
-            row_a = random.choice(isolate_groups[aid1])
-            row_b = random.choice(isolate_groups[aid2])
+            row_a = rng.choice(isolate_groups[aid1])
+            row_b = rng.choice(isolate_groups[aid2])
```

Also update the docstring (line 373) — the claim that `seed` is unused becomes
false.

### Properties after the fix

- Within a single bundle: negatives in fold_k depend only on
  `(master_seed, fold_id, schema_pair, df, cooccur_pairs)`. Rerunning one fold in
  isolation produces the same negatives as that fold produced in a full
  12-fold sweep.
- No cross-fold state leakage: fold_k's RNG is independent of how many draws
  fold_{k-1} consumed.
- No cross-bundle state leakage: prior Python-random consumption outside this
  function cannot shift the stream.

### What the fix does not achieve

Realized `(aid_a, aid_b)` negative-pair sets will **still differ across pair
bundles** at the same fold_id, because of source (B): `missing_left_func` /
`missing_right_func` rejection fires after `rng.sample(...)` but before any
`rng.choice(...)`, and the set of isolates lacking `func_left` depends on
`schema_pair`. So even with identical initial RNG state, iteration 2 onward
diverges between, say, ha_na and pb2_pb1.

## Optional: cross-bundle isolate-pair identity (larger change)

If we want the `(aid_a, aid_b)` negative set to be identical across all 28 pair
bundles at the same fold_id (not just reproducible within each), we need to
decouple isolate-pair sampling from function-availability rejection. Sketch:

1. Pre-compute a fold-keyed pool of `(aid_a, aid_b)` tuples using only
   `isolate_ids` — schema-agnostic, seeded by `(master_seed, fold_id)`.
2. For each tuple, map to `(protein_a, protein_b)` using a second RNG; if the
   tuple lacks the required `schema_pair` functions, reject and draw a
   replacement isolate pair from the schema-agnostic pool (not from a fresh
   `rng.sample` call).

This is a structural change with several secondary decisions: how to handle
replacement draws, whether the 1:1 mapping survives when the same pair needs a
second-choice replacement in one bundle but not another, and how to keep the
`duplicate_brc` / `duplicate_seq` / `blocked_cooccur` rejections from re-introducing
schema-specific desync. Probably warrants its own plan if pursued.

My recommendation: do the minimal fix first (it's a clear bug — plumbed `seed`
argument that's silently discarded) and only pursue the cross-bundle identity
change if a downstream analysis actually needs it.

## Implementation steps

1. Apply the diff above to `src/datasets/dataset_segment_pairs.py`.
2. Update the `create_negative_pairs` docstring (`seed: Random seed ...`,
   line 373) to say the seed is now used to derive a local RNG.
3. Run a small regression check: generate one small bundle
   (e.g., `flu_pb2_ha_na_5ks`) pre- and post-fix with the same `master_seed`. Pre-fix
   and post-fix negatives will differ (expected — the whole point is to change the
   RNG plumbing). Verify: running post-fix *twice* produces identical negatives
   per fold, and running only fold_k in isolation matches the fold_k negatives
   from the full CV sweep.
4. Document in `.claude/memory.md` and `docs/SEED_SYSTEM.md` that negative-pair
   RNG is now fold-scoped.

## Risk / blast radius

- All previously generated datasets remain bit-identical; the fix only affects
  datasets generated after merge.
- Downstream training runs use whatever dataset they were generated against —
  no retraining is forced. A dataset re-generated post-fix will produce different
  negatives, so any training run that depends on a specific cached dataset must
  either pin the dataset directory or accept the reroll.
- The paper's Task 11 sweep (`allpairs_prod_*`) used pre-fix datasets. Results
  stand; the fix is a correctness improvement for future runs, not a retroactive
  invalidation.

## Open questions

- Should positives also get a local RNG for symmetry, even though they're
  currently deterministic (no random calls)? Probably yes as a matter of hygiene,
  so future edits to `create_positive_pairs` can't accidentally couple to global
  state.
- Should we also re-seed inside `generate_all_cv_folds` at each fold iteration
  (belt-and-suspenders), so the local RNG in `create_negative_pairs` is the only
  Python-random consumer per fold? Probably overkill given the local-RNG fix, but
  worth noting.
