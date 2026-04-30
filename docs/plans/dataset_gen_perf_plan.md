# Stage 3 `split_dataset` performance plan

**Status: IN PROGRESS**

## Context

Profiling run: `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260427_164925/stage3_dataset.log`
(bundle `flu_ha_na`, full Flu A, schema_ordered HA/NA, single split, 1.79M protein rows
→ 868K after function filter → 173,648 train / 15,062 val / 15,066 test pairs).

Total Stage 3 runtime: ~14:40. `split_dataset` alone consumes **674s (~76% of total)**.

CPU contention can balloon the whole run to 55+ min, but on a quiet host the breakdown
below is the steady-state cost.

## Where the time goes

### `split_dataset` — 674s

| Step | Time | Output |
|---|---|---|
| `build_cooccurrence_set` | 12s | 1.3M cooccurring pairs |
| `compute_isolate_pair_counts` | 21s | 108,530 isolates |
| stratified isolate split | 0.05s | — |
| **`hard_partition_isolates`** | **138s** | — |
| **`create_positive_pairs` train** | **255s** | 86,824 pairs |
| `create_positive_pairs` val | 32s | 10,853 pairs |
| `create_positive_pairs` test | 32s | 10,853 pairs |
| **`create_negative_pairs` train** | **145s** | 86,824 pairs (93,168 attempts) |
| `create_negative_pairs` val | 16s | 10,853 pairs |
| `create_negative_pairs` test | 22s | 10,853 pairs |
| pair_key validation | 0.06s | — |

### Other notable costs (outside `split_dataset`)

| Step | Time | Notes |
|---|---|---|
| `attach_dna_to_prot_df` | 17s | md5 over 1.79M rows; quiet host |
| `save_split_output` total | 133s | dominated by visualization |
| └─ `visualize_dataset_stats` | 104s | k-mer PCA loads full 4096-dim matrix |

## Suggested fixes (priority order)

### 1. `hard_partition_isolates` — drop list-based `not in` (≈ 138s → <1s)

The current block does O(N²) Python `in` over lists at lines 1049–1051 of
`src/datasets/dataset_segment_pairs.py`:

```python
val_isolates  = [aid for aid in val_isolates if (aid not in train_isolates)]
test_isolates = [aid for aid in test_isolates if (aid not in train_isolates) and (aid not in val_isolates)]
unassigned    = [aid for aid in unique_isolates if ... triple `not in list` ...]
```

With ~87K train, ~11K val, ~11K test, that's billions of Python comparisons.

**Fix:** build a set once per partition and reuse it. Use `*_iso_set` names to
avoid colliding with the existing `train_set`/`val_set`/`test_set` defined later
in the function for pair-level overlap diagnostics:

```python
train_iso_set = set(train_isolates)
val_isolates  = [aid for aid in val_isolates if aid not in train_iso_set]
val_iso_set   = set(val_isolates)
test_isolates = [aid for aid in test_isolates if aid not in train_iso_set and aid not in val_iso_set]
test_iso_set  = set(test_isolates)
unassigned    = [aid for aid in unique_isolates
                 if aid not in train_iso_set
                 and aid not in val_iso_set
                 and aid not in test_iso_set]
```

Same logic, O(N). Trivial, no behavior change.

**Status: APPLIED** (2026-04-29).

### 2. `create_positive_pairs` — vectorize per-isolate pair construction (≈ 255s + 32s + 32s → seconds)

Schema-ordered HA/NA produces at most one positive pair per isolate, yet the train call
takes 255s for 87K pairs (~3ms/pair). The cost is per-isolate Python iteration over
`df[df['assembly_id'].isin(...)]` rather than the actual pair construction.

**Investigate:** read `create_positive_pairs` body. Likely candidates for vectorization:
- single `groupby('assembly_id')` over the filtered df instead of per-isolate masking.
- in `schema_ordered` mode: filter to func_left rows + func_right rows, merge on
  `assembly_id`, then concatenate the two slot columns.

Expected: collapses to a couple of pandas joins. Will scale to multi-pair schemas
(more `selected_functions`) without per-isolate Python loops.

### 3. `create_negative_pairs` — vectorize rejection sampling (≈ 145s + 16s + 22s → seconds)

93,168 attempts producing 86,824 accepts is only a 7% rejection rate, so the cost is
**per-attempt Python overhead**, not the cooccur-set lookups themselves.

**Investigate:** read `create_negative_pairs` body. Likely candidates:
- sample two batches of `assembly_id`s with `np.random.choice` once per call.
- pull `func_left` rows and `func_right` rows by single boolean mask.
- mask cooccurring pairs with a vectorized `np.isin` against the cooccur set
  (or `cooccur_pairs` cast to a `frozenset` of canonical-keyed tuples for O(1) lookup).
- top up with a small loop only if the first vector batch under-delivers.

### 4. `visualize_dataset_stats` — 104s; flip to opt-in or cache (lower priority)

K-mer PCA loads the full 4096-dim k-mer matrix every Stage 3 run regardless of need.
For routine reruns, `dataset.skip_kmer_pca_plots: true` shaves ~90s. Reserve full
visualizations for runs where the plots are actually consumed.

## Notes / non-issues

- `attach_dna_to_prot_df` (17s, quiet host) and `enrich_prot_data_with_metadata` (5s)
  are not bottlenecks under normal conditions. The earlier 224s and 54s for these
  on the 55-min run reflect host CPU contention, not code paths.
- `build_cooccurrence_set` (12s for 1.3M pairs) and `compute_isolate_pair_counts`
  (21s for 108K isolates) are reasonable; not worth touching before 1–3.

## Out of scope / deferred

- Parallelizing `create_positive_pairs` / `create_negative_pairs` across train/val/test
  (only worth doing once each is vectorized).
- Replacing the schema-ordered branch with multi-function combinatorial pair generation
  — keep as-is; current usage is HA/NA only.
- Tuning `compute_isolate_pair_counts` — 21s is acceptable; revisit only if
  multi-pair schemas push it higher.

## Verification plan

After each fix:
1. Re-run `./scripts/stage3_dataset.sh flu_ha_na`.
2. Confirm `[diag] split_dataset:` line for the touched step drops to expected range.
3. Diff resulting `train_pairs.csv` / `val_pairs.csv` / `test_pairs.csv` against the
   pre-fix run (same seed). Should be byte-identical (or row-set-identical for
   sampling-based steps with the same RNG).
4. Diff `dataset_stats.json` — must match.
