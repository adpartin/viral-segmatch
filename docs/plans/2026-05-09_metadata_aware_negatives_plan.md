# Metadata-aware negative sampling — Plan

**Status: IN PROGRESS**
**Date:** 2026-05-09 (plan); 2026-05-10 (implementation start, branch
`feature/metadata-aware-negative-sampling`)

## Context

The 2026-05-07 metadata-shortcut writeup
(`docs/results/2026-05-07_metadata_shortcut_negatives.md`) quantified mode #5
leakage on HA/NA mixed: false-positive rate climbs 30–50× across `match_count`
buckets, and capacity *amplifies* the shortcut (h=[200] is more committed to
using metadata as a label proxy than h=[10]). The shortcut lives in the
dataset's emergent metadata-match distribution: random negative sampling over a
population skewed toward 2024-Human-H3N2 makes "all metadata aligned" negatives
rare, so the model never has to learn to discriminate them.

This plan adds **deterministic, metadata-aware negative sampling**: at
construction time, target a configurable mix of metadata-match regimes (e.g.,
30% "all axes match" hard negatives, 20% "no axes match" easy negatives) so the
test-set metadata footprint is calibrated rather than emergent.

The v2 builder already anticipates this. `create_negative_pairs_v2`,
`split_dataset_v2`, and `generate_all_cv_folds_v2` all carry an
`axis_quotas: Optional[dict] = None` parameter that currently raises
`NotImplementedError`. This plan implements that slot.

## Glossary

Plain definitions, matching the terminology used elsewhere in this doc.

| Term | Meaning |
|---|---|
| **Pair** | A `(slot_a, slot_b)` row carrying two protein sequences, one per `schema_pair` function. |
| **Positive pair** | A pair drawn from one isolate (`assembly_id_a == assembly_id_b`). Label = 1. |
| **Negative pair** | A pair drawn from two different isolates (`assembly_id_a != assembly_id_b`). Label = 0. |
| **Candidate (negative)** | A hypothetical `(isolate_i, isolate_j)` with `i != j` that passes basic constraints (not in `cooccur_pairs`, schema-compatible, not seq-excluded). The "candidate universe" is all such (i, j) before regime stratification. |
| **Sampling axes** | The metadata fields the *sampler* uses to define regimes. Default: `{host, hn_subtype, year}`. |
| **Annotation axes** | The metadata fields attached to every pair as `<axis>_a`, `<axis>_b`, `same_<axis>` columns for downstream eval. Default: `{host, hn_subtype, year, geo_location, passage}`. Superset of sampling axes. |
| **`year_match` policy** | Either `exact` (equality on integer year) or `binned` (equality on `year_bin` derived from `YEAR_BIN_EDGES`). Default: `binned`. |
| **Metadata cell** | A single tuple `(host_value, hn_subtype_value, year_or_year_bin)`. Each isolate maps to exactly one cell. There are typically O(100) distinct cells across Flu A. |
| **Regime** | One of nine mutually-exclusive labels assigned to a candidate based on which sampling-axis values match between its two isolates. See "Regimes" below. |
| **Available count** | For a regime R, the closed-form number of candidates with regime R, computed by combinatorics over metadata cells. Reported in the manifest. |
| **Target count** | `round(num_negatives × target_fraction[R])`. What the sampler aims to draw for regime R. |
| **Achieved count** | What the sampler actually produced for regime R after coverage and exclusions. Reported alongside target. |
| **Shortfall** | Positive integer `target_count[R] - achieved_count[R]`. Reported with a reason. |
| **`neg_to_pos_ratio`** | Existing config knob. Sets `num_negatives = round(neg_to_pos_ratio × |pos|)` per split. Controls the SIZE of the negative budget; orthogonal to regime quotas (which control the MIX). |
| **Coverage phase** | Existing v2 first phase. Walks every (slot, dna_hash) target and produces ≥1 negative pair per target. Regime-blind by design. |
| **Fill phase** | Existing v2 second phase. Tops up to `num_negatives`. **Becomes regime-aware** in this plan. |
| **Coverage floor** | `max(|unique_slot_A_dnas|, |unique_slot_B_dnas|)`. The minimum number of negatives the coverage phase will produce regardless of `num_negatives`. |
| **Coverage override** | Condition where `coverage_floor > num_negatives`. Coverage wins; fill phase is a no-op. Logged via `coverage_overrode_ratio`. |
| **Manifest** | A JSON/CSV artifact summarizing per-regime target/available/achieved/shortfall, written next to existing `dataset_stats.json`. |
| **Gold-standard metadata lookup** | `assembly_id → metadata` via `isolate_metadata.csv`. The correct lookup for pair-level metadata. (Contrast: the seq_hash lookup currently used in `compute_axis_flags`, which is wrong; see "Pre-requisite bug fixes".) |

## Regimes

Nine mutually-exclusive, collectively-exhaustive labels for negative pairs over
the three sampling axes `{host, hn_subtype, year}` (with `year` matched per the
`year_match` policy). For each candidate, classify by comparing isolate_i and
isolate_j on each axis.

| Regime | Match condition | Hardness |
|---|---|---|
| `none_match` | host differs AND subtype differs AND year differs | easiest — every metadata shortcut available |
| `host_only` | host equal; subtype differs; year differs | one shortcut |
| `subtype_only` | subtype equal; host differs; year differs | one shortcut |
| `year_only` | year equal; host differs; subtype differs | one shortcut |
| `host_subtype_only` | host AND subtype equal; year differs | two shortcuts |
| `host_year_only` | host AND year equal; subtype differs | two shortcuts |
| `subtype_year_only` | subtype AND year equal; host differs | two shortcuts |
| `host_subtype_year` | all three match | hardest — no metadata shortcut, model must use sequence content |
| `unknown_metadata_neg` | at least one sampling-axis value is null/unparseable on either side | catch-all |

Concrete examples (HA/NA + binned year):

- `(Pig, H1N1, 2018)` × `(Human, H3N2, 2024)` → `none_match`
- `(Human, H1N1, 2024)` × `(Human, H3N2, 2018)` → `host_only`
- `(Pig, H3N2, 2024)` × `(Pig, H3N2, 2024)` → `host_subtype_year` (hardest negative)
- `(NaN, H3N2, 2024)` × `(Human, H3N2, 2024)` → `unknown_metadata_neg`

## Pre-requisite bug fixes (BLOCKING)

These must land before metadata-aware sampling does. Otherwise, regime
assignment derives from wrong inputs.

### Bug 1: `compute_axis_flags` keys on `seq_hash` instead of `assembly_id`

`src/datasets/dataset_segment_pairs_v2.py:compute_axis_flags` builds
`<axis>_a`, `<axis>_b`, `same_<axis>` via `df.groupby('seq_hash')[axis].first()`.
The same protein appearing across isolates with different metadata
(cross-host transmission, multi-year strains, multi-subtype reassortment)
collapses to one arbitrary representative. The function's own comments
(lines 798–824) document the right answer ("look up by assembly_id_a/_b on the
pair") but the code never adopted it.

Empirical impact (probe `/tmp/probe_metadata_lookup.py`, run 2026-05-09):

| `same_<axis>` disagreement vs gold (HA/NA) | All pairs | Negatives | Positives |
|---|---|---|---|
| `same_host` | 4.6% | 1.0% | 9.1% |
| `same_year` | 8.2% | 1.8% | 16.5% |
| `same_hn_subtype` | 1.5% | 0.4% | 3.0% |

| `same_<axis>` disagreement vs gold (PB2/PB1) | All pairs | Negatives | Positives |
|---|---|---|---|
| `same_host` | 7.3% | 1.5% | 14.7% |
| `same_year` | 16.5% | 3.3% | 33.6% |
| `same_hn_subtype` | 5.6% | 1.5% | 10.9% |

The positive-side numbers are large because positives have
`assembly_id_a == assembly_id_b`, so the gold-standard `same_<axis>` is always
True; the seq_hash lookup can collapse to a different isolate's metadata and
falsely return False. **One in three positive pairs in PB2/PB1 has a wrong
`same_year` flag.**

`dna_hash` lookup is dramatically better (PB2/PB1 `same_year` disagreement
drops from 16.5% to 0.8%, a 21× improvement) but still imperfect. The truly
correct lookup is `assembly_id` — pairs already carry `assembly_id_a/_b` and
every Stage-3 dataset already saves `isolate_metadata.csv`.

**Fix.** Replace seq_hash-based lookup in `compute_axis_flags` with
`assembly_id`-based lookup against `isolate_metadata.csv`. One line per axis.

**Backfill.** Either (a) regenerate `<axis>_a/_b/same_<axis>` columns on
existing pair CSVs from saved `isolate_metadata.csv`, or (b) document existing
datasets as known-stale and fix forward. Default: (b) + footnote on the
2026-05-07 result, revisit if a downstream analysis is materially affected by
the negative-side error rate (1–3% — still nonzero).

### Bug 2: `compute_metadata_coverage` per-seq null undercount

`compute_metadata_coverage` computes `per_seq = df.groupby('seq_hash')[col].first()`
then counts `per_seq.isna().sum()`. If the first row for a seq_hash is null and
a later row has a value, the seq is counted as null in the per-seq view even
though a non-null value exists in the data.

**Fix.** Drop nulls before groupby:
`df.dropna(subset=[col]).groupby('seq_hash')[col].first()`, then recompute null
count as `n_unique_seqs_total - len(per_seq)`. Or rephrase the field as
"n_unique_seqs with any non-null value seen".

### Bug 3 (cosmetic): per-seq view documentation

Per-seq views in both functions are misleading for internal-protein bundles
where one seq_hash legitimately spans multiple isolates with different
metadata. Update docstrings and `metadata_coverage.json` field names to make
clear that per-seq is "first-seen representative", not "every isolate carrying
this seq has this value". The right substrate for sampling decisions is
per-isolate (`assembly_id`), not per-seq.

## Design decisions

Resolved in 2026-05-09 conversation; do not re-litigate.

1. **Three sampling axes, five annotation axes.** Sampling regimes use
   `{host, hn_subtype, year}` (8 + 1-unknown = 9 regimes). Annotation columns
   continue to cover `{host, hn_subtype, year, geo_location, passage}`.
   Rationale: 5-axis sampling blows up to 32 regimes, mostly empty;
   geo_location / passage are higher-cardinality and lower-quality, fine for
   diagnostics, bad for sampling targets.
2. **Year match: binned by default.** Use the existing
   `YEAR_BIN_EDGES = [<=2015, 2016-2020, 2021+]` (already in
   `analyze_stage4_train.py`). Make `year_match: exact | binned` a knob;
   default `binned`.
3. **Closed-form candidate counts via metadata-cell groupby.** Do NOT enumerate
   the O(N²) candidate space. Compute `n_isolates_per_cell`, then count per
   regime via combinatorics over (cell_i, cell_j) pairs. Cost: O(K²) where
   K = distinct cells (~100s). Runs in milliseconds.
4. **Targets as fractions of `num_negatives` summing to 1.0.** Per-regime
   ceilings emerge naturally: each regime's
   `target = round(num_negatives × target_fraction[r])`. No separate ceiling knob.
5. **`on_shortfall: redistribute` is the default.** When a regime can't be
   filled, its remaining budget reapportions pro-rata to non-saturated regimes.
   Alternatives `warn_only` (leave under-budget) and `error` (fail loudly) are
   knobs.
6. **Coverage > quotas.** Coverage is a hard correctness invariant
   (mode #2 leakage protection); regime mix is a soft target (mode #5
   calibration). Coverage phase stays regime-blind. Fill phase becomes
   regime-aware.
7. **`neg_to_pos_ratio` keeps current semantics.** It controls the SIZE of the
   negative budget. Quotas control the MIX. Orthogonal.
8. **Implement under the existing `axis_quotas` slot.** Don't introduce a
   parallel name. The placeholder is already plumbed through
   `create_negative_pairs_v2`, `split_dataset_v2`, `generate_all_cv_folds_v2`.
   Replace the `NotImplementedError` body.

## Configuration schema

```yaml
# conf/dataset/default.yaml — new section under axes_for_flags
negative_sampling:
  # Sampling axes used to define mutually-exclusive metadata-match regimes.
  # Subset of axes_for_flags. Defaults to the 3-axis taxonomy.
  axes: [host, hn_subtype, year]

  # year matching policy.
  year_match: binned                # exact | binned
  year_bin_edges:                   # only used when year_match=binned
    - [-inf, 2015, "<=2015"]
    - [2016, 2020, "2016-2020"]
    - [2021,  inf, "2021+"]

  # Target distribution. Must sum to 1.0 (validated). All 9 regime keys must be
  # present (validation aids reproducibility -- silent omissions were the bug
  # cluster the 2026-05-07 writeup is about).
  regime_targets:
    none_match:           0.20
    host_only:            0.10
    subtype_only:         0.10
    year_only:            0.10
    host_subtype_only:    0.10
    host_year_only:       0.05
    subtype_year_only:    0.05
    host_subtype_year:    0.30      # the hardest-negative regime
    unknown_metadata_neg: 0.00      # 0 by default; non-zero allowed for stress tests

  # Behavior when a regime's available_count < target_count.
  on_shortfall: redistribute        # redistribute | warn_only | error
```

When `negative_sampling: null` (default for backward-compat), the sampler runs
in legacy mode (regime-blind fill phase, current behavior). Setting any value
enables the new path; the bundle must provide a complete `regime_targets` dict
(no defaults inserted).

## Sampler logic

Per split:

```
1. Build pos_df (existing).
2. Build per-isolate metadata cells via assembly_id lookup against
   isolate_metadata.csv:
       cell(iso) = (host, hn_subtype, year_or_bin)
   Cells where any axis is null map to a special UNKNOWN_CELL.
3. Compute available_count[regime] for every regime via closed-form
   combinatorics over (cell_i, cell_j). Cost O(K^2).
4. Compute target_count[regime] = round(num_negatives x target_fraction[r]).
   If sum(target_count) != num_negatives due to rounding, adjust the largest
   target by +/-1 deterministically (sorted by regime name).
5. Coverage phase (existing, regime-blind). Walk (slot, dna_hash) targets;
   for each, draw partners from sorted isolate_ids_list. Each accepted neg is
   tagged with its regime. coverage_placed[r]++ per accepted pair.
6. Fill phase (regime-aware). For each regime:
       residual[r] = max(target_count[r] - coverage_placed[r], 0)
   Build a per-regime candidate iterator (deterministic): for each compatible
   cell-pair in the regime, iterate isolate-pair products in sorted order;
   attempt _try_accept until residual is filled or supply exhausted.
7. Shortfall handling. Sum unfilled residuals; redistribute pro-rata of
   original target_fraction over non-saturated regimes. Repeat until either
   total = num_negatives, all regimes saturated, or fixed-point reached.
8. Manifest. Per regime, write target / available / coverage_placed /
   fill_placed / achieved / shortfall_reason.
```

**Determinism.** Every iteration order is `sorted(...)` with stable tie-breaks.
Every sampling step uses `random.Random(seed).shuffle(...)` on a sorted list
rather than `set` iteration. Re-running on the same `(seed, df, config)`
produces byte-identical pair CSVs.

**Closed-form counting (step 3 detail).** With sampling axes
A = (host, hn_subtype, year_match) and `cells = pos_df.groupby([h, s, y]).size()`:

```
N(host_subtype_year)    = sum_c n_c * (n_c - 1)
N(host_subtype_only)    = sum over c1, c2 with c1.h==c2.h, c1.s==c2.s, c1.y!=c2.y of n1*n2
N(host_year_only)       = (analogous, swap roles of s and y)
N(subtype_year_only)    = (analogous)
N(host_only)            = sum over c1, c2 with c1.h==c2.h, c1.s!=c2.s, c1.y!=c2.y of n1*n2
N(subtype_only)         = (analogous)
N(year_only)            = (analogous)
N(none_match)           = sum over c1, c2 with all three differ of n1*n2
N(unknown_metadata_neg) = sum involving any UNKNOWN_CELL on either side
```

Cooccur and seq-exclusion rejections are uniform-in-expectation across regimes,
so the closed-form over-estimates by a known factor. Acceptable for
feasibility checking and manifest reporting.

## Manifest and provenance

Per dataset, written to `<dataset_dir>/negative_regime_manifest.json`:

```json
{
  "config": {
    "axes": ["host", "hn_subtype", "year"],
    "year_match": "binned",
    "year_bin_edges": [[-Infinity, 2015, "<=2015"], [2016, 2020, "2016-2020"], [2021, Infinity, "2021+"]],
    "regime_targets": {"...": "..."},
    "on_shortfall": "redistribute"
  },
  "splits": {
    "train": {
      "num_negatives_requested": 47060,
      "num_negatives_achieved": 47060,
      "regimes": [
        {"regime": "none_match", "target": 9412, "available": 38291,
         "coverage_placed": 1247, "fill_placed": 8165, "achieved": 9412,
         "shortfall_reason": null},
        {"regime": "host_subtype_year", "target": 14118, "available": 4502,
         "coverage_placed": 392, "fill_placed": 4110, "achieved": 4502,
         "shortfall_reason": "supply_exhausted; redistributed_to=[none_match]"}
      ]
    },
    "val":  {"...": "..."},
    "test": {"...": "..."}
  }
}
```

Companion CSV `negative_regime_manifest.csv` (one row per split × regime) for
human reading.

The pair CSVs gain two columns:

- `neg_regime`: one of the 9 regime names; `pd.NA` for positives.
- `metadata_match_count`: integer 0..3 (count of sampling-axis matches);
  `pd.NA` for positives.

`<axis>_a / <axis>_b / same_<axis>` columns continue to cover all 5 annotation
axes (after Bug 1 fix).

## Edge cases and robustness

- **Single-cell bundle** (e.g., `flu_ha_na_human_h3n2_2024`): only
  `host_subtype_year` is feasible. All other regime targets get redistributed
  there. The dataset is valid; it's just that all negatives are the hardest
  type. No crash.
- **All-unknown bundle**: every isolate has at least one null axis, so every
  candidate is `unknown_metadata_neg`. Sampler honors that. The dataset is
  usable with the caveat that no metadata stratification is possible.
- **Coverage-overrides-num_negatives**: existing `coverage_overrode_ratio`
  semantics preserved. Fill phase is a no-op; achieved counts will diverge
  from targets; manifest records the gap.
- **Cross-split feasibility**: negatives are sampled per-split with
  `forbidden_pair_keys` threading. If train consumes most `none_match`
  candidates, val/test see reduced supply. The closed-form `available_count`
  is recomputed per split AFTER subtracting forbidden pair keys, so the
  manifest is honest about per-split feasibility.
- **Empty regime in target dict**: explicitly setting a regime to 0.0 means
  "don't sample this regime". Validation rejects negative or NaN values.
- **Target sum != 1.0**: hard error at config validation. No silent
  normalization.

## Test plan

Unit tests in `tests/datasets/test_negative_regime_sampling.py`:

1. **Regime assignment for all 8 axis-match combinations.** Construct a
   16-isolate synthetic df where every (host, subtype, year_bin) cell is
   populated. Sample with uniform target = 1/9. Assert every regime has at
   least one negative; assert the `neg_regime` column matches independent
   recomputation.
2. **Unknown metadata handling.** Force one axis null on a subset of isolates.
   Assert those candidates land in `unknown_metadata_neg` regardless of other
   axes.
3. **Determinism.** Run twice with same seed; assert byte-identical pair CSVs.
4. **`year_match=exact` vs `binned`.** Same data, two policies. Assert
   different regime distributions; assert binned mode collapses 2023-2024 into
   the same year_bin.
5. **Single-cell bundle.** Filter df to one (host, subtype, year_bin) cell.
   Assert all candidates land in `host_subtype_year`; assert no crash.
6. **Empty bundle / all-unknown bundle.** Assert graceful behavior, manifest
   reports zero candidates for impossible regimes, no crash.
7. **Coverage > quotas.** Construct a tight bundle where coverage forces a
   regime over-target. Assert achieved[regime_X] > target[regime_X]; assert
   manifest reports the override.
8. **Cross-split disjointness preserved.** Run full split; assert pair_keys
   disjoint across train/val/test (existing v2 invariant holds under the new
   sampler).
9. **`on_shortfall=error` triggers.** Set a target above the closed-form
   available. Assert raises with regime name and counts.
10. **Validation errors.** Targets summing to 0.99 or 1.01 raise. Negative or
    NaN target raises. Missing regime key in target dict raises.
11. **Bug 1 regression test.** Construct synthetic df with one seq_hash mapped
    to two isolates with different host. Assert `compute_axis_flags`
    (post-fix) returns the correct host per side based on assembly_id.
12. **Bug 2 regression test.** Construct df where seq_hash X has
    `[NaN, "Human"]` across two rows. Assert `compute_metadata_coverage`'s
    per-seq null count is 0 (post-fix).

Integration tests:

- Build a small `flu_ha_na` dataset with the new path and the legacy path;
  assert all hard invariants hold (no pair_key overlap, every seq_hash
  covered, etc.).
- Run `analyze_negative_hardness` on the new dataset; assert match_count
  distribution matches the manifest.

## Future extensions: per-split regime control

Optional. Considered but deferred. Scenarios where test (and possibly val)
evaluates a *specific* metadata-match regime while train remains
unrestricted. The core sampler in this plan does not yet support per-split
regime targets; this section documents four candidate scenarios so we can
decide which (if any) to add later. Of these, **(a) is the immediate target
use case** — held-out-cell OOD evaluation, the strongest 'biology learning'
claim available in this dataset.

| Scenario | What it tests | What it implies for train | Supported by current plan? |
|---|---|---|---|
| (a) Held-out-cell OOD test | "Does the model generalize to a population (e.g., Human-H3N2-2021+) it never saw during training?" The strongest 'biology learning' claim — performance on isolates from a population train was deliberately blocked from seeing. | Train contains zero isolates from the test cell, by isolate-level filter. | **No.** Needs `test_filter` / `val_filter` config + isolate-level partitioning in `split_dataset_v2`. |
| (b) Stratified eval on mixed test | "Does the model fail on hard pairs drawn from the same population it trained on, when no metadata shortcut is available?" Within-distribution stress test on the hardest regime. | Train is normally-mixed. No constraint. | **Yes, no plan changes.** Existing `analyze_negative_hardness` and `analyze_predictions_stratified.py` already compute per-stratum metrics post-hoc on a normally-built test. |
| (c) Min-count hard-regime guarantee | "Same scientific question as (b), but guarantees enough statistical power on the hard regime when normal random sampling leaves too few examples to draw a conclusion." | Train is normally-mixed. Test gets a minimum count of `host_subtype_year` pairs. | **Partially.** Needs per-split `regime_targets` overrides and a `min_count` semantics (absolute counts in addition to fractions). |
| (d) Hard-regime test with train-test isolate overlap | "Within-distribution evaluation of the hardest regime on the *same* isolates train saw." Requires isolate overlap to keep population identical. | Train and test share isolates. Mode #1 / #3 / #4 leakage by construction. | **No, and not recommended.** Conflicts with `hard_partition_isolates: true` (v2 hard-coded). Reintroduces leakage. The cleaner answer is (b). |

### Main algorithmic changes per scenario

**(a) Held-out-cell test, filter-based.** Smallest change; recommended path
for the OOD goal.

- Config additions:
  - `dataset.test_filter: {host: "Human", hn_subtype: "H3N2", year_bin: "2021+"}`
    — isolates matching this go to test.
  - `dataset.val_filter: null | {axis: value}` — optional; `null` means val
    draws from the non-test pool.
- `split_dataset_v2`:
  - Before the auto `train_test_split`, partition isolates by filter —
    matching go to test (and optionally val); rest go to train.
  - Validate non-empty pools and a configurable minimum isolate count per
    pool (small pools give noisy metrics).
- `create_negative_pairs_v2`: **no logic change** — operates on whatever
  isolate pool a split has.
- Manifest: per-split isolate-pool composition (cell counts, regime
  distribution within the pool).
- Coverage: works inside each pool unchanged. If the test pool is one
  metadata cell, all test negatives are `host_subtype_year` by
  construction; coverage finds same-cell partners via the existing
  `seq_to_isolates` exclusion.

**(b) Post-hoc stratification on a normally-built test.** Already supported.

- No code changes. Use:
  - `analyze_negative_hardness` (in `src/analysis/analyze_stage4_train.py`)
    for `match_count` / `match_pattern` tables.
  - `analyze_predictions_stratified.py` (committed 2026-05-09) for
    per-regime metrics.
- Reportable metric is `host_subtype_year`-only AUC / FP-rate — already
  produced by the existing outputs.

**(c) Per-split min-count quotas.** Larger change; defer until (b)
demonstrably lacks statistical power.

- Config additions:
  - `negative_sampling.per_split_overrides: {test: {regime_targets:
    {host_subtype_year: {min_count: 1000}, ...}}}`.
  - Allow `target_count[r]` to be a fraction (default), an absolute integer,
    or a `{min: int}` directive (sampler must hit minimum, redistributes
    from other regimes if needed).
- `split_dataset_v2`: look up per-split targets before each
  `create_negative_pairs_v2` call.
- `create_negative_pairs_v2`: in the `axis_quotas` body, resolve fractions
  / absolutes / mins into final per-regime target counts before the fill
  phase.
- Coverage tension: if test pool can't supply the minimum (e.g., 50
  isolates in the cell give `50×49 = 2450` candidate pairs max), manifest
  reports shortfall; `on_shortfall` policy decides redistribute / warn /
  error.

**(d) Same-isolate train+test overlap.** Not recommended; conflicts with
hard invariants.

- Would require dropping `hard_partition_isolates: true`. v2 hard-codes this;
  reintroduces mode #1 / #3 / #4 leakage.
- Cleaner alternative: scenario (b). The same population *is* represented
  in train (different isolates of the same cell), without the literal
  isolate overlap that creates leakage.

### Priority within future work

- (a) lands after the core sampler is working. Small additive change to
  `split_dataset_v2` (~30 lines) + two config knobs; no change to
  `create_negative_pairs_v2`. Recommended for the immediate OOD-eval
  experiment.
- (c) is larger and depends on the per-split-overrides config plumbing.
  Land only if (b) post-hoc analysis lacks the statistical power for a
  real finding.
- (d) is not on the roadmap.

## Out of scope

- Hard-negative mining beyond regime stratification (curriculum learning,
  focal loss, adversarial training, gradient reversal). Tracked in
  `roadmap_v2.md` Task 12.
- Cross-bundle isolate-pair identity. Different problem, different plan
  (`docs/plans/negative_pair_rng_fix_plan.md`).
- Modifying the coverage phase. It stays regime-blind by design.
- Sampling for positives. Positives are deterministic combinatorial pairs
  within each isolate; no sampling decisions to make.

## Open questions

1. **Year bin edges.** Keep `[<=2015, 2016-2020, 2021+]` or revisit? They were
   chosen for the post-hoc analyzer; the sampling use-case might want finer
   or coarser bins. Default: keep current; configurable via
   `year_bin_edges`.
2. **`unknown_metadata_neg` target.** 0 by default rejects all-unknown
   candidates from the sampler; on Flu A this drops ~0.15% of host-null
   candidates silently. Acceptable for clean bundles. Or set to ~0.01 to keep
   visibility? Default: 0; document as an override.
3. **Manifest format.** JSON for downstream tooling, CSV for human reading —
   ship both.
4. **Backfill policy for Bug 1.** Regenerate all existing pair CSVs
   (~6 datasets, expensive) or document existing as known-stale and fix
   forward (cheap, but old result writeups need a footnote)? Default:
   fix-forward + footnote; revisit if a downstream analysis is materially
   affected.

## Dependencies and order of implementation

1. Pre-requisite bug fixes (`compute_axis_flags`, `compute_metadata_coverage`).
   Land first; ship regression tests; backfill or footnote existing artifacts.
2. Closed-form regime-count utility. Pure function, easy to test in isolation.
3. Per-isolate metadata cell builder (wraps `isolate_metadata.csv` lookup).
4. Regime-aware fill phase. Replaces the `NotImplementedError` body in
   `axis_quotas`.
5. Manifest writer.
6. Pair-CSV column additions (`neg_regime`, `metadata_match_count`).
7. Config validation.
8. End-to-end test on a small bundle.
9. Documentation update (`docs/methods/leakage_definitions.md` mode #5 entry
   should reference this plan).

## See also

- `docs/methods/leakage_definitions.md` — mode #5 demographic shortcut.
- `docs/results/2026-05-07_metadata_shortcut_negatives.md` — the trigger result.
- `docs/plans/2026-05-07_leakage_diagnostics_plan.md` — broader leakage taxonomy.
- `docs/plans/design_dataset_gen_v2.md` — v2 builder spec.
- `docs/plans/negative_pair_rng_fix_plan.md` — RNG determinism in v2.
- `src/datasets/dataset_segment_pairs_v2.py` — current `axis_quotas` placeholder.
- `src/datasets/_pair_helpers.py` — shared helpers (`canonical_pair_key`, etc.).
- `src/analysis/analyze_stage4_train.py:analyze_negative_hardness` — post-hoc
  consumer of `match_count`.
- `/tmp/probe_metadata_lookup.py` — empirical probe for Bug 1 / Bug 2;
  promote to repo before implementation.
