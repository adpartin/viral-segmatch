# K-fold variance estimation for single-slot cluster_disjoint

**Status: IN PROGRESS**

## Motivation

Single-slot `cluster_disjoint` (`split_strategy.single_slot ∈ {'a','b'}`)
produces a deterministic single 80/10/10 partition under LPT-greedy
bin-packing. The current builder rejects `n_folds > 1` for any
`cluster_disjoint` mode (`dataset_segment_pairs_v2.py:2643`), so variance
across positive partitions is not estimable. Paper-grade claims about
"MLP F1 = X.XX ± std under single-slot HA-only at id095" need that
variance.

This plan extends Stage 3 to support k-fold cross-validation under
single-slot `cluster_disjoint`, using sklearn `GroupKFold(n_splits=k)`
keyed on the constrained slot's cluster id. The new path produces
variance source **(i) positive-routing variance**, which combined with
existing **(iii) training-seed variance** (via the existing
`stage4_sweep.sh` seed sweep) gives the per-(pair, alphabet, slot,
threshold) variance estimate the paper requires.

Variance source **(ii) negative-resampling variance** is explicitly
out of scope (see § "Out of scope").

## Design decisions (locked)

### D1. Primitive: sklearn `GroupKFold`

Use `sklearn.model_selection.GroupKFold(n_splits=k)` with
`groups = pos_df['cluster_id_{single_slot}']` to partition the
constrained slot's clusters into k non-overlapping test folds. Each
fold's `train` is the union of the other k−1 fold's clusters.

Why GroupKFold and not LPT-greedy-with-atom-shuffle:

- Standard, recognized variance estimator. Reviewer expectation aligns.
- Deterministic given atom order; the variance estimate isn't
  contaminated by routing-seed noise.
- Composes cleanly with `max_feasible_k` (D2 below): k is bounded
  by atom geometry, not by the number of random shuffles we choose.

**Caveat (acknowledged, addressed by D3 below).** sklearn `GroupKFold`
equalizes *group count* per fold, not *pair count*. With skewed atom
sizes, per-fold pair counts drift more than the LPT-greedy single-shot
drift on the same configuration. This is exactly why per-fold
feasibility checks (D3) are non-negotiable — aggregate checks miss it.

**Per-fold target ratios are k-dependent.** Derived from the existing
CV math at `dataset_segment_pairs_v2.py:1895` (`val_frac = val_ratio /
(1 - 1/n_folds)`). Canonical 0.8/0.1/0.1 holds only at k=10; at k=5
the effective per-fold ratios are 0.7/0.1/0.2 (train/val/test). The
per-fold feasibility checks (D3) apply to these k-derived targets,
not the bundle-config defaults.

**Atom ordering is pinned** to `(-size, cluster_id)` ascending — same
key as the existing LPT path (`_split_helpers.py:233-238`). This makes
the GroupKFold partition bit-reproducible across runs and architectures
(sklearn `GroupKFold` preserves input order). The chosen key is
recorded in the per-fold audit JSON as `atom_ordering_key` for
unambiguous debugging.

### D2. Per-(configuration) `max_feasible_k`

`max_feasible_k = floor(1 / (max_atom_frac - drift_pp))` for
`max_atom_frac > drift_pp`, where `max_atom_frac` is the largest
single cluster's fraction of pairs (already computed in
`single_slot_cluster_disjoint_feasibility.py`) and `drift_pp` is the
configured `max_acceptable_drift_pp` from D3 (default `0.05`). The
strict formula (zero-drift case) is `floor(1 / max_atom_frac)`.

Derivation. The strict constraint is that no fold's test bin
overflows the per-fold target. With per-fold target = `1/k` and drift
allowed up to `drift_pp`, the test bin can absorb up to
`1/k + drift_pp`. Solving for k: `k ≤ 1/(max_atom - drift_pp)`.

Worked example. HA-only id095 has `max_atom_frac = 0.135` (per
`clustering_overview.md` §10.3). At zero-drift: `floor(1/0.135) = 7`.
At default `drift_pp = 0.05`: `floor(1/0.085) = 11`.

D4's pre-build gate evaluates this formula at the **build-time
configured `drift_pp`** (not necessarily the default). Phase 1 emits
both `max_feasible_k_strict` (zero-drift) and
`max_feasible_k_at_default_drift` (drift-aware at the configured
default) so feasibility tables remain pre-publishable for the
deployed-defaults case, while runtime behavior follows the user's
actual config.

Necessary, not sufficient. Counterexample: 6 atoms of sizes
`[0.20, 0.20, 0.20, 0.14, 0.13, 0.13]` (sum 1.00, max 0.20) satisfy
`max_atom ≤ 1/k` for k=5, but LPT-greedy packs them as
`[0.20, 0.20, 0.20, 0.14, 0.26]` — bin 5 absorbs `0.13+0.13`, landing
30% over target. The *sufficient* test is per-fold drift, applied at
build time (D3).

`max_feasible_k` is therefore a **drift-aware upper bound under the
configured drift tolerance**, surfaced in the feasibility pre-flight
as guidance. Other failure modes (uneven medium-atom packing per the
counterexample above) can still trigger per-fold check failures at
build time — the per-fold check is the authoritative gate.

### D3. Two-knob feasibility check (per fold)

Every fold's bins (train, val, test) must independently satisfy:

- **`max_acceptable_drift_pp`** (default `0.05`) — max absolute pp
  deviation from target on any bin (train, val, OR test).
- **`min_test_frac`** (default `0.05`) — test bin ≥ 5% of total pairs.

Both knobs apply to all k folds **and to the single-shot path
(`n_folds=null` or `n_folds=1`)** — D3 enforces consistent
feasibility regardless of how the dataset is built. If any fold
fails either knob, the build raises with the informative menu (D4).

Why both knobs (partial redundancy, but they answer different
questions):

- `max_acceptable_drift_pp` catches the **train-too-small** case
  (e.g. 0.65/0.20/0.15: drift 15 pp fails, test fine). Undertrained
  model means worse metrics across all folds even with adequate test
  sample.
- `min_test_frac` catches the **test-too-small for metric stability**
  case (e.g. 0.85/0.11/0.04: drift 6 pp passes at default,
  test_frac 4% fails). A test bin below ~5% means seed-to-seed
  variance dominates the signal.

All bins are checked symmetrically — val drift propagates to test
metrics via early-stopping signal noise, so val gets no exemption.

### D4. k-fold-infeasible behavior: refuse with informative menu

When the requested `k` exceeds `max_feasible_k`, OR when any per-fold
feasibility check (D3) fails at build time, raise with a structured
message exposing the user's options:

```
Configuration (pair=HA-NA, alphabet=aa, slot=a, threshold=id093)
  requested k=5.
  max_feasible_k = 3 at this configuration.
  Options (require explicit config change):
    - reduce k to 3 (smaller N, same threshold)
    - relax to threshold id094 (k=5 feasible)
    - accept larger min_test_frac / max_acceptable_drift_pp (loosens
      feasibility but produces noisier per-fold metrics)
```

Why refuse over (a) silent single-shot fallback or (c) silent
auto-downshift to `k = max_feasible_k`:

- **(a)** replaces fold variance with seed variance and labels them
  identically in downstream tables. A reader of "HA-NA single-slot,
  mean F1 ± std across N folds" should be able to assume N is
  consistent across thresholds. (a) breaks that silently.
- **(c)** silently produces non-comparable fold counts across
  configurations in the same result table. Same comparability break.
- **(b) refuse** forces the user to commit per-config, which is the
  right place for the trade-off.

Two trigger points:

- **Pre-build (`k > max_feasible_k`)**: detected before any routing
  work, just a comparison against the Phase 1 pre-flight column.
  Cheap; no output written.
- **Mid-build (per-fold check fails)**: detected after all k folds
  are constructed in memory but before any disk write. The builder
  runs all per-fold checks first, then either writes all folds or
  raises naming the failing folds (e.g., "folds 2 and 4 fail
  max_acceptable_drift_pp at 0.067 and 0.082 respectively"). **No
  partial output** — either all k fold subdirs are written or none
  are. The failure is **deterministic for a given (config, atom
  set)** because atom ordering is pinned per D1; re-running with
  the same config produces the same pass/fail outcome (not flaky
  across runs).

### D5. Existing-field rename + rescale

Rename `max_target_deviation_pct` → `max_target_deviation_pp` in the
audit JSON schema (`_split_helpers.py:320`) **and rescale from 0-100
to 0-1 fractions in the same patch**. The current field computes
`max(abs(achieved_pct - target_pct))` where both inputs are on a
0-100 scale (`_split_helpers.py:300-301`); the new knobs
(`max_acceptable_drift_pp`, `min_test_frac`) are on a 0-1 fraction
scale. Keeping the current 0-100 scale while introducing 0-1 knobs
would create silent unit mismatches.

Schema note for the rescaled field: `max_target_deviation_pp ∈ [0, 1]`
where `0.05` means "achieved bin fraction deviated by up to 5
percentage points from target." The `_pp` semantics in this codebase
are therefore "absolute difference of two fractions, expressed as a
fraction itself" — slightly non-canonical (canonical `pp` implies
percentage inputs) but matches the agreed knob convention and stays
internally consistent.

## Implementation phases

### Phase 1: Feasibility pre-flight `max_feasible_k` columns

Extend `src/analysis/single_slot_cluster_disjoint_feasibility.py` to
emit **two** columns in
`single_slot_feasibility_{pair}_{alphabet}.csv` (under
`results/{virus}/{data_version}/runs/cluster_disjoint_feasibility/`):

- **`max_feasible_k_strict`** = `floor(1 / max_atom_frac)`. Zero-drift
  formula. Matches deployed behavior when the user configures
  `max_acceptable_drift_pp = 0`.
- **`max_feasible_k_at_default_drift`** = `floor(1 / (max_atom_frac -
  default_drift_pp))` where `default_drift_pp = 0.05` (the config
  default from D3). Matches deployed behavior at default settings.

Column-header note: "Necessary condition only — per-fold drift check
at build time is authoritative. D4's pre-build gate evaluates the
drift-aware formula at the build-time configured `drift_pp`; the
two columns bracket the common cases (`drift_pp = 0` and `drift_pp
= 0.05`)."

Why two columns and not one parameterized lookup: feasibility tables
get pasted into result writeups for cross-config comparison; readers
need to see the gate value without re-deriving it. The two-column
form is also one-step-CSV-readable.

No code changes outside this script. Estimated effort: ~40 lines.

### Phase 2: GroupKFold integration in `cluster_disjoint_route_pos_df`

Modify `src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df`
to optionally return k folds instead of a single (train, val, test)
triple. Dispatch by the `n_folds` value:

- **Single-shot** (`n_folds=null` or `n_folds=1`): existing
  LPT-greedy path (current behavior, preserved). Returns
  `(train_pos, val_pos, test_pos, audit)` as today. The bin-packing
  logic itself is unchanged; what's new is the **post-routing D3
  check** that raises if the achieved partition violates either
  knob. Note: `n_folds=1` does NOT route through `GroupKFold`
  (sklearn requires `n_splits >= 2`); it uses LPT-greedy and is
  semantically identical to `n_folds=null`.
- **k-fold** (`n_folds=k` for k>=2, requires `single_slot ∈
  {'a','b'}`): new GroupKFold path. Returns `list[(train_pos,
  val_pos, test_pos, audit)]` of length k. Each fold uses
  `GroupKFold` to select test atoms, then partitions the remaining
  k−1 fold's atoms into train/val by LPT-greedy with target ratios
  `train = (k-1)/k − val_frac`, `val = val_frac`. D3 feasibility
  checks apply per fold.

`val_frac` is a new arg, defaulting to `val_ratio / (1 - 1/n_folds)`
— matches the existing CV math in `generate_all_cv_folds_v2` at
line 1895. The k-fold path guarantees `n_folds >= 2` (single-shot
dispatches to LPT before this math runs), so the divide-by-zero
edge case at `n_folds = 1` doesn't fire.

Bilateral mode (single_slot=None) is **not** extended in this phase —
out of scope (see § "Out of scope"). Calling `cluster_disjoint_route_pos_df`
with `n_folds > 1` and `single_slot=None` raises NotImplementedError.

Per-fold D3 evaluation uses **collect-all** semantics: the routing
helper computes per-fold D3 metrics (`drift_pp`, `test_frac`) and
includes them in each fold's audit dict but does **NOT** raise on
individual fold failures. The dispatch caller in
`dataset_segment_pairs_v2.py` (Phase 3) collects D3 outcomes across
all folds, then either writes the dataset (if all pass) or raises
with the D4 menu listing all failing folds and their values. Better
debug UX than fail-fast at trivial cost (k ≤ ~10 fold routings before
raising), matching D4's "runs all per-fold checks first" specification.

### Phase 3: V2 builder CV path for single-slot cluster_disjoint

Modify `src/datasets/dataset_segment_pairs_v2.py`:

- Lift the rejection at line 2643 conditional on
  `single_slot ∈ {'a','b'}`. With `single_slot=None`, the rejection
  stays (out of scope).
- Extend `generate_all_cv_folds_v2` (line 1856) to dispatch to the new
  k-fold path in `cluster_disjoint_route_pos_df` when
  `split_mode == 'cluster_disjoint'` and `single_slot is not None`.
  Existing random-mode CV path is unchanged.
- Each fold's negatives are sampled via the existing per-split flow
  (`create_negative_pairs_v2` × 3) with `forbidden_pair_keys` threaded
  **within each fold** (across that fold's three splits) and **reset
  between folds**. Each fold is a statistically independent replicate;
  cross-fold negative coupling would bias later folds toward harder
  negatives (easier candidates are drained from earlier folds),
  biasing both the mean and the std of the per-fold metric estimate.
  Within-fold threading remains — required to prevent intra-model
  same-pair leakage (mode #1). Matches the existing
  `generate_all_cv_folds_v2` (line 1856) no-cross-fold-threading
  behavior.

Estimated effort: ~80 lines plus tests.

### Phase 4: Per-fold audit JSON schema

`cluster_disjoint_audit.json` gains a `folds` array (one entry per
fold) when k > 1. Each fold's entry mirrors the current
single-shot audit schema but adds:

- `fold_id`: 0..k−1
- `feasibility_check`: `{max_acceptable_drift_pp: {pass: bool, achieved: float, threshold: float}, min_test_frac: {pass: bool, achieved: float, threshold: float}}`
- `max_target_deviation_pp` (renamed per D5)

`dataset_stats.json` gains a `kfold_summary` headline block with
per-fold pass/fail summary, mirroring the existing `slot_leakage_summary`
pattern from yesterday's audit-symmetry work. Schema:

```json
"kfold_summary": {
  "k": 5,
  "max_feasible_k": 7,
  "all_folds_pass": true,
  "single_slot": "a",
  "cluster_alphabet": "aa",
  "atom_ordering_key": "(-size, cluster_id)",
  "composition_mode": null,
  "per_fold": [
    {"fold_id": 0, "drift_pp": 0.024, "test_frac": 0.198, "pass": true},
    {"fold_id": 1, "drift_pp": 0.031, "test_frac": 0.205, "pass": true}
  ]
}
```

Including `single_slot`, `cluster_alphabet`, and `atom_ordering_key`
makes the headline block self-contained for cross-config disambiguation.

`composition_mode` is **reserved for forward compatibility** with the
OoS #7 composition lever (cluster_disjoint(slot a) + seq_disjoint(slot
b)). In this plan it is always written as `null`, meaning
"constrained-slot-only atoms; no seq_disjoint augmentation on the
unconstrained slot." If the composition lever is later implemented,
the value extends to `'cluster_a_seq_b'` or `'cluster_b_seq_a'` to
record which slot got the augmentation, **without requiring a schema
migration** on existing audit JSONs. Rationale: zero implementation
cost now, avoids a breaking schema change later if multi-fold results
motivate building the composition path.

### Phase 5: Configs + defaults

Add to `conf/dataset/default.yaml`:

```yaml
split_strategy:
  feasibility:
    max_acceptable_drift_pp: 0.05
    min_test_frac: 0.05
```

`n_folds` already exists at the dataset config level (line 91) and
already supports `null` (single-shot) | int N. No new top-level knob
needed.

Two new bundles for smoke tests:

- `conf/bundles/flu_ha_na_cluster_aa_id095_HAonly_k5.yaml` — extends
  the existing id095 HAonly bundle with `dataset.n_folds: 5`.
- `conf/bundles/flu_pb2_pb1_cluster_aa_id095_PB1only_k5.yaml` —
  intentionally infeasible (`max_feasible_k = 1` at PB2-PB1 PB1-only
  id095 per §10.3 of `clustering_overview.md`; `max_atom_frac ≈
  0.784`). Verifies the D4 refuse path. (PB2-only at the same
  threshold has `max_atom_frac ≈ 0.155` and would build successfully —
  wrong direction for this smoke test.)

### Phase 6: Stage 4 integration

`stage4_sweep.sh` already iterates over dataset directories. When
Phase 3 emits `fold_{0..k-1}/` subdirs under the Stage 3 output dir,
the sweep loops over them naturally. Two integration points:

- `train_pair_classifier.py --fold_id` (lines 1060, 1202-1203 —
  verified) already resolves `--dataset_dir / "fold_{fold_id}"`.
  Matches the Phase 3 emitted layout exactly; no changes needed.
- `stage4_sweep.sh` may need a glob-expansion tweak so the per-fold
  subdirs are iterated alongside the per-seed loop.

Estimated effort: 0-5 lines, depending on whether the existing glob
expansion already covers `fold_*/` subdirs.

### Phase 7: Smoke tests

Two configurations, three checks each:

**Smoke 1: HA-NA HA-only aa id095 k=5 (expected: feasible)**
- Per-fold feasibility check passes on all 5 folds.
- `kfold_summary` in dataset_stats.json reports k=5, all passing.
- Stage 4 sweep produces 5 × 3 seed runs, all completing.

**Smoke 2: PB2-PB1 PB1-only aa id095 k=5 (expected: refuse)**
- Phase 1 pre-flight reports `max_feasible_k_strict = 1` AND
  `max_feasible_k_at_default_drift = 1` (PB2-PB1 PB1-only at id095
  has max_atom_frac ≈ 0.784 per §10.3 of `clustering_overview.md`;
  drift-aware `floor(1/(0.784 − 0.05)) = floor(1.362) = 1`, same as
  zero-drift `floor(1/0.784) = floor(1.276) = 1`).
- Stage 3 build raises with the D4 menu before any folds are written
  (pre-build trigger per D4).
- Test that the error message text matches the D4 template.

**Smoke 3: HA-NA HA-only aa id095 k=12 at default `drift_pp = 0.05` (expected: refuse)**
- `max_feasible_k_at_default_drift` from Phase 1 pre-flight = 11
  (HA-NA HA-only id095 has max_atom_frac ≈ 0.135 → drift-aware
  `floor(1/(0.135 − 0.05)) = 11`); k=12 > 11 → pre-build refuse.
- Stage 3 build raises with the D4 menu before any folds are written
  (pre-build trigger per D4).
- Exercises the drift-aware pre-build gate at deployed defaults.
  Complementary to Smoke 2 (which fails on the strict formula too).

## Files affected

| File | Change | Effort |
|---|---|---|
| `src/analysis/single_slot_cluster_disjoint_feasibility.py` | Add `max_feasible_k_strict` + `max_feasible_k_at_default_drift` columns | ~40 lines |
| `src/datasets/_split_helpers.py` | Extend `cluster_disjoint_route_pos_df` for k-fold; rename `_pct` → `_pp` | ~60 lines |
| `src/datasets/dataset_segment_pairs_v2.py` | Lift n_folds rejection for single-slot cluster_disjoint; dispatch k-fold path | ~80 lines |
| `conf/dataset/default.yaml` | Add feasibility knobs | ~5 lines |
| `conf/bundles/flu_ha_na_cluster_aa_id095_HAonly_k5.yaml` | New bundle | ~10 lines |
| `conf/bundles/flu_pb2_pb1_cluster_aa_id095_PB1only_k5.yaml` | New bundle (refuse-case smoke) | ~10 lines |
| `scripts/stage4_sweep.sh` | Probably no change; verify glob expansion | ~0–5 lines |
| `docs/methods/clustering_overview.md` | §7.2 update with k-fold semantics | ~20 lines |
| `docs/methods/leakage_definitions.md` | "Routing equivalence" subsection note for k-fold | ~10 lines |
| `docs/methods/dataset_construction_v2_workflow.md` | Phase 6 audit list gains kfold_summary mention | ~5 lines |

Total: ~260 LoC code, ~50 lines doc updates, plus tests.

## Explicitly out of scope

1. **Variance source (ii) negative-resampling.** Per cross-cutting A
   decision: deferred. The current Stage 3 negative sampling consumes
   `master_seed`; varying the seed re-rolls negatives against a fixed
   positive partition. This would produce a different variance estimand
   (Monte Carlo over negative draws given one positive partition), not
   comparable to (i) k-fold variance. Don't mix in the same report.
2. **E1 (LPT + atom-shuffle).** Modifying the bin-packer to consume the
   seed and produce different valid bin-packings per seed. Revisit only
   if k-fold variance produced by E2 is too thin (N=3 or N=5) to
   support paper claims; then E1 could supplement as a Monte Carlo
   estimator with overlapping test sets. Don't build speculatively.
3. **E1 hybrid (GroupKFold for test partition + LPT for train/val
   within each fold).** Adds complexity for marginal balance gain over
   the default GroupKFold. Revisit only if per-fold drift shows up as a
   real problem in audits.
4. **E3 (GroupShuffleSplit).** Random group assignment without balance
   optimization. Overshoots ratios badly when atoms are skewed
   (PB2-PB1 PB1-only at id095 with max_atom 78% would put that atom in
   the test bin most of the time). Not useful for our atom geometry.
5. **k-fold for bilateral cluster_disjoint. Not currently supported.**
   Bilateral atoms (bipartite CCs) collapse to a mega-component at
   most thresholds below id099 on Flu A; k-fold feasibility is
   unattainable at the interesting thresholds. At id100 it's
   technically feasible (max_feasible_k=5 on HA-NA per §10.2 of
   `clustering_overview.md`) but redundant with seq_disjoint k-fold
   (item 6). Design extends naturally — same GroupKFold + per-fold
   check pattern, atom = bipartite CC.
6. **k-fold for seq_disjoint. Not currently supported.** A separate
   work item with its own feasibility profile. Design extends
   naturally — bipartite CCs as atoms (per
   `_pair_helpers.seq_disjoint_route_pos_df`), same GroupKFold +
   per-fold check pattern, ~40 LoC. Skipped because seq_disjoint
   atoms on Flu A are typically small (largest CC ~20% of pairs at
   HA-NA hash_key=seq) so k-fold feasibility is rarely the
   bottleneck — the variance question is less interesting empirically.
7. **Composition of cluster_disjoint(slot a) + seq_disjoint(slot b).**
   The unconstrained-slot mitigation from item 5 of the design
   conversation. Audit-first (already done via `slot_leakage_summary`);
   compose-if-needed depending on multi-fold results. When implemented,
   integrates with this plan's k-fold framework via an extended atom
   definition (bipartite CC on `(cluster_id_a, seq_hash_b)` rather
   than just `cluster_id_a`); same GroupKFold + D3 + D4 pattern, with
   composed-atom feasibility computed in Phase 1 pre-flight as a
   separate `max_feasible_k_composed` column. Tracked separately in
   BACKLOG.md "Single-slot routing follow-ups" #4 (to be added).
8. **Item 4 reporting convention** ("control / quantify /
   acknowledge"). Reserved as a placeholder section below; content
   depends on whether multi-fold results show unconstrained-slot
   leakage as a meaningful per-fold variance contributor.
9. **DataSAIL ILP path.** L(π) minimizer collapsed all routings to a
   partition-shape constant on Flu A (see `docs/results/2026-05-24_datasail_lpi_results.md`).
   Not a viable alternative primitive.
10. **N-candidate-orderings fallback for mid-build D4 failures.** If
    Phase 1 deployment shows the mid-build refuse rate exceeds ~20%
    of configs, add a Phase 8 that shuffles the atom list N times
    (e.g., N=10), runs GroupKFold on each, picks the partition with
    the best worst-fold drift. Introduces a meta-seed dependency
    (the partition depends on N and a shuffle seed), but variance
    semantics stay clean as long as the chosen partition is recorded
    in the audit. Adds ~20 LoC. Don't build speculatively — current
    Phase 1 ships with deterministic D4 refuse only.

## Reporting convention (RESERVED)

**Status: empty pending multi-fold results.**

Populated post-results, once Phase 7 smoke tests + first real multi-fold
sweep produce concrete numbers. The convention's emphasis depends on
whether unconstrained-slot leakage materializes as a meaningful
per-fold variance contributor:

- If leakage shifts metrics noticeably across folds (variance contribution
  exceeds e.g. 0.5 pp on F1), the convention emphasizes **acknowledge**
  ("MLP F1 = X.XX ± std under single-slot HA-only at id095, with
  NA seq_hash leakage 5.1%; unbiased estimator would require composing
  NA seq_disjoint per BACKLOG.md #4").
- If leakage is inside seed noise (variance contribution within
  per-fold std), the convention shifts to **verified empirically
  negligible** ("MLP F1 = X.XX ± std under single-slot HA-only at
  id095; NA seq_hash leakage 5.1% confirmed within per-fold variance,
  not a metric-shift confounder").

Write the convention after the first multi-fold result lands. Locking
prose pre-results risks anchoring on the wrong frame.

## Testing strategy

- **Unit tests** for `cluster_disjoint_route_pos_df` k-fold path:
  per-fold non-overlap on the constrained slot's cluster ids;
  pair_key uniqueness across folds; per-fold feasibility check
  pass/fail logic.
- **Integration tests** via Phase 7 smoke tests.
- **Regression test**: single-shot path (`n_folds=1` or
  `n_folds=null`) produces bit-identical output to current master
  before the patch **for configs that pass the D3 feasibility checks**.
  Configs that previously succeeded but now violate D3 (e.g., HA-NA
  aa id095 at 98.5/0.76/0.76) will raise by-design under the new
  behavior — the regression test framework should mark those configs
  as expected-raise rather than expected-pass. This is the intentional
  D3-applies-to-single-shot consequence.

## See also

- `docs/methods/clustering_overview.md` § 7.2 — LPT-greedy framing and
  multi-shuffle distinction.
- `docs/methods/leakage_definitions.md` § "Routing equivalence" — the
  bilateral vs single-slot table.
- `BACKLOG.md` § "Single-slot routing follow-ups" #3 — the originally
  scoped GroupKFold work that this plan implements.
- `data_split_design_prompt1.md`, `data_split_design_resp1.md`,
  `data_split_design_prompt2_draft.md`,
  `data_split_design_prompt2_partial.md` — design conversation
  artifacts that produced these decisions (scratch files in repo
  root; not in git).
- `_split_helpers.py:126-331` — `cluster_disjoint_route_pos_df`
  implementation.
- `dataset_segment_pairs_v2.py:1856-1947` — existing CV path
  (`generate_all_cv_folds_v2`, random mode only).
- `dataset_segment_pairs_v2.py:2643-2648` — current n_folds>1
  rejection for cluster_disjoint, to be lifted conditionally.
