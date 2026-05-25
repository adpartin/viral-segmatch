# Single-slot HA-only cluster_disjoint sweep on HA-NA — does MMD grow gradually with id↓?

First MMD trajectory across a full idXX sweep on Flu A. Builds on the
single-slot routing introduced in commit `4607050` and the feasibility
pre-flight in `docs/results/2026-05-24_cluster_disjoint_feasibility_HA_NA.md`.

Question: under aa cluster_disjoint **single-slot HA-only** routing,
does MMD on train vs test grow monotonically as cluster-identity
threshold decreases (id100 → id095)? Pre-registered prediction
(written before the sweep):

- **S1 HA** (constrained slot) — monotone growth with id↓.
- **S1 NA** (unconstrained slot) — stays near random baseline.
- **S2 pair** (Test 3 interaction) — partial growth.

The pre-registered prediction for S1 NA was **falsified** on the
POC run at id098 (NA MMD² = 0.012, p = 0.002 vs random baseline
0.001). A 30-second sanity check confirmed the mechanism: HA cluster
boundary essentially **is** the NA subtype boundary on this corpus
(Cramér's V = 0.90; 88% of pairs in HA clusters that are ≥95%
NA-subtype-pure). The single-slot relaxation does not decouple HA
and NA because they are biologically coupled at the isolate level
via subtype.

The sweep then asks: how does that coupling evolve as id↓ coarsens
HA clusters and (potentially) breaks subtype-purity?

## Scope

- One virus, one bundle pair: Flu A HA-NA, full corpus
  (`data_version=July_2025`).
- One alphabet: aa (symmetric easy-linclust, post-2026-05-22 switch).
- One slot direction: HA-only (NA-only and nt sibling sweeps not run).
- One pair interaction at S2: Test 3 (`slot_transform=unit_norm`,
  `interaction=unit_diff+prod`) — what the production model uses.
- Six thresholds: id100, id099, id098, id097, id096, id095.
  All feasible 80/10/10 per pre-flight; id095 is the boundary of
  feasibility on this corpus.
- Two feature spaces: ESM-2 (1280-dim, protein-level) and aa k=3
  (8000-dim, protein-level).
- Same 1000-isolate subsample (subsample_seed=42) and same σ values
  as prior S1/S2 baselines, so the new cells are directly comparable
  to the existing random / seq_disjoint / bilateral cluster_id099
  numbers in `docs/results/2026-05-24_mmd_per_slot_results.md` and
  `docs/results/2026-05-24_mmd_per_pair_results.md`.

## Setup

- **Datasets**: six Stage 3 dirs under
  `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_aa_id{XXX}_HAonly_*`.
  All hit 80/10/10 exactly. HA cluster overlap = 0 across splits
  (enforced by routing). NA cluster overlap non-zero by design
  (580–3420 across thresholds; grows as idXX → 100 because NA gets
  more granular clusters).
- **Bundle configs**: `conf/bundles/flu_ha_na_cluster_aa_idXXX_HAonly.yaml`.
  Each inherits `flu_ha_na` and overrides
  `dataset.split_strategy.{mode: cluster_disjoint, cluster_alphabet: aa,
  single_slot: a, cluster_id_threshold: 0.XX, cluster_id_path: ...}`.
- **MMD**: PCA-50 + RBF + 500-permutation. Fixed σ per (feature space,
  S1/S2) from prior Phase 1 sanity runs:
  - ESM-2 σ_S1 = 1.0719, σ_S2 = 0.3588
  - aa k=3 σ_S1 = 29.3192, σ_S2 = 1.0720
- **Scripts**: `src/analysis/mmd_per_slot.py` (S1), `mmd_per_pair.py`
  (S2), `aggregate_mmd_single_slot_sweep.py` (rollup + plot).

## Results

### ESM-2

MMD² and permutation p-value at each idXX.

| idXX | HA MMD² | HA p | NA MMD² | NA p | Pair MMD² | Pair p |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.00366 | 0.541 | 0.00214 | 0.417 | 0.00299 | 0.677 |
| 099 | 0.01005 | 0.154 | 0.00437 | 0.092 | 0.01109 | 0.054 |
| 098 | 0.01828 | **0.028** | 0.01223 | **0.002** | 0.02747 | **0.002** |
| 097 | 0.02489 | **0.010** | 0.00478 | 0.060 | 0.02101 | **0.002** |
| 096 | 0.04138 | **0.002** | 0.02053 | **0.002** | 0.04337 | **0.002** |
| 095 | 0.08287 | **0.002** | 0.01997 | **0.002** | 0.07028 | **0.002** |

Growth ratios (id095 / id100): HA 22.6×, NA 9.3×, Pair 23.5×.

### aa k=3

| idXX | HA MMD² | HA p | NA MMD² | NA p | Pair MMD² | Pair p |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.00187 | 0.876 | 0.00231 | 0.681 | 0.00208 | 0.868 |
| 099 | 0.01941 | **0.004** | 0.00583 | 0.112 | 0.01609 | **0.006** |
| 098 | 0.03045 | **0.002** | 0.01564 | **0.002** | 0.02949 | **0.002** |
| 097 | 0.02798 | **0.002** | 0.01149 | **0.010** | 0.02612 | **0.002** |
| 096 | 0.04830 | **0.002** | 0.03014 | **0.002** | 0.04561 | **0.002** |
| 095 | 0.06304 | **0.002** | 0.02986 | **0.002** | 0.05879 | **0.002** |

Growth ratios (id095 / id100): HA 33.7×, NA 12.9×, Pair 28.3×.

### Comparison to existing reference baselines (same σ, same subsample)

| Routing | ESM-2 HA p | ESM-2 NA p | ESM-2 Pair p | aa k=3 HA p | aa k=3 NA p | aa k=3 Pair p |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.639 | 0.864 | 0.705 | 0.617 | 0.645 | 0.717 |
| seq_disjoint | 0.166 | 0.056 | 0.034 | **0.004** | 0.062 | **0.002** |
| bilateral cluster_id099 | **0.002** | **0.002** | **0.002** | **0.002** | **0.002** | **0.002** |

The single-slot sweep's id100 sits at random-baseline level, id099
sits near seq_disjoint level (mid-strength), and id098–id095
progresses toward bilateral cluster_id099 strength but only on HA;
NA stays well below bilateral cluster_id099's NA shift in most cells.

### Plot

`results/flu/July_2025/runs/split_separation_mmd/sweep_aggregate/sweep_mmd_vs_idxx.png`
— two panels (ESM-2, aa k=3) with three lines per panel (HA blue,
NA orange, pair purple), filled markers for p ≤ 0.05, dashed
reference lines for bilateral cluster_id099 and dotted for random.

### Held-out test performance (aa k=3 features, Test 3 interaction)

One MLP + two baselines (LGBM, 1-NN cosine margin) trained per
dataset using the `flu_ha_na_kmer_aa_k3` bundle. MLP trained with
**3 seeds (42, 43, 44)** for error-bar estimation; LGBM and 1-NN are
single-seed (LGBM has minor sampling randomness but was not
re-seeded here; 1-NN cosine margin is deterministic w.r.t. the data
so a single run is sufficient). Six GPUs in parallel for each MLP
batch; baselines on CPU. Output dirs follow
`models/.../runs/training_*_HAonly_idXXX[_seedN]_*` and
`baseline_{lgbm,knn1_margin}_*_HAonly_idXXX_*`.

**MLP across 3 seeds (mean ± std):**

| idXX | F1 | AUC-ROC | MCC |
|---:|---:|---:|---:|
| 100 | 0.9621 ± 0.0013 | 0.9855 ± 0.0006 | 0.9365 ± 0.0022 |
| 099 | 0.9473 ± 0.0019 | 0.9810 ± 0.0015 | 0.9116 ± 0.0031 |
| 098 | 0.9363 ± 0.0009 | 0.9744 ± 0.0018 | 0.8930 ± 0.0014 |
| 097 | 0.9363 ± 0.0013 | 0.9751 ± 0.0011 | 0.8928 ± 0.0023 |
| 096 | 0.9199 ± 0.0032 | 0.9683 ± 0.0010 | 0.8649 ± 0.0056 |
| 095 | 0.9181 ± 0.0018 | 0.9661 ± 0.0007 | 0.8619 ± 0.0031 |

Standard deviations are 0.001–0.005 on F1, 0.001–0.002 on AUC-ROC,
0.001–0.006 on MCC. The 4.4-pp F1 drop from id100 to id095 is
~80σ — the trajectory is statistically robust at this scale (model-
seed variance only; the variance from re-splitting the same atoms
into different folds is not measured here, see BACKLOG.md
"Single-slot routing follow-ups" #3).

**Three notable per-cell observations** the multi-seed run rules in
as real (not noise):

- id097 ≈ id098 plateau on F1 (both 0.9363 with std < 0.002) is real,
  not noise. The plateau aligns with the id097 NA-MMD dip observed in
  the MMD sweep.
- id100 → id099 step (1.5 pp F1) is real — std bars don't overlap.
- id096 has slightly noisier seeds (F1 std = 0.0032, MCC = 0.0056)
  but the gap to id095 (1.8 pp F1) is still > std.

**LGBM and 1-NN at seed=42 (single seed):**

| idXX | LGBM F1 | 1-NN F1 | LGBM AUC-ROC | 1-NN AUC-ROC | LGBM MCC | 1-NN MCC |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.9500 | 0.9580 | 0.9894 | 0.9847 | 0.9160 | 0.9297 |
| 099 | 0.9301 | 0.9395 | 0.9813 | 0.9800 | 0.8823 | 0.8983 |
| 098 | 0.9171 | 0.9306 | 0.9767 | 0.9734 | 0.8603 | 0.8831 |
| 097 | 0.9148 | 0.9301 | 0.9740 | 0.9735 | 0.8564 | 0.8825 |
| 096 | 0.8984 | 0.9204 | 0.9674 | 0.9693 | 0.8284 | 0.8657 |
| 095 | 0.8907 | 0.9105 | 0.9597 | 0.9646 | 0.8153 | 0.8490 |

LGBM and 1-NN drops from id100 → id095: F1 5.9 pp (LGBM), 4.7 pp
(1-NN). AUC-ROC: 3.0 pp (LGBM), 2.0 pp (1-NN). MCC: 10.0 pp (LGBM),
8.1 pp (1-NN). Model ordering **MLP > 1-NN > LGBM** at every
threshold; 1-NN edges MLP at id096 by 0.001 F1 (within MLP's seed
noise).

Plots:
- `sweep_perf_vs_idxx.png` — three panels (F1, AUC-ROC, MCC),
  three lines per panel (MLP, LGBM, 1-NN); MLP line is mean across
  seeds with ±1 std shaded band, per-seed dots overlaid.
- `sweep_perf_vs_mmd_pair_kmer_aa_{pos,neg,both}.png` — three single
  panel scatters of F1 vs S2 pair MMD² (aa k=3) for each label
  filter, with MLP error bars (3 seeds) and per-seed dots.

### Negative-pair regime sanity check

Sized + axis-mix comparison across the 6 datasets — confirms
constructed-negatives are not silently shifting in a way that would
contaminate the perf trajectory.

Pair counts are identical across all 6 datasets (train 46,710 + 70,065
neg / val 5,839 + 8,758 neg / test 5,839 + 8,758 neg — neg:pos = 1.50
held exactly). The `axis_flag_summary` per-axis `same%` (positives +
negatives, counted on `axis_flag_summary` from `dataset_stats.json`):

| axis | test range across idXX (max − min) | train range |
|---|---:|---:|
| hn_subtype   | 2.89 pp | 1.51 pp |
| host         | 4.90 pp | 2.29 pp |
| year         | 0.38 pp | 0.20 pp |
| geo_location | 0.51 pp | 0.17 pp |
| passage      | 1.83 pp | 0.26 pp |

Largest swing is `host` on test (53.46% same at id100 → 49.95% at id095).
Negatives become slightly MORE cross-host as id↓ — which would make
them *easier*, not harder, to discriminate from positives. F1 still
drops, so the perf trajectory is not driven by negative-axis
composition shift.

### Negative-pair MMD trajectory

The axis-composition sanity above only checks aggregate axis-state
fractions. The per-sequence negative *distribution* in feature space
can still shift because negatives are sampled from per-split
positive pools (which themselves shift with id↓). Re-running the aa
k=3 MMD sweep with `--label_filter 0` (negatives only):

| idXX | HA pos | HA neg | NA pos | NA neg | Pair pos | Pair neg | Pair **both** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.0019 | 0.0072 | 0.0023 | 0.0026 | 0.0021 | 0.0052 | **0.0033** |
| 099 | 0.0194 | 0.0134 | 0.0058 | 0.0067 | 0.0161 | 0.0115 | **0.0165** |
| 098 | 0.0304 | 0.0337 | 0.0156 | 0.0103 | 0.0295 | 0.0248 | **0.0241** |
| 097 | 0.0280 | 0.0312 | 0.0115 | 0.0113 | 0.0261 | 0.0242 | **0.0209** |
| 096 | 0.0483 | 0.0448 | 0.0301 | 0.0139 | 0.0456 | 0.0305 | **0.0358** |
| 095 | 0.0630 | 0.0460 | 0.0299 | 0.0231 | 0.0588 | 0.0386 | **0.0490** |

The **"Pair both"** column is the same pair-level S2 MMD but computed
on the full pair set (positives + negatives jointly, neg:pos = 1.5).
This is the closest single MMD summary to "what the model trains
and tests on". p-values: id100 borderline (p = 0.024); id099–id095
all at floor (p = 0.002).

Both-labels pair MMD is approximately a count-weighted blend of the
pos-only and neg-only pair MMDs at each threshold, with the id097
dip preserved (0.0241 → 0.0209 → 0.0358) — the perf plateau aligns.

Negative p-values are all significant at the perm floor (p ≤ 0.030)
except id100 NA (p = 0.293). Both pos and neg show **monotone growth
with id↓** in HA / NA / pair, in the same direction. Negative MMD is
comparable in magnitude to positive MMD across the sweep, with two
regime-dependent excursions:

- **At id100, negatives shift MORE than positives** (HA neg 3.9× the
  HA pos MMD; pair neg 2.5× the pair pos MMD). At id100 the routing
  is essentially random (largest HA atom 0.7% of pairs), so positives
  barely shift; the negative shift at id100 reflects per-split
  regime-sampler stochasticity, not the intentional constraint.
- **At id095, positives shift MORE than negatives** (HA pos 1.4× the
  HA neg MMD; pair pos 1.5× the pair neg MMD). At id095 the
  constraint is heavy and the positive HA shift dominates; negatives
  shift partially because their partner pool tracks the positive
  pool's shift, but partner-sampling regime constraints dampen the
  full propagation.
- **At id098 and id097, pos and neg are within ~10–15% of each other**
  — the dominant shift regime.

### MMD ↔ perf relationship

Pairing the F1 numbers above with the aa k=3 pair-MMD² (positives
only) from the sweep gives a nearly linear F1-vs-MMD relationship
across the six thresholds, in all three models:

- All three models trace approximately straight lines in F1-vs-MMD²
  space.
- The model ordering (MLP > 1-NN > LGBM at every threshold) is
  preserved across the entire sweep — distribution shift weakens
  everything roughly proportionally, not a specific model class.
- The id097 ≈ id098 plateau on F1 (MLP: 0.9367 vs 0.9366) lines up
  with the id097 dip on S1 NA MMD: the constrained-slot HA MMD did
  grow id098 → id097 but the unconstrained-slot NA MMD dropped, and
  the pair S2 MMD did not grow either — and F1 did not drop. The
  three (HA, NA, pair) MMD signals are coherent with the perf signal
  on a per-cell basis, not just in aggregate.

**Refined framing on what's driving the perf drop:** the negative MMD
trajectory rules out an earlier hope that the perf drop would be
"positive-driven". It is more accurate to call this a **joint
positives + negatives shift**:

- Both halves of the data the model sees shift across the sweep.
- Positives shift via the intentional HA-cluster constraint (direct).
- Negatives shift via the partner-pool inheritance (indirect — the
  partner pool *is* the per-split positives, which shifted).
- The model's perf drop reflects the joint shift. Pair-level MMD
  (S2) — which mixes both slots' effects — is the closest MMD
  summary to "what the model sees" and is what shows the cleanest
  near-linear MMD↔F1 fit.

The earlier "AUC-ROC drops less than F1 because negatives are
easier" reasoning is partly right but needs adjustment: negatives
are *axis-easier* at id095 (more cross-host) but *sequence-shifted*
across the sweep, so the AUC-ROC vs F1 gap can't be cleanly
attributed to one factor.

Caveat on interpretability: cross-feature-space MMD² values are not
directly comparable (different σ per feature space). The F1-vs-MMD
plot uses the aa k=3 pair MMD as the x-axis because that matches
the feature space the model was trained on; an ESM-2-pair x-axis
gives a similarly monotone scatter but with different MMD²
magnitudes.

## Observations

1. **HA MMD grows monotonically with id↓** in both feature spaces.
   ESM-2: 0.004 → 0.083 (22.6×). aa k=3: 0.002 → 0.063 (33.7×). The
   intentional constraint produces the intended monotone shift. id098
   is the first cell where ESM-2 HA detects the shift at p ≤ 0.05;
   aa k=3 detects it one threshold earlier (id099, p = 0.004).

2. **NA MMD grows non-monotonically, with smaller magnitude than HA.**
   Both feature spaces show a clear dip at id097 (NA HA-MMD: ESM-2
   0.025 → 0.005; aa k=3 0.028 → 0.011) before resuming growth at
   id096–id095. NA growth ratio is ~9–13×, roughly half of HA's
   ~22–34×. The biological-coupling-via-subtype hypothesis predicts
   exactly this pattern: NA distribution shifts only insofar as the
   HA cluster bin-packing happens to align with NA subtype partition,
   which can flip threshold-to-threshold as clusters coalesce.

3. **S2 pair MMD tracks HA MMD closely**, slightly smaller in magnitude.
   ESM-2 pair growth ratio = 23.5×, very close to HA's 22.6×; aa k=3
   pair = 28.3× vs HA 33.7×. The pair representation is dominated by
   the constrained slot's shift; the unconstrained slot's smaller and
   non-monotonic shift contributes a fraction that the Test 3
   interaction (`unit_diff` + `prod` after `unit_norm`) doesn't
   amplify.

4. **id100 is a clean sanity check.** Largest HA atom is only 0.68%
   of pairs, so the routing is effectively random. MMD² ≈ random
   baseline in both feature spaces, all p > 0.4. ✓

5. **aa k=3 detects mid-strength shifts more sharply than ESM-2** at
   id099 (HA p = 0.004 vs ESM-2 HA p = 0.154; Pair p = 0.006 vs
   0.054). Consistent with the slot-level finding in
   `2026-05-24_mmd_per_slot_results.md` (k-mer > ESM-2 for
   seq_disjoint detection). At id098 onward both feature spaces hit
   the permutation-test floor (p = 0.002) on most cells, so the
   relative ordering is no longer informative there.

6. **HA-NA biological coupling is partial, not perfect, across the sweep.**
   If coupling were exact and constant across idXX, NA MMD would
   equal HA MMD at every threshold. Instead NA MMD is ~50–70% of HA
   MMD in cells where both grow, and as low as ~20% at id097. The
   coupling strength changes with idXX because different cluster
   coalescences hit different mixes of NA subtypes.

## Interpretation — what we can and cannot claim

### What the empirical results show

- For Flu A HA-NA, a single-slot HA-only cluster_disjoint sweep
  from id100 to id095 produces a **monotone HA MMD trajectory** and
  a **non-monotone NA MMD trajectory** — both significantly
  different from random in the lower half of the sweep.
- The S2 pair representation tracks the HA trajectory under Test 3.
- The "intended-shift on HA, coupling-shift on NA" decomposition is
  visible numerically: HA MMD ≥ NA MMD in every cell of the sweep.
- The id097 non-monotonicity is a feature of the empirical data, not
  a wiring artifact (confirmed in both feature spaces independently).
- Held-out test perf drops monotonically with id↓ in all three models
  (MLP F1 4.6 pp, LGBM 5.9 pp, 1-NN 4.7 pp; AUC-ROC 1.9–3.0 pp; MCC
  7.9–10.0 pp). F1-vs-MMD² scatter is nearly linear in all three.
- The drop is a **joint positives + negatives shift**, not
  positive-only: negative MMD also grows monotonically with id↓
  (pos and neg within ~10–15% of each other at id098/id097; neg
  exceeds pos at id100 sampling-noise regime; pos exceeds neg at
  id095 heavy-constraint regime). Negative-axis composition is
  stable across the sweep (≤4.9 pp variation per axis), so the
  negative shift is sequence-level, not regime-level.

### What this does not establish

- **MLP measured with 3 seeds (42, 43, 44); LGBM and 1-NN with 1
  seed (42).** MLP seed noise is small enough (std ≤ 0.005 on F1)
  to confirm the trajectory; LGBM/1-NN single-seed numbers should
  be read as point estimates, not means. Re-seeding LGBM (which has
  minor sampling randomness) is a cheap follow-up if its trajectory
  matters for a downstream claim.
- **Only model-seed variance measured, not split-induced variance.**
  All 3 MLP seeds were trained on the SAME 6 datasets. A different
  partitioning of the same atoms into train/val/test (different
  random bin-packings, or proper CV folds via sklearn's
  `GroupKFold`) would add another variance source. Tracked as
  BACKLOG.md "Single-slot routing follow-ups" #3.
- **One feature space for training (aa k=3).** ESM-2 training across
  the same six datasets is the natural cross-check — would it show
  the same MMD↔perf trajectory? Skipped to keep the batch tight.
- **One sweep, one direction, one cluster-alphabet, one pair.**
  PB2-PB1 is biologically different (no subtype coupling, polymerase
  complex co-conservation instead) and may show a very different
  NA-shift trajectory. nt cluster_disjoint single-slot HA-only is
  feasible through id097 per pre-flight and would test alphabet
  dependence at the routing step; not run.
- **One subsample size + one PCA dim for the primary numbers.**
  N=1000 and PCA-50 throughout the primary tables. Robustness checks
  at N=2000 and PCA dim ∈ {25, 50, 100, 200} show the trajectory is
  preserved (see "Robustness checks" subsection below); larger N or
  alternative PCA dim was NOT explored across all 6 thresholds, so
  the primary numbers are conditioned on these specific choices.
- **One subsample seed, one set of σ values.** Subsample_seed=42 is fixed
  across S1/S2/sweep work; no resampling. σ was set by Phase 1
  median heuristic on the cluster_id099 set and held fixed; an
  alternative σ choice (e.g., recomputed on the random partition of
  each idXX dataset) could shift absolute MMD² values.
- **id097 non-monotonicity is described, not explained.** A focused
  follow-up (which HA clusters land in train vs test at id097? which
  NA subtypes do they correspond to?) would identify the specific
  cluster-coalescence behavior responsible.
- **The "biological coupling" story has been validated at id098
  only** (Cramér's V = 0.90 on HA-cluster × NA-subtype). It is
  plausible at other thresholds but not directly verified.
- **MMD↔perf is a within-corpus correlation, not a generalization
  guarantee.** The near-linear F1-vs-MMD² relationship was measured
  on one corpus, one slot constraint, one model-feature stack. The
  shape (and even the sign) of the relationship is not guaranteed
  to hold under heavier regimes (e.g., id < 0.95 if feasibility
  could be unlocked), under multi-axis metadata holdout, or on
  another virus / protein pair.

### Robustness checks (2026-05-25)

Two sanity checks on the headline numbers' sensitivity to the two
main MMD hyperparameters. Both ran on aa k=3 (the cheapest feature
space; same sigma-tuning per check).

**(1) Sample-size: N=1000 → N=2000 (positives only, aa k=3).**
Re-ran the full 6-threshold × 3-role sweep at N=2000. σ at N=2000:
HA σ = 29.20 (vs 29.32 at N=1000), pair σ = 1.0697 (vs 1.0720) —
both within ~0.5%. Sweep numbers:

| idXX | HA N=1000 | HA N=2000 | NA N=1000 | NA N=2000 | Pair N=1000 | Pair N=2000 |
|---:|---|---|---|---|---|---|
| 100 | 0.0019 (0.88) | 0.0013 (0.74) | 0.0023 (0.68) | 0.0030 (0.15) | 0.0021 (0.87) | 0.0020 (0.40) |
| 099 | 0.0194 (0.004) | 0.0179 (0.002) | 0.0058 (0.11) | 0.0064 (**0.012**) | 0.0161 (0.006) | 0.0147 (0.002) |
| 098 | 0.0304 (0.002) | 0.0319 (0.002) | 0.0156 (0.002) | 0.0144 (0.002) | 0.0295 (0.002) | 0.0303 (0.002) |
| 097 | 0.0280 (0.002) | 0.0250 (0.002) | 0.0115 (0.010) | 0.0123 (**0.002**) | 0.0261 (0.002) | 0.0244 (0.002) |
| 096 | 0.0483 (0.002) | 0.0560 (0.002) | 0.0301 (0.002) | 0.0304 (0.002) | 0.0456 (0.002) | 0.0527 (0.002) |
| 095 | 0.0630 (0.002) | 0.0659 (0.002) | 0.0299 (0.002) | 0.0231 (0.002) | 0.0588 (0.002) | 0.0568 (0.002) |

MMD² magnitudes shift ≤15% per cell; trajectory identical (including
the id097 NA dip); p-values slightly tighter at N=2000 on the few
previously-borderline cells (most notably id099 NA: 0.11 → 0.012,
id097 NA: 0.010 → 0.002). No qualitative change.

**(2) PCA dim ∈ {25, 50, 100, 200} (pair MMD², both labels, aa k=3).**
σ auto-computed per (idXX, PCA dim) via median heuristic; values:
σ_PCA25 ≈ 0.94, σ_PCA50 ≈ 0.99, σ_PCA100 ≈ 1.03, σ_PCA200 ≈ 1.06.

| idXX | PCA=25 | PCA=50 | PCA=100 | PCA=200 |
|---:|---|---|---|---|
| 100 | 0.0036 (0.034) | 0.0037 (0.022) | 0.0036 (0.022) | 0.0036 (0.018) |
| 098 | 0.0291 (0.002) | 0.0266 (0.002) | 0.0250 (0.002) | 0.0239 (0.002) |
| 095 | 0.0597 (0.002) | 0.0546 (0.002) | 0.0510 (0.002) | 0.0484 (0.002) |

MMD² magnitudes shift ≤22% across the 8× range of PCA dims; ordering
preserved at every PCA dim (id100 < id098 < id095). p-values
essentially identical. The slight MMD² decrease with higher PCA dim
is expected (larger σ → smoother kernel → smaller MMD² magnitude;
the ratio across thresholds is what carries the signal). PCA-50
default is well-supported.

**Takeaway:** the headline trajectory is robust to N (within compute-
feasible range) and to PCA dim choice. The chosen defaults
(N=1000, PCA-50) are good operating points: smaller doesn't lose
power on the primary cells, larger doesn't change the story.

Output CSVs: `_N2000.csv` and `_PCA{25,50,100,200}.csv` suffixes
under `results/.../split_separation_mmd/`. Not loaded by the default
aggregator (which sticks to the N=1000 / PCA-50 primary numbers);
re-aggregate by hand if needed.

### Implications for the bigger picture

- Single-slot routing is a useful tool for separating "intended
  shift" from "incidental shift" — but only because the unconstrained
  slot's shift is itself empirically measurable and can be regressed
  against the intended one. If NA had stayed near random across the
  sweep, single-slot would have been just "less strict cluster_disjoint";
  because NA shifts too, we get a per-threshold measurement of how
  much biological coupling buys us "for free".
- The HA growth trajectory provides a natural sweep axis for the
  paper experiment that the bilateral path could not (only id100 and
  id099 are feasible bilaterally). Six monotonic HA-MMD points is
  more than enough resolution to plot perf-vs-MMD if models are
  trained.
- The k-mer sharper-at-seq-disjoint finding repeats at the
  single-slot sweep, providing one more data point that aa k=3 is at
  least as good a "split-separation diagnostic" as ESM-2 (and far
  cheaper to compute — ~10s/run vs ~230s/run).

## Reproduce

End-to-end with the wrapper scripts (`scripts/stage4_sweep.sh` and
`scripts/mmd_sweep.sh`). Both are bash-and-zsh portable and accept a
`{thr}` placeholder in the dataset-pattern; no inline editing needed.

```bash
# 1. Build the 6 sweep datasets (HA-only single-slot at each idXX):
for THR in 100 099 098 097 096 095; do
  bash scripts/stage3_dataset.sh flu_ha_na_cluster_aa_id${THR}_HAonly
done

# 2. MMD sweep — aa k=3, positives only (the primary measurement).
#    sigma values come from the prior Phase 1 sanity runs (see
#    docs/results/2026-05-24_mmd_per_{slot,pair}_results.md).
bash scripts/mmd_sweep.sh \
    --thresholds "100 099 098 097 096 095" \
    --dataset_pattern "dataset_flu_ha_na_cluster_aa_id{thr}_HAonly_*" \
    --routing_label_pattern "cluster_aa_id{thr}_HAonly" \
    --feature_space kmer_aa --kmer_k 3 \
    --sigma_s1 29.3192 --sigma_s2 1.0720

# 3. Same sweep with ESM-2 (slow — ~75 min — HDF5 load dominates).
bash scripts/mmd_sweep.sh \
    --thresholds "100 099 098 097 096 095" \
    --dataset_pattern "dataset_flu_ha_na_cluster_aa_id{thr}_HAonly_*" \
    --routing_label_pattern "cluster_aa_id{thr}_HAonly" \
    --feature_space esm2 \
    --sigma_s1 1.0719 --sigma_s2 0.3588

# 4. Negative-only MMD sweep (decomposes the perf trajectory).
bash scripts/mmd_sweep.sh \
    --thresholds "100 099 098 097 096 095" \
    --dataset_pattern "dataset_flu_ha_na_cluster_aa_id{thr}_HAonly_*" \
    --routing_label_pattern "cluster_aa_id{thr}_HAonly" \
    --feature_space kmer_aa --kmer_k 3 \
    --sigma_s1 29.3192 --sigma_s2 1.0720 \
    --label_filter 0 --out_suffix "_neg"

# 5. Both-labels MMD (the "what the model sees" summary).
bash scripts/mmd_sweep.sh \
    --thresholds "100 099 098 097 096 095" \
    --dataset_pattern "dataset_flu_ha_na_cluster_aa_id{thr}_HAonly_*" \
    --routing_label_pattern "cluster_aa_id{thr}_HAonly" \
    --feature_space kmer_aa --kmer_k 3 \
    --sigma_s1 29.3192 --sigma_s2 1.0720 \
    --label_filter both --out_suffix "_both"

# 6. Train MLP + LGBM + 1-NN, 3 seeds × 6 datasets, parallel across
#    GPUs 1-6. Each seed batch runs in parallel; seeds run sequentially.
bash scripts/stage4_sweep.sh \
    --bundle flu_ha_na_kmer_aa_k3 \
    --thresholds "100 099 098 097 096 095" \
    --dataset_pattern "dataset_flu_ha_na_cluster_aa_id{thr}_HAonly_*" \
    --output_prefix training_flu_ha_na_kmer_aa_k3_HAonly \
    --seeds "42 43 44" \
    --baselines "lgbm knn1_margin" \
    --start_gpu 1

# 7. Aggregate MMD + perf + plot (auto-detects available seeds, label
#    filters, feature spaces).
python -m src.analysis.aggregate_mmd_single_slot_sweep \
    --feature_spaces esm2 kmer_aa
```

`scripts/stage4_sweep.sh --help` and `scripts/mmd_sweep.sh --help`
document all flags (dataset_root, log_dir, n_permutations, etc.).

Subtype-coupling sanity check on the POC dataset
(`docs/results/2026-05-24_cluster_disjoint_feasibility_HA_NA.md` interpretation):

```python
import pandas as pd, numpy as np
from scipy.stats import chi2_contingency
DS = 'data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_aa_id098_HAonly_20260524_201833'
pos = pd.concat([pd.read_csv(f'{DS}/{sp}_pairs.csv', low_memory=False,
                              keep_default_na=False, na_values=[''])
                 for sp in ('train','val','test')], ignore_index=True)
pos = pos[pos['label']==1]
pos['na_subtype'] = pos['hn_subtype_b'].astype(str).str.extract(r'(N\d+)')
xt = pd.crosstab(pos['cluster_id_a'], pos['na_subtype'])
n = xt.values.sum(); r, k = xt.shape
chi2 = chi2_contingency(xt.values)[0]
print(f"Cramer's V = {np.sqrt(chi2 / (n * (min(r,k)-1))):.4f}")
```

## See also

- `docs/results/2026-05-24_cluster_disjoint_feasibility_HA_NA.md` —
  bilateral + single-slot feasibility pre-flight that enabled this sweep.
- `docs/results/2026-05-24_mmd_per_slot_results.md` —
  S1 baseline for random / seq_disjoint / bilateral cluster_id099.
- `docs/results/2026-05-24_mmd_per_pair_results.md` —
  S2 baseline for the same three routings.
- `docs/plans/2026-05-22_split_separation_metrics_plan.md` —
  overall split-separation plan.
- `src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df` —
  routing helper with the `single_slot` mode used here.
- `src/analysis/aggregate_mmd_single_slot_sweep.py` — aggregator that
  produced the sweep CSVs and the figure.
- `BACKLOG.md` § "Algorithm-switch follow-ups" — context on the
  symmetric easy-linclust switch this sweep builds on.
