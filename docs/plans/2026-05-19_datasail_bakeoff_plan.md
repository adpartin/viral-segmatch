# DataSAIL vs segmatch bicc — Bake-off plan

**Status: PAUSED after Phase 0** (decision criterion 2 met: DataSAIL C2 drop rate >50% across all configs tested; see `docs/results/2026-05-20_datasail_phase0_results.md`)
**Date:** 2026-05-19; Phase 0 results documented 2026-05-20.
**Branch:** `feature/datasail-bakeoff`
**Parent doc:** `docs/methods/leakage.md` § "Relation to prior-art split taxonomies"

## One-line framing

Empirically compare DataSAIL's `S2` routing (cluster + ILP + drop)
against segmatch's `cluster_disjoint` routing (bicc — bipartite-CC
LPT-greedy + route-atomically) on the same Flu A HA/NA positives, to
decide whether DataSAIL is a useful complement (or replacement) for
bicc in our cluster-leakage mitigation story.

## Context

`docs/methods/leakage.md` § "Relation to prior-art split
taxonomies" establishes that DataSAIL `S2` and segmatch
`cluster_disjoint id<100` share *intent* (test pairs in P&M class C3
with cluster-novel components on both slots) but differ in
*algorithm* in two ways:

1. **Route-not-drop vs drop-on-straddle.** bicc routes whole
   bipartite-CCs atomically; DataSAIL drops interactions whose two
   entities land in different folds (`_split_helpers.py:267` vs
   DataSAIL main p3).
2. **LPT-greedy heuristic vs cluster+ILP heuristic.** bicc uses a
   sort-and-pack bin-packer at the (cluster_a, cluster_b)
   bipartite-component level; DataSAIL solves an NP-hard ILP on a
   precomputed cluster set (default K=50 clusters per entity type)
   to minimize total cross-fold similarity `L(π)`.

The empirical question is whether (1) and (2) produce meaningfully
different splits / downstream generalization on our corpus. This
matters for paper writeups (defensible methodology choice) and for
tooling (do we adopt DataSAIL as a leakage-measurement primitive,
keep bicc as the production routing, or both?).

## Three workflows considered

DataSAIL operates on a similarity matrix over entities (clusters);
for 2D data it ALSO consumes interactions to decide which pairs
straddle and get dropped. Three viable workflows for hooking it
into our pipeline:

| Workflow | What DataSAIL sees | Output | Negatives handled by |
|---|---|---|---|
| **(a) Positives-only routing** | Positive pairs (entity_a, entity_b) + similarity | Per-pair fold assignment for positives (some dropped) | segmatch's `regime_aware_coverage` *after* routing, within each fold |
| **(b) Pos+neg routing together** | All pairs (pos + neg) + similarity + label as C-class | Per-pair fold assignment for both (some dropped) | DataSAIL stratifies pos:neg ratio per fold |
| **(c) Entity-list routing (1D mode)** | Just an entity list per slot, no pairs | Per-entity fold assignment | segmatch's pair construction (pos + neg) within each fold |

**First experiment will be workflow (a).** Reasons:

- Keeps segmatch's `regime_aware_coverage` negative sampler intact
  (we know it works; DataSAIL has no analog for our 8-regime
  metadata structure).
- DataSAIL only does what it's good at: routing the protein-level
  structure with leakage minimization.
- The drop rate on positives is directly interpretable as the
  feasibility cost of strict S2 splitting — the same kind of
  feasibility question we already mapped for bicc in
  `data/processed/flu/July_2025/clusters_aa/redundancy_summary.md`.

(b) and (c) are deferred until (a) is validated. (b) couples the
bake-off to whether DataSAIL handles our 8-regime structure as a
usable C-class space, which is a separate risk. (c) sidesteps the
route-vs-drop issue but loses bipartite coupling — two proteins that
pair together could end up in different folds, then those pairs are
lost downstream anyway.

## Scope of the first experiment

| Dimension | Choice | Rationale |
|---|---|---|
| Virus | Flu A | Production corpus |
| Pair | HA + NA | Matches our published cluster_disjoint baselines |
| Alphabet | aa | DataSAIL natively supports MMseqs2 for proteins (Table 2, main p8); nt support is only via CD-HIT, not MMseqs2 — would require a custom precomputed similarity matrix for apples-to-apples comparison |
| DataSAIL mode | S2 (similarity-based, 2D) | Direct analog to our `cluster_disjoint id<100` |
| DataSAIL R | R=1 (test first) | Same-entity-type pair-input data is mathematically R=1; DataSAIL paper doesn't explicitly benchmark this configuration. Worth a sanity check on a small slice before committing to a large run. |
| DataSAIL K | K=50 (paper default) | Page 5: *"the quality of the splits does not improve for K > 150 and is already good for K ≈ 50"* — start at the paper's sweet spot |
| DataSAIL C-class | Single axis (`host`) | Smallest viable stratification config. Multi-axis (host × subtype × year) deferred to a follow-up |
| DataSAIL ε, δ | 0.1 each (paper default) | Hyperparameter Table 3 default |
| Input | Positives only from one existing Stage 3 output | Workflow (a). Will pick a recent HA/NA run with `cluster_disjoint id095` already done so the bicc baseline is comparable. |
| Comparison baseline | segmatch `cluster_disjoint cluster_alphabet=aa id095` on the same positives | The most directly analogous configuration; both are "cluster sequences at threshold, then split disjointly." |

## Open questions to resolve in this experiment

The questions below cannot be answered from the DataSAIL paper alone
and need empirical answers from the first run:

1. **R=1 vs R=2 behavior for same-entity-type pair data.** DataSAIL's
   2D experiments are all drug-target (R=2 with distinct types). For
   our HA-NA pairs, R=1 (both are "proteins") is the natural formal
   framing, but the practical behavior under DataSAIL's clustering
   pipeline is untested. Verify on a small slice (≤100 isolates)
   first.
2. **K vs id-threshold calibration.** DataSAIL's K (number of
   clusters) is granularity, not threshold. The paper doesn't expose
   a "cluster at ≥X% identity" knob like our mmseqs2 `--min-seq-id`.
   Need to empirically map K to approximate aa identity on our
   corpus (e.g. "at K=50, the average within-cluster identity is
   ~Y%"). This calibration determines what we're comparing
   apples-to-apples to in our `id095` baseline.
3. **Drop rate at S2 on HA/NA aa.** memory.md "Experiment B-nt
   feasibility ceiling" shows bicc hits a mega-component collapse
   below id099. DataSAIL's drop-on-straddle policy should hit the
   same corpus structure but express it as lost pairs rather than
   routing infeasibility. The drop fraction at K=50 is the
   first-order question.
4. **C-class spec for multi-axis stratification.** DataSAIL's C is
   a single discrete class space. Single-axis (host alone) is
   straightforward. Multi-axis would require either (a) Cartesian
   product as C-classes (potentially hundreds of cells, risk of
   ILP infeasibility), or (b) σ(x) ⊆ [C] with each axis-value as a
   class (DataSAIL allows multi-class membership per data point).
   Test single axis first; decide on multi-axis after.

## Implementation sketch

The bake-off code lives in segmatch but invokes DataSAIL via
subprocess into the `datasail` conda env (since DataSAIL isn't
importable from `segmatch` — see `.claude/memory.md` § "Env
Management"). Outline:

1. **Wrapper script:** `src/analysis/datasail_bakeoff.py`
   - Reads positives CSV from an existing Stage 3 dataset directory.
   - Writes a DataSAIL-compatible input file (CSV with the required
     entity/interaction columns).
   - Subprocesses into `datasail` env to run DataSAIL via its
     Python API (or CLI — verify which is more stable).
   - Reads the fold assignment back, joins to the original positives.
   - Emits per-fold positive CSVs + a `datasail_audit.json` with
     drop counts, fold sizes, `L(π)`, and the lost-pair list.

2. **Downstream harness:** reuse Stage 4 training script with the
   DataSAIL-routed positives substituted in. Sample negatives via
   segmatch's `regime_aware_coverage` within each DataSAIL fold.

3. **Comparison aggregator:** small script `src/analysis/aggregate_datasail_vs_bicc.py`
   reads both runs' `post_hoc/metrics.csv` + per-regime CSVs and
   emits a side-by-side comparison table.

4. **Output location:** `results/flu/July_2025/runs/datasail_bakeoff/`
   (matches the aggregator-output convention from CLAUDE.md).

## Metrics to collect

| Metric | What it tells us |
|---|---|
| Drop rate (n_lost / n_input) | Feasibility cost of S2 at K=50 |
| Fold sizes vs target (80/10/10) | Did the ILP hit the ratios? |
| Scaled `L(π)` per Eq. 20 | DataSAIL's own leakage measure on its splits AND on bicc's splits (cross-applied) |
| Per-regime TPR/TNR (level1_neg_regimes) | Whether the route-vs-drop difference changes per-regime behavior |
| Aggregate AUC-ROC, AUC-PR, F1, MCC | Downstream generalization |
| 1-NN cosine margin | Leakage upper bound (memory.md 2026-05-15) — does it agree with `L(π)` on the same splits? |

Cross-applying `L(π)` (DataSAIL's leakage measure) to bicc's splits
is the cleanest "shared yardstick" — both methods can be scored on
the same metric, computed by DataSAIL's code, regardless of which
produced the split.

## Decision criteria

After workflow (a) completes:

- **DataSAIL drops are tolerable (<20%) AND downstream metrics within
  seed-noise of bicc:** use DataSAIL's `L(π)` as a reportable leakage
  metric in the paper; keep bicc as production routing because of
  no-drop guarantee.
- **DataSAIL drops are heavy (>50%):** DataSAIL hits the same
  feasibility ceiling as bicc on Flu A. Document this as a parallel
  finding to `docs/results/2026-05-15_cluster_disjoint_nt_results.md`.
  Decide whether to lower threshold or accept the drop rate.
- **Downstream metrics differ substantially in either direction:**
  root-cause analysis. Likely culprits: K-vs-identity-threshold
  mismatch, C-class stratification interaction, or pair-drop
  selection bias. Iterate on calibration before any conclusion.

## Phases

| Phase | Scope | Exit criterion |
|---|---|---|
| Phase 0 | Sanity run on ~100 isolates HA/NA aa, R=1, C=1 (no stratification), workflow (a) | **COMPLETE 2026-05-20.** Wrapper works; I2 drop rate 2-7% across K and ε, C2 drop rate 51-71% (decision criterion #2 met). Two unexpected findings: (i) DataSAIL package uses C1/C2 where paper uses S1/S2; (ii) I2 ILP has no objective function — fold sizes drift unpredictably from 80/10/10 target depending on ε. See `docs/results/2026-05-20_datasail_phase0_results.md`. |
| Phase 1 | Full HA/NA aa, R=1, C=1, workflow (a) | Full positives routed; per-fold CSVs written; bicc baseline (`cluster_disjoint id095`) run on same positives |
| Phase 2 | Full HA/NA aa, R=1, C=2 (host axis), workflow (a) | Same as Phase 1 + stratification check (host distribution preserved across folds within ε) |
| Phase 3 | Full evaluation: train MLP, LGBM, 1-NN on both DataSAIL and bicc splits; aggregate via heatmap | Side-by-side comparison ready for writeup |
| Phase 4 (optional) | R=2 trial on the same data, comparison to R=1 | Decide which framing to recommend for paper |

## Risks and what we'll do about them

- **DataSAIL Python API instability / undocumented behavior.** The
  paper cites a specific code version (GitHub `kalininalab/DataSAIL`).
  Pin a version in the `datasail` env; record commit hash in the
  audit JSON.
- **ILP solver failures or timeouts.** Page 9 reports SCIP/MOSEK/GUROBI
  used with a 1000s time limit per split. If SCIP (the free fallback)
  times out, we'd need to either reduce dataset size or get a
  GUROBI/MOSEK academic license.
- **Large drop fraction blocks Phase 1.** If S2 at K=50 drops >80% of
  positives, the experiment is uninformative — fall back to a higher
  K (more clusters, less aggressive splits) and document the
  trade-off.

## Files involved (planned)

- `src/analysis/datasail_bakeoff.py` (new wrapper)
- `src/analysis/aggregate_datasail_vs_bicc.py` (new comparison)
- `results/flu/July_2025/runs/datasail_bakeoff/` (outputs)
- `docs/results/2026-05-XX_datasail_bakeoff_results.md` (post-experiment writeup)

## See also

- `docs/methods/leakage.md` § "Relation to prior-art split taxonomies"
- `docs/methods/clusters.md` § 4.4 (bicc naming)
- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` (Experiment B = cluster_disjoint, predecessor)
- `docs/results/2026-05-15_cluster_disjoint_nt_results.md` (bicc feasibility ceiling — likely also DataSAIL's ceiling on the same corpus)
- DataSAIL paper: `refs/joeres2025_datasail.pdf` (Nat Commun 2025) + `refs/joeres2025_datasail_supp.pdf`
- Park & Marcotte: `refs/park2012_pair_input_flaws.pdf` (Nat Methods Correspondence) + `refs/park2012_pair_input_flaws_supp.pdf`
- `.claude/memory.md` § "Env Management" (datasail env setup)
