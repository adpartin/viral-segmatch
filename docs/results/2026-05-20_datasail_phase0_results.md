# DataSAIL vs segmatch bicc — Phase 0 results

**Date.** 2026-05-20.
**Scope.** Phase 0 sanity run of the bake-off plan
(`docs/plans/2026-05-19_datasail_bakeoff_plan.md`) on 100 HA/NA Flu A
isolates. Goal: validate the wrapper script, measure drop rate, and
verify the round-trip back to segmatch format works.
**Dataset.** Subsampled from
`data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_racov_20260515_070044`
(positives only): 100 isolates → 100 positive pairs, 98 unique slot_a
proteins (HA), 99 unique slot_b proteins (NA).
**Tool versions.** DataSAIL 1.x from the `datasail` conda env built
2026-05-19 (kalininalab + conda-forge + bioconda channels); SCIP 9.x
as the ILP solver; mmseqs2 invoked internally by DataSAIL for protein
clustering.
**Wrapper.** `src/analysis/datasail_bakeoff.py` (runs in the
`datasail` env, not in `segmatch`).
**Status: COMPLETE.** Decision: pause the bake-off, document findings.
The plan's 20% drop threshold (decision criterion) is exceeded by
DataSAIL's similarity-based mode at every config tested in Phase 0.

## TL;DR

DataSAIL's similarity-based 2D mode (`C2` in the package, `S2` in
the paper) is **structurally a poor fit for our HA/NA Flu A corpus
at the scale we care about**. On 100 positives, C2 drops 51–71% of
pairs at every K and ε we tested, sometimes outright fails (solver
returns `None`), and produces unpredictable entity-fold distributions
(e.g., the test fold ends up at 39% of entities when the target is
10%). The corpus's cluster-collapse property (memory.md "Experiment
B-nt feasibility ceiling": >80% of pairs in a single bipartite
mega-component below id099 at the full-corpus scale) appears to
manifest in DataSAIL as massive pair drops at this sub-sample scale
too.

DataSAIL's identity-based 2D mode (`I2`) is much better-behaved
(2–7% drop rate) but is the algorithmic analog of our
`seq_disjoint hash_key=seq`, not of `cluster_disjoint id<100`. So
I2 doesn't measure the cluster-leakage axis we are trying to compare.

The bake-off as planned (DataSAIL C2 head-to-head against bicc
`cluster_disjoint id095` on the same positives) is therefore not the
right comparison structure on this corpus. Recommendations below.

## Findings

### Drop rate and fold distribution

Each row is a single DataSAIL run with `splits=[0.8, 0.1, 0.1]`,
`solver=SCIP`, `max_sec=300`, and N=100 input positives.

| Mode | K | ε | Pair drop | HA entity tr/v/te | Pair tr/v/te | Pair frac (of assigned) |
|---|---|---|---|---|---|---|
| I2 | 5–50 | 0.50 | 7 (7%) | 73/13/13 | 70/11/11 | 0.76 / 0.12 / 0.12 |
| I2 | 5–50 | 0.20 | 6 (6%) | 57/21/21 | 54/20/19 | 0.58 / 0.22 / 0.20 |
| I2 | 5–50 | 0.10 | 6 (6%) | 51/24/24 | 49/22/22 | 0.53 / 0.24 / 0.24 |
| I2 | 5–50 | 0.05 | 4 (4%) | 49/26/26 | 48/24/23 | 0.51 / 0.26 / 0.24 |
| I2 | 5–50 | 0.01 | 2 (2%) | 47/27/27 | 46/26/25 | 0.47 / 0.27 / 0.26 |
| C2 | 10 | 0.50 | **65 (65%)** | 42/19/39 | 28/2/4 | 0.82 / 0.06 / 0.12 |
| C2 | 10 | 0.20 | 51 (51%) | 63/18/18 | 43/5/0 | 0.90 / 0.10 / 0.00 |
| C2 | 10 | 0.10 | 69 (69%) | 46/30/24 | 20/5/5 | 0.67 / 0.17 / 0.17 |
| C2 | 10 | 0.05 | **FAILED** | — | — | (SCIP returned `None`; wrapper raised `'NoneType' object is not subscriptable`) |
| C2 | 50 | 0.50 | 56 (56%) | 64/13/22 | 37/1/5 | 0.86 / 0.02 / 0.12 |
| C2 | 50 | 0.20 | 57 (57%) | 53/24/22 | 31/7/4 | 0.74 / 0.17 / 0.10 |
| C2 | 50 | 0.10 | 71 (71%) | 45/30/26 | 18/6/4 | 0.64 / 0.21 / 0.14 |
| C2 | 50 | 0.05 | 69 (69%) | 41/26/34 | 17/6/7 | 0.57 / 0.20 / 0.23 |

Observations:

- **I2 has no objective function.** Source code in
  `solver/id_2d.py` calls `solve(1, constraints, ...)` — the
  optimization is "minimize the constant 1," i.e., find any feasible
  point. The constraint is a per-fold lower bound only
  (`compute_limits` in `solver/utils.py:30`:
  `[int(split * (1 - ε) * total) for split in splits]`). With no
  preference toward 80/10/10, the solver returns whatever feasible
  assignment it hits. This explains why I2 entity-fold sizes drift
  toward 50/25/25 (equal-ish split) when ε is tight, and toward
  73/13/13 (target-like) only when ε is loose enough to give the
  solver freedom but not so loose that any solution would do.

- **C2 has an objective** (minimize total cross-fold cluster
  similarity), but on our corpus it tends to find solutions that
  satisfy the objective by **dropping pairs aggressively**. The
  bipartite-component structure of HA/NA Flu A is such that any
  cluster-cluster routing that respects the size constraint must
  separate clusters that participate in many shared pairs, and those
  shared pairs become "lost samples."

- **The K parameter does not matter for I2 at this scale.** Sweep
  K=5,10,20,50 returned bit-for-bit identical fold assignments. For
  I2 there is no clustering — entities go directly into the ILP, so
  K is unused.

- **K does matter for C2** (different drop rates and fold sizes at
  K=10 vs K=50), but neither value yields a workable split.

- **DataSAIL "S2" is renamed "C2" in the package.** The paper uses
  `S1`/`S2` (similarity-based 1D/2D); the implementation uses
  `C1`/`C2` (cluster-based 1D/2D). Confirmed in
  `datasail/settings.py:TEC_C2 = SRC_CL + DIM_2`. Passing the paper
  name `'S2'` produces a confusing error
  (`"Technique S2 is not a two-dimensional technique"`).

- **One pair is silently missing from every result dict** (not
  marked `'not selected'`, just absent). Reproducible at every K
  and ε; same pair (`a3c7dd14fcc15358` × `87a4171e5d4b0220` in our
  subsample). Likely an artifact of the `solver/overflow.py`
  pre-assignment, which sorts entities by weight and reroutes
  "overflow" ones before the ILP runs. Not investigated further at
  this stage.

### Why C2 drops so much: corpus structure

Our prior cluster-disjoint feasibility study
(`data/processed/flu/July_2025/clusters_aa/redundancy_summary.md` §
feasibility table, and `docs/results/2026-05-15_cluster_disjoint_nt_results.md`)
established that the Flu A HA/NA bipartite-cluster graph collapses
into one mega-component containing >80% of pairs below id099 on the
full corpus. The 100-isolate sub-sample preserves this density: a
small number of large clusters dominate, and any cluster-disjoint
routing forces large fractions of pairs into the boundary region.

Our `cluster_disjoint` routing handles this by routing whole
bipartite-CCs atomically — the mega-component is assigned to one
split (typically train) and the smaller components fill val/test.
DataSAIL's C2 instead drops the pairs that straddle. So the same
corpus property that limits our feasibility ceiling
to id099 manifests as ~60% pair drops in DataSAIL C2.

This is the **route-vs-drop tradeoff** from
`docs/methods/leakage.md` § "Relation to prior-art split
taxonomies", quantified empirically for the first time on our
corpus.

### Sanity check: would scaling to Phase 1 help?

Unlikely. The drop rate is driven by the bipartite-cluster
mega-component, which scales with corpus density, not corpus size.
At full scale (~58K positives) the largest component is ~20% of pairs
on HA/NA at id100 (memory.md "seq_disjoint scales to conserved
proteins"), and grows with looser thresholds. DataSAIL's C2 would
drop everything that straddles this structure.

That said, the 100-isolate result is a **small-sample observation**.
A targeted Phase 1 run at K=50, ε=0.5 (the best-behaved config in
Phase 0) would settle the question rigorously. The compute cost is
low (DataSAIL ran in seconds per config at this scale; even
30–60 minutes per config at full scale is acceptable).

## What this means for the bake-off plan

Per the plan's decision criteria
(`docs/plans/2026-05-19_datasail_bakeoff_plan.md` § "Decision
criteria"):

> **DataSAIL drops are heavy (>50%):** DataSAIL hits the same
> feasibility ceiling as bicc on Flu A. Document this as a parallel
> finding to `docs/results/2026-05-15_cluster_disjoint_nt_results.md`.

Phase 0 satisfies this branch. The bake-off in its original form
(DataSAIL C2 routing on Flu A HA/NA, head-to-head with bicc
`cluster_disjoint id095`) does not yield a useful comparison: the
two algorithms are addressing different operational tradeoffs of
the same fundamental corpus structure (drop pairs vs route
mega-components).

## Recommended next steps

In rough priority order:

1. **Use DataSAIL's `L(π)` leakage measure as a shared yardstick on
   bicc splits.** This is the "shared metric" idea from the plan's
   metrics table. We can compute `L(π)` on any split (bicc-produced
   or otherwise) and report it as a leakage score independent of
   the algorithm that produced the split. Concretely: take an
   existing `cluster_disjoint id095` dataset, expose its train/val/
   test assignment to DataSAIL, and compute `L(π)`. Compare to the
   `L(π)` DataSAIL would have achieved on the same positives via
   its own routing (Phase 0 already gives us this number).

2. **A targeted full-scale C2 confirmation.** Run DataSAIL C2 once at
   K=50, ε=0.5 on the full HA/NA positives set. If the drop rate
   holds at ~60% (as predicted by the mega-component analysis),
   we have a strong publishable finding. If it changes substantially,
   we learn something about scale dependence.

3. **I2 vs `seq_disjoint hash_key=seq` comparison.** This is a
   cleaner head-to-head than C2 vs `cluster_disjoint id095`. Both
   are identity-based 2D, both should give similar drop rates and
   similar downstream performance. Mainly serves as a sanity check
   that DataSAIL and bicc converge when the algorithmic difference
   (route vs drop) is small.

4. **Investigate the C2 SCIP-returns-`None` failure mode.** Phase 0
   hit one outright crash (K=10, ε=0.05). Worth understanding before
   any paper claim. Could be a SCIP licensing issue, a numerical
   infeasibility, or a wrapper bug.

5. **Pivot toward paper writing.** The Phase 0 result is itself a
   paper-worthy data point: "We attempted to compare our routing
   to DataSAIL's S2 (paper) / C2 (package) and found that DataSAIL
   drops 60% of pairs on our corpus due to the cluster-collapse
   structure; bicc routes the same pairs without drop." That's a
   useful position for the methods section and could close out the
   bake-off thread without a deeper experiment.

## Files referenced

- `src/analysis/datasail_bakeoff.py` — Phase 0 wrapper
- `results/flu/July_2025/runs/datasail_bakeoff/phase0_20260519_235443/` —
  original Phase 0 output (I2, K=50, ε=0.1)
- `/tmp/datasail_K_sweep.py`, `/tmp/datasail_eps_sweep.py`,
  `/tmp/datasail_c2_test.py` — ad-hoc diagnostic scripts (not
  in repo)

## See also

- `docs/plans/2026-05-19_datasail_bakeoff_plan.md` — parent plan
- `docs/methods/leakage.md` § "Relation to prior-art split taxonomies"
- `docs/methods/clusters.md` § 4.4 (bicc naming)
- `docs/results/2026-05-15_cluster_disjoint_nt_results.md` (bicc feasibility ceiling)
- `data/processed/flu/July_2025/clusters_aa/redundancy_summary.md` (corpus redundancy structure)
- DataSAIL paper: `refs/joeres2025_datasail.pdf` and `refs/joeres2025_datasail_supp.pdf`
- DataSAIL source (in conda env): `/homes/apartin/miniconda3/envs/datasail/lib/python3.10/site-packages/datasail/`
