# K-fold variance estimation — remaining work

**Status: IN PROGRESS**

The k-fold variance plan
(`docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md`)
landed its implemented core 2026-05-27: single-slot
`cluster_disjoint` k-fold via sklearn `GroupKFold` + per-fold D3
feasibility + D4 refuse menu + per-fold audit schema (Phases 1–7).
The implemented design is documented in `docs/methods/splits.md` § 2.

**This doc tracks the deferred items** — three k-fold modes that are
not yet built, the variance-source extensions, and the reporting
convention that depends on multi-fold results.

---

## Not-yet-built k-fold paths

Three of the five dispatch cells in `splits.md` § 2.2 raise
`NotImplementedError` for `n_folds >= 2`:

### 1. `seq_disjoint` k-fold

Routing: bipartite CCs as atoms (per
`_pair_helpers.py::seq_disjoint_route_pos_df`); sklearn `GroupKFold`
+ per-fold LPT-greedy + D3 + D4, same pattern as single-slot
cluster_disjoint. Effort: ~ 40 LoC plus tests.

Motivation is weak: seq_disjoint atoms on Flu A are typically small
(largest CC ~ 20 % of pairs at HA-NA `hash_key=seq`), so k-fold
feasibility is rarely the bottleneck — the variance question is less
empirically interesting than for cluster_disjoint where the atoms
are larger and the per-fold drift can swing more.

Build only if a downstream result calls for seq_disjoint variance
estimates that the existing seed-sweep cannot answer.

### 2. Bilateral `cluster_disjoint` k-fold

Atom = bipartite CC on `(cluster_id_a, cluster_id_b)`; same
GroupKFold + per-fold check pattern as single-slot.

Currently infeasible at the interesting thresholds on Flu A:
bipartite CC atoms collapse to a mega-component at most thresholds
below id099 (see `splits.md` § 1.9.2). At id100 technically feasible
(max_feasible_k = 5 on HA-NA per the feasibility CSV) but redundant
with seq_disjoint k-fold (item 1) since id100 ≈ seq_disjoint hash_key=seq.

Build only if a future corpus has a different bipartite collapse
trajectory or if cross-corpus comparison demands a "comparable
bilateral CV" estimand.

### 3. `metadata_holdout` k-fold

Currently `metadata_holdout` raises `NotImplementedError` on
`n_folds >= 2` (`dataset_segment_pairs_v2.py:2947`).

Design unfinished. Two reasonable extensions:

- **GroupKFold on isolates within the filter pool**: take the
  filter's resulting isolate sets and apply GroupKFold-on-isolates
  within them. Gives per-fold variance for the metadata-holdout
  estimand. Cleanest if the user wants "variance under this
  cross-population split."
- **Per-axis stratified KFold**: stratify on the held-out axis
  (e.g. by year bin) to produce folds that systematically vary the
  out-of-distribution direction. Different estimand: "how does
  performance vary across slices of the held-out axis."

Pick the design that matches the publication use case; build at that
point. Don't build speculatively.

---

## Variance-source extensions

The implemented k-fold path measures **variance source (i):
positive-routing variance** — different positive partitions across
GroupKFold folds, combined with training-seed variance via the
existing `stage4_sweep.sh` seed sweep (variance source (iii)). Three
other variance sources were considered and deferred:

### 4. Variance source (ii): negative-resampling

Re-roll negatives against a fixed positive partition by varying
`master_seed`. The current Stage 3 negative sampling already consumes
`master_seed` for this. Measuring (ii) is therefore "free" in the
sense that no new code is needed — just multiple Stage 3 builds with
different master seeds on the same positive partition.

Deferred because the variance estimand is **not comparable to (i)
k-fold variance**: (ii) is Monte Carlo over negative draws given one
positive partition; (i) is over partitions. Mixing them in the same
report would conflate two different variance sources and would
inflate the apparent variance compared to the genuine routing
variance the paper claims should bound.

If a downstream paper needs a combined estimand, write the math first
(per CLAUDE.md "claims match verified evidence" — no synthesizing two
variance sources into one number without justification).

### 5. E1 (LPT + atom-shuffle)

Modify the bin-packer to consume the seed and produce different
valid bin-packings per seed. Would supplement k-fold variance as a
Monte Carlo estimator with overlapping test sets.

Revisit only if k-fold variance produced by the current GroupKFold
path is too thin (e.g. N = 3 or N = 5 too few to support paper
claims). Adds ~ 30 LoC. Don't build speculatively.

### 6. E1 hybrid (GroupKFold for test + LPT for train/val)

Adds complexity for marginal balance gain over the default
GroupKFold. Revisit only if per-fold drift shows up as a real
problem in production audits.

### 7. E3 (GroupShuffleSplit)

Random group assignment without balance optimization. Overshoots
ratios badly when atoms are skewed (PB2-PB1 PB1-only at id095 with
max_atom 78 % would put that atom in the test bin most of the time).
Not useful for the segmatch atom geometry. Rejected.

### 8. N-candidate-orderings fallback for mid-build D4 failures

If the mid-build refuse rate exceeds ~ 20 % of configs in production,
add a phase that shuffles the atom list N times (e.g. N = 10), runs
GroupKFold on each, picks the partition with the best worst-fold
drift. Introduces a meta-seed dependency (the partition depends on N
and a shuffle seed), but variance semantics stay clean as long as
the chosen partition is recorded in the audit. Adds ~ 20 LoC.

Don't build speculatively — the current Phase 1 ships with
deterministic D4 refuse only. Build when the refuse rate is
empirically a problem.

---

## Composition of split modes

### 9. `cluster_disjoint(slot a)` + `seq_disjoint(slot b)`

The unconstrained-slot mitigation for single-slot cluster_disjoint.
Audit-first work (already done via `slot_leakage_summary`); compose-
if-needed depending on multi-fold results. When implemented,
integrates with the k-fold framework via an extended atom definition
(bipartite CC on `(cluster_id_a, seq_hash_b)` rather than just
`cluster_id_a`); same GroupKFold + D3 + D4 pattern, with
composed-atom feasibility computed in Phase 1 pre-flight as a
separate `max_feasible_k_composed` column.

The k-fold plan's per-fold audit schema reserves `composition_mode`
for this — currently always `null`, extending to `'cluster_a_seq_b'`
or `'cluster_b_seq_a'` if implemented (no schema migration needed).

Build when single-slot multi-fold results show the unconstrained-slot
leakage as a meaningful per-fold variance contributor (per the
reporting convention below).

---

## Alternative primitives evaluated and rejected

### 10. DataSAIL ILP path

L(π) minimizer collapsed all routings to a partition-shape constant
on Flu A — see `docs/results/2026-05-24_datasail_lpi_results.md`.
Not a viable alternative primitive on this corpus. The bicc
LPT-greedy heuristic is the chosen alternative.

Re-evaluate only if the corpus changes substantially (different
virus or different metadata structure) such that the L(π) landscape
is no longer constant.

---

## Reporting convention (RESERVED)

**Status: empty pending multi-fold results.**

Populated post-results, once Phase 7 smoke tests + first real
multi-fold sweep produce concrete numbers. The convention's emphasis
depends on whether unconstrained-slot leakage materializes as a
meaningful per-fold variance contributor:

- If leakage shifts metrics noticeably across folds (variance
  contribution exceeds e.g. 0.5 pp on F1), the convention emphasizes
  **acknowledge** ("MLP F1 = X.XX ± std under single-slot HA-only
  at id095, with NA seq_hash leakage 5.1 %; unbiased estimator would
  require composing NA seq_disjoint per item 9 above").
- If leakage is inside seed noise (variance contribution within
  per-fold std), the convention shifts to **verified empirically
  negligible** ("MLP F1 = X.XX ± std under single-slot HA-only at
  id095; NA seq_hash leakage 5.1 % confirmed within per-fold
  variance, not a metric-shift confounder").

Write the convention after the first multi-fold result lands.
Locking prose pre-results risks anchoring on the wrong frame.

---

## See also

- `docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md` —
  historical design log for the implemented core (D1–D5 mechanics,
  Phase 1–7 implementation, smoke-test results).
- `docs/methods/splits.md` § 2 — live reference for the implemented
  k-fold path (per-mode dispatch table, D3 / D4 mechanics, per-fold
  audit schema, Stage 4 integration).
