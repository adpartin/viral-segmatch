# Regime-aware coverage phase — Plan

**Status: IMPLEMENTED**
**Date:** 2026-05-14 (plan), 2026-05-15 (implemented).
**Branch:** `feature/cluster-disjoint-splits` (carried alongside the cluster-disjoint work).
**Results doc:** `docs/results/2026-05-15_regime_aware_coverage_validation.md`

## Context

The 2026-05-13 8-cell experiment matrix
(`docs/results/2026-05-13_full_8cell_matrix.md`) and the 2026-05-13 ratio=3.0
test both confirmed that regime-aware **fill** alone is a weak lever for
shifting the achieved per-regime distribution toward the configured
`regime_targets`. The dominant reason is that the **coverage phase is
regime-blind**: it walks every `(slot, seq_hash)` cell that needs ≥1 negative
and picks any valid partner uniformly at random. Since random across-isolate
partners overwhelmingly mismatch on all three axes, coverage natural-skews
heavily into `none_match` (and `year_only` once the year_bin axis is binned).

Verified numbers on `dataset_flu_ha_na_regimes_ratio3_20260513_211559`
(`negative_regime_manifest.csv`):

| Regime | Target | Coverage placed | Fill placed | Achieved | Shortfall? |
|---|---:|---:|---:|---:|---|
| `none_match`         | 7,006 | **23,583** | 0     | 23,583 | overshoot |
| `host_only`          | 7,006 | 6,116      | 3,226 | 7,006  | — |
| `host_subtype_year`  | 21,023 | 2,922     | 0     | **2,922** | supply_exhausted |

Coverage placed `none_match` at 3.4× its target *before* fill saw it; fill can
only add, never remove. By the time fill runs, `host_subtype_year` candidate
supply is depleted (most rare-cell isolates have already been used as
partners in coverage, just not for the regime that needed them).

**This plan adds regime-aware partner selection inside the coverage phase**
so coverage placements are biased toward harder regimes (where supply is
scarce), and `none_match` is filled only as needed and last. The fill phase
stays as it is — it'll naturally do less work when coverage already approximates
the target distribution.

The user's tight bundles (`flu_ha_na_tight`, `flu_pb2_pb1_tight`) recently
added `negative_sampling.regime_targets` blocks (2026-05-14), making them
the first natural targets for this new mechanism.

## Glossary

| Term | Meaning |
|---|---|
| **Cell** | An isolate's `(host, hn_subtype, year_bin)` tuple under the configured `sampling_axes`. |
| **Cell-pair regime** | The classification of a `(self_cell, partner_cell)` pair by `classify_pair_regime` — one of the 8 in `REGIME_NAMES`. |
| **Per-cell feasibility** | For a given `self_cell`, the set of regimes that have ≥1 partner cell available (i.e., at least one other cell whose pair with `self_cell` lands in that regime). Filter-dependent: a tight bundle with only 2 hosts × 2 subtypes will have several regimes infeasible for many cells. |
| **Priority chain** | An ordering of the 8 regimes for sampling preference. By default: hardest → easiest. The chain is consulted per-cell; infeasible regimes are skipped automatically. |
| **Per-regime attempt budget** | Number of sampling attempts inside one regime before the sampler falls over to the next regime in the priority chain. |
| **Last-resort fallback** | If no regime yields a valid partner within budget, the original uniform-random sampler runs to guarantee coverage (mode #2 protection — every `(slot, seq_hash)` must get ≥1 negative). |

## Decisions (locked in)

These are the design choices already committed to in the conversation that
motivated this plan.

1. **Strict priority** (not weighted random, not quota-gated) — coverage
   tries the hardest feasible regime first, falls back through the chain.
   Reason: simplest mechanism with maximum impact on hard-regime supply.
2. **Priority order** — hardest first, axis-count primary, with the
   subtype>host>year tie-break the user suggested:
   ```
   ['host_subtype_year', 'subtype_year_only', 'host_year_only',
    'host_subtype_only', 'year_only', 'subtype_only', 'host_only',
    'none_match']
   ```
3. **Opt-in flag** — new bundle knob `negative_sampling.regime_aware_coverage:
   true|false`, default **`false`**. Existing regime-aware bundles (the
   2026-05-09 design) continue to use regime-blind coverage unless the flag
   is flipped.
4. **Fill phase unchanged** — keeps its existing per-regime greedy +
   redistribute logic. When coverage over-fills hard regimes, fill's residuals
   for those regimes will be 0 and it will spend its budget on under-target
   regimes via the `redistribute` pass.
5. **No `none_match` floor on first cut** — let coverage skew as hard as
   supply allows. If post-hoc evaluation shows generalization regression on
   easy-regime negs, revisit and add a floor as a separate opt-in knob.
6. **Per-regime attempt budget = 10** — total fallback budget stays at
   `max_attempts_per_seq=50`. Chain of 8 regimes × 10 attempts = 80 worst-case;
   the last-resort uniform sampler adds another 50, but in practice coverage
   accepts long before any of these caps.

## Open questions (resolve during implementation)

- **Are there cells with zero feasible regimes other than `none_match`?**
  Likely yes on tight bundles. In that case, last-resort fallback handles them
  — but we should log how often this happens (`rejection_stats[
  'coverage_no_feasible_regime']` counter) so the user can see if the priority
  chain is being bypassed often.
- **Should DNA-level coverage (per-`dna_hash`) also be regime-aware?**
  The current coverage iterates both `(slot, seq_hash)` and `(slot, dna_hash)`.
  Regime classification operates at the isolate level (cells are
  isolate-level), so changing the iteration unit doesn't change regime logic.
  Both DNA-level and protein-level cells will benefit identically.
- **Manifest accounting** — the regime manifest already breaks `coverage_placed`
  and `fill_placed` per regime. The new code must continue to update those
  correctly so plots and the manifest remain truthful.

## Implementation phases

### Phase 1 — Pre-compute per-cell partner pools

In `_negative_regime_sampling.py` (or a new helper there), add:

```python
def build_cell_regime_partners(
    isolate_to_cell: dict,
    *,
    axes: Iterable[str],
) -> dict:
    """Returns {self_cell -> {regime -> [partner_cells]}} for fast lookup
    during regime-aware coverage. Each partner_cells list is in deterministic
    order (sorted) for reproducibility. Empty regimes are omitted from the
    inner dict.
    """
```

Cost: O(n_cells^2). With ~24 cells (3 hosts × 4 subtypes × 2 year_bins) it
takes microseconds. For the full-flu corpus where there could be 20-40 distinct
cells across 8 hosts × 18 subtypes × 3 year_bins, still trivial.

This map is reusable across all `(slot, seq_hash)` iterations.

### Phase 2 — Modify coverage loop

In `dataset_segment_pairs_v2.py:568-619` add a branch keyed on the new
`regime_aware_coverage` flag.

```python
regime_aware_cov = bool(getattr(neg_sampling_cfg, 'regime_aware_coverage', False))
priority_chain = [...]   # the locked-in list above
per_regime_budget = 10
cell_partner_map = build_cell_regime_partners(...) if regime_aware_cov else None

for (d, slot) in target_dna_iter:
    if already_covered(d, slot): continue
    self_cell = isolate_to_cell[self_aid]
    excluded = same_seq_isolates(self_seq)

    accepted = False
    if regime_aware_cov:
        for regime in priority_chain:
            partner_cells = cell_partner_map.get(self_cell, {}).get(regime, [])
            if not partner_cells: continue
            for _attempt in range(per_regime_budget):
                # Weighted choice by cell size (so larger cells aren't
                # under-sampled when a small cell is on the candidate list).
                cell_p = weighted_choice(partner_cells, weights=cell_sizes)
                other_aid = rng.choice(cell_to_isolates[cell_p])
                if other_aid in excluded: continue
                if _try_accept(self, other):
                    accepted = True
                    break
            if accepted: break

    if not accepted:
        # Last-resort uniform: identical to the current coverage logic.
        for _attempt in range(max_attempts_per_seq):
            other_aid = rng.choice(isolate_ids_list)
            ...
```

### Phase 3 — Counter for diagnostics

Add a `rejection_stats['coverage_regime_aware']` substats dict:

```python
{
  'regime_aware_attempts_per_regime': {r: int_count for r in REGIME_NAMES},
  'regime_aware_acceptances_per_regime': {r: int_count for r in REGIME_NAMES},
  'fell_back_to_uniform': int_count,   # cells where the priority chain found nothing
}
```

Surface these in `dataset_stats.json` under the existing `coverage.<split>`
section so the manifest plot in `visualize_dataset_stats.py` can show the
regime-aware coverage breakdown alongside fill placements.

### Phase 4 — Tests

Unit tests in `tests/datasets/test_regime_aware_coverage.py`:

1. **Priority order is respected.** Construct 4 isolates in 3 cells where
   `host_subtype_year` and `none_match` are both feasible from a given
   `self_cell`. Force `regime_aware_coverage=true`; verify the placed
   negative lands in `host_subtype_year` (the higher-priority regime).
2. **Fallback when only `none_match` is feasible.** Construct isolates such
   that only `none_match` is feasible for a given `self_cell`; verify the
   priority chain skips infeasible regimes and accepts `none_match`.
3. **Last-resort uniform fires when no regime has feasible+available
   partners.** Force all partner cells to be in the excluded set for a given
   `self_seq`. Verify last-resort runs and counter increments.
4. **Coverage invariant holds.** End-to-end: build a tiny dataset with
   `regime_aware_coverage=true` and verify
   `n_seqs_with_zero_negatives == 0` per split.

### Phase 5 — End-to-end validation on tight bundles

Build all four cells (the two tight bundles, with and without regime-aware
coverage):

| Cell | Bundle | regime_aware_coverage |
|---|---|---|
| A | `flu_ha_na_tight`    | false (baseline) |
| B | `flu_ha_na_tight`    | **true** |
| C | `flu_pb2_pb1_tight`  | false (baseline) |
| D | `flu_pb2_pb1_tight`  | **true** |

Stage 4: LGBM on each (4 runs).

Report:
1. **Achieved regime distribution per split** — compare A vs B and C vs D.
   Expect `host_subtype_year` share to climb materially under regime-aware
   coverage; `none_match` share to drop.
2. **Per-regime test TNR** — compare A vs B and C vs D. Hypothesis: hard-regime
   TNR rises on tight bundles where the natural distribution had very few
   `host_subtype_year` examples (tight HA/NA achieved 3.6%; PB2/PB1 tight
   achieved 3.8% under regime-blind coverage — see
   `docs/results/2026-05-14_tight_lgbm_results.md`).
3. **Aggregate metrics (F1, AUC, MCC)** — should hold or improve. If they
   regress, decision #5 (no floor) needs to be revisited.
4. **Coverage stats** — total coverage placements, fallback counts. If
   fallback fires for >5% of cells, the priority chain is leaving feasibility
   on the table — investigate.

## Acceptance criteria

The plan is **implemented** when:

- [ ] Phase 1 helper added to `_negative_regime_sampling.py` with unit test.
- [ ] Phase 2 branch in `dataset_segment_pairs_v2.py` is gated on
      `regime_aware_coverage` flag (default false; existing builds reproduce
      bit-identical results when off).
- [ ] Phase 3 diagnostic counters wired through to `dataset_stats.json` and
      visible in the regime manifest plot.
- [ ] Phase 4 tests pass (`pytest tests/datasets/test_regime_aware_coverage.py`).
- [ ] Phase 5 validation: 4 fresh datasets built, 4 LGBMs trained, side-by-side
      results written to `docs/results/2026-05-XX_regime_aware_coverage_*.md`.
- [ ] Status flipped to `IMPLEMENTED` and the plan moved to
      `docs/plans/done/`.

## Risks and how to detect them

| Risk | Symptom | Mitigation |
|---|---|---|
| Coverage runs much slower (per-regime budget × 8 regimes per cell) | Stage 3 wall-clock 5× higher | Profile in Phase 5; reduce `per_regime_budget` from 10 → 5; consider holdout weighted sampling instead of strict priority |
| Last-resort fallback fires often (cells with no feasible non-`none_match` partners) | High `fell_back_to_uniform` count in stats | Could indicate too-narrow filter; expected behavior on extreme filters (e.g. single-host bundle). Logged not raised. |
| Easy-regime TNR regresses (model never sees easy negatives during training) | Drop in `none_match` / `host_only` test TNR | Add `none_match_floor: 0.1` knob as a follow-up; coverage allocates at least 10% of placements to `none_match` regardless of priority. |
| RNG determinism breaks (different results across runs with same seed) | Stage 3 stats vary run-to-run | Use the same `rng` instance and keep weighted_choice calls deterministic; covered by Phase 4 tests |
| Interaction with `hash_key=seq` routing produces drops | `pairs_dropped > 0` in `seq_disjoint_audit.json` | Coverage placements happen *after* routing; routing already partitioned positives, coverage only chooses partners. No interaction expected. Verified by Phase 5 validation. |

## Out of scope (deferred)

- **Cluster-disjoint splits** via mmseqs2 — addresses mode #4 (cluster
  leakage), orthogonal to this plan. See
  `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`.
- **Train-time loss reweighting by regime** — an alternative path that
  doesn't touch the sampler. See discussion in the 2026-05-13 8-cell results
  doc.
- **Adaptive `per_regime_budget`** — scale budget by per-cell available count
  so common cells get fewer attempts and rare cells get more. Worth doing if
  Phase 5 shows excess wall-clock; deferred until measured.

## Cross-references

- `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md` — the
  metadata-aware fill design this plan extends.
- `docs/results/2026-05-13_full_8cell_matrix.md` — the 8-cell matrix that
  established regime-aware fill alone is insufficient.
- `docs/results/2026-05-14_tight_lgbm_results.md` — the tight-bundle
  baselines that the validation phase compares against.
- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` — cluster-
  disjoint routing (mode #4 mitigation). Sequential with this plan:
  cluster-disjoint operates at Phase 3 (positive routing); regime-aware
  coverage operates at Phase 4. Both knobs are independent — can be
  enabled separately or together.
- `docs/methods/dataset_construction_v2_workflow.md` §27 — current
  Phase 4 (coverage) + Phase 5 (fill) walkthrough.
- `docs/methods/leakage_definitions.md` mode #5 — the demographic-shortcut
  leakage this mechanism mitigates at construction time.
- `src/datasets/_negative_regime_sampling.py` — home of the cell helpers and
  REGIME_NAMES.
- `src/datasets/dataset_segment_pairs_v2.py:568-619` — the coverage loop
  to modify.
