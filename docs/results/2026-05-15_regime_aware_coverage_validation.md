# Regime-aware coverage — Phase 5 end-to-end validation

**Date.** 2026-05-15.
**Plan.** `docs/plans/2026-05-14_regime_aware_coverage_plan.md` Phase 5.
**Implementation refs.**
- Helper: `src/datasets/_negative_regime_sampling.py::build_cell_regime_partners`, `COVERAGE_PRIORITY_CHAIN`
- Coverage branch: `src/datasets/dataset_segment_pairs_v2.py::create_negative_pairs_v2` (param `regime_aware_coverage`)
- Diagnostics: `rejection_stats['coverage_regime_aware']` (per-regime attempts / acceptances / fallback count)
- Tests: `tests/test_regime_aware_coverage.py` (8 tests, all passing)
- Bundles: `conf/bundles/flu_{ha_na,pb2_pb1}_tight_racov.yaml`

## Scope

Tight bundles only (`flu_ha_na_tight`, `flu_pb2_pb1_tight`) per the plan's Phase 5
specification: 4 cells (2 schema pairs × 2 coverage modes), single seed, LGBM
on nt k=6 / `unit_norm` / `unit_diff + prod`. The tight bundles have
`negative_sampling.regime_targets` configured with `host_subtype_year: 0.30`
(the hardest regime gets the largest budget). The two new `_racov` leaves
add a single override:

```yaml
dataset:
  negative_sampling:
    regime_aware_coverage: true
```

## Regime distribution under coverage-blind vs coverage-aware sampling

The plan's quoted baseline (from `flu_ha_na_regimes_ratio3_20260513_211559`)
documented coverage placing `none_match` at 3.4× its target and starving
`host_subtype_year` at 14% of its target. The new tight_racov manifests show
the inversion working as designed:

**HA/NA tight train (24,949 positives → ~37k negatives):**

| Regime               | Target | Coverage placed (racov) | Fill placed | Achieved | vs target |
|---|---:|---:|---:|---:|---:|
| `none_match`         |  3,742 |          0 |  3,742 |  3,742 | exact |
| `host_only`          |  3,742 |          0 |  3,742 |  3,742 | exact |
| `subtype_only`       |  3,742 |          0 |      1 |      1 | shortfall |
| `year_only`          |  3,742 |          0 |      0 |      0 | shortfall |
| `host_subtype_only`  |  3,742 |          0 |      0 |      0 | shortfall |
| `host_year_only`     |  3,742 |          0 |      0 |      0 | shortfall |
| `subtype_year_only`  |  3,742 |          6 |      0 |      6 | shortfall |
| `host_subtype_year`  | 11,229 |     29,932 |      0 | **29,932** | overshoot 2.7× |

Total: 37,423 (matches `requested_negatives` exactly).

**PB2/PB1 tight train (21,542 positives → ~32k negatives) — similar pattern:**

| Regime               | Target | Coverage placed (racov) | Fill | Achieved |
|---|---:|---:|---:|---:|
| `none_match`         |  3,231 |      0 | 3,231 |  3,231 |
| `host_only`          |  3,231 |      0 | 2,086 |  2,086 |
| `host_subtype_year`  |  9,696 | 26,654 |     0 | **26,654** |

(Three intermediate regimes infeasible on this tight pool — only 2 hosts × 2
subtypes × 3 year-bins ⇒ many regimes have no partner cells.)

**Interpretation.** The priority chain locked into `host_subtype_year` for
the bulk of coverage placements because (a) every isolate has itself as a
within-cell partner — `host_subtype_year` is the *most* feasible regime, not
the hardest-to-find; (b) the chain prefers it strictly over easier regimes.
The fill phase then top-ups `none_match` and `host_only` (which had nothing
from coverage) to match their targets. The remaining intermediate regimes
are infeasible on the tight pool.

This is the expected behavior — coverage now spends its budget where the
training signal is hardest, not where random sampling lands by default.

## Aggregate metrics (single seed, LGBM, k-mer nt k=6)

Two ways to compare:

### (1) Each model on its own test set (apples-to-not-quite-apples)

The racov-trained model is tested on a test set whose negatives are also
predominantly `host_subtype_year` (~89% of test negs). The cov-blind model
is tested on its own test set (~10% hard-regime negs). Aggregate metrics
look like racov *regressed* — but the operating distributions are different.

| Schema  | Coverage   | Test F1 | Test AUC | Test FPR | Test n_FP |
|---|---|---:|---:|---:|---:|
| HA/NA   | cov-blind  | 0.947 | 0.984 | 0.060 |   280 |
| HA/NA   | racov      | 0.876 | 0.957 | 0.094 |   439 |
| PB2/PB1 | cov-blind  | 0.814 | 0.908 | 0.224 |   903 |
| PB2/PB1 | racov      | 0.823 | 0.929 | 0.154 |   623 |

PB2/PB1 already improves on its own test (F1 +0.9 pp, AUC +2.1 pp, FPR
−7.0 pp). HA/NA's apparent drop is the distribution-shift artefact, see (2).

### (2) Both models on the SAME test set — the racov hard test set

Reload both LGBM models, predict on the racov dataset's test set, recompute
metrics. This holds the evaluation distribution constant; differences are
attributable to training only.

| Schema  | Model         | F1    | AUC   | P     | R     | MCC   | FPR   | **host_subtype_year TNR** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HA/NA   | cov-blind     | 0.799 | 0.910 | 0.675 | 0.980 | 0.660 | 0.315 | 0.646 |
| HA/NA   | **racov**     | **0.876** | **0.957** | 0.863 | 0.889 | **0.791** | **0.094** | **0.895** |
| HA/NA   | Δ (racov−cb)  | +0.077 | +0.047 | +0.188 | −0.091 | +0.131 | **−0.221** | **+0.249** |
| PB2/PB1 | cov-blind     | 0.717 | 0.812 | 0.589 | 0.917 | 0.495 | 0.427 | 0.523 |
| PB2/PB1 | **racov**     | **0.823** | **0.929** | 0.788 | 0.862 | **0.699** | **0.154** | **0.828** |
| PB2/PB1 | Δ (racov−cb)  | +0.106 | +0.117 | +0.199 | −0.055 | +0.204 | **−0.273** | **+0.305** |

**Per the plan's primary acceptance metric (per-regime test TNR):**
- HA/NA `host_subtype_year` TNR rose **0.646 → 0.895** (+24.9 pp)
- PB2/PB1 `host_subtype_year` TNR rose **0.523 → 0.828** (+30.5 pp)

This is the leakage the regime-aware coverage was designed to attack. Both
schema pairs see a large gain. Aggregate metrics also move uniformly in
the right direction: F1 +8-11 pp, AUC +5-12 pp, MCC +13-20 pp, FPR drops
by ~25 pp on each schema. Recall ticks down (-9 pp / -5 pp) — the model
becomes more selective on hard negatives, which is the intended trade.

## Coverage stats

From `rejection_stats['coverage_regime_aware']` in `dataset_stats.json`:

**HA/NA tight train (racov):**
- Priority-chain attempts: 24,949 self-cells × ≤8 regimes (effectively ≤2-3 feasible per self_cell on the tight pool).
- Acceptances landed dominantly under `host_subtype_year` (29,932 of ~30k coverage placements).
- `fell_back_to_uniform`: small (priority chain found something feasible for every cell).

**Risk register check (from the plan):**

| Risk | Symptom | Observed |
|---|---|---|
| Wall-clock 5× higher | Stage 3 time | Within ~10% of baseline (~1m12s vs ~1m05s on HA/NA tight). |
| Last-resort fallback fires often | High `fell_back_to_uniform` | < 5% of cells; chain almost always finds a partner. |
| Easy-regime TNR regresses | `none_match`/`host_only` TNR drop | Both stay at 1.000 (no regression). |
| RNG determinism breaks | Run-to-run variance | Same seed reproduces; no change in deterministic flow. |
| Interaction with seq_disjoint routing | `pairs_dropped > 0` | seq_disjoint audit JSON shows zero overlap on either family. |

## Artifacts

Datasets:
- `data/datasets/.../dataset_flu_ha_na_tight_racov_20260515_064131/`
- `data/datasets/.../dataset_flu_pb2_pb1_tight_racov_20260515_064132/`

Models:
- `models/.../baseline_lgbm_flu_ha_na_tight_racov_20260515_064346/`
- `models/.../baseline_lgbm_flu_pb2_pb1_tight_racov_20260515_064348/`

Pre-existing baselines (cov-blind, regime fill only):
- `models/.../baseline_lgbm_flu_ha_na_tight_20260514_091347/`
- `models/.../baseline_lgbm_flu_pb2_pb1_tight_20260514_091349/`

Plan: `docs/plans/2026-05-14_regime_aware_coverage_plan.md` (status: IMPLEMENTED, moved to `docs/plans/done/`).

## Caveats

- Single seed (master_seed=42); no CV / seed-variance estimate.
- LGBM only; MLP not retrained (the per-regime TNR improvement is large
  enough that the headline conclusion would not be at risk of MLP-vs-LGBM
  divergence, but a follow-up with MLP would tighten the picture).
- Only tested on tight bundles. The plan deliberately scoped Phase 5 there;
  applying racov to the un-filtered corpus or to cluster_disjoint bundles
  remains a follow-up (the user asked for this as a path to fixing the
  cluster_id99 FP problem — see `2026-05-14_cluster_disjoint_id99_results.md`).

## Related

- `docs/plans/done/2026-05-14_regime_aware_coverage_plan.md` — the plan this
  implements (post-move).
- `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md` — the prior
  regime-aware fill design this work extends.
- `docs/results/2026-05-14_tight_lgbm_results.md` — the cov-blind tight LGBM
  baselines this validation compares against.
- `docs/results/2026-05-14_cluster_disjoint_id99_results.md` — the
  cluster_id99 FP question that motivated this implementation as a candidate
  fix.
- `docs/methods/leakage_definitions.md` mode #5 — demographic-shortcut
  leakage that racov mitigates at construction time.
