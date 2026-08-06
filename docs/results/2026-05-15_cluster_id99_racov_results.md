# regime_aware_coverage on cluster_disjoint id099 — results

**Date.** 2026-05-15.
**Scope.** Applied regime-aware coverage to the cluster_id99 bundles (full
corpus, the configurations where we previously observed the FP problem). Tested
both `neg_to_pos_ratio` values that exist for cluster_id99:

| Bundle                              | Routing            | Ratio | Coverage    |
|---|---|---|---|
| `flu_ha_na_cluster_id99`            | cluster_disjoint @id099 | 1.5 | cov-blind (baseline) |
| `flu_ha_na_cluster_id99_racov`      | cluster_disjoint @id099 | 1.5 | **racov** |
| `flu_ha_na_cluster_id99_r3`         | cluster_disjoint @id099 | 3.0 | cov-blind (baseline) |
| `flu_ha_na_cluster_id99_r3_racov`   | cluster_disjoint @id099 | 3.0 | **racov** |
| `flu_pb2_pb1_*_cluster_id99[_r3][_racov]` | parallel set on PB2/PB1 polymerase pair |

LGBM only, k-mer nt k=6, `unit_norm` + `unit_diff+prod`, single seed=42.

## Headline: FP rate drops 65–84% across all four cells

Same hard (racov) test set, both models evaluated head-to-head:

| Schema | Ratio | Mode | F1 | AUC | **AUC-PR** | Precision | Recall | MCC | **n_FP** | **host_subtype_year TNR** |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HA/NA   | 1.5 | baseline | 0.559 | 0.708 | 0.605 | 0.567 | 0.552 | 0.272 | 2,457 | 0.689 |
| HA/NA   | 1.5 | **racov** | 0.369 | 0.715 | **0.621** | 0.702 | 0.251 | 0.252 | **620**   | **0.918** |
| HA/NA   | 3.0 | baseline | 0.461 | 0.771 | 0.529 | 0.574 | 0.385 | 0.336 | 1,672 | 0.834 |
| HA/NA   | 3.0 | **racov** | 0.213 | 0.758 | 0.521 | 0.730 | 0.124 | 0.234 | **269**   | **0.974** |
| PB2/PB1 | 1.5 | baseline | 0.648 | 0.765 | 0.634 | 0.596 | 0.710 | 0.381 | 1,636 | 0.651 |
| PB2/PB1 | 1.5 | **racov** | 0.502 | 0.792 | **0.677** | 0.698 | 0.392 | 0.327 | **574**   | **0.878** |
| PB2/PB1 | 3.0 | baseline | 0.566 | 0.827 | 0.569 | 0.557 | 0.575 | 0.418 | 1,550 | 0.750 |
| PB2/PB1 | 3.0 | **racov** | 0.420 | 0.845 | **0.614** | 0.685 | 0.303 | 0.354 | **474**   | **0.919** |

![cluster_id99 racov comparison](../../results/flu/July_2025/runs/cluster_id99_racov_comparison.png)

**Headline deltas (racov − baseline, same test set):**

| Cell | ΔF1 | ΔAUC-PR | Δprecision | Δrecall | Δn_FP | ΔFPR | Δ host_subtype_year TNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| HA/NA r=1.5     | −0.190 | +0.016 | +0.135 | −0.301 | **−75 %** | **−0.210** | **+0.229** |
| HA/NA r=3.0     | −0.249 | −0.008 | +0.156 | −0.261 | **−84 %** | **−0.080** | **+0.140** |
| PB2/PB1 r=1.5   | −0.146 | +0.043 | +0.103 | −0.319 | **−65 %** | **−0.209** | **+0.227** |
| PB2/PB1 r=3.0   | −0.146 | +0.045 | +0.128 | −0.272 | **−69 %** | **−0.106** | **+0.169** |

## Interpretation

The FP question the user originally raised on the cluster_id99 datasets
(HA/NA r=1.5 baseline had 2,357 FPs at thr=0.5) is **substantially answered**
by regime-aware coverage:

- **HA/NA r=1.5: 2,457 → 620 FPs (−75 %)** at the same default threshold.
- **HA/NA r=3.0: 1,672 → 269 FPs (−84 %)**.
- **PB2/PB1 r=1.5: 1,636 → 574 FPs (−65 %)**.
- **PB2/PB1 r=3.0: 1,550 → 474 FPs (−69 %)**.

The cost is a substantial recall drop (−26 to −32 pp). The model trained on
hard-regime-dominated negatives becomes much more cautious — it has learned
that "shares host + subtype + year_bin with the other isolate" is no longer
a reliable signal for "same isolate", so it withholds positive calls when
the metadata cells overlap.

**AUC-PR is flat-to-slightly-improved** (up 1.6 / down 0.8 / up 4.3 / up 4.5 pp).
The discrimination quality is essentially preserved — what shifts is the
operating point. AUC-ROC also stays in the same neighborhood.

This matches what we observed on the tight bundles (Phase 5 of the
regime-aware coverage plan) — but the magnitude here is even larger because
the full-corpus cluster_disjoint id99 splits have a more aggressive regime
distribution under racov (76–81% of test negatives end up in
`host_subtype_year`, vs ~89% on tight bundles).

## How to read the F1 drop

The default threshold (0.5) is increasingly wrong as the training-set
positive prior drops. At ratio=1.5 the train has 40% positives; at ratio=3
it's 25%. Under racov the model learns that a "matching-metadata" pair is
NOT a positive cue — the calibrated probability for such pairs drops below
0.5 even when they are actually positive, hence the recall collapse.

Threshold tuning (val-best) would recover much of the F1, since AUC-PR is
intact. The right operational comparison is **precision at fixed-FP-rate**
or **threshold-tuned F1**, not raw F1 at 0.5.

## Comparison to the previous "ratio=3 fixes FPs?" hypothesis

The earlier ratio sweep at r=3 alone (no racov) saw FPs drop modestly
(HA/NA: 2,357 → 2,109, only −10 %) while AUC-PR went DOWN. Adding racov
delivers the FP fix the ratio bump did not:

- HA/NA r=1.5 racov beats r=1.5 baseline FPs by 75 % with AUC-PR ↑.
- HA/NA r=3 alone reduced FPs only 10 % with AUC-PR ↓.

i.e., the regime-aware coverage was the missing knob; ratio alone is the
wrong lever.

## Stage 3 cost note (and a future-work bullet)

Stage 3 build time on the full corpus jumped from ~1.5 min (regime-blind
cov on cluster_id99) to **16–60 min** with regime-aware coverage:

| Bundle | Build time | n_cells |
|---|---|---:|
| `flu_ha_na_cluster_id99_racov`        | 16 min | 2,916 |
| `flu_pb2_pb1_cluster_id99_racov`      | 22 min | 2,916 |
| `flu_ha_na_cluster_id99_r3_racov`     | 43 min | 2,916 |
| `flu_pb2_pb1_cluster_id99_r3_racov`   | 62 min | 2,916 |

Compare to the tight bundles (12 cells, ~1 min each). The slowdown is
mostly from the priority-chain re-classification per (slot, dna_hash)
cell when the partner map is large. The plan's risk register listed this
as "≤5× slowdown" — observed: roughly 10–40× on the full corpus.

Two follow-ups worth investigating:
1. The plan's risk-mitigation suggestion: **lower per-regime budget**
   (10 → 5). Most acceptances happen on the first attempt; the rest is
   wasted budget on unlucky draws. Should cut runtime roughly in half.
2. **Cache the priority chain's first successful regime per `self_cell`** —
   if cell A's first iteration accepted in `host_subtype_year`, cell A's
   second iteration likely will too. Skip the chain prefix.

Neither is required for correctness; both are pure perf optimizations.

## Artifacts

Datasets (Stage 3 outputs):
- `data/datasets/.../dataset_flu_ha_na_cluster_id99_racov_20260515_070044/`
- `data/datasets/.../dataset_flu_ha_na_cluster_id99_r3_racov_20260515_070047/`
- `data/datasets/.../dataset_flu_pb2_pb1_cluster_id99_racov_20260515_070046/`
- `data/datasets/.../dataset_flu_pb2_pb1_cluster_id99_r3_racov_20260515_070048/`

Models:
- `models/.../baseline_lgbm_flu_ha_na_cluster_id99_racov_20260515_064346/` *(typo in dir, actually at 080…)*  — see CSV for exact paths
- All listed in `results/flu/July_2025/runs/cluster_id99_racov_comparison.csv`

Plot: `results/flu/July_2025/runs/cluster_id99_racov_comparison.png`

## Caveats

- Single seed (master_seed=42); no seed-variance estimate.
- LGBM only; MLP not retrained here (the LGBM signal is consistent enough
  that the headline conclusion would not be at risk; an MLP follow-up
  would tighten the recall-precision tradeoff measurement).
- Threshold tuning on val omitted from this report (would shift F1
  upward but leaves AUC-PR unchanged).
- Full-corpus racov is computationally heavier than the tight bundles
  (see above); production use should either accept the wall-clock or
  apply the listed optimizations.

## Related

- `docs/plans/done/2026-05-14_regime_aware_coverage_plan.md` — plan being applied
- `docs/results/2026-05-15_regime_aware_coverage_validation.md` — tight-bundle validation (Phase 5 of the plan)
- `docs/results/2026-05-14_cluster_disjoint_id99_results.md` — original cluster_id99 FP problem; includes the ratio=3 ablation
- `docs/methods/leakage.md` mode #5 — the demographic-shortcut leakage racov mitigates
