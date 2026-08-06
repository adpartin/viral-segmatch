# CC within_fold threshold sweep on OOD clusters (nt_cds): performance tracks #atoms, not the identity threshold

**Date:** 2026-07-14. **Config:** nt_cds · HA–NA · within_fold · LGBM(concat) · m_pos=1 ·
neg_to_pos_ratio=1.0 · drop_negative_infeasible_ccs=false · 5-fold CV. Clusters: **OOD
connected-component** (`clusters_nt_cds_ood`, the across-cluster separation guarantee), via bundle
`flu_ha_na_cc_nt_cds_ood`.

The OOD re-run of `docs/results/2026-07-02_cc_within_fold_size_decoupling.md` (set-cover clusters) —
same conclusion, now on clusters where the threshold genuinely controls train/test dissimilarity.

## Question

The set-cover study found performance tracks #atoms, not `t`. But set-cover clustering is leaky
(linclust misses ~28% of ≥50%-identity pairs; easy-cluster fragments), so its "cluster-disjoint"
split was not truly OOD — lowering `t` may not have cleanly increased train/test dissimilarity. OOD
(connected-component) clusters remove that confound: across clusters, sequences are `< t` identical
by construction, so t095 is a strictly more-OOD split than t099. Does a stricter OOD threshold make
prediction harder once training size is held fixed?

## Structure: the mega-CC collapse (worse than set-cover)

2D bigraph (HA-cluster × NA-cluster; one edge per positive pair) collapse as `t` loosens:

| t | #CCs (atoms) | largest-CC share of pairs | set-cover ref |
|---|---|---|---|
| t099 | 3,350 | 85.7% | 5,716 / 65% |
| t098 | 947 | 95.1% | 1,644 / 89% |
| t097 | 387 | 97.9% | 690 / 95% |
| t095 | 108 | 98.7% | 218 / 98% |

OOD (single-linkage connected components) merges harder than set-cover, so at each `t` it leaves
**fewer independent atoms and a bigger mega-CC** — the collapse bites sooner.

## Result: still sample size, not threshold difficulty

| t | naive AUC-ROC (atoms vary) | cap387 AUC-ROC (387 atoms fixed) |
|---|---|---|
| t099 | 0.930 (3,350 atoms) | 0.776 ± 0.03 |
| t098 | — | 0.751 ± 0.03 |
| t097 | 0.756 (387 atoms)* | 0.756 ± 0.08 |
| t095 | 0.565 (108 atoms) | — |

*naive t097 = cap387 t097 (`max_atoms` is a no-op when the budget equals the atom count).

Holding #atoms fixed at 387, AUC-ROC is **flat across t099–t097** (0.776 / 0.751 / 0.756; spread
0.025 < the ±0.03–0.08 per-fold std, and not even monotonic). F1-macro likewise (0.694 / 0.656 /
0.680). The naive decline (0.930 → 0.565) is a **learning-curve in #atoms** — capping t099 from
3,350 → 387 atoms drops it 0.930 → 0.776, onto the level of t098/t097.

A cap-to-108 check (t099 vs t095 at 108 atoms) was also flat — 0.554 vs 0.565 — but at chance level
(108 atoms is data-starved), so it confirms size-dominance without resolving a threshold effect;
**cap387 is the informative probe** (real ~0.75 signal, still flat).

## Conclusion + implication

On clean OOD clusters — where `t` genuinely dials train/test dissimilarity — the identity threshold
has **no measurable effect on performance once training size is matched**. The set-cover conclusion
holds, now free of the leaky-clustering confound and at a non-starved budget: **2D-CD performance is
governed by the number of independent atoms, not the OOD threshold per se.** "More OOD ⇒ harder" is
not supported at equal training size; the apparent difficulty of low-`t` splits is a
data-availability artifact of the mega-CC collapse.

The lever is therefore #atoms, not `t`. Recovering atoms at low `t` is the path — but **2D
cut-fragmentation** (`_megacc_cut.apply_drop_budget_cut`, edge min-cut) is capped by the largest
**single-segment** cluster (an indivisible bigraph node; e.g. HA_0 nt_cds t095 ≈ 32% of HA seqs), so
**single-side (single-segment) fragmentation** is required to push below t097 at a usable sample size.

## Caveats / scope

- One config: nt_cds · HA–NA · within_fold · LGBM · m=1. May differ for aa, other pairs, within_cc,
  or other models.
- cap387 covers t099/t098/t097 (387 = t097's atom count); t095 (108) stays below the budget and can't
  join at 387.
- within_cc + OOD not tested — `_cc_helpers.build_cc_isolate_pool` reads the set-cover membership;
  running within_cc on `_ood` now raises (guard added) until the `_ood` membership is wired in.

## Artifacts

- `results/flu/July_2025/runs/cc_ntcds_ood_threshold_sweep/` — naive (t099, t095).
- `results/flu/July_2025/runs/cc_ntcds_ood_cap387_threshold_sweep/` — size-controlled (t099/t098/t097
  @ 387); summary + plot.
- `results/flu/July_2025/runs/cc_ntcds_ood_cap108_threshold_sweep/` — cap108 (t099/t095 @ 108; chance floor).
- OOD clusters: `data/processed/flu/July_2025/clusters_nt_cds_ood/{t099,t098,t097,t095}/`.
- Bundle: `conf/bundles/flu_ha_na_cc_nt_cds_ood.yaml`.

Set-cover analog: `docs/results/2026-07-02_cc_within_fold_size_decoupling.md`. Cluster-build plan:
`docs/plans/2026-07-08_single_segment_ood_clusters_plan.md`.
