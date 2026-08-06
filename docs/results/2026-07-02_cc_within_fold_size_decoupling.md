# CC within_fold threshold sweep (nt_cds): performance tracks #atoms, not the identity threshold

**Date:** 2026-07-02. **Config:** nt_cds · HA–NA · within_fold · LGBM(concat) · m_pos=1 ·
neg_to_pos_ratio=1.0 · drop_negative_infeasible_ccs=false · 5-fold CV.

## Question

Under 2D cluster-disjoint CV, test performance falls as the mmseqs identity threshold loosens
(t099→t095). Is the split getting *harder* (train/test more dissimilar), or does looser clustering
just leave *fewer independent clusters* (atoms) to train on? At m_pos=1, #positives = #atoms = #CCs,
so the two move together and the naive AUC-vs-t curve conflates them.

## Structure: the mega-CC collapse

The 2D cluster bigraph (slot-a cluster × slot-b cluster; one edge per positive pair) collapses into
a single mega-CC as t loosens:

| t | #CCs (atoms) | largest-CC share of pairs |
|---|---|---|
| t099 | 5,716 | 65% |
| t098 | 1,644 | 89% |
| t097 | 690 | 95% |
| t096 | 370 | 97% |
| t095 | 218 | 98% |
| t090 | 22 | 99% |

Cluster-disjoint CV routes whole CCs to folds, so the mega-CC swallows the independent units:
5,716→218 atoms across t099→t095 (→22 at t090). Source: `results/.../2D_cluster_sizes_nt_cds`.

## Result: it's sample size, not threshold difficulty

Naive sweep (atoms vary with t) vs two size-controlled sweeps (atoms held fixed via `max_atoms`):

| t | drop0 AUC (atoms vary) | cap690 AUC (fixed 690) | cap218 AUC (fixed 218) |
|---|---|---|---|
| t099 | 0.956 (5,708 atoms) | 0.832 | 0.575 |
| t098 | 0.903 (1,643) | 0.833 | 0.669 |
| t097 | 0.833 (690) | 0.833 | 0.573 |
| t096 | 0.689 (372) | — | 0.566 |
| t095 | 0.611 (218) | — | 0.611 |

Holding #atoms fixed, the identity threshold has **no measurable effect on AUC**: cap690 is flat at
0.83 (±0.001) across t099–t097; cap218 is flat (noisier, ~0.6) across t099–t095. The naive decline
(0.956→0.611) is a **learning-curve effect in #atoms** — read down the t099 column:

```
#atoms:  218  →  690  →  5,708
AUC:    0.575   0.832    0.956
```

Internal check: cap690-t097 = 0.833 = drop0-t097, and cap218-t095 = 0.611 = drop0-t095 — because
`max_atoms` is a no-op when the budget equals the atom count.

## Conclusion + implication

Within the tested range, 2D-CD performance is governed by the number of independent clusters
(atoms), not by the identity threshold per se. The mega-CC is the bottleneck: loosening t (for a
more-dissimilar, more-rigorous split) collapses atoms and with them apparent performance.

This **motivates cut-fragmentation** (`_megacc_cut.apply_drop_budget_cut`, BACKLOG "CC dataset CV"
#3): splitting the mega-CC back into usable atoms should recover effective sample size — and hence
performance — at low t while keeping the rigorous split.

## Caveats / scope

- One config: nt_cds · HA–NA · within_fold · LGBM · m=1. May differ for aa, other pairs,
  within_cc, or other models.
- cap690 covers t099–t097 (690 = t097's atom count); cap218 covers t099–t095. 218 atoms is a
  data-starved floor (AUC ~0.6, higher variance); 690 is the cleaner probe.
- F1-macro drifts slightly in cap690 (0.73→0.70) but AUC (threshold-free) is flat — a
  decision-threshold artifact, not a difficulty signal.
- Structure #CCs (from `bigraph_pair_feasibility`, 5,716 @ t099) and perf n_atoms (from the builder,
  5,708) differ ~0.1% — the analysis loader's coarser pair-dedup vs the builder's cds_dna_hash
  universe; identical at t095/t097. Use the builder's n_atoms for the perf axis.

## Artifacts

- `results/flu/July_2025/runs/cc_ntcds_wf_drop0_threshold_sweep/` — naive sweep (atoms vary).
- `results/flu/July_2025/runs/cc_ntcds_wf_cap690_threshold_sweep/` — fixed 690 atoms.
- `results/flu/July_2025/runs/cc_ntcds_wf_cap218_threshold_sweep/` — fixed 218 atoms.
- `results/flu/July_2025/runs/2D_cluster_sizes_nt_cds/` — CC-count-vs-t + mega-CC structure.

Plan: `docs/plans/done/2026-07-01_cc_within_fold_threshold_sweep_plan.md`.
