# CC within_fold threshold sweep (nt_cds) — slides

**Status: IN PROGRESS**

Branch: `feature/cc-within-fold-tsweep`. Goal: a 1–2 slide report of pair-classification
performance vs cluster identity threshold under 2D cluster-disjoint 5-fold CV, with the
sample-size confound shown explicitly.

## Config

Builder `src/datasets/dataset_pairs_cc.py`, bundle `flu_ha_na_cc_nt_cds` (HA–NA). No
builder-logic change — every knob exists.

- negative_scope=within_fold, m_pos_per_cc=1, neg_to_pos_ratio=1.0
- **drop_negative_infeasible_ccs=false** — keeps infeasible CCs as atoms; within_fold gives
  them cross-CC negatives, so the set stays 1:1 balanced and gains atoms. Verified at aa t099:
  4,099 atoms (drop=false) vs 928 (drop=true), both exactly 1:1. The on-disk sweep used
  drop=true; this is a re-run.
- Thresholds t099, t098, t097, t096, t095 (`clusters_nt_cds/tXXX`).
- Model LGBM: `train_pair_baselines.py --baseline lgbm --override training.interaction=concat`.

## Confound (the slide's point)

At m=1, #positives = #atoms = #CCs, which shrinks as t tightens. Prior nt_cds sweep
(drop=true): test AUC-ROC 0.932→0.549 while n_train 2,706→146 across t099→t095 — performance
and sample size fall together, so a bare AUC-vs-t curve conflates "harder split" with "fewer
independent units". aa structural reference: #CCs 4,099→384→20 (t099→t095→t090). drop=false
raises atoms per t but does not remove the confound.

## Steps

1. Build nt_cds within_fold drop=false, t099–t095 → `data/.../runs/dataset_cc_nt_cds_wf_drop0_tXXX`
   (distinct dir; leave the drop=true runs intact). Record #CCs + n_train per t.
2. Train LGBM per (t, fold) → `models/.../runs/cc_ntcds_wf_drop0_concat_tXXX_foldK`.
3. Aggregate: extend `src/analysis/aggregate_cc_threshold_sweep.py` to read the drop0 runs;
   plot metric vs t with #atoms (CC count) overlaid.
4. Disentangling point: subsample every t to a common atom budget (= smallest usable t's atom
   count), rebuild+retrain+re-plot. AUC still dropping at fixed #atoms ⇒ genuine split
   difficulty; flattening ⇒ size-driven. Small-N caveat.

## Reuse vs new

- Reuse: builder (unchanged), bundle `flu_ha_na_cc_nt_cds`, aggregator (extend), LGBM path.
- New: sweep driver (build+train over t×fold), nt_cds #CC-vs-t for the overlay, slide artifacts
  under `results/.../cc_ntcds_wf_drop0_threshold_sweep`.

## Out of scope

Builder-logic changes; within_cc; aa runs. Cluster regeneration deferred (only if results warrant).
