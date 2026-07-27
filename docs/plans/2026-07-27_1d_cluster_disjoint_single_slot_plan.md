# 1D cluster-disjoint (DataSAIL S1) — HA–NA single-slot CV, nt_cds / cm0

**Status: IN PROGRESS** — code landed; 10 datasets built + validated (2026-07-27); Stage-4 next.

## Goal
Does HA–NA co-occurrence generalize when ONE slot's clusters are held out (DataSAIL **S1**
= 1D cluster-disjoint), unlike the conclusive 2D-CD (**S2**) collapse to chance? **Win:**
better-than-chance on 1D-CD at high threshold, and **degradation as t decreases** (looser →
bigger clusters → harder). Substrate: flu HA–NA, **nt_cds, cm0** (linclust set-cover). cm1 is
out of scope — its transitive clusters saturate too early (t099 HA already 19% pair mass).

## Audit result (this session, verified against code)
The 1D-CD path is functional and wired: `dataset_segment_pairs.py:708-740` →
`generate_all_cluster_disjoint_cv_folds_v2`. The mature routing IS present — it lives in the
shared `_split_helpers.cluster_disjoint_route_pos_df`: GroupKFold on the constrained slot
(`:578`, no shuffle → deterministic, size-balanced folds), D3/D4 feasibility guard
(`:535`), nt_cds via `cds_dna_hash`, and constrained-side cross-split overlap asserted in
`save_split_output_v2:2394-2416`. Two differences from the 2D builder `dataset_pairs_cc.py`:
- **Negatives:** no within_cc/within_fold toggle; current path uses `create_negative_pairs_v2`
  (mandatory coverage phase, no skip flag; optional regime). → **the one change below.**
- **No `m_pos_per_cc` cap** (all positives routed, pair-mass weighted). Accepted for now.

## Design (locked)
- **Positives:** reuse `cluster_disjoint_route_pos_df(single_slot='a'|'b')` — unchanged.
- **Negatives:** **within_fold, ratio-driven, no coverage, no regime.** Reuse the existing
  primitive `within_fold_negatives` (`dataset_pairs_cc.py:394-452`): budget =
  `round(ratio × n_split_pos)`; pairs a random slot-a × slot-b hash from *that split's own
  positives*; rejects cooccur + neg-dups; nt_cds via `hash_col='cds_dna_hash'`. Both endpoints
  stay in-split → folds remain cluster-disjoint on the constrained slot. Negatives are
  regenerated **per fold** (not a shared pool), consistent with within_fold semantics.
- **k = 4** for BOTH slots across the whole sweep (pre-flight below).
- **Sweep:** t099 → t095. `neg_to_pos_ratio = 1.0`. Total pos (79,347) and neg (= ratio × pos)
  are **identical across t** — positives are threshold-independent (`dataset_segment_pairs_v2.py:2121`
  builds pos_df once; threshold enters only at routing), so degradation is purely OOD signal,
  not a size artifact.

## Pair-mass pre-flight (cm0, nt_cds, HA–NA; drift_pp = 0.05; n_pairs = 79,347, constant across t)

| t | slot a (HA) largest% / max-k@0.05 | slot b (NA) largest% / max-k@0.05 |
|---|---|---|
| t099 | 8.2% / 30 | 11.4% / 15 |
| t098 | 11.2% / 16 | 13.4% / 11 |
| t097 | 14.1% / 11 | 22.3% / 5 |
| t096 | 14.0% / 11 | 24.7% / 5 |
| t095 | 22.9% / 5 | **28.4% / 4** |

Binding = min over both slots = **4** (t095/NA). k=4 feasible for `single_slot='a'` and `'b'`
across all of t099..t095; k=5 fails D4 at t095/NA. (Pre-flight uses the bilateral module's
alphabet-aware primitives `build_isolate_pairs` + `load_cluster_lookup_for_schema`; the
`single_slot_cluster_disjoint_feasibility.py` CLI hardcodes `prot_hash` at `:84` and can't read
the nt_cds `cds_dna_hash` cluster layout — a stale-tool gap, not fixed here.)

## Implementation (LANDED 2026-07-27) — reuse-based
`negative_scope` knob (`'coverage'` default | `'within_fold'`) on `split_dataset_v2` +
`generate_all_cluster_disjoint_cv_folds_v2` (`dataset_segment_pairs_v2.py`), read from
`conf/dataset/split_strategy/cluster_disjoint.yaml` via `dataset_segment_pairs.py`.
- The within_fold branch swaps **only** the sampler inside `split_dataset_v2` — all its
  audit/stats/overlap machinery is reused. It calls `within_fold_negatives`
  (`dataset_pairs_cc.py:394`; lazy-imported to avoid the module cycle) per split, enriching from a
  df restricted to that split's isolates so the isolate-disjoint tripwire holds, and synthesizes a
  minimal reject_stats (`_within_fold_reject_stats`). Coverage path unchanged (wrapped in `else`).
- **Validated** (cm0/nt_cds/t099/single_slot=`a`/k=4): routing OK, negs ratio 1.0, isolate-disjoint
  held, HA cluster overlap 0%, NA recurs 14.6%; 10/10 existing tests pass.

## Experiments

### Datasets (BUILT 2026-07-27) — cm0/nt_cds, k=4, within_fold, ratio 1.0
10 datasets at `data/datasets/flu/July_2025/runs/dataset_1dcd_nt_cds_cm0_slot{a,b}_t{099..095}`
(bundle `flu_ha_na_1dcd_nt_cds`; t + slot varied via `--override`). Validation across all folds:
- **`total_pos = 78,764` constant** across all t and both slots (positives are threshold- and
  slot-independent; only routing changes) → Stage-4 degradation is pure OOD signal, not a size artifact.
- **Constrained-slot cross-split cluster overlap = 0 everywhere**; **k=4 feasible everywhere**,
  incl. t095/slot-b (the pair-mass boundary — its test folds spread 18.8k–22.5k, within the 5pp drift bound).
- **Unconstrained-slot cluster leakage rises monotonically as t↓**: HA-side 14.5→39.5%, NA-side 13.7→35.4%.

### Stage-4 (gated) — LGBM
Matrix: 2 slots × 5 t × 4 folds. Model: LGBM on nt_cds k-mer k6 `concat` (pre-req: the
`kmer_features_nt_cds_k6` corpus cache). At neg:pos = 1:1 the **chance floor is AUC-PR ≈ 0.50, MCC ≈ 0**
(the better-than-chance bar; no baseline arm needed for the floor). Reads: (1) does each slot clear
chance at t099? (2) mean ± range AUC-PR & MCC vs t (t099→t095) per slot — the degradation curve.
Aggregate `(slot, t) → mean±range` reusing the `test_predicted.csv` parse from `aggregate_cm_stage4.py`.
Run slot a first, then slot b.

## Caveats
- **Subtype correlation** (`docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md`): HA clusters
  are ≥95% NA-subtype-pure (Cramér's V 0.90) → holding out HA also removes NA subtypes; 1D-CD
  approaches 2D-CD as clusters grow. Degradation is expected; the open question is whether t099
  (small clusters) clears chance.
- **Negative regime differs** from the 2D within_cc collapse — 1D within_fold numbers are NOT
  directly comparable to the 2D-CD results.

## Validation
- Constrained-side cross-split cluster overlap = 0 (builder asserts); unconstrained side recurs.
- n_pairs = 79,347 stable across t; D3/D4 feasibility passes at k=4.
