# 1D cluster-disjoint (DataSAIL S1) — HA–NA single-slot CV, nt_cds / cm0

**Status: IMPLEMENTED** — code landed; 10 datasets built + validated (2026-07-27); Stage-4 run
(2026-07-27) across both slots and six further HA partners. **1D-CD clears chance at t099 (AUC-PR
0.90–0.93 vs a 0.500 floor) and degrades monotonically as `t` loosens; NA is the harder axis to hold
out than HA.** Remaining: promote the result to `docs/results/`.

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

### Stage-4 — LGBM — **DONE 2026-07-27**
Matrix: 2 slots × 5 t × 4 folds, plus an extension holding HA out against the other six partners.
Model: LGBM on nt_cds k-mer k6 `concat`. Every fold is exactly 1:1, so the **AUC-PR chance floor is
0.500** (measured: positive fraction 0.5000 in all folds).

**Both reads answered: yes at t099, and the degradation is monotone.**

AUC-PR, mean [min–max] over 4 folds:

| held out | t099 | t098 | t097 | t096 | t095 |
|---|---|---|---|---|---|
| HA (of HA-NA) | 0.918 [0.887–0.940] | 0.894 | 0.874 | 0.811 | 0.755 [0.622–0.816] |
| **NA (of HA-NA)** | 0.903 [0.881–0.925] | 0.861 | 0.815 | 0.780 | **0.623 [0.555–0.714]** |
| HA (of HA-PB2) | 0.918 | 0.893 | 0.847 | 0.804 | 0.766 |
| HA (of HA-PB1) | 0.909 | 0.898 | 0.864 | 0.793 | 0.751 |
| HA (of HA-PA) | 0.906 | 0.893 | 0.855 | 0.792 | 0.752 |
| HA (of HA-NP) | 0.910 | 0.888 | 0.865 | 0.806 | 0.786 |
| HA (of HA-M1) | 0.934 | 0.902 | 0.834 | 0.802 | 0.801 |
| HA (of HA-NS1) | 0.920 | 0.905 | 0.864 | 0.804 | 0.770 |

MCC follows the same shape — HA-of-HA-NA 0.816 → 0.454, NA-of-HA-NA 0.768 → 0.160.

**Findings.**
- **1D-CD clears chance comfortably at t099** — every family lands at 0.90–0.93 AUC-PR (MCC
  0.77–0.86) against a 0.500 floor. This is the contrast with 2D-CD under within_cc negatives, which
  sits at chance.
- **Degradation is monotone in `t` for all eight families**, as designed: looser threshold → coarser
  clusters → a more OOD split.
- **The constrained slot matters, and NA is the harder axis.** Holding out NA decays faster than
  holding out HA (0.903 → 0.623 vs 0.918 → 0.755; MCC 0.768 → 0.160 vs 0.816 → 0.454), and at
  t095/slot-b one fold reaches MCC **−0.042** — at chance. Consistent with the caveat below: HA
  clusters are ≥95% NA-subtype-pure, so constraining NA removes more of the shared signal.
- **Per-fold spread widens as `t` loosens** (HA-of-HA-NA: ±0.03 at t099 → ±0.10 at t095), which is
  what few, large held-out clusters look like.

Runs: `models/flu/July_2025/runs/lgbm_1dcd_cm0_{slota,slotb,ha_*}_t{099..095}_fold{0..3}`; figures in
`tmp/score/` via `src/analysis/score_vs_threshold.py`. **Not yet promoted to a `docs/results/` entry.**

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
