# OOD vs random split: is the OOD split itself hard?

**Status: IN PROGRESS** — builder implemented; datasets generated + validated (deterministic) 2026-07-25; Stage-4 exploratory runs done 2026-07-26: **OOD collapses to chance across every model/feature tested** (see Stage-4 result).

## Question
At a **fixed size**, is a cluster-disjoint (**OOD**) split harder than a **random** (non-OOD) split? If
OOD is harder at matched size, the gap is the split itself, not data scarcity. Follows
`docs/results/2026-07-14_cc_ood_threshold_size_decoupling.md` (performance tracks the atom count, not
the OOD threshold).

## Design (HA-NA, nt_cds, `_ood` clusters, t095)
Edge-cut the mega-CC into atoms (`fragment_until`). At t095 this yields 125 atoms, of which the **3
largest CCs carry 95.6%** of pairs: the HA_0×NA_0 hub, a multi-cluster CC, and the HA_1×NA_1 hub
(glossary: *bipartite hub*). Build **both arms from ONE fixed pool** (positives + within-CC negatives),
differing only in the partition:
- **OOD arm** — leave-one-CC-out: each of the 3 largest CCs is the sole test fold once; the other CCs
  + the tail are train. Cluster-disjoint by construction.
- **Random arm** — the same 3-CC pairs shuffled into folds whose test sizes match the OOD folds; the
  tail is always train. In-distribution (clusters recur across folds).

**Negatives: `within_cc`** — drawn within the connected component, so cross-cluster in the multi-cluster
CC and within-cluster in the hub CCs. Generated once into the pool and **reused by both arms**, so the
only difference between arms is the partition. `m_pos_per_cc: null` (keep all pairs — recurrence is the
point); `neg_to_pos_ratio: 1`. Report per-fold (the 3 folds differ: 2 subtype-pure hubs + 1 mixed CC)
**and** the mean.

## Implementation (`src/datasets/dataset_pairs_cc.py`, `src/datasets/_cc_helpers.py`)
- `pick_largest_atoms`, `_carve_val_pairs` (pair-level val), `make_folds_leave_cc_out`,
  `make_folds_random`, `_partition_full` (arms selector). `_make_folds_for_scope` builds `full` once
  then partitions; `main` writes `out_dir/{ood,random}/fold_k/`.
- `build_cc_isolate_pool(membership_path=...)` — within_cc now runs on `_ood` clusters via
  `cluster_memb_nt_cds_ood.parquet`, and drops **edge-cut straddler** isolates (slot-a/slot-b clusters
  in different atoms) so no within-CC negative leaks a sequence across atoms.
- Bundle `flu_ha_na_cc_nt_cds_ood_ood_vs_random` (t095, `leave_cc_out`, `within_cc`, `paired_random`,
  `edge_cut.max_drop_frac 0.10`). `--override dataset.split_strategy.tail_ccs_to_train=false` measures
  the tail's effect.

## Verified (2026-07-25, deterministic rebuild)
Generated both arms to `runs/dataset_cc_nt_cds_ood_ood_vs_random_t095/{ood,random}/fold_{0,1,2}/` from ONE
fixed pool of **151,409** pairs (75,740 pos / 75,669 within-CC neg; 3,580 straddler isolates dropped; 125
atoms; test CCs = atoms 60/66/40 sized 26,611/25,179/20,608). Both arms partition the identical pool with
identical negatives; per-fold test sizes matched (**53,222 / 50,358 / 41,216**). The **only** difference is
CC recurrence across test↔train: **OOD 0** clusters / 0 sequences in every fold; **random 247–266 clusters /
36–41k sequences** (the in-distribution control) — `pair_key` overlap is 0 in both arms. Deterministic: a
fresh-process rebuild reproduced all 18 arm×fold×split assignments byte-identically (the spectral edge-cut
is now a dense eigensolve -- see `2026-07-17_2d_cc_edge_cut_fragmentation_plan.md`).

## Remaining
- **DONE** — split-geometry kmer_nt_cds UMAPs (per fold × arm × slot, 12 figs) via
  `src/analysis/umap_ood_vs_random.py` (commit `98657fd`): OOD test = one contiguous held-out CC
  (extrapolation); random test = scattered across the hubs (interpolation).
- **ON HOLD** — persist per-CC modal subtype labels (would extend `plot_cc_metadata.py` to emit a
  `cc_metadata_modal_*.csv`, reusing `_pair_helpers.pair_key_to_metadata`). Values already computed for the
  3 held-out CCs at t095: **CC1 (cc_id 60) = H3N2 99.6%**, **CC2 (cc_id 66) = avian mix (124 subtypes, no
  majority; top H5N1 30% / H9N2 12%)**, **CC3 (cc_id 40) = H1N1 99.2%**. Full per-subtype breakdown is in
  `cc_nt_cds_ood/HA-NA/t095/fragmented/figures/cc_metadata_hn_subtype_*.png`.
## Stage-4 result (2026-07-26, exploratory)
LGBM on kmer_nt_cds, all 6 t095 arm×folds (test): **OOD mean AUC-PR 0.51 / MCC 0.03 (chance) vs random
0.97 / 0.88** — same pool, same size, only the partition differs. Every OOD fold collapses, including the
124-subtype avian tangle (fold_1); the random arm is near-perfect and fold-stable. Models train fine
in-distribution (val AUC-PR ~0.96–0.98) but cannot extrapolate to a held-out CC — an OOD-specific failure,
not a training failure. Read AUC-PR/MCC, not F1 (OOD F1 is a degenerate predict-all-positive at the 0.5
cutoff).

**Robust across every axis** (fold_0 ablation): model (LGBM, MLP), threshold (t095 ≈ t099), interaction
(concat ≈ unit_diff+prod), feature (kmer_nt_cds; **ESM-2 aa** only nudges AUC-ROC 0.52→0.56 — a sliver,
still chance), scaling (none / unit_norm / standard). Conclusion: in-distribution performance is **entirely**
the cluster shortcut; no representation tested carries a generalizable HA–NA co-occurrence signal. Runs are
local artifacts under `models/flu/July_2025/runs/` (not committed); aggregator in scratchpad.

## Related
`docs/plans/2026-07-17_2d_cc_edge_cut_fragmentation_plan.md` (fragmentation mechanism);
`docs/plans/2026-07-08_single_segment_ood_clusters_plan.md` (`_ood` clusters);
`docs/plans/2026-06-09_cc_dataset_cv_plan.md` (2D-CD builder); `docs/methods/glossary.md`.
