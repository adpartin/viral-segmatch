# OOD vs random split: is the OOD split itself hard?

**Status: IN PROGRESS** — builder implemented + verified 2026-07-24; Stage-4 training GATED (not launched).

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

## Verified (2026-07-24)
Built both arms (151,163 pairs). OOD arm: **0** test↔train sequence overlap in every fold; random arm:
34k–38k overlap; **identical pool and identical negatives** across arms; per-fold test sizes matched
(52,978 / 50,358 / 41,216).

## Remaining
- Subtype-label each held-out CC (H_x) via `src/analysis/bigraph_cut_subtype.py::pair_key_to_subtype`.
- Split-colored kmer_nt_cds UMAPs (per arm × slot); reuse `plot_utils.umap_scatter`.
- **(GATED)** Stage 4: train both arms, compare per-fold + mean AUC-PR/MCC over cut-seed × fold-seed.
  No launch without explicit confirmation.

## Related
`docs/plans/2026-07-17_2d_cc_edge_cut_fragmentation_plan.md` (fragmentation mechanism);
`docs/plans/2026-07-08_single_segment_ood_clusters_plan.md` (`_ood` clusters);
`docs/plans/2026-06-09_cc_dataset_cv_plan.md` (2D-CD builder); `docs/methods/glossary.md`.
