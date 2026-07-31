# Bigraph / cluster-disjoint code consolidation

**Status: IN PROGRESS**

Date: 2026-07-30

Scope: consolidate the code that builds the **cluster-level bigraph** and fragments it by **edge
min-cut**, remove stale analysis scripts, and bring the production split path in line with the
project's coding conventions. Covers `src/datasets/` (the splitters) and the `src/analysis/bigraph_*`
family.

Related: `docs/plans/2026-07-21_cc_structure_prestep_plan.md` (the `cc_{source}` artifacts these
analyses should read), `docs/methods/glossary.md` (canonical terms).

---

## 1. Decisions (locked — do not re-litigate)

**Layering.** "Production" means code that participates in producing train/val/test splits,
regardless of directory. It may live in `src/datasets/` or `src/utils/`; `src/analysis/` may import
from it, never the reverse. Testable form:

> No module in `src/datasets/` or `src/utils/` may import from `src/analysis/`.

**Terminology.** Use **bigraph**, never "bipartite". Glossary changes land before code changes.

**CC builders.** Two siblings in `_pair_helpers.py`, chosen by what the nodes are:
- `cluster_ccs` — cluster-level bigraph (nodes = mmseqs clusters), networkx via `build_pair_bigraph`.
- `sequence_ccs` — sequence-level bigraph (nodes = sequence hashes), union-find (~3x faster at
  78k-123k nodes). Owns the `hash_key` families; its `col_a`/`col_b` mode is test-only.

**Naming inside vs across layers.** CC functions return **`cc_id`** (they compute components).
`_split_helpers` assigns **`atom_id`** (the routing unit: a CC in 2D-CD, one slot's cluster in
1D-CD). Summary keys use the mode-neutral `*_atom*` form so both branches share one schema.

**CC label ordering.** `(-pair_count, min node id)` — order-invariant. Required because
`_lpt_bin_pack` sorts atoms by `(-size, cc_id)`, so the id decides placement among equal-sized CCs.

**Archiving.** A `docs/results/` reference is *not* a reason to keep code — those docs are the
historical record. Archive to `src/archive/` via `git mv`; never import from it.

**seq_disjoint** is retained as an option and supports nt_cds via the `cds` hash family.

## 2. Done

- Archived `bigraph_pair_metadata`, `bigraph_pair_feasibility`, `bigraph_reassort_check`,
  `bigraph_cut_subtype` to `src/archive/` (+ README). Ported `bigraph_pair_2d` off the archived
  helper and `bigraph_cc_count_vs_threshold` onto `cc_summary.json`.
- `seq_disjoint` `hash_key='cds'`; `HASH_FAMILY_ALPHABET` as the single token→alphabet map;
  `_ensure_cds_dna_hash` shared by the two routes needing CDS-DNA identity.
- `build_pair_bigraph`: unique-index assert + phase comments. `attach_cluster_ids`: `Args:`, split
  merges, historical marker removed.
- `bipartite_components` → `cluster_ccs` + `sequence_ccs`; 5 cluster-disjoint call sites redirected;
  summary keys renamed; equivalence + order-invariance tests added.

## 3. Validation record

The label-ordering change is **partition-preserving, not split-preserving**. Measured impact
depends entirely on the fold-assignment strategy:

| config | strategy | test atoms chosen by | impact |
|---|---|---|---|
| `cc_nt_cds_cm0_wf` t099 | `groupkfold` | id order via `(-size, cc_id)` | **72% of pairs changed fold**; 4-fold mean test F1 +0.034 — inside the baseline's own between-fold sd (0.034) |
| `cc_nt_cds_ood_ood_vs_random` t095 | `leave_cc_out` | **size** (`pick_largest_atoms`) | **positives identical in all 6 folds**; only negatives resample (RNG walks CCs in id order) |

Cause of the 72%: 99% of the 5,731 atoms share a pair count, so `cc_id` is the tie-break for
essentially all of them. The prior labelling came from union-find root order and was itself
row-order dependent (99.4% churn under a row shuffle), so it was never a stable baseline.

**LGBM test metrics, `ood_vs_random` t095** (positives identical, so every delta below is
negative-resampling variation alone):

| arm | F1 | F1-macro | AUC-ROC |
|---|---|---|---|
| random | 0.9419 → 0.9489 (+0.007) | 0.9391 → 0.9470 (+0.008) | 0.9802 → 0.9823 (+0.002) |
| ood | 0.4583 → 0.2992 (**−0.159**) | 0.4780 → 0.4400 (−0.038) | 0.5258 → 0.4898 (−0.036) |

The random arm is stable. The **ood arm sits at chance in both versions** (AUC-ROC 0.53 → 0.49),
where F1 is highly unstable — its −0.159 is dominated by one fold (−0.34) and is not a capability
change. Open control: re-run the ood arm at several negative-sampling seeds to establish its
natural spread, so this delta can be attributed rather than assumed.

Baseline digests: `tests/golden/megacc_cut/fold_baseline_*.json`, captured by
`scripts/capture_2dcd_fold_baseline.py`.

## 4. Remaining work (in order)

1. **Close the last layering violation.** `_cv_sampling.assign_atoms` produces GroupKFold splits
   (production) but lives in `src/analysis` and imports `bigraph_min_cut`. Move
   `fragment_weighted` / `uniform_targets` / `lpt_max_drift` into `_megacc_cut`, relocate
   `_cv_sampling` to `src/datasets/`. Unblocks retiring `bigraph_min_cut`.
2. **Hygiene.**
   - `_megacc_cut` module docstring, "Dependency note" — claims the bisection core is duplicated
     for the analysis diagnostics; it is not (they import `_bisect` from here).
   - `cluster_pair_weight_topk.load_pair_universe` dedups on `prot_hash` for *every* alphabet →
     nt_cds analyses undercount by 25% (58,826 vs 78,764). Guard or fix.
   - `--clusters_nt` defaults to `clusters_nt`, which does not exist on disk (the real dirs are
     `clusters_nt_cds*`), in 5 live scripts: `bigraph_min_cut`, `bigraph_hub_peel`,
     `bigraph_properties`, `cluster_pair_weight_topk`, `cluster_analysis_summary`.
   - `load_cluster_map` defined identically in `bigraph_properties` and `cluster_pair_weight_topk`.
3. **`bipartite` → `bigraph` pass.** Glossary first (`Bipartite multigraph` → `Multigraph bigraph`,
   `Bipartite hub` → `Bigraph hub`), then `build_bipartite_multigraph` (18 uses) and the
   `bipartite_largest_pct_vs_threshold` output filename. **Blocked on 6.1.**
4. **Gen-2 ports** of the surviving `bigraph_*` scripts onto `cc_{source}` artifacts:
   `bigraph_properties` (per-CC λ / bridges / cut nodes / hub Gini), `bigraph_hub_peel` (node-peel —
   the only implementation of that route), `bigraph_min_cut`, `bigraph_pair_2d` (no Gen-2
   equivalent exists).
5. **Reconcile** `docs/plans/2026-07-21_cc_structure_prestep_plan.md` — its §7 build order is
   complete; either close it or record what remains.

## 5. Walkthrough (understanding pass, interleaves with the above)

`build_pair_bigraph` done. Remaining: `_bisect` → `fragment_largest_cc` (one cut); `fragment_until`
vs `apply_drop_budget_cut` (the two stop conditions); `route_holdout` + `make_folds` (atoms →
splits); `_cv_sampling.assign_atoms` (the third split path).

## 6. Open questions (need a decision)

6.1 **Persisted audit strings.** The split audits record an `algorithm` value of
`'bipartite_cc_lpt_greedy'` (`_pair_helpers.seq_disjoint_route_pos_df`) and
`'bipartite_cc_lpt_greedy_on_cluster_ids'` (`_split_helpers.cluster_disjoint_route_pos_df`).
Renaming makes new run dirs disagree with existing ones. Rename, or leave the persisted values
alone? Item 3 above is blocked on this.

6.2 **Default split strategy.** `conf/dataset/default.yaml:10` makes `seq_disjoint` the repo-wide
default, while 45 of 46 recorded runs are `cluster_disjoint*`. A new bundle that omits `mode`
silently gets protein-level seq routing even on an nt_cds experiment. Retarget the default?
