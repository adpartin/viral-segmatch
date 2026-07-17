# 2D-CC edge-cut fragmentation: recover atoms below t097 on the OOD line

**Status: IN PROGRESS**

Date: 2026-07-17

Scope: fragment the **2D bipartite mega-CC** by **edge min-cut** (drop straddling pairs) to
recover independent **atoms** at low `t` on the production **OOD nt_cds** line, so we can test
whether the "size, not threshold" result holds in the most-OOD regime. **Edge-cut only** —
single-side (single-segment / mega-cluster) **node-cut** is a *future, separate* plan.

Related:
- **Supersedes** `docs/plans/2026-06-06_fragmentation_cv_plan.md` (analysis-harness; aa +
  set-cover clusters). Kept for its edge-cut design + the K-uniform reasoning. [→ move to
  `done/` with a superseded-by note; task 1.]
- `docs/plans/2026-06-04_2d_cd_drop_budget_router_plan.md` — the drop-budget router
  (`src/datasets/_megacc_cut.apply_drop_budget_cut`), **CORE IMPLEMENTED**; wired into
  `_split_helpers.cluster_disjoint_route_pos_df` (bilateral holdout, L459-477), **not** into
  `dataset_pairs_cc` (its L154 is the TODO this plan closes).
- `docs/results/2026-07-14_cc_ood_threshold_size_decoupling.md` — the finding this extends.
- `docs/plans/2026-07-08_single_segment_ood_clusters_plan.md` — the OOD clusters we operate on.
- `docs/methods/glossary.md` — canonical terms: **edge weight**, **pair mass**, **straddling
  pair**, **mega-CC**, **mega-cluster**, **atom**, **edge min-cut**.

---

## 1. Motivation

`2026-07-14` established: 2D-CD test performance tracks the **number of independent atoms**, not
the OOD threshold `t` (flat AUC across t099-t097 at a fixed 387 atoms). The lever is #atoms. At
low `t` the **mega-CC collapse** starves atoms (nt_cds HA-NA: t099=3,350 → t097=387 →
t095=**108**), so the most-OOD splits are data-starved and can't be tested at adequate size by
subsampling alone.

**Edge-cut fragmentation recovers atoms** at fixed `t` by bisecting the mega-CC into smaller CCs,
dropping only the straddling positive pairs. This unlocks the controlled test: at t095 (and
below), recover atoms to an adequate count and ask whether difficulty finally rises — i.e., does
"size, not threshold" hold in the most-OOD regime, or break?

## 2. Mechanism (grounded)

`apply_drop_budget_cut` (`src/datasets/_megacc_cut.py`; proven in `_split_helpers.py:459-477`):
- builds the pair-weighted **simple bigraph** (nodes = single-segment clusters, slot-prefixed
  `a:`/`b:`; **edge weight** = #positive pairs on the cluster-pair, `weight=len(rows)` L149);
- repeatedly **`_bisect`** (spectral Fiedler / KL) the largest CC → a node partition A/not-A
  (L56-75), then the caller **drops the crossing edges** (`H.remove_edges_from(cross)` L190) →
  the CC splits into two (sometimes more) CCs;
- until the kept CC sizes LPT-pack the target ratios within `drift_pp`, else raises
  `DropBudgetExceeded` past `max_drop_frac`.
- **`_bisect` = partition only; the caller's edge-drop realizes the cut.** Dropping an edge drops
  the positive-pair **rows** on that cluster pair (`kept_pos = pos_with_ids.drop(index=drop_idx)`
  L194); clusters and their sequences persist (a dropped pair's sequences may live in other kept
  pairs) — the cost is counted in **pairs**.

**Hard cap:** `_bisect` partitions **nodes**, so it can never split a single cluster. The largest
atom cannot drop below the **largest node's pair mass** — the *edge-cut floor*. Going below that
floor needs **single-side node-cut** (out of scope; future plan).

## 3. Scope

- IN: edge-cut of the 2D mega-CC on **OOD nt_cds HA-NA** via the production `dataset_pairs_cc`
  builder; single-cut validation; the score-vs-`t` experiment.
- OUT: single-side / mega-cluster **node-cut** (future `..._single_segment_node_cut_..._plan.md`);
  metadata-aware sampling; nt_ctg; the analysis harness (superseded `2026-06-06`).

## 4. Design decisions

- **D1. Cut target = K-uniform `1/K`** (tighter than 80/10/10; a single 80%-mass atom is
  LPT-feasible for 80/10/10 but not K equal folds). The K-uniform drop-% is unknown and `> 0.9%`
  (the 80/10/10 figure) — measure it (Q1).
- **D2. Cut method = spectral (default)**, expose `kl`. Spectral drops fewer pairs (glossary:
  0.9% vs 10.1% at aa t095) but is unbalanced; fold balance is handled by the K-uniform LPT step,
  not the cut. (METIS/KaHIP held in reserve if the measured drop-% is unacceptable *and* the cap
  isn't binding.)
- **D3. Reuse `apply_drop_budget_cut`** for the production wiring (P2). For the P1 single-cut
  validation, call the `_bisect`+drop-crossing-edges primitive directly in a small standalone —
  the 80/10/10 (or K-uniform) target is just the loop's stopping rule, not intrinsic to the cut,
  so don't fight the budget loop to force "one cut".
- **D4. Docs:** `glossary.md` *Edge weight* — **DONE** (commit 9784692). `_megacc_cut` /
  bigraph-builder docstrings for the same clarifications — still owed (task 4).

## 5. Phases

**P0 — measure the cap (analysis-only; no builder change).** On OOD nt_cds HA-NA per `t`
(t099/t098/t097/t095): the **largest node's pair-mass fraction** (the edge-cut floor) vs `1/K`,
and how many atoms edge-cut *alone* recovers before the floor. Decides whether the cap binds (→
future node-cut) and the achievable atom count. Reuse `cc_pair_sizes.csv` + a `_bisect` dry-run.

**P1 — single-cut validation (standalone; START HERE; gates P4).** Bisect the OOD mega-CC **once**
and inspect:
- the two fragments + the dropped set;
- size of each fragment (clusters + pairs);
- per-slot (HA / NA) drop accounting — dropped straddling pairs and the sequences they touch on
  each side;
- **UMAP** of the two fragments (ESM-2, colored by fragment) — do they separate? (does the nt_cds
  OOD cut align with antigenic subtypes as the aa cut did — `2026-06-04`?);
- **OOD-verify** across the two fragments (`verify_ood_clusters.py` check fn) → 0 cross-fragment
  `>= t` hits (expected by construction; the value is catching a bug where a cluster spans both
  fragments).

**P2 — wire edge-cut into `dataset_pairs_cc`** (closes its L154 TODO). Mirror `_split_helpers`:
after atom assignment, if `drop_budget.enabled`, call `apply_drop_budget_cut` (K-uniform target,
`cut_method`) → re-derive atoms via `bipartite_components` → GroupKFold. Add the knobs to the OOD
bundle. Rebuild OOD nt_cds HA-NA t097/t095 with the cut → atoms recovered + AUC.

**P4 — score-vs-`t` experiment (gated on P1).** With atoms recovered to an adequate count at low
`t` (t095 and below), vary `t` → does the flat "size, not threshold" curve hold in the most-OOD
regime, or does difficulty rise? Report per `t`: n_atoms, largest_atom_frac, dropped_frac
(K-uniform drop cost), AUC / F1.

*(No P3 — node-cut out of scope.)*

## 6. Verification

- Cluster-disjoint invariant: no cluster spans two fragments/atoms (assert in the cut path).
- P1's `verify_ood_clusters` cross-fragment check = 0.
- Regression: at the 80/10/10 target, reproduce the known 0.9% spectral drop on aa t095 (guards
  the reused cut core).

## 7. Open questions / risks

- **Q1. K-uniform drop-% (blocks the ceiling).** Unknown, `> 0.9%`. Sets the recoverable-atom
  ceiling and the cut-bias size. Measure in P0/P1 before committing to an atom target.
- **Q2. Does the cap bind at t095?** If the largest node's pair mass `> 1/K`, edge-cut cannot
  reach K folds at t095 → the experiment is floor-limited there, and node-cut (future) is needed
  to go further. P0 answers this.
- **Q3. Cut-bias direction.** Dropped straddling pairs are the cross-subtype reassortant bridges
  (aa `2026-06-04` finding); the atoms get more subtype-pure. Report `dropped_frac` so it is
  legible — it is a feature (cleaner atoms) as much as a confound.
- **Q4. nt_cds OOD subtype alignment.** "Atoms ≈ antigenic subtypes" was found on **aa +
  set-cover**; P1's UMAP checks whether it holds for **nt_cds OOD**.
- **Q5. Seed dependence.** Spectral/KL are seeded; the fragmentation (hence the folds) depends on
  the cut seed. Note it; repeated-CV-over-cut-seeds is a later variance study.

## 8. Task list

1. Move `2026-06-06_fragmentation_cv_plan.md` → `done/` with a superseded-by note.
2. **P0** — largest-node pair-mass vs `1/K` table (OOD nt_cds HA-NA, per `t`).
3. **P1** — standalone single-cut + the 5 inspections; **stop and review before P2**.
4. `_megacc_cut` / bigraph-builder docstrings for *edge weight* (glossary already done).
5. **P2** — wire `apply_drop_budget_cut` into `dataset_pairs_cc`; OOD-bundle knobs; rebuild t097/t095.
6. **P4** — score-vs-`t` at t095 and below (gated on P1). **Do not launch the full sweep without
   explicit confirmation** (standing instruction).
