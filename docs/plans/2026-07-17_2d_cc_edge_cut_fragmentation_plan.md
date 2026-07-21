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

Cut primitives in `src/datasets/_megacc_cut.py` (Phase R; see their docstrings): `build_pair_bigraph`
(pair-weighted simple bigraph, `weight` = #pairs on a cluster-pair) then `fragment_largest_cc` (one
edge min-cut of the largest CC via `_bisect`, spectral Fiedler / KL -- partition only; dropping the
crossing edges realizes the cut) then `edges_to_row_index` (crossing edges to dropped pair rows).
Two loops over the primitive: `apply_drop_budget_cut` (routing-A, holdout 80/10/10 feasibility stop,
wired at `_split_helpers.cluster_disjoint_route_pos_df`) and `fragment_until` (routing-B, count stop; P2).

**Hard cap:** `_bisect` partitions NODES, so it never splits a single cluster -- the largest atom
cannot drop below the largest node's pair mass (the *edge-cut floor*; going below needs node-cut, out
of scope).

## 3. Scope

- IN: edge-cut of the 2D mega-CC on **OOD nt_cds HA-NA** via the production `dataset_pairs_cc`
  builder; single-cut validation; the score-vs-`t` experiment.
- OUT: single-side / mega-cluster **node-cut** (future `..._single_segment_node_cut_..._plan.md`);
  metadata-aware sampling; nt_ctg; the analysis harness (superseded `2026-06-06`).

## 4. Design decisions

- **D1. Route by atom COUNT (routing-B: GroupKFold + `m_pos_per_cc`), not pair mass.** The CV
  builder gives every atom `m` pairs regardless of its size, so the pair-mass floor (Q2) never
  binds — what matters is the NUMBER of atoms, which we grow by fragmenting. `apply_drop_budget_cut`
  is routing-A (pair-mass LPT for a *holdout*, floor-limited at t095) and is NOT used by P2; it
  stays the holdout tool at lower priority. Layers: **L1** `fragment_largest_cc` (cut a CC, have)
  → **L2** a count-based stop (`fragment_until`: target #atoms / N cuts) → **L3** the existing
  routers (`make_folds` GroupKFold+`m_pos` for CV; `route_holdout` LPT for a holdout). The stop
  rule is decoupled from the router, except the feasibility stop, which is holdout-specific.
- **D2. Cut method = spectral (default)**, expose `kl`. Spectral drops fewer pairs (glossary:
  0.9% vs 10.1% at aa t095) but is unbalanced; for routing-B, fold balance is by atom count in
  GroupKFold, not the cut. (METIS/KaHIP held in reserve if the measured drop-% is unacceptable.)
- **D3. One shared cut core (Phase R).** Extract the primitive out of `apply_drop_budget_cut` so
  the single cut (P1, via `fragment_once`), the routing-A budget loop (holdout), and P2's routing-B
  count fragmentation all call it — the stop rule (feasibility / #atoms / N cuts) is the loop's
  business, not intrinsic to the cut. Supersedes the earlier "small standalone" P1 approach (P1's
  `p1_single_cut.py` is repointed at `fragment_once`).

## 5. Phases

**P0 — measure the cap (analysis-only; no builder change).** On OOD nt_cds HA-NA per `t`
(t099/t098/t097/t095): the **largest node's pair-mass fraction** (the edge-cut floor) vs `1/K`,
and how many atoms edge-cut *alone* recovers before the floor. Decides whether the cap binds (→
future node-cut) and the achievable atom count. Reuse `cc_pair_sizes.csv` + a `_bisect` dry-run.

**P1 — single-cut validation DONE (2026-07-17; gated P4).** Bisected the OOD nt_cds mega-CC **once**
(t095, spectral): 77,731-pair mega-CC → 25,179 / 52,538, 14 straddling pairs dropped. Finding (from
the fragments, per-slot HA/NA drop accounting, ESM-2 UMAP, and `verify_ood_clusters`): an **avian vs
mammalian** split with the 14 dropped bridges on Duck/Dog/Mink/Turkey host co-occurrences, and
cross-fragment OOD holds by construction (0 `>= t` hits).

**Phase R DONE (2026-07-19) — modular fragmentation primitive (behavior-preserving).** Extracted the
reusable edge-min-cut core from `apply_drop_budget_cut` into `_megacc_cut.py` — `build_pair_bigraph`,
`fragment_largest_cc`, `edges_to_row_index`, and the standalone `fragment_once` (P0/P1; signatures in
their docstrings) — so the P1 single cut, P2's routing-B count fragmentation, and the analysis
`bigraph_*` twins share one implementation. `apply_drop_budget_cut` was rewritten over them with
**signature and return unchanged** (the `_split_helpers` bilateral-holdout caller at L459-477 is
untouched); absorbs task 4's *edge weight* docstrings. **Deferred to P2:** the count-based stop
(`fragment_until`) — R kept the existing 80/10/10 loop verbatim.

**P2 DONE (2026-07-20; commit `f0e9ce9`) -- atom-count fragmentation wired into `dataset_pairs_cc`
(routing-B; closed the L154 TODO; validated on OOD t095: 108 -> 124 atoms @ 1.8%, cluster-disjoint,
+ a full 5-fold build; `tests/test_dataset_pairs_cc_cut.py`).** Grow
the atom count by fragmenting the mega-CC, then use the existing folds machinery unchanged. In
`assign_atoms_prod`: after the natural CCs, if fragmentation is enabled, loop `fragment_largest_cc`
to a target #atoms (L2 count stop) → drop the straddling pairs → re-derive atoms via
`bipartite_components` → `make_folds` (GroupKFold + `m_pos`). Does NOT call `apply_drop_budget_cut`
(routing-A). Add the knobs to the OOD bundle; fix the L154 note (it still points at
`apply_drop_budget_cut`). Rebuild OOD nt_cds HA-NA at t095 and below → recover atoms toward the
t097 count. **Reality (measured): edge-cut is floor-limited** -- t095 goes 108 -> ~124 within a 2%
drop budget (`pairs_dropped` ~1.8%), NOT the t097 ~387 (that needs node-cut, Q2). So the P4
comparison holds `t` at a **common ~120 atoms** (fragment t095 up; `max_atoms` caps higher-`t` down).

**P2 design decisions (settled 2026-07-20):**
- `cc_id = atom_id =` the post-cut fragment (overwrite both -- negatives + `m_pos` group per
  fragment). Keep the pre-cut CC on `pos_ids['natural_cc_id']` (analysis-only; excluded from the
  fold CSVs by the `_PAIR_COLUMNS` re-select at `_write_output`).
- Config `split_strategy.edge_cut: {enabled, cut_method, target_atoms, max_drop_frac}` -- an optional
  block (via `OmegaConf.select`, existing bundles untouched); `edge_cut`, not `cut`, to disambiguate
  from a future `node_cut`. Composes with `max_atoms` (grow to target, then cap down for a common N).
- Before/after artifacts via `_write_cc_pair_sizes`: `cc_pair_sizes.csv` (natural, pre-cut) +
  `cc_pair_sizes_post_edge_cut.csv` (fragmented, post-cut); `plot_cc_sizes.py` overlays them.
- Verify with `src/analysis/cluster_disjoint_cv_experiment._assert_fold_disjoint` (no shared cluster
  across folds on either slot) -- reuse, don't reinvent.

**P4 (OPTIONAL) — score-vs-`t` experiment.** P5 (the matched-size contrast) is the more direct
primary test; run the full `t`-sweep only if the P5 result motivates it. With atoms recovered to an
adequate count at low
`t` (t095 and below), vary `t` → does the flat "size, not threshold" curve hold in the most-OOD
regime, or does difficulty rise? Report per `t`: n_atoms, largest_atom_frac, dropped_frac
(fragmentation drop cost), AUC / F1.

**P5 -- is the OOD split itself hard? OOD vs non-OOD dataset at matched size (primary; simpler than
P4).** Build two datasets from the **same** positive pairs -- an **OOD** one (train/val/test
cluster-disjoint) and a **non-OOD** one (train/test share clusters) -- train both, read the gap. Same
pairs and same size, so the gap is the split's effect (the OOD penalty), not size (2026-07-14 covered
size).

Why not the earlier "reshuffle the fold CSVs" idea: at `m_pos_per_cc=1` each cluster sits in exactly
one positive pair (atom = one bipartite CC, which owns its clusters), so **any** split of those
positives is already cluster-disjoint -- reshuffling gives a second OOD dataset, and the only
cross-split mixing would be via relocated negatives (a confound). A real non-OOD dataset needs
clusters to recur across positives, which `m_pos=1` forbids.

Construction -- one pool, two splits:
- **Pool.** Edge-cut fragment the mega-CC at `t` into N atoms and keep **all** pairs per fragment
  (`m_pos_per_cc` well above 1, not 1 -- recurrence is the point). Only the fragmentation straddlers
  are dropped, from both datasets. (Small CCs are ~1% of pairs at low `t`; exclude them at higher `t`,
  where they would sit in one fold and dilute the non-OOD split.)
- **OOD dataset.** Route whole fragments to folds -- the existing GroupKFold-by-`atom_id` path
  (`make_folds_within_fold`). No cluster spans folds.
- **non-OOD dataset.** Split the same pairs at the **pair** level (random K-fold, ignoring fragment
  boundaries). A dominant cluster (NA_0, HA_0) is in many pairs within its fragment, so a pair-level
  split scatters those pairs across folds and the cluster lands in more than one fold -- train/test
  now overlap.
- **Negatives.** `within_fold` negatives per split for **both** datasets -- same procedure, so the
  split is the only difference. within_fold is *required* for the OOD dataset (a cross-split negative
  would put a test cluster in train); it is **not required** for the non-OOD dataset, and is used
  there only to build both datasets the same way.

Notes:
- **Not every pair overlaps -- fine.** Many pairs inside the mega-CC share no cluster, so the non-OOD
  dataset is *mostly* in-distribution (what a real random split looks like); the overlap comes from
  the dominant clusters.
- **`t` sets a trade.** Lower `t` -> larger, denser mega-CC -> richer pool and more recurrence, but
  the edge-cut floor (NA_0 ~37% at t095) caps N and leaves one ~37% fragment that cannot be split --
  it lands whole in one fold and unbalances the OOD folds. Higher `t` -> thinner pool, more even
  atoms. Pick `t` from the per-`t` mega-CC fractions (to measure).
- **Fold balance.** OOD folds are uneven (whole fragments of different sizes), non-OOD folds even;
  stratify or subsample if it matters.
- **Verify.** OOD: no cluster spans folds on either slot (`_assert_fold_disjoint`); non-OOD: clusters
  do span folds (its inverse). Fold CSVs carry no `cluster_id`, so join hash->cluster with
  `load_cluster_lookup` on the per-slot hash columns; do not add `cluster_id` to the CSVs (Stage-4
  schema).
- **Repeats.** Over cut-seed x fold-seed for a confidence interval on the gap (Q5).
- **Optional 3rd arm.** Cluster-disjoint folds on **set-cover** clusters (`clusters_nt_cds`) at the
  same N separates the OOD penalty from the plain disjointness penalty.

New vs reuse: the non-OOD **pair-level random split** is new (a small builder over the pool + the
existing `within_fold` negatives, not a CSV reshuffle); the OOD dataset, edge-cut, `within_fold`, and
`_assert_fold_disjoint` already exist.

*(No P3 — node-cut out of scope.)*

## 6. Verification

- Cluster-disjoint invariant: no cluster spans two fragments/atoms (assert in the cut path).
- P1's `verify_ood_clusters` cross-fragment check = 0.
- **Phase R (no drift):** `tests/test_megacc_cut.py` guards the reused cut core -- synthetic
  property tests for the primitives + two OOD integration checks on the production
  `clusters_nt_cds_ood`: `apply_drop_budget_cut` reproduces the pre-refactor t099 digest
  (`tests/golden/megacc_cut/ood_nt_cds_t099.json`) and `fragment_once` reproduces the P1 t095
  numbers. (The retired aa `drop_budget_2d_aa` harness guard + its deleted bundle are gone; the
  glossary keeps the aa t095 0.9%-spectral figure as the reference.)

## 7. Open questions / risks

- **Q1. Drop-% to reach the target #atoms (routing-B) -- ANSWERED (2026-07-20).** OOD nt_cds t095
  spectral: 108 -> 124 atoms costs ~1.8% (1,394 straddling pairs); the cheap knee is ~125 @ ~2%,
  past which the floor makes further atoms prohibitive (~130 @ 38%). This is the recoverable-atom
  ceiling and the cut-bias size.
- **Q2. Does the cap bind at t095? YES (measured 2026-07-19).** The largest single-side cluster's
  pair mass is the edge-cut floor: NA_0 = 37.1%, HA_0 = 33.7% of all pairs at t095 (t099: 29.9% /
  29.1%). Since every K-fold bin (`1/K = 20%` at K=5) `< 37.1%`, edge-cut alone cannot reach it --
  `apply_drop_budget_cut` even raises `DropBudgetExceeded` for the looser 80/10/10 (37.6% dropped
  after 34 cuts, largest CC floored ~51%). This limits **routing-A** (pair-mass holdout) — not our
  route: **routing-B** (GroupKFold + `m_pos`, D1) sidesteps the floor (the dense core just stays one
  atom; we grow the count by fragmenting the rest), so node-cut is not needed for the CV line.
  Corroborated by the HA/NA t095 cluster-size barplots (HA_0 31.9% + HA_1 25.6%; NA_0 35.1% +
  NA_1 22.0% of unique sequences).
- **Q3. Cut-bias direction.** Dropped straddling pairs are the cross-subtype reassortant bridges
  (aa `2026-06-04` finding); the atoms get more subtype-pure. Report `dropped_frac` so it is
  legible — it is a feature (cleaner atoms) as much as a confound.
- **Q4. nt_cds OOD subtype alignment.** "Atoms ≈ antigenic subtypes" was found on **aa +
  set-cover**; P1's UMAP checks whether it holds for **nt_cds OOD**.
- **Q5. Seed dependence.** Spectral/KL are seeded; the fragmentation (hence the folds) depends on
  the cut seed. Note it; repeated-CV-over-cut-seeds is a later variance study.

## 8. Task list

1. **DONE** — `2026-06-06_fragmentation_cv_plan.md` moved to `done/` (superseded-by note added).
2. **P0 — largely answered (Q1/Q2):** the floor binds at t095 (NA_0 37.1% > 1/K) and edge-cut
   recovers ~124 atoms @ 1.8%; a full per-`t` floor table is optional.
3. **P1 DONE (2026-07-17)** — single-cut + the 5 inspections; reviewed. Avian-vs-mammalian split,
   14 dropped bridges (Duck/Dog/Mink/Turkey hosts), cross-fragment OOD holds by construction.
4. **Phase R DONE (2026-07-19)** — extracted `build_pair_bigraph` / `fragment_largest_cc` /
   `edges_to_row_index` / `fragment_once` in `_megacc_cut.py` (+ full docstring rewrite); rewrote
   `apply_drop_budget_cut` over them, behavior-preserving. Verified by `tests/test_megacc_cut.py`
   (bit-exact OOD t099 digest + P1 t095 reproduction). Glossary: added *LPT bin-pack*, *drop-budget*.
5. **P2 (routing-B)** — atom-count fragmentation in `dataset_pairs_cc.assign_atoms_prod` (L2 count
   stop → `bipartite_components` → GroupKFold+`m_pos`); OOD-bundle knobs; fix the L154 note; rebuild
   t095 to ~124 (floor-limited @ 1.8%; not the t097 ~387). **DONE 2026-07-20, commit `f0e9ce9`** --
   `split_strategy.edge_cut` config + bundle `flu_ha_na_cc_nt_cds_ood_edge_cut` + before/after
   `cc_pair_sizes*.csv` + `tests/test_dataset_pairs_cc_cut.py`. Not `apply_drop_budget_cut` (routing-A).
6. **P4** — (OPTIONAL, superseded as primary by P5) score-vs-`t` at t095 and below. **Do not launch the full sweep without
   explicit confirmation** (standing instruction).
7. **P5** -- OOD vs non-OOD dataset at matched size, same pool (mega-CC fragmented, `m_pos` raised to
   keep all pairs): OOD = whole-fragment folds; non-OOD = pair-level random split; `within_fold`
   negatives for both; verify clusters do/don't span folds (hash->cluster join-back). New: the
   pair-level random split builder. Optional set-cover 3rd arm. Gated with P4.
