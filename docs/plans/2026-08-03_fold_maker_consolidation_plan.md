# Fold-maker and fragmentation consolidation

**Status: IN PROGRESS**

Date: 2026-08-03

Consolidate the code that turns *atoms* into folds and that fragments the mega-CC to create
those atoms. Three phases are done; testing the production paths is agreed and unstarted
(§5); two behaviour-changing decisions remain open (§6).

Related: `docs/plans/done/2026-07-30_bigraph_consolidation_plan.md` (the graph/CC layer below
this), `docs/methods/glossary.md`, `docs/methods/splits.md`.

---

## 1. The two production paths

Every change is judged against these. Both cluster nt_cds at **t099** on the `cm0` root, k=4,
`neg_to_pos_ratio: 1.0`, and draw within-fold negatives from the same sampler.

| | **2D-CD** | **1D-CD (HA axis)** |
|---|---|---|
| bundle | `flu_ha_na_cc_nt_cds_cm0_wf.yaml` | `flu_ha_na_1dcd_nt_cds.yaml` |
| builder | `dataset_pairs_cc.py` | `dataset_segment_pairs_v2.py` |
| mode | `cluster_disjoint_cc` | `cluster_disjoint` + `single_slot: a` — **HA** is the held-out axis |
| atom | one CC on (`cluster_id_a`, `cluster_id_b`); under `edge_cut`, a post-cut fragment | one HA cluster |
| router | `make_folds_within_fold` (`dataset_pairs_cc.py:469`) | k-fold branch of `_split_helpers.cluster_disjoint_route_pos_df:591-628` |
| test folds | `GroupKFold(shuffle=True, random_state=seed)` on `atom_id` | `GroupKFold` (**unshuffled**) on `cluster_id_a` |
| val carve | `_carve_val_atoms:328` — seeded atom shuffle | `_lpt_bin_pack`, two bins |
| determinism | seeded | **seed-independent** — `seed` is audited, never consumed |
| negatives | `within_fold_negatives:393` | same function, lazily imported by the v2 builder |

Both partition positives only; both keep atoms whole in every split, so val is cluster-disjoint
from train on both paths. `cluster_disjoint_route_pos_df` raises `NotImplementedError` for k-fold
with bilateral atoms, which is why `dataset_pairs_cc.py` exists.

Six sibling bundles (`flu_ha_{m1,np,ns1,pa,pb1,pb2}_1dcd_nt_cds.yaml`, untracked) pair HA with a
different protein. **HA is always the held-out axis**, but its slot follows `protein_order`:
`single_slot: b` for PA / PB1 / PB2, `a` for NA / NP / NS1 / M1. Code must not assume slot `a`.

## 2. Constraint

**Bit-exact on both production bundles.** Every change must reproduce the existing
`fold_k/{train,val,test}_pairs.csv` byte-for-byte, evidenced by a before/after capture rather than
an argument. `pytest tests/ -q` green before each commit. Anything that cannot meet this goes to §6.

## 3. Current state

**Fold-makers** — four, over one shared core, `groupkfold_by_atom` (`dataset_pairs_cc.py:362`).
The **leave-cc-out** column marks the `# ===` block at `dataset_pairs_cc.py:767-916`, which serves
only `flu_ha_na_cc_nt_cds_ood_ood_vs_random` — not to be confused with the other senses of "OOD"
(the `clusters_*_ood` root; the general property that any cluster-disjoint split holds out unseen
clusters, 2D-CD included).

| function | file:line | reached by | leave-cc-out? | test unit | val carve |
|---|---|---|---|---|---|
| `make_folds_within_fold` | `dataset_pairs_cc.py:469` | **2D-CD production** | no | k atom-groups | atoms |
| `_split_helpers` k-fold branch | `_split_helpers.py:591-628` | **1D-CD production** | no | k atom-groups | atoms (LPT) |
| `_partition_full` groupkfold arm | `dataset_pairs_cc.py:874` | no bundle | in the block, generic branch | k atom-groups | atoms |
| `make_folds_leave_cc_out` | `dataset_pairs_cc.py:816` | the OOD-vs-random bundle | **yes** | **one atom** | **rows** |
| `make_folds_random` | `dataset_pairs_cc.py:840` | same bundle, control arm | **yes** | **rows** | **rows** |

**Fragmentation** — two loops over one cut, `fragment_largest_cc` (`_megacc_cut.py:173`).

| function | file:line | stops on | budget | used by |
|---|---|---|---|---|
| `fragment_until` | `_megacc_cut.py:268` | `stop_fn` predicate | `max_drop_frac` | **2D-CD production** |
| `fragment_to_targets` | `_megacc_cut.py:376` | LPT feasibility vs `targets` | none | `bigraph_min_cut` |

`fragment_once` (`:202`) applies the cut once and returns the `CutStep`; the loops mutate a graph
across cuts so none can use it. `route_holdout` (`_pair_helpers.py:877`) produces a holdout, not
folds; neither production path reaches it.

## 4. Completed

**Phase 1 — fold-maker consolidation.** Extracted `groupkfold_by_atom` as the shared routing core
and removed `make_folds`, which had become a pure delegate no bundle reached. Renamed the
leave-cc-out arms' `cc`→`atom` (they filter `atom_id`, and under `edge_cut` that is a post-cut
fragment). Corrected "cross-CC" to "within-fold" in three places — the sampler ignores CC
membership. Documented two invariants the code relied on silently: why `within_fold_negatives`'
per-split dedup set is sufficient, and why `_make_folds_for_scope` may ignore `fold_assignment`.
Added a **Within-fold negative** glossary entry, and corrected `splits.md` §3.2, which listed
bilateral `cluster_disjoint` k-fold as "not built" when it is the primary production path.

**Phase 2 — experiment subtree.** `_partition_full`, `pick_largest_atoms`,
`make_folds_leave_cc_out`, `make_folds_random`, `_carve_val_pairs` and four config knobs all
arrived in one commit (`c1f7afd`) for one bundle, with no test coverage. Co-located behind a
`# ===` banner, documented in the module docstring, `KEEP` header on the bundle, and pinned by
`tests/test_partition_full_arms.py` — which asserts the experiment's premise, that **both arms
partition the same rows**, previously unchecked. Verified by mutation, not by passing.

**Phase 3 — fragmentation.** `fragment_to_targets` (renamed from `fragment_weighted`, whose
contrast died with `weighted_simple`) carried a private copy of the cut; it now calls
`fragment_largest_cc`, so the module's "one shared cut" claim is true. Archived
`apply_drop_budget_cut` with its wiring — the hardcoded-80/10/10 form of `fragment_to_targets`,
serving the superseded 2D-CD holdout, with no config declaring its knob. Fixed: the budget being a
trip-wire rather than a cap, an atom count that included stranded nodes, an `IndexError` on empty
input in every entry point, and `fragment_until` shredding single-pair components (3 atoms → 0).
Moved `uniform_targets` to the archived harness that was its only caller. Removed the plan labels
(`routing-A/B`, `L2/L3`) and `_bisect`'s history essay.

Closed the three ways the analysis loop diverged from production: it defaulted to `kl`, which no
production path uses and which the 07-30 plan measured as order-sensitive; it lacked the cut floor;
and it reported raw components where production reports routable atoms — `bigraph_min_cut` printed
that raw count under the label "atoms". `fragment_to_targets` now defaults to `spectral`, carries
the floor, and emits an `n_atoms` column beside `n_pieces`. What still differs is deliberate: it
stops on LPT feasibility rather than an atom count, and carries no drop budget.

Verification across all three: 2D-CD and 1D-CD t099 HA-NA rebuild md5-identical to the pre-Phase-1
baseline; `pytest tests/ -q` green. Per-finding detail is in the commit messages, not restated here.

## 5. Phase 4 — test the production paths (agreed 2026-08-05, not started)

Neither production path is adequately tested. 1D-CD has **no tests at all** — nothing exercises
`cluster_disjoint_route_pos_df`, `_lpt_bin_pack`, the drift check or `route_holdout`. 2D-CD is
tested where it is least needed: `assign_atoms_prod`, the cut primitives and the leave-cc-out
experiment arms are covered, while `groupkfold_by_atom`, `make_folds_within_fold`,
`_carve_val_atoms` and both negative samplers — the code that decides what lands in train/val/test
— are not. Every bit-exactness claim in §4 came from building and diffing by hand.

`scripts/production_split_harness.py` (added 2026-08-05) closes the end-to-end hole for both paths
but is a manual script.

**P4.1 — end-to-end, both paths.** A test per path that builds the production bundle and compares
per-fold `pair_key` digests against `tests/golden/production_splits/`. Invokes the harness rather
than re-implementing the digest, so the two cannot drift. Three constraints:
- **Skip, don't fail, when the corpus is absent** (`protein_final`, `cds_dna_final`, the cluster
  parquets), matching the rest of the suite.
- **Deselected by default.** 2D-CD ~105s and 1D-CD ~125s, and both run longer under load; adding
  them unconditionally takes `pytest tests/ -q` from ~3 min to ~8-10. Mark them so the default
  suite stays fast and the guard is one explicit command.
- **The goldens pin the current corpus.** Rebuilding Stage 1 changes the pairs legitimately and
  will turn these red for a good reason. The failure message must say so and name the `capture`
  regeneration path, or the next reader will "fix" the code instead.

**P4.2 — unit tests, Tier 1 (pure decision functions).** Cheap with synthetic frames, and a
failure names the culprit — which the end-to-end test cannot.
- 2D-CD: `groupkfold_by_atom`, `make_folds_within_fold`, `_carve_val_atoms`,
  `within_fold_negatives`, `within_cc_negatives`, `compute_negative_infeasible_ccs`
- 1D-CD: the k-fold branch of `cluster_disjoint_route_pos_df`, `_lpt_bin_pack`, the drift check
- shared: `_resolve_spec`'s validation, which is the guard that makes F6 safe

Properties worth pinning, not just smoke: atoms whole in every split; val ≈ `val_ratio` of the
whole set, not of the non-test pool; negatives drawn only from their own split; folds partitioning
the positives exactly once. Each test verified able to fail, as in `test_partition_full_arms.py`.

**Tier 2 — orchestration, covered by P4.1 rather than unit-tested.** `_build_positives`,
`_make_folds_for_scope`, `split_dataset_v2`, `generate_all_cluster_disjoint_cv_folds_v2`. Heavy
I/O and mostly wiring; their bugs surface as wrong output, which is what the end-to-end digest
already catches.

**Tier 3 — not tested.** `_side_rep`, `_write_output`, `_write_cc_sizes`, `_write_cc_pair_sizes`.
A test costs maintenance and catches nothing P4.1 misses.

Order: P4.1 first — it protects everything else while the unit tests land — then Tier 1 for 2D-CD,
then Tier 1 for 1D-CD, which is the largest gap.

## 6. Open

- **D1 — vary the val-carve seed per fold.** `make_folds_within_fold:480` passes the bare `seed` to
  `_carve_val_atoms` on every fold while its negatives vary. Both schemes are equally reproducible;
  only fold-to-fold independence differs. 2D-CD only — the 1D-CD router consumes no seed. Cheap and
  correct, but it changes every built dataset.
- **D2 — one router for both production paths.** Both route positives only, but disagree on
  shuffled vs unshuffled `GroupKFold` and on seeded atom shuffle vs LPT for val, so D2 cannot be
  bit-exact on both. LPT is not arbitrary: `_build_audit:179-180` measures drift on all three bins
  and LPT minimizes it, while `_carve_val_atoms` can overshoot by a whole atom (the 2D-CD path has
  no such check). Carries the `cluster_disjoint_route_pos_df` naming question — it implements
  bilateral holdout, single-slot holdout and single-slot k-fold, but every bundle sets
  `single_slot`, so only the single-slot paths are reachable. D2 is a **subset of P3** in
  `docs/plans/2026-06-03_dataset_split_refactor_plan.md`, which additionally wants
  `generate_all_cv_folds_v2` and `generate_all_cluster_disjoint_cv_folds_v2` retired and CV
  validated end-to-end.
- **Recorded, not fixed** — plan labels off the fragmentation path: `D1`–`D4`, `OoS`, `P2` (~35
  sites in `_split_helpers` and `dataset_segment_pairs{,_v2}`). `D3`/`D4` appear in raised error
  text, so a user tripping the feasibility guard is pointed at a plan label; that one has a real
  cost.

## 7. Standard

`CLAUDE.md` § Conventions, in particular: docstrings state what the function does with accurate
`Args:`/`Returns:` and no history; names consistent across production scripts and inferable from
the code; function names describe current behaviour, and renames need approval; dense statements
broken into named steps, never a return that also does the work; and no plan-only vocabulary in
code, comments or error text.

## 8. Out of scope

`route_holdout` and the `seq_disjoint` router; the sampling logic inside the negative samplers
(their docstrings are in scope); `dataset_segment_pairs_v2`'s coverage sampler; `src/archive/`.
