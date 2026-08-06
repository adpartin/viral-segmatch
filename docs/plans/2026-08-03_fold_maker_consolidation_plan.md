# Fold-maker and fragmentation consolidation

**Status: IN PROGRESS**

Date: 2026-08-03

Consolidate the code that turns *atoms* into folds and that fragments the mega-CC to create
those atoms. Four phases are done, including test coverage for both paths (§5). Two
behaviour-changing decisions remain open (§6); nothing else is outstanding.

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

## 5. Phase 4 — test the production paths — **DONE (2026-08-05)**

Before this phase, 1D-CD had **no tests at all** and 2D-CD was tested where it mattered least:
`assign_atoms_prod`, the cut primitives and the leave-cc-out experiment arms were covered, while
the functions deciding what lands in train/val/test were not. Every bit-exactness claim in §4 came
from building and diffing by hand.

**P4.1 — end-to-end, both paths.** `tests/test_production_splits.py` rebuilds each production
bundle and diffs per-fold `pair_key` digests against `tests/golden/production_splits/`. It invokes
`scripts/production_split_harness.py` as a subprocess rather than re-implementing the digest, so
test and script cannot drift. All three constraints held:
- Skips rather than fails when the corpus or a golden is absent; all three skip branches verified.
- Deselected by default via a `production_split` marker (`pyproject.toml`'s first
  `[tool.pytest.ini_options]`). Default suite 2:45; the guards 5:09 — adding them unconditionally
  would have tripled it. A command-line `-m` overrides the `addopts` filter, verified both ways.
- The harness failure message names the `capture` regeneration path and separates the two causes:
  split code moving pairs is the regression, a rebuilt corpus is legitimate.

**P4.2 — Tier 1 unit tests.** `tests/test_fold_makers_2d_cd.py` (15 tests, 3.3s) and
`tests/test_fold_makers_1d_cd.py` (16 tests, 1.1s), covering every function listed for both paths.
They pin properties, not values: atoms whole in every split, val measured against the whole set,
negatives drawn only from their own split, folds partitioning the positives exactly once. For
1D-CD the central pair asserts the guarantee **and its asymmetry** — the constrained slot never
spans splits while the unconstrained slot recurs; if the second ever failed, the split would have
silently become 2D-CD.

Every invariant was verified able to fail by mutating the code and re-running: 4/4 for 2D-CD, 5/5
for 1D-CD. That pass caught a test that passed for the wrong reason — with atom sizes
`(40,30,20,6,4)`, the correct val target and the incorrect one land on the same accumulation
boundary, so the assertion held either way; single-row atoms make it discriminate 10 vs 8. The two
`pytest.raises` tests were separately checked for firing on the condition they name rather than an
unrelated error.

`_resolve_spec`'s validation was listed for this phase and is **not** covered: it is exercised
indirectly wherever a test resolves a bundle, but has no test of its own. Small, and the F6 safety
argument rests on it.

Suite: 152 → 183 tests, still 2:45.

**Tier 2 — orchestration, covered by P4.1 rather than unit-tested.** `_build_positives`,
`_make_folds_for_scope`, `split_dataset_v2`, `generate_all_cluster_disjoint_cv_folds_v2`. Heavy
I/O and mostly wiring; their bugs surface as wrong output, which is what the end-to-end digest
already catches.

**Tier 3 — not tested.** `_side_rep`, `_write_output`, `_write_cc_sizes`, `_write_cc_pair_sizes`.
A test costs maintenance and catches nothing P4.1 misses.

The Tier 2 / Tier 3 boundary held up in practice: the end-to-end guards did catch things during
P4.2 — every fixture mistake showed up as a build or digest failure — and no orchestration bug
appeared that a unit test would have localized better.

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
