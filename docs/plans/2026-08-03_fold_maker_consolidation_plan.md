# Fold-maker and fragmentation consolidation

**Status: IN PROGRESS**

Date: 2026-08-03

Consolidate the code that turns *atoms* into folds and that fragments the mega-CC to create
those atoms. Three phases are done; two behaviour-changing decisions remain open (§5).

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
an argument. `pytest tests/ -q` green before each commit. Anything that cannot meet this goes to §5.

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

Verification across all three: 2D-CD and 1D-CD t099 HA-NA rebuild md5-identical to the pre-Phase-1
baseline; `pytest tests/ -q` green. Per-finding detail is in the commit messages, not restated here.

## 5. Open

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
  `single_slot`, so only the single-slot paths are reachable.
- **Analysis fidelity.** `fragment_to_targets` reports raw `n_pieces` where production reports live
  atoms, so `min_cut_*.csv` over-reports. An additive `n_atoms` column would close it. (The `kl`
  default and the missing cut floor, the other two gaps, are fixed.)
- **Recorded, not fixed** — plan labels off the fragmentation path: `D1`–`D4`, `OoS`, `P2` (~35
  sites in `_split_helpers` and `dataset_segment_pairs{,_v2}`). `D3`/`D4` appear in raised error
  text, so a user tripping the feasibility guard is pointed at a plan label; that one has a real
  cost.

## 6. Standard

`CLAUDE.md` § Conventions, in particular: docstrings state what the function does with accurate
`Args:`/`Returns:` and no history; names consistent across production scripts and inferable from
the code; function names describe current behaviour, and renames need approval; dense statements
broken into named steps, never a return that also does the work; and no plan-only vocabulary in
code, comments or error text.

## 7. Out of scope

`route_holdout` and the `seq_disjoint` router; the sampling logic inside the negative samplers
(their docstrings are in scope); `dataset_segment_pairs_v2`'s coverage sampler; `src/archive/`.
