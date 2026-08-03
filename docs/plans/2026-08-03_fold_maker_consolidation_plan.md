# Fold-maker consolidation (2D-CD and 1D-CD routing)

**Status: IN PROGRESS**

Date: 2026-08-03

Consolidate the functions that turn *atoms* into folds. Five fold-makers exist across two builders;
two of them are production. This plan removes duplication where it costs no behaviour change, fixes
names and docstrings against the current code, and records the divergences that are deliberate.

Related: `docs/plans/done/2026-07-30_bigraph_consolidation_plan.md` (the graph/CC layer this sits on
top of), `docs/methods/glossary.md` (canonical terms), `docs/methods/splits.md`.

---

## 1. The two production paths

Every change is judged against these two. Both cluster nt_cds at **t099** on the `cm0` root, use k=4
and `neg_to_pos_ratio: 1.0`, and draw **within-fold negatives** from the same sampler.

| | **2D-CD** | **1D-CD (HA axis)** |
|---|---|---|
| bundle | `flu_ha_na_cc_nt_cds_cm0_wf.yaml` | `flu_ha_na_1dcd_nt_cds.yaml` |
| builder | `dataset_pairs_cc.py` | `dataset_segment_pairs_v2.py` |
| mode | `cluster_disjoint_cc` | `cluster_disjoint` + `single_slot: a` — **HA** is the cluster-disjoint axis, NA unconstrained |
| atom | one CC on (`cluster_id_a`, `cluster_id_b`); under `edge_cut`, a post-cut fragment of one | one HA cluster |
| router | `make_folds_within_fold` (`dataset_pairs_cc.py:456`) | k-fold branch of `_split_helpers.cluster_disjoint_route_pos_df:591-628` — inline, unnamed |
| test folds | `GroupKFold(shuffle=True, random_state=seed)` on `atom_id` | `GroupKFold` (**unshuffled**) on `cluster_id_a` (`:596`) |
| val carve | `_carve_val_atoms:310` — seeded atom shuffle | `_lpt_bin_pack` over non-test atoms, two bins (`:621`) |
| determinism | seeded | **seed-independent** — `seed` is audited, never consumed (`_split_helpers.py:380-382`) |
| negatives | `within_fold_negatives` (`dataset_pairs_cc.py:395`) | same function, lazily imported at `dataset_segment_pairs_v2.py:1699` |

Both routers partition positives only. Both val carves take whole atoms, so val is cluster-disjoint
from train on both paths — same guarantee, different mechanism.

The builders are not interchangeable: `cluster_disjoint_route_pos_df` raises `NotImplementedError`
for k-fold with bilateral atoms, which is why `dataset_pairs_cc.py` exists.

Six sibling bundles (`flu_ha_{m1,np,ns1,pa,pb1,pb2}_1dcd_nt_cds.yaml`, untracked) pair HA with a
different protein on the 1D-CD path. **HA is always the held-out axis**, but its slot follows
`protein_order`: `single_slot: b` for PA / PB1 / PB2, `a` for NA / NP / NS1 / M1. Code must not
assume slot `a`.

## 2. Constraint

**Bit-exact on both production bundles.** Every Phase-1 change must reproduce the existing
`fold_k/{train,val,test}_pairs.csv` byte-for-byte for `flu_ha_na_cc_nt_cds_cm0_wf` and
`flu_ha_na_1dcd_nt_cds`. Evidence is a before/after capture, not an argument that the change looks
safe. Anything that cannot meet this moves to §6. `pytest tests/ -q` must pass before each commit.

## 3. Current state — four fold-makers over one shared core

Post-Phase-1. `groupkfold_by_atom` (`dataset_pairs_cc.py:378`) is the shared GroupKFold-by-atom core;
both negative scopes route through it. Phase 1 removed a fifth fold-maker, `make_folds`, which had
become a pure delegate to that core and which no bundle reached (F5).

| function | file:line | reached by | test unit | val carve | atoms whole in val? |
|---|---|---|---|---|---|
| `make_folds_within_fold` | `dataset_pairs_cc.py:560` | **2D-CD production** | k atom-groups | `_carve_val_atoms` | yes |
| `_split_helpers` k-fold branch | `_split_helpers.py:591-628` | **1D-CD production** | k atom-groups | `_lpt_bin_pack` | yes |
| `_partition_full` groupkfold arm | `dataset_pairs_cc.py:876` | no bundle (F5) | k atom-groups | `_carve_val_atoms` | yes |
| `make_folds_leave_cc_out` | `dataset_pairs_cc.py:425` | `flu_ha_na_cc_nt_cds_ood_ood_vs_random` | **one atom** per fold | `_carve_val_pairs` | **no** |
| `make_folds_random` | `dataset_pairs_cc.py:449` | same bundle, `paired_random` arm | **rows** | `_carve_val_pairs` | **no** |

`route_holdout` (`_pair_helpers.py:877`) is a sixth router but produces one holdout, not folds.
Callers: `_split_helpers.py:520` and `seq_disjoint_route_pos_df` (`_pair_helpers.py:994`). Neither
production path reaches it; out of scope.

## 4. Findings

Each is verified against the code. The supporting reasoning goes into the code as part of §5, not
here.

**F1 — `make_folds:332-339` and `make_folds_within_fold:471-480` route identically**, differing only
in the frame passed (`full` vs `pos_full`) and local names. The real difference is *when negatives
enter*, which is forced: within-fold negatives come from a split's own positives. Extractable
without behaviour change.

**F2 — both negative scopes are cluster-disjoint; they differ on the cluster shortcut.** Within-CC
negatives take both endpoints from one CC (`_cc_helpers.py:197-201`); within-fold negatives take both
from the split's own positives (`:414-415`), which can be either within or across CCs. `within_cc` is the stricter scope;
both production paths use `within_fold`.

**F3 — within-CC negative budgets are per-CC proportional** (`within_cc_negatives:271`,
`round(r × n_pos_in_cc)`), so `neg_to_pos_ratio` alone leaves the atom size ordering unchanged. Each
function already weights correctly for its scope. **No production change follows.**

**F4 — names say CC where the code means atom.** `test_cc_ids`, `main_cc_pairs`, `tail_cc_pairs` and
the locals in `_partition_full:743` all filter on `atom_id`. Their only caller sets
`edge_cut.enabled: true`, under which `atom_id` is a post-cut fragment, not a CC
(`assign_atoms_prod:178`).

**F5 — no bundle reaches the `within_cc + groupkfold` combination.** It appears only in the group
default `conf/dataset/split_strategy/cluster_disjoint_cc.yaml`, which no bundle selects. Reachable
by `--override`. Says nothing about past builds. *Resolved:* Phase 1 removed the `make_folds`
wrapper this finding was written against; the arm itself stays, inlined in `_partition_full`.

**F6 — `fold_assignment` is unread on the `within_fold` branch** (`_make_folds_for_scope:836`), safe
only because `_resolve_spec:648` rejects the one value that would diverge — 188 lines away.

**F7 — cross-split duplicate negatives are impossible, by an invariant the code never states.**
`within_fold_negatives:421` dedups per call, one call per split (`:483`). Safety rests on every row
carrying a given hash sharing one `atom_id`, plus `_resolve_schema_pair:538` forcing two distinct
proteins.

**F8 — the 2D-CD val carve uses a fixed seed across folds** (`make_folds_within_fold:480`) while its
negatives vary (`:486`); `make_folds_leave_cc_out:370` and `make_folds_random:389` use `seed + i`.
Equally reproducible either way — only fold-to-fold independence differs. 2D-CD only; the 1D-CD
router has no seed. See D1.

**F9 — `cluster_disjoint_route_pos_df` is named for more than it serves.** It implements bilateral
holdout, single-slot holdout and single-slot k-fold, but every `cluster_disjoint` bundle sets
`single_slot`, so only the single-slot paths are reachable. The bilateral holdout code is
live-but-unreached, the way the removed `make_folds` wrapper was (F5). Fix folded into D2.

**F10 — the code calls within-fold negatives "cross-CC".** `_make_folds_for_scope:831` and `:836`,
and `make_folds_within_fold`'s docstring, all say cross-CC; the sampler draws both endpoints from the
split's positives without regard to CC, so a negative may fall within or across CCs.

**Not a defect.** `negative_scope` takes different value sets per mode, but `within_fold` names the
same function in both.

## 5. Phase 1 — bit-exact work items — **DONE (2026-08-03)**

All nine landed, plus the `make_folds` removal agreed after the first pass. Verified: both
production bundles rebuilt and every `fold_k/{train,val,test}_pairs.csv` md5-identical to the
pre-change build (12 files each); `pytest tests/ -q` 146 passed.

1. **Extract the shared GroupKFold-by-atom core** (F1) as `groupkfold_by_atom`. Takes a frame with
   an `atom_id` column, returns the per-fold `(train, val, test)` row partition; each caller passes
   its own frame and keeps its own negative handling. `make_folds` was left as a one-line delegate
   in the first pass, then **removed** — `_partition_full`'s groupkfold arm now calls
   `groupkfold_by_atom` directly, with the "negatives carry their CC's atom_id" note moved to that
   call site.
2. **Move `_carve_val_pairs` next to `_carve_val_atoms`**, which `make_folds` had separated.
3. **Rename cc → atom** (F4) in `make_folds_leave_cc_out`, `make_folds_random`, `_partition_full`:
   `test_cc_ids` → `test_atom_ids`, `main_cc_pairs` → `main_atom_pairs`, `tail_cc_pairs` →
   `tail_atom_pairs`, plus locals. Keep `cc_id` where the code means the CC.
4. **Docstring pass** to the §7 standard over the `dataset_pairs_cc.py` functions this plan touches:
   the three `make_folds_*`, `groupkfold_by_atom`, `_carve_val_atoms`, `_carve_val_pairs`,
   `pick_largest_atoms`, `within_fold_negatives`, `within_cc_negatives`, `_partition_full`,
   `_make_folds_for_scope`. The 1D-CD router is a branch, so its share is
   `cluster_disjoint_route_pos_df`'s docstring (`_split_helpers.py:355-399`) and the comment at
   `:591-594`.
5. **Fix the "cross-CC" wording** (F10) in `_make_folds_for_scope` (comment, runtime print) and
   `make_folds_within_fold`'s docstring: within-fold negatives may fall within or across CCs.
6. **Break down dense statements** per §7, starting with the `return {'': make_folds_within_fold(…)}`
   in `_make_folds_for_scope`. Grepped both builders and `_pair_helpers` for the same shape; the only
   other hits return literal dicts, so nothing else needed changing.
7. **Write F7's invariant and F6's guard into the code** — the reasoning this plan deliberately omits.
8. **Glossary**: add **Within-fold negative**. `docs/methods/glossary.md` defines *Within-CC negative*
   but not its sibling, which is what both production paths use.
9. **Record the router divergence** from §1 in `docs/methods/splits.md`.

Verification for every item: rebuild both production bundles, diff the fold CSVs byte-for-byte, run
`pytest tests/ -q`.

## 6. Deferred — behaviour-changing, each its own decision

- **D1 — vary the val-carve seed per fold** (F8). Cheap and correct, but changes every built dataset.
- **D2 — one router for both production paths.** Both already route positives only, but they disagree
  on two mechanisms: shuffled vs unshuffled `GroupKFold`, and seeded atom shuffle vs LPT for val.
  Either choice changes one path's folds, so D2 cannot be bit-exact on both. Two questions belong
  here: **is LPT required on the 1D-CD val carve?** — not arbitrary, since `_build_audit:179-180`
  measures drift on all three bins and LPT minimizes it while `_carve_val_atoms` can overshoot by a
  whole atom (the 2D-CD path has no such check); and **where should 1D-CD routing live?** (F9) —
  renaming `cluster_disjoint_route_pos_df` now, then restructuring in D2, is churn. If D2 is
  declined, the rename becomes a standalone Phase-1 item.
- **D3 — decide `make_folds`'s fate** (F5). **Closed in Phase 1**: removed. The `within_cc +
  groupkfold` arm it wrapped survives, inlined in `_partition_full`.

D2 carries one further open question: the within_cc arm routes positives + negatives through
`groupkfold_by_atom` while within_fold routes positives only, so a single router for both scopes
would need to know whether `GroupKFold(shuffle=True)` partitions identically when every group size
is scaled by the same constant (F3's `(1+r)` factor). Not checked.

## 7. Standard for every function touched

Full rules: `CLAUDE.md` § Conventions. The four that bite here:

- **Docstring** — first sentence says what the function does; `Args:` and `Returns:` present and
  correct; every claim checked against the code, never inferred from the name; current state only.
- **Names** — consistent across production scripts, meaning inferable from the code. Watch atom vs
  CC vs cluster, row/pair vs atom, `cc_id` vs `atom_id`.
- **Function name** — must describe what the code does now; rename the function when the code significantly changed and the name does not reflect the code functionality.
- **Statement complexity** — break a dense statement into named steps with a brief comment. Never
  return an expression that also does the work: bind the call to a named variable, then return it.
  Example to fix: `_make_folds_for_scope:838`, `return {'': make_folds_within_fold(...)}`.

## 8. Out of scope

`route_holdout` and the `seq_disjoint` router; the sampling algorithms inside the negative samplers
(their docstrings are in scope); `dataset_segment_pairs_v2`'s coverage sampler; `src/archive/`.
