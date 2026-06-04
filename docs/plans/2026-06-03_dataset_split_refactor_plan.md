# Dataset split refactor — design plan

**Status: DRAFT — proposed for review; not yet approved. Implementation NOT in progress.**

Scope: `src/datasets/` split generation (Stage 3 routing), measured against
`docs/methods/splits.md` §1.1. Assessment 2026-06-03.

---

## 1. Motivation

`splits.md` §1.1–1.3 specifies **one engine** — *define atoms, then LPT-greedy
bin-pack into 80/10/10* — shared by every non-trivial routing mode. The code
implements that engine several times instead of once; each mode was added as a
parallel branch / function / file. Verified evidence (2026-06-03):

- **Bin-packer duplicated.** `_split_helpers.py:148` `_lpt_bin_pack` is an
  extracted helper, but `seq_disjoint_route_pos_df` hand-rolls the identical loop
  inline (`_pair_helpers.py:791-799`) instead of calling it. One algorithm, two
  maintenance points that can drift.
- **Routing split across files by no principle.** `seq_disjoint_route_pos_df` in
  `_pair_helpers.py:716`; `cluster_disjoint_route_pos_df` in `_split_helpers.py:359`.
  Sibling modes, same pattern, different files, no shared base.
- **Dispatch is a 5-branch if/elif** (`dataset_segment_pairs_v2.py:1505-1664`) with
  copy-pasted glue — `train_isolates = sorted(train_pos['assembly_id_a'].tolist())`
  repeated at `:1515`, `:1564`, `:1644`, and a differently-named audit var per branch.
- **CV bolted on as two generators.** `generate_all_cv_folds_v2` (`:1996`, random,
  sklearn KFold + isolate-override) and `generate_all_cluster_disjoint_cv_folds_v2`
  (`:2093`, cluster, LPT route + routed-pos-override), with two matching override
  branches in `split_dataset_v2` (`:1505`, `:1523`). seq_disjoint CV unsupported (`:3047`).
- **Dimensions wired non-uniformly.** `single_slot` is consumed only by
  cluster_disjoint; `seq_disjoint_route_pos_df` doesn't take it (`:716`), so §1.1 row 2
  (`1D-SD`) is permanently "planned". `pair_key_alphabet` threaded by hand with
  repeated `if 'aa'/elif 'nt_cds'` at `:234`, `:413`, `:1436`, `:2027`.
- **v1 carries a full parallel copy** — `dataset_segment_pairs.py` `split_dataset`
  (`:536`) + `generate_all_cv_folds` (`:1193`).

Net: altering one routing mode touches a branch + a file + a CV generator + several
alphabet conditionals, and the duplicated packer can silently drift.

## 2. Target architecture

Separate **what an atom is** (per mode) from **how atoms become splits** (shared,
written once).

**Per-mode — the only mode-specific code:**
```
assign_atoms(pos_df, params) -> Series[atom_id]   # one row per pair -> its atom id
```
- `random` → atom_id = `assembly_id` (isolate)
- `seq_disjoint` → bipartite-CC id on (hash_a, hash_b); `single_slot` → constrained-slot hash
- `cluster_disjoint` → bipartite-CC id on (cluster_a, cluster_b); `single_slot` → constrained-slot cluster
- `metadata_holdout` → emits the split label directly (sentinel that skips packing)

**Shared engine — one implementation each:**
- `lpt_bin_pack(atom_sizes, targets, bin_order)` — the single packer
  (`_split_helpers._lpt_bin_pack`); seq stops hand-rolling its copy.
- `route_holdout(pos_df, atom_id, ratios) -> (train, val, test, audit)` — pack atoms,
  map rows to splits, build audit. Replaces the per-branch routing + glue.
- `route_kfold(pos_df, atom_id, n_folds, ...)` — GroupKFold over the SAME atom_id.
  Replaces both CV generators.
- `build_audit(...)` — one builder, parameterized by which key families to
  overlap-check (seq / dna / cluster).

**Dispatch** becomes a registry `{mode: atom_provider}` — no if/elif. Cross-cutting
params (`pair_key_alphabet`, `hash_key`, `cluster_alphabet`, `single_slot`) collapse
into one `PairKeySpec`/params object that knows its hash column, replacing the
scattered alphabet conditionals.

This maps **1:1 to §1.1**: each row = one atom-provider + flags. The two planned
rows get cheap:
- `1D-SD` = `seq_disjoint` with `single_slot` set (same atom logic as cluster single-slot).
- `2D-CD-test` = a `route_holdout` flag (pack test-only, subsample val from train
  scope), not a new mode file.

**Extensibility — the graph-surgery seam.** Operations that fragment a mega-CC by
cutting bridges / articulation points (`src/analysis/bipartite_graph_properties.py`;
BACKLOG "BiCC improvements"; `docs/results/2026-05-21_bicc_pair_drop_audit.md`) plug
in as a preprocessing step *inside* a cluster atom-provider — it changes how atom_ids
are computed, nothing downstream. The atom-provider boundary is exactly where the
clustering/graph-theory methodology thread (tracked separately) attaches. The refactor
must keep that seam clean; it does **not** block on that research.

## 3. The hard constraint: bit-exact outputs

Like the Phase 2 pair_key migration (regression-guarded at ε=0), every phase must
produce **byte-identical split assignments** (`pair_key → split`, and per-fold) on a
protected bundle set — or the change is flagged as output-affecting and re-baselined
deliberately.

### 3.1 Regression-guard set — by CODE PATH, not by STATUS
The `active` STATUS set (35 of 83 bundles) is almost all `seq_disjoint` + the on-hold
28p random-CV family; the patchy paths live in `experimental` bundles. So cover every
distinct routing path with one representative each:

| Code path | Representative bundle |
|---|---|
| random holdout | `flu_ha_na_random.yaml` |
| random CV (n_folds≥2) | `flu_28_major_protein_pairs_master.yaml` (n_folds=12) |
| seq_disjoint hash_key=seq | `flu_ha_na.yaml`, `flu_pb2_pb1.yaml` |
| seq_disjoint hash_key=dna | *(no bundle — override only; add a test fixture)* |
| cluster_disjoint bilateral aa | `flu_ha_na_cluster_t099.yaml` |
| cluster_disjoint bilateral nt_cds | `flu_ha_na_cluster_nt_t099.yaml` |
| cluster_disjoint single_slot=a aa | `flu_ha_na_cluster_aa_t095_HAonly.yaml` |
| cluster_disjoint single_slot=b aa | *fresh-validate only — no holdout bundle (NA-only/PB1-only untested; only a `_k5` CV build exists)* |
| cluster_disjoint single_slot CV | `flu_ha_na_cluster_aa_t095_HAonly_k5.yaml` |
| metadata_holdout | `flu_ha_na_holdout_year.yaml` (+ `_host`, `_subtype`) |
| pair_key_alphabet=nt_cds | via the `cluster_nt` bundles (default inference) |

*Fresh-validate rows (Track B, §4 P3) — NOT bit-exact baselines: random CV, single_slot CV,
and single_slot=b (v2 CV never validated, §5; NA-only/PB1-only is an untested combination).
The bit-exact guard covers only the holdout rows above.*

### 3.2 Published set (CONFIRMED 2026-06-03)
Bundles whose exact numbers appear in `paper_outline_v2` / `docs/results/` and must
reproduce byte-identical: `flu_ha_na`, `flu_pb2_pb1`, `flu_{ha_na,pb2_pb1}_regimes`,
`flu_{ha_na,pb2_pb1}_tight*`, the `cluster_aa_t*_{HAonly,PB2only}` single-slot sweep,
`flu_{ha_na,pb2_pb1}_cluster_nt_t099/t100`, `flu_ha_na_kmer_aa_k3`,
`flu_{ha_na,pb2_pb1}_human_h3n2_2024`. **Confirmed 2026-06-03 (user).** All are
holdout-mode, so the bit-exact guard is holdout-only — CV is a redesign, not a preserve
target (§5).

### 3.3 Two baselines per guard bundle
Each guard bundle gets two independent baselines that test different things:

- **Fresh golden (the control) — mandatory, every bundle.** Built at current HEAD; the
  refactor must reproduce it bit-exact. Isolates the refactor as the only variable.
- **Historical anchor (known-good) — bonus, on unchanged paths.** An existing dated run
  reused as a baseline. Catches the "both wrong" trap the fresh golden is blind to: if HEAD
  has already drifted from a published result, the fresh golden silently locks it in. Used as
  a **pre-refactor gate** — confirm HEAD reproduces the historical run bit-exact *before*
  touching code; a mismatch with no intervening change = a pre-existing regression caught early.

A historical anchor is valid only if nothing output-affecting changed since it was built:

- ✅ `seq_disjoint` / `random` / `metadata_holdout` / `regimes` (aa; no cluster dependency):
  runs 2026-05-13 → 06-03 reproduce. Proven anchor: `flu_ha_na_regimes_ratio3` already
  reproduces bit-exact under current code.
- ✅ aa cluster_disjoint built **≥ 2026-05-22** (symmetric-linclust switch changed aa cluster
  IDs); nt_cds built **≥ 2026-06-02** (Phase 2 inflated the nt pair universe). E.g.
  `cluster_{t099,nt_t099}` @06-02; `cluster_aa_id095_HAonly` @05-27.
- ❌ aa cluster runs < 05-22, any nt_cds run < 06-02 — stale by documented change; would
  false-alarm. Skip (do not use as anchors).

**Provenance.** Rebuilding a historical run needs its exact (commit, config/overrides, data
version, seed). Older dataset dirs may lack a resolved config — the Phase 2 repro inferred
`neg_to_pos_ratio=3.0` from the `_ratio3_` dir token. If a run's inputs aren't recoverable, it
is not usable as an anchor.

**Threshold realignment.** §3.1 picks the cluster thresholds that already have a valid recent
run (`t099`, `nt_t099`, `t095_HAonly`) so one bundle yields both baselines; the code path is
identical across thresholds, so this costs no coverage.

**Anchor gate — PASSED 2026-06-03 (seq_disjoint-aa).** `flu_ha_na_regimes_ratio3`: the 05-13
run vs its 06-03 rebuild have byte-identical split assignments via `harness extract`
(pos `8acb8b8c`, full `49c9ac2d`, n_pos 58,388) — HEAD reproduces the 3-week-old full-corpus
run bit-exact, positives and negatives. This exercises the full build machinery, so it
confirms no latent HEAD drift on the main path before refactoring. The other paths have no
strong anchor: old cluster runs are stale-by-design (invalid), valid cluster runs (06-02) are
too recent to test drift, and metadata_holdout lost its anchor when its bundle was fixed.

## 4. Staged migration (each phase bit-exact-gated)

- **P0 — harness + baselines.** Build datasets for the §3.1 set at current HEAD and snapshot
  `pair_key→split` (and per-fold) as **fresh goldens** (the control). Also capture the
  **historical anchors** (§3.3) and confirm HEAD reproduces them bit-exact *before* any
  refactor. Acceptance = diff-empty against both.
- **P1 — dedupe packer.** seq_disjoint calls `_lpt_bin_pack`. Smallest, highest-value;
  independently shippable. Verify bit-exact.
- **P2 — atom-provider interface.** Reimplement each mode as `assign_atoms` + shared
  `route_holdout`. Verify per bundle.
- **P3 — CV redesign (separate track, fresh-validated, NOT bit-exact-gated).** v2 CV was never
  validated (§5), so there is no golden to preserve. Build the unified `route_kfold` (GroupKFold
  over `atom_id`) on the atom-provider, validate end-to-end (fold-size balance, per-fold leakage
  audit, a completed Stage-4 aggregation), and retire `generate_all_cv_folds_v2` +
  `generate_all_cluster_disjoint_cv_folds_v2`. The CV guard bundles (random-CV master, cluster
  single-slot `_k5`) are validation inputs here, not bit-exact baselines.
- **P4 — PairKeySpec.** Centralize the alphabet/param conditionals. Verify.
- **P5 — relocate + retire.** One routing module (seq + cluster together); retire v1
  split machinery if unused.
- **P6 (optional) — wire planned modes.** `1D-SD`, `2D-CD-test` — now cheap.

P1 is shippable on its own. P0 must precede everything.

## 5. Open questions / risks

- **CV is a redesign, not a bit-exact-preserve target — RESOLVED 2026-06-03.** v2 CV was
  never validated end-to-end: `dataset_segment_pairs_v2.py:1521` says *"haven't tested the
  CV with v2 yet"* (random path); cluster single-slot CV built fold datasets on 2026-05-27
  (`*_k5_*/cv_info.json`) but none completed Stage-4 aggregation (no `cv_summary`); the only
  complete CV artifact (`old_runs/...cv5_20260226...`) is a Feb-2026 v1-era run on the retired
  `flu_schema_raw_*` bundle; Task 11's Polaris CV failed (0/336 folds). So there is **no
  trustworthy v2 CV baseline to reproduce** — the bit-exact guard does not apply to CV. The
  unified `route_kfold` (GroupKFold over `atom_id`) falls out of the atom-provider refactor and
  is **validated fresh** (end-to-end run, fold-balance + leakage audit), retiring both untested
  generators. See §4 P3.
- **Not yet traced** (confirm before P2): metadata_holdout's exact dispatch (appears to
  flow through `mode: random` + isolate overrides — `splits.md:57`, and the holdout
  bundles set `mode: random`); the negative-sampler coupling (sampler runs per-split on
  `pos_df` — confirm orthogonal to routing so the refactor doesn't perturb negatives).
- **Determinism.** `seed` is recorded but not consumed by the packer (`splits.md` §1.3);
  the refactor must preserve that (introduce no accidental shuffle).

## 6. Non-goals
- No change to clustering, negative sampling, or feature construction.
- No new routing science — graph-surgery is a separate thread; this only keeps the seam open.
- Not a v1 rewrite beyond retiring dead split code.

## 7. See also
- `docs/methods/splits.md` §1.1 (mode catalog), §2 (cluster-disjoint variants)
- `docs/results/2026-05-21_bicc_pair_drop_audit.md`, `BACKLOG.md` (BiCC improvements)
- `docs/plans/done/2026-06-02_phase2_pair_key_migration_plan.md` (bit-exact guard template)
