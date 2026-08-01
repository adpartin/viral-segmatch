# Bigraph / cluster-disjoint code consolidation

**Status: IN PROGRESS**

Date: 2026-07-30

Scope: consolidate the code that builds the **cluster-level bigraph** and fragments it by **edge
min-cut**, remove stale analysis scripts, and bring the production split path in line with the
project's coding conventions. Covers `src/datasets/` (the splitters) and the `src/analysis/bigraph_*`
family.

Related: `docs/plans/2026-07-21_cc_structure_prestep_plan.md` (the `cc_{source}` artifacts these
analyses should read), `docs/methods/glossary.md` (canonical terms).

Environment: `/nfs/lambda_stor_01/homes/apartin/miniconda3/envs/cepi/bin/python` (has omegaconf +
networkx; the system python does not). Ruff lives in the `segmatch` env.

---

## 1. Decisions (locked — do not re-litigate)

**Layering.** "Production" means code that participates in producing train/val/test splits,
regardless of directory. It may live in `src/datasets/` or `src/utils/`; `src/analysis/` may import
from it, never the reverse. Testable form:

> No module in `src/datasets/` or `src/utils/` may import from `src/analysis/`.

One exception, documented in `docs/architecture.md` § Layering and at the call site:
`dataset_segment_pairs_v2` → `visualize_dataset_stats`. Plotting is not split-producing, the import
is function-local under `if generate_visualizations`, and a failure degrades to a warning. Any new
such import is a violation, not a second exception — so the grep above is a review aid, not a
CI gate, unless it is taught to skip that one line.

**Terminology.** Use **bigraph** to name variables, functions, scripts, never "bipartite" (unless "bipartite" is an existing package modeules or attributes). Glossary changes land before code changes.

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

**seq_disjoint** is retained as an option and supports nt_cds via the `cds` hash family. It also
stays the repo-wide default (`conf/dataset/default.yaml`), even though nearly every recorded run is
`cluster_disjoint*` — bundles set `mode` explicitly.

**Persisted audit values are frozen.** The split audits keep `algorithm =
'bipartite_cc_lpt_greedy'` / `'bipartite_cc_lpt_greedy_on_cluster_ids'`. These are recorded data,
not identifiers, so the bigraph naming rule does not reach them; renaming would only make new run
dirs disagree with existing ones.

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
- **Layering violation (a) closed** (item 1a). `_cv_sampling` → `src/datasets/`; `fragment_weighted`
  + `uniform_targets` → `_megacc_cut` (its `method=` knob renamed `cut_method=` to match the
  module's other three loops); `cluster_source` → `src/utils/` (its `cluster_map_for_root` is what
  `bigraph_properties.load_cluster_map` delegated to, so `_cv_sampling` now imports it directly).
  `lpt_max_drift` did NOT need moving — `_megacc_cut._lpt_max_drift` was already bit-identical
  (checked on 3,000 random size vectors × 6 target schemes), so the analysis copy was deleted
  rather than relocated. `bigraph_min_cut` keeps only `weighted_simple` + `min_cut_recursive` +
  the CLI and imports the loop back from `_megacc_cut`.
- **Layering violation (b) recorded as the one documented exception** (item 1b) —
  `docs/architecture.md` § Layering (new; carries the rule, the grep, the exception, and the
  `src/archive/` note) plus a call-site comment. The source-file map there was corrected for both
  moves. **Item 1 is closed.**
- **Gen-2 ports** (item 4b). All four `bigraph_*` scripts now read the persisted CC artifacts via
  a new `src/analysis/_cc_artifacts.py` (`cc_dir` / `load_cc_pairs` / `load_cc_bigraph` /
  `add_cc_source_args`) instead of rebuilding the pair universe and re-deriving clusters.
  `--cds_final` / `--clusters_*` / `--schema_pair` give way to `--cc_source` / `--pair` /
  `--threshold(s)`, matching the convention `plot_cc_composition` and `umap_cc` already set.
  Default is **nt_cds / `cc_nt_cds_cm0` / HA-NA / t099..t095** (user decision, 2026-07-31).
  **Item 4b is closed** — and with it, item 4.

  - `hub_peel`, `properties`, `min_cut` are clean ports; `pair_2d` is a documented **hybrid** —
    the artifact is deduped to one row per `pair_key`, so the isolate set behind each pair is
    gone and the modal subtype still comes from `cds_dna_final`.
  - `build_cluster_bigraph` (the Gen-1 "look up clusters, then build" adapter) moved to
    `src/archive/_gen1_bigraph.py` — after the ports its only callers were the four archived
    scripts, so it was a live function serving dead code.
  - Dead-code audit over all 85 graph-related functions in live `src/`: exactly two had zero live
    callers, both removed — `_cv_sampling.assign_cc` (a back-compat alias for
    `assign_atoms(strategy='natural')`) and `_cc_artifacts.load_cc_summary` (added and never used
    in the same session). Confirmed by a filesystem-wide grep, not just a tracked-file one: every
    directory including the ruff-excluded `notebooks/`, `eda/`, `examples/`, `documentation/`,
    `reports/`, and `llms/` (excluding only `.git/` and `data/`) has zero code references to
    either. Caveat: `assign_cc` existed as a back-compat alias, so code OUTSIDE this repo that
    imported it would now break.
  - This closes §6.2 for these four callers by construction — they read the production universe
    (78,764) rather than `load_pair_universe`'s aa-keyed 58,826.
- **"bipartite" retired** (item 3), as the two rules. By the time it ran, 4a had deleted
  `build_bipartite_multigraph` and the Plot-C rename had taken `plot_bipartite_largest_pct`, so
  **no live Python identifier contained the word** — it was purely a prose pass. 65 substitutions
  across the glossary, `splits.md` / `clusters.md` / `dataset_construction_v2_workflow.md`,
  `docs/architecture.md`, `CLAUDE.md`, `.claude/memory.md`, one config comment, and ~20 source
  docstrings/comments. The glossary now states both rules and their exceptions under
  *Bigraph (bipartite graph)*, and `Bipartite multigraph` → `Multigraph bigraph` was rewritten to
  say the project does not build it (true since 4a). Deliberately untouched: the persisted
  `algorithm` audit values and the published algorithm names **bipartite-CC LPT-greedy** /
  **BiCC-Split** (`splits.md` § 1.7.1) that mirror them; `docs/results/`, `docs/plans/`, and
  `docs/project_changelog.md` as historical record; and `preprocess_bunya_protein.py`, where
  "bipartite" means a 2-segment **genome**, not a graph. **Item 3 is closed.**
- **One bigraph builder** (item 4a). New leaf `src/datasets/_bigraph.py` holds
  `build_pair_bigraph`, `edges_to_row_index`, and `ranked_ccs`; `_megacc_cut`, `_pair_helpers`
  (lazy import dropped), `_cv_sampling` (two hand-rolled graph loops dropped), and the four
  `bigraph_*` analysis scripts all consume it. `build_bipartite_multigraph` (45 lines) and
  `weighted_simple` (11) deleted; the hash→cluster mapping adapter `build_cluster_bigraph` was
  introduced here and later moved to `src/archive/` (see the 4b entry). `per_cc_stats` /
  `hub_peel` converted to `weight=` (this only changed the representation they read). **Item 4a is
  closed.** Two determinism bugs fixed on the way: `hub_peel` picked its heaviest candidate by
  scanning a `set` (hash-seed dependent), and `bridges.csv` row/endpoint order followed DFS
  traversal — both now canonical.
- **Hygiene** (item 2). `cluster_source.CLUSTERS_ROOT` is now the single per-alphabet cluster-root
  map, consumed by `_cv_sampling` (its hardcoded `_ROOT` deleted) and the `--clusters_{aa,nt}` CLI
  defaults of `bigraph_min_cut` / `bigraph_hub_peel` / `bigraph_properties` /
  `cluster_pair_weight_topk` / `bigraph_pair_2d`. Both `load_cluster_map` delegates deleted; all 8
  call sites use `cluster_map_for_root` directly. `load_pair_universe` gained
  `pair_key_alphabet` (below). The 4 archived scripts were repointed off the deleted delegate and
  all import again; `src/archive/README.md` now states that repointing is a courtesy, not a
  guarantee. **Item 2 is closed** except the caller flip in §6.2.

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

**Stored `cc_nt_cds_*` artifacts vs current code** (checked 2026-07-31, all 3 sources × 5
thresholds × {natural, fragmented} = 30 artifacts, read-only re-derivation):
- **Partition identical in all 30** — every CC holds the same `pair_key` set. Zero partition changes.
- **All 15 `fragment_audit.json` identical** on `n_cuts` / `n_atoms` / `pairs_dropped` /
  `dropped_frac` / `stopped_reason`. All were built with `cut_method: spectral`, the path item 4a
  leaves untouched.
- **`cc_id` labels differ** (so `cc_sizes.csv` does too). The cause is `6055a85`, not item 4a:
  the artifacts date to 2026-07-26 and the label-ordering change landed 2026-07-30. Confirmed by
  replicating `cluster_ccs`'s pre-4a body verbatim and diffing — identical in 15/15.
- **Caveat**: the CC structure reproduces, a split derived from it need not. `_lpt_bin_pack` sorts
  atoms by `(-size, cc_id)`, so relabelling can move size-tied CCs between folds (§3's 72% churn).

**Item 4a is behaviour-preserving on every production path** (before/after capture of 4 scripts
× 25 artifacts, plus the `assign_atoms` digest):
- Byte-identical: `min_cut_*_spectral.csv` (aa + nt_cds), both `hub_peel_*.csv`.
- Same content, canonical relabelling: `graph_props.csv` (all 5,700 aa / 5,934 nt stat rows equal
  as a multiset with `cc_id` excluded), `node_degrees.csv`, `cut_nodes.csv`, `bridges.csv` (same
  bridge SET — the diff was endpoint order), `pair_2d cells` (same `(node_a, node_b, n_pairs)`
  and the same `kept` flag on all 10,756 cells), and `assign_atoms` spectral on all 10 configs
  (audits identical).
- **Changed: Kernighan-Lin only.** KL depends on node iteration order, so canonical ordering
  moves it — `min_cut --method kl` t095 went from 1 cut / 5,925 dropped to 2 cuts / 2,640, and
  `assign_atoms(cut_method='kl')` moved in both directions. Neither is "correct"; KL is a
  heuristic. **Unreachable from production**: `_megacc_cut`'s three loops, `_cv_sampling`, and
  `_split_helpers` all default to `spectral`, and all four configs naming `cut_method` set
  `spectral`. Spectral is unaffected because `_bisect` already sorted nodes before the eigensolve.

**Item 1a is behaviour-preserving** (verified, not assumed):
- `assign_atoms` on aa HA-NA × t099..t095 × {natural, cut}: the full
  `(pair_key, cluster_a/b, cluster_pair_id, cc_id, atom_id)` assignment is byte-identical
  before/after, all 10 configs.
- `bigraph_min_cut` CLI (`--alphabet aa --threshold t095 --method spectral`) reproduces the
  committed 2026-06-07 `min_cut_ha_na_aa_t095_spectral.csv` exactly — this is the path that
  exercises the `targets=None` → 80/10/10 default.
- `pytest tests/ -q`: 146 passed.

## 4. Remaining work (in order)

3. **Retire "bipartite"** -- DONE (see §2), as two rules with the algorithm names kept.
4. **One bigraph builder + Gen-2 ports** -- DONE (see §2). Items 4a and 4b both closed.
5. **Reconcile** `docs/plans/2026-07-21_cc_structure_prestep_plan.md` — its §7 build order is
   complete; either close it or record what remains.

## 5. Walkthrough (understanding pass, interleaves with the above)

`build_pair_bigraph` done. Remaining: `_bisect` → `fragment_largest_cc` (one cut); `fragment_until`
vs `apply_drop_budget_cut` (the two stop conditions); `route_holdout` + `make_folds` (atoms →
splits); `_cv_sampling.assign_atoms` (the third split path).

## 6. Open questions (need a decision)

6.2 **`load_pair_universe` nt_cds undercount — which callers to flip.** The function now takes
`pair_key_alphabet`, but its default is still `'aa'`, so nothing changed yet: every current caller
builds the universe once with the aa key and then maps nt_cds hashes off it. `keep='first'` picks
one arbitrary CDS representative per protein pair, so those nt_cds analyses see **58,826** HA-NA
pairs where the nt_cds-keyed universe has **79,347** — a 26% undercount (both measured on
`cds_dna_final.parquet` via `load_pair_universe`). Not to be confused with the **78,764**
recorded elsewhere in this plan and in `fragment_audit.json`: that is the *production* HA-NA
nt_cds universe (`build_pair_universe`, after the v2 filters), a different and equally correct
quantity. Compare like with like when judging the undercount.

Flipping a caller changes its published nt_cds numbers, so it is a decision, not a cleanup. The
callers, and what each would need:
- `cluster_pair_weight_topk.main` — loops BOTH alphabets off one universe; would need one universe
  per alphabet, i.e. a real restructure of the loop.
- `bigraph_properties`, `bigraph_hub_peel`, `bigraph_min_cut`, `bigraph_pair_2d` — one universe per
  run, `--alphabet` already selects the cluster side; a one-line change each.
- `cluster_disjoint_cv_experiment`, `cluster_disjoint_regime_cv`, `_cv_sampling` callers — the aa
  path is the maintained one and nt is documented as not wired, so likely leave.

Cheapest honest next step: flip the four single-alphabet bigraph scripts, re-run one nt_cds slice,
and diff against the recorded figures before touching anything else.

6.1 **ood-arm attribution.** The `ood_vs_random` t095 ood arm moved −0.159 test F1 on
negative-resampling alone (§3). The arm is at chance in both versions, so this is most likely
instability rather than a capability change — but that is untested. Control: re-run the ood arm
across several negative-sampling seeds and compare the spread to the observed delta.
