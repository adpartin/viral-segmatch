# Bigraph / cluster-disjoint code consolidation

**Status: IMPLEMENTED**

Date: 2026-07-30 (closed 2026-07-31)

## Closing summary (2026-07-31)

All five work items (§4) and both questions (§6) are closed. What changed, in one place:

**One builder.** `src/datasets/_bigraph.py` is the single home of `build_pair_bigraph` /
`edges_to_row_index` / `ranked_ccs`. It replaced three independent derivations of the same
cluster-level bigraph, and the representation is now a weighted simple `nx.Graph` everywhere —
the multigraph is gone, having been shown equal on every statistic that used it.

**One layer.** Split-producing code lives in `src/datasets` / `src/utils` and never imports
`src/analysis`; the single exception (optional plotting) is documented in `docs/architecture.md`
§ Layering and at its call site.

**One input for the analyses.** The four `bigraph_*` scripts read the persisted `cc_{source}`
artifacts via `src/analysis/_cc_artifacts.py`, so they see the pairs the splitter actually routes
rather than a re-derived universe. This is what closed §6.2 and what the
`2026-07-21_cc_structure_prestep_plan` had set out to enable.

**One vocabulary.** "bipartite" is retired as a graph adjective (→ *bigraph*) and as a component
qualifier (→ *CC*), with the persisted `algorithm` audit values and the published algorithm names
deliberately kept, and virology's "bipartite genome" untouched.

Behaviour was preserved and verified at each step, not assumed: `assign_atoms` byte-identical
across 10 configs; the `bigraph_min_cut` CLI reproducing its committed 2026-06-07 output; all 30
stored `cc_nt_cds_*` artifacts partition-identical with all 15 fragmentation audits matching; and
146 tests passing throughout. The only genuine behaviour change is confined to Kernighan-Lin,
which no production path uses (all default to spectral).

Follow-ups deliberately left open, none blocking:
- ~~`cc_aa_*` artifacts do not exist, so the aa CV harness still uses the Gen-1
  `load_pair_universe` path.~~ **Moot since 2026-07-31**: `cluster_disjoint_cv_experiment`,
  `cluster_disjoint_regime_cv` and `cluster_pair_weight_topk` were archived to `src/archive/`
  (`4e41dfb`), so no live code takes that path and no `cc_aa_*` artifacts are needed.
- **(Corrected 2026-07-31: an earlier draft of this list claimed `_cv_sampling` still carries a
  hardcoded `_ROOT` map. It does not — item 2 replaced it with
  `from src.utils.cluster_source import CLUSTERS_ROOT as _ROOT`.)** The real residual is one level
  down: `data/processed/flu/July_2025` is baked into 21 live `src/` files. 17 are `src/analysis`
  CLI defaults (overridable by flag, normal for a CLI); the production dataset builders derive the
  path from config (`build_cc_structure`: `processed_base = cluster_id_path.parents[2]`). Only two
  are module-level constants on a production path — `cluster_source._PROC` and
  `_cc_helpers._MEMB_DIR` (the latter defaults `build_cc_isolate_pool`'s membership table, already
  overridable via `membership_path`). Latent portability, not a correctness bug: it bites only
  when a second virus or `data_version` appears, and there is none to test a fix against.
- CC artifacts cover HA-NA only, t099..t095; other pairs/thresholds on demand via
  `build_cc_structure.py`.

Scope: consolidate the code that builds the **cluster-level bigraph** and fragments it by **edge
min-cut**, remove stale analysis scripts, and bring the production split path in line with the
project's coding conventions. Covers `src/datasets/` (the splitters) and the `src/analysis/bigraph_*`
family.

Related: `docs/plans/done/2026-07-21_cc_structure_prestep_plan.md` (the `cc_{source}` artifacts these
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
change. A seed-spread control (re-run the ood arm at several negative-sampling seeds) would let the
delta be attributed rather than assumed; §6.1 records the decision not to run it.

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
5. **Reconcile the cc-structure prestep plan** -- DONE. All seven of its build-order steps verified
   complete and its §6.3 table reproduced exactly from `cc_summary.json`; marked IMPLEMENTED and
   moved to `docs/plans/done/`. Its long-term goal (retire the ad-hoc `bigraph_*` analysis in
   favour of these artifacts) was met by item 4b.

**All items are closed.**

## 5. Walkthrough (understanding pass, interleaves with the above)

Covered while this plan ran: `build_pair_bigraph`. Covered afterwards, under
`docs/plans/2026-08-03_fold_maker_consolidation_plan.md`: `route_holdout` and the fold-makers
(atoms → splits). Dropped: `_cv_sampling.assign_atoms`, archived 2026-07-31 (`4e41dfb`).

The remainder — `_bisect` → `fragment_largest_cc`, and `fragment_until` vs `apply_drop_budget_cut`
— moved to §6b of the 2026-08-03 plan, so this closed plan holds no live work.

## 6. Questions — both closed (2026-07-31)

6.1 **ood-arm attribution — accepted, no further work.** The `ood_vs_random` t095 ood arm moved
−0.159 test F1 on negative-resampling alone (§3). The arm sits at chance in both versions
(AUC-ROC 0.53 → 0.49), where F1 is unstable, and the delta is dominated by a single fold (−0.34).
User decision: chance is chance — the seed-spread control is not worth running.

6.2 **`load_pair_universe` nt_cds undercount — resolved by item 4b; production was never
affected.** The concern was that the function's aa-keyed default dedup collapses each protein pair
onto one arbitrary CDS representative, so an nt_cds analysis run off the default sees **58,826**
HA-NA pairs where the nt_cds-keyed universe has **79,347** (both via `load_pair_universe` on
`cds_dna_final.parquet`; the production universe, after the v2 filters, is a third quantity,
**78,764**).

Two findings closed it:

- **No production code calls it.** `dataset_segment_pairs_v2`, `dataset_pairs_cc`, and
  `build_cc_structure` build the universe with `create_positive_pairs_v2` /
  `build_pair_universe`. The three `load_pair_universe` hits under `src/datasets/` and
  `src/utils/` are comments naming it, not calls (checked line by line).
- **The exposed callers are gone.** The four `bigraph_*` scripts stopped using it when 4b pointed
  them at the CC artifacts. What remains is `cluster_disjoint_cv_experiment`,
  `cluster_disjoint_regime_cv`, `cluster_pair_weight_topk`, plus two `scripts/verify_*` checkers
  and the archive — all **aa-only** paths (the CV harness documents nt as "NOT wired"), and on aa
  the default is the correct choice.

The `pair_key_alphabet` parameter stays, so the correct nt_cds behaviour is one argument away if
an nt analysis is ever wired up.

**Not moved to `src/archive/`**, unlike `build_cluster_bigraph`: that had zero live callers, this
has five. Archiving it would force live code to import from the archive, which the layering rule
forbids. If it should eventually go, the order is the one 4b used — retire or port the callers
first, then the function moves on its own. That would need `cc_aa_*` artifacts, which do not exist.
