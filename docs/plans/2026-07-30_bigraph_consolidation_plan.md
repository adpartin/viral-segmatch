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

**Item 1a is behaviour-preserving** (verified, not assumed):
- `assign_atoms` on aa HA-NA × t099..t095 × {natural, cut}: the full
  `(pair_key, cluster_a/b, cluster_pair_id, cc_id, atom_id)` assignment is byte-identical
  before/after, all 10 configs.
- `bigraph_min_cut` CLI (`--alphabet aa --threshold t095 --method spectral`) reproduces the
  committed 2026-06-07 `min_cut_ha_na_aa_t095_spectral.csv` exactly — this is the path that
  exercises the `targets=None` → 80/10/10 default.
- `pytest tests/ -q`: 146 passed.

## 4. Remaining work (in order)

3. **Retire "bipartite" — as TWO rules, not one.** The word does two different jobs, and they get
   different replacements:

   a. **Graph-structure adjective → `bigraph`.** Where it describes the graph being two-sided:
      glossary first (`Bipartite multigraph` → `Multigraph bigraph`, `Bipartite hub` →
      `Bigraph hub`), then `build_bipartite_multigraph` (18 uses — but see item 4a, which deletes
      it outright, so do 4a first and this shrinks).

   b. **`bipartite component`/`bipartite CC` → `CC`.** Where it names the routing unit, the
      "bipartite" is redundant: `glossary.md` already defines unqualified **CC** as exactly this
      cluster-level bigraph, and reserves *cluster*/*mega-cluster* for the single-segment
      similarity graph — so the qualifier adds nothing the canonical term does not already carry.
      Done for `cluster_analysis_summary.py` + `cluster_disjoint_feasibility.py` (incl. the Plot-C
      output rename `bipartite_largest_pct_vs_threshold.png` →
      `largest_cc_pct_vs_threshold.png`, with `clusters.md` / `splits.md` updated).

   Out of scope for both: the persisted `algorithm` audit values (§1), and the glossary lines where
   "bipartite" is the *standard graph-theory* term being defined (`**Bigraph (bipartite graph)**`)
   or contrasted — those are definitions, not usages, and a blind replace destroys them.

   Total surface: 254 occurrences across 70 tracked files, but ~232 are prose and roughly half of
   those sit in `docs/results/` + `docs/plans/done/`, which are historical record and should be
   left alone (CLAUDE.md: docs describe current state, not history). Only two live Python
   identifiers exist — `build_bipartite_multigraph` and (now renamed) `plot_bipartite_largest_pct`.
   No `nx.bipartite` / `bipartite=` usage anywhere, so nothing is blocked by networkx's own naming.
4. **One bigraph builder** (4a), then the Gen-2 ports (4b).

   a. **Collapse the three graph builders into one.** Three implementations build the same
      cluster-level bigraph: `_megacc_cut.build_pair_bigraph`, `bigraph_properties.
      build_bipartite_multigraph`, and a hand-rolled loop in `_cv_sampling._fragment_atoms`.
      Measured on aa HA-NA t095: all three yield the **same node set, edge set, and edge weights**
      (9,712 nodes / 10,756 edges / total weight 58,826 = the row count).

      **Standardise on the weighted simple `nx.Graph`** (edge `weight` = pairs). Every multigraph
      statistic is recoverable from it via stock networkx `weight=` args — verified equal on the
      real graph for pair mass (`degree(weight=)`), simple degree, CC partition, per-CC pair count
      (`size(weight=)`), bridges, and cut nodes. The multigraph is strictly lossier: it needs an
      `nx.Graph(...)` conversion at every CC just to get bridges/cut nodes back.

      Three blocks:
      - **map** — rows + cluster source → `cluster_id_a/b`. Extend `_split_helpers.
        attach_cluster_ids` (parquet-lookup flavour) with a dict-map flavour, so the analysis and
        CV paths stop inlining `.map()`. One place decides which hash column per alphabet
        (`schema.SCHEMA` is already the source of truth).
      - **build** — `build_pair_bigraph`, unchanged logic, moved with `edges_to_row_index` into a
        new leaf `src/datasets/_bigraph.py` (imports nothing from `src`, so `_megacc_cut`,
        `_pair_helpers`, `_cv_sampling`, and analysis all import it cleanly; removes the lazy
        function-local import at `_pair_helpers.cluster_ccs`). `cluster_ccs` does **not** move — it
        stays beside its locked sibling `sequence_ccs`.
      - **consume** — `cluster_ccs`, the `_megacc_cut` cut loops, and `per_cc_stats` all take `H`.

      Deletes `build_bipartite_multigraph` (45 lines + 18 refs), `weighted_simple` (11 lines), the
      hand-rolled loop in `_fragment_atoms`, and the per-CC `nx.Graph(...)` conversions. Free win:
      the analysis min-cut path inherits canonical node order and stops being row-order sensitive
      (today `build_bipartite_multigraph`'s node order is NOT stable under a row shuffle — measured
      — which is the exact hazard `build_pair_bigraph`'s sorted-insertion comment warns about).

      Verify by diffing `graph_props.csv` / `hub_peel_*.csv` before and after.

   b. **Gen-2 ports** of the surviving `bigraph_*` scripts onto `cc_{source}` artifacts:
      `bigraph_properties` (per-CC λ / bridges / cut nodes / hub Gini), `bigraph_hub_peel`
      (node-peel — the only implementation of that route), `bigraph_min_cut`, `bigraph_pair_2d`
      (no Gen-2 equivalent exists). **Deferred** — converting `bigraph_properties` /
      `bigraph_hub_peel` to `weight=` under 4a is work on scripts this item may rewrite anyway, so
      4a lands the builder and leaves those two consumers for here.
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
pairs where the true nt_cds universe has **79,347** — a 26% undercount (measured on
`cds_dna_final.parquet`; the plan previously recorded 78,764, which did not reproduce).

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
