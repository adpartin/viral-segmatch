# CC-structure pre-step: persist the pair universe + per-t CC data

**Status: IMPLEMENTED**

Date: 2026-07-21 (closed 2026-07-31)

## Closing note (2026-07-31)

All seven §7 build-order steps are done, verified against the repo and the artifacts on disk:

1-4. `pair_universe_nt_cds/HA-NA/pairs.parquet` (78,764 pairs) plus, per threshold,
   `pairs_with_cc.parquet` / `cc_sizes.csv` / `cc_cluster_composition.csv` / `cc_summary.json`.
5. `stacked_composition_barplot` is in `src/utils/plot_utils.py`; `pair_key_to_metadata` is in
   `src/datasets/_pair_helpers.py` (and is alphabet-agnostic via `hash_col`).
6. `src/analysis/plot_cc_sizes.py` and `src/analysis/plot_cc_composition.py`.
7. `src/datasets/build_cc_structure.py` is `--config_bundle` driven.

**§6.3 reproduces exactly** — `cc_summary.json` for `cc_nt_cds_ood/HA-NA` returns the table's own
numbers at all four thresholds (t099 85.7% / NA_0 29.9% / K=3; t098 95.1% / 34.2% / K=2;
t097 97.9% / 35.8% / K=2; t095 98.7% / 37.1% / K=2), so the finding it records still holds: one
diffuse mega-CC at every `t`, NA_0 the pair-mass floor throughout, no `t` supporting balanced
5-fold.

Deviations from the plan as written, all in the "did more" direction:
- **§8.2 superseded.** "`_ood` only for now" — three cluster sources are now built
  (`cc_nt_cds_cm0`, `_cm1`, `_ood`), each × 5 thresholds (t099..t095, one more than the four
  planned), each with a `fragmented/` sibling holding the post-edge-cut structure.
- **§4 `figures/` subdir not used.** Plots are written under `results/…` by the plot scripts'
  `--out_dir` / `--out_png` rather than beside the data.
- **§5 is stale in one name only**: it lists `bipartite_components` as the CC helper; that became
  `cluster_ccs` (2026-07-30, `6055a85`), reached through `dataset_pairs_cc.assign_atoms_prod`.

The plan's stated long-term goal — "a correct nt_cds replacement for the ad-hoc `bigraph_*`
analysis (long-term retirement target)" — was reached on 2026-07-31: all four `bigraph_*` scripts
now read these artifacts (`8c421ca`, item 4b of
`docs/plans/2026-07-30_bigraph_consolidation_plan.md`).

Scope: a Stage-2.5 pre-step that builds and persists the positive **pair universe** (per alphabet,
schema-pair) and the per-`t` **CC structure** (sizes, single-side-cluster composition, floor), so CC
analysis -- and the OOD-vs-random experiment -- consume **artifacts** instead of re-running the
Stage-3 builder. Uses the CC-builder's correct positive path (`create_positive_pairs_v2`), not the
analysis `load_pair_universe` (which mislabels/dedups nt_cds). First target: HA-NA nt_cds on the
`_ood` clusters; parametrized for other alphabets/pairs.

Related:
- `docs/plans/2026-07-21_ood_vs_random_split_plan.md` -- the experiment that will sit on top of these
  artifacts (its §7 CC plots need exactly this data).
- `docs/plans/2026-07-08_single_segment_ood_clusters_plan.md` -- the cluster artifacts this mirrors.
- `docs/methods/glossary.md` -- pair universe, connected component (CC), mega-CC, cluster.

---

## 1. Why

Single-side **clusters** are artifact-backed (`clusters_{alphabet}_ood/tXXX/`): you can study any
(alphabet, `t`) straight from disk. The 2D **CC structure** is not -- to see CC sizes, per-CC
composition, or the positive pairs you must run `dataset_pairs_cc.py`, the full Stage-3 builder
(front-end load + fragmentation + fold CSVs). Every CC analysis this session re-did the ~100s
front-end via a scratch script. A persisted CC-structure artifact (the analog of the cluster
artifacts) fixes that, is the foundation the experiment's CC plots need, and is a correct nt_cds
replacement for the ad-hoc `bigraph_*` analysis (long-term retirement target).

## 2. Key structure: the pair universe is `t`-invariant

The positive universe is the **same at every `t`** (78,764 HA-NA nt_cds pairs; only the cluster join
changes). So build the positives **once per (alphabet, schema-pair)** -- the expensive front-end step
-- persist them, then **layer the per-`t` cluster/CC assignment** on top (cheap: join + CCs).

## 3. Artifacts produced

3.1 **Pair universe** (cluster-independent, `t`-invariant) -- built once per (alphabet, schema-pair):
the positive pairs with `cds_dna_hash_a/b` + `prot_hash_a/b` + `assembly_id_a/b` + `func_a/b`.

3.2 **Per (alphabet, schema-pair, cluster-source, `t`):**
- `pairs_with_cc.parquet` -- `pair_key` + `cluster_id_a/b` + `cc_id` (the per-`t` assignment, slim;
  join the universe on `pair_key` for the rest).
- `cc_sizes.csv` -- `cc_id, n_pairs` (feeds `plot_cc_sizes.py`).
- `cc_cluster_composition.csv` -- long-form `cc_id, slot, cluster_id, n_pairs, pct_of_cc` (the
  hub-dominance record: shows "NA_0 = 97% of this CC").
- `cc_summary.json` -- `n_ccs`, largest-CC pair fraction, per-slot **largest-cluster pair mass** (the
  edge-cut floor), and `max_balanced_k = floor(total / largest_cluster_pairs)`.

## 4. Artifact layout (proposed -- confirm names/paths)

```
data/processed/flu/July_2025/
  pair_universe_{alphabet}/{pair}/pairs.parquet     # 3.1  t-invariant, cluster-independent
  cc_{alphabet}_ood/{pair}/                         # CCs on the _ood clusters (cluster-source in the name)
    tXXX/
      pairs_with_cc.parquet                         # 3.2
      cc_sizes.csv
      cc_cluster_composition.csv
      cc_summary.json
    figures/
      cc_sizes_{pair}_{alphabet}_tXXX.png
      cc_cluster_composition_HA_{pair}_{alphabet}_tXXX.png
      cc_cluster_composition_NA_{pair}_{alphabet}_tXXX.png
```
`{alphabet}`=nt_cds, `{pair}`=HA-NA. `pair_universe_*` has no `t`/cluster-source (it depends on
neither); `cc_*_ood` is per cluster-source + `t`. `pairs_with_cc.parquet` is slim --
`(pair_key, cluster_id_a/b, cc_id)`, keyed to the universe by `pair_key` (the full frame is
~144 MB/threshold; slim is ~5 MB).

## 5. Reuse vs new (verified against source)

**Reuse (the whole compute path -- all used in this session's scratch scripts, proven to work):**
- `build_frontend` (`dataset_pairs_cc.py:86`) + `create_positive_pairs_v2`
  (`dataset_segment_pairs_v2.py:112`) -> the pair universe.
- `load_cluster_lookup` / `attach_cluster_ids` (`_split_helpers.py:35`/`:81`) -> per-`t` cluster join.
- `bipartite_components` (`_pair_helpers.py:603`) -> CCs + a size summary.
- `plot_cc_sizes.py` -> the CC-size barplot.

**New (plumbing + two extractions -- no new algorithms):**
- the **orchestration script** (build positives once -> loop `t` -> persist); `dataset_pairs_cc.py`
  builds datasets, not clean CC artifacts, and re-does the front-end each run.
- **persistence + the canonical layout** (§4).
- a **per-CC cluster-composition helper** (currently only inline `groupby`+`value_counts`).
- **extractions to durable homes:** `stacked_composition_barplot` -> `src/utils/plot_utils.py`;
  `pair_key_to_metadata` -> `src/datasets/_pair_helpers.py` (out of `bigraph_pair_metadata.py`, which
  rides the nt_cds-mislabeling `load_pair_universe`). These two are already on the experiment plan.

Script home: `src/datasets/build_cc_structure.py` (Hydra/`--config_bundle` driven, reusing the
dataset-side front-end) -- it is a dataset-side artifact builder, not analysis.

## 6. CC-analysis outputs

6.1 **CC-size barplot** per (alphabet, `t`, pair) -- reuse `plot_cc_sizes.py` on `cc_sizes.csv`. The
`_ood` version of the old `2D_cluster_sizes`.

6.2 **Per-CC cluster-composition** -- the `cc_cluster_composition.csv` + **two stacked-bar plots
per (alphabet, `t`, pair): one HA-side, one NA-side**, top-N CCs as bars stacked by `top-k clusters +
"other"` (a hub-core is one tall solid block; a diffuse CC is a short top block over a big "other").
Same x-order in both so hub-vs-diffuse reads at a glance. Reuses `stacked_composition_barplot`.

6.3 **Floor / K-feasibility summary** -- per `t`, the largest single-cluster pair mass and
`max_balanced_k`. Already measured by scratch (the pre-step formalizes it as `cc_summary.json`):

| t | pairs | largest natural CC | largest cluster (floor) | max balanced K |
|---|---|---|---|---|
| t099 | 78,764 | 85.7% | NA_0 29.9% | 3 |
| t098 | 78,764 | 95.1% | NA_0 34.2% | 2 |
| t097 | 78,764 | 97.9% | NA_0 35.8% | 2 |
| t095 | 78,764 | 98.7% | NA_0 37.1% | 2 |

Finding this makes durable: at every `t` there is one **diffuse mega-CC** (85.7%->98.7%) + a tiny
tail; NA_0 is the pair-mass floor everywhere (29.9%->37.1%), so no `t` supports balanced 5-fold. This
is exactly the input the experiment design needs.

## 7. Build order (incremental -- examine at each step)

1. **Pair universe** -- build + persist `pairs.parquet` for HA-NA nt_cds (3.1). *Examine:* 78,764
   pairs, columns present.
2. **Per-`t` CC assignment** -- join clusters + `bipartite_components`, persist `pairs_with_cc.parquet`
   + `cc_sizes.csv` for t099/t098/t097/t095. *Examine:* CC counts (108->3,350), largest-CC %.
3. **`cc_summary.json`** (floor + maxK) -- *Examine:* reproduces the §6.3 table.
4. **`cc_cluster_composition.csv`** -- *Examine:* NA_0's share of the top CCs (the hub record).
5. **Extract `stacked_composition_barplot`** -> `plot_utils.py` (+ move `pair_key_to_metadata` ->
   `_pair_helpers.py`).
6. **6.1 CC-size barplot** (reuse `plot_cc_sizes.py`) and **6.2 composition plots** (HA-side, NA-side).
   *Examine:* the plots.
7. Wire a `--config_bundle` CLI so it runs per (alphabet, pair) like the other stages.

## 8. Decisions

8.1 **Names/paths** (§4): `pair_universe_*` / `cc_*_ood`; script at
`src/datasets/build_cc_structure.py`.

8.2 **Cluster sources**: `_ood` only for now (the dir name encodes the source, leaving room for
set-cover later).

8.3 **First build**: HA-NA nt_cds at the four `t`; other alphabets/pairs on demand.

8.4 **`pairs_with_cc` columns**: slim -- `(pair_key, cluster_id_a/b, cc_id)`, keyed to the universe
by `pair_key` (the full frame is ~144 MB/threshold vs ~5 MB slim; measured 2026-07-21). Add per-`t`
columns as needed.
