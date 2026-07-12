# Single-segment OOD clusters (across-cluster separation)

**Status: IN PROGRESS.** `build_ood_clusters.py` builds OOD clusters as connected components of the
`≥ t`/cov similarity graph (search → union-find). Clusters exist for **M1, HA, NA** at **t099 + t095**
in **aa and nt_cds** (Results), each 0 cross-cluster by construction; **aa M1 t099** is additionally
verified independently (0 violations). `plot_clusters.py` renders separation, umap, and barplot
figures (`--plots separation umap barplot`); OOD hits are stored as compact `hits.parquet`
(`--delete_hits` to drop). **Remaining:** the `results/1D_clusters_sizes` convention decision, the
other majors, nt_ctg, and wiring a bundle's `cluster_id_path` at an OOD combined parquet.

Date: 2026-07-08

## Goal

Generate per-(alphabet, segment, threshold `t`) sequence clusters that guarantee
**across clusters: different sequences** — any two sequences in *different* clusters are less than `t` identical.
These clusters are the **prerequisite input for both 1D cluster-disjoint (1D-CD) and 2D cluster-disjoint (2D-CD)** (OOD)
splits: the router places whole clusters on one fold, so no test sequence is `≥ t` identical to
any train sequence — the split is only as OOD as the clusters feeding it.

**Within-cluster tightness is not a goal** — OOD does not need it (GraphPart's clustering step is
likewise single-linkage — it does not pursue within-tightness either). Scope is single-segment
clusters only (one alphabet, one protein/segment, one `t`); this is GraphPart steps
(i) Alignment + (ii) Clustering.

## Why the current clusters don't qualify

The one clustering path (`build_mmseqs_clusters.py` → `run_mmseqs_easy_clust`) pins two settings
that break the guarantee:

- **`--cluster-mode 0`** (set-cover / star, the `run_mmseqs_easy_clust` default) — links each sequence
  member to a representative sequence only. Two sequences `≥ t` identical to each other can land in different clusters.
- **`easy-linclust`** (the `build_mmseqs_clusters.py` default, `--algorithm linclust`) — linear-time k-mer heuristic;
  misses ~28% of `≥50%`-identity pairs (Linclust, Steinegger & Söding 2018). Its similarity graph
  is incomplete, so "different cluster" does not imply "`< t`".

Neither is reachable from a bundle (bundles only point `cluster_id_path` at prebuilt parquets), so
every experiment today inherits clusters without the guarantee.

## Method

Cluster each segment's sequences (per alphabet, per `t`) as **connected components of the `≥ t`/cov
similarity graph**, computed by `build_ood_clusters.py` (details in Implementation B):

1. **dedup** — front-end → FASTA of unique sequences (header = alphabet hash).
2. **all-vs-all** — `mmseqs easy-search` (`--min-seq-id t -c 0.8 --cov-mode 0 -s 7.5 --max-seqs ≥N`;
   nt adds `--search-type 3`) → directed hits, each an `≥ t`/cov edge.
3. **union-find** over the full node set (isolated seqs → singletons) → `{hash → cluster_id}` with
   canonical size-ranked labels. A cluster **is** a connected component of this graph, so no hit
   crosses a cluster boundary **by construction**; `verify_ood_clusters.py` certifies it independently.

**`--max-seqs ≥ N`** (per-function unique-seq count) **is required** and the builder enforces it
(`eff_max_seqs = max(max_seqs, len(nodes))`): the `mmseqs cluster` default of 20 truncates neighbor
lists and drops true `≥ t` edges (see Findings).

> The builder uses union-find, **not** `mmseqs easy-cluster --cluster-mode 1`: set-cover cluster-mode
> does not return true connected components — it *fragments* the graph (M1: 566 clusters, 3,797
> cross-cluster violations vs 234 / 0 for union-find; Findings).

## Guarantee & bounds

(Carry this block into the output/methods docs.)

**Link rule:** two sequences are *linked* when mmseqs finds an alignment with `identity ≥ t` and
both sequences `≥ 80%` covered. **Cluster** = connected component of the link graph.

- **Across: different (the guarantee).** If the link graph is complete, any two sequences in
  different clusters have `identity < t` **OR** `coverage < 0.8`. Relax coverage so it never binds
  and this reduces to: **every cross-cluster pair is `< t` identical.**
- **Within: not guaranteed.** Same-cluster sequences are only chain-connected; a within-cluster
  pair can be far below `t`.
- **Bounds / where it leaks:**
  1. **Completeness (two ways the graph is truncated).** (a) `--max-seqs` caps neighbors per
     sequence — the `cluster` default of 20 is far too low on dense inputs and drops true `≥ t`
     edges; set it `≥ N`. (b) `-s 7.5` can still miss a few `≥ t` pairs. The verify pass measures
     the residual; for a hard guarantee re-run with true-exhaustive alignment.
  2. **Coverage.** "Different cluster" includes "`≥ t` identical but `< 80%` covered" (a fragment
     or one shared domain). Set `-c` / `--cov-mode` to match your OOD notion.
  3. **Scope.** `t` is measured over aligned columns and is alphabet-specific (`t` on aa ≠ `t` on nt).

## Steps

1. **[DONE]** Set-cover wrapper params (`cluster_mode`, `sensitivity`, `single_step_clustering`,
   `max_seqs`) exist on `run_mmseqs_easy_clust` / `build_mmseqs_clusters.py` (back-compat byte-exact).
   OOD does **not** use them — the guarantee needs union-find (Method; Findings), so it lives in the
   dedicated **`build_ood_clusters.py`** (Implementation B).
2. **[DONE]** **Output namespace** — `clusters_{alphabet}_ood/tXXX/` (`_ood` not `_cc`, to avoid the
   bipartite mega-CC; set-cover parquets kept alongside, never overwritten). Bundles opt in via
   `cluster_id_path`.
3. **[DONE]** **Build** — `build_ood_clusters.py`, e.g. HA + NA, aa, both thresholds:

   ```
   python -m src.preprocess.build_ood_clusters \
     --protein_final data/processed/flu/July_2025/protein_final.parquet \
     --out_root      data/processed/flu/July_2025/clusters_aa_ood \
     --thresholds 0.99 0.95 --functions HA NA --threads 32 \
     --mmseqs_bin /nfs/lambda_stor_01/homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs
   ```
   nt_cds: swap `--protein_final` → `--cds_dna_final …/cds_dna_final.parquet`, `--out_root` →
   `clusters_nt_cds_ood`.
4. **Verify** (`verify_ood_clusters.py`) a sample before wiring a bundle — see Verification.
   **[aa M1 t099 independently verified → 0. HA/NA: construction-only (each set = the CCs of its own
   `-s 7.5` hits) + M1's `-s 7.5 ≡ --prefilter-mode 2` proof; no independent HA/NA verify run — see
   Results.]**

## Verification (certifies the guarantee)

Tool: `src/analysis/verify_ood_clusters.py` — all-vs-all (`mmseqs easy-search`) over the same FASTA
under the clustering's own rule (`--min-seq-id t -c 0.8 --cov-mode 0 --seq-id-mode 0`), then count
cross-cluster pairs. **Guarantee holds ⇔ count = 0.**

1. Run all-vs-all with a **high `--max-seqs`** (`≥ N`) so no neighbor is truncated, at `-s 7.5`
   (for high `t`, any sensitivity finds the near-identical pairs). Do **not** use
   `--exhaustive-search` — in mmseqs 18 it is a slow profile-based iterative search, not a plain
   all-vs-all. **Caveat: `verify_ood_clusters.py` *defaults* to `--exhaustive-search`** — always
   pass `--sensitivity 7.5 --max_seqs ≥N`. It does not expose `--prefilter-mode 2` (the
   provable-complete search); for that, extend the tool or run the builder's search with
   `prefilter_mode=2`.
2. Every returned hit already meets `id ≥ t` and `cov ≥ 0.8`; any hit whose endpoints are in
   different clusters is a violation. Report the count (+ cluster-size stats).
3. **Compute:** all-vs-all is `~O(N²)` — CPU is fine for small/dense functions (~12 min for M1's
   4,771 seqs) but slow for HA/NA (~40k, hours; the `-s 7.5` prefilter passes the whole set to a full
   CPU alignment). **GPU (unresolved):** this env's mmseqs has GPU support (`--gpu` / `--gpu-server`;
   Kallenborn 2024) and lambda13 is 8× H100 NVL, but a quick `easy-search --gpu 1 --createdb-mode 2`
   test did **not** engage the GPU (0% util — the flag is "use GPU *if possible*", and it fell back to
   CPU). Real acceleration needs the mmseqs2-GPU path (`makepaddedseqdb` / `gpu-server`) — **untested**;
   treat CPU as the only confirmed route for now.

## Pipeline — generation vs verification (union-find approach)

The M1 diagnostic (2026-07-08) showed `easy-cluster --cluster-mode 1` does **not** return
connected components of the ≥t/cov graph (566 clusters, 3,797 cross-cluster violations), whereas
connected components computed by **union-find over the same `easy-search` all-vs-all graph** give
234 clusters with **0** violations (tool-verified). The two pipelines below share one step — the
all-vs-all `easy-search` (**G2 = V2**) — which is why union-find clusters verify to 0 **by
construction**. (G3 union-find is now implemented in `build_ood_clusters.py`; Implementation B.)

**Cluster generation** (produces `{hash → cluster_id}`):

| Step | In → Out | mmseqs? | union-find? | Deterministic? |
|---|---|---|---|---|
| **G1** dedup | protein_final (M1: 108,530 rows) → FASTA of **4,771** unique seqs (header=`prot_hash`) | no (pandas+md5) | no | **yes** |
| **G2** all-vs-all search | FASTA → **466,874** directed hits (id≥t & cov≥0.8) = graph edges | **yes** (`easy-search`; prefilter GPU + align CPU) | no | search **empirically yes** (466,874 on 4/4 runs), not proven; **completeness: heuristic** k-mer prefilter — provable completeness needs `--prefilter-mode 2` (nofilter), *not* `-s` or `--exhaustive-search` |
| **G3** connected components | edges + 4,771 nodes → `{hash→cluster_id}`, **234** clusters | no (Python) | **yes** | **yes** (partition order-invariant) |
| **G4** write parquet | mapping → `M1_cluster.parquet` | no | no | **yes** |

**Verification** (checks any cluster parquet; `verify_ood_clusters.py`):

| Step | In → Out | mmseqs? | union-find? | Deterministic? |
|---|---|---|---|---|
| **V1** load | parquet → `{hash→cluster_id}` + nodes | no | no | **yes** |
| **V2** all-vs-all search | same FASTA → **466,874** hits | **yes** (`easy-search`) | no | same as G2 |
| **V3** count cross-cluster hits | hits + mapping → violation count (**0 = holds**) | no | no | **yes** |

All-vs-all appears in **both** pipelines (G2 = V2, one `easy-search`). The only step that is not
fully deterministic/complete is that search; every other step is deterministic. Verification logic:
`easy-search` returns only pairs already meeting id≥t & cov≥0.8, so any returned pair whose two
endpoints carry different `cluster_id`s is a violation of "across clusters: different"; **0 such
pairs ⇔ guarantee holds.**

## Figure — cluster separation (production QC)

`plot_clusters.py --plots separation` → `figures/<short>_t<NN>_separation.png`: the `≥ t`/cov
similarity matrix, sequences ordered by cluster (largest first), drawn as a **scatter** — every hit
a dot (cross-cluster hits ringed **red**, faint identity diagonal). A cluster = a diagonal **block**;
an off-block dot = a cross-cluster link (0 expected). Feasible where the hit set is small enough to
scatter (aa t099; Results). It reads the build's *own* hits, so its 0 is by construction (a structure
view, not a completeness check) — but it would still ring red on a union-find/labeling **bug**.

> Historical: the original two-panel explainer (`M1_separation_heatmap.png`) contrasted union-find
> (234 clusters, 0 off-diagonal) with `easy-cluster --cluster-mode 1` (566, 3,797 off-diagonal) on
> M1. Regenerable, no longer on disk; the contrast now lives in Findings + Method.

## Findings (2026-07-08)

- **The default `--max-seqs 20` breaks the guarantee.** First smoke on M1 aa t099 (OOD recipe
  *without* `--max-seqs`) gave 593 clusters, and `verify_ood_clusters.py` found cross-cluster pairs
  at 99.2–99.6% identity with full coverage — real violations. Root cause: `mmseqs cluster` defaults
  to `--max-seqs 20`; on dense/conserved M1 that truncates each sequence's neighbor list, so true
  `≥ t` edges are never evaluated and `--cluster-mode 1` fragments the components. Re-clustering with
  `--max-seqs 100000` merged 593 → 566, confirming dropped edges. Hence `--max-seqs ≥ N` is now part
  of the recipe.
- **CONFIRMED (2026-07-08/09, H100): `--max-seqs 100000` does NOT reach 0** — the 566-cluster set has
  **3,797** cross-cluster ≥0.99/full-cov pairs (GPU verify). Root cause is not `--max-seqs`: `easy-cluster
  --cluster-mode 1` **fragments** the ≥t/cov components (566 clusters vs **234** true connected components;
  the largest true component of 2,013 seqs is split into 150 easy-cluster clusters). **Fix: build clusters
  as connected components of the `easy-search` all-vs-all graph via union-find** → M1 = 234 clusters, **0**
  violations (tool-verified). So the OOD builder must be search→union-find, not `easy-cluster`.
- **COMPLETENESS proven on M1:** `-s 7.5` and exhaustive `--prefilter-mode 2` return the **identical**
  edge set (233,885 undirected edges; 0 missed, 0 extra), so the 234 clusters are **provably** OOD, not
  just self-consistent. `--prefilter-mode 2` (nofilter) is the provable-complete search — *not* `-s`
  (heuristic, no guarantee) and *not* `--exhaustive-search` (profile-iterative; see below). Cost: M1
  exhaustive all-vs-all ≈ 1.5 min at 32 threads; HA (~42k, longer seqs) is far costlier — for HA, rely on
  `-s 7.5` (empirically complete at high *t*) and spot-check completeness on a sample.
- **`--exhaustive-search` is not a plain all-vs-all** in mmseqs 18 — it builds a profileDB and
  iterates (>14 min on M1). Use `-s 7.5 --max-seqs <big>` instead.
- **GPU (unresolved).** mmseqs here has GPU support (`--gpu` / `--gpu-server`; Kallenborn 2024) and
  lambda13 is **8× H100 NVL (cc 9.0)**. An earlier note claimed "V100 (cc 7.0) fails / H100 works",
  but a quick `easy-search --gpu 1` test on H100 fell back to CPU (0% GPU util — "use GPU *if
  possible*" wasn't met). The proper GPU workflow (`makepaddedseqdb` / `gpu-server`) is untested —
  don't assume a GPU speedup until verified.

## Guardrails

1. **Back-compat / byte-exact.** Default args emit the *identical* mmseqs command; new
   behavior only when opted in. Locked by a unit test on the pure command builder
   (`_build_mmseqs_clust_cmd`).
2. **Test coverage.** `tests/test_clustering_utils.py` locks the pure command builders byte-exact
   (set-cover default; OOD `easy-search`) and the union-find CC labeling (Implementation B → Tests).
3. **Ruff-clean.** Run `ruff check` on edited files. `.claude/hooks/ruff_check.sh` is now a
   **wired** PostToolUse hook (2026-07-09; matcher `Edit|Write|MultiEdit`) — parses its JSON via
   `python3` (no `jq` dep) and resolves `ruff` via `$RUFF_BIN` → PATH → `python -m ruff` → the NFS
   `segmatch` env; silent no-op if none found. Verified live 2026-07-09 (a `Write` of an
   unused-import `.py` was blocked with ruff `F401`; a clean `.py` passed silently).
4. **Commits explicit-only.** Do not commit without instruction; work on a feature branch, not
   `master`.
5. **Runtime (Steps 2–4).** mmseqs via `MMSEQS_BIN` / the dedicated `mmseqs2` env; `-s` /
   `--single-step-clustering` are `cluster`-only (guarded in the wrapper); write to
   `clusters_{alphabet}_ood/`, never overwrite the set-cover parquets.

## Implementation (B) — `build_ood_clusters.py` (search → union-find)

**Status: IMPLEMENTED** (2026-07-09/10; commits `e681e64` / `09d7709` / `e88fb03`, per-threshold
`runtime.json` + preserve-on-cache). A **dedicated** builder — not a new `--algorithm`
on `build_mmseqs_clusters.py`, which would bloat that script's `main`, its command-centric
`write_results_markdown`, and its "redundancy-assessment" identity. The linclust default stays
byte-exact (Guardrail #1). Reusable primitives already live in `clustering_utils.py`, so the
dedicated builder reuses them with no duplication.

**New in `src/utils/clustering_utils.py`** (siblings to `run_mmseqs_easy_clust` / `parse_cluster_tsv`):
- `load_sequence_frame(...)` — input-load / alphabet-resolve / hash / function-skip; **shared** by
  `build_mmseqs_clusters.py` and `build_ood_clusters.py` (one implementation, byte-identical output).
- `_build_mmseqs_search_cmd(...)` — pure, byte-exact-testable builder for `easy-search`
  (`--min-seq-id t -c 0.8 --cov-mode 0 --seq-id-mode 0 -e 0.001 --format-output
  query,target,fident,qcov,tcov`; `-s` *or* `--prefilter-mode 2`; `--max-seqs`; `--gpu` →
  `--createdb-mode 2`; `--dbtype 2` for nt; `--threads`).
- `run_mmseqs_search(...)` — runs it; returns the hits-TSV path.
- `connected_components_from_hits(hits_tsv, nodes, prefix)` — union-find over the **full node set**
  (isolated seqs → singletons) → `{hash → cluster_id}` with a **canonical deterministic label**
  (component rank by size desc, min-hash tiebreak) so IDs are stable across runs.

**`build_ood_clusters.py`** — per (function, threshold): `export_function_fasta` → `run_mmseqs_search`
→ `connected_components_from_hits` → write the standard parquet
(`hash, cluster_id, function, function_short, threshold, alphabet`) → `cluster_size_distribution` →
**assert 0 cross-cluster hits** (a free self-check; clusters are CCs of that graph). Then combined
parquet + stats CSV + runtime.json. **Supports all three alphabets and any `t`**; built for M1, HA,
NA in aa + nt_cds at t099 + t095 (Results).

`build_mmseqs_clusters.py`'s `--cluster_mode 1` / `--max_seqs` help directs OOD users to
`build_ood_clusters.py` (doc-only; its set-cover command path is unchanged).

**Tests** (`tests/test_clustering_utils.py`): `_build_mmseqs_search_cmd` byte-exact (default /
`--prefilter-mode 2` / `--gpu` / nt `--search-type 3`); `connected_components_from_hits` on a
synthetic graph (known CCs, singletons, both edge directions, deterministic labels);
`load_sequence_frame` parity. Oracle: M1 aa t099 → **234** clusters, `verify_ood_clusters` → **0**.

**Terminology.** `glossary.md`'s *Cluster* entry covers both build methods (set-cover vs the
connected-component OOD variant), and *Connected component* / *Mega-cluster* keep these
single-segment similarity-graph clusters distinct from the bipartite mega-CC — so "CC" here never
means an OOD cluster.

## Results (2026-07-12) — M1, HA, NA built for aa + nt_cds at t099 + t095

Cluster counts (unique sequences → connected components; **0 cross-cluster hits by construction** in
every cell). Full stats in each `clusters_{aa,nt_cds}_ood/cluster_stats.csv` + `tXXX/runtime.json`.

| segment | uniq aa / nt_cds | aa t099 | aa t095 | nt_cds t099 | nt_cds t095 |
|---|---|---|---|---|---|
| M1 | 4,771 / 32,413 | 234 | 4 | 2,066 | 21 |
| HA | 41,896 / 65,414 | 4,839 | 201 | 6,409 | 367 |
| NA | 37,488 / 58,887 | 4,889 | 178 | 5,566 | 303 |

- **Diversity vs conservation.** HA/NA fragment into thousands of clusters at t099, hundreds at t095
  (antigenic subtype diversity); conserved M1 collapses to 234 → 4 (aa). Monotonic t099 → t095
  (looser → fewer, larger) everywhere. At nt_cds t095, two mega-clusters dominate HA (HA_0 31.9% +
  HA_1 25.6%; top-12 = 84% of unique).
- **nt finer than aa** — nt_cds keeps codon/synonymous variants → ~1.5× more unique seqs and more
  clusters than aa for the same segment/threshold.
- **Runtime: aa was SLOWER than nt** (aa ~16 h vs nt ~7 h wall; **32 threads, CPU**). At `-s 7.5` the
  aa prefilter passes the *whole* set as candidates → a full all-vs-all alignment (~N² pairs); nt's
  nucleotide k-mers are more selective (~17k of 65k candidates/query), so its alignment is lighter.
  (GPU: `easy-search --gpu 1` did **not** engage the H100s in a quick test — CPU fallback; a proper
  mmseqs2-GPU path is untested, so no confirmed GPU speedup yet — see Findings.)
- **Hit-parquet footprint** (`<short>_hits.parquet`, query/target/fident): aa t099 ~0.02–0.15 GB,
  nt t099 ~2 GB, aa t095 ~2.5–3 GB, **nt t095 13–15 GB** (464 M / 416 M rows — a near-complete graph
  on the mega-clusters). Kept for now; `--delete_hits` (or a later cleanup) drops them.
- **Figures** (`clusters_{aa,nt_cds}_ood/figures/`): **barplot** for all 8 (segment × alphabet ×
  threshold); **ESM-2 UMAP** for aa (4); **separation** map for **aa t099** only (10 M / 7.6 M hits —
  feasible). The rest (98 M–464 M hits) are too dense for a per-hit scatter, and their 0 is
  tautological on the build's own hits anyway (Figure section), so barplots + UMAPs carry them.
- **Verification stance (HA/NA = construction-only, by choice).** Only **aa M1 t099** is
  independently verified (fresh search → 0). HA/NA rest on the **construction guarantee** (each set =
  the connected components of its own build `-s 7.5` hits → 0 cross-cluster by construction) plus M1's
  proof that `-s 7.5 ≡ --prefilter-mode 2` (complete) at high *t*. No independent HA/NA verify was run:
  a fresh `-s 7.5` re-verify is near-tautological (same search) and costly (~10 h CPU at 16 threads),
  and the only check that adds information — `--prefilter-mode 2` (nofilter, provably complete) — is
  not exposed by `verify_ood_clusters.py`. Revisit with a `--prefilter-mode 2` sample if HA/NA ever
  need certification.

## Next steps — remaining scale-out

Done: **M1, HA, NA** for **aa + nt_cds** at **t099 + t095** (Results); viz
(`plot_clusters.py`: separation / umap / barplot) complete. Remaining:

1. **`results/1D_clusters_sizes` convention (OPEN).** Is that easy-linclust set-cover dir a
   tracked/curated (paper) output, or regenerable → migrate under `clusters_aa/figures/`?
   (Leaning migrate, to unify the convention.)
2. **Remaining majors** — `build_ood_clusters` for PB2 / PB1 / PA / NP (and M2 / NEP where present)
   into `clusters_aa_ood/` + `clusters_nt_cds_ood/`, same shape as M1/HA/NA (nt_cds skips any segment
   absent from `cds_dna_final`). `nt_ctg` after.
3. **Wire a bundle** — point a bundle's `cluster_id_path` at an OOD `combined_cluster.parquet`;
   confirm the router places whole clusters on one fold.

**UMAP note:** raw 1280-d ESM-2 → UMAP, **no PCA pre-reduction — kept by choice**: at 37–65k points it
runs in minutes and full-dim keeps all ESM-2 signal (PCA→~50-d would trade faithfulness for speed we
don't need here). Revisit only if a much larger N makes the kNN build the bottleneck. MDS /
identity-metric / k-mer projections remain optional.

## Out of scope (follow-ups)

- **Fragmenting the single-segment mega-cluster.** On a drift continuum (influenza HA) a protein's
  sequences collapse into one giant **mega-cluster**; splitting it means **dropping bridging
  sequences** — a **node/vertex cut** costing *nodes* (sequences), vs the bipartite mega-CC's **edge
  min-cut** (*pairs*). Separate plan (distinct from `mega-CC`; see `glossary.md` *Mega-cluster*).
- **Within-cluster tightness** (complete-linkage / pruning). Not needed for OOD.
- **nt_cds / nt_ctg rollout** beyond the first validated segment.
