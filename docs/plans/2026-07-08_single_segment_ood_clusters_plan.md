# Single-segment OOD clusters (across-cluster separation)

**Status: IN PROGRESS** — the OOD builder (`build_ood_clusters.py`, search → union-find) is
implemented, committed, and **verified** on aa M1 t099 (**234 clusters, 0 cross-cluster violations**);
per-threshold `runtime.json` + preserve-on-cache landed; visualization (`plot_ood_clusters.py`:
separation map + ESM-2 UMAP) done (Next steps 1–3). **Remaining:** the `results/1D_clusters_sizes`
convention decision, the 8-major scale-out (held by choice), and the nt_cds/nt_ctg rollout.

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

- **`--cluster-mode 0`** (Set-Cover / star, `clustering_utils.py:414`) — links each sequence member to a
  representative sequence only. Two sequences `≥ t` identical to each other can land in different clusters.
- **`easy-linclust`** (default, `build_mmseqs_clusters.py:445`) — linear-time k-mer heuristic;
  misses ~28% of `≥50%`-identity pairs (Linclust, Steinegger & Söding 2018). Its similarity graph
  is incomplete, so "different cluster" does not imply "`< t`".

Neither is reachable from a bundle (bundles only point `cluster_id_path` at prebuilt parquets), so
every experiment today inherits clusters without the guarantee.

## Method

Cluster each segment's sequences (per alphabet, per `t`) as **connected components of a
(near-)complete similarity graph**:

```
mmseqs easy-cluster <segment.fasta> <out> <tmp> \
  --min-seq-id t -c 0.8 --cov-mode 0 \
  --cluster-mode 1 --single-step-clustering -s 7.5 --max-seqs <N>
```

- `--cluster-mode 1` — connected-component clustering (the across-different topology).
- `--single-step-clustering` — clusters are components of one alignment graph (no cascade merge).
- `-s 7.5` — most sensitive prefilter → near-complete graph (`linclust` has no `-s`; that is why we
  switch to `easy-cluster`).
- **`--max-seqs <N>` — REQUIRED, set `≥` the per-function unique-seq count.** `mmseqs cluster`
  defaults to `--max-seqs 20`, which truncates each sequence's neighbor list; on dense proteins that
  drops true `≥ t` edges and fragments the connected components — the guarantee fails without it
  (see Findings).
- `-c 0.8 --cov-mode 0`, `--min-seq-id t` — unchanged from today.

Then **verify** the guarantee with a full all-vs-all pass (see Verification).

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

1. **[DONE]** **`clustering_utils.py::run_mmseqs_easy_clust`** — add params `cluster_mode: int = 0`
   (→ `--cluster-mode`), `sensitivity: float | None = None` (→ `-s`), `single_step_clustering: bool = False`
   (→ `--single-step-clustering`); thread into the command.
   Defaults reproduce today's set-cover/linclust bytes (back-compatible). Guard: `-s` /
   `--single-step-clustering` apply to `algorithm='cluster'` only.
2. **[DONE]** **`build_mmseqs_clusters.py`** + **`run_mmseqs_easy_clust`** — add CLI/params
   `--cluster_mode`, `--sensitivity`, `--single_step_clustering`, **`--max_seqs`**. OOD recipe =
   `--algorithm cluster --cluster_mode 1 --single_step_clustering --sensitivity 7.5 --max_seqs <N>`.
3. **Output namespace** — write OOD clusters to `clusters_{alphabet}_ood/tXXX/` (`_ood` not `_cc`,
   to avoid confusion with the bipartite mega-CC; keep the existing set-cover parquets alongside,
   do not replace). Bundles opt in by pointing `cluster_id_path` at the new dir.
4. **Build** OOD clusters for the target segments/alphabets/thresholds. Point
   `--out_root` at the `_ood` namespace and pass the OOD recipe, e.g. HA at t099:

   ```
   python -m src.preprocess.build_mmseqs_clusters \
     --protein_final data/processed/flu/July_2025/protein_final.parquet \
     --out_root      data/processed/flu/July_2025/clusters_aa_ood \
     --thresholds 0.99 --functions HA \
     --algorithm cluster --cluster_mode 1 --single_step_clustering --sensitivity 7.5 \
     --max_seqs 100000 \
     --mmseqs_bin /homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs
   ```
5. **Verify** (`src/analysis/verify_ood_clusters.py`, below) on at least one (segment, alphabet,
   `t`) before wiring any bundle. **[built; not yet run on a `--max_seqs`-fixed set]**

## Verification (certifies the guarantee)

Tool: `src/analysis/verify_ood_clusters.py` — all-vs-all (`mmseqs easy-search`) over the same FASTA
under the clustering's own rule (`--min-seq-id t -c 0.8 --cov-mode 0 --seq-id-mode 0`), then count
cross-cluster pairs. **Guarantee holds ⇔ count = 0.**

1. Run all-vs-all with a **high `--max-seqs`** (`≥ N`) so no neighbor is truncated, at `-s 7.5`
   (for high `t`, any sensitivity finds the near-identical pairs). Do **not** use
   `--exhaustive-search` — in mmseqs 18 it is a slow profile-based iterative search, not a plain
   all-vs-all.
2. Every returned hit already meets `id ≥ t` and `cov ≥ 0.8`; any hit whose endpoints are in
   different clusters is a violation. Report the count (+ cluster-size stats).
3. **Compute:** all-vs-all is `~O(N²)` — CPU is fine for small/dense functions (~12 min for M1's
   4,771 seqs) but impractical for HA (~42k). **GPU** (`--gpu 1`, adds `--createdb-mode 2`)
   accelerates `easy-search`, but the conda build's kernels need Ampere+/Hopper: **V100 (cc 7.0)
   fails** ("invalid device symbol"), **H100 works**. On CPU-only hardware, verify small functions
   fully and sample the large ones.

## Pipeline — generation vs verification (union-find approach)

The M1 diagnostic (2026-07-08) showed `easy-cluster --cluster-mode 1` does **not** return
connected components of the ≥t/cov graph (566 clusters, 3,797 cross-cluster violations), whereas
connected components computed by **union-find over the same `easy-search` all-vs-all graph** give
234 clusters with **0** violations (tool-verified). The two pipelines below share one step — the
all-vs-all `easy-search` (**G2 = V2**) — which is why union-find clusters verify to 0 **by
construction**. Step **G3 (union-find) is not yet in the codebase** (this is the productization
follow-up).

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

## Figure — cluster separation, M1 (explainer; two-panel `M1_separation_heatmap.png`)

> **Production figure (2026-07-10):** the single-panel QC figure is now
> `clusters_aa_ood/figures/M1_t099_separation.png` (via `src/analysis/plot_ood_clusters.py
> --plots separation`). The two-panel union-find-vs-easy-cluster comparison described below
> is explainer-only and no longer on disk (regenerable). See Next steps 2.

**What it shows (plain terms).** A similarity matrix over M1's 4,771 unique aa sequences: a dot at
row *i*, column *j* marks a pair that is **≥ 0.99 identical with ≥ 0.8 coverage** — exactly the pairs
the guarantee constrains. Rows and columns are ordered so each cluster's sequences are contiguous, so
a cluster appears as a square **block on the diagonal**. Both panels plot the *same* dots; only the
cluster labeling (hence the ordering) differs.

- **Left — union-find connected components (234 clusters):** every dot lies inside a diagonal block,
  **nothing off-diagonal**. No two sequences in *different* clusters are ≥ t identical → the guarantee
  holds. The block-diagonal *is* the visual form of "0 cross-cluster pairs."
- **Right — `easy-cluster --cluster-mode 1` (566 clusters):** the same similarity, but **3,797 orange
  dots fall off the diagonal blocks** — pairs of ≥ 0.99-identical sequences split across different
  clusters (the guarantee fails). The largest true component is visibly shattered into strips.

**How it was generated.** Exhaustive mmseqs `easy-search` all-vs-all over the M1 FASTA
(`--min-seq-id 0.99 -c 0.8 --cov-mode 0 --prefilter-mode 2`) yields the edges; each edge is drawn blue
if its endpoints share a cluster, orange if not (`-s 7.5` gives the identical edge set on M1 — see
Findings). The diagnostic script is pending productization (part of the search→union-find builder, B).

**Conclusion.** The union-find clusters are separable by eye *and* provably (0 off-diagonal); the
`easy-cluster` clusters are not. Qualitative companion to the numeric verify (234 → 0 vs 566 → 3,797).

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
- **GPU:** the 8× V100 here (cc 7.0) can't run the conda GPU kernels; H100 (Hopper) can.

## Guardrails

1. **Back-compat / byte-exact.** Default args emit the *identical* mmseqs command; new
   behavior only when opted in. Locked by a unit test on the pure command builder
   (`_build_mmseqs_clust_cmd`).
2. **Test coverage.** `tests/test_clustering_utils.py` did not cover the wrapper; add
   builder tests (default command byte-exact; OOD emits exactly `--cluster-mode 1`, `-s`,
   `--single-step-clustering`).
3. **Ruff-clean.** Run `ruff check` on edited files. `.claude/hooks/ruff_check.sh` is now a
   **wired** PostToolUse hook (2026-07-09; matcher `Edit|Write|MultiEdit`) — parses its JSON via
   `python3` (no `jq` dep) and resolves `ruff` via `$RUFF_BIN` → PATH → `python -m ruff` → the NFS
   `segmatch` env; silent no-op if none found. Verified live 2026-07-09 (a `Write` of an
   unused-import `.py` was blocked with ruff `F401`; a clean `.py` passed silently).
4. **Commits explicit-only; on `master`.** Do not commit without instruction; branch first.
   Keep this change separate from the two pre-staged docstring edits (`_pair_helpers.py`,
   `_split_helpers.py`).
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
- `load_sequence_frame(...)` — the input-load / alphabet-resolve / hash / function-skip logic
  factored out of `build_mmseqs_clusters.main` (491–566); **both** scripts call it (shared, DRY).
  `build_mmseqs_clusters.py` switches to it under a byte-identical-output regression check.
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
parquet + stats CSV + runtime.json. **Supports all three alphabets and any `t`**; first run is aa /
t099 only.

**Also:** relabel `build_mmseqs_clusters.py`'s `--cluster_mode 1` / `--max_seqs` help (458–459,
467–473, 484–489) from "OOD" to "not OOD; use `build_ood_clusters.py`" (doc-only; command path
unchanged).

**Tests.** Unit: `_build_mmseqs_search_cmd` byte-exact (default / `--prefilter-mode 2` / `--gpu` /
nt); `connected_components_from_hits` on a synthetic graph (known CCs, singletons, both edge
directions, deterministic labels); `load_sequence_frame` parity. **Oracle (primary gate):** build M1
aa t099 → assert **234** clusters, run `verify_ood_clusters` → **0** violations. A single
`/code-review` pass is the secondary gate (not a writer↔critic loop).

**Terminology (verify-terminology gate).** The glossary's `Cluster` (`glossary.md:54`) is pinned to
`easy-linclust` output and `Connected component` (`glossary.md:25`) is bigraph-specific; **add
glossary entries** for the OOD notion (single-segment similarity-graph connected component) before
authoring B's docstrings, so "CC" here stays distinct from the bipartite mega-CC.

## Next steps (2026-07-10) — visualization DONE; scale-out pending

**Do NOT re-run the full 8 majors yet.** Validated state: aa **M1 t099 only** (234
clusters, 0 violations). The 8-major run is `build_ood_clusters` with the default
`--functions` into `clusters_aa_ood/`; hold until the viz work below settles.

**1. [DONE] Cluster visualizations — `src/analysis/plot_ood_clusters.py`** (commits `e88fb03`,
`17c9210`). A dedicated, regenerable script that READS the cluster parquet (+ the
`t<NN>/<short>_hits.tsv` graph) and writes figures into
`data/processed/.../clusters_<alphabet>_ood/figures/` **without recomputing clusters**; the
production builders stay figure-free. `--plots {separation,umap}` selects figures.
**DECISION STILL OPEN:** is `results/1D_clusters_sizes` (easy-linclust set-cover) a
tracked/curated (paper) output, or regenerable → migrate under `clusters_aa/figures/`?
(Leaning migrate, to unify the convention.)

**2. [DONE] Separation map (production QC).** `plot_separation_map` →
`figures/<short>_t<NN>_separation.png`: the cluster set's ≥t/cov similarity matrix ordered by
cluster, drawn as a **scatter** (every hit shown — not a shrunk image that could hide a lone
violation) with cross-cluster violations ringed in **red** and a faint identity diagonal.
M1 t099 → 0 off-diagonal (separation holds). This is the single-panel "left-style" production
figure; the old two-panel union-find-vs-easy-**cluster** comparison (`M1_separation_heatmap.png`)
was explainer-only (no longer on disk; regenerable).

**3. [DONE] ESM-2 UMAP colored by cluster.** `plot_cluster_umap` →
`figures/<short>_t<NN>_umap.png`: 2-D UMAP of the **existing** ESM-2 embeddings (no new compute),
joined via `prot_hash → protein_final.esm2_ready_seq → sha1::model_sig → row`. M1 t099:
4,771/4,771 embedded; mid-size clusters form coherent blobs, while the mega-cluster M1_0 (2,013)
spreads across ESM-2 space — the single-linkage-chaining signature at t=0.99 (graph-connected ≠
embedding-compact). Qualitative companion; the block-diagonal separation map stays the rigorous
view. MDS / identity-metric / k-mer projections (options a, c) remain optional follow-ups.

**Remaining scale-out.** After the `results/1D_clusters_sizes` decision: run the 8 majors into
`clusters_aa_ood/` (then nt_cds / nt_ctg), and wire a bundle's `cluster_id_path` at the OOD
combined parquet for an OOD split.

## Out of scope (follow-ups)

- **Fragmenting the single-segment mega-cluster.** On a drift continuum (influenza HA) a protein's
  sequences collapse into one giant **mega-cluster**; splitting it means **dropping bridging
  sequences** — a **node/vertex cut** costing *nodes* (sequences), vs the bipartite mega-CC's **edge
  min-cut** (*pairs*). Separate plan (distinct from `mega-CC`; see `glossary.md` *Mega-cluster*).
- **Within-cluster tightness** (complete-linkage / pruning). Not needed for OOD.
- **nt_cds / nt_ctg rollout** beyond the first validated segment.
