# Single-segment OOD clusters (across-cluster separation)

**Status: IN PROGRESS**

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
  --cluster-mode 1 --single-step-clustering -s 7.5
```

- `--cluster-mode 1` — connected-component clustering (the across-different topology).
- `--single-step-clustering` — clusters are components of one alignment graph (no cascade merge).
- `-s 7.5` — most sensitive prefilter → near-complete graph (`linclust` has no `-s`; that is why we
  switch to `easy-cluster`).
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
  1. **Completeness.** `-s 7.5` can miss a few `≥ t` pairs → rare false splits (two similar
     sequences in different clusters). The verify pass measures this; for a hard guarantee re-run
     with `--exhaustive-search` / needleall.
  2. **Coverage.** "Different cluster" includes "`≥ t` identical but `< 80%` covered" (a fragment
     or one shared domain). Set `-c` / `--cov-mode` to match your OOD notion.
  3. **Scope.** `t` is measured over aligned columns and is alphabet-specific (`t` on aa ≠ `t` on nt).

## Steps

1. **[DONE]** **`clustering_utils.py::run_mmseqs_easy_clust`** — add params `cluster_mode: int = 0`
   (→ `--cluster-mode`), `sensitivity: float | None = None` (→ `-s`), `single_step_clustering: bool = False`
   (→ `--single-step-clustering`); thread into the command.
   Defaults reproduce today's set-cover/linclust bytes (back-compatible). Guard: `-s` /
   `--single-step-clustering` apply to `algorithm='cluster'` only.
2. **[DONE]** **`build_mmseqs_clusters.py`** — add CLI `--cluster_mode`, `--sensitivity`, `--single_step_clustering`.
   OOD recipe = `--algorithm cluster --cluster_mode 1 --single_step_clustering --sensitivity 7.5`.
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
     --mmseqs_bin /homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs
   ```
5. **Verify** (below) on at least one (segment, alphabet, `t`) before wiring any bundle.

## Verification (certifies the guarantee)

1. Exhaustive all-vs-all on the segment's sequences (`--exhaustive-search` or needleall) →
   per-pair identity + coverage.
2. Count cross-cluster pairs with `identity ≥ t` **and** `coverage ≥ 0.8`.
   **Guarantee holds ⇔ count = 0.** (Exhaustive clusters → exactly 0; `-s 7.5` clusters → report
   the residual; any `> 0` are the missed-edge false splits.)
3. Report cluster-size distribution + largest-cluster fraction (feeds the fragmentation follow-up).

## Guardrails

1. **Back-compat / byte-exact.** Default args emit the *identical* mmseqs command; new
   behavior only when opted in. Locked by a unit test on the pure command builder
   (`_build_mmseqs_clust_cmd`).
2. **Test coverage.** `tests/test_clustering_utils.py` did not cover the wrapper; add
   builder tests (default command byte-exact; OOD emits exactly `--cluster-mode 1`, `-s`,
   `--single-step-clustering`).
3. **Ruff-clean.** Run `ruff check` on edited files. (`.claude/hooks/ruff_check.sh` is a
   PostToolUse ruff hook but is currently unwired.)
4. **Commits explicit-only; on `master`.** Do not commit without instruction; branch first.
   Keep this change separate from the two pre-staged docstring edits (`_pair_helpers.py`,
   `_split_helpers.py`).
5. **Runtime (Steps 2–4).** mmseqs via `MMSEQS_BIN` / the dedicated `mmseqs2` env; `-s` /
   `--single-step-clustering` are `cluster`-only (guarded in the wrapper); write to
   `clusters_{alphabet}_ood/`, never overwrite the set-cover parquets.

## Out of scope (follow-ups)

- **Fragmenting the single-segment mega-CC.** Connected components on a drift continuum (influenza
  HA) collapse into one giant cluster; splitting it means **dropping bridging sequences** — the
  single-segment analog of the bipartite cut costs *nodes* (sequences), not *edges* (pairs).
  Separate plan.
- **Within-cluster tightness** (complete-linkage / pruning). Not needed for OOD.
- **nt_cds / nt_ctg rollout** beyond the first validated segment.
