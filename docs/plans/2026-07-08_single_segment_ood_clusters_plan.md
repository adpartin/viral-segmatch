# Single-segment OOD clusters (across-cluster separation)

**Status: IN PROGRESS** — code done (Steps 1–2, incl. `--max_seqs`); the across-cluster guarantee
is **not yet verified** (the default `--max-seqs 20` fails it — see Findings; GPU re-verify pending
on an H100).

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

## Findings (2026-07-08)

- **The default `--max-seqs 20` breaks the guarantee.** First smoke on M1 aa t099 (OOD recipe
  *without* `--max-seqs`) gave 593 clusters, and `verify_ood_clusters.py` found cross-cluster pairs
  at 99.2–99.6% identity with full coverage — real violations. Root cause: `mmseqs cluster` defaults
  to `--max-seqs 20`; on dense/conserved M1 that truncates each sequence's neighbor list, so true
  `≥ t` edges are never evaluated and `--cluster-mode 1` fragments the components. Re-clustering with
  `--max-seqs 100000` merged 593 → 566, confirming dropped edges. Hence `--max-seqs ≥ N` is now part
  of the recipe.
- **NOT yet confirmed:** whether `--max-seqs ≥ N` reaches **0** violations (the 566 set is
  unverified), or whether `-s 7.5` leaves a residual needing true-exhaustive. First task on the H100.
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
