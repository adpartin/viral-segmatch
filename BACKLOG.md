# segmatch — Backlog

Forward-looking todos that aren't roadmap-level and don't yet warrant a
full plan doc. Bullets graduate to `docs/plans/<name>_plan.md` (Status:
IN PROGRESS) when ready to design + implement, then to
`docs/plans/done/` when complete. Conventions:

- Items are numbered per section so they can be referred to as
  "Cluster sweep #2" or "DataSAIL #1" in conversation. Numbering is
  stable: don't renumber when adding new items — append to the end.
  When an item is done, strike it through with `~~done text~~` or
  delete and bump a "DONE on YYYY-MM-DD" note inline.
- Each item names a concrete **next action**, not just an idea.
- Scope estimate inline in parens (`~5 min`, `needs design`, etc.).
- If an item starts wanting sub-headings, promote it to a plan doc.
- Triage periodically; delete dead items rather than letting them
  moulder.

See also: `roadmap_v1.md` / `roadmap_v2.md` (big-picture experimental
plan from meetings) and `docs/plans/` (active and completed plans).

---

## Cluster sweep cleanup

Triggered by the 2026-05-19/20 one-unit-increment sweep + the script
fixes that landed alongside it.

1. ~~**Rebuild cumulative `redundancy_stats.csv`** by re-running
   `seq_redundancy_per_function.py` once for aa and once for nt with
   the full threshold list (1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.90,
   0.85, 0.80). Cluster parquets are cached so each run is a few
   minutes. The merge-fix (commit `74e798e`) now appends rather than
   overwrites. (~5 min compute total.)~~ — **DONE 2026-05-20**:
   aa = 80 rows (8 thresholds × 10 functions); nt = 72 rows
   (9 thresholds × 8 functions; M2/NEP auto-skipped). All cached
   reads, ~1 min total. Outputs live under `data/processed/.../
   clusters_{aa,nt}/redundancy_stats.csv` (gitignored).
2. ~~**Regenerate `mutations_tolerated_table.csv`** by re-running
   `cluster_analysis_summary.py` after the CSV rebuild. The float
   precision fix (commit `67ddb62`) is in place; CSV needs to refresh
   to pick it up. (~1 min.)~~ — **DONE 2026-05-20**: 144 rows
   (8 functions × 2 alphabets × 9 thresholds). Verified PB2 at id090
   now = 76 (previously 75 under the float bug). Bonus: collapse plots
   `cluster_counts_vs_threshold.png` and `bipartite_largest_pct_vs_threshold.png`
   regenerated with the granular id097/id096 trajectory — feeds item #3.
3. ~~**Update `clustering_overview.md` §6 + §8** with new cluster counts
   (per-function redundancy table) + collapse plots covering the
   full id100→id080 trajectory at one-unit resolution. The PB2/PB1
   collapse signature at id097→id096 (89%/79% drop) is the headline
   finding to surface. (~30 min after the CSV rebuild.)~~ — **DONE
   2026-05-20** (commit `f91b805`): §8 rewritten into 4 subsections
   (one-unit-resolution n_clusters table; two-collapse-modes prose;
   largest-cluster-%% companion table; routing implications). §6 left
   unchanged — its id100 retention numbers don't depend on sweep
   granularity. Concrete signature surfaced: PB2 717→77 at id097→id096.
4. **Migrate machine-generated files out of `docs/results/`** —
   scope expanded by the 2026-05-20 audit (Smaller #2). Two
   generators involved:
   - `seq_redundancy_per_function.py` writes the
     `2026-05-15_seq_redundancy_per_function*.md` (2 files). New
     default already points at `<out_root>/redundancy_summary.md`
     (commit `74e798e`); old files just need to be migrated and 9
     references updated in `docs/methods/`, `CLAUDE.md`,
     `.claude/memory.md`.
   - `cluster_disjoint_feasibility.py:279` writes the
     `*_cluster_disjoint_feasibility_*.csv` files (4 files: aa and
     nt × HA/NA and PB2/PB1). Script default **still points at
     `docs/results/`** per docstring lines 32, 40 — needs the same
     out-of-docs/ redirect we did for the seq_redundancy script.
     Referenced from `cluster_analysis_summary.py:115-117` and
     `plot_aa_vs_nt_cluster_disjoint.py:28`; both call sites need
     updating.

   **Plan doc ready: `docs/plans/2026-05-20_docs_results_migration_plan.md`
   (Status: PROPOSED).** Three phases sketched (feasibility-CSV
   redirect, seq_redundancy markdown cleanup, figs/ triage). Total
   estimated ~1 hour active work.
   Open question (deferred to Phase 3): what to do with
   `docs/results/figs/` (5 PNGs, 0 references, unclear provenance).

## DataSAIL follow-ups

Bake-off paused after Phase 0 (see
`docs/plans/2026-05-19_datasail_bakeoff_plan.md`). These are the
items worth revisiting before deciding to fully retire the bake-off.

1. **Compute L(π) on bicc splits at id100, id099, id095** as a shared
   yardstick metric for the paper. Reuse the existing
   `src/analysis/datasail_bakeoff.py` wrapper with DataSAIL in
   measurement-only mode (load our cluster IDs as the partition, ask
   DataSAIL to compute the leakage score). Paper-worthy result.
   (~30–60 min compute when revisited.)
2. (Tier C, low priority) **Resolve the 1-pair-silently-missing edge
   case** from `solver/overflow.py` pre-assignment. Reproducible
   across all 13 Phase 0 configs; same pair every time. Likely a
   small bug in DataSAIL's overflow handling.
3. (Tier C, low priority) **C2 SCIP-returns-None failure** at
   K=10, ε=0.05 — wrapper crashes with
   `'NoneType' object is not subscriptable`. Either fix the wrapper
   to handle the None return gracefully, or characterize when SCIP
   fails.
4. (Tier C, low priority) **Why does I2's ILP have no objective
   function?** `src/solver/id_2d.py` calls `solve(1, constraints,
   ...)` — minimize the constant 1. This explains the unpredictable
   fold-size drift we observed but isn't documented in the paper.
   Worth a GitHub issue against `kalininalab/DataSAIL` if we ever
   re-engage.

## Methodology ideas — possible paper contributions

1. **Boundary-sample drop scheme** — identify sequences at the edge of
   clusters whose presence keeps clusters "close" in similarity space,
   drop a small fraction of them to unlock cleaner bicc routing at
   lower identity thresholds (where bicc currently mega-collapses).
   Hybrid between bicc (no drop) and DataSAIL/LoHi (drop pairs /
   vertices). Could be a genuine methodological contribution. First
   action: **draft a plan doc** outlining the algorithm + a
   small-scale feasibility test on HA/NA at id095.
2. **t-SNE / UMAP train/val/test cluster visualization** with
   confounder overlays (host, hn_subtype, year). Inspired by DataSAIL
   Fig. 3 and P&M's framing of "how representative is your test set."
   First action: **decide on the feature space** (ESM-2 embeddings,
   k-mer features, or both) and the coloring scheme.

## Infrastructure

1. ~~**Validate `segmatch` env end-to-end** by running one Stage 3 +
   Stage 4 cycle with an existing bundle. Env was built 2026-05-19
   but only smoke-tested at the import layer. Until this completes
   successfully we cannot retire the `cepi` env. Suggested bundle:
   `flu_ha_na` (production default).~~ — **DONE 2026-05-20**: ran
   `flu_ha_na` with `dataset.max_isolates_to_process=2000
   training.epochs=5 training.patience=10`. Stage 3 done in 2 min;
   Stage 4 done in 2 min on GPU; model learned (Test F1=0.802,
   AUC-ROC=0.92). No new errors — only pre-existing pandas dtype +
   matplotlib deprecation warnings. **`cepi` env is unblocked for
   retirement.**

## Smaller items / minor polish

1. ~~**`cluster_analysis_summary.py:141`** reads `protein_final.csv`
   for length stats but doesn't pass `keep_default_na=False`. The
   `function` column uses full names (no `'NA'` trap), so it's safe
   today, but adding the kwarg for defensive consistency would
   prevent a future-self foot-gun if anyone ever adds a column with
   the literal `'NA'`. (~2 lines.)~~ — **DONE 2026-05-20** (commit `25280e2`).
2. ~~**`docs/results/` machine-vs-handauthored audit**. The 2026-05-15
   seq_redundancy markdowns aren't the only machine-generated files
   in `docs/results/` — the `*_cluster_disjoint_feasibility_*.csv`
   files are likely also machine outputs. Audit + plan migration to
   `results/` if so.~~ — **DONE 2026-05-20**: 22 files audited.
   15 hand-authored markdowns confirmed (keep). 6 machine-generated
   (2 seq_redundancy markdowns + 4 feasibility CSVs) — migration
   work folded into Cluster sweep #4's expanded scope. 5 orphan PNGs
   in `docs/results/figs/` flagged as unclear (0 references,
   provenance unknown).
