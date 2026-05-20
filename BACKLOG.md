# segmatch — Backlog

Forward-looking todos that aren't roadmap-level and don't yet warrant a
full plan doc. Bullets graduate to `docs/plans/<name>_plan.md` (Status:
IN PROGRESS) when ready to design + implement, then to
`docs/plans/done/` when complete. Conventions:

- Each bullet names a concrete **next action**, not just an idea.
- Scope estimate inline in parens (`~5 min`, `needs design`, etc.).
- If a bullet starts wanting sub-headings, promote it to a plan doc.
- Triage periodically; delete dead items rather than letting them
  moulder.

See also: `roadmap_v1.md` / `roadmap_v2.md` (big-picture experimental
plan from meetings) and `docs/plans/` (active and completed plans).

---

## Cluster sweep cleanup

Triggered by the 2026-05-19/20 one-unit-increment sweep + the script
fixes that landed alongside it.

- **Rebuild cumulative `redundancy_stats.csv`** by re-running
  `seq_redundancy_per_function.py` once for aa and once for nt with
  the full threshold list (1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.90,
  0.85, 0.80). Cluster parquets are cached so each run is a few
  minutes. The merge-fix (commit `74e798e`) now appends rather than
  overwrites. (~5 min compute total.)
- **Regenerate `mutations_tolerated_table.csv`** by re-running
  `cluster_analysis_summary.py` after the CSV rebuild. The float
  precision fix (commit `67ddb62`) is in place; CSV needs to refresh
  to pick it up. (~1 min.)
- **Update `clustering_overview.md` §6 + §8** with new cluster counts
  (per-function redundancy table) + collapse plots covering the
  full id100→id080 trajectory at one-unit resolution. The PB2/PB1
  collapse signature at id097→id096 (89%/79% drop) is the headline
  finding to surface. (~30 min after the CSV rebuild.)
- **Move `docs/results/2026-05-15_seq_redundancy_per_function*.md`
  to `data/processed/.../clusters_{aa,nt}/`** (matching the convention
  we adopted in commit `74e798e`) and update the 9 references in
  committed docs (`docs/methods/`, `CLAUDE.md`, `.claude/memory.md`).
  Mid-size refactor — propose a plan doc before doing it.

## DataSAIL follow-ups

Bake-off paused after Phase 0 (see
`docs/plans/2026-05-19_datasail_bakeoff_plan.md`). These are the
items worth revisiting before deciding to fully retire the bake-off.

- **Compute L(π) on bicc splits at id100, id099, id095** as a shared
  yardstick metric for the paper. Reuse the existing
  `src/analysis/datasail_bakeoff.py` wrapper with DataSAIL in
  measurement-only mode (load our cluster IDs as the partition, ask
  DataSAIL to compute the leakage score). Paper-worthy result.
  (~30–60 min compute when revisited.)
- (Tier C, low priority) **Resolve the 1-pair-silently-missing edge
  case** from `solver/overflow.py` pre-assignment. Reproducible
  across all 13 Phase 0 configs; same pair every time. Likely a
  small bug in DataSAIL's overflow handling.
- (Tier C, low priority) **C2 SCIP-returns-None failure** at
  K=10, ε=0.05 — wrapper crashes with
  `'NoneType' object is not subscriptable`. Either fix the wrapper
  to handle the None return gracefully, or characterize when SCIP
  fails.
- (Tier C, low priority) **Why does I2's ILP have no objective
  function?** `src/solver/id_2d.py` calls `solve(1, constraints,
  ...)` — minimize the constant 1. This explains the unpredictable
  fold-size drift we observed but isn't documented in the paper.
  Worth a GitHub issue against `kalininalab/DataSAIL` if we ever
  re-engage.

## Methodology ideas — possible paper contributions

- **Boundary-sample drop scheme** — identify sequences at the edge of
  clusters whose presence keeps clusters "close" in similarity space,
  drop a small fraction of them to unlock cleaner bicc routing at
  lower identity thresholds (where bicc currently mega-collapses).
  Hybrid between bicc (no drop) and DataSAIL/LoHi (drop pairs /
  vertices). Could be a genuine methodological contribution. First
  action: **draft a plan doc** outlining the algorithm + a
  small-scale feasibility test on HA/NA at id095.
- **t-SNE / UMAP train/val/test cluster visualization** with
  confounder overlays (host, hn_subtype, year). Inspired by DataSAIL
  Fig. 3 and P&M's framing of "how representative is your test set."
  First action: **decide on the feature space** (ESM-2 embeddings,
  k-mer features, or both) and the coloring scheme.

## Infrastructure

- **Validate `segmatch` env end-to-end** by running one Stage 3 +
  Stage 4 cycle with an existing bundle. Env was built 2026-05-19
  but only smoke-tested at the import layer. Until this completes
  successfully we cannot retire the `cepi` env. Suggested bundle:
  `flu_ha_na` (production default).
- **Decide on tracking `refs/*.md` notes**. The PDFs are now
  gitignored (commit `3761d6a`); the hand-authored notes
  (`joeres2025_datasail_notes.md`,
  `park2012_pair_input_flaws_notes.md`) are still untracked. Worth
  `git add`-ing if they should travel with the repo as project IP;
  leaving untracked if they're personal scratchpads. No action from
  Claude until you decide.

## Smaller items / minor polish

- **`cluster_analysis_summary.py:141`** reads `protein_final.csv`
  for length stats but doesn't pass `keep_default_na=False`. The
  `function` column uses full names (no `'NA'` trap), so it's safe
  today, but adding the kwarg for defensive consistency would
  prevent a future-self foot-gun if anyone ever adds a column with
  the literal `'NA'`. (~2 lines.)
- **`docs/results/` machine-vs-handauthored audit**. The 2026-05-15
  seq_redundancy markdowns aren't the only machine-generated files
  in `docs/results/` — the `*_cluster_disjoint_feasibility_*.csv`
  files are likely also machine outputs. Audit + plan migration to
  `results/` if so.
