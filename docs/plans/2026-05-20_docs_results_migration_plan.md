# Migration of machine-generated files out of `docs/results/`

**Status: PROPOSED**
**Date:** 2026-05-20.
**Branch:** `master` (will branch off when implementation starts).
**Closes:** BACKLOG.md → Cluster sweep cleanup #4.
**Parent:** the "Aggregator Output Convention" in CLAUDE.md, which
says machine-generated outputs live under `results/{virus}/{data_version}/runs/`
or `data/processed/{virus}/{data_version}/...`, not under `docs/`.
The `docs/` tree is reserved for hand-authored writeups.

## One-line framing

`docs/results/` currently mixes 15 hand-authored experiment writeups
with 6 machine-generated files (2 markdowns + 4 CSVs) and 5
unreferenced PNGs. Migrate the machine-generated files to their
data-companion directories, update the 5 committed references, and
make a per-PNG decision on the orphans.

## Scope of files

### A. Already-redirected-but-orphaned (2 files)

`seq_redundancy_per_function.py` writes its summary markdown. Default
output path was redirected to `<out_root>/redundancy_summary.md` in
commit `74e798e`, but the old outputs are still in `docs/results/`:

- `docs/results/2026-05-15_seq_redundancy_per_function.md` (aa)
- `docs/results/2026-05-15_seq_redundancy_per_function_nt.md` (nt)

Replacement files exist at:
- `data/processed/flu/July_2025/clusters_aa/redundancy_summary.md`
- `data/processed/flu/July_2025/clusters_nt/redundancy_summary.md`

### B. Generator still writes to `docs/results/` (4 files)

`cluster_disjoint_feasibility.py:279` does `df_out.to_csv(args.out_csv)`
and its docstring (lines 32, 40) sets the default `--out_csv` to a
`docs/results/<date>_cluster_disjoint_feasibility*.csv` path:

- `docs/results/2026-05-14_cluster_disjoint_feasibility_ha_na.csv`
- `docs/results/2026-05-14_cluster_disjoint_feasibility_pb2_pb1.csv`
- `docs/results/2026-05-15_cluster_disjoint_feasibility_nt_ha_na.csv`
- `docs/results/2026-05-15_cluster_disjoint_feasibility_nt_pb2_pb1.csv`

The script itself needs the same redirect treatment we did for
`seq_redundancy_per_function.py`.

### C. Unreferenced PNGs (5 files) — to be triaged individually

`docs/results/figs/` contains 5 PNGs with 0 references from
committed code/docs and no script writes to that directory:

- `2026-05-14_cluster_id99_calibration.png`
- `2026-05-14_clustering_schematic.png`
- `2026-05-14_redundancy_bipartite_collapse.png`
- `2026-05-14_redundancy_largest_pct.png`
- `2026-05-14_redundancy_n_clusters.png`

Names suggest paper figures or one-off analysis snapshots. Provenance
not recoverable from the repo. Need per-PNG decisions from the user.

## Target directory choices

For the feasibility CSVs (Section B), there's a precedent in commit
`b7f8510`: `cluster_analysis_summary.py` writes its outputs (tables +
plots) to `results/flu/July_2025/runs/cluster_analysis/`. The
feasibility CSVs are sibling analyses (cluster-pair feasibility for
the same alphabet × pair × threshold space), so a parallel directory
fits naturally:

```
results/flu/July_2025/runs/cluster_disjoint_feasibility/
    feasibility_<pair>_<alphabet>.csv
```

Filename change: drop the date prefix (the script overwrites the file
each run, so the date in the filename is misleading). Add `<alphabet>`
to disambiguate aa vs nt.

For the seq_redundancy markdowns (Section A), the new location is
already in place — just delete the stale `docs/results/` copies.

## Implementation plan

### Phase 1 — `cluster_disjoint_feasibility.py` redirect (Section B)

1. **Change the default `--out_csv` in `cluster_disjoint_feasibility.py`**
   from `docs/results/<date>_*.csv` to
   `results/flu/July_2025/runs/cluster_disjoint_feasibility/feasibility_<pair>_<alphabet>.csv`.
   The script currently requires `--out_csv` to be passed explicitly
   (no auto-derive); add a default that constructs the new path from
   `--pair_short` and `--alphabet`. Update the docstring's example
   invocations (lines 32, 40).
2. **Regenerate the 4 feasibility CSVs** at the new locations by
   re-running `cluster_disjoint_feasibility.py` for each
   (pair, alphabet) combination. Inputs (cluster artifacts) are all
   cached, so each run is seconds. Expected outputs:
   - `results/.../feasibility_ha_na_aa.csv`
   - `results/.../feasibility_pb2_pb1_aa.csv`
   - `results/.../feasibility_ha_na_nt.csv`
   - `results/.../feasibility_pb2_pb1_nt.csv`
3. **Update 2 consumer scripts**:
   - `src/analysis/cluster_analysis_summary.py:115-117` —
     `load_feasibility()` builds paths like
     `feasibility_dir / f'2026-05-14_cluster_disjoint_feasibility_{pair_short}.csv'`.
     Change to `results/.../cluster_disjoint_feasibility/feasibility_{pair_short}_{alphabet}.csv`.
   - `src/analysis/cluster_analysis_summary.py:378-379` — `--help`
     text mentions the old `docs/results/` paths. Update.
   - `src/analysis/plot_aa_vs_nt_cluster_disjoint.py:28` — docstring
     mentions `docs/results/2026-05-15_cluster_disjoint_feasibility_nt_*.csv`.
     Update.
4. **Verify** by re-running `cluster_analysis_summary.py` and
   confirming feasibility-line plots still render correctly.
5. **Delete the 4 old CSVs from `docs/results/`** after verification.

### Phase 2 — old seq_redundancy markdowns (Section A)

1. **Confirm the new files at `<out_root>/redundancy_summary.md`**
   are present and current (they should be — were written during the
   Cluster sweep #1 rebuild on 2026-05-20).
2. **Find and update any references** to the old
   `docs/results/2026-05-15_seq_redundancy_per_function*.md` paths in
   `docs/methods/`, `CLAUDE.md`, `.claude/memory.md`. The audit
   counted 6+3 refs but it's worth grepping once more before edit.
   For each reference, decide: drop it (the new file is companion to
   the data, may not need a doc-level callout) or update it (point at
   the new path).
3. **Delete the 2 old markdowns from `docs/results/`**.

### Phase 3 — `docs/results/figs/` triage (Section C)

5 PNGs, all 2026-05-14, all unreferenced. Per-PNG decision tree:

1. **`clustering_schematic.png`** — possibly hand-drawn for a paper
   figure. Without provenance, hard to recreate. Default: keep, but
   move to a clearer location (`docs/figs/` or `paper_figures/`).
2. **`cluster_id99_calibration.png`** — calibration plot from a
   specific run. If unreferenced, likely safe to delete; superseded
   by current plots in `results/.../cluster_analysis/`.
3. **`redundancy_bipartite_collapse.png`**, **`redundancy_largest_pct.png`**,
   **`redundancy_n_clusters.png`** — naming matches the
   `cluster_analysis_summary.py` outputs at
   `bipartite_largest_pct_vs_threshold.png` / `cluster_counts_vs_threshold.png` /
   `unique_sequence_retention.png`. Likely earlier versions of those
   plots. Default: delete (superseded by current versions at
   `results/.../cluster_analysis/`).

Asks the user to confirm each before deleting.

## Verification checklist (at the end)

- [ ] No file in `docs/results/` is machine-generated (i.e., no
      committed script writes to that path by default).
- [ ] All references to the old paths updated (grep `docs/results/` in
      `src/`, `docs/`, `CLAUDE.md`, `.claude/memory.md` returns only
      references to hand-authored `*.md` files that legitimately live
      there).
- [ ] `cluster_analysis_summary.py` re-runs cleanly using only files
      from `results/.../cluster_disjoint_feasibility/` and
      `data/processed/.../clusters_{aa,nt}/`.
- [ ] CLAUDE.md "Aggregator Output Convention" updated if the canonical
      target dir is something other than what's currently named.

## Risks and edge cases

- **Two run-dates baked into filenames** (`2026-05-14_*` for aa,
  `2026-05-15_*` for nt) — these dates reflect when the feasibility
  analysis was first run. New filename convention drops the date, so
  re-runs are idempotent. Loss: historical timestamp is in git
  history instead.
- **Other generators we missed**: the audit found 2 generators
  (`seq_redundancy_per_function.py`, `cluster_disjoint_feasibility.py`).
  Worth a final grep for `docs/results` in `src/` after Phase 1+2 to
  catch anything else.
- **Hand-authored docs cite frozen filenames**: e.g.,
  `docs/results/2026-05-15_cluster_disjoint_nt_results.md` may quote
  feasibility CSV rows. Renaming the CSV doesn't change the cited
  numbers, but if any doc uses a literal CSV path as a "see this
  file" reference, those break. Grep for the old CSV filenames in
  `docs/results/*.md` before deleting.

## Estimated time

- Phase 1 (cluster_disjoint_feasibility migration): ~30-45 min
  (small script edit + 4 quick re-runs + 3 reference updates + verify).
- Phase 2 (seq_redundancy markdown cleanup): ~10-15 min (mostly
  grep + decide-per-reference).
- Phase 3 (figs/ triage): depends on user input per-PNG.
- Total: ~1 hour active work assuming no surprises.

## See also

- `BACKLOG.md` — Cluster sweep cleanup #4 (will mark DONE when this
  plan is implemented).
- Commit `74e798e` — the seq_redundancy script redirect. Sets the
  pattern this plan extends.
- Commit `b7f8510` — Cluster sweep #1+#2 outputs, which validated the
  `results/flu/July_2025/runs/cluster_analysis/` location as the
  precedent.
- `CLAUDE.md` — "Aggregator Output Convention" section.
