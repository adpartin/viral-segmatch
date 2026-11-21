# viral-segmatch — Project Memory (compact working state)

In the repo so it travels across machines. Read it at session start. Update it when a production
setting changes, when work starts or finishes, or when a decision is made that will still matter
later.

**What belongs here**: current production state, work in progress, environment rules, user
preferences — things that change and that you cannot work out by reading the code. It does not
repeat what these already say:
- **`CLAUDE.md`** — behavioural rules, core vocabulary, conventions. Read it first.
- **`docs/architecture.md`** — descriptive/reference material: pipeline stages, config system,
  source-file map, layering rule, results for the paper plan, roadmap, HPC.
- **`docs/results/`** — canonical writeup for every headline finding.
- **`docs/plans/`** (+ `done/`, `shelved/`) — design/implementation plans.
- **`docs/project_changelog.md`** — relocated implementation log, Removed-keys log, Polaris/Task-11
  history, CV/temporal-holdout implementation detail, pre-writeup exploration notes. Reference
  material, not session-startup reading.

---

## Current Production State
- **Base HA/NA bundle** (`flu_ha_na.yaml`) sets `slot_transform=unit_norm` and
  `interaction=unit_diff+prod`. Every H3N2 result to date used `interaction: concat`, so a bundle
  that inherits `flu_ha_na` must set concat itself to stay comparable;
  `flu_ha_na_h3n2_2024_random_cv4` does, so the Task 1 and Task 2 chains get concat. There is no
  `flu_pb2_pb1.yaml` base bundle.
- **Cluster roots** (`src/preprocess/build_clusters.py`, with the binary from the `mmseqs2` env):
  - A filtered build (`--hn_subtype`, `--year`, `--year_range`) needs its own `--out_root`, named
    by appending the filters, e.g. `clusters_nt_cds_cm0_h3n2_2024`. `verify_out_root_filters`
    refuses to mix two filter sets in one root.
  - When adding a function to an existing cluster root, rerun with the full function list. The
    combined lookup (`combined_cluster.parquet`) and `runtime.json` contain only the functions
    requested in the last run.
- **Routing modes**: `cluster_disjoint_test_only` is described in `docs/methods/splits.md` and the
  glossary but has no code. `single_slot` has been run only on HA-only and PB2-only.
- **Subtype context for the HA-NA CCs**: the HA_0×NA_0 hub is H3N2, HA_1×NA_1 is H1N1, and the
  remaining multi-cluster group is an avian mix. Never call a 95%-nt cluster a "lineage".

## Work In Progress
Each plan's own `**Status:**` line is the truth. This table names the work and its plan only; read
the plan for its state. Shelved plans (stopped before they were finished; may resume) are in
`docs/plans/shelved/`. Most are 2D-CD, OOD-cluster and split-refactor work.

| work | plan |
|---|---|
| Paper: Task 1 (Human-H3N2-2024), Task 2 (4 hosts, 3 subtypes, 2010–2025), Task 3 (future years) | `docs/plans/2026-10-08_paper_plan.md` |
| Aligned codon features | `docs/plans/2026-11-17_aligned_codon_features_plan.md` |
| Per-site codon features | `docs/plans/2026-09-14_codon_site_features_plan.md` |
| Per-site nucleotide features | `docs/plans/2026-08-28_per_site_nt_features_plan.md` |
| Hamming 1-NN baseline on codon features | `docs/plans/2026-10-01_codon_hamming_1nn_baseline_plan.md` |

## Forward-looking work
- Todos: `BACKLOG.md` (numbered and triaged; the single source of truth). Bigger experiments:
  `docs/plans/2026-10-08_paper_plan.md`. Put new items in those files, not here, so this one
  does not fill up with stale lists.
- In-development modules + k-mer scaling limits: `docs/architecture.md`.

## Moving to another machine
Building and training Task 2 reads these untracked inputs, about 3.6 GB. Copy them to the same
paths under the repo root; the readers use relative paths and checksums, so the root may differ.
- `data/processed/flu/July_2025/protein_final.parquet` (578 MB), `ctg_dna_final.parquet`
  (683 MB) and `cds_dna_final.parquet` (836 MB)
- `data/processed/flu/metadata_eda/flu_genomes_metadata_parsed.csv` (37 MB)
- `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/`
  (1.4 GB): the protein alignments, `selected_cds.parquet`, and the aligned codon caches
  (`site_features_codon_<protein>_aligned*`, built 2026-11-18) in each protein's final alignment
  subdirectory.
- `results/flu/July_2025/pair_capacity_8_proteins_4host_3subtype_2010_2025_aligned/` (192 KB)
- Task 1 only: `results/flu/July_2025/pair_capacity_8_proteins_aligned/` (148 KB)

`environment.yml` pins Linux build strings and CUDA packages. On macOS, use the smaller recipe in
`docs/installation.md`.

## Env Management
**Rule**: bioconda / kalininalab CLI tools and experimental Python packages live in **dedicated
conda envs**, never in the `segmatch` pipeline env. **Why**: bioconda pulls a different `libhdf5`
than conda-forge and breaks the precompiled `h5py` wheel (broke the pipeline env twice — mmseqs2
2026-05-15, datasail 2026-05-19). **Never** `conda remove --force` to undo bioconda damage —
rebuild from `environment.yml`.
- `segmatch`: clean pipeline env (conda-forge only, `environment.yml`). Validated 2026-05-20.
- `mmseqs2`: CLI-only, v18.8cc5c. On lambda13 use the NFS path,
  `/nfs/lambda_stor_01/homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs` — the `/homes/...` form
  does not resolve there.
- `datasail`: dedicated env for comparing DataSAIL's splits against ours.
- On lambda13 `$HOME` has no miniconda — use NFS absolute binaries
  (`/nfs/lambda_stor_01/homes/apartin/miniconda3/envs/<env>/bin/python`); bare `conda activate` fails.

## HPC (ALCF Polaris)
- PBS job arrays, not SLURM. Do NOT use Hydra's submitit launcher (SLURM-only).
- Batch mode doesn't source dotfiles: use `#!/bin/bash -l` (login shell) to load PrgEnv/CUDA.
- Stage 2 (embeddings) is GPU-heavy; Stage 4 (training) is modest. The 8-GPU dev cluster has no
  scheduler — use a `subprocess.Popen`-per-fold launcher with `CUDA_VISIBLE_DEVICES`.
- Refs: `polaris_plan.md` (Task 11 phases), `speed_up.md`, `docs/hardware_notes.md`.

## User Preferences
- No emojis in replies unless asked. No refactoring beyond what was asked.
- **One script per purpose**: follow the existing pattern in `src/analysis/` — propose a dedicated
  script with a clear name rather than stretching an existing script to cover a second job. Extend
  an existing script when its purpose is unchanged, and share code through functions rather than
  copies. Pick the obvious answer rather than offering several.
- **Performance changes**: make one only when it is measured, or when speed is required for the
  result to be correct. (CLAUDE.md sets the order correctness > readability > efficiency.)
- **Explaining**: assume the reader does not know the codebase. Cut filler and hedges, not
  technical content.
- **Terms across sources**: check before treating two terms as the same. Superficially synonymous
  terms (DataSAIL I2, Park & Marcotte C3, segmatch seq_disjoint) may differ in dimensionality or in
  which axes they cover.
- **Plots**: don't split by slot when the data is per-pair.
- **Refer to Claude as "Claude"** in committed docs and writeups, not "I" or "my proposal".
