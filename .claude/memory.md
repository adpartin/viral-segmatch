# viral-segmatch — Project Memory (compact working state)

Version-controlled (`.claude/memory.md`) so it travels across machines. Read at session start.
Update when production settings change, work moves in/out of flight, or a durable decision is made.

**Scope**: current production state, work in flight, env rules, user preferences — things that
change and aren't derivable from code. This file does NOT duplicate:
- **`CLAUDE.md`** — behavioural rules, core vocabulary, conventions. Read it first.
- **`docs/architecture.md`** — descriptive/reference material: pipeline stages, config system,
  source-file map, layering rule, Key Experimental Findings, Recent Run Outputs, roadmap, HPC.
- **`docs/results/`** — canonical writeup for every headline finding.
- **`docs/plans/`** (+ `done/`) — design/implementation plans.
- **`docs/project_changelog.md`** — relocated implementation log, Removed-keys log, Polaris/Task-11
  history, CV/temporal-holdout implementation detail, pre-writeup exploration notes. Reference
  material, not session-startup reading.

---

## Current Production State
- **Builder**: v2 is the only builder (v1 retired 2026-06-03); the `dataset_segment_pairs.py` CLI
  dispatches to `dataset_segment_pairs_v2.py`. Stage 3/4 are decoupled — Stage 4 takes
  `--dataset_dir` explicitly; provenance in `training_info.json`. **The v1 CLI path is not
  supported for new work** — e.g. its default `pair_key_alphabet` inference is intentionally
  unfixed (`cluster_alphabet=nt_ctg` without an explicit `pair_key_alphabet` silently falls to `aa`).
- **Active HA/NA + PB2/PB1 bundles** (`flu_ha_na.yaml`, `flu_pb2_pb1.yaml`) bake in
  `split_strategy.mode=seq_disjoint`, `hash_key=seq` (protein-level, stricter), and the "Test 3"
  interaction (`slot_transform=unit_norm`, `interaction=unit_diff+prod`).
- **Clustering**: symmetric mmseqs2 `easy-linclust` on BOTH alphabets (since 2026-05-22). Artifacts
  at `clusters_aa/tXXX/<func>_cluster.parquet` (col `prot_hash`) and
  `clusters_nt_cds/tXXX/<func>_cluster.parquet` (col `cds_dna_hash`); pre-Phase-2 easy-cluster +
  idXXX artifacts archived under `clusters_*_archive_*`. Binary via the dedicated `mmseqs2` env,
  resolved through `MMSEQS_BIN` / `--mmseqs_bin` / PATH.
- **pair_key + axis consistency**: `split_strategy.pair_key_alphabet` ∈ `{aa, nt_cds, nt_ctg}` (`aa`
  default). Non-`aa` pair_keys make finer variants distinct positives (nt_cds: codon variants;
  nt_ctg: +UTR), inflating the universe / opening DNA-variant leakage — cite the alphabet in any
  post-2026-06-03 experiment. **`dataset.molecule` master knob** (opt-in): derives
  `cluster_alphabet` + `pair_key_alphabet` + `kmer.alphabet` from one value with a config-load guard
  (`dataset.allow_alphabet_mismatch` to override a deliberate mix); legacy bundles are untouched.
  See `config_hydra._resolve_molecule_alphabets`.
- **Two DNA notions, by purpose**: **contig** DNA (`ctg_dna_seq`/`ctg_dna_hash` — full submitted
  contig; k-mer *features* + nt_ctg clustering) vs **CDS** DNA (`cds_dna_seq`/`cds_dna_hash` —
  coding-only; nt_cds clustering). History not derivable from code: DNA *clustering* was switched to
  CDS on the assumption clustering should be coding-only — never tested vs contig clustering — while
  k-mer *features* stayed contig. Resolved by the nt_cds/nt_ctg refactor
  (`docs/plans/done/2026-06-21_nt_cds_ctg_hash_refactor_plan.md`, closed 2026-06-25): explicit names
  everywhere via the `src/utils/schema.py` registry, `nt_ctg` enabled end-to-end, `dataset.molecule`
  added. Phase C tested CDS-vs-contig on HA/NA t100: feature axis ~flat (~0.3 pp), all three configs
  0.95–0.97 LGBM.
- **Routing modes**: `random`; `seq_disjoint` (hash_key seq|dna); `cluster_disjoint` (bilateral /
  `single_slot: a|b` / planned `cluster_disjoint_test_only`); `metadata_holdout`. `single_slot`
  exercised on HA-only and PB2-only; NA-only / PB1-only and nt single_slot untested.
- **Graph + CC layer** (consolidated 2026-07-31,
  `docs/plans/done/2026-07-30_bigraph_consolidation_plan.md`): ONE builder,
  `src/datasets/_bigraph.build_pair_bigraph`, returning a weighted simple `nx.Graph` (edge `weight`
  = pairs) — no multigraph anywhere. Split-producing code lives in `src/datasets`/`src/utils` and
  never imports `src/analysis` (one documented exception: optional plotting; see
  `docs/architecture.md` § Layering). The four `bigraph_*` analyses read persisted `cc_{source}`
  artifacts via `src/analysis/_cc_artifacts.py` (default `nt_cds_cm0` / HA-NA / t099..t095), not a
  re-derived universe. `m_pos_per_cc` default is `null` (no cap).
- **2D-CD fold balance**: `groupkfold_by_atom` routes with unshuffled `GroupKFold`, whose assignment
  is LPT (largest atom to the lightest fold) — every fold receives one of the k largest atoms and the
  folds come out equal-sized. **Feasibility rule**: an edge cut cannot split a cluster, so balanced
  k-fold needs the heaviest single-side cluster's pair mass `<= 1/k`. HA-NA nt_cds cm0: 8.8% at t099
  rising to 25.8% at t095 — so k=4 is unreachable at t095 whatever the router does. Check the floor
  before choosing k. Detail: `docs/results/2026-08-09_2d_cd_fold_balance.md`.
- **Threshold notation `tXXX`**: on-disk cluster parquets at `clusters_*/tXXX/`; pre-Phase-2
  dataset/model run dirs keep their `idXXX` names.
- **Best-model finding** (slot_norm + unit_diff for ESM-2 on HA/NA): see `docs/architecture.md`
  § Key Experimental Findings. The `flu_schema_raw_slot_norm_unit_diff` bundle was retired
  2026-05-12 — the finding stands, the bundle file no longer exists.
- **2D-CD builder** (`src/datasets/dataset_pairs_cc.py`): Stage-3 builder for bilateral
  cluster-disjoint holdout/K-fold. All three alphabets (aa / nt_cds / nt_ctg) build end-to-end;
  `negative_scope` within_cc|within_fold, with `drop_negative_infeasible_ccs` unified across both.
- **Single-segment OOD clusters** (`src/preprocess/build_ood_clusters.py`): the "across clusters,
  sequences are `< t` identical" guarantee requires clusters that are **connected components of the
  `>= t`/coverage all-vs-all graph**. mmseqs `easy-cluster --cluster-mode 1` does NOT deliver it (M1
  aa t099: 566 clusters containing 3,797 cross-cluster `>= 0.99` pairs); the correct build is
  `easy-search` all-vs-all -> threshold -> **union-find** (M1: 234 clusters, 0 violations).
  `--exhaustive-search` is profile-iterative, NOT all-vs-all; `--prefilter-mode 2` is the
  provable-complete search. Writes `clusters_{alphabet}_ood/tXXX/`, never overwriting the set-cover
  parquets. Figures `src/analysis/plot_clusters.py`; verifier `src/analysis/verify_ood_clusters.py`.
- **Subtype context for the HA-NA CCs**: the HA_0×NA_0 hub is **H3N2**, HA_1×NA_1 is **H1N1**, and
  the multi-cluster tangle is an avian mix. Never call a 95%-nt cluster a "lineage".
- **CV output shape**: nested `fold_k/` dirs + `cv_summary.*`; launchers `scripts/run_cv_lambda.py`
  and `scripts/run_cv_polaris.pbs`.
- **Temporal holdout**: implemented. Known issue — pair_key dedup removes ~42% of val/test positives
  (the same strains recur across years), creating label imbalance; disable dedup for temporal mode
  before publication. K-mer beats ESM-2 here (AUC 0.941 vs 0.891).
- **Plot helpers**: don't split by slot when the data is per-pair.

## Work In Flight

**Status lives in each plan's own `**Status:**` line — this table only says what the work is, so
there is one place to update.** Read the plan before assuming a state.

| work | plan |
|---|---|
| Data-split refactor (**top priority**): one atom-provider + one packer + one CV path, bit-exact on a code-path-coverage bundle set | `docs/plans/2026-06-03_dataset_split_refactor_plan.md` |
| K-fold: remaining validation of the v2 CV path | `docs/plans/2026-05-28_kfold_remaining.md` |
| Fold-maker consolidation + the two production split paths | `docs/plans/2026-08-03_fold_maker_consolidation_plan.md` |
| Single-segment OOD clusters: 8-major scale-out + nt rollout | `docs/plans/2026-07-08_single_segment_ood_clusters_plan.md` |
| OOD-vs-random CV at matched size (leave-one-CC-out vs random) | `docs/plans/2026-07-21_ood_vs_random_split_plan.md` |
| 1D cluster-disjoint single-slot (HA held out vs each partner) | `docs/plans/2026-07-27_1d_cluster_disjoint_single_slot_plan.md` |
| Task 11 / 28-pair sweep; throughput fix lives on the unmerged branch `fix/mpiexec-cpu-binding` | `polaris_plan.md`, `docs/project_changelog.md` |

- **Stage-4 training is GATED** — no launch without explicit OK.

## Forward-looking work
- Todos: `BACKLOG.md` (numbered, triaged — the single source of truth). Big-picture experiments:
  `roadmap_v2.md`. Keep new items there, not here, so this file doesn't re-accumulate stale lists.
- In-development modules + k-mer scaling limits: `docs/architecture.md`.

## Env Management
**Rule**: bioconda / kalininalab CLI tools and experimental Python packages live in **dedicated
conda envs**, never in the `segmatch` pipeline env. **Why**: bioconda pulls a different `libhdf5`
than conda-forge and breaks the precompiled `h5py` wheel (broke the pipeline env twice — mmseqs2
2026-05-15, datasail 2026-05-19). **Never** `conda remove --force` to undo bioconda damage —
rebuild from `environment.yml`.
- `segmatch`: clean pipeline env (conda-forge only, `environment.yml`). Validated 2026-05-20.
- `mmseqs2`: CLI-only, v18.8cc5c, `/homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs`.
- `datasail`: dedicated env for the DataSAIL bake-off.
- On lambda13 `$HOME` has no miniconda — use NFS absolute binaries
  (`/nfs/lambda_stor_01/homes/apartin/miniconda3/envs/<env>/bin/python`); bare `conda activate` fails.

## HPC (ALCF Polaris)
- PBS job arrays, not SLURM. Do NOT use Hydra's submitit launcher (SLURM-only).
- Batch mode doesn't source dotfiles: use `#!/bin/bash -l` (login shell) to load PrgEnv/CUDA.
- Stage 2 (embeddings) is GPU-heavy; Stage 4 (training) is modest. The 8-GPU dev cluster has no
  scheduler — use a `subprocess.Popen`-per-fold launcher with `CUDA_VISIBLE_DEVICES`.
- Refs: `polaris_plan.md` (Task 11 phases), `speed_up.md`, `docs/hardware_notes.md`.

## User Preferences
- Concise responses, no emojis unless asked
- No unnecessary refactoring beyond what's asked
- Always ask before destructive operations (rm, git reset --hard, git push --force, etc.)
- CLAUDE.md is the authoritative project context; .claude/memory.md is the compact working memory
- Both files are in the repo — update them when decisions change
- **One script per purpose**: follow the existing pattern in `src/analysis/` — propose a dedicated
  script with a clear name rather than hedging between existing scripts. Commit to the obvious answer.
- **Code priority order**: correctness > readability > efficiency. Optimize for the next reader, not
  the next clock cycle. Reach for performance changes only when measured (or when efficiency is
  correctness-critical).
- **Communication style**: prefer common words; use jargon only when it carries meaning the plain
  term doesn't. Don't cut technical content; cut hedges and filler. Concrete numbers, file:line refs,
  and observed data beat hedged adjectives. When explaining, assume the reader does not carry the
  codebase in their head — lead with the plain-language answer, then the evidence.
- **Terminology**: use canonical terms from `docs/methods/glossary.md`; add new terms there first.
  (Enforced as CLAUDE.md Conventions § Terminology.)
- **Accuracy over confidence**: state only what is verified against a source actually checked in this
  session (paper passage, code at file:line, observed command output). When uncertain, say so with
  what would resolve it. Don't pattern-match across sources without verification — superficially
  synonymous terms (DataSAIL I2, Park & Marcotte C3, segmatch seq_disjoint) may differ in
  dimensionality or which axes they cover. (Full rule: CLAUDE.md Conventions § Verify before asserting.)
- **Commits are explicit-only** (full rule: CLAUDE.md Conventions § Commits are explicit-only): never
  run `git commit`/`--amend` on Claude's own initiative. Otherwise stage + prep the diff and wait.
- **Refer to Claude as "Claude"** in committed docs and writeups, not "I" or "my proposal".
