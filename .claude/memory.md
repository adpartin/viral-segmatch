# viral-segmatch — Project Memory (compact working state)

Version-controlled (`.claude/memory.md`) so it travels across machines. Read at session start.
Update when production settings change, work moves in/out of flight, or a durable decision is made.

**Scope**: current production state, work in flight, env rules, user preferences — things that
change and aren't derivable from code. This file does NOT duplicate:
- **`CLAUDE.md`** — authoritative project context (pipeline, config system, conventions, Key
  Experimental Findings, roadmap). Read it first.
- **`docs/results/`** — canonical writeup for every headline finding (index in CLAUDE.md
  §Recent run outputs).
- **`docs/plans/`** (+ `done/`) — design/implementation plans.
- **`docs/project_changelog.md`** — relocated implementation log, Removed-keys log,
  Polaris/Task-11 history, CV/decoupling/temporal-holdout implementation detail, and
  pre-writeup exploration notes (HA-NA co-occurrence matrix, coverage `-c` sweep,
  bipartite-graph properties, level-0 multiplicity, nt_ctg pivot, DataSAIL C1 scaling).
  Reference material, not session-startup reading.

---

## Current Production State
- **Builder**: v2 is the default (`pair_builder_version: v2`); the `dataset_segment_pairs.py` CLI
  dispatches to `dataset_segment_pairs_v2.py`. Stage 3/4 are decoupled — Stage 4 takes
  `--dataset_dir` explicitly; provenance in `training_info.json`.
- **Active HA/NA + PB2/PB1 bundles** (`flu_ha_na.yaml`, `flu_pb2_pb1.yaml`) bake in
  `split_strategy.mode=seq_disjoint`, `hash_key=seq` (protein-level, stricter), and the "Test 3"
  interaction (`slot_transform=unit_norm`, `interaction=unit_diff+prod`). Verified in
  `flu_ha_na.yaml` 2026-06-03.
- **Regime-aware leaves**: `flu_ha_na_regimes`, `flu_pb2_pb1_regimes` — inherit the parent and add
  `dataset.negative_sampling.regime_targets` (uniform 0.10×7 + 0.30 on `host_subtype_year`).
- **Clustering**: symmetric mmseqs2 `easy-linclust` on BOTH alphabets (since 2026-05-22; replaced
  the asymmetric easy-cluster+linclust setup). Artifacts at `clusters_aa/tXXX/<func>_cluster.parquet`
  (col `seq_hash`) and `clusters_nt_cds/tXXX/<func>_cluster.parquet` (col `cds_dna_hash`); pre-Phase-2
  easy-cluster + idXXX artifacts archived under `clusters_*_archive_*`. Binary via the dedicated
  `mmseqs2` env, resolved through `MMSEQS_BIN` env var / `--mmseqs_bin` / PATH.
- **pair_key**: `split_strategy.pair_key_alphabet` — `aa` default (protein pair_key); `nt_cds` opt-in
  for nt_cds cluster_disjoint bundles, where silent codon variants become distinct positives (inflates
  the pair universe, opens DNA-variant leakage). Cite the alphabet in any post-2026-06-03 experiment.
- **Routing modes**: `random`; `seq_disjoint` (hash_key seq|dna); `cluster_disjoint` (bilateral /
  `single_slot: a|b` / planned `cluster_disjoint_test_only`); `metadata_holdout`. `single_slot`
  exercised on HA-only and PB2-only; NA-only / PB1-only and nt single_slot untested.
- **Threshold notation `tXXX`** (CLAUDE.md convention): on-disk cluster parquets are at `clusters_*/tXXX/`
  (confirmed on disk 2026-06-03); pre-Phase-2 dataset/model run dirs keep their `idXXX` names.
- **Best-model finding** (slot_norm + unit_diff for ESM-2 on HA/NA): see CLAUDE.md §Key Experimental
  Findings. The `flu_schema_raw_slot_norm_unit_diff` bundle of that name was retired 2026-05-12 — the
  finding stands, the bundle file no longer exists.

## Work In Flight
- **Phase 2 pair_key migration** (branch `feature/phase2-pair-key-migration`): plan DONE
  (`docs/plans/done/2026-06-02_phase2_pair_key_migration_plan.md`); commits 1–6 landed by 2026-06-03
  (commit 6 = post-migration metrics; the tXXX-rename spans commits 3–7 per CLAUDE.md). aa pipeline
  regression-guarded at ε=0 (pure refactor, bytes identical); nt_cds runs degrade per the expected
  pair-universe inflation (+34.9% HA-NA, +57.0% PB2-PB1) plus routing shift — not a bug. The 2026-05-13
  regimes run reproduces bit-exactly under 2026-06-03 code. Metrics:
  `docs/results/2026-06-03_phase2_postmigration_metrics.md`.
- **Cross-validation**: IMPLEMENTED, needs an end-to-end run. Canonical reference
  `docs/methods/splits.md` §2; remaining work `docs/plans/2026-05-28_kfold_remaining.md`. Output is
  nested `fold_k/` dirs under one Stage-3 run + `cv_summary.*`; launchers `scripts/run_cv_lambda.py`
  (subprocess.Popen per fold) and `scripts/run_cv_polaris.pbs` (PBS array). Full impl detail in
  `docs/project_changelog.md`.
- **Temporal holdout**: IMPLEMENTED. Known issue: pair_key dedup removes ~42% of val/test positives
  (same strains across years), creating label imbalance — fix (disable dedup for temporal mode) needed
  before publication. K-mer beats ESM-2 (AUC 0.941 vs 0.891; gap wider than on random splits). Detail
  in `docs/project_changelog.md` §Temporal holdout.
- **Task 11 (28 protein pairs, 8×8 heatmap)**: ON HOLD since 2026-04-06. Built (master bundle +
  28 children + Polaris launchers); Phase 3 failed on Polaris (training data-loading bound, post-mortem
  + phase log in `docs/project_changelog.md`). Resume by swapping the master bundle's
  `dataset.negative_sampling` block to the regime-aware one.

## Forward-looking work
- Todos: `BACKLOG.md` (numbered, triaged — the single source of truth). Big-picture experiments:
  `roadmap_v2.md`. Keep new items there, not here, so this file doesn't re-accumulate stale lists.
- In-development modules + k-mer scaling limits: CLAUDE.md §What Is In Development.

## Env Management
**Rule**: bioconda / kalininalab CLI tools and experimental Python packages live in **dedicated conda
envs**, never in the `segmatch` pipeline env. **Why**: bioconda pulls a different `libhdf5` than
conda-forge and breaks the precompiled `h5py` wheel (broke the pipeline env twice this way — mmseqs2 2026-05-15, datasail
2026-05-19; `conda remove --force` left transitive deps orphaned). **Never** `conda remove --force` to
undo bioconda damage — rebuild from `environment.yml`.
- `segmatch`: clean pipeline env (conda-forge only, `environment.yml`). Validated end-to-end 2026-05-20.
- `mmseqs2`: CLI-only, v18.8cc5c, `/homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs`.
- `datasail`: dedicated env for the DataSAIL bake-off.

## HPC (ALCF Polaris)
- PBS job arrays, not SLURM. Do NOT use Hydra's submitit launcher (SLURM-only).
- Batch mode doesn't source dotfiles: use `#!/bin/bash -l` (login shell) to load PrgEnv/CUDA.
- Stage 2 (embeddings) is GPU-heavy; Stage 4 (training) is modest. The 8-GPU dev cluster has no
  scheduler — use a `subprocess.Popen`-per-fold launcher with `CUDA_VISIBLE_DEVICES`.
- Refs: `polaris_plan.md` (Task 11 phases), `speed_up.md` (training opts), `docs/hardware_notes.md`.

## User Preferences
- Concise responses, no emojis unless asked
- No unnecessary refactoring beyond what's asked
- Always ask before destructive operations (rm, git reset --hard, git push --force, etc.)
- CLAUDE.md is the authoritative project context; .claude/memory.md is the compact working memory
- Both files are in the repo -- update them when decisions change
- **One script per purpose**: follow the existing pattern in `src/analysis/` — propose a dedicated script with a clear name (e.g., `aggregate_cv_results.py`) rather than hedging between existing scripts. Commit to the obvious answer.
- **Code priority order**: correctness > readability > efficiency. Optimize for the next reader, not the next clock cycle. Reach for performance changes only when measured (or when efficiency is correctness-critical, e.g., when the naive version is intractable at expected input sizes).
- **Communication style**: prefer common words; use jargon only when it carries meaning the plain term doesn't. Don't cut technical content; cut hedges and filler. Concrete numbers, file:line refs, and observed data beat hedged adjectives.
- **Terminology**: use canonical terms from `docs/methods/glossary.md`; add new terms there first. Don't coin a synonym for a term that already has an entry. (Enforced as CLAUDE.md Conventions §Terminology.)
- **Accuracy over confidence**: state only what is verified against a source actually checked in this session (paper passage, code at file:line, observed command output). When uncertain, say so with what would resolve it ("haven't read §X", "need to grep Y"). Don't pattern-match across sources without verification — superficially synonymous terms (e.g. DataSAIL I2, Park & Marcotte C3, segmatch seq_disjoint) may differ in dimensionality, what gets discarded, or which axes they cover. Hedges are fine when the uncertainty is named; vague hedging is not. **Verify before asserting**: for claims about what exists / current values / what code does, check the source in the same turn before stating it, and flag anything unverified (full rule: CLAUDE.md Conventions §Verify before asserting).
- **No commits without explicit instruction**: never run `git commit` until the user says "commit" (or equivalent) for the specific change. Stage and prepare diffs freely, but wait for the green light.
- **Refer to Claude as "Claude"** in committed docs and writeups, not "I" or "my proposal." First-person is fine in conversation; documents persist and read better in third-person.
