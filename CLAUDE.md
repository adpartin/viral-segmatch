# CLAUDE.md — Project Context for Claude Code

This file provides persistent context for Claude Code sessions. Update it as the project evolves.
Also read `.claude/memory.md` for compact, up-to-date project memory (pipeline state, recent decisions).

---

## Session Startup Checklist

At the start of each session:
1. Read `.claude/memory.md` (in the repo) for current project state.
2. Check if the machine-local auto-memory `MEMORY.md` (path provided in the system prompt) exists.
   If it does not exist, create it with exactly this content:
   ```
   Project memory has moved into the repo for portability across machines.
   Read: .claude/memory.md (in the repo root)
   This machine-local file is no longer updated.
   ```
3. Check `docs/plans/` for any in-progress plans (status != IMPLEMENTED).
   If found, read the plan and offer to resume implementation.

### Plans directory (`docs/plans/`)

Plans are saved in the repo (not machine-local `~/.claude/`) for cross-machine portability.
When creating a plan during plan mode, save it to `docs/plans/<descriptive_name>_plan.md` before
starting implementation. Mark the plan's status right after the title heading:
- `**Status: IN PROGRESS**` — plan approved, implementation underway
- `**Status: IMPLEMENTED**` — implementation complete

When a plan is fully implemented, mark `**Status: IMPLEMENTED**` and move it to `docs/plans/done/`.
Active plans stay in `docs/plans/`; completed plans live in `docs/plans/done/`.

---

## Approval Required

Always ask for explicit confirmation before running any of the following — even if you think it is safe:

- `rm *` or any file/directory deletion
- `git rm *`
- `git reset --hard *` or `git reset --mixed *`
- `git push --force *` or `git push -f *`
- `git branch -D *`
- `git rebase *`
- `git clean *`
- Any command that modifies shared infrastructure, sends messages, or affects state outside this repo

---

## What This Project Does

**viral-segmatch**: Predicts whether two viral protein segments co-occur in the same isolate (binary classification). Given embeddings of two protein segments from the same virus, the model learns to distinguish true co-occurring pairs (positive) from artificially mixed pairs (negative).

**Approach**: Frozen ESM-2 protein embeddings (1280-dim, `esm2_t33_650M_UR50D`) → pairwise interaction feature (e.g., `unit_diff`, `concat`) → MLP binary classifier.

**Primary virus**: Influenza A (Flu A). Bunyavirales support exists but is not actively maintained.

---

## Pipeline Stages

Stages 1–2 run once per dataset (shared across experiments). Stages 3–4 are experiment-specific.

| Stage | Script | Output | Runs |
|-------|--------|--------|------|
| 1. Preprocess | `src/preprocess/preprocess_flu.py` | `data/processed/flu/{version}/protein_final.csv` + `genome_final.csv` | Once |
| 2. Embeddings | `src/embeddings/compute_esm2_embeddings.py` | `data/embeddings/flu/{version}/master_esm2_embeddings.h5` | Once |
| 3. Dataset | `src/datasets/dataset_segment_pairs.py` | `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/` | Per experiment |
| 4. Train | `src/models/train_pair_classifier.py` | `models/flu/{version}/runs/training_{bundle}_{ts}/` | Per experiment |

Shell wrappers: `scripts/stage1_preprocess_flu.sh`, `scripts/stage2_esm2.sh`, `scripts/stage3_dataset.sh`, `scripts/stage4_train.sh`.

**Stage 3/4 decoupling**: Stage 4's shell script requires `--dataset_dir` explicitly and
does not extract or validate a bundle name from the dataset path. This allows running
Stage 3 once and Stage 4 multiple times with different training bundles against the same
dataset. Provenance is tracked via `training_info.json` saved in the training output dir.

---

## Configuration System

**Hydra** with a bundle-per-experiment pattern.

- Bundles: `conf/bundles/{bundle_name}.yaml` — one file per named experiment
- Virus configs: `conf/virus/flu.yaml`, `conf/virus/bunya.yaml`
- Data paths: `conf/paths/flu.yaml`, `conf/paths/bunya.yaml`
- Defaults: `conf/dataset/default.yaml`, `conf/embeddings/default.yaml`, `conf/training/base.yaml`
- Config loader: `src/utils/config_hydra.py`

**Convention**: One bundle = one reproducible experiment. Bundle names encode the experiment:
`flu_{proteins}_{n_isolates}[_{modifiers}]`, e.g., `flu_ha_na_5ks`, `flu_schema_raw_slot_norm_unit_diff_h3n2`.

Key bundle parameters: `virus.selected_functions`, `dataset.max_isolates_to_process`, `dataset.hn_subtype`, `dataset.year`, `dataset.host`, `training.slot_transform`, `training.interaction`.

**Bundle organization** (see `conf/bundles/README.md` for full detail):
- Each bundle has a `# STATUS: active|ablation|experimental|legacy|not maintained` header comment.
- Bundles form inheritance chains via Hydra `defaults`. Base bundles (e.g., `flu_schema.yaml`,
  `flu_schema_raw_slot_norm_unit_diff.yaml`) must stay flat; only leaf bundles can be moved to subdirs.
- `conf/bundles/paper/` — reserved for publication experiments (CV, temporal holdout, large dataset).
- Three generations: Gen1 (`flu.yaml` base, no schema ordering), Gen2 (`flu_schema.yaml` base,
  none+concat), Gen3 (`flu_schema_raw_*`, current best: `slot_norm + unit_diff`).


---

## Active Source Files

```
src/
  preprocess/
    preprocess_flu.py               # Stage 1 (ACTIVE): GTO → protein_final.csv + genome_final.csv
    preprocess_flu_protein.py       # Older protein-only variant; superseded by preprocess_flu.py (kept for reference)
    flu_genomes_eda.py              # Generates flu_genomes_metadata_parsed.csv (run once)
    preprocess_bunya_protein.py     # Bunya preprocessing (NOT actively maintained)
    preprocess_bunya_genome.py      # Bunya genome preprocessing (NOT actively maintained; renamed from preprocess_bunya_dna.py)
  embeddings/
    compute_esm2_embeddings.py      # Stage 2: sequences → ESM-2 HDF5 cache
    compute_kmer_features.py        # Stage 2b: DNA → k-mer sparse matrix
  datasets/
    dataset_segment_pairs.py        # Stage 3: pairs + train/val/test splits
  models/
    train_pair_classifier.py  # Stage 4: MLP classifier training
  analysis/
    analyze_stage4_train.py         # Confusion matrix, ROC, FP/FN analysis
    visualize_dataset_stats.py      # PCA plots, interaction diagnostics
    aggregate_experiment_results.py # Cross-bundle summary tables
    visualize_cv_results.py         # CV visualization (error bars, ROC curves)
    analyze_stage1_preprocess.py    # Preprocessing QC
    analyze_stage2_embeddings.py    # Embedding quality checks
    analyze_stage3_datasets.py      # Dataset balance/distribution
    create_presentation_plots.py    # PPT-ready figures
  utils/
    config_hydra.py                 # Hydra config loader (primary)
    esm2_utils.py                   # ESM-2 tokenization, batch embedding
    embedding_utils.py              # Load/index embeddings from HDF5
    metadata_enrichment.py          # Load flu_genomes_metadata_parsed.csv
    experiment_registry.py          # experiments/registry.yaml tracking
    seed_utils.py                   # Hierarchical seed system
    learning_verification_utils.py  # Karpathy-style sanity checks
    plot_config.py                  # Colors, protein name mapping
    gto_utils.py, protein_utils.py, path_utils.py, timer_utils.py
    kmer_utils.py                   # Load k-mer features, pair construction
    dna_utils.py                    # DNA QC utilities (in development)
    dim_reduction_utils.py          # PCA/UMAP wrappers
```

---

## Key Experimental Findings (as of Mar 2026)

- **`unit_diff` > `concat` on homogeneous data (ESM-2 only)**: ESM-2 `concat` fails on H3N2-only HA/NA (AUC=0.50); `unit_diff` succeeds (AUC=0.96). Direction of embedding difference carries genuine biological signal; magnitude is a shortcut.
- **ESM-2 `concat` collapse is not H3N2-specific**: Task 11 all-pairs sweeps (Apr 2026) show PB1/PA collapses on BOTH the full unfiltered dataset (AUC=0.4971) and the H3N2-filtered dataset (AUC=0.4981) with ESM-2 concat. K-mer concat never collapses on any pair in any sweep. The subspace-offset failure is a property of ESM-2's protein-type geometry, not of a specific subtype.
- **K-mer concat does NOT collapse on H3N2**: K-mer concat achieves AUC=0.985 on H3N2-only, proving the concat failure is specific to ESM-2's embedding geometry (subspace offset between protein types), not concatenation as an interaction.
- **K-mer dominates ESM-2 on homogeneous data**: On H3N2-only, k-mer unit_diff AUC=0.988 vs ESM-2 unit_diff AUC=0.957. K-mer features are interaction-agnostic (unit_diff≈concat) because sparse frequency vectors don't have ESM-2's subspace offset problem.
- **LayerNorm (`slot_norm`) is critical for homogeneous subsets**: Without it, raw HA/NA embeddings live in slightly different subspaces; `unit_diff` then picks up slot offset rather than biological signal.
- **Delayed learning on H3N2 + unit_diff**: Characteristic plateau-then-breakthrough (~epochs 10–32, seed-dependent). Increase `patience` to 40+ for H3N2-only runs.
- **High FP rate on filtered datasets**: Likely due to subtype/host confounders in negative pairs. Hypothesis: model learns "same population" rather than "same isolate." Hard negatives or strict metadata filtering can test this.
- **K-mer (k=6, 4096-dim) matches or exceeds ESM-2 on mixed-subtype HA/NA**: AUC 0.982 vs 0.966–0.975. Both interactions work with k-mers.

---

## Roadmap (from 02/10/2026 meeting)

Priority experiments for publication:
1. **Cross-validation** (N splits, mean ± std metrics) — needs `fold_id`/`n_folds` in dataset config + job array
2. **Large dataset** (full Flu A, ~100K isolates) — HPC required (Polaris)
3. **Temporal holdout** (train 2021–2023, test 2024) — `year_train`/`year_test` config fields
4. **Genome features** (k-mers + XGBoost/LightGBM, then GenSLM) — unified `preprocess_flu.py` now in production (emits `genome_final.csv` alongside `protein_final.csv`)
5. **PB2/PB1 + H3N2 bundle** — trivial; one new bundle
6. **Accuracy vs genetic distance** — needs clade metadata from BV-BRC

---

## HPC (ALCF Polaris)

- PBS job arrays, not SLURM. Hydra's `submitit` launcher is SLURM-only — do not use it.
- For CV/parallel runs: PBS job array where `PBS_ARRAY_INDEX` maps to fold ID; pass as Hydra override `dataset.fold_id=${FOLD}`.
- For 8-GPU dev cluster (no scheduler): Python launcher using `subprocess.Popen` per fold, each with `CUDA_VISIBLE_DEVICES=K`.
- Stage 2 (embeddings) is the most GPU-intensive. Stage 4 (training) is modest.

---

## What Is NOT Maintained

- `old_scripts/` — superseded by current stage scripts; see `old_scripts/README.md`
- `src/preprocess/preprocess_bunya_protein.py` — Bunya preprocessing; see maintenance note in file
- `conf/bundles/bunya.yaml` — Bunya experiment config; see maintenance note in file

## What Is In Development (Not Yet Production)

- `src/utils/dna_utils.py` — DNA sequence QC utilities
- Temporal holdout split logic (`year_train`/`year_test`)

Note: unified Flu preprocessing (`preprocess_flu.py`, protein + genome in one pass) was previously listed here; it is now in production and is the entry point invoked by `scripts/stage1_preprocess_flu.sh`.

---

## Directory Layout

```
viral-segmatch/
├── CLAUDE.md                   # This file (auto-loaded by Claude Code)
├── .claude/
│   ├── settings.json           # Claude Code permissions (deny/allow rules)
│   └── memory.md               # Compact project memory — read this every session
├── README.md                   # Project overview
├── roadmap_v1.md               # Experiment plan v1 (02/10/2026 meeting)
├── roadmap_v2.md               # Experiment plan v2 (updated 03/12/2026 meeting)
├── paper_outline_v1.md         # Paper outline v1 (initial draft)
├── paper_outline_v2.md         # Paper outline v2 (updated 03/12/2026 meeting)
├── _ongoing_work.md            # Technical notes on interactions, findings
├── _notes.txt                  # Ad-hoc questions and TODOs
├── src/                        # Python source code
├── scripts/                    # Shell pipeline wrappers (stage2–4)
├── conf/                       # Hydra configs
│   └── bundles/                # One YAML per named experiment
├── eda/                        # Exploratory analysis scripts (not pipeline)
├── examples/                   # HuggingFace reference scripts (not pipeline)
├── old_scripts/                # Superseded scripts (not maintained)
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Technical docs
├── documentation/              # User guides
├── experiments/
│   └── registry.yaml           # Experiment tracking (all pipeline runs)
└── data/                       # Not in git; symlinked raw data
```

---

## Per-machine Git Setup

Run once after cloning on each new machine (writes to `.git/config`, not tracked by git):
```bash
git config pull.rebase true   # avoid "need to reconcile divergent branches" on git pull
```

---

## Conventions

- **Experiment naming**: `{virus}_{proteins}_{n_isolates}[_{modifiers}]`
- **Timestamps**: All run directories include `YYYYMMDD_HHMMSS`
- **Shared vs. run-specific**: Preprocessing and embeddings are shared per `{virus}/{data_version}`. Datasets and models are per run in `runs/` subdirectories.
- **Seed system**: Hierarchical — `master_seed` derives all process seeds. See `docs/SEED_SYSTEM.md`.
- **Metrics**: F1, AUC-ROC, Brier score. Val imbalance is intentional (realistic); train is balanced.
- **Proteins**: `preprocess_flu.py` maps GTO replicon functions to standard protein names (PB2, PB1, PA, HA, NP, NA, M1, M2, NEP).
- **Log messages**: No emojis in print/log output. Use text prefixes instead: `ERROR:` (fatal, script will raise/exit), `WARNING:` (non-fatal but noteworthy), `Done.` (success). Decorative emojis (`📊`, `🔍`, etc.) should be removed — the surrounding text is sufficient.
