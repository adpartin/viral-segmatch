# CLAUDE.md — Project Context for Claude Code

This file provides persistent context for Claude Code sessions. Update it as the project evolves.

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
| 1. Preprocess | `src/preprocess/preprocess_flu_protein.py` | `data/processed/flu/{version}/protein_final.csv` | Once |
| 2. Embeddings | `src/embeddings/compute_esm2_embeddings.py` | `data/embeddings/flu/{version}/master_esm2_embeddings.h5` | Once |
| 3. Dataset | `src/datasets/dataset_segment_pairs.py` | `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/` | Per experiment |
| 4. Train | `src/models/train_esm2_frozen_pair_classifier.py` | `models/flu/{version}/runs/training_{bundle}_{ts}/` | Per experiment |

Shell wrappers: `scripts/stage2_esm2.sh`, `scripts/stage3_dataset.sh`, `scripts/stage4_train.sh`.

There is no stage1 shell script; preprocessing is run directly.

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

Key bundle parameters: `virus.selected_functions`, `dataset.max_isolates_to_process`, `dataset.hn_subtype`, `dataset.year`, `dataset.host`, `training.pre_mlp_mode`, `training.interaction`.


---

## Active Source Files

```
src/
  preprocess/
    preprocess_flu_protein.py       # Stage 1: GTO → protein_final.csv
    flu_genomes_eda.py              # Generates flu_genomes_metadata_parsed.csv (run once)
    preprocess_bunya_protein.py     # Bunya preprocessing (NOT actively maintained)
    preprocess_bunya_dna.py         # DNA preprocessing template (in development; will become preprocess_flu_dna.py)
  embeddings/
    compute_esm2_embeddings.py      # Stage 2: sequences → ESM-2 HDF5 cache
  datasets/
    dataset_segment_pairs.py        # Stage 3: pairs + train/val/test splits
  models/
    train_esm2_frozen_pair_classifier.py  # Stage 4: MLP classifier training
  analysis/
    analyze_stage4_train.py         # Confusion matrix, ROC, FP/FN analysis
    visualize_dataset_stats.py      # PCA plots, interaction diagnostics
    aggregate_experiment_results.py # Cross-bundle summary tables
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
    dna_utils.py                    # DNA QC utilities (in development)
    dim_reduction_utils.py          # PCA/UMAP wrappers
```

---

## Key Experimental Findings (as of Feb 2026)

- **`unit_diff` > `concat` on homogeneous data**: `concat` fails on H3N2-only (AUC=0.50); `unit_diff` succeeds (AUC=0.96). Direction of embedding difference carries genuine biological signal; magnitude is a shortcut.
- **LayerNorm (`slot_norm`) is critical for homogeneous subsets**: Without it, raw HA/NA embeddings live in slightly different subspaces; `unit_diff` then picks up slot offset rather than biological signal.
- **Delayed learning on H3N2 + unit_diff**: Characteristic plateau-then-breakthrough (~epochs 10–32, seed-dependent). Increase `patience` to 40+ for H3N2-only runs.
- **High FP rate on filtered datasets**: Likely due to subtype/host confounders in negative pairs. Hypothesis: model learns "same population" rather than "same isolate." Hard negatives or strict metadata filtering can test this.

---

## Roadmap (from 02/10/2026 meeting)

Priority experiments for publication:
1. **Cross-validation** (N splits, mean ± std metrics) — needs `fold_id`/`n_folds` in dataset config + job array
2. **Large dataset** (full Flu A, ~100K isolates) — HPC required (Polaris)
3. **Temporal holdout** (train 2021–2023, test 2024) — `year_train`/`year_test` config fields
4. **Genome features** (k-mers + LightGBM, then GenSLM) — start from `preprocess_bunya_dna.py`
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

- `src/preprocess/preprocess_bunya_dna.py` — template for `preprocess_flu_dna.py` (DNA k-mer pipeline)
- `src/utils/dna_utils.py` — DNA sequence QC utilities
- Cross-validation support (`fold_id`/`n_folds` in dataset config)
- Temporal holdout split logic (`year_train`/`year_test`)

---

## Directory Layout

```
viral-segmatch/
├── CLAUDE.md                   # This file
├── README.md                   # Project overview
├── _roadmap.md                 # Experiment plan (02/10/2026 meeting)
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

## Conventions

- **Experiment naming**: `{virus}_{proteins}_{n_isolates}[_{modifiers}]`
- **Timestamps**: All run directories include `YYYYMMDD_HHMMSS`
- **Shared vs. run-specific**: Preprocessing and embeddings are shared per `{virus}/{data_version}`. Datasets and models are per run in `runs/` subdirectories.
- **Seed system**: Hierarchical — `master_seed` derives all process seeds. See `docs/SEED_SYSTEM.md`.
- **Metrics**: F1, AUC-ROC, Brier score. Val imbalance is intentional (realistic); train is balanced.
- **Proteins**: `preprocess_flu_protein.py` maps GTO replicon functions to standard protein names (PB2, PB1, PA, HA, NP, NA, M1, M2, NEP).
