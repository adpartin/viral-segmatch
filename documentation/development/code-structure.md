# Code Structure

High-level orientation for new contributors. The **authoritative
module-by-module reference** lives in
[`../../CLAUDE.md`](../../CLAUDE.md) ("Active Source Files" section);
this doc is the conceptual / "how does it all fit together" overview.

## Top-level layout

```
viral-segmatch/
├── src/                # Python source code (per-stage pipeline + utilities)
├── scripts/            # Lean shell wrappers for each pipeline stage
├── conf/               # Hydra configuration (bundles, virus, dataset, training, baselines)
├── data/               # Pipeline outputs (not in git; symlink raw GTOs in)
├── models/             # Trained models (created by Stage 4)
├── results/            # Cross-model heatmaps and aggregator outputs
├── docs/               # Methods / technical / plans documentation
└── documentation/      # User guides (this directory)
```

## The 4 pipeline stages

| Stage | Entry script | Output |
|---|---|---|
| 1. Preprocess | `src/preprocess/preprocess_flu.py` | `data/processed/flu/{ver}/protein_final.csv` + `genome_final.csv` |
| 2. Embeddings | `src/embeddings/compute_esm2_embeddings.py` | `data/embeddings/flu/{ver}/master_esm2_embeddings.h5` |
| 2b. K-mer features | `src/embeddings/compute_kmer_features.py` | `data/embeddings/flu/{ver}/kmer_features_k6.npz` + sidecar parquet index |
| 3. Dataset | `src/datasets/dataset_segment_pairs.py` (CLI) → dispatches to `dataset_segment_pairs_v2.py` (default since 2026-05-11) | `data/datasets/flu/{ver}/runs/dataset_<bundle>_<TS>/` |
| 4. Train MLP | `src/models/train_pair_classifier.py` | `models/flu/{ver}/runs/training_<bundle>_<TS>/` |
| 4. Train baselines | `src/models/train_pair_baselines.py` | `models/flu/{ver}/runs/baseline_<name>_<bundle>_<TS>/` |

Shell wrappers in `scripts/` that wrap each stage and call from
`./scripts/stage*_*.sh`:

- `stage1_preprocess_flu.sh`
- `stage2_esm2.sh` / `stage2b_kmer.sh`
- `stage3_dataset.sh`
- `stage4_train.sh` / `stage4_baselines.sh`

## Source-tree map (high level)

### `src/preprocess/`

GTO JSON → tabular protein + genome outputs. `preprocess_flu.py` is the
active script (unifies the older protein-only and genome-only
preprocessors). See
[`../../docs/methods/preprocess.md`](../../docs/methods/preprocess.md)
for the detailed parse map and filter pipeline.

### `src/embeddings/`

Stage 2 featurization. ESM-2 path (`compute_esm2_embeddings.py`) is
GPU-heavy. K-mer path (`compute_kmer_features.py`) is CPU-only, ~5–10
min on full Flu-A. As of 2026-05-12 the same machinery supports
protein k-mers via an `alphabet` parameter, though no active bundle
uses that yet. See
[`../../docs/methods/kmer_features.md`](../../docs/methods/kmer_features.md).

### `src/datasets/`

Stage 3 pair construction + train/val/test split.
`dataset_segment_pairs.py` is the CLI entry point;
`dataset_segment_pairs_v2.py` is the default builder. Shared helpers
in `_pair_helpers.py`. Negative-regime sampler in
`_negative_regime_sampling.py` (8 regimes: `none_match`, `host_only`,
`subtype_only`, `year_only`, `host_subtype_only`, `host_year_only`,
`subtype_year_only`, `host_subtype_year`).

### `src/models/`

Stage 4 training. `train_pair_classifier.py` is the MLP path;
`train_pair_baselines.py` runs sklearn baselines from
`baselines/{logistic,lgbm,knn1_margin,knn_vote}.py`. Shared feature
loading in `_pair_features.py`; shared metric helpers in
`_pair_metrics.py`. Defaults centralized in
`conf/baselines/default.yaml`.

### `src/analysis/`

Post-hoc analysis and visualization. Key entry points:
- `analyze_stage4_train.py` — per-run analysis (confusion matrix, ROC,
  per-regime heatmap, FP/FN breakdown).
- `aggregate_baselines_vs_mlp.py` — cross-model heatmap aggregator.
- `exp3_cosine_deciles.py` — leakage diagnostic Exp 3.

### `src/utils/`

Pipeline-wide utilities. Most-used:
- `config_hydra.py` — Hydra config loader (primary entry).
- `path_utils.py` — path generation for inputs / outputs.
- `seed_utils.py` — hierarchical seed system (see
  [`../../docs/SEED_SYSTEM.md`](../../docs/SEED_SYSTEM.md)).
- `esm2_utils.py`, `embedding_utils.py` — ESM-2 + embedding I/O.
- `kmer_utils.py` — k-mer feature loading + pair construction.
- `metadata_enrichment.py` — joins host/year/subtype/etc. onto the
  per-isolate dataframe.
- `gto_utils.py`, `protein_utils.py` — GTO parsing + protein QC.
- `learning_verification_utils.py` — Karpathy-style training-sanity
  checks (see
  [`../../docs/plans/2026-05-12_model_validation_plan.md`](../../docs/plans/2026-05-12_model_validation_plan.md)).

For the authoritative file-by-file list with descriptions, see the
"Active Source Files" section of [`../../CLAUDE.md`](../../CLAUDE.md).

## Configuration

Hydra **bundle-per-experiment**. Each experiment is one YAML in
`conf/bundles/`. Bundles inherit from a chain of base configs in
`conf/{virus,paths,dataset,embeddings,training,baselines}/`. See
[`../../conf/bundles/README.md`](../../conf/bundles/README.md) for the
inheritance chain and bundle status conventions (`active` /
`ablation` / `experimental` / `legacy`).

Detailed config docs: [`../../docs/conf_guide.md`](../../docs/conf_guide.md).

## Design principles

1. **Configuration-driven.** Every stage takes a single `--config_bundle`
   argument; everything else flows from there.
2. **Stage decoupling.** Stages 1 and 2 are shared per
   `(virus, data_version)`. Stages 3 and 4 are per-experiment.
   Stage 3 produces a dataset directory; Stage 4's `--dataset_dir` is
   required and is how training picks up that build. Same dataset, N
   training runs.
3. **Master cache for embeddings.** ESM-2 embeddings (and k-mer
   features) are computed once per data version and reused across all
   downstream bundles. The protein-level row index lives alongside the
   HDF5 / NPZ for O(1) lookup.
4. **Hierarchical seeding.** `master_seed` → per-process seeds via
   `seed_utils.resolve_process_seed`. See
   [`../../docs/SEED_SYSTEM.md`](../../docs/SEED_SYSTEM.md).
5. **Lean shell wrappers.** Each `stage*_*.sh` is a thin wrapper around
   the Python entry point with provenance logging. No registry; no
   experiment-tracking system. Stages 1, 2b, 3, 4 follow the lean
   pattern at <100 lines each; stage2_esm2 is the one outlier still
   on the older verbose pattern (see
   [`../../docs/plans/code_cleanup_plan.md`](../../docs/plans/code_cleanup_plan.md)
   item 3).

## Adding a new bundle

1. Create `conf/bundles/<name>.yaml` with a `# STATUS: active|ablation|…`
   header comment.
2. Set up the `defaults:` chain — typically `flu_base` or
   `flu_schema_raw_slot_norm_unit_diff` for current Gen-3 bundles, or
   inherit from `flu_28_major_protein_pairs_master` for an all-pairs
   sweep variant.
3. Set the bundle-specific overrides (`virus.selected_functions`,
   `dataset.split_strategy.{mode,hash_key}`, `training.slot_transform`,
   `training.interaction`, etc.).
4. Smoke-test the config load:
   ```bash
   python -c "
   from src.utils.config_hydra import get_virus_config_hydra, print_config_summary
   print_config_summary(get_virus_config_hydra('<name>'))
   "
   ```
5. Run Stages 3 + 4 on the new bundle.

## Adding a new analysis script

1. Place under `src/analysis/`.
2. Take `--config_bundle` (and optionally `--model_dir`) at minimum.
3. Read inputs from the conventional locations (use
   `path_utils.build_training_paths` to derive them).
4. Write outputs under the run directory (`models/.../runs/.../post_hoc/`
   for per-run; `results/.../runs/...` for cross-bundle aggregators).

## Related documentation

- [`../../CLAUDE.md`](../../CLAUDE.md) — authoritative module-by-module file list.
- [`../../docs/methods/pipeline_overview.md`](../../docs/methods/pipeline_overview.md) — pipeline architecture deep dive.
- [`../../docs/conf_guide.md`](../../docs/conf_guide.md) — Hydra configuration system.
- [`../../docs/methods/`](../../docs/methods/) — per-topic methods reference docs.
