# Quick Start Guide

End-to-end run of the viral-segmatch pipeline on the production Flu-A
dataset. For the pipeline architecture and methodology, see
[`../docs/methods/pipeline_overview.md`](../docs/methods/pipeline_overview.md).

## Prerequisites

Set up your environment first — see [`installation.md`](installation.md).
You need:
- conda env active, dependencies installed
- `data/raw/Full_Flu_Annos/July_2025/*.gto` populated (or the equivalent
  for whatever data version your bundles point at)

## The 4 stages

| Stage | What it does | Run-once or per-experiment? |
|---|---|---|
| 1. Preprocess (`stage1_preprocess_flu.sh`) | GTO JSON → `protein_final.csv` + `genome_final.csv` | Run once per data version |
| 2. Embeddings (`stage2_esm2.sh`) + k-mer (`stage2b_kmer.sh`) | Sequences → ESM-2 HDF5 cache and/or k-mer NPZ | Run once per data version |
| 3. Dataset (`stage3_dataset.sh`) | Pair construction + train/val/test split | Per experiment (per bundle) |
| 4. Train (`stage4_train.sh`, `stage4_baselines.sh`) | MLP + sklearn baselines | Per experiment, per model |

## First-time setup: Stages 1 + 2

Stages 1 and 2 are run once and shared across all downstream experiments:

```bash
# Stage 1 — Preprocess GTOs into protein_final.csv + genome_final.csv
./scripts/stage1_preprocess_flu.sh --config_bundle flu_base

# Stage 2 — ESM-2 embeddings (uses GPU; takes hours on full Flu-A)
./scripts/stage2_esm2.sh --config_bundle flu_base --cuda_name cuda:0

# Stage 2b — K-mer features (CPU; ~5-10 min on full Flu-A)
./scripts/stage2b_kmer.sh --config_bundle flu_base
```

Outputs:
- `data/processed/flu/July_2025/protein_final.csv` (one row per CDS, ~1.79 M rows)
- `data/processed/flu/July_2025/genome_final.csv` (one row per contig, 868,240 rows)
- `data/embeddings/flu/July_2025/master_esm2_embeddings.h5`
- `data/embeddings/flu/July_2025/kmer_features_k6.npz` (1.78 GB sparse, 868,240 × 4,096)

## Per-experiment: Stages 3 + 4

Current production bundles (HA/NA and PB2/PB1 deep dives):

```bash
# HA/NA experiment
./scripts/stage3_dataset.sh --config_bundle flu_ha_na
./scripts/stage4_train.sh   --config_bundle flu_ha_na --cuda_name cuda:0 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_YYYYMMDD_HHMMSS

# Same dataset, sklearn baselines (LightGBM, 1-NN cosine-margin, k-NN vote, logistic regression)
./scripts/stage4_baselines.sh --config_bundle flu_ha_na \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_YYYYMMDD_HHMMSS
```

Stages 3 and 4 are **decoupled** — Stage 3 produces a dataset directory
that any number of Stage 4 runs (different bundles, different model
configs) can train against. Stage 4's `--dataset_dir` is required and is
how the training script knows which build to read.

### Finding the dataset directory

`stage3_dataset.sh` prints the output path. To find the most recent run
for a bundle:

```bash
ls -td data/datasets/flu/July_2025/runs/dataset_flu_ha_na_* | head -1
```

## Available production bundles (as of 2026-05-12)

| Bundle | Schema | Notes |
|---|---|---|
| `flu_ha_na` | HA + NA | Variable surface antigens; current Test 3 production setting (`unit_norm + unit_diff + prod`, `seq_disjoint` routing) |
| `flu_pb2_pb1` | PB2 + PB1 | Conserved RdRp subunits; same Test 3 production setting |
| `flu_ha_na_regimes` | HA + NA | Same as `flu_ha_na` but with regime-aware negative sampling (`negative_sampling.regime_targets`) at `neg_to_pos_ratio: 2.0` |
| `flu_pb2_pb1_regimes` | PB2 + PB1 | Same regime-aware variant for the conserved-protein bundle |
| `flu_ha_na_holdout_{host,subtype,year}` | HA + NA | Metadata-holdout variants for cross-population generalization tests |
| `flu_28p_*` | 28 pair-bundles | C(8,2) all-pairs sweep (Task 11 publication experiment) |

See [`../conf/bundles/README.md`](../conf/bundles/README.md) for the full
inheritance chain and the bundle status (`active` / `ablation` /
`experimental` / `legacy`).

## Analyze results

After training, run the per-model post-hoc analysis:

```bash
# Per-model analysis (confusion matrix, ROC, per-regime TPR/TNR heatmap, FP/FN breakdown)
python src/analysis/analyze_stage4_train.py --config_bundle flu_ha_na

# Or specify the model directory explicitly
python src/analysis/analyze_stage4_train.py --config_bundle flu_ha_na \
    --model_dir models/flu/July_2025/runs/training_flu_ha_na_YYYYMMDD_HHMMSS
```

Then aggregate baselines vs MLP into a heatmap:

```bash
# Autodiscovers the matching baseline + MLP runs for the bundle
python src/analysis/aggregate_baselines_vs_mlp.py --bundle flu_ha_na
```

Outputs land under `results/flu/July_2025/runs/baselines_vs_mlp_flu_ha_na_*`:
- `baselines_vs_mlp_heatmap.png` (5-row heatmap: 1 MLP + 4 baselines × 9 columns: positive TPR + 8 negative regimes' TNR)
- `baselines_vs_mlp_overall.png` (aggregate AUC / F1 / MCC bars)
- `baselines_vs_mlp.csv` (raw numbers)

## What "good" performance looks like (current production)

Verified 2026-05-12 from
`results/flu/July_2025/runs/baselines_vs_mlp_*_20260512_*/baselines_vs_mlp.csv`:

| Model | HA/NA AUC-ROC | HA/NA MCC | PB2/PB1 AUC-ROC | PB2/PB1 MCC |
|---|---:|---:|---:|---:|
| MLP | 0.9771 | 0.885 | 0.9760 | 0.887 |
| LightGBM | 0.9830 | 0.881 | 0.9824 | 0.879 |
| 1-NN margin | 0.9771 | 0.892 | 0.9815 | 0.900 |

Note: on PB2/PB1, 1-NN edges MLP on MCC — consistent with the
conservation hypothesis (fewer truly-novel test sequences). See
[`../docs/methods/leakage_definitions.md`](../docs/methods/leakage_definitions.md)
for the "biology learning" criterion this comparison anchors.

## Output structure (full pipeline)

```
data/
├── processed/flu/July_2025/                                   # Stage 1 outputs (shared)
├── embeddings/flu/July_2025/                                  # Stage 2 + 2b outputs (shared)
└── datasets/flu/July_2025/runs/dataset_<bundle>_YYYYMMDD_HHMMSS/    # Stage 3 outputs (per experiment)

models/flu/July_2025/runs/
├── training_<bundle>_YYYYMMDD_HHMMSS/                         # MLP training
└── baseline_<name>_<bundle>_YYYYMMDD_HHMMSS/                  # Sklearn baseline training (per baseline)

results/flu/July_2025/runs/
└── baselines_vs_mlp_<bundle>_YYYYMMDD_HHMMSS/                 # Cross-model heatmap aggregator output
```

## Common next steps

- **Troubleshoot a failure:** [`troubleshooting.md`](troubleshooting.md)
- **Read the analysis outputs:** [`analysis/results-analysis.md`](analysis/results-analysis.md)
- **Configure a new experiment bundle:** [`../docs/conf_guide.md`](../docs/conf_guide.md)
- **Deep dive on the pipeline:** [`../docs/methods/pipeline_overview.md`](../docs/methods/pipeline_overview.md)
- **Understand the leakage / "biology learning" framing:** [`../docs/methods/leakage_definitions.md`](../docs/methods/leakage_definitions.md)
