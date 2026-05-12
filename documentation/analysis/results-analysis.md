# Results Analysis

How to interpret the outputs of a training run + the post-hoc analysis
pipeline. For methodology, see
[`../../docs/methods/pipeline_overview.md`](../../docs/methods/pipeline_overview.md)
and
[`../../docs/post_hoc_analysis_design.md`](../../docs/post_hoc_analysis_design.md).

## What gets produced after a training run

A successful Stage 4 run (`stage4_train.sh` for the MLP, or
`stage4_baselines.sh` for the sklearn baselines) writes to:

```
models/flu/{data_version}/runs/training_<bundle>_<TS>/        # MLP run
models/flu/{data_version}/runs/baseline_<name>_<bundle>_<TS>/ # Each baseline (logistic, lgbm, knn1_margin, knn_vote)
```

Each run directory contains at least:

- `best_model.pt` (MLP) or `best_model.joblib` (sklearn baselines) — the trained model
- `test_predicted.csv` — per-pair predictions on the test split
- `training_info.json` — provenance (config bundle, dataset directory, hyperparameters, threshold, seed)
- `post_hoc/` — analysis artifacts (see below)

## Per-run post-hoc analysis

```bash
# Autodiscover the most recent training run for a bundle
python src/analysis/analyze_stage4_train.py --config_bundle flu_ha_na

# Or specify an explicit run directory
python src/analysis/analyze_stage4_train.py --config_bundle flu_ha_na \
    --model_dir models/flu/July_2025/runs/training_flu_ha_na_YYYYMMDD_HHMMSS
```

Outputs (under `<run_dir>/post_hoc/`):

| File | Content |
|---|---|
| `metrics.csv` | Summary: F1 (binary + macro), precision, recall, AUC-ROC, AUC-PR, MCC, Brier, BCE loss |
| `confusion_matrix.png` / `.csv` | Test-set confusion matrix |
| `roc_curve.png` | ROC curve with AUC |
| `precision_recall_curve.png` | PR curve with average precision |
| `learning_curves.png` (MLP only) | Train/val loss + F1 + AUC per epoch |
| `level1_neg_regimes.csv` / `.png` | Per-metadata-regime TPR/TNR (8 negative regimes — see leakage_definitions.md for the taxonomy) |
| `error_analysis_summary.csv` | FP / FN counts by stratum |
| `fp_fn_analysis_*.png` | FP/FN distribution plots |

## Cross-model heatmap (the headline diagnostic)

After running both the MLP and at least one baseline for the same bundle:

```bash
python src/analysis/aggregate_baselines_vs_mlp.py --bundle flu_ha_na
```

Outputs land in `results/flu/{data_version}/runs/baselines_vs_mlp_<bundle>_<TS>/`:

| File | Content |
|---|---|
| `baselines_vs_mlp_heatmap.png` | (n_models × 9 regimes) heatmap. Columns: `positive` (TPR) + 8 negative regimes (TNR each, ascending hardness from `none_match` to `host_subtype_year`). Rows: typically MLP + 4 baselines (logistic, lgbm, knn1_margin, knn_vote). |
| `baselines_vs_mlp_overall.png` | Aggregate metrics (AUC-ROC, AUC-PR, F1, MCC) per model, side-by-side bars. |
| `baselines_vs_mlp.csv` | The raw numbers from both plots. |

## How to read the per-regime heatmap

The **headline column** is `host_subtype_year` (TNR) — the hardest
regime, where the two sides of a negative pair share host AND subtype
AND year. The model has no metadata shortcut; whatever discrimination
shows here must come from sequence content.

- **1-NN baseline (`knn1_margin`)** is the **leakage anchor**. A high
  `host_subtype_year` TNR for 1-NN means the test set is densely
  connected to train in feature space — even nearest-neighbor lookup
  gets it right. This is the leakage signal.
- For the MLP to claim "biology learning," it needs to **beat 1-NN by
  ≥0.02 on AUC** on sequence-disjoint splits. If MLP ≈ 1-NN, the MLP
  is doing soft memorization with extra steps. See
  [`../../docs/methods/leakage_definitions.md`](../../docs/methods/leakage_definitions.md)
  ("biology learning" criterion).
- A model that scores ~1.0 on `none_match` but drops sharply on
  `host_subtype_year` is using metadata coincidence as a shortcut.

## Aggregate metrics — current production reference points

Verified 2026-05-12 from the production HA/NA and PB2/PB1 builds
(`baselines_vs_mlp_*_20260512_*/baselines_vs_mlp.csv`):

| Model | HA/NA AUC-ROC | HA/NA MCC | PB2/PB1 AUC-ROC | PB2/PB1 MCC |
|---|---:|---:|---:|---:|
| MLP | 0.9771 | 0.885 | 0.9760 | 0.887 |
| LightGBM | 0.9830 | 0.881 | 0.9824 | 0.879 |
| 1-NN margin | 0.9771 | 0.892 | 0.9815 | 0.900 |

On PB2/PB1 the 1-NN baseline edges the MLP on MCC (0.900 vs 0.887) —
consistent with the conservation hypothesis (fewer truly-novel test
sequences on conserved proteins) and inside the seed-noise band on a
single-seed run. See
[`../../docs/results/2026-05-11_exp4a_seq_disjoint_results.md`](../../docs/results/2026-05-11_exp4a_seq_disjoint_results.md)
for the earlier HA/NA result under looser routing.

## Caveat baked into the heatmap caption

Per-slot preprocessing isn't uniform across models. The MLP applies
its `slot_transform` (current production: `unit_norm`, parameter-free
L2 row-norm). The sklearn baselines apply each model's natural
preprocessing (`StandardScaler` for LR, none for LightGBM, cosine
metric for 1-NN/k-NN which normalizes internally). The heatmap caption
flags this so row-to-row score differences aren't read as pure
model-quality differences when they could partly reflect featurization
differences. See
[`../../docs/methods/feature_normalization.md`](../../docs/methods/feature_normalization.md)
for the full (model × feature_source) defaults matrix.

## Debugging analysis

### Verify the test-predictions file

```python
import pandas as pd
from pathlib import Path

model_dir = Path('models/flu/July_2025/runs/training_flu_ha_na_YYYYMMDD_HHMMSS')
df = pd.read_csv(model_dir / 'test_predicted.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'Label distribution: {df["label"].value_counts().to_dict()}')
print(f'Pred prob range: {df["pred_prob"].min():.3f} - {df["pred_prob"].max():.3f}')
```

### Verify metrics

```python
import pandas as pd
metrics = pd.read_csv(model_dir / 'post_hoc' / 'metrics.csv')
print(metrics.to_string())
```

## Related

- [`../../docs/methods/pipeline_overview.md`](../../docs/methods/pipeline_overview.md) — pipeline architecture and §9 evaluation methodology
- [`../../docs/methods/leakage_definitions.md`](../../docs/methods/leakage_definitions.md) — 5-mode taxonomy + "biology learning" criterion
- [`../../docs/post_hoc_analysis_design.md`](../../docs/post_hoc_analysis_design.md) — Level 1 / Level 2 stratified-eval design
