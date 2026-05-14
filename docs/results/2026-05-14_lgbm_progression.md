# LGBM progression — `neg_to_pos_ratio` and `regime_targets` tradeoffs

**Date.** 2026-05-14.
**Scope.** LightGBM trained on nt 6-mer features across **2 schema pairs ×
3 settings = 6 cells**. All cells use `seq_disjoint` routing with
`hash_key=seq` and `unit_norm` slot transform (consistent across the matrix).

## The question

Two levers control how the negative pool is shaped during dataset
construction:

1. **`neg_to_pos_ratio`** — how many negatives per positive. Larger ratio
   gives more training negatives, but also shifts the class balance:
   the model becomes more conservative on positive predictions.
2. **`regime_targets`** — when set, the regime-aware fill phase biases
   the negatives toward harder metadata-match patterns (especially
   `host_subtype_year`, where host AND subtype AND year all match
   between the two isolates of a negative pair, so the model has no
   metadata shortcut and must rely on sequence content).

We isolate each lever via a 3-step progression per schema pair:

```
r=1.5 regime-blind  →  r=3.0 regime-blind  →  r=3.0 regime-aware
   (baseline)          (more negatives)        (+ targeted allocation)
```

## Headline tradeoff — what to look at

| Cell | TPR (positive) | TNR (`none_match`) | **TNR (`host_subtype_year`)** | F1 | F1 macro | MCC | AUC-ROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HA/NA**, r=1.5, regime-blind   | **0.944** | 0.977 | 0.561 | **0.931** | **0.942** | **0.885** | 0.984 |
| **HA/NA**, r=3.0, regime-blind   | 0.900 | 0.991 | 0.685 | 0.901 | 0.934 | 0.868 | 0.985 |
| **HA/NA**, r=3.0, regime-aware   | 0.892 | 0.982 | **0.817** | 0.851 | 0.899 | 0.800 | 0.968 |
| **PB2/PB1**, r=1.5, regime-blind | **0.953** | 0.978 | 0.713 | **0.930** | **0.941** | **0.883** | 0.983 |
| **PB2/PB1**, r=3.0, regime-blind | 0.909 | 0.988 | 0.762 | 0.894 | 0.929 | 0.858 | 0.984 |
| **PB2/PB1**, r=3.0, regime-aware | 0.891 | 0.982 | **0.821** | 0.836 | 0.889 | 0.780 | 0.967 |

Two trends are visible across the progression on both schema pairs:

- **TPR drops monotonically** (0.944 → 0.900 → 0.892 on HA/NA; 0.953 → 0.909 → 0.891 on PB2/PB1) — going up in `neg_to_pos_ratio` makes the model more conservative on positive predictions. The threshold tuned on val shifts upward as the val set gains negatives.
- **`host_subtype_year` TNR rises monotonically** (0.561 → 0.685 → 0.817 on HA/NA; 0.713 → 0.762 → 0.821 on PB2/PB1) — both levers contribute additively. Doubling the ratio (regime-blind) gives ~+0.12 on HA/NA and ~+0.05 on PB2/PB1; layering regime-aware allocation on top adds another ~+0.13 and ~+0.06 respectively.

The aggregate F1 drops along the same axis (0.931 → 0.901 → 0.851 on HA/NA), but **this is partly a test-set composition artifact**: at r=3.0 the test set has 2× more negatives than at r=1.5, mechanically inflating the FP denominator in precision. The per-regime TNR breakdown is the apples-to-apples comparison. AUC-ROC is composition-invariant: it stays flat from r=1.5 to r=3.0 regime-blind (0.984 → 0.985) but drops modestly under regime-aware sampling (0.985 → 0.968) — a real but small ranking-quality cost for the hard-regime accuracy gain.

## HA/NA progression

### Step 1 — HA/NA, `neg_to_pos_ratio=1.5`, regime-blind (baseline)

Dataset: `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260513_201046/` &nbsp;&nbsp;|&nbsp;&nbsp; Model: `models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_20260513_202802/`

![split composition](../../data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260513_201046/plots/split_composition.png)

![confusion matrix](../../models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_20260513_202802/post_hoc/confusion_matrix.png)

![per-regime TPR/TNR](../../models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_20260513_202802/post_hoc/level1_neg_regimes.png)

The natural baseline. Model predicts positives confidently (TPR 0.944), but the `host_subtype_year` regime is the visible weak point — when negatives have all three metadata axes matched between the two isolates, the model classifies them correctly only 56.1% of the time. Below random would be a real failure; this is "model uses metadata as shortcut and loses signal when the shortcut isn't available."

### Step 2 — HA/NA, `neg_to_pos_ratio=3.0`, regime-blind

Dataset: `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_ratio3_20260513_222429/` &nbsp;&nbsp;|&nbsp;&nbsp; Model: `models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_20260513_225631/`

![split composition](../../data/datasets/flu/July_2025/runs/dataset_flu_ha_na_ratio3_20260513_222429/plots/split_composition.png)

![confusion matrix](../../models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_20260513_225631/post_hoc/confusion_matrix.png)

![per-regime TPR/TNR](../../models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_20260513_225631/post_hoc/level1_neg_regimes.png)

Only the dataset size changed (ratio 1.5 → 3.0). No regime targeting. The `host_subtype_year` test TNR jumps 0.561 → 0.685 (+0.124) — even regime-blind, simply having more training negatives gives the model more examples of the hardest regime to learn from (because the regime composition in the natural distribution scales proportionally with size). TPR drops 0.944 → 0.900 from the more-conservative threshold.

### Step 3 — HA/NA, `neg_to_pos_ratio=3.0`, regime-aware

Dataset: `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_regimes_ratio3_20260513_211559/` &nbsp;&nbsp;|&nbsp;&nbsp; Model: `models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_regimes_20260513_212352/`

![split composition by regime](../../data/datasets/flu/July_2025/runs/dataset_flu_ha_na_regimes_ratio3_20260513_211559/plots/split_composition_by_regime.png)

![confusion matrix](../../models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_regimes_20260513_212352/post_hoc/confusion_matrix.png)

![per-regime TPR/TNR](../../models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_regimes_20260513_212352/post_hoc/level1_neg_regimes.png)

Same ratio, but `regime_targets` reweights the fill phase toward `host_subtype_year` (30% target). Train neg composition shifts from naturally-dominant `none_match` to a forced 30% of host_subtype_year — about 11× more hard-regime training negatives than under the regime-blind run. `host_subtype_year` test TNR jumps another 0.685 → 0.817 (+0.132). The model genuinely learns to use sequence content for the hardest regime. Cost: aggregate F1 drops to 0.851 (partly composition artifact, partly real ranking-quality loss visible in AUC-ROC 0.968).

## PB2/PB1 progression

### Step 1 — PB2/PB1, `neg_to_pos_ratio=1.5`, regime-blind (baseline)

Dataset: `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_20260513_201050/` &nbsp;&nbsp;|&nbsp;&nbsp; Model: `models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_20260513_202806/`

![split composition](../../data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_20260513_201050/plots/split_composition.png)

![confusion matrix](../../models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_20260513_202806/post_hoc/confusion_matrix.png)

![per-regime TPR/TNR](../../models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_20260513_202806/post_hoc/level1_neg_regimes.png)

Same setup as HA/NA baseline but on the conserved schema pair. `host_subtype_year` TNR starts higher (0.713 vs 0.561 on HA/NA) — but this is consistent with PB2/PB1's higher cluster-leakage (more isolates share near-identical proteins, so the model has near-neighbor anchors even on metadata-matched negatives). The model isn't necessarily learning more biology on PB2/PB1; it's exploiting denser sequence clusters.

### Step 2 — PB2/PB1, `neg_to_pos_ratio=3.0`, regime-blind

Dataset: `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_ratio3_20260513_222431/` &nbsp;&nbsp;|&nbsp;&nbsp; Model: `models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_20260513_225633/`

![split composition](../../data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_ratio3_20260513_222431/plots/split_composition.png)

![confusion matrix](../../models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_20260513_225633/post_hoc/confusion_matrix.png)

![per-regime TPR/TNR](../../models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_20260513_225633/post_hoc/level1_neg_regimes.png)

Doubling the ratio lifts hard-regime TNR by +0.049 (less than HA/NA's +0.124) — consistent with PB2/PB1 having less headroom because much of its performance was already coming from sequence clustering rather than from regime-blind training data.

### Step 3 — PB2/PB1, `neg_to_pos_ratio=3.0`, regime-aware

Dataset: `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_regimes_ratio3_20260513_222433/` &nbsp;&nbsp;|&nbsp;&nbsp; Model: `models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_regimes_20260513_225635/`

![split composition by regime](../../data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_regimes_ratio3_20260513_222433/plots/split_composition_by_regime.png)

![confusion matrix](../../models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_regimes_20260513_225635/post_hoc/confusion_matrix.png)

![per-regime TPR/TNR](../../models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_regimes_20260513_225635/post_hoc/level1_neg_regimes.png)

Regime-aware sampling layered on top adds another +0.059 on `host_subtype_year` TNR (0.762 → 0.821). Final hard-regime TNR is essentially tied between HA/NA and PB2/PB1 at this setting (0.817 vs 0.821), even though they started in very different places.

## What the progression tells us

**`neg_to_pos_ratio` controls the TPR vs hard-regime-TNR tradeoff axis.** Higher ratio → more conservative threshold → fewer positive predictions, but more training exposure to hard negatives. The model goes from "confident on positives, weak on hard negatives" to "balanced confidence, stronger on hard negatives."

**`regime_targets` accelerates the hard-regime lift without requiring even more dataset size.** At r=3.0, switching from regime-blind to regime-aware shifts the train neg distribution: `host_subtype_year` jumps from naturally-emergent ~4% to a targeted 30%. The model uses the extra hard-regime examples to push test TNR another ~13 percentage points on HA/NA without changing the total number of training pairs.

**The two levers are partially-substitutable.** On HA/NA, going r=1.5 blind → r=3.0 blind buys 0.12 of hard-regime TNR; going r=3.0 blind → r=3.0 regimes buys another 0.13. They're additive but neither alone closes the gap from the 0.561 baseline to the 0.817 final.

**Aggregate F1 is the wrong headline metric for this progression**, because the test composition changes with ratio. AUC-ROC is the more honest aggregate signal: stable across r=1.5 vs r=3.0 (regime-blind), small drop under regime-aware (0.985 → 0.968).

## Caveats

- **Single seed (42)**. Per-cell metric differences within ~0.01 AUC are likely within seed noise.
- **No 1-NN baseline.** The "biology-learning" criterion (MLP / LGBM > 1-NN by ≥0.02 AUC on `seq_disjoint`) requires that comparison; this matrix alone can't say whether the hard-regime TNR lift comes from learned representations or from improved cluster-membership confidence.
- **Test set composition differs across the 3 settings** by ratio. F1 comparisons across the rows of the headline table over-state the regression because the FP denominator grows.
- **Cluster leakage (mode #4) not addressed.** All cells use `seq_disjoint` to bound the exact-hash case; mmseqs2 cluster-disjoint splits remain pending.

## Related

- `docs/results/2026-05-13_full_8cell_matrix.md` — the full 8-cell matrix including holdout cells.
- `docs/2026-05-13-in_depth.md` — methods walkthrough.
- `docs/methods/leakage_definitions.md` — the leakage taxonomy this progression exercises (mode #5 demographic shortcut).
- `docs/plans/2026-05-14_regime_aware_coverage_plan.md` — proposed next-step mechanism that extends regime targeting into the coverage phase.
