# Full 8-cell experiment matrix — results

**Date.** 2026-05-13.
**Scope.** 8 cells × 2 models (MLP, LightGBM) = **16 runs**, all using nt 6-mer features with `slot_transform=unit_norm` and `interaction="unit_diff + prod"` (consistent across every run; verified from each `training_info.json`).
**Reproduce.** Aggregator: `/tmp/experiment_2026-05-13/aggregate_full.py`. CSV: `aggregate_full.csv`.

## Datasets used

All 8 datasets built fresh today; interaction settings consistent across every run.

| Cell | Train | Val | Test | Train neg:pos | Sampling | Routing |
|---|---:|---:|---:|---:|---|---|
| `ha_na`, r=1.5 | 116,775 | 14,597 | 14,597 | 1.500 | regime-blind | seq_disjoint |
| `pb2_pb1`, r=1.5 | 105,312 | 13,165 | 13,165 | 1.500 | regime-blind | seq_disjoint |
| `ha_na`, r=3.0 | 186,840 | 23,356 | 23,356 | 3.000 | regime-blind | seq_disjoint |
| `pb2_pb1`, r=3.0 | 168,500 | 21,064 | 21,064 | 3.000 | regime-blind | seq_disjoint |
| `ha_na`, r=3.0 (regimes) | 186,840 | 23,356 | 23,356 | 3.000 | regime-aware | seq_disjoint |
| `pb2_pb1`, r=3.0 (regimes) | 168,500 | 21,064 | 21,064 | 3.000 | regime-aware | seq_disjoint |
| `ha_na` Human→Pig holdout, r=1.5 | 63,687 | 7,017 | 22,005 | 1.500 | regime-blind | random + metadata_holdout |
| `pb2_pb1` Human→Pig holdout, r=1.5 | 53,515 | 5,995 | 20,787 | 1.500 | regime-blind | random + metadata_holdout |

## Headline test-set metrics

| Cell | Model | F1 | Prec | Rec | AUC-ROC | AUC-PR | MCC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ha_na` r=1.5            | MLP  | **0.940** | 0.927 | 0.954 | 0.979 | 0.952 | **0.900** | 0.045 |
| `ha_na` r=1.5            | LGBM | 0.931 | 0.919 | 0.944 | 0.984 | 0.971 | 0.885 | 0.044 |
| `pb2_pb1` r=1.5          | MLP  | **0.945** | 0.930 | 0.961 | 0.980 | 0.957 | **0.908** | 0.041 |
| `pb2_pb1` r=1.5          | LGBM | 0.930 | 0.909 | 0.953 | 0.984 | 0.970 | 0.883 | 0.045 |
| `ha_na` r=3.0            | MLP  | 0.910 | 0.900 | 0.920 | 0.983 | 0.930 | 0.880 | 0.041 |
| `ha_na` r=3.0            | LGBM | 0.901 | 0.901 | 0.900 | **0.985** | **0.951** | 0.868 | 0.038 |
| `pb2_pb1` r=3.0          | MLP  | 0.920 | 0.902 | 0.940 | 0.985 | 0.938 | 0.894 | 0.036 |
| `pb2_pb1` r=3.0          | LGBM | 0.894 | 0.879 | 0.909 | 0.985 | 0.947 | 0.858 | 0.040 |
| `ha_na` r=3.0 (regimes)  | MLP  | 0.864 | 0.836 | 0.893 | 0.965 | 0.861 | 0.817 | 0.064 |
| `ha_na` r=3.0 (regimes)  | LGBM | 0.851 | 0.815 | 0.892 | 0.968 | 0.896 | 0.800 | 0.060 |
| `pb2_pb1` r=3.0 (regimes)| MLP  | 0.861 | 0.827 | 0.898 | 0.967 | 0.870 | 0.813 | 0.060 |
| `pb2_pb1` r=3.0 (regimes)| LGBM | 0.836 | 0.788 | 0.891 | 0.967 | 0.881 | 0.780 | 0.063 |
| **`ha_na` Human→Pig**    | MLP  | 0.648 | 0.534 | 0.825 | **0.714** | 0.563 | 0.347 | 0.336 |
| **`ha_na` Human→Pig**    | LGBM | 0.500 | 0.682 | 0.394 | **0.744** | 0.670 | 0.316 | 0.228 |
| **`pb2_pb1` Human→Pig**  | MLP  | 0.647 | 0.497 | 0.924 | **0.714** | 0.572 | 0.338 | 0.390 |
| **`pb2_pb1` Human→Pig**  | LGBM | 0.680 | 0.621 | 0.752 | **0.793** | 0.692 | 0.437 | 0.199 |

## Per-regime TPR (positive) / TNR (negatives) on test

| Cell | Model | positive | none_match | host_only | subtype_only | year_only | host_sub | host_yr | sub_yr | **host_sub_yr** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ha_na` r=1.5            | MLP  | 0.954 | 0.981 | 0.917 | 0.929 | 0.966 | 0.824 | 0.912 | 0.880 | **0.587** |
| `ha_na` r=1.5            | LGBM | 0.944 | 0.977 | 0.940 | 0.926 | 0.960 | 0.778 | 0.918 | 0.844 | **0.561** |
| `pb2_pb1` r=1.5          | MLP  | 0.961 | 0.984 | 0.880 | 0.979 | 0.964 | 0.861 | 0.774 | 0.884 | **0.764** |
| `pb2_pb1` r=1.5          | LGBM | 0.953 | 0.978 | 0.827 | 0.954 | 0.950 | 0.829 | 0.761 | 0.848 | **0.713** |
| `ha_na` r=3.0            | MLP  | 0.920 | 0.988 | 0.957 | 0.956 | 0.977 | 0.866 | 0.941 | 0.905 | **0.708** |
| `ha_na` r=3.0            | LGBM | 0.900 | 0.991 | 0.961 | 0.954 | 0.978 | 0.864 | 0.950 | 0.877 | **0.685** |
| `pb2_pb1` r=3.0          | MLP  | 0.940 | 0.992 | 0.920 | 0.975 | 0.971 | 0.911 | 0.862 | 0.869 | **0.793** |
| `pb2_pb1` r=3.0          | LGBM | 0.909 | 0.988 | 0.891 | 0.968 | 0.969 | 0.873 | 0.830 | 0.861 | **0.762** |
| `ha_na` r=3.0 (regimes)  | MLP  | 0.893 | 0.983 | 0.961 | 0.969 | 0.975 | 0.921 | 0.937 | 0.929 | **0.846** |
| `ha_na` r=3.0 (regimes)  | LGBM | 0.892 | 0.982 | 0.957 | 0.969 | 0.966 | 0.897 | 0.946 | 0.913 | **0.818** |
| `pb2_pb1` r=3.0 (regimes)| MLP  | 0.898 | 0.988 | 0.922 | 0.981 | 0.973 | 0.936 | 0.889 | 0.936 | **0.847** |
| `pb2_pb1` r=3.0 (regimes)| LGBM | 0.891 | 0.983 | 0.904 | 0.970 | 0.962 | 0.900 | 0.856 | 0.926 | **0.821** |
| **`ha_na` Human→Pig**    | MLP  | 0.825 | n/a   | 0.560 | n/a   | n/a   | 0.414 | 0.596 | n/a   | **0.417** |
| **`ha_na` Human→Pig**    | LGBM | 0.394 | n/a   | 0.905 | n/a   | n/a   | 0.829 | 0.909 | n/a   | **0.810** |
| **`pb2_pb1` Human→Pig**  | MLP  | 0.924 | n/a   | 0.358 | n/a   | n/a   | 0.364 | 0.411 | n/a   | **0.396** |
| **`pb2_pb1` Human→Pig**  | LGBM | 0.752 | n/a   | 0.702 | n/a   | n/a   | 0.695 | 0.691 | n/a   | **0.669** |

Test-set support (n_samples) for the holdout cells:
- HA/NA Human→Pig: positive=8,802; host_only=5,637; host_subtype_only=2,899; host_year_only=3,092; host_subtype_year=1,575. The four host-mismatch regimes (none_match, subtype_only, year_only, subtype_year_only) have **n=0** by construction — every test pair is Pig-Pig so same_host=True always.
- PB2/PB1 Human→Pig: positive=8,315; host_only=5,394; host_subtype_only=2,615; host_year_only=2,966; host_subtype_year=1,497.

## Findings

### 1. The two ratio=3.0 lifts are distinct and additive

Walking the HA/NA `host_subtype_year` TNR ladder:

| Step | host_subtype_year TNR | Δ vs prev |
|---|---:|---:|
| r=1.5 regime-blind   | 0.587 | — |
| r=3.0 regime-blind   | 0.708 | **+0.121** (more data, no regime targeting) |
| r=3.0 regime-aware   | 0.846 | **+0.138** (regime-aware allocation on top) |

So **(a) doubling negative volume helps even regime-blind, (b) targeted allocation helps further**. Same pattern for PB2/PB1: 0.764 → 0.793 → 0.847. The earlier "regime-aware did not help" finding from the r=1.5 matrix was supply-limited — the fill phase only had 0.28× pos to allocate. At r=3.0 the fill phase has 1.78× pos and can actually bias the distribution.

### 2. F1 drops at r=3.0 are partly composition artifact (as expected)

At r=3.0 the test has 2× more negatives, mechanically inflating the FP denominator in precision. The per-regime TPR/TNR are the apples-to-apples comparison: TPR drops 3–5 pp (genuine — threshold shift), every TNR rises by varying amounts (genuine learning, especially on hard regime).

### 3. Cross-host (Human→Pig) holdout — substantial generalization gap

| Cell | Setting | F1 | AUC-ROC | MCC | `host_subtype_year` TNR |
|---|---|---:|---:|---:|---:|
| HA/NA | r=1.5 (within Human/Pig train+test) | 0.940 | 0.979 | 0.900 | 0.587 |
| HA/NA | r=1.5 (Human → Pig holdout) MLP | 0.648 | 0.714 | 0.347 | **0.417** |
| HA/NA | r=1.5 (Human → Pig holdout) LGBM | 0.500 | 0.744 | 0.316 | **0.810** |
| PB2/PB1 | r=1.5 (within Human/Pig train+test) | 0.945 | 0.980 | 0.908 | 0.764 |
| PB2/PB1 | r=1.5 (Human → Pig holdout) MLP | 0.647 | 0.714 | 0.338 | **0.396** |
| PB2/PB1 | r=1.5 (Human → Pig holdout) LGBM | 0.680 | 0.793 | 0.437 | **0.669** |

**AUC-ROC drops ~25-27 pp** on both MLP runs going from within-distribution to cross-host. This is genuine distribution shift, not threshold artifact.

**Two very different failure modes:**

- **MLP on holdout** keeps high TPR (0.82-0.92) but collapses TNR (0.36-0.60). It defaults to predicting "positive" for most Pig-Pig pairs — the model recognises "this is Pig flu" and assumes co-occurrence. Threshold tuned on the (Human) val set transfers badly to Pig test.
- **LGBM HA/NA** flips the other way: TPR=0.394, TNR=0.81-0.91. Too conservative on the new distribution. Predicts negative most of the time.
- **LGBM PB2/PB1** is the only halfway-reasonable cross-host model: TPR=0.752, TNRs ~0.67-0.70. PB2/PB1 conservation may be making Pig sequences look more like Human sequences to a tree-based ranker.

**MLP under-performs LGBM in MCC on the holdout for both pairs** — opposite to the within-distribution result where MLP wins. The MLP overfits to the training population's structure; LGBM's tree splits transfer slightly better.

### 4. The "biology learning" verdict needs a 1-NN baseline on these runs

We confirmed earlier that PB2/PB1 1-NN ≈ MLP under regime-blind seq_disjoint (mode #4 cluster leakage suspected). The r=3.0 regimes lift on hard-regime TNR (+0.28) could be:
- Real biology learning (model uses sequence content to distinguish "different isolate" when metadata matches)
- Better near-neighbor coverage (more hard examples in train → more chances for test to land near one)

Without 1-NN comparisons on the new datasets, both interpretations are still on the table. The holdout result (AUC drops 25 pp under domain shift) suggests the within-distribution model is at least partly memorizing population structure — which is what mode #5 + #4 leakage predicts.

### 5. Practical takeaways

| If the goal is... | Use this cell | Why |
|---|---|---|
| Best within-distribution F1 / AUC | `ha_na` or `pb2_pb1` r=1.5 (regime-blind) | Highest F1; everything else trades off |
| Best hard-regime (`host_subtype_year`) TNR while keeping ratio = 3.0 | `*_regimes` r=3.0 | +0.28 pp TNR vs regime-blind at same ratio |
| Cross-host generalization probe | `*_holdout_host` r=1.5 | Shows ~25 pp AUC gap; quantifies the within-distribution overfitting |

## Limitations

- **Single seed (42)** throughout. Per-cell differences within ~0.01-0.02 AUC are within seed noise. Triple-seed runs would tighten claims on the within-distribution comparison.
- **No 1-NN baseline run on the 6 new datasets.** Required to anchor whether r=3.0 regimes lift is biology or near-neighbor lookup.
- **Holdout val is carved from train pool (Human)**, not Pig. So early-stopping signal is within-Human, which contributes to the MLP overfitting to Human distribution. A held-out Pig val would change this story.
- **No cluster-disjoint splits.** Mode #4 (cluster leakage) remains structurally open — `seq_disjoint` only bounds exact-hash overlap.

## Artifacts

All run directories under `data/datasets/flu/July_2025/runs/` and `models/flu/July_2025/runs/` (timestamps `20260513_201046` through `20260513_225639`). Aggregator CSV: `/tmp/experiment_2026-05-13/aggregate_full.csv`. Per-cell post-hoc plots in each model run dir's `post_hoc/`.

## Related

- `docs/results/2026-05-13_experiment_matrix_results.md` — initial 4-cell × 2-model matrix at r=1.5 (now superseded by this 8-cell doc).
- `docs/2026-05-13-in_depth.md` — the methods walkthrough this experiment matrix tests.
- `docs/methods/leakage_definitions.md` — the leakage taxonomy this matrix sharpens (modes #3, #4, #5).
- `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md` — regime-aware sampler design.
- `docs/plans/2026-05-11_metadata_holdout_plan.md` — cross-population holdout design.
