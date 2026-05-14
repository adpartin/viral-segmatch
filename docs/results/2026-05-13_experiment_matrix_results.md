# Experiment matrix results — 4 bundles × {MLP, LightGBM}

**Date.** 2026-05-13.
**Scope.** 4 bundles × 2 models = 8 training runs, all using `nt k=6` k-mer features at `neg_to_pos_ratio: 1.5`.
**Goal.** Test whether regime-aware negative sampling materially shifts hard-regime TNR vs the regime-blind baseline, at equal dataset size.
**Reproduce.** Stage 3 builds + Stage 4 training launchers are in `/tmp/experiment_2026-05-13/` (aggregator: `aggregate.py`; output CSVs: `aggregate_metrics.csv`, `dataset_summary.csv`). All commits up to `b9580fa`.

## Dataset construction (Stage 3, `neg_to_pos_ratio: 1.5`)

All four datasets built fresh today; **equal-size control held by construction** (pos/neg/pair counts identical between regimes and non-regimes per schema pair, by design — they share the same positives, the same Phase 0–3 pipeline, and only differ in *how* Phase 5 distributes the same total `|neg|`).

| Bundle | Train pairs | Val pairs | Test pairs | Pos:Neg ratio | CC count | Largest CC | Pairs dropped |
|---|---:|---:|---:|---|---:|---:|---:|
| `flu_ha_na`          | 116,775 | 14,597 | 14,597 | 1 : 1.500 | 21,719 | 11,748 (20.1%) | **0** |
| `flu_ha_na_regimes`  | 116,775 | 14,597 | 14,597 | 1 : 1.500 | 21,719 | 11,748 (20.1%) | **0** |
| `flu_pb2_pb1`        | 105,312 | 13,165 | 13,165 | 1 : 1.500 | 14,924 | 20,214 (38.4%) | **0** |
| `flu_pb2_pb1_regimes`| 105,312 | 13,165 | 13,165 | 1 : 1.500 | 14,924 | 20,214 (38.4%) | **0** |

Audits passed cleanly on all four runs (cross-split `seq_hash` overlap = 0 on both slots; `n_seqs_with_zero_negatives = 0` per split; `pair_key` cross-split overlap = 0). Build wall-clock for all 4 in parallel: **2 min 41 s**.

### Train-set negative distribution by regime (the actual difference)

The non-regimes pipeline does not record `neg_regime` on training pairs, so the distribution below is **derived** from `(same_host, same_hn_subtype, same_year)` on the saved pairs.

| Regime | `flu_ha_na` (regime-blind) | `flu_ha_na_regimes` | `flu_pb2_pb1` (regime-blind) | `flu_pb2_pb1_regimes` |
|---|---:|---:|---:|---:|
| `none_match`         | **58.4%** (40,924) | 33.7% (23,583) | **61.4%** (38,769) | 36.3% (22,918) |
| `host_only`          | 15.6% (10,921) | 10.0% (7,006) | 14.3% (9,056) | 10.0% (6,319) |
| `subtype_only`       | 6.2% (4,377)  | 10.0% (7,006)  | 6.1% (3,886) | 10.0% (6,319) |
| `year_only`          | 3.1% (2,174)  | 16.7% (11,729) | 3.3% (2,062) | 18.3% (11,542) |
| `host_subtype_only`  | 14.0% (9,815) | 10.0% (7,006)  | 12.3% (7,786) | 10.0% (6,319) |
| `host_year_only`     | 1.0% (725)    | 10.0% (7,006)  | 1.0% (653)   | 9.5% (6,004) |
| `subtype_year_only`  | 0.5% (319)    | 5.4% (3,807) † | 0.5% (296)   | 2.2% (1,379) † |
| **`host_subtype_year`** | **1.2% (810)** | **4.2% (2,922) †** | **1.1% (679)** | **3.8% (2,387) †** |

† = `shortfall_reason: supply_exhausted_after_redistribute` (manifest target was 30% for `host_subtype_year` = ~21K negs, but supply ran out at ~3K). The regimes sampler boosted the hardest-regime training negatives by **~3.6× on HA/NA and ~3.5× on PB2/PB1**, but absolute counts remain small relative to the easy regimes.

## Test-set headline metrics

All metrics computed from `post_hoc/metrics.csv` (test split). MLP optimal threshold tuned on val-F1; LightGBM uses 0.5.

| Bundle | Model | F1 | Precision | Recall | AUC-ROC | AUC-PR | MCC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `flu_ha_na`         | **MLP**  | **0.9403** | 0.9272 | 0.9538 | 0.9787 | 0.9520 | **0.8998** | 0.0449 |
| `flu_ha_na`         | LGBM     | 0.9314 | 0.9188 | 0.9443 | **0.9836** | **0.9714** | 0.8849 | 0.0438 |
| `flu_ha_na_regimes` | **MLP**  | **0.9366** | 0.9224 | 0.9512 | 0.9785 | 0.9519 | **0.8935** | 0.0468 |
| `flu_ha_na_regimes` | LGBM     | 0.9270 | 0.9139 | 0.9404 | **0.9831** | **0.9706** | 0.8774 | 0.0450 |
| `flu_pb2_pb1`       | **MLP**  | **0.9452** | 0.9297 | 0.9613 | 0.9804 | 0.9570 | **0.9080** | 0.0413 |
| `flu_pb2_pb1`       | LGBM     | 0.9303 | 0.9090 | 0.9525 | **0.9835** | **0.9697** | 0.8826 | 0.0445 |
| `flu_pb2_pb1_regimes` | **MLP**  | **0.9401** | 0.9216 | 0.9594 | 0.9770 | 0.9468 | **0.8993** | 0.0457 |
| `flu_pb2_pb1_regimes` | LGBM     | 0.9257 | 0.9005 | 0.9523 | **0.9830** | **0.9695** | 0.8748 | 0.0463 |

**Per-cell winner.** MLP wins on F1 / Recall / MCC; LightGBM wins on AUC-ROC / AUC-PR. The pattern is consistent across all four cells: LightGBM is the stronger ranker (lower precision, higher AUC) but the MLP makes better discrete decisions at the tuned threshold.

## Per-regime breakdown — test set (the leakage story)

`positive` column reports TPR; the 8 regime columns report TNR (negatives correctly classified as different-isolate).

| Bundle | Model | positive | none_match | host_only | subtype_only | year_only | host_sub | host_yr | sub_yr | **host_sub_yr** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flu_ha_na`         | MLP  | 0.954 | 0.981 | 0.917 | 0.929 | 0.966 | 0.824 | 0.912 | 0.880 | **0.587** |
| `flu_ha_na`         | LGBM | 0.944 | 0.977 | 0.940 | 0.926 | 0.960 | 0.778 | 0.918 | 0.844 | **0.561** |
| `flu_ha_na_regimes` | MLP  | 0.951 | 0.978 | 0.926 | 0.946 | 0.961 | 0.816 | 0.899 | 0.898 | **0.567** |
| `flu_ha_na_regimes` | LGBM | 0.940 | 0.974 | 0.942 | 0.939 | 0.951 | 0.784 | 0.934 | 0.824 | **0.549** |
| `flu_pb2_pb1`       | MLP  | 0.961 | 0.984 | 0.880 | 0.979 | 0.964 | 0.861 | 0.774 | 0.884 | **0.764** |
| `flu_pb2_pb1`       | LGBM | 0.953 | 0.978 | 0.827 | 0.954 | 0.950 | 0.829 | 0.761 | 0.848 | **0.713** |
| `flu_pb2_pb1_regimes` | MLP  | 0.959 | 0.979 | 0.870 | 0.962 | 0.960 | 0.882 | 0.789 | 0.852 | **0.764** |
| `flu_pb2_pb1_regimes` | LGBM | 0.952 | 0.975 | 0.825 | 0.954 | 0.943 | 0.825 | 0.745 | 0.832 | **0.736** |

### The demographic-shortcut signature

**TNR drops 28–43 pp from `none_match` to `host_subtype_year` in every cell.** This is the leakage-taxonomy mode #5 signature (demographic shortcut): when host AND subtype AND year all match, the model can no longer use metadata correlation as a "different isolate" cue, and the TNR floor is set by sequence content alone.

| Bundle | Model | `none_match` TNR | `host_subtype_year` TNR | Δ (drop) |
|---|---|---:|---:|---:|
| `flu_ha_na`           | MLP  | 0.981 | 0.587 | **−0.394** |
| `flu_ha_na`           | LGBM | 0.977 | 0.561 | **−0.416** |
| `flu_ha_na_regimes`   | MLP  | 0.978 | 0.567 | **−0.411** |
| `flu_ha_na_regimes`   | LGBM | 0.974 | 0.549 | **−0.425** |
| `flu_pb2_pb1`         | MLP  | 0.984 | 0.764 | −0.220 |
| `flu_pb2_pb1`         | LGBM | 0.978 | 0.713 | −0.265 |
| `flu_pb2_pb1_regimes` | MLP  | 0.979 | 0.764 | −0.215 |
| `flu_pb2_pb1_regimes` | LGBM | 0.975 | 0.736 | −0.239 |

**PB2/PB1 has a smaller drop than HA/NA** (~22 pp vs ~40 pp). Consistent with PB2/PB1 being more conserved — sequence signal is more discriminative on its own, even when metadata coincidence is held constant. HA/NA is under positive selection and more variable; the model leans harder on metadata when sequence alone is ambiguous.

## Did regime-aware sampling help? (the matched-pair test)

For each (schema_pair, model) cell, compare non-regimes vs regimes:

| Comparison | Non-regimes `host_subtype_year` TNR | Regimes `host_subtype_year` TNR | Δ |
|---|---:|---:|---:|
| HA/NA, MLP    | 0.587 | 0.567 | **−0.020** |
| HA/NA, LGBM   | 0.561 | 0.549 | **−0.012** |
| PB2/PB1, MLP  | 0.764 | 0.764 | **±0.000** |
| PB2/PB1, LGBM | 0.713 | 0.736 | **+0.023** |

**Surprise: the regime-aware sampler did NOT improve hard-regime TNR in this matrix.** Three of four cells are flat or slightly *worse* under regimes; the PB2/PB1 LGBM cell improves by ~2.3 pp (likely within single-seed noise).

**Why the null result.** Two compounding effects warp the achieved distribution off the config target:

- **Coverage phase overshoots easy regimes.** Phase 4 walks every `(slot, seq_hash)` cell and grabs *any* valid negative — random partners across isolates overwhelmingly mismatch on all three axes, so coverage lands ~42% `none_match` and ~28% `year_only` "for free" on the test set, before fill phase even starts. The 10% targets for those regimes are already exceeded by coverage; the fill phase can only add on top, not subtract.
- **Supply exhaustion on the hard regime.** `host_subtype_year` test target is 2,626 negatives; only 173 valid candidates exist after `forbidden_pair_keys` blocking from train (`supply_exhausted_after_redistribute` flag in the manifest). The candidate space is genuinely small — you need two *different* isolates sharing host AND subtype AND year — and train consumes most of it.

Net effect: the regimes sampler boosted `host_subtype_year` train negatives 3.6× (HA/NA: 810 → 2,922 — see distribution table above), but absolute counts remain small (4.2% of train negs vs the 30% target). The level1_neg_regimes plots show this clearly — what looks like "widely varying numbers" is the achieved distribution being pushed by coverage overshoot and capped by supply, not the 30% / 10% targets the config requested.

**Aggregate metrics dropped slightly** on the regimes runs (e.g. HA/NA MLP F1 0.940 → 0.937). Plausible explanation: with `host_subtype_year` capped by supply, the regimes sampler shifted training distribution toward `subtype_only`, `year_only`, `host_year_only` etc. (compare regime mix tables above) — which are intermediate-difficulty negatives. The net effect on the test set, which is dominated by `none_match` (4,103 / 8,758 = 47%), is a small drag, not a win.

## What this means for the slide deck

1. **Pipeline mechanics work as designed.** Equal dataset sizes confirmed; all leakage audits pass.
2. **Mode #5 demographic-shortcut leakage is confirmed visible** — TNR drops 28–43 pp from easiest to hardest regime in every cell.
3. **Regime-aware sampling alone (at `neg_to_pos_ratio: 1.5`) does NOT close the gap** in this corpus. The hardest regime is supply-limited. To make this lever work, options are:
   - Increase `neg_to_pos_ratio` substantially (e.g. 3.0 or higher) to give the fill phase more headroom — at the cost of larger datasets and a different size class.
   - Train-time reweighting of the loss by regime, independent of sampling.
   - Adversarial / gradient-reversal training (`roadmap_v2.md` Task 12).
   - Or a stricter routing scheme (e.g. demographic-disjoint splits, similar in spirit to mmseqs2 cluster-disjoint splits).
4. **PB2/PB1 retains higher hard-regime TNR than HA/NA — but this is most plausibly cluster leakage, NOT better biology learning.** The 2026-05-11 PB2/PB1 result under `seq_disjoint` (`docs/results/2026-05-11_exp4a_seq_disjoint_results.md`) had the 1-NN cosine-margin baseline **edging the MLP on aggregate MCC (0.900 vs 0.887)**. If PB2/PB1's higher TNR came from learned biology, the MLP would beat 1-NN by a meaningful margin — it doesn't. Mechanism: PB2/PB1 is under purifying selection → fewer distinct unique proteins per slot (33,414 / 30,996 vs HA/NA's 41,629 / 37,207) → test sequences are on average closer to their nearest train neighbor → mode #4 cluster leakage. The 0.71–0.76 TNR partly reflects lookup, not representation. HA/NA's lower TNR (0.55–0.59) is consistent: HA/NA is under positive selection → more truly novel test sequences → when metadata also matches, the model has nothing to lean on. Same story as the aa-vs-nt similarity-leakage diagnostic (`docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md`).
5. **MLP vs LightGBM is bundle-stable.** MLP wins F1/MCC, LightGBM wins AUC-ROC/AUC-PR, in every cell. No model dominates across all metrics. For deployment under a single threshold, MLP is better; for ranking, LightGBM.

## Run artifacts

Dataset runs:
- `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260513_201046/`
- `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_regimes_20260513_201048/`
- `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_20260513_201050/`
- `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_regimes_20260513_201052/`

Model runs:
- `models/flu/July_2025/runs/training_flu_ha_na_20260513_201506/` (MLP)
- `models/flu/July_2025/runs/training_flu_ha_na_regimes_20260513_201508/` (MLP)
- `models/flu/July_2025/runs/training_flu_pb2_pb1_20260513_201510/` (MLP)
- `models/flu/July_2025/runs/training_flu_pb2_pb1_regimes_20260513_201512/` (MLP)
- `models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_20260513_202802/` (LGBM)
- `models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_regimes_20260513_202804/` (LGBM)
- `models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_20260513_202806/` (LGBM)
- `models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_regimes_20260513_202808/` (LGBM)

Each MLP run dir carries `best_model.pt`, `training_history.csv`, `test_predicted.csv`, `post_hoc/metrics.csv`, `post_hoc/level1_neg_regimes.csv`, plus the standard analyzer plots (`level1_neg_regimes.png`, `level2_by_subtype.png`, `confusion_matrix.png`, etc.).

Each LightGBM run dir carries `best_model.joblib`, `metrics_summary.json`, `test_predicted.csv`, and the same `post_hoc/` analyzer outputs.

## Caveats

- **Single seed (42).** Per-cell differences within ~0.01–0.02 are within seed noise. Confirming the "regimes did not help" finding would need ≥3 seeds per cell.
- **`host_subtype_year` shortfall** caps the regimes-sampler test. A version run at `neg_to_pos_ratio: 3.0` would give a fairer test of the mechanism.
- **No 1-NN baseline in this matrix.** The biology-learning criterion (`leakage_definitions.md`) needs MLP vs 1-NN comparison — outside the scope of this run.
- **Cluster leakage (mode #4) not addressed.** All four bundles use `seq_disjoint` routing which bounds the *exact-hash* case; cluster-disjoint splits via mmseqs2 still pending.
- The aggregator CSVs are in `/tmp/experiment_2026-05-13/` rather than `results/` because they're auxiliary; the canonical numeric source is the per-run `post_hoc/metrics.csv` and `post_hoc/level1_neg_regimes.csv` in each model directory.

## Related

- `docs/2026-05-13-in_depth.md` §37 — the experiment matrix specification that this doc reports on.
- `docs/methods/dataset_construction_v2_workflow.md` — phase-by-phase Stage 3 walkthrough.
- `docs/methods/leakage_definitions.md` — the 5-mode taxonomy. Mode #5 is the one this matrix sharpens.
- `docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md` — companion mode #4 diagnostic.
- `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md` — regime-aware sampler design.
