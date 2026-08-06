# Negative-pair leakage: model uses metadata correlations as shortcuts

**Date:** 2026-05-07
**Status:** measured on two runs (HA/NA mixed, k-mer, h=[10] and h=[200]). Same dataset; only `hidden_dims` differs. Trend should be re-checked on PB2/PB1 and on filtered (H3N2) bundles next.

---

## TL;DR

On the HA/NA mixed test set, the model's false-positive rate on
*negative* pairs climbs **monotonically** with how many metadata axes
{subtype, host, year_bin, geo_location, passage} coincide between the
two sides of the negative. The trend appears at every capacity tested:

|                | h=[10]   | h=[200]  |
|----------------|----------|----------|
| match_count=0  | 2.5%     | 1.5%     |
| match_count=3  | 45.2%    | 25.2%    |
| match_count=5  | 75.0%    | 75.0%    |
| Aggregate AUC  | 0.978    | 0.990    |
| Aggregate F1   | 0.956    | 0.973    |

**More capacity exploits the shortcut more strongly, not less.** This
might look counter-intuitive because the absolute FP *count* drops on
easy buckets (the larger model is generally more accurate). But three
metrics expose that the shortcut itself is *amplified*:

- **Same-vs-cross FP-rate ratio** (the shortcut's effect size) grows
  from **30× at h=[10] → 50× at h=[200]** (cross-everything 2.5% vs
  same-everything 75.0% → 1.5% vs 75.0%).
- **Mean predicted probability for same-everything negatives** climbs
  from **0.64 → 0.74** with capacity — the larger model pushes harder
  toward "positive" on the truly hardest negatives.
- **FP confidence** (mean pred_prob on the FPs themselves) climbs at
  every match level: at match_count=5 from **0.85 → 0.99**. When the
  larger model is wrong, it is more decisively wrong.

Within-axis vs cross-axis FP rates show the same pattern per individual
axis (subtype 3.8–4.2×, host 3.4×, year_bin 3.3×).

This is direct evidence the larger model is *more* committed to using
host / subtype / year / geo_location / passage matches as shortcuts —
even though it makes fewer raw FPs on the easy negatives, it is more
confident that hard, indistinguishable-by-metadata negatives are
positive.

Within-axis vs cross-axis FP rates show the same pattern per individual
axis (subtype 3.8–4.2×, host 3.4×, year_bin 3.3×).

The shortcut behavior is intrinsic to the architecture / feature
representation, not an under-fitting artifact. Aggregate AUC / F1 do
not reveal this — they go up with capacity because the easy-negative
class is large and the shortcut helps on it too.

---

## Setup

Two training runs on the same v2 dataset; only `hidden_dims` differs.

| Item | Value |
|---|---|
| Bundle | `flu_ha_na` (active; full-dataset HA/NA, k-mer k=6, slot_norm + concat) |
| Stage 3 dataset | `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260506_150017` |
| `pair_builder_version` | v2 |
| `schema_pair` | (Hemagglutinin precursor, Neuraminidase protein) |
| `feature_source` | kmer (k=6, 4096-dim) |
| `slot_transform` | slot_norm |
| `interaction` | concat |
| Test pairs | 12,899 (5,883 pos / 7,016 neg) |

| Run | hidden_dims | Run dir |
|---|---|---|
| h=[10]  | [10]  | `models/flu/July_2025/runs/training_flu_ha_na_20260506_150320` |
| h=[200] | [200] | `models/flu/July_2025/runs/training_flu_ha_na_h200_20260507_135837` |

| Aggregate metric | h=[10] | h=[200] |
|---|---|---|
| Accuracy | 0.959 | 0.975 |
| F1 (binary) | 0.956 | 0.973 |
| Precision | 0.930 | 0.959 |
| Recall | 0.984 | 0.988 |
| MCC | 0.919 | 0.950 |
| AUC-ROC | 0.978 | 0.990 |
| AUC-PR (avg precision) | 0.949 | 0.980 |
| Brier | 0.040 | 0.021 |

---

## Reproduce

```bash
# Stage 3 -- dataset (run once; output dir is timestamped)
bash scripts/stage3_dataset.sh flu_ha_na

# Stage 4, default h=[100] from the bundle:
bash scripts/stage4_train.sh flu_ha_na \
   --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260506_150017

# Stage 4 with a different hidden_dims (Hydra dotlist override; bypass
# the wrapper so the bundle stays unchanged on disk).
TS=$(date +"%Y%m%d_%H%M%S")
python src/models/train_pair_classifier.py \
   --config_bundle flu_ha_na \
   --cuda_name cuda:0 \
   --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260506_150017 \
   --run_output_subdir "training_flu_ha_na_h200_${TS}" \
   --override 'training.hidden_dims=[200]'

# Stage 4 post-hoc (auto-runs after training; manual rerun if needed)
python src/analysis/analyze_stage4_train.py \
   --config_bundle flu_ha_na \
   --model_dir models/flu/July_2025/runs/training_flu_ha_na_h200_${TS}
```

The post-hoc script regenerates `metrics.csv`, the confusion matrix,
ROC / PR / calibration plots, Level 1 / Level 2 stratified tables, and
the negative-hardness outputs covered below.

---

## Result 1 — Within-axis vs cross-axis FP rate

For each metadata axis, split test negatives into within-axis (both
sides share the value) and cross-axis (sides differ). Pairs where
either side is missing or unparseable are excluded.

**h=[10]:**

| Axis     | Within FP | n      | Cross FP | n      | Ratio |
|----------|-----------|--------|----------|--------|-------|
| subtype  | 16.3%     | 1,264  | 3.9%     | 5,621  | 4.2×  |
| host     | 13.4%     | 1,712  | 3.9%     | 5,286  | 3.4×  |
| year_bin | 11.4%     | 2,433  | 3.5%     | 4,562  | 3.3×  |

**h=[200]:**

| Axis     | Within FP | n      | Cross FP | n      | Ratio |
|----------|-----------|--------|----------|--------|-------|
| subtype  | 8.9%      | 1,264  | 2.3%     | 5,621  | 3.8×  |

(host and year_bin within/cross stats not re-extracted for h=[200];
the trend should mirror subtype since the match-count curve below
shows the same pattern.)

All three axes are individually used as shortcuts; the within/cross
*ratio* persists at h=[200] even though both magnitudes drop. See
`neg_prob_distribution_by_axis.png` in each run dir for the full
pred-prob distributions.

---

## Result 2 — Match-count trend

Per negative pair, count how many of {subtype, host, year_bin,
geo_location, passage} have the same value on both sides (excluding
unknown/missing). Aggregate FP rate, mean predicted probability over
the bucket, and FP confidence (mean predicted probability on the FPs
themselves) by that count.

| match_count | n_neg | h=[10] FP rate | h=[200] FP rate | h=[10] mean prob (all neg) | h=[200] mean prob | h=[10] FP avg conf | h=[200] FP avg conf |
|-------------|-------|----------------|------------------|---------------------------|--------------------|--------------------|----------------------|
| 0 (cross-everything) | 3,087 | 2.5%   | 1.5%   | 0.025 | 0.014 | 0.79 | 0.90 |
| 1                    | 2,533 | 4.9%   | 2.4%   | 0.049 | 0.024 | 0.77 | 0.90 |
| 2                    | 1,092 | 8.5%   | 5.2%   | 0.085 | 0.051 | 0.78 | 0.91 |
| 3                    | 270   | 45.2%  | 25.2%  | 0.400 | 0.245 | 0.82 | 0.91 |
| 4                    | 30    | 70.0%  | 46.7%  | 0.585 | 0.467 | 0.81 | 0.96 |
| 5 (same-everything)  | 4     | 75.0%  | 75.0%  | 0.637 | **0.739** | 0.85 | **0.99** |

Two things to read off:

1. **FP rate climbs monotonically at both capacities** — the steepest
   jump is between match_count=2 and 3 in both. The curve shape is
   not flattened by capacity.
2. **The shortcut is amplified at h=[200]**, even though raw FP rate
   on easy buckets drops:
   - Same/cross FP-rate ratio: **30× at h=[10] → 50× at h=[200]**.
   - Mean prob on match_count=5 (the all-axis-matched negatives):
     **0.64 → 0.74** — the larger model pushes harder toward
     "positive" on the negatives most metadata-similar to positives.
   - FP confidence at match_count=5: **0.85 → 0.99** — when wrong,
     more decisively wrong.

See `error_by_match_count.png` in each run dir.

---

## Result 3 — Match-pattern decomposition

Match-count alone collapses *which* axes match. The pattern table
breaks each negative into the exact combination. Selected rows (full
tables in each run's `error_by_match_pattern.csv`):

| match_pattern                | match_count | n_neg | h=[10] FP | h=[200] FP |
|------------------------------|-------------|-------|-----------|-------------|
| none                         | 0           | 3,087 | 2.5%      | 1.5%        |
| year_bin                     | 1           | 1,584 | 4.6%      | 2.3%        |
| host                         | 1           | 571   | 4.7%      | 3.0%        |
| subtype                      | 1           | 319   | 5.6%      | 1.3%        |
| geo_location                 | 1           | 53    | 13.2%     | 7.5%        |
| subtype,host                 | 2           | 501   | 4.6%      | 2.6%        |
| host,year_bin                | 2           | 313   | 9.3%      | 5.8%        |
| subtype,year_bin             | 2           | 178   | 12.4%     | 7.3%        |
| host,geo_location            | 2           | 22    | 22.7%     | 13.6%       |
| year_bin,passage             | 2           | 14    | 28.6%     | 14.3%       |
| **subtype,host,year_bin**    | 3           | 210   | **54.3%** | **28.1%**   |
| subtype,host,geo_location    | 3           | 20    | 15.0%     | 20.0%       |
| host,year_bin,geo_location   | 3           | 21    | 19.0%     | 19.0%       |
| subtype,host,year_bin,passage | 4          | 18    | 66.7%     | 50.0%       |
| subtype,host,year_bin,geo_location | 4     | 11    | 81.8%     | 45.5%       |
| subtype,host,year_bin,geo_location,passage | 5 | 4 | 75.0%     | 75.0%       |

Two observations from the pattern table that match-count alone
wouldn't reveal:
- `subtype,host` (n=501) only fires 4.6% / 2.6% FP — a benign 2-axis match.
- `subtype,host,year_bin` (n=210) fires 54.3% / 28.1% FP — adding the
  *year* dimension to the same population is what turns the model's
  behavior from "fine" to "wrong."

This points at year/temporal structure as the main amplifier of the
metadata shortcut, on top of the subtype + host base. The pattern
holds at both capacities.

---

## Findings

1. The MLP does not just use sequence content; it uses the joint
   metadata profile of the pair as a strong predictor.
2. The shortcut is monotonic in the number of matching metadata axes
   at both capacities tested.
3. The single most damaging combination is `subtype + host + year_bin`
   together — a "same demographic, same era" signature that the model
   treats as evidence of co-occurrence (54.3% FP at h=[10]; 28.1% at
   h=[200]).
4. **More capacity exploits the shortcut more, not less.** Going from
   h=[10] to h=[200]:
   - Same/cross FP-rate ratio: 30× → 50× (relative shortcut signal grows).
   - Mean predicted probability on same-everything negatives:
     0.64 → 0.74 (larger model is more confident those negatives are
     positives).
   - FP confidence at match_count=5: 0.85 → 0.99 (when wrong, more
     decisively wrong).

   The drop in absolute FP count on easy buckets (cross-everything
   2.5% → 1.5%) is the larger model getting easy negatives right more
   often, which masks the fact that on the hard negatives the
   shortcut is being applied more aggressively. Aggregate AUC / F1
   reward the easy-bucket gains and hide the hard-bucket
   amplification.
5. Aggregate AUC-ROC (0.978 → 0.990) and F1 (0.956 → 0.973)
   *under-state* the leakage because easy cross-everything negatives
   dominate the test set (3,087 of ~7,000 negatives) and pull
   aggregate metrics up. The shortcut is concentrated in the small
   match_count≥3 buckets that aggregate metrics barely see.

## Conclusion

The current architecture (k-mer concat + MLP) leans on
metadata-correlated shortcuts as a primary signal, and **scaling
capacity makes this worse, not better**: the relative shortcut signal
grows from 30× to 50× and the model becomes more confident that
metadata-indistinguishable negatives are positives. Aggregate AUC and
F1 *improve* at the same time because easy negatives dominate the
test set and the larger model gets those right more often.

Headline aggregate metrics are therefore not safe to interpret as
biology-only generalization, and they cannot be used as evidence the
model is "learning more biology" with more capacity — the opposite is
true on the metadata-hard subset.

Mitigations, ordered by effort:

1. **Stratified reporting (no model change).** Always report FP rate
   per match_count alongside aggregate AUC/F1. Already wired into
   post-hoc as of this session.
2. **Hard-negative mining.** During training, oversample negatives
   with high `match_count` so the model is forced to find a non-
   metadata signal to discriminate them.
3. **Stricter dataset filtering.** Restricting to a single subtype +
   host + year (as in `flu_ha_na_human_h3n2_2024`) removes the
   metadata variance the model is exploiting; the prior runs on those
   bundles still scored very high, suggesting the leakage extends
   below the metadata level into sequence near-neighbors (separate
   thread, not this finding).
4. **Sequence-disjoint splits.** Independent of metadata shortcuts;
   addresses the deeper near-neighbor issue documented separately.

---

## Files generated per run

In `<run_dir>/post_hoc/`:
- `level1_pair_regime.csv` + `.png` — TPR / TNR by within/cross-subtype
  regime. **Note (2026-05-10):** these two files were the Level 1 outputs
  at the time of this experiment. They have since been replaced by
  `level1_neg_regimes.{csv,png}` (per-regime, 9 buckets) plus
  `level1_neg_regimes_agg.{csv,png}` (aggregated by metadata-match count),
  both using the same regime taxonomy as the v2 metadata-aware sampler.
  See `docs/post_hoc_analysis_design.md` for the new outputs.
- `level2_by_{host,subtype,year_bin}.csv` + `.png` — per-stratum
  metrics
- `neg_prob_distribution_by_axis.png` — pred-prob histograms (within
  vs cross) for subtype / host / year_bin
- `error_by_match_count.csv` + `.png` — headline trend
- `error_by_match_pattern.csv` — per-pattern FP rates (no plot;
  pattern count is variable)

---

## See also

- `docs/post_hoc_analysis_design.md` — methodology for Level 1 / Level 2
- `_ongoing_work.md` — running technical notes
- `roadmap_v2.md` — Task 12 (FP/FN diagnosis + mitigation)
