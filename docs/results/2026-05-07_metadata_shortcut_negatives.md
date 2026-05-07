# Negative-pair leakage: model uses metadata correlations as shortcuts

**Date:** 2026-05-07
**Status:** measured on one run (HA/NA mixed, k-mer, h=[10]); trend should be re-checked on other runs.

---

## TL;DR

On the HA/NA mixed test set, the model's false-positive rate on
*negative* pairs climbs from **2.5%** (no metadata axes match between A
and B) to **45.2%** (3 axes match) to **75.0%** (all 5 axes match) —
an 18× increase that tracks how many metadata fields coincide between
the two sides of the negative pair.

Within-axis vs cross-axis FP rates show the same pattern per individual
axis (subtype 4.2×, host 3.4×, year_bin 3.3×).

This is direct evidence the model uses host / subtype / year /
geo_location / passage matches between the pair's two halves as
shortcuts to predict "same isolate," instead of learning a purely
sequence-based co-occurrence signal.

---

## Setup

| Item | Value |
|---|---|
| Bundle | `flu_ha_na` (active; full-dataset HA/NA, k-mer k=6, slot_norm + concat, h=[10]) |
| Stage 3 dataset | `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260506_150017` |
| Stage 4 training run | `models/flu/July_2025/runs/training_flu_ha_na_20260506_150320` |
| `pair_builder_version` | v2 |
| `schema_pair` | (Hemagglutinin precursor, Neuraminidase protein) |
| `feature_source` | kmer (k=6, 4096-dim) |
| `slot_transform` | slot_norm |
| `interaction` | concat |
| `hidden_dims` | [10] |
| Test pairs | 12,899 (5,883 pos / 7,016 neg) |
| Test AUC-ROC | 0.978 |
| Test F1 | 0.956 |
| Analysis tools introduced this session | commit 4161ce6 and one follow-up (negative-hardness + per-axis prob distribution) |

---

## Reproduce

```bash
# Stage 3 -- dataset (run once; output dir is timestamped)
bash scripts/stage3_dataset.sh flu_ha_na

# Stage 4 -- training (point at the Stage 3 dataset built above)
bash scripts/stage4_train.sh \
   --config_bundle flu_ha_na \
   --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260506_150017

# Stage 4 post-hoc (generates everything below into <run>/post_hoc/)
python src/analysis/analyze_stage4_train.py \
   --config_bundle flu_ha_na \
   --model_dir models/flu/July_2025/runs/training_flu_ha_na_20260506_150320
```

The post-hoc script regenerates `metrics.csv`, the confusion matrix,
ROC / PR / calibration plots, Level 1 / Level 2 stratified tables, and
the negative-hardness outputs covered below.

---

## Result 1 — Within-axis vs cross-axis FP rate

For each metadata axis, split test negatives into within-axis (both
sides share the value) and cross-axis (sides differ). Pairs where
either side is missing or unparseable are excluded.

| Axis     | Within FP | n      | Cross FP | n      | Ratio |
|----------|-----------|--------|----------|--------|-------|
| subtype  | 16.3%     | 1,264  | 3.9%     | 5,621  | 4.2×  |
| host     | 13.4%     | 1,712  | 3.9%     | 5,286  | 3.4×  |
| year_bin | 11.4%     | 2,433  | 3.5%     | 4,562  | 3.3×  |

All three axes are individually used as shortcuts. See
`neg_prob_distribution_by_axis.png` for the full pred-prob distributions
that produced these numbers.

---

## Result 2 — Match-count trend

Per negative pair, count how many of {subtype, host, year_bin,
geo_location, passage} have the same value on both sides (excluding
unknown/missing). Aggregate FP rate by that count.

| match_count | n_neg  | n_FP | FP rate | mean pred_prob |
|-------------|--------|------|---------|----------------|
| 0 (cross-everything) | 3,087  | 76   | **2.5%**  | 0.025 |
| 1                    | 2,533  | 124  | 4.9%      | 0.049 |
| 2                    | 1,092  | 93   | 8.5%      | 0.085 |
| 3                    | 270    | 122  | **45.2%** | 0.400 |
| 4                    | 30     | 21   | 70.0%     | 0.585 |
| 5 (same-everything)  | 4      | 3    | **75.0%** | 0.637 |

The rate climbs monotonically; the steepest jump is between 2 and 3
axes matching (8.5% → 45.2%). See `error_by_match_count.png`.

---

## Result 3 — Match-pattern decomposition

Match-count alone collapses *which* axes match. The pattern table
breaks each negative into the exact combination. Selected rows (full
table in `error_by_match_pattern.csv`):

| match_pattern             | match_count | n_neg | FP rate |
|---------------------------|-------------|-------|---------|
| none                      | 0           | 3,087 | 2.5%    |
| year_bin                  | 1           | 1,584 | 4.6%    |
| host                      | 1           | 571   | 4.7%    |
| subtype                   | 1           | 319   | 5.6%    |
| geo_location              | 1           | 53    | 13.2%   |
| subtype,host              | 2           | 501   | 4.6%    |
| host,year_bin             | 2           | 313   | 9.3%    |
| subtype,year_bin          | 2           | 178   | 12.4%   |
| host,geo_location         | 2           | 22    | 22.7%   |
| year_bin,passage          | 2           | 14    | 28.6%   |
| **subtype,host,year_bin** | 3           | 210   | **54.3%** |
| subtype,host,year_bin,passage | 4       | 18    | 66.7%   |
| subtype,host,year_bin,geo_location | 4  | 11    | 81.8%   |
| subtype,host,year_bin,geo_location,passage | 5 | 4 | 75.0% |

Two observations from the pattern table that match-count alone
wouldn't reveal:
- `subtype,host` (n=501) only fires 4.6% FP — a benign 2-axis match.
- `subtype,host,year_bin` (n=210) fires 54.3% FP — adding the
  *year* dimension to the same population is what turns the model's
  behavior from "fine" to "wrong."

This points at year/temporal structure as the main amplifier of the
metadata shortcut, on top of the subtype + host base.

---

## Findings

1. The MLP does not just use sequence content; it uses the joint
   metadata profile of the pair as a strong predictor.
2. The shortcut is monotonic in the number of matching metadata axes.
3. The single most damaging combination on this run is
   `subtype + host + year_bin` together — a "same demographic, same
   era" signature that the model treats as evidence of co-occurrence.
4. AUC-ROC 0.978 and F1 0.956 *under-state* the leakage because the
   easy cross-everything negatives dominate the test set (3,087 of
   ~7,000 negatives) and pull aggregate metrics up.

## Conclusion

The current architecture (k-mer concat + MLP) leans on
metadata-correlated shortcuts as a primary signal. Headline aggregate
metrics are not safe to interpret as biology-only generalization.

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
  regime
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
