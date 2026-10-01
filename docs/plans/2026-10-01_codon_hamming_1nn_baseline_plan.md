# Hamming 1-NN baseline on per-site codon features

**Status: IN PROGRESS**

Run the 1-NN baseline on the per-site codon datasets with Hamming distance over the codon codes
instead of cosine. One pilot, on PB2-HA fold 0.

All six implementation items and the fold 0 pilot are complete; the three follow-up items are not
started.

## Why

`docs/methods/leakage.md` uses the 1-NN baseline as a lookup floor. It was last run in May 2026 on
cluster-disjoint aa and nt routings, matching or beating LightGBM in all 8 pair-by-routing cells, and
never on Hopcroft-Karp positives, pinned-length CDS, or per-site codon features.

`knn1_margin` offered cosine distance only, built for k-mer counts. Per-site codon features are
nominal codes, not magnitudes, so cosine over them measures nothing meaningful; the
`CATEGORICAL_FEATURES` block in `train_pair_baselines.main` already says so and declares every
ordinal site column categorical for LightGBM. Hamming only tests
whether two codes are equal, so it reads them the same way. One-hot plus cosine gives identical
distances, verified to 1.8e-6, but needs 81,835 columns against 1,327.

Codon agreement is not nucleotide identity, since one disagreeing codon can hold up to three
disagreeing bases, nor alignment identity with coverage, since the sites are pinned and compared
position by position. It pools both slots, so high agreement on one can hide low agreement on the
other.

## Notation

For a test pair `q` and a training pair `t`, over the 1,327 concatenated codon sites of PB2-HA:

- `D(q, t)` = disagreeing codon sites / 1,327. Codon agreement is `1 - D`.
- `A_pair = (760 * A_pb2 + 567 * A_ha) / 1327`, the length-weighted mean of the per-slot agreements
  `A_pb2` and `A_ha`. It equals `1 - D`.
- `d_pos = min D(q, t)` over positive training pairs, `d_neg = min` over negative ones. These are the
  two distances `_distances_pos_neg` returns.
- `k_pos = round(1327 * d_pos)` and `k_neg = round(1327 * d_neg)`, the integer counts of disagreeing
  sites. Rounded because sklearn's fraction drifts off the lattice. Every Hamming comparison uses
  these counts, never the floats.
- `C(q)`, the co-nearest set, is every `t` whose disagreeing-site count is at the minimum. That
  minimum equals `min(k_pos, k_neg)`, so when `k_pos < k_neg` every member of `C(q)` is positive.
- Hard label transfer predicts 1 when `k_pos < k_neg`, and 0 otherwise, which covers both a
  nearer negative and a tie.
- The distance margin is `m = (k_neg - k_pos) / 1327`. `predict_proba` returns `(m + 2) / 4`, which is
  exactly 0.5 on a count tie. `m` lies in [-1, 1], so probabilities fall in [0.25, 0.75]; AUC is
  rank-invariant, so the map does not affect it.
- For any binary prediction, `AUC = (sensitivity + specificity) / 2 = balanced accuracy`, because a
  two-valued score puts a single interior point on the ROC curve. The two hard-decision rows below
  therefore report one number twice.

## Preliminary results, PB2-HA fold 0

Direct calculation outside the harness, recorded so the pilot has something to reproduce. Dataset
`data/datasets/flu/July_2025/runs/exp3_28p_codon_pb2_ha/fold_0`: train 2,652 rows (1,326 / 1,326),
test 1,022 rows (511 / 511). Three separate results plus a sensitivity reading of the second, which
must not be merged into one number.

| result | F1 | F1 macro | balanced acc | AUC |
|---|---|---|---|---|
| LightGBM, from `models/flu/July_2025/runs/lgbm_exp3_28p_codon_pb2_ha_fold0` | 0.9104 | 0.9047 | 0.9051 | 0.9458 |
| hard label transfer, `k_pos < k_neg`, ties to class 0 | 0.8479 | 0.8433 | 0.8434 | 0.8434 |
| the same decision with ties to class 1, as a sensitivity check | 0.8326 | 0.7971 | 0.8033 | 0.8033 |
| distance-margin score, the continuous `m` ranking | — | — | — | 0.9219 |

LightGBM's balanced accuracy came from the `label` and `pred_label` columns of its
`test_predicted.csv`, since `metrics_summary.json` does not carry it. The margin AUC is the integer
form's, which item 4 adopts. The two baselines come from one search but are different estimators: the
margin uses both distances, so it ranks more finely, while hard label transfer copies a neighbour's
label only when the nearer class is unambiguous.

Counts on the 1,022 test rows:

- 149 rows have `k_pos == k_neg`, called cross-class ties here; their true labels are 54 positive and
  95 negative.
- The other 873 are called unambiguous-decision rows, not unique-neighbour rows: 398 of the 1,022
  rows hold more than one pair in `C(q)`.
- Nearest-training codon agreement runs 0.7943 to 0.9985, median 0.9947, 42 rows below 0.99 and 6
  below 0.98. No test row has an exact training match.

## What these numbers do not say

They do not establish how LightGBM works. A small gap would show a simple neighbour baseline scoring
similarly on the folds measured, not that LightGBM performs lookup. The ΔAUC < 0.02 bar under
"The 1-NN lookup gauge" in `docs/methods/leakage.md` does not apply here; it was set for an MLP
against a cosine 1-NN in the same feature space. And nearly every test row has a training pair
above 0.99 codon agreement, so a result here describes lookup under near-clone conditions, not
performance on distant sequences.

Where the gap between LightGBM and hard label transfer comes from is unknown, and these figures
cannot answer it: LightGBM's cover all 1,022 rows and cannot be compared against a subset. Until both
models are scored on the same subsets, do not attribute the gap to the coarseness of Hamming.

## Implementation

All six items land before the pilot runs.

1. **Configurable metric.** `KNN1Margin.fit` passed `metric='cosine'` to both searches with no way
   to change it. Add a `metric` argument to `KNN1Margin.__init__`, validate it against `METRICS`,
   read it in `get_estimator`, default it to `cosine` so existing k-mer and ESM-2 runs are
   unchanged, add it to the module docstring's override table, and add `metric: cosine` to the
   `baseline_knn1_margin` block in `conf/baselines/default.yaml`.
   `feature_scaling` stays `none`: StandardScaler is injective per column, so it would leave Hamming
   distances unchanged at no benefit.


2. **Bounded-metric margin.** Record the margin bullet above in `knn1_margin`'s module docstring,
   including that a count tie maps to exactly 0.5. No code change beyond item 4.

3. **Leave-one-out by training row index.** `_distances_pos_neg` skips any neighbour closer than
   `LOO_EPS` as a self-match, which discards two things it should keep: another training row at
   distance 0, and an exact validation or test match the audit should report. Fix: `fit` stores
   `np.flatnonzero(pos_mask)` and `np.flatnonzero(~pos_mask)` to map a subset index back to a
   training row index, since `nn_pos_` and `nn_neg_` are fitted on subsets of `X`; add a method for
   scoring training rows that excludes only the query's own index; leave
   `predict` and `predict_proba` excluding nothing; drop the `LOO_EPS` test.
   `train_pair_baselines._run_one_baseline` scores all three splits, so it detects an estimator that
   accepts `train_rows` and passes the training row indices for that split alone.

   No distance-0 case arises on PB2-HA fold 0, in either direction. It is reachable elsewhere:
   `flu_ha_na_h3n2_2024_random_cv4_pinned_length_hopcroft_karp_site_aa.yaml` sets `site.unit: aa` and
   inherits `pair_key_alphabet: nt_cds`, so two pairs with different CDS can translate to the same
   amino-acid pair and share a feature vector.

4. **Integer counts for every Hamming comparison.** Under `metric='hamming'`, derive `k_pos` and
   `k_neg` as `round(distance * n_features_in_)`, decide the hard label on `k_pos < k_neg`, and
   compute the margin as `(k_neg - k_pos) / n_features_in_`. A count tie then yields a margin of
   exactly 0, so `predict_proba` returns exactly 0.5 and `compute_pair_metrics`'s strict
   `y_probs > threshold` assigns class 0 — the stated rule holding by construction rather than by
   luck. Leave cosine on the float path, which has no integer lattice. It matters because sklearn's
   fraction drifts off the lattice by up to 3.05e-05 here and the two distances come from separately
   fitted searches, so equal counts can give unequal floats on another fold; on this fold all 149
   count-ties happen to be float-equal.

   The form also sets the margin AUC: integer counts give 0.921929 over 39 distinct values, today's
   `(1 - d_pos) - (1 - d_neg)` gives 0.922494 over 60, and `d_neg - d_pos` gives 0.922082 over 93.
   On PB2-HA fold 0 the hard decision is identical under all three. Report the tie count and the
   ties-to-1 metrics as the table above does. Do not drop tied rows, which would change the
   evaluation population, and do not break the tie with the second neighbour, which would change the
   baseline.

5. **Neighbour export.** The prediction path produces probabilities only, so this needs its own
   write path, `train_pair_baselines.write_neighbor_report`, fed by `KNN1Margin.co_nearest`. It
   writes `neighbors_<split>.csv` beside each `<split>_predicted.csv`, one row per scored pair, with
   these columns.

   | column | meaning |
   |---|---|
   | `pair_key`, `label` | the scored pair and its true label |
   | `distance` | `D(q, t)` at the minimum, the metric's own value |
   | `n_co_nearest` | size of `C(q)` |
   | `n_co_nearest_pos`, `n_co_nearest_neg` | its class split |
   | `representative_pair_key` | the member with the lexicographically smallest `pair_key`, following `create_positive_pairs_v2`'s representative-isolate rule |
   | `representative_label` | that member's label |
   | `mismatch_count` | `k(q, t)` at the minimum; `hamming` only |
   | `agreement_pooled` | `A_pair`, equal to `1 - distance`; `hamming` only |
   | `agreement_slot_a`, `agreement_slot_b` | `A_pb2` and `A_ha` against the representative; per-site ordinal features only |

   `distance` and `mismatch_count` are separate columns because they are different quantities: a
   Hamming distance of 6 sites is `6 / 1327`, not 6. Reporting one number under one name would
   invite reading a count as a distance.

   These are audit fields, not the prediction. On an unambiguous-decision row every member of `C(q)`
   carries the nearer class, so the representative's label equals the prediction; on a cross-class tie
   `C(q)` holds both classes, the rule predicts 0, and the representative's label may be 1. The
   representative is also one choice among equals whenever `C(q)` holds more than one pair, which its
   size makes visible.

6. **Separate output directories.** One per metric, so a cosine run cannot overwrite a Hamming run.

## Pilot

PB2-HA fold 0 only. `--skip_post_hoc` because the population is one host, one subtype and one year,
which leaves `analyze_stage4_train.py`'s metadata strata degenerate.

```
python src/models/train_pair_baselines.py \
  --config_bundle flu_28p_codon_pb2_ha \
  --baseline knn1_margin \
  --dataset_dir data/datasets/flu/July_2025/runs/exp3_28p_codon_pb2_ha/fold_0 \
  --run_output_subdir knn1_margin_hamming_exp3_28p_codon_pb2_ha_fold0 \
  --override baseline_knn1_margin.metric=hamming \
  --skip_post_hoc
```

1. **Harness reproduction.** `metrics_summary.json` test F1 0.8479, F1 macro 0.8433, and `auc_roc`
   0.9219, the last being the margin AUC under the integer form. That file carries no balanced
   accuracy, so compute it from the `label` and `pred_label` columns of `test_predicted.csv` and
   check it against 0.8434; by the identity above that also checks AUC of the hard labels.
2. **Metric is threaded.** Re-run with `metric=cosine` into its own directory; the two must differ.
3. **Count tie behaviour.** 149 rows of `test_predicted.csv` must carry `pred_prob` exactly 0.5, and
   every one of them `pred_label` 0. All `pred_prob` must fall inside [0.25, 0.75].
4. **Determinism.** `get_estimator` ignores `random_state`, so two identical runs must give identical
   predictions.

## Follow-up, in order

1. Score LightGBM and hard label transfer on the same 873 unambiguous-decision rows, then on the same
   149 cross-class-tie rows.
2. Bin both models on nearest-training codon agreement, pooling the 4 folds for roughly 170 rows
   below 0.99. Report the count in every bin.
3. Extend to the remaining 27 pairs, repeating the distinct-feature-vector check per pair rather than
   assuming PB2-HA fold 0 generalises.

Optional and exploratory, outside the work above: the positive fraction within `C(q)` is a third
estimator the export makes available at no extra cost. How it ranks against the other two has not
been measured. If tried, name and report it separately from both.
