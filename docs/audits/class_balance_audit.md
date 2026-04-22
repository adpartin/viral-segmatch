# Class-Balance Audit — Task 11 Sweeps (val_unfilt, val_h3n2)

**Date:** 2026-04-22
**Scope:** 28 pair bundles × 12 CV folds × 2 sweeps (val_unfilt, val_h3n2).
Dataset generation is feature-source-agnostic, so `val_unfilt_esm2` and
`val_h3n2_esm2` share datasets with `val_unfilt` and `val_h3n2` respectively
(verified: no separate `*_esm2` dataset dirs exist).

## Finding — TL;DR

**Post-filter class imbalance in val/test is material and systematic**, driven
entirely by the cross-split `pair_key` overlap removal step. Train remains a
pristine 1:1 (ratio = 1.000 exactly, std = 0.0, every pair × every fold). Val
and test have a `pos:neg` ratio range of:

- val_unfilt: val **0.279 – 0.508**, test **0.283 – 0.507** (median ~0.43)
- val_h3n2:   val **0.193 – 0.506**, test **0.199 – 0.508** (median ~0.42)

Filtering removes **30–67% of pre-filter val/test pairs** (pair_key collisions
across splits), and the removal is asymmetric — positives collide far more
often than negatives because many positive sequence pairs recur across isolates
(conserved segments), while negatives are re-sampled and blocked from
co-occurring sequence pairs at generation time.

**In every pair × both sweeps, the val/test mean ratio deviates from 1.00 by
more than 0.49 on average**, which far exceeds the 0.05 threshold flagged in
the task brief. Not a single pair × fold achieves post-filter balance.

## Methodology

- For each pair × fold, load `dataset_stats.json` (post-filter pos/neg per
  split) and `duplicate_stats.json` (pre-filter counts, pair_key overlap
  counts, rejection stats).
- Define ratio = positive_pairs / negative_pairs (so 1.00 = balanced;
  0.5 ≈ 1:2 pos:neg; 0.2 ≈ 1:5 pos:neg).
- Aggregate per pair across 12 folds: mean, std.
- Spot-check CSV vs stats for fold_0 of one pair (`ha_na`, `val_unfilt`):
  counts match exactly (train 88,632/88,632; val 5,029/10,042;
  test 4,181/8,338).

Generating configuration (relevant):
- `neg_to_pos_ratio: 1.0` (default, confirmed in `conf/dataset/default.yaml`
  and `conf/bundles/flu.yaml`) — generation targets 1:1.
- `pair_key` = canonical hash of the two protein sequences. Positives from
  isolate A can share pair_key with positives from isolate B if both isolates
  carry identical HA and NA sequences (common for conserved segments).
- The pair_key removal step (`split_dataset` in `dataset_segment_pairs.py`)
  drops any val/test pair whose pair_key appears in any other split. This
  runs *after* pair generation and hits post-sampling populations.

## Summary tables

Full per-pair tables saved as:

- `docs/audits/class_balance_val_unfilt.csv` — 28 rows, aggregated over 12 folds
- `docs/audits/class_balance_val_h3n2.csv` — same
- `docs/audits/class_balance_val_unfilt_per_fold.csv` — per-fold raw
  (28 × 12 = 336 rows)
- `docs/audits/class_balance_val_h3n2_per_fold.csv` — same

### val_unfilt — 28 pairs, mean ± std across 12 folds

All train ratios = 1.000 ± 0.000 (omitted from table).

| Pair | val ratio | test ratio | val removed % | test removed % | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| ha_m1 | 0.464 ± 0.005 | 0.471 ± 0.009 | 46.3 | 46.9 | 0.978 | 0.995 |
| ha_na | 0.494 ± 0.004 | 0.495 ± 0.007 | 30.7 | 30.9 | 0.974 | 0.993 |
| ha_np | 0.464 ± 0.006 | 0.466 ± 0.006 | 38.2 | 38.7 | 0.972 | 0.994 |
| ha_ns1 | 0.465 ± 0.004 | 0.468 ± 0.006 | 36.0 | 36.3 | 0.968 | 0.993 |
| m1_ns1 | 0.307 ± 0.004 | 0.315 ± 0.005 | 59.9 | 60.8 | 0.966 | 0.995 |
| na_m1 | 0.424 ± 0.008 | 0.429 ± 0.008 | 49.3 | 50.1 | 0.974 | 0.995 |
| na_ns1 | 0.433 ± 0.005 | 0.435 ± 0.007 | 38.0 | 38.4 | 0.966 | 0.993 |
| np_m1 | 0.279 ± 0.004 | 0.283 ± 0.007 | 63.0 | 63.9 | 0.962 | 0.995 |
| np_na | 0.430 ± 0.005 | 0.432 ± 0.007 | 40.4 | 41.0 | 0.967 | 0.993 |
| np_ns1 | 0.350 ± 0.004 | 0.352 ± 0.005 | 48.8 | 49.4 | 0.964 | 0.994 |
| pa_ha | 0.506 ± 0.004 | 0.507 ± 0.007 | 31.6 | 31.8 | 0.972 | 0.994 |
| pa_m1 | 0.401 ± 0.007 | 0.404 ± 0.007 | 51.2 | 51.9 | 0.975 | 0.996 |
| pa_na | 0.477 ± 0.005 | 0.477 ± 0.005 | 33.3 | 33.6 | 0.971 | 0.994 |
| pa_np | 0.419 ± 0.005 | 0.420 ± 0.005 | 42.1 | 42.6 | 0.971 | 0.995 |
| pa_ns1 | 0.429 ± 0.004 | 0.429 ± 0.004 | 39.5 | 39.9 | 0.969 | 0.994 |
| pb1_ha | 0.503 ± 0.003 | 0.505 ± 0.005 | 32.6 | 32.9 | 0.971 | 0.993 |
| pb1_m1 | 0.378 ± 0.003 | 0.383 ± 0.007 | 53.0 | 53.6 | 0.972 | 0.996 |
| pb1_na | 0.476 ± 0.005 | 0.475 ± 0.006 | 34.4 | 34.6 | 0.970 | 0.993 |
| pb1_np | 0.407 ± 0.005 | 0.408 ± 0.006 | 43.5 | 44.0 | 0.970 | 0.995 |
| pb1_ns1 | 0.417 ± 0.004 | 0.418 ± 0.005 | 41.0 | 41.4 | 0.968 | 0.994 |
| pb1_pa | 0.468 ± 0.004 | 0.468 ± 0.005 | 35.8 | 36.2 | 0.972 | 0.995 |
| pb2_ha | 0.508 ± 0.004 | 0.507 ± 0.006 | 31.6 | 31.8 | 0.972 | 0.993 |
| pb2_m1 | 0.403 ± 0.004 | 0.405 ± 0.004 | 51.3 | 52.1 | 0.974 | 0.996 |
| pb2_na | 0.480 ± 0.007 | 0.479 ± 0.007 | 33.3 | 33.6 | 0.970 | 0.993 |
| pb2_np | 0.419 ± 0.004 | 0.417 ± 0.005 | 42.1 | 42.8 | 0.971 | 0.995 |
| pb2_ns1 | 0.431 ± 0.003 | 0.430 ± 0.005 | 39.6 | 40.1 | 0.970 | 0.994 |
| pb2_pa | 0.477 ± 0.006 | 0.473 ± 0.005 | 34.7 | 35.1 | 0.974 | 0.994 |
| pb2_pb1 | 0.467 ± 0.004 | 0.467 ± 0.006 | 35.7 | 36.2 | 0.973 | 0.995 |

**Every val and test mean ratio deviates from 1.00 by more than 0.05.** The
two columns to flag are every pair × val and every pair × test.

### val_h3n2 — worse imbalance on filtered subset

| Pair | val ratio | test ratio | val removed % | test removed % | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| ha_m1 | 0.416 ± 0.014 | 0.421 ± 0.009 | 50.0 | 50.4 | 0.970 | 0.995 |
| ha_na | 0.497 ± 0.008 | 0.495 ± 0.009 | 35.1 | 35.5 | 0.975 | 0.994 |
| ha_np | 0.456 ± 0.010 | 0.460 ± 0.008 | 45.6 | 46.0 | 0.972 | 0.994 |
| ha_ns1 | 0.461 ± 0.007 | 0.465 ± 0.014 | 41.8 | 42.3 | 0.969 | 0.993 |
| m1_ns1 | 0.246 ± 0.005 | 0.254 ± 0.012 | 63.2 | 63.8 | 0.946 | 0.993 |
| na_m1 | 0.370 ± 0.008 | 0.372 ± 0.012 | 53.9 | 54.4 | 0.962 | 0.994 |
| na_ns1 | 0.427 ± 0.007 | 0.431 ± 0.008 | 44.8 | 45.6 | 0.959 | 0.991 |
| **np_m1** | **0.193 ± 0.006** | **0.199 ± 0.009** | **67.1** | **67.7** | **0.931** | 0.993 |
| np_na | 0.421 ± 0.011 | 0.420 ± 0.009 | 49.1 | 49.5 | 0.961 | 0.992 |
| np_ns1 | 0.320 ± 0.011 | 0.324 ± 0.009 | 57.4 | 58.0 | 0.954 | 0.992 |
| pa_ha | 0.498 ± 0.008 | 0.505 ± 0.009 | 37.7 | 38.1 | 0.972 | 0.994 |
| pa_m1 | 0.308 ± 0.006 | 0.314 ± 0.009 | 57.0 | 57.2 | 0.956 | 0.994 |
| pa_na | 0.471 ± 0.007 | 0.473 ± 0.012 | 40.6 | 41.0 | 0.966 | 0.992 |
| pa_np | 0.376 ± 0.009 | 0.381 ± 0.011 | 51.6 | 51.9 | 0.966 | 0.993 |
| pa_ns1 | 0.402 ± 0.007 | 0.409 ± 0.008 | 47.7 | 48.2 | 0.963 | 0.992 |
| pb1_ha | 0.506 ± 0.007 | 0.508 ± 0.008 | 38.0 | 38.3 | 0.971 | 0.993 |
| pb1_m1 | 0.299 ± 0.007 | 0.306 ± 0.009 | 57.5 | 58.0 | 0.954 | 0.994 |
| pb1_na | 0.477 ± 0.009 | 0.481 ± 0.011 | 40.7 | 41.2 | 0.969 | 0.993 |
| pb1_np | 0.374 ± 0.007 | 0.378 ± 0.009 | 51.9 | 52.6 | 0.965 | 0.994 |
| pb1_ns1 | 0.401 ± 0.007 | 0.408 ± 0.009 | 48.1 | 48.6 | 0.963 | 0.992 |
| pb1_pa | 0.440 ± 0.006 | 0.447 ± 0.012 | 43.8 | 44.1 | 0.969 | 0.993 |
| pb2_ha | 0.505 ± 0.009 | 0.506 ± 0.010 | 37.8 | 38.2 | 0.973 | 0.994 |
| pb2_m1 | 0.316 ± 0.008 | 0.315 ± 0.010 | 57.5 | 58.0 | 0.955 | 0.994 |
| pb2_na | 0.479 ± 0.006 | 0.478 ± 0.007 | 40.4 | 41.1 | 0.966 | 0.992 |
| pb2_np | 0.378 ± 0.009 | 0.376 ± 0.010 | 52.0 | 52.7 | 0.962 | 0.992 |
| pb2_ns1 | 0.404 ± 0.008 | 0.407 ± 0.007 | 47.9 | 48.5 | 0.960 | 0.992 |
| pb2_pa | 0.447 ± 0.010 | 0.448 ± 0.008 | 43.4 | 43.9 | 0.970 | 0.993 |
| pb2_pb1 | 0.447 ± 0.010 | 0.448 ± 0.007 | 43.4 | 44.1 | 0.968 | 0.993 |

**`np_m1` is the most extreme case**: val pos:neg = 1 : 5.2 (193 pos per 1000
neg), 67% of pre-filter pairs removed, F1 = 0.931 while AUC = 0.993.

### Rejection breakdown (val_unfilt, `ha_na` fold_0, representative)

From `duplicate_stats.json`:

- `pair_key_overlaps.val_pairs_before_removal` = 21,706
- `pair_key_overlaps.val_pairs_after_removal`  = 15,071 (30.6% removed)
- `pair_key_overlaps.train_val.count` = 3,744 (17.2% of val)
- `pair_key_overlaps.val_test.count`  = 1,271 (7.0% of test)
- Negative-sampling rejections per split are small (< 6K out of 95K attempts
  for train); no `same_func_limit` or `missing_*_func` rejections in this
  sweep (pair_mode=schema_ordered with all isolates pre-filtered by function
  availability at isolate assignment stage).

So the imbalance source is **almost entirely** the cross-split pair_key
overlap removal, not intra-sampling rejection.

## Does imbalance explain F1 variance across pairs?

Cross-reference post-filter val_ratio against sweep F1 and AUC.

**val_unfilt:** Pearson correlations across 28 pairs:

- r(val_ratio, F1)         = **0.59**  (p = 0.0009)
- r(val_ratio, precision)  = 0.26   (p = 0.18)
- r(val_ratio, recall)     = 0.88   (p < 1e-9)
- r(val_ratio, AUC-ROC)    = **−0.59** (p = 0.0009) — note sign

The negative F1 ↔ ratio correlation in val_unfilt is driven by a confound:
M1-containing pairs (most neg-heavy) are also the *biologically easiest* (M1
is the most conserved segment), so they get high AUC despite low F1. Imbalance
and task difficulty are entangled. Don't over-interpret this sign.

**val_h3n2:** cleaner signal, because H3N2 filtering blows up the imbalance
without materially changing task difficulty for most pairs:

- r(val_ratio, F1)         = **0.92**  (p < 1e-11) — very strong
- r(val_ratio, precision)  = **0.96**  (p < 1e-15) — near-perfect
- r(val_ratio, recall)     = 0.63   (p = 0.0003)
- r(val_ratio, AUC-ROC)    = −0.05  (p = 0.80)   — none

That's the textbook behavior: AUC-ROC is threshold-free and class-prior
invariant, so it doesn't track the ratio. F1 at threshold=0.5 is sensitive
to class prior, so it tracks almost perfectly. **In val_h3n2, class imbalance
explains ~85% of the between-pair variance in F1.**

Concretely: most-balanced pair (`pb1_ha`, val_ratio 0.506) reports F1 = 0.971;
most-imbalanced pair (`np_m1`, val_ratio 0.193) reports F1 = 0.931. The F1 gap
between the "easiest" and "hardest" pair in val_h3n2 is ~4 points; the AUC gap
for the same two pairs is ~0.1 points (0.993 vs 0.993). **Most of that F1 gap
reflects class prior, not model quality.**

## Implications for paper methods reporting

**This is material, not cosmetic.** Three concrete recommendations for
paper-ready reporting:

1. **Lead with AUC-ROC** in cross-pair comparisons. AUC is the imbalance-robust
   summary metric; it's what the 8×8 heatmap should report as the primary cell.
   The flatness of the AUC values across pairs (0.991–0.996) is a real
   result, and it's obscured by the much-noisier F1 heatmap.

2. **Report F1 alongside the post-filter positive rate per pair**, or
   explicitly state that F1 is computed at threshold 0.5 against a pair-specific
   class prior that ranges from 1:2 to 1:5. Otherwise readers will attribute
   the F1 spread to model behavior rather than to filtering.

3. **Consider reporting balanced F1** (or precision@recall=0.95, or PR-AUC)
   for cross-pair comparison in the paper's main tables. The audit CSVs are
   sufficient to recompute any of these from the stored per-fold predictions
   without re-running training.

Optional future work (out of scope for this audit):

- Investigate why conserved segments (M1, NS1, NP) collide so heavily in
  `pair_key`. If many isolates carry identical M1 sequences, the pair_key
  removal is acting as an inadvertent near-duplicate filter — arguably doing
  the right thing scientifically, but at the cost of skewing the class prior.
- Consider whether to target a 1:1 post-filter ratio by oversampling negatives
  during generation (e.g., `neg_to_pos_ratio` > 1, then truncate post-filter).
  This would preserve the dedup guarantee while restoring balance.

## Files

- `docs/audits/class_balance_val_unfilt.csv` — per-pair summary (28 rows)
- `docs/audits/class_balance_val_h3n2.csv` — per-pair summary
- `docs/audits/class_balance_val_unfilt_per_fold.csv` — raw (336 rows)
- `docs/audits/class_balance_val_h3n2_per_fold.csv` — raw
- Source script (not committed, one-shot): `/tmp/class_balance_audit.py`
