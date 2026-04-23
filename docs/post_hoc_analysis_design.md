# Post-hoc Analysis Design

**Status:** DRAFT (initial design for stratified evaluation extension)

This document describes the post-hoc analyses run on a single trained pair classifier
(one `training_flu_28p_*_fold*_*` run). It replaces nothing — it consolidates the
existing analyses in `src/analysis/analyze_stage4_train.py` and extends them with
**stratified evaluation** (Level 1 and Level 2) for subtype / host / year axes.

Scope boundary: this is **evaluation-time** stratification of already-trained
all-pairs models. Training-time stratified datasets (filter train to a single
stratum, or train-on-X / test-on-Y) are a separate topic covered by
`docs/STRATIFIED_EXPERIMENTS_PLAN.md`.

---

## Inputs

Each training run directory contains:

- `test_predicted.csv` — one row per test pair. Columns:
  `pair_key, assembly_id_a, assembly_id_b, brc_a, brc_b, ctg_a, ctg_b, seq_a, seq_b,
  seg_a, seg_b, func_a, func_b, seq_hash_a, seq_hash_b, label, pred_label, pred_prob, pred_logit`
- `training_info.json` — includes `optimal_threshold` (0.5 across the pipeline as of
  2026-04; `threshold_metric: null` for all 28p sweeps).
- Companion dataset dir contains `isolate_metadata.csv` with columns:
  `assembly_id, host, hn_subtype, year, geo_location_clean, passage`
  (no `isolation_year`, no `region`; `year` is float64 with ~0.16% NaN on val_unfilt;
  `host` has ~0.15% NaN; `hn_subtype` has 0 NaN but ~1% malformed — see below).

### Threshold handling

`pred_label` in `test_predicted.csv` is already computed with the run's
`optimal_threshold` (0.5 everywhere today). All stratified metrics use
`y_pred = df['pred_label']` directly. **Note:** if the pipeline later adopts a
non-0.5 threshold (e.g., F1-optimal on val), the upstream code will re-derive
`pred_label`; the post-hoc functions stay threshold-agnostic because they never
re-threshold.

---

## Metadata enrichment (fix + extension)

Current code in `analyze_errors_by_metadata` (line 548) joins `isolate_metadata.csv`
**only on `assembly_id_a`**, silently dropping side-B metadata. For any pair
where the two isolates differ in host/subtype/year, the current analysis uses
side-A alone — this is wrong for stratified evaluation of cross-stratum negatives.

**Fix:** merge metadata twice, producing per-side columns:

```
df = df.merge(meta, left_on='assembly_id_a', right_on='assembly_id',
              how='left').rename(columns={'host':'host_a','hn_subtype':'hn_subtype_a','year':'year_a'})
df = df.merge(meta, left_on='assembly_id_b', right_on='assembly_id',
              how='left').rename(columns={'host':'host_b','hn_subtype':'hn_subtype_b','year':'year_b'})
```

For **positive pairs** `assembly_id_a == assembly_id_b`, so the two sides are
identical by construction — `host_a == host_b`, `hn_subtype_a == hn_subtype_b`,
`year_a == year_b`.

---

## Subtype parsing

`hn_subtype` values in val_unfilt (108,530 isolates):

| Pattern | Count | % | Bucket |
|---|---|---|---|
| `^H\d+N\d+$` (e.g., H3N2, H1N1, H5N1) | 107,524 | 99.07% | parsed |
| `"HN"` (untypable) | 1,004 | 0.93% | unknown |
| `"H1N"`, `"H3N"` (truncated) | 2 | 0.00% | unknown |
| NaN | 0 | 0.00% | (n/a) |

Parsing rule: `re.match(r'^(H\d+)(N\d+)$', s)` — anything that fails → `"unknown"`.
Expect ~1% of isolates in the unknown bucket; all others parse cleanly.

---

## Level 1: pair-regime stratification

Classifies each test pair into one of four mutually exclusive regimes:

| Regime | Definition | Why it matters |
|---|---|---|
| `positive` | `label == 1` (same isolate) | Reference for the positive class |
| `within_subtype_neg` | `label == 0` and both sides parsed and `hn_subtype_a == hn_subtype_b` | "Hard" negatives — model must use signal beyond subtype |
| `cross_subtype_neg` | `label == 0` and both sides parsed and `hn_subtype_a != hn_subtype_b` | "Easy" negatives — subtype difference is a shortcut |
| `unknown_neg` | `label == 0` and at least one side is `"unknown"` | Small residual bucket, reported separately |

For each regime, report: `n_samples, accuracy, precision, recall, f1, auc_roc,
auc_pr, fp_rate, fn_rate, fp_avg_confidence, fn_avg_confidence`.
`precision` and positive-class metrics are defined only for regimes that contain
`label == 1` (i.e., `positive`); negative regimes report FP rate / confidence only.

**Why the gap between `within_subtype_neg` and `cross_subtype_neg` matters:**
if accuracy on `cross_subtype_neg` ≈ 1.0 but accuracy on `within_subtype_neg` is
much lower, the model is relying on subtype as a shortcut rather than true
isolate-pair co-occurrence signal.

---

## Level 2: per-axis marginal stratification

Three axes, analyzed independently: **host, hn_subtype, year_bin**.

For each axis, a pair is assigned to the stratum value only if **both sides share
that value**. Pairs where the two sides differ are aggregated into a `"mixed"`
stratum (reported once per axis). Pairs with NaN/unknown on either side go to
`"unknown"`. This produces a compact table per axis.

### Axis: host
- Strata: one per common host ({`Human`, `Pig`, `Chicken`, `Cow`, ...}) — keep those
  with `n_samples >= 20` after pairing; collapse rarer hosts into `"other"`.
- Special strata: `"mixed"` (host_a != host_b), `"unknown"` (either side NaN).

### Axis: hn_subtype
- Strata: one per common subtype in the parsed set (`H3N2`, `H1N1`, `H5N1`, ...) —
  same `n_samples >= 20` cutoff, rarer → `"other"`.
- Special strata: `"mixed"` (subtype_a != subtype_b, both parsed), `"unknown"`
  (either side malformed).
- **Note:** for positives this reduces to per-isolate subtype; for negatives it
  isolates the within-subtype population.

### Axis: year_bin
- Binning: `<=2015`, `2016-2020`, `2021+`. Rationale: Flu A data is
  heavily skewed toward recent years; 3 bins keep strata dense while separating
  pre- and post-pandemic eras.
- Strata: `"<=2015-<=2015"`, `"2016-2020-2016-2020"`, `"2021+-2021+"` (only
  same-bin pairs count per stratum).
- Special strata: `"mixed"` (different bins), `"unknown"` (either side NaN).

For each stratum, report the same metrics as Level 1. CSV output per axis; a
combined summary markdown table per run.

---

## Reading the plots: why bars can be missing

A missing F1 / AUC-ROC / AUC-PR bar on a Level 1 or Level 2 plot does **not**
mean the model had no false predictions — it means the metric is mathematically
undefined for that stratum. Two distinct reasons:

1. **Single-class stratum (affects AUC-ROC and AUC-PR).** Both AUC metrics
   require at least one positive AND one negative to be defined (you can't
   draw a ROC/PR curve from a single-class sample). Rare strata where all
   pairs happen to be positive (e.g., a subtype with a handful of isolates
   that only shows up in positive pairs) will have F1 but no AUC. Negative-
   only strata such as `mixed` on the subtype axis (see point 2) will have
   no AUC either.

2. **All-negative stratum (affects F1 too).** The `mixed` bucket on any axis
   is always all-negative: positive pairs have `assembly_id_a == assembly_id_b`,
   so both sides necessarily share host / subtype / year and can never land
   in `mixed`. On Level 1, the three `*_neg` regimes are all-negative by
   construction. In these buckets F1 is also undefined (no TPs possible →
   precision=0, recall undefined) and only TNR / FP rate / accuracy carry
   signal.

Plots render undefined metrics as thin grey placeholder bars labeled `N/A`
so single-class / all-negative strata are visually distinguishable from
strata where the metric really is zero.

## Outputs

Colocated with existing post-hoc artifacts in `<training_run>/post_hoc/`:

- **Existing (preserved):** `confusion_matrix.png`, `roc_curve.png`,
  `precision_recall_curve.png`, `prediction_distribution.png`, `fp_fn_analysis.png`,
  `error_analysis_by_host.csv`, `error_analysis_by_hn_subtype.csv`,
  `error_analysis_by_year.csv`, error-analysis bar plots.
- **New:**
  - `level1_pair_regime.csv` — 4 rows (positive, within_subtype_neg, cross_subtype_neg, unknown_neg).
  - `level2_by_host.csv` — one row per host stratum + `mixed` + `unknown`.
  - `level2_by_hn_subtype.csv` — one row per subtype stratum + `mixed` + `unknown`.
  - `level2_by_year_bin.csv` — one row per year bin stratum + `mixed` + `unknown`.
  - `level1_pair_regime.png` — bar plot of accuracy/F1/AUC per regime.
  - `level2_by_{host,hn_subtype,year_bin}.png` — bar plots per axis.
  - `stratified_eval_summary.md` — human-readable summary per run.

The existing `error_analysis_by_{host,hn_subtype,year}.csv` files stay for
backwards compatibility but will be augmented to honor both-sides matching
(the current files silently use side-A only, which is incorrect).

---

## Aggregation across the 28-pair sweep (future, not in this pass)

A companion script (`src/analysis/aggregate_stratified_eval.py`, to be built)
will gather per-run `level1_pair_regime.csv` and `level2_*.csv` files across
all 28 pairs × 12 folds for a sweep (e.g., `val_unfilt`) and produce a
cross-pair summary with mean ± std per regime and per stratum.

---

## Open questions / defaults to revisit

- **Year binning** is currently 3 fixed bins. If publication plots need finer
  granularity, bump to 4 or 5 bins; sample counts per bin should stay ≥500.
- **Rarer-host collapse threshold** (`n_samples >= 20`) is a guess; adjust
  after seeing per-fold counts.
- **Mixed-stratum interpretation** — mixed-host and mixed-year pairs for
  negatives could be further decomposed (e.g., which specific host-pairs are
  easiest). Deferred.

---

## Style / logging conventions

- No emojis in print/log output (CLAUDE.md). Existing `⚠️` in
  `analyze_stage4_train.py` will be replaced with `WARNING:` prefixes as part of
  this edit.
- `ERROR:` for fatal, `WARNING:` for noteworthy non-fatal, `Done.` for success.
