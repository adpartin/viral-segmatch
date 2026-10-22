# Paper plan: influenza segment-pair classification

**Status: IN PROGRESS**

This plan is based on Jim's plan of 2026-10-08 (`plan-Jim-10-8-26-original.pdf`). That PDF is not
tracked in the repo.

## What the current labels mean

In the current experiments, positives are observed same-isolate segment pairs. Negatives are pairs
not observed together in the corpus. A negative label does not establish biological
incompatibility.

## 1. Establish and demonstrate the method on `Human-H3N2-2024`

**Status: IMPLEMENTED (2026-10-15).** The 28 datasets were built and audited. All 140 `1A` fits and
all 1,400 `1B` redraw fits completed at commit `ba9b108`. The final report is
in `results/flu/July_2025/task1_28p_codon_matched_20261015_194607_report/`.

**Goal.** Establish the method on a narrow population that contains a single metadata group.

### 1A. Performance for all 28 schema pairs

**Goal.** Show how performance changes between different schema pairs.

**Methods.**
- 5-fold CV
- LightGBM
- Positives: HK selection of positives (each sequence appears once)
- Negatives: matched balanced-usage negatives, `matched_balanced_usage_negatives`
  (`split_strategy.negative_scope=balanced` with a `dataset.negative_sampling` block). Each
  negative pairs a slot-A sequence and a slot-B sequence from the `HK selected` positives of the
  same split, and the two sides must share host and HN subtype and have the same year. At a 1:1
  ratio, the accepted datasets must use each selected sequence exactly once per slot in each
  split's negatives.
- The Task 1 population is a single metadata group, so the matching requirement removes no
  candidate negative here. Across the 28 `1A` datasets and the 28 redraw datasets, all 329,410
  saved rows carry `Human`, `H3N2` and 2024 on both sides, and every negative is drawn in the
  `host_subtype_year` regime.

**Results.** 
- experiment spec `conf/experiments/human_h3n2_2024_28p_codon_matched_cv5.yaml`
- dataset prefix `task1_28p_codon_matched_20261015_194607_`
- training run `task1_28p_codon_matched_20261015_194607_lgbm_20261015_195321`
- report `results/flu/July_2025/task1_28p_codon_matched_20261015_194607_report/`

<img src="../results/figs/2026-10-15_h3n2_2024_codon_matched_28pairs_f1_macro_matrix.png" width="600" alt="28-pair matched-negative F1 macro matrix, per-site codon LightGBM, Human-H3N2-2024">

- The strictly lower triangle holds the 28 schema pairs. Each cell shows that pair's mean test F1
  macro across the 5 folds, with ± SD across those folds on the next line, and the cell is colored
  by the mean. The layout follows Fig 5 in
  https://academic.oup.com/bib/article/27/1/bbaf667/8422735, with the ± SD on a separate line.
- In the table below, mean ± SD is across the 5 folds, and pairs are ranked by mean F1 macro.
- All 8 Experiment 1 capacity columns sit beside these metrics in `allpairs_summary.csv`.

| Rank | Schema pair | HK selected | Features | F1 macro | AUC-ROC | AUC-PR | Precision | Recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2-PB1 | 1,404 | 1,519 | 0.895 ± 0.009 | 0.944 ± 0.007 | 0.923 ± 0.018 | 0.859 ± 0.012 | 0.947 ± 0.015 |
| 2 | PB1-HA | 1,392 | 1,326 | 0.895 ± 0.016 | 0.948 ± 0.007 | 0.928 ± 0.009 | 0.858 ± 0.010 | 0.948 ± 0.038 |
| 3 | PB2-HA | 2,042 | 1,327 | 0.893 ± 0.022 | 0.941 ± 0.014 | 0.918 ± 0.020 | 0.847 ± 0.028 | 0.963 ± 0.012 |
| 4 | HA-NA | 1,698 | 1,037 | 0.886 ± 0.012 | 0.940 ± 0.012 | 0.918 ± 0.019 | 0.843 ± 0.011 | 0.950 ± 0.013 |
| 5 | PB1-NA | 1,212 | 1,229 | 0.885 ± 0.033 | 0.929 ± 0.023 | 0.900 ± 0.018 | 0.849 ± 0.037 | 0.938 ± 0.027 |
| 6 | PB2-NA | 1,745 | 1,230 | 0.874 ± 0.018 | 0.933 ± 0.012 | 0.913 ± 0.020 | 0.831 ± 0.023 | 0.942 ± 0.017 |
| 7 | PB2-PA | 2,030 | 1,477 | 0.850 ± 0.023 | 0.929 ± 0.013 | 0.914 ± 0.018 | 0.810 ± 0.033 | 0.919 ± 0.008 |
| 8 | PB1-PA | 1,341 | 1,476 | 0.850 ± 0.021 | 0.918 ± 0.017 | 0.896 ± 0.025 | 0.800 ± 0.020 | 0.936 ± 0.026 |
| 9 | PA-HA | 1,944 | 1,284 | 0.842 ± 0.025 | 0.917 ± 0.016 | 0.892 ± 0.014 | 0.798 ± 0.030 | 0.919 ± 0.016 |
| 10 | PA-NA | 1,689 | 1,187 | 0.827 ± 0.009 | 0.905 ± 0.004 | 0.877 ± 0.013 | 0.782 ± 0.010 | 0.911 ± 0.011 |
| 11 | PB2-NP | 1,512 | 1,259 | 0.823 ± 0.017 | 0.901 ± 0.015 | 0.880 ± 0.017 | 0.759 ± 0.018 | 0.954 ± 0.003 |
| 12 | PB1-NP | 1,041 | 1,258 | 0.818 ± 0.019 | 0.900 ± 0.016 | 0.886 ± 0.024 | 0.769 ± 0.018 | 0.916 ± 0.024 |
| 13 | HA-NP | 1,482 | 1,066 | 0.817 ± 0.019 | 0.898 ± 0.011 | 0.875 ± 0.020 | 0.765 ± 0.021 | 0.923 ± 0.016 |
| 14 | PA-NP | 1,459 | 1,216 | 0.813 ± 0.022 | 0.902 ± 0.016 | 0.889 ± 0.016 | 0.765 ± 0.021 | 0.907 ± 0.019 |
| 15 | NP-NA | 1,287 | 969 | 0.806 ± 0.029 | 0.888 ± 0.021 | 0.864 ± 0.023 | 0.751 ± 0.031 | 0.927 ± 0.011 |
| 16 | HA-M1 | 720 | 820 | 0.778 ± 0.020 | 0.855 ± 0.020 | 0.833 ± 0.024 | 0.727 ± 0.023 | 0.904 ± 0.029 |
| 17 | NA-M1 | 657 | 723 | 0.746 ± 0.035 | 0.842 ± 0.030 | 0.809 ± 0.030 | 0.698 ± 0.041 | 0.901 ± 0.042 |
| 18 | PB2-M1 | 726 | 1,013 | 0.729 ± 0.027 | 0.812 ± 0.013 | 0.790 ± 0.016 | 0.680 ± 0.024 | 0.894 ± 0.028 |
| 19 | PB1-NS1 | 704 | 990 | 0.728 ± 0.039 | 0.823 ± 0.032 | 0.815 ± 0.030 | 0.687 ± 0.039 | 0.859 ± 0.027 |
| 20 | PB1-M1 | 517 | 1,012 | 0.709 ± 0.058 | 0.799 ± 0.055 | 0.770 ± 0.055 | 0.666 ± 0.047 | 0.897 ± 0.045 |
| 21 | HA-NS1 | 959 | 798 | 0.699 ± 0.055 | 0.789 ± 0.044 | 0.776 ± 0.040 | 0.657 ± 0.043 | 0.891 ± 0.014 |
| 22 | PB2-NS1 | 995 | 991 | 0.691 ± 0.060 | 0.789 ± 0.052 | 0.781 ± 0.055 | 0.657 ± 0.053 | 0.851 ± 0.054 |
| 23 | NA-NS1 | 853 | 701 | 0.656 ± 0.041 | 0.760 ± 0.042 | 0.749 ± 0.043 | 0.626 ± 0.033 | 0.826 ± 0.032 |
| 24 | NP-NS1 | 806 | 730 | 0.628 ± 0.022 | 0.719 ± 0.037 | 0.714 ± 0.040 | 0.602 ± 0.015 | 0.846 ± 0.031 |
| 25 | PA-M1 | 707 | 970 | 0.601 ± 0.076 | 0.710 ± 0.058 | 0.698 ± 0.056 | 0.591 ± 0.044 | 0.894 ± 0.046 |
| 26 | PA-NS1 | 952 | 948 | 0.594 ± 0.050 | 0.701 ± 0.051 | 0.694 ± 0.051 | 0.581 ± 0.030 | 0.868 ± 0.025 |
| 27 | NP-M1 | 627 | 752 | 0.589 ± 0.056 | 0.694 ± 0.046 | 0.681 ± 0.044 | 0.581 ± 0.040 | 0.850 ± 0.070 |
| 28 | M1-NS1 | 440 | 484 | 0.564 ± 0.058 | 0.683 ± 0.064 | 0.684 ± 0.062 | 0.564 ± 0.034 | 0.848 ± 0.055 |

### 1B. Measure sensitivity to the draw of training negative pairs

**Goal.** Measure how much model predictions vary on a fixed test set when we vary only the
training negative pairs.

**Methods.**
- Same as in `1A`.
- For every fold, redraw the training negatives `r` times, each time with a different random seed,
  and train one model per draw. Each redraw uses the sampler the dataset was built with, which for
  these datasets is `matched_balanced_usage_negatives`. The training positives, the val and test
  pairs, and the model seed stay the same. This results in `r` models per fold.
- Use `r = 10` draws with seeds 42–51, yielding 1,400 redraw fits (28 pairs × 5 folds × 10 draws).
- The same negative pair can appear in more than one draw. We allow this, and we report how often
  it happens: the most negative pairs that any two draws of a fold share, and how many negative
  pairs appear in more than one draw.
- Each of the `r` models predicts the same test pairs. This results in `r` predictions for each
  test pair.

**Results.** The redraws used the same datasets as `1A`, under the run prefix
`task1_28p_codon_matched_20261015_194607_negredraw_20261015_200417_`, fitted as
`task1_28p_codon_matched_20261015_194607_negredraw_20261015_200417_lgbm`. Each redraw drew its
negatives with `matched_balanced_usage_negatives`, the sampler the source datasets were built with.
The report is in `results/flu/July_2025/task1_28p_codon_matched_20261015_194607_report/`.

<img src="../results/figs/2026-10-15_h3n2_2024_codon_matched_28pairs_f1_macro_sd_draws_matrix.png" width="600" alt="28-pair matrix of the mean across folds of the within-fold SD of test F1 macro across 10 negative draws, Human-H3N2-2024">

- The matrix is laid out as in `1A`. Each cell shows one value: the mean across the 5 folds of the
  within-fold SD of test F1 macro across the 10 draws. It measures the variation from redrawing the
  training negatives, which is a different source of variation from the SD across folds in `1A`.
- The table below holds fold 0 of each of the 28 pairs, sorted by F1 macro. The other four folds
  are in `redraw_by_pair_fold.csv`, which has all 140 rows. The matrix averages the within-fold
  draw SD across folds. How to summarize the other columns across folds is not yet decided, so the
  table shows one fold rather than an average.

| Schema pair | F1 macro | prob SD pos / neg | logit SD pos / neg | Max shared by two draws | In >1 draw |
|---|---:|---:|---:|---:|---:|
| PB1-HA | 0.909 ± 0.007 | 0.051 / 0.057 | 0.370 / 0.615 | 5 | 40 |
| PB2-HA | 0.894 ± 0.010 | 0.059 / 0.061 | 0.397 / 0.652 | 3 | 37 |
| PB1-NA | 0.894 ± 0.009 | 0.061 / 0.068 | 0.365 / 0.498 | 4 | 50 |
| PB2-PB1 | 0.885 ± 0.015 | 0.071 / 0.079 | 0.361 / 0.445 | 4 | 35 |
| HA-NA | 0.868 ± 0.020 | 0.072 / 0.086 | 0.335 / 0.453 | 5 | 39 |
| PB2-PA | 0.863 ± 0.007 | 0.045 / 0.048 | 0.273 / 0.340 | 4 | 50 |
| PB2-NA | 0.858 ± 0.007 | 0.058 / 0.070 | 0.300 / 0.451 | 4 | 44 |
| PA-HA | 0.857 ± 0.024 | 0.064 / 0.080 | 0.355 / 0.582 | 3 | 42 |
| PB1-PA | 0.836 ± 0.020 | 0.053 / 0.056 | 0.323 / 0.366 | 4 | 37 |
| PA-NA | 0.831 ± 0.009 | 0.046 / 0.048 | 0.252 / 0.397 | 3 | 43 |
| HA-NP | 0.824 ± 0.008 | 0.057 / 0.055 | 0.350 / 0.548 | 4 | 47 |
| NP-NA | 0.815 ± 0.006 | 0.039 / 0.047 | 0.184 / 0.331 | 3 | 42 |
| PB1-NP | 0.814 ± 0.020 | 0.045 / 0.054 | 0.226 / 0.344 | 3 | 33 |
| PB2-NP | 0.813 ± 0.014 | 0.066 / 0.090 | 0.312 / 0.641 | 5 | 49 |
| HA-M1 | 0.795 ± 0.009 | 0.026 / 0.035 | 0.117 / 0.167 | 3 | 43 |
| PA-NP | 0.770 ± 0.023 | 0.055 / 0.077 | 0.232 / 0.330 | 4 | 42 |
| HA-NS1 | 0.759 ± 0.018 | 0.040 / 0.050 | 0.175 / 0.313 | 4 | 51 |
| PB2-M1 | 0.737 ± 0.031 | 0.036 / 0.049 | 0.147 / 0.204 | 3 | 45 |
| PB2-NS1 | 0.732 ± 0.023 | 0.030 / 0.042 | 0.120 / 0.176 | 3 | 40 |
| NA-M1 | 0.703 ± 0.018 | 0.030 / 0.033 | 0.122 / 0.135 | 5 | 51 |
| PB1-NS1 | 0.696 ± 0.016 | 0.026 / 0.030 | 0.107 / 0.124 | 5 | 52 |
| PB1-M1 | 0.678 ± 0.023 | 0.028 / 0.028 | 0.112 / 0.115 | 3 | 44 |
| NA-NS1 | 0.662 ± 0.032 | 0.030 / 0.033 | 0.122 / 0.134 | 5 | 58 |
| M1-NS1 | 0.651 ± 0.041 | 0.011 / 0.016 | 0.044 / 0.064 | 3 | 45 |
| NP-NS1 | 0.630 ± 0.030 | 0.021 / 0.025 | 0.085 / 0.100 | 5 | 42 |
| PA-NS1 | 0.597 ± 0.031 | 0.019 / 0.022 | 0.076 / 0.089 | 3 | 47 |
| NP-M1 | 0.567 ± 0.031 | 0.012 / 0.013 | 0.048 / 0.052 | 3 | 49 |
| PA-M1 | 0.507 ± 0.015 | 0.014 / 0.014 | 0.055 / 0.055 | 4 | 35 |

How to read the table, using the PB1-HA row:

- Every row is fold 0 of that schema pair. Fold 0 has 10 models, one per draw with seeds 42-51, and
  all 10 are scored on the same test pairs. PB1-HA fold 0 has 558 test pairs, 279 positive and 279
  negative, and 973 training positives and 973 training negatives per draw.
- `F1 macro` = 0.909 ± 0.007: the mean ± SD of test F1 macro over the 10 models. Here the ± is the
  spread across negative draws, not across folds as in `1A`.
- `prob SD pos / neg` = 0.051 / 0.057: each test pair gets one SD, computed from its 10 predicted
  probabilities. 0.051 is the median of these SDs over the 279 positive test pairs, and 0.057 is
  the median over the 279 negative test pairs.
- `logit SD pos / neg` = 0.370 / 0.615: the same medians, computed on LightGBM's raw score instead
  of the probability. The raw score is `logit(p) = log(p / (1 - p))`.
- `Max shared by two draws` = 5 and `In >1 draw` = 40: the most negative pairs that any two of the
  10 draws share, and the number of distinct negative pairs that appear in more than one draw. The
  10 draws of this fold use 9,690 distinct negative pairs out of 9,730 drawn, so each of the 40
  repeated pairs appears in exactly two draws.

Relative to the 2026-10-09 balanced-negative 5-fold run (`h3n2_2024_28p_codon_cv5_20261009_141504_`),
the per-pair mean over folds of the draw mean changes by 0.0074 on average in absolute value, with a
mean change of −0.0006 and a range of −0.0218 to +0.0278. That run differs from this one in
both the negative sampler and the co-occurrence blocking set, which was population-scoped there
and corpus-wide here, so these differences do not isolate either change.

### Implementation record

Experiments 1, 2 and 6 in `docs/plans/2026-09-14_codon_site_features_plan.md` supplied the
initial 28-pair screen and training-negative redraws. Task 1 added 5-fold experiment specs,
manifest-based dataset and fit selection, strict config and input checks, a 28-pair audit and
report, and a redraw coordinator. The final matched-negative spec is
`conf/experiments/human_h3n2_2024_28p_codon_matched_cv5.yaml`.

`scripts/run_allpairs_datasets.py` built the 28 datasets;
`src/analysis/audit_pair_datasets.py` audited them;
`scripts/run_allpairs_baselines.py` ran `1A`;
`scripts/run_allpairs_balanced_redraws.py` ran `1B`; and
`src/analysis/aggregate_allpairs_results.py --final` produced the report. The dataset, training
and redraw manifests record the selected runs, settings, code commit and input checksums. The
reported datasets and all fits used commit `ba9b108`. The full output locations and results are
identified above.

## 2. Evaluate across years, hosts, and HN subtypes

**Status: IMPLEMENTED (2026-10-15).** The 21 datasets were built and audited. All 105 five-fold
LightGBM fits completed at commit `8b604ed`. The final report is in
`results/flu/July_2025/task2_21p_codon_matched_20261015_151826/`.

This run replaced a first one built on 2026-10-14 at commit `7d7b3b6`, so that Task 2 and Task 1
share both the negative sampler and the corpus-wide co-occurrence blocking set. The rebuild used
the same population, seed, sampler and spec, and the audit confirmed each pair's pooled positive
test pair keys against the earlier build. It changed no positive and 1,795 of the 382,885 saved
negatives, in 6 of the 21 pairs. Those 6 pairs are the only ones whose scores moved, by at most
0.0042 F1 macro, which is inside every pair's fold SD. The earlier run's outputs are kept in
`results/flu/July_2025/task2_21p_codon_matched_20261014_090208/`.

**Goal.** Train on a *population* containing several hosts, HN subtypes or years, and measure
performance within each *metadata group*.

* *Population*: a dataset containing all the isolates selected by the dataset filters.
* *Metadata group*: a dataset subset with one exact `(host, HN subtype, year)` combination.

Task 1 set the dataset filters to (Human, H3N2, 2024), so its population held one metadata group.
The Task 2 population holds several metadata groups, and its reports distinguish the two terms.

**Methods.** The build filters to the eight groups below, excluding isolates missing host, HN
subtype or year. `filter_by_metadata` keeps only values in the three filter lists. The Task 2
experiment spec sets `dataset.n_folds=5` for both dataset building and training; the bundles retain
their inherited four-fold setting. Positives are selected by Hopcroft–Karp after global pair-key
deduplication. Each split receives one matched balanced-usage negative per positive. Its two
sequences come from different selected positives in the same split and metadata group. The sampler
matches exact years with `year_match: exact`, rather than the default year bins. The saved negative
rows were checked using both endpoints' isolate metadata, not just the requested regime target or
the `neg_regime` field. The implementation is described below.

**Reports.** `allpairs_summary.csv` gives one row per schema pair for the whole population.
`src/analysis/metadata_group_report.py` writes `metadata_group_counts.csv` from saved train,
validation and test pairs, with positive and negative counts by schema pair, fold, split and group.
The counts sum to each split's totals. It writes `test_metrics_by_metadata.csv` from each fold's
`test_predicted.csv`, using that model's decision threshold. Its 1,470 rows cover the eight joint
groups and six pooled host, subtype or year values for each of 21 pairs and five folds. Each row
contains the group values, class counts, confusion counts, F1 macro, AUC-ROC, precision and recall.
All metric rows in this run have both classes. If another run has a one-class group, its metrics
are NaN for that fold. The figure code excludes those folds from its mean and shows how many folds
contributed.

The report contains four figures: `matrix_f1_macro.png` for the whole population,
`metrics_by_group_f1_macro.png` for the eight joint groups,
`metrics_by_axis_f1_macro.png` for six pooled values, and `counts_by_group.png` for test positives
summed over folds. The metric figures show each cell's fold mean above its fold SD and share a
color scale. `--metric` can draw other supported metrics. `--min_test_pairs` can mask thin group
cells; its default of 0 left every cell visible in this report. Figure titles and CSV rows identify
the population. NS1 and its seven schema pairs are excluded for the reason below.

### The Task 2 population

The build uses `[Human, Pig]`, `[H3N2, H1N1]`, `[2022, 2023]`. All eight metadata groups have isolates:
10,992 in total, from 4,864 in Human-H3N2-2022 down to 157 in Pig-H3N2-2022. The measurements
behind this choice are in `results/flu/July_2025/pair_capacity_task2_2x2x2/`, under
`cds_length_survey_grid/` for each group on its own and `cds_length_survey_candidates/` for the
pooled population.

This population supports the 21 schema pairs among PB2, PB1, PA, HA, NP, NA and M1. It does not
support the seven pairs that contain NS1. Per-site codon features need one pinned CDS length per
protein, and NS1 has two lengths here: 693 nt in Human-H3N2 and 660 nt in the other three host and
subtype combinations. The two counts are close, at 1,552 of the 3,062 unique complete sequences on
693 nt and 1,485 on 660 nt, so neither length reaches the 0.90 share of unique complete sequences
that `check_cds_length` requires. The lengths are 231 and 220 codons, so they differ in the protein
itself rather than in sequence quality, and no single pin covers both. Excluding NS1 is the cost of
keeping Human-H3N2 in the population. Task 1 reports all 28 pairs on its own population, so Task 2
does not repeat that.

The year choice is what allows a single PB1 pin. Human-H3N2 PB1 is 2,274 nt through 2023 and
2,277 nt in 2024 and 2025, while every other population in the survey stays at 2,274 nt in every
year measured. Over 2022-2023, all eight groups have 2,274 nt as their own modal length and the
pooled share at that length is 0.912, so the pin holds. Adding 2024 drops that share to 0.782,
which fails the check. Using 2021-2022 instead raises it to 0.944, but Human-H1N1-2021 has only
11 isolates.

The pins are PB2 2,280, PB1 2,274, PA 2,151, HA 1,701, NP 1,497, NA 1,410 and M1 759 nt.
`conf/bundles/flu_7_major_proteins_human_pig_h3n2_h1n1_2022_2023_pinned_length.yaml` sets the
three filters and overrides PB1 to 2,274 nt; `conf/virus/flu.yaml` supplies the other six pins.
The pair-capacity run uses this lower bundle. The master bundle
`conf/bundles/flu_21p_human_pig_h3n2_h1n1_2022_2023_site_codon.yaml` adds Hopcroft–Karp
selection, codon-site features and matched balanced-usage negatives. Its 21 children each set one
schema pair. The experiment spec
`conf/experiments/human_pig_h3n2_h1n1_2022_2023_21p_codon_matched_cv5.yaml` supplies five folds
and checks the settings used by the build and fits.

### Matched balanced-usage negatives

An ordinary balanced negative can join isolates from different groups. The matched sampler in
`src/datasets/_negative_sampling.py` instead assigns each selected positive to its retained source
isolate's group, then pairs slot-A and slot-B sequences from different positives within that group
and split. Global pair-key deduplication remains in place. If a sequence pair was observed in
several groups before deduplication, only its retained isolate's group receives it; the retained
isolate depends on input row order.

The Task 2 master sets all eight regime targets, with `host_subtype_year: 1.0` and the others zero,
plus `year_match: exact` and `on_shortfall: error`. The config guard accepts this profile with
`negative_scope: balanced`, three matching axes and a 1:1 negative ratio. It rejects other
`negative_sampling` profiles with balanced scope. Task 1's master bundle sets the same block since
2026-10-15, so both screens draw their negatives with this sampler. Task 1's population is one
metadata group, so the matching rules nothing out there; what it changes is the sampler, the
`neg_regime` and `metadata_match_count` columns on each negative row, and the build stopping rather
than warning when a group cannot give each of its sequences one negative use.

For each group and split, the sampler finds a one-to-one assignment from selected slot-A sequences
to selected slot-B sequences. It excludes observed positive pairs and negatives already drawn in
the fold. An augmenting-path search proves whether the assignment exists. A group with one
positive, no valid assignment, repeated selected sequences in a slot or more than 5,000 positives
raises an error naming the group and split. `duplicate_stats.json` records the group sizes and
successful negative counts. Each endpoint retains its own source isolate's metadata. Saved
negatives carry `neg_regime: host_subtype_year` and `metadata_match_count: 3`; the saved endpoint
metadata was checked independently for equal host, HN subtype and exact year.

### Observed results

The final report covers all 21 pairs and 105 completed fits, with no failed fits. The whole
population's fold-mean F1 macro ranged from 0.889 ± 0.013 for PB2-PA to 0.782 ± 0.012 for NA-M1.
All six M1 pairs ranked below the other 15. M1 also had the fewest unique slot sequences and the
smallest HK-selected sets; these results do not isolate sequence capacity as the cause of their
lower scores.

The following group values average the reported metrics over 21 pairs and five folds. Test
positives sum across the 21 pairs, with each positive appearing in one test fold. These are
descriptive means, not fitted comparisons among groups.

| Metadata group | F1 macro | AUC-ROC | Test positives |
|---|---:|---:|---:|
| Human-H3N2-2022 | 0.862 | 0.907 | 28,745 |
| Human-H1N1-2023 | 0.847 | 0.901 | 19,204 |
| Pig-H1N1-2022 | 0.840 | 0.900 | 3,239 |
| Human-H3N2-2023 | 0.837 | 0.889 | 8,441 |
| Human-H1N1-2022 | 0.825 | 0.872 | 5,711 |
| Pig-H3N2-2023 | 0.818 | 0.883 | 4,761 |
| Pig-H1N1-2023 | 0.818 | 0.878 | 4,784 |
| Pig-H3N2-2022 | 0.815 | 0.888 | 1,692 |

The largest and smallest group means differ by 0.047 F1 macro. The marginal means were Human
0.851 versus Pig 0.823, H3N2 0.851 versus H1N1 0.840, and 2022 0.852 versus 2023 0.840.
Group size did not order the scores: Pig-H1N1-2022 scored 0.840 on 3,239 test positives, while
Human-H1N1-2022 scored 0.825 on 5,711. The thinnest cells were Pig-H3N2-2022 on the six M1 pairs,
averaging 12.2 to 12.4 test positives per fold. The lowest single fold held 7 test positives, on
NP-M1. Small cells remain visible with their counts; the Pig-H3N2-2022 fold SD reached ±0.125 for
PB1-M1. No group was single-class in a test fold.

### Open questions

The first run used the builder's positive-pair fold assignment without metadata-group
stratification. It removed no group, masked no figure cell and scored every group. Matching was
feasible in all saved splits. The questions below concern possible changes for later populations
or for interpreting thin cells in this one.

1. **Small groups in the test split.** Folds are assigned by positive, not by metadata group. In
   this run Pig-H3N2-2022 averaged 12.2 NA-M1 test positives per fold, with 8 in its lowest fold,
   although all eight groups appeared in the test metrics. In another population, a group could
   have test positives but no training positives in a fold. That score would measure performance
   on an unseen group.
   The matching check does not catch this case because a training split with no positives in a
   group needs no negatives there. Options for a later run:
   - Stratify the folds by metadata group.
   - Set a minimum group size and do not score groups below it. `analyze_level2_by_axis` in
     `src/analysis/analyze_stage4_train.py` merges strata with fewer than 20 pairs into an
     `other` stratum. The report's `--min_test_pairs` option can mask group cells below a chosen
     threshold and state it in the figure title; no threshold was used here.
   - Score every group, but report the unseen-group scores separately.
     `metadata_group_counts.csv` gives each group's training positives per fold, so these scores
     can be flagged.
2. **A rule for removing small groups.** If a later run removes small groups, set the rule before
   creating the folds. Record the positives each group lost, by schema pair and metadata
   group. Never remove positives from just one split, because each positive has only one test
   fold. The CC builder likewise removes infeasible components before generating its splits.
   Removing groups can still leave a group with one positive in a validation or test split. If the
   matching check then fails, change the rule or the split and rebuild; do not patch one split.
   Do not call this rule an eligibility rule. In the glossary, *eligible isolates* are the isolates
   that carry a schema pair's required sequences, such as complete CDS at the pinned length.
3. **Coarser host labels.** Our metadata carries one host string per isolate, such as `Chicken`
   or `Mallard`, and most non-human hosts have too few isolates. Grouping them into a coarser label
   (especially `Avian`) would give a host group large enough to compare against `Human`. Marcus's
   paper reports performance over human, avian, environmental, sea-mammal, lab-derived and
   nonhuman-mammal samples, plus unknown and other. Jamie's paper uses human, swine, duck, chicken
   and other. We have no host-grouping map in the repo yet.



## 3. Test on future-year pairs

**Goal.** Train on past-year pairs and test on future-year pairs whose endpoint sequences were not
selected for training or validation. Report performance separately for each test year.

**Example run.** For one schema pair, use 2011–2013 as the past pool and 2014–2015 as the future
pool. Set `dataset.metadata_holdout.train: {year_range: [2011, 2013]}` and
`dataset.metadata_holdout.test: {year_range: [2014, 2015]}`. These are example years; choose a
population with enough selected positives in each split before fixing the experiment bundle. Use
the same sequence filters and `pair_key_alphabet` in both pools. The Task 1/2 codon setting uses
`nt_cds`.

1. Build positive candidates from the past pool only. Deduplicate by `pair_key`, then run
   Hopcroft–Karp (HK) once on those candidates. Randomly assign `dataset.val_ratio` of the selected
   positives to validation with a recorded seed; the rest go to training. Freeze both sets.
2. Build positive candidates from both future years. Drop a candidate if its slot-A sequence was
   selected in past training or validation, or if its slot-B sequence was selected there. Compare
   hashes in the configured `pair_key_alphabet`. Deduplicate the remaining future candidates by
   `pair_key`, then run HK once across both future years. These selected pairs form the test set;
   future selection cannot change the frozen past positives.
3. Assert that no selected `pair_key`, slot-A hash or slot-B hash occurs in more than one split.
   Record candidate, deduplicated, excluded-for-past-overlap and HK-selected counts by period.
   Also report HK-selected counts by test year. Use the source year of each retained positive for
   its test-year group.
4. Draw 1:1 balanced-usage negatives separately in each split. Use Task 2's matched profile so
   both endpoints of each negative share host, HN subtype and exact year. Fail with the split and
   group named if a one-use matching cannot be made. Train one model on the past training set,
   select its settings on past validation, and evaluate it once on the combined future test set.

**Required code changes.**
- Add a single-split temporal HK path and allow active `positive_pair_selection` with
  `metadata_holdout` and `dataset.n_folds: 1`. The current validator requires at least two folds
  for active selection. The current holdout path divides past isolates into train and validation
  before positive selection, so it must instead divide the past HK-selected pairs afterward.
- Build and select past positives before examining future candidates. Do not globally deduplicate
  positive pairs across both periods or rerun HK on their union. Reuse the positive selector and
  the existing prepartitioned-positive input to `split_dataset_v2` for negative sampling and
  audits. Save the period-specific selection counts and cross-split overlap checks.
- Negatives are already blocked in every split against a corpus-wide co-occurrence set, so this
  needs no Task 3 work. `_pair_helpers.build_corpus_cooccurrence_set` builds it on the two
  proteins of the schema pair and the configured `pair_key_alphabet`, reading the alphabet's own
  `*_final` table, so neither the population filters nor the pinned-length filter narrow it. A
  sequence pair observed together anywhere in the corpus never receives a negative label, in the
  test split as well as in training. The set spans both periods, so a co-occurrence first seen in
  a future isolate can reject a past training negative. That is accepted here: it only removes
  candidate negatives and corrects labels, and it never puts information about a test pair into a
  training row. The scope and size are recorded with the dataset statistics as
  `co_occurrence_blocking.scope` and `total_cooccur_pairs`, and the audit count of observed
  positives labeled negative is measured against this set.
- Add a bundle and experiment spec for the chosen years and pair. Exercise the single-split
  dataset audit, fit runner and aggregator end to end; adjust any fold-count assumptions or
  figure labels that call a single holdout `CV`. The existing temporal HA-NA bundle uses a
  different year window and does not select HK positives.
- Use the existing year rows in `test_metrics_by_metadata.csv` for per-test-year scores. Add a
  plot of F1 macro against test year from this one held-out model; `plot_year_sweep.py` reads
  separate random-split runs and does not draw this result.

**Results.** Report test positives, negatives, F1 macro and AUC-ROC for 2014 and 2015, alongside
the combined test score. Plot F1 macro by test year with the counts visible. This single split
has no across-fold SD. A difference between the two years describes this held-out test set; it
does not isolate elapsed time from changes in the sampled sequences or metadata groups.

**Notes.**
- The corpus-wide blocking set changes few labels. Checked directly against the Task 2 run
  `task2_21p_codon_matched_20261014_090208_*`, 6 of 382,885 saved negatives across 21 pairs and 5
  folds are corpus co-occurrences, and in the Task 1 run
  `h3n2_2024_28p_codon_cv5_20261009_141504_*`, 6 of 164,705 across 28 pairs and 5 folds, with
  per-pair counts of 0 or 1. On the Task 2 population, the sequence pairs that co-occur outside
  that population while both of their sequences are present inside it number 68 for PB2-NP, 129
  for NP-M1, 69 for HA-NA and 47 for PB2-PB1, against slot universes of 12 to 33 million candidate
  cells. Under uniform draws that is 0.01 to 0.06 expected mislabeled negatives per split. These
  counts use complete CDS at the modal length per protein in place of the pin, so they are
  candidate counts rather than counts from a build.
- A single split is `dataset.n_folds: null` in both existing holdout bundles. The validator
  accepts null or 1, so the Task 3 bundle should state which one it uses.
- Validation drives LightGBM early stopping on AUC, over 50 rounds by default. The decision
  threshold stays at 0.5, because `conf/training/base.yaml` sets `threshold_metric: null`. A fixed
  threshold is what lets F1 macro drift when the score distribution shifts between years while
  AUC-ROC holds, which is the reason to report both.
- `dataset.val_ratio` is 0.1 and currently means a fraction of the train isolates. Step 1 above
  reuses it as a fraction of the selected past pairs.
- Two temporal HA-NA bundles already exist:
  `conf/bundles/flu_ha_na_h3n2_nt_cds_temporal_2025.yaml` (train 2015-2024, test 2025) and
  `conf/bundles/flu_ha_na_h3n2_temporal_2024_to_2025.yaml` (train 2024, test 2025). Both leave
  `dataset.positive_pair_selection.method` at `all`.
