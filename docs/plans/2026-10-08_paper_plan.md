# Paper plan: influenza segment-pair classification

**Status: IN PROGRESS**

This plan is based on `plan-Jim-10-8-26-original.pdf`.

## What the current labels mean

In the current experiments, positives are observed same-isolate segment pairs. Negatives are pairs
not observed together in the source corpus. A negative label does not establish biological
incompatibility.

## 1. Establish and demonstrate the method on `Human-H3N2-2024`

**Goal.** Establish the method on one narrow population.

### 1A. Performance for all 28 schema pairs

**Goal.** Show how performance changes between different schema pairs.

**Methods.**
- 5-fold CV (try larger `k` for the final paper)
- LightGBM
- Positives: HK selection of positives (each sequence appears once)
- Negatives: balanced-usage negatives (`split_strategy.negative_scope=balanced`). Each negative
  pairs a slot-A sequence and a slot-B sequence from the `HK selected` positives of the same split.
  At a 1:1 ratio, each sequence is normally used once.

**Results.**
- A detailed table as shown in `2026-09-14_codon_site_features_plan.md` Exp 2 with 28 rows.
  Supp materials.
- A schema-pair matrix: an 8-by-8 matrix that shows all 28 schema pairs with mean and std. Main
  paper. It should be a strictly lower triangular matrix, where each cell shows the mean and, on
  the next line, ± std. See Fig 5 in https://academic.oup.com/bib/article/27/1/bbaf667/8422735,
  but with ± std instead of parentheses. We need to update the code that generates the matrix
  (`src/analysis/aggregate_allpairs_results.py`).

### 1B. Measure sensitivity to the draw training negative pairs

**Goal.** Measure how much model predictions vary on a fixed test set when we vary only the
training negative pairs.

**Methods.**
- Same as in `1A`.
- For every fold, redraw the balanced-usage training negatives `r` times, each time with a
  different random seed, and train one model per draw. The training positives, the val and test
  pairs, and the model seed stay the same. This results in `r` models per fold.
- The same negative pair can appear in more than one draw. We allow this, and we report how often
  it happens: the most negative pairs that any two draws of a fold share, and how many negative
  pairs appear in more than one draw.
- Each of the `r` models predicts the same test pairs. This results in `r` predictions for each
  test pair.

**Results.**
- A detailed table as shown in `2026-09-14_codon_site_features_plan.md` Exp 6, with one row per
  schema pair and fold: 28 × `k` rows (140 at `k` = 5). Supp materials.
- A schema-pair matrix as in `1A`. `todo`: decide what each cell shows.

### Existing materials
Experiments 1, 2, and 6 in `docs/plans/2026-09-14_codon_site_features_plan.md` cover a lot of
this on Human-H3N2-2024. Experiment 2 covered 28 schema pairs with 4-fold CV. Experiment 6 redrew **training** negatives with fixed test pairs for PB2-HA and PB2-NP.

### Code gaps
The code covers the methods in `1A` and `1B`. These pieces are missing:

- `1A`: the schema-pair matrix (see `1A` Results).
- `1A`: 5-fold CV. The 28-pair bundles set `n_folds: 4`, and `scripts/run_allpairs_datasets.py`
  cannot override it, so the bundle must change. `src/analysis/audit_pair_datasets.py` expects
  `dataset.n_folds: 4`, so it will fail every 5-fold dataset until that line changes.
- `1A`: the detailed table. `src/analysis/aggregate_allpairs_results.py` writes the mean and std of
  each metric to `allpairs_summary.csv`, but not the `HK selected` and `Features` columns.
- `1B`: a script that runs the redraws for all 28 pairs.
  `scripts/resample_balanced_train_negatives.py` and `scripts/run_balanced_train_redraw.py` run one
  pair and one fold per call. `scripts/analyze_balanced_train_redraw.py` runs one pair per call.
- `1B`: the overlap counts. `redraw_summary.json` gives the number of distinct negative pairs and
  the pairwise Jaccard. It does not give the most pairs that two draws share, or the number of pairs
  that appear in more than one draw.
- `1B`: the detailed table. `scripts/analyze_balanced_train_redraw.py` writes one
  `per_pair_sd_by_class.csv` per schema pair, and nothing combines them into one table.



## 2. Evaluate across years, hosts, and HN subtypes

**Goal.** Build one model (or several) that covers several hosts, subtypes and years, and check
how well it predicts for each of them.

**Methods.**
- Metadata-aware negative sampling: each negative pairs sequences from two isolates with the same
  host, subtype and year. This is the `host_subtype_year` regime in
  `src/datasets/_negative_regime_sampling.py`, set with `dataset.negative_sampling.regime_targets`.
  The code exists, but we have not checked that it works with balanced-usage negatives.
- By default the sampler matches the year bin (`<=2015`, `2016-2020`, `2021+`), not the exact year.
  `dataset.negative_sampling.year_match: exact` matches the exact year.
- Determine largest groups, and experiment with them. For example: host=[Human, Pig],
subtype=[H3N2, H5N1], year_range=[2015, 2025].

**Results.**
- The schema-pair matrix
- Report F1 scores over metadata categories in test set (HN-type, host, year)
- Show performance for different populations.



## 3. Test on future-year pairs

**Goal.** Using model(s) trained on prior years, how well can we predict pairs
from future years?

**Methods.**
- Temporal holdout: keep future-year pairs out of model training. Set with
  `dataset.metadata_holdout`, which takes a `train` and a `test` year filter.
- E.g., train_years=[2015, 2024], test_years=[2025]. The bundle
  `conf/bundles/flu_ha_na_h3n2_nt_cds_temporal_2025.yaml` already does this for HA-NA in H3N2.
- Compare the above with a random split within 2025: train_years=[2025], test_years=[2025].
- E.g., train_years=[2015, 2020], test_years=[2021, 2022, 2023, 2024, 2025].
  Break down performance by test year.
- Does prediction uncertainty increase as we predict further into the future?

**Results.**
- Plot of F1 vs Year.
- ...