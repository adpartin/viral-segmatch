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
The dataset builder and single-fold redraw code exist. Task 1 still needs:

- `1A` matrix: `src/analysis/aggregate_allpairs_results.py` makes a symmetric matrix of means.
  Change it to show only the lower triangle, with mean and ± std on separate lines.
- `1A` 5-folds: the dataset builder accepts `dataset.n_folds=5`, but the 28-pair bundles set 4
  and `scripts/run_allpairs_datasets.py` cannot pass an override. The training runner reads the
  fold count from the bundle, so dataset generation and training need the same 5-fold config.
  Going from 4 to 5 folds keeps the same `HK selected` positives, because they are selected before
  folds are assigned. It changes the split shares of the selected positives: test goes from 25% to
  20%, and train from 65% to 70%. Validation stays at 10% (`val_ratio: 0.1`).
- `1A` dataset audit: `src/analysis/audit_pair_datasets.py` expects 4 folds; make the fold count
  configurable. `--expect_pairs 28` counts run directories, not distinct schema pairs, so two runs
  of one pair plus a missing pair would still pass. Training settings such as `site.unit: codon`
  are printed as notes and never fail the audit.
- `1A` table: extend `src/analysis/aggregate_allpairs_results.py` to add all eight columns in
  Experiment 1's pair-capacity table to `allpairs_summary.csv`, alongside its metric means and stds.
  Pass `pair_capacity.csv` explicitly and check the schema pairs and population before joining.
  Add `Features` from the per-pair feature widths.
- `1A` run selection: `aggregate_allpairs_results.py` takes the latest CV directory per bundle by
  default. `--tag` and `--timestamp` select the CV directories written by one call of
  `scripts/run_allpairs_baselines.py`. Nothing checks that they hold 28 distinct schema pairs, each
  with 5 completed folds. The aggregator stops only when no pair is complete. A timestamp also does
  not identify one set of fits: a resumed call writes CV directories under a new timestamp, and
  those directories list folds that an earlier call fit. With `--pairs`, a call covers only the
  listed pairs. `--manifest` sets only the output directory and does not select inputs.
- Resume: `scripts/run_allpairs_baselines.py` skips a fold when its `test_predicted.csv` exists,
  and `scripts/run_balanced_train_redraw.py` skips a run when its `metrics_summary.json` exists.
  Neither checks the config, seed or code commit of the existing output.
- `1B`: a script that runs the redraws for all 28 pairs.
  `scripts/resample_balanced_train_negatives.py` and `scripts/run_balanced_train_redraw.py` run one
  pair and one fold per call. `scripts/analyze_balanced_train_redraw.py` runs one pair per call.
- `1B` overlap counts: add both requested counts to `redraw_summary.json`. The maximum shared
  count can be calculated from its maximum Jaccard because all draws have the same size. Counting
  distinct negative pairs that appear in more than one draw requires a frequency tally.
- `1B`: the detailed table. `scripts/analyze_balanced_train_redraw.py` writes one
  `per_pair_sd_by_class.csv` per schema pair, and nothing combines them into one table.

The `1B` matrix still needs a chosen cell value before its output can be specified.

### Addressing code gaps

Use the existing 28-pair build and training scripts. The dataset manifest already lists the pair
bundles, dataset directories and build exit codes. Extend it to also record the capacity CSV, the
`dataset.n_folds=5` override and the Git commit. The training runner reads it, applies the same
override when planning and fitting, and records the exact model and CV directories it uses.

Use a new dataset prefix for Task 1. Model run IDs are built from the dataset directory name, and
`run_allpairs_datasets.py` does not write into an existing dataset directory, so a new prefix keeps
Task 1 apart from Experiment 2. Never run the trainer with `--force` on Experiment 2's dataset
prefix.

Reuse `audit_pair_datasets.py` to check each dataset before fitting. It already checks the capacity
counts, the population, the pinned length of each site-feature cache and the balanced-usage
negatives. Add three checks: the expected fold count (5 for Task 1), that the audited runs cover the
28 distinct schema pairs, and that the training config matches. The training-config check must
fail, not print a note, when a codon setting differs. Require all 28 datasets to pass before the
full screen. Check that each pair's selected positive pair keys match Experiment 2; only their fold
assignment should change.

Add one `1B` coordinator that reads the dataset manifest and runs the existing redraw, fit and
analysis steps for each pair and fold with bounded concurrency. The redraw already uses
`balanced_usage_negatives` and `compute_axis_flags`; keep those implementations. Both training
scripts launch `train_pair_baselines.py`, so share that launch code. Resume a job only when its
saved config, source fold, seed, code revision and required outputs match the manifest.
Add a pair filter to the dataset builder and `1B` coordinator so a one-pair pilot uses the same
path as the full screen; the training runner already has this filter.

Build reports from the recorded run directories. Record the actual feature count at fit time and
require it to agree across folds. Make `aggregate_allpairs_results.py` select the exact CV
directories recorded by the trainer, require 28 distinct schema pairs with 5 completed folds each,
check that every fit records the same code commit in its `training_info.json`, and join all eight
Experiment 1 table columns from an explicit capacity CSV after checking pair and population
matches. Split its `main` into discovery, table and output functions; compute each matrix once.
Keep means and stds numeric in CSV. Color cells by mean and use separate text for the figure.
Extend `plot_utils.annotated_heatmap` as needed for the lower-triangle mean and ± std display.
For `1B`, write the two overlap counts during redraw and combine draw metrics and per-class
prediction SDs into one row per pair and fold. Name SD columns by what varied: folds in `1A`,
draws in `1B`.

Check changed code paths against selected four-fold Experiment 2 and 6 outputs. Then run one
five-fold pair through `1A` and `1B` before the full 28 × 5 datasets, 140 baseline fits and
28 × 5 × `r` redraw fits. Verify that training positives, validation and test pairs, codon feature
encoding and caches, and model seed remain fixed across a fold's redraws and match the source fold.
Report overlap between negative draws and flag any identical draws.
Use one Git commit with no uncommitted code changes for the reported run. Choose `r` (at least two),
its seeds and the `1B` matrix value before running all redraws.



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
