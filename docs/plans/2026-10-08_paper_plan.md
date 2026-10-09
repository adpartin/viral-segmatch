# Paper plan: influenza segment-pair classification

**Status: IN PROGRESS**

This plan is based on Jim's plan of 2026-10-08 (`plan-Jim-10-8-26-original.pdf`). That PDF is not
tracked in the repo.

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
  At a 1:1 ratio, the accepted datasets must use each selected sequence exactly once per slot in
  each split's negatives.

**Results.**
- A 28-row supplementary table with the Experiment 2 display columns: Rank, Schema pair,
  HK selected, Features, F1 macro, AUC-ROC, Precision and Recall. Keep all 8 Experiment 1
  capacity columns in the underlying CSV.
- An 8-by-8 F1 macro matrix for the main paper. Use the strictly lower triangle for the 28 schema
  pairs. Each pair's cell shows its mean test F1 macro across the 5 folds. The next line shows
  ± SD across those folds. Color the cell by the mean. See Fig 5 in
  https://academic.oup.com/bib/article/27/1/bbaf667/8422735, but show ± SD on a separate line.
  Update `src/analysis/aggregate_allpairs_results.py` to generate it.

### 1B. Measure sensitivity to the draw of training negative pairs

**Goal.** Measure how much model predictions vary on a fixed test set when we vary only the
training negative pairs.

**Methods.**
- Same as in `1A`.
- For every fold, redraw the balanced-usage training negatives `r` times, each time with a
  different random seed, and train one model per draw. The training positives, the val and test
  pairs, and the model seed stay the same. This results in `r` models per fold.
- Use `r` = 10 draws with seeds 42–51, the seeds Experiment 6 used. This is subject to the pilot's
  runtime: the full screen needs 1,400 redraw fits (28 pairs × 5 folds × 10 draws).
- The same negative pair can appear in more than one draw. We allow this, and we report how often
  it happens: the most negative pairs that any two draws of a fold share, and how many negative
  pairs appear in more than one draw.
- Each of the `r` models predicts the same test pairs. This results in `r` predictions for each
  test pair.

**Results.**
- A supplementary table with one row per schema pair and fold: 28 × `k` rows (140 at `k` = 5).
  Show the mean ± SD of test F1 macro across draws, the median per-pair probability and logit SD
  for each test class, and the two negative-pair overlap counts for that fold. These are the
  Experiment 6 measures with the overlap counts added.
- A schema-pair matrix laid out as in `1A`. Each pair's cell shows one value: the mean across
  folds of the within-fold SD of test F1 macro across draws. Label the figure with that definition.
  It measures the variation from redrawing the training negatives, which is a different source of
  variation from the SD across folds in `1A`.

### Existing materials
Experiments 1, 2, and 6 in `docs/plans/2026-09-14_codon_site_features_plan.md` cover a lot of
this on Human-H3N2-2024. Experiment 2 covered 28 schema pairs with 4-fold CV. Experiment 6 redrew **training** negatives with fixed test pairs for PB2-HA and PB2-NP.

### Code gaps
The dataset builder and single-fold redraw code exist. Task 1 still needs:

- `1A` matrix: `src/analysis/aggregate_allpairs_results.py` makes a symmetric matrix of means.
  Change the F1 macro figure to show only the lower triangle, with the 5-fold mean and SD on
  separate lines in each pair's cell.
- `1A` 5-folds: the dataset builder accepts `dataset.n_folds=5`, but the 28-pair bundles set 4
  and `scripts/run_allpairs_datasets.py` cannot pass an override. The training runner reads the
  fold count from the bundle, so dataset generation and training need the same 5-fold config.
  Going from 4 to 5 folds keeps the same `HK selected` positives, because they are selected before
  folds are assigned. It changes the split shares of the selected positives: test goes from 25% to
  20%, and train from 65% to 70%. Validation stays at 10% (`val_ratio: 0.1`).
- `1A` dataset audit: `src/analysis/audit_pair_datasets.py` expects 4 folds. The experiment
  spec will supply the fold count (see "Audit the datasets and training settings").
  `--expect_pairs 28` counts run directories, not distinct schema pairs, so two runs of one pair
  plus a missing pair would still pass. Feature settings such as `site.unit: codon`
  are printed as notes and never fail the audit.
- `1A` table: extend `src/analysis/aggregate_allpairs_results.py` to add all 8 columns in
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
- Failures: `src/models/train_pair_baselines.py` catches an error from a baseline, prints it and
  exits 0. `scripts/run_allpairs_baselines.py` marks a fold failed only on a non-zero exit, so it
  records that fit as complete and aggregates the pair. `scripts/aggregate_cv_results.py` then
  prints a warning for the fold without `test_predicted.csv` and aggregates the other folds. If
  aggregation fails, the runner prints an error but does not count it in its exit status.
- `1B`: a script that runs the redraws for all 28 pairs.
  `scripts/resample_balanced_train_negatives.py` and `scripts/run_balanced_train_redraw.py` run one
  pair and one fold per call. `scripts/analyze_balanced_train_redraw.py` runs one pair per call.
- `1B` 5-folds: `run_balanced_train_redraw.py` does not pass the 5-fold config override to the
  trainer, so a redraw fit currently resolves the bundle's 4-fold setting.
- `1B` overlap counts: add both requested counts to `redraw_summary.json`. The redraw already
  holds each draw's negative pair keys, so it can count the largest intersection between two draws
  and the distinct negative pair keys used in at least two draws.
- `1B`: the detailed table. `scripts/analyze_balanced_train_redraw.py` writes one
  `per_pair_sd_by_class.csv` per schema pair, and nothing combines them into one table.
- `1B` matrix: `scripts/analyze_balanced_train_redraw.py` writes each fold's SD of test F1 macro
  across draws (`f1_macro_std` under `test_metrics_across_seeds` in `summary.json`), one pair per
  call. Nothing averages it over folds for all 28 pairs or draws the matrix.

### Addressing code gaps

Run the work in this order:

1. Make the code changes below and write both experiment specs.
2. Run the 4-fold regression check on PB2-HA and PB2-NP (see "Verify before the full screen").
3. Run the one-pair 5-fold pilot through `1A` and `1B` (same section).
4. Commit the code and experiment specs, and confirm a clean worktree.
5. Build and audit all 28 5-fold datasets.
6. Run the 140 `1A` fits.
7. Run the 1,400 `1B` redraw fits.
8. Build the tables and figures.

#### Build and identify the 5-fold datasets

Use the existing 28-pair build and training scripts and the 28 `flu_28p_codon_*` child bundles. Add
no new bundles. Before the build, write an experiment spec (see `docs/methods/glossary.md`)
for each run: `conf/experiments/human_h3n2_2024_28p_codon_balanced_cv5.yaml` for Task 1 and
`conf/experiments/human_h3n2_2024_28p_codon_balanced_cv4.yaml` for the 4-fold regression check
below. Write each one independently of any dataset, so the audit never checks a dataset against
settings copied from that dataset. Each experiment spec has two sections:

- `overrides`: an ordered list of the config values that differ from the child bundles. The Task 1
  experiment spec sets only `dataset.n_folds=5`. The 4-fold experiment spec sets none.
- `expect`: the expected dataset and training settings: population, positive selection, negative
  sampler and ratio, pinned-length requirement, fold count, feature settings and LightGBM row
  sampling.

A key that appears in both sections must have the same value in both. A small experiment spec reader
returns the override list. The dataset build and every fit pass that list as `--override` to the
builder (`src/datasets/dataset_segment_pairs.py`) and the trainer
(`src/models/train_pair_baselines.py`), which already accept it. Each runner also uses the same
list for its planned config: the config it composes from a child bundle and the override list
before it launches a fit. Keep `get_virus_config_hydra` unchanged.

The dataset manifest already lists the pair bundles, dataset directories and build exit codes.
Extend it with:

- Settings: the experiment spec's path and checksum, the capacity CSV, the override list
  read from the experiment spec, the fold count the builder used (`dataset.n_folds`) and the Git
  commit. Record the fold count itself, not only an override: the 4-fold regression rebuild uses
  the bundle's setting and has no override.
- Split files: each dataset's split directories, relative to the dataset directory (`fold_0` to
  `fold_4` for Task 1; `.` for a single holdout split, such as Task 3's temporal holdout, which
  writes its files at the dataset root). Add a checksum of each dataset's `resolved_config.yaml`,
  and checksums of each split directory's `{train,val,test}_pairs.csv` and
  `cooccurring_sequence_pairs.csv`.
- Cache files: for each selected protein `P`, checksums of `site_features_codon_{P}.npz`,
  `site_features_codon_{P}_index.parquet` and `site_features_codon_{P}_metadata.json` under
  `data/embeddings/flu/July_2025/`. Check these before training and after all fits.

Keep the dataset manifest unchanged after the build.

The audit, the training runner (`scripts/run_allpairs_baselines.py`) and the `1B` coordinator
(`scripts/run_allpairs_balanced_redraws.py`, described below) read the split directories and fold
count from the dataset manifest, and the required fold count from the experiment spec.
Each fails unless they match: 5 folds under the Task 1 experiment spec and 4 under the regression
experiment spec. Each also requires `dataset.n_folds` in the dataset's `resolved_config.yaml` and in
every planned fit config to equal the manifest's fold count. A single holdout split is not
cross-validation and must never be reported as 5-fold CV.

Use a new dataset prefix for Task 1. Model run IDs are built from the dataset directory name, and
`run_allpairs_datasets.py` does not write into an existing dataset directory, so a new prefix keeps
Task 1 apart from Experiment 2. Never run the training runner (`scripts/run_allpairs_baselines.py`)
with `--force` on Experiment 2's dataset prefix. A pilot with code edits must use a new dataset
prefix.

#### Audit the datasets and training settings

Reuse `audit_pair_datasets.py` to check each dataset before fitting. It already checks the capacity
counts, the population, the pinned length of each site-feature cache and the balanced-usage
negatives. Replace its hard-coded `DATASET_EXPECT` and `TRAINING_EXPECT` values with the
experiment spec's `expect` section. Check that every split directory in the dataset manifest exists,
and for Task 1 require exactly `fold_0` through `fold_4`. Read the dataset runs from the dataset
manifest, not a directory glob. Take the expected pair set from the selected child bundles; require
exactly one run per selected pair, no extra runs, and exactly one matching row per population and
pair in the capacity CSV. The audit currently indexes that CSV by `Schema pair` alone, which works
only while it holds one population. The full run must cover all 28 capacity pairs. Audit the
dataset's resolved config, and make the training runner check its planned config before launching
any fits. Fail if either differs from the experiment spec's `expect` section. For Task 1, `expect`
sets 5 folds, the feature settings (`training.feature_source=site`, `site.unit=codon`,
`site.encoding=ordinal`, `training.interaction=concat`, `training.slot_transform=none`) and the
LightGBM row-sampling settings (`baseline_lgbm.subsample=0.7`, `baseline_lgbm.subsample_freq=1`). Require
every selected dataset to pass before fitting; the full screen requires all 28. For each pair,
require the union of positive test pair keys across its 5 folds to equal the union of positive test
pair keys across the 4 folds of `exp2_28p_codon_balanced_20261007_180037_{pair}`. Fold assignments
and sampled negative pairs may differ.

#### Run the `1A` fits

The training runner, `scripts/run_allpairs_baselines.py`, reads the dataset manifest, applies the
experiment spec's override list when planning and fitting, and writes a separate
`training_run_manifest.json` with the dataset manifest path, code commit, and exact model and CV
paths.

A fit's input checksums cover its fold's `{train,val,test}_pairs.csv` and
`cooccurring_sequence_pairs.csv`, the dataset's `resolved_config.yaml`, and the codon cache files
listed in the dataset manifest. A fit's required outputs are `best_model.joblib`,
`resolved_config.yaml`, `metrics_summary.json`, `training_info.json` and
`{train,val,test}_predicted.csv`.

Save the input checksums in each fit's `training_info.json`. Before fitting or skipping a `1A`
fold, require the current commit and the dataset files to match the dataset manifest. Before
skipping, require the fit's saved resolved config to match the planned config, its input checksums
and code commit to match the dataset manifest, and all required outputs to exist. Otherwise fail
without overwriting the fit. The runner must make this check before it launches the trainer,
because the trainer reuses an existing run directory and overwrites its files. The runner creates
each fit's directory and `train.log` before launching the trainer, so a fit that crashes leaves a
partial directory, and a resume stops there. Removing that directory is a deletion and needs the
user's approval. A pilot can use a new dataset prefix instead.

A failed fit or aggregation must end with a non-zero exit status:

- The trainer runs every requested baseline, then exits non-zero if any of them failed. One failed
  baseline still does not stop the others.
- `scripts/aggregate_cv_results.py` gets a strict mode that fails when a fold has no
  `test_predicted.csv`. Task 1 uses it. Its other callers, `scripts/run_cv_lambda.py` and
  `scripts/run_cv_polaris.pbs`, keep the current behaviour, which aggregates the folds that exist.
- The training runner exits non-zero when aggregation fails.

`training_run_manifest.json` assigns one CV directory to each pair. A resume call must name the
existing `training_run_manifest.json`, reuse its model and CV directories, and require the current
commit to match its recorded commit. On resume, the runner rebuilds each CV directory's summary
files (`cv_run_manifest.json`, `status.json` and the `aggregate_cv_results.py` outputs) from its
folds. These summary files are the only outputs a resume overwrites; fits and redraw directories
are never overwritten.

#### Run the balanced negative redraws

Add one `1B` coordinator, `scripts/run_allpairs_balanced_redraws.py`. It reads the dataset manifest
and runs the existing steps for each pair and fold with bounded concurrency: the redraw script
(`scripts/resample_balanced_train_negatives.py`), the fits, and the analysis script
(`scripts/analyze_balanced_train_redraw.py`). The redraw script copies `val_pairs.csv`,
`test_pairs.csv` and `cooccurring_sequence_pairs.csv` from the dataset fold it redraws, called its
source fold, into each seed directory, and writes a new `train_pairs.csv` there. A `1B` fit reads
the files in its seed directory, and its input checksums cover those files instead of the source
fold's. Before any fit, check the planned config, current commit and source-fold file hashes
against the dataset manifest and the experiment spec. Reuse the functions the redraw script already
calls: `balanced_usage_negatives` from `src/datasets/_negative_sampling.py` and
`compute_axis_flags` from `src/datasets/dataset_segment_pairs_v2.py`.

Before redrawing, require the dataset to meet the redraw script's assumptions: Hopcroft-Karp
positives (`dataset.positive_pair_selection.method=hopcroft_karp`), balanced-usage negatives
(`dataset.split_strategy.negative_scope=balanced`) at a 1:1 ratio (`dataset.neg_to_pos_ratio=1.0`),
`dataset.negative_sampling: null`, and a passed one-use audit. The one-use audit is
`check_balanced_usage` in `audit_pair_datasets.py`. In every split of every fold, it requires each
slot's sequences to appear as often among the negatives as among the positives (once each, for
`HK selected` positives), and no negative to be duplicated, observed as a co-occurrence or reused
across splits. Fail otherwise. The redraw script always uses `balanced_usage_negatives`, so on a
metadata-matched dataset it could produce unmatched negatives.

`scripts/run_allpairs_baselines.py` and `scripts/run_balanced_train_redraw.py` both launch
`train_pair_baselines.py`. Move that launch code into one shared function, which the coordinator
also uses. The existing single-fold redraw commands keep working through it. Every `1A` and `1B` fit
gets the experiment spec's override list, which sets `dataset.n_folds=5` for Task 1.

The coordinator writes `redraw_run_manifest.json` with the dataset manifest path, redraw seeds,
code commit, model directories, analysis output directories and checksums of the redrawn
`train_pairs.csv` files. Add a `--neg_redraw_seed` argument to the trainer. The trainer only
records the value: it saves it as `neg_redraw_seed` in `training_info.json`, with the fit's input
checksums. A fit whose training negatives come from the builder records `neg_redraw_seed: null`.
The coordinator checks each fit's `neg_redraw_seed` against `redraw_run_manifest.json`.

A resume call must name the existing `redraw_run_manifest.json`; reuse its run prefix without
creating a new timestamped run, append missing job records and fail if the current commit differs
from its recorded commit. Before skipping an existing redraw directory, require its seed
directories to match the seed list in `redraw_run_manifest.json`, and require the checksums of all
its files to match that manifest: each seed's `{train,val,test}_pairs.csv` and
`cooccurring_sequence_pairs.csv`, and the fold's `redraw_summary.json`. Fail without overwriting a
partial or mismatched redraw directory. Before skipping a fit, require its source fold's files to
match the dataset manifest and its redrawn `train_pairs.csv` to match `redraw_run_manifest.json`.
Also require its saved config, input checksums, `neg_redraw_seed`, code commit and required outputs
to match the planned job. Otherwise fail without overwriting. As for `1A`, make these checks before
launching the trainer; a crashed fit leaves a partial directory that stops a resume. The
coordinator exits non-zero when any redraw, fit or analysis step fails. If two seeds for the same
fold produce identical sets of training negative pairs, record the seed pair in the manifest and
fail without assuming why.

Add a pair filter to `scripts/run_allpairs_datasets.py` and the `1B` coordinator so a one-pair pilot
uses the same path as the full screen; the training runner already has this filter (`--pairs`).
Also give `run_allpairs_datasets.py` an optional `--bundle-prefix` argument whose default is its
current prefix, `flu_28p_codon_`. Existing commands keep working, and later populations can use the
same runner.

#### Build the tables and figures

`src/analysis/aggregate_allpairs_results.py` builds both reports from the recorded run directories.
Its `--manifest` argument already sets the output directory, so add two new arguments:
`--training_run_manifest` selects the `1A` CV directories listed in `training_run_manifest.json`,
and `--redraw_run_manifest` selects the `1B` model and analysis directories listed in
`redraw_run_manifest.json`. The coordinator only runs jobs and does not build reports.

For both reports:

- For the final report, require every fit's `training_info.json` to record the same `code.commit`
  as the dataset manifest and `code.is_dirty=false`.
- Name SD columns by what varied: folds in `1A`, draws in `1B`.
- Split `main` into discovery, table and output functions, and compute each matrix once.
- Extend `plot_utils.annotated_heatmap` as needed for the lower-triangle display.
- Leave per-category reporting to Task 2, which must first decide how to assign a pair whose two
  sides differ in host, subtype or year, or lack that metadata.

For `1A`:

- Recompute each fold's test metrics from its `test_predicted.csv` and require them to match its
  `metrics_summary.json`.
- Require 5 completed folds for each selected pair. Require all 28 pairs for the final report, but
  allow the selected pair set for a pilot report.
- Save `estimator.n_features_in_` as `n_features` in each fit's `training_info.json`. Require the
  value to agree across folds and use it for the `Features` column.
- Join all 8 Experiment 1 table columns from an explicit capacity CSV after checking pair and
  population matches.
- Keep the F1 macro fold means and SDs numeric in CSV.
- In the figure, color each pair's cell by its mean and annotate it with the mean and ± SD on
  separate lines.

For `1B`:

- The redraw script writes the two overlap counts to `redraw_summary.json` during the redraw.
- Combine the draw metrics, per-class prediction SDs and overlap counts into one row per pair and
  fold.
- In the figure, each pair's cell shows one value: the mean across folds of the within-fold SD of
  test F1 macro across draws. Label the figure with that definition. The per-fold values stay in
  the supplementary table.

#### Verify before the full screen

Before the full run, rebuild PB2-HA and PB2-NP under the 4-fold experiment spec
(`human_h3n2_2024_28p_codon_balanced_cv4.yaml`), and run the audit, the training runner and the
`1B` coordinator with it. Require their `{train,val,test}_pairs.csv` files to match the
corresponding fold files under
`exp2_28p_codon_balanced_20261007_180037_pb2_{ha,np}` byte-for-byte. Redraw seed 42 for fold 0
of each pair and require `train_pairs.csv` to match the corresponding file under
`exp6_negative_redraw_20261007_224316_pb2_{ha,np}/fold_0/seed_42/` byte-for-byte. Run one new fit
twice on the same inputs and require identical test predictions. Compare a new refit with the
historical Experiment 2 predictions, but treat that comparison as diagnostic: those fits recorded
`code.is_dirty=true`, so a difference needs an explanation rather than an automatic failure.

Run one 5-fold pair through `1A` and `1B` with at least two redraw seeds. The pilot audit and
report must accept that one selected pair. Verify that training positives, validation and test
pairs, codon feature settings and caches, and model seed remain fixed across each fold's redraws
and match the source fold. The development pilot may record a dirty worktree, but its outputs must
not enter the final report. If the pilot's runtime makes 10 draws too costly, choose a smaller `r`
(at least two) before running all redraws.

Commit the code changes before building the final datasets, so their manifest and every final fit
record the same commit. Make no tracked edits during the reported run. Require empty output from
`git status --porcelain --untracked-files=all -- src scripts conf` before and after it; this checks
the code and config directories used by Task 1.

Then build and audit all 28 datasets (5 folds each), run 140 baseline fits and 1,400 redraw fits
(140 × 10 draws), and produce the full report.



## 2. Evaluate across years, hosts, and HN subtypes

**Goal.** Build one model (or several) that covers several hosts, subtypes and years, and check
how well it predicts for each of them.

**Methods.**
- Metadata-aware negative sampling: each negative should pair sequences from two isolates with the
  same host, subtype and year. This is the `host_subtype_year` regime in
  `src/datasets/_negative_regime_sampling.py`, requested with
  `dataset.negative_sampling.regime_targets`.
- Only the default sampler (`negative_scope: coverage`) takes regime targets. The `balanced`
  sampler does not: `src/datasets/_negative_sampling.py` sets `neg_regime` to a placeholder marked
  "not wired". Balanced-usage negatives that also match on metadata need new sampler code.
- Regime targets do not guarantee matched negatives. The coverage sampler's coverage phase ignores
  regimes unless its `regime_aware_coverage` option is on. With that option on, it still falls back
  to a uniform draw when no regime yields a partner. Audit the achieved regime of every negative
  before reporting that the negatives are matched.
- Regime comparisons use year bins by default (`<=2015`, `2016-2020`, `2021+`), not the exact year.
  `dataset.negative_sampling.year_match: exact` compares exact years.
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
- Hopcroft-Karp selection and the temporal holdout cannot be combined today. Active
  `dataset.positive_pair_selection` requires `dataset.n_folds >= 2`, and `dataset.metadata_holdout`
  allows only a single split (`n_folds` null or 1). Using Task 1's `HK selected` positives in Task 3
  needs a builder change.
- E.g., `train: {year_range: [2015, 2024]}` and `test: {year: [2025]}`; `year_range` is inclusive.
  The bundle `conf/bundles/flu_ha_na_h3n2_nt_cds_temporal_2025.yaml` already does this for HA-NA in
  H3N2.
- Compare the above with a random split within 2025: filter the population to 2025
  (`dataset.year: [2025]`) and split it at random. Do not give `metadata_holdout` the same train and
  test years: it requires disjoint train and test isolate sets and fails otherwise.
- E.g., `train: {year_range: [2015, 2020]}` and `test: {year_range: [2021, 2025]}`.
  Break down performance by test year.
- Does prediction uncertainty increase as we predict further into the future?

**Results.**
- Plot of F1 vs Year.
- ...
