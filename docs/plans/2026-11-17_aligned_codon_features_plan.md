# Aligned codon features

**Status: IN PROGRESS**

## Goal

Build per-site codon features from the final protein alignments of the Task 2 population, so
that every protein has one shared set of sites without a pinned CDS length. Tasks 1 and 2 in
`docs/plans/2026-10-08_paper_plan.md` both use these features.

The alignments are done for all 8 proteins. Their method and results are in
`docs/results/2026-11-09_protein_alignment.md`.

## Decisions

- Features: codons placed in protein-alignment columns, not at a pinned CDS length.
- Records: the CDS kept by the retention rule (`docs/results/2026-11-09_protein_alignment.md`,
  section "Retention rule").
- Sites: every column of the final alignment, plus one column after the last alignment column.
  Stop codons are placed by the **stop codon placement rule**, defined in
  `docs/results/2026-11-09_protein_alignment.md`, section "Method". The extra column holds the
  stop codon of each CDS whose last protein symbol is in the last alignment column.
- Proteins: the 8 major proteins.
- Alignments: Tasks 1 and 2 both use the alignments of the Task 2 population, so a feature
  column is the same alignment column in both tasks.
- Site settings: aligned mode requires `site.unit: codon` and `site.encoding: ordinal`.
  `get_virus_config_hydra` in `src/utils/config_hydra.py` rejects any other combination.
- Provenance: aligned features are fit only on datasets built in aligned mode on the same final
  alignments. A pinned dataset is never fit with aligned features, including as a comparison.
  `check_aligned_dataset` in `src/models/train_pair_baselines.py` enforces it (step 6).

The stop codon placement rule is a convention. The downstream contig check in
`docs/results/2026-11-05_ns1_alignment_pilot.md` supports it for the 654- and 660-nt NS1 CDS of
13 Human-H3N2-2024 isolates. That covers 9 of the 25,506 Task 2 isolates with a 660-nt NS1 CDS.
The rule is unverified for the other proteins.

The alignment runs once per population and protein with `src/analysis/align_proteins.py`,
outside the dataset build. The dataset build has the steps [1] to [8] shown under "Dataset build
steps" in `docs/plans/2026-10-08_paper_plan.md`. There, the retention rule filter (step 6) is
applied in pair eligibility [3], where it replaces the requirement of complete CDS at pinned
lengths. Building, filtering and loading in aligned mode use no pinned CDS length.

The caches are written beside the alignment they come from, not in `data/embeddings/`, because
each cache belongs to one alignment of one population.

## Steps

1. Configure the alignment directories and create the master bundles for Tasks 1 and 2.

   The final alignments are under
   `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/`:
   - `<protein>/length_share_above_0.02/` for PA, NP, M1 and NS1.
   - `<protein>/length_share_above_0.02/gap_pattern_share_above_0.02/` for PB2, PB1, HA and NA.

   A new site config, `conf/site/aligned_4host_3subtype_2010_2025.yaml`, starts from `default`,
   sets `site.mode: aligned`, and maps each protein to its directory in `site.alignment_dirs`.
   Building, filtering and loading all read this one mapping. `conf/site/default.yaml` sets
   `site.mode: pinned`. Both master bundles select the aligned config with
   `override /site: aligned_4host_3subtype_2010_2025` in their defaults. This keeps inherited
   settings such as `site.unit`.

   A *master bundle* holds the settings shared by all schema pairs of one experiment and sets no
   schema pair. This step creates two:
   - Task 1: `flu_28p_human_h3n2_2024_site_codon_aligned`. It inherits from
     `flu_28p_human_h3n2_2024_site_codon`, which keeps its name for the pinned runs. It sets
     `dataset.require_complete_cds_at_pinned_length: false`, because the parent sets it to true
     and step 6 rejects it in aligned mode.
   - Task 2: `flu_28p_4host_3subtype_2010_2025_site_codon_aligned`. It inherits from the Task 1
     aligned master and sets only the population: hosts Human, Swine, Chicken and Duck; subtypes
     H3N2, H1N1 and H5N1; `year: null` and `year_range: [2010, 2025]`. It therefore also
     inherits Task 1's 4-fold CV, Hopcroft-Karp positive selection and negative-sampling
     settings. These are inherited defaults pending Task 2's experiment configuration, which
     sets the fold, negative-sampler, year-matching and thin-group settings described under
     Task 2 in `docs/plans/2026-10-08_paper_plan.md`.

   Each run uses one child bundle per schema pair, as `flu_28p_codon_*.yaml` do for the pinned
   master. This plan creates one child per master, to test step 6. The other children are
   created with the runs.

2. Add the mapping script `src/embeddings/compute_protein_aligned_codon_features.py`. Read
   `aligned_proteins.fasta` with `read_fasta` in `src/utils/fasta_utils.py`, and read
   `selected_cds.parquet`. Join each CDS to its aligned row by `prot_hash`, which names the
   FASTA records. Require `all_passed: true` in `alignment_validation.json` and require its
   `aligned_fasta_sha256` to equal the FASTA's sha256. Check the FASTA and selected-CDS hashes
   against `alignment_manifest.json`. Reuse `encode_codon` unchanged. Group CDS by nucleotide
   length before encoding, because `sequences_to_byte_matrix` requires equal-length sequences.

3. Place codons in alignment columns. Each feature row represents one unique CDS, keyed by
   `cds_dna_hash`. Place every codon except the stop codon in its corresponding non-gap
   protein column, including codons represented by X. Place the stop codon by the stop codon
   placement rule. Fill remaining empty columns before the stop codon with `GAP`, and those after
   it with `AFTER_STOP`.

   Each protein's feature width is W + 1, where W is `aligned_width` in its final
   `alignment_validation.json`. On 2026-11-17 the final alignments gave:

   | | PB2 | PB1 | PA | HA | NP | NA | M1 | NS1 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|
   | W | 759 | 758 | 716 | 581 | 498 | 493 | 252 | 230 |
   | W + 1 | 760 | 759 | 717 | 582 | 499 | 494 | 253 | 231 |

   Neither config nor code sets these widths.

4. Write caches and a readable table beside each final alignment. Use the stem
   `site_features_codon_<protein>_aligned` with the existing `.npz`, `_index.parquet` and
   `_metadata.json` layout.

   Keep the existing metadata fields whose meaning does not change: `unit`, `protein`,
   `function`, `n_sites`, `n_sequences`, `source_column`, `index_key`, `code_map`, `other_code`,
   `n_sites_on_other_code` and `timestamp`. Omit `pinned_nt`, `n_unique_cds`,
   `n_dropped_incomplete`, `n_dropped_off_pinned_length`, `n_rows_decode_checked` and
   `input_file`, which describe the pinned build's corpus input, filters and sample decode check.
   Add these fields:
   - `site_mode`: `"aligned"`;
   - `alignment_dir`: the final alignment directory;
   - `aligned_fasta_sha256` and `selected_cds_sha256`: the sha256 of `aligned_proteins.fasta`
     and of `selected_cds.parquet`, under the names `alignment_manifest.json` uses;
   - `stop_codon_placement_rule`: the rule in one sentence;
   - `n_rows_validated`: the number of rows that passed the row and position checks of step 7;
   - `config_bundle` and `code`: the bundle the script ran with, and the commit record from
     `get_git_info` in `src/utils/experiment_utils.py`.

   Give `GAP` the id 97 and `AFTER_STOP` the id 98. Both are above every id in GenSLM's tokenizer
   (0–96) and fit in the cache's uint8 codes. Add both to `code_map`, so that it declares every
   code in the matrix. `read_aligned_codon_cache` rejects a cache whose matrix holds an undeclared
   code. The two ids are defined once, as `GAP_CODE` and `AFTER_STOP_CODE` in
   `src/utils/site_utils.py`.

   The readable table is `site_features_codon_<protein>_aligned_table.parquet`, beside the other
   cache files. It is keyed by `cds_dna_hash`, with one row per unique CDS and one text column per
   feature column, named `site_1` to `site_<W + 1>`. Each cell contains the codon triplet in
   uppercase, `GAP` or `AFTER_STOP`, preserving ambiguous triplets.

5. Update loading and training. `site.mode` chooses **aligned** or **pinned** mode. Its default
   is `pinned`, so existing bundles keep their current behaviour. Aligned loading reads each
   protein's cache from `site.alignment_dirs` and needs no pin. Neither mode falls back to the
   other. The code to change:
   - `load_site_cache` in `src/utils/site_utils.py`, which reads `metadata['pinned_nt']`;
   - `load_pair_features_for_baselines` in `src/models/_pair_features.py` and `input_checksums`
     in `src/models/train_pair_baselines.py`, which take one site directory plus pins;
   - the error text in `_slot_matrix` in `src/utils/site_utils.py`, which names the pinned
     length.

   `load_site_cache` must tell the two kinds of cache apart before it resolves a pin. Today,
   when no pin is given, `resolve_site_cache_pin` looks only for
   `site_features_<unit>_<protein>_*nt_metadata.json`, so an aligned cache is reported as
   missing, with advice to build a pinned one. In pinned mode, when the directory holds
   `site_features_<unit>_<protein>_aligned_metadata.json`, raise an error that says the caller
   supports pinned caches only. Aligned caches load only in aligned mode, and their metadata
   must have `site_mode: aligned`. A cache whose metadata has no `site_mode` is a pinned cache,
   so caches built before this change still load.

   The batch scripts in `scripts/` find caches by pin through `site_cache_files` in
   `src/utils/dataset_manifest.py` and `check_caches` in `src/analysis/audit_pair_datasets.py`.
   Both must raise in aligned mode until they support aligned caches.

   Four analysis scripts in `src/analysis/` stay pinned-only: `plot_site_shuffle_refit.py`,
   `plot_site_shuffle_fixed.py`, `plot_negative_pair_ambiguity.py` and
   `compare_negative_pair_distances.py`. They take the site directory and pin from their own
   arguments, so they cannot read `site.mode`. They load caches through `load_site_cache`, so its
   pinned-mode check rejects an aligned cache in all four.

   Done on 2026-11-19. `site_mode`, `configured_site_cache_paths` and `load_configured_site_cache`
   in `src/utils/site_utils.py` pick each protein's cache by `site.mode`. Aligned caches are read
   by `read_aligned_codon_cache`. Besides the metadata, the matrix and the index, it checks that
   the width is `aligned_width` + 1 and that `aligned_proteins.fasta` and `selected_cds.parquet`
   still have the sha256 the cache recorded. `load_pair_features_for_baselines` takes the two
   loaded caches, and `train_pair_baselines` loads either kind. `site_cache_files` and
   `check_caches` still raise in aligned mode. Step 9 adds aligned support to them.

6. Add the retention rule filter to pair eligibility [3]. In aligned mode, a record is
   eligible only if its `(assembly_id, function, cds_dna_hash)` is in the final
   `selected_cds.parquet`. This replaces the requirement of complete CDS at pinned lengths.
   It reads `cds_dna_hash`, `cds_length` and `is_complete_cds` from `cds_dna_final.parquet`
   itself, because `src/datasets/dataset_segment_pairs.py` attaches `cds_dna_hash` only when
   `pair_key_alphabet` is `nt_cds`. Require the dataset's population to lie inside
   the alignment's: every host, subtype and year the dataset selects must be among those in
   `selection_manifest.json`. Compare years as sets, so that `year: [2024]` lies inside
   `year_range: [2010, 2025]`. A dataset that leaves host, subtype or year unset fails, because it
   would select records outside the alignment's population. Require `geo_location` and `passage`
   to be unset, because the selection does not filter on them. Reject use together with
   `dataset.require_complete_cds_at_pinned_length`.

   The filter must explain every record it drops. A record kept by population filtering [2] may
   be absent from the final `selected_cds.parquet` only for one of these reasons, read from the
   saved selections:
   - its CDS is not complete (`is_complete_cds` is false);
   - the retention rule dropped its CDS length: its CDS is complete and its length is in
     `length_filter.dropped_lengths` of `<protein>/length_share_above_0.02/selection_manifest.json`.
     The plain selection `<protein>/selected_cds.parquet` is not used, because
     `align_proteins.py` writes it in a separate run, not as the parent of the length-filtered
     selection;
   - the retention rule dropped its gap pattern (PB2, PB1, HA and NA only): the record is in
     `length_share_above_0.02/selected_cds.parquet` but not in
     `length_share_above_0.02/gap_pattern_share_above_0.02/selected_cds.parquet`. The parent's
     file must still have the `parent_selected_cds_sha256` the gap-pattern selection recorded.

   Any other absent record fails the build, because dropping it silently would change the
   population. A record with no row in `cds_dna_final.parquet` also fails. Report the number of
   records dropped for each reason. The 2D-CD builder, `src/datasets/dataset_pairs_cc.py`, does
   not apply the filter and rejects aligned mode.

   Add the provenance check (see Decisions) to `train_pair_baselines`. In aligned mode it fits
   only a dataset whose `resolved_config.yaml` has `site.mode: aligned` and the same
   `site.alignment_dirs` entries as the training config for the fit's two proteins. It reads that
   file from `--dataset_resolved_config`, which is therefore required in aligned mode.

   Done on 2026-11-19. `filter_records_kept_by_retention_rule` and
   `check_population_inside_alignments` are in `src/datasets/_pair_helpers.py`.
   `dataset_segment_pairs.py` calls both in aligned mode and writes the drop counts to
   `records_kept_by_retention_rule.json` in the dataset directory. `_validate_v2_config` rejects
   `require_complete_cds_at_pinned_length` in aligned mode, and `check_aligned_dataset` is in
   `src/models/train_pair_baselines.py`.

7. Validate and update documentation.

   On 2026-11-19 all seven pinned-cache comparisons passed (157,831 rows).
   See [Codon cache checks](../results/2026-11-09_protein_alignment.md#codon-cache-checks).
   The aligned Task 1 PB1-HA build is described in `docs/plans/2026-10-08_paper_plan.md`. The
   aligned Task 2 build has not been run.

   The mapping script runs the row and position checks below on every row before it writes.

   Row check: after removing `GAP` and `AFTER_STOP`, the row's codes must equal the codes
   `encode_codon` gives for the original CDS, and the readable table's joined triplets must
   equal `cds_dna_seq.upper()`, including ambiguous bases.

   Position check: compare both outputs with the row's protein in `aligned_proteins.fasta`.
   Every column must hold what step 3 places there.

   Compare with the existing pinned caches, `site_features_codon_<protein>_<pin>nt` in
   `data/embeddings/flu/July_2025/`. Every aligned row at the lengths below must be in the pinned
   cache for that length and must match it. A missing hash fails the check. Compare full rows,
   except for PB1 2274 nt: its first 758 columns, including the stop, must match, and column 759
   must be `AFTER_STOP`. HA and NA are not compared, because every kept HA and NA row has internal
   gaps. NS1 660 nt and HA 1704 nt have no pinned cache.

   | Protein | CDS length (nt) | Unique CDS in the current final selection |
   |---|---:|---:|
   | PB2 | 2280 | 36,878 |
   | PA | 2151 | 35,609 |
   | NP | 1497 | 27,086 |
   | M1 | 759 | 14,572 |
   | PB1 | 2277 | 3,767 |
   | PB1 | 2274 | 30,822 |
   | NS1 | 693 | 9,097 |

   The counts were measured on 2026-11-17 from each protein's final `selected_cds.parquet`. The
   check tests the rule above, not these numbers.

   Record the check results in `docs/results/2026-11-09_protein_alignment.md`. Update the
   `site_feature_columns` docstring, the comments in `conf/site/default.yaml`, and the glossary
   entries for *Site*, *Trailing, internal and leading gap*, `GAP` and `AFTER_STOP`.

   In `docs/plans/2026-10-08_paper_plan.md`, describe the aligned dataset builds of Tasks 1 and
   2. Each description uses the stages of the existing "Dataset build steps" diagram, with that
   build's counts, plus the number of records step 6 dropped for each reason. Keep the existing
   pinned-length diagram, labelled as the record of the pinned Human-H3N2-2024 run.

8. Update `src/analysis/plot_site_importance.py` for aligned caches. Its site labels must name
   protein-alignment columns 1 to W, plus column W + 1, the extra column after the alignment.
   Until this step is done, the pinned-mode check in `load_site_cache` (step 5) rejects aligned
   caches in this script too.

9. Add aligned support to the batch runners and the dataset audit. `site_cache_files` in
   `src/utils/dataset_manifest.py` and `check_caches` in `src/analysis/audit_pair_datasets.py`
   name and check caches by pin, and both raise in aligned mode. `scripts/run_allpairs_datasets.py`
   and `scripts/run_balanced_train_redraw.py` call `site_cache_files` directly.
   `scripts/run_allpairs_baselines.py` and `scripts/run_allpairs_balanced_redraws.py` reach it
   through `pair_cache_checksums`. The 28-pair runs of Tasks 1 and 2 therefore cannot start in
   aligned mode until this step is done.

   Both functions take each protein's cache files from `configured_site_cache_paths`. The step is
   done when two tests pass. For an aligned fit, the checksums `fit_input_checksums` expects must
   equal the ones `train_pair_baselines` records. The audit must reject an aligned dataset whose
   cache is missing or fails `read_aligned_codon_cache`.

## Evidence

- Config selection (2026-11-18). On a copy of `conf/`, a bundle inheriting from
  `flu_28p_human_h3n2_2024_site_codon` selected a test site config with `override /site:`. It
  kept its parent's `site.unit: codon` and population (Human, H3N2, 2024). A bundle inheriting
  from that bundle kept the test site config, replaced the population, and inherited
  `n_folds: 4`. Without the `override` keyword, Hydra fails with "Multiple values for site".
- CDS rows (2026-11-18). All 868,240 records of the 8 proteins in `protein_final.parquet` had a
  row in `cds_dna_final.parquet`.
- Task 1 records under the Task 2 retention rule (2026-11-18). Counted on Human-H3N2-2024
  records in each protein's selections:

  | | PB2 | PB1 | PA | HA | NP | NA | M1 | NS1 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | Records with a complete CDS | 5,341 | 2,958 | 5,341 | 5,339 | 5,327 | 5,181 | 5,345 | 5,337 |
  | Dropped by the retention rule | 8 | 0 | 5 | 0 | 0 | 5 | 0 | 4 |
- Aligned feature loading (2026-11-19). Features were loaded for all 5 folds of the pinned PB1-HA
  Task 1 dataset `task1_28p_codon_matched_20261015_194607_pb1_ha`, and no model was fit. Every
  pair's CDS hashes were in the aligned caches. Every PB1 CDS in that dataset is 2,277 nt, and its
  759 aligned PB1 columns equal the pinned cache's for all 2,784 pairs of each fold. This checks
  loading only. The provenance rule forbids fitting a model on this combination.
- Retention rule filter on the two populations (2026-11-19). The filter ran on the builder's
  population records of all 8 proteins, without building pairs. On Human-H3N2-2024 its counts
  of complete CDS and of records dropped by the retention rule equal the table above for every
  protein. On the Task 2 population it kept 475,383 of 486,008 records, and every dropped record
  had one of the three reasons. HA's counts there equal its selection manifest: 60,751 records,
  469 incomplete and 59,807 kept.
- Aligned Task 1 PB1-HA build (2026-11-19), `task1_28p_aligned_codon_matched_20261119_145832_pb1_ha`,
  with `dataset.n_folds=5`. The retention rule dropped 0 PB1 and 0 HA records. The filter
  dropped 2,388 PB1 and 7 HA records for incomplete CDS. Aligned features loaded for all 5 folds,
  1,341 columns each, and no model was fit. The build differs from the pinned Task 1 build by 13
  PB1 records at 2,274 nt, which the retention rule keeps and the 2,277-nt pin drops, so its
  selected positives are 1,401 rather than 1,392.

## Code and artifacts

| item | status | purpose and contract |
|---|---|---|
| `src/analysis/align_proteins.py` | built | selects and validates complete CDS, prepares unique protein sequences, runs MAFFT with `--align_only`, checks exact input recovery with `--validate_only`, and records gap positions with `--inspect_only` |
| `results/flu/July_2025/cds_alignment_pilot/` | alignment inspected | `<population>/<protein>/` holds the selection, alignment and validation outputs plus `gap_locations.csv` and `inspection_manifest.json`. All 8 proteins for the Task 2 population; NS1 only for `Human-H3N2-2024` |
| `conf/site/aligned_4host_3subtype_2010_2025.yaml` and the two master bundles (step 1) | built | the aligned site config, the Task 1 and Task 2 master bundles, and their PB1-HA children `flu_28p_aligned_codon_pb1_ha` and `flu_28p_4host_aligned_codon_pb1_ha` |
| `src/embeddings/compute_protein_aligned_codon_features.py` (steps 2 to 4) | built | writes the aligned codon caches and readable tables, after the row and position checks of step 7 |
| `site_features_codon_<protein>_aligned*` in each final alignment directory | built 2026-11-18 | all 8 proteins of the Task 2 population. All 231,782 rows passed the row and position checks |
| `read_aligned_codon_cache`, `configured_site_cache_paths` and `load_configured_site_cache` in `src/utils/site_utils.py` (step 5) | built 2026-11-19 | read and check one aligned cache, and pick each protein's cache by `site.mode`. `train_pair_baselines` loads pinned or aligned caches through them |
| `filter_records_kept_by_retention_rule` and `check_population_inside_alignments` in `src/datasets/_pair_helpers.py`, called by `dataset_segment_pairs.py`; `check_aligned_dataset` in `src/models/train_pair_baselines.py` (step 6) | built 2026-11-19 | keep only the records of each final alignment, explain every dropped record, and fit aligned features only on datasets built on the same final alignments |
| `data/datasets/flu/July_2025/runs/task1_28p_aligned_codon_matched_20261119_145832_pb1_ha` | built 2026-11-19 | the aligned Task 1 PB1-HA dataset, 5 folds |
| aligned support in `site_cache_files` and `check_caches` (step 9) | not started | lets the 28-pair batch runners and the dataset audit use aligned caches |
