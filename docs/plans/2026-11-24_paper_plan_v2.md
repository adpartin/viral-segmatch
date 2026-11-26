# Paper plan v2: influenza segment-pair classification

**Status: IN PROGRESS**

The paper asks whether a classifier can distinguish an observed same-isolate segment pair (*positive pair*) from a generated pair unobserved in the corpus (*negative pair*). We evaluate each of the 28 schema pairs separately. For each schema pair, the modeling approach is to use aligned per-site codon as features and a LightGBM classifier on the concatenated slot features. Host, HN subtype and year are metadata used to define the datasets.

This is the current paper plan. The task numbers follow the [October methods-development record](2026-10-08_paper_plan.md). The results appear in this order:

| Task | Question | Test pairs |
|---|---|---|
| 2. Performance across hosts, subtypes and years | How well does one classifier per schema pair perform across a wide population and within each metadata group? | All test pairs for that schema pair, then the same test pairs grouped by host, subtype and exact year. |
| 1A. Performance within Human/H3N2/2024 | How well does a classifier trained within this one metadata group perform? How does its performance compare with the wide model on the same pairs? | The Human/H3N2/2024 test pairs from each Task 2 fold. |
| 1B. Sensitivity to the sampling of training negatives | How much do test scores and individual predictions vary when the training negatives change? | The same Task 1A test pairs for every draw within a fold. |
| 3. Test on future-year pairs | How well does a classifier trained on past years perform on held-out future years? | A separate holdout, reported by test year. |

## Task 2. Performance across hosts, subtypes and years

### Population and labels

A *schema pair* names the two slots, such as PB1 in slot A and HA in slot B. A *pair* is one instance with a CDS sequence in each slot. A *positive pair* is an observed same-isolate segment pair. A *negative pair* is a generated pair unobserved in any isolate in the corpus. This label describes the corpus observation. Biological incompatibility is a different question.

The *population* contains the isolates selected by the dataset filters. Task 2 covers 4 hosts (Human, Swine, Chicken, Duck), 3 subtypes (H3N2, H1N1, H5N1), and years 2010–2025 inclusive. A *metadata group* contains one exact combination of host group, subtype and year (e.g., Human/H3N2/2023 and Human/H3N2/2024 are different groups). The population filters select 61,635 isolates in the July 2025 corpus before *eligibility* is checked for either slot.

**Population coverage.** The host × subtype heatmap counts isolates across 2010–2025, before CDS completeness checks or sequence deduplication. It shows four sparse combinations: Chicken/H1N1, Chicken/H3N2, Human/H5N1 and Swine/H5N1. Before the CV folds are assigned, the metadata group minimum is applied separately for each schema pair: each exact host/subtype/year group needs at least 20 HK-selected positives. None of the four sparse combinations has a year that reaches 20 for any schema pair, so no separate host/subtype filter is needed. Give group counts alongside group scores. When comparing hosts, name the subtype and years covered. [Heatmap](../results/figs/2026-11-12_full_population_host_subtype_heatmap.png) · [Counts after the metadata group minimum](2026-10-08_paper_plan.md#build-settings)

### Dataset construction and size

An *eligible isolate* has a record kept by the alignment retention rule for both slots. *Unique positives* counts distinct `nt_cds` pair keys among eligible isolates. The pair key uses the two CDS nucleotide hashes, so different CDS sequences remain distinct even when they encode the same protein sequence. *Unique A* and *Unique B* count distinct CDS sequences in their respective slots among the unique positives. [Aligned codon features plan](2026-11-17_aligned_codon_features_plan.md)

*HK selected* counts the largest set of positives in which each CDS sequence occurs at most once per slot, selected by Hopcroft–Karp (HK). Selection takes place across the wide population for each schema pair. When the same pair key occurs in several metadata groups, deduplication keeps one row. The source isolate in slot A of that row (`assembly_id_a`) determines the positive's metadata group. The *metadata group minimum* drops whole groups with fewer than 20 HK-selected positives. *Positives kept* counts the positives that enter the folds after this step.

The test folds and the validation split are stratified by metadata group. Within each split, matched balanced-usage negatives give one negative per positive. Both endpoints of each negative belong to the same metadata group. Each sequence used in a positive occurs once in a negative in the same slot and split. Observed pair keys from the whole corpus are excluded from the negative draw.

For PB1-HA the counts follow this order:

```text
55,294 eligible isolates
   → 43,315 unique positives
   → 27,060 HK-selected positives in 139 metadata groups
   → 26,613 positives kept in 79 groups (60 groups dropped, containing 447 positives)
   → 26,613 positives and 26,613 negatives across the five test folds
```

Each kept positive occurs in one test fold. The last line counts 53,226 test pairs across five folds. A training split has its own, smaller count.

| # | Pair ID | Schema pair | Eligible isolates | Unique positives | Unique A | Unique B | HK selected | Positives kept |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1-2 | PB2-PB1 | 55,903 | 44,382 | 34,823 | 34,488 | 28,203 | 27,769 |
| 2 | 1-3 | PB2-PA | 60,105 | 46,906 | 36,683 | 35,446 | 28,879 | 28,435 |
| 3 | 1-4 | PB2-HA | 59,606 | 46,455 | 36,245 | 34,977 | 28,485 | 28,028 |
| 4 | 1-5 | PB2-NP | 59,833 | 43,798 | 36,516 | 26,976 | 23,428 | 22,962 |
| 5 | 1-6 | PB2-NA | 58,960 | 44,568 | 35,906 | 30,501 | 25,635 | 25,346 |
| 6 | 1-7 | PB2-M1 | 60,307 | 40,737 | 36,813 | 14,463 | 13,506 | 13,049 |
| 7 | 1-8 | PB2-NS1 | 58,768 | 40,518 | 35,723 | 17,223 | 15,727 | 15,366 |
| 8 | 2-3 | PB1-PA | 55,833 | 43,798 | 34,453 | 33,594 | 27,372 | 26,891 |
| 9 | 2-4 | PB1-HA | 55,294 | 43,315 | 33,992 | 33,251 | 27,060 | 26,613 |
| 10 | 2-5 | PB1-NP | 55,615 | 40,991 | 34,388 | 25,895 | 22,475 | failed |
| 11 | 2-6 | PB1-NA | 55,018 | 41,750 | 33,833 | 29,191 | 24,527 | 24,227 |
| 12 | 2-7 | PB1-M1 | 55,992 | 38,043 | 34,562 | 14,055 | 13,092 | 12,637 |
| 13 | 2-8 | PB1-NS1 | 54,421 | 37,803 | 33,447 | 16,628 | 15,148 | 14,788 |
| 14 | 3-4 | PA-HA | 59,492 | 45,769 | 34,958 | 34,926 | 27,650 | 27,163 |
| 15 | 3-5 | PA-NP | 59,728 | 42,912 | 35,286 | 26,946 | 22,982 | 22,496 |
| 16 | 3-6 | PA-NA | 58,932 | 43,739 | 34,704 | 30,505 | 25,064 | 24,746 |
| 17 | 3-7 | PA-M1 | 60,273 | 39,475 | 35,574 | 14,491 | 13,369 | 12,888 |
| 18 | 3-8 | PA-NS1 | 58,677 | 39,568 | 34,504 | 17,235 | 15,423 | 15,050 |
| 19 | 4-5 | HA-NP | 59,262 | 42,407 | 34,801 | 26,503 | 22,487 | 22,041 |
| 20 | 4-6 | HA-NA | 58,522 | 43,256 | 34,347 | 30,139 | 24,783 | 24,515 |
| 21 | 4-7 | HA-M1 | 59,675 | 38,856 | 35,033 | 14,109 | 13,017 | 12,587 |
| 22 | 4-8 | HA-NS1 | 58,246 | 38,986 | 33,999 | 16,876 | 15,115 | 14,762 |
| 23 | 5-6 | NP-NA | 58,767 | 39,902 | 26,382 | 30,427 | 20,786 | 20,509 |
| 24 | 5-7 | NP-M1 | 59,938 | 33,079 | 27,042 | 14,446 | 12,078 | failed |
| 25 | 5-8 | NP-NS1 | 58,439 | 33,708 | 26,182 | 17,181 | 13,670 | 13,326 |
| 26 | 6-7 | NA-M1 | 59,083 | 35,431 | 30,584 | 14,042 | 12,366 | 12,109 |
| 27 | 6-8 | NA-NS1 | 57,839 | 36,087 | 29,867 | 16,924 | 14,298 | 14,016 |
| 28 | 7-8 | M1-NS1 | 58,905 | 26,157 | 13,953 | 17,302 | 9,296 | 8,964 |

The columns through *HK selected* come from the capacity table. Each of the 28 builds logged the same eligible isolate, unique positive and HK-selected counts as the capacity table. *Positives kept* comes from each completed dataset's count after the metadata group minimum. *Failed* marks a build that stopped before it wrote all five folds. For each completed schema pair, report the groups kept, the groups dropped and the positives dropped alongside the dataset record. [Capacity table](../../results/flu/July_2025/pair_capacity_8_proteins_4host_3subtype_2010_2025_aligned/pair_capacity.csv) · [Per-pair build counts](../results/2026-11-24_task2_28p_build_20261124_205806/per_pair_counts.csv) · [PB1-HA group counts](../../data/datasets/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261123_183854_pb1_ha/metadata_groups_dropped.json)

### Build status of the 28 datasets

One job attempted all 28 schema pairs on 2026-11-24 under the dataset prefix `task2_28p_aligned_codon_matched_20261124_205806_`. Twenty-six wrote all five folds. PB1-NP and NP-M1 stopped partway, so 26 of the 28 schema pairs have a dataset. The manifest records an exit code for each of the 28 schema pairs and carries dataset entries for the 26 that finished. The dataset directories are untracked, so the manifest, the two failure logs and the per-pair counts are copied into a tracked record. [Build record](../results/2026-11-24_task2_28p_build_20261124_205806/README.md)

Both failures came from the matched negative draw on a validation split. The matched sampler gives each slot-A sequence of a metadata group one slot-B sequence taken from another positive of the same group, and it uses each sequence once. A group's own positives are observed pairs, so no sequence may take its own partner. When a split holds only 2 positives of a group, exchanging the two partners is the only assignment left, and the draw fails if that pair is observed anywhere in the corpus or was already drawn. PB1-NP failed on Duck/H1N1/2012 in the validation split of fold 3 of 5 (`fold_2`). NP-M1 failed on Duck/H1N1/2014 in the validation split of fold 1 of 5 (`fold_0`). In PB1-NP, Duck/H1N1/2012 held exactly 20 positives, which is the metadata group minimum, and the validation split took 2 of them. The minimum of 20 therefore admits groups whose validation split reaches the smallest size the sampler can assign at all.

The configuration that failed is not rare. Across the 132 fold-by-schema-pair combinations that were written, 638 validation-split group draws held exactly 2 positives, and every one of the 27 schema pairs with at least one written fold had such a draw. The 26 completed builds ran the same single-candidate draw and the exchange happened not to be blocked.

### Proposed rule: exclude a metadata group that cannot be matched

The rule is implemented in the builder and no dataset has been built with it yet, so it is still proposed as a change to the study's population. The builds described above all ran with `dataset.on_infeasible_metadata_group='error'`, under which such a failure stops the build for that schema pair.

Under the proposed rule, `dataset.on_infeasible_metadata_group='drop_group'`, a group whose matched negative draw has no valid assignment in any split is excluded from all three splits of all five folds for that schema pair, and the build continues. Positives of the other groups keep the fold and split assignment they already had. A group is excluded for sparsity, not for a fault, so the other reasons the draw can stop, such as a positive whose two sides are different isolates or a sequence repeated within a slot, stay build errors. The minimum stays at 20, because the minimum and this rule answer different questions: the minimum sets population coverage, while a group can pass any minimum and still have no valid assignment. [Unmatchable metadata group](../methods/glossary.md)

Each build records the groups it removed, with their positives and the folds and splits they were unmatchable in, in `metadata_groups_infeasible.json`. Report those counts alongside *Positives kept*, which counts the positives after the metadata group minimum and before this removal.

The cost for PB1-NP would be 20 of its 22,012 positives and 1 of its 80 kept groups. NP-M1's figure is unknown, because it wrote no group file. Kept group coverage would then differ between schema pairs, which it already does: the 26 completed datasets keep between 73 and 83 groups.

The 26 completed datasets are the reference for checking that the rule leaves an unaffected build unchanged, since none of them contains a group the rule would exclude. Their saved files were verified against the manifest on 2026-11-25. On the builder's test fixture, turning the rule on with nothing to remove reproduces the default folds pair table for pair table, which is the property that check relies on. [Baseline verification](../results/2026-11-24_task2_28p_build_20261124_205806/baseline_verification.md)

### Performance per schema pair

There is one model per schema pair per fold. For the whole-population score, each fold includes all its test pairs across retained metadata groups. The table gives the mean and sample SD of the five fold scores. A larger group contributes more test pairs to the fold score. In each completed run, name the fitting rule and the classification threshold. The PB1-HA fits monitored validation AUC-ROC with 50 rounds of early-stopping patience, a 1,000-tree cap and a decision threshold of 0.5.

| Pair ID | Schema pair | Positives kept | Features | F1 macro | Precision | Recall | AUC-ROC | AUC-PR | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2-4 | PB1-HA | 26,613 | 1,341 | 0.841 ± 0.004 | 0.781 ± 0.005 | 0.953 ± 0.001 | 0.902 ± 0.003 | 0.868 ± 0.003 | 0.126 ± 0.002 |

A built dataset is not a fitted model, and the two states are counted separately. 26 of the 28 schema pairs have a dataset, as the build status above records. One schema pair has fitted models, the five PB1-HA folds in the table above, so 27 schema pairs have no scores. The PB1-HA fits read the earlier pilot dataset `task2_28p_aligned_codon_matched_20261123_183854_pb1_ha`. Every file of that pilot matches the PB1-HA dataset of the 2026-11-24 run except `cv_info.json`, `runtime.json` and `stage3.log`, so both builds supply the same pairs, labels and fold assignments. When the table is complete, rank schema pairs by mean test F1 macro, as in the October plan. [PB1-HA CV summary](../../models/flu/July_2025/cv_runs/cv_flu_28p_pb1_ha_codon_aligned_4host_20261123_190333/cv_summary.json)

For the metadata group breakdown, report positive and negative counts by schema pair, fold, split and group. Report F1 macro, precision, recall, AUC-ROC, AUC-PR and Brier on the held-out pairs within each group. Show the whole-population F1 macro matrix and the metadata group results with their test counts. Scores for groups represented in training answer how well the wide model performs *within* those groups. Task 3 addresses the separate question of performance in future years.

The full PB1-HA report covers all 79 retained metadata groups. Its 395 group-by-fold test rows each have equal positive and negative counts. The other 115 test rows pool groups by host, subtype or year. [Counts by group, fold and split](../../results/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261123_183854_lgbm_20261123_190333_pb1_ha_metadata_report/metadata_group_counts.csv) · [Full PB1-HA test metrics](../../results/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261123_183854_lgbm_20261123_190333_pb1_ha_metadata_report/test_metrics_by_metadata.csv)

## Task 1A. Performance within Human/H3N2/2024

For each schema pair, take the Human/H3N2/2024 rows from the training, validation and test splits of each Task 2 fold. Keep each row's pair key, label and split assignment. The two analyses share the positive selection, fold assignments and Human/H3N2/2024 negatives from the Task 2 build. Here, *wide* describes the model fitted on the Task 2 population and *narrow* describes the model fitted on the extracted Human/H3N2/2024 pairs. [Dataset extraction script](../../scripts/extract_metadata_group_dataset.py)

In both tasks, the classifier receives sequence features. The two endpoints share host, subtype and exact year in both positive and negative pairs. Agreement on those fields therefore occurs with both labels. Task 1A limits training, validation and evaluation to one metadata group. Sequence patterns can still distinguish pairs within that group. Its test scores summarize comparisons within Human/H3N2/2024.

The paired comparison asks how performance on these test pairs changes when other metadata groups are included in training and validation. Validation AUC-ROC determines the selected number of trees, so the validation population is part of the comparison. Training-set size and composition change together.

### PB1-HA dataset and paired scores

The five folds had the following split sizes. Across the five test folds, the derived dataset contains 1,305 distinct test positives. [PB1-HA extraction record](../../data/datasets/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261123_183854_pb1_ha_human_h3n2_2024/extraction_info.json)

| Split in one fold | Positives | Negatives | Total pairs |
|---|---:|---:|---:|
| Training | 913 | 913 | 1,826 |
| Validation | 131 | 131 | 262 |
| Test | 261 | 261 | 522 |

In each fold, the two models scored the same 522 test pairs, with matching pair keys and labels. The wide column below uses only its Human/H3N2/2024 predictions. Differences were calculated within each fold before taking the mean and sample SD. [Narrow CV summary](../../models/flu/July_2025/cv_runs/cv_flu_28p_pb1_ha_codon_aligned_4host_human_h3n2_2024_20261124_120000/cv_summary.json) · [Five-row extract for Human/H3N2/2024](../../results/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261123_183854_lgbm_20261123_190333_pb1_ha_human_h3n2_2024/test_metrics_by_metadata.csv)

To reproduce the wide column, use the joint Human/H3N2/2024 rows for PB1-HA from the full PB1-HA test metrics above. Match the five fold IDs to the narrow CV summary for the paired differences. [Report](../../src/analysis/metadata_group_report.py) · [Wide training run manifest](../../models/flu/July_2025/training_runs/task2_28p_aligned_codon_matched_20261123_183854_lgbm_20261123_190333/training_run_manifest.json)

| Metric | Narrow model | Wide model on Human/H3N2/2024 | Paired difference (narrow minus wide) |
|---|---:|---:|---:|
| F1 macro | 0.8986 ± 0.0143 | 0.8545 ± 0.0121 | +0.0441 ± 0.0153 |
| Precision | 0.8647 ± 0.0207 | 0.7981 ± 0.0136 | +0.0666 ± 0.0134 |
| Recall | 0.9464 ± 0.0197 | 0.9533 ± 0.0074 | −0.0069 ± 0.0267 |
| AUC-ROC | 0.9476 ± 0.0063 | 0.9310 ± 0.0069 | +0.0166 ± 0.0066 |
| AUC-PR | 0.9250 ± 0.0103 | 0.9095 ± 0.0128 | +0.0155 ± 0.0138 |
| Brier | 0.1049 ± 0.0251 | 0.1446 ± 0.0055 | −0.0397 ± 0.0226 |

The narrow model had higher F1 macro in all five folds. At the common threshold of 0.5 it made 194 false positives and 70 false negatives across the five folds. The wide model made 315 false positives and 61 false negatives on those same pairs. AUC-PR summarizes ranking across thresholds and also favored the narrow model on average. The difference in AUC-PR cannot be used to apportion the difference in F1 macro.

Both arms used the same tree cap. The narrow fits selected 115–418 trees. All five wide fits reached the cap, with four selecting round 1,000 and one selecting round 993. The effect of the cap on the paired scores is unknown. These results cover one schema pair and one metadata group.

### Earlier standalone Task 1

The October plan records a separate Human/H3N2/2024 build with its own HK selection, folds and negative draws. The October pinned-length results and the November pinned-length versus aligned comparison are development results. Mean F1 macro over the 28 standalone schema pairs was 0.767 with pinned lengths and 0.769 with aligned codons. The standalone aligned PB1-HA dataset had 1,401 positives. Its test pairs differ from those in the derived Task 1A comparison. [Standalone Task 1 results](../results/2026-11-19_task1_aligned_codon_28p.md)

## Task 1B. Sensitivity to the sampling of training negatives

Measure how much test performance and each test pair's predicted probability vary across training-negative draws. Within each derived Task 1A fold, draw 1 is the training-negative set inherited from Task 2. Nine further draws give ten in total. The training positives, validation pairs, test pairs and LightGBM seed are fixed across draws. Draw 1 supplies the narrow fit in the paired comparison above.

For each schema pair and fold, report the mean and sample SD of test F1 macro across the ten draws. For each test pair, calculate the sample SD of its ten predicted probabilities. Report the median of those SDs separately for true positives and true negatives. The first statistic measures variation in test performance. The other two measure variation in individual predictions. All ten predictions for a given pair use the same test example.

The ten-draw score table for the derived population is pending. Decide whether to report 1B for PB1-HA alone or for all 28 schema pairs as their derived datasets become available. The standalone 1B table in the October plan describes the earlier datasets.

## Task 3. Test on future-year pairs

Evaluate on future-year pairs whose endpoint sequences were absent from the corresponding slots in past training and validation. Use a separate temporal holdout for each schema pair studied. The years and schema pairs remain open. The October plan gives 2011–2013 and 2014–2015 as example past and future periods.

Report the past and future periods, counts excluded for overlap with the past, test positives and negatives, and test F1 macro and AUC-ROC for each future year and for the combined test set. Show the test counts alongside the scores. A single temporal holdout has no across-fold SD. Year-to-year differences can also reflect which sequences and metadata groups were sampled. No future-year result is reported here. [Task 3 in the October plan](2026-10-08_paper_plan.md#3-test-on-future-year-pairs)
