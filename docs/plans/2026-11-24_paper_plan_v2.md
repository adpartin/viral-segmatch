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

| # | Pair ID | Schema pair | Eligible isolates | Unique positives | Unique A | Unique B | HK selected | Positives kept | Positives in folds |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1-2 | PB2-PB1 | 55,903 | 44,382 | 34,823 | 34,488 | 28,203 | 27,769 | 27,769 |
| 2 | 1-3 | PB2-PA | 60,105 | 46,906 | 36,683 | 35,446 | 28,879 | 28,435 | 28,435 |
| 3 | 1-4 | PB2-HA | 59,606 | 46,455 | 36,245 | 34,977 | 28,485 | 28,028 | 28,028 |
| 4 | 1-5 | PB2-NP | 59,833 | 43,798 | 36,516 | 26,976 | 23,428 | 22,962 | 22,962 |
| 5 | 1-6 | PB2-NA | 58,960 | 44,568 | 35,906 | 30,501 | 25,635 | 25,346 | 25,346 |
| 6 | 1-7 | PB2-M1 | 60,307 | 40,737 | 36,813 | 14,463 | 13,506 | 13,049 | 13,049 |
| 7 | 1-8 | PB2-NS1 | 58,768 | 40,518 | 35,723 | 17,223 | 15,727 | 15,366 | 15,366 |
| 8 | 2-3 | PB1-PA | 55,833 | 43,798 | 34,453 | 33,594 | 27,372 | 26,891 | 26,891 |
| 9 | 2-4 | PB1-HA | 55,294 | 43,315 | 33,992 | 33,251 | 27,060 | 26,613 | 26,613 |
| 10 | 2-5 | PB1-NP | 55,615 | 40,991 | 34,388 | 25,895 | 22,475 | 22,012 | 21,992 |
| 11 | 2-6 | PB1-NA | 55,018 | 41,750 | 33,833 | 29,191 | 24,527 | 24,227 | 24,227 |
| 12 | 2-7 | PB1-M1 | 55,992 | 38,043 | 34,562 | 14,055 | 13,092 | 12,637 | 12,637 |
| 13 | 2-8 | PB1-NS1 | 54,421 | 37,803 | 33,447 | 16,628 | 15,148 | 14,788 | 14,788 |
| 14 | 3-4 | PA-HA | 59,492 | 45,769 | 34,958 | 34,926 | 27,650 | 27,163 | 27,163 |
| 15 | 3-5 | PA-NP | 59,728 | 42,912 | 35,286 | 26,946 | 22,982 | 22,496 | 22,496 |
| 16 | 3-6 | PA-NA | 58,932 | 43,739 | 34,704 | 30,505 | 25,064 | 24,746 | 24,746 |
| 17 | 3-7 | PA-M1 | 60,273 | 39,475 | 35,574 | 14,491 | 13,369 | 12,888 | 12,888 |
| 18 | 3-8 | PA-NS1 | 58,677 | 39,568 | 34,504 | 17,235 | 15,423 | 15,050 | 15,050 |
| 19 | 4-5 | HA-NP | 59,262 | 42,407 | 34,801 | 26,503 | 22,487 | 22,041 | 22,041 |
| 20 | 4-6 | HA-NA | 58,522 | 43,256 | 34,347 | 30,139 | 24,783 | 24,515 | 24,515 |
| 21 | 4-7 | HA-M1 | 59,675 | 38,856 | 35,033 | 14,109 | 13,017 | 12,587 | 12,587 |
| 22 | 4-8 | HA-NS1 | 58,246 | 38,986 | 33,999 | 16,876 | 15,115 | 14,762 | 14,762 |
| 23 | 5-6 | NP-NA | 58,767 | 39,902 | 26,382 | 30,427 | 20,786 | 20,509 | 20,509 |
| 24 | 5-7 | NP-M1 | 59,938 | 33,079 | 27,042 | 14,446 | 12,078 | 11,610 | 11,590 |
| 25 | 5-8 | NP-NS1 | 58,439 | 33,708 | 26,182 | 17,181 | 13,670 | 13,326 | 13,326 |
| 26 | 6-7 | NA-M1 | 59,083 | 35,431 | 30,584 | 14,042 | 12,366 | 12,109 | 12,109 |
| 27 | 6-8 | NA-NS1 | 57,839 | 36,087 | 29,867 | 16,924 | 14,298 | 14,016 | 14,016 |
| 28 | 7-8 | M1-NS1 | 58,905 | 26,157 | 13,953 | 17,302 | 9,296 | 8,964 | 8,964 |

The columns through *HK selected* come from the capacity table. *Positives kept* counts positives after the metadata group minimum. *Positives in folds* counts positives after the split feasibility check. The two counts differ for PB1-NP and NP-M1. [Capacity table](../../results/flu/July_2025/pair_capacity_8_proteins_4host_3subtype_2010_2025_aligned/pair_capacity.csv) · [Build manifest](../../data/datasets/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261126_101242_manifest.json)

### Build status of the 28 datasets

All 28 schema pairs completed five folds on 2026-11-25 at commit `f4409bb`. They were rebuilt on 2026-11-26 at commit `faa3f44`, which raised the LightGBM tree cap, because the training script requires the fits to run at the commit the datasets were built at. All 560 split files of the rebuild match the 2026-11-25 build byte for byte (15 pair tables and five blocking files per dataset). The only change in the resolved dataset configs is the tree cap. The audit of the rebuild reported zero failures across the 28 datasets, including the check for excluded groups in training, validation and test. Both manifests record a dirty working tree, because documentation edits were uncommitted. [Build manifest](../../data/datasets/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261126_101242_manifest.json) · [Audit log](../../data/datasets/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261126_101242_audit.log) · [2026-11-25 build manifest](../../data/datasets/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261125_193635_manifest.json)

**Split feasibility.** The metadata group minimum remains 20. If a retained metadata group cannot supply valid matched negatives in any split, Task 2 excludes that group from all three splits of all five folds for that schema pair. Positives from the other groups keep their original fold and split assignments. [Glossary](../methods/glossary.md)

| Schema pair | Metadata group excluded | Split where matching failed | Positives kept after minimum | Positives excluded | Positives in folds |
|---|---|---|---:|---:|---:|
| PB1-NP | Duck/H1N1/2012 | `fold_2` validation | 22,012 | 20 | 21,992 |
| NP-M1 | Duck/H1N1/2014 | `fold_0` validation | 11,610 | 20 | 11,590 |

These were the only two exclusions after the metadata group minimum. PB1-NP retained 79 of the 80 groups that passed the minimum. NP-M1 retained 75 of 76. [PB1-NP exclusion record](../../data/datasets/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261126_101242_pb1_np/metadata_groups_infeasible.json) · [NP-M1 exclusion record](../../data/datasets/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261126_101242_np_m1/metadata_groups_infeasible.json)

The 2026-11-24 build stopped on those two validation splits and completed the other 26 datasets. Small validation groups were common in that run: 638 group draws had two positives across the 132 folds written. For all 26 completed datasets, the replacement manifest matches the earlier SHA-256 checksums for all 520 split files (15 pair tables and five blocking files per dataset). Across all 28 schema pairs, the only change in the resolved dataset configs was the added `on_infeasible_metadata_group: drop_group` setting. [Earlier build record](../results/2026-11-24_task2_28p_build_20261124_205806/README.md) · [Baseline verification](../results/2026-11-24_task2_28p_build_20261124_205806/baseline_verification.md)

### Performance per schema pair

There is one model per schema pair per fold. For the whole-population score, each fold includes all its test pairs across retained metadata groups. The table gives the mean and sample SD of the five fold scores. A larger group contributes more test pairs to the fold score. In each completed run, name the fitting rule and the classification threshold. The fits monitored validation AUC-ROC with 50 rounds of early-stopping patience, a 5,000-tree cap and a decision threshold of 0.5. The selected tree counts ranged from 760 to 4,108, so no fit reached the cap.

| Pair ID | Schema pair | Positives in folds | Features | F1 macro | Precision | Recall | AUC-ROC | AUC-PR | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-2 | PB2-PB1 | 27,769 | 1,519 | 0.865 ± 0.004 | 0.812 ± 0.005 | 0.952 ± 0.002 | 0.923 ± 0.004 | 0.895 ± 0.004 | 0.105 ± 0.003 |
| 1-3 | PB2-PA | 28,435 | 1,477 | 0.860 ± 0.005 | 0.806 ± 0.007 | 0.951 ± 0.001 | 0.919 ± 0.003 | 0.890 ± 0.004 | 0.108 ± 0.003 |
| 1-4 | PB2-HA | 28,028 | 1,342 | 0.868 ± 0.002 | 0.816 ± 0.002 | 0.953 ± 0.004 | 0.921 ± 0.002 | 0.890 ± 0.006 | 0.105 ± 0.002 |
| 1-5 | PB2-NP | 22,962 | 1,259 | 0.847 ± 0.005 | 0.792 ± 0.007 | 0.946 ± 0.006 | 0.906 ± 0.004 | 0.873 ± 0.007 | 0.118 ± 0.003 |
| 1-6 | PB2-NA | 25,346 | 1,254 | 0.857 ± 0.004 | 0.805 ± 0.007 | 0.943 ± 0.004 | 0.915 ± 0.003 | 0.883 ± 0.005 | 0.111 ± 0.003 |
| 1-7 | PB2-M1 | 13,049 | 1,013 | 0.797 ± 0.008 | 0.751 ± 0.009 | 0.897 ± 0.002 | 0.868 ± 0.005 | 0.836 ± 0.005 | 0.148 ± 0.005 |
| 1-8 | PB2-NS1 | 15,366 | 991 | 0.803 ± 0.006 | 0.752 ± 0.007 | 0.911 ± 0.006 | 0.874 ± 0.004 | 0.843 ± 0.003 | 0.143 ± 0.004 |
| 2-3 | PB1-PA | 26,891 | 1,476 | 0.863 ± 0.003 | 0.808 ± 0.004 | 0.956 ± 0.003 | 0.921 ± 0.003 | 0.891 ± 0.007 | 0.106 ± 0.002 |
| 2-4 | PB1-HA | 26,613 | 1,341 | 0.862 ± 0.006 | 0.807 ± 0.010 | 0.955 ± 0.003 | 0.918 ± 0.004 | 0.884 ± 0.003 | 0.107 ± 0.004 |
| 2-5 | PB1-NP | 21,992 | 1,258 | 0.849 ± 0.008 | 0.794 ± 0.010 | 0.946 ± 0.003 | 0.909 ± 0.005 | 0.875 ± 0.007 | 0.116 ± 0.005 |
| 2-6 | PB1-NA | 24,227 | 1,253 | 0.854 ± 0.005 | 0.800 ± 0.006 | 0.947 ± 0.003 | 0.912 ± 0.004 | 0.880 ± 0.007 | 0.112 ± 0.003 |
| 2-7 | PB1-M1 | 12,637 | 1,012 | 0.789 ± 0.007 | 0.741 ± 0.011 | 0.896 ± 0.010 | 0.861 ± 0.007 | 0.824 ± 0.010 | 0.151 ± 0.005 |
| 2-8 | PB1-NS1 | 14,788 | 990 | 0.808 ± 0.012 | 0.755 ± 0.013 | 0.920 ± 0.004 | 0.877 ± 0.007 | 0.843 ± 0.010 | 0.141 ± 0.006 |
| 3-4 | PA-HA | 27,163 | 1,299 | 0.864 ± 0.005 | 0.810 ± 0.007 | 0.954 ± 0.003 | 0.920 ± 0.003 | 0.888 ± 0.003 | 0.106 ± 0.004 |
| 3-5 | PA-NP | 22,496 | 1,216 | 0.847 ± 0.003 | 0.793 ± 0.003 | 0.944 ± 0.011 | 0.909 ± 0.002 | 0.876 ± 0.001 | 0.116 ± 0.001 |
| 3-6 | PA-NA | 24,746 | 1,211 | 0.852 ± 0.005 | 0.799 ± 0.007 | 0.944 ± 0.003 | 0.913 ± 0.005 | 0.880 ± 0.006 | 0.113 ± 0.005 |
| 3-7 | PA-M1 | 12,888 | 970 | 0.784 ± 0.009 | 0.737 ± 0.011 | 0.890 ± 0.005 | 0.857 ± 0.008 | 0.825 ± 0.009 | 0.156 ± 0.007 |
| 3-8 | PA-NS1 | 15,050 | 948 | 0.806 ± 0.009 | 0.755 ± 0.008 | 0.910 ± 0.008 | 0.877 ± 0.006 | 0.843 ± 0.007 | 0.140 ± 0.005 |
| 4-5 | HA-NP | 22,041 | 1,081 | 0.847 ± 0.007 | 0.792 ± 0.010 | 0.945 ± 0.002 | 0.904 ± 0.007 | 0.868 ± 0.012 | 0.119 ± 0.006 |
| 4-6 | HA-NA | 24,515 | 1,076 | 0.864 ± 0.005 | 0.808 ± 0.006 | 0.957 ± 0.002 | 0.919 ± 0.005 | 0.885 ± 0.007 | 0.106 ± 0.004 |
| 4-7 | HA-M1 | 12,587 | 835 | 0.790 ± 0.007 | 0.740 ± 0.008 | 0.901 ± 0.003 | 0.858 ± 0.007 | 0.819 ± 0.009 | 0.154 ± 0.006 |
| 4-8 | HA-NS1 | 14,762 | 813 | 0.798 ± 0.009 | 0.746 ± 0.010 | 0.910 ± 0.002 | 0.867 ± 0.004 | 0.827 ± 0.003 | 0.146 ± 0.004 |
| 5-6 | NP-NA | 20,509 | 993 | 0.839 ± 0.003 | 0.785 ± 0.003 | 0.937 ± 0.003 | 0.899 ± 0.003 | 0.862 ± 0.004 | 0.123 ± 0.002 |
| 5-7 | NP-M1 | 11,590 | 752 | 0.778 ± 0.013 | 0.730 ± 0.013 | 0.891 ± 0.009 | 0.854 ± 0.010 | 0.823 ± 0.011 | 0.156 ± 0.007 |
| 5-8 | NP-NS1 | 13,326 | 730 | 0.788 ± 0.008 | 0.738 ± 0.008 | 0.901 ± 0.007 | 0.866 ± 0.006 | 0.831 ± 0.008 | 0.148 ± 0.004 |
| 6-7 | NA-M1 | 12,109 | 747 | 0.781 ± 0.003 | 0.735 ± 0.006 | 0.886 ± 0.012 | 0.851 ± 0.007 | 0.814 ± 0.009 | 0.158 ± 0.003 |
| 6-8 | NA-NS1 | 14,016 | 725 | 0.803 ± 0.006 | 0.753 ± 0.007 | 0.907 ± 0.003 | 0.872 ± 0.007 | 0.836 ± 0.010 | 0.145 ± 0.003 |
| 7-8 | M1-NS1 | 8,964 | 484 | 0.756 ± 0.005 | 0.713 ± 0.007 | 0.864 ± 0.006 | 0.834 ± 0.007 | 0.804 ± 0.008 | 0.166 ± 0.003 |

All 140 fits come from one training run at commit `faa3f44`. Rank schema pairs by mean test F1 macro. [Training run manifest](../../models/flu/July_2025/training_runs/task2_28p_aligned_codon_matched_20261126_101242_lgbm_20261126_103606/training_run_manifest.json) · [PB1-HA CV summary](../../models/flu/July_2025/cv_runs/cv_flu_28p_pb1_ha_codon_aligned_4host_20261126_103606/cv_summary.json)

The PB1-HA pilot used the same test pairs with a 1,000-tree cap, and its five fits selected round 993 or 1,000. It obtained an F1 macro of 0.841 ± 0.004, and the 5,000-tree fits obtained 0.862 ± 0.006. The paired fold difference was +0.021 ± 0.003, and F1 macro rose in all five folds. AUC-ROC rose by 0.016 ± 0.001 and Brier fell by 0.020 ± 0.003. This measures the effect of the cap for PB1-HA only. [PB1-HA pilot CV summary](../../models/flu/July_2025/cv_runs/cv_flu_28p_pb1_ha_codon_aligned_4host_20261123_190333/cv_summary.json)

For the metadata group breakdown, report positive and negative counts by schema pair, fold, split and group. Report F1 macro, precision, recall, AUC-ROC, AUC-PR and Brier on the held-out pairs within each group. Show the whole-population F1 macro matrix and the metadata group results with their test counts. Scores for groups represented in training answer how well the wide model performs *within* those groups. Task 3 addresses the separate question of performance in future years.

The full PB1-HA report covers all 79 retained metadata groups. Its 395 group-by-fold test rows each have equal positive and negative counts. The other 115 test rows pool groups by host, subtype or year. [Counts by group, fold and split](../../results/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261123_183854_lgbm_20261123_190333_pb1_ha_metadata_report/metadata_group_counts.csv) · [Full PB1-HA test metrics](../../results/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261123_183854_lgbm_20261123_190333_pb1_ha_metadata_report/test_metrics_by_metadata.csv)

## Task 1A. Performance within Human/H3N2/2024

For each schema pair, take the Human/H3N2/2024 rows from the training, validation and test splits of each Task 2 fold. Keep each row's pair key, label and split assignment. The two analyses share the positive selection, fold assignments and Human/H3N2/2024 negatives from the Task 2 build. Here, *wide* describes the model fitted on the Task 2 population and *narrow* describes the model fitted on the extracted Human/H3N2/2024 pairs. [Dataset extraction script](../../scripts/extract_metadata_group_dataset.py)

In both tasks, the classifier receives sequence features. The two endpoints share host, subtype and exact year in both positive and negative pairs. Agreement on those fields therefore occurs with both labels. Task 1A limits training, validation and evaluation to one metadata group. Sequence patterns can still distinguish pairs within that group. Its test scores summarize comparisons within Human/H3N2/2024.

The paired comparison asks how performance on these test pairs changes when other metadata groups are included in training and validation. Validation AUC-ROC determines the selected number of trees, so the validation population is part of the comparison. Training-set size and composition change together.

### Extracted datasets

Human/H3N2/2024 datasets were extracted from the Task 2 folds for all 28 schema pairs. Across the five test folds, the number of distinct positives per schema pair ranged from 365 for M1-NS1 to 1,911 for PB2-HA. The PB1-HA extraction matches the pilot extraction byte for byte in all 20 split files. Only PB1-HA has narrow fits, so the comparison for the other 27 schema pairs is pending. [Extraction log](../../data/datasets/flu/July_2025/runs/task2_28p_aligned_codon_matched_20261126_101242_human_h3n2_2024_extraction.log)

*Positives in folds* counts the Human/H3N2/2024 positives across the five test folds. Each positive occurs in one test fold. The last three columns give positive counts within one fold, with the minimum and maximum shown when counts vary. Every split has the same number of negatives as positives.

| # | Pair ID | Schema pair | Positives in folds | Training positives per fold | Validation positives per fold | Test positives per fold |
|---:|---|---|---:|---:|---:|---:|
| 1 | 1-2 | PB2-PB1 | 1,325 | 927–928 | 132–133 | 265 |
| 2 | 1-3 | PB2-PA | 1,895 | 1,326 | 190 | 379 |
| 3 | 1-4 | PB2-HA | 1,911 | 1,337–1,338 | 191 | 382–383 |
| 4 | 1-5 | PB2-NP | 1,381 | 966–967 | 138 | 276–277 |
| 5 | 1-6 | PB2-NA | 1,609 | 1,126–1,127 | 161 | 321–322 |
| 6 | 1-7 | PB2-M1 | 623 | 436–437 | 62 | 124–125 |
| 7 | 1-8 | PB2-NS1 | 871 | 609–610 | 87 | 174–175 |
| 8 | 2-3 | PB1-PA | 1,257 | 879–880 | 126 | 251–252 |
| 9 | 2-4 | PB1-HA | 1,305 | 913 | 131 | 261 |
| 10 | 2-5 | PB1-NP | 963 | 674–675 | 96 | 192–193 |
| 11 | 2-6 | PB1-NA | 1,140 | 798 | 114 | 228 |
| 12 | 2-7 | PB1-M1 | 432 | 302–303 | 43 | 86–87 |
| 13 | 2-8 | PB1-NS1 | 619 | 433–434 | 62 | 123–124 |
| 14 | 3-4 | PA-HA | 1,793 | 1,255–1,256 | 179 | 358–359 |
| 15 | 3-5 | PA-NP | 1,325 | 927 | 133 | 265 |
| 16 | 3-6 | PA-NA | 1,567 | 1,096–1,097 | 157 | 313–314 |
| 17 | 3-7 | PA-M1 | 595 | 416 | 60 | 119 |
| 18 | 3-8 | PA-NS1 | 826 | 578 | 82–83 | 165–166 |
| 19 | 4-5 | HA-NP | 1,333 | 933–934 | 133 | 266–267 |
| 20 | 4-6 | HA-NA | 1,555 | 1,088 | 156 | 311 |
| 21 | 4-7 | HA-M1 | 610 | 427 | 61 | 122 |
| 22 | 4-8 | HA-NS1 | 836 | 584–585 | 84 | 167–168 |
| 23 | 5-6 | NP-NA | 1,163 | 814–815 | 116 | 232–233 |
| 24 | 5-7 | NP-M1 | 531 | 371–372 | 53 | 106–107 |
| 25 | 5-8 | NP-NS1 | 709 | 496–497 | 71 | 141–142 |
| 26 | 6-7 | NA-M1 | 550 | 385 | 55 | 110 |
| 27 | 6-8 | NA-NS1 | 738 | 516–517 | 74 | 147–148 |
| 28 | 7-8 | M1-NS1 | 365 | 255 | 37 | 73 |

### PB1-HA paired scores

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
