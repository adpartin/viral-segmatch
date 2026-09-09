# Segment matching: progress report and scope decision

**Date:** 2026-09-08

**Question.** Given two influenza A gene-segment sequences, can a classifier distinguish an observed same-isolate pair from a generated pair that was not observed together?

a) **Within-season matching**: Train and test on disjoint pairs from the same year. This represents an established season with existing paired sequences for training.

b) **One-season-ahead matching**: Train on year T and test on year T+1, without using year T+1 data for model training or fine-tuning. This represents the beginning of a new season, before existing paired sequences have accumulated.

**Purpose of this report.** Summarize the completed Human H3N2 2024 work and decide whether three
bounded follow-ups justify a publication, or whether the project should be archived as a technical
report. The within-season setting has now been tested with unique nucleotide sequences on both
sides. The one-season-ahead result predates those controls and remains preliminary.

---

## 1. Dataset populations used

Two dataset populations appear below. They are not interchangeable, so each result names its own.

| label | filter | positives | used for |
|---|---|---:|---|
| **A. Original** | H3N2 2024, all hosts, complete CDS at pinned length | 3,580 | feature importance, shuffling, false-positive analysis |
| **B. Equal-count** | Human H3N2 2024, complete CDS at pinned length, Hopcroft-Karp matched, sampled to a common size | 1,698 per pair | the four-pair experiment |

Population B enforces that **no nucleotide CDS is used in more than one retained positive pair**.
Generating negatives within each fold then gives zero exact sequence overlap between training and
test. Population B has about half as many HA-NA positives as population A.

---

## 2. Four-pair experiment (population B)

Four schema pairs over four proteins, arranged so that each protein appears twice (HA and NA: variable surface proteins; PB2 and PA: conserved polymerase proteins).

|  | PA | NA |
|---|---|---|
| **PB2** | PB2-PA | PB2-NA |
| **HA** | PA-HA | HA-NA |

### Population sizes

Table columns:
* **Eligible isolates**: isolates satisfying the metadata filters (Human, H3N2, 2024) with both required segments represented by complete CDS sequences at their pinned lengths.
* **Unique positive pairs**: eligible isolates after deduplicating isolates containing the same exact pair of nucleotide sequences (i.e., sequence pair deduplication).
* **HK-selected positives**: a maximum-size subset in which each nucleotide sequence occurs in at most one positive pair on each side. Hopcroft–Karp (HK) finds the maximum possible count under this constraint.
* **Min-count sample**: the HK-selected population randomly sampled to 1,698 positives, the smallest HK count across the four schemas. HA–NA already has 1,698 and therefore is not reduced further.

```
                                              PB2-PA
  [ Human, H3N2, 2024 + complete CDS at pinned length ]
                      ↓
  Eligible isolates                            5,324
                      ↓  dedup identical nucleotide sequence pairs
  Unique positive pairs                        3,837
                      ↓  Hopcroft-Karp: each sequence at most once per side
  HK-selected positives                        2,030
                      ↓  sample to the smallest HK count across schemas
  Min-count sampled                            1,698
```

| Schema pair | Eligible isolates | Unique positive pairs | HK-selected positives | Min-count sampled |
|---|---:|---:|---:|---:|
| HA-NA | 5,173 | 3,466 | 1,698 | 1,698 |
| PB2-PA | 5,324 | 3,837 | 2,030 | 1,698 |
| PB2-NA | 5,167 | 3,532 | 1,745 | 1,698 |
| PA-HA | 5,329 | 3,805 | 1,944 | 1,698 |

* HA-NA has the smallest HK-selected count, so it set the min-count sample size of 1,698. Eligible isolates span 5,167 to 5,329, a spread of about 3%.

*  Note that while the final positive count is the same across the four schema pairs (Min-count sampled), the underlying isolates do not necessarily match, because filtering, HK selection, and sampling are each done independently per schema.

### Setup

**Datasets**: Within a schema pair, all four feature representations use the SAME dataset rows and the SAME
CV folds.

**Models**: LightGBM classifier with threshold 0.5, and a 1:1 neg-to-pos ratio. A total of 64 model trainings across schemas and folds.

### Results

Mean ± std across four test folds.

_"Per-site features"_ in this experiment use one feature column for each aligned position of a fixed-length CDS: a nucleotide, codon, or translated amino acid (aa), depending on the feature type.

We avoid the term _"positional encoding"_ because it usually means adding positional information to sequence tokens in transformer architectures; here, position (or site) is represented directly by the feature column itself.

| Schema pair | Feature type | Features | F1 macro | AUC-ROC | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| HA-NA | nt 6-mer | 8,192 | 0.8635 ± 0.0158 | 0.9250 ± 0.0113 | 0.8059 | 0.9617 |
| HA-NA | site nt | 3,111 | 0.8713 ± 0.0037 | 0.9358 ± 0.0061 | 0.8199 | 0.9541 |
| HA-NA | site codon | 1,037 | 0.8777 ± 0.0134 | 0.9349 ± 0.0068 | 0.8288 | 0.9541 |
| HA-NA | site aa | 1,037 | 0.7605 ± 0.0116 | 0.8321 ± 0.0175 | 0.7174 | 0.8687 |
| PB2-PA | nt 6-mer | 8,192 | 0.7673 ± 0.0345 | 0.8642 ± 0.0261 | 0.7183 | 0.8928 |
| PB2-PA | site nt | 4,431 | 0.8300 ± 0.0332 | 0.9093 ± 0.0233 | 0.7818 | 0.9217 |
| PB2-PA | site codon | 1,477 | 0.8122 ± 0.0283 | 0.9036 ± 0.0196 | 0.7598 | 0.9211 |
| PB2-PA | site aa | 1,477 | 0.4526 ± 0.0214 | 0.5263 ± 0.0132 | 0.5028 | 0.7001 |
| PB2-NA | nt 6-mer | 8,192 | 0.8580 ± 0.0167 | 0.9244 ± 0.0124 | 0.8035 | 0.9517 |
| PB2-NA | site nt | 3,690 | 0.8819 ± 0.0167 | 0.9331 ± 0.0149 | 0.8370 | 0.9505 |
| PB2-NA | site codon | 1,230 | 0.8726 ± 0.0192 | 0.9277 ± 0.0163 | 0.8240 | 0.9505 |
| PB2-NA | site aa | 1,230 | 0.5751 ± 0.0392 | 0.6467 ± 0.0312 | 0.5725 | 0.7850 |
| PA-HA | nt 6-mer | 8,192 | 0.8113 ± 0.0191 | 0.8979 ± 0.0113 | 0.7581 | 0.9217 |
| PA-HA | site nt | 3,852 | 0.8438 ± 0.0235 | 0.9179 ± 0.0162 | 0.7946 | 0.9323 |
| PA-HA | site codon | 1,284 | 0.8296 ± 0.0177 | 0.9120 ± 0.0166 | 0.7826 | 0.9175 |
| PA-HA | site aa | 1,284 | 0.5353 ± 0.0248 | 0.5591 ± 0.0393 | 0.5337 | 0.6438 |

### What the table shows

* Performance for Human-H3N2-2024 varies across schema pairs. With per-site nucleotide features, F1 macro ranges: 0.8300 for PB2-PA to 0.8819 for PB2-NA.

* Per-site nucleotide has a higher mean than 6-mers for every schema pair. `TODO`: Use more folds

* Amino-acid (aa) performance is lower and varies by pair, and the comparison is not clean. Some schema pairs fall to near-chance. The aa arm likely violates two constraints that the nucleotide arm satisfies: (a) no sequence is reused across pairs, and (b) no negative pair matches a positive pair. Both were enforced when the dataset was built, but enforced on nucleotide sequences, and translation collapses synonymous variants, so there is no guarantee either constraint should carry into aa space. Part of the performance loss therefore likely reflects how the dataset was constructed rather than biological signal lost in translation. `TODO`: For a fair aa experiment, rebuild the dataset using aa identity for positive-pair deduplication, and negative-pair blocking. Report this as a separate population because it cannot use exactly the same rows as the nucleotide experiments.

* Mean precision is below mean recall in all 16 cells. See §5.

---

## 3. Open question 1: where the signal sits

Measured on **population A**, HA-NA, codon features, 1,037 positions.

### Which positions the model uses (gain)

Gain is the reduction in training loss attributed to every tree split using a feature. It is read
from each fitted model, normalized within fold, and then averaged across the four folds. It measures
what the fitted trees used; it does not establish biological importance.

| rank | protein | position | share of total gain |
|---:|---|---:|---:|
| 1 | HA | 544 | 6.25% |
| 2 | NA | 310 | 4.81% |
| 3 | NA | 244 | 4.00% |
| 4 | HA | 36 | 3.70% |
| 5 | NA | 284 | 3.16% |

Cumulative gain: top 10 positions hold **35.2%**, top 25 hold 54.1%, top 100 hold 82.7%.

### Whether a retrained model can compensate after top sites are corrupted

The completed retraining experiment used sites ranked by **SHAP**, not gain. Gain and SHAP were
strongly correlated (Spearman 0.97) and shared 12 of their top 15 sites, but the two top-N sets are
not identical. The selected positions are shuffled in train, validation and test, and a new model
is fitted from scratch. AUC-ROC loss relative to the uncorrupted baseline:

| positions corrupted | top-ranked | random |
|---:|---:|---:|
| 10 | 15.8% | 0.8% |
| 25 | 33.3% | 0.4% |
| 50 | 56.1% | 1.5% |
| 100 | 89.2% | 2.0% |
| all 1,037 | 100.7% | — |

The SHAP-ranked top 10 cost a retrained model 15.8% of above-chance AUC-ROC, compared with 49.5%
when the fitted model is tested without retraining. The model can therefore recover much of the
lost performance from other positions. Corrupting the top 100 removes 89.2%, while 100 random sites
remove only 2.0%.

**Open.** The current result is exploratory because held-out test inputs contributed to the SHAP
ranking used to choose the perturbed sites. Test labels were not used, but test features still
influenced experiment selection. Repeat the experiment with gain-ranked sites on the
uniqueness-controlled HA-NA population. The ranked sites may identify lineage or clade rather than
segment compatibility; this analysis alone cannot assign biological meaning.

---

## 4. Open question 2: predicting the next season

Run on an earlier population (H3N2, HA-NA, nucleotide 6-mers), **before** the unique-sequence
controls in population B. Treat as indicative.

| training window | test | F1 macro | precision |
|---|---|---:|---:|
| 2024 | 2025 | 0.8940 | 0.837 |
| 2015-2024 | 2025 | 0.8240 | 0.736 |

Within-season results on the same setup, for reference: 2022 0.9065, 2023 0.8847, 2024 0.9177,
2025 0.8872. PB2-PB1 on 2024 gives 0.9236.

The earlier run suggests that one recent season may be sufficient, but it does not yet answer the
PI's prospective question under the current controls. It used all hosts, allowed exact sequences
to occur across periods, and used a different population size. The 0.8940 result is therefore not
directly comparable with §2.

**Open.** Repeat HA-NA with Human H3N2, complete pinned-length CDS, and nucleotide-unique endpoints.
Train on 2024 and test on 2025, remove exact train-test sequence overlap, and generate negatives
within each year so year alone cannot separate the classes. Compare it with a size-matched
within-2024 control. Start with site nucleotide and nucleotide 6-mer features; expand only if the
result is informative.

---

## 5. Open question 3: why so many false positives

Precision is below recall in every configuration measured. On population A, HA-NA, per-site
nucleotide features, pooled over four folds:

| | predicted positive | predicted negative |
|---|---:|---:|
| **actual positive** | 3,500 | 80 |
| **actual negative** | 496 | 3,084 |

Precision 0.876, recall 0.978. False positives outnumber false negatives 6.2 to 1.

### Where they sit

Each negative pair was scored by its Hamming distance to the nearest observed positive pair, in
nucleotides:

| distance to nearest positive | negatives | false positives | false-positive rate |
|---|---:|---:|---:|
| 0-2 nt | 176 | 145 | **0.824** |
| 3-5 nt | 847 | 198 | 0.234 |
| 6-10 nt | 1,748 | 136 | 0.078 |
| 11-20 nt | 547 | 15 | 0.027 |
| >20 nt | 262 | 2 | 0.008 |

**False positives are concentrated on negatives that are nearly identical to real positives.** The
closest 4.9% of negatives produce 29.2% of the false positives, a 5.95-fold enrichment, and the rate
falls monotonically with distance.

### What this means, and what it does not

This says the errors are not spread randomly: they are enriched among generated negatives close
to an observed positive. The distance is based on the same aligned sequences available to the
model, so this association is not independent evidence that the negative labels are wrong.

It does **not** establish that these pairs are unlabelable. A substantial minority of the closest
negatives are still classified correctly. The precision-recall asymmetry is also threshold
dependent: at 0.70 the counts are 312 false positives against 310 false negatives.

**Open.** First, tune the operating threshold on validation data only and report its test
precision-recall tradeoff. Second, repeat the distance analysis on the uniqueness-controlled
four-pair datasets to test whether the same pattern holds by schema. If it does, redraw negatives
within narrow collection-time windows and controlled distance bins, retrain, and determine whether
performance changes. These steps separate a threshold effect from a sampler effect; they still
cannot establish biological compatibility without external labels.

---

## 6. Relation to the collection-date study

The useful analogy with Jamie's draft is the experimental structure, not the prediction target.
That work compares segments, nucleotide and amino-acid representations, importance-guided
retraining, and transfer across populations. Its weaker representations and transfer failures help
define where collection-date prediction works.

The analogous segmatch results are the four schema pairs, the nucleotide/codon/amino-acid
comparison, importance-guided corruption and retraining, and the proposed 2024-to-2025 test. The
important difference is that segmatch negatives are generated rather than externally observed
incompatible pairs. Negative construction is therefore part of the scientific question, not only
a technical detail.

---

## 7. Publication-or-archive decision

### What has held up

- Within-season co-assignment remains predictable after exact train-test sequence reuse is removed:
  per-site nucleotide F1 macro is 0.83 to 0.88 across four schema pairs.
- Results depend on the schema and representation. Nucleotide and codon features remain strong,
  while amino-acid features weaken sharply for three pairs.
- Top-ranked positions matter, but a retrained model can recover from corruption of a small number
  of them by using other positions.
- False positives are strongly enriched among negatives close to observed positives.

### What is not established

The ranked positions have no biological validation. The model may be recognizing lineage,
population structure, or collection-time proximity rather than segment compatibility. The
prospective result predates the current controls. The amino-acid comparison also contains label
collisions created by defining identity in nucleotide space. The equal-count experiment uses one
sampling seed and four folds.

### Candidate scope

If the group considers evaluation and failure analysis a sufficient contribution, a defensible
working claim is:

> Within-season influenza A segment co-assignment remains predictable after exact sequence reuse is
> removed, but performance depends on the segment pair and sequence representation. Nucleotide
> information discarded during translation and the construction of negative pairs both materially
> affect the result.

This is an evaluation result, not a claim of biological compatibility or coevolution.

### Bounded work needed to decide

1. **Importance and retraining:** rerun the top-site retraining experiment using gain-ranked sites
   on the uniqueness-controlled HA-NA population. This aligns the importance measure and the
   perturbed positions in one clean experiment.
2. **Prospective prediction:** run the controlled Human H3N2 2024-to-2025 experiment described in
   section 4, with a size-matched within-2024 reference.
3. **False positives:** calibrate the threshold using validation data, repeat the distance
   diagnostic on the four uniqueness-controlled schemas, and run a controlled negative-sampling
   experiment only if the same distance pattern remains.

If these results are intended for a paper, repeat the equal-count sample with additional seeds
after the three analyses define the final design. Do not expand now to a 28-pair sweep, ESM-2, or
additional metadata axes.

### The question for the PIs

Is a careful evaluation of when segment co-assignment is predictable, and when it fails, a
publication contribution for this group? If yes, complete only the three bounded analyses above and
then write. If biological interpretation of individual sites is required, these experiments are
unlikely to provide it; summarize the existing results as a technical report and archive the
project.

---

## Artifacts

Datasets `data/datasets/flu/July_2025/runs/dataset_{ha_na,pb2_pa,pb2_na,pa_ha}_human_h3n2_2024_random_cv4_pinned_length_hopcroft_karp_n1698_seed42`.
Models under `models/flu/July_2025/runs/` with `human_h3n2_2024_n1698_seed42` in their names.
Figures under `results/flu/July_2025/dataset_ha_na_h3n2_2024_random_cv4_pinned_length/`:
`site_importance/site_importance_codon_gain_trace.png`,
`site_importance/site_shuffle_refit_codon_shap.png`,
`negative_pair_ambiguity_site_nt/negative_pair_ambiguity_nt_min.png`.
Method detail in `docs/plans/2026-08-28_per_site_nt_features_plan.md`, capacity in
`docs/results/2026-09-08_cds_pair_capacity.md`, population definition in
`docs/results/2026-09-07_cds_length_survey.md`.
