# Segment matching: progress report and scope decision

**Date:** 2026-09-08

**Question.** Given two influenza A gene-segment sequences, can a classifier distinguish an observed same-isolate pair from a generated pair that was not observed together?

a) **Within-season matching**: Train and test on disjoint pairs from the same year. This represents an established season with existing paired sequences for training.

b) **One-season-ahead matching**: Train on year T and test on year T+1, without using year T+1 data for model training or fine-tuning. This represents the beginning of a new season, before existing paired sequences have accumulated.

**Purpose of this report.** Summarize the completed Human H3N2 2024 work and decide whether two
bounded follow-ups justify a publication, or whether the project should be archived as a technical
report. The within-season setting has now been tested with unique nucleotide sequences on both
sides. The one-season-ahead setting remains open.

---

## 1. Dataset population

The completed experiments in sections 2 and 3 use the following population.

| filter | positives | used for |
|---|---:|---|
| Human H3N2 2024, complete CDS at pinned length, Hopcroft-Karp matched, sampled to a common size | 1,698 per pair | four-pair comparison, feature importance, shuffling and refitting |

The dataset construction enforces that **no nucleotide CDS is used in more than one retained positive pair**.
Generating negatives within each fold then gives zero exact sequence overlap between training and
test.

---

## 2. Four-pair experiment

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

## 3. Where the codon-site signal sits

Gain importance was computed for all four schema pairs. Each codon position is one feature. Gain is
the total reduction in training loss from tree splits using that feature. Gain was normalized
within each fold and then averaged across the four folds.

| Schema pair | Gain by protein | Gain in top 25 sites |
|---|---:|---:|
| HA-NA | HA 55.5%; NA 44.5% | 59.1% |
| PB2-PA | PB2 50.8%; PA 49.2% | 50.1% |
| PB2-NA | PB2 52.9%; NA 47.1% | 60.8% |
| PA-HA | PA 42.5%; HA 57.5% | 46.2% |

Both proteins contribute in every schema, while a relatively small set of sites carries much of
the total gain.

The same protein also receives a similar site ranking when paired with a different protein.
Across the four proteins, cross-partner Spearman correlations are 0.79-0.80, and 6-8 of each
protein's top 10 sites are shared between its two schema pairs. The important sites are therefore
largely protein-specific rather than unique to one schema pair.

### HA-NA shuffle and refit

For HA-NA, sites were ranked by gain. The selected values were shuffled across pair rows in the
training, validation, and test splits, and the model was then trained again. Results are reported
as the fraction of above-chance AUC-ROC lost:

`(baseline AUC - refit AUC) / (baseline AUC - 0.5)`

| Sites shuffled | Top-ranked sites | Random sites |
|---:|---:|---:|
| 10 | 0.287 | 0.006 |
| 25 | 0.749 | 0.000 |
| 50 | 0.954 | 0.026 |
| 100 | 1.008 | 0.031 |

A retrained model recovers most of the signal after the top 10 sites are shuffled. Shuffling the
top 50 removes almost all above-chance performance, whereas shuffling 100 random sites removes
only 3.1%.

These results identify sites used by the classifier, but they do not establish biological
importance or distinguish segment compatibility from lineage or population structure. The ranking
was averaged across all folds, so the shuffle/refit result should be treated as descriptive. A
fully held-out estimate would rank sites independently within each training fold.

---

## 4. Open question 1: one-season-ahead matching

Can a classifier trained on Human H3N2 sequences from 2024 match segments collected in 2025?

Build the 2024 and 2025 populations using the same complete-CDS, pinned-length, and
nucleotide-uniqueness controls used in the within-season experiment. Remove exact sequence overlap
between years,
generate negatives within each year, and use only 2024 data for training, validation, threshold
selection, and model tuning. Evaluate the final model on 2025.

Compare this result with a size-matched within-2024 experiment to measure the effect of transferring
to a new season. Start with per-site nucleotide and nucleotide 6-mer features.

---

## 5. Open question 2: why is precision lower than recall?

At the default threshold of 0.5, mean precision is lower than mean recall for all 16 configurations
in the four-pair experiment. Determine whether this is primarily an operating-threshold effect or
a consequence of how negative pairs are generated.

First, select the classification threshold using validation data only and report the test
precision-recall tradeoff. Second, stratify false positives by their sequence distance from
observed positive pairs in each schema pair. If the pattern persists, generate negatives
within controlled collection-time and distance ranges, retrain the models, and measure the effect
on performance.

The current datasets have a 1:1 class ratio. Precision in an application will also depend on the
prevalence of true matches in that setting.

---

## 6. Internal framing note: lessons from the collection-date study

**Do not include this section in the shared report.**

Jamie's collection-date study is used only as an example of a project whose scope and experimental
evidence were considered sufficient to proceed toward publication. Its prediction target is
different, and its results do not provide evidence for segment matching.

The useful lessons for designing the segmatch study are:

- Define a narrow biological population and prediction setting.
- Compare multiple segment pairs and sequence representations.
- Include controlled tests outside the training population, such as one-season-ahead matching.
- Report informative performance losses and failure modes, not only high average scores.
- Separate predictive performance from biological interpretation.
- Treat dataset construction, especially generated negatives, as part of the scientific design.

These principles motivate the completed four-pair comparison and the two open questions. They do
not require the segmatch experiments to reproduce the collection-date experiments.

---

## 7. Publication-or-archive decision

### What has held up

- Within-season co-assignment remains predictable after exact train-test sequence reuse is removed:
  per-site nucleotide F1 macro is 0.83 to 0.88 across four schema pairs.
- Results depend on the schema and representation. Nucleotide and codon features remain strong,
  while amino-acid features weaken sharply for three pairs.
- Top-ranked positions matter, but a retrained model can recover from corruption of a small number
  of them by using other positions.

### What is not established

The ranked positions have no biological validation. Section 3 shows which positions the model
depends on, not what they mean, and it does not separate segment compatibility from lineage or
population structure. One-season-ahead performance has not been evaluated under the current
controls, and the cause of the precision-recall difference remains unknown. The amino-acid
comparison also contains label collisions created by defining identity in nucleotide space. The
four-pair experiment uses one min-count sample (seed 42) and four folds.

### Candidate scope

If the group considers evaluation and failure analysis a sufficient contribution, a defensible
working claim is:

> Within-season influenza A segment co-assignment remains predictable after exact sequence reuse is
> removed, but performance depends on the segment pair and sequence representation. Nucleotide
> information discarded during translation and the construction of negative pairs both materially
> affect the result.

This is an evaluation result, not a claim of biological compatibility or coevolution.

### Bounded work needed to decide

1. **Prospective prediction:** run the controlled Human H3N2 2024-to-2025 experiment described in
   section 4, with a size-matched within-2024 reference.
2. **False positives:** calibrate the threshold using validation data, repeat the distance
   diagnostic on the four uniqueness-controlled schemas, and run a controlled negative-sampling
   experiment only if the same distance pattern remains.

The gain-ranked shuffle/refit experiment is complete and reported in section 3. If these results
are intended for a paper, repeat the min-count sampling with additional seeds after the two
remaining analyses define the final design. Do not expand now to a 28-pair sweep, ESM-2, or
additional metadata axes.

### The question for the PIs

Is a careful evaluation of when segment co-assignment is predictable, and when it fails, a
publication contribution for this group? If yes, complete only the two bounded analyses above and
then write. If biological interpretation of individual sites is required, these experiments are
unlikely to provide it; summarize the existing results as a technical report and archive the
project.

---

## Artifacts

Datasets `data/datasets/flu/July_2025/runs/dataset_{ha_na,pb2_pa,pb2_na,pa_ha}_human_h3n2_2024_random_cv4_pinned_length_hopcroft_karp_n1698_seed42`.
Models under `models/flu/July_2025/runs/` with `human_h3n2_2024_n1698_seed42` in their names.
Codon-site importance maps are under
`results/flu/July_2025/dataset_{ha_na,pb2_pa,pb2_na,pa_ha}_human_h3n2_2024_random_cv4_pinned_length_hopcroft_karp_n1698_seed42/site_importance/`.
The HA-NA figures used in section 3 are `site_importance_codon_gain_trace.png` and
`site_shuffle_refit_codon_gain.png`.
Method detail in `docs/plans/2026-08-28_per_site_nt_features_plan.md`, capacity in
`docs/results/2026-09-08_cds_pair_capacity.md`, population definition in
`docs/results/2026-09-07_cds_length_survey.md`.
