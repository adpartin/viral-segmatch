# Cross-year site feature importance, 28-pairs, and aligned CDS features

**Status: IN PROGRESS**

## Goal

Follow up questions/tasks following the `docs/results/2026-09-08_h3n2_2024_progress_report.md` report (Human-H3N2-2024):

1. Do the same sequence sites dominate feature importance in different years? E.g., compare Human-H3N2-2024 vs Human-H3N2-2025.
2. How do prediction performance, pair capacity, and the share of gain on each protein vary across 28-pairs; the 8 major proteins, c(8,2)?
3. Can codon-preserving sequence alignment retain records that pinned-length filtering drops? We need to determine whether the alignment effort is worth it. Consider `2026-09-07_cds_length_survey.md` in general, and specifically its "Pin reach by year" section, which covers 2015-2025.

The cross-year comparison and the initial 28-pairs screen can use the existing pinned-length pipeline.

Alignment will be evaluated separately and will enter the production pipeline only if it retains meaningful data.

Alignment is not required for Experiment 1, because HA and NA keep the same pin in 2024 and 2025.
What a validated alignment would add is evidence that a shared site is a shared *homologous*
position. `docs/results/2026-09-07_cds_length_survey.md` states that equal CDS length does not prove
positional homology, and nothing in the current pipeline checks it.

## Scope

- Population (metadata filters): Human-H3N2.
- Baseline year for the 28-pairs screen: 2024.
- Years for the feature importance comparison: 2024 and 2025.
- Proteins: PB2, PB1, PA, HA, NP, NA, M1, NS1.
- Baseline cross-year schema pair: HA-NA.
- Positive pairs: observed same-isolate pairs, deduplicated by `nt_cds` pair key.
- Positive selection: Hopcroft-Karp, so each retained CDS occurs at most once in each slot.
- Splitting: 4-fold random CV with negatives generated within each fold.
- Primary features: nucleotide 6-mers and per-site codons.
- Primary importance measure: fold-averaged LightGBM gain.

Experiments 1 to 3 share one pin table, so their site coordinates and importance maps stay
comparable. Every protein is pinned to its Human-H3N2-2024 modal complete-CDS length, and the same
value is used in 2025. Pinning two compared years to lengths that differ by an indel would shift
every position downstream of it, so a site index would not mean the same place in both.

| Segment ID | protein | pin (nt) | source |
| --- |---|---:|---|
| 1 | PB2 | 2,280 | `conf/virus/flu.yaml` |
| 2 | PB1 | 2,277 | bundle override |
| 3 | PA | 2,151 | `conf/virus/flu.yaml` |
| 4 | HA | 1,701 | `conf/virus/flu.yaml` |
| 5 | NP | 1,497 | `conf/virus/flu.yaml` |
| 6 | NA | 1,410 | `conf/virus/flu.yaml` |
| 7 | M1 | 759 | `conf/virus/flu.yaml` |
| 8 | NS1 | 693 | bundle override |

The six values in `conf/virus/flu.yaml` already equal the Human-H3N2-2024 mode, so that file needs
no change. PB1 and NS1 come from a bundle-level `virus.cds_length` override, which merges with the
six rather than replacing them. How far each pin reaches into earlier years is measured in the
"Pin reach by year" section of `docs/results/2026-09-07_cds_length_survey.md`.

Note that the "July 2025" corpus contains a partial 2025 season. We have to state in our results (we reserve this for final publication).

## Current evidence

### PB1 is the one protein the pins cannot rescue

In Human-H3N2-2024 only 55.3% of isolates have a complete PB1, and alignment cannot change that,
because the missing bases are absent from the assemblies. See "Why PB1 retains about half its
isolates" in `docs/results/2026-09-07_cds_length_survey.md`.

### Pair capacity differs even under the same metadata filters

The existing six-protein survey found 616-1,987 Hopcroft-Karp positives across 15 pairs in Human-H3N2-2024. M1 pairs had the smallest populations because M1 has few distinct sequences. The 28-pairs experiment must therefore report native sample size and sequence diversity beside model performance.

### Existing code can be reused

- `conf/bundles/flu_28_major_protein_pairs_master.yaml` and its 28 child bundles enumerate all
  protein pairs. Their current population and training settings are not the settings in this plan,
  so new experiment bundles must override them explicitly.
- `src/analysis/aggregate_allpairs_results.py` already builds a 28-pair summary and heatmaps. It
  should be extended only where the current LightGBM/site-feature outputs require it.
- `src/analysis/summarize_cds_lengths.py` provides the per-protein completeness and length audit.
- `src/analysis/summarize_pair_capacity.py` provides positive-pair and Hopcroft-Karp counts. Its
  current common-cohort design is useful for controlled comparisons but is not the primary
  population definition for the 28-pairs screen.
- `src/analysis/plot_site_importance.py` already writes per-site gain, SHAP, and permutation
  importance by fold.

## Definitions and interpretation

### Population for a schema pair

Each schema pair will be built independently from Human-H3N2 isolates in the specified year. An isolate is eligible for a pair when it has both required proteins as complete CDS records in the coordinate system used by that experiment. This holds host, subtype, and year fixed, but it does not force different schema pairs to retain the same isolates.

For each pair, report these counts in order:

1. eligible isolates;
2. unique observed positive pairs after `nt_cds` pair-key deduplication;
3. distinct slot-A and slot-B sequences;
4. positives retained by Hopcroft-Karp;
5. positives and negatives in each CV fold. This one needs built CV folds, so Experiment 3
   produces it and Experiment 2 does not.

### Pinned-length site features

Pinned-length site features assign one feature column to each nucleotide, codon, or amino-acid (aa) position after retaining one configured CDS length. This is the current production method and the one experiments 1 to 3 use.

### Dataset checks every experiment must pass

- Positive pair keys are unique.
- No CDS hash occurs in more than one CV split within a fold.
- No generated negative is an observed positive from the full pre-selection positive universe.
- Every importance row can be traced to a model run, fold, protein, and site.

### What the model predicts

The label records whether two segment sequences were observed together in one isolate. Strong performance does not by itself establish biochemical compatibility, coevolution, or reassortment fitness. Shared lineage, time, geography, and sampling structure can all contribute to the signal.

## Experiment 1: cross-year HA-NA importance — DONE (2026-09-15)

### Question

Do models trained separately on Human-H3N2-2024 and Human-H3N2-2025 rely on similar HA and NA
codon sites?

### Methods

The 2024 analysis used
`conf/bundles/flu_ha_na_human_h3n2_2024_random_cv4_pinned_length_hopcroft_karp.yaml` and the saved
`resolved_config.yaml` files from its codon runs. The 2025 analysis used a sibling bundle that
changed only the year and otherwise reused the same feature and model settings. Both analyses kept
their full Hopcroft-Karp populations: 1,698 positives in 2024 and 1,337 in 2025. They were not
downsampled to the same size.
The July 2025 corpus contains only a partial 2025 season.

`src/analysis/plot_site_importance.py` computed gain, SHAP, and permutation importance. The
cross-year ranking uses fold-averaged, normalized gain. The comparison was run three ways:

- `combined`: HA and NA compete for the same top-N positions;
- `HA`: sites are ranked within HA only;
- `NA`: sites are ranked within NA only.

The null comparison treats a site as eligible when it has more than one observed value
(`n_values > 1`). If `V_2024` and `V_2025` are the eligible sets, two independent random top-N
lists have expected overlap
`|V_2024 ∩ V_2025| × (N / |V_2024|) × (N / |V_2025|)`.
This is a descriptive baseline, not a significance test: it treats eligible sites as independent
and equally likely to be selected, which is not true for correlated sites.

Both datasets contain complete sequences at the same pins and use the same feature coordinates:
567 HA sites and 470 NA sites.

### Results

The barplots show gain, SHAP, and permutation importance for Human-H3N2-2024 and
Human-H3N2-2025.

![HA-NA codon-site importance, Human-H3N2-2024](../results/figs/2026-09-08_ha_na_codon_importance_barplot.png)

![HA-NA codon-site importance, Human-H3N2-2025](../results/figs/2026-09-15_ha_na_2025_codon_importance_barplot.png)

The gain traces show where gain falls along HA and NA.

![HA-NA codon-site gain trace, Human-H3N2-2024](../results/figs/2026-09-08_ha_na_codon_gain_trace.png)

![HA-NA codon-site gain trace, Human-H3N2-2025](../results/figs/2026-09-15_ha_na_2025_codon_gain_trace.png)

The division of gain between the two proteins changed modestly, while its concentration in the
combined top 25 sites was similar.

| year | gain by protein | gain in combined top 25 sites |
|---|---|---:|
| 2024 | HA 55.5%; NA 44.5% | 59.1% |
| 2025 | HA 60.4%; NA 39.6% | 60.6% |

The two years shared 13 of their combined top 25 sites, 15 of the top 25 HA sites, and 13 of the
top 25 NA sites. All three overlaps were much larger than expected under the varying-site null.

| ranking | eligible in 2024 | eligible in 2025 | eligible in both | shared top 25 | shared/N | expected | enrichment |
|---|---:|---:|---:|---:|---:|---:|---:|
| combined | 987 | 946 | 931 | 13 | 52% | 0.62 | 20.9x |
| HA | 542 | 519 | 511 | 15 | 60% | 1.14 | 13.2x |
| NA | 445 | 427 | 420 | 13 | 52% | 1.38 | 9.4x |

The complete top-10-to-top-50 comparison is in
`results/flu/July_2025/cross_year_site_importance/ha_na_human_h3n2_2024_vs_2025/site_importance_comparison.csv`.

The leading sites therefore recur across the two annual fits more often than expected if varying
sites were selected uniformly. This does not show that the full rankings are identical or that the
difference is a biological year effect. The comparison does not control for the smaller 2025
population, the partial 2025 season, or correlation among sites.

## Experiment 2 (prerequisite to Experiment 3): 28-pair capacity audit — DONE (2026-09-16)

### Question

How much usable and sequence-unique Human-H3N2-2024 data is available for each of the 28 pairs
formed from PB2, PB1, PA, HA, NP, NA, M1, and NS1?

### Population rules

Use the same metadata filters for every pair, but build each pair from its own eligible isolates.
Do not require a common 8-protein isolate cohort for the primary analysis. Requiring PB1 from
every isolate would remove about 45% of the population from pairs that do not contain PB1.

Use the existing pins for PB2, PA, HA, NP, NA, and M1, and add the two the config does not carry.
The two are not the same kind of addition. NS1 at 693 nt is population-specific but not
year-specific: it is the modal complete-CDS length in every Human-H3N2 year from 2015 to 2025, and
what prevents a corpus-wide value is the subtype split, since H1N1 is predominantly 660 nt. PB1 at
2,277 nt is both population- and year-specific, because the Human-H3N2 mode is 2,274 nt through
2023. Recheck both against the input data before use. PB1 pairs must be marked as
completeness-selected because their eligible population is much smaller.

For every pair, report counts 1 to 4 defined above. Also report the distribution of per-sequence
reuse before matching, and the retained-isolate overlap with the combinations whose two schema
pairs share a protein marked. The distinct-sequence counts and the reuse distribution answer
different questions. Hopcroft-Karp keeps at most one positive per distinct sequence, so the
smaller of a pair's two counts is a hard ceiling on its matched count. The reuse distribution
describes how concentrated the observed positives are, which helps explain a low retained share,
because a sequence with many distinct partners contributes only one of them. How close the
matching comes to its ceiling depends on the whole bigraph rather than on the reuse distribution
alone. Reuse is counted after the positives are deduplicated on the pair key, so a sequence's
reuse count is the number of distinct partner sequences it was observed with rather than the
number of isolates it occurs in.

### Implementation

The override in Scope is used rather than an edit to `conf/virus/flu.yaml` because that file is
per-virus and shared with H1N1 work, where NS1 is 660 nt and PB1 is 2,274 nt, so writing
Human-H3N2 values there would make `check_cds_length` raise on those populations. The override
reaches both `summarize_pair_capacity.py:224` and `dataset_segment_pairs.py:694`, which read the
same key.

Adapt `src/analysis/summarize_pair_capacity.py` so it can produce a pair-specific-cohort table for
all eight proteins. Preserve its current common-cohort mode because that remains useful as a
sensitivity analysis. Do not silently change the meaning of its existing results.

The capacity audit is a gate before training. Review the table before deciding whether very small
pairs should be trained, grouped into a low-capacity stratum, or reported as data-limited. Do not
choose a minimum count before measuring the 28 populations.

### Required outputs

- one 28-row pair-capacity CSV;
- a readable capacity table sorted by segment number and a second view sorted by matched count;
- an 8 x 8 matrix of Hopcroft-Karp counts, written as a CSV and a heatmap;
- a short audit of PB1 and NS1 eligibility that cites the length survey instead of restating it;
- a check of whether any pair needs aligned rather than pinned-length coordinates.

The last check is expected to return no pairs. Among complete CDS in Human-H3N2-2024 the lowest
`frac at mode` is NS1 at 0.991, so the pins already retain almost every complete CDS. What PB1
loses is incompleteness, which alignment cannot recover. Alignment is examined in Experiment 4.

`results/` is not tracked by git, so copy the heatmap into `docs/results/figs/` and record the
capacity table in this plan.

### Results

One command produced every output, using the defaults in
`src/analysis/summarize_pair_capacity.py`:

```
python -m src.analysis.summarize_pair_capacity \
  --out_dir results/flu/July_2025/pair_capacity_8_proteins
```

The 21 pairs without PB1 drew on 5,156 to 5,338 eligible isolates. The 7 pairs with PB1 drew on
2,937 to 2,945, because only 55.1% of Human-H3N2-2024 isolates have a complete PB1 CDS at the
2,277 nt pin. See "Why PB1 retains about half its isolates" in
`docs/results/2026-09-07_cds_length_survey.md`. NS1 cost
almost no isolates at the 693 nt pin, so its pairs kept full-size populations and lost their
capacity at the matching step instead.

Hopcroft-Karp kept 440 positives at the least (M1-NS1), 1,126 at the median, and 2,042 at the most
(PB2-HA).

![28-pair Hopcroft-Karp capacity, Human-H3N2-2024](../results/figs/2026-09-16_h3n2_2024_pair_capacity_matrix.png)

| Pair ID | pair | eligible isolates | positives | Unique slot-A | Unique slot-B | HK matched | HK share |
|---|---|---:|---:|---:|---:|---:|---:|
| 1-4 | PB2-HA | 5,329 | 3,796 | 2,810 | 2,681 | 2,042 | 53.8% |
| 1-3 | PB2-PA | 5,324 | 3,837 | 2,808 | 2,712 | 2,030 | 52.9% |
| 3-4 | PA-HA | 5,329 | 3,805 | 2,711 | 2,682 | 1,944 | 51.1% |
| 1-6 | PB2-NA | 5,167 | 3,532 | 2,745 | 2,203 | 1,745 | 49.4% |
| 4-6 | HA-NA | 5,173 | 3,466 | 2,634 | 2,203 | 1,698 | 49.0% |
| 3-6 | PA-NA | 5,167 | 3,520 | 2,649 | 2,202 | 1,689 | 48.0% |
| 1-5 | PB2-NP | 5,318 | 3,484 | 2,800 | 1,830 | 1,512 | 43.4% |
| 4-5 | HA-NP | 5,323 | 3,382 | 2,673 | 1,830 | 1,482 | 43.8% |
| 3-5 | PA-NP | 5,318 | 3,455 | 2,703 | 1,832 | 1,459 | 42.2% |
| 1-2 | PB2-PB1 | 2,945 | 2,349 | 1,788 | 1,790 | 1,404 | 59.8% |
| 2-4 | PB1-HA | 2,945 | 2,327 | 1,790 | 1,756 | 1,392 | 59.8% |
| 2-3 | PB1-PA | 2,939 | 2,332 | 1,786 | 1,710 | 1,341 | 57.5% |
| 5-6 | NP-NA | 5,162 | 3,091 | 1,794 | 2,197 | 1,287 | 41.6% |
| 2-6 | PB1-NA | 2,939 | 2,237 | 1,786 | 1,480 | 1,212 | 54.2% |
| 2-5 | PB1-NP | 2,942 | 2,162 | 1,787 | 1,227 | 1,041 | 48.1% |
| 1-8 | PB2-NS1 | 5,313 | 3,204 | 2,800 | 1,120 | 995 | 31.1% |
| 4-8 | HA-NS1 | 5,318 | 3,151 | 2,676 | 1,119 | 959 | 30.4% |
| 3-8 | PA-NS1 | 5,315 | 3,214 | 2,705 | 1,121 | 952 | 29.6% |
| 6-8 | NA-NS1 | 5,156 | 2,794 | 2,196 | 1,092 | 853 | 30.5% |
| 5-8 | NP-NS1 | 5,307 | 2,540 | 1,824 | 1,114 | 806 | 31.7% |
| 1-7 | PB2-M1 | 5,332 | 3,128 | 2,813 | 812 | 726 | 23.2% |
| 4-7 | HA-M1 | 5,338 | 3,018 | 2,686 | 811 | 720 | 23.9% |
| 3-7 | PA-M1 | 5,335 | 3,082 | 2,716 | 812 | 707 | 22.9% |
| 2-8 | PB1-NS1 | 2,937 | 2,029 | 1,788 | 781 | 704 | 34.7% |
| 6-7 | NA-M1 | 5,176 | 2,620 | 2,206 | 800 | 657 | 25.1% |
| 5-7 | NP-M1 | 5,326 | 2,356 | 1,833 | 808 | 627 | 26.6% |
| 2-7 | PB1-M1 | 2,945 | 1,992 | 1,790 | 582 | 517 | 26.0% |
| 7-8 | M1-NS1 | 5,323 | 1,840 | 806 | 1,122 | 440 | 23.9% |

Sequence diversity explains the ordering better than the eligible count does. The smaller of a
pair's two distinct-sequence counts is a hard ceiling on its matched count. Reuse describes how
concentrated the observed positives are and helps explain the `HK share`. How close the matching
comes to its ceiling depends on the whole bigraph, so neither number predicts it on its own.
Positives are deduplicated on the `nt_cds` pair key before reuse is counted, so a sequence's reuse
count is the number of distinct partner sequences it was observed with, not the number of isolates
it occurs in. The ranges below run over the 7 pairs each protein takes part in. The distribution is long-tailed, so the median is 1 for every protein and pair, and
the mean and the maximum are what separate them.

| protein | distinct sequences | mean reuse | max reuse | one partner only |
|---|---:|---:|---:|---:|
| PB2 | 1,788-2,813 | 1.11-1.37 | 108 | 87.6-94.5% |
| PB1 | 1,786-1,790 | 1.11-1.31 | 51 | 87.7-94.0% |
| PA | 1,710-2,716 | 1.13-1.41 | 117 | 86.1-93.6% |
| HA | 1,756-2,686 | 1.12-1.42 | 136 | 84.7-93.5% |
| NP | 1,227-1,833 | 1.29-1.90 | 398 | 80.1-88.8% |
| NA | 1,480-2,206 | 1.19-1.60 | 138 | 84.0-92.1% |
| M1 | 582-812 | 2.28-3.85 | 860 | 72.3-79.5% |
| NS1 | 781-1,122 | 1.64-2.87 | 977 | 73.4-82.3% |

The two bottlenecks act at different stages. M1 and NS1 limit the maximum matching, because they
supply relatively few distinct sequences. Their high partner counts also contribute to low
`HK share` values. The 13 pairs containing one of them are the 13 lowest matched counts in the
table, and the first pair containing neither is PB1-NP at 1,041. M1-NS1 is
the floor at 440, and the bound above it is loose, because the pair has 806 distinct M1 sequences
and keeps 440 positives. PB1 limits eligibility instead, because 44.7% of 2024 isolates have no
complete PB1 CDS. Among the isolates that remain, PB1 has low sequence reuse, so PB1-HA still
keeps 1,392 positives from 2,945 isolates.

The matchings do not retain the same isolates. Over all 378 combinations of two schema pairs,
isolate Jaccard runs from 0.112 to 0.568 with a median of 0.234. Over the 168 combinations whose
schema pairs share a protein, it runs from 0.188 to 0.568 with a median of 0.353. The lowest of
all is M1-NS1 against PB1-PA and the highest is PB1-HA against PB2-PB1. Two pairs are therefore
less comparable than the shared metadata filters suggest, and a performance difference between
them is not measured on one population.

No pair needs aligned rather than pinned-length coordinates for this screen. Among complete CDS in
Human-H3N2-2024 the lowest `frac at mode` is NS1 at 0.991 and PB1 is at 0.994, so the pins retain
almost every complete CDS. PB1's loss comes from incomplete assemblies, which alignment cannot
recover.

The six output files are in `results/flu/July_2025/pair_capacity_8_proteins/`, which is not
tracked by git. The heatmap is copied to
`docs/results/figs/2026-09-16_h3n2_2024_pair_capacity_matrix.png`.

## Experiment 3: pinned-length 28-pairs screen

### Question

Does within-season segment-matching performance differ across the 28 schema pairs, and are weak
pairs associated with particular proteins or limited pair capacity?

### Dataset and model design

For every pair that passes the capacity audit:

- use its native Hopcroft-Karp population;
- use identical dataset rules and 4 CV folds;
- train nucleotide 6-mer and per-site codon LightGBM models;
- use threshold 0.5 and 1:1 class balance;
- save raw test predictions and standard dataset audits.

This is 28 pairs x 2 feature representations x 4 folds, or 224 model fits if all pairs pass.
Run the dataset audits before starting the full training matrix.

Per-site nucleotide models are deferred from the complete screen because codons retained similar
performance with one third as many columns in the four-pair experiment. Add nucleotide-site models
for selected strong, weak, or discrepant pairs after the first screen.

aa models are also deferred. Negatives built and blocked in nucleotide space can collapse
to the same aa pair with conflicting labels. A fair aa experiment must construct,
deduplicate, and block pairs in aa space and must be reported as a different dataset
population.

### Comparisons

Report, for every pair and feature representation:

- AUC-ROC, F1 macro, precision, recall, and Brier score as fold mean and standard deviation;
- pooled confusion counts;
- eligible, unique-positive, and Hopcroft-Karp counts;
- the difference between codon and k-mer performance;
- each protein's share of total gain in the codon model, one number per slot.

Save the normalized per-fold gain for every codon model. `src/analysis/plot_site_importance.py`
already writes it, so this costs no extra model fits. The first screen reports the per-protein
share of gain only. It does not interpret 56 per-site importance maps, because the pairs differ in
population size, capacity and sequence diversity, and Experiment 1 is where importance is examined
carefully.

Plot symmetric 8 x 8 heatmaps for AUC-ROC, F1 macro, precision, and recall. Plot performance against
Hopcroft-Karp count and against per-protein sequence diversity. Treat these as descriptive
associations, not explanations of performance.

Use `src/analysis/aggregate_allpairs_results.py` where possible. Keep the older 28-pair experiment
separate by using a new bundle tag and output namespace.

### Primary interpretation limits

- Native pair counts make the screen representative of available data but do not isolate protein
  identity from sample size or diversity.
- A fixed-count sensitivity analysis can compare pairs within capacity strata. It cannot equalize
  sequence diversity or negative difficulty.
- Poor performance can reflect weak pairing signal, label ambiguity, limited diversity, or limited
  sample size. The first heatmap will not distinguish these explanations by itself.

## Experiment 4: codon-preserving alignment pilot

### Question

Does alignment add enough valid site-feature data to justify new production-pipeline support?

### Aligned site features

Aligned site features assign one feature column to each homologous alignment position. Coding sequences must be aligned in a way that preserves the reading frame. The proposed pilot translates each CDS, aligns the proteins, and projects protein gaps back to codon triplets. An unrestricted nucleotide alignment is not acceptable because it can introduce frame-breaking gaps.

An alignment gap, an unknown base, and unobserved sequence are different states:

- a gap represents an inferred biological insertion or deletion relative to other sequences;
- an unknown base is present in the record but unresolved;
- unobserved sequence is absent because the assembly or CDS is truncated.

These states must not be encoded as the same category. In particular, a truncated PB1 record must not be presented as evidence of a biological deletion.

### Audit before alignment

Seven of the eight proteins already have a modal complete-CDS length that does not change across
Human-H3N2 2015-2025, so this audit reduces to PB1. Extend the length survey by protein and year
for 2023-2025 to confirm that, then, for every non-modal length, separate:

1. complete CDS records with plausible biological insertions or deletions;
2. incomplete CDS records caused by missing start or stop sequence;
3. records with internal stops or unresolved bases;
4. possible annotation inconsistencies.

Report both isolates and distinct CDS sequences. Alignment can place complete biological length
variants into a common coordinate system. It cannot recover unsequenced bases, improve assembly
completeness, or create new sequence diversity.

### Pilot proteins

Pilot the method on PB1 and NS1 because their lengths change across the years of interest. Use
PB2-PB1 as the first paired modeling case if PB1 passes alignment validation; PB2 supplies a stable
partner and the pair has a direct polymerase interpretation. Keep the existing pinned-length
population as the control.

### Alignment method

Prototype the alignment outside the dataset builder first:

1. translate each complete CDS using the current translation rules;
2. align aa sequences with a reproducible tool and version. MAFFT and pyFAMSA
   (https://github.com/althonos/pyfamsa) are the progressive-aligner candidates. pyHMMER
   (https://github.com/althonos/pyhmmer) is a separate option if a profile-HMM alignment, which
   gives a column space that does not shift as sequences are added, is under consideration;
3. project each aa gap back to a three-nucleotide codon gap;
4. retain an explicit mapping from alignment column to original residue/codon coordinate;
5. encode gaps, unknown residues, and missing sequence separately;
6. record the exact input sequence hashes and alignment command.

Do not add general alignment knobs to the production pipeline until the pilot passes its checks and
shows a useful increase in eligible data.

### Validation

- Removing alignment gaps reproduces the original input sequence exactly.
- Every inserted CDS gap has a length divisible by three.
- Translating the ungapped aligned CDS reproduces the original protein.
- The procedure introduces no internal stop codons.
- Known PB1 and NS1 length forms align to plausible terminal or internal locations rather than
  being spread across many arbitrary gaps.
- Repeated runs with the same inputs and tool version produce identical coordinates.
- Gap-heavy and missing-heavy columns are reported and can be excluded in a sensitivity analysis.

For a descriptive comparison across years, a pooled 2023-2025 alignment may be used if it is
labeled as pooled. For a prospective train-on-year-T, test-on-year-T+1 experiment, test-year
sequences must not determine the training feature coordinates. That setting needs a training-only
reference/profile and a documented rule for insertions not represented in the training alignment.

### Decision gate

Promote aligned features into the dataset pipeline only if:

- the validation checks pass;
- alignment retains a meaningful number of complete sequences or enables a previously excluded
  protein/year comparison;
- the added columns have a stable biological interpretation;
- conclusions are not driven only by gap or missingness indicators.

PB1 records truncated before the terminal stop remain excluded from the primary analysis even if
they can be padded. They may be examined only in a clearly labeled missing-data sensitivity arm.

## Execution order

1. Build and audit the pinned-length 2024 and 2025 HA-NA datasets.
2. Run the codon cross-year importance comparison.
3. Produce the 2024 eight-protein, 28-pair capacity audit.
4. Build and audit the 28 pinned-length datasets.
5. Run the k-mer and codon 28-pairs screen and aggregate the results.
6. Complete the PB1/NS1 alignment feasibility audit and prototype.
7. Rerun only the pairs or years for which alignment materially improves eligibility.
8. Decide whether the evidence supports a publication scope, a narrower follow-up, or an archived
   negative/benchmark result.

Steps 1-3 produce useful results without waiting for alignment. Steps 4-5 can proceed for pairs
with validated pinned-length coordinates while the alignment pilot is being evaluated.

## Reproducibility and reporting

Every dataset and model run must record:

- input corpus version and git commit;
- metadata filters;
- CDS completeness and coordinate rules;
- pair-key alphabet and full positive-blocking universe;
- positive-selection method and seed;
- fold assignments and negative-generation scope;
- feature representation and alignment version, if used;
- model configuration and output paths.

Keep per-fold data. Do not report only means. Any strict held-out perturbation or masking analysis
must derive its site ranking inside each training fold; a ranking averaged over all folds is
descriptive and must not be used to select features for that same held-out data.

The final report should contain:

1. cross-year HA and NA importance comparisons;
2. the 28-pair capacity and performance matrices;
3. a direct statement of which pairs fail or weaken;
4. the alignment yield and validation results;
5. limitations from sampling, partial 2025 coverage, correlated sites, and metadata shortcuts;
6. a recommendation to continue, narrow the scope, or archive the project.

## Non-goals

- Inferring missing PB1 bases from neighboring sequences.
- Treating padding as biological alignment.
- Building one alignment across subtypes in this first effort.
- Running every feature representation over every pair and year before the screening results are
  known.
- Claiming that a predicted observed pair is biologically compatible.

## Planned code and artifacts

Names are provisional until implementation begins.

| item | purpose |
|---|---|
| `src/analysis/compare_site_importance_across_years.py` | compare annual per-fold importance tables and produce shared-coordinate plots |
| `src/analysis/summarize_pair_capacity.py` | add an explicit pair-specific-cohort mode while preserving the current common-cohort mode |
| `src/analysis/aggregate_allpairs_results.py` | support the new LightGBM k-mer/codon run naming and additional metric heatmaps |
| `src/preprocess/align_cds_by_protein.py` | alignment pilot; added only after its input/output contract is fixed |
| `results/flu/July_2025/cross_year_site_importance/` | cross-year tables, audits, and figures |
| `results/flu/July_2025/pair_capacity_8_proteins/` | 28-pair capacity audit |
| `results/flu/July_2025/all_pairs_human_h3n2_2024/` | 28-pairs summaries and figures |
| `results/flu/July_2025/cds_alignment_pilot/` | alignment audit, mappings, and validation results |

