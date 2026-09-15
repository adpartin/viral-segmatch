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
5. positives and negatives in each CV fold.

### Pinned-length site features

Pinned-length site features assign one feature column to each nucleotide, codon, or amino-acid (aa) position after retaining one configured CDS length. This is the current production method and the one experiments 1 to 3 use.

### Dataset checks every experiment must pass

- Positive pair keys are unique.
- No CDS hash occurs in more than one CV split within a fold.
- No generated negative is an observed positive from the full pre-selection positive universe.
- Every importance row can be traced to a model run, fold, protein, and site.

### What the model predicts

The label records whether two segment sequences were observed together in one isolate. Strong performance does not by itself establish biochemical compatibility, coevolution, or reassortment fitness. Shared lineage, time, geography, and sampling structure can all contribute to the signal.

## Experiment 1: cross-year HA-NA importance

### Question

Do models trained separately on Human-H3N2-2024 and Human-H3N2-2025 rely on similar HA and NA
codon sites?

### Design

Use `conf/bundles/flu_ha_na_human_h3n2_2024_random_cv4_pinned_length_hopcroft_karp.yaml` as the
2024 dataset reference. Use the saved `resolved_config.yaml` files from the existing 2024 HA-NA
codon runs as the feature and model references. Reuse the 2024 results. These
runs are summarized in `docs/results/2026-09-08_h3n2_2024_progress_report.md`.

For 2025:

- create a sibling dataset bundle that changes only the year
- train per-site codon models for the performance and importance comparison
- keep the full Hopcroft-Karp population

The 2024 and 2025 populations contain 1,698 and 1,337 positives, respectively. They will not be
downsampled to the same size, so sample size may contribute to differences in their gain rankings.
The July 2025 corpus contains only a partial 2025 season.

### Comparisons

Produce for 2025 year the figures as section 3 of
`docs/results/2026-09-08_h3n2_2024_progress_report.md`, so that 2024 and 2025 can be read side by side:
the 3-panel importance barplot of gain, SHAP and permutation, and the gain trace along HA and
NA. `src/analysis/plot_site_importance.py` writes both. The shuffle-and-refit figure from that
section is out of scope here (computationally expensive).

Report the share of total gain as a two-row table, 2024 and 2025, giving the HA and NA shares and
the share falling in the top 25 sites (check first table in section 3). Fold-to-fold variation is the error bar on the barplot,
which covers the sites shown; the per-fold CSV carries it for every site.

Then compare the years directly, reporting HA and NA separately:

- Spearman correlation across all sites
- Overlap and Jaccard similarity for the top 12 and top 25 sites

### Required outputs

Beyond the figures and tables named above:

- a dataset audit for each year;
- a cross-year comparison CSV;
- a results note that separates measurements from interpretation.

### Acceptance checks

In addition to the shared checks, HA and NA must have identical site counts and coordinates in both
years. Every retained sequence must be complete and at its configured pin.

### Results

Both populations use the same HA and NA pins, so the coordinates match: HA 567 sites, NA 470.

Gain, SHAP and permutation importance. Human-H3N2-2024 above, Human-H3N2-2025 below.

![HA-NA codon-site importance, Human-H3N2-2024](../results/figs/2026-09-08_ha_na_codon_importance_barplot.png)

![HA-NA codon-site importance, Human-H3N2-2025](../results/figs/2026-09-15_ha_na_2025_codon_importance_barplot.png)

Gain along each protein. Human-H3N2-2024 above, Human-H3N2-2025 below.

![HA-NA codon-site gain trace, Human-H3N2-2024](../results/figs/2026-09-08_ha_na_codon_gain_trace.png)

![HA-NA codon-site gain trace, Human-H3N2-2025](../results/figs/2026-09-15_ha_na_2025_codon_gain_trace.png)

| year | gain by protein | gain in top 25 sites |
|---|---|---:|
| 2024 | HA 55.5%; NA 44.5% | 59.1% |
| 2025 | HA 60.4%; NA 39.6% | 60.6% |

Sites the two years share in their top-N lists. Each column ranks over a different set of
candidates, so `top N` selects different sites in each. `combined` ranks HA and NA together over
all 1,037 sites, so the two proteins compete for the same N slots and a shift in the gain balance
costs shared sites on its own. `HA` ranks over its 567 sites and `NA` over its 470, so each of
those columns compares N sites of that protein alone.

The three top-10 lists for 2024 show what that means. Entries are (protein, site) pairs, so HA239
and NA239 are different sites.

```
combined top-10:  HA544, HA36, NA24, NA284, NA310, NA400, HA531, NA223, HA129, NA239
HA       top-10:  HA544, HA36, HA531, HA129, HA239, HA87, HA286, HA95, HA390, HA14
NA       top-10:  NA24, NA284, NA310, NA400, NA223, NA239, NA140, NA308, NA244, NA462
```

The combined list holds 4 HA and 6 NA. The HA column adds six HA sites the combined list has no
room for, and the NA column adds four. A row of the table therefore compares three separate
questions at the same N, not one question three ways.

| top N | combined | HA | NA |
|---:|---:|---:|---:|
| 10 | 4 | 6 | 6 |
| 15 | 8 | 10 | 7 |
| 20 | 11 | 12 | 10 |
| 25 | 13 | 15 | 13 |
| 30 | 16 | 18 | 17 |
| 35 | 20 | 24 | 22 |
| 40 | 22 | 27 | 24 |
| 45 | 24 | 32 | 26 |
| 50 | 29 | 37 | 28 |

## Experiment 2 (rerequisite to Exp. 3): 28-pair capacity audit

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

For every pair, report the five population counts defined above. Also report the distribution of
per-sequence reuse before matching and the retained-isolate overlap for pairs that share a protein.

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
- a matrix of Hopcroft-Karp counts;
- a short audit of PB1 and NS1 eligibility;
- a list of pairs that need aligned rather than pinned-length coordinates.

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

