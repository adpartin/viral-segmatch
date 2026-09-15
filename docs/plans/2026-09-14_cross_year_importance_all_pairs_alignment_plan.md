# Cross-year site importance, all 28 pairs, and aligned CDS features

**Status: IN PROGRESS**

## Goal

Follow up questions/tasks following the `docs/results/2026-09-08_h3n2_2024_progress_report.md` report (Human-H3N2-2024):

1. Are the same sequence sites dominate in terms of feature importance in different years? E.g., compare Human-H3N2-2024 vs Human-H3N2-{2023,2025}.
2. Extend the prediction performance and feature imparance analysis to all 28 pairs; the 8 major proteins, c(8,2).
3. Can codon-preserving sequence alignment retain useful records that pinned-length filtering drops? We need to understand if we really need this. Consider `2026-09-07_cds_length_survey.md` in general, and specifically table 3 which focuses on years 2015-2025. We need to determine whether alignment effort worth it.

These tasks are related but do not need to be answered in one experiment. The cross-year comparison and the initial 28-pairs screen will use the current pinned-length pipeline.

Alignment will be evaluated separately and will enter the production pipeline only if it retains meaningful data without making site coordinates ambiguous. -> `Q:` isn't alignment required for feature importance if we want to do some interpretation with feature importance?

Alignment tools to consider:
- pyhmmer: https://github.com/althonos/pyhmmer; https://pyhmmer.readthedocs.io/en/stable/
- pyfamsa: https://github.com/althonos/pyfamsa; https://pyfamsa.readthedocs.io/en/stable/

## Scope

- Population (metadata filters): Human-H3N2.
- Baseline year for the 28-pairs screen: 2024.
- Years for the feature importance comparison: 2023, 2024, 2025.
- Proteins: PB2, PB1, PA, HA, NP, NA, M1, NS1.
- Baseline cross-year schema pair: HA-NA.
- Positive pairs: observed same-isolate pairs, deduplicated by `nt_cds` pair key.
- Positive selection: Hopcroft-Karp, so each retained CDS occurs at most once in each slot.
- Splitting: 5-fold random CV with negatives generated within each fold.
- Primary features: nucleotide 6-mers and per-site codons.
- Primary importance measure: fold-averaged LightGBM gain.

Note that the "July 2025" corpus contains a partial 2025 season. We have to state in our results (we reserve this for final publication).

## Current evidence

### Pinned-length site features are already feasible for most proteins

`docs/results/2026-09-07_cds_length_survey.md` shows that PB2, PA, HA, NP, NA, and M1 retain at
least 90% of Human-H3N2 isolates at their current pins in 2023-2025. NS1 also retains at least
99% at a population-specific length of 693 nt in those years.

PB1 is different. In Human-H3N2-2024, 55.3% of isolates have a complete PB1. Most 2,274-nt PB1
records are missing the terminal stop because the contig ends there. Alignment cannot reconstruct
those missing bases or make the records complete. Complete 2024 PB1 records are predominantly
2,277 nt and can still be analyzed as a smaller, explicitly selected population.

### Pair capacity differs even under the same metadata filters

The existing six-protein survey found 616-1,987 Hopcroft-Karp positives across 15 pairs in Human-H3N2-2024. M1 pairs had the smallest populations because M1 has few distinct sequences. An 28-pairs experiment must therefore report native sample size and sequence diversity beside model performance.

The primary 28-pairs analysis will not downsample every pair to the smallest pair. A global minimum would discard most observations from the larger pairs and would not equalize sequence diversity or negative difficulty. A fixed-count sensitivity analysis can be added within sensible capacity groups after the full capacity table is available.

### Existing code can be reused

- `conf/bundles/flu_28_major_protein_pairs_master.yaml` and its 28 child bundles enumerate all
  protein pairs. Their current population and training settings are not the settings in this plan,
  so new experiment bundles must override them explicitly.
- `src/analysis/aggregate_allpairs_results.py` already builds a 28-pair summary and heatmaps. It
  should be extended only where the current LightGBM/site-feature outputs require it.
- `src/analysis/summarize_cds_lengths.py` provides the per-protein completeness and length audit.
- `src/analysis/summarize_pair_capacity.py` provides positive-pair and Hopcroft-Karp counts. Its
  current common-cohort design is useful for controlled comparisons but is not the primary
  population definition for all 28 pairs.
- `src/analysis/plot_site_importance.py` already writes per-site gain, SHAP, and permutation
  importance by fold.

## Definitions and interpretation

### Population for a schema pair

Each schema pair will be built independently from Human H3N2 isolates in the specified year. An isolate is eligible for a pair when it has both required proteins as complete CDS records in the coordinate system used by that experiment. This holds host, subtype, and year fixed, but it does not force different schema pairs to retain the same isolates.

For each pair, report these counts in order:

1. eligible isolates;
2. unique observed positive pairs after `nt_cds` pair-key deduplication;
3. distinct slot-A and slot-B sequences;
4. positives retained by Hopcroft-Karp;
5. positives and negatives in each CV fold.

### Fixed-length and aligned site features

Fixed-length site features assign one feature column to each nucleotide, codon, or amino-acid (aa) position after retaining one CDS length. This is the current production method.

Aligned site features assign one feature column to each homologous alignment position. Coding sequences must be aligned in a way that preserves the reading frame. The proposed pilot translates each CDS, aligns the proteins, and projects protein gaps back to codon triplets. An unrestricted nucleotide alignment is not acceptable because it can introduce frame-breaking gaps.

An alignment gap, an unknown base, and unobserved sequence are different states:

- a gap represents an inferred biological insertion or deletion relative to other sequences;
- an unknown base is present in the record but unresolved;
- unobserved sequence is absent because the assembly or CDS is truncated.

These states must not be encoded as the same category. In particular, a truncated PB1 record must not be presented as evidence of a biological deletion.

### What the model predicts

The label records whether two segment sequences were observed together in one isolate. Strong performance does not by itself establish biochemical compatibility, coevolution, or reassortment fitness. Shared lineage, time, geography, and sampling structure can all contribute to the signal.

## Experiment 1: cross-year HA-NA importance

### Question

When the same model and population definition are applied to Human H3N2 HA-NA data from 2023, 2024, and 2025, do the fitted models use the same codon sites?

### Dataset construction

Build one HA-NA dataset per year using:

- complete CDS at the existing HA and NA pins;
- `pair_key_alphabet: nt_cds`;
- Hopcroft-Karp positive selection;
- 4-fold random CV;
- `negative_scope: within_fold`;
- a 1:1 negative-to-positive ratio;
- the same fold and sampling seeds in every year.

The current 2023 and 2025 HA-NA bundles do not include all these controls. Add dedicated bundles
rather than treating their older results as directly comparable.

Use the native Hopcroft-Karp population as the primary analysis. Also sample every year to the
smallest annual Hopcroft-Karp count as a sample-size sensitivity analysis. The fixed-count samples
must be deterministic and recorded in their manifests.

### Models and importance

Train per-site codon LightGBM models first. Codons give one column per residue coordinate, retain synonymous nucleotide information, and are narrower than per-nucleotide features. Run nucleotide 6-mers as a performance reference; k-mer importance is not part of the positional comparison.

For every fold:

1. read LightGBM gain from the fitted trees;
2. normalize gain to sum to 1 within the fold;
3. retain the full per-fold table;
4. average normalized gain across folds only for the descriptive annual map.

Gain is computed from the training process. It is not test-set importance. SHAP on held-out rows and permutation importance can be used as confirmation if the gain results are unstable or if the top sites drive a biological claim.

### Comparisons

Compare 2023 against 2024, 2024 against 2025, and 2023 against 2025. Report HA and NA separately:

- Spearman correlation across all sites;
- overlap and Jaccard similarity of the top 10 and top 25 sites;
- each protein's share of total gain;
- fold-to-fold variation within each year;
- native-count and fixed-count results side by side.

Also plot the three gain traces on shared coordinates. Label sites by 1-based residue number, as in the current importance outputs.

Correlated sites can substitute for each other in tree models. A low exact top-site overlap does not necessarily mean that the underlying sequence signal changed. If exact ranks differ, inspect whether importance moved among nearby or strongly correlated sites before interpreting the change.

### Required outputs

- one annual dataset audit per year;
- one per-fold importance CSV per year;
- one annual mean importance CSV per year;
- one cross-year comparison CSV;
- an overlaid gain trace for each protein;
- a top-site overlap plot or table;
- a short results note that separates measured results from interpretation.

### Acceptance checks

- HA and NA site counts and coordinates are identical across years.
- Each retained sequence is complete and at the configured pin.
- Positive pair keys are unique.
- No CDS hash occurs in more than one CV split within a fold.
- No generated negative is an observed positive from the full pre-selection positive universe.
- Every importance row can be traced to a model run, fold, protein, and site.

## Experiment 2: capacity audit for all 28 pairs

### Question

How much usable and sequence-unique Human H3N2 2024 data is available for each of the 28 pairs
formed from PB2, PB1, PA, HA, NP, NA, M1, and NS1?

### Population rules

Use the same metadata filters for every pair, but build each pair from its own eligible isolates.
Do not require a common eight-protein isolate cohort for the primary analysis. Requiring PB1 from
every isolate would remove about 45% of the population from pairs that do not contain PB1.

Use the existing pins for PB2, PA, HA, NP, NA, and M1. Add experiment-specific 2024 pins for NS1
(693 nt) and complete PB1 (2,277 nt), after rechecking them from the input data. PB1 pairs must be
marked as completeness-selected because their eligible population is much smaller.

For every pair, report the five population counts defined above. Also report the distribution of
per-sequence reuse before matching and the retained-isolate overlap for pairs that share a protein.

### Implementation

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
- a list of pairs that need aligned rather than fixed-length coordinates.

## Experiment 3: fixed-length 28-pairs screen

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

Amino-acid models are also deferred. Negatives built and blocked in nucleotide space can collapse
to the same amino-acid pair with conflicting labels. A fair amino-acid experiment must construct,
deduplicate, and block pairs in amino-acid space and must be reported as a different dataset
population.

### Comparisons

Report, for every pair and feature representation:

- AUC-ROC, F1 macro, precision, recall, and Brier score as fold mean and standard deviation;
- pooled confusion counts;
- eligible, unique-positive, and Hopcroft-Karp counts;
- the difference between codon and k-mer performance.

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

### Audit before alignment

Extend the length survey by protein and year for 2023-2025. For every non-modal length, separate:

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
partner and the pair has a direct polymerase interpretation. Keep the existing fixed-length
population as the control.

### Alignment method

Prototype the alignment outside the dataset builder first:

1. translate each complete CDS using the current translation rules;
2. align amino-acid sequences with a reproducible tool and version, with MAFFT as the first tool to
   evaluate;
3. project each amino-acid gap back to a three-nucleotide codon gap;
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

1. Build and audit the fixed-length 2023, 2024, and 2025 HA-NA datasets.
2. Run the codon cross-year importance comparison at native and fixed counts.
3. Produce the 2024 eight-protein, 28-pair capacity audit.
4. Build and audit the 28 fixed-length datasets.
5. Run the k-mer and codon 28-pairs screen and aggregate the results.
6. Complete the PB1/NS1 alignment feasibility audit and prototype.
7. Rerun only the pairs or years for which alignment materially improves eligibility.
8. Decide whether the evidence supports a publication scope, a narrower follow-up, or an archived
   negative/benchmark result.

Steps 1-3 produce useful results without waiting for alignment. Steps 4-5 can proceed for pairs
with validated fixed-length coordinates while the alignment pilot is being evaluated.

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

