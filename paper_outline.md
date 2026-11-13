# Paper outline

Sections, figures and tables of the paper. Most of the methods and results are
described in the *paper plan* in `docs/plans/2026-10-08_paper_plan.md`.

Settings used throughout, unless a section says otherwise:

- **Narrow population**: `Human-H3N2-2024` (Task 1), 5,346 isolates.
- **Full population**: the Task 2 filter, host `[Human, Swine, Chicken, Duck]`, subtype
  `[H3N2, H1N1, H5N1]`, years 2010–2025; 60,751 isolates.
- **Features**: per-site codons.
- **Model**: LightGBM on the concatenated slot features `[x_A, x_B]`.
- **Negatives**: matched balanced-usage negatives (`matched_balanced_usage_negatives`), one per
  positive, in both populations.
- **Metric**: F1 macro.



## Methods

### Core concepts

- [Fig] Flu anatomy (simple schematic). The goal is to show that a viral isolate
  consists of 8 segments, and each segment has a major protein representing it.
- [Fig] Model anatomy. Schematic of the prediction model: codon features for slot A
  and slot B, concatenated, LightGBM, threshold. See `2026-10-08_paper_plan.md`.

### Dataset generation flowchart

- [Fig] Dataset-generation flowchart. Shows the step-by-step process going from
  full dataset down to 5-fold CV splits with positive and negative pairs. The
  flowchart is shown in the paper plan 1A "Dataset Construction (methods)". Each
  step shows the counts representative of a specific schema pair.

### Protein sequence alignment and per-site codon featurization 

- [Fig] Flowchart. Keep the CDS lengths above 2% of unique CDS -> align the
  kept proteins -> drop the CDS of gap patterns at or below 2% -> if anything
  dropped, realign the remaining proteins -> each protein's final alignment is
  the input to codon featurization. The steps are the retention rule in
  `docs/results/2026-11-09_protein_alignment.md`.



## Results

### Exploratory data analysis

The figures and tables are drafted for the full population (task 2 in the paper
plan). Counts are isolates (before any filtering on CDS completeness or
sequence deduplication).

- [Fig] Isolates-vs-year barplot (2010-2025)
- [Fig] Subtype-vs-host heatmap (with numbers)

### Sequence length analysis

- [Table] Stats of sequence counts and lengths (EDA for alignment analysis)
  for the full population (task 2 in the paper plan).

| Segment ID | Protein | Unique CDS | Complete CDS | Min length | Max length | Mode length | CDS lengths | Complete CDS at mode | Fraction @ mode | 
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 67,341 | 66,356 | 2,199 | 2,283 | 2,280 | 18 | 66,210 | 0.998 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 8 | NS1 | 38,039 | 37,843 | 609 | 717 | 693 | 16 | 21,576 | 0.570 |

### Alignment results and decisions

- Equal CDS or protein length does not guarantee that residues at the same
  position correspond (insertions and deletions can shift homologous residues).
  In HA and NA, the two largest gap patterns come from CDS of the same length,
  yet their gaps fall in different columns.
- [Table] Report alignment results and decisions.
  - Stats of unique lengths (see the tables in `2026-11-09_protein_alignment.md`)
  - Dominant lengths per protein
  - Gap patterns per protein
- [Fig] Interesting alignment patterns from Jalview.
- [Text] Report the decisions about what sequences to drop, and final alignment.

PB2 (segment 1).

| Source CDS length | Isolates | Unique CDS | Unique proteins | % of unique proteins | Leading AA-gap columns | Internal AA-gap columns | Trailing AA-gap columns |
|---|---:|---:|---:|---:|---|---|---|
| 2280 nt | 60,413 | 36,879 | 15,230 | 99.61% | None | 7 | None |
| 17 other lengths | 61 | 60 | 60 | 0.39% | 0 to 10 | 6 to 31 | 0 to 10 |

The 759 aa sequences have 2 observed gap patterns:

| Source CDS length | AA gap columns | AA residues | Unique proteins |
|---|---|---:|---:|
| 2280 nt | 42–45, 348, 575, 678 | 759 | 15,229 |
| 2280 nt | 42–45, 348, 582, 678 | 759 | 1 |

### Model performance results for the narrow population (Human-H3N2-2024)

- [Table] Per schema pair performance scores.
- [Fig] The F1 macro matrix: all 28 schema pairs (one triangle filled). E.g.,
  `docs/results/figs/2026-10-15_h3n2_2024_codon_matched_28pairs_f1_macro_matrix.png`.
- [Table] Sensitivity to negative resampling (1B). Maybe supp materials.
- [Fig] Sensitivity to negative resampling (1B). Per-class scatter plots with
  median. E.g.,
  `docs/results/figs/2026-10-15_h3n2_2024_codon_matched_28pairs_f1_macro_sd_draws_matrix.png`.

### Model performance results for the full population

Not run yet.

- [Table] Per schema pair performance scores.
- [Fig] The F1 macro matrix: all 28 schema pairs (one triangle filled).
- [Fig] Performance breakdown by metadata group. A metadata group is one exact
  (host, HN subtype, year). The full population holds about 147 of them. Thus,
  we may show performance by host-subtype while combining all years.
  `metadata_group_report.py` scores each group and also each single host,
  subtype and year.

### Predicting future years.

TBD.

### [Optional] Feature importance

- Gain, SHAP and permutation importance (`plot_site_importance.py`)
- Shuffle and refit: shuffle the top-N gain-ranked sites in train, val and test, retrain, and
  report the share of above-chance AUC-ROC lost (`plot_site_shuffle_refit.py`).
