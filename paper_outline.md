# Paper outline

Sections, figures and tables of the paper. Most of the methods and results are
described in the [current paper plan](docs/plans/2026-11-24_paper_plan_v2.md).

Settings used throughout, unless a section says otherwise:

- **Task 2 population**: Human, Swine, Chicken and Duck, H3N2, H1N1 and H5N1, 2010–2025. The metadata filters select 61,635 isolates before eligibility is checked for each schema pair.
- **Task 1 population**: Human/H3N2/2024 pairs extracted from each Task 2 split.
- **Features**: aligned per-site codons.
- **Model**: LightGBM on the concatenated slot features `[x_A, x_B]`.
- **Negatives**: matched balanced-usage negatives (`matched_balanced_usage_negatives`), one per
  positive, in both populations.
- **Metric**: F1 macro.



## Methods

### Core concepts

- [Fig] Flu anatomy (simple schematic). The goal is to show that a viral isolate
  consists of 8 segments, and each segment has a major protein representing it.
- [Fig] Model anatomy. Schematic of the prediction model: codon features for slot A
  and slot B, concatenated, LightGBM, threshold.

Schematic from the [October methods-development record](docs/plans/2026-10-08_paper_plan.md#flu-a-data).

```text
        Slot A CDS                Slot B CDS
            |                         |
            v                         v
    generate features         generate features
            |                         |
            v                         v
           x_A                       x_B
            |                         |
            +------------+------------+
                         |
                         v
              concatenate: [x_A, x_B]
                         |
                         v
                LightGBM classifier
                         |
                         v
           probability of a positive pair
                         |
                         v
                     threshold
                         |
                         v
              predicted label: 1 or 0
```

### Dataset generation flowchart

- [Fig] Show the Task 2 dataset counts for PB1-HA from eligible isolates through positives kept. Show how the Human/H3N2/2024 pairs retain their split assignments in Task 1. [Dataset counts](docs/plans/2026-11-24_paper_plan_v2.md#dataset-construction-and-size)

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

<img src="docs/results/figs/2026-11-12_full_population_isolates_per_year.png" width="600" alt="Isolates per year, 2010-2025, full population">

<img src="docs/results/figs/2026-11-12_full_population_isolates_per_year_by_subtype.png" width="600" alt="Isolates per year, stacked by H/N subtype, full population">

<img src="docs/results/figs/2026-11-12_full_population_isolates_per_year_by_host.png" width="600" alt="Isolates per year, stacked by host, full population">

<img src="docs/results/figs/2026-11-12_full_population_host_subtype_heatmap.png" width="600" alt="Host x H/N subtype isolate counts, full population">

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

### Task 2. Performance across hosts, subtypes and years

- [Table] Dataset sizes and whole-population scores per schema pair. PB1-HA has five completed folds.
- [Fig] F1 macro matrix across schema pairs and a breakdown by metadata group with test counts.

### Task 1. Performance within Human/H3N2/2024

- [Table] Task 1A scores and paired differences on the same Human/H3N2/2024 test pairs in each fold.
- [Table or supplementary figure] Task 1B sensitivity to the training-negative draw within a fold.

### Predicting future years.

TBD.

### [Optional] Feature importance

- Gain, SHAP and permutation importance (`plot_site_importance.py`)
- Shuffle and refit: shuffle the top-N gain-ranked sites in train, val and test, retrain, and
  report the share of above-chance AUC-ROC lost (`plot_site_shuffle_refit.py`).



## Discussion

### Limitations

This is a POC study. For an actual tool, several things could be done differently

#### 1. 28 different models

#### 2. K-mers rather than aligned codons

#### 3. ...

