# Paper outline

Sections, figures and tables of the paper. Most of the methods and results are
described in the *paper plan* in `docs/plans/2026-10-08_paper_plan.md`.

Settings used throughout, unless a section says otherwise:

- **Narrow population**: `Human-H3N2-2024` (Task 1), 5,346 isolates.
- **Full population**: the Task 2 filter, host `[Human, Swine, Chicken, Duck]`, subtype
  `[H3N2, H1N1, H5N1]`, years 2010–2025; 60,751 isolates. It is not the whole corpus (108,530
  isolates).
- **Features**: per-site codons. ESM-2 and k-mers are not in the paper unless brought back.
- **Model**: LightGBM on the concatenated slot features `[x_A, x_B]`.
- **Negatives**: matched balanced-usage negatives (`matched_balanced_usage_negatives`), one per
  positive, in both populations.
- **Metric**: F1 macro.

## Methods

### Core concepts

- Flu anatomy. Schematic of a flu showing 8 segments and 8 major proteins.
- Model anatomy. Schematic of the prediction model: codon features for slot A
  and slot B, concatenated, LightGBM, threshold. See `2026-10-08_paper_plan.md`.

### Exploratory data analysis

The figures/tables are drafted for the full population (task 2 in the paper
plan). Counts are isolates (before any filtering on CDS completeness or
sequences deduplication).

- Isolates vs year (2010-2025)
- Heatmap (with numbers) subtype-vs-host

### Dataset generation flowchart

- Based on the flowchart in the paper plan, 1A "Dataset Construction (methods)".

### Sequence length and alignment analysis

- Sequence length distributions (prep for alignment analysis), starting with
  the full population (task 2 in the paper plan). We have different distributions
  across the proteins, so we need to determine how to show it.
- Per-protein alignment. Explain the reasons for alignment and the alignment
  process.

## Results

### Alignment results and decisions

- Equal CDS or protein length does not guarantee that residues at the same
  position correspond, because insertions and deletions can shift homologous
  residues. In HA and NA, the two largest gap patterns come from CDS of the
  same length, yet their gaps fall in different columns.
- Report alignment results.
  - How many gap patterns per protein?
  - How many unique proteins dropped.

### Model training and performance analysis for the narrow population (Human-H3N2-2024)

- 8x8 F1 macro matrix: 28 schema pairs, one triangle filled. Current figure:
  `docs/results/figs/2026-10-15_h3n2_2024_codon_matched_28pairs_f1_macro_matrix.png`.
- Table with performance scores (maybe supp materials)
- Sensitivity to negative resampling (1B). Current figure:
  `docs/results/figs/2026-10-15_h3n2_2024_codon_matched_28pairs_f1_macro_sd_draws_matrix.png`.

### Model training and performance analysis for the full population

Not run yet.

- 8x8 F1 macro matrix
- Table with performance scores (maybe supp materials)
- Performance breakdown by metadata group, to be discussed. A metadata group is one exact
  (host, HN subtype, year). The full population holds about 147 of them, many small.
  `metadata_group_report.py` scores each group and also each single host, subtype and year.

### [Optional] Feature importance

- Gain, SHAP and permutation importance (`plot_site_importance.py`)
- Shuffle and refit: shuffle the top-N gain-ranked sites in train, val and test, retrain, and
  report the share of above-chance AUC-ROC lost (`plot_site_shuffle_refit.py`).
