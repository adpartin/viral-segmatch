# Paper outline

Sections, figures and tables of the paper. The work behind them, by task, is in
`docs/plans/2026-10-08_paper_plan.md`.

## Methods

### Core concepts

- Flu anatomy. Schematic of a flu showing 8 segments and 8 major proteins.
- Model anatomy. Schematic of the prediction model.

### Exploratory data analysis

- Isolates vs year (2010-2025)
- Heatmap (with numbers) subtype-vs-host

### Dataset generation flowchart

### Sequence length and alignment analysis

- Sequence length distributions (prep for alignment analysis).
- Per-protein alignment analysis.

## Results

### Alignment results and decisions

- Same length sequences does not mean positional correspondence. For example, in HA and NA, the
  two largest patterns come from CDS of the same length.

### Model training and performance analysis for a narrow population (H3N2-Human-2024)

- 8x8 F1 matrix
- Table with performance scores (maybe supp materials)
- Sensitivity to negative resampling.

### Model training and performance analysis for the full population

- 8x8 F1 matrix
- Table with performance scores (maybe supp materials)
- Performance breakdown by metadata groups.

### [Optional] Feature importance

- SHAP importance
- Permutation importance
- Performance drop when feature importance are shuffled
