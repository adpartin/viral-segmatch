# eda/

Exploratory analysis scripts. **Not part of the pipeline.** These were
used to understand raw data formats, validate methodology before
implementation, or reproduce specific results docs. Each script has a
self-contained header docstring explaining purpose, status, and inputs.

## Files

### Bunya exploration (Status: not maintained — Bunya is in CLAUDE.md's "What Is NOT Maintained" list)

| Script | Purpose |
|---|---|
| `bunya_gto_eda.py` | Inspects per-assembly outputs of the Bunya quality-annotation pipeline (`.contig_quality`, `.feature_quality`, `.qual.gto` triplet). |
| `bunya_pssm_eda.py` | Parses `Viral_PSSM.json` (BV-BRC's reference schema for segmented RNA virus families) into a flat per-segment / per-feature DataFrame. |

### DNA-level coverage feasibility (reproducibility hooks for the results docs that motivated the v2 DNA-level coverage fix, 2026-05-08)

| Script | Purpose |
|---|---|
| `dna_coverage_feasibility.py` | DNA-level coverage feasibility analysis on one specific dataset build. Companion to `docs/results/2026-05-08_dna_coverage_feasibility.md`. The hard-coded `DATASET_DIR` constant is stale; update before re-running. |
| `dna_coverage_feasibility_sweep.py` | Apriori cross-bundle feasibility sweep. Companion to `docs/results/2026-05-08_dna_coverage_feasibility_sweep.md`. The hard-coded `BUNDLES` list includes retired bundles; trim before re-running. |

## Note on `flu_genomes_eda.py`

This script lives in `src/preprocess/` (not here) because it generates
a required pipeline artifact:
`data/processed/flu/metadata_eda/flu_genomes_metadata_parsed.csv`
— parsed Flu A genome metadata used by
`src/utils/metadata_enrichment.py` for host / year / subtype /
geography filters. It must be run once before metadata-based filtering
is available. See `docs/methods/pipeline_overview.md` §4 for where
metadata enrichment fits in the Stage 3 flow.
