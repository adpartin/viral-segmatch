# eda/

Exploratory data analysis scripts. These are **not part of the pipeline** — they were used to understand raw data formats and structures during early development.

## Files

| Script | Purpose |
|--------|---------|
| `bunya_gto_eda.py` | Explores Bunyavirales GTO file structure (contig_quality, feature_quality fields) |
| `bunya_pssm_eda.py` | Explores Viral_PSSM.json reference schema for Bunyavirales segment/protein mapping |

## Note on `flu_genomes_eda.py`

This script lives in `src/preprocess/` (not here) because it generates a required pipeline artifact:
`flu_genomes_metadata_parsed.csv` — parsed Flu A genome metadata used by `src/utils/metadata_enrichment.py`
for host/year/subtype/geography filters. It must be run once before metadata filtering is available.
