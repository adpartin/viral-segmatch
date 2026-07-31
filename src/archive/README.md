# src/archive — retired scripts

Scripts kept for provenance only. **Nothing here is imported by live code**, and nothing here is
maintained: paths, CLI defaults, and cluster sources are frozen as of the archive date and are
expected to rot. Read them as a record of how a result was produced, not as runnable tools.

A `docs/results/` write-up citing one of these is not a reason to restore it — those docs are the
historical record, and this directory is where the code they cite lives.

## Contents

Archived 2026-07-30, during the bigraph/fragmentation consolidation:

| Script | Why archived | Replacement |
|---|---|---|
| `bigraph_pair_metadata.py` | superseded plotting; carried an aa-only `pair_key_to_metadata` copy | `src/analysis/plot_cc_metadata.py`; helper generalized into `_pair_helpers.pair_key_to_metadata` |
| `bigraph_pair_feasibility.py` | superseded barplot; rode the protein-dedup pair universe (25% nt_cds undercount) and a hardcoded cluster root | `src/analysis/plot_cc_sizes.py` |
| `bigraph_reassort_check.py` | one-off aa-only validation of the cut vs reassortment signal | none — finding recorded in `docs/results/2026-06-04_bigraph_megacc_structure_and_cutting.md` |
| `bigraph_cut_subtype.py` | its `pair_key_to_subtype` was absorbed by the generalized helper | `_pair_helpers.pair_key_to_metadata(..., fields=('hn_subtype',))` |

All four read the analysis-side `load_pair_universe`, which by default dedups on `prot_hash` for
**every** alphabet — so their nt_cds numbers count 58,826 pairs where the true nt_cds universe for
HA-NA has 79,347 (measured on `cds_dna_final.parquet`). Treat any nt_cds figure they produced as
aa-deduped.

Imports here are repointed when a live symbol they depend on moves — enough to keep them readable
and importable, nothing more. As of 2026-07-31 all four import cleanly, but that is a courtesy, not
a guarantee: no test covers them and they are not run.
