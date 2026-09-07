# Per-protein CDS completeness and length across the Flu A corpus

```yaml
# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-09-07
population:     all           # every subtype, host and year in the corpus
proteins:       PB2, PB1, PA, HA, NP, NA, M1, NS1
source:         data/processed/flu/July_2025/cds_dna_final.parquet
script:         src/analysis/summarize_cds_lengths.py
script_commit:  c58937d
artifact:       results/flu/July_2025/cds_length_survey/cds_length_survey.csv
depends_on:     [src/utils/cds_utils.py, src/utils/config_hydra.py]
```

## Why this was measured

The current per-site implementation does not align or pad sequences. It therefore retains one
pinned CDS length per protein so that a column has the same index in every sequence. A dominant
length is a necessary screen for this implementation, but it does not establish that equal-index
positions are homologous. This survey reports the length distributions used to screen the 28
schema pairs before a run is attempted.

## What was done

```bash
python -m src.analysis.summarize_cds_lengths
```

Every statistic counts each distinct CDS sequence once. The script currently deduplicates on
`cds_dna_hash` before grouping by protein; no hash occurs under more than one protein in this
corpus, so this gives the same result as deduplicating on `(function, cds_dna_hash)`. Counting rows
instead would let a heavily sampled strain decide the answer on its own, because one sequence
appears once per isolate carrying it. The `min`, `max`, `median` and `mode` columns describe
complete sequences only. A sequence is complete when `is_complete_cds` holds, which is
`starts_with_m & has_terminal_stop & ~has_internal_stop`.

## Results

| Segment ID | protein | unique seqs | complete seqs | min | max | median | mode | seqs at mode | frac at mode | distinct lengths |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 67,341 | 66,356 | 2,199 | 2,283 | 2,280 | 2,280 | 66,210 | 0.998 | 23 |
| 2 | PB1 | 67,034 | 63,574 | 2,193 | 2,292 | 2,274 | 2,274 | 58,925 | 0.927 | 23 |
| 3 | PA | 65,242 | 64,670 | 2,073 | 2,163 | 2,151 | 2,151 | 64,576 | 0.999 | 18 |
| 4 | HA | 65,414 | 64,125 | 1,659 | 1,713 | 1,701 | 1,701 | 44,202 | 0.689 | 17 |
| 5 | NP | 52,800 | 51,749 | 1,446 | 1,500 | 1,497 | 1,497 | 51,681 | 0.999 | 7 |
| 6 | NA | 58,887 | 57,278 | 1,341 | 1,428 | 1,410 | 1,410 | 46,175 | 0.806 | 29 |
| 7 | M1 | 32,413 | 32,119 | 726 | 762 | 759 | 759 | 32,117 | 1.000 | 3 |
| 8 | NS1 | 38,039 | 37,843 | 609 | 717 | 693 | 693 | 21,576 | 0.570 | 25 |

The median equals the mode for all eight proteins. No protein had a tie for the most common
length, so the tie-breaking rule in `modal_length` was never exercised on this population.

## Completeness, counted two ways

`cds_dna_final` holds exactly the 8 modelled proteins at 108,530 rows each, which is 868,240 rows
in total. Completeness has a different value depending on whether rows or distinct sequences are
counted, and the two numbers are not interchangeable.

At the row level, 855,695 of 868,240 rows are complete, which is 98.56%. This is the figure
already recorded in `docs/plans/2026-08-28_per_site_nt_features_plan.md` step 0, alongside
`starts_with_m` at 864,444, `has_terminal_stop` at 858,776 and `has_internal_stop` at 0.

At the distinct-sequence level, 437,714 of 447,170 unique sequences are complete, which is 97.89%.
This is the denominator the table above uses. The row-level figure is higher because a sequence
carried by many isolates is counted many times, and frequently observed sequences are more often
complete.

## What this says about the pinned lengths, and what it does not

The corpus-wide mode agrees with the pin in `conf/virus/flu.yaml` for all six pinned proteins:
PB2 2,280, PA 2,151, HA 1,701, NP 1,497, NA 1,410 and M1 759. That agreement is worth recording,
because the selected lengths are also the most common lengths in the broader corpus. This is not
independent confirmation. H3N2 and H1N1 are 36,949 and 29,833 of the 108,530 isolates in
`cds_dna_final`, or 61.5% together, so they can set the corpus mode on their own.

The share at that mode is a different matter. The `seq_frac` values in `conf/virus/flu.yaml` run
from 0.977 to 1.000, and they were measured on H3N2 plus H1N1 only. Corpus-wide, HA reaches 0.689
and NA reaches 0.806. The gap is expected, because the corpus spans subtypes with genuinely
different lengths. H5N1 HA is 1,704 nt, H9 and H7 HA are 1,683 nt, and N8, N6 and N9 NA are
1,413 nt. This survey therefore supports the selected lengths as corpus-wide modes but cannot
confirm the population-specific `seq_frac` floors.

The same limit applies to the two proteins that carry no pin. PB1 sits at 0.927 corpus-wide, which
would pass a 0.90 floor, so its exclusion is not explained by any corpus-wide share. The reason
recorded in `conf/virus/flu.yaml` is that H3N2 switched from 2,274 to 2,277 nt between the 2023 and
2024 seasons, with 2023 mid-turnover at 68%, so no single value is right for every year. NS1 sits
at 0.570 and does fail a 0.90 floor corpus-wide, but its recorded reason is also population-specific:
H1N1 is 660 nt, H5N1 is 693 nt, and H3N2 moved from 660 to 693 around 2021. Neither exclusion can be
derived from this table. Both need a population-specific breakdown; the required restrictions
depend on the population being studied.

## Reading the table as a screen

`frac at mode` is the main screening statistic; `distinct lengths` supplies context but does not
show whether one length dominates. PB2, PA, NP and M1 sit at 0.998 or above corpus-wide, so pairs
drawn from those proteins are good candidates for the current pinned-length pipeline. They still
need checks for positional consistency, pair count and the number of positives retained by any
uniqueness constraint. HA and NA need a narrower population, as in the existing HA-NA H3N2 2024
runs. PB1 and NS1 need population-specific length analysis before selecting a pin.

This is a screen, not a decision. A high `frac at mode` says a pin is possible for that protein. It
says nothing about whether the resulting pair has enough positives to train on, which is a separate
question answered by the pair universe rather than by this table.

## Extension to metadata: not yet run

The breakdown by subtype, year and host is the next step, starting with H3N2 2024. It needs no
change to `summarize_cds_lengths`, which already takes an already-filtered frame and carries a
`population` label into its output so that several tables can be stacked and still be told apart.

`cds_dna_final` has no subtype, year or host column. Its 15 columns are `assembly_id`,
`genbank_ctg_id`, `brc_fea_id`, `function`, `canonical_segment`, `prot_hash`, `prot_seq`,
`cds_dna_seq`, `cds_dna_hash`, `length`, `cds_length`, `starts_with_m`, `has_terminal_stop`,
`has_internal_stop` and `is_complete_cds`. The three axes come from an isolate-level join on
`assembly_id`, which `src/utils/metadata_enrichment.py` already provides:

```python
from src.utils.metadata_enrichment import attach_isolate_metadata, filter_by_metadata

enriched = attach_isolate_metadata(cds)                                   # adds hn_subtype, host, year
h3n2_2024 = filter_by_metadata(enriched, hn_subtype='H3N2', year=2024)
table = summarize_cds_lengths(h3n2_2024, function_to_short, population='H3N2 2024')
```

`filter_by_metadata` filters at the isolate level, so an isolate matching every stated criterion
contributes all of its records. It accepts a scalar for an exact match or a list for set
membership on each axis, and `year_range` accepts an inclusive `[min, max]` pair on the year axis
only.

Two changes to the CLI are needed before this can be run from the command line. The argument
parser currently exposes only `--cds_final`, `--config_bundle`, `--population` and `--out_dir`, so
it needs metadata arguments. The parquet read also selects five columns explicitly and does not
include `assembly_id`, so it cannot currently be joined against the metadata table.

## Limitations

Every number here is corpus-wide, covering every subtype, host and year at once. A mode taken
across subtypes is a mixture rather than a fact about any one population, so nothing in this
document alone should be used to choose a pin. The lengths describe stored CDS records, not an
alignment. Equal length therefore does not establish positional homology or verify the annotation.
`has_internal_stop` is 0 across the whole corpus, but that does not rule out errors that preserve
the reading frame.

## Notes for whoever uses this next

Read the CSV with `keep_default_na=False`. Otherwise pandas reads the protein name `NA`
(Neuraminidase) as a missing value and drops the row.

`summarize_cds_lengths` deduplicates on `cds_dna_hash` alone, before grouping by protein. That is
safe only because no hash occurs under two proteins here, which was checked. Deduplicating on
`(function, cds_dna_hash)` would make it safe by construction.
