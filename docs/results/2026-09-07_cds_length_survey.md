# Per-protein CDS completeness and length, corpus-wide and for human H3N2 2024

```yaml
# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-09-08
populations:    all; human H3N2 2024
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
python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --host Human --year 2024
```

The survey reports two different fractions and they can disagree sharply. `frac at mode` is the
share of distinct complete sequences at the modal length. `frac isolates at mode` is the share of
isolates whose CDS is complete and at that length. A dataset is built from isolates, so the isolate
column is the one that decides whether a protein can be pinned. The human H3N2 2024 section below
shows a case where the two differ by 0.44.

Every statistic counts each distinct CDS sequence once. The script currently deduplicates on
`cds_dna_hash` before grouping by protein; no hash occurs under more than one protein in this
corpus, so this gives the same result as deduplicating on `(function, cds_dna_hash)`. Counting rows
instead would let a heavily sampled strain decide the answer on its own, because one sequence
appears once per isolate carrying it. The `min`, `max`, `median` and `mode` columns describe
complete sequences only. A sequence is complete when `is_complete_cds` holds, which is
`starts_with_m & has_terminal_stop & ~has_internal_stop`.

## Results: whole corpus

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

## Results: human H3N2 2024

This is the population the current runs are built on. It holds 5,346 isolates. Every isolate has a
record for all 8 proteins, so nothing here is caused by a missing segment.

| Segment ID | protein | isolates | complete seqs | mode | frac at mode | frac isolates at mode |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PB2 | 5,346 | 2,821 | 2,280 | 0.997 | 0.998 |
| 2 | PB1 | 5,346 | 1,800 | 2,277 | 0.994 | **0.551** |
| 3 | PA | 5,346 | 2,721 | 2,151 | 0.998 | 0.998 |
| 4 | HA | 5,346 | 2,687 | 1,701 | 1.000 | 0.999 |
| 5 | NP | 5,346 | 1,834 | 1,497 | 1.000 | 0.996 |
| 6 | NA | 5,346 | 2,211 | 1,410 | 0.998 | 0.968 |
| 7 | M1 | 5,346 | 814 | 759 | 1.000 | 1.000 |
| 8 | NS1 | 5,346 | 1,132 | 693 | 0.991 | 0.996 |

Every pin in `conf/virus/flu.yaml` holds on this population, and `check_cds_length` passes for all
six pinned proteins. NS1 also reaches 0.996 by isolate here, so it can be pinned at 693 nt for this
population even though it carries no corpus-wide pin.

### PB1 cannot be pinned, and the sequence column hides it

PB1 reads 0.994 by sequence and 0.551 by isolate. Read on the sequence column alone it looks like
one of the better proteins in the set. It is the worst.

The cause is not length and not duplication. Of the 5,346 isolates, only 2,958 have a complete PB1.
Of the 2,388 incomplete records, 2,386 fail `has_terminal_stop` and 2,339 of those sit at 2,274 nt.
The usable length is 2,277 nt. The difference is exactly one codon, and that codon is the stop.

`has_terminal_stop` is set at Stage 1 from whether `prot_seq` ends in `*`, and Stage 1.5 cuts the
CDS at the coordinates the protein record supplies. So these records are a faithful copy of an
upstream annotation that excludes the stop codon. This is not a defect in the extraction code.

Whether the annotation is merely short is not yet known. The check is to take a 2,274 nt record and
read the next three bases from `ctg_dna_final`. If they are a stop codon, the CDS can be extended
and PB1 becomes pinnable at roughly 0.99, which would open 13 further schema pairs. If they are
not, the record is truncated and cannot be recovered. Until that is settled, PB1 stays excluded.

The same question applies to the comment in `conf/virus/flu.yaml`, which attributes the 2,274 to
2,277 shift to H3N2 gaining one codon between the 2023 and 2024 seasons. The share of isolates with
PB1 complete at 2,277 rises from 0.000 in 2010 to 0.619 in 2025, which is consistent with a change
in annotation practice rather than with viral evolution. The contig check settles this too.

### Pin stability by year

Share of each year's human H3N2 isolates with a complete CDS at the 2024 pin:

| year | isolates | PB2 | PA | HA | NP | NA | M1 | NS1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 1,335 | 0.998 | 0.999 | 1.000 | 0.999 | 1.000 | 0.999 | 0.993 |
| 2016 | 1,416 | 0.999 | 0.999 | 0.997 | 1.000 | 1.000 | 1.000 | 0.985 |
| 2017 | 2,845 | 1.000 | 0.998 | 1.000 | 0.999 | 0.998 | 1.000 | 0.958 |
| 2018 | 1,639 | 0.999 | 0.999 | 1.000 | 0.999 | 0.999 | 1.000 | 0.961 |
| 2019 | 2,755 | 1.000 | 0.998 | 0.999 | 0.926 | 0.996 | 1.000 | 0.723 |
| 2020 | 189 | 0.995 | 0.995 | 0.995 | 0.905 | 0.979 | 1.000 | 0.688 |
| 2021 | 1,089 | 0.999 | 0.981 | 0.999 | 1.000 | 0.999 | 0.999 | 0.981 |
| 2022 | 4,864 | 0.998 | 0.998 | 0.998 | 0.996 | 0.996 | 0.999 | 0.993 |
| 2023 | 1,347 | 0.997 | 0.999 | 0.999 | 0.998 | 0.936 | 1.000 | 0.996 |
| 2024 | 5,346 | 0.998 | 0.998 | 0.999 | 0.996 | 0.968 | 1.000 | 0.996 |
| 2025 | 3,434 | 0.999 | 0.998 | 1.000 | 0.999 | 0.999 | 1.000 | 0.999 |

The six pinned proteins hold above 0.90 in every year from 2010 to 2025. The weakest cells are NP
in 2019 and 2020 and NA in 2023. Adding NS1 restricts the range to 2021 to 2025, because NS1 falls
to 0.723 and 0.688 in 2019 and 2020. That range holds 16,080 isolates. 2020 holds only 189 isolates
and is worth dropping on size alone.

PB1 is omitted from this table because it has no pin. Its column would read 0.000 before 2021 and
0.619 at best in 2025.

## Reading the table as a screen

Screen on `frac isolates at mode`. PB1 is the reason: it reads 0.994 by sequence and 0.551 by
isolate, so the sequence column ranks it second-best when it is in fact the only protein in the set
that cannot be pinned.

`distinct lengths` screens nothing. PB2 and PB1 both show 23 distinct lengths corpus-wide but sit
at 0.998 and 0.927, and HA has fewer distinct lengths than PB2 while reaching only 0.689. A count
of lengths says nothing about whether one of them dominates.

On human H3N2 2024, seven of the eight proteins pass: PB2, PA, HA, NP, NA and M1 on their existing
pins, and NS1 on a population-specific pin at 693 nt. PB1 fails. Corpus-wide, PB2, PA, NP and M1
pass on their own, and HA and NA reach their pins only inside a subtype.

This is a screen, not a decision. A protein passing means a pin is possible. It says nothing about
whether a pair built from it has enough positives to train on. That is a pair-level question this
table cannot answer, and the counts differ a great deal: on human H3N2 2024 the fifteen pairs over
the six pinned proteins range from 2,293 to 3,723 positives, and from 616 to 1,987 after
unique-sequence matching.

## Metadata filtering

`summarize_cds_lengths` takes the three axes directly:

```bash
python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --host Human --year 2024
python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --year_range 2021 2025
```

`cds_dna_final` carries no subtype, host or year, so they are joined per isolate on `assembly_id`
through `src/utils/metadata_enrichment.py`. Filtering is at the isolate level, so an isolate
matching every stated criterion contributes all of its records. Each axis takes a value or a list;
`year_range` takes an inclusive `[min, max]` pair and applies to the year axis only. The
`population` column defaults to the filters that were applied, so stacked tables stay
distinguishable.

Filter to human hosts when comparing schema pairs. Human H3N2 2024 holds 5,346 human isolates
against 130 swine, 4 turkey and 2 duck. Swine H3N2 diverged from the human lineage long ago, so a
swine HA pairs with a swine NA in a way that is separable by lineage alone. That is a small
shortcut, and 2.5% of the data is cheap to give up.

## Limitations

The corpus-wide table mixes every subtype, host and year. A mode taken across subtypes is a
mixture rather than a fact about any one population, so it should not be used to choose a pin. Use
the human H3N2 2024 table for that.

The lengths describe stored CDS records, not an alignment. Equal length does not establish
positional homology or verify the annotation. `has_internal_stop` is 0 across the whole corpus, but
that does not rule out errors that preserve the reading frame.

Completeness is inherited from the source annotation, as the PB1 case shows. A protein can fail
this screen because of how its records were annotated rather than because of anything about the
virus, and this table cannot tell the two apart.

## Notes for whoever uses this next

Read the CSV with `keep_default_na=False`. Otherwise pandas reads the protein name `NA`
(Neuraminidase) as a missing value and drops the row.

`summarize_cds_lengths` deduplicates on `cds_dna_hash` alone, before grouping by protein. That is
safe only because no hash occurs under two proteins here, which was checked. Deduplicating on
`(function, cds_dna_hash)` would make it safe by construction.

Sequence statistics dedup on `cds_dna_hash`; isolate statistics do not, because deduplicating
collapses the isolates that share a sequence. Quoting one where the other is meant is the error
this document exists to prevent.
