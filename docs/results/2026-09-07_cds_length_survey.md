# Per-protein CDS completeness and length, corpus-wide and for Human-H3N2-2024

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

Per-site features use one column per sequence position and do not align or pad sequences. Each protein therefore needs a single CDS length shared by every record. A dominant length is necessary, but equal-length sequences are not necessarily aligned at homologous positions. This survey checks CDS completeness and length before schema-pair experiments.

## What was done

```bash
python -m src.analysis.summarize_cds_lengths
python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --host Human --year 2024
```

All counts refer to CDS DNA, keyed by `cds_dna_hash`. Sequence statistics count each unique CDS once; isolate statistics count every isolate carrying it. These can differ greatly: 5,346 human H3N2 isolates from 2024 carry only 815 unique M1 sequences.

## Results: whole corpus

All 108,530 isolates carry a record for every protein.

A CDS is complete if: `starts_with_m & has_terminal_stop & ~has_internal_stop`.

Columns:

- `unique CDS`: how many unique CDS sequences the protein has. Records carrying the same sequence count once.
- `complete CDS`: out of all the `unique CDS` sequences how many are complete.
- `min`, `max`, `median`: considering `complete CDS` sequences, determine the min, max and median CDS length, in nucleotides.
- `mode`: most common CDS length, over the `complete CDS` sequence set.
- `complete CDS at mode`: how many complete sequences are at the mode.
- `frac at mode`: share of complete sequences that are at the mode. `complete CDS at mode` / `complete CDS`.
- `frac isolates complete`: share of isolates whose record for this protein is complete. Isolates with a complete record / isolates carrying the protein.
- `frac isolates at mode`: share of isolates whose record is complete AND at the mode. This is the primary screening measure, because an isolate with an incomplete record stays in the denominator.

Clarifying `frac isolates complete` and `frac isolates at mode` with PB1 on the whole corpus. Both fractions share the same denominator — isolates, not sequences:

```
isolates carrying PB1:                     108,530 (denominator for both)
isolates whose PB1 is complete:            102,943
isolates complete AND at mode (2,274 nt):   95,768

frac isolates complete = 102,943 / 108,530 = 0.948521  ->  0.949
frac isolates at mode  =  95,768 / 108,530 = 0.882410  ->  0.882
```

The two fractions separate two different failures. 5,587 isolates have no complete PB1 at all, and
a further 7,175 have a complete PB1 at a length other than 2,274 nt. `frac at mode` sees only the
second of these, because its denominator is the complete sequences.

| Segment ID | protein | unique CDS | complete CDS | min | max | median | mode | complete CDS at mode | frac at mode | frac isolates complete | frac isolates at mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 67,341 | 66,356 | 2,199 | 2,283 | 2,280 | 2,280 | 66,210 | 0.998 | 0.990 | 0.989 |
| 2 | PB1 | 67,034 | 63,574 | 2,193 | 2,292 | 2,274 | 2,274 | 58,925 | 0.927 | 0.949 | 0.882 |
| 3 | PA | 65,242 | 64,670 | 2,073 | 2,163 | 2,151 | 2,151 | 64,576 | 0.999 | 0.994 | 0.993 |
| 4 | HA | 65,414 | 64,125 | 1,659 | 1,713 | 1,701 | 1,701 | 44,202 | 0.689 | 0.987 | 0.659 |
| 5 | NP | 52,800 | 51,749 | 1,446 | 1,500 | 1,497 | 1,497 | 51,681 | 0.999 | 0.988 | 0.987 |
| 6 | NA | 58,887 | 57,278 | 1,341 | 1,428 | 1,410 | 1,410 | 46,175 | 0.806 | 0.983 | 0.826 |
| 7 | M1 | 32,413 | 32,119 | 726 | 762 | 759 | 759 | 32,117 | 1.000 | 0.996 | 0.996 |
| 8 | NS1 | 38,039 | 37,843 | 609 | 717 | 693 | 693 | 21,576 | 0.570 | 0.998 | 0.600 |

The median equals the mode for all 8 proteins. No protein had a tie for the most common
length, so the tie-breaking rule in `modal_length` was never exercised on this population.

## Completeness, counted two ways

The corpus in cds_dna_final.parquet contains 868,240 rows (8 major proteins x 108,530 isolates = 868,240 records).

1) Out of 868,240 records, 855,695 are complete CDS based on bool column `is_complete_cds` in cds_dna_final.parquet (98.56%).
2) Out of 868,240 records, 447,170 are unique. Out of 447,170 unique, 437,714 are complete CDS (97.89%).

## What the results say about the pinned lengths

The corpus-wide mode values (i.e., regardless of the metadata) match all 6 length pins in `conf/virus/flu.yaml`: PB2 2,280, PA 2,151, HA 1,701, NP 1,497, NA 1,410, M1 759 nt. Note that variation in terms of mode length might be observed across different metadata (e.g., H3N2 and H1N1 together make up 61.5% of the corpus and can determine those modes).

PB1 and NS1 have no global pins because their dominant lengths change across the populations and years.

## Results: Human-H3N2-2024

This is the population used by the current experiments. It contains 5,346 isolates, each with a record for all 8 proteins.

Clarifying `frac isolates complete` and `frac isolates at mode` with PB1 on this population. Both fractions share the same denominator — isolates, not sequences:

```
isolates carrying PB1:                     5,346 (denominator for both)
isolates whose PB1 is complete:            2,958
isolates complete AND at mode (2,277 nt):  2,945

frac isolates complete = 2,958 / 5,346 = 0.553311  ->  0.553
frac isolates at mode  = 2,945 / 5,346 = 0.550879  ->  0.551
```

The two fractions are nearly equal because only 13 isolates have a complete PB1 at a length other
than 2,277 nt. Almost all of the loss is the 2,388 isolates with no complete PB1 at all, which is
a different failure from the one `frac at mode` reports.

| Segment ID | protein | isolates | unique CDS | complete CDS | mode | complete CDS at mode | frac at mode | frac isolates complete | frac isolates at mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 5,346 | 2,826 | 2,821 | 2,280 | 2,813 | 0.997 | 0.999 | 0.998 |
| 2 | PB1 | 5,346 | 3,061 | 1,800 | 2,277 | 1,790 | 0.994 | **0.553** | **0.551** |
| 3 | PA | 5,346 | 2,726 | 2,721 | 2,151 | 2,716 | 0.998 | 0.999 | 0.998 |
| 4 | HA | 5,346 | 2,694 | 2,687 | 1,701 | 2,687 | 1.000 | 0.999 | 0.999 |
| 5 | NP | 5,346 | 1,851 | 1,834 | 1,497 | 1,834 | 1.000 | 0.996 | 0.996 |
| 6 | NA | 5,346 | 2,320 | 2,211 | 1,410 | 2,206 | 0.998 | 0.969 | 0.968 |
| 7 | M1 | 5,346 | 815 | 814 | 759 | 814 | 1.000 | 1.000 | 1.000 |
| 8 | NS1 | 5,346 | 1,141 | 1,132 | 693 | 1,122 | 0.991 | 0.998 | 0.996 |

All 6 existing pins pass `check_cds_length` in this population. NS1 also reaches 0.996 by isolate,
so 693 nt is a suitable population-specific pin. The large gap between isolate and unique-CDS
counts, especially for M1, shows why both units are reported.

### Why PB1 is excluded

PB1 has `frac at mode = 0.994` among complete CDS, but only 55.3% of isolates have a complete PB1
and 55.1% have a complete CDS at the mode. The first fraction alone is therefore misleading.

Only 2,958 of 5,346 isolates have a complete PB1. Of the 2,388 incomplete records, 2,386 lack a
terminal stop and 2,339 are 2,274 nt. The mode among complete records is 2,277 nt, one codon
longer.

The 2,274-nt records appear to be truncated at the contig boundary rather than merely annotated
too short. Of 2,352 records at this length, 99.7% have no downstream contig sequence. Among the
2,339 incomplete records tested, none has a downstream stop codon; all but one has no downstream
bases to examine. By comparison, the 2,945 records at 2,277 nt have a median of 27 downstream
bases. The 2,274-nt records therefore cannot be extended from the current assemblies, so PB1
remains excluded.

In human H3N2 2024, 2,277 nt is the dominant length among complete PB1 records. About 44% of PB1
records are incomplete because their contigs end three bases before the expected terminal stop.
This assembly truncation complicates interpretation of the observed change from 2,274 to 2,277 nt
across years.

### Pin stability by year

Share of each year's Human H3N2 isolates with a complete CDS at the 2024 pin:

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

For the years shown, all six existing pins retain at least 90% of isolates. NP is lowest in 2019 and
2020, and NA is lowest in 2023. Adding NS1 limits a contiguous recent range to 2021-2025 because
NS1 falls below 90% in 2019 and 2020. PB1 is omitted because it has no stable pin.

## Using the survey as a screen

Screen proteins using `frac isolates at mode`, with `frac isolates complete` beside it to identify
whether failures come from incomplete records or length variation. In human H3N2 2024, the six
existing pins pass, NS1 passes with a population-specific 693-nt pin, and PB1 fails.

Passing this screen means that fixed-length site features are feasible. It does not guarantee
enough training pairs. `docs/results/2026-09-08_cds_pair_capacity.md` measures that per pair, on a
cohort built from the proteins this screen passes. Among the 15 pairs formed from the 6 pinned proteins, human H3N2 2024 has
2,293-3,723 positives before unique-sequence matching and 616-1,987 afterward.

## Metadata filtering

The script accepts subtype, host, and year filters:

```bash
python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --host Human --year 2024
python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --host Human --year_range 2021 2025
```

Because `cds_dna_final` lacks these fields, the script joins metadata by `assembly_id` through
`src/utils/metadata_enrichment.py`, then filters isolates. `year_range` is inclusive.

Human H3N2 2024 contains 5,346 human and 136 non-human isolates (130 swine, 4 turkey, and 2
duck). Restricting schema-pair comparisons to human isolates removes host-associated sequence
signal at a small cost in sample size.

## Limitations

- The corpus-wide table mixes subtypes, hosts, and years. Use a population-specific table to choose
  a pin.
- Equal CDS length does not prove positional homology or validate the annotation.
- Completeness comes from the source annotation. As PB1 shows, this survey cannot distinguish an
  annotation problem from biological length variation.

## Notes

Read the CSV with `keep_default_na=False`; otherwise pandas interprets the protein name `NA` as a
missing value.

The script deduplicates sequences using `cds_dna_hash`. No hash occurs under more than one protein
in this corpus, so this is currently safe. Deduplicating on `(function, cds_dna_hash)` would make
that assumption explicit.

Sequence statistics deduplicate by hash; isolate statistics do not. Report which denominator is
used.
