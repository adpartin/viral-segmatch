# Per-protein CDS completeness and length, corpus-wide and for Human-H3N2-2024

```yaml
# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-09-08
populations:    all; Human-H3N2-2024
proteins:       PB2, PB1, PA, HA, NP, NA, M1, NS1
source:         data/processed/flu/July_2025/cds_dna_final.parquet
script:         src/analysis/summarize_cds_lengths.py
script_commit:  c58937d
artifact:       results/flu/July_2025/cds_length_survey/cds_length_survey.csv
depends_on:     [src/utils/cds_utils.py, src/utils/config_hydra.py]
```

The `script` and `artifact` produce only the whole-corpus and Human-H3N2-2024 per-protein tables. The year-specific tables and PB1 downstream-contig analysis were rechecked directly against the processed data on 2026-09-14, but their commands and outputs have not been preserved as artifacts. They therefore need separate provenance before publication.

## Problems and Solutions

Per-site features need every retained record for a protein to be complete and at one length. Three
problems prevent that. Each has a different solution.

1. **Incomplete CDS.** Part of the coding sequence (CDS) is absent, so the record fails the completeness
   check. This affects `PB1`: in recent Human-H3N2 years most of its incomplete records end at the
   contig boundary, before the expected terminal stop. There is no solution. Alignment cannot recover
   bases that were never sequenced, so these records must be excluded.

2. **Complete CDS at more than one length.** The records contain complete CDS but have different lengths,
   either from an insertion or deletion or because the length differs between populations. Pinning
   keeps one length and drops the rest. Corpus-wide this is largest for `NS1`, `HA` and `NA`, whose
   lengths differ by subtype. Restricting the population to specific Host-Subtype-Year removes
   nearly all of it. What remains is a year in which the length is changing, which is `PB1` in 2023.
   That is the one case a codon-preserving alignment can potentially recover, by putting homologous positions
   in shared columns and writing indels as gaps.

3. **Low sequence diversity.** Many isolates carry the same CDS, so there are far fewer distinct
   sequences than isolates. `M1` and `NS1` are the main examples. This drops no isolate, so it does not
   show up in the fractions below. It caps how many distinct pairs can be built, and is measured in
   `docs/results/2026-09-08_cds_pair_capacity.md`. Alignment does not help, because it cannot create
   diversity.

## Why this was measured

Per-site features use one column per sequence position and do not align or pad sequences. Every retained sequence for a protein must therefore be complete and at the same configured *pinned length*. A dominant length is necessary for this method. This survey checks completeness and length; **it is NOT an alignment validation**.

A *pin*, or *pinned length*, is a protein-specific CDS length recorded in the config rather than chosen for each run. The configured `cds_length` values in `conf/virus/flu.yaml` provide these lengths. When `dataset.require_complete_cds_at_pinned_length` is enabled, the pipeline retains only complete CDS records whose length equals the pin. It also checks that the pin is the modal length and holds for the required share of unique complete CDS in the population being processed. A pin also has a scope, the set of populations it was measured over, and it is valid only inside it. See *Pinned CDS length* in `docs/methods/glossary.md` for the canonical definition.

Using a configured pin gives runs the same feature dimensions and sequence coordinates. Otherwise, populations with different modal lengths could produce incompatible feature matrices. Positions downstream of an insertion or deletion could also refer to different biological sites. A pin does not itself prove that corresponding positions are homologous; that requires separate validation or alignment.

## What was done

```bash
python -m src.analysis.summarize_cds_lengths
python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --host Human --year 2024
```

All counts refer to CDS DNA, keyed by `cds_dna_hash`.
- Sequence statistics count each unique CDS once
- Isolate statistics count every isolate carrying it

The two statistics can differ greatly: 5,346 Human-H3N2-2024 isolates carry only 815 unique M1 sequences.

## Results: whole corpus

All 108,530 isolates carry a record for every protein.

A CDS is complete if: `starts_with_m & has_terminal_stop & ~has_internal_stop`, and it's recorded in a column `is_complete_cds`.

Table columns below:

- `unique CDS`: how many unique CDS sequences the protein has. Records carrying the same sequence count once.
- `complete CDS`: out of all the `unique CDS`, how many are complete.
- `min`, `max`, `median`: considering `complete CDS`, determine the min, max and median CDS length, in nucleotides.
- `mode`: most common CDS length, over the `complete CDS` set.
- `complete CDS at mode`: how many of the `complete CDS` are at the mode.
- `frac at mode`: share of `complete CDS` that are at the mode. `complete CDS at mode` / `complete CDS`.
- `frac isolates complete`: share of isolates (the 108,530) whose record for this protein is complete. Isolates with a complete record / isolates carrying the protein.
- `frac isolates at mode`: share of isolates (the 108,530) whose record is complete and at the mode. It measures
  how much of the isolate population a pin at the modal length would retain.

Clarifying `frac isolates complete` and `frac isolates at mode` columns with PB1 on the whole corpus. Both fractions share the same denominator — isolates, not sequences:

```
isolates carrying PB1:                     108,530 (denominator for both)
isolates whose PB1 is complete:            102,943 (is_complete_cds==True)
isolates complete AND at mode (2,274 nt):   95,768

frac isolates complete = 102,943 / 108,530 = 0.948521  ->  0.949
frac isolates at mode  =  95,768 / 108,530 = 0.882410  ->  0.882
```

The two isolate fractions separate two sources of data loss. Of the 108,530 isolates, 5,587 have no
complete PB1, and a further 7,175 have a complete PB1 at a length other than 2,274 nt. `frac at
mode` is not another view of these isolate counts: it excludes incomplete CDS and counts each
unique complete CDS once. It measures how concentrated the complete sequence set is at one length.

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

The corpus-wide modes match the 6 lengths currently configured in `conf/virus/flu.yaml`: PB2
2,280, PA 2,151, HA 1,701, NP 1,497, NA 1,410, and M1 759 nt. These corpus-wide modes are
descriptive, not universal pins: the corpus mixes subtypes, hosts, and years, and some proteins have
different dominant lengths in narrower populations.

No corpus-wide pin is configured for PB1 or NS1, but the two are absent for different reasons.

Within Human-H3N2, the modal complete-CDS length of NS1 is 693 nt in every year from 2015 through
2025. In a broader H3N2 data, 3,120 of the 3,276 isolates with a complete 660-nt NS1 are labelled
`Pig`. The available data therefore do not support a year-specific NS1 mode change within
Human-H3N2; the absence of a global pin reflects population mixing rather than instability within
this filter.

Among the 8 proteins surveyed, PB1 is the only one whose modal complete-CDS length changes
within Human-H3N2 from 2015 through 2025: 2,274 nt through 2023 and 2,277 nt in 2024 and 2025.
These are properties of the records in this corpus, so neither value should be treated as a
universal pin length.

## Results: Human-H3N2-2024

This is the population used by the current experiments. It contains 5,346 isolates, each with a record for all 8 proteins.

Clarifying `frac isolates complete` and `frac isolates at mode` with PB1 on this population. Both fractions share the same denominator — isolates, not sequences:

```
isolates carrying PB1:                     5,346 (denominator for both)
isolates whose PB1 is complete:            2,958 (is_complete_cds==True)
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

At the default 90% unique-sequence floor, all 6 configured pins pass `check_cds_length` in this
population. NS1 at 693 nt contains 99.1% of unique complete CDS and 99.6% of isolates, so it also
passes as a Human-H3N2-2024-specific pin. This does not make 693 nt a corpus-wide NS1 pin. The
large gap between isolate and unique-CDS counts, especially for M1, shows why both units are
reported.

### Why PB1 requires a population-specific pin

PB1 has `frac at mode = 0.994` among unique complete CDS, but only 55.3% of isolates have a
complete PB1 and 55.1% have a complete CDS at the mode. The sequence-level fraction establishes a
clear modal length among complete sequences; it does not describe isolate retention.

Only 2,958 of 5,346 isolates have a complete PB1. Of the 2,388 incomplete records, 2,386 lack a
terminal stop and 2,339 are 2,274 nt. The mode among unique complete CDS is 2,277 nt, one codon
longer.

The 2,274-nt records appear to be truncated at the contig boundary rather than merely annotated
too short. Of 2,352 records at this length, 99.7% have no downstream contig sequence. Among the
2,339 incomplete records tested, none has a downstream stop codon; all but one has no downstream
bases to examine. By comparison, the 2,945 records at 2,277 nt have a median of 27 downstream
bases. The 2,274-nt records therefore cannot be extended from the current assemblies.

In Human-H3N2-2024, 2,277 nt is the dominant length among complete PB1 records. About 44% are
incomplete 2,274-nt records whose contigs end at the CDS boundary and lack a terminal stop. The
mixture of these incomplete records and complete length variants complicates interpretation of the
apparent change across years. The measurement shows where the available sequence stops; it does not
establish why the sequence is absent or measure the biological prevalence of the two forms.

### Pin stability by year (additional measurement)

For seven proteins, the modal complete-CDS length is unchanged in every Human-H3N2 year from 2015
through 2025: PB2 2,280, PA 2,151, HA 1,701, NP 1,497, NA 1,410, M1 759, and NS1 693 nt. The
table reports the share of each year's isolates with a complete CDS at that length:

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

For the years shown, the 6 configured pins retain at least 90% of isolates. At the same threshold,
NS1 passes in 2015-2018 and 2021-2025 but not in 2019-2020. This illustrates that a stable modal
length does not guarantee uniformly high isolate retention.

PB1 requires a separate view because both its modal length and its isolate retention change:

| year | isolates | complete at 2,274 nt | complete at 2,277 nt | complete at either length |
|---:|---:|---:|---:|---:|
| 2021 | 1,089 | 0.956 | 0.042 | 0.998 |
| 2022 | 4,864 | 0.905 | 0.067 | 0.972 |
| 2023 | 1,347 | 0.432 | 0.391 | 0.823 |
| 2024 | 5,346 | 0.002 | 0.551 | 0.553 |
| 2025 | 3,434 | 0.003 | 0.619 | 0.622 |

No single PB1 length retains at least 90% of Human-H3N2 isolates across 2023-2025. For a
Human-H3N2-2024 analysis, 2,277 nt is nevertheless a clear population-specific pin among complete
PB1 records. Analyses that use it retain about 55% of isolates and should describe the resulting
PB1 population as completeness-selected. The 2025 values come from the partial 2025 season in the
July 2025 corpus.

## Using the survey as a screen

Use the sequence- and isolate-level measures together. `frac at mode` checks whether one length
dominates among unique complete CDS, as required by the production pin guard. `frac isolates at
mode`, with `frac isolates complete` beside it, measures how much of the isolate population would
remain and whether losses come from incomplete records or other lengths.

In Human-H3N2-2024, the 6 configured pins pass both checks, and NS1 passes with a
population-specific 693-nt pin. PB1 at 2,277 nt passes the unique-sequence concentration check but
retains only 55.1% of isolates. It therefore does not pass a 90% isolate-retention screen, but it
can still be used as a smaller, explicitly completeness-selected population. This distinction
allows PB1-containing pairs to remain in the 28-pairs capacity audit without implying that PB1 has
the same coverage as the other proteins.

Passing this screen means that per-site features on a single shared CDS length are feasible. It does
not guarantee enough training pairs. The companion capacity audit
(`docs/results/2026-09-08_cds_pair_capacity.md`) measures this per pair, including the smaller
completeness-selected PB1 pairs.

## Metadata filtering

The script accepts subtype, host, and year filters:

```bash
python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --host Human --year 2024
python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --host Human --year_range 2021 2025
```

Because `cds_dna_final` lacks these fields, the script joins metadata by `assembly_id` through
`src/utils/metadata_enrichment.py`, then filters isolates. `year_range` is inclusive.

H3N2-2024 contains 5,346 human and 136 non-human isolates (130 swine, 4 turkey, and 2 duck).
Restricting the experiment to Human-H3N2-2024 holds the recorded host category fixed and prevents a
classifier from directly separating human from non-human records. It does not remove lineage,
geography, passage, or other structure within the human population.

## Limitations

- The corpus-wide table mixes subtypes, hosts, and years. Use a population-specific table to choose
  a pin.
- Equal CDS length does not prove positional homology or validate the annotation.
- `is_complete_cds` is an operational check derived from the annotated protein: it requires an
  initial methionine, a terminal stop, and no internal stop. It is not independent evidence that the
  assembly contains every biologically expected base.
- The PB1 contig-boundary check shows that the current assemblies provide no downstream bases for
  most incomplete 2,274-nt records. It does not establish why those bases are absent.
- The year-specific and PB1 contig-boundary measurements do not yet have saved scripts and output
  artifacts; see the provenance note above.

## Notes

Read the CSV with `keep_default_na=False`; otherwise pandas interprets the protein name `NA` as a
missing value.

The script deduplicates sequences using `cds_dna_hash`. No hash occurs under more than one protein
in this corpus, so this is currently safe. Deduplicating on `(function, cds_dna_hash)` would make
that assumption explicit.

Sequence statistics deduplicate by hash; isolate statistics do not. Report which denominator is
used.
