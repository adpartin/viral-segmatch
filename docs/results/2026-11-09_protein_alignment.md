# Protein alignment for codon features

```yaml
# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-11-09
populations:    Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025; Human-H3N2-2024
proteins:       NS1
source:         data/processed/flu/July_2025/cds_dna_final.parquet
scripts:        src/analysis/align_proteins.py; src/analysis/summarize_protein_alignment.py; src/analysis/audit_ns1_downstream_contig.py
artifacts:      results/flu/July_2025/cds_alignment_pilot/<population>/<protein>/
```

## Why this is measured

Per-site codon features need the same sites in every record, so that position
200 represents the same place in each record. For each protein, there are two
options to obtain those sites:

1. **Pinned CDS length.** Use the fixed length for the protein from
   `conf/virus/flu.yaml`, then keep only complete CDS at that length.
   `check_cds_length` accepts the pin only when it equals the observed mode and
   at least 90% of unique complete CDS have the pinned length. The mode checks
   the configured pin; it does not set it.
2. **Protein alignment.** Align the translated proteins across all observed
   complete CDS lengths. The alignment columns estimate which sites correspond
   across records of different lengths. Under the proposed codon mapping, an
   alignment column with no codon in a record becomes `GAP` or `AFTER_STOP`,
   rather than causing the record to be dropped.

If a protein has no usable pin, the alternatives are to align it or to drop it.
NS1 has no configured pin because its CDS length varies across subtypes and
years, so this analysis measures the alignment route.


## The population

Every protein section below is measured on one population, the one Task 2 uses,
set in `docs/plans/2026-10-08_paper_plan.md`. Filters: host `[Human, Swine,
Chicken, Duck]`, subtype `[H3N2, H1N1, H5N1]`, year `2010–2025`. `host` is a
host group, so `Duck` covers 54 animal names and `Chicken` covers 3.

The one section on a different population names it in the title: NS1 on
Human-H3N2-2024, the Task 1 population where the method was first run.


## How to read these sections

Each protein has a section with 6 numbered parts: 1. Inputs, 2. Lengths and
gaps, 3. Stop codon columns, 4. By host, subtype and year, 5. Findings, 6.
Artifacts. If a part has nothing to report, it says so in one sentence rather
than being left out. A section may add one unnumbered part for a check
specific to it.

Gap terms, as defined in [the
glossary](../methods/glossary.md#per-site-features): a **trailing gap** is a `-`
after the last protein symbol, an **internal gap** is a `-` between protein
symbols, a **leading gap** is a `-` before the first, and a protein symbol is an
amino acid or an `X`. A **gap pattern** is one protein's full set of gap columns.
In the schematics, `res` stands for one protein symbol.

This alignment does not:

* Place CDS stop codons. The terminal `*` marker is removed before MAFFT, so no
  column holds a stop codon. Where a section gives stop codon columns, that is a
  proposal read off the gap pattern, not a measurement.
* Show that a column represents the same biological position in every record.
  The alignment is an estimate of positional correspondence.
* Recover bases absent from a truncated record. These runs use complete CDS only.


## Method

`src/analysis/align_proteins.py` does all 6 steps, over 4 runs. The plain run
does steps 1 to 3; `--align_only`, `--validate_only` and `--inspect_only` do
steps 4, 5 and 6.

1. Select complete CDS for the protein and population, keeping all observed
   lengths.
2. Check every selected unique CDS: it translates to its recorded `prot_seq`,
   ends with a `*` and has no other `*`, its length matches both the stored
   `cds_length` and 3 times the protein length including the `*`, and
   `prot_hash` is the md5 of `prot_seq`. One CDS hash must not carry conflicting
   records, and one protein hash must not name two sequences.
3. Remove the terminal `*` from the protein sent to MAFFT, leaving the CDS
   unchanged. Keep `X` symbols and count them by CDS length.
4. Align the unique proteins with MAFFT `--retree 2 --maxiterate 0`, recording
   the version, command, strategy and input hashes.
5. Check that removing `-` from each aligned protein recovers the exact input
   protein.
6. Record every gap position.

Every run used MAFFT v7.526, so the sections below give only the strategy and
the column count.

`src/analysis/summarize_protein_alignment.py` then writes
`group_length_summary.csv`, `gap_pattern_summary.csv` and
`column_occupancy.csv`.

The saved protein is the exact MAFFT input, which is what step 5 compares
against. With the options used here, [MAFFT would remove `*`
anyway](https://mafft.cbrc.jp/alignment/software/anysymbol.html).

An `X` means the codon does not translate to a single amino acid. It does not
remove the codon and does not create a gap. It does give the aligner less
information at that position. Whether any gap placement falls next to an `X` has
not been checked, except in the Human-H3N2-2024 section where it is reported.


## Summary across proteins

One row per protein, on the Task 2 population.

| Protein | Isolates | Unique CDS | Unique proteins | CDS lengths | Columns | `frac at mode` | Largest internal gap | One-residue columns |
|---|---:|---:|---:|---:|---:|---:|---|---|
| NS1 | 60,654 | 18,107 | 10,130 | 16 | 240 | 693 nt, 0.502 | columns 81–85, 5 codons, 206 proteins | 74, 112, 132 |

"Largest internal gap" gives the longest run of neighbouring internal gap columns
any protein has, as column numbers rather than a count of columns, followed by how
long the run is and how many proteins carry it. `columns 81–85, 5 codons` means
five gaps, one in each of the columns numbered 81 to 85. Each section lists all of
its internal gaps; the table gives only the longest.

A **one-residue column** is an alignment column where exactly one protein has a
protein symbol and every other protein has a gap.

NS1 has no pin on the Task 2 population. It has a usable pin at 0.991 on the
Task 1 population, reported in the last section. Widening the population removes
the pin.


## NS1

### 1. Inputs

Of 60,751 filtered isolates, 60,654 (99.8%) had a complete CDS. They contained
18,107 unique CDS and 10,130 unique proteins across 16 CDS lengths and 147
host–subtype–year groups. 957 isolates and 551 unique proteins contain an `X`.

No single pinned length is available. `frac at mode` is 0.502, at 693 nt, then
0.455 at 660 nt, against the 0.90 `check_cds_length` needs.

MAFFT used FFT-NS-2 and produced 240 columns. Removing `-` recovered the
exact input protein for all 10,130.

### 2. Lengths and gaps

| CDS length | Isolates | Unique CDS | Unique proteins | Internal gap columns | Trailing gap columns |
|---|---:|---:|---:|---|---|
| 654 nt | 389 | 236 | 207 | 74, 112, 132 | 221–240 |
| 660 nt | 25,506 | 8,243 | 4,601 | 74, 112, 132 | 223–240 |
| 672 nt | 806 | 235 | 144 | 74, 112, 132 | 227–240 |
| 678 nt | 341 | 246 | 205 | 74, 81–85, 112, 132 | 234–240 |
| 693 nt | 33,553 | 9,097 | 4,926 | 74, 112, 132 | 234–240 |
| 714 nt | 16 | 11 | 11 | 74, 112, 132 | None |
| Ten other lengths | 43 | 39 | 36 | Varies | Varies |

The 678-nt form is not a truncated 693-nt form. It is the 693-nt form with 5
codons missing in the middle, at columns 81–85:

```text
Column:     71         81         91
693 nt:     EEY-DESLKM TIASVPASCY LTEM
678 nt:     EEP-DEALKM -----PTSRY LTEM
```

All 205 of its unique proteins share this pattern, over 341 isolates.

### 3. Stop codon columns

The column after a form's last protein symbol is where that form's stop codon
would go.

```text
Column:     216        226        236
714 nt:     PFTTKQKRKM ERTIGPEVRR NKMAN     last symbol 240, stop in 241
693 nt:     PFPPKQKREM ARTTGPKV-- -----     last symbol 233, stop in 234
678 nt:     PLPPNQKRKL ARAIESEV-- -----     last symbol 233, stop in 234
672 nt:     PLTPKQKREM A--------- -----     last symbol 226, stop in 227
660 nt:     SLLPEQE--- ---------- -----     last symbol 222, stop in 223
654 nt:     PLTPK----- ---------- -----     last symbol 220, stop in 221
```

The alignment has 240 amino-acid columns, but the codon representation needs 241
site features to accommodate the terminal stop codon of the 714-nt form. This
mapping is not implemented.

The 693-nt and 678-nt forms put their stop codons in the same column, 234. That
follows from the internal deletion: the two forms differ in the middle, not at
the end.

### 4. By host, subtype and year

"Other lengths" covers the ten lengths not listed separately and the 714-nt form,
so its column totals 59 isolates.

| Host | Subtype | Complete NS1 isolates | 654 nt | 660 nt | 672 nt | 678 nt | 693 nt | Other lengths |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Chicken | H1N1 | 2 | 0 | 1 | 0 | 0 | 1 | 0 |
| Chicken | H3N2 | 17 | 0 | 0 | 0 | 1 | 16 | 0 |
| Chicken | H5N1 | 1,825 | 16 | 1 | 0 | 206 | 1,600 | 2 |
| Duck | H1N1 | 266 | 0 | 0 | 0 | 0 | 266 | 0 |
| Duck | H3N2 | 217 | 0 | 0 | 0 | 0 | 217 | 0 |
| Duck | H5N1 | 1,899 | 34 | 4 | 0 | 121 | 1,739 | 1 |
| Human | H1N1 | 20,115 | 6 | 19,672 | 2 | 0 | 430 | 5 |
| Human | H3N2 | 29,423 | 161 | 124 | 804 | 0 | 28,319 | 15 |
| Human | H5N1 | 37 | 0 | 0 | 0 | 11 | 26 | 0 |
| Swine | H1N1 | 3,662 | 138 | 2,746 | 0 | 0 | 752 | 26 |
| Swine | H3N2 | 3,175 | 34 | 2,958 | 0 | 0 | 173 | 10 |
| Swine | H5N1 | 16 | 0 | 0 | 0 | 2 | 14 | 0 |

The 678-nt form is an H5N1 form: 340 of its 341 isolates are H5N1, of which 206
are Chicken and 121 Duck.

The 672-nt form is concentrated in Human-H3N2 in 2019–2020, with 753 of its 806
isolates.

By year, with the same "Other lengths" column as above:

| Year | Complete NS1 isolates | 654 nt | 660 nt | 672 nt | 678 nt | 693 nt | Other lengths |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2010 | 1,548 | 29 | 976 | 0 | 18 | 520 | 5 |
| 2011 | 1,473 | 30 | 815 | 0 | 45 | 582 | 1 |
| 2012 | 1,645 | 22 | 619 | 0 | 39 | 960 | 5 |
| 2013 | 1,822 | 11 | 780 | 0 | 51 | 977 | 3 |
| 2014 | 1,468 | 4 | 512 | 0 | 36 | 911 | 5 |
| 2015 | 2,175 | 5 | 630 | 0 | 116 | 1,419 | 5 |
| 2016 | 3,317 | 27 | 1,779 | 1 | 23 | 1,485 | 2 |
| 2017 | 4,142 | 121 | 1,125 | 0 | 11 | 2,878 | 7 |
| 2018 | 3,504 | 14 | 1,711 | 48 | 2 | 1,723 | 6 |
| 2019 | 5,255 | 19 | 2,069 | 698 | 0 | 2,462 | 7 |
| 2020 | 1,723 | 13 | 1,405 | 57 | 0 | 245 | 3 |
| 2021 | 1,538 | 12 | 380 | 0 | 0 | 1,146 | 0 |
| 2022 | 7,145 | 22 | 1,201 | 2 | 0 | 5,916 | 4 |
| 2023 | 5,137 | 36 | 3,483 | 0 | 0 | 1,616 | 2 |
| 2024 | 11,096 | 10 | 4,561 | 0 | 0 | 6,522 | 3 |
| 2025 | 7,666 | 14 | 3,460 | 0 | 0 | 4,191 | 1 |

The 660-nt and 693-nt forms are present in all 16 years. The other three listed
forms are not. The 678-nt form runs from 2010 to 2018 and does not appear after
that, and the 672-nt form appears only in 2016 and 2018–2020 and 2022.

### 5. Findings

* The 678-nt internal deletion is the first internal gap in this protein carried
  by more than a handful of records: 205 unique proteins over 341 isolates, plus
  the single 699-nt isolate, so 206 proteins and 342 isolates have a gap at
  columns 81–85. The Human-H3N2-2024 run had no internal gaps at all. Under
  the proposed codon feature mapping these 5 columns take the `GAP` state, not
  `AFTER_STOP`, because they sit between protein symbols rather than after a
  stop codon.
* The birds and H5N1 do not bring a third common length. Chicken-H5N1 is 1,600
  of 1,825 at 693 nt and Duck-H5N1 is 1,739 of 1,899, the same form as
  Human-H3N2. They bring the 678-nt form, and 33,553 isolates sit at 693 nt.
* The 660-nt and 693-nt forms cover 59,059 of the 60,654 complete records
  (97.4%). Adding 654, 672 and 678 nt reaches 60,595 (99.9%). The 714-nt form
  accounts for 16 isolates and the ten other lengths for 43.
* Columns 74, 112 and 132 each hold a protein symbol in exactly one protein; the
  other 10,129 have a gap.
* Columns 234–240 hold a protein symbol in only 12 proteins, from 17 isolates,
  which are the 714-nt and 699-nt forms. 188 of the 240 columns hold a protein
  symbol in every protein.
* There are 19 gap patterns over the 16 CDS lengths, so three lengths carry two
  patterns each: 663, 687 and 696 nt.
* Four patterns put internal gaps outside columns 74, 112 and 132: columns
  81–85 in the 678-nt and 699-nt forms, columns 77–80 in the 648-nt form, and
  columns 79–80 in the second 687-nt pattern. 2 proteins over 4 isolates have a
  gap at columns 79–80. The 645-nt form is the only one with leading gaps, at
  columns 1–5, in 1 protein from 1 isolate.
* Still open before codon features are built: whether to keep the 43 isolates
  outside the six main length forms, and whether to keep the three one-residue
  columns. Uncommon length alone does not establish an error.

### 6. Artifacts

`results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/NS1/`.
`align_proteins.py` wrote the selection, alignment, validation and gap
locations. `summarize_protein_alignment.py` wrote `group_length_summary.csv`,
`gap_pattern_summary.csv` and `column_occupancy.csv`. The manifests record file
hashes and code state.

## NS1 on Human-H3N2-2024 (Task 1 population)

The first alignment run, on 2026-11-05, where the method was worked out. It is
on the population Task 1 uses, so it is the one section not measured on the
Task 2 population.

### 1. Inputs

Of 5,346 filtered isolates, 5,337 (99.8%) had a complete CDS. They contained
1,132 unique CDS and 560 unique proteins in one host–subtype–year group. 48
isolates and 48 unique proteins contain an `X`, all at 693 nt.

This population has a usable pin. `frac at mode` is 0.991, at 693 nt, against the
0.90 `check_cds_length` needs.

MAFFT used FFT-NS-2 and produced 230 columns. Removing `-` recovered the
exact input protein for all 560. This run asked for `--auto` rather than the
flags in Method.

### 2. Lengths and gaps

The last three columns count symbols per protein in the alignment, not codon
feature values.

| CDS length | Isolates | Unique CDS | Unique proteins | Internal gap columns | Trailing gap columns | Codons incl. stop | Symbols sent to MAFFT | `-` symbols |
|---|---:|---:|---:|---|---|---:|---:|---:|
| 693 nt | 5,324 | 1,122 | 551 | None | None | 231 | 230 | 0 |
| 660 nt | 9 | 7 | 6 | None | 220–230 | 220 | 219 | 11 |
| 654 nt | 4 | 3 | 3 | None | 218–230 | 218 | 217 | 13 |

The examples below show input amino acids from position 205 to the end and their
placement in columns 205–230. Protein hash prefixes identify records in the saved
FASTA files.

| CDS length | Protein hash prefix | Input amino acids 205–end | Aligned columns 205–230 |
|---|---|---|---|
| 693 nt | `47566821` | `SSHENGGPPLTPKQKQEMARTARSEV` | `SSHENGGPPLTPKQKQEMARTARSEV` |
| 693 nt | `046b53e6` | `SSNENGGPPLTPKQKREMARTARSEV` | `SSNENGGPPLTPKQKREMARTARSEV` |
| 660 nt | `40191b0e` | `SCDENGRPSLPPEQK` | `SCDENGRPSLPPEQK-----------` |
| 654 nt | `6a1f0bc8` | `SSHENGGPPLTPK` | `SSHENGGPPLTPK-------------` |

The 654-nt example matches the first 217 amino acids of the `47566821` protein
from a 693-nt CDS. The trailing gaps are consistent with the 654-nt CDS ending
earlier.

### 3. Stop codon columns

```text
Column:     1–217       218   219   220–230
693 nt:     res...      res   res   res...res
660 nt:     res...      res   res   - ... -
654 nt:     res...      -     -     - ... -
```

The 660-nt CDS records end with a stop codon at codon 220 and the 654-nt records
at codon 218. The downstream contig check below supports placing those stop
codons opposite the same-numbered codons of the 693-nt reference. The proposed
codon feature mapping is in the NS1 alignment pilot section of [the codon site
features plan](../plans/2026-09-14_codon_site_features_plan.md#next-step). This
mapping is not implemented.

### 4. By host, subtype and year

One metadata group, so there is no breakdown.

### 5. Findings

* All 551 proteins from 693-nt CDS have no gaps, and each shorter protein has one
  run of trailing gaps.
* No protein has an `X` in columns 218–230, where one or both shorter forms have
  trailing gaps.
* This population has a usable pin at 693 nt, so the alignment is not needed to
  model NS1 here. It was run to establish the method.

### Downstream contig check

The protein alignment shows where amino acids end, but it cannot place CDS stop
codons or show whether bases remain in the contig after a stop. We checked the
source contig for all 13 isolates with a 654- or 660-nt NS1 CDS. We first
verified that each annotated location recovers its CDS exactly. We then compared
the bases after its stop with the same positions in the most frequent 693-nt CDS,
which 2,095 isolates carry. This comparison assumes that codons with the same
number, counted from the CDS start, correspond across these sequences. The
downstream matches support that placement for these 13 isolates.

| CDS length | Isolates | Contig bases after stop | Bases compared with 693-nt CDS | Matching bases |
|---|---:|---:|---:|---:|
| 654 nt | 4 | 197 | 39 | 34–39 |
| 660 nt | 9 | 178–191 | 33 | 25–27 |

At the short CDS's terminal codon, all four 654-nt sequences have `TAG` where the
693-nt reference has `CAG` (codon 218); all nine 660-nt sequences have `TGA`
where the reference has `CGG` (codon 220). The 693-nt reference ends with `TGA`
at codon 231. That triplet remains `TGA` at the corresponding contig position for
all nine 660-nt isolates and three of the four 654-nt isolates; it is `CGA` in
the remaining 654-nt isolate.

These findings support mapping the 654- and 660-nt CDS stop codons opposite the
same-numbered codons in the 693-nt reference rather than aligning all terminal
stop codons. They do not establish the exact evolutionary history. Bases beyond a
short CDS's stop are in the contig, outside that annotated CDS.

### 6. Artifacts

`results/flu/July_2025/cds_alignment_pilot/Human-H3N2-2024/NS1/`. The contig
check is `src/analysis/audit_ns1_downstream_contig.py`; its per-isolate
comparisons, summary and manifest are `ns1_downstream_contig.csv`,
`ns1_downstream_summary.csv` and `ns1_downstream_manifest.json` in the same
directory.

## Limitations

* Only NS1 has been aligned. The other seven modelled proteins have no alignment
  on any population.
* No codon features and no models were produced from any of these alignments. The
  mapping from alignment columns to codon features is proposed in
  `docs/plans/2026-09-14_codon_site_features_plan.md` and is not implemented.
* Every run used the same MAFFT strategy, FFT-NS-2. The Task 2 runs asked
  for it with `--retree 2 --maxiterate 0` and the Human-H3N2-2024 run let
  `--auto` choose it. We did not test a second strategy. Agreement between two
  strategies would not establish that a placement is correct.
* The one-residue columns and the uncommon gap patterns have not been reviewed
  sequence by sequence.
