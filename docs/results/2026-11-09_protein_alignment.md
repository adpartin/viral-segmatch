# Protein alignment for codon features

```yaml

# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-11-09
populations:    Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025; Human-H3N2-2024
proteins:       PB2; PB1; PA; HA; NP; NA; M1; NS1
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

Each protein has a section with 5 numbered parts: 1. Inputs, 2. Lengths and
gaps, 3. Stop codon columns, 4. By host, subtype and year, 5. Findings. If a
part has nothing to report, it says so in one sentence rather than being left
out. A section may add one unnumbered part for a check specific to it. The
files every section is built from are listed once, under "Artifacts".

Gap terms, as defined in [the
glossary](../methods/glossary.md#per-site-features):
- a **trailing gap** is a `-` after the last protein symbol,
- an **internal gap** is a `-` between protein symbols,
- a **leading gap** is a `-` before the first,
- and a protein symbol is an amino acid or an `X`.

A **gap pattern** is one protein's full set of gap columns. In the schematics,
`res` stands for one protein symbol.

```text
Column:     1   2   3   4   5   6   7   8   9  10
Protein A: res res res res res res res res res res
Protein B:  -   -  res res  -  res res res  -   -
```

Protein B has leading gaps in columns 1–2, an internal gap in column 5, and
trailing gaps in columns 9–10. Its gap pattern is all five: 1, 2, 5, 9 and 10.
Protein A has no gaps.

A **leading gap** does not mean a missing start codon. It means another protein's
first residue sits further left in the alignment. Both records with leading
gaps here begin with ATG: NP's 1461-nt record starts 12 codons downstream of
the other NP records, and one HA record starts 4 codons upstream of the others.

Part 2's table works the same way in every section. A CDS length gets its own
row if it holds at least 0.1% of that protein's complete records. If a protein
has 3 lengths or fewer, every length gets a row. The rest share one "Other
lengths" row, and the text below calls them the rare lengths.

The three gap columns give counts, not lists, because one NA protein gaps 93
columns. A count can be read down the column across sections. A range such as
`66 to 73` means that length holds several gap patterns.

Putting a rare length in the "Other lengths" row can hide why an alignment is
as wide as it is. So each section also says how many columns the listed lengths
use, and which rare lengths reach the rest.

This alignment does not:

* Place CDS stop codons. The terminal `*` marker is removed before MAFFT, so no
  column holds a stop codon. Where a section gives stop codon columns, that is a
  proposal read off the gap pattern, not a measurement.
* Show that a column represents the same biological position in every record.
  The alignment is an estimate of positional correspondence.
* Recover bases absent from a truncated record. These runs use complete CDS
  only.


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
3. Make the MAFFT input: copy each unique protein and remove the terminal `*`
   from the copy. Keep `X` symbols and count them by CDS length.
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

The strip applies only to that copy. The stored `prot_seq` keeps its terminal
`*` and `cds_dna_seq` keeps its terminal stop codon. So each protein carries two
hashes here. The FASTA header is `prot_hash`, the md5 of the stored protein
including the `*`. `protein_input_hashes.csv` holds `alignment_seq_sha256`, the
sha256 of the stop-free sequence MAFFT received. The stop codon the CDS keeps is
what each section's part 3 counts from when it proposes stop codon columns.

The saved protein is the exact MAFFT input, which is what step 5 compares
against. With the options used here, [MAFFT would remove `*`
anyway](https://mafft.cbrc.jp/alignment/software/anysymbol.html).

An `X` is the protein symbol for a codon that does not resolve to one amino
acid. The codon remains in the CDS, and the `X` is sent to MAFFT rather than
converted to a gap. Because `X` does not identify a single amino acid, it gives
MAFFT less information for placing nearby gaps. The M1 and Human-H3N2-2024
sections report whether any gap falls next to an `X`. The NS1 section on the
Task 2 population has not checked this.


## Summary across proteins

One row per protein, on the Task 2 population.

| Segment ID | Protein | Isolates | Unique CDS | Unique proteins | CDS lengths | Columns | `frac at mode` | Largest internal gap | One-residue columns |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | PB2 | 60,474 | 36,939 | 15,290 | 18 | 766 | 2280 nt, 36,879 of 36,939 | columns 692–715, 24 codons, 1 protein | 42–45, 575, 678 |
| 2 | PB1 | 56,093 | 34,642 | 13,220 | 18 | 776 | 2274 nt, 30,824 of 34,642 | columns 736–764, 29 codons, 1 protein | 77–79, 389, 718–719 |
| 3 | PA | 60,396 | 35,659 | 15,939 | 14 | 744 | 2151 nt, 35,609 of 35,659 | columns 80–107, 28 codons, 1 protein | 25 columns, listed in part 5 |
| 4 | HA | 60,282 | 35,509 | 19,841 | 10 | 612 | 1701 nt, 33,492 of 35,509 | columns 20–27, 8 codons, 1 protein | 14 columns, listed in part 5 |
| 5 | NP | 60,040 | 27,087 | 7,250 | 2 | 498 | 1497 nt, 27,086 of 27,087 | None | None |
| 6 | NA | 59,856 | 31,157 | 17,267 | 16 | 543 | 1410 nt, 30,630 of 31,157 | columns 68–94, 27 codons, 1 protein | 16 columns, listed in part 5 |
| 7 | M1 | 60,566 | 14,573 | 1,948 | 2 | 252 | 759 nt, 14,572 of 14,573 | columns 31–41, 11 codons, 1 protein | None |
| 8 | NS1 | 60,654 | 18,107 | 10,130 | 16 | 240 | 693 nt, 9,097 of 18,107 | columns 81–85, 5 codons, 206 proteins | 74, 112, 132 |

The table below counts the complete records a usable pin would drop. H5N1
counts include all four host groups and use complete records for that protein.

| Protein | Usable pin | Complete records dropped by pin | H5N1 records kept by pin |
|---|---:|---:|---:|
| PB2 | 2280 nt | 61 of 60,474 | 3,759 of 3,761 |
| PB1 | None | Not applicable | Not applicable |
| PA | 2151 nt | 64 of 60,396 | 3,764 of 3,766 |
| HA | 1701 nt | 4,059 of 60,282 | 13 of 3,754 |
| NP | 1497 nt | 1 of 60,040 | 3,766 of 3,766 |
| NA | 1410 nt | 702 of 59,856 | 3,211 of 3,754 |
| M1 | 759 nt | 1 of 60,566 | 3,776 of 3,776 |
| NS1 | None | Not applicable | Not applicable |

"Largest internal gap" gives the longest run of neighbouring internal gap
columns any protein has, as column numbers rather than a count of columns,
followed by how long the run is and how many proteins carry it.
`columns 81–85, 5 codons` means five gaps, one in each of the columns numbered
81 to 85. The table gives only the longest run. Each section lists all of its
internal gaps, except NA and HA, whose lists are too long to print and are in
`gap_pattern_summary.csv`.

A **one-residue column** is an alignment column where exactly one protein has a
protein symbol and every other protein has a gap.

Six of the eight proteins have a usable pin on this population: PB2, PA, HA,
NP, NA and M1. The two without one are PB1 at 0.890 and NS1 at 0.502, which are
also the two proteins `conf/virus/flu.yaml` pins no length for.
HA is the extreme: its 1701-nt pin keeps 13 of 3,754 complete H5N1 records
because the main H5 form sits at 1704 nt.

Two of the six pinned proteins hold more than one form at the pinned length.
NA's 1410 nt holds N1 and N2, and HA's 1701 nt holds H1 and H3 with H5 at
1704 nt. Those sections give the measured column overlap between the forms.

Six of the eight need one column more than the alignment has, to hold the stop
codon of their longest form. PB1 and NA do not, because their longest form's
stop codon already falls inside the alignment width.

NS1 has a usable pin at 0.991 on the Task 1 population, reported in the last
section. However, extending the population to the one Task 2 uses drops
`frac at mode` at 693 nt to 0.502, so NS1 has no pin there.


## Artifacts

Every Task 2 run wrote to
`results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/<protein>/`,
one directory per protein. The Task 1 pilot wrote to
`results/flu/July_2025/cds_alignment_pilot/Human-H3N2-2024/NS1/`.

`align_proteins.py` wrote the selection, alignment, validation and gap
locations. `summarize_protein_alignment.py` wrote `group_length_summary.csv`,
`gap_pattern_summary.csv` and `column_occupancy.csv`. The manifests record file
hashes and code state.

Two manifests record a MAFFT runtime: NA at 365.6 seconds and M1 at 3.9. The
other seven ran before `mafft_seconds` was added, so they have no runtime.

The Task 1 pilot directory also holds the downstream contig check, written by
`src/analysis/audit_ns1_downstream_contig.py` as `ns1_downstream_contig.csv`,
`ns1_downstream_summary.csv` and `ns1_downstream_manifest.json`.


## PB2 (Segment 1)

### 1. Inputs

* **Records.** Of 60,751 filtered isolates, 60,474 (99.5%) had a complete CDS
  and 277 were dropped as incomplete.
* **Sequences.** 36,939 unique CDS and 15,290 unique proteins, across 18 CDS
  lengths and 147 host–subtype–year groups.
* **Unknown residues.** 1,296 isolates and 1,197 unique proteins contain an
  `X`, all 1,296 at 2280 nt.
* **Pin.** PB2 has a usable pin at 2280 nt. 36,879 of the 36,939 unique
  complete CDS are 2280 nt, a share of 0.998, against the 0.90
  `check_cds_length` needs. `conf/virus/flu.yaml` pins PB2 at 2280 nt and the
  observed mode agrees.
* **Alignment.** MAFFT used FFT-NS-2 and produced `766` columns.
  - Removing `-` from the aligned proteins recovered the exact input protein
    for all 15,290.
  - A 2280-nt CDS is 759 amino acids, so the alignment adds 7 columns.

### 2. Lengths and gaps

| CDS length | Isolates | Unique CDS | Unique proteins | Leading gap columns | Internal gap columns | Trailing gap columns |
|---|---:|---:|---:|---|---|---|
| 2280 nt | 60,413 | 36,879 | 15,230 | None | 7 | None |
| 17 other lengths | 61 | 60 | 60 | 0 to 10 | 6 to 31 | 0 to 10 |

The 2280-nt proteins have 2 gap patterns. The biggest pattern covers 15,229 of
the 15,230 and has 7 internal gap columns: 42–45, 348, 575 and 678.

The 2280-nt proteins don't have residues in columns 42–45, 348 and 678, so these
proteins use 760 of the 766 columns. Those 6 columns are in the alignment
because the rare 2268-nt and 2283-nt forms do have residues there.

### 3. Stop codon columns

The modal form has no trailing gap, so it puts its last protein symbol in
column 766 and its stop codon in column 767. A codon feature table for PB2
needs 767 columns, not 766. This mapping is not implemented.

### 4. By host, subtype and year

"Other lengths" covers the 17 lengths not listed separately, so its
column totals 61 isolates.

| Host | Subtype | Complete PB2 isolates | 2280 nt | Other lengths |
|---|---|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 0 |
| Chicken | H3N2 | 17 | 17 | 0 |
| Chicken | H5N1 | 1,815 | 1,814 | 1 |
| Duck | H1N1 | 264 | 264 | 0 |
| Duck | H3N2 | 216 | 215 | 1 |
| Duck | H5N1 | 1,894 | 1,894 | 0 |
| Human | H1N1 | 20,017 | 19,994 | 23 |
| Human | H3N2 | 29,439 | 29,409 | 30 |
| Human | H5N1 | 36 | 36 | 0 |
| Swine | H1N1 | 3,599 | 3,595 | 4 |
| Swine | H3N2 | 3,159 | 3,158 | 1 |
| Swine | H5N1 | 16 | 15 | 1 |

| Year | Complete PB2 isolates | 2280 nt | Other lengths |
|---|---:|---:|---:|
| 2010 | 1,542 | 1,540 | 2 |
| 2011 | 1,460 | 1,458 | 2 |
| 2012 | 1,635 | 1,633 | 2 |
| 2013 | 1,810 | 1,808 | 2 |
| 2014 | 1,463 | 1,462 | 1 |
| 2015 | 2,173 | 2,170 | 3 |
| 2016 | 3,317 | 3,316 | 1 |
| 2017 | 4,130 | 4,129 | 1 |
| 2018 | 3,499 | 3,497 | 2 |
| 2019 | 5,231 | 5,230 | 1 |
| 2020 | 1,628 | 1,628 | 0 |
| 2021 | 1,536 | 1,536 | 0 |
| 2022 | 7,157 | 7,148 | 9 |
| 2023 | 5,136 | 5,127 | 9 |
| 2024 | 11,093 | 11,074 | 19 |
| 2025 | 7,664 | 7,657 | 7 |

The off-mode lengths are spread across hosts, subtypes and years, with no group
holding more than 30 of the 61 isolates.

### 5. Findings

* PB2 has a pin on this population.
* Pinning at 2280 nt would drop 61 of the 60,474 complete records and keep
  3,759 of 3,761 H5N1 records.
* The 2280-nt form holds 60,413 of the 60,474 complete records. The other 17
  lengths hold 61 isolates between them, 6 at most each.
* The alignment adds 7 columns. 6 of them are one-residue columns: 42–45, 575
  and 678.
* 576 of the 766 columns hold a protein symbol in all 15,290 proteins.
* The longest internal gap run is columns 692 to 715, 24 codons, in one protein
  from one isolate at 2208 nt.
* Still open before codon features are built: whether to keep the 61 isolates
  outside the modal length, and whether to keep the 6 one-residue columns.

## PB1 (Segment 2)

### 1. Inputs

* **Records.** Of 60,751 filtered isolates, 56,093 (92.3%) had a complete CDS
  and 4,658 were dropped as incomplete.
* **Sequences.** 34,642 unique CDS and 13,220 unique proteins, across 18 CDS
  lengths and 147 host–subtype–year groups.
* **Unknown residues.** 1,325 isolates and 1,158 unique proteins contain an
  `X`, 1,136 of them at 2274 nt.
* **Pin.** PB1 has no usable pin. 30,824 of the 34,642 unique complete CDS are
  2274 nt, a share of 0.890, below the 0.90 `check_cds_length` needs.
  `conf/virus/flu.yaml` pins no length for PB1.
* **Alignment.** MAFFT used FFT-NS-2 and produced `776` columns.
  - Removing `-` from the aligned proteins recovered the exact input protein
    for all 13,220.
  - A 2274-nt CDS is 757 amino acids, so the alignment adds 19 columns.

### 2. Lengths and gaps

| CDS length | Isolates | Unique CDS | Unique proteins | Leading gap columns | Internal gap columns | Trailing gap columns |
|---|---:|---:|---:|---|---|---|
| 2274 nt | 49,948 | 30,824 | 12,060 | None | 15 to 16 | 3 to 4 |
| 2277 nt | 6,095 | 3,768 | 1,110 | None | 14 to 15 | 3 to 4 |
| 16 other lengths | 50 | 50 | 50 | None | 15 to 41 | 0 to 11 |

The 2274-nt proteins have 3 gap patterns. The biggest pattern covers 12,058 of
the 12,060 and has 15 internal gap columns: 77–79, 389 and 718–728.

The listed lengths don't have residues in columns 718–728 or 774–776, so these
lengths use 762 of the 776 columns. Those 14 columns are in the alignment
because the rare 2217-nt and 2232-nt forms do have residues there.

The two main lengths are the same form, one codon apart at the C-terminus.
Their internal gaps are identical and only their trailing run differs, by the
single column 773:

```text
Column:     772 773 774 775 776
2274 nt:    res  -   -   -   -     last symbol 772, stop in 773
2277 nt:    res res  -   -   -     last symbol 773, stop in 774
```

Together they hold 56,043 of the 56,093 complete records.

### 3. Stop codon columns

The two main forms end one column apart, so they would put their stop codons in
columns 773 and 774. Both fall inside the 776 columns, so a codon feature table
for PB1 needs 776 columns and no more. This mapping is not implemented.

### 4. By host, subtype and year

"Other lengths" covers the sixteen lengths not listed separately, so its column
totals 50 isolates.

| Host | Subtype | Complete PB1 isolates | 2274 nt | 2277 nt | Other lengths |
|---|---|---:|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 0 | 0 |
| Chicken | H3N2 | 17 | 17 | 0 | 0 |
| Chicken | H5N1 | 1,798 | 1,796 | 2 | 0 |
| Duck | H1N1 | 263 | 259 | 4 | 0 |
| Duck | H3N2 | 217 | 212 | 4 | 1 |
| Duck | H5N1 | 1,871 | 1,869 | 1 | 1 |
| Human | H1N1 | 19,761 | 19,706 | 28 | 27 |
| Human | H3N2 | 25,390 | 19,394 | 5,984 | 12 |
| Human | H5N1 | 35 | 35 | 0 | 0 |
| Swine | H1N1 | 3,578 | 3,530 | 41 | 7 |
| Swine | H3N2 | 3,148 | 3,115 | 31 | 2 |
| Swine | H5N1 | 13 | 13 | 0 | 0 |

The 2277-nt form is a Human-H3N2 form: 5,984 of its 6,095 isolates are
Human-H3N2, which is 23.6% of that group's 25,390 complete PB1 records.

| Year | Complete PB1 isolates | 2274 nt | 2277 nt | Other lengths | 2277 nt share |
|---|---:|---:|---:|---:|---:|
| 2010 | 1,542 | 1,512 | 30 | 0 | 1.9% |
| 2011 | 1,471 | 1,466 | 4 | 1 | 0.3% |
| 2012 | 1,634 | 1,628 | 6 | 0 | 0.4% |
| 2013 | 1,813 | 1,808 | 5 | 0 | 0.3% |
| 2014 | 1,461 | 1,453 | 8 | 0 | 0.5% |
| 2015 | 2,169 | 2,165 | 4 | 0 | 0.2% |
| 2016 | 3,315 | 3,303 | 10 | 2 | 0.3% |
| 2017 | 4,125 | 4,109 | 14 | 2 | 0.3% |
| 2018 | 3,487 | 3,472 | 9 | 6 | 0.3% |
| 2019 | 5,205 | 5,190 | 10 | 5 | 0.2% |
| 2020 | 1,529 | 1,517 | 10 | 2 | 0.7% |
| 2021 | 1,537 | 1,488 | 48 | 1 | 3.1% |
| 2022 | 7,010 | 6,675 | 328 | 7 | 4.7% |
| 2023 | 4,831 | 4,289 | 533 | 9 | 11.1% |
| 2024 | 8,677 | 5,720 | 2,948 | 9 | 34.0% |
| 2025 | 6,287 | 4,153 | 2,128 | 6 | 33.9% |

The last column is the 2277-nt share of the two main lengths. It stays between
0.2% and 1.9% from 2010 to 2020, then rises each year to 34.0% in 2024 and
33.9% in 2025.

### 5. Findings

* PB1 has no pin on this population. The modal length holds 0.890 of unique
  complete CDS, below the 0.90 floor, and `conf/virus/flu.yaml` pins no length
  for it.
* No pin is available, so the pinning cost does not apply.
* The two main lengths differ by one codon at the C-terminus and share every
  internal gap column, so the pin fails on a three-nucleotide difference rather
  than on two different proteins.
* The 2277-nt form was 0.2% to 1.9% of the two main lengths from 2010 to 2020
  and is 34.0% in 2024 and 33.9% in 2025. It is concentrated in Human-H3N2.
* PB1 lost 4,658 of 60,751 isolates to the completeness filter, 7.7%.
* 563 of the 776 columns hold a protein symbol in all 13,220 proteins.
* 6 columns hold a protein symbol in exactly one protein: 77–79, 389 and
  718–719.
* The longest internal gap run is columns 736 to 764, 29 codons, in one protein
  from one isolate at 2217 nt.
* Still open before codon features are built: whether the two main lengths can
  share one set of columns, given that they differ only in the last codon.

## PA (Segment 3)

### 1. Inputs

* **Records.** Of 60,751 filtered isolates, 60,396 (99.4%) had a complete CDS
  and 355 were dropped as incomplete.
* **Sequences.** 35,659 unique CDS and 15,939 unique proteins, across 14 CDS
  lengths and 147 host–subtype–year groups.
* **Unknown residues.** 1,349 isolates and 1,203 unique proteins contain an
  `X`, 1,348 of them at 2151 nt.
* **Pin.** PA has a usable pin at 2151 nt. 35,609 of the 35,659 unique complete
  CDS are 2151 nt, a share of 0.999, against the 0.90 `check_cds_length` needs.
  `conf/virus/flu.yaml` pins PA at 2151 nt and the observed mode agrees.
* **Alignment.** MAFFT used FFT-NS-2 and produced `744` columns.
  - Removing `-` from the aligned proteins recovered the exact input protein
    for all 15,939.
  - A 2151-nt CDS is 716 amino acids, so the alignment adds 28 columns.

### 2. Lengths and gaps

| CDS length | Isolates | Unique CDS | Unique proteins | Leading gap columns | Internal gap columns | Trailing gap columns |
|---|---:|---:|---:|---|---|---|
| 2151 nt | 60,332 | 35,609 | 15,897 | None | 26 | 2 |
| 13 other lengths | 64 | 50 | 42 | 0 to 20 | 22 to 49 | 0 to 25 |

The 2151-nt proteins have 2 gap patterns. The biggest pattern covers 15,896 of
the 15,897 and has 26 internal gap columns: 48–49, 77, 82–83, 94–98, 624,
641–650, 660–663 and 703.

The 2151-nt proteins don't have residues in columns 48–49, 77, 82–83, 624,
641–650, 660–663, 703 or 743–744, so these proteins use 721 of the 744 columns.
Those 23 columns are in the alignment because the rare 2073-nt, 2154-nt,
2157-nt, 2160-nt and 2163-nt forms do have residues there.

The biggest of the rare lengths is 2157 nt, at 39 isolates. Its main form is
the modal form with two more codons at the C-terminus, which is how it fills
columns 743 and 744 that the modal form leaves empty.

### 3. Stop codon columns

```text
Column:     742 743 744
2151 nt:    res  -   -      last symbol 742, stop in 743
2157 nt:    res res res     last symbol 744, stop in 745
```

A codon feature table for PA needs 745 columns, not 744, because the 2157-nt
form fills the last column. This mapping is not implemented.

### 4. By host, subtype and year

"Other lengths" covers the twelve lengths not listed separately, so its column
totals 25 isolates.

| Host | Subtype | Complete PA isolates | 2151 nt | 2157 nt | Other lengths |
|---|---|---:|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 0 | 0 |
| Chicken | H3N2 | 17 | 17 | 0 | 0 |
| Chicken | H5N1 | 1,824 | 1,823 | 0 | 1 |
| Duck | H1N1 | 265 | 265 | 0 | 0 |
| Duck | H3N2 | 217 | 216 | 0 | 1 |
| Duck | H5N1 | 1,898 | 1,897 | 0 | 1 |
| Human | H1N1 | 19,896 | 19,892 | 0 | 4 |
| Human | H3N2 | 29,450 | 29,399 | 39 | 12 |
| Human | H5N1 | 36 | 36 | 0 | 0 |
| Swine | H1N1 | 3,619 | 3,616 | 0 | 3 |
| Swine | H3N2 | 3,164 | 3,161 | 0 | 3 |
| Swine | H5N1 | 8 | 8 | 0 | 0 |

Every one of the 39 isolates at 2157 nt is Human-H3N2.

| Year | Complete PA isolates | 2151 nt | 2157 nt | Other lengths |
|---|---:|---:|---:|---:|
| 2010 | 1,546 | 1,545 | 0 | 1 |
| 2011 | 1,476 | 1,475 | 0 | 1 |
| 2012 | 1,646 | 1,645 | 0 | 1 |
| 2013 | 1,822 | 1,821 | 0 | 1 |
| 2014 | 1,469 | 1,467 | 0 | 2 |
| 2015 | 2,179 | 2,177 | 0 | 2 |
| 2016 | 3,318 | 3,317 | 0 | 1 |
| 2017 | 4,144 | 4,139 | 3 | 2 |
| 2018 | 3,504 | 3,502 | 1 | 1 |
| 2019 | 5,152 | 5,147 | 3 | 2 |
| 2020 | 1,563 | 1,562 | 0 | 1 |
| 2021 | 1,537 | 1,516 | 19 | 2 |
| 2022 | 7,152 | 7,144 | 4 | 4 |
| 2023 | 5,135 | 5,135 | 0 | 0 |
| 2024 | 11,089 | 11,082 | 3 | 4 |
| 2025 | 7,664 | 7,658 | 6 | 0 |

The 2157-nt form appears in 8 of the 16 years, with 19 of its 39 isolates in
2021.

### 5. Findings

* PA has a pin on this population.
* Pinning at 2151 nt would drop 64 of the 60,396 complete records and keep
  3,764 of 3,766 H5N1 records.
* The 2151-nt form holds 60,332 of the 60,396 complete records.
* The alignment adds 28 columns, and 25 of them are one-residue columns: 49,
  77, 82–83, 94–98, 624, 641–650, 660–663 and 703. So nearly all of the extra
  width comes from single proteins.
* 587 of the 744 columns hold a protein symbol in all 15,939 proteins.
* The longest internal gap run is columns 80 to 107, 28 codons, in one protein
  from one isolate at 2088 nt.
* Still open before codon features are built: whether to keep the 25
  one-residue columns, which would cut the table from 745 columns to 720.

## HA (Segment 4)

### 1. Inputs

* **Records.** Of 60,751 filtered isolates, 60,282 (99.2%) had a complete CDS
  and 469 were dropped as incomplete.
* **Sequences.** 35,509 unique CDS and 19,841 unique proteins, across 10 CDS
  lengths and 147 host–subtype–year groups.
* **Unknown residues.** 1,797 isolates and 1,613 unique proteins contain an
  `X`, 1,756 of them at 1701 nt.
* **Pin.** HA has a usable pin at 1701 nt. 33,492 of the 35,509 unique complete
  CDS are 1701 nt, a share of 0.943, against the 0.90 `check_cds_length` needs.
  `conf/virus/flu.yaml` pins HA at 1701 nt and the observed mode agrees.
* **Alignment.** MAFFT used FFT-NS-2 and produced `612` columns.
  - Removing `-` from the aligned proteins recovered the exact input protein
    for all 19,841.
  - A 1701-nt CDS is 566 amino acids, so the alignment adds 46 columns.

### 2. Lengths and gaps

| CDS length | Isolates | Unique CDS | Unique proteins | Leading gap columns | Internal gap columns | Trailing gap columns |
|---|---:|---:|---:|---|---|---|
| 1695 nt | 244 | 203 | 179 | 4 | 44 | None |
| 1698 nt | 71 | 67 | 62 | 4 | 43 | None |
| 1701 nt | 56,223 | 33,492 | 18,606 | 4 | 42 | None |
| 1704 nt | 3,632 | 1,643 | 901 | 4 | 41 | None |
| 1707 nt | 63 | 56 | 50 | 0 to 4 | 40 to 44 | None |
| 5 other lengths | 49 | 48 | 43 | 4 to 14 | 39 to 48 | None |

The 1701-nt proteins have 6 gap patterns. The biggest pattern covers 10,274 of
the 18,606 and has 42 internal gap columns, which `gap_pattern_summary.csv`
lists.

The listed lengths don't have residues in columns 20, 321, 335–337, 375 or
601–603, so these lengths use 603 of the 612 columns. Those 9 columns are in the
alignment because the rare 1683-nt, 1686-nt and 1710-nt forms do have residues
there.

Every form begins at column 5, because columns 1 to 4 belong to one protein
only.

HA holds three forms, which split by HA subtype. The two at 1701 nt are H3 and
H1, and the 1704-nt form is H5:

| Form | CDS length | Unique proteins | Isolates | H3N2 | H1N1 | H5N1 |
|---|---:|---:|---:|---:|---:|---:|
| H3 | 1701 nt | 10,274 | 32,783 | 32,778 | 3 | 2 |
| H1 | 1701 nt | 8,326 | 23,428 | 4 | 23,424 | 0 |
| H5 | 1704 nt | 870 | 3,596 | 0 | 0 | 3,596 |

These three cover 59,807 of the 60,282 complete records. H3 and H1 are both 566
amino acids and H5 is 567, and they do not use the same columns. Of the 583
columns the three use between them, 550 are filled by all three, 14 by H3 only,
3 by H5 only, 14 by H1 and H5 but not H3, and 2 by H1 and H3 but not H5. No
column is filled by H1 alone.

### 3. Stop codon columns

All three forms put their last protein symbol in column 612, so all three would
put their stop codon in column 613. A codon feature table for HA needs 613
columns, not 612. This mapping is not implemented.

### 4. By host, subtype and year

"Other lengths" covers the seven lengths not listed separately, so its column
totals 183 isolates.

| Host | Subtype | Complete HA isolates | 1695 nt | 1701 nt | 1704 nt | Other lengths |
|---|---|---:|---:|---:|---:|---:|
| Chicken | H1N1 | 2 | 0 | 2 | 0 | 0 |
| Chicken | H3N2 | 17 | 0 | 17 | 0 | 0 |
| Chicken | H5N1 | 1,816 | 3 | 0 | 1,791 | 22 |
| Duck | H1N1 | 265 | 0 | 265 | 0 | 0 |
| Duck | H3N2 | 217 | 0 | 215 | 2 | 0 |
| Duck | H5N1 | 1,886 | 24 | 13 | 1,780 | 69 |
| Human | H1N1 | 20,013 | 0 | 19,991 | 1 | 21 |
| Human | H3N2 | 29,422 | 0 | 29,420 | 0 | 2 |
| Human | H5N1 | 37 | 0 | 0 | 29 | 8 |
| Swine | H1N1 | 3,454 | 217 | 3,170 | 11 | 56 |
| Swine | H3N2 | 3,138 | 0 | 3,130 | 4 | 4 |
| Swine | H5N1 | 15 | 0 | 0 | 14 | 1 |

The 1704-nt form is an H5N1 form: 3,614 of its 3,632 isolates are H5N1. The
1695-nt form is a Swine-H1N1 form, with 217 of its 244 isolates.

| Year | Complete HA isolates | 1695 nt | 1701 nt | 1704 nt | Other lengths |
|---|---:|---:|---:|---:|---:|
| 2010 | 1,518 | 2 | 1,493 | 12 | 11 |
| 2011 | 1,432 | 11 | 1,370 | 27 | 24 |
| 2012 | 1,636 | 2 | 1,588 | 35 | 11 |
| 2013 | 1,809 | 3 | 1,737 | 41 | 28 |
| 2014 | 1,464 | 0 | 1,416 | 38 | 10 |
| 2015 | 2,168 | 4 | 2,044 | 115 | 5 |
| 2016 | 3,312 | 2 | 3,264 | 41 | 5 |
| 2017 | 4,121 | 6 | 4,064 | 44 | 7 |
| 2018 | 3,466 | 1 | 3,437 | 23 | 5 |
| 2019 | 5,215 | 3 | 5,192 | 12 | 8 |
| 2020 | 1,667 | 2 | 1,642 | 17 | 6 |
| 2021 | 1,526 | 7 | 1,478 | 35 | 6 |
| 2022 | 7,115 | 15 | 6,089 | 1,006 | 5 |
| 2023 | 5,097 | 139 | 4,674 | 283 | 1 |
| 2024 | 11,081 | 37 | 9,848 | 1,190 | 6 |
| 2025 | 7,655 | 10 | 6,887 | 713 | 45 |

The 1704-nt form was 12 to 115 isolates a year from 2010 to 2021, then 1,006 in
2022 and 1,190 in 2024. The 1695-nt form has 139 of its 244 isolates in 2023.

### 5. Findings

* HA has a pin on this population, at 1701 nt for 0.943 of unique complete CDS.
* Pinning at 1701 nt would drop 4,059 of the 60,282 complete records and keep
  13 of 3,754 H5N1 records.
* The modal length holds two forms, H3 and H1, and a third form, H5, sits at
  1704 nt. Of the 583 columns the three use between them, 550 are filled by all
  three and 33 are not.
* The 1704-nt form's rise tracks the H5N1 isolates in the population, from 12
  to 115 a year through 2021 to 1,190 in 2024.
* 47 gap patterns cover the 10 CDS lengths. The modal length alone holds 6.
* 522 of the 612 columns hold a protein symbol in all 19,841 proteins.
* 14 columns hold a protein symbol in exactly one protein: 1–4, 191–192, 198,
  381, 383, 601–603 and 608–609. Columns 1 to 4 sit before every other
  protein's first symbol.
* The longest internal gap run is columns 20 to 27, 8 codons, in one protein
  from one isolate at 1680 nt.
* Still open before codon features are built: whether one set of columns can
  serve H1, H3 and H5, and whether to keep the 14 one-residue columns.

## NP (Segment 5)

### 1. Inputs

* **Records.** Of 60,751 filtered isolates, 60,040 (98.8%) had a complete CDS
  and 711 were dropped as incomplete.
* **Sequences.** 27,087 unique CDS and 7,250 unique proteins, across 2 CDS
  lengths and 147 host–subtype–year groups.
* **Unknown residues.** 541 isolates and 436 unique proteins contain an `X`,
  all 541 at 1497 nt.
* **Pin.** NP has a usable pin at 1497 nt. 27,086 of the 27,087 unique complete
  CDS are 1497 nt, a share of 1.000, against the 0.90 `check_cds_length` needs.
  `conf/virus/flu.yaml` pins NP at 1497 nt and the observed mode agrees.
* **Alignment.** MAFFT used FFT-NS-2 and produced `498` columns.
  - Removing `-` from the aligned proteins recovered the exact input protein
    for all 7,250.
  - A 1497-nt CDS is 498 amino acids, so the alignment adds no columns.

### 2. Lengths and gaps

| CDS length | Isolates | Unique CDS | Unique proteins | Leading gap columns | Internal gap columns | Trailing gap columns |
|---|---:|---:|---:|---|---|---|
| 1461 nt | 1 | 1 | 1 | 12 | None | None |
| 1497 nt | 60,039 | 27,086 | 7,249 | None | None | None |

The 1497-nt proteins have 1 gap pattern. It covers all 7,249 and has no internal
gap columns.

The listed lengths use all 498 columns, so no rare length adds width.

The 1461-nt form is the 1497-nt form without its first 12 codons. Its gaps are
leading rather than internal, so it starts at column 13 and ends at column 498
with every other protein:

```text
Column:     1          11         21
1497 nt:    MASQGTKRSY EQMETDGDRQ NATEIR
1461 nt:    ---------- --METGGERQ NATEIR
```

One isolate carries this form, from Swine-H3N2 in 2011. The 1497-nt row above
is the most common 1497-nt protein, which 9,227 isolates carry.

### 3. Stop codon columns

Neither form has a trailing gap, so both put their last protein symbol in
column 498 and their stop codon in column 499. A codon feature table for NP
needs 499 columns, not 498. This mapping is not implemented.

### 4. By host, subtype and year

"Other lengths" is the 1461-nt form, the only length not listed separately.

| Host | Subtype | Complete NP isolates | 1497 nt | Other lengths |
|---|---|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 0 |
| Chicken | H3N2 | 17 | 17 | 0 |
| Chicken | H5N1 | 1,819 | 1,819 | 0 |
| Duck | H1N1 | 263 | 263 | 0 |
| Duck | H3N2 | 216 | 216 | 0 |
| Duck | H5N1 | 1,895 | 1,895 | 0 |
| Human | H1N1 | 19,824 | 19,824 | 0 |
| Human | H3N2 | 29,177 | 29,177 | 0 |
| Human | H5N1 | 36 | 36 | 0 |
| Swine | H1N1 | 3,608 | 3,608 | 0 |
| Swine | H3N2 | 3,167 | 3,166 | 1 |
| Swine | H5N1 | 16 | 16 | 0 |

By year: every year is entirely 1497 nt except 2011, which holds the single
1461-nt isolate among its 1,472.

### 5. Findings

* NP has a pin on this population.
* Pinning at 1497 nt would drop 1 of the 60,040 complete records and keep
  3,766 of 3,766 H5N1 records.
* The 1497-nt form holds 60,039 of the 60,040 complete records. The 1461-nt
  form is one isolate.
* The alignment adds no columns. Its 498 columns are the 498 amino acids of a
  1497-nt CDS.
* NP has no internal gaps at any length. The only gaps in the alignment are the
  12 leading columns of the 1461-nt form.
* 486 of the 498 columns hold a protein symbol in all 7,250 proteins. The other
  12 are columns 1–12, which hold one in 7,249.
* There are no one-residue columns.
* Still open before codon features are built: whether to keep the single
  1461-nt isolate. Uncommon length alone does not establish an error.

## NA (Segment 6)

### 1. Inputs

* **Records.** Of 60,751 filtered isolates, 59,856 (98.5%) had a complete CDS
  and 895 were dropped as incomplete.
* **Sequences.** 31,157 unique CDS and 17,267 unique proteins, across 16 CDS
  lengths and 146 host–subtype–year groups.
* **Unknown residues.** 1,637 isolates and 1,137 unique proteins contain an
  `X`, 1,627 of them at 1410 nt.
* **Pin.** NA has a usable pin at 1410 nt. 30,630 of the 31,157 unique complete
  CDS are 1410 nt, a share of 0.983, against the 0.90 `check_cds_length` needs.
  `conf/virus/flu.yaml` pins NA at 1410 nt and the observed mode agrees.
* **Alignment.** MAFFT used FFT-NS-2 and produced `543` columns.
  - Removing `-` from the aligned proteins recovered the exact input protein
    for all 17,267.
  - A 1410-nt CDS is 469 amino acids, so the alignment adds 74 columns.

### 2. Lengths and gaps

| CDS length | Isolates | Unique CDS | Unique proteins | Leading gap columns | Internal gap columns | Trailing gap columns |
|---|---:|---:|---:|---|---|---|
| 1350 nt | 505 | 407 | 319 | None | 93 | 1 |
| 1410 nt | 59,154 | 30,630 | 16,843 | None | 66 to 73 | 1 to 8 |
| 1413 nt | 129 | 59 | 55 | None | 70 to 73 | 0 to 3 |
| 13 other lengths | 68 | 61 | 50 | 0 to 14 | 67 to 93 | 0 to 2 |

The 1410-nt proteins have 9 gap patterns. The biggest pattern covers 8,510 of
the 16,843 and has 72 internal gap columns, which `gap_pattern_summary.csv`
lists.

The listed lengths don't have residues in columns 65, 456–457, 467–471 or 501,
so these lengths use 534 of the 543 columns. Those 9 columns are in the
alignment because the rare 1353-nt, 1416-nt, 1422-nt and 1428-nt forms do have
residues there.

Every length carries internal gaps, including the modal one.

NA holds two forms at the modal length, which split by neuraminidase subtype.
They cover 59,117 of the 59,154 isolates at 1410 nt:

| Form | CDS length | Unique proteins | Isolates | H3N2 | H1N1 | H5N1 |
|---|---:|---:|---:|---:|---:|---:|
| N2 | 1410 nt | 8,510 | 32,486 | 32,462 | 11 | 13 |
| N1 | 1410 nt | 8,305 | 26,631 | 8 | 23,447 | 3,176 |

H5N1 and H1N1 both carry N1, as their names say. Both forms are 469 amino acids
in 543 columns, but they do not use the same columns:

```text
Column:     86         96         106        116
N2 (H3N2):  LT-NTT---- ---------- -I-------- -EKEI
N1 (H1N1):  IS-NTN---S AARQSVASVK -LAG------ -NSSL
```

Of the 493 columns the two forms use between them, 445 are filled by both, 24
by N2 only and 24 by N1 only.

### 3. Stop codon columns

The two modal forms end in different columns, so they would put their stop
codons in different columns too.

```text
Column:     541 542 543
N2, 1410:   res  -   -      last symbol 541, stop in 542
N1, 1410:   res res  -      last symbol 542, stop in 543
1350 nt:    res res  -      last symbol 542, stop in 543
```

Both stop codons fall inside the 543 columns, so a codon feature table for NA
needs 543 columns and no more. This mapping is not implemented.

### 4. By host, subtype and year

"Other lengths" covers the twelve lengths not listed separately, so its column
totals 30 isolates.

| Host | Subtype | Complete NA isolates | 1350 nt | 1407 nt | 1410 nt | 1413 nt | Other lengths |
|---|---|---:|---:|---:|---:|---:|---:|
| Chicken | H1N1 | 2 | 0 | 0 | 2 | 0 | 0 |
| Chicken | H3N2 | 17 | 0 | 0 | 16 | 0 | 1 |
| Chicken | H5N1 | 1,817 | 227 | 1 | 1,584 | 0 | 5 |
| Duck | H1N1 | 263 | 0 | 0 | 263 | 0 | 0 |
| Duck | H3N2 | 217 | 0 | 0 | 215 | 0 | 2 |
| Duck | H5N1 | 1,890 | 266 | 0 | 1,592 | 28 | 4 |
| Human | H1N1 | 19,686 | 0 | 0 | 19,630 | 53 | 3 |
| Human | H3N2 | 29,162 | 0 | 4 | 29,138 | 18 | 2 |
| Human | H5N1 | 36 | 10 | 0 | 26 | 0 | 0 |
| Swine | H1N1 | 3,607 | 0 | 32 | 3,566 | 5 | 4 |
| Swine | H3N2 | 3,148 | 0 | 1 | 3,113 | 25 | 9 |
| Swine | H5N1 | 11 | 2 | 0 | 9 | 0 | 0 |

The 1350-nt form is an H5N1 form: all 505 of its isolates are H5N1, of which
266 are Duck, 227 Chicken, 10 Human and 2 Swine.

| Year | Complete NA isolates | 1350 nt | 1407 nt | 1410 nt | 1413 nt | Other lengths |
|---|---:|---:|---:|---:|---:|---:|
| 2010 | 1,539 | 19 | 0 | 1,520 | 0 | 0 |
| 2011 | 1,455 | 45 | 1 | 1,406 | 3 | 0 |
| 2012 | 1,629 | 35 | 0 | 1,587 | 7 | 0 |
| 2013 | 1,804 | 51 | 3 | 1,744 | 4 | 2 |
| 2014 | 1,460 | 33 | 1 | 1,373 | 52 | 1 |
| 2015 | 2,177 | 115 | 2 | 2,046 | 10 | 4 |
| 2016 | 3,316 | 41 | 0 | 3,266 | 3 | 6 |
| 2017 | 4,129 | 46 | 2 | 4,077 | 3 | 1 |
| 2018 | 3,495 | 23 | 1 | 3,468 | 1 | 2 |
| 2019 | 5,184 | 12 | 10 | 5,156 | 4 | 2 |
| 2020 | 1,547 | 15 | 5 | 1,527 | 0 | 0 |
| 2021 | 1,537 | 21 | 0 | 1,514 | 2 | 0 |
| 2022 | 7,144 | 13 | 5 | 7,113 | 9 | 4 |
| 2023 | 4,961 | 22 | 2 | 4,936 | 0 | 1 |
| 2024 | 10,895 | 12 | 6 | 10,870 | 3 | 4 |
| 2025 | 7,584 | 2 | 0 | 7,551 | 28 | 3 |

The 1350-nt form is present in all 16 years. The 1413-nt form peaks twice, with
52 of its 129 isolates in 2014 and 28 in 2025.

### 5. Findings

* NA has a pin on this population, at 1410 nt for 0.983 of unique complete CDS.
* Pinning at 1410 nt would drop 702 of the 59,856 complete records and keep
  3,211 of 3,754 H5N1 records.
* The modal length holds two forms, N1 and N2, which do not fill the same
  columns. Of the 493 columns they use between them, 24 are N2 only and 24 are
  N1 only.
* The 1410-nt length holds 9 gap patterns, of which the N2 and N1 forms cover
  59,117 of its 59,154 isolates.
* 45 gap patterns cover the 16 CDS lengths. The 1413-nt length alone holds 11
  and the 1410-nt length 9, so a length does not determine a pattern.
* 379 of the 543 columns hold a protein symbol in all 17,267 proteins.
* 16 columns hold a protein symbol in exactly one protein: 93, 94, 456, 457,
  467 to 471, 494, and 527 to 532.
* The longest internal gap run is columns 68 to 94, 27 codons, in one protein
  from one isolate at 1353 nt.
* Still open before codon features are built: whether one set of columns can
  serve both N1 and N2, whether to keep the 30 isolates outside the four listed
  lengths, and whether to keep the 16 one-residue columns.

## M1 (Segment 7)

### 1. Inputs

* **Records.** Of 60,751 filtered isolates, 60,566 (99.7%) had a complete CDS
  and 185 were dropped as incomplete.
* **Sequences.** 14,573 unique CDS and 1,948 unique proteins, across 2 CDS
  lengths and 147 host–subtype–year groups.
* **Unknown residues.** 425 isolates and 185 unique proteins contain an `X`,
  all 425 at 759 nt.
* **Pin.** M1 has a usable pin at 759 nt. 14,572 of the 14,573 unique complete
  CDS are 759 nt, a share of 1.000, against the 0.90 `check_cds_length` needs.
  `conf/virus/flu.yaml` pins M1 at 759 nt and the observed mode agrees.
* **Alignment.** MAFFT used FFT-NS-2 and produced `252` columns.
  - Removing `-` from the aligned proteins recovered the exact input protein
    for all 1,948.
  - A 759-nt CDS is 252 amino acids, so the alignment adds no columns.

### 2. Lengths and gaps

| CDS length | Isolates | Unique CDS | Unique proteins | Leading gap columns | Internal gap columns | Trailing gap columns |
|---|---:|---:|---:|---|---|---|
| 726 nt | 1 | 1 | 1 | None | 11 | None |
| 759 nt | 60,565 | 14,572 | 1,947 | None | None | None |

The 759-nt proteins have 1 gap pattern. It covers all 1,947 and has no internal
gap columns.

The listed lengths use all 252 columns, so no rare length adds width.

The 726-nt form is not a truncated 759-nt form. It is the 759-nt form with 11
codons missing in the middle, at columns 31–41:

```text
Column:     21         31         41         51
759 nt:     KAEIAQRLED VFAGKNTDLE ALMEWLKTRP ILSP
726 nt:     KAEIAQRLED ---------- -LMEWLKTRP ILSP
```

One isolate carries this form, from Swine-H1N1 in 2011. The 759-nt row above is
the most common 759-nt protein, which 27,661 isolates carry.

### 3. Stop codon columns

Neither form has a trailing gap, so both put their last protein symbol in
column 252 and their stop codon in column 253. A codon feature table for M1
needs 253 columns, not 252. This mapping is not implemented.

### 4. By host, subtype and year

"Other lengths" is 0 throughout, because M1 has only the two lengths listed.

| Host | Subtype | Complete M1 isolates | 726 nt | 759 nt | Other lengths |
|---|---|---:|---:|---:|---:|
| Chicken | H1N1 | 2 | 0 | 2 | 0 |
| Chicken | H3N2 | 17 | 0 | 17 | 0 |
| Chicken | H5N1 | 1,824 | 0 | 1,824 | 0 |
| Duck | H1N1 | 265 | 0 | 265 | 0 |
| Duck | H3N2 | 217 | 0 | 217 | 0 |
| Duck | H5N1 | 1,900 | 0 | 1,900 | 0 |
| Human | H1N1 | 20,001 | 0 | 20,001 | 0 |
| Human | H3N2 | 29,456 | 0 | 29,456 | 0 |
| Human | H5N1 | 37 | 0 | 37 | 0 |
| Swine | H1N1 | 3,658 | 1 | 3,657 | 0 |
| Swine | H3N2 | 3,174 | 0 | 3,174 | 0 |
| Swine | H5N1 | 15 | 0 | 15 | 0 |

By year: every year is entirely 759 nt except 2011, which holds the single
726-nt isolate among its 1,473.

### 5. Findings

* M1 has a pin on this population.
* Pinning at 759 nt would drop 1 of the 60,566 complete records and keep
  3,776 of 3,776 H5N1 records.
* The 759-nt form holds 60,565 of the 60,566 complete records. The 726-nt form
  is one isolate.
* The alignment adds no columns. Its 252 columns are the 252 amino acids of a
  759-nt CDS, because the one protein that differs is shorter in the middle
  rather than longer at either end.
* 241 of the 252 columns hold a protein symbol in all 1,948 proteins. The other
  11 are columns 31–41, which hold one in 1,947, every protein but the 726-nt
  one.
* There are no one-residue columns.
* The gap columns carry `X` symbols in other records. Columns 31–41 hold an `X`
  in 9 of the other 1,947 proteins, and columns 30 and 42 hold one in 7.
* Still open before codon features are built: whether to keep the single 726-nt
  isolate. Uncommon length alone does not establish an error.

## NS1 (Segment 8)

### 1. Inputs

* **Records.** Of 60,751 filtered isolates, 60,654 (99.8%) had a complete CDS
  and 97 were dropped as incomplete.
* **Sequences.** 18,107 unique CDS and 10,130 unique proteins, across 16 CDS
  lengths and 147 host–subtype–year groups.
* **Unknown residues.** 957 isolates and 551 unique proteins contain an `X`,
  679 of them at 693 nt.
* **Pin.** NS1 has no usable pin. 9,097 of the 18,107 unique complete CDS are
  693 nt, a share of 0.502, below the 0.90 `check_cds_length` needs.
  `conf/virus/flu.yaml` pins no length for NS1.
* **Alignment.** MAFFT used FFT-NS-2 and produced `240` columns.
  - Removing `-` from the aligned proteins recovered the exact input protein
    for all 10,130.
  - A 693-nt CDS is 230 amino acids, so the alignment adds 10 columns.

### 2. Lengths and gaps

| CDS length | Isolates | Unique CDS | Unique proteins | Leading gap columns | Internal gap columns | Trailing gap columns |
|---|---:|---:|---:|---|---|---|
| 654 nt | 389 | 236 | 207 | None | 3 | 20 |
| 660 nt | 25,506 | 8,243 | 4,601 | None | 3 | 18 |
| 672 nt | 806 | 235 | 144 | None | 3 | 14 |
| 678 nt | 341 | 246 | 205 | None | 8 | 7 |
| 693 nt | 33,553 | 9,097 | 4,926 | None | 3 | 7 |
| 11 other lengths | 59 | 50 | 47 | 0 to 5 | 2 to 8 | 0 to 35 |

The 693-nt proteins have 1 gap pattern. It covers all 4,926 and has 3 internal
gap columns: 74, 112 and 132.

The listed lengths don't have residues in columns 74, 112, 132 or 234–240, so
these lengths use 230 of the 240 columns. Those 10 columns are in the alignment
because the rare 663-nt, 696-nt, 699-nt and 714-nt forms do have residues there.

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

* NS1 has no pin on this population. The modal length holds 0.502 of unique
  complete CDS, below the 0.90 floor, and `conf/virus/flu.yaml` pins no length
  for it.
* No pin is available, so the pinning cost does not apply.
* The 678-nt internal deletion is the first internal gap in this protein
  carried
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
* Columns 74, 112 and 132 each hold a protein symbol in exactly one protein;
  the
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

## NS1 on Human-H3N2-2024 (Segment 8, Task 1 population)

The first alignment run, on 2026-11-05, where the method was worked out. It is
on the population Task 1 uses, so it is the one section not measured on the
Task 2 population.

### 1. Inputs

* **Records.** Of 5,346 filtered isolates, 5,337 (99.8%) had a complete CDS and
  9 were dropped as incomplete.
* **Sequences.** 1,132 unique CDS and 560 unique proteins, across 3 CDS lengths
  in one host–subtype–year group.
* **Unknown residues.** 48 isolates and 48 unique proteins contain an `X`, all
  48 at 693 nt.
* **Pin.** NS1 clears the floor here but has no configured pin. 1,122 of the
  1,132 unique complete CDS are 693 nt, a share of 0.991, above the 0.90
  `check_cds_length` needs. `conf/virus/flu.yaml` pins no length for NS1, so
  there is no pin for `check_cds_length` to check.
* **Alignment.** MAFFT used FFT-NS-2 and produced `230` columns.
  - Removing `-` from the aligned proteins recovered the exact input protein
    for all 560.
  - A 693-nt CDS is 230 amino acids, so the alignment adds no columns.
  - This run asked for `--auto` rather than the flags in Method.

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

* All 551 proteins from 693-nt CDS have no gaps, and each shorter protein has
  one
  run of trailing gaps.
* No protein has an `X` in columns 218–230, where one or both shorter forms
  have
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

## Limitations

* All 8 modelled proteins are aligned on the Task 2 population and written up
  here. Six of the eight ran before `mafft_seconds` was added, so only NA and
  M1 report a runtime.
* No codon features and no models were produced from any of these alignments.
  The
  mapping from alignment columns to codon features is proposed in
  `docs/plans/2026-09-14_codon_site_features_plan.md` and is not implemented.
* Every run used the same MAFFT strategy, FFT-NS-2. The Task 2 runs asked
  for it with `--retree 2 --maxiterate 0` and the Human-H3N2-2024 run let
  `--auto` choose it. We did not test a second strategy. Agreement between two
  strategies would not establish that a placement is correct.
* The one-residue columns and the uncommon gap patterns have not been reviewed
  sequence by sequence.
