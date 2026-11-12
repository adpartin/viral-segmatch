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

In all eight, some form fills the last column of the alignment, so a codon
feature table needs one column more than the alignment has: PB2 767, PB1 777,
PA 745, HA 613, NP 499, NA 544, M1 253 and NS1 241.

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

PB2 sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Isolates and CDS records.** Of 60,751 filtered isolates, 60,474 (99.5%)
  have a complete PB2 CDS record; 277 isolates were excluded because their PB2
  CDS was incomplete. Thus, counts of complete records in this section are
  counts of isolates with a complete PB2 CDS, before sequences are
  deduplicated.
* **Nucleotide sequences.** The 60,474 complete PB2 CDS records contain 36,939
  unique CDS sequences across 18 CDS lengths and 147 host–subtype–year groups.
* **Protein sequences.** Those CDS sequences translate to 15,290 unique PB2
  protein sequences.
* **Unknown amino acids.** The translated proteins for 1,296 isolates,
  representing 1,197 unique proteins, contain an `X`. All 1,296 isolates have
  a 2280-nt CDS.
* **Modal CDS length.** PB2 has a usable configured length of 2280 nt.
  This length includes 759 sense codons and 1 terminal stop codon. It occurs
  in 36,879 of 36,939 unique complete CDS sequences (0.998), exceeding the
  `check_cds_length` requirement of 0.90. The configured value in
  `conf/virus/flu.yaml` agrees with the observed mode.
* **MAFFT input.** Before alignment, the terminal `*` was removed from each
  protein. MAFFT therefore aligned 15,290 unique stop-free aa sequences. A
  protein from a 2280-nt CDS contributes 759 aa residues to this alignment.
* **Alignment result.** MAFFT used FFT-NS-2 and produced 766 aa alignment
  columns.
  - Removing alignment gaps (`-`) from every aligned sequence recovered its
    exact stop-free protein input.
  - Each protein from a 2280-nt CDS contains 759 aa residues and therefore has
    gaps in 7 of the 766 alignment columns.
  - The proteins from 2280-nt CDS records have 2 observed gap patterns.

### 2. Lengths and gaps

The CDS lengths in this table identify the nt sequences from which the proteins
were translated. All gap counts refer to the resulting aa alignment, not to an
nt alignment.

| Source CDS length | Isolates | Unique CDS | Unique proteins | Leading AA-gap columns | Internal AA-gap columns | Trailing AA-gap columns |
|---|---:|---:|---:|---|---|---|
| 2280 nt | 60,413 | 36,879 | 15,230 | None | 7 | None |
| 17 other lengths | 61 | 60 | 60 | 0 to 10 | 6 to 31 | 0 to 10 |

The 15,230 unique proteins translated from 2280-nt CDS sequences have 2 observed
aa gap patterns:

| Source CDS length | AA gap columns | AA residues | Unique proteins |
|---|---|---:|---:|
| 2280 nt | 42–45, 348, 575, 678 | 759 | 15,229 |
| 2280 nt | 42–45, 348, 582, 678 | 759 | 1 |

Thus, each of these proteins contributes 759 aa residues to the 766-column
alignment and has an aa alignment gap in 7 columns. These gaps are alignment
placeholders; they are not gaps inserted into the source CDS sequence.

### 3. Projection to codons and the terminal stop

For codon featurization, each aa alignment column becomes a projected codon
column, and each aa residue is mapped back to the codon that encoded it. An aa
alignment gap has no source codon, so its projected codon column holds no codon
and takes one of two proposed states. It is `GAP` when the projected column
comes before that record's terminal stop codon, and `AFTER_STOP` when it comes
after. Both comparisons are made in the projected codon frame, because the
terminal stop codon has no position in the aa alignment.

The terminal `*` was excluded from the MAFFT input, so none of the 766 aa
alignment columns corresponds to a terminal stop codon. Proteins from 2280-nt
CDS sequences have their last aa residue in alignment column 766, so their
terminal stop codon would be placed in projected codon column 767.

The PB2 protein alignment has 766 aa columns, whereas the proposed PB2 codon
feature representation requires 767 columns. This projection has not been
implemented.

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

The off-mode lengths are spread across hosts, subtypes and years, with no group
holding more than 30 of the 61 isolates.

### 5. Findings

* PB2 has a usable pin at 2280 nt. Pinning would keep 60,413 of the 60,474
  isolates with a complete PB2 CDS and drop 61 isolates at 17 other CDS
  lengths.
* The required feature width depends on which proteins are retained. An aa
  alignment column that becomes all-gap after filtering can be removed. The
  projected codon representation adds one column for the terminal stop codon.

| Proteins retained | Unique proteins | All-gap aa columns removed | AA alignment columns | Projected codon columns |
|---|---:|---|---:|---:|
| All 18 source CDS lengths | 15,290 | None | 766 | 767 |
| 2280-nt source CDS only | 15,230 | 42–45, 348, 678 | 760 | 761 |
| Dominant 2280-nt gap pattern only | 15,229 | 42–45, 348, 575, 678 | 759 | 760 |

* The proteins from 2280-nt CDS records retain 2 gap patterns. One protein has
  its aa alignment gap at column 582 instead of 575 and therefore accounts for
  the additional aa column in the 760-column representation. Excluding it
  would require a separate criterion beyond the CDS-length pin.
* The protein-to-codon projection has not been implemented. It must represent
  aa alignment gaps for any retained non-dominant patterns.
* Still open before codon features are built: whether to keep the 61 isolates
  outside the modal CDS length.

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

```text
Length   Empty columns                              Residues   Proteins   Stop column
2274 nt  77-79, 389, 718-728, 773-776                    757     12,058           773
2274 nt  62-63, 68, 389, 718-728, 773-776                757          1           773
2274 nt  77-79, 94, 389, 718-728, 774-776                757          1           774
2277 nt  77-79, 389, 718-728, 774-776                    758      1,109           774
2277 nt  77-79, 718-728, 773-776                         758          1           773
```

The 2274-nt proteins have 757 residues, so 19 of the 776 columns are empty.
The 2274-nt proteins have 3 gap patterns.

The 2277-nt proteins have 758 residues, so 18 of the 776 columns are empty.
The 2277-nt proteins have 2 gap patterns.

The two main lengths are the same form, one codon apart at the C-terminus.
Their biggest patterns have identical internal gaps and differ only in the
trailing run, by the single column 773:

```text
Column:     772 773 774 775 776
2274 nt:    res  -   -   -   -     last symbol 772, stop in 773
2277 nt:    res res  -   -   -     last symbol 773, stop in 774
```

A codon feature table for PB1 needs 777 columns, not 776.
One 2217-nt protein and one 2232-nt protein fill column 776, so their stop codon
falls one column past the alignment.
This mapping is not implemented.

### 3. By host, subtype and year

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

The 2277-nt share of the two main lengths stays between 0.2% and 1.9% from 2010
to 2020. It then rises each year, to 34.0% in 2024 and 33.9% in 2025.

### 4. Findings

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

PA sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Isolates and CDS records.** Of 60,751 filtered isolates, 60,396 (99.4%)
  have a complete PA CDS record; 355 isolates were excluded because their PA
  CDS was incomplete. Thus, counts of complete records in this section are
  counts of isolates with a complete PA CDS, before sequences are
  deduplicated.
* **Nucleotide sequences.** The 60,396 complete PA CDS records contain 35,659
  unique CDS sequences across 14 CDS lengths and 147 host–subtype–year groups.
* **Protein sequences.** Those CDS sequences translate to 15,939 unique PA
  protein sequences.
* **Unknown amino acids.** The translated proteins for 1,349 isolates,
  representing 1,203 unique proteins, contain an `X`. 1,348 of those isolates
  have a 2151-nt CDS.
* **Modal CDS length.** PA has a usable configured length of 2151 nt. This
  length includes 716 sense codons and 1 terminal stop codon. It occurs in
  35,609 of 35,659 unique complete CDS sequences (0.999), exceeding the
  `check_cds_length` requirement of 0.90. The configured value in
  `conf/virus/flu.yaml` agrees with the observed mode.
* **MAFFT input.** Before alignment, the terminal `*` was removed from each
  protein. MAFFT therefore aligned 15,939 unique stop-free aa sequences. A
  protein from a 2151-nt CDS contributes 716 aa residues to this alignment.
* **Alignment result.** MAFFT used FFT-NS-2 and produced 744 aa alignment
  columns.
  - Removing alignment gaps (`-`) from every aligned sequence recovered its
    exact stop-free protein input.
  - Each protein from a 2151-nt CDS contains 716 aa residues and therefore has
    gaps in 28 of the 744 alignment columns.
  - The proteins from 2151-nt CDS records have 2 observed gap patterns.

### 2. Lengths and gaps

The CDS lengths in this table identify the nt sequences from which the proteins
were translated. All gap counts refer to the resulting aa alignment, not to an
nt alignment.

| Source CDS length | Isolates | Unique CDS | Unique proteins | Leading AA-gap columns | Internal AA-gap columns | Trailing AA-gap columns |
|---|---:|---:|---:|---|---|---|
| 2151 nt | 60,332 | 35,609 | 15,897 | None | 26 | 2 |
| 13 other lengths | 64 | 50 | 42 | 0 to 20 | 22 to 49 | 0 to 25 |

The 2151-nt CDS length holds 2 aa gap patterns. The table also lists the two at
2157 nt, the most frequent of the other lengths at 39 isolates, because that is
the length whose proteins reach the last alignment column.

| Source CDS length | AA gap columns | AA residues | Unique proteins | Isolates | H3N2 | H1N1 | H5N1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2151 nt | 48–49, 77, 82–83, 94–98, 624, 641–650, 660–663, 703, 743–744 | 716 | 15,896 | 60,331 | 32,792 | 23,775 | 3,764 |
| 2151 nt | 48–49, 77, 82–88, 624, 641–650, 660–663, 703, 743–744 | 716 | 1 | 1 | 1 | 0 | 0 |
| 2157 nt | 48–49, 77, 82–83, 94–98, 624, 641–650, 660–663, 703 | 718 | 16 | 38 | 38 | 0 | 0 |
| 2157 nt | 77, 82–83, 94–98, 624, 641–650, 660–663, 703, 743–744 | 718 | 1 | 1 | 1 | 0 | 0 |

Thus, each protein translated from a 2151-nt CDS contributes 716 aa residues to
the 744-column alignment and has an aa alignment gap in 28 columns. These gaps
are alignment placeholders; they are not gaps inserted into the source CDS
sequence.

PA's gap patterns do not split by subtype. The first pattern carries all three
subtypes and all four hosts, and holds 60,331 of the 60,396 isolates with a
complete PA CDS. Only the rare 2157-nt proteins are subtype-specific, and all
39 of their isolates are Human-H3N2.

PA differs from PB2 in where those gaps fall. Two of the 28 are trailing gaps,
in columns 743 and 744, so a protein translated from a 2151-nt CDS does not
reach the last alignment column.

Both 2157-nt patterns are the modal pattern with one gap run removed. A 2157-nt
CDS has 718 sense codons, two more than the modal length. In the pattern held by
16 proteins the two extra codons sit at the C-terminus, which removes the
743–744 trailing gap and is how those proteins reach the last alignment column.
In the pattern held by 1 protein they sit at the N-terminus instead, filling
columns 48 and 49.

### 3. Projection to codons and the terminal stop

For codon featurization, each aa alignment column becomes a projected codon
column, and each aa residue is mapped back to the codon that encoded it. An aa
alignment gap has no source codon, so its projected codon column holds no codon
and takes one of two proposed states. It is `GAP` when the projected column
comes before that record's terminal stop codon, and `AFTER_STOP` when it comes
after. Both comparisons are made in the projected codon frame, because the
terminal stop codon has no position in the aa alignment.

The terminal `*` was excluded from the MAFFT input, so none of the 744 aa
alignment columns corresponds to a terminal stop codon. Proteins from 2151-nt
CDS sequences have their last aa residue in alignment column 742, so their
terminal stop codon would be placed in projected codon column 743.

Alignment columns 743 and 744 hold an aa residue in 16 unique proteins, all
translated from 2157-nt CDS sequences. Those proteins have their last aa residue
in alignment column 744, so their terminal stop codon would be placed in
projected codon column 745.

```text
Projected codon column:  742  743   744         745
2151 nt:                 res  stop  AFTER_STOP  AFTER_STOP
2157 nt:                 res  res   res         stop
```

PA therefore needs both proposed codon-level states in its last columns. For a
2151-nt record, projected column 743 holds its terminal stop codon, and columns
744 and 745 fall after it, so they receive `AFTER_STOP` rather than `GAP`.

The PA protein alignment has 744 aa columns, whereas the proposed PA codon
feature representation requires 745 columns when the 2157-nt records are kept.
This projection has not been implemented.

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

The 2157-nt form appears in 7 of the 16 years, with 19 of its 39 isolates in
2021.

### 5. Findings

* PA has a usable pin at 2151 nt. Pinning would keep 60,332 of the 60,396
  isolates with a complete PA CDS and drop 64 isolates at 13 other CDS lengths.
* The required feature width depends on which proteins are retained. An aa
  alignment column that becomes all-gap after filtering can be removed. The
  projected codon representation adds one column for the terminal stop codon.

| Proteins retained | Unique proteins | All-gap aa columns removed | AA alignment columns | Projected codon columns |
|---|---:|---:|---:|---:|
| All 14 source CDS lengths | 15,939 | None | 744 | 745 |
| 2151-nt source CDS only | 15,897 | 23 columns | 721 | 722 |
| Dominant 2151-nt gap pattern only | 15,896 | 28 columns | 716 | 717 |

* The 28 columns removed in the last row are the dominant gap pattern itself,
  listed in part 2. The 23 columns removed in the middle row are those 28
  without columns 94 to 98.
* The proteins from 2151-nt CDS records retain 2 gap patterns. One protein has
  its aa alignment gaps at columns 82 to 88 instead of 82 to 83 and 94 to 98,
  and therefore accounts for the 5 additional aa columns in the 721-column
  representation. Excluding it would require a separate criterion beyond the
  CDS-length pin.
* 25 aa alignment columns hold an aa residue in exactly one protein: 49, 77,
  82–83, 94–98, 624, 641–650, 660–663 and 703. 20 of them hold residues from 5
  off-mode proteins and become all-gap once the 2151-nt pin is applied. Columns
  94–98 hold residues from the exceptional 2151-nt protein and remain after
  pinning. Cutting all 25 would take the projected codon representation from 745
  columns to 720, but it would delete residues from all 6 of those proteins.
  That is a site-exclusion decision, not a consequence of the CDS-length pin.
* The protein-to-codon projection has not been implemented. It must represent
  aa alignment gaps for any retained non-dominant patterns. For retained
  proteins with different C-terminal lengths, it must also distinguish `GAP`,
  the terminal stop codon and `AFTER_STOP`.
* Still open before codon features are built: whether to keep the 64 isolates
  outside the modal CDS length.

## HA (Segment 4)

HA sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Isolates and CDS records.** Of 60,751 filtered isolates, 60,282 (99.2%)
  have a complete HA CDS record; 469 isolates were excluded because their HA
  CDS was incomplete. Thus, counts of complete records in this section are
  counts of isolates with a complete HA CDS, before sequences are deduplicated.
* **Nucleotide sequences.** The 60,282 complete HA CDS records contain 35,509
  unique CDS sequences across 10 CDS lengths and 147 host–subtype–year groups.
* **Protein sequences.** Those CDS sequences translate to 19,841 unique HA
  protein sequences.
* **Unknown amino acids.** The translated proteins for 1,797 isolates,
  representing 1,613 unique proteins, contain an `X`. 1,756 of those isolates
  have a 1701-nt CDS.
* **Modal CDS length.** HA has a usable configured length of 1701 nt. This
  length includes 566 sense codons and 1 terminal stop codon. It occurs in
  33,492 of 35,509 unique complete CDS sequences (0.943), exceeding the
  `check_cds_length` requirement of 0.90. The configured value in
  `conf/virus/flu.yaml` agrees with the observed mode.
* **MAFFT input.** Before alignment, the terminal `*` was removed from each
  protein. MAFFT therefore aligned 19,841 unique stop-free aa sequences. A
  protein from a 1701-nt CDS contributes 566 aa residues to this alignment.
* **Alignment result.** MAFFT used FFT-NS-2 and produced 612 aa alignment
  columns.
  - Removing alignment gaps (`-`) from every aligned sequence recovered its
    exact stop-free protein input.
  - Each protein from a 1701-nt CDS contains 566 aa residues and therefore has
    gaps in 46 of the 612 alignment columns.
  - The proteins from 1701-nt CDS records have 6 observed gap patterns. PB2 and
    PA also hold 2 patterns at their modal length, but in both the second is a
    single protein. In HA the second holds 8,326 proteins.

### 2. Lengths and gaps

The CDS lengths in this table identify the nt sequences from which the proteins
were translated. All gap counts refer to the resulting aa alignment, not to an
nt alignment.

| Source CDS length | Isolates | Unique CDS | Unique proteins | Leading AA-gap columns | Internal AA-gap columns | Trailing AA-gap columns |
|---|---:|---:|---:|---|---|---|
| 1695 nt | 244 | 203 | 179 | 4 | 44 | None |
| 1698 nt | 71 | 67 | 62 | 4 | 43 | None |
| 1701 nt | 56,223 | 33,492 | 18,606 | 4 | 42 | None |
| 1704 nt | 3,632 | 1,643 | 901 | 4 | 41 | None |
| 1707 nt | 63 | 56 | 50 | 0 to 4 | 40 to 44 | None |
| 5 other lengths | 49 | 48 | 43 | 4 to 14 | 39 to 48 | None |

The 6 gap patterns at 1701 nt are not six variants of one form. The two that
hold the most unique proteins each sit almost entirely in one HA subtype, and a
third pattern at 1704 nt sits entirely in a third subtype. Their column lists
run to 151 to 187 characters each, so `gap_pattern_summary.csv` holds them
rather than this table. This section calls the three patterns H3, H1 and H5
after the subtype that dominates each, which is a label for the gap pattern and
not a claim that the pattern and the subtype are the same set.

| Source CDS length | Gap pattern | AA residues | Unique proteins | Isolates | H3N2 | H1N1 | H5N1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1701 nt | H3 | 566 | 10,274 | 32,783 | 32,778 | 3 | 2 |
| 1701 nt | H1 | 566 | 8,326 | 23,428 | 4 | 23,424 | 0 |
| 1701 nt | 4 rare patterns | 566 | 6 | 12 | 0 | 1 | 11 |
| 1704 nt | H5 | 567 | 870 | 3,596 | 0 | 0 | 3,596 |

H3 and H1 each fill 566 aa alignment columns, as their residue count requires,
but not the same 566. They share 552 columns. H3 fills 14 that H1 does not:
8, 10, 12, 21–26, 86–88, 158 and 176. H1 fills 14 that H3 does not: 77–80, 109,
122, 149–152, 161, 178, 302 and 575. Between them they fill 580 columns.

The clearest of those differences is columns 20 to 27, where the H1 form has its
8 consecutive aa alignment gaps and the H3 form has 6 residues:

```text
Alignment column:   19  20  21  22  23  24  25  26  27  28
H3, 1701 nt:       res   -  res res res res res res  -  res
H1, 1701 nt:       res   -   -   -   -   -   -   -   -  res
H5, 1704 nt:       res   -   -   -   -   -   -   -   -  res
```

Taking all three subtype forms together, they cover 59,807 of the 60,282
complete records and fill 583 aa alignment columns. 550 of those columns are
filled by all three forms, 14 by H3 only, 3 by H5 only, 14 by H1 and H5 but not
H3, and 2 by H1 and H3 but not H5. No column is filled by H1 alone.

The five CDS lengths with their own row above have no aa residue in columns 20,
321, 335–337, 375 or 601–603, so they use 603 of the 612 columns. The sixth row
holds the forms that do fill them. Those 9 columns are in the
alignment because rarer forms fill them: 1683-nt proteins fill column 20,
1683-nt and 1686-nt proteins fill 321 and 375, and 1710-nt proteins fill
335–337 and 601–603.

19,837 of the 19,841 proteins have their first aa residue in alignment column 5,
so columns 1 to 4 are leading gaps for nearly every protein. One 1707-nt protein
fills all four, two proteins start in column 6, and one starts in column 15.

### 3. Projection to codons and the terminal stop

For codon featurization, each aa alignment column becomes a projected codon
column, and each aa residue is mapped back to the codon that encoded it. An aa
alignment gap has no source codon, so its projected codon column holds no codon
and takes one of two proposed states. It is `GAP` when the projected column
comes before that record's terminal stop codon, and `AFTER_STOP` when it comes
after. Both comparisons are made in the projected codon frame, because the
terminal stop codon has no position in the aa alignment.

The terminal `*` was excluded from the MAFFT input, so none of the 612 aa
alignment columns corresponds to a terminal stop codon. Every one of the 19,841
proteins has its last aa residue in alignment column 612, and none has a
trailing aa alignment gap. So every record's terminal stop codon would be placed
in projected codon column 613, and no record needs `AFTER_STOP` at its
C-terminus.

HA instead makes the leading gaps the common case. Projected codon columns 1 to
4 hold no codon for 19,840 of the 19,841 unique proteins, because one 1707-nt
protein is the only one with an aa residue there. `GAP` in those columns
therefore applies to nearly every protein rather than to an exception.

The HA protein alignment has 612 aa columns, whereas the proposed HA codon
feature representation requires 613 columns. This projection has not been
implemented.

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

These counts are by CDS length, not by gap pattern, so the 1704-nt column holds
3,632 isolates against the 3,596 of the H5 pattern. The 1704-nt records are
almost all H5N1, 3,614 of 3,632. The 1695-nt records are almost all Swine-H1N1,
217 of 244.

The 1704-nt records were 12 to 115 isolates a year from 2010 to 2021. They then
went to 1,006 in 2022, 283 in 2023, 1,190 in 2024 and 713 in 2025, so the rise
is large but not steady. The 1695-nt records have 139 of their 244 isolates in
2023.

### 5. Findings

* HA has a usable pin at 1701 nt, but the pin does not reduce HA to one form.
  Pinning would keep 56,223 of the 60,282 isolates with a complete HA CDS and
  drop 4,059. It would also drop the H5 form, keeping 13 of the 3,754 H5N1
  isolates.
* The required feature width depends on which proteins are retained. An aa
  alignment column that becomes all-gap after filtering can be removed. The
  projected codon representation adds one column for the terminal stop codon.

| Proteins retained | Unique proteins | All-gap aa columns removed | AA alignment columns | Projected codon columns |
|---|---:|---:|---:|---:|
| All 10 source CDS lengths | 19,841 | None | 612 | 613 |
| 1701-nt source CDS only | 18,606 | 24 columns | 588 | 589 |
| The H3, H1 and H5 forms | 19,470 | 29 columns | 583 | 584 |
| Dominant 1701-nt gap pattern only, which is H3 | 10,274 | 46 columns | 566 | 567 |

* Selecting by form beats selecting by CDS length. The three subtype forms keep
  19,470 unique proteins over 59,807 isolates in 583 aa columns, while the
  1701-nt pin keeps 18,606 proteins over 56,223 isolates in 588 aa columns.
  That is 3,584 more isolates in 5 fewer columns.
* The last row is not a usable option. The dominant gap pattern is the H3 form
  alone, so reaching 566 columns means dropping H1 and H5.
* 14 aa alignment columns hold an aa residue in exactly one protein: 1–4,
  191–192, 198, 381, 383, 601–603 and 608–609. They belong to 7 proteins, one
  at 1701 nt, one at 1704 nt, one at 1710 nt and four at 1707 nt. Columns 1 to 4
  sit before every other protein's first aa residue.
* The longest internal run of aa alignment gaps is columns 20 to 27, 8 columns.
  It is not an outlier: 9,541 unique proteins over 27,471 isolates have it,
  including the whole H1 form at 1701 nt and the H5 form at 1704 nt.
* The protein-to-codon projection has not been implemented. For HA it must
  represent `GAP` for the leading columns of nearly every record, and for the
  columns where the retained subtype forms disagree.
* Still open before codon features are built: whether one set of columns can
  serve H1, H3 and H5 given that they disagree in 33 of the 583 columns they
  use, and whether to select records by subtype form rather than by CDS length.

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

```text
Length   Empty columns                              Residues   Proteins
1461 nt  1-12                                            486          1
1497 nt  None                                            498      7,249
```

The 1461-nt proteins have 486 residues, so 12 of the 498 columns are empty.
The 1461-nt proteins have 1 gap pattern.

The 1497-nt proteins have 498 residues, so no column is empty.
The 1497-nt proteins have 1 gap pattern.

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

Both of those stop codons fall inside the 543 columns. Five rare proteins do
not: one at 1353 nt, one at 1413 nt and three at 1422 nt fill column 543, so
their stop codon falls one column past the alignment. A codon feature table for
NA needs 544 columns, not 543. This mapping is not implemented.

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

```text
Length   Empty columns                              Residues   Proteins
726 nt   31-41                                           241          1
759 nt   None                                            252      1,947
```

The 726-nt proteins have 241 residues, so 11 of the 252 columns are empty.
The 726-nt proteins have 1 gap pattern.

The 759-nt proteins have 252 residues, so no column is empty.
The 759-nt proteins have 1 gap pattern.

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

```text
Length   Empty columns                              Residues   Proteins
654 nt   74, 112, 132, 221-240                           217        207
660 nt   74, 112, 132, 223-240                           219      4,601
672 nt   74, 112, 132, 227-240                           223        144
678 nt   74, 81-85, 112, 132, 234-240                    225        205
693 nt   74, 112, 132, 234-240                           230      4,926
```

The 654-nt proteins have 217 residues, so 23 of the 240 columns are empty.
The 654-nt proteins have 1 gap pattern.

The 660-nt proteins have 219 residues, so 21 of the 240 columns are empty.
The 660-nt proteins have 1 gap pattern.

The 672-nt proteins have 223 residues, so 17 of the 240 columns are empty.
The 672-nt proteins have 1 gap pattern.

The 678-nt proteins have 225 residues, so 15 of the 240 columns are empty.
The 678-nt proteins have 1 gap pattern.

The 693-nt proteins have 230 residues, so 10 of the 240 columns are empty.
The 693-nt proteins have 1 gap pattern.

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
