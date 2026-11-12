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
gaps, 3. Projection to codons and the terminal stop, 4. By host, subtype and
year, 5. Findings. If a
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

Part 3 is the same proposal in every section. For codon featurization, each aa
alignment column becomes a projected codon column, and each aa residue is mapped
back to the codon that encoded it. An aa alignment gap has no source codon, so
its projected codon column holds no codon and takes one of two proposed states.
It is `GAP` when the projected column comes before that record's terminal stop
codon, and `AFTER_STOP` when it comes after. Both comparisons are made in the
projected codon frame, because the terminal stop codon has no position in the aa
alignment. Each section's part 3 gives only what is specific to that protein:
which alignment column each form's last aa residue sits in, and therefore which
projected codon column would hold its stop codon.

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
4. Align the unique proteins with MAFFT `--retree 2 --maxiterate 0 --reorder`,
   recording the version, command, strategy, runtime and input hashes.
   `--reorder` writes the rows in MAFFT's guide-tree order instead of input
   order.
5. Check that removing `-` from each aligned protein recovers the exact input
   protein.
6. Record every gap position, and name each gap pattern by rank. `gap_pat001`
   is the pattern held by the most unique proteins, and ties go to the pattern
   with more isolates.

Every run used MAFFT v7.526, so the sections below give only the strategy and
the column count.

The eight Task 2 alignments were first run without `--reorder` and rerun with
it on 2026-11-12. Every aligned row, every width and every summary table came
out identical. Only the row order changed.

Step 6 also writes two copies of the alignment, with the same rows in the same
order and a label on each. The label fields are the protein hash, the protein,
its gap pattern, its source CDS length, its aa residue count and the number of
isolates carrying it. `aligned_proteins.fasta` stays keyed by `prot_hash` alone.

* `aligned_proteins_labelled.fasta` starts each label with the protein hash, for
  example `5315b8760b3edab2dcf60f53f0d8d4c8|PB2|gap_pat001|2280nt|759aa|5046iso`.
* `aligned_proteins_jalview.fasta` starts each label with the protein and ends
  it with the protein hash, for example
  `PB2|gap_pat001|2280nt|759aa|5046iso|5315b8760b3edab2dcf60f53f0d8d4c8`. Sorting
  these labels as text puts each gap pattern in one block, in rank order.

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

| Segment ID | Protein | Isolates | Unique CDS | Unique proteins | CDS lengths | AA columns | `frac at mode` | Longest internal AA-gap run | One-residue columns |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | PB2 | 60,474 | 36,939 | 15,290 | 18 | 766 | 2280 nt, 36,879 of 36,939 | columns 692–715, 24 columns, 1 protein, 1 isolate | 42–45, 575, 678 |
| 2 | PB1 | 56,093 | 34,642 | 13,220 | 18 | 776 | 2274 nt, 30,824 of 34,642 | columns 736–764, 29 columns, 2 proteins, 2 isolates | 77–79, 389, 718–719 |
| 3 | PA | 60,396 | 35,659 | 15,939 | 14 | 744 | 2151 nt, 35,609 of 35,659 | columns 80–107, 28 columns, 1 protein, 1 isolate | 25 columns, listed in part 5 |
| 4 | HA | 60,282 | 35,509 | 19,841 | 10 | 612 | 1701 nt, 33,492 of 35,509 | columns 20–27, 8 columns, 9,541 proteins, 27,471 isolates | 14 columns, listed in part 5 |
| 5 | NP | 60,040 | 27,087 | 7,250 | 2 | 498 | 1497 nt, 27,086 of 27,087 | None | None |
| 6 | NA | 59,856 | 31,157 | 17,267 | 16 | 543 | 1410 nt, 30,630 of 31,157 | columns 68–94, 27 columns, 1 protein, 1 isolate | 16 columns, listed in part 5 |
| 7 | M1 | 60,566 | 14,573 | 1,948 | 2 | 252 | 759 nt, 14,572 of 14,573 | columns 31–41, 11 columns, 1 protein, 1 isolate | None |
| 8 | NS1 | 60,654 | 18,107 | 10,130 | 16 | 240 | 693 nt, 9,097 of 18,107 | columns 81–85, 5 columns, 206 proteins, 342 isolates | 74, 112, 132 |

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

"Longest internal AA-gap run" gives the longest run of neighbouring internal aa
alignment gap columns any protein has, as column numbers, followed by how long
the run is and how many unique proteins and isolates carry it.
`columns 81–85, 5 columns` means five aa alignment gaps, one in each of the
columns numbered 81 to 85. The carrier count is not always small: HA's run is
carried by 9,541 of its 19,841 unique proteins, because it is where the H1 form
differs from the H3 form. The table gives only the longest run.

Each section's part 2 gives the gap columns of its main forms, not of every
pattern: only NP and M1 have few enough to list in full. HA and NA give their
patterns by name rather than by column list, because those lists run to 151 to
193 characters. `gap_pattern_summary.csv` holds every pattern for all eight.

A **one-residue column** is an aa alignment column where exactly one protein has
an aa residue and every other protein has a gap.

Six of the eight proteins have a usable pin on this population: PB2, PA, HA,
NP, NA and M1. The two without one are PB1 at 0.890 and NS1 at 0.502, which are
also the two proteins `conf/virus/flu.yaml` pins no length for.
HA is the extreme: its 1701-nt pin keeps 13 of 3,754 complete H5N1 records
because the main H5 form sits at 1704 nt.

Four of the six pinned proteins hold more than one gap pattern at the pinned
length: PB2 2, PA 2, HA 6 and NA 9. In PB2 and PA the second pattern is a single
protein. In HA and NA the two biggest are both large and split by subtype:
HA's 1701 nt holds 10,274 H3 proteins and 8,326 H1, with H5 at 1704 nt, and
NA's 1410 nt holds 8,510 N2 proteins and 8,305 N1. Those two sections give the
measured column overlap between the forms.

In all eight, some form fills the last column of the alignment, so a codon
feature table needs one column more than the alignment has: PB2 767, PB1 777,
PA 745, HA 613, NP 499, NA 544, M1 253 and NS1 241. Those figures assume every
CDS length is retained. Each section's part 5 gives the width for narrower
retention sets, which is lower for six of the eight.

NS1 has a usable pin at 0.991 on the Task 1 population, 1,122 of 1,132 unique
complete CDS at 693 nt, reported in the last section. Extending the population
to the one Task 2 uses drops `frac at mode` at 693 nt to 0.502, so NS1 has no
pin there.


## Artifacts

Every Task 2 run wrote to
`results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/<protein>/`,
one directory per protein. The Task 1 pilot wrote to
`results/flu/July_2025/cds_alignment_pilot/Human-H3N2-2024/NS1/`.

`align_proteins.py` wrote the selection, alignment, validation and gap
locations. `summarize_protein_alignment.py` wrote `group_length_summary.csv`,
`gap_pattern_summary.csv` and `column_occupancy.csv`. The manifests record file
hashes and code state.

The eight Task 2 manifests record a MAFFT runtime. The eight runs ran at the
same time, one thread each: PB2 438.7 seconds, PB1 337.3, PA 446.0, HA 563.2,
NP 69.9, NA 351.9, M1 4.1 and NS1 60.5. The Task 1 pilot ran before
`mafft_seconds` was added, so it has no runtime.

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

PB1 sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Isolates and CDS records.** Of 60,751 filtered isolates, 56,093 (92.3%)
  have a complete PB1 CDS record; 4,658 isolates were excluded because their
  PB1 CDS was incomplete. That is 7.7% of the population, the largest
  completeness loss of the eight proteins. Thus, counts of complete records in
  this section are counts of isolates with a complete PB1 CDS, before sequences
  are deduplicated.
* **Nucleotide sequences.** The 56,093 complete PB1 CDS records contain 34,642
  unique CDS sequences across 18 CDS lengths and 147 host–subtype–year groups.
* **Protein sequences.** Those CDS sequences translate to 13,220 unique PB1
  protein sequences.
* **Unknown amino acids.** The translated proteins for 1,325 isolates,
  representing 1,158 unique proteins, contain an `X`. 1,136 of those isolates
  have a 2274-nt CDS.
* **No usable pin.** PB1 is one of two proteins with no configured CDS length,
  and the measurement shows why. Its modal length, 2274 nt, occurs in 30,824 of
  34,642 unique complete CDS sequences, a share of 0.890. That is below the
  `check_cds_length` requirement of 0.90, so the pin would be rejected even if
  one were configured. The `cds_length` block in `conf/virus/flu.yaml` lists
  PB2, PA, NP, M1, HA and NA, and no length for PB1.
* **MAFFT input.** Before alignment, the terminal `*` was removed from each
  protein. MAFFT therefore aligned 13,220 unique stop-free aa sequences. A
  protein from a 2274-nt CDS contributes 757 aa residues to this alignment, and
  one from a 2277-nt CDS contributes 758.
* **Alignment result.** MAFFT used FFT-NS-2 and produced 776 aa alignment
  columns.
  - Removing alignment gaps (`-`) from every aligned sequence recovered its
    exact stop-free protein input.
  - Each protein from a 2274-nt CDS contains 757 aa residues and therefore has
    gaps in 19 of the 776 alignment columns. A 2277-nt protein has 758 residues
    and gaps in 18.
  - The proteins from 2274-nt CDS records have 3 observed gap patterns, and
    those from 2277-nt records have 2.

### 2. Lengths and gaps

The CDS lengths in this table identify the nt sequences from which the proteins
were translated. All gap counts refer to the resulting aa alignment, not to an
nt alignment.

| Source CDS length | Isolates | Unique CDS | Unique proteins | Leading AA-gap columns | Internal AA-gap columns | Trailing AA-gap columns |
|---|---:|---:|---:|---|---|---|
| 2274 nt | 49,948 | 30,824 | 12,060 | None | 15 to 16 | 3 to 4 |
| 2277 nt | 6,095 | 3,768 | 1,110 | None | 14 to 15 | 3 to 4 |
| 16 other lengths | 50 | 50 | 50 | None | 15 to 41 | 0 to 11 |

The two most frequent source CDS lengths are 2274 nt and 2277 nt, and this
section calls them the two main lengths. Within a length, the biggest pattern
means the one held by the most unique proteins. The three subtype columns count
unique proteins, not isolates. In the two multi-protein rows they sum to more
than the unique protein count, by 95 for 2274 nt and by 1 for 2277 nt, because
91 proteins at 2274 nt and 1 at 2277 nt occur in isolates of more than one
subtype, and 4 of those 91 occur in all three.

| Source CDS length | AA gap columns | AA residues | Unique proteins | Isolates | H3N2 proteins | H1N1 proteins | H5N1 proteins |
|---|---|---:|---:|---:|---:|---:|---:|
| 2274 nt | 77–79, 389, 718–728, 773–776 | 757 | 12,058 | 49,946 | 4,897 | 6,200 | 1,056 |
| 2274 nt | 62–63, 68, 389, 718–728, 773–776 | 757 | 1 | 1 | 0 | 1 | 0 |
| 2274 nt | 77–79, 94, 389, 718–728, 774–776 | 757 | 1 | 1 | 1 | 0 | 0 |
| 2277 nt | 77–79, 389, 718–728, 774–776 | 758 | 1,109 | 6,094 | 1,065 | 43 | 2 |
| 2277 nt | 77–79, 718–728, 773–776 | 758 | 1 | 1 | 0 | 0 | 1 |

The two biggest patterns, one per length, have identical internal gaps at
columns 77–79, 389 and 718–728. They differ only in the trailing run, by the
single column 773:

```text
Alignment column:  770 771 772 773 774 775 776
2274 nt:           res res res  -   -   -   -
2277 nt:           res res res res  -   -   -
```

The two lengths are three nucleotides apart, and 3,768 of the 34,642 unique
complete CDS sequences are at 2277 nt, a share of 0.109. That group is what
holds the modal share below the 0.90 floor: the 16 rare lengths together account
for 50 unique CDS.

The identical gap structure does not make the proteins the same. Deleting
alignment column 773 from each of the 1,109 proteins in the biggest 2277-nt
pattern reproduces an existing aligned 2274-nt protein for 52 of them. The other
1,057 differ at other columns as well, so the two lengths hold related but
distinct sequences.

The claim of identical internal gaps covers the two biggest patterns, not all
five. Columns 62, 63, 68 and 94 are each gapped in exactly one 2274-nt protein
and in no 2277-nt protein: 62, 63 and 68 in one of the two single-protein
2274-nt patterns, and 94 in the other.

The 2274-nt length holds 6,201 distinct H1N1 proteins, 4,898 H3N2 and 1,056
H5N1, so it is not one subtype's length. The 2277-nt length is different:
1,065 of the 1,109 proteins in its biggest pattern are H3N2, and 5,984 of the
length's 6,095 isolates are Human-H3N2.

### 3. Projection to codons and the terminal stop

The terminal `*` was excluded from the MAFFT input, so none of the 776 aa
alignment columns corresponds to a terminal stop codon. PB1's proteins do not
all end in the same alignment column. Their last aa residue falls in one of five
columns, so their stop codons would land in five different projected codon
columns:

| Last aa residue column | Projected stop column | Unique proteins | Isolates | Source CDS lengths |
|---:|---:|---:|---:|---|
| 765 | 766 | 1 | 1 | 2253 nt |
| 770 | 771 | 1 | 1 | 2268 nt |
| 772 | 773 | 12,105 | 49,993 | 16 lengths, including 2274 nt and 2277 nt |
| 773 | 774 | 1,111 | 6,096 | 2268 nt, 2274 nt, 2277 nt |
| 776 | 777 | 2 | 2 | 2217 nt, 2232 nt |

The three cases that carry almost every record:

```text
Projected codon column:        772 773 774 775 776 777
2274 nt, biggest pattern:      res  *   -   -   -   -
2277 nt, biggest pattern:      res res  *   -   -   -
2217 nt and 2232 nt:           res res res res res  *
```

`*` marks the projected column holding that protein's terminal stop codon, and
`-` a column after it, which receives `AFTER_STOP` rather than `GAP`. The two
proteins in the last row, one from a 2217-nt CDS and one from a 2232-nt CDS, are
the only two with no trailing aa alignment gap, so they fill alignment column
776 and their stop codon falls one column past the alignment.

A codon feature table for PB1 therefore needs 777 columns, not 776. The other
13,218 of the 13,220 unique proteins need `AFTER_STOP` for their last columns, a
run of 3, 4, 6 or 11 columns depending on the protein. This projection has not
been implemented.

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

The 2277-nt records are concentrated in Human-H3N2: 5,984 of 6,095, which is
23.6% of that group's 25,390 complete PB1 records.

The 2277-nt share of the isolates at the two main lengths is still moving:

```text
2010   2011-2019   2020   2021   2022   2023   2024   2025
1.9%   under 0.6%  0.7%   3.1%   4.7%  11.1%  34.0%  33.9%
```

Within Human-H3N2 the change is far sharper. The 2277-nt share of that group's
isolates at the two main lengths was 6.9% in 2022, 47.5% in 2023, 99.6% in 2024
and 99.5% in 2025. By 2024 almost every Human-H3N2 record is 2277 nt.

### 5. Findings

* PB1 has no usable pin. Its modal length holds 0.890 of unique complete CDS,
  below the 0.90 floor, and `conf/virus/flu.yaml` configures no length for it.
  So the choice is not whether to pin but which lengths to align together.
* The modal share falls below the floor because of one group three nucleotides
  longer, not because PB1 holds two unrelated proteins. The 2277-nt group is
  0.109 of unique complete CDS, the 16 rare lengths together are 50 sequences,
  and the biggest pattern at each main length has identical internal gaps,
  differing only in the trailing run by the single column 773. The two lengths
  are still distinct sequences: dropping column 773 makes only 52 of the 1,109
  biggest-pattern 2277-nt proteins match an existing 2274-nt protein.
* The required feature width depends on which proteins are retained. An aa
  alignment column that becomes all-gap after filtering can be removed. The
  projected codon representation adds one column for the terminal stop codon.

| Proteins retained | Unique proteins | Isolates | All-gap aa columns removed | AA alignment columns | Projected codon columns |
|---|---:|---:|---:|---:|---:|
| All 18 source CDS lengths | 13,220 | 56,093 | None | 776 | 777 |
| 2274-nt and 2277-nt source CDS | 13,170 | 56,043 | 14 columns | 762 | 763 |
| 2274-nt source CDS only | 12,060 | 49,948 | 15 columns | 761 | 762 |
| Biggest 2274-nt gap pattern only | 12,058 | 49,946 | 19 columns | 757 | 758 |

* Keeping both main lengths costs one column against keeping only the modal
  length, 762 against 761, and keeps 1,110 more unique proteins over 6,095 more
  isolates. A modal-length rule would drop the 2277-nt records, which are 34.0%
  of all 2024 isolates at the two main lengths and 99.6% of the Human-H3N2 ones.
  For recent Human-H3N2 the modal length is the minority form.
* 6 aa alignment columns hold an aa residue in exactly one protein: 77–79, 389
  and 718–719. They belong to 3 proteins: 77–79 to a 2274-nt protein, 389 to a
  2277-nt protein, and 718–719 to a 2232-nt protein. Two of those three are at a
  main length, so dropping the 16 rare lengths does not free the columns.
* 563 of the 776 aa alignment columns hold an aa residue in all 13,220 proteins.
* The longest internal run of aa alignment gaps is columns 736 to 764, 29
  columns. Two proteins have it, one from a 2217-nt CDS and one from a 2232-nt
  CDS, over 2 isolates. Those are the same two proteins that fill column 776.
* The protein-to-codon projection has not been implemented. For PB1 it must
  distinguish `GAP` from `AFTER_STOP` for almost every record, because all but
  two proteins have a trailing aa alignment gap.
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
the length whose proteins reach the last alignment column. The three subtype
columns count unique proteins, not isolates. They sum to more than the unique
protein count because 70 of the 2151-nt proteins occur in isolates of more than
one subtype, one of them in all three.

| Source CDS length | AA gap columns | AA residues | Unique proteins | Isolates | H3N2 proteins | H1N1 proteins | H5N1 proteins |
|---|---|---:|---:|---:|---:|---:|---:|
| 2151 nt | 48–49, 77, 82–83, 94–98, 624, 641–650, 660–663, 703, 743–744 | 716 | 15,896 | 60,331 | 7,075 | 7,614 | 1,279 |
| 2151 nt | 48–49, 77, 82–88, 624, 641–650, 660–663, 703, 743–744 | 716 | 1 | 1 | 1 | 0 | 0 |
| 2157 nt | 48–49, 77, 82–83, 94–98, 624, 641–650, 660–663, 703 | 718 | 16 | 38 | 16 | 0 | 0 |
| 2157 nt | 77, 82–83, 94–98, 624, 641–650, 660–663, 703, 743–744 | 718 | 1 | 1 | 1 | 0 | 0 |

Thus, each protein translated from a 2151-nt CDS contributes 716 aa residues to
the 744-column alignment and has an aa alignment gap in 28 columns. These gaps
are alignment placeholders; they are not gaps inserted into the source CDS
sequence.

PA's gap patterns do not split by subtype. The first pattern holds 7,075
distinct H3N2 proteins, 7,614 H1N1 and 1,279 H5N1, so it is shared by thousands
of sequences in every subtype rather than being one subtype's form. It covers
all four hosts and 60,331 of the 60,396 isolates with a complete PA CDS. Only
the rare 2157-nt proteins are subtype-specific, and all 39 of their isolates are
Human-H3N2.

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
not a claim that the pattern and the subtype are the same set. The three subtype
columns count unique proteins, not isolates. They sum to more than the unique
protein count for the first two patterns, because 2 proteins in each occur in
isolates of two subtypes.

| Source CDS length | Gap pattern | AA residues | Unique proteins | Isolates | H3N2 proteins | H1N1 proteins | H5N1 proteins |
|---|---|---:|---:|---:|---:|---:|---:|
| 1701 nt | H3 | 566 | 10,274 | 32,783 | 10,272 | 3 | 1 |
| 1701 nt | H1 | 566 | 8,326 | 23,428 | 4 | 8,324 | 0 |
| 1701 nt | 4 rare patterns | 566 | 6 | 12 | 0 | 1 | 5 |
| 1704 nt | H5 | 567 | 870 | 3,596 | 0 | 0 | 870 |

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

NP sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Isolates and CDS records.** Of 60,751 filtered isolates, 60,040 (98.8%)
  have a complete NP CDS record; 711 isolates were excluded because their NP
  CDS was incomplete. Thus, counts of complete records in this section are
  counts of isolates with a complete NP CDS, before sequences are deduplicated.
* **Nucleotide sequences.** The 60,040 complete NP CDS records contain 27,087
  unique CDS sequences across 2 CDS lengths and 147 host–subtype–year groups.
  Two lengths is the fewest of the eight proteins, tied with M1.
* **Protein sequences.** Those CDS sequences translate to 7,250 unique NP
  protein sequences.
* **Unknown amino acids.** The translated proteins for 541 isolates,
  representing 436 unique proteins, contain an `X`. All 541 of those isolates
  have a 1497-nt CDS.
* **Modal CDS length.** NP has a usable configured length of 1497 nt. This
  length includes 498 sense codons and 1 terminal stop codon. It occurs in
  27,086 of 27,087 unique complete CDS sequences (1.000), exceeding the
  `check_cds_length` requirement of 0.90. The configured value in
  `conf/virus/flu.yaml` agrees with the observed mode.
* **MAFFT input.** Before alignment, the terminal `*` was removed from each
  protein. MAFFT therefore aligned 7,250 unique stop-free aa sequences. A
  protein from a 1497-nt CDS contributes 498 aa residues to this alignment.
* **Alignment result.** MAFFT used FFT-NS-2 and produced 498 aa alignment
  columns.
  - Removing alignment gaps (`-`) from every aligned sequence recovered its
    exact stop-free protein input.
  - Each protein from a 1497-nt CDS contains 498 aa residues and therefore has
    gaps in none of the 498 alignment columns. The alignment is 498 columns wide
    because no input protein is longer than 498 aa residues.
  - The proteins from 1497-nt CDS records have 1 observed gap pattern, which is
    the pattern with no gaps.

### 2. Lengths and gaps

The CDS lengths in this table identify the nt sequences from which the proteins
were translated. All gap counts refer to the resulting aa alignment, not to an
nt alignment.

| Source CDS length | Isolates | Unique CDS | Unique proteins | Leading AA-gap columns | Internal AA-gap columns | Trailing AA-gap columns |
|---|---:|---:|---:|---|---|---|
| 1461 nt | 1 | 1 | 1 | 12 | None | None |
| 1497 nt | 60,039 | 27,086 | 7,249 | None | None | None |

NP holds 2 gap patterns in the whole alignment, one per CDS length, which is the
fewest of the eight proteins alongside M1.

| Source CDS length | AA gap columns | AA residues | Unique proteins | Isolates |
|---|---|---:|---:|---:|
| 1497 nt | None | 498 | 7,249 | 60,039 |
| 1461 nt | 1–12 | 486 | 1 | 1 |

NP and M1 are the only two of the eight proteins whose modal form has no aa
alignment gap at all. Every host and subtype in the population has records at
1497 nt, as part 4 shows.

The 1461-nt protein is 12 codons shorter than a 1497-nt one, and MAFFT placed
all 12 of its aa alignment gaps at the leading end, so it starts at column 13
and ends at column 498 with every other protein. It is not a 1497-nt protein
with 12 residues removed: none of the 7,249 proteins at 1497 nt matches it once
columns 1 to 12 are deleted, and against the most common one it differs at 45
residue positions outside the gap. Two of those differences are visible below:

```text
Alignment column:  1          11         21
1497 nt:           MASQGTKRSY EQMETDGDRQ NATEIR
1461 nt:           ---------- --METGGERQ NATEIR
```

One isolate carries this form, from Swine-H3N2 in 2011. The 1497-nt row above is
the most common 1497-nt protein, which 9,227 isolates carry.

A leading gap here does not mean a missing start codon. The 1461-nt CDS begins
with ATG; its first aa residue sits in column 13 because the other records have
12 aa residues further left.

### 3. Projection to codons and the terminal stop

The terminal `*` was excluded from the MAFFT input, so none of the 498 aa
alignment columns corresponds to a terminal stop codon. Every one of the 7,250
proteins has its last aa residue in alignment column 498, and none has a
trailing aa alignment gap. So every record's terminal stop codon would be placed
in projected codon column 499, and no record needs `AFTER_STOP`. NP, HA and M1
are the three proteins of the eight where that holds.

The only projected columns that need `GAP` are 1 to 12, for the single 1461-nt
record. Those columns also hold an `X` in 10 of the 7,249 other proteins, so a
column that is a leading gap for one record can hold an unresolved aa residue in
another.

A codon feature table for NP therefore needs 499 columns, not 498. This
projection has not been implemented.

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

* NP has a usable pin at 1497 nt. Pinning would keep 60,039 of the 60,040
  isolates with a complete NP CDS and drop 1. It would keep all 3,766 H5N1
  isolates.
* The required feature width depends on which proteins are retained, except for
  NP, where neither choice below changes it. The only gaps in the alignment are
  the 12 leading columns of one record, and every other protein fills them, so
  dropping that record frees no column. Retaining only that record would be the
  one filtering that narrows the width, to 486 columns.

| Proteins retained | Unique proteins | Isolates | All-gap aa columns removed | AA alignment columns | Projected codon columns |
|---|---:|---:|---:|---:|---:|
| Both source CDS lengths | 7,250 | 60,040 | None | 498 | 499 |
| 1497-nt source CDS only | 7,249 | 60,039 | None | 498 | 499 |

* The alignment adds no columns, so the pin costs none. Every one of the 7,249
  proteins at 1497 nt holds an aa residue in all 498 columns, so within that set
  positions correspond by index and no gap is inserted. That is not the same as
  the proteins being identical: there are 7,249 distinct sequences at that
  length, and 436 of them carry an `X`.
* NP has no internal aa alignment gaps at any length. The only gaps anywhere in
  the alignment are the 12 leading columns of the 1461-nt form.
* 486 of the 498 aa alignment columns hold an aa residue in all 7,250 proteins.
  The other 12 are columns 1 to 12, which hold one in 7,249. That share, 0.976,
  is the highest of the eight proteins.
* There are no one-residue columns.
* Still open before codon features are built: whether to keep the single
  1461-nt isolate. Uncommon length alone does not establish an error.

## NA (Segment 6)

NA sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Isolates and CDS records.** Of 60,751 filtered isolates, 59,856 (98.5%)
  have a complete NA CDS record; 895 isolates were excluded because their NA
  CDS was incomplete. Thus, counts of complete records in this section are
  counts of isolates with a complete NA CDS, before sequences are deduplicated.
* **Nucleotide sequences.** The 59,856 complete NA CDS records contain 31,157
  unique CDS sequences across 16 CDS lengths and 146 host–subtype–year groups.
  NA is the only one of the eight proteins with 146 rather than 147.
* **Protein sequences.** Those CDS sequences translate to 17,267 unique NA
  protein sequences.
* **Unknown amino acids.** The translated proteins for 1,637 isolates,
  representing 1,137 unique proteins, contain an `X`. 1,627 of those isolates
  have a 1410-nt CDS.
* **Modal CDS length.** NA has a usable configured length of 1410 nt. This
  length includes 469 sense codons and 1 terminal stop codon. It occurs in
  30,630 of 31,157 unique complete CDS sequences (0.983), exceeding the
  `check_cds_length` requirement of 0.90. The configured value in
  `conf/virus/flu.yaml` agrees with the observed mode.
* **MAFFT input.** Before alignment, the terminal `*` was removed from each
  protein. MAFFT therefore aligned 17,267 unique stop-free aa sequences. A
  protein from a 1410-nt CDS contributes 469 aa residues to this alignment.
* **Alignment result.** MAFFT used FFT-NS-2 and produced 543 aa alignment
  columns.
  - Removing alignment gaps (`-`) from every aligned sequence recovered its
    exact stop-free protein input.
  - Each protein from a 1410-nt CDS contains 469 aa residues and therefore has
    gaps in 74 of the 543 alignment columns. That is the largest gap count of
    the eight proteins' modal lengths.
  - The proteins from 1410-nt CDS records have 9 observed gap patterns, and two
    of them are large: 8,510 proteins and 8,305. PB2, PB1 and PA also hold more
    than one pattern at their modal length, but their second pattern is a single
    protein. NA and HA are the two where the modal length splits in two.

### 2. Lengths and gaps

The CDS lengths in this table identify the nt sequences from which the proteins
were translated. All gap counts refer to the resulting aa alignment, not to an
nt alignment.

| Source CDS length | Isolates | Unique CDS | Unique proteins | Leading AA-gap columns | Internal AA-gap columns | Trailing AA-gap columns |
|---|---:|---:|---:|---|---|---|
| 1350 nt | 505 | 407 | 319 | None | 93 | 1 |
| 1410 nt | 59,154 | 30,630 | 16,843 | None | 66 to 73 | 1 to 8 |
| 1413 nt | 129 | 59 | 55 | None | 70 to 73 | 0 to 3 |
| 13 other lengths | 68 | 61 | 50 | 0 to 14 | 67 to 93 | 0 to 2 |

Every length carries internal aa gaps, including the modal one.

The 9 gap patterns at 1410 nt are not nine variants of one form. The two that
hold the most unique proteins each sit almost entirely in one neuraminidase
subtype, and the 1350-nt length holds a single pattern that is a shorter variant
of the second. Their column lists run to 166 to 193 characters each, so
`gap_pattern_summary.csv` holds them rather than this table. This section calls
the patterns N2 and N1 after the neuraminidase subtype that dominates each,
which is a label for the gap pattern and not a claim that the pattern and the
subtype are the same set. The three subtype columns count unique proteins, not
isolates. In the N2 and N1 rows they sum to more than the unique protein count,
by 4 and by 8, because that many proteins occur in isolates of more than one
subtype.

| Source CDS length | Gap pattern | AA residues | Unique proteins | Isolates | H3N2 proteins | H1N1 proteins | H5N1 proteins |
|---|---|---:|---:|---:|---:|---:|---:|
| 1410 nt | N2 | 469 | 8,510 | 32,486 | 8,495 | 9 | 10 |
| 1410 nt | N1 | 469 | 8,305 | 26,631 | 6 | 7,531 | 776 |
| 1410 nt | 7 rare patterns | 469 | 28 | 37 | 10 | 1 | 17 |
| 1350 nt | the only pattern | 449 | 319 | 505 | 0 | 0 | 319 |

Both patterns occur in all three HN subtypes, but not evenly. The N1 pattern is
almost entirely H1N1 and H5N1, which both carry N1 as their names say, and the
N2 pattern is almost entirely H3N2.

N2 and N1 each fill 469 aa alignment columns, as their residue count requires,
but not the same 469. They share 445 columns. N2 fills 24 that N1 does not: 38,
39, 53, 55, 123–136, 353, 378–379, 384, 442 and 498. N1 fills 24 that N2 does
not: 69, 80–83, 95–105, 108–109, 208, 463–465, 472 and 542. Between them they
fill 493 columns.

Their longest run of disagreeing columns is 123 to 136, where N2 has 14
residues and N1 has 14 aa alignment gaps:

```text
Alignment column:    118-122   123-136   137-140
N2, 1410 nt:         res       res       res
N1, 1410 nt:         res       gap       res
1350 nt:             res       gap       res
```

The 1350-nt form is an N1 form, 20 codons shorter. It has 449 aa residues, and
every column it fills is already filled by N2 or N1, so it adds no column to the
alignment.

The three listed CDS lengths have no aa residue in columns 65, 456–457, 467–471
or 501, so they use 534 of the 543 columns. Those 9 columns are in the alignment
because rarer lengths fill them: 1422-nt proteins fill column 65, 1416-nt
proteins fill 456 and 457, 1428-nt proteins fill 467 to 471, and 1353-nt and
1422-nt proteins fill 501.

### 3. Projection to codons and the terminal stop

The terminal `*` was excluded from the MAFFT input, so none of the 543 aa
alignment columns corresponds to a terminal stop codon. NA's proteins do not all
end in the same alignment column. Their last aa residue falls in one of five
columns, so their stop codons would land in five different projected codon
columns:

| Last aa residue column | Projected stop column | Unique proteins | Isolates | Source CDS lengths |
|---:|---:|---:|---:|---|
| 535 | 536 | 1 | 1 | 1410 nt |
| 540 | 541 | 16 | 18 | 1413 nt |
| 541 | 542 | 8,574 | 32,580 | 7 lengths, including 1410 nt |
| 542 | 543 | 8,671 | 27,252 | 12 lengths, including 1410 nt and 1350 nt |
| 543 | 544 | 5 | 5 | 1353 nt, 1413 nt, 1422 nt |

The two forms at the modal length are in different rows. N2 ends in column 541
and its stop codon would be placed in projected column 542. N1 and the 1350-nt
form end in column 542, so their stop codon would be placed in projected column
543.

```text
Projected codon column:   541  542  543  544
N2, 1410 nt:              res   *    -    -
N1, 1410 nt:              res  res   *    -
1350 nt:                  res  res   *    -
1353, 1413 and 1422 nt:   res  res  res   *
```

`*` marks the projected column holding that protein's terminal stop codon, and
`-` a column after it, which receives `AFTER_STOP` rather than `GAP`. The last
row is 5 proteins over 5 isolates, one at 1353 nt, one at 1413 nt and three at
1422 nt. They fill alignment column 543, so their stop codon falls one column
past the alignment.

A codon feature table for NA therefore needs 544 columns, not 543. This
projection has not been implemented.

### 4. By host, subtype and year

"Other lengths" covers the thirteen lengths not listed separately, so its column
totals 68 isolates.

| Host | Subtype | Complete NA isolates | 1350 nt | 1410 nt | 1413 nt | Other lengths |
|---|---|---:|---:|---:|---:|---:|
| Chicken | H1N1 | 2 | 0 | 2 | 0 | 0 |
| Chicken | H3N2 | 17 | 0 | 16 | 0 | 1 |
| Chicken | H5N1 | 1,817 | 227 | 1,584 | 0 | 6 |
| Duck | H1N1 | 263 | 0 | 263 | 0 | 0 |
| Duck | H3N2 | 217 | 0 | 215 | 0 | 2 |
| Duck | H5N1 | 1,890 | 266 | 1,592 | 28 | 4 |
| Human | H1N1 | 19,686 | 0 | 19,630 | 53 | 3 |
| Human | H3N2 | 29,162 | 0 | 29,138 | 18 | 6 |
| Human | H5N1 | 36 | 10 | 26 | 0 | 0 |
| Swine | H1N1 | 3,607 | 0 | 3,566 | 5 | 36 |
| Swine | H3N2 | 3,148 | 0 | 3,113 | 25 | 10 |
| Swine | H5N1 | 11 | 2 | 9 | 0 | 0 |

The 1350-nt records are all H5N1: 227 Chicken, 266 Duck, 10 Human and 2 Swine,
505 in total. They appear in all 16 years.

The 1413-nt records have two peaks in a 13-year series: 52 of their 129 isolates
in 2014 and 28 isolates in 2025.

### 5. Findings

* NA has a usable pin at 1410 nt, but the pin does not reduce NA to one form.
  Pinning would keep 59,154 of the 59,856 isolates with a complete NA CDS and
  drop 702. It would drop all 505 isolates of the 1350-nt form, and 543 H5N1
  isolates in total, keeping 3,211 of the 3,754.
* The required feature width depends on which proteins are retained. An aa
  alignment column that becomes all-gap after filtering can be removed. The
  projected codon representation adds one column for the terminal stop codon.

| Proteins retained | Unique proteins | Isolates | All-gap aa columns removed | AA alignment columns | Projected codon columns |
|---|---:|---:|---:|---:|---:|
| All 16 source CDS lengths | 17,267 | 59,856 | None | 543 | 544 |
| 1410-nt source CDS only | 16,843 | 59,154 | 33 columns | 510 | 511 |
| The N2, N1 and 1350-nt forms | 17,134 | 59,622 | 50 columns | 493 | 494 |
| The N2 and N1 forms | 16,815 | 59,117 | 50 columns | 493 | 494 |
| N2 form only | 8,510 | 32,486 | 74 columns | 469 | 470 |

* Selecting by form beats selecting by CDS length, and for NA it wins on every
  count. The three forms keep 17,134 unique proteins over 59,622 isolates in 493
  aa columns, while the 1410-nt pin keeps 16,843 proteins over 59,154 isolates
  in 510 columns. That is 291 more proteins and 468 more isolates in 17 fewer
  columns.
* The 1350-nt form costs no column. Adding it to N2 and N1 leaves the count at
  493, because every column it fills is already filled by one of them, and it
  brings 319 unique proteins over 505 isolates, all H5N1. It is not free in the
  feature matrix: its records gap 20 columns that N1 fills, columns 66 to 85.
* The last row is not a usable option, but not because it excludes a subtype.
  N2 alone still holds 11 H1N1 and 13 H5N1 isolates, and N1 alone also fills 469
  columns, so 469 does not identify N2. The row is unusable because keeping one
  pattern drops the 8,305 unique proteins and 26,631 isolates of the other.
* 16 aa alignment columns hold an aa residue in exactly one protein: 93–94,
  456–457, 467–471, 494 and 527–532. They belong to 5 proteins. Two of those are
  at the modal length and own 7 of the 16 columns, 494 and 527 to 532, so the
  pin does not free them.
* 379 of the 543 aa alignment columns hold an aa residue in all 17,267 proteins.
  That is the lowest share of the eight proteins.
* 45 gap patterns cover the 16 CDS lengths. The 1413-nt length alone holds 11,
  the 1410-nt length 9 and the 1407-nt length 8, so a CDS length does not
  determine a gap pattern.
* The longest internal run of aa alignment gaps is columns 68 to 94, 27 columns,
  in one protein from one isolate at 1353 nt.
* The protein-to-codon projection has not been implemented. For NA it must
  handle five different stop columns and must distinguish `GAP` from
  `AFTER_STOP` for the records that end before column 543.
* Still open before codon features are built: whether one set of columns can
  serve both N1 and N2 given that they disagree in 48 of the 493 columns they
  use, and whether to select records by form rather than by CDS length.

## M1 (Segment 7)

M1 sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Isolates and CDS records.** Of 60,751 filtered isolates, 60,566 (99.7%)
  have a complete M1 CDS record; 185 isolates were excluded because their M1
  CDS was incomplete. That is 0.3% of the population, the smallest completeness
  loss of the eight proteins after NS1. Thus, counts of complete records in this
  section are counts of isolates with a complete M1 CDS, before sequences are
  deduplicated.
* **Nucleotide sequences.** The 60,566 complete M1 CDS records contain 14,573
  unique CDS sequences across 2 CDS lengths and 147 host–subtype–year groups.
  Two lengths is the fewest of the eight proteins, tied with NP.
* **Protein sequences.** Those CDS sequences translate to 1,948 unique M1
  protein sequences, the fewest of the eight proteins.
* **Unknown amino acids.** The translated proteins for 425 isolates,
  representing 185 unique proteins, contain an `X`. All 425 of those isolates
  have a 759-nt CDS.
* **Modal CDS length.** M1 has a usable configured length of 759 nt. This length
  includes 252 sense codons and 1 terminal stop codon. It occurs in 14,572 of
  14,573 unique complete CDS sequences (1.000), exceeding the `check_cds_length`
  requirement of 0.90. The configured value in `conf/virus/flu.yaml` agrees with
  the observed mode.
* **MAFFT input.** Before alignment, the terminal `*` was removed from each
  protein. MAFFT therefore aligned 1,948 unique stop-free aa sequences. A
  protein from a 759-nt CDS contributes 252 aa residues to this alignment.
* **Alignment result.** MAFFT used FFT-NS-2 and produced 252 aa alignment
  columns.
  - Removing alignment gaps (`-`) from every aligned sequence recovered its
    exact stop-free protein input.
  - Each protein from a 759-nt CDS contains 252 aa residues and therefore has
    gaps in none of the 252 alignment columns. The alignment is 252 columns wide
    because no input protein is longer than 252 aa residues.
  - The proteins from 759-nt CDS records have 1 observed gap pattern, which is
    the pattern with no gaps.

### 2. Lengths and gaps

The CDS lengths in this table identify the nt sequences from which the proteins
were translated. All gap counts refer to the resulting aa alignment, not to an
nt alignment.

| Source CDS length | Isolates | Unique CDS | Unique proteins | Leading AA-gap columns | Internal AA-gap columns | Trailing AA-gap columns |
|---|---:|---:|---:|---|---|---|
| 726 nt | 1 | 1 | 1 | None | 11 | None |
| 759 nt | 60,565 | 14,572 | 1,947 | None | None | None |

M1 holds 2 gap patterns in the whole alignment, one per CDS length, which is the
fewest of the eight proteins alongside NP.

| Source CDS length | AA gap columns | AA residues | Unique proteins | Isolates |
|---|---|---:|---:|---:|
| 759 nt | None | 252 | 1,947 | 60,565 |
| 726 nt | 31–41 | 241 | 1 | 1 |

Every host and subtype in the population has records at 759 nt, as part 4
shows.

The 726-nt protein is not a truncated 759-nt one. It is 11 codons shorter, and
MAFFT placed all 11 of its aa alignment gaps in the middle, at columns 31 to 41,
so it starts at column 1 and ends at column 252 with every other protein. Only
1 of the 1,947 proteins at 759 nt matches it once those columns are deleted, and
that is not the one shown below, which differs at 11 residue positions outside
the gap:

```text
Alignment column:  21         31         41         51
759 nt:            KAEIAQRLED VFAGKNTDLE ALMEWLKTRP ILSP
726 nt:            KAEIAQRLED ---------- -LMEWLKTRP ILSP
```

One isolate carries this form, from Swine-H1N1 in 2011. The 759-nt row above is
the most common 759-nt protein, which 27,661 isolates carry.

### 3. Projection to codons and the terminal stop

The terminal `*` was excluded from the MAFFT input, so none of the 252 aa
alignment columns corresponds to a terminal stop codon. Every one of the 1,948
proteins has its last aa residue in alignment column 252, and none has a
trailing aa alignment gap. So every record's terminal stop codon would be placed
in projected codon column 253, and no record needs `AFTER_STOP`. M1, HA and NP
are the three proteins of the eight where that holds.

The only projected columns that need `GAP` are 31 to 41, for the single 726-nt
record. Those columns also hold an `X` in 9 of the 1,947 other proteins, and the
columns on either side, 30 and 42, hold an `X` in 7 of them. So a column that is
an internal gap for one record can hold an unresolved aa residue in another.

A codon feature table for M1 therefore needs 253 columns, not 252. This
projection has not been implemented.

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

* M1 has a usable pin at 759 nt. Pinning would keep 60,565 of the 60,566
  isolates with a complete M1 CDS and drop 1. It would keep all 3,776 H5N1
  isolates.
* The required feature width depends on which proteins are retained, except for
  M1, where neither choice below changes it. The only gaps in the alignment are
  the 11 internal columns of one record, and every other protein fills them, so
  dropping that record frees no column. Retaining only that record would be the
  one filtering that narrows the width, to 241 columns.

| Proteins retained | Unique proteins | Isolates | All-gap aa columns removed | AA alignment columns | Projected codon columns |
|---|---:|---:|---:|---:|---:|
| Both source CDS lengths | 1,948 | 60,566 | None | 252 | 253 |
| 759-nt source CDS only | 1,947 | 60,565 | None | 252 | 253 |

* The alignment adds no columns, so the pin costs none. The width is 252
  because no input protein is longer than 252 aa residues. Every one of the
  1,947 proteins at 759 nt holds an aa residue in all 252 columns, so within
  that set positions correspond by index and no gap is inserted. That is not the
  same as the proteins being identical: there are 1,947 distinct sequences at
  that length, and 185 of them carry an `X`.
* 241 of the 252 aa alignment columns hold an aa residue in all 1,948 proteins.
  The other 11 are columns 31 to 41, which hold one in 1,947, every protein but
  the 726-nt one.
* There are no one-residue columns.
* Still open before codon features are built: whether to keep the single 726-nt
  isolate. Uncommon length alone does not establish an error.

## NS1 (Segment 8)

NS1 sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Isolates and CDS records.** Of 60,751 filtered isolates, 60,654 (99.8%)
  have a complete NS1 CDS record; 97 isolates were excluded because their NS1
  CDS was incomplete. That is 0.2% of the population, the smallest completeness
  loss of the eight proteins. Thus, counts of complete records in this section
  are counts of isolates with a complete NS1 CDS, before sequences are
  deduplicated.
* **Nucleotide sequences.** The 60,654 complete NS1 CDS records contain 18,107
  unique CDS sequences across 16 CDS lengths and 147 host–subtype–year groups.
* **Protein sequences.** Those CDS sequences translate to 10,130 unique NS1
  protein sequences.
* **Unknown amino acids.** The translated proteins for 957 isolates,
  representing 551 unique proteins, contain an `X`. 679 of those isolates have
  a 693-nt CDS.
* **No usable pin.** NS1 is one of two proteins with no configured CDS length,
  and the two are not close: PB1's modal length reaches 0.890 of unique complete
  CDS. NS1's modal length, 693 nt, occurs in 9,097 of 18,107 unique complete CDS
  sequences, a share of 0.502. The next length, 660 nt, holds 8,243, a share of
  0.455. Neither is close to the
  `check_cds_length` requirement of 0.90, so no pin is available and the
  alignment is the only route to shared sites. The `cds_length` block in
  `conf/virus/flu.yaml` lists PB2, PA, NP, M1, HA and NA, and no length for
  NS1.
* **MAFFT input.** Before alignment, the terminal `*` was removed from each
  protein. MAFFT therefore aligned 10,130 unique stop-free aa sequences. A
  protein from a 693-nt CDS contributes 230 aa residues to this alignment, and
  one from a 660-nt CDS contributes 219.
* **Alignment result.** MAFFT used FFT-NS-2 and produced 240 aa alignment
  columns.
  - Removing alignment gaps (`-`) from every aligned sequence recovered its
    exact stop-free protein input.
  - Each protein from a 693-nt CDS contains 230 aa residues and therefore has
    gaps in 10 of the 240 alignment columns. A 660-nt protein has 219 residues
    and gaps in 21.
  - The proteins from 693-nt CDS records have 1 observed gap pattern, and so do
    those from 660-nt records. Of the 16 CDS lengths, 13 hold one pattern each
    and 3 hold two: 663, 687 and 696 nt, one protein apiece in the second. So
    the population is split across lengths rather than within them.

### 2. Lengths and gaps

The CDS lengths in this table identify the nt sequences from which the proteins
were translated. All gap counts refer to the resulting aa alignment, not to an
nt alignment.

| Source CDS length | Isolates | Unique CDS | Unique proteins | Leading AA-gap columns | Internal AA-gap columns | Trailing AA-gap columns |
|---|---:|---:|---:|---|---|---|
| 654 nt | 389 | 236 | 207 | None | 3 | 20 |
| 660 nt | 25,506 | 8,243 | 4,601 | None | 3 | 18 |
| 672 nt | 806 | 235 | 144 | None | 3 | 14 |
| 678 nt | 341 | 246 | 205 | None | 8 | 7 |
| 693 nt | 33,553 | 9,097 | 4,926 | None | 3 | 7 |
| 11 other lengths | 59 | 50 | 47 | 0 to 5 | 2 to 8 | 0 to 35 |

Each of the five listed lengths holds exactly one gap pattern, so for every
length in the table above the CDS length fixes the gap pattern. HA's modal
length holds 6 patterns and NA's holds 9, two of them large in each case.

The three subtype columns count unique proteins, not isolates. They sum to more than the unique protein
count in four rows, by 1 for 654 nt, 102 for 660 nt, 1 for 672 nt and 45 for
693 nt. For the first three that is the number of proteins occurring in isolates
of more than one subtype. At 693 nt it is 38 such proteins, 7 of which occur in
all three subtypes and so add 2 each.

| Source CDS length | AA gap columns | AA residues | Unique proteins | Isolates | H3N2 proteins | H1N1 proteins | H5N1 proteins |
|---|---|---:|---:|---:|---:|---:|---:|
| 654 nt | 74, 112, 132, 221–240 | 217 | 207 | 389 | 74 | 119 | 15 |
| 660 nt | 74, 112, 132, 223–240 | 219 | 4,601 | 25,506 | 1,060 | 3,641 | 2 |
| 672 nt | 74, 112, 132, 227–240 | 223 | 144 | 806 | 144 | 1 | 0 |
| 678 nt | 74, 81–85, 112, 132, 234–240 | 225 | 205 | 341 | 1 | 0 | 204 |
| 693 nt | 74, 112, 132, 234–240 | 230 | 4,926 | 33,553 | 3,618 | 686 | 667 |
| 11 other lengths | 14 patterns | 202 to 237 | 47 | 59 | 20 | 24 | 3 |

No length belongs to one HN subtype, but four of the five lean hard. By
isolates: 693 nt is 28,725 H3N2, 3,379 H5N1 and 1,449 H1N1; 660 nt is 22,419 of
25,506 H1N1; 672 nt is 804 of 806 H3N2; and 678 nt is 340 of 341 H5N1.

Each of the five listed lengths gaps columns 74, 112 and 132, as do 11 of the 14
patterns in the last row. Those three columns are in the alignment for three
rare proteins only: a 696-nt protein fills column 74, a 663-nt protein fills
112, and a different 696-nt protein fills 132. Each is the sole protein in its
column, and each is one of the three patterns in the last row that does not gap
all three.

The 678-nt form is not a truncated 693-nt form. Its proteins are 5 codons
shorter, and MAFFT placed all 5 of those aa alignment gaps in the middle, at
columns 81 to 85:

```text
Alignment column:  71-73      74      75-80      81-85      86-94
693 nt:            res        gap     res        res        res
678 nt:            res        gap     res        gap        res
```

Column 74 is a gap in both forms, because it is one of the three columns held by
a single rare protein.

All 205 of its unique proteins share this pattern, over 341 isolates. One
699-nt protein from one isolate has the same 81–85 gap, so 206 proteins over 342
isolates gap those columns. Under the proposed codon mapping these 5 columns
take `GAP` and not `AFTER_STOP`, because they sit between aa residues rather
than after a stop codon. Three other patterns put internal gaps outside columns
74, 112 and 132: columns 81 to 85 in that 699-nt protein, columns 77 to 80 in a
648-nt protein over 3 isolates, and columns 79 to 80 in a 687-nt protein over 1
isolate.

One protein has leading gaps: a 645-nt protein from one isolate, at columns 1
to 5.

### 3. Projection to codons and the terminal stop

The terminal `*` was excluded from the MAFFT input, so none of the 240 aa
alignment columns corresponds to a terminal stop codon. NS1 ends in more
different places than any other of the eight proteins. Its proteins' last aa
residue falls in one of eleven columns, so their stop codons would land in
eleven different projected codon columns:

| Last aa residue column | Projected stop column | Unique proteins | Isolates | Source CDS lengths |
|---:|---:|---:|---:|---|
| 205 | 206 | 10 | 14 | 609 nt |
| 206 | 207 | 3 | 3 | 612 nt |
| 219 | 220 | 5 | 6 | 651 nt |
| 220 | 221 | 207 | 389 | 654 nt |
| 222 | 223 | 4,604 | 25,511 | 645, 648, 660, 663 nt |
| 223 | 224 | 4 | 4 | 663 nt |
| 226 | 227 | 144 | 806 | 672 nt |
| 230 | 231 | 3 | 3 | 684 nt |
| 231 | 232 | 4 | 4 | 687 nt |
| 233 | 234 | 5,134 | 33,897 | 678, 687, 693, 696 nt |
| 240 | 241 | 12 | 17 | 699, 714 nt |

The two forms that carry almost every record are in two of those rows. The
660-nt form ends in column 222 and its stop codon would be placed in projected
column 223. The 693-nt and 678-nt forms both end in column 233, so both would
place their stop codon in projected column 234. That the 678-nt form ends with
the 693-nt one follows from its deletion being internal rather than terminal.

```text
Projected codon column:   222  223  234  241
660 nt:                   res   *    -    -
693 nt and 678 nt:        res  res   *    -
699 nt and 714 nt:        res  res  res   *
```

`*` marks the projected column holding that protein's terminal stop codon, and
`-` a column after it, which receives `AFTER_STOP` rather than `GAP`.

A codon feature table for NS1 therefore needs 241 columns, not 240. The 241st
column exists only for the 699-nt and 714-nt forms, which are 12 unique proteins
over 17 isolates. 10,118 of the 10,130 unique proteins need `AFTER_STOP` for
their last columns, a run of 7 to 35 columns depending on the protein. This
projection has not been implemented.

### 4. By host, subtype and year

"Other lengths" covers the eleven lengths not listed separately, so its column
totals 59 isolates. The 714-nt form is one of them. It is one of the two forms
that reach alignment column 240, the other being the single 699-nt protein.

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

The 678-nt records are almost all H5N1, 340 of 341, of which 206 are Chicken and
121 Duck.

The 672-nt records are concentrated in Human-H3N2 in 2019 and 2020, with 753 of
their 806 isolates.

The 654-nt, 660-nt and 693-nt records are present in all 16 years. The other two
listed lengths are not. The 678-nt records run from 2010 to 2018 and do not
appear after that, in 9 years, and the 672-nt records appear only in 2016, 2018
to 2020 and 2022, in 5 years.

### 5. Findings

* NS1 has no usable pin, and it is the only protein where no length holds even
  three fifths of unique complete CDS. The modal length holds 0.502 and the next
  holds 0.455, both below the 0.90 floor, and `conf/virus/flu.yaml` configures
  no length for it. The alignment is the only route to shared sites.
* NS1 splits across lengths rather than within one. Each of the five listed
  lengths holds exactly one gap pattern, so for those lengths a CDS length fixes
  the gap pattern. Only 3 of the 16 lengths hold two patterns, one protein
  apiece in the second: 663, 687 and 696 nt.
* The required feature width depends on which proteins are retained. An aa
  alignment column that becomes all-gap after filtering can be removed. The
  projected codon representation adds one column for the terminal stop codon.

| Proteins retained | Unique proteins | Isolates | All-gap aa columns removed | AA alignment columns | Projected codon columns |
|---|---:|---:|---:|---:|---:|
| All 16 source CDS lengths | 10,130 | 60,654 | None | 240 | 241 |
| The 5 listed CDS lengths | 10,083 | 60,595 | 10 columns | 230 | 231 |
| 660-nt and 693-nt source CDS | 9,527 | 59,059 | 10 columns | 230 | 231 |
| 693-nt source CDS only | 4,926 | 33,553 | 10 columns | 230 | 231 |
| 660-nt source CDS only | 4,601 | 25,506 | 21 columns | 219 | 220 |

* 10 of the 240 aa alignment columns exist for 15 unique proteins in total.
  Columns 234 to 240 exist for the 12 proteins at 699 and 714 nt, over 17
  isolates, and columns 74, 112 and 132 for one protein each, over 3 isolates.
  Dropping those 15 proteins takes the width to 230, and it stays at 230 for
  every narrower set that keeps the modal length, down to 693 nt alone. A set
  without it can be narrower: 660 nt alone is 219 columns.
  Keeping the 5 listed lengths therefore costs nothing in width against keeping
  the modal length alone, and keeps 5,157 more unique proteins over 27,042 more
  isolates.
* The 660-nt and 693-nt lengths cover 59,059 of the 60,654 complete records,
  97.4%. Adding 654, 672 and 678 nt reaches 60,595, 99.9%. The 714-nt form
  accounts for 16 isolates and the ten remaining lengths for 43.
* Columns 74, 112 and 132 each hold an aa residue in exactly one of the 10,130
  proteins, and every one of the common patterns gaps all three.
* 188 of the 240 aa alignment columns hold an aa residue in all 10,130 proteins.
* There are 19 gap patterns over the 16 CDS lengths. Three lengths carry two
  patterns each: 663, 687 and 696 nt.
* The birds and H5N1 do not bring a third common length. Chicken-H5N1 is 1,600
  of 1,825 at 693 nt and Duck-H5N1 is 1,739 of 1,899, the same length as
  Human-H3N2. They bring the 678-nt length, which holds 341 isolates, 328 of
  them Chicken or Duck.
* The protein-to-codon projection has not been implemented, and NS1 asks the
  most of it on two measures. It has eleven different projected stop columns,
  against at most five for any other of the eight. And `AFTER_STOP` is needed
  for 10,118 of its 10,130 unique proteins over runs of 7 to 35 columns, where
  no other protein's run exceeds 25.
* Still open before codon features are built: whether to keep the 59 isolates
  outside the five listed lengths, and whether to keep the three one-residue
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
  here.
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
