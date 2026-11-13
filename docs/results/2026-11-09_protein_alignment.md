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

## Goal

This analysis determines which CDS records and alignment columns to retain for
per-site codon features, and how to place the retained records in a common
coordinate system. Equal CDS or protein length does not guarantee that residues
at the same position correspond, because insertions and deletions can shift
homologous residues. Protein alignment estimates positional correspondence
across sequences of both equal and unequal length.

The retention decision is protein-specific. When one gap pattern covers nearly
all records and all other patterns together cover only a small fraction, as in
PB2, we may retain only the common pattern and drop the uncommon records. When
multiple gap patterns are each represented by many records, as in NS1, HA and
NA, retaining only one would discard many records and could bias the retained
population. Alignment allows these patterns to be retained in one shared
coordinate system. In HA and NA, the two largest patterns come from CDS of the
same length, yet their gaps fall in different columns, so residues at the same
position do not all correspond.

How many records are enough to retain a gap pattern is a decision informed by
this analysis. For each protein, the document reports population coverage,
gap-pattern support, feature-width cost, the proposed placement of terminal
stop codons, and whether an absent codon would be encoded as `GAP` or
`AFTER_STOP`. The goal is to preserve well-supported sequence forms without
adding records or alignment columns whose small coverage does not justify their
complexity. The projection from alignment columns to codons is proposed in part
3 of each section and is not implemented.


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


## Retention rule

This rule decides which CDS records each protein keeps for per-site codon
features.

1. Use only complete CDS.
2. For each protein, compute each CDS length's share of the protein's unique
   CDS. Keep the lengths above 2%.
3. Translate the kept CDS and align the unique proteins with MAFFT.
4. In that alignment, compute each gap pattern's share of the unique CDS kept
   in step 2. Drop the CDS of gap patterns at or below 2%.
5. If any CDS were dropped, realign the remaining proteins. That alignment is
   the input to the per-site codon features.

Steps 3 and 4 have been run for all eight proteins, and step 5 for none.
Applied to the alignment of all lengths reported in this document, steps 2 and 4 keep
exactly the lengths and gap patterns listed in each protein's Decision section.
On that alignment, any threshold from 1.36% to 4.58% gives the same result in
all eight proteins. The largest share dropped is NS1's 678-nt length, at 1.36%
of NS1's unique CDS. The smallest share kept is HA's H5 gap pattern at 1704 nt,
at 4.59% of HA's unique CDS at the kept lengths.

The rule pools all years. It drops NS1's 672-nt length, which holds 698 of the
5,255 isolates with a complete NS1 CDS in 2019.


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

* **Complete CDS.** 60,474 of the 60,751 filtered isolates (99.5%) have a
  complete PB2 CDS. The other 277 have an incomplete CDS and are excluded.
* **Unique sequences.** The 60,474 complete CDS hold 36,939 unique CDS at 18 CDS
  lengths. They translate to 15,290 unique proteins.

### 2. Retention rule

#### Steps 1 and 2. Complete CDS and lengths above 2%

The CDS lengths in this table identify the nt sequences from which the proteins
were translated.

| CDS length | Isolates | Unique CDS | % of unique CDS | Unique proteins | % of unique proteins | Step 2 |
|---|---:|---:|---:|---:|---:|---|
| 2280 nt | 60,413 | 36,879 | 99.84% | 15,230 | 99.61% | keep |
| 17 other lengths | 61 | 60 | 0.16% | 60 | 0.39% | drop |

#### Step 3. Alignment of the kept lengths

* MAFFT aligned the 15,230 unique proteins from 2280-nt CDS with FFT-NS-2 and
  produced 760 aa alignment columns. MAFFT took 420.6 seconds.
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* Each protein has 759 aa residues, so it has a gap in 1 of the 760 columns.
* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/PB2/length_share_above_0.02/`.

#### Step 4. Gap patterns above 2%

| Gap pattern | AA gap columns | AA residues | Unique proteins | Unique CDS | % of unique CDS | Isolates | Step 4 |
|---|---|---:|---:|---:|---:|---:|---|
| `gap_pat001` | 570 | 759 | 15,229 | 36,878 | >99.99% | 60,412 | keep |
| `gap_pat002` | 577 | 759 | 1 | 1 | <0.01% | 1 | drop |

* `% of unique CDS` divides each gap pattern's unique CDS by 36,879, the unique
  CDS kept in step 2.

#### Step 5. Realignment

TODO.

### 4. By host, subtype and year

| Host | Subtype | Complete PB2 isolates | Kept isolates | % kept |
|---|---|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 100% |
| Chicken | H3N2 | 17 | 17 | 100% |
| Chicken | H5N1 | 1,815 | 1,813 | 99.89% |
| Duck | H1N1 | 264 | 264 | 100% |
| Duck | H3N2 | 216 | 215 | 99.54% |
| Duck | H5N1 | 1,894 | 1,894 | 100% |
| Human | H1N1 | 20,017 | 19,994 | 99.89% |
| Human | H3N2 | 29,439 | 29,409 | 99.90% |
| Human | H5N1 | 36 | 36 | 100% |
| Swine | H1N1 | 3,599 | 3,595 | 99.89% |
| Swine | H3N2 | 3,159 | 3,158 | 99.97% |
| Swine | H5N1 | 16 | 15 | 93.75% |
| Total | | 60,474 | 60,412 | 99.90% |

* Every host–subtype–year group with at least 100 isolates keeps at least
  99.53% of its isolates. The lowest is Swine-H1N1-2010, which keeps 213 of 214.

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

### Decision

* Keep 2280 nt CDS (760 aa proteins).
  * Within the 2280 nt CDS, keep the "42–45, 348, 575, 678" chunk.


## PB1 (Segment 2)

PB1 sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Complete CDS.** 56,093 of the 60,751 filtered isolates (92.3%) have a
  complete PB1 CDS. The other 4,658 have an incomplete CDS and are excluded.
* **Unique sequences.** The 56,093 complete CDS hold 34,642 unique CDS at 18 CDS
  lengths. They translate to 13,220 unique proteins.

### 2. Retention rule

#### Steps 1 and 2. Complete CDS and lengths above 2%

The CDS lengths in this table identify the nt sequences from which the proteins
were translated.

| CDS length | Isolates | Unique CDS | % of unique CDS | Unique proteins | % of unique proteins | Step 2 |
|---|---:|---:|---:|---:|---:|---|
| 2274 nt | 49,948 | 30,824 | 88.98% | 12,060 | 91.23% | keep |
| 2277 nt | 6,095 | 3,768 | 10.88% | 1,110 | 8.40% | keep |
| 16 other lengths | 50 | 50 | 0.14% | 50 | 0.38% | drop |

#### Step 3. Alignment of the kept lengths

* MAFFT aligned the 13,170 unique proteins from 2274-nt and 2277-nt CDS with
  FFT-NS-2 and produced 762 aa alignment columns. MAFFT took 343.9 seconds.
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* A protein from a 2274-nt CDS has 757 aa residues, so it has gaps in 5 of the
  762 columns. A protein from a 2277-nt CDS has 758 aa residues and gaps in 4.
* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/PB1/length_share_above_0.02/`.

#### Step 4. Gap patterns above 2%

| Gap pattern | CDS length | AA gap columns | AA residues | Unique proteins | Unique CDS | % of unique CDS | Isolates | Step 4 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `gap_pat001` | 2274 nt | 77–79, 389, 762 | 757 | 12,058 | 30,822 | 89.10% | 49,946 | keep |
| `gap_pat002` | 2277 nt | 77–79, 389 | 758 | 1,109 | 3,767 | 10.89% | 6,094 | keep |
| `gap_pat003` | 2274 nt | 62–63, 68, 389, 762 | 757 | 1 | 1 | <0.01% | 1 | drop |
| `gap_pat004` | 2277 nt | 77–79, 762 | 758 | 1 | 1 | <0.01% | 1 | drop |
| `gap_pat005` | 2274 nt | 77–79, 94, 389 | 757 | 1 | 1 | <0.01% | 1 | drop |

* `% of unique CDS` divides each gap pattern's unique CDS by 34,592, the unique
  CDS kept in step 2.

#### Step 5. Realignment

TODO.

### 4. By host, subtype and year

| Host | Subtype | Complete PB1 isolates | Kept isolates | % kept |
|---|---|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 100% |
| Chicken | H3N2 | 17 | 17 | 100% |
| Chicken | H5N1 | 1,798 | 1,797 | 99.94% |
| Duck | H1N1 | 263 | 263 | 100% |
| Duck | H3N2 | 217 | 216 | 99.54% |
| Duck | H5N1 | 1,871 | 1,870 | 99.95% |
| Human | H1N1 | 19,761 | 19,734 | 99.86% |
| Human | H3N2 | 25,390 | 25,377 | 99.95% |
| Human | H5N1 | 35 | 35 | 100% |
| Swine | H1N1 | 3,578 | 3,570 | 99.78% |
| Swine | H3N2 | 3,148 | 3,146 | 99.94% |
| Swine | H5N1 | 13 | 13 | 100% |
| Total | | 56,093 | 56,040 | 99.91% |

* Every host–subtype–year group with at least 100 isolates keeps at least
  99.09% of its isolates. The lowest is Swine-H1N1-2022, which keeps 219 of 221.

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

### Decision

* Keep 2274 nt CDS (758 aa proteins).
  * Within the 2274 nt CDS, keep the "77–79, 389, 718–728, 773–776" chunk.
* Keep 2277 nt CDS (759 aa proteins).
  * Within the 2277 nt CDS, keep the "77–79, 389, 718–728, 774–776" chunk.




## PA (Segment 3)

PA sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Complete CDS.** 60,396 of the 60,751 filtered isolates (99.4%) have a
  complete PA CDS. The other 355 have an incomplete CDS and are excluded.
* **Unique sequences.** The 60,396 complete CDS hold 35,659 unique CDS at 14 CDS
  lengths. They translate to 15,939 unique proteins.

### 2. Retention rule

#### Steps 1 and 2. Complete CDS and lengths above 2%

The CDS lengths in this table identify the nt sequences from which the proteins
were translated.

| CDS length | Isolates | Unique CDS | % of unique CDS | Unique proteins | % of unique proteins | Step 2 |
|---|---:|---:|---:|---:|---:|---|
| 2151 nt | 60,332 | 35,609 | 99.86% | 15,897 | 99.74% | keep |
| 13 other lengths | 64 | 50 | 0.14% | 42 | 0.26% | drop |

#### Step 3. Alignment of the kept lengths

* MAFFT aligned the 15,897 unique proteins from 2151-nt CDS with FFT-NS-2 and
  produced 716 aa alignment columns. MAFFT took 447.6 seconds.
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* Each protein has 716 aa residues, so no protein has a gap.
* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/PA/length_share_above_0.02/`.

#### Step 4. Gap patterns above 2%

| Gap pattern | AA gap columns | AA residues | Unique proteins | Unique CDS | % of unique CDS | Isolates | Step 4 |
|---|---|---:|---:|---:|---:|---:|---|
| `gap_pat001` | None | 716 | 15,897 | 35,609 | 100% | 60,332 | keep |

* `% of unique CDS` divides each gap pattern's unique CDS by 35,609, the unique
  CDS kept in step 2.

#### Step 5. Realignment

Not applicable: step 4 drops no gap pattern.

### 4. By host, subtype and year

| Host | Subtype | Complete PA isolates | Kept isolates | % kept |
|---|---|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 100% |
| Chicken | H3N2 | 17 | 17 | 100% |
| Chicken | H5N1 | 1,824 | 1,823 | 99.95% |
| Duck | H1N1 | 265 | 265 | 100% |
| Duck | H3N2 | 217 | 216 | 99.54% |
| Duck | H5N1 | 1,898 | 1,897 | 99.95% |
| Human | H1N1 | 19,896 | 19,892 | 99.98% |
| Human | H3N2 | 29,450 | 29,399 | 99.83% |
| Human | H5N1 | 36 | 36 | 100% |
| Swine | H1N1 | 3,619 | 3,616 | 99.92% |
| Swine | H3N2 | 3,164 | 3,161 | 99.91% |
| Swine | H5N1 | 8 | 8 | 100% |
| Total | | 60,396 | 60,332 | 99.89% |

* Every host–subtype–year group with at least 100 isolates keeps at least
  98.25% of its isolates. The lowest is Human-H3N2-2021, which keeps 1,068 of
  1,087.

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

### Decision

* Keep 2151 nt CDS (717 aa proteins).
  * Within the 2152 nt CDS, keep the "48–49, 77, 82–83, 94–98, 624, 641–650, 660–663, 703, 743–744" chunk.


## HA (Segment 4)

HA sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Complete CDS.** 60,282 of the 60,751 filtered isolates (99.2%) have a
  complete HA CDS. The other 469 have an incomplete CDS and are excluded.
* **Unique sequences.** The 60,282 complete CDS hold 35,509 unique CDS at 10 CDS
  lengths. They translate to 19,841 unique proteins.

### 2. Retention rule

#### Steps 1 and 2. Complete CDS and lengths above 2%

The CDS lengths in this table identify the nt sequences from which the proteins
were translated.

| CDS length | Isolates | Unique CDS | % of unique CDS | Unique proteins | % of unique proteins | Step 2 |
|---|---:|---:|---:|---:|---:|---|
| 1695 nt | 244 | 203 | 0.57% | 179 | 0.90% | drop |
| 1698 nt | 71 | 67 | 0.19% | 62 | 0.31% | drop |
| 1701 nt | 56,223 | 33,492 | 94.32% | 18,606 | 93.78% | keep |
| 1704 nt | 3,632 | 1,643 | 4.63% | 901 | 4.54% | keep |
| 1707 nt | 63 | 56 | 0.16% | 50 | 0.25% | drop |
| 5 other lengths | 49 | 48 | 0.14% | 43 | 0.22% | drop |

#### Step 3. Alignment of the kept lengths

* MAFFT aligned the 19,507 unique proteins from 1701-nt and 1704-nt CDS with
  FFT-NS-2 and produced 590 aa alignment columns. MAFFT took 525.3 seconds.
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* A protein from a 1701-nt CDS has 566 aa residues, so it has gaps in 24 of the
  590 columns. A protein from a 1704-nt CDS has 567 aa residues and gaps in 23.
* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/HA/length_share_above_0.02/`.

#### Step 4. Gap patterns above 2%

| Gap pattern | CDS length | AA gap columns | AA residues | Unique proteins | Unique CDS | % of unique CDS | Isolates | Step 4 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `gap_pat001` (H3) | 1701 nt | 22, 25, 72–75, 104, 117, 134–136, 157, 168, 173, 186–187, 190, 300–301, 363–366, 575 | 566 | 10,274 | 18,560 | 52.82% | 32,783 | keep |
| `gap_pat002` (H1) | 1701 nt | 4, 6, 8, 19–25, 28, 81–83, 168, 173, 186–187, 190, 301, 363–366 | 566 | 8,326 | 14,926 | 42.48% | 23,428 | keep |
| `gap_pat003` (H5) | 1704 nt | 4, 6–8, 19–25, 28, 81–83, 151, 168, 173, 186–187, 190, 301, 366 | 567 | 870 | 1,612 | 4.59% | 3,596 | keep |
| 12 other patterns | 1701 and 1704 nt | | 566 or 567 | 37 | 37 | 0.04% or less each | 48 | drop |

* `% of unique CDS` divides each gap pattern's unique CDS by 35,135, the unique
  CDS kept in step 2.
* H3, H1 and H5 name each kept pattern after the subtype that dominates it.
  32,778 of the 32,783 isolates of `gap_pat001` are H3N2, 23,424 of the 23,428
  of `gap_pat002` are H1N1, and all 3,596 of `gap_pat003` are H5N1.

#### Step 5. Realignment

TODO.

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

### Decision

* Keep 1701 nt CDS (567 aa proteins).
  * Within the 1701 nt CDS, keep the H3 gap pattern chunk.
  * Within the 1701 nt CDS, keep the H1 gap pattern chunk.
* Keep 1704 nt CDS (568 aa proteins).
  * Within the 1704 nt CDS, keep the H5 gap pattern chunk.


## NP (Segment 5)

NP sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Complete CDS.** 60,040 of the 60,751 filtered isolates (98.8%) have a
  complete NP CDS. The other 711 have an incomplete CDS and are excluded.
* **Unique sequences.** The 60,040 complete CDS hold 27,087 unique CDS at 2 CDS
  lengths. They translate to 7,250 unique proteins.

### 2. Retention rule

#### Steps 1 and 2. Complete CDS and lengths above 2%

The CDS lengths in this table identify the nt sequences from which the proteins
were translated.

| CDS length | Isolates | Unique CDS | % of unique CDS | Unique proteins | % of unique proteins | Step 2 |
|---|---:|---:|---:|---:|---:|---|
| 1461 nt | 1 | 1 | <0.01% | 1 | 0.01% | drop |
| 1497 nt | 60,039 | 27,086 | >99.99% | 7,249 | 99.99% | keep |

#### Step 3. Alignment of the kept lengths

* MAFFT aligned the 7,249 unique proteins from 1497-nt CDS with FFT-NS-2 and
  produced 498 aa alignment columns. MAFFT took 70.3 seconds.
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* Each protein has 498 aa residues, so no protein has a gap.
* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/NP/length_share_above_0.02/`.

#### Step 4. Gap patterns above 2%

| Gap pattern | AA gap columns | AA residues | Unique proteins | Unique CDS | % of unique CDS | Isolates | Step 4 |
|---|---|---:|---:|---:|---:|---:|---|
| `gap_pat001` | None | 498 | 7,249 | 27,086 | 100% | 60,039 | keep |

* `% of unique CDS` divides each gap pattern's unique CDS by 27,086, the unique
  CDS kept in step 2.

#### Step 5. Realignment

Not applicable: step 4 drops no gap pattern.

### 4. By host, subtype and year

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

### Decision

* Keep 1497 nt CDS (599 aa proteins).
  * Within the 1497 nt CDS, keep all seqs (same pattern).


## NA (Segment 6)

NA sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Complete CDS.** 59,856 of the 60,751 filtered isolates (98.5%) have a
  complete NA CDS. The other 895 have an incomplete CDS and are excluded.
* **Unique sequences.** The 59,856 complete CDS hold 31,157 unique CDS at 16 CDS
  lengths. They translate to 17,267 unique proteins.

### 2. Retention rule

#### Steps 1 and 2. Complete CDS and lengths above 2%

The CDS lengths in this table identify the nt sequences from which the proteins
were translated.

| CDS length | Isolates | Unique CDS | % of unique CDS | Unique proteins | % of unique proteins | Step 2 |
|---|---:|---:|---:|---:|---:|---|
| 1350 nt | 505 | 407 | 1.31% | 319 | 1.85% | drop |
| 1410 nt | 59,154 | 30,630 | 98.31% | 16,843 | 97.54% | keep |
| 1413 nt | 129 | 59 | 0.19% | 55 | 0.32% | drop |
| 13 other lengths | 68 | 61 | 0.20% | 50 | 0.29% | drop |

#### Step 3. Alignment of the kept lengths

* MAFFT aligned the 16,843 unique proteins from 1410-nt CDS with FFT-NS-2 and
  produced 514 aa alignment columns. MAFFT took 332.9 seconds.
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* Each protein has 469 aa residues, so it has gaps in 45 of the 514 columns.
* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/NA/length_share_above_0.02/`.

#### Step 4. Gap patterns above 2%

| Gap pattern | AA gap columns | AA residues | Unique proteins | Unique CDS | % of unique CDS | Isolates | Step 4 |
|---|---|---:|---:|---:|---:|---:|---|
| `gap_pat001` (N2) | 46–55, 102–119, 193, 201, 303, 442–445, 468, 471, 499–505, 514 | 469 | 8,520 | 16,092 | 52.54% | 32,498 | keep |
| `gap_pat002` (N1) | 40–53, 64–70, 87–93, 201, 303, 338, 366–367, 369, 424, 468, 471–472, 499–505 | 469 | 8,305 | 14,518 | 47.40% | 26,631 | keep |
| 4 other patterns | | 469 | 18 | 20 | 0.05% or less each | 25 | drop |

* `% of unique CDS` divides each gap pattern's unique CDS by 30,630, the unique
  CDS kept in step 2.
* N2 and N1 name each kept pattern after the neuraminidase subtype of its
  isolates. 32,474 of the 32,498 isolates of `gap_pat001` are H3N2, and 26,623
  of the 26,631 of `gap_pat002` are H1N1 or H5N1, which both carry N1.

#### Step 5. Realignment

TODO.

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

### Decision

* Keep 1410 nt CDS (470 aa proteins).
  * Within the 1410 nt CDS, keep the N2 gap pattern seqs.
  * Within the 1410 nt CDS, keep the N1 gap pattern seqs.


## M1 (Segment 7)

M1 sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Complete CDS.** 60,566 of the 60,751 filtered isolates (99.7%) have a
  complete M1 CDS. The other 185 have an incomplete CDS and are excluded.
* **Unique sequences.** The 60,566 complete CDS hold 14,573 unique CDS at 2 CDS
  lengths. They translate to 1,948 unique proteins.

### 2. Retention rule

#### Steps 1 and 2. Complete CDS and lengths above 2%

The CDS lengths in this table identify the nt sequences from which the proteins
were translated.

| CDS length | Isolates | Unique CDS | % of unique CDS | Unique proteins | % of unique proteins | Step 2 |
|---|---:|---:|---:|---:|---:|---|
| 726 nt | 1 | 1 | 0.01% | 1 | 0.05% | drop |
| 759 nt | 60,565 | 14,572 | 99.99% | 1,947 | 99.95% | keep |

#### Step 3. Alignment of the kept lengths

* MAFFT aligned the 1,947 unique proteins from 759-nt CDS with FFT-NS-2 and
  produced 252 aa alignment columns. MAFFT took 3.9 seconds.
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* Each protein has 252 aa residues, so no protein has a gap.
* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/M1/length_share_above_0.02/`.

#### Step 4. Gap patterns above 2%

| Gap pattern | AA gap columns | AA residues | Unique proteins | Unique CDS | % of unique CDS | Isolates | Step 4 |
|---|---|---:|---:|---:|---:|---:|---|
| `gap_pat001` | None | 252 | 1,947 | 14,572 | 100% | 60,565 | keep |

* `% of unique CDS` divides each gap pattern's unique CDS by 14,572, the unique
  CDS kept in step 2.

#### Step 5. Realignment

Not applicable: step 4 drops no gap pattern.

### 4. By host, subtype and year

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

### Decision

* Keep 759 nt CDS (253 aa proteins).
  * Within the 759 nt CDS, keep all seqs (same pattern).


## NS1 (Segment 8)

NS1 sequences are aligned in amino-acid (aa) space. The corresponding CDS
sequences are retained so that each aa alignment column can later be projected
back to the codon that encoded that aa residue. The CDS sequences are not
aligned in this analysis.

### 1. Inputs

* **Complete CDS.** 60,654 of the 60,751 filtered isolates (99.8%) have a
  complete NS1 CDS. The other 97 have an incomplete CDS and are excluded.
* **Unique sequences.** The 60,654 complete CDS hold 18,107 unique CDS at 16 CDS
  lengths. They translate to 10,130 unique proteins.

### 2. Retention rule

#### Steps 1 and 2. Complete CDS and lengths above 2%

The CDS lengths in this table identify the nt sequences from which the proteins
were translated.

| CDS length | Isolates | Unique CDS | % of unique CDS | Unique proteins | % of unique proteins | Step 2 |
|---|---:|---:|---:|---:|---:|---|
| 654 nt | 389 | 236 | 1.30% | 207 | 2.04% | drop |
| 660 nt | 25,506 | 8,243 | 45.52% | 4,601 | 45.42% | keep |
| 672 nt | 806 | 235 | 1.30% | 144 | 1.42% | drop |
| 678 nt | 341 | 246 | 1.36% | 205 | 2.02% | drop |
| 693 nt | 33,553 | 9,097 | 50.24% | 4,926 | 48.63% | keep |
| 11 other lengths | 59 | 50 | 0.28% | 47 | 0.46% | drop |

#### Step 3. Alignment of the kept lengths

* MAFFT aligned the 9,527 unique proteins from 660-nt and 693-nt CDS with
  FFT-NS-2 and produced 230 aa alignment columns. MAFFT took 52.6 seconds.
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* A protein from a 693-nt CDS has 230 aa residues, so it has no gap. A protein
  from a 660-nt CDS has 219 aa residues and gaps in 11 of the 230 columns.
* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/NS1/length_share_above_0.02/`.

#### Step 4. Gap patterns above 2%

| Gap pattern | CDS length | AA gap columns | AA residues | Unique proteins | Unique CDS | % of unique CDS | Isolates | Step 4 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `gap_pat001` | 693 nt | None | 230 | 4,926 | 9,097 | 52.46% | 33,553 | keep |
| `gap_pat002` | 660 nt | 220–230 | 219 | 4,601 | 8,243 | 47.54% | 25,506 | keep |

* `% of unique CDS` divides each gap pattern's unique CDS by 17,340, the unique
  CDS kept in step 2.

#### Step 5. Realignment

Not applicable: step 4 drops no gap pattern.

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

### Decision

* Keep 660 nt CDS (220 aa proteins).
  * Within the 660 nt CDS, keep the only gap pattern.
* Keep 693 nt CDS (231 aa proteins).
  * Within the 693 nt CDS, keep the only gap pattern.


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
