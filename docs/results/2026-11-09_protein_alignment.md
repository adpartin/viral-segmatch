# Protein alignment for codon features

```yaml

# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-11-09
populations:    Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025; Human-H3N2-2024
proteins:       PB2; PB1; PA; HA; NP; NA; M1; NS1
source:         data/processed/flu/July_2025/cds_dna_final.parquet
scripts:        src/analysis/align_proteins.py; src/analysis/summarize_protein_alignment.py
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

For each protein, the document reports population coverage, gap-pattern support,
and the final alignment. The goal is to preserve well-supported sequence forms
without adding records or alignment columns whose small coverage does not justify
their complexity. Projection from alignment columns to codons is not implemented.


## Why this is measured

Per-site codon features need the same sites in every record, so that position
200 represents the same place in each record. For each protein, there are two
options to obtain those sites:

1. **Pinned CDS length.** Use the fixed length for the protein from
   `conf/virus/flu.yaml`, then keep only complete CDS at that length.
   `check_cds_length` accepts the pin only when it equals the observed mode and
   at least 90% of unique complete CDS have the pinned length. The mode checks
   the configured pin; it does not set it.
2. **Protein alignment.** Align translated proteins to estimate which sites
   correspond across records. Under the proposed codon mapping, a column with
   no codon would be `GAP` before a record's stop or `AFTER_STOP` after it.


## The population

The eight Task 2 protein sections use one population,
set in `docs/plans/2026-10-08_paper_plan.md`. Filters: host `[Human, Swine,
Chicken, Duck]`, subtype `[H3N2, H1N1, H5N1]`, year `2010–2025`. `host` is a
host group, so `Duck` covers 54 animal names and `Chicken` covers 3.

## How to read these sections

Each Task 2 protein section has five parts: inputs, retention rule, breakdown by
host, subtype and year, findings, and decision. A **gap pattern** is one
protein's full set of gap columns. An internal gap lies between protein symbols;
see the [glossary](../methods/glossary.md#per-site-features) for gap terminology.

This alignment does not:

* Place CDS stop codons. The terminal `*` marker is removed before MAFFT, so no
  column holds a stop codon.
* Show that a column represents the same biological position in every record.
  The alignment is an estimate of positional correspondence.
* Recover bases absent from a truncated record. These runs use complete CDS
  only.


## Method

`src/analysis/align_proteins.py` prepares, aligns, validates and inspects each
set of proteins. These stages use separate commands. We first aligned all
complete CDS lengths, then the lengths kept by the retention rule. When the
rule dropped a gap pattern, we prepared and aligned the remaining proteins
again.

* **Prepare.** Select complete CDS for the protein and population, applying a
  length or gap-pattern filter when needed. Check each unique CDS against its
  recorded length, protein, terminal stop and protein hash. Remove the terminal
  `*` from each unique protein for alignment; retain any `X` symbols. The stored
  CDS and protein keep their stop codon and marker.
* **Align.** Align the unique proteins with MAFFT and record the command,
  version, runtime and file hashes. The Task 2 runs used MAFFT v7.526 with
  `--retree 2 --maxiterate 0 --reorder`, which gives FFT-NS-2 and writes rows
  in guide-tree order.
* **Validate.** Check file hashes, protein IDs and alignment widths. Remove
  gaps from each aligned protein and check that it matches its input exactly.
* **Inspect.** Record gap positions and count the proteins, CDS and isolates
  in each pattern. Name patterns by rank: most unique proteins first, then
  most isolates, then the gap-column string to break any remaining ties.
  Names such as `gap_pat001` apply only to the alignment that produced them.

The protein-to-CDS mapping is retained for later codon features. FASTA headers
use `prot_hash`, the hash of the stored protein including its terminal `*`;
`protein_input_hashes.csv` records hashes of the stop-free alignment inputs.
An `X` represents a codon that does not resolve to one amino acid, so it gives
MAFFT less information for placing nearby gaps.

The inspection also writes labelled copies of the alignment for viewing.
`src/analysis/summarize_protein_alignment.py` writes group, gap-pattern and
column-occupancy summaries.

The eight all-length Task 2 alignments were first run without `--reorder` and
rerun with it on 2026-11-12. Every aligned row, width and summary table was
identical; only the row order changed.


## Summary across proteins

The table below gives the complete CDS inputs for each protein on the Task 2
population. Counts include all observed lengths before applying the retention
rule. The last two columns compare the original and final protein alignment
widths; they do not include stop codons.

| Segment | Protein | Complete isolates | Unique CDS | Unique proteins | CDS lengths | All-length AA columns | Final AA columns |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 60,474 | 36,939 | 15,290 | 18 | 766 | 759 |
| 2 | PB1 | 56,093 | 34,642 | 13,220 | 18 | 776 | 758 |
| 3 | PA | 60,396 | 35,659 | 15,939 | 14 | 744 | 716 |
| 4 | HA | 60,282 | 35,509 | 19,841 | 10 | 612 | 581 |
| 5 | NP | 60,040 | 27,087 | 7,250 | 2 | 498 | 498 |
| 6 | NA | 59,856 | 31,157 | 17,267 | 16 | 543 | 493 |
| 7 | M1 | 60,566 | 14,573 | 1,948 | 2 | 252 | 252 |
| 8 | NS1 | 60,654 | 18,107 | 10,130 | 16 | 240 | 230 |

The retention rule keeps the records below. Percentages use the isolates with
a complete CDS for that protein as the denominator. A kept length can still
lose records when the rule removes uncommon gap patterns.

| Protein | Kept CDS lengths (nt) | Kept isolates | % kept |
|---|---|---:|---:|
| PB2 | 2280 | 60,412 | 99.90% |
| PB1 | 2274, 2277 | 56,040 | 99.91% |
| PA | 2151 | 60,332 | 99.89% |
| HA | 1701, 1704 | 59,807 | 99.21% |
| NP | 1497 | 60,039 | >99.99% |
| NA | 1410 | 59,129 | 98.79% |
| M1 | 759 | 60,565 | >99.99% |
| NS1 | 660, 693 | 59,059 | 97.37% |

The final alignments have no gaps for PB2, PA, NP and M1. PB1 and NS1 retain
two lengths that differ only at the end of the alignment. HA and NA retain
different gap patterns even among proteins of the same length, so length
alone does not give them a shared set of positions.

Pinning one CDS length would miss some supported forms. PB1 and NS1 have no
configured pin, and their modal lengths fall below the 90% floor on this
population. HA clears that floor overall, but its 1701-nt pin would keep only
13 of 3,754 complete H5N1 records; the retention rule also keeps the main
1704-nt form.

The next table shows coverage by year. "Filtered isolates" counts all isolates
in the Task 2 population that year; each protein column counts those kept by
the retention rule. The last row uses a different denominator: that year's
isolates with a complete CDS for the protein.

| Year | Filtered isolates | PB2 | PB1 | PA | HA | NP | NA | M1 | NS1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2010 | 1,553 | 1,540 | 1,542 | 1,545 | 1,498 | 1,505 | 1,517 | 1,549 | 1,496 |
| 2011 | 1,476 | 1,458 | 1,470 | 1,475 | 1,393 | 1,471 | 1,406 | 1,472 | 1,397 |
| 2012 | 1,649 | 1,633 | 1,634 | 1,645 | 1,615 | 1,638 | 1,587 | 1,641 | 1,579 |
| 2013 | 1,825 | 1,808 | 1,813 | 1,821 | 1,767 | 1,819 | 1,744 | 1,825 | 1,757 |
| 2014 | 1,471 | 1,462 | 1,461 | 1,467 | 1,452 | 1,467 | 1,373 | 1,469 | 1,423 |
| 2015 | 2,179 | 2,170 | 2,168 | 2,177 | 2,159 | 2,177 | 2,046 | 2,177 | 2,049 |
| 2016 | 3,319 | 3,316 | 3,313 | 3,317 | 3,305 | 3,318 | 3,266 | 3,319 | 3,264 |
| 2017 | 4,145 | 4,129 | 4,123 | 4,139 | 4,107 | 4,128 | 4,077 | 4,144 | 4,003 |
| 2018 | 3,509 | 3,497 | 3,481 | 3,502 | 3,460 | 3,499 | 3,468 | 3,507 | 3,434 |
| 2019 | 5,268 | 5,230 | 5,200 | 5,147 | 5,204 | 5,025 | 5,156 | 5,221 | 4,531 |
| 2020 | 1,737 | 1,628 | 1,527 | 1,562 | 1,657 | 1,624 | 1,527 | 1,635 | 1,650 |
| 2021 | 1,540 | 1,536 | 1,536 | 1,516 | 1,513 | 1,539 | 1,514 | 1,539 | 1,526 |
| 2022 | 7,160 | 7,148 | 7,003 | 7,144 | 7,094 | 7,134 | 7,111 | 7,156 | 7,117 |
| 2023 | 5,145 | 5,127 | 4,822 | 5,135 | 4,957 | 5,052 | 4,936 | 5,142 | 5,099 |
| 2024 | 11,108 | 11,074 | 8,668 | 11,082 | 11,038 | 11,058 | 10,870 | 11,103 | 11,083 |
| 2025 | 7,667 | 7,656 | 6,279 | 7,658 | 7,588 | 7,585 | 7,531 | 7,666 | 7,651 |
| Lowest yearly % kept | | 99.82% (2023) | 99.81% (2023) | 98.63% (2021) | 97.25% (2023) | 99.93% (2011) | 93.98% (2015) | 99.93% (2011) | 86.22% (2019) |

No year loses all of its records for any protein. Coverage is lowest for NS1
in 2019, when the rule keeps 86.22% of isolates with a complete NS1 CDS.
The protein sections give the host and subtype breakdowns.


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

Steps 3 and 4 have been run for all eight proteins. Step 5 has realigned PB2,
PB1, HA and NA. It does not apply to PA, NP, M1 and NS1, because their step 4
drops no gap pattern, so their step-3 alignments are final.
Using the step-3 alignments, thresholds from 1.36% to 4.58% give the same
retained lengths and gap patterns across all eight proteins. The largest share
dropped is NS1's 678-nt length, at 1.36%
of NS1's unique CDS. The smallest share kept is HA's H5 gap pattern at 1704 nt,
at 4.59% of HA's unique CDS at the kept lengths.

The rule pools all years. It drops NS1's 672-nt length, which holds 698 of the
5,255 isolates with a complete NS1 CDS in 2019.


## Artifacts

Every Task 2 run wrote to
`results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/<protein>/`,
one directory per protein.

`align_proteins.py` wrote the selection, alignment, validation and gap
locations. `summarize_protein_alignment.py` wrote `group_length_summary.csv`,
`gap_pattern_summary.csv` and `column_occupancy.csv`. The manifests record file
hashes and code state.

The eight Task 2 manifests record a MAFFT runtime. The eight runs ran at the
same time, one thread each, and took (m:ss): PB2 7:19, PB1 5:37, PA 7:26, HA
9:23, NP 1:10, NA 5:52, M1 0:04 and NS1 1:01.


## PB2 (Segment 1)

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
  produced 760 aa alignment columns. MAFFT took 7:01 (m:ss).
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

* MAFFT realigned the 15,229 unique proteins kept in step 4 with FFT-NS-2 and
  produced 759 aa alignment columns. MAFFT took 7:07 (m:ss).
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* Each protein has 759 aa residues, so no protein has a gap. The alignment has
  one gap pattern, the one with no gaps.
* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/PB2/length_share_above_0.02/gap_pattern_share_above_0.02/`.

### 3. By host, subtype and year

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

* Every host–subtype–year group with a complete PB2 CDS keeps at least one
  isolate.

### 4. Findings

* The retention rule keeps 60,412 of the 60,474 isolates with a complete PB2 CDS
  (99.90%). They hold 36,878 unique CDS and 15,229 unique proteins.
* Step 2 drops the 17 CDS lengths other than 2280 nt. They hold 60 unique CDS and
  61 isolates.
* Step 4 drops one protein, carried by one isolate. Its gap is at column 577 of
  the step-3 alignment, where every other protein has its gap at column 570.
* Step 5 realigns the 15,229 kept proteins into 759 columns, with no gap in any
  protein.

### 5. Decision

* Keep the 2280-nt CDS, whose proteins have 759 aa residues.
  * Within them, keep `gap_pat001` of the step-3 alignment, which gaps column
    570. Drop `gap_pat002`, the single protein that gaps column 577 instead.
* Use the step-5 realignment as the final alignment.


## PB1 (Segment 2)

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
  FFT-NS-2 and produced 762 aa alignment columns. MAFFT took 5:44 (m:ss).
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

* MAFFT realigned the 13,167 unique proteins kept in step 4 with FFT-NS-2 and
  produced 758 aa alignment columns. MAFFT took 5:45 (m:ss).
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* A protein from a 2277-nt CDS has 758 aa residues, so it has no gap. A protein
  from a 2274-nt CDS has 757 aa residues and one gap, in the last column, 758.
  The alignment has these two gap patterns only.
* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/PB1/length_share_above_0.02/gap_pattern_share_above_0.02/`.

### 3. By host, subtype and year

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

* Every host–subtype–year group with a complete PB1 CDS keeps at least one
  isolate.

### 4. Findings

* The retention rule keeps 56,040 of the 56,093 isolates with a complete PB1 CDS
  (99.91%). They hold 34,589 unique CDS and 13,167 unique proteins.
* Step 2 drops the 16 CDS lengths other than 2274 and 2277 nt. They hold 50
  unique CDS and 50 isolates.
* Step 4 drops 3 gap patterns, each held by one protein and one isolate.
* Step 5 realigns the 13,167 kept proteins into 758 columns. A 2277-nt protein
  has no gap, and a 2274-nt protein's only gap is the last column, so the two
  lengths align position by position.
* Both lengths are needed across years. The 2277-nt share of Human-H3N2 isolates
  at the two lengths was 6.9% in 2022, 47.5% in 2023, 99.6% in 2024 and 99.5% in
  2025.

### 5. Decision

* Keep the 2274-nt and 2277-nt CDS, whose proteins have 757 and 758 aa residues.
  * Within them, keep `gap_pat001` (2274 nt) and `gap_pat002` (2277 nt) of the
    step-3 alignment. Drop the 3 single-protein patterns.
* Use the step-5 realignment as the final alignment.


## PA (Segment 3)

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
  produced 716 aa alignment columns. MAFFT took 7:28 (m:ss).
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

Not applicable: step 4 drops no gap pattern, so the step 3 alignment is the
final alignment.

### 3. By host, subtype and year

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

* Every host–subtype–year group with a complete PA CDS keeps at least one
  isolate.

### 4. Findings

* The retention rule keeps 60,332 of the 60,396 isolates with a complete PA CDS
  (99.89%). They hold 35,609 unique CDS and 15,897 unique proteins.
* Step 2 drops the 13 CDS lengths other than 2151 nt. They hold 50 unique CDS and
  64 isolates.
* Step 4 drops no gap pattern, so the step-3 alignment is final: 716 columns,
  with no gap in any protein.

### 5. Decision

* Keep the 2151-nt CDS, whose proteins have 716 aa residues.
  * Within them, keep `gap_pat001`, the only gap pattern of the step-3 alignment.
* Use the step-3 alignment as the final alignment.


## HA (Segment 4)

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
  FFT-NS-2 and produced 590 aa alignment columns. MAFFT took 8:45 (m:ss).
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

* MAFFT realigned the 19,470 unique proteins kept in step 4 with FFT-NS-2 and
  produced 581 aa alignment columns. MAFFT took 8:48 (m:ss).
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* A protein from a 1701-nt CDS has 566 aa residues, so it has gaps in 15 of the
  581 columns. A protein from a 1704-nt CDS has 567 aa residues and gaps in 14.

| Gap pattern | CDS length | AA gap columns | Unique proteins | Unique CDS | Isolates |
|---|---|---|---:|---:|---:|
| `gap_pat001` (H3) | 1701 nt | 70–73, 102, 115, 132–134, 155, 287, 352–354, 566 | 10,274 | 18,560 | 32,783 |
| `gap_pat002` (H1) | 1701 nt | 7–9, 19–24, 79–81, 352–354 | 8,326 | 14,926 | 23,428 |
| `gap_pat003` (H5) | 1704 nt | 5, 7–9, 19–24, 79–81, 149 | 870 | 1,612 | 3,596 |

* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/HA/length_share_above_0.02/gap_pattern_share_above_0.02/`.

### 3. By host, subtype and year

| Host | Subtype | Complete HA isolates | Kept isolates | % kept |
|---|---|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 100% |
| Chicken | H3N2 | 17 | 17 | 100% |
| Chicken | H5N1 | 1,816 | 1,782 | 98.13% |
| Duck | H1N1 | 265 | 265 | 100% |
| Duck | H3N2 | 217 | 215 | 99.08% |
| Duck | H5N1 | 1,886 | 1,774 | 94.06% |
| Human | H1N1 | 20,013 | 19,991 | 99.89% |
| Human | H3N2 | 29,422 | 29,420 | 99.99% |
| Human | H5N1 | 37 | 28 | 75.68% |
| Swine | H1N1 | 3,454 | 3,169 | 91.75% |
| Swine | H3N2 | 3,138 | 3,130 | 99.75% |
| Swine | H5N1 | 15 | 14 | 93.33% |
| Total | | 60,282 | 59,807 | 99.21% |

* 7 host–subtype–year groups with a complete HA CDS keep none of their isolates,
  24 in all:
  - Swine-H1N1 in 2025: 9 isolates.
  - Human-H5N1 in 2011–2014: 8 isolates.
  - Duck-H5N1 in 2010: 6 isolates.
  - Swine-H5N1 in 2011: 1 isolate.

### 4. Findings

* The retention rule keeps 59,807 of the 60,282 isolates with a complete HA CDS
  (99.21%). They hold 35,098 unique CDS and 19,470 unique proteins.
* Step 2 drops the 8 CDS lengths other than 1701 and 1704 nt. They hold 374
  unique CDS and 427 isolates. The largest is 1695 nt, with 244 isolates, 217 of
  them Swine-H1N1.
* Step 4 drops 12 gap patterns, which hold 37 unique proteins and 48 isolates.
* Step 5 realigns the 19,470 kept proteins into 581 columns. H3 and H1 proteins
  both have 566 aa residues, but their gaps fall in different columns, so equal
  length does not make their residues correspond position by position.

### 5. Decision

* Keep the 1701-nt and 1704-nt CDS, whose proteins have 566 and 567 aa residues.
  * Within them, keep `gap_pat001` (H3), `gap_pat002` (H1) and `gap_pat003` (H5)
    of the step-3 alignment. Drop the 12 other patterns.
* Use the step-5 realignment as the final alignment.


## NP (Segment 5)

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
  produced 498 aa alignment columns. MAFFT took 1:10 (m:ss).
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

Not applicable: step 4 drops no gap pattern, so the step 3 alignment is the
final alignment.

### 3. By host, subtype and year

| Host | Subtype | Complete NP isolates | Kept isolates | % kept |
|---|---|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 100% |
| Chicken | H3N2 | 17 | 17 | 100% |
| Chicken | H5N1 | 1,819 | 1,819 | 100% |
| Duck | H1N1 | 263 | 263 | 100% |
| Duck | H3N2 | 216 | 216 | 100% |
| Duck | H5N1 | 1,895 | 1,895 | 100% |
| Human | H1N1 | 19,824 | 19,824 | 100% |
| Human | H3N2 | 29,177 | 29,177 | 100% |
| Human | H5N1 | 36 | 36 | 100% |
| Swine | H1N1 | 3,608 | 3,608 | 100% |
| Swine | H3N2 | 3,167 | 3,166 | 99.97% |
| Swine | H5N1 | 16 | 16 | 100% |
| Total | | 60,040 | 60,039 | >99.99% |

* Every host–subtype–year group with a complete NP CDS keeps at least one
  isolate.

### 4. Findings

* The retention rule keeps 60,039 of the 60,040 isolates with a complete NP CDS
  (>99.99%). They hold 27,086 unique CDS and 7,249 unique proteins.
* Step 2 drops the 1461-nt length, which holds 1 unique CDS and 1 isolate.
* Step 4 drops no gap pattern, so the step-3 alignment is final: 498 columns,
  with no gap in any protein.

### 5. Decision

* Keep the 1497-nt CDS, whose proteins have 498 aa residues.
  * Within them, keep `gap_pat001`, the only gap pattern of the step-3 alignment.
* Use the step-3 alignment as the final alignment.


## NA (Segment 6)

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
  produced 514 aa alignment columns. MAFFT took 5:33 (m:ss).
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

* MAFFT realigned the 16,825 unique proteins kept in step 4 with FFT-NS-2 and
  produced 493 aa alignment columns. MAFFT took 5:23 (m:ss).
* Removing alignment gaps (`-`) from every aligned sequence recovered its exact
  stop-free protein input.
* Each protein has 469 aa residues, so it has gaps in 24 of the 493 columns.

| Gap pattern | AA gap columns | Unique proteins | Unique CDS | Isolates |
|---|---|---:|---:|---:|
| `gap_pat001` (N2) | 92–109, 183, 430–433, 493 | 8,520 | 16,092 | 32,498 |
| `gap_pat002` (N1) | 40–43, 54–60, 77–83, 326, 354–355, 357, 412, 460 | 8,305 | 14,518 | 26,631 |

* The outputs are in
  `results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/NA/length_share_above_0.02/gap_pattern_share_above_0.02/`.

### 3. By host, subtype and year

| Host | Subtype | Complete NA isolates | Kept isolates | % kept |
|---|---|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 100% |
| Chicken | H3N2 | 17 | 16 | 94.12% |
| Chicken | H5N1 | 1,817 | 1,583 | 87.12% |
| Duck | H1N1 | 263 | 263 | 100% |
| Duck | H3N2 | 217 | 215 | 99.08% |
| Duck | H5N1 | 1,890 | 1,571 | 83.12% |
| Human | H1N1 | 19,686 | 19,630 | 99.72% |
| Human | H3N2 | 29,162 | 29,138 | 99.92% |
| Human | H5N1 | 36 | 26 | 72.22% |
| Swine | H1N1 | 3,607 | 3,563 | 98.78% |
| Swine | H3N2 | 3,148 | 3,113 | 98.89% |
| Swine | H5N1 | 11 | 9 | 81.82% |
| Total | | 59,856 | 59,129 | 98.79% |

* 19 host–subtype–year groups with a complete NA CDS keep none of their
  isolates, 202 in all:
  - Chicken-H5N1 in 2010–2014 and 2016–2018: 121 isolates.
  - Duck-H5N1 in 2013, 2018 and 2020: 68 isolates.
  - Human-H5N1 in 2010 and 2012–2015: 10 isolates.
  - Swine-H5N1 in 2011 and 2015: 2 isolates.
  - Chicken-H3N2 in 2015: 1 isolate.
* Human-H5N1-2011 is not in this list: its 1 isolate has no complete NA CDS.

### 4. Findings

* The retention rule keeps 59,129 of the 59,856 isolates with a complete NA CDS
  (98.79%). They hold 30,610 unique CDS and 16,825 unique proteins.
* Step 2 drops the 15 CDS lengths other than 1410 nt. They hold 527 unique CDS
  and 702 isolates. The largest are 1350 nt, with 505 isolates, all H5N1, and
  1413 nt, with 129.
* Step 4 drops 4 gap patterns, which hold 18 unique proteins and 25 isolates.
* Step 5 realigns the 16,825 kept proteins into 493 columns. N2 and N1 proteins
  both have 469 aa residues, but their gaps fall in different columns, so equal
  length does not make their residues correspond position by position.

### 5. Decision

* Keep the 1410-nt CDS, whose proteins have 469 aa residues.
  * Within them, keep `gap_pat001` (N2) and `gap_pat002` (N1) of the step-3
    alignment. Drop the 4 other patterns.
* Use the step-5 realignment as the final alignment.


## M1 (Segment 7)

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
  produced 252 aa alignment columns. MAFFT took 0:04 (m:ss).
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

Not applicable: step 4 drops no gap pattern, so the step 3 alignment is the
final alignment.

### 3. By host, subtype and year

| Host | Subtype | Complete M1 isolates | Kept isolates | % kept |
|---|---|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 100% |
| Chicken | H3N2 | 17 | 17 | 100% |
| Chicken | H5N1 | 1,824 | 1,824 | 100% |
| Duck | H1N1 | 265 | 265 | 100% |
| Duck | H3N2 | 217 | 217 | 100% |
| Duck | H5N1 | 1,900 | 1,900 | 100% |
| Human | H1N1 | 20,001 | 20,001 | 100% |
| Human | H3N2 | 29,456 | 29,456 | 100% |
| Human | H5N1 | 37 | 37 | 100% |
| Swine | H1N1 | 3,658 | 3,657 | 99.97% |
| Swine | H3N2 | 3,174 | 3,174 | 100% |
| Swine | H5N1 | 15 | 15 | 100% |
| Total | | 60,566 | 60,565 | >99.99% |

* Every host–subtype–year group with a complete M1 CDS keeps at least one
  isolate.

### 4. Findings

* The retention rule keeps 60,565 of the 60,566 isolates with a complete M1 CDS
  (>99.99%). They hold 14,572 unique CDS and 1,947 unique proteins.
* Step 2 drops the 726-nt length, which holds 1 unique CDS and 1 isolate.
* Step 4 drops no gap pattern, so the step-3 alignment is final: 252 columns,
  with no gap in any protein.

### 5. Decision

* Keep the 759-nt CDS, whose proteins have 252 aa residues.
  * Within them, keep `gap_pat001`, the only gap pattern of the step-3 alignment.
* Use the step-3 alignment as the final alignment.


## NS1 (Segment 8)

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
  FFT-NS-2 and produced 230 aa alignment columns. MAFFT took 0:53 (m:ss).
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

Not applicable: step 4 drops no gap pattern, so the step 3 alignment is the
final alignment.

### 3. By host, subtype and year

| Host | Subtype | Complete NS1 isolates | Kept isolates | % kept |
|---|---|---:|---:|---:|
| Chicken | H1N1 | 2 | 2 | 100% |
| Chicken | H3N2 | 17 | 16 | 94.12% |
| Chicken | H5N1 | 1,825 | 1,601 | 87.73% |
| Duck | H1N1 | 266 | 266 | 100% |
| Duck | H3N2 | 217 | 217 | 100% |
| Duck | H5N1 | 1,899 | 1,743 | 91.79% |
| Human | H1N1 | 20,115 | 20,102 | 99.94% |
| Human | H3N2 | 29,423 | 28,443 | 96.67% |
| Human | H5N1 | 37 | 26 | 70.27% |
| Swine | H1N1 | 3,662 | 3,498 | 95.52% |
| Swine | H3N2 | 3,175 | 3,131 | 98.61% |
| Swine | H5N1 | 16 | 14 | 87.50% |
| Total | | 60,654 | 59,059 | 97.37% |

* 13 host–subtype–year groups with a complete NS1 CDS keep none of their
  isolates, 118 in all:
  - Chicken-H5N1 in 2010–2012 and 2014: 74 isolates.
  - Duck-H5N1 in 2013: 31 isolates.
  - Human-H5N1 in 2010–2015: 11 isolates.
  - Swine-H5N1 in 2011 and 2015: 2 isolates.

### 4. Findings

* The retention rule keeps 59,059 of the 60,654 isolates with a complete NS1 CDS
  (97.37%). They hold 17,340 unique CDS and 9,527 unique proteins.
* Step 2 drops the 14 CDS lengths other than 660 and 693 nt. They hold 767
  unique CDS and 1,595 isolates. The largest are 672 nt, with 806 isolates, 804
  of them Human-H3N2; 654 nt, with 389; and 678 nt, with 341, 340 of them H5N1.
* Step 4 drops no gap pattern, so the step-3 alignment is final: 230 columns. A 693-nt
  protein has no gap, and a 660-nt protein's gaps are the last 11 columns, so
  the two lengths align position by position.

### 5. Decision

* Keep the 660-nt and 693-nt CDS, whose proteins have 219 and 230 aa residues.
  * Within them, keep both gap patterns of the step-3 alignment.
* Use the step-3 alignment as the final alignment.


## NS1 on Human-H3N2-2024 (Segment 8, Task 1 population)

The first alignment run, on 2026-11-05, used the Task 1 population to develop
the method. Its outputs are in
`results/flu/July_2025/cds_alignment_pilot/Human-H3N2-2024/NS1/`. The run
predated `mafft_seconds`, so it has no recorded runtime. It used `--auto` to
select FFT-NS-2; no second strategy was tested.

The directory also holds the downstream contig check, written by
`src/analysis/audit_ns1_downstream_contig.py` as `ns1_downstream_contig.csv`,
`ns1_downstream_summary.csv` and `ns1_downstream_manifest.json`.

A trailing gap is a `-` after the last protein symbol; an internal gap is
between protein symbols. `res` in the schematic below stands for one amino acid
or `X`. An `X` does not identify a single amino acid and gives MAFFT less
information for placing nearby gaps.

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
  there is no pin for `check_cds_length` to check. On the Task 2 population,
  the modal share at 693 nt is 0.502.
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
* The 693-nt length clears the pinning floor in this population. The alignment
  was run to establish the method.

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
* The Task 2 runs used FFT-NS-2. We did not test a second strategy. Agreement
  between two strategies would not establish that a placement is correct.
* The one-residue columns and the uncommon gap patterns have not been reviewed
  sequence by sequence.
