# Protein alignment for codon features

```yaml
# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-11-09
populations:    Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025; Human+Pig-H1N1+H3N2-yr2010-2025; Human-H3N2-2024
proteins:       NS1
source:         data/processed/flu/July_2025/cds_dna_final.parquet
scripts:        src/analysis/align_proteins.py; src/analysis/summarize_protein_alignment.py; src/analysis/audit_ns1_downstream_contig.py
artifacts:      results/flu/July_2025/cds_alignment_pilot/<population>/<protein>/
```

## Why this is measured

Per-site codon features need one pinned CDS length per protein, so that every
record has the same number of columns. A population with several CDS lengths for
a protein has no such pin. Aligning the proteins is the alternative: it gives a
set of columns that records of different lengths can share.

The alignment is also what decides whether a protein can be modelled at all on a
given population. `check_cds_length` needs `frac at mode` to reach 0.90 before it
will accept a pin, where `frac at mode` is the share of unique complete CDS whose
length equals the mode. Where no length reaches that, the choice is to align or
to drop the protein.

## How to read these sections

Each section covers one protein on one population, and carries the same six
parts: Inputs, Lengths and gaps, Stop codon columns, By host and subtype,
Findings and Artifacts. Where a part has nothing to report, it says so in one
sentence rather than being left out, so two sections can be read side by side. A
section may add a further part for a check specific to it.

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

The same five steps for every protein and population, from
`src/analysis/align_proteins.py`:

1. Select complete CDS for the protein and population, keeping all observed
   lengths.
2. Check every selected unique CDS: it translates to its recorded `prot_seq`,
   ends with exactly one `*`, has no internal `*`, and its length is three times
   the protein length including the `*`. Remove the terminal `*` only from the
   protein sent to MAFFT and leave the CDS unchanged. Keep `X` symbols and count
   them by CDS length.
3. Align the unique proteins with MAFFT `--auto`, recording the version, command,
   selected strategy and input hashes.
4. Check that removing `-` from each aligned protein recovers the exact input.
5. Record every gap position.

`src/analysis/summarize_protein_alignment.py` then writes
`group_length_summary.csv`, `gap_pattern_summary.csv` and
`column_occupancy.csv`.

The saved protein is the exact MAFFT input, which is what step 4 compares
against. With the options used here, [MAFFT would remove `*`
anyway](https://mafft.cbrc.jp/alignment/software/anysymbol.html).

An `X` means the codon does not translate to a single amino acid. It does not
remove the codon and does not create a gap. It does give the aligner less
information at that position. Whether any gap placement falls next to an `X` has
not been checked, except in the Human-H3N2-2024 section where it is reported.

## Summary across proteins and populations

| Protein | Population | Isolates | Unique CDS | Unique proteins | CDS lengths | Columns | `frac at mode` | Largest internal gap | One-residue columns |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| NS1 | Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025 | 60,654 | 18,107 | 10,130 | 16 | 240 | 693 nt, 0.502 | columns 81–85, 5 codons, 206 proteins | 74, 112, 132 |
| NS1 | Human+Pig-H1N1+H3N2-yr2010-2025 | 56,417 | 16,540 | 9,039 | 14 | 240 | 660 nt, 0.499 | columns 77–80, 4 codons, 1 protein | 74, 112, 132 |
| NS1 | Human-H3N2-2024 | 5,337 | 1,132 | 560 | 3 | 230 | 693 nt, 0.991 | None | None |

"Largest internal gap" gives the longest run of neighbouring internal gap columns
any protein has, as column numbers rather than a count of columns, followed by how
long the run is and how many proteins carry it. `columns 81–85, 5 codons` means
five gaps, one in each of the columns numbered 81 to 85. Each section lists all of
its internal gaps; the table gives only the longest.

A **one-residue column** is an alignment column where exactly one protein has a
protein symbol and every other protein has a gap.

NS1 has a usable pin on Human-H3N2-2024 at 0.991 and no pin on either wider
population. Widening the population removes the pin. The two wide populations
also differ in which length holds the largest share of unique CDS, 660 nt in
Human+Pig against 693 nt in the Task 2 population, although 693 nt has the most
isolates in both.

## NS1

### NS1 on Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025

This is the population Task 2 uses, set in
`docs/plans/2026-10-08_paper_plan.md`. Filters: host `[Human, Swine, Chicken,
Duck]`, subtype `[H3N2, H1N1, H5N1]`, year `2010–2025`. `host` is a host group,
so `Duck` covers 54 animal names and `Chicken` covers 3.

#### Inputs

Of 60,751 filtered isolates, 60,654 (99.8%) had a complete CDS. They contained
18,107 unique CDS and 10,130 unique proteins across 16 CDS lengths and 147
host–subtype–year groups. 957 isolates and 551 unique proteins contain an `X`.

No single pinned length is available. `frac at mode` is 0.502, at 693 nt, then
0.455 at 660 nt, against the 0.90 `check_cds_length` needs.

MAFFT 7.526 with `--auto` selected FFT-NS-2 and produced 240 columns. Removing
`-` recovered the exact input protein for all 10,130.

#### Lengths and gaps

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

#### Stop codon columns

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

A codon feature table for this population needs 241 columns, not 240. The 714-nt
form fills every protein column, so its stop codon needs one more. This mapping
is not implemented.

The 693-nt and 678-nt forms put their stop codons in the same column, 234. That
follows from the internal deletion: the two forms differ in the middle, not at
the end.

#### By host and subtype

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

#### Findings

* The 678-nt internal deletion is the first internal gap in this protein carried
  by more than a handful of records: 205 unique proteins over 341 isolates, plus
  the single 699-nt isolate, so 206 proteins and 342 isolates have a gap at
  columns 81–85. The Human-H3N2-2024 pilot had no internal gaps, and the
  Human+Pig population's only internal gap run, at columns 77–80, was carried
  by one protein. Under the proposed codon feature mapping these 5 columns take the
  `GAP` state, not `AFTER_STOP`, because they sit between protein symbols rather
  than after a stop codon.
* Adding the birds and H5N1 does not add a third common length. Chicken-H5N1 is
  1,600 of 1,825 at 693 nt and Duck-H5N1 is 1,739 of 1,899, the same form as
  Human-H3N2. They add the 678-nt form and more 693-nt isolates, which rise from
  29,690 in the Human+Pig population to 33,553 here.
* The 660-nt and 693-nt forms cover 59,059 of the 60,654 complete records
  (97.4%). Adding 654, 672 and 678 nt reaches 60,595 (99.9%). The 714-nt form
  accounts for 16 isolates and the ten other lengths for 43.
* Columns 74, 112 and 132 each hold a protein symbol in exactly one protein; the
  other 10,129 have a gap. These are the same three columns as the Human+Pig
  population.
* Columns 234–240 hold a protein symbol in only 12 proteins, from 17 isolates,
  which are the 714-nt and 699-nt forms. 188 of the 240 columns hold a protein
  symbol in every protein.
* There are 19 gap patterns over the 16 CDS lengths, against 16 patterns in the
  Human+Pig population. The 3 new ones are the 678-nt form, a second 687-nt
  pattern with internal gaps at columns 79–80, and the 699-nt form. The last two are
  one isolate each.
* Columns 81–85 are the only internal gap columns new to this population. The
  columns 77–80 in the 648-nt form and the 5 leading gaps in the 645-nt form were
  already in the Human+Pig population. The new gap run at columns 79–80 falls inside columns
  that were already internal gaps there, and 2 proteins over 4 isolates carry it.
* Still open before codon features are built: whether to keep the 43 isolates
  outside the six main length forms, and whether to keep the three one-residue
  columns. Uncommon length alone does not establish an error.

#### Artifacts

`results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/NS1/`.
`align_proteins.py` wrote the selection, alignment, validation and gap
locations. `summarize_protein_alignment.py` wrote `group_length_summary.csv`,
`gap_pattern_summary.csv` and `column_occupancy.csv`. The manifests record file
hashes and code state.

### NS1 on Human+Pig-H1N1+H3N2-yr2010-2025

This population was a candidate for Task 2 and is not the one Task 2 uses.
Filters: host `[Human, Pig]`, subtype `[H3N2, H1N1]`, year `2010–2025`. It
predates `host` holding the host group, so `Pig` here is the animal name in what
is now the `Swine` group.

#### Inputs

Of 56,512 filtered isolates, 56,417 (99.8%) had a complete CDS. They contained
16,540 unique CDS and 9,039 unique proteins across 14 CDS lengths and 63
host–subtype–year groups. 938 isolates and 532 unique proteins contain an `X`.

No single pinned length is available. `frac at mode` is 0.499, at 660 nt, against
the 0.90 `check_cds_length` needs.

MAFFT 7.526 with `--auto` selected FFT-NS-2 and produced 240 columns. Removing
`-` recovered the exact input protein for all 9,039.

#### Lengths and gaps

| CDS length | Isolates | Unique CDS | Unique proteins | Internal gap columns | Trailing gap columns |
|---|---:|---:|---:|---|---|
| 654 nt | 339 | 219 | 192 | 74, 112, 132 | 221–240 |
| 660 nt | 25,526 | 8,248 | 4,604 | 74, 112, 132 | 223–240 |
| 672 nt | 806 | 235 | 144 | 74, 112, 132 | 227–240 |
| 693 nt | 29,690 | 7,791 | 4,055 | 74, 112, 132 | 234–240 |
| 714 nt | 15 | 10 | 10 | 74, 112, 132 | None |
| Nine other lengths | 41 | 37 | 34 | Varies | Varies |

#### Stop codon columns

```text
Column:          222   223–233      234–240
660 nt:          res   - ... -      - ... -
693 nt:          res   res ... res  - ... -
714 nt:          res   res ... res  res ... res
```

For a 660-nt CDS the last protein symbol is in column 222, so column 223 is its
first trailing `-` and would hold its stop codon. Later empty columns would be
`AFTER_STOP`. This mapping is not implemented.

#### By host and subtype

"Other lengths" covers the nine lengths not listed separately.

| Host | Subtype | Complete NS1 isolates | 654 nt | 660 nt | 672 nt | 693 nt | 714 nt | Other lengths |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Human | H1N1 | 20,141 | 6 | 19,698 | 2 | 430 | 0 | 5 |
| Human | H3N2 | 29,439 | 161 | 124 | 804 | 28,335 | 6 | 9 |
| Pig | H1N1 | 3,662 | 138 | 2,746 | 0 | 752 | 4 | 22 |
| Pig | H3N2 | 3,175 | 34 | 2,958 | 0 | 173 | 5 | 5 |

The 672-nt form is concentrated in Human-H3N2 in 2019–2020, with 753 of its 806
isolates.

#### Findings

* The 660-nt and 693-nt forms cover 55,216 of the 56,417 complete records
  (97.9%).
* The 654-, 660-, 672- and 693-nt forms cover 56,361 of the 56,417 complete
  records (99.9%). The other lengths account for 56 isolates. Whether to exclude
  those 56 from an initial feature set is still open; uncommon length alone does
  not establish an error.
* Every protein at each of the 5 listed CDS lengths has the internal and trailing
  gaps shown in the table. Some proteins at the 9 other lengths have different
  leading or internal gap patterns.
* Columns 74, 112 and 132 each hold a protein symbol in exactly one protein; the
  other 9,038 have a gap. The three protein symbols occur in three separate
  proteins, from Human-H3N2 at 696 nt in 2017 and 2018 and Human-H1N1 at 663 nt
  in 2017.
* Only the 714-nt form occupies columns 234–240: 10 unique proteins from 15
  isolates.

#### Artifacts

`results/flu/July_2025/cds_alignment_pilot/Human+Pig-H1N1+H3N2-yr2010-2025/NS1/`.
The same scripts and outputs as the section above.

### NS1 on Human-H3N2-2024

The first alignment run, on 2026-11-05. It is the pilot for the method, on the
population Task 1 uses.

#### Inputs

Of 5,346 filtered isolates, 5,337 (99.8%) had a complete CDS. They contained
1,132 unique CDS and 560 unique proteins in one host–subtype–year group. 48
isolates and 48 unique proteins contain an `X`, all at 693 nt.

This population has a usable pin. `frac at mode` is 0.991, at 693 nt, against the
0.90 `check_cds_length` needs.

MAFFT 7.526 with `--auto` selected FFT-NS-2 and produced 230 columns. Removing
`-` recovered the exact input protein for all 560.

#### Lengths and gaps

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

#### Stop codon columns

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

#### By host and subtype

One metadata group, so there is no breakdown.

#### Findings

* All 551 proteins from 693-nt CDS have no gaps, and each shorter protein has one
  run of trailing gaps.
* No protein has an `X` in columns 218–230, where one or both shorter forms have
  trailing gaps.
* This population has a usable pin at 693 nt, so the alignment is not needed to
  model NS1 here. It was run to establish the method.

#### Downstream contig check

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

#### Artifacts

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
* Every run used one MAFFT strategy, FFT-NS-2, chosen by `--auto`. We did not
  test a second one. Agreement between two strategies would not establish that a
  placement is correct.
* The one-residue columns and the uncommon gap patterns have not been reviewed
  sequence by sequence.
