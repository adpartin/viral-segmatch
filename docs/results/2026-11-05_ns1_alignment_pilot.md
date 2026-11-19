# NS1 protein alignment pilot on Human-H3N2-2024

```yaml

# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-11-05
population:     Human-H3N2-2024
proteins:       NS1
source:         data/processed/flu/July_2025/ (cds_dna_final, protein_final, ctg_dna_final)
scripts:        src/analysis/align_proteins.py; src/analysis/audit_ns1_downstream_contig.py
artifacts:      results/flu/July_2025/cds_alignment_pilot/Human-H3N2-2024/NS1/
```

This was the first protein alignment run. It aligned NS1 on the Task 1
population, Human-H3N2-2024, to develop the method. The alignment of all 8
proteins on the Task 2 population is in
[Protein alignment for codon features](2026-11-09_protein_alignment.md).

This doc also holds the downstream contig check. The
[aligned codon features plan](../plans/2026-11-17_aligned_codon_features_plan.md)
cites it as the support for the stop codon placement rule in the 654- and 660-nt
NS1 CDS.

## Inputs

* 5,337 of the 5,346 isolates have a complete NS1 CDS. The other 9 are
  incomplete and excluded.
* The 5,337 complete CDS hold 1,132 unique CDS at 3 lengths. They translate to
  560 unique proteins.
* 693 nt holds 1,122 of the 1,132 unique CDS (99.1%). On the Task 2
  population, it holds 50.2%.
* 48 isolates have an `X` in their protein, all at 693 nt. An `X` represents a
  codon that does not resolve to one amino acid, so it gives MAFFT less
  information for placing nearby gaps.

## Alignment

MAFFT v7.526 aligned the 560 unique proteins with `--auto`, which chose
FFT-NS-2. The Task 2 runs set the same strategy with explicit flags. The
alignment has 230 columns. Removing the gaps from each aligned protein gives
back its exact input.

| CDS length | Isolates | Unique CDS | Unique proteins | Amino acids | Gap columns |
|---|---:|---:|---:|---:|---|
| 693 nt | 5,324 | 1,122 | 551 | 230 | none |
| 660 nt | 9 | 7 | 6 | 219 | 220–230 |
| 654 nt | 4 | 3 | 3 | 217 | 218–230 |

Each shorter protein has gaps only at its end. No protein has an `X` in columns
218–230.

The examples below show amino acids 205 to the end. The hash prefixes identify
the proteins in the saved FASTA files.

| CDS length | Protein hash prefix | Amino acids 205–end | Alignment columns 205–230 |
|---|---|---|---|
| 693 nt | `47566821` | `SSHENGGPPLTPKQKQEMARTARSEV` | `SSHENGGPPLTPKQKQEMARTARSEV` |
| 693 nt | `046b53e6` | `SSNENGGPPLTPKQKREMARTARSEV` | `SSNENGGPPLTPKQKREMARTARSEV` |
| 660 nt | `40191b0e` | `SCDENGRPSLPPEQK` | `SCDENGRPSLPPEQK-----------` |
| 654 nt | `6a1f0bc8` | `SSHENGGPPLTPK` | `SSHENGGPPLTPK-------------` |

The 654-nt protein `6a1f0bc8` equals the first 217 amino acids of the 693-nt
protein `47566821`.

## Downstream contig check

MAFFT received the proteins without their terminal `*`, so the alignment has no
stop codons. A 660-nt CDS has its stop at codon 220. The alignment does not show
whether that stop belongs opposite codon 220 of the 693-nt CDS or opposite the
693-nt stop at codon 231. The same question applies to the 654-nt stop at codon
218.

The contig of a short CDS continues after its stop codon.
`audit_ns1_downstream_contig.py` took the 13 isolates with a 654- or 660-nt
CDS. For each, it checked that the annotated location recovers the CDS from the
contig. It then compared the contig bases after the stop codon with the same
positions in the most frequent 693-nt CDS, which 2,095 isolates carry.

```text
               codon 220   codons 221–231 (33 bases)
693-nt CDS:    CGG         sense codons, then the stop TGA
660-nt contig: TGA         bases after the stop, compared with the row above
```

| CDS length | Isolates | Contig bases after stop | Bases compared | Matching bases |
|---|---:|---:|---:|---:|
| 654 nt | 4 | 197 | 39 | 34–39 |
| 660 nt | 9 | 178–191 | 33 | 25–27 |

* The 654-nt CDS end with `TAG` at codon 218, where the 693-nt CDS has `CAG`.
* The 660-nt CDS end with `TGA` at codon 220, where the 693-nt CDS has `CGG`.
* The 693-nt stop, `TGA` at codon 231, is also `TGA` at the same contig position
  in all nine 660-nt isolates and in three of the four 654-nt isolates. The
  fourth has `CGA`.

Most bases after each short stop codon match the 693-nt CDS at the same
positions. This supports placing each short stop codon opposite the codon with
the same number in the 693-nt CDS: column 218 for 654 nt and column 220 for
660 nt. It does not support putting all stop codons in one column.

Limits:

* The comparison assumes that codons with the same number, counted from the CDS
  start, correspond. It does not establish the evolutionary history.
* The bases after a short stop codon are in the contig, outside the annotated
  CDS.
* On the Task 2 population, the retention rule keeps 660 nt and drops 654 nt.
  25,506 Task 2 isolates have a 660-nt CDS, and this check covers 9 of them.
