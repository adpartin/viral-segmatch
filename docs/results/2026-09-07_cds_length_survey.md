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

The `script` and `artifact` produce only the whole-corpus and Human-H3N2-2024 per-protein tables. The year-specific tables and PB1 downstream contig analysis were rechecked directly against the processed data on 2026-09-14, but their commands and outputs have not been preserved as artifacts. They therefore need separate provenance before publication.

## Problems and possible solutions

Per-site features need every retained record for a protein to be complete and at one length. This survey identifies 3 separate limitations that affect the construction of per-site datasets. Each requires a different response.

1. **Incomplete CDS.** A record is considered incomplete when its annotated CDS fails the completeness check: an initial methionine (start codon), a terminal stop, and no internal stop. Alignment cannot reconstruct bases that are absent. These records are therefore excluded. `PB1` is the main example in Human-H3N2-2024: most of its incomplete 2,274-nt records end at the contig boundary, before the expected terminal stop.

2. **Complete CDS at more than one length.** Complete CDS can have different lengths because of indels (insertions, deletions) or population-specific sequence forms. The current pinned-length method retains one length and excludes the others. In the whole corpus, length heterogeneity is most evident for `NS1`, `HA` and `NA`, partly because the corpus combines subtypes and hosts. Restricting the data to Human-H3N2-2024 largely removes this heterogeneity. `PB1` remains unusual: its modal Human-H3N2 length changes from 2,274 to 2,277 nt, with both forms common in 2023. **Possible solution:** a codon-aware alignment may allow complete sequences of different lengths to share a common coordinate system, with alignment gaps representing length differences. The alignment would need to be validated before using its columns as features.

3. **Low sequence diversity.** Many isolates can carry the same CDS, so there are far fewer unique CDS than isolates. `M1` and `NS1` are the main examples. This limitation is visible in the `unique CDS` counts, but not in the completeness or mode length fractions. It reduces the number of independent examples and limits how many positive pairs remain after requiring each CDS to appear at most once per side. Pair capacity is measured separately in
`docs/results/2026-09-08_cds_pair_capacity.md`. Alignment cannot increase sequence diversity.
**Possible solution:** use Hopcroft-Karp selection to retain as many positive pairs as possible under the constraint that no CDS appears twice in a slot, report the resulting pair capacity, and treat small populations as a limitation of the experiment.

## Why this was measured

Per-site features use one column per CDS position and do not align or pad sequences. Every retained CDS for a protein must therefore be complete and have the same configured *pinned length*. This survey checks completeness and length; it does not validate positional alignment.

A *pin*, or *pinned length*, is a protein-specific CDS length recorded in the config rather than selected independently for each run. The `cds_length` values in `conf/virus/flu.yaml` provide these lengths. When `dataset.require_complete_cds_at_pinned_length` is enabled, the pipeline retains only complete CDS records whose length equals the pin. It also checks that the pin is the modal length and accounts for the required share of unique complete CDS in the population being processed. A pin has a scope: the population in which it was measured. It must be revalidated before being applied outside that scope. See *Pinned CDS length* in `docs/methods/glossary.md` for the canonical definition.

Using a configured pin gives comparable runs the same feature dimensions and consistent column numbering. A pin does not by itself establish that corresponding positions are homologous; that requires separate validation or alignment.

## What was done

```bash
python -m src.analysis.summarize_cds_lengths
python -m src.analysis.summarize_cds_lengths --hn_subtype H3N2 --host Human --year 2024
```

All counts refer to CDS DNA, keyed by `cds_dna_hash`.
- Sequence statistics count each unique CDS once
- Isolate statistics count every isolate carrying it

The two statistics can differ greatly: 5,346 Human-H3N2-2024 isolates carry only 815 unique M1 sequences.

## Results: whole-corpus

All 108,530 isolates carry a record for every protein.

A CDS is complete if: `starts_with_m & has_terminal_stop & ~has_internal_stop`, and it's recorded in a column `is_complete_cds`.

The 3 problems:
- **Incomplete CDS (P1):** measured at the isolate level by `1 - frac isolates complete`. It is small in the whole corpus; `PB1` has the largest loss at 5.1%.
- **Complete CDS at more than one length (P2):** measured among unique complete CDS by `1 - frac at mode`. It is most evident for `NS1`, `HA`, and `NA`. At the isolate level, `frac isolates complete - frac isolates at mode` gives the additional loss caused by retaining only the modal length.
- **Low sequence diversity (P3):** indicated by the number of `unique CDS` relative to 108,530 isolates. `M1` and `NS1` have the fewest unique CDS per isolate. This is a different quantity from the per-sequence reuse reported in `pair_sequence_reuse.csv`, which counts a sequence's unique partner sequences in a schema pair's positives rather than the isolates carrying it.

Table columns below:

- `unique CDS`: number of unique CDS sequences. Identical CDS are counted once.
- `complete CDS`: number of `unique CDS` that pass the completeness check.
- `min`, `max`, `median`: min, max, and median length, in nucleotides, among unique complete CDS.
- `mode`: most common length among unique complete CDS.
- `complete CDS at mode`: number of unique complete CDS whose length equals the mode.
- `frac at mode`: share of unique complete CDS whose length equals the mode: `complete CDS at mode / complete CDS`.
- `frac isolates complete`: share of isolates (the 108,530) whose CDS for the protein passes the completeness check: isolates with a complete CDS / isolates carrying the protein.
- `frac isolates at mode`: share of isolates (the 108,530) whose CDS is complete and has the modal length: isolates with a complete CDS at the mode / isolates carrying the protein. This is the share of the isolate population retained by pinning the protein to its modal length.

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

Clarifying `frac isolates complete` and `frac isolates at mode` with PB1 on the whole corpus. Both fractions share the same denominator — isolates, not sequences:

```
isolates carrying PB1:                     108,530 (denominator for both)
isolates whose PB1 is complete:            102,943 (is_complete_cds==True)
isolates complete AND at mode (2,274 nt):   95,768

frac isolates complete = 102,943 / 108,530 = 0.948521  ->  0.949
frac isolates at mode  =  95,768 / 108,530 = 0.882410  ->  0.882
```

## What the results say about the pinned lengths

This section concerns P2. A validated codon-aware alignment could allow complete CDS of different lengths to share one coordinate system and be retained together. It would not recover incomplete CDS or establish that combining different subtypes or years is scientifically appropriate.

The corpus-wide modes match the 6 lengths configured in `conf/virus/flu.yaml`: PB2 2,280, PA 2,151, HA 1,701, NP 1,497, NA 1,410, and M1 759 nt. This agreement does not make the lengths universal. The whole corpus combines subtypes, hosts, and years, so a pin must be validated in the population where it will be used.

`PB1` and `NS1` have no configured pins, but the source of the length variation differs:

- **NS1 varies mainly across subtypes.** H1N1 is dominated by 660 nt and H3N2 by 693 nt. Because `cds_length` is keyed only by protein, one configured `NS1` length cannot represent both subtypes. Within Human-H3N2, the mode remains 693 nt from 2015 through 2025, although other complete lengths reduce retention to 72.3% in 2019 and 68.8% in 2020.

- **PB1 varies across years within Human-H3N2.** Its modal length is 2,274 nt through 2023 and 2,277 nt in 2024-2025, with both lengths common in 2023. Neither length provides one `PB1` pin for the full period.

## Results: Human-H3N2-2024

This section concerns P1 and P3: incomplete CDS in PB1, and low sequence diversity in M1 and NS1.
Complete CDS at more than one length (i.e., P2) nearly disappears here, because the population is
restricted to one subtype, host and year.

This is the population used by the current experiments. It contains 5,346 isolates, each with a record for all 8 proteins.

Clarifying `frac isolates complete` and `frac isolates at mode` with PB1 on this population. Both fractions share the same denominator — isolates, not sequences:

```
isolates carrying PB1:                     5,346 (denominator for both)
isolates whose PB1 is complete:            2,958 (is_complete_cds==True)
isolates complete AND at mode (2,277 nt):  2,945

frac isolates complete = 2,958 / 5,346 = 0.553311  ->  0.553
frac isolates at mode  = 2,945 / 5,346 = 0.550879  ->  0.551
```

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

## Why PB1 retains about half its isolates

This section concerns P1. PB1 has the largest completeness loss in the Human-H3N2-2024 table. Only
2,958 of 5,346 isolates have a complete PB1, and 2,945 have a complete PB1 at the 2,277-nt mode.
Pinning therefore removes only 13 additional isolates after incomplete records have been excluded.

Of the 2,388 incomplete PB1 records, 2,386 lack a terminal stop and 2,339 are 2,274 nt. The
2,274-nt records appear to end at the contig boundary: of the 2,352 records at this length, 99.7%
have no downstream contig sequence. All but one of the 2,339 incomplete records have no downstream
bases at all, so there is nothing to extend them with; the exception has 32 downstream bases whose
first codon is unresolved. By comparison, the 2,945 records at 2,277 nt have a median of 27
downstream bases. The incomplete 2,274-nt records therefore cannot be extended from the current
assemblies.

This measurement describes the available records. It does not establish why the bases are absent
or the biological prevalence of the 2,274- and 2,277-nt forms.

## Pin reach by year in Human-H3N2 (additional measurement)

This table covers Human-H3N2 isolates from 2015 through 2025; it does not include other hosts or
subtypes. This section concerns P1 and P2. The experiments in
`docs/plans/2026-09-14_codon_site_features_plan.md` pin all 8 proteins to
their Human-H3N2-2024 modal complete-CDS length: PB2 2,280, PB1 2,277, PA 2,151, HA 1,701,
NP 1,497, NA 1,410, M1 759, and NS1 693 nt. Six are in `conf/virus/flu.yaml`; PB1 and NS1 come
from a bundle-level `virus.cds_length` override. The table reports the share of each year's
isolates with a complete CDS at that length, so it shows how far the 2024 pins reach:

| year | isolates | PB2 | PB1 | PA | HA | NP | NA | M1 | NS1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 1,335 | 0.998 | 0.001 | 0.999 | 1.000 | 0.999 | 1.000 | 0.999 | 0.993 |
| 2016 | 1,416 | 0.999 | 0.004 | 0.999 | 0.997 | 1.000 | 1.000 | 1.000 | 0.985 |
| 2017 | 2,845 | 1.000 | 0.000 | 0.998 | 1.000 | 0.999 | 0.998 | 1.000 | 0.958 |
| 2018 | 1,639 | 0.999 | 0.001 | 0.999 | 1.000 | 0.999 | 0.999 | 1.000 | 0.961 |
| 2019 | 2,755 | 1.000 | 0.001 | 0.998 | 0.999 | 0.926 | 0.996 | 1.000 | 0.723 |
| 2020 | 189 | 0.995 | 0.000 | 0.995 | 0.995 | 0.905 | 0.979 | 1.000 | 0.688 |
| 2021 | 1,089 | 0.999 | 0.042 | 0.981 | 0.999 | 1.000 | 0.999 | 0.999 | 0.981 |
| 2022 | 4,864 | 0.998 | 0.067 | 0.998 | 0.998 | 0.996 | 0.996 | 0.999 | 0.993 |
| 2023 | 1,347 | 0.997 | 0.391 | 0.999 | 0.999 | 0.998 | 0.936 | 1.000 | 0.996 |
| 2024 | 5,346 | 0.998 | 0.551 | 0.998 | 0.999 | 0.996 | 0.968 | 1.000 | 0.996 |
| 2025 | 3,434 | 0.999 | 0.619 | 0.998 | 1.000 | 0.999 | 0.999 | 1.000 | 0.999 |

PB2, PA, HA, NP, NA and M1 retain at least 90% of isolates in every year. The other two fail for
different reasons. NS1 stays the modal length in every year and only retains fewer isolates in 2019
and 2020. PB1 is a different length before 2024, so its pin does not apply to earlier years: 0.619
in 2025 and 0.551 in 2024, 0.391 in 2023, and effectively zero before 2021.

The low cells do not all have the same cause, and the table alone does not separate them. For NS1,
2,754 of 2,755 isolates in 2019 and all 189 isolates in 2020 have a complete CDS. Complete NS1 at a
length other than 693 nt accounts for 27.6% of isolates in 2019 and 31.2% in 2020, so these are P2
losses. By contrast, the lower NP values in 2019-2020 and the lower NA value in 2023 are caused by
incomplete CDS (P1), not by other complete lengths.

The next table separates the two PB1 lengths. Each entry is the share of that year's isolates with
a complete PB1 at the stated length.

| year | isolates | complete at 2,274 nt | complete at 2,277 nt | complete at either length |
|---:|---:|---:|---:|---:|
| 2015 | 1,335 | 0.999 | 0.001 | 1.000 |
| 2016 | 1,416 | 0.996 | 0.004 | 1.000 |
| 2017 | 2,845 | 0.999 | 0.000 | 0.999 |
| 2018 | 1,639 | 0.998 | 0.001 | 0.998 |
| 2019 | 2,755 | 0.996 | 0.001 | 0.997 |
| 2020 | 189 | 0.989 | 0.000 | 0.989 |
| 2021 | 1,089 | 0.956 | 0.042 | 0.998 |
| 2022 | 4,864 | 0.905 | 0.067 | 0.972 |
| 2023 | 1,347 | 0.432 | 0.391 | 0.823 |
| 2024 | 5,346 | 0.002 | 0.551 | 0.553 |
| 2025 | 3,434 | 0.003 | 0.619 | 0.622 |

No single PB1 length retains at least 90% of Human-H3N2 isolates across 2023-2025. For a
Human-H3N2-2024 analysis, 2,277 nt is nevertheless a clear population-specific pin among complete
PB1 records. Analyses that use it retain about 55% of isolates and should describe the resulting
set of retained PB1 isolates as completeness-selected. The 2025 values come from the partial 2025
season in the July 2025 corpus.

## Using the survey as a screen

Use the sequence- and isolate-level measures together. `frac at mode` checks whether one length
dominates among unique complete CDS, as required by the production pin guard. `frac isolates at
mode`, with `frac isolates complete` beside it, measures how much of the isolate population would
remain and whether losses come from incomplete records or other lengths.

In Human-H3N2-2024, the 6 configured pins pass both checks, and NS1 passes with a
population-specific 693-nt pin. PB1 at 2,277 nt passes the unique-sequence concentration check but
retains only 55.1% of isolates. It therefore does not pass a 90% isolate-retention screen, but it
can still be used as a smaller, explicitly completeness-selected set of isolates. This distinction
allowed PB1-containing pairs to enter the 28-pairs capacity audit without implying that PB1 has
the same coverage as the other proteins.

Passing this screen means that per-site features on a single shared CDS length are feasible. It does
not guarantee enough training pairs. The companion capacity audit
(`docs/results/2026-09-08_cds_pair_capacity.md`) measures that for all 28 pairs, including those
with PB1 and NS1.

## NS1 protein alignment pilot: Human-H3N2-2024

On 2026-11-05, `src/analysis/align_proteins.py` selected 5,337 complete NS1 CDS from 5,346
isolates. These records contained 1,132 unique CDS and 560 unique proteins. MAFFT 7.526 used
`--auto`, selected FFT-NS-2, and produced an alignment with 230 amino-acid columns. Removing
alignment gaps recovered the exact input sequence for all 560 proteins.

Each CDS includes a terminal stop codon. We removed the final `*` from the protein before
alignment but kept the stop codon in the CDS. The table counts gaps in the protein alignment.

| CDS length | Unique proteins | Codons including stop | Amino acids aligned | Gaps | Gap location |
|---|---:|---:|---:|---:|---|
| 693 nt | 551 | 231 | 230 | 0 | — |
| 660 nt | 6 | 220 | 219 | 11 | Columns 220–230, at the end |
| 654 nt | 3 | 218 | 217 | 13 | Columns 218–230, at the end |

All 551 proteins at 693 nt have no gaps. Each shorter protein has one gap block at the end, with
no internal gaps. No protein has an `X` in columns 218–230, where the shorter proteins have gaps.
The 660-nt CDS end with a stop at codon 220; the 654-nt CDS end with a stop at codon 218.
These stop codons are not represented by the protein alignment gaps.

The examples show original protein residues from position 205 to the end and their placement
in alignment columns 205–230. Protein hash prefixes identify records in the saved FASTA files.

| CDS length | Protein hash prefix | Input residues 205–end | Aligned columns 205–230 |
|---|---|---|---|
| 693 nt | `47566821` | `SSHENGGPPLTPKQKQEMARTARSEV` | `SSHENGGPPLTPKQKQEMARTARSEV` |
| 693 nt | `046b53e6` | `SSNENGGPPLTPKQKREMARTARSEV` | `SSNENGGPPLTPKQKREMARTARSEV` |
| 660 nt | `40191b0e` | `SCDENGRPSLPPEQK` | `SCDENGRPSLPPEQK-----------` |
| 654 nt | `6a1f0bc8` | `SSHENGGPPLTPK` | `SSHENGGPPLTPK-------------` |

The 654-nt example matches the first 217 amino acids of the `47566821` long protein. The
terminal gap placement is consistent with the short CDS ending earlier. This alignment is an
estimate of positional correspondence, not a recovered evolutionary history. We found no
scattered or internal gaps that called for a second MAFFT strategy in this pilot. The proposed
codon and stop mapping is in [Experiment 5](../plans/2026-09-14_codon_site_features_plan.md#next-step).

The saved inputs, aligned FASTA, validation report, per-protein `gap_locations.csv` and
`inspection_manifest.json` are under
`results/flu/July_2025/cds_alignment_pilot/Human-H3N2-2024/NS1/`. The manifests record file
hashes, MAFFT settings and the code working-tree state for this run. The provenance block at the
top of this document applies to the earlier length-survey tables, not this pilot.

### NS1 downstream contig check

The protein alignment shows where amino acids end, but it does not show where a stop codon
belongs or whether bases remain in the segment after that stop. We checked the source contig
for all 13 short-form isolates. We first verified that the annotated location recovers each
short CDS exactly. We then compared the bases immediately after its stop with the same
positions in the most frequent 693-nt CDS (2,095 isolates). The comparison assumes codons
share a start position; matching downstream sequence provides evidence for this placement.

| Short CDS | Isolates | Contig bases after stop | Compared with long CDS | Matching bases |
|---|---:|---:|---:|---:|
| 654 nt | 4 | 197 | 39 | 34–39 |
| 660 nt | 9 | 178–191 | 33 | 25–27 |

At the short CDS's terminal codon, all four 654-nt sequences have `TAG` where the long
reference has `CAG` (codon 218); all nine 660-nt sequences have `TGA` where the reference
has `CGG` (codon 220). The long reference ends with `TGA` at codon 231. That triplet remains
`TGA` at the corresponding contig position for all nine 660-nt isolates and three of the
four 654-nt isolates; it is `CGA` in the remaining 654-nt isolate. These findings support
placing each short CDS stop at its own codon position rather than lining up all stop codons
at the end. They do not establish the exact evolutionary history. Bases beyond a short CDS's
stop belong to the contig, not to that annotated CDS.

The reproducible check is `src/analysis/audit_ns1_downstream_contig.py`. Its per-isolate
comparisons, summary and provenance manifest are `ns1_downstream_contig.csv`,
`ns1_downstream_summary.csv` and `ns1_downstream_manifest.json` in the pilot artifact
directory above. The manifest records the input hashes and code working-tree state.

## NS1 alignment for Task 2 candidate population

Filters: host `[Human, Pig]`, subtype `[H3N2, H1N1]`, year `2010–2025` (inclusive).
This is separate from the reported 2022–2023 Task 2 run. Of 56,512 NS1 isolate records,
56,417 had a complete CDS. They contained 16,540 unique CDS and 9,039 unique proteins
across 63 host–subtype–year groups.

A trailing protein-alignment gap is a `-` after that protein's last amino acid. Stop markers
were removed before alignment, so these gaps are not stop codons.

| CDS length | Isolates | Unique CDS | Unique proteins | Trailing protein-alignment gaps |
|---|---:|---:|---:|---|
| 654 nt | 339 | 219 | 192 | 221–240 |
| 660 nt | 25,526 | 8,248 | 4,604 | 223–240 |
| 672 nt | 806 | 235 | 144 | 227–240 |
| 693 nt | 29,690 | 7,791 | 4,055 | 234–240 |
| 714 nt | 15 | 10 | 10 | None |
| Nine other lengths | 41 | 37 | 34 | Varies |

The final columns of the 240-column protein alignment show the trailing gaps. Here `aa`
means an amino acid:

```text
Column:          222   223–233     234–240
660-nt CDS:      aa    - ... -     - ... -
693-nt CDS:      aa    aa ... aa   - ... -
714-nt CDS:      aa    aa ... aa   aa ... aa
```

The 660-nt and 693-nt CDS account for 55,216 of the 56,417 complete records (97.9%).
The 672-nt form is concentrated in Human-H3N2 in 2019–2020 (753 of 806 records).
The table below shows length by host and subtype. "Other lengths" includes 654 and 714 nt:

| Host | Subtype | Complete NS1 isolates | 660 nt | 672 nt | 693 nt | Other lengths |
|---|---|---:|---:|---:|---:|---:|
| Human | H1N1 | 20,141 | 19,698 | 2 | 430 | 11 |
| Human | H3N2 | 29,439 | 124 | 804 | 28,335 | 176 |
| Pig | H1N1 | 3,662 | 2,746 | 0 | 752 | 164 |
| Pig | H3N2 | 3,175 | 2,958 | 0 | 173 | 44 |

- MAFFT 7.526 used `--auto` (FFT-NS-2). Removing gaps recovered every input protein.
- Each of the five listed CDS lengths has one gap pattern. Some proteins at the nine
  other lengths have additional leading or internal gaps.
- All five listed length forms have gaps at internal columns 74, 112 and 132. Each column
  holds a residue in just one protein (one isolate); all other proteins have a gap there.
  Those three isolates are Human-H3N2 at 696 nt in 2017 and 2018, and Human-H1N1 at
  663 nt in 2017. Only the 714-nt form occupies columns 234–240: 10 unique proteins
  from 15 isolates.
- The common trailing-gap pattern suggests the main length forms may share columns.
  This single protein alignment does not prove positional correspondence. Review the
  three singleton columns and uncommon gap patterns before defining codon features.
  Stop codons were not aligned, so their placement remains to be checked. No codon
  features or models were produced here.

The inputs and outputs are under
`results/flu/July_2025/cds_alignment_pilot/Human+Pig-H1N1+H3N2-yr2010-2025/NS1/`.
`src/analysis/align_proteins.py` generated the alignment, validation and gap locations.
`src/analysis/summarize_protein_alignment.py` generated `group_length_summary.csv`,
`gap_pattern_summary.csv` and `column_occupancy.csv`. The manifests record file hashes
and code state. The provenance block at the top of this document does not cover this audit.

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

- Equal CDS length does not prove positional homology or validate the annotation.
- `is_complete_cds` is an operational check derived from the annotated protein: it requires an
  initial methionine, a terminal stop, and no internal stop. It is not independent evidence that the
  assembly contains every biologically expected base.
- The year-specific and PB1 contig-boundary measurements do not yet have saved scripts and output
  artifacts; see the provenance note above.

## Notes

- Read the CSV with `keep_default_na=False`; otherwise pandas interprets the protein name `NA` as a
missing value.
- The script deduplicates sequences using `cds_dna_hash`. No hash occurs under more than one protein
in this corpus, so this is currently safe. Deduplicating on `(function, cds_dna_hash)` would make
that assumption explicit.
- Sequence statistics deduplicate by hash; isolate statistics do not.
