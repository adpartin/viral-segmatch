# Positive-pair capacity of the 28 schema pairs, Human-H3N2-2024

```yaml
# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-09-16
population:     Human-H3N2-2024
proteins:       PB2, PB1, PA, HA, NP, NA, M1, NS1
alphabet:       nt_cds
script:         src/analysis/summarize_pair_capacity.py
script_commit:  67e73c5
bundle:         flu_8_major_proteins_human_h3n2_2024_pinned_length   # supplies the pins
artifacts:      results/flu/July_2025/pair_capacity_8_proteins/
depends_on:     [docs/results/2026-09-07_cds_length_survey.md]
```

Replaces the 2026-09-08 six-protein audit, which gave every pair one shared set of isolates.
This 28-pair audit decides eligibility per pair.

## Question

- `docs/results/2026-09-07_cds_length_survey.md` says which proteins can be pinned. It works per
  protein and says nothing about how many training pairs a schema pair built from them would have.
- This measures that, so pairs can be chosen before any model is trained.

## Methods

```bash
python -m src.analysis.summarize_pair_capacity \
  --config_bundle flu_8_major_proteins_human_h3n2_2024_pinned_length \
  --proteins PB2 PB1 PA HA NP NA M1 NS1 \
  --out_dir results/flu/July_2025/pair_capacity_8_proteins
```

- **Population**: Human-H3N2-2024, 5,346 isolates, each carrying a record for all 8 proteins.
- **Eligible isolates**: each pair uses its own eligible isolates. An isolate needs the pair's
  two proteins as a complete CDS at the pinned length, not the other six.
- **Pins**: the six in `conf/virus/flu.yaml`, plus PB1 at 2,277 nt and NS1 at 693 nt from the
  bundle. Reach by year is in the survey's "Pin reach by year" section.
- **`Unique positives`**: observed same-isolate pairs, deduplicated on the `nt_cds` pair key.
- **`HK selected`**: the positives left once no slot-A and no slot-B sequence is used twice.
  Hopcroft-Karp returns a maximum matching, so it is the largest such set; the sequential-dedup
  selectors retain fewer. `HK share` is that count over `Unique positives`.

## Results

<img src="figs/2026-09-16_h3n2_2024_pair_capacity_matrix.png" width="550" alt="28-pair Hopcroft-Karp capacity, Human-H3N2-2024">

| Pair ID | Schema pair | Eligible isolates | Unique positives | Unique slot-A | Unique slot-B | HK selected | HK share |
|---|---|---:|---:|---:|---:|---:|---:|
| 1-2 | PB2-PB1 | 2,945 | 2,349 | 1,788 | 1,790 | 1,404 | 59.8% |
| 1-3 | PB2-PA | 5,324 | 3,837 | 2,808 | 2,712 | 2,030 | 52.9% |
| 1-4 | PB2-HA | 5,329 | 3,796 | 2,810 | 2,681 | 2,042 | 53.8% |
| 1-5 | PB2-NP | 5,318 | 3,484 | 2,800 | 1,830 | 1,512 | 43.4% |
| 1-6 | PB2-NA | 5,167 | 3,532 | 2,745 | 2,203 | 1,745 | 49.4% |
| 1-7 | PB2-M1 | 5,332 | 3,128 | 2,813 | 812 | 726 | 23.2% |
| 1-8 | PB2-NS1 | 5,313 | 3,204 | 2,800 | 1,120 | 995 | 31.1% |
| 2-3 | PB1-PA | 2,939 | 2,332 | 1,786 | 1,710 | 1,341 | 57.5% |
| 2-4 | PB1-HA | 2,945 | 2,327 | 1,790 | 1,756 | 1,392 | 59.8% |
| 2-5 | PB1-NP | 2,942 | 2,162 | 1,787 | 1,227 | 1,041 | 48.1% |
| 2-6 | PB1-NA | 2,939 | 2,237 | 1,786 | 1,480 | 1,212 | 54.2% |
| 2-7 | PB1-M1 | 2,945 | 1,992 | 1,790 | 582 | 517 | 26.0% |
| 2-8 | PB1-NS1 | 2,937 | 2,029 | 1,788 | 781 | 704 | 34.7% |
| 3-4 | PA-HA | 5,329 | 3,805 | 2,711 | 2,682 | 1,944 | 51.1% |
| 3-5 | PA-NP | 5,318 | 3,455 | 2,703 | 1,832 | 1,459 | 42.2% |
| 3-6 | PA-NA | 5,167 | 3,520 | 2,649 | 2,202 | 1,689 | 48.0% |
| 3-7 | PA-M1 | 5,335 | 3,082 | 2,716 | 812 | 707 | 22.9% |
| 3-8 | PA-NS1 | 5,315 | 3,214 | 2,705 | 1,121 | 952 | 29.6% |
| 4-5 | HA-NP | 5,323 | 3,382 | 2,673 | 1,830 | 1,482 | 43.8% |
| 4-6 | HA-NA | 5,173 | 3,466 | 2,634 | 2,203 | 1,698 | 49.0% |
| 4-7 | HA-M1 | 5,338 | 3,018 | 2,686 | 811 | 720 | 23.9% |
| 4-8 | HA-NS1 | 5,318 | 3,151 | 2,676 | 1,119 | 959 | 30.4% |
| 5-6 | NP-NA | 5,162 | 3,091 | 1,794 | 2,197 | 1,287 | 41.6% |
| 5-7 | NP-M1 | 5,326 | 2,356 | 1,833 | 808 | 627 | 26.6% |
| 5-8 | NP-NS1 | 5,307 | 2,540 | 1,824 | 1,114 | 806 | 31.7% |
| 6-7 | NA-M1 | 5,176 | 2,620 | 2,206 | 800 | 657 | 25.1% |
| 6-8 | NA-NS1 | 5,156 | 2,794 | 2,196 | 1,092 | 853 | 30.5% |
| 7-8 | M1-NS1 | 5,323 | 1,840 | 806 | 1,122 | 440 | 23.9% |

- `HK selected` runs from 440 (M1-NS1) to 2,042 (PB2-HA), median 1,126.5 — a 4.6x range.
- `Pair ID` is the two segment numbers and is stable. Rows are in segment order here;
  `pair_capacity.csv` is sorted by descending `HK selected`.

### Eligibility and capacity are separate limits

- The 21 pairs without PB1 have 5,156 to 5,338 eligible isolates. The 7 with PB1 have 2,937 to
  2,945, because only 55.1% of isolates have a complete PB1 CDS at the 2,277 nt pin. See "Why PB1
  retains about half its isolates" in the survey.
- Eligible isolates do not predict capacity: PB1-HA keeps 1,392 positives from 2,945 isolates,
  while M1-NS1 keeps 440 from 5,323.
- The ceiling is the smaller of a pair's two unique-sequence counts, since the matching keeps at
  most one positive per unique sequence. How close a pair comes to that ceiling is a property of
  its positive-pair bigraph, not of either count alone.
- **M1 and NS1 limit the matching.** They supply the fewest unique sequences — M1 582 to 812 and
  NS1 781 to 1,122 across the 7 pairs each takes part in. The 13 pairs containing one of them hold
  the 13 lowest counts; the first pair containing neither is PB1-NP at 1,041.
- **PB1 limits eligibility instead.** 44.7% of isolates have no complete PB1 CDS. What remains is
  a smaller population, not a less diverse one: PB1 pairs that avoid M1 and NS1 keep 48.1% to
  59.8% of their positives, the same band as the rest of the table.

### Pairs do not share their retained isolates

- Each matching is solved on its own bigraph, so nothing forces two pairs to keep the same
  isolates. `pair_isolate_overlap.csv` holds all 378 combinations, with `shares protein` marked.
- Isolate Jaccard runs 0.112 to 0.568, median 0.234. Over the 168 combinations whose pairs share a
  protein it runs 0.188 to 0.568, median 0.353.
- A performance difference between two pairs is therefore not measured on one set of isolates, and
  a head-to-head should be read alongside its Jaccard.

## What this does not settle

- Which pairs will train well. Capacity is a count, not a difficulty.
- `HK selected` is an upper bound on what the unique-sequence constraint allows, not a target. A
  run may retain fewer for other reasons.

## Limitations

- One population: one year, one subtype, one host. Nothing here says whether the ordering holds for
  another year or subtype.
- The counts assume the `nt_cds` pair-key alphabet. Under `aa`, synonymous variants collapse, so
  the unique-sequence, `Unique positives` and `HK selected` counts would be equal or lower.
  `Eligible isolates` is unaffected, since eligibility is decided on the CDS.
- PB1 pairs are completeness-selected, so their population size is not comparable to the others'.
