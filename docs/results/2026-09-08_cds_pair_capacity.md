# Positive-pair capacity of the 15 schema pairs, human H3N2 2024

```yaml
# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-09-08
population:     human H3N2 2024
proteins:       PB2, PA, HA, NP, NA, M1
alphabet:       nt_cds
script:         src/analysis/summarize_pair_capacity.py
script_commit:  434bb00
bundle:         flu_ha_na_h3n2_2024_random_cv4_pinned_length   # supplies the pins
artifacts:      results/flu/July_2025/pair_capacity/{pair_capacity.csv,pair_isolate_overlap.csv}
depends_on:     [docs/results/2026-09-07_cds_length_survey.md]
```

## Question

`docs/results/2026-09-07_cds_length_survey.md` says which proteins can be pinned. It works per
protein, and it says nothing about how many training pairs a schema pair built from them would
have. This measures that, so the pairs can be chosen before any model is trained.

It also fixes a comparability problem. Two schema pairs trained on different isolates differ in
host, subtype and year as well as in their proteins, so a score difference cannot be attributed.
Every row here is built on one common cohort, defined below.

## Methods

```bash
python -m src.analysis.summarize_pair_capacity --hn_subtype H3N2 --host Human --year 2024
```

The **common cohort** is the isolates carrying a complete CDS at the pinned length for all six
proteins at once. On this population that is **5,143 of 5,346 isolates**, so the constraint costs
3.8%. Every pair is then built from those same isolates, which holds host, subtype and year fixed
by construction.

`HK selected` is how many positives survive the unique-sequence constraint, where no slot-A
sequence and no slot-B sequence is used twice. It comes from Hopcroft-Karp, a maximum matching, so
it is the largest such set. The sequential-dedup selectors retain fewer.

PB1 and NS1 are absent, because `conf/virus/flu.yaml` carries pins for the other six and not for
these two. Both were pinned later, on 2026-09-14, in
`conf/bundles/flu_8_major_proteins_human_h3n2_2024_pinned_length.yaml`: NS1 at 693 nt and PB1 at
2,277 nt. Neither value belongs in the per-virus file, which is shared with H1N1 work where NS1 is
660 nt and PB1 is 2,274 nt. The NS1 pin costs almost no isolates, while the PB1 pin retains 55.1%
of them, so PB1 pairs are completeness-selected rather than unpinnable. The 28-pair audit in
`docs/plans/2026-09-14_cross_year_importance_all_pairs_alignment_plan.md` covers all eight
proteins.

## Results

| ID | Pair ID | Schema pair | Unique positives | Unique slot-A | Unique slot-B | HK selected | HK share |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 1-4 | PB2-HA | 3,693 | 2,727 | 2,615 | 1,987 | 53.8% |
| 2 | 1-3 | PB2-PA | 3,723 | 2,727 | 2,636 | 1,972 | 53.0% |
| 3 | 3-4 | PA-HA | 3,699 | 2,636 | 2,615 | 1,894 | 51.2% |
| 4 | 1-6 | PB2-NA | 3,512 | 2,727 | 2,189 | 1,732 | 49.3% |
| 5 | 4-6 | HA-NA | 3,444 | 2,615 | 2,189 | 1,686 | 49.0% |
| 6 | 3-6 | PA-NA | 3,504 | 2,636 | 2,189 | 1,677 | 47.9% |
| 7 | 1-5 | PB2-NP | 3,392 | 2,727 | 1,787 | 1,478 | 43.6% |
| 8 | 4-5 | HA-NP | 3,301 | 2,615 | 1,787 | 1,454 | 44.0% |
| 9 | 3-5 | PA-NP | 3,359 | 2,636 | 1,787 | 1,429 | 42.5% |
| 10 | 5-6 | NP-NA | 3,080 | 1,787 | 2,189 | 1,282 | 41.6% |
| 11 | 1-7 | PB2-M1 | 3,032 | 2,727 | 793 | 708 | 23.4% |
| 12 | 4-7 | HA-M1 | 2,936 | 2,615 | 793 | 703 | 23.9% |
| 13 | 3-7 | PA-M1 | 2,990 | 2,636 | 793 | 688 | 23.0% |
| 14 | 6-7 | NA-M1 | 2,602 | 2,189 | 793 | 650 | 25.0% |
| 15 | 5-7 | NP-M1 | 2,293 | 1,787 | 793 | 616 | 26.9% |

`ID` ranks the table as sorted, by descending `HK selected`, so it moves when the population
changes. `Pair ID` is the two segment numbers and is stable.

### A shared cohort does not give equal counts

Every row draws on the same 5,143 isolates, yet unique positives range from 2,293 to 3,723 and
selected positives from 616 to 1,987. Two isolates that share both proteins of a pair collapse
into one positive, and how often that happens depends on the proteins. M1 has only 793 unique
sequences across the cohort, so M1 pairs collapse hardest.

Holding the isolates fixed therefore removes the metadata confound. It does not remove the
diversity difference, which is what drives both the dedup step and the matching step.

### M1 sets the floor for any equal-count design

Every M1 pair caps near 700, because a matching cannot exceed the 793 unique M1 sequences.
Downsampling all fifteen pairs to a common count would cap the experiment at **616**, which costs
PB2-HA 69% of its data and the median pair 58%.

Excluding M1 leaves ten pairs spanning **1,282 to 1,987**, a 1.55x range rather than 3.2x. That is
the set to screen first, with M1 pairs run separately if they are wanted at all.

HA-NA sits fifth of fifteen at 1,686. The pair with the most existing results is unremarkable in
capacity, which matters when reading the weaker PB2-PA result as schema-specific rather than as a
size effect. PB2-PA is second at 1,972, so it has 17% more selected positives than HA-NA and still
scored lower.

### Two pairs sharing a protein do not share their isolates

`pair_isolate_overlap.csv` holds all 105 combinations. Each matching is solved on its own bigraph,
so nothing forces two pairs to keep the same isolates, and they do not:

| | isolate Jaccard |
|---|---:|
| median across 105 combinations | 0.301 |
| highest, HA-M1 against PB2-M1 | 0.556 |
| PA-NP against NP-NA, which share NP | 0.470 |
| lowest, NP-M1 against PB2-HA | 0.167 |

Even for two pairs that share a protein, fewer than half the isolates are common. A shared cohort
makes pairs more comparable than separately built populations would be. It does not make them
identical, and a head-to-head should be read alongside its Jaccard.

## What this does not settle

It does not say which pairs will train well. Capacity is a count, not a difficulty.

The PB2-PA result is still unexplained. Two candidate explanations were tested and both failed.
PB2 and PA carry more per-position Shannon entropy than HA and NA, not less (137.0 and 128.1 bits
against 97.2 and 81.8), so it is not a diversity deficit. Cross-slot coupling is also comparable
between the two pairs. Those two measurements were taken on the all-host H3N2 2024 population,
before the human-only restriction, so they are indicative rather than exact for this cohort.

Matched counts here are not the ones the existing HA-NA and PB2-PA runs used. Those were built
without the six-protein cohort and without the human-only filter, giving 1,782 and 2,127. The
cohort and host restrictions bring them to 1,686 and 1,972. Scores from the earlier runs cannot be
compared directly against models trained on this cohort.

## Limitations

One population, one year, one subtype, one host. Nothing here says whether the ordering holds for
another year or subtype.

The counts assume the `nt_cds` pair-key alphabet. Under `aa`, codon variants collapse and every
count would be lower.

`HK selected` is an upper bound on what the unique-sequence constraint allows, not a target. A run
may retain fewer for other reasons.
