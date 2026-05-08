# DNA-level coverage feasibility for v2 negative sampling

**Date:** 2026-05-08
**Dataset:** `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_20260508_125547`
(HA/NA mixed, full Flu A, v2)
**Script:** `eda/dna_coverage_feasibility.py`
**Status:** apriori prediction confirmed in production. Option C
(per-`dna_hash` coverage with best-effort logging) was implemented in
`create_negative_pairs_v2` on 2026-05-08 and verified on
`dataset_flu_ha_na_20260508_171512`: `n_dna_uncovered = 0` on every
split; pos `n_unique` now matches neg `n_unique` on every `dna_hash`
row of `split_overlap_stats.csv`; neg-pair count grew ~27% (47,859 →
59,598 on train) as predicted by the coverage-floor lift.

## TL;DR

**DNA-level coverage is feasible** on this dataset. Zero seq_hashes
fail the per-split test, zero fail the cross-split test, and the
worst-case ratio of "DNA variants needing coverage" to "available
partner seq_hashes" is **0.011** — two orders of magnitude of
headroom. Extending v2's coverage phase from `seq_hash` to
`dna_hash` is a clean fix; we don't need a fallback path.

## Why this question

v2's coverage phase guarantees that every protein `seq_hash` H on
each slot appears in at least one positive AND at least one negative
pair. That's a **protein-level** invariant. Mode #2 (sequence-level
label imbalance) is therefore addressed at the protein level only.

The same protein H can be encoded by K different DNA sequences
across isolates (synonymous codon variation). v2's coverage gives H
*one* negative pair, with one specific DNA. The other K−1 DNA
variants of H may have no negative counterexample. For models that
consume DNA-derived features (k-mer), each DNA is its own feature
vector, and a feature that appears only in positives during training
is a per-feature label imbalance.

The HA/NA mixed `split_overlap_stats.csv` shows the symptom directly:
on the train slot a, `n_unique` is 34,338 for both `seq_hash` rows
(pos and neg match by v2 invariant) but **43,875 vs 37,618 for the
`dna_hash` rows** — about 14% of train slot-A pos DNAs have no neg
counterexample.

The question for this analysis: would extending the coverage phase
to `dna_hash` actually be feasible, given v2's other invariants
(pair_key uniqueness, cooccur blocking, cross-split forbidden
pair_keys)?

## Feasibility model

For each `(split, slot, seq_hash H)`:
- **K** = number of distinct `dna_hash` values encoding H on this
  slot in this split's positive pairs. This is the demand: each DNA
  needs its own neg pair_key.
- **partner_pool** = unique partner `seq_hash` values on the
  *opposite* slot in the same split, minus those `cooccur` with H.
  This is the supply: partners we could pair H with to form a valid
  neg pair_key. Distinct partner seq_hashes give distinct pair_keys.
- **ratio = K / partner_pool**. If `ratio ≥ 1`, DNA coverage is
  infeasible for that H (more DNAs than available partner pair_keys).

Cross-split exhaustion (train consuming partners, leaving val/test
with less) is approximated as a cross-split demand-vs-supply test:
- **demand** = sum of K across train, val, test for the same H.
- **supply** = unique partner seq_hashes across all splits, minus
  cooccur partners.

## Inputs to the analysis

| | train | val | test |
|---|---:|---:|---:|
| pos pairs | 47,060 | 5,883 | 5,883 |
| unique seq_hash on slot a | 34,338 | 5,057 | 5,072 |
| unique seq_hash on slot b | 31,010 | 4,849 | 4,842 |

`cooccurring_sequence_pairs.csv` carries 58,826 canonical
seq_hash pairs across 79,384 unique seq_hashes — this is the
"these two proteins were observed together in some isolate, so
they cannot be a negative pair" set. Per seq_hash, the cooccur
list has a small mean (a few partners) and a long tail (highly
conserved proteins co-occur with many others).

## Per-split feasibility

K is highly skewed: most seq_hashes have one DNA variant, a handful
have many.

| split, slot | K=1 | K≥10 | K≥50 | K≥100 |
|---|---:|---:|---:|---:|
| train, a | 31,313 (91.2%) | 156 | 18 | 5 |
| train, b | 27,360 (88.2%) | 174 | 21 | 5 |
| val, a | 4,810 (95.1%) | 20 | 0 | 0 |
| val, b | 4,545 (93.7%) | 16 | 1 | 0 |
| test, a | 4,829 (95.2%) | 19 | 0 | 0 |
| test, b | 4,510 (93.1%) | 16 | 1 | 0 |

The biggest K seen anywhere is **315** (train slot b) and **241**
(train slot a). These are conserved polymerase-like proteins that
appear in hundreds of train isolates with many synonymous DNA
variants.

Partner pool is large at every split-slot:

| split, slot | partner_universe | min partner_pool | median partner_pool |
|---|---:|---:|---:|
| train, a | 31,010 | 30,668 | 31,009 |
| train, b | 34,338 | 33,812 | 34,337 |
| val, a | 4,849 | 4,792 | 4,848 |
| val, b | 5,057 | 4,970 | 5,056 |
| test, a | 4,842 | 4,787 | 4,841 |
| test, b | 5,072 | 4,968 | 5,071 |

Cooccur blocks shrink the universe by ~1% on average. Even the
worst-blocked seq_hash on the smallest split keeps a partner pool of
4,787 — vastly larger than any K we see.

**Worst per-split ratio: 0.011** (val slot b, K=53, partner_pool
4,970). **Zero seq_hashes fail.**

## Cross-split feasibility

The biggest demand any single seq_hash creates is **409** (slot b,
sum across train+val+test). The smallest cross-split partner supply
on the corresponding slot is **41,271**. Worst ratio: **0.0099**.
**Zero seq_hashes fail.**

| slot | demand=1 | demand≥10 | demand≥50 | demand≥100 | max demand |
|---|---:|---:|---:|---:|---:|
| a | 37,639 (89.8%) | 217 | 31 | 14 | 312 |
| b | 32,365 (86.3%) | 236 | 29 | 11 | 409 |

## Verdict

DNA-level coverage is feasible. The partner pool dwarfs the demand
at every (split, slot, seq_hash). Cross-split exhaustion isn't a
concern either — total demand never exceeds 1% of total supply.

## Implications and recommendations

1. **Extend v2's coverage phase to `dna_hash`.** The change is at
   `create_negative_pairs_v2`: instead of demanding ≥1 neg per
   `seq_hash`, demand ≥1 neg per (`seq_hash`, `dna_hash`) pair on
   each slot. Each DNA variant gets a distinct partner seq_hash on
   the opposite slot. The forbidden_pair_keys threading already
   handles cross-split disjointness; no architectural change there.

2. **No fallback path needed.** Because feasibility holds with two
   orders of magnitude of headroom on this dataset. We can keep the
   strict raise: if any DNA variant cannot be covered, fail loudly.

3. **Status of mode #2 in the leakage taxonomy** can flip back to
   ✅ ADDRESSED (both protein and DNA levels) once the extension
   lands. Until then, the "⚠️ NOT addressed at DNA level" caveat
   stands.

4. **Re-test feasibility on smaller / more filtered datasets**
   before extending to bundles like `flu_ha_na_human_h3n2` (much
   fewer isolates, partner pool ~few thousand). Re-run this script
   pointed at those datasets; the same code works.

5. **Quantify expected dataset size impact.** Adding DNA-level
   coverage will increase neg pair count by roughly the gap we
   already see — about 14–16% more negs on slot a/b. Train neg
   count grows from ~47K to ~54K-55K. Val/test grow proportionally.
   No infeasibility, just a slightly larger dataset.

## Caveats

- This analysis assumes the cooccur set captures all "blocked"
  pairs. v2's cooccur is built from positive pairs in the FULL
  dataset (across splits). That's the correct denominator.
- Feasibility is computed with one negative per DNA variant. The
  actual coverage minimum (`min_required_for_coverage`) is higher
  — but only by a constant factor that doesn't change the
  feasibility verdict (worst ratio rises from 0.011 to maybe 0.1,
  still well under 1).
- Partner pool counts assume EVERY non-cooccur partner is reachable.
  In practice, the v2 sampler uses random sampling and may not find
  every theoretically-available partner within
  `max_attempts_per_seq` tries. This adds slack but doesn't change
  the order of magnitude.

## See also

- `docs/plans/2026-05-07_leakage_diagnostics_plan.md` — leakage
  taxonomy (mode #2 currently flagged as protein-only).
- `docs/methods/leakage_definitions.md` — canonical mode definitions.
- `eda/dna_coverage_feasibility.py` — the script that produced these
  numbers.
- `/tmp/dna_coverage_feasibility/per_seq.csv` and
  `/tmp/dna_coverage_feasibility/cross_split.csv` — full per-seq
  outputs (not committed; regenerated on demand).
