# aa-vs-nt similarity leakage in the HA/NA regimes dataset

**Date.** 2026-05-13.
**Dataset.** `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_regimes_20260512_114205` (HA/NA, regimes-based negatives, `split_strategy.mode=seq_disjoint`, `hash_key=seq`, neg_to_pos_ratio=2.0).
**Script.** `src/analysis/similarity_leakage_aa_vs_nt.py`.
**Reproduce.**
```
python src/analysis/similarity_leakage_aa_vs_nt.py \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_regimes_20260512_114205
```

## Question

Our production split is **identity-disjoint** at the aa level via
`hash_key=seq` (verified: `set(train_seq_hash_a) ∩ set(test_seq_hash_a) = ∅`
and the same for slot_b and for `dna_hash`). But identity-disjointness
does not prevent **similarity leakage** — a test protein 1 aa away from
some train protein, or a test contig 1 nt away from some train contig.

Because amino acids are more conserved than DNA across flu isolates
(synonymous mutations create silent dna variation that leaves aa
unchanged), we conjectured the aa representation should see *closer*
nearest-train-neighbors than the dna representation. This was raised
during the nt-vs-aa k-mer comparison (see
`docs/plans/2026-05-13_aa_kmer_and_cache_symmetry_plan.md`) — if the
conjecture holds, part of aa k=3's measured F1 lead on HA/NA may come
from similarity leakage rather than pure representation power.

## Method

For each unique test sequence (slot a or b, aa or dna), compute the
maximum percent identity over all unique train sequences using

  ```
  matches / max(L_train, L_test)
  ```

where `matches` is per-position equality of byte-encoded sequences.
Train and test arrays are right-padded with *distinct* pad bytes
(train=0, test=255) so pad-pad cells never inflate the match count.
The denominator `max(L_train, L_test)` penalises length mismatches
fairly. Pairwise alignment is not performed (proteins are essentially
constant-length per function in this corpus — see
`docs/methods/gto_format_reference.md §6.5`).

The metric is computed independently for:
- `slot_a aa` — HA protein sequences (`seq_a`)
- `slot_b aa` — NA protein sequences (`seq_b`)
- `slot_a dna` — HA contig DNA (`dna_seq_a`)
- `slot_b dna` — NA contig DNA (`dna_seq_b`)

Train and test sequence sets are de-duplicated before comparison.

## Results

`n` columns: number of **unique** sequences on the test side after
dedup. The dataset has 17,517 test pairs that collapse to 5,839
unique HA proteins and 5,839 unique NA proteins (and similarly
5,839 unique HA/NA contigs after deduplication).

| Side / level | Train unique | Test unique | Mean % | p50 % | p90 % | p99 % | Max % | % ≥ 99.5% | % < 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HA **aa** | 29,951 | 5,839 | 98.91 | 99.47 | 99.82 | 99.82 | 99.82 | **47.83** | 1.99 |
| HA **dna** | 42,247 | 5,839 | 97.97 | 99.32 | 99.88 | 99.94 | 99.94 | 41.58 | 6.79 |
| NA **aa** | 25,529 | 5,839 | 98.98 | 99.57 | 99.79 | 99.79 | 99.79 | **55.06** | 2.02 |
| NA **dna** | 39,417 | 5,839 | 98.35 | 99.37 | 99.86 | 99.93 | 99.93 | 44.89 | 5.20 |

## Interpretation

1. **The split is identity-disjoint as designed.** Max identity is
   capped just below 100% in every row:
   HA aa max = 99.82% (one aa change in 567), HA dna max = 99.94%
   (one nt change in ~1800), NA aa max = 99.79%, NA dna max = 99.93%.
   No exact sequence is shared between train and test on either side
   or level — consistent with the upstream guarantee from
   `hash_key=seq` (which automatically implies dna-identity disjointness
   since identical DNA implies identical AA).

2. **AA-level similarity leakage exceeds DNA-level similarity leakage.**
   For both HA and NA, the mean nearest-neighbor identity is 0.6–0.9
   percentage points higher at the aa level, and the fraction of test
   sequences with a ≥99.5% identical train neighbor is **6–10 pp
   higher on aa**:
   - HA: 47.83% aa vs 41.58% dna at ≥99.5%.
   - NA: 55.06% aa vs 44.89% dna at ≥99.5%.
   This matches the biological prediction: synonymous variation makes
   the dna representation more diverse than the aa representation,
   pushing dna nearest-neighbors slightly farther away.

3. **DNA has a longer left tail.** The "< 95% identity to nearest
   train neighbor" bucket is roughly 3× larger for dna than for aa
   (HA: 6.79% vs 1.99%; NA: 5.20% vs 2.02%). DNA carries more
   "moderate-distance" cases because identical-AA proteins can have
   substantially different DNA encodings.

## Implication for the nt-vs-aa k-mer comparison

The 2026-05-13 HA/NA training comparison (see the conversation log;
also `docs/methods/kmer_features.md`) reported:

  - aa k=3 + unit_diff+prod: Test F1 = 0.925
  - nt k=6 + unit_diff+prod: Test F1 = 0.911 (Δ = −0.014 vs aa)

The numbers in this note **do not prove** the aa k=3 lead is due to
similarity leakage, but they make it **plausible**: aa has a
non-trivially higher rate of near-identical train neighbors that the
aa k=3 model could exploit, and the magnitude of the asymmetry
(+0.6 to +0.9 pp on mean identity, +6 to +10 pp on the high-similarity
bucket) is in the same order as the F1 gap.

The clean test is **cluster-disjoint splits**: cluster proteins at,
e.g., 95% or 90% identity (mmseqs2 / CD-HIT) and require no cluster
to span train/test. Retrain both nt and aa k-mer models on those
splits. If aa's lead survives, the representation advantage is real;
if it shrinks or disappears, it was the similarity leakage. Plan:
`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`.

## Caveats

- The metric is the per-position character-equality identity after
  end-truncation to `max(L_train, L_test)`. For flu majors where
  lengths are nearly constant (std ≤ 2.8 aa) this is essentially
  optimal-alignment identity; for sequences with N- or C-terminal
  extensions / truncations, a Needleman–Wunsch alignment would give a
  more accurate identity score. No actively studied flu majors have
  large length variation, so the simplification is justified for
  this corpus.
- The diagnostic uses **unique** test sequences after dedup. Pair-level
  counts (17,517 test pairs) are larger because multiple pairs may
  reference the same test protein.
- Only the HA/NA dataset was evaluated. PB2/PB1 may behave
  differently — protein-level conservation is tighter on the
  polymerase subunits, so the aa-vs-nt asymmetry could be more
  pronounced there. Worth re-running this diagnostic with
  `--dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_regimes_20260512_114204`
  to confirm.

## Related project entries

- `docs/methods/leakage_definitions.md` — leakage mode #4 ("cluster
  leakage") is what this diagnostic targets; the current pipeline
  does not mitigate it.
- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` —
  proposed mitigation via cluster-disjoint splits.
- `docs/plans/2026-05-13_aa_kmer_and_cache_symmetry_plan.md` — the
  aa k-mer work that motivated this diagnostic.
- `docs/results/2026-05-11_exp4a_seq_disjoint_results.md` — original
  seq_disjoint routing results, which established the
  identity-disjointness guarantee that this diagnostic operates
  beyond.
