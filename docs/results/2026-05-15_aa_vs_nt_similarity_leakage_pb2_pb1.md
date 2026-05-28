# aa-vs-nt similarity leakage on the PB2/PB1 regimes dataset

**Date.** 2026-05-15.
**Dataset.** `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_regimes_ratio3_20260513_222433` (PB2/PB1, regimes-based negatives, `split_strategy.mode=seq_disjoint`, `hash_key=seq`, neg_to_pos_ratio=3.0).
**Script.** `src/analysis/similarity_leakage_aa_vs_nt.py`.
**Reproduce.**
```
python src/analysis/similarity_leakage_aa_vs_nt.py \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_regimes_ratio3_20260513_222433
```

## Why this run

The companion HA/NA diagnostic (`docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md`)
reported a 6–10 pp aa-tail-excess and flagged PB2/PB1 as the next pair to
check: "protein-level conservation is tighter on the polymerase subunits,
so the aa-vs-nt asymmetry could be more pronounced there." This is the
PB2/PB1 follow-up, run before any commitment to Experiment B-nt
(`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`).

## Method

Same metric as HA/NA: for each unique test sequence (slot a or b, aa or
dna), compute the maximum percent identity over all unique train
sequences via per-position byte-equality after right-padding with
distinct pad bytes (train=0, test=255). Train and test sequence sets are
deduplicated before comparison.

The dataset has 21,064 test pairs that collapse to 4,993 unique PB2
proteins and 4,938 unique PB1 proteins (and similarly ~5,200 unique
contigs on either side after deduplication).

## Results

| Side / level | Train unique | Test unique | Mean % | p50 % | p90 % | p99 % | Max % | % ≥ 99.5% | % < 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PB2 **aa**  | 23,444 | 4,993 | 99.57 | 99.74 | 99.87 | 99.87 | 99.87 | **71.70** | 1.34 |
| PB2 **dna** | 38,623 | 5,210 | 98.68 | 99.36 | 99.91 | 99.96 | 99.96 | 42.71 | 15.53 |
| PB1 **aa**  | 21,104 | 4,938 | 99.59 | 99.74 | 99.87 | 99.87 | 99.87 | **74.34** | 1.07 |
| PB1 **dna** | 38,304 | 5,198 | 98.72 | 99.35 | 99.87 | 99.96 | 99.96 | 42.61 | 14.08 |

## Interpretation

1. **Identity-disjointness holds**, as required by `hash_key=seq`. Max
   identity is capped just below 100% on every row (PB2 aa max =
   99.87% = one aa change in ~759, etc.). No exact protein or DNA
   sequence is shared between train and test.

2. **aa-tail-excess on PB2/PB1 is ~3–5× larger than on HA/NA.**
   At the ≥99.5% nearest-train-neighbor bucket:
   - PB2: 71.70% aa vs 42.71% nt → **+28.99 pp aa-tail-excess**
   - PB1: 74.34% aa vs 42.61% nt → **+31.73 pp aa-tail-excess**
   - HA (reference): 47.83% aa vs 41.58% nt → +6.25 pp
   - NA (reference): 55.06% aa vs 44.89% nt → +10.17 pp

3. **Polymerase subunits are far more conserved at the aa level than HA
   or NA.** 72–74% of test PB2/PB1 proteins have a ≥99.5%-identical
   train neighbor — almost three quarters. On HA/NA the equivalent
   figure is 48–55%.

4. **At the nt level, similarity leakage is roughly equal across all
   pairs** (~42% on PB2/PB1, ~42–45% on HA/NA). The PB2/PB1 asymmetry
   is driven entirely by aa-side conservation; nt diversity is similar
   across functions.

5. **The left tail diverges.** 14–16% of PB2/PB1 test DNA contigs are
   <95% identical to their nearest train neighbor, vs only ~1% on the
   aa side. Synonymous variation in conserved polymerase ORFs creates a
   long left tail at the nt level that disappears at the aa level.

## Implication for Experiment B-nt on PB2/PB1

The 2026-05-08 cluster-disjoint plan proposes mmseqs2 clustering at
lower nt identity thresholds as a way to block residual similarity
leakage. The mechanism only works to the extent that aa-similar
proteins co-cluster at the chosen nt threshold:

- For HA/NA, aa-tail-excess is small (6–10 pp). An nt cluster at, say,
  95% identity captures most of the aa-near-clones (which are typically
  also nt-near-clones), so cluster-disjoint splits on nt approximate
  cluster-disjoint splits on aa.
- For PB2/PB1, aa-tail-excess is large (29–32 pp). An nt cluster at 95%
  catches only the ~43% of test proteins that are nt-near-clones to
  train; the remaining ~29 pp of aa-near-clone leakage rides through
  because synonymous mutations push the matching DNA out of cluster.

Concretely: on PB2/PB1, an nt-cluster-disjoint split is a **partial
mitigation** for similarity leakage — it blocks roughly the same nt
fraction as HA/NA, but leaves a sizeable aa-near-clone residual that
nt clustering structurally cannot address.

The user opted (2026-05-15) to proceed with Experiment B-nt on PB2/PB1
anyway, with the partial-mitigation status flagged in the bundles
(`conf/bundles/flu_pb2_pb1_cluster_nt_id*.yaml`). Results should be read
as a directional comparison vs aa cluster_id99, not as a clean
leakage-blocker.

## Caveats

- Same as the HA/NA note: per-position character-equality identity
  after end-truncation, not full Needleman–Wunsch alignment. Flu
  polymerase functions have very low length variation
  (`docs/methods/gto_format_reference.md` §6.5), so this is essentially
  optimal-alignment identity.
- Unique-sequence-level metric. Pair-level counts (21,064 test pairs)
  are larger because multiple pairs may reference the same test
  protein.

## Related project entries

- `docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md` — the HA/NA
  companion that motivated this run.
- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` —
  Experiment B (aa-level cluster_disjoint, implemented 2026-05-15) and
  Experiment B-nt (nt-level cluster_disjoint, currently underway).
- `docs/methods/leakage.md` § cluster leakage (mode #4).
