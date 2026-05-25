# cluster_disjoint feasibility on HA-NA — bilateral vs single-slot, aa vs nt

Pre-flight feasibility for the planned idXX-sweep experiment. Two
questions:

1. Under symmetric `easy-linclust` on both alphabets (post-2026-05-22
   algorithm switch), which idXX thresholds support 80/10/10
   cluster_disjoint splits on Flu A HA-NA at full-corpus scale?
2. If we relax to **single-slot** cluster_disjoint (only one of HA or
   NA constrained, the other unconstrained), how much idXX headroom
   do we gain?

Motivation: previous bilateral cluster_disjoint experiments on Flu A
HA-NA (`docs/results/2026-05-15_cluster_disjoint_nt_results.md`)
showed only id100 and id099 were operable. The idXX-sweep narrative
("gradual MMD shift → gradual perf drop") needs more than two
cluster points. Single-slot was raised as the candidate path to a
real sweep, and was already on the table as one of three ways to push
below id099 (others: boundary-drop, absolute-mutation-tolerance
clustering).

## Setup

- Scripts:
  - `src/analysis/cluster_disjoint_feasibility.py` — bilateral
    bipartite-component pre-flight (existing).
  - `src/analysis/single_slot_cluster_disjoint_feasibility.py` —
    single-slot atom pre-flight (new in this commit). Reuses
    `build_isolate_pairs` and the cluster lookup loader from the
    bilateral script so the dedup, hashing, and cluster artifact
    paths match exactly.
- Corpus: Flu A `data_version=July_2025`, full corpus.
  - aa input: `data/processed/flu/July_2025/protein_final.parquet`
    (109,418 sequences across the 8 major proteins).
  - nt input: `data/processed/flu/July_2025/cds_final.parquet`.
- Schema pair: `("Hemagglutinin precursor", "Neuraminidase protein")`.
- Cluster artifacts: symmetric `easy-linclust` for both alphabets,
  written under `data/processed/flu/July_2025/clusters_{aa,nt}/idNN/`.
- Thresholds tested: 1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.90, 0.85.
- 80/10/10 feasibility = (largest atom ≤ 80% of pairs) AND (second
  atom ≤ 20%).
  - Bilateral: "atom" = bipartite component of (HA cluster, NA cluster).
  - Single-slot: "atom" = single-slot cluster + all its pairs (the
    other slot is free to be random).

n_pairs after dedup on `(seq_hash_a, seq_hash_b)`:
- aa: 58,826 unique protein pairs.
- nt: 79,347 unique DNA pairs (higher because synonymous-codon
  variants are distinct at the nt level — consistent with
  `clustering_overview.md` § 3.2 ratio ~1.5×).

## Results — bilateral

Bipartite-component largest CC as a fraction of n_pairs at each idXX.

| idXX | aa: n_CC | aa: largest CC % | aa: fea | nt: n_CC | nt: largest CC % | nt: fea |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 11,743 |  48.98 | ✓ | 44,396 |  1.52 | ✓ |
| 0.99 |  4,125 |  79.58 | ✓ |  5,710 | 69.35 | ✓ |
| 0.98 |  2,235 |  88.38 | ✗ |  1,641 | 91.03 | ✗ |
| 0.97 |  1,185 |  92.70 | ✗ |    689 | 95.66 | ✗ |
| 0.96 |    643 |  95.82 | ✗ |    371 | 97.75 | ✗ |
| 0.95 |    390 |  97.79 | ✗ |    218 | 98.16 | ✗ |
| 0.90 |     20 |  99.75 | ✗ |     22 | 99.12 | ✗ |
| 0.85 |      4 |  99.99 | ✗ |      5 | 99.56 | ✗ |

Bilateral verdict — **same ceiling as the 2026-05-15 result**: only
id100 and id099 are operable, on both alphabets. The 2026-05-22
algorithm switch did not unlock id098 at full-corpus scale on
HA-NA. Both alphabets cliff sharply between id099 and id098 (aa
80% → 88%; nt 69% → 91%, even larger jump).

nt has slightly more bilateral headroom at id099 (largest CC 69.35%
vs aa 79.58%) but ties or loses at every other threshold. nt at id100
is ~1.5% largest CC vs aa 48.98% — the aa space at exact-match
collapses many identical proteins across isolates that nt does not
collapse because of synonymous-codon variation.

## Results — single-slot

Single-slot atom largest as a fraction of n_pairs. "HA-only" =
constrain HA clusters only (NA free); "NA-only" = constrain NA
clusters only (HA free).

**aa (58,826 pairs):**

| idXX | HA-only largest % | HA-only fea | NA-only largest % | NA-only fea |
|---:|---:|---:|---:|---:|
| 1.00 |  0.68 | ✓ |  8.89 | ✓ |
| 0.99 |  6.08 | ✓ | 10.75 | ✓ |
| 0.98 |  8.02 | ✓ | 10.72 | ✓ |
| 0.97 | 11.29 | ✓ | 10.67 | ✓ |
| 0.96 | 11.52 | ✓ | 15.51 | ✓ |
| 0.95 | 13.47 | ✓ | 15.49 | ✓ |
| 0.90 | 25.42 | ✗ (2nd=21.35>20) | 22.65 | ✓ |
| 0.85 | 27.12 | ✗ (2nd=21.68>20) | 34.02 | ✓ (2nd=18.80<20) |

**nt (79,347 pairs):**

| idXX | HA-only largest % | HA-only fea | NA-only largest % | NA-only fea |
|---:|---:|---:|---:|---:|
| 1.00 |  0.32 | ✓ |  0.49 | ✓ |
| 0.99 |  8.25 | ✓ | 11.37 | ✓ |
| 0.98 | 11.23 | ✓ | 13.45 | ✓ |
| 0.97 | 14.07 | ✓ | 22.34 | ✓ |
| 0.96 | 13.96 | ✓ | 24.74 | ✓ |
| 0.95 | 22.90 | ✓ | 28.39 | ✗ (2nd=21.37>20) |
| 0.90 | 30.62 | ✗ | 24.16 | ✗ (2nd=24.07>20) |
| 0.85 | 33.04 | ✗ | 35.99 | ✗ |

Single-slot verdict — **a real sweep is now operable**:

- **aa is the more permissive alphabet.** On both HA-only and NA-only
  every threshold from id100 down to id095 is feasible. That is 6
  monotonic thresholds × 2 slots = 12 operable configurations.
- nt is feasible down to id097 on both slots; id096 OK on HA-only
  only; id095 OK on HA-only only (NA-only fails at id095 because
  2nd=21.37%); id090 and id085 both fail.
- aa-NA-disjoint stays feasible all the way down to id085 — the
  most relaxed single-axis available.

## Headline observations

1. **Bilateral cluster_disjoint on HA-NA Flu A: id099 is still the
   floor.** Both aa and nt collapse at id098. The bilateral
   idXX-sweep is limited to {id100, id099} — two points, insufficient
   for a "gradual perf drop" story.
2. **Single-slot cluster_disjoint on aa unlocks the full intended
   sweep range.** id100 / id099 / id098 / id097 / id096 / id095 are
   all feasible on both HA-only and NA-only. 6 monotonic thresholds
   per slot.
3. **The constrained-slot atom grows smoothly with id↓ on aa
   single-slot,** unlike bilateral where it cliffs at id098. HA-only
   aa largest atom: 0.68% → 6.08% → 8.02% → 11.29% → 11.52% → 13.47%
   across id100/099/098/097/096/095. NA-only aa: 8.89% → 10.75% →
   10.72% → 10.67% → 15.51% → 15.49%. Both are gradients, not cliffs.
4. **nt single-slot is feasible to id097 on both slots and id095 on
   HA-only.** Less headroom than aa but still meaningfully wider than
   the bilateral floor.

## Implications for the idXX-sweep experiment

The natural primary sweep is **aa, HA-only, id100/099/098/097/096/095**
— 6 monotonically increasing constrained-atom percentages, all
feasible at 80/10/10. The aa, NA-only path is a sibling cross-check
(also 6 monotonic points, also feasible) and helps disambiguate
HA-specific vs slot-symmetric effects.

Per the "MMD on both slots + pair" framing this opens:
- The constrained-slot S1 MMD (e.g., HA under HA-only) is expected
  to grow monotonically with id↓ — it is the metric most directly
  sensitive to the cluster split.
- The unconstrained-slot S1 MMD (e.g., NA under HA-only) is expected
  to stay near the random baseline — no cluster constraint applied
  to that slot.
- The pair (S2) MMD is expected to grow partially — joint effect of
  the constrained-slot shift dampened by the unconstrained-slot
  invariance.
- Model performance is expected to degrade partially, because the
  unconstrained-slot signal still generalizes from train to test.

This experimental design produces a sharper validation of MMD as a
split-separation diagnostic than bilateral cluster_disjoint does:
"the constrained slot moves; the unconstrained slot doesn't; the
pair moves a known fraction of the constrained-slot shift" is a
falsifiable prediction, not an after-the-fact rationalization.

What this pre-flight does NOT establish:

- We have not measured MMD or trained any model on a single-slot
  dataset yet. The feasibility is structural (cluster sizes), not
  empirical (we have not built a single-slot dataset under the v2
  builder; the routing helpers will need a `single_slot` mode added).
- The 80/10/10 feasibility criterion is the same as for bilateral
  (largest atom ≤ 80%, second ≤ 20%). In practice the bin-packing
  may also need to honor the "isolate share" target per split; this
  is more permissive than the atom criterion when atoms are small
  but can be tighter when many similar-size atoms compete for the
  same split.
- PB2-PB1 single-slot feasibility was not tested here. PB2 and PB1
  are more conserved than HA / NA so the single-slot relaxation may
  buy more or less depending on the cluster size distribution; would
  need its own pre-flight.

## Reproduce

```bash
# Bilateral feasibility (existing script):
python -m src.analysis.cluster_disjoint_feasibility \
    --protein_final data/processed/flu/July_2025/protein_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_aa \
    --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85

python -m src.analysis.cluster_disjoint_feasibility \
    --cds_final     data/processed/flu/July_2025/cds_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_nt \
    --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85

# Single-slot feasibility (new script):
python -m src.analysis.single_slot_cluster_disjoint_feasibility \
    --protein_final data/processed/flu/July_2025/protein_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_aa \
    --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85

python -m src.analysis.single_slot_cluster_disjoint_feasibility \
    --cds_final     data/processed/flu/July_2025/cds_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_nt \
    --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85
```

Outputs land under
`results/flu/July_2025/runs/cluster_disjoint_feasibility/` —
`feasibility_ha_na_{aa,nt}.csv` (bilateral) and
`single_slot_feasibility_ha_na_{aa,nt}.csv` (single-slot).

## See also

- `docs/results/2026-05-15_cluster_disjoint_nt_results.md` —
  Experiment B-nt: pre-switch bilateral feasibility ceiling on Flu A.
- `docs/results/2026-05-22_aa_cluster_algorithm_validation_results.md` —
  symmetric `easy-linclust` switch validation; this pre-flight uses
  the post-switch aa artifacts.
- `docs/results/2026-05-21_bicc_pair_drop_audit.md` — boundary-sample
  drop audit; alternative path to push below id099 bilaterally.
- `docs/results/2026-05-24_mmd_per_slot_results.md` and
  `docs/results/2026-05-24_mmd_per_pair_results.md` — the MMD metric
  this pre-flight prepares datasets for.
- `BACKLOG.md` § "Algorithm-switch follow-ups" — rebuild &
  re-validate tasks the upcoming sweep work absorbs.
- `src/analysis/cluster_disjoint_feasibility.py` —
  `src/analysis/single_slot_cluster_disjoint_feasibility.py` — the
  scripts.
