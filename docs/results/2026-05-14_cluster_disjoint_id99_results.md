# Exp B: cluster_disjoint routing at aa identity ≥ 0.99 — results

**Date.** 2026-05-14.
**Scope.** Cluster-disjoint splits (Experiment B from
`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`) on two schema
pairs (`flu_ha_na`, `flu_pb2_pb1`). Models: **LGBM** (k-mer nt k=6,
`unit_norm` + `unit_diff + prod`) and **MLP** (same features,
hidden_dims=[512,256,64], patience=20). All runs at `neg_to_pos_ratio=1.5`,
single seed (master_seed=42). Compares cluster-disjoint routing at aa
identity ≥ 0.99 against the existing seq_disjoint baseline (both using
`hash_key=seq` protein-level partitioning).

## Headline numbers (single seed)

| Schema  | Routing                 | Model | Test F1 | Test AUC | Test P | Test R | Test MCC |
|---|---|---|---:|---:|---:|---:|---:|
| HA/NA   | seq_disjoint            | LGBM | **0.931** | **0.984** | 0.919 | 0.944 | 0.885 |
| HA/NA   | seq_disjoint            | MLP  | **0.940** | **0.979** | 0.927 | 0.954 | 0.900 |
| HA/NA   | cluster_disjoint @id099 | LGBM | **0.660** | **0.869** | 0.820 | 0.552 | 0.520 |
| HA/NA   | cluster_disjoint @id099 | MLP  | **0.771** | **0.883** | 0.686 | 0.881 | 0.600 |
| PB2/PB1 | seq_disjoint            | LGBM | **0.930** | **0.983** | 0.909 | 0.953 | 0.883 |
| PB2/PB1 | seq_disjoint            | MLP  | **0.945** | **0.980** | 0.930 | 0.961 | 0.908 |
| PB2/PB1 | cluster_disjoint @id099 | LGBM | **0.759** | **0.902** | 0.814 | 0.710 | 0.619 |
| PB2/PB1 | cluster_disjoint @id099 | MLP  | **0.815** | **0.897** | 0.744 | 0.900 | 0.680 |

![Cluster-disjoint id99 vs seq_disjoint](../../results/flu/July_2025/runs/cluster_disjoint_id99_vs_seq_disjoint.png)

**Gaps (cluster_id099 − seq_disjoint, same model):**

| Schema  | Model | ΔF1 | ΔAUC | ΔMCC |
|---|---|---:|---:|---:|
| HA/NA   | LGBM | −0.272 | −0.114 | −0.365 |
| HA/NA   | MLP  | −0.169 | −0.096 | −0.300 |
| PB2/PB1 | LGBM | −0.171 | −0.081 | −0.263 |
| PB2/PB1 | MLP  | −0.131 | −0.083 | −0.228 |

**Cluster-disjoint at 99% aa identity drops test F1 by 13–27 pp and AUC
by 8–11 pp relative to seq_disjoint on the same corpus, same features,
and same model.** The seq_disjoint baseline was using a substantial
fraction of its measured accuracy on near-identical near-neighbors of
training proteins that did not share a `seq_hash` (mode #4 cluster
leakage from `docs/methods/leakage_definitions.md`).

**MLP is more robust to the stricter routing than LGBM.** On both
schema pairs the MLP gap is smaller by 4-10 pp F1:
- HA/NA MLP loses 17 pp F1; LGBM loses 27 pp.
- PB2/PB1 MLP loses 13 pp F1; LGBM loses 17 pp.

The signature of the difference is in precision vs recall: under
cluster_disjoint, LGBM keeps its precision (~0.81-0.82) but recall
collapses (0.55-0.71) — it makes confident negative calls on unfamiliar
cluster combinations. MLP does the opposite: recall stays high
(0.88-0.90) but precision drops (0.69-0.74) — it stays "willing to call
positive" on novel clusters and pays in false positives. AUC behaves
similarly across the two models, suggesting both score positives higher
than negatives in the right rank order, but their default threshold
biases differ.

## The threshold-sweep plan didn't survive contact with the data

The original sweep at thresholds {1.00, 0.95, 0.90, 0.80} (plus the
optional 0.99) cannot be done as designed on the unfiltered flu A corpus.
The per-function cluster-size distribution
(`2026-05-14_protein_redundancy_per_function.md`) suggested workable
counts at 0.95 (HA/NA largest 11–13% of unique seqs), but **bipartite-
component analysis on the actual routing input
(`src/analysis/cluster_disjoint_feasibility.py`) reveals the binding
constraint**:

| schema_pair | id100 (largest bipartite-CC % of deduped pos_df) | id099 | id095 | id090 | id080 |
|---|---|---|---|---|---|
| HA/NA   | 20.2% **FEASIBLE** | 80.0% **FEASIBLE** | 98.5% degenerate | 99.3% degenerate | 100% degenerate |
| PB2/PB1 | 38.4% **FEASIBLE** | 87.1% borderline   | 100% degenerate  | 100% degenerate  | 100% degenerate |

**Why it collapses so fast.** Flu A is dominated by a few major subtypes.
Within each subtype, both slot-a and slot-b proteins are highly conserved
across many isolates. Even at 1% drift, the slot-a clusters of H1N1
connect to the slot-b clusters of H1N1 via the thousands of H1N1
isolates → one mega-component. Below 0.99 the H1N1 and H3N2
mega-components merge into a single bipartite component that's
effectively the whole corpus. Per-function cluster sizes only constrain
the partition in marginal cases; the bipartite structure is the load-
bearing factor.

So **id099 is the strictest meaningful cluster-disjoint test on this
corpus**, and id100 is by construction nearly identical to seq_disjoint.

Each schema-pair therefore reduces to a **two-point comparison**:
seq_disjoint (≈ id100) versus cluster_disjoint id99. The threshold "sweep"
collapses to a single delta, but that delta is informative — it directly
quantifies the leakage seq_disjoint allowed.

## Why the leakage signal is bigger on HA/NA than PB2/PB1

HA/NA: 0.99-clustering folds 41,896 unique HA proteins into 11,039
clusters and 37,488 unique NA proteins into 10,184 clusters. The
seq_disjoint baseline had ~50% of its test HA proteins within ≥99.5%
aa identity of a train protein (`2026-05-13_aa_vs_nt_similarity_leakage.md`).
Cluster_disjoint id99 forces those near-neighbors into the same split.

PB2/PB1: 0.99-clustering produces fewer surviving distinct lineages
(PB2 → 7,935 clusters; PB1 → 10,782 clusters), but PB2/PB1's bipartite
graph at id099 already has its largest component covering 87% of pairs —
the actual val/test (6.4% each) only sample the long tail of rare cluster
combinations. The signal is real (Δ AUC = −0.083) but the effect size is
smaller because:
1. Most isolates already share dominant PB2/PB1 clusters, so removing
   near-neighbors trims less unique training info than on HA/NA.
2. Val/test become a niche-clade evaluation rather than a balanced one
   (train holds 87% of pairs; val and test are slivers).

The HA/NA drop is the cleaner read on leakage; the PB2/PB1 drop is
suggestive but confounded by the train-heavy partition.

## Split shapes

| Dataset                                              | Train | Val | Test | n_components | largest_comp % |
|---|---:|---:|---:|---:|---:|
| `dataset_flu_ha_na_cluster_id99_20260514_215801`     | 46,710 | 5,839 | 5,839 | 4,057 | 79.9% |
| `dataset_flu_pb2_pb1_cluster_id99_20260514_215910`   | 45,874 | 3,392 | 3,391 | 3,587 | 87.1% |

HA/NA achieved a clean 80/10/10 because the largest component (79.9%)
just fit inside the train target; LPT bin-packing filled val and test
from the tail. PB2/PB1's largest (87.1%) overflowed train; LPT
distributed the remaining ~13% across val/test, resulting in ~6.4%
each.

## What this validates

- **The cluster_disjoint routing wiring works correctly.** Audit confirms
  `cluster_id` overlap = 0 across all splits at both schema pairs, on
  positives-only and full-pairs scopes (negatives sampler doesn't pull
  partners from other splits).
- **Sanity check at id100 ≈ seq_disjoint.** Per-function id100 clustering
  produces 41,708 HA clusters and 37,102 NA clusters (vs 41,896 / 37,488
  unique sequences) — collapse ratio ~1.00. The 236 extra collapses come
  from mmseqs2's `-c 0.8` coverage merging length-related sequences.
  cluster_disjoint id100 will be within seed noise of seq_disjoint.

## Caveats

- **Single seed (master_seed=42).** No CV / no seed variance.
- **k-mer nt k=6 only.** Same conclusion may or may not hold on ESM-2 or
  k-mer aa features. 1-NN baseline not run at id099 yet (would test the
  biology-learning criterion in `docs/methods/leakage_definitions.md`).
- **PB2/PB1 id099 split is train-heavy** (87/6.4/6.4). The signal is
  reliable but val/test sizes are smaller than nominal 80/10/10 would give.
- **mmseqs2 settings: `--min-seq-id <th> -c 0.8 --cov-mode 0`** (per
  plan). At 100% identity with `-c 0.8`, mmseqs2 still merges
  length-variants; a strict-identity baseline would need `-c 1.0`.
- **Default threshold (0.5)** for binary classification on all models.
  The MLP's lower precision under cluster_disjoint may partially reflect
  a sub-optimal threshold rather than a calibration issue — AUC stays
  comparable across MLP and LGBM, so a tuned threshold could close part
  of the precision gap.

## Artifacts

Cluster artifacts (one-time, per data version):
- `data/processed/flu/July_2025/clusters/id{100,099,095,090,080}/`
- `data/processed/flu/July_2025/clusters/redundancy_stats.csv`

Datasets:
- `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_20260514_215801/`
- `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_cluster_id99_20260514_215910/`

Models — LGBM:
- `models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_20260514_225408/` (seq_disjoint ratio=1.5)
- `models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_20260514_225410/` (seq_disjoint ratio=1.5)
- `models/flu/July_2025/runs/baseline_lgbm_flu_ha_na_cluster_id99_20260514_220604/`
- `models/flu/July_2025/runs/baseline_lgbm_flu_pb2_pb1_cluster_id99_20260514_220606/`

Models — MLP:
- `models/flu/July_2025/runs/training_flu_ha_na_20260513_201506/` (seq_disjoint ratio=1.5)
- `models/flu/July_2025/runs/training_flu_pb2_pb1_20260513_201510/` (seq_disjoint ratio=1.5)
- `models/flu/July_2025/runs/training_flu_ha_na_cluster_id99_20260514_225013/`
- `models/flu/July_2025/runs/training_flu_pb2_pb1_cluster_id99_20260514_225015/`

Comparison CSV + plot:
- `results/flu/July_2025/runs/cluster_disjoint_id99_full_comparison.csv` (8 rows)
- `results/flu/July_2025/runs/cluster_disjoint_id99_vs_seq_disjoint.png`

Feasibility analysis:
- `src/analysis/cluster_disjoint_feasibility.py`
- `docs/results/2026-05-14_cluster_disjoint_feasibility_ha_na.csv`
- `docs/results/2026-05-14_cluster_disjoint_feasibility_pb2_pb1.csv`

## Followup: ratio=1.5 vs ratio=3.0 on cluster_id99 (2026-05-14)

Tested the hypothesis "more negatives should reduce FPs from the cluster_disjoint
failure mode". Built `flu_{ha_na,pb2_pb1}_cluster_id99_r3` bundles
(inheriting cluster_id99 parents, overriding `neg_to_pos_ratio: 3.0`).
Re-trained LGBM + MLP on both schema pairs. Same routing, same features,
single seed.

| Schema | Model | ratio | F1 (thr=0.5) | AUC-ROC | **AUC-PR** | FP count | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| HA/NA   | LGBM | 1.5 | 0.660 | 0.869 | **0.823** | 707  | 0.081 |
| HA/NA   | LGBM | 3.0 | 0.528 | 0.862 | **0.722** | 427  | 0.024 |
| HA/NA   | MLP  | 1.5 | 0.771 | 0.883 | **0.797** | 2357 | 0.269 |
| HA/NA   | MLP  | 3.0 | 0.703 | 0.905 | **0.728** | 2109 | 0.120 |
| PB2/PB1 | LGBM | 1.5 | 0.759 | 0.902 | **0.854** | 549  | 0.108 |
| PB2/PB1 | LGBM | 3.0 | 0.653 | 0.900 | **0.757** | 632  | 0.062 |
| PB2/PB1 | MLP  | 1.5 | 0.815 | 0.897 | **0.798** | 1050 | 0.206 |
| PB2/PB1 | MLP  | 3.0 | 0.723 | 0.907 | **0.710** | 1347 | 0.132 |

![cluster_disjoint id99 — effect of neg ratio](../../results/flu/July_2025/runs/cluster_disjoint_id99_ratio_sweep.png)

**Headline: more negatives didn't help.** AUC-PR drops by 7-10 pp on every
schema × model combination, and threshold-tuned F1 drops too (val-tuned
LGBM F1 fell from 0.734 → 0.642 on HA/NA, from 0.777 → 0.691 on PB2/PB1).
AUC-ROC stayed flat or improved 2 pp for the MLP — but AUC-ROC is
misleading on imbalanced test sets; AUC-PR is the operative measure when
positives are scarce, and it deteriorates uniformly.

**Why ratio alone backfires.** The extra negatives generated at r=3 are
mostly **easy** ones — random metadata combinations the model already
separates trivially. They don't teach the decision boundary to handle the
**hard** cluster_disjoint failure mode (test proteins that are
*similar-but-not-identical* to training proteins). The model becomes more
conservative overall — FPR drops (good), but precision-recall area
shrinks (bad), because the easy negatives crowd out the gradient signal
that would have come from hard negatives.

FP count comparison confirms this for HA/NA MLP: 2,357 → 2,109 (−10 %)
— modest reduction — but FN count rose 693 → 1,530 (+121 %). The
threshold-tuning analysis shows the operating point shifts from 0.5 to
~0.20-0.25, but even at the optimal threshold, F1 is still worse than
r=1.5's optimum.

**Operational read.** Ratio=3 isn't a useful FP-fix on cluster_disjoint
data with the current random-negative sampler. The *kind* of negatives
matters more than the *count*. This validates that the regime-aware
coverage plan (`docs/plans/2026-05-14_regime_aware_coverage_plan.md`)
is the right next step: force the additional negatives to be hard ones
(high metadata-match-count) so they directly target the failure mode.

**Artifacts (r=3):**
- `data/datasets/.../dataset_flu_ha_na_cluster_id99_r3_20260514_232151/`
- `data/datasets/.../dataset_flu_pb2_pb1_cluster_id99_r3_20260514_232152/`
- `models/.../baseline_lgbm_flu_ha_na_cluster_id99_r3_20260514_232433/`
- `models/.../baseline_lgbm_flu_pb2_pb1_cluster_id99_r3_20260514_232435/`
- `models/.../training_flu_ha_na_cluster_id99_r3_20260514_232438/`
- `models/.../training_flu_pb2_pb1_cluster_id99_r3_20260514_232439/`
- `results/flu/July_2025/runs/cluster_disjoint_id99_ratio_sweep.csv`
- `results/flu/July_2025/runs/cluster_disjoint_id99_ratio_sweep.png`

## Related

- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` — Experiment B plan
- `docs/results/2026-05-14_protein_redundancy_per_function.md` — per-function
  cluster sizes at multiple thresholds + bipartite feasibility table
- `docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md` — diagnostic
  that motivated this experiment (HA/NA had ~48% of test proteins with
  a ≥99.5% aa-identity near-neighbor in train under seq_disjoint)
- `docs/methods/leakage_definitions.md` — mode #4 (cluster leakage)
  taxonomy
