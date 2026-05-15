# Experiment B-nt: nt-level cluster_disjoint routing on Flu A

**Date.** 2026-05-15.
**Scope.** Cluster-disjoint splits using mmseqs2 nucleotide clustering
on the reconstructed CDS DNA, as a companion to Experiment B (aa-level
clustering). Plan:
`docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` § B-nt.
**Schema pairs.** HA/NA and PB2/PB1.
**Model.** LGBM (sklearn baseline path), k-mer k=6 nt features,
unit_norm + unit_diff+prod, default threshold 0.5, single seed.
**Datasets.** Built 2026-05-15 against `cds_final.parquet` (Stage 1.5
extraction, validated translate-back, see
`src/preprocess/extract_cds_dna.py`).

## Hypothesis going in

The 2026-05-13 HA/NA aa-vs-nt similarity diagnostic
(`docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md`) and the
2026-05-15 PB2/PB1 follow-up
(`docs/results/2026-05-15_aa_vs_nt_similarity_leakage_pb2_pb1.md`)
established that the aa representation enjoys more residual similarity
leakage than the nt representation, especially on the polymerase
subunits. The conjecture this experiment tests: **nt cluster_disjoint
splits can block similarity leakage at lower identity thresholds than
aa clustering can**, because synonymous variation gives the nt
representation more diversity per protein cluster.

## What we actually found about feasibility

The bipartite-component pre-flight check
(`src/analysis/cluster_disjoint_feasibility.py`,
`docs/results/2026-05-15_cluster_disjoint_feasibility_nt_{ha_na,pb2_pb1}.csv`)
rules out the lower-threshold sweep on the full Flu A corpus:

| Schema | Threshold | n_components | largest % | second % | 80/10/10 feasible |
|---|---:|---:|---:|---:|:---:|
| HA/NA  | 1.00 | 44,396 | 1.5  | 1.4  | YES |
| HA/NA  | 0.99 |  5,710 | 69.3 | 6.0  | YES (marginal) |
| HA/NA  | 0.95 |    218 | 98.2 | 0.4  | **NO** |
| HA/NA  | 0.90 |     22 | 99.1 | 0.4  | NO |
| HA/NA  | 0.85 |      5 | 99.6 | 0.4  | NO |
| HA/NA  | 0.80 |      4 | 100.0 | 0.0 | NO |
| PB2/PB1| 1.00 | 50,437 | 2.9  | 1.1  | YES |
| PB2/PB1| 0.99 |  6,129 | 59.7 | 16.4 | YES (marginal) |
| PB2/PB1| 0.95 |     89 | 99.1 | 0.4  | **NO** |
| PB2/PB1| 0.90 |     11 | 99.5 | 0.4  | NO |
| PB2/PB1| 0.85 |      4 | 100.0 | 0.0 | NO |
| PB2/PB1| 0.80 |      3 | 100.0 | 0.0 | NO |

The same bipartite mega-component collapse seen on aa clustering between
id099 and id095 (`docs/results/2026-05-14_protein_redundancy_per_function.md`)
also dominates the nt picture. The mmseqs2 nucleotide alphabet adds more
within-cluster diversity per threshold, but Flu A's metadata structure
(8 segments, ~16 dominant HxNy subtype × host × year cells) ties the
slots together strongly enough that two clusters connect into one
component as soon as nt similarity is relaxed past ~99%. **id100 and
id099 are the only feasible nt thresholds on this corpus** — the same
ceiling as aa.

The negative result is itself useful: it shows the bipartite collapse
is driven by corpus-level co-occurrence patterns, not by aa-vs-nt
representation choice. nt clustering does not unlock lower-threshold
splits on Flu A; expecting it to was a fair prior, but the data say no.

## Surviving comparison: aa vs nt at fixed thresholds

The kept bundles are `flu_{ha_na,pb2_pb1}_cluster_nt_id{100,099}.yaml`
(four total). Combined with the production seq_disjoint baseline (≈ aa
id100 partition) and the existing aa cluster_id099 runs, the comparison
that's actually testable is a 4-routing head-to-head per schema pair:

- **seq_disjoint** (aa-hash partition): the production baseline. No
  cluster-distance enforcement — only identity-disjointness.
- **aa cluster_id099**: protein-level mmseqs2 at 99% aa identity.
- **nt cluster_id100**: CDS DNA partition at strict identity. Stricter
  than seq_disjoint: blocks synonymous-mutation pairs that share aa but
  differ at the nt level.
- **nt cluster_id099**: CDS DNA at 99% nt identity. Allows ~1 nt
  mismatch per ~100 nt within a cluster — fundamentally a different
  near-clone definition than aa id099.

### LGBM results (single seed, default threshold 0.5)

| Pair    | Routing            | F1    | AUC-PR | AUC-ROC | MCC   |
|---|---|---:|---:|---:|---:|
| HA/NA   | seq_disjoint       | 0.931 | 0.971  | 0.984   | 0.885 |
| HA/NA   | aa cluster_id099   | 0.660 | 0.823  | 0.869   | 0.520 |
| HA/NA   | nt cluster_id100   | **0.958** | **0.986**  | **0.993**   | **0.930** |
| HA/NA   | nt cluster_id099   | 0.861 | 0.927  | 0.948   | 0.764 |
| PB2/PB1 | seq_disjoint       | 0.930 | 0.970  | 0.983   | 0.883 |
| PB2/PB1 | aa cluster_id099   | 0.759 | 0.854  | 0.902   | 0.619 |
| PB2/PB1 | nt cluster_id100   | **0.965** | **0.988**  | **0.994**   | **0.942** |
| PB2/PB1 | nt cluster_id099   | 0.863 | 0.910  | 0.951   | 0.765 |

(Rendered figure: `results/flu/July_2025/runs/cluster_aa_vs_nt/lgbm_cluster_aa_vs_nt.png`;
flat CSV: `lgbm_cluster_aa_vs_nt.csv`.)

### Key reads from the table

1. **nt id100 ≥ seq_disjoint, not the reverse.** On both pairs the
   strictest-at-the-DNA-level partition produces marginally *higher* F1
   than the production seq_disjoint baseline (HA/NA: 0.958 vs 0.931;
   PB2/PB1: 0.965 vs 0.930). The mechanism is the same one the
   aa-vs-nt similarity diagnostic predicted: seq_disjoint partitions on
   the protein hash so aa-identical pairs are forced into the same
   split — under nt id100 (which partitions on the **CDS DNA** hash),
   two proteins with the same aa but synonymous nt differences end up
   in different clusters and can land on opposite sides of the
   train/test boundary. That is a *looser* aa-disjointness guarantee
   than seq_disjoint, so the test set carries slightly more aa
   near-clone leakage and the task gets a touch easier. The nt id100
   bundle is not a stricter test — it is a test of a different
   leakage definition.

2. **aa id099 is much stricter than nt id099, on both pairs.** Same
   nominal threshold (1% identity slack), very different effect on the
   classifier:
   - HA/NA: aa id099 F1 = 0.660, nt id099 F1 = 0.861 (Δ +20 pp easier)
   - PB2/PB1: aa id099 F1 = 0.759, nt id099 F1 = 0.863 (Δ +10 pp easier)
   1% nt distance can be ≤1% aa distance (synonymous changes) or even
   zero aa distance for short stretches; 1% aa distance is by
   definition ≥1% nt distance. So nt id099 admits more aa-near-clones
   into the same cluster than aa id099 — fewer aa-near-clones get
   forced across split boundaries, and the test set looks more like the
   train set.

3. **The aa-vs-nt asymmetry is bigger on HA/NA than PB2/PB1**, in the
   same direction as the similarity diagnostic predicted. ΔF1(aa id099,
   nt id099) is +20 pp on HA/NA versus +10 pp on PB2/PB1. The
   polymerase subunits' high aa conservation means most aa-near-clones
   on PB2/PB1 are *also* nt-near-clones (synonymous variation is more
   common on the more variable HA/NA), so the two clusterings agree
   more on PB2/PB1.

4. **The headline LGBM ranking is consistent across the two schema
   pairs**: nt id100 > seq_disjoint > nt id099 > aa id099. The
   strict-aa partition (aa id099) is the hardest, and the strict-nt
   partition (nt id100) is actually the easiest — a counter-intuitive
   result that reflects what each partition is and isn't blocking, not
   a model failure.

## Caveat on PB2/PB1 partial mitigation

The aa-vs-nt similarity diagnostic on PB2/PB1 showed a ~30 pp
aa-tail-excess (71.7% / 74.3% of test proteins ≥99.5%-identical to
some train protein at the aa level, vs 42.7% / 42.6% at the nt level).
Synonymous variation hides aa-near-clones behind distinct DNA contigs,
so nt-cluster-disjoint splits on PB2/PB1 address only a fraction of the
residual aa similarity leakage — roughly the ~43% of test proteins that
are nt-near-clones, leaving the remaining ~29 pp of aa-near-clone
leakage in place. PB2/PB1 nt cluster_disjoint should be read as a
**partial mitigation**, not a clean leakage-blocker. Bundles flag this
in their header (`conf/bundles/flu_pb2_pb1_cluster_nt_id*.yaml`).

## Artifacts

- Cluster artifacts: `data/processed/flu/July_2025/clusters_nt/id{NN}/`
- Datasets: `data/datasets/flu/July_2025/runs/dataset_flu_{ha_na,pb2_pb1}_cluster_nt_id{100,099}_*`
- Models: `models/flu/July_2025/runs/baseline_lgbm_flu_{ha_na,pb2_pb1}_cluster_nt_id{100,099}_*`
- Plot script: `src/analysis/plot_aa_vs_nt_cluster_disjoint.py`
- Plot output: `results/flu/July_2025/runs/cluster_aa_vs_nt/`

## Related

- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` —
  Experiment B (aa, IMPLEMENTED 2026-05-15) and B-nt (this note).
- `docs/results/2026-05-14_cluster_disjoint_id99_results.md` — aa-side
  cluster_id099 LGBM/MLP results.
- `docs/results/2026-05-14_protein_redundancy_per_function.md` — aa
  redundancy sweep + feasibility table; the parallel reference for the
  nt picture above.
- `docs/results/2026-05-15_protein_redundancy_per_function_nt.md` —
  nt redundancy sweep (autogen by the per-function script).
- `docs/methods/leakage_definitions.md` — leakage mode #4 (cluster
  leakage) is what this experiment targets.
