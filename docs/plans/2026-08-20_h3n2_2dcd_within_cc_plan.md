# H3N2 2D-CD with within-CC negatives

**Status: IN PROGRESS**

## Research question

Can the model make accurate segment matching for H3N2 2D-CD?

## What has been run

All on HA-NA, nt_cds, cm0 clusters built from H3N2 isolates only
(`clusters_nt_cds_cm0_h3n2`), t099, k=3, LGBM on k-mer features.

**Exp 1 — 2D-CD vs random, same rows.** `dataset_cc_nt_cds_cm0_h3n2_t099` against
`..._t099_random`, which re-cuts each fold's own rows at that fold's own split sizes, so the two
arms hold identical rows and only the partition differs.

| arm | fold | F1 macro | precision | recall |
|---|---|---|---|---|
| 2D-CD | 0 | 0.4163 | 0.518 | 0.981 |
| 2D-CD | 1 | 0.7583 | 0.697 | 0.938 |
| 2D-CD | 2 | 0.7749 | 0.740 | 0.851 |
| random | 0-2 | 0.9501 / 0.9539 / 0.9584 | 0.918-0.931 | ~0.99 |

The random arm is flat, so the whole difference comes from the partition. The 2D-CD arm is not:
fold 0 is at chance and folds 1 and 2 are around 0.76. No answer can be read off three folds that
disagree this much, hence: what is different about fold 0?

**Exp 2 — metadata per fold.** Host is ~87% Human everywhere and explains nothing. Year does not:
2D-CD fold 0 trains at a median year of 2017 (14.5% of isolates >= 2021) and tests on 2024 (97.3%),
while folds 1 and 2 train on recent isolates and test on older ones. The random arm sits at ~42%
>= 2021 on both sides of every fold. Routing whole CCs routes time as a side effect, because CCs
are year-concentrated. That suggested the model simply cannot match pairs from future years.

**Exp 3 — temporal split, no 2D-CD.** `dataset_h3n2_nt_cds_temporal_2025`: H3N2, train 2015-2024,
test 2025, single split. **F1 macro=0.8039, AUC-ROC=0.8937, precision=0.714, recall=0.991.**
Forward-in-time prediction is not the problem, so Exp 2's explanation is wrong.

## Diagnosis

2D-CD routes positives so that no cluster is shared between splits. It does not constrain the
negatives: `negative_scope: within_fold` draws them at random inside each split, ignoring CC
membership. Whether a negative is within-CC (both slots in one CC, a recombination of similar
sequences) or cross-CC (the cluster pairing alone predicts the label -- the cluster shortcut) is
then decided by which CCs happened to land in that fold's test set.

Scoring the trained models by negative type:

| fold | within-CC negatives | FPR | AUC-ROC cross-CC | AUC-ROC within-CC |
|---|---|---|---|---|
| 0 | 93.4% | 91.4% | 0.9930 | 0.5140 |
| 1 | 41.4% | 40.7% | 0.9312 | 0.7057 |
| 2 | 18.7% | 29.9% | 0.9145 | 0.6089 |

The FPR (FPR=FP/N) tracks the within-CC share almost exactly on folds 0 and 1. Cross-CC
negatives are nearly always classified correctly; within-CC negatives are not. Each fold's score is
a weighted average of those two, with weights set by the sampling rather than by design.

So fold 0 is not anomalous -- it is the only fold whose negatives are mostly within-CC, which is
what 2D-CD is meant to test. Folds 1 and 2 reach 0.76 largely on cross-CC negatives that the
cluster-disjoint routing was supposed to prevent. Positives obey the cluster-disjoint constraint
and negatives do not, so the 3 folds are measuring 3 different tasks.

## What this experiment changes

`negative_scope: within_cc` draws every negative inside one CC, so no fold has cross-CC negatives
and all three measure the same task.

This experiment is worth running rather than reading off the table above, because those numbers come from
models trained mostly on cross-CC negatives. A model trained on within-CC negatives throughout may
learn a different decision function.

## Prediction (recorded before running)

We expect that the spread of scores across the folds will collapse, since the thing that caused it is removed. Three outcomes:

1. **All folds low** (F1 macro ~0.5-0.6, AUC-ROC ~0.5-0.7). Answer: no -- with the cluster
   shortcut removed, the model cannot match H3N2 segments much above chance.
2. **All folds high** (F1 macro >= 0.85). Answer: yes -- the model generalizes, and Exp 1's spread
   was an artifact of how negatives were sampled.
3. **All folds middling** (F1 macro ~0.65-0.8, consistent). Answer: modest but real signal on
   within-CC negatives. Not chance, not strong. This is neither of the first two and should be
   reported as its own result rather than argued into one of them.

Any of the three is a usable answer, because all three are consistent across folds.

## Steps

1. Build `cluster_memb_nt_cds_cm0_h3n2.parquet`:
   `build_cluster_membership.py --alphabet nt_cds --clusters_root .../clusters_nt_cds_cm0_h3n2
   --out .../cluster_memb_nt_cds_cm0_h3n2.parquet`.

   A within-fold negative is drawn from rows already in the split, so nothing extra is needed --
   which is why `dataset_cc_nt_cds_cm0_h3n2_t099` did not require this table. A within-CC negative
   is different: it recombines one isolate's slot-A sequence with another isolate's slot-B sequence
   from the same CC, so the builder needs to know which isolate carries which sequence. That
   mapping is the membership table, and none exists for these clusters
   (`_cc_helpers.build_cc_isolate_pool`, called only under `within_cc` at
   `dataset_pairs_cc.py:975`).

   **Fragmentation does not affect this table.** It stores cluster ids, one column per threshold
   (`t099`, `t098`, `t097` here -- one per tXXX subdir of the cluster root). CC ids and atom ids are
   not in it. The builder attaches those at run time: `dataset_pairs_cc.py:976-979` builds
   `cluster_id -> atom_id` from the positives *after* the edge cut, and passes it in alongside the
   table. This is safe because an edge cut splits the bigraph between nodes, and a cluster IS a
   node -- so a cut can never divide a cluster, and every cluster ends up in exactly one atom.
   (Same fact as the edge-cut floor: the heaviest cluster's pair mass cannot be split.) So build
   the table once per cluster set; changing `max_drop_frac`, the cut method or the seed does not
   require rebuilding it.

   Isolates outside H3N2 get empty cluster ids, because their sequences are not in these cluster
   parquets, and the build will report them as unmapped. That is harmless -- they are absent from
   the `cluster_id -> atom_id` map and get dropped, and the pool is restricted to the front-end
   population regardless (`dataset_pairs_cc.py:982-986`).

2. New bundle `flu_ha_na_cc_nt_cds_cm0_h3n2_within_cc.yaml`, inheriting
   `flu_ha_na_cc_nt_cds_cm0_h3n2` and setting `negative_scope: within_cc`, `membership_path` to the
   table from step 1, and `drop_negative_infeasible_ccs: true`. The three go together: the existing
   `false` is justified in the cm0 bundle by within_fold giving singleton CCs cross-CC negatives,
   which no longer holds.

   `membership_path` must be set explicitly. Cluster ids are unique across slots (`HA_123` vs
   `NA_45`) but NOT across cluster sets: `HA_0` exists in both `clusters_nt_cds_cm0` and
   `clusters_nt_cds_cm0_h3n2` and means a different set of sequences in each. Pointing at the wrong
   membership table would join cleanly and build pools from the wrong sequences -- a wrong answer
   rather than an error.
3. Build the 2D-CD dataset, then the random arm from it (`build_random_arm.py`), then train LGBM on
   both arms x 3 folds.
4. Compare against the Exp 1 table above, same metrics.

## Risk to check, not assume

A CC needs at least two pair_keys to yield a within-CC negative, so `drop_negative_infeasible_ccs`
removes the singletons: 493 of 786 CCs, holding 493 pairs (1.9%). The 293 CCs with >= 2 pairs hold
25,737 pairs (98.1%), so almost no data is lost -- but the smaller denominator raises the edge-cut
floor from 8,464/26,230 = 32.3% to 8,464/25,737 = 32.9%, against the 33.3% that k=3 needs. The
margin drops to about 0.4 points. Read `max_balanced_k` and `floor_frac_joined` out of the new
`cc_summary.json` before training; if k=3 no longer fits, that is a result about the design, not a
bug to work around.

## Already ruled out -- do not re-run

- **Time.** Exp 3 scores 0.8039 with a strict 2025 holdout. Independently, holding negative type
  fixed, fold 0 scores 0.5403 on test rows inside its training year range and 0.4935 outside, so it
  is at chance even on years it trained on.
- **Demographic shortcut leakage.** Within-CC negatives that match on host and year still score
  0.6715 (fold 1) and 0.7024 (fold 2); fold 2 scores higher on matched than on mismatched
  negatives, the opposite of a demographic explanation.
- **Cluster structure inside a CC.** Restricting to within-CC negatives that sit on a cluster pair
  the positives also use leaves fold 1 at 0.6883.

## When done

Mark this plan IMPLEMENTED and move it to `docs/plans/done/`. Write the finding to
`docs/results/` with the Exp 1-4 chain, since the answer to the research question rests on all
four and on which explanations were eliminated.
