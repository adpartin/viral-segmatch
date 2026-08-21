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

The two arms hold identical rows, so the gap between them comes from the partition. The random arm
is also flat across folds (std 0.0042); the 2D-CD arm is not -- fold 0 is near chance (AUC-ROC
0.5454) and folds 1 and 2 are around 0.76. No answer can be read off three folds that disagree this
much, hence: what is different about fold 0?

**Exp 2 — metadata per fold.** Host is ~87% Human throughout, so it cannot separate the folds. Year
can: 2D-CD fold 0 trains at a median year of 2017 (14.5% of isolates >= 2021) and tests on 2024
(97.3%), while folds 1 and 2 train on recent isolates and test on older ones. The random arm sits
at ~42% >= 2021 on both sides of every fold. CCs tend to be year-concentrated -- median year span
within a CC is 6 against 91 across the dataset, though two of the eight largest CCs span ~80 years
-- so routing whole CCs also sorts by time to a degree. That suggested the model cannot match pairs
from future years.

**Exp 3 — temporal split, no 2D-CD.** `dataset_h3n2_nt_cds_temporal_2025`: H3N2, train 2015-2024,
test 2025, single split. **F1 macro=0.8039, AUC-ROC=0.8937, precision=0.714, recall=0.991.** So
predicting a later year is not by itself what fold 0 fails at, and Exp 2's explanation does not
hold. Note this split is not cluster-disjoint (`metadata_holdout` forces `mode: random`), so a 2025
sequence may share a cluster with a 2024 training sequence; 6.3% of test HA and 7.9% of test NA
sequences appear verbatim in train.

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

The FPR (FPR=FP/N) tracks the within-CC share almost exactly on folds 0 and 1, less so on fold 2.
Positives are separated from cross-CC negatives almost perfectly (AUC-ROC 0.91-0.99) and from
within-CC negatives barely (0.51-0.71), so each fold's score reflects the mix of the two -- a mix
set by which CCs landed in test, not by design.

Fold 0 is then not anomalous: it is the only fold whose negatives are mostly within-CC, which is
what 2D-CD is meant to test. Folds 1 and 2 reach 0.76 on test sets where most negatives are
cross-CC. Positives obey the cluster-disjoint constraint and negatives do not, so the 3 folds are
not measuring the same task.

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

## Result

**Outcome 1**, with one number to note: the folds agree, and they agree just below the band that
outcome was written with. F1 macro came out at 0.4926 mean against the stated ~0.5-0.6, with two of
the three folds under 0.5. AUC-ROC is inside its ~0.5-0.7 band, at the bottom of it.

| scope | arm | F1 macro by fold | mean | std |
|---|---|---|---|---|
| within_fold | 2D-CD | 0.4163 / 0.7583 / 0.7749 | 0.6498 | 0.2024 |
| within_cc | 2D-CD | 0.4625 / 0.5284 / 0.4868 | **0.4926** | **0.0334** |
| within_fold | random | 0.9501 / 0.9539 / 0.9584 | 0.9541 | 0.0042 |
| within_cc | random | 0.8588 / 0.8552 / 0.8516 | 0.8552 | 0.0036 |

2D-CD AUC-ROC under within_cc is 0.5169 / 0.5356 / 0.5147 -- at or just above 0.5 in each fold.
The spread across folds fell 6x (std 0.2024 -> 0.0334). Fold 0 is no longer the odd one out; all
three folds now score about what fold 0 scored before.

**Answer to the research question: no.** With negatives drawn inside the same CC as the positives,
the model does not match H3N2 HA-NA segments better than chance at t099.

What supports reading it that way:

- **The random arm on the SAME rows still works** (0.8552, std 0.0036). Both arms hold identical
  rows -- `build_random_arm` re-cuts each fold's own rows -- so the gap between them is the
  partition, not the data. Within-CC negatives are therefore learnable in principle; they are not
  learnable across a cluster-disjoint split.
- **Training on within-CC negatives did not change the outcome.** This was the reason to run rather
  than read the stratification off the Exp 1 models, which had seen mostly cross-CC negatives.
  These models saw within-CC negatives throughout and still sit at ~0.52 AUC-ROC, so the earlier
  0.5140 / 0.7057 / 0.6089 was not an artifact of the training distribution.
- **The errors moved from one side to the other.** Under within_fold the model over-predicted
  positives (fold 0: recall 0.981, 7,994 FP). Under within_cc fold 0 has recall 0.227 with 6,614
  FN. With AUC-ROC at 0.52 the ranking carries almost no signal either way, so which side the
  errors fall on says little.

Comparing the two random arms (0.9541 vs 0.8552) is NOT a like-for-like measure of what harder
negatives cost: the two scopes hold different rows, since within_cc drops the negative-infeasible
CCs and draws different negatives. The within-scope arm comparison is the controlled one.

Feasibility, replacing the risk flagged before running: k=3 held. 520 CCs were dropped, not the 493
singletons predicted. Negative-infeasible is a superset of singleton -- per the glossary it also
covers dense CCs where every cross pairing co-occurs -- which is the likely source of the extra 27,
though they were not inspected individually. The drop cost 551 positives (2.1% against a 1.9%
estimate), leaving 25,679 positives and 25,657 within-CC negatives across 266 CCs. Folds came out
balanced to within 4 rows, and no cluster is shared between test and train/val on either slot in
any fold.

## What was built

1. `cluster_memb_nt_cds_cm0_h3n2.parquet`, via
   `build_cluster_membership.py --alphabet nt_cds --clusters_root .../clusters_nt_cds_cm0_h3n2
   --out .../cluster_memb_nt_cds_cm0_h3n2.parquet`. Only HA and NA are populated; the other six
   functions have no cluster parquet in this root and come out empty, which is expected.

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

2. Bundle `flu_ha_na_cc_nt_cds_cm0_h3n2_within_cc.yaml`, inheriting
   `flu_ha_na_cc_nt_cds_cm0_h3n2` and setting `negative_scope: within_cc`, `membership_path` to the
   table from step 1, and `drop_negative_infeasible_ccs: true`. The three go together: the existing
   `false` is justified in the cm0 bundle by within_fold giving singleton CCs cross-CC negatives,
   which no longer holds.

   `membership_path` must be set explicitly. Cluster ids are unique across slots (`HA_123` vs
   `NA_45`) but NOT across cluster sets: `HA_0` exists in both `clusters_nt_cds_cm0` and
   `clusters_nt_cds_cm0_h3n2` and means a different set of sequences in each. Pointing at the wrong
   membership table would join cleanly and build pools from the wrong sequences -- a wrong answer
   rather than an error.
3. `dataset_cc_nt_cds_cm0_h3n2_t099_within_cc`, and `..._within_cc_random` from it via
   `build_random_arm.py`, which re-cuts each fold's own rows at that fold's own sizes so both arms
   hold identical rows.
4. Six LGBM runs, `lgbm_cc_nt_cds_cm0_h3n2_t099_within_cc[_random]_fold{0,1,2}`.

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
