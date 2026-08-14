# 2D-CD vs random splits on identical rows, HA-NA nt_cds, t099-t097

```yaml
# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-08-13
schema_pair:    HA-NA
alphabet:       nt_cds
clusters:       clusters_nt_cds_cm0/{t099,t098,t097}
bundle:         flu_ha_na_cc_nt_cds_cm0
k_folds:        4
negative_scope: within_fold
edge_cut:       spectral, max_drop_frac 0.05, seed 42
model:          LGBM on kmer_nt_cds k=6, interaction concat, slot_transform unit_norm
builder_commit: cde91b0        # all 6 datasets and all 24 runs; see each artifact's own `code`
depends_on:     [src/datasets/dataset_pairs_cc.py, src/datasets/build_random_arm.py]
datasets:       data/datasets/flu/July_2025/runs/dataset_cc_nt_cds_cm0_{t099,t098,t097}[_random]
models:         models/flu/July_2025/runs/lgbm_cc_nt_cds_cm0_{t099,t098,t097}[_random]_fold{0..3}
```

## What was done

For each threshold, one 2D-CD dataset was built and a random control arm derived from it:

1. **2D-CD arm** — `dataset_pairs_cc.py`. Atoms are connected components of the cluster-level
   bigraph; the mega-CC is fragmented by spectral edge min-cut within a 5% drop budget; folds come
   from `GroupKFold(shuffle=False)` on `atom_id`. Val is carved from the non-test rows at row level.
2. **Random arm** — `build_random_arm.py`. For each fold independently, its train/val/test rows are
   pooled, shuffled (seed 42), and re-cut at that fold's original three sizes. Both arms therefore
   hold **the same rows** in **the same per-split counts**; only fold membership differs.
3. **Training** — LGBM, identical configuration for both arms, `--dataset_dir` the only difference.

Negatives are drawn per split by `within_fold_negatives` during the 2D-CD build. The random arm
**moves** those rows; it does not redraw them.

## Results (test split)

| t | arm | macro F1 | sd | AUC-ROC | sd |
|---|---|---:|---:|---:|---:|
| t099 | 2D-CD | 0.8545 | 0.0224 | 0.9005 | 0.0130 |
| t099 | random | 0.9542 | 0.0029 | 0.9880 | 0.0005 |
| t098 | 2D-CD | 0.7767 | 0.0331 | 0.8508 | 0.0207 |
| t098 | random | 0.9544 | 0.0015 | 0.9874 | 0.0005 |
| t097 | 2D-CD | 0.4907 | 0.1506 | 0.6074 | 0.0963 |
| t097 | random | 0.9529 | 0.0035 | 0.9869 | 0.0014 |

Macro F1 gap (random minus 2D-CD): **+0.0997**, **+0.1777**, **+0.4622** at t099, t098, t097.

Per-fold macro F1:

| t | 2D-CD | random |
|---|---|---|
| t099 | 0.8725 / 0.8258 / 0.8720 / 0.8476 | 0.9572 / 0.9527 / 0.9510 / 0.9561 |
| t098 | 0.7899 / 0.7322 / 0.8103 / 0.7743 | 0.9527 / 0.9550 / 0.9535 / 0.9562 |
| t097 | 0.3644 / 0.3673 / 0.5621 / 0.6689 | 0.9525 / 0.9512 / 0.9499 / 0.9579 |

The random arm ranges 0.9499-0.9579 across all 12 of its folds. The 2D-CD arm ranges 0.3644-0.8725.

## Split structure

| t | atoms after cut | pairs kept | dropped | largest atom | fold_0 train / val / test |
|---|---:|---:|---:|---:|---|
| t099 | 5,731 | 75,248 | 3,516 (4.46%) | 9.75% | 97,822 / 15,050 / 37,624 |
| t098 | 1,670 | 75,947 | 2,817 (3.58%) | 13.84% | 98,730 / 15,190 / 37,974 |
| t097 | 707 | 77,458 | 1,306 (1.66%) | 26.92% | 97,718 / 15,492 / 41,706 |

Routed pairs span 75,248-77,458 across the three thresholds (2.9%). Val is the same size in all
four folds of a threshold. Test folds are 25.0% each at t099 and t098; at t097 they are
26.9 / 25.0 / 24.0 / 24.0%, and the largest test fold consists of a single connected component.

## Verification

- **Row identity.** `build_random_arm` asserts per fold that no `pair_key` repeats in the pooled
  rows, that each output split matches its input size, and that the fold read back from disk covers
  the source's rows. All three passed for every fold.
- **Reproducibility.** An independent rebuild and retrain gave **0 mismatches across 72 fold CSVs**
  (md5) and **0 mismatches across 24 runs** (every test metric). The datasets reported here were
  then rebuilt a further time under new names and reproduced the table to four decimal places.
- **Split guard.** `pytest -m production_split` passes against
  `tests/golden/production_splits/2d_cd_t099.json`, re-captured at `cde91b0`.
- **Provenance.** Every dataset and run records `cde91b0`, clean.

## Scope

- One schema pair (HA-NA), one alphabet (nt_cds), one cluster construction (cm0), one model (LGBM
  on k-mer k=6). Not tested on other pairs, alphabets, cluster sources, or models.
- Three thresholds. t096 and t095 were not built.
- Val is in-distribution in both arms: it is carved at row level, so it shares atoms with train.
  Early stopping therefore uses an in-distribution signal in both arms, including the 2D-CD arm
  whose test set is not.
- The random arm's negatives were manufactured under the 2D-CD partition and moved, not redrawn.
  This is what makes the row sets identical; it also means the comparison holds the negative set
  fixed rather than re-deriving it per partition.
- At t097 the 2D-CD per-fold values span 0.3644-0.6689, so the four folds are not interchangeable
  and the mean summarises tests of differing composition.
