# 2D-CD K-fold routing assigned atoms by count, not size — folds were badly unbalanced

```yaml
# Provenance. status: current | at-risk (inputs changed, not rebuilt) | superseded (replaced)
status:         current
date:           2026-08-09
schema_pair:    HA-NA
alphabet:       nt_cds
clusters:       clusters_nt_cds_cm0/t099
bundle:         flu_ha_na_cc_nt_cds_cm0
k_folds:        4
negative_scope: within_fold
model:          LGBM on kmer_nt_cds k=6, interaction concat
builder_commit: 2eea3b6        # see each artifact's own cv_info.json / training_info.json `code`
depends_on:     [src/datasets/dataset_pairs_cc.py, src/datasets/_megacc_cut.py]
dataset:        data/datasets/flu/July_2025/runs/dataset_cc_nt_cds_cm0_t099
models:         models/flu/July_2025/runs/lgbm_cc_cm0_t099_concat_fold{0..3}
```

**Scope:** every 2D-CD build routed by `fold_assignment: groupkfold` (the default), i.e. all of
`dataset_pairs_cc.py` except `leave_cc_out` bundles.

## What was wrong

`groupkfold_by_atom` built its splitter as `GroupKFold(n_splits=k, shuffle=True, random_state=seed)`.
In scikit-learn 1.6 those two settings select genuinely different algorithms:

- `shuffle=False` — sort atoms by pair count descending, assign each to the currently lightest
  fold. This is **LPT** greedy bin-packing; sklearn's own comment is *"Distribute samples by
  adding the largest weight to the lightest fold."*
- `shuffle=True` — permute the atoms and `np.array_split` them into k chunks of equal **atom
  count**, blind to atom size.

With atom sizes spanning three orders of magnitude (median 1 pair, max 7,340), equal-count
chunking produces wildly unequal folds.

## Measured effect on the split

| | test-fold share of positives | spread |
|---|---|---:|
| before (`shuffle=True`) | 7.8% / 19.7% / 27.9% / 44.5% | 5.7× |
| after (`shuffle=False`) | 25.0% / 25.0% / 25.0% / 25.0% | 1.00× |

Fold composition was the more serious problem. Before the fix, **fold 1 contained no atom larger
than rank 13** (its biggest was 1,367 pairs) while **fold 3 held ranks 1, 3, 4 and 7** — four of
the seven largest components in one test set. The CV was averaging over four qualitatively
different tests. After the fix each fold receives exactly one of the top-4 atoms, then the next
tier snakes back (ranks 5→fold 3, 6→fold 2, 7→fold 1, 8→fold 0), so no fold can be all-tail.

Per-fold training data went from spanning **45.4–78.8%** of the dataset to **60.8–65.0%**.

Cluster-disjointness was never at risk and is unaffected: it comes from GroupKFold's group
integrity, which is independent of `shuffle`. Verified directly — 0 clusters span two folds on
either slot, under both settings.

## Measured effect on scores

LGBM, k-mer nt_cds k=6, **interaction `concat`** in both arms, same seed, same 75,248 positives —
so the fold assignment is the only thing that differs:

| folds | macro F1 | sd | range | AUC-ROC |
|---|---:|---:|---:|---:|
| unbalanced (`shuffle=True`) | 0.7652 | 0.0528 | 0.118 | 0.8571 |
| **balanced (`shuffle=False`)** | **0.8373** | **0.0213** | **0.049** | **0.8921** |

**The variance reduction is the headline** — per-fold sd fell **60%** and the range 58%. The mean
also rose 0.072, which is partly a real gain (no fold now trains on only 45% of the data) and
partly the estimate simply being better conditioned; it is not a better model.

Both production bundles now declare `training.interaction: concat`. That is the repo default
(`flu_28_major_protein_pairs_master`); `flu_ha_na` overrides it to `unit_diff + prod` for its own
k-mer interaction sweep and the `cc` family inherits that, so production had been restoring `concat`
via a CLI override that lived nowhere in the repo. Declaring it makes the runs reproducible. On
these folds `concat` also scored higher (macro F1 0.8373 vs 0.7996), but that gap is about one
per-fold sd — read it as "not worse", not as a winner; the k-mer interaction sweep put all four
variants within seed noise.

## Fix

`src/datasets/dataset_pairs_cc.py` — `GroupKFold(n_splits=k_folds, shuffle=False)`, written
explicitly rather than relying on the default. `random_state` had to go with it (sklearn raises
if it is set while `shuffle=False`). `seed` remains a parameter: it still seeds `_carve_val_atoms`,
but no longer influences the test-fold assignment, which is now fully deterministic given the
atom sizes.

## When balance is unreachable at all

An edge cut partitions nodes, so it can never split a single cluster: the largest atom can never
fall below the heaviest single-side cluster's share of pairs (the **edge-cut floor**). Balanced
K-fold therefore requires `floor <= 1/K`, which is checkable before any build:

| t | heaviest HA cluster | heaviest NA cluster | floor | max K balanced |
|---|---|---|---:|---:|
| t099 | HA_8155 8.7% | NA_2949 8.8% | 8.8% | 11 |
| t098 | HA_560 9.1% | NA_2 11.1% | 11.1% | 9 |
| t097 | HA_4 11.4% | NA_699 19.9% | 19.9% | 5 |
| t096 | HA_1073 12.7% | NA_258 22.4% | 22.4% | 4 |
| t095 | HA_0 20.4% | NA_814 25.8% | 25.8% | 3 |

**t096 is the last threshold where k=4 is achievable.** At t095 the floor exceeds 25%, so no
assignment algorithm can balance four folds — one must carry at least a quarter of the data.
`shuffle=False` is still correct there (it is never worse), but balance is a property of the
data, not of the router. The floor is a lower bound, so max-K is an upper bound: at t099 the
floor is 8.8% but the achieved largest atom was 9.8%, capping real K at 10 rather than 11.

## Bearing on earlier 2D-CD results

`groupkfold` is the default `fold_assignment`, so every 2D-CD result except the OOD-vs-random
experiment was produced on unbalanced folds — including
`2026-06-29_cc_within_cc_vs_within_fold.md`, `2026-07-02_cc_within_fold_size_decoupling.md` and
`2026-07-14_cc_ood_threshold_size_decoupling.md`. The OOD-vs-random plan is unaffected: its
`leave_cc_out` / `paired_random` arms never call `groupkfold_by_atom`
(`dataset_pairs_cc.py:893`).

Qualitative conclusions are very unlikely to move — the within_cc-vs-within_fold gap
(~0.50 vs 0.87 AUC) is far larger than the 0.072 the fold fix moved the mean here. **The per-fold
± values are the part to distrust**, since they were inflated by uneven training sizes. That
matters most for `2026-07-14`, whose central inference compares a 0.025 threshold spread against a
±0.03–0.08 per-fold std; the std measured here fell **60%** once folds were balanced, so that
comparison needs re-checking before the "flat across t099–t097" claim is restated. None of these
have been rebuilt.

## Artifacts

- Dataset: `data/datasets/flu/July_2025/runs/dataset_cc_nt_cds_cm0_t099/fold_{0..3}`.
- Models: `models/flu/July_2025/runs/lgbm_cc_cm0_t099_fold{0..3}` — `concat`, resolved from the
  bundle rather than a CLI override, which is why the run names carry no interaction suffix.
- The pre-fix (unbalanced) build and its runs, and the `unit_diff + prod` runs behind the 0.7996
  comparison, were deleted; their numbers survive only in this document. Every artifact here
  records its builder commit in `cv_info.json` / `training_info.json`.
- Golden: `tests/golden/production_splits/2d_cd_t099.json`.
- The 1D-CD path was never affected — `_split_helpers.py:568` already used
  `GroupKFold(n_splits=...)` with no shuffle.
