# CC 2D-CD negatives: within_cc vs within_fold (HA–NA, k-mers, t099)

**Date:** 2026-06-29
**Status:** EXPLORATORY (single seed; t099; k-mer features aa k=3 + nt_cds k=6; 5-fold LGBM +
MLP, aa MLP fold_0). Smoke that also validated the 2D-CD builder → Stage-4 path
(plan §Phase-1(b) RESOLVED).
**Bundles:** `flu_ha_na_cc_aa` (aa) / `flu_ha_na_cc_nt_cds` (nt_cds); builder
`src/datasets/dataset_pairs_cc.py`.

---

## Headline

Same positives, only the **negatives** differ. Under cluster-disjoint within-CC negatives,
**every model × k-mer feature combination sits at chance**; the cross-CC (within_fold) variant —
which leaves the cluster shortcut intact — reaches **AUC-ROC 0.87**.

| Negatives | Model | Features | Folds | Test AUC-ROC | Test macro-F1 |
|---|---|---|---|---|---|
| `within_cc` | MLP | aa k=3 | fold_0 | 0.498 | 0.381 |
| `within_cc` | LGBM | aa k=3 | 5-fold | 0.507 ± 0.030 | 0.501 ± 0.018 |
| `within_cc` | MLP | nt_cds k=6 | 5-fold | **0.494 ± 0.005** | 0.333 (MCC 0.0) |
| `within_cc` | LGBM | nt_cds k=6 | 5-fold | **0.496 ± 0.009** | 0.475 ± 0.020 |
| `within_fold` | MLP | aa k=3 | fold_0 | **0.870** | 0.822 |

Per-fold within_cc AUC-ROC — aa LGBM: 0.483, 0.524, 0.461, 0.539, 0.528; nt_cds-k6 LGBM:
0.491, 0.483, 0.504, 0.506, 0.497; nt_cds-k6 MLP: 0.496, 0.499, 0.485, 0.495, 0.493.

**Reading:** the **cluster shortcut is worth ≈ 0.37 AUC**. Same-CC recombinant negatives strip
cluster identity, leaving only the specific HA–NA pairing — which **neither aa k=3 nor nt_cds
k=6 k-mers carry under within_cc at t099** (both models, both alphabets ≈ chance; the nt_cds MLP
even collapses to one class, MCC 0.0). within_fold's 0.87 is the shortcut-inflated number, not
real specific-pairing capability ⇒ a feature/setup limit, not model capacity. Richer DNA 6-mers
do **not** recover what aa 3-mers miss.

---

## Configuration (for reproducibility)

Shared (from `conf/bundles/flu_ha_na_cc_aa.yaml`, seed = 42, device = CPU):

| Knob | Value |
|---|---|
| `split_strategy.mode` | `cluster_disjoint_cc` |
| `split_strategy.cluster_alphabet` / pair_key | `aa` / `aa` (enforced equal) |
| `cluster_id_path` | `data/processed/flu/July_2025/clusters_aa/t099/combined_cluster.parquet` (t099) |
| `m_pos_per_cc` | 1 (one positive per CC → uniform atoms) |
| `neg_to_pos_ratio` | 1.0 |
| `drop_negative_infeasible_ccs` | `true` (→ same 928 atoms for both scopes) |
| `n_folds` / `val_ratio` | 5 / 0.1 |
| `negative_scope` | **`within_cc`** (default) vs **`within_fold`** (override) |
| Features | **aa**: kmer `aa` k=3 (`kmer_features_aa_k3`); **nt_cds**: kmer `nt_cds` k=6 (`kmer_features_nt_cds_k6`, via `--override kmer.alphabet=nt_cds kmer.k=6`). Both: `interaction=unit_diff`, `slot_transform=none`, `feature_scaling=none` |
| MLP | hidden `[512,256,64]`, dropout 0.2, adamw lr 1e-3, batch 128, epochs 100, early-stop f1 (patience 20) |
| LGBM | `baseline_lgbm` defaults in `conf/baselines/default.yaml` |

Dataset — **aa** (both scopes): 58,388 positives → 4,099 CCs → drop 3,171 negative-infeasible →
**928 atoms** (928/928); fold_0 test 372. **nt_cds** within_cc (bundle `flu_ha_na_cc_nt_cds`,
`clusters_nt_cds/t099`; CDS-level dedup → more unique positives): → **1,934 atoms** (1,934/1,934);
fold_0 test 774.

### Commands

```bash
PY=python   # use the segmatch env; runs done on CPU via CUDA_VISIBLE_DEVICES=""

# within_cc dataset (default scope)
$PY src/datasets/dataset_pairs_cc.py --config_bundle flu_ha_na_cc_aa \
    --out_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cc_aa_within_cc_smoke

# within_fold dataset (one override)
$PY src/datasets/dataset_pairs_cc.py --config_bundle flu_ha_na_cc_aa \
    --override dataset.split_strategy.negative_scope=within_fold \
    --out_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cc_aa_within_fold_smoke

# MLP on one fold (fold_id appends fold_K/)
CUDA_VISIBLE_DEVICES="" $PY src/models/train_pair_classifier.py \
    --config_bundle flu_ha_na_cc_aa --dataset_dir <DATASET_DIR> \
    --fold_id 0 --skip_post_hoc --run_output_subdir cc_aa_<scope>_smoke_fold0

# LGBM per fold (baseline runner has no --fold_id; point dataset_dir at the fold subdir)
for K in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES="" $PY src/models/train_pair_baselines.py \
      --config_bundle flu_ha_na_cc_aa --baseline lgbm \
      --dataset_dir <DATASET_DIR>/fold_$K --skip_post_hoc \
      --run_output_subdir cc_aa_within_cc_lgbm_fold$K
done

# nt_cds k=6 variant — nt_cds split bundle, then add the kmer override to the Stage-4 commands:
$PY src/datasets/dataset_pairs_cc.py --config_bundle flu_ha_na_cc_nt_cds \
    --out_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cc_nt_cds_within_cc
#   train with: --config_bundle flu_ha_na_cc_nt_cds --override kmer.alphabet=nt_cds kmer.k=6
```

Metrics per run: `metrics_summary.json` (`test` split) under the `--run_output_subdir`.

---

## Caveats

- **Single seed** (42); **t099**. k-mer feature family covered (aa k=3, nt_cds k=6); aa MLP is
  fold_0, the rest are 5-fold.
- Still untested: **ESM-2** features (the key one — protein semantics, not a bag of k-mers),
  **within_fold LGBM** (symmetry), **nt_ctg**, **id100** threshold, multi-seed. "No learnable
  signal under within_cc" is now established for the *k-mer family* at t099; ESM-2 may differ.
- within_fold's high score is **not** model quality; it is the magnitude of the cluster
  shortcut. The rigorous metric is within_cc.
