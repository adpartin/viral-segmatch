# CC 2D-CD negatives: within_cc vs within_fold (HA–NA, aa, t099)

**Date:** 2026-06-29
**Status:** EXPLORATORY (single seed; aa k=3 / t099; MLP fold_0 + 5-fold LGBM). Smoke that
also validated the 2D-CD builder → Stage-4 path (plan §Phase-1(b) RESOLVED).
**Bundle:** `flu_ha_na_cc_aa` (builder `src/datasets/dataset_pairs_cc.py`).

---

## Headline

Same positives, only the **negatives** differ. Under cluster-disjoint within-CC negatives,
both a kmer-MLP and a 5-fold LGBM sit at **chance**; the cross-CC (within_fold) variant —
which leaves the cluster shortcut intact — reaches **AUC-ROC 0.87**.

| Negatives | Model | Folds | Test AUC-ROC | Test macro-F1 |
|---|---|---|---|---|
| `within_cc` | MLP | fold_0 | **0.498** | 0.381 |
| `within_cc` | LGBM | 5-fold | **0.507 ± 0.030** | 0.501 ± 0.018 |
| `within_fold` | MLP | fold_0 | **0.870** | 0.822 |

within_cc LGBM per-fold AUC-ROC: 0.483, 0.524, 0.461, 0.539, 0.528.

**Reading:** the **cluster shortcut is worth ≈ 0.37 AUC**. Same-CC recombinant negatives strip
cluster identity, leaving only the specific HA–NA pairing — which **aa k=3 kmers do not carry
under within_cc at t099**. within_fold's 0.87 is the shortcut-inflated number, not real
specific-pairing capability. Both models at chance ⇒ this is feature/setup, not model capacity.

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
| Features | kmer `aa` k=3 (`kmer_features_aa_k3`), `interaction=unit_diff`, `slot_transform=none`, `feature_scaling=none` |
| MLP | hidden `[512,256,64]`, dropout 0.2, adamw lr 1e-3, batch 128, epochs 100, early-stop f1 (patience 20) |
| LGBM | `baseline_lgbm` defaults in `conf/baselines/default.yaml` |

Dataset (both scopes): 58,388 positives → 4,099 CCs → drop 3,171 negative-infeasible →
**928 atoms** (928 pos / 928 neg); fold_0 test = 372 (186/186).

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
```

Metrics per run: `metrics_summary.json` (`test` split) under the `--run_output_subdir`.

---

## Caveats

- **Single seed** (42); **aa k=3 / t099** only. The MLP rows are **fold_0 only**; the 5-fold
  aggregate is LGBM.
- Not tested: **within_fold LGBM** (symmetry), **ESM-2** features, higher k, **nt_cds/nt_ctg**
  alphabets, **id100** threshold, multi-seed. "No learnable signal under within_cc" is scoped
  to *these features at t099* — richer features may differ.
- within_fold's high score is **not** model quality; it is the magnitude of the cluster
  shortcut. The rigorous metric is within_cc.
