# Phase 2 post-migration metrics (2026-06-03)

Post-migration measurements for the Phase 2 pair_key alphabet migration
(see `docs/plans/2026-06-02_phase2_pair_key_migration_plan.md`). Pairs
1-to-1 with `docs/results/2026-06-02_phase2_preflight_baselines.md` for
the regression-guard comparison.

Captured 2026-06-03 against the post-Phase-2 code (commits 1-5 of 7,
HEAD at commit 5 `feature/phase2-pair-key-migration`) and the
regenerated test datasets from commit 5.

## Two distinct experiments in one run set

| Bundle group | What's being measured | Anchor |
|---|---|---|
| aa regimes (HA-NA, PB2-PB1) | Regression guard: aa pipeline must be byte-identical to pre-flight | ε = 0 (per pre-flight repro check) |
| nt_cds cluster (HA-NA, PB2-PB1) and aa cluster (HA-NA, PB2-PB1) | Headline: bias-direction of pair_key migration on cluster_disjoint test sets | nt_cds vs aa delta on the same routing |

The cluster bundles are rebuilt under the new pair_key semantics
(commit 4 + 5). The aa cluster runs are the new aa baseline (no
pre-flight aa cluster anchor exists — those clusters were rebuilt
under linclust during commit 3, see plan §3.2). The nt_cds cluster
runs are the new nt_cds measurement. The two cluster routings differ
in BOTH cluster atoms AND pair_key alphabet — interpretation
combines both effects.

## Section A — aa regression guard (regimes bundles, ε = 0)

Pre-flight aa baseline md5 vs post-Phase-2 aa md5, on three
artifacts: `metrics.csv` (model-level metrics), `level1_neg_regimes.csv`
(per-regime negative-pair stratification), `level1_neg_regimes_agg.csv`
(regime-level rollup).

| Bundle | Algorithm | Pre-flight run dir | Post-P2 run dir | metrics.csv | level1_neg_regimes.csv | level1_neg_regimes_agg.csv |
|---|---|---|---|:---:|:---:|:---:|
| flu_ha_na_regimes | mlp | `training_flu_ha_na_regimes_20260602_204217` | `training_flu_ha_na_regimes_20260603_000002` | ✓ match | ✓ match | ✓ match |
| flu_ha_na_regimes | lgbm | `baseline_lgbm_flu_ha_na_regimes_20260602_204220` | `baseline_lgbm_flu_ha_na_regimes_20260603_000121` | ✓ match | ✓ match | ✓ match |
| flu_pb2_pb1_regimes | mlp | `training_flu_pb2_pb1_regimes_20260602_204219` | `training_flu_pb2_pb1_regimes_20260603_000802` | ✓ match | ✓ match | ✓ match |
| flu_pb2_pb1_regimes | lgbm | `baseline_lgbm_flu_pb2_pb1_regimes_20260602_204221` | `baseline_lgbm_flu_pb2_pb1_regimes_20260603_000610` | ✓ match | ✓ match | ✓ match |

(Recorded md5 values for reproducibility: HA-NA MLP metrics.csv =
`1bf8d3dd9a7d477b9dbc7745f0f37076`; HA-NA LGBM = `9339de6e2b48b1eb31f90c63b696dc43`;
PB2-PB1 MLP = `013e78ff44d47df8baddba6e32ff0942`; PB2-PB1 LGBM =
`218e2cb7b88a5202f1d01c9cc66c6d5a`.)

**Verdict: regression guard PASSES at ε = 0** across all 4 (bundle ×
algorithm) cells × all 3 post-hoc artifacts (12 md5 checks; all
match). The pair_key migration is a pure refactor for the aa
pipeline at the model-output level: identical bits in, identical
bits out.

This is consistent with the dataset-level check (commit 5): the new
aa pair tables have byte-identical `(pair_key, label)` content vs
pre-flight; only the schema differs (added `cds_dna_hash_{a,b}` as
NA on the aa flow), which doesn't reach any downstream consumer.

## Section B — nt_cds vs aa delta on cluster_disjoint (headline)

Cluster bundles trained on each of the two pair_key alphabets.
"Cluster atoms" differ (aa = protein clusters from `easy-cluster`,
nt_cds = CDS clusters from `easy-linclust`), and the pair_key
universe differs (nt_cds is inflated +34.9 % on HA-NA, +57.0 % on
PB2-PB1, per pair_key plan §4).

### F1 / MCC / AUC-ROC

| Bundle | Algorithm | aa F1 | nt_cds F1 | ΔF1 | aa MCC | nt_cds MCC | ΔMCC | aa AUC-ROC | nt_cds AUC-ROC | ΔAUC-ROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HA-NA cluster_t099 | mlp | 0.7940 | 0.8635 | **+0.0695** | 0.6427 | 0.7672 | **+0.1245** | 0.8978 | 0.9379 | **+0.0400** |
| HA-NA cluster_t099 | lgbm | 0.6938 | 0.8419 | **+0.1481** | 0.5592 | 0.7290 | **+0.1698** | 0.8707 | 0.9446 | **+0.0739** |
| PB2-PB1 cluster_t099 | mlp | 0.8382 | 0.8448 | +0.0066 | 0.7244 | 0.7362 | +0.0119 | 0.9207 | 0.9215 | +0.0009 |
| PB2-PB1 cluster_t099 | lgbm | 0.8006 | 0.7302 | **−0.0704** | 0.6751 | 0.5354 | **−0.1396** | 0.9239 | 0.8511 | **−0.0729** |

Bold = absolute delta ≥ 0.05.

### Pattern (descriptive, not yet interpreted)

- **HA-NA**: nt_cds shifts UP on both algorithms (+7 to +15 pp F1).
  Larger shift on LGBM than on MLP.
- **PB2-PB1**: MLP barely moves (+0.7 pp F1, within noise); LGBM
  shifts DOWN substantially (−7 pp F1, −14 pp MCC, −7 pp AUC-ROC).
- The cluster-atom change (aa easy-cluster → nt linclust) and the
  pair_key change (seq_hash → cds_dna_hash) are confounded in these
  rows; the delta reflects their combined effect, not either in
  isolation. Disentangling requires running aa cluster_t099 under
  the nt_cds pair_key (or vice versa), which is not in scope here.

### Run dirs

| Bundle | Algorithm | Run dir |
|---|---|---|
| HA-NA cluster_t099 (aa) | mlp | `training_flu_ha_na_cluster_t099_20260602_234436` |
| HA-NA cluster_t099 (aa) | lgbm | `baseline_lgbm_flu_ha_na_cluster_t099_20260602_234438` |
| HA-NA cluster_nt_t099 (nt_cds) | mlp | `training_flu_ha_na_cluster_nt_t099_20260602_234437` |
| HA-NA cluster_nt_t099 (nt_cds) | lgbm | `baseline_lgbm_flu_ha_na_cluster_nt_t099_20260602_234850` |
| PB2-PB1 cluster_t099 (aa) | mlp | `training_flu_pb2_pb1_cluster_t099_20260602_235037` |
| PB2-PB1 cluster_t099 (aa) | lgbm | `baseline_lgbm_flu_pb2_pb1_cluster_t099_20260602_235425` |
| PB2-PB1 cluster_nt_t099 (nt_cds) | mlp | `training_flu_pb2_pb1_cluster_nt_t099_20260602_235522` |
| PB2-PB1 cluster_nt_t099 (nt_cds) | lgbm | `baseline_lgbm_flu_pb2_pb1_cluster_nt_t099_20260602_235743` |

## Datasets used

All 6 cluster + regimes datasets are gitignored. New runs at:

- `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_t099_20260602_225401`
- `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_nt_t099_20260602_231059`
- `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_cluster_t099_20260602_233009`
- `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_cluster_nt_t099_20260602_233011`
- `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_regimes_20260602_233012`
- `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_regimes_20260602_233013`

## Implications

1. **aa pipeline regression-free** (Section A). Commit 7 (squashing
   the alphabet enum change + pair_key change + idXXX/tXXX rename
   into the main history) carries zero risk to existing aa
   experiments — the bits flowing out of Stage 4 are identical for
   the bundles tested.

2. **nt_cds is a different measurement, not "better cluster_disjoint"**
   (Section B). The headline pattern (HA-NA up, PB2-PB1 mixed) is a
   composite of (i) cluster-atom changes (aa protein cluster vs nt
   linclust CDS cluster, with different test-set composition) and
   (ii) pair_key universe inflation (positive count grows because
   silent variants of the same protein-pair now count as distinct
   positives). The right framing in the writeup is "bias-direction
   measurement", not "improvement vs aa".

3. **Disentanglement experiment not in scope** for Phase 2. Would
   need: (a) aa cluster_t099 trained under nt_cds pair_key (same
   cluster atoms, different pair-universe dedup), and (b) nt
   cluster_t099 trained under aa pair_key (different atoms, same
   dedup). Backlog as a follow-up.

## See also

- `docs/plans/2026-06-02_phase2_pair_key_migration_plan.md` §§ 3.4, 4.4, 4.5
- `docs/plans/2026-06-02_pair_key_alphabet_plan.md` § 4 (universe-size
  prediction: +34.9 % HA-NA, +57.0 % PB2-PB1 — observed to the
  decimal in commit 5)
- `docs/results/2026-06-02_phase2_preflight_baselines.md` (anchors)
- `docs/plans/2026-06-02_clustering_cleanup_plan.md` (parent plan)
