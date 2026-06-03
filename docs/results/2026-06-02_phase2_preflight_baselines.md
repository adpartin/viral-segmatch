# Phase 2 pre-flight baselines (2026-06-02)

Pre-migration anchors for the Phase 2 pair_key alphabet migration
(see `docs/plans/done/2026-06-02_phase2_pair_key_migration_plan.md`).

Captured 2026-06-02 against current (linclust) cluster artifacts.
These are the regression-guard anchors for aa (must match
byte-identically post-migration) and the comparison points for
nt_cds (expected shift; magnitude is the headline result).

## Test bundle set

| Bundle | Feature space | Routing | Coverage role |
|---|---|---|---|
| `flu_ha_na_cluster_nt_t099` | k-mer nt k=6 | cluster_disjoint (nt) | cluster routing path |
| `flu_pb2_pb1_cluster_nt_t099` | k-mer nt k=6 | cluster_disjoint (nt) | same, different biology |
| `flu_ha_na_regimes` | k-mer nt k=6 | random + regime-aware negatives | pair-universe dedup path (no cluster routing) |
| `flu_pb2_pb1_regimes` | k-mer nt k=6 | random + regime-aware negatives | same, different biology |

The cluster vs regimes split exercises the two surfaces Phase 2
touches: cluster_disjoint atom derivation (cluster bundles) and
pair-universe dedup (regimes bundles, which use random routing so
their pair_key universe size IS what shifts under the migration,
independent of cluster routing).

## Baseline metrics

| Bundle | Algorithm | Fresh? | F1 | AUC-ROC | AUC-PR | MCC | Run dir |
|---|---|---|---:|---:|---:|---:|---|
| flu_ha_na_cluster_nt_t099 | mlp | ✓ | 0.8745 | 0.9478 | 0.8982 | 0.7865 | `training_flu_ha_na_cluster_nt_id099_20260602_203518` |
| flu_ha_na_cluster_nt_t099 | lgbm | 2026-05-15 | 0.8606 | 0.9483 | — | 0.7644 | `baseline_lgbm_flu_ha_na_cluster_nt_id099_20260515_115224` |
| flu_pb2_pb1_cluster_nt_t099 | mlp | ✓ | 0.8927 | 0.9523 | 0.8979 | 0.8180 | `training_flu_pb2_pb1_cluster_nt_id099_20260602_203520` |
| flu_pb2_pb1_cluster_nt_t099 | lgbm | 2026-05-15 | 0.8625 | 0.9509 | — | 0.7654 | `baseline_lgbm_flu_pb2_pb1_cluster_nt_id099_20260515_115232` |
| flu_ha_na_regimes | mlp | ✓ | 0.9366 | 0.9785 | 0.9519 | 0.8935 | `training_flu_ha_na_regimes_20260602_204217` |
| flu_ha_na_regimes | lgbm | ✓ | 0.9270 | 0.9831 | 0.9706 | 0.8774 | `baseline_lgbm_flu_ha_na_regimes_20260602_204220` |
| flu_pb2_pb1_regimes | mlp | ✓ | 0.9401 | 0.9770 | 0.9468 | 0.8993 | `training_flu_pb2_pb1_regimes_20260602_204219` |
| flu_pb2_pb1_regimes | lgbm | ✓ | 0.9257 | 0.9830 | 0.9695 | 0.8748 | `baseline_lgbm_flu_pb2_pb1_regimes_20260602_204221` |

"Fresh ✓" rows are 2026-06-02 runs; the two LGBM rows for
`*_cluster_nt_id099` reuse the existing 2026-05-15 runs (their
datasets were built against current linclust artifacts and remain
valid; both were re-verified byte-identical via fresh re-runs as
part of the reproducibility check below). AUC-PR is "—" on those
two because the 2026-05-15 LGBM runs predate the
`avg_precision → auc_pr` rename (2026-05-18); the value is in the
old CSV under the old column name.

## Observations

1. **MLP > LGBM on cluster_nt_id099** by 1.4 pp F1 on HA-NA, 3.0
   pp F1 on PB2-PB1; AUC-ROC essentially matching. Consistent with
   the published "MLP > 1-NN > LGBM at every threshold" pattern on
   HA-NA single-slot sweeps (memory.md).
2. **Regimes scores noticeably higher** than cluster_nt_id099 (~7
   pp F1 gap). Expected: regimes uses random split (easier than
   cluster_disjoint), and the regime-aware negative sampler is the
   feature under test rather than the routing.
3. **HA-NA ≈ PB2-PB1** within each bundle pair. Performance is
   comparable across the two biological systems for both routing
   modes.

## Reproducibility check (full pipeline)

Three independent runs of `flu_ha_na_regimes` to characterize
nondeterminism in Stages 3 + 4:

- **Run A**: original 2026-06-02 baseline (dataset 203521, MLP
  204217, LGBM 204220).
- **Run B**: same dataset, fresh MLP only (MLP 210230). Isolates
  Stage 4 training determinism.
- **Run C**: fresh Stage 3 + fresh MLP + fresh LGBM (dataset
  210627, MLP 211209, LGBM newest). Isolates end-to-end pipeline
  determinism.

**Result: A == B == C, byte-identical.**

| Comparison | What's varied | Result |
|---|---|---|
| Stage 3 pair tables (A's dataset vs C's dataset) | Stage 3 rerun on same input | md5 match on train/val/test_pairs.csv |
| Stage 4 MLP metrics.csv (A vs B vs C) | Stage 4 rerun on same OR fresh dataset | every digit identical across all three |
| Stage 4 LGBM metrics.csv (A vs C) | Stage 4 rerun on fresh dataset | every digit identical |

Two independent LGBM repros of the `*_cluster_nt_id099` bundles
against the 2026-05-15 datasets also matched the 2026-05-15
metrics.csv files byte-for-byte (every digit; only the
`avg_precision → auc_pr` column rename differs).

## Implication for Phase 2 validation

Regression-guard tolerance threshold collapses from "≤ seed-variance
band (~0.005-0.01)" to **ε = 0**:

- Same seed + same data + same code = bit-exact metrics.csv on V100
  with this workload. No GPU atomic-op nondeterminism observed.
- Phase 2 aa metrics (under the new code, same datasets equivalent
  by construction) MUST match these baselines byte-for-byte. Any
  drift is a real signal — a Phase 2 bug, not rerun noise.
- Phase 2 nt_cds metrics will differ by construction (inflated
  pair-key universe). Direction + magnitude vs these anchors is the
  headline result.

## See also

- `docs/plans/done/2026-06-02_phase2_pair_key_migration_plan.md` §§ 2, 4
- `docs/plans/2026-06-02_pair_key_alphabet_plan.md` § 4 (universe-
  size deltas: +34.9 % on HA-NA, +57.0 % on PB2-PB1)
- `models/flu/July_2025/runs/` for the source run dirs (paths in
  the table above)
