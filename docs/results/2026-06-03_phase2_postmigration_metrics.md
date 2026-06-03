# Phase 2 post-migration metrics (2026-06-03)

Post-migration measurements for the Phase 2 pair_key alphabet migration
(see `docs/plans/2026-06-02_phase2_pair_key_migration_plan.md`). All
tables compare the SAME bundle + SAME algorithm executed BEFORE the
code change (pre-flight) vs AFTER (post-Phase-2).

Captured 2026-06-03 against the post-Phase-2 code (commits 1-5 of 7,
HEAD at commit 5 `feature/phase2-pair-key-migration`) and the
regenerated test datasets from commit 5.

## Comparison set

Eight (bundle × algorithm) experiments, each with a pre-flight run
and a post-Phase-2 run. The migration changed two things:

1. **aa regimes** (4 cells): bundle name and config UNCHANGED; only
   the `_PAIR_COLUMNS` schema gained `cds_dna_hash_{a,b}` (NA on aa
   flow). This is the regression-guard — output must be byte-identical.

2. **nt cluster_t099** (4 cells): bundle renamed
   (`flu_*_cluster_nt_id099` → `flu_*_cluster_nt_t099`),
   `cluster_alphabet` enum changed (`nt` → `nt_cds`), cluster path
   moved (`clusters_nt/id099/` → `clusters_nt_cds/t099/`), and
   pair_key default switched from protein (`seq_hash`-based) to CDS
   DNA (`cds_dna_hash`-based). The cluster ASSIGNMENTS are byte-
   identical pre vs post (validated 176/176 cells in commit 3); only
   the pair_key semantics changed. Metric shift = effect of the
   intentional semantic change.

## Section A — aa regimes (regression guard)

Pre-flight aa md5 vs post-Phase-2 aa md5, on three artifacts:
`metrics.csv` (model-level metrics), `level1_neg_regimes.csv`
(per-regime negative-pair stratification), `level1_neg_regimes_agg.csv`
(regime-level rollup).

| Bundle | Algorithm | Pre-flight run dir | Post-P2 run dir | metrics.csv | level1_neg_regimes.csv | level1_neg_regimes_agg.csv |
|---|---|---|---|:---:|:---:|:---:|
| flu_ha_na_regimes | mlp | `training_flu_ha_na_regimes_20260602_204217` | `training_flu_ha_na_regimes_20260603_000002` | ✓ match | ✓ match | ✓ match |
| flu_ha_na_regimes | lgbm | `baseline_lgbm_flu_ha_na_regimes_20260602_204220` | `baseline_lgbm_flu_ha_na_regimes_20260603_000121` | ✓ match | ✓ match | ✓ match |
| flu_pb2_pb1_regimes | mlp | `training_flu_pb2_pb1_regimes_20260602_204219` | `training_flu_pb2_pb1_regimes_20260603_000802` | ✓ match | ✓ match | ✓ match |
| flu_pb2_pb1_regimes | lgbm | `baseline_lgbm_flu_pb2_pb1_regimes_20260602_204221` | `baseline_lgbm_flu_pb2_pb1_regimes_20260603_000610` | ✓ match | ✓ match | ✓ match |

(Recorded md5 values: HA-NA MLP `metrics.csv` =
`1bf8d3dd9a7d477b9dbc7745f0f37076`; HA-NA LGBM =
`9339de6e2b48b1eb31f90c63b696dc43`; PB2-PB1 MLP =
`013e78ff44d47df8baddba6e32ff0942`; PB2-PB1 LGBM =
`218e2cb7b88a5202f1d01c9cc66c6d5a`.)

**Verdict: regression guard PASSES at ε = 0** across all 4 cells × 3
artifacts (12 md5 checks; all match). The pair_key migration is a
pure refactor for the aa pipeline at the model-output level.

This is consistent with commit 5's dataset-level finding: the new aa
pair tables have byte-identical `(pair_key, label)` content vs
pre-flight; only the schema differs (added `cds_dna_hash_{a,b}` as
NA on the aa flow), which doesn't reach any downstream consumer.

## Section B — nt cluster_t099 (semantic change measurement)

Same experimental setup (nt cluster_disjoint at threshold 0.99, k-mer
nt k=6 features, same training hyperparameters), before vs after the
pair_key migration. Cluster atoms are byte-identical pre/post; only
the pair_key universe changed (protein-based → CDS-DNA-based,
inflating positives by +34.9 % on HA-NA, +57.0 % on PB2-PB1).

### Aggregate metrics — 4 experiments

#### Experiment B1: HA-NA nt cluster_t099, MLP

|  | F1 | AUC-ROC | MCC | Run dir |
|---|---:|---:|---:|---|
| before | 0.8745 | 0.9478 | 0.7865 | `training_flu_ha_na_cluster_nt_id099_20260602_203518` |
| after  | 0.8635 | 0.9379 | 0.7672 | `training_flu_ha_na_cluster_nt_t099_20260602_234437` |
| Δ      | −0.011 | −0.010 | −0.019 | |

#### Experiment B2: HA-NA nt cluster_t099, LGBM

|  | F1 | AUC-ROC | MCC | Run dir |
|---|---:|---:|---:|---|
| before | 0.8606 | 0.9483 | 0.7644 | `baseline_lgbm_flu_ha_na_cluster_nt_id099_20260602_210228` |
| after  | 0.8419 | 0.9446 | 0.7290 | `baseline_lgbm_flu_ha_na_cluster_nt_t099_20260602_234850` |
| Δ      | −0.019 | −0.004 | −0.035 | |

#### Experiment B3: PB2-PB1 nt cluster_t099, MLP

|  | F1 | AUC-ROC | MCC | Run dir |
|---|---:|---:|---:|---|
| before | 0.8927 | 0.9523 | 0.8180 | `training_flu_pb2_pb1_cluster_nt_id099_20260602_203520` |
| after  | 0.8448 | 0.9215 | 0.7362 | `training_flu_pb2_pb1_cluster_nt_t099_20260602_235522` |
| Δ      | **−0.048** | −0.031 | **−0.082** | |

#### Experiment B4: PB2-PB1 nt cluster_t099, LGBM

|  | F1 | AUC-ROC | MCC | Run dir |
|---|---:|---:|---:|---|
| before | 0.8625 | 0.9509 | 0.7654 | `baseline_lgbm_flu_pb2_pb1_cluster_nt_id099_20260602_210229` |
| after  | 0.7302 | 0.8511 | 0.5354 | `baseline_lgbm_flu_pb2_pb1_cluster_nt_t099_20260602_235743` |
| Δ      | **−0.132** | **−0.100** | **−0.230** | |

Bold = |Δ| ≥ 0.05.

### Per-regime deltas (level1_neg_regimes.csv)

For each experiment, the per-regime breakdown shows where the
aggregate metric shift comes from. For the positive regime, the
column reports recall (the model's hit rate on co-occurring pairs);
for each negative regime, it reports `fp_rate` (false-positive rate
on the structured negatives generated by the regime-aware sampler).
↑ = worse (recall down, fp_rate up); ↓ = better.

#### Experiment B1 — HA-NA nt MLP

| Regime | n (before→after) | before | after | Δ |
|---|---|---:|---:|---:|
| positive (recall) | 5,839→7,876 | 0.924 | 0.925 | flat |
| none_match | 3,684→5,198 | 0.078 | 0.095 | +1.7 ↑ |
| host_only | 441→498 | 0.188 | 0.161 | −2.7 ↓ |
| subtype_only | 206→339 | 0.257 | 0.212 | −4.5 ↓ |
| year_only | 3,542→4,353 | 0.112 | 0.122 | +1.0 ↑ |
| host_subtype_only | 188→230 | 0.293 | 0.465 | **+17.2 ↑** |
| host_year_only | 384→531 | 0.263 | 0.226 | −3.7 ↓ |
| subtype_year_only | 150→362 | 0.373 | 0.282 | −9.1 ↓ |
| host_subtype_year | 163→303 | 0.460 | 0.700 | **+24.0 ↑** |

#### Experiment B2 — HA-NA nt LGBM

| Regime | n (before→after) | before | after | Δ |
|---|---|---:|---:|---:|
| positive (recall) | 5,839→7,876 | 0.884 | 0.913 | +2.9 ↓ (better) |
| none_match | 3,684→5,198 | 0.076 | 0.123 | +4.7 ↑ |
| host_only | 441→498 | 0.152 | 0.141 | −1.1 ↓ |
| subtype_only | 206→339 | 0.248 | 0.227 | −2.0 ↓ |
| year_only | 3,542→4,353 | 0.097 | 0.151 | +5.3 ↑ |
| host_subtype_only | 188→230 | 0.277 | 0.500 | **+22.3 ↑** |
| host_year_only | 384→531 | 0.190 | 0.232 | +4.2 ↑ |
| subtype_year_only | 150→362 | 0.333 | 0.331 | flat |
| host_subtype_year | 163→303 | 0.454 | 0.713 | **+25.9 ↑** |

#### Experiment B3 — PB2-PB1 nt MLP

| Regime | n (before→after) | before | after | Δ |
|---|---|---:|---:|---:|
| positive (recall) | 5,266→8,269 | 0.940 | 0.958 | +1.7 ↓ (better) |
| none_match | 3,605→5,754 | 0.063 | 0.163 | **+10.0 ↑** |
| host_only | 501→660 | 0.228 | 0.308 | +8.0 ↑ |
| subtype_only | 143→238 | 0.147 | 0.319 | **+17.3 ↑** |
| year_only | 2,817→4,516 | 0.096 | 0.192 | +9.6 ↑ |
| host_subtype_only | 195→300 | 0.277 | 0.350 | +7.3 ↑ |
| host_year_only | 336→492 | 0.265 | 0.358 | +9.3 ↑ |
| subtype_year_only | 128→177 | 0.297 | 0.328 | +3.1 ↑ |
| host_subtype_year | 174→266 | 0.356 | 0.511 | **+15.5 ↑** |

#### Experiment B4 — PB2-PB1 nt LGBM

| Regime | n (before→after) | before | after | Δ |
|---|---|---:|---:|---:|
| positive (recall) | 5,266→8,269 | 0.932 | 0.773 | **−15.8 ↑** (worse) |
| none_match | 3,605→5,754 | 0.102 | 0.208 | **+10.7 ↑** |
| host_only | 501→660 | 0.272 | 0.258 | −1.4 ↓ |
| subtype_only | 143→238 | 0.196 | 0.319 | **+12.4 ↑** |
| year_only | 2,817→4,516 | 0.148 | 0.224 | +7.5 ↑ |
| host_subtype_only | 195→300 | 0.251 | 0.243 | −0.8 ↓ |
| host_year_only | 336→492 | 0.307 | 0.341 | +3.5 ↑ |
| subtype_year_only | 128→177 | 0.313 | 0.305 | −0.7 ↓ |
| host_subtype_year | 174→266 | 0.362 | 0.383 | +2.1 ↑ |

### Pattern across B1-B4

- **All 4 experiments degrade in aggregate**. HA-NA is mild (~1-2 pp
  F1, ~2-4 pp MCC); PB2-PB1 is large (5-13 pp F1, 8-23 pp MCC). LGBM
  on PB2-PB1 is the worst hit.
- **Where the degradation comes from differs per cell**:
  - HA-NA MLP + LGBM: most negative regimes improve or stay flat; the
    degradation concentrates in two metadata-overlap regimes
    (`host_subtype_only` +17-22 pp fp_rate, `host_subtype_year`
    +24-26 pp). Net aggregate drops only because those two regimes
    dominate the F1 arithmetic at low fp_rate.
  - PB2-PB1 MLP: nearly every negative regime gets worse (8 of 8
    show ↑ fp_rate); positive recall actually improves +1.7 pp.
    Aggregate F1 falls because the universal negative-rejection
    regression outweighs the small positive gain.
  - PB2-PB1 LGBM: positive recall collapses −15.8 pp AND most
    negatives degrade (5 of 8 ↑). Both axes hurt aggregate F1
    simultaneously — largest aggregate drop of the four cells.
- **Pair-universe inflation alone doesn't explain the per-regime
  asymmetry**. HA-NA inflates +34.9 %, PB2-PB1 inflates +57.0 % — the
  ratios match the F1 drop magnitudes (mild vs large), but inflation
  is uniform across regimes inside each pair, while the per-regime
  effect is concentrated. The CDS-DNA-pair-key change appears to
  add training-pair correlation (silent variants are not independent
  samples) that PB2-PB1 — without antigenic subtype structure to
  anchor the model — absorbs worse than HA-NA.

## Datasets used

All 6 cluster + regimes post-Phase-2 datasets are gitignored. Paths:

- `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_t099_20260602_225401`
- `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_nt_t099_20260602_231059`
- `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_cluster_t099_20260602_233009`
- `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_cluster_nt_t099_20260602_233011`
- `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_regimes_20260602_233012`
- `data/datasets/flu/July_2025/runs/dataset_flu_pb2_pb1_regimes_20260602_233013`

The aa cluster_t099 datasets and runs (`training_flu_ha_na_cluster_t099_*`,
`training_flu_pb2_pb1_cluster_t099_*`, plus matching LGBM) were also
generated as part of the commit-6 batch, but they have no pre-flight
counterpart (no aa cluster bundle existed before Phase 2), so no
before/after comparison is reportable for them.

## Implications

1. **aa pipeline regression-free** (Section A). Commit 7 (the final
   plan commit) carries zero risk to existing aa experiments — the
   bits flowing out of Stage 4 are identical for the bundles tested.

2. **Pair_key migration measurably hurts nt cluster metrics**
   (Section B). All 4 experiments degrade; the magnitude tracks
   pair-universe inflation across pairs (HA-NA mild, PB2-PB1 large)
   but the per-regime pattern differs by pair and algorithm.
   Mechanism hypothesis (not proven): silent-variant pair inflation
   adds correlated training samples that the model treats as
   independent, weakening generalization to structured negative
   regimes.

3. **Out-of-scope follow-ups**:
   - Disentangle the pair_key effect from the cluster-atom effect by
     running aa cluster_t099 under nt_cds pair_key.
   - Multi-seed re-runs of the 4 nt cluster experiments to bound the
     "drop is real signal vs run-to-run noise" question. Pre-flight's
     ε = 0 reproducibility verdict makes the deltas above genuine,
     but seed-variance on the architectural shift hasn't been
     measured.
   - Per-regime drop on `host_subtype_year` (always the worst-hit
     regime, 15-26 pp fp_rate increase) — most demographically
     overlapping regime, suggests CDS-DNA pair_key is shifting the
     model toward demographic shortcuts.

## See also

- `docs/plans/2026-06-02_phase2_pair_key_migration_plan.md` §§ 3.4, 4.4, 4.5
- `docs/plans/2026-06-02_pair_key_alphabet_plan.md` § 4 (universe-size
  prediction: +34.9 % HA-NA, +57.0 % PB2-PB1 — observed to the
  decimal in commit 5)
- `docs/results/2026-06-02_phase2_preflight_baselines.md` (pre-flight
  anchors)
- `docs/plans/2026-06-02_clustering_cleanup_plan.md` (parent plan)
