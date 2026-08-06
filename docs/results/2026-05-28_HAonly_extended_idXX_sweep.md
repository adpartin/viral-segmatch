# HA-NA HA-only single-slot cluster_disjoint — extended idXX sweep (id100..id080)

Holdout-only sweep extending the 2026-05-24 id100..id095 work down to the
lowest cluster-artifact threshold available (id080). Goal: identify where
the model still beats memorization (MLP > 1-NN) and where 1-NN drops
to chance.

## TL;DR

| | F1 at id100 | F1 at id080 | Δ (id100 → id080) |
|---|---:|---:|---:|
| MLP (mean ± std, 3 seeds) | 0.9621 ± 0.0013 | 0.4839 ± 0.0342 | −0.478 |
| LGBM | 0.9500 | 0.3556 | −0.594 |
| 1-NN cosine margin | 0.9580 | 0.4321 | −0.526 |
| **MLP − 1-NN gap** | +0.004 | +0.052 | — |

The MLP − 1-NN gap is the "biology learning beyond memorization" signal
(see `leakage.md` § "When we say 'the model learned
biology'"). The going-in hypothesis was that the gap would widen at
low idXX, where memorization shortcuts get harder.

**The data does not support that hypothesis.** Across id094..id085 —
the regime where cluster_disjoint actually dislodges trivial
memorization — 1-NN cosine margin matches or beats MLP at five of the
seven new thresholds (id094, id093, id092, id085). MLP only pulls
clearly ahead at id080 (+0.052), where absolute F1 ≈ 0.48 sits just
above chance. The gap oscillates around 0 rather than widening
monotonically — MLP is tracking 1-NN within noise across the
informative middle band.

## Scope

- **Pair**: Flu A HA-NA, full corpus (`data_version=July_2025`).
- **Alphabet**: aa (symmetric easy-linclust, post-2026-05-22).
- **Slot direction**: HA-only (constrained=slot a; NA unconstrained).
- **Routing**: single-slot cluster_disjoint, holdout (k=1) per
  `cluster_disjoint_route_pos_df` with `n_folds=None`.
- **Thresholds**: id100, id099, id098, id097, id096, id095 (from
  2026-05-24), id094, id093, id092, id091, id090, id085, id080 (new
  2026-05-27/28). Thirteen thresholds total.
- **Feature**: aa k=3 (8000-dim k-mer count vector), Test 3 interaction
  (`slot_transform=unit_norm`, `interaction=unit_diff+prod`).
- **Negatives**: plain coverage-first sampling, `neg_to_pos_ratio=1.5`,
  no `regime_targets`.

## Setup

- **Datasets**: thirteen Stage 3 dirs under
  `data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_aa_id{XXX}_HAonly_*`.
- **Bundles**: `conf/bundles/flu_ha_na_cluster_aa_id{XXX}_HAonly.yaml`
  (one per threshold; the seven new id094..id080 bundles created
  2026-05-27).
- **Training**: `flu_ha_na_kmer_aa_k3` bundle. MLP 3 seeds (42, 43,
  44); LGBM and 1-NN cosine margin single-seed each (baselines don't
  take a master_seed override, so the seed dimension would be
  redundant — `stage4_sweep.sh`'s `FIRST_SEED` guard runs them only on
  seed=42).
- **Stage 4 sweep**: `scripts/stage4_sweep.sh` with
  `--baselines "lgbm"` for parallel sweep, then knn1_margin run
  sequentially after to avoid 5×16-thread CPU thrashing (the same
  thrashing that caused the k5 1-NN sweep to stall on 2026-05-27).
- **Per-threshold D3 verification**: all thirteen thresholds achieve
  EXACT 80.00 / 10.00 / 10.00 partition (max_target_deviation_pp =
  0.0). Confirms holdout under the modern D3 check has no practical
  lower bound on this corpus — even id080 with 35.4% max-atom-frac
  packs cleanly thanks to the long tail of small clusters.

## Results

### Per-threshold metrics

_See `perf_vs_idxx.csv` for the long-form table (F1, AUC-ROC, MCC
per model); the markdown rendering below is the F1 headline view._

| idXX | MLP F1 (mean ± std) | LGBM F1 | 1-NN F1 | MLP − 1-NN | source |
|---:|---:|---:|---:|---:|:---|
| 100 | 0.9621 ± 0.0013 | 0.9500 | 0.9580 | +0.004 | May 24 |
| 099 | 0.9473 ± 0.0019 | 0.9301 | 0.9395 | +0.008 | May 24 |
| 098 | 0.9363 ± 0.0009 | 0.9171 | 0.9306 | +0.006 | May 24 |
| 097 | 0.9363 ± 0.0013 | 0.9148 | 0.9301 | +0.006 | May 24 |
| 096 | 0.9199 ± 0.0032 | 0.8984 | 0.9204 | −0.000 | May 24 |
| 095 | 0.9181 ± 0.0018 | 0.8907 | 0.9105 | +0.008 | May 24 |
| 094 | 0.7607 ± 0.0044 | 0.7528 | 0.7708 | −0.010 | May 27/28 |
| 093 | 0.7372 ± 0.0351 | 0.7388 | 0.7919 | −0.055 | May 27/28 |
| 092 | 0.7978 ± 0.0014 | 0.7369 | 0.8287 | −0.031 | May 27/28 |
| 091 | 0.7349 ± 0.0127 | 0.6798 | 0.7218 | +0.013 | May 27/28 |
| 090 | 0.7009 ± 0.0066 | 0.6499 | 0.6637 | +0.037 | May 27/28 |
| 085 | 0.6498 ± 0.0193 | 0.4835 | 0.6728 | −0.023 | May 27/28 |
| 080 | 0.4839 ± 0.0342 | 0.3556 | 0.4321 | +0.052 | May 27/28 |

AUC-ROC and MCC track F1 closely (see the plot and the CSV); the
per-metric numbers don't change the qualitative ordering at any
threshold.

### MLP does not consistently beat 1-NN in the extended range

| Range | MLP − 1-NN F1 | What it says |
|---|---|---|
| id100..id095 | +0.000 to +0.008 (mean +0.005) | MLP slightly ahead but within seed noise |
| id094..id092 | −0.010 to −0.055 (mean −0.032) | **1-NN ahead** — partition newly hard, MLP behind memorization |
| id091..id090 | +0.013 to +0.037 (mean +0.025) | MLP recovers small lead |
| id085 | −0.023 | 1-NN ahead again |
| id080 | +0.052 | MLP ahead, but absolute F1 ≈ 0.48 ≈ chance |

There is no monotone widening of the gap with decreasing idXX. The
MLP doesn't pick up biological signal that 1-NN cosine margin misses
on this corpus — across the regime where cluster_disjoint actually
displaces trivial memorization (id094..id085), MLP tracks 1-NN
within ~0.05 F1 and loses head-to-head at id094, id093, id092, and
id085. The "model learned biology beyond memorization" criterion
from `leakage.md` is not met here under aa k=3 + Test 3.

This is a real negative result for the MLP > 1-NN hypothesis on
HA-NA HA-only single-slot. Possible reasons to investigate next:
the k=3 feature is too low-resolution for the MLP to outperform
nearest-neighbor; the MLP architecture (default `[512, 256]` hidden)
is under-capacity for the harder partitions; or HA-NA biological
coupling at low idXX is itself near-memorization-shaped, so no
classifier built on this feature can beat 1-NN by much.

### Cliff at id095 → id094 — structural, not model-specific

| Model | F1 at id095 | F1 at id094 | Δ |
|---|---:|---:|---:|
| MLP | 0.9181 | 0.7607 | −0.157 |
| LGBM | 0.8907 | 0.7528 | −0.138 |
| 1-NN | 0.9105 | 0.7708 | −0.140 |

All three models drop 0.14–0.16 pp together. The HA cluster count
drops 7,578 → 2,368 between id095 → id094 and the bipartite
coupling on the NA side shifts; max_atom_frac stays near 13–14% so
the cliff is not driven by a single dominant cluster. This is a
shared structural cliff in the routing — three independent
estimators see it identically.

### Non-monotone dip-and-recover at id093 / id092

| Model | id094 | id093 | id092 |
|---|---:|---:|---:|
| MLP | 0.7607 ± 0.0044 | 0.7372 ± 0.0351 | 0.7978 ± 0.0014 |
| LGBM | 0.7528 | 0.7388 | 0.7369 |
| 1-NN | 0.7708 | 0.7919 | 0.8287 |

MLP and 1-NN both dip at id093 and recover (or surpass) at id092;
LGBM dips and stays flat. MLP std at id093 (0.0351) is 8× the std
at id094 — the id093 partition is hard enough that initialization
meaningfully affects MLP convergence, while 1-NN (deterministic)
just reads the partition and reports a higher F1. Same shape as the
2026-05-24 id097 plateau.

### Seed variance grows at low idXX (non-monotone)

| idXX | MLP F1 std (3 seeds) |
|---:|---:|
| 100 | 0.0013 |
| 095 | 0.0018 |
| 094 | 0.0044 |
| 093 | 0.0351 |
| 091 | 0.0127 |
| 090 | 0.0066 |
| 085 | 0.0193 |
| 080 | 0.0342 |

At high idXX (≥095) std ≤ 0.005 (matching the 2026-05-24 result).
At low idXX seed variance is ~10–25× larger, with the maximum at
id093 — the partition is hard enough that model initialization
meaningfully affects convergence. The non-monotone std (id092
falls back to 0.0014) tracks the partition-geometry story rather
than a smooth degradation with idXX.

## Plot

`results/flu/July_2025/runs/HAonly_extended_idXX/perf_vs_idxx.png`

Three panels (F1, AUC-ROC, MCC). Three lines per panel (MLP with
±1σ error bars, LGBM single-seed, 1-NN single-seed). Grey dotted
vertical at id095 marks the prior sweep boundary. x-axis inverted so
id100 is on the left and id080 on the right.

## D3 holdout feasibility — a finding

The 2026-05-24 work cited id094 as the lower bound for HA-only
holdout based on a pre-flight heuristic; D3 disagrees. All thirteen
thresholds (down to id080) achieve EXACT 80.00/10.00/10.00 partition
— single-slot has no practical lower bound on this corpus, because
the long tail of small clusters fills val/test cleanly even when the
top atoms are large. The bilateral routing still hits a hard
structural ceiling at id099 (bipartite mega-CC absorbs > 85% of
pairs); only single-slot benefits from the long-tail headroom.

## Reproduce

```bash
# 1. Build new datasets (or reuse if already in data/datasets/...)
for ID in 094 093 092 091 090 085 080; do
    scripts/_with_segmatch.sh scripts/stage3_dataset.sh \
        flu_ha_na_cluster_aa_id${ID}_HAonly
done

# 2. Stage 4 sweep (MLP + LGBM in parallel across thresholds; 3 seeds)
scripts/_with_segmatch.sh scripts/stage4_sweep.sh \
    --bundle flu_ha_na_kmer_aa_k3 \
    --thresholds "094 093 092 091 090 085 080" \
    --dataset_pattern "dataset_flu_ha_na_cluster_aa_id{thr}_HAonly_2026052*" \
    --seeds "42 43 44" \
    --baselines "lgbm" \
    --start_gpu 0 \
    --output_prefix training_flu_ha_na_kmer_aa_k3_HAonly_EXT

# 3. knn1_margin sequentially (one threshold at a time, n_jobs=16)
for ID in 094 093 092 091 090 085 080; do
    DS=$(ls -dt data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_aa_id${ID}_HAonly_2026052*/ | head -1 | sed 's:/$::')
    bash scripts/stage4_baselines.sh flu_ha_na_kmer_aa_k3 \
        --baseline knn1_margin --dataset_dir "$DS" \
        --output_dir "models/flu/July_2025/runs/baseline_knn1_margin_flu_ha_na_kmer_aa_k3_HAonly_EXT_id${ID}_<TS>"
done

# 4. Aggregate + plot
scripts/_with_segmatch.sh python /tmp/aggregate_haonly_ext.py
# Writes:
#   results/flu/July_2025/runs/HAonly_extended_idXX/perf_vs_idxx.csv
#   results/flu/July_2025/runs/HAonly_extended_idXX/perf_vs_idxx.png
```

## Pointers

- **Reference sweep (id100..id095)**:
  `docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md`
- **Routing helper**:
  `src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df`
- **D3 check**:
  `src/datasets/_split_helpers.py::_compute_d3_check`
  + `docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md` D3/D4.
- **Feasibility pre-flight**:
  `src/analysis/single_slot_cluster_disjoint_feasibility.py`;
  CSV at
  `results/flu/July_2025/runs/cluster_disjoint_feasibility/single_slot_feasibility_ha_na_aa.csv`.
- **Aggregate-and-plot**: `/tmp/aggregate_haonly_ext.py` (one-shot;
  consider promoting to `src/analysis/` if used again).
