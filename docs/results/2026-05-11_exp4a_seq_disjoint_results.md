# Exp 4a results — seq_disjoint vs random split (HA/NA)

**Date:** 2026-05-11
**Branch:** feature/sequence-disjoint-splits
**Plan:** `docs/plans/done/2026-05-10_seq_disjoint_routing_plan.md`
**Parent plan:** `docs/plans/2026-05-07_leakage_diagnostics_plan.md` (Exp 4)
**Leakage mode addressed:** #3 sequence-level leakage (see `docs/methods/leakage_definitions.md`)

---

## Setup

Two builds of the HA/NA regime-aware dataset, identical except for the
positive-pair routing into train/val/test:

| Build | Bundle | Dataset dir | Routing |
|---|---|---|---|
| Random | `flu_ha_na_neg_regimes` | `dataset_flu_ha_na_neg_regimes_20260510_120116` | row-level shuffle on `pos_df` |
| seq_disjoint | `flu_ha_na_seq_disjoint` | `dataset_flu_ha_na_seq_disjoint_20260511_000517` | bipartite-CC LPT-greedy on (HA-DNA, NA-DNA) graph |

Same Stage-3 negative sampler (regime targets unchanged), same k-mer
features (k=6), same MLP config (hidden_dims=[512,256,64],
slot_norm + concat, 100 epochs), same baselines (lgbm, knn1_margin,
knn_vote).

`flu_ha_na_seq_disjoint` was built and verified by the routing audit
(`seq_disjoint_audit.json`): 46,954 components on 58,826 positives,
largest 178 pairs (0.30%), zero pair drops, max target deviation 0.001%,
all 12 cross-split dna_hash overlap counts = 0 (before AND after
negatives were added).

The logistic baseline was NOT re-run under seq_disjoint (it collapsed
to near-zero TPR under random split — re-running on this dataset would
not change the leakage story). The four-model row order in the
heatmaps is MLP → lgbm → knn1_margin → knn_vote.

---

## Headline numbers (per-regime TPR / TNR)

Heatmaps:
- `docs/results/baselines_vs_mlp_flu_ha_na_neg_regimes_20260511_071453/baselines_vs_mlp_heatmap.png`
- `docs/results/baselines_vs_mlp_flu_ha_na_seq_disjoint_20260511_071425/baselines_vs_mlp_heatmap.png`

Selected columns side-by-side (positive = TPR; negatives = TNR):

| Model | regime | random | seq_disjoint | Δ |
|---|---|---|---|---|
| MLP | positive | 0.9772 | 0.9701 | -0.0071 |
| MLP | none_match | 0.9890 | 0.9880 | -0.0010 |
| MLP | host_subtype_only | 0.9873 | 0.9745 | -0.0128 |
| MLP | **host_subtype_year** | **0.8723** | **0.8341** | **-0.0382** |
| MLP | unknown_metadata_neg | 1.0000 | 0.9872 | -0.0128 |
| lgbm | positive | 0.9738 | 0.9699 | -0.0039 |
| lgbm | host_subtype_year | 0.7730 | 0.7812 | +0.0082 |
| knn1_margin | positive | 0.9769 | 0.9733 | -0.0036 |
| knn1_margin | host_subtype_year | 0.8444 | 0.8481 | +0.0037 |
| knn_vote | positive | 0.9806 | 0.9810 | +0.0004 |
| knn_vote | host_subtype_year | 0.7936 | 0.7972 | +0.0036 |

The remaining regimes (`host_only`, `subtype_only`, `year_only`,
`host_year_only`, `subtype_year_only`) all show MLP changes
within ±0.013. Full tables in
`docs/results/baselines_vs_mlp_*_*.csv`.

---

## Three findings

### 1. MLP loses 3.8pp on the hardest regime, baselines don't move.

`host_subtype_year` is the negative regime where the random isolate
shares **all three** metadata axes (host, subtype, year-bin) with the
positive's anchor isolate. It is the closest in-distribution negative
the sampler can produce and the only regime where a model is forced to
use sequence content rather than a metadata shortcut.

The MLP's drop of 0.038 on this column, with positive TPR essentially
unchanged (-0.007), localizes the leakage signal cleanly: previous
performance on the hardest regime depended on sequences that ALSO
appeared somewhere in train. Once the same `dna_hash` cannot be in two
splits, the MLP can no longer rely on that near-neighbor memory.

The baselines (lgbm, knn1_margin, knn_vote) move by < 0.01 in either
direction. This is consistent with their design: tree- and
neighborhood-based predictors are insensitive to "did I see this exact
DNA?" — they already operate on the local cosine landscape, which is
not changed by routing a specific DNA into one split vs another.

### 2. MLP falls **below** 1-NN on host_subtype_year under seq_disjoint.

`docs/methods/leakage_definitions.md` operationalizes "model learned
biology" as MLP beating 1-NN by ≥ 0.02. On host_subtype_year:

| Split | MLP | knn1_margin (1-NN) | gap |
|---|---|---|---|
| random | 0.8723 | 0.8444 | **+0.0279** (above bar) |
| seq_disjoint | 0.8341 | 0.8481 | **−0.0140** (MLP loses to 1-NN) |

Under random split the MLP cleared the biology-learning bar on this
regime by a small margin. Under seq_disjoint it does not: the MLP's
host_subtype_year accuracy is **lower** than a cosine 1-NN classifier
on the same features. The "biology generalization" story on the
hardest-negative regime was load-bearing on cross-split DNA reuse.

This is the headline scientific finding of Exp 4a. It does not say the
MLP learns nothing (it still wins on most regimes); it says the MLP's
edge on the hardest regime, where biology generalization matters most,
was an artifact of memorization.

### 3. Easy regimes are robust.

`none_match`, `host_only`, `subtype_only`, `year_only` show MLP changes
under 0.012 across the board. The model continues to nail
metadata-mismatched negatives (the "easy" regimes) under seq_disjoint.
The penalty is concentrated in the regimes where the answer requires
sequence-level discrimination — exactly the regimes where leakage was
suspected.

---

## What this does and does not establish

**Establishes:**
- Mode #3 (sequence-level leakage) was inflating the MLP's headline
  metric on the regime that matters most for the biology claim.
- The construction-time mitigation works as designed: cross-split DNA
  overlap is zero, with zero pair drops, on this dataset.
- Baselines under seq_disjoint provide a fair "soft 1-NN" reference;
  the MLP currently does not beat them on host_subtype_year.

**Does not establish:**
- Whether the remaining ~0.834 host_subtype_year TNR is genuine biology
  signal or residual *cluster*-level leakage (mode #4, near-neighbor
  DNAs that escape exact-`dna_hash` dedup). Exp 3 (cosine deciles) and
  Exp 5 (mmseqs2 cluster splits) address that.
- Whether a different MLP configuration (more capacity, different
  interaction, longer training, ESM-2 instead of k-mer) recovers a
  gap above 1-NN under seq_disjoint. Worth one or two follow-up runs.

---

## Reproducibility

```
# Stage 3 build (writes seq_disjoint_audit.json and asserts zero overlap):
bash scripts/stage3_dataset.sh flu_ha_na_seq_disjoint

# Stage 4 — MLP + baselines (the user's actual command stream):
bash scripts/stage4_train.sh flu_ha_na_seq_disjoint --dataset_dir <DS>
python src/models/train_pair_baselines.py --config_bundle flu_ha_na_seq_disjoint --dataset_dir <DS>

# Aggregator (uses autodiscovery — picks the latest training_/baseline_
# runs for this bundle and refuses to mix dataset_dirs):
python src/analysis/aggregate_baselines_vs_mlp.py \
    --bundle flu_ha_na_seq_disjoint \
    --output_dir docs/results/baselines_vs_mlp_flu_ha_na_seq_disjoint_<TS>
```

The companion `flu_ha_na_neg_regimes` (random-split) heatmap above was
generated by the same autodiscovery command with `--bundle
flu_ha_na_neg_regimes`.
