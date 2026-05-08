# Cosine-controlled and cluster-based splits — Plan

**Status: PROPOSED** (pending approval)
**Date:** 2026-05-08

## Context

`docs/plans/2026-05-07_leakage_diagnostics_plan.md` lays out the broader
diagnostic agenda. This plan zooms in on two split-construction
experiments that directly test cluster leakage (taxonomy #4):

- **Cosine-controlled splits** — build train/val/test such that every
  test pair's joint feature vector is at least some cosine distance
  away from any training pair's feature vector. Tests near-neighbor
  leakage at the *feature* level, where the threshold is tunable.
- **Cluster-based splits (mmseqs2)** — build train/val/test such that
  no two pairs across splits share a sequence cluster at a chosen
  protein-identity threshold. Tests near-neighbor leakage at the
  *sequence* level, where the threshold is biological (% identity).

Both experiments answer the same headline question — *does test
performance hold when we forbid near-neighbors of training pairs from
appearing in test?* — but along different similarity axes. Reporting
both lets us distinguish "the model memorizes feature-vector lookups"
from "the model relies on phylogenetic neighbors regardless of feature
representation."

Pre-conditions assumed:
- Exp 1 from the parent plan (`split_overlap_stats.csv`) has landed,
  so we have visibility into seq/dna overlap on the baseline run.
- Exp 4 from the parent plan (`seq_disjoint` + `strict_dedup` split
  modes in `split_dataset_v2`) has landed, providing the routing
  scaffolding that these new modes plug into.

If Exp 4 results already settle the leakage question (e.g., AUC
crashes from 0.99 to chance under strict dedup), this plan can be
de-prioritized — its value is in *quantifying* the near-neighbor
threshold below which performance breaks, not in proving leakage
exists.

---

## Experiment A — Cosine-controlled splits

### Goal

Build alternative train/val/test such that every test pair has
`max_train_cosine ≤ τ` for some threshold τ ∈ {0.95, 0.9, 0.8, 0.7}.
Re-train under each threshold; plot AUC / FP rate vs τ.

A monotonic AUC drop as τ decreases is direct evidence the model
relies on feature-space near-neighbors. A flat AUC across τ would
suggest the model has learned a generalizable representation.

### Method

1. **Compute joint features** for every pair in the existing v2
   dataset. For k-mer feature_source, this is
   `concat(kmer_features[ctg_a], kmer_features[ctg_b])`,
   L2-normalized. Cache as a sparse matrix.
2. **Pair-similarity graph**: for each pair `i`, compute its nearest
   pair `j ≠ i` in feature space. Edge `(i, j)` exists if their
   cosine ≥ τ.
3. **Component-aware split**: connected components of the graph go to
   one split. Random shuffle of components into train/val/test
   respecting the target ratios. Leftover within-component pairs all
   land in the same split, by construction.
4. **Sanity check**: re-compute `max_train_cosine` for each test pair
   and assert all are < τ.

### Files to modify / create

| File | Change |
|---|---|
| `src/datasets/_split_helpers.py` (new) | Hosts `cosine_controlled_split(pos_df, features, tau, ratios, seed)`. Returns three index lists or boolean masks. |
| `src/datasets/dataset_segment_pairs_v2.py` | Add `split_mode: cosine_controlled` branch in `split_dataset_v2`; reads `dataset.cosine_threshold` (default 0.9). |
| `src/utils/similarity_utils.py` (new) | Shared helpers: `pair_features(df, kmer_or_esm)`, `nearest_cosine(matrix_test, matrix_train)`. Used by both Exp A and Exp B and by the parent plan's Exp 2. |
| `conf/dataset/default.yaml` | Add `split_mode` and `cosine_threshold` defaults (`assembly` / `null` to preserve existing behavior). |
| `conf/bundles/flu_ha_na_cosine_t{N}.yaml` (4 new) | Bundles for τ ∈ {0.95, 0.9, 0.8, 0.7}. Inherit from `flu_ha_na`; override `split_mode` and `cosine_threshold`. |

### Datasets generated

For each τ a new Stage-3 directory:

```
data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cosine_t095_<TS>/
data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cosine_t090_<TS>/
data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cosine_t080_<TS>/
data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cosine_t070_<TS>/
```

Each carries the standard v2 outputs (train/val/test pairs CSV,
`dataset_stats.json`, etc.) plus a new `cosine_threshold_audit.json`
documenting the achieved max-train-cosine on the resulting test set.

### Training runs

Four MLP runs at `h=[100]` (matches the current default):

```
models/flu/July_2025/runs/training_flu_ha_na_cosine_t{tau}_<TS>/
```

Plus the existing `training_flu_ha_na_*` baseline as the τ = 1.0
control.

### Outputs and success criteria

- A 5-row CSV `docs/results/cosine_controlled_sweep.csv` with columns
  `tau`, `n_pairs_train/val/test`, `auc`, `f1`, `mcc`, `match_count_3_FP_rate`.
- A line plot `cosine_controlled_sweep.png` (AUC vs τ, dual-axis with
  match_count=3 FP rate).
- **Success criterion**: monotonic AUC drop as τ decreases.
  Quantitatively, if AUC at τ=0.7 drops below 0.7, near-neighbor
  leakage was the dominant signal at the standard split.

### Effort

- Routing implementation + helpers: ~half day.
- Four Stage-3 builds + four training runs: ~hour wall-clock total.
- Analysis + plot: ~half hour.

### Risks

- **Component sizes blow up at low τ.** When most pairs are in one
  giant connected component (likely for highly conserved proteins),
  cosine-controlled splits become infeasible — train consumes
  everything, leaving an empty test. Mitigation: cap by random
  sub-sampling within components and report achieved test size; if
  test < 500 pairs we abort that τ and document.

---

## Experiment B — mmseqs2 cluster-based splits

### Goal

Build alternative train/val/test such that no two pairs across splits
share a sequence cluster at a chosen protein-identity threshold
(default 95%). Re-train and compare to baseline.

This addresses near-neighbor leakage at the *biological* level —
phylogenetically close sequences with synonymous codon variation that
escape both `dna_hash` dedup and feature-cosine controls (because they
might still be cosine-near *or* not, depending on representation).

### Method

1. Cluster all unique protein sequences across the relevant functions
   (HA + NA, or all 8 segments) using mmseqs2 at min-seq-id 0.95,
   coverage 0.8.
2. For each cluster, every member sequence inherits the same
   `cluster_id`.
3. **Cluster-aware split**: partition `cluster_id`s across
   train/val/test (e.g., 60/20/20 by count). Each pair routes to the
   split where BOTH its sequences' clusters land. Drop split-mixed
   pairs.

### Files to modify / create

| File | Change |
|---|---|
| `scripts/cluster_proteins_mmseqs.sh` (new) | Wrapper that exports protein FASTA from `protein_final.csv`, runs `mmseqs easy-cluster`, parses cluster TSV into `data/processed/flu/{version}/protein_clusters_id{N}.parquet` (columns: `seq_hash`, `cluster_id`). One-time per dataset version + threshold. |
| `src/datasets/_split_helpers.py` | Add `cluster_disjoint_split(pos_df, cluster_lookup, ratios, seed)`. |
| `src/datasets/dataset_segment_pairs_v2.py` | Add `split_mode: cluster_disjoint` branch; reads `dataset.cluster_id_path` (path to the parquet). |
| `conf/dataset/default.yaml` | Document the new `split_mode` value. |
| `conf/bundles/flu_ha_na_cluster_id{N}.yaml` | One bundle per identity threshold (95%, 90%, 80%). Inherit `flu_ha_na`; override `split_mode` and `cluster_id_path`. |

### One-time prerequisite

Install mmseqs2:

```bash
mamba install -n cepi -c bioconda mmseqs2
```

Cluster command (script encapsulates this):

```bash
mmseqs easy-cluster \
    proteins.fasta \
    out_clusters_id095 \
    tmp_mmseqs \
    --min-seq-id 0.95 -c 0.8 --cov-mode 0
```

### Datasets generated

```
data/processed/flu/July_2025/protein_clusters_id095.parquet  (one-time)
data/processed/flu/July_2025/protein_clusters_id090.parquet
data/processed/flu/July_2025/protein_clusters_id080.parquet

data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id095_<TS>/
data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id090_<TS>/
data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id080_<TS>/
```

### Training runs

Three MLP runs at `h=[100]`:

```
models/flu/July_2025/runs/training_flu_ha_na_cluster_id{N}_<TS>/
```

### Outputs and success criteria

- A 4-row CSV `docs/results/cluster_split_sweep.csv` with columns
  `id_threshold`, `n_pairs_train/val/test`, `auc`, `f1`, `mcc`,
  `match_count_3_FP_rate`. Includes the τ=1.0 baseline.
- Line plot `cluster_split_sweep.png` (AUC vs identity threshold).
- **Success criterion**: monotonic AUC drop as identity threshold
  decreases (looser clusters → train and test pulled further apart in
  sequence space). At 80% identity, AUC drop quantifies how much of
  the headline metric depended on close phylogenetic relatives.

### Effort

- mmseqs2 install + cluster script: ~hour.
- Routing implementation: ~half day (less if Exp A's helpers are
  reusable, which is the design intent).
- Three Stage-3 builds + three training runs: ~hour total.
- Analysis: ~half hour.

### Risks

- **mmseqs2 cluster file misalignment** — if the cluster TSV's
  `seq_hash`es don't all appear in `protein_final.csv` (e.g., due to
  different length/QC filters), the join silently drops rows. Hard
  assertion in the routing helper: every seq_hash in `pos_df` must
  appear in the cluster lookup; raise on missing.
- **Highly conserved functions degenerate**. M1 (4,771 unique
  proteins covering 114K isolates) might collapse into very few
  clusters at 95% identity, leaving few clusters to partition.
  Mitigation: report `n_clusters` per function; if any function has
  fewer than (say) 100 clusters, exclude it from the cluster-disjoint
  experiment and note in the writeup.

---

## Shared infrastructure

A small refactor pays off for both experiments and the parent plan's
Exp 2:

- `src/utils/similarity_utils.py` — pure functions for k-mer / ESM-2
  joint feature construction and cosine math. Unit-testable.
- `src/datasets/_split_helpers.py` — pure functions implementing the
  four new split modes (`seq_disjoint`, `strict_dedup`,
  `cosine_controlled`, `cluster_disjoint`). `split_dataset_v2`
  becomes a thin dispatcher.

This keeps the v2 builder readable as the experiments multiply.

## Execution order within this plan

1. Land `_split_helpers.py` skeleton + `seq_disjoint` and
   `strict_dedup` modes (parent plan Exp 4) — gives the dispatcher
   shape to extend.
2. Cosine-controlled split (Experiment A) — uses helpers from step 1
   and similarity utilities.
3. mmseqs2 cluster install + cluster script + cluster_disjoint mode
   (Experiment B).
4. Run all four sweeps; produce both result CSVs and plots.
5. Combine into a single `docs/results/<date>_split_robustness_sweep.md`
   that interprets all four split modes side-by-side.

## Out of scope

- ESM-2 features for these experiments. Once helpers exist, swapping
  `feature_source: esm2` is one config flag — but ESM-2's
  cosine-distance behavior may differ enough to deserve its own
  thread.
- Re-running Bunya bundles. Leave to a future plan.
- Per-fold variance / cross-validation under these split modes.

## See also

- `docs/plans/2026-05-07_leakage_diagnostics_plan.md` — parent plan;
  this is the specific design for its Exp 2 and Exp 7.
- `docs/results/2026-05-07_metadata_shortcut_negatives.md` — the
  analysis that motivated all of this.
