# Temporal Holdout Split — Task 3

**Status: IMPLEMENTED** (initial runs complete, pair_key dedup issue pending)

## Goal

Train on isolates from 2021-2023, test on 2024. Assesses whether the model generalizes
to future flu seasons — a strong publication story ("does the model predict the next
flu season?").

## Implementation

- Config fields: `dataset.year_train`, `dataset.year_test` in `conf/dataset/default.yaml`
- Split logic: `generate_temporal_split()` in `src/datasets/dataset_segment_pairs.py`
- Bundle: `conf/bundles/flu_schema_raw_slot_norm_unit_diff_temporal.yaml`
- Val isolates are drawn from the **test years** (2024), not train years. This is a
  deliberate design choice: if val were sampled from the train years (2021-2023), it
  would represent the training distribution, not the future-season distribution. Early
  stopping and threshold tuning would then be optimistically biased — the model would
  select hyperparameters that work well on the past but may not transfer to the shifted
  test distribution.
- Uses the existing `*_isolates_override` mechanism in `split_dataset()`, the same hook
  used by CV. This keeps the implementation clean and reusable for future tasks (large
  test sets, learning curves).

Mutually exclusive with `year` (single-value filter) and `n_folds` (CV mode).

## Feasibility Analysis (March 2026)

Data source: `protein_final.csv` (July_2025 version), 108,530 isolates total.
Every isolate has exactly one HA and one NA record, so isolate count = positive pair count.

### Isolates per year (HA+NA, all subtypes)

| Year | Isolates |
|------|----------|
| 2019 | 6,693 |
| 2020 | 2,813 |
| 2021 | 2,623 |
| 2022 | 11,191 |
| 2023 | 6,309 |
| **2024** | **17,053** |
| 2025 | 9,961 |

### Expected split sizes (train 2021-2023, val+test 2024)

| Split | Years | Isolates | Pos Pairs | Total Pairs (1:1 neg) |
|-------|-------|----------|-----------|----------------------|
| Train | 2021-2023 | 20,123 | ~20,123 | ~40,246 |
| Val | 2024 (50%) | ~8,527 | ~8,527 | ~17,054 |
| Test | 2024 (50%) | ~8,526 | ~8,526 | ~17,052 |

Val/test split is 50/50 of 2024 isolates (from `val_frac = val_ratio / (1 - train_ratio) = 0.5`).

**Scale comparison**: Previous experiments used `max_isolates_to_process: 5000` (the "5ks"
bundles), producing ~5K positive pairs and ~10K total pairs. This temporal experiment uses
all matching isolates (`max_isolates_to_process: null`), producing ~40K train pairs and
~34K val+test pairs — roughly **8x larger** than 5K-sample runs. Training time will scale
accordingly but should still be feasible on a single GPU (the MLP classifier is lightweight;
the bottleneck is pair construction in Stage 3, not training in Stage 4).

**Train:test ratio is ~2.4:1** (20K train vs 8.5K test), which is more balanced than the
typical 80/10/10 random split. This is inherent to the year-based partitioning — 2024
happens to be the largest single year in the dataset.

### Subtype distribution shift (train vs test)

| Subtype | Train (2021-2023) | Test (2024) |
|---------|------------------|-------------|
| H3N2 | 8,041 (40%) | 5,482 (32%) |
| H5N1 | 4,826 (24%) | 6,917 (41%) |
| H1N1 | 4,520 (22%) | 4,439 (26%) |
| Other | ~2,736 (14%) | ~215 (1%) |

Key observations:

- **H5N1 jumps from 24% to 41%** between train and test periods, reflecting the 2024
  avian flu surge. H3N2 drops from 40% to 32%. This is a realistic temporal shift that
  directly tests whether the model generalizes across changing subtype prevalence —
  exactly the kind of distribution shift we want to evaluate.
- **"Other" subtypes nearly vanish** in 2024 (14% → 1%), meaning the test set is more
  concentrated on the three major subtypes. The model will need to handle this shift
  in subtype diversity.
- This subtype shift also means that if the model has learned subtype-specific shortcuts
  (e.g., "H5N1 pairs look different from H3N2 pairs"), those shortcuts may hurt
  generalization. The temporal holdout will reveal this.

### COVID-era dip in 2020-2021

2020 and 2021 have notably fewer isolates (2,813 and 2,623 respectively) compared to
surrounding years. This likely reflects reduced flu surveillance and circulation during
the COVID-19 pandemic — fewer flu samples were collected and sequenced worldwide during
lockdowns and mask mandates. The 2021 count is the lowest in the entire 2019-2025 range.

This means the 2021-2023 training window is somewhat front-loaded: 2022 alone contributes
56% of training isolates (11,191 / 20,123). If this imbalance is a concern, the train
window could be extended to 2019-2023 (adding 9,506 isolates for 29,629 total), though
20K should be sufficient for the HA/NA binary task.

### 2024 is the largest year

2024 has 17,053 isolates — the single largest year in the dataset, likely reflecting
both resumed surveillance post-COVID and the H5N1 avian flu surge. This means we have
ample test data, which is a strength: the temporal holdout results will have narrow
confidence intervals.

## What to Watch For in Results

- **AUC drop vs random split**: If temporal AUC is substantially lower than random-split
  AUC (~0.97), the model may be exploiting population-level confounders (e.g., "same
  subtype mix" rather than "same isolate") that shift between years. This would connect
  to the existing high-FP-rate observation on filtered datasets.
- **FP rate on H5N1 pairs**: H5N1 is underrepresented in training (24%) but dominant in
  test (41%). If FP rate is disproportionately high on H5N1 test pairs, the model may
  not have seen enough H5N1 diversity during training.
- **Delayed learning**: The H3N2-only experiments showed a plateau-then-breakthrough
  pattern requiring patience=40+. With a larger, mixed-subtype dataset this may not
  occur, but worth monitoring. The temporal bundle currently inherits the default
  patience (no override).
- **Comparison to 5K-sample results**: The ~8x increase in training data should improve
  or at least match 5K-sample performance. If it doesn't, data quality or label noise
  at scale may be an issue.

## How to Run

```bash
# Stage 3: generate temporal dataset
conda run -n cepi ./scripts/stage3_dataset.sh flu_schema_raw_slot_norm_unit_diff_temporal

# Stage 4: train on temporal dataset
conda run -n cepi ./scripts/stage4_train.sh flu_schema_raw_slot_norm_unit_diff_temporal --dataset_dir <path_from_stage3>
```

## Pair-Key Overlap Issue (discovered March 2026)

### Observation

Stage 3 run (`20260306_150934`) revealed massive pair_key overlap between splits:

- 2,719 overlapping pair_keys found across splits
  - Train-Val: 226 (1.33% of val)
  - Train-Test: 229 (1.34% of test)
  - Val-Test: 2,264 (13.28% of test)
- After removal: Val lost 42.6% of pairs (17,052 → 9,786), Test lost 42.2% (17,054 → 9,855)
- 3,387 isolates became unassigned (all their pairs were removed)
- Val/test positive rate dropped to ~25% (from expected ~50%) because overlapping
  pair_keys are disproportionately positives (same strain circulating across years
  produces identical positive pair_keys)

Train remained perfectly balanced (20,123 pos / 40,246 total = 50.00%).

### Root cause

Many 2024 isolates share identical HA+NA sequences with 2021–2023 isolates (same
circulating strains). Their positive pair_keys (based on seq_hash) match train pair_keys
and get removed from val/test. Negatives survive because they're randomly sampled and
unlikely to collide, creating the ~25/75 label imbalance in val/test.

### Candidate approaches

**A. Disable pair_key dedup for temporal holdout (recommended).** The pair_key dedup was
designed for random splits where the same sequence pair in train+test is true leakage.
In temporal holdout, the goal is operational: "can the model predict co-occurrence for a
2024 isolate?" If a 2024 isolate shares sequences with a 2021 isolate, that's realistic —
not leakage. The model should handle it. Downside: slightly inflated test metrics if the
model memorizes specific pairs, though with 15K+ co-occurring pairs that's unlikely.

**B. Deduplicate isolates before splitting.** Remove 2024 isolates whose (HA, NA) sequence
pair already appeared in 2021–2023. Keeps dedup logic intact but shrinks the test set by
design. More conservative.

**C. Move overlapping 2024 isolates into training.** If a 2024 isolate has the same
pair_key as a train isolate, it doesn't add new information to val/test — absorb it into
train. Keeps val/test clean and doesn't waste data.

### Decision

TBD — selecting approach before re-running Stage 3.

## Initial Results (March 7, 2026)

Results with pair_key dedup active (25/75 imbalanced val/test). Threshold=0.5 for both.
Dataset: `dataset_flu_schema_raw_slot_norm_unit_diff_temporal_20260307_184509`

### ESM-2 (slot_norm + unit_diff)

Run: `training_flu_schema_raw_slot_norm_unit_diff_temporal_20260307_190355`

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.891 |
| F1 | 0.734 |
| Precision | 0.583 |
| Recall | 0.989 |
| FP/FN ratio | 64.1 |
| TP / FP / FN / TN | 2511 / 1795 / 28 / 5521 |

### K-mer (k=6, slot_norm + unit_diff)

Run: `training_flu_schema_raw_kmer_k6_slot_norm_unit_diff_temporal_20260307_215857`

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.941** |
| F1 | **0.832** |
| Precision | **0.729** |
| Recall | 0.969 |
| FP/FN ratio | **11.6** |
| TP / FP / FN / TN | 2460 / 913 / 79 / 6403 |

### Key observations

- **K-mers substantially outperform ESM-2 on temporal holdout** (AUC 0.941 vs 0.891).
  K-mers cut false positives in half (913 vs 1,795) while maintaining high recall.
  The AUC gap (+5 points) is wider than on random splits, suggesting k-mer frequency
  profiles generalize better across flu seasons than ESM-2 embedding geometry.
- **Both models show AUC drop vs random splits** (~0.97 → 0.89/0.94), confirming
  genuine temporal generalization difficulty from the subtype distribution shift
  (H5N1 24%→41%).
- **Both models are FP-heavy**, partly due to the pair_key dedup creating 25/75
  imbalanced test sets. Disabling dedup (approach A) would restore 50/50 balance
  and give cleaner metrics.
- **Both models overfit**: val_loss diverges from train_loss early while val_F1
  plateaus. Early stopping by val_loss instead of val_F1 may help calibration.
- **Caveat**: These results are confounded by the pair_key dedup imbalance.
  Re-running with dedup disabled is needed for publication-quality numbers.

## Verification Checklist

- [ ] Stage 3 logs show correct year-based partitioning
- [ ] Test pairs contain only 2024 isolates
- [ ] Train pairs contain only 2021-2023 isolates
- [ ] Val pairs contain only 2024 isolates (same distribution as test)
- [ ] No isolate overlap between train/val/test
- [ ] Setting both `year` and `year_train`/`year_test` raises an error
- [ ] Stage 4 trains successfully on temporal dataset
