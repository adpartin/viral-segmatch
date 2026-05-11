# Pipeline overview — data, models, evaluation

> A multi-audience synthesis of the project's data construction, model
> pipeline, and evaluation. For deep dives, see the linked methods docs
> and plans (each section ends with pointers).
>
> Audience cues:
> - **Biology:** sections 1, 2, 4 (the "what this task represents")
> - **Bioinformatics:** sections 3, 5, 6 (the "what flows where")
> - **Data science / ML:** sections 6, 7, 8, 9 (splits, features, models, evaluation)

---

## 1. The task

Binary classification of viral protein **pairs**: given two protein
segments, predict whether they co-occur in the same viral isolate.

- **Positive** pair: both proteins come from the same isolate
  (`assembly_id_a == assembly_id_b`).
- **Negative** pair: the two proteins come from different isolates
  (`assembly_id_a != assembly_id_b`).

Each experiment fixes a **schema pair** (`func_left`, `func_right`) —
e.g., `("Hemagglutinin precursor", "Neuraminidase protein")` for
HA/NA, or `("PB2 subunit", "PB1 subunit")`. Slot A is always
`func_left`; slot B is always `func_right`. This removes "input
direction" as a free variable for directed features like `unit_diff`.

Primary virus is **Influenza A**. The 8-segment Flu A genome lets us
ask 28 = C(8,2) such pair questions; the project today is built around
HA/NA and PB2/PB1 plus a 28-pair sweep.

---

## 2. Why this question is biologically interesting

Reassortment — the swapping of segments between co-infecting Flu A
strains — is the proximate cause of pandemic emergence (1918, 1957,
1968, 2009). Surveillance pipelines need to know which segment
combinations are plausible (have been seen together) and which are
novel reassortants. A model that scores arbitrary segment pairs for
"do these belong together?" supports:

- **Data remediation**: flagging mis-joined or contaminated genome
  records in BV-BRC and similar repositories.
- **Wastewater surveillance**: assembling consensus segments from
  fragmented metagenomic data and asking whether the inferred
  segment-pair memberships are coherent.
- **Reassortant detection**: flagging genuinely-novel cross-strain
  combinations as they emerge.

The model is intentionally trained to discriminate at the **isolate**
level (same vs different isolate), not at the population level (same
vs different host/subtype/year). The line between "biology learning"
and "metadata memorization" is the central evaluation question — see
section 9.

---

## 3. Pipeline stages

Four stages, two run-once-per-dataset (1, 2) and two per-experiment
(3, 4):

| Stage | Script | Output | Cost |
|---|---|---|---|
| 1. Preprocess | `src/preprocess/preprocess_flu.py` | `data/processed/flu/{ver}/protein_final.csv` + `genome_final.csv` (one row per protein / contig) | Run once per data version |
| 2. Embeddings | `src/embeddings/compute_esm2_embeddings.py` (+ k-mer Stage 2b) | `master_esm2_embeddings.h5` (1280-dim per protein) and/or `kmer_features_k6_*` | Run once per data version |
| 3. Dataset (pair construction) | `src/datasets/dataset_segment_pairs.py` | `data/datasets/flu/{ver}/runs/dataset_<bundle>_<TS>/{train,val,test}_pairs.{csv,parquet}` plus stats / manifests | Per experiment (~minutes on full Flu A) |
| 4. Train | `src/models/train_pair_classifier.py` (MLP) and `src/models/train_pair_baselines.py` (sklearn baselines) | `models/flu/{ver}/runs/training_<bundle>_<TS>/{best_model.pt,test_predicted.csv,post_hoc/}` | Per experiment, per model |

Configuration is **Hydra bundle-per-experiment**: one YAML under
`conf/bundles/<bundle>.yaml` fully specifies an experiment. Bundles
inherit from a chain of base configs (see
`conf/bundles/README.md`).

**See also**: `docs/methods/kmer_features.md` for stage 2b (k-mer
counting); `docs/SEED_SYSTEM.md` for reproducibility seeding.

---

## 4. Filtering and isolate sampling

Stage 3 narrows the per-isolate dataframe in this fixed order before
pair construction (see `dataset_segment_pairs.py:1503-1599`):

1. **Metadata enrichment** — joins host, year, hn_subtype, geo_location,
   passage from the parsed metadata.
2. **Metadata filtering** — `dataset.host/year/hn_subtype/...` keep only
   matching isolates.
3. **Subtype balancing** (optional) — downsamples to equal per-subtype
   representation. Mutually exclusive with `hn_subtype` filter.
4. **`max_isolates_to_process`** — optional random isolate cap.
5. **Function selection** — keep only `virus.selected_functions`.
6. **v2-only**: narrow to `schema_pair` functions (reduces co-occurrence
   set and per-sequence index sizes).

Each step shrinks the pool the pair builder sees. The pair counts
downstream are sensitive to all of these — comparing two bundles is
only meaningful when steps 1-6 produce comparable isolate pools.

---

## 5. Pair construction (Stage 3 in detail)

### 5.1 Positive pairs

For each isolate with both `func_left` and `func_right`, take the
cross-product of slot-A rows × slot-B rows. In Flu A this is usually
1 × 1 = 1 pair per isolate (one HA, one NA).

After construction we **deduplicate on `pair_key`** (canonical
`(seq_hash_a, seq_hash_b)`). Two isolates that carry sequence-identical
HA + NA proteins collapse to one positive pair_key — keeping both
would let the model see the same example twice and inflate metrics.

The dedup rate depends on protein conservation:

| Schema pair | Dedup rate | Why |
|---|---|---|
| HA / NA | ~46% dropped | Surface antigens under positive selection → diverse → many distinct pair_keys |
| PB2 / PB1 | ~51% dropped | Internal RdRp subunits under purifying selection → conserved → many isolates share the same protein-level pair |

So a fixed isolate pool produces fewer positive pairs for conserved
schema pairs than for diverse ones. Numerical example from the
metadata-aware regime bundles (108,530 starting isolates):

| Schema pair | unique positive pair_keys |
|---|---|
| HA / NA | 58,826 |
| PB2 / PB1 | 53,078 |

### 5.2 Negative pairs

Sampled across isolates with three guarantees:

1. **Co-occurrence blocking.** A candidate `(seq_hash_a, seq_hash_b)`
   is rejected if it appears as a positive in *any* isolate (not just
   the current split). Without this, "negatives" might actually be
   true positives that landed in a different isolate — a contradictory
   label.
2. **Coverage** (mode #2 leakage protection): every protein
   sequence (and DNA variant) that appears in positives must appear in
   ≥1 negative pair. Without this, a sequence could end up
   "positive-only" and the model would learn to memorize "I've seen
   this protein → predict 1" — sequence-level label imbalance.
3. **Regime mix** (opt-in): per-bundle target distribution over 9
   mutually-exclusive metadata-match regimes (`none_match`,
   `host_only`, ..., `host_subtype_year`, `unknown_metadata_neg`). See
   §5.3.

The sampler runs in two phases per split:
- **Coverage phase** (regime-blind): walks every (slot, dna_hash) target
  and finds at least one valid partner. Dominates for tight bundles.
- **Fill phase** (regime-aware when `negative_sampling.regime_targets`
  is set): tops up to `num_negatives = round(neg_to_pos_ratio × |pos|)`
  while biasing toward the configured per-regime mix.

Coverage is a **hard correctness invariant** (raises if violated at
the protein level). Regime mix is a **soft target** (logged in the
manifest with shortfalls). When `neg_to_pos_ratio` is small enough
that coverage produces all negatives, the fill phase is a no-op and
the regime mix is whatever coverage yields naturally. Setting
`neg_to_pos_ratio: 2.0` (rather than 1.0) gives the regime-aware fill
phase room to work.

### 5.3 The 9 metadata-match regimes

For each candidate negative `(isolate_i, isolate_j)`, classify by
whether the host, hn_subtype, and year_bin axes match between the two
sides:

| Regime | Match condition | Hardness |
|---|---|---|
| `none_match` | host ≠ AND subtype ≠ AND year ≠ | easiest — every shortcut available |
| `host_only` / `subtype_only` / `year_only` | exactly one axis matches | one shortcut |
| `host_subtype_only` / `host_year_only` / `subtype_year_only` | exactly two axes match | two shortcuts |
| `host_subtype_year` | all three match | hardest — model must use sequence content |
| `unknown_metadata_neg` | ≥1 axis is null on either side | catch-all |

Why this taxonomy? Earlier work (`docs/results/2026-05-07_metadata_shortcut_negatives.md`)
showed FP rate climbs **30–50× from `none_match` to `host_subtype_year`**
on legacy random-sampled datasets. The model uses metadata coincidence
as a shortcut. The regime-aware sampler addresses this at construction
time by guaranteeing a minimum population of hard negatives in every
split (default mix has 30% `host_subtype_year`).

**See also**: `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md`
(full design); `docs/methods/leakage_definitions.md` (the 5-mode
taxonomy this fits into).

---

## 6. Train / val / test split

### How the split is formed

- **Isolate-disjoint by construction** (`hard_partition_isolates: true`,
  hard-coded in v2). Train and val/test never share an isolate.
- Default ratio: 80 / 10 / 10 (`dataset.train_ratio` / `val_ratio`).
- Isolate-level partition first, then per-split positive + negative
  generation. The same `pair_key` cannot appear in two splits
  (cross-split overlap is detected and would raise).

### What the regime sampler does to per-split balance

The same `regime_targets` dict applies to all three splits — there's
no per-split override. Practical effects:

- **`neg_to_pos_ratio: 2.0`** → train/val/test each aim for `2 × |pos|`
  negatives. Actual achieved ratio drifts a bit because the coverage
  floor can exceed the requested count on tight splits.
- **Cross-split feasibility shrinks supply.** Negatives generated for
  train are added to a `forbidden_pair_keys` set that val and test
  must avoid. The closed-form `available_count` per regime is *not*
  recomputed against this set, so the val/test manifest can show
  `shortfall_reason: supply_exhausted` even when the train manifest is
  clean. The hardest regime (`host_subtype_year`) typically falls
  short by ~50% in val/test on full Flu A — the per-cell isolate count
  in the dominant Human-H3N2-2024 cell isn't infinite.

### Resulting label balance

For a typical regime-aware run on full Flu A (HA/NA, neg_to_pos=2):

```
train: 141,180 pairs  (pos 47,060, neg 94,120, ratio 2.0)
val  :  17,649 pairs  (pos  5,883, neg 11,766, ratio 2.0)
test :  17,649 pairs  (pos  5,883, neg 11,766, ratio 2.0)
```

Negative composition by regime is in `negative_regime_manifest.csv`
(per split). For metric reporting we don't reweight by regime; the
analyzer instead reports per-regime TPR / TNR explicitly (§8).

**See also**: `docs/methods/leakage_definitions.md` (modes #1 and #2);
`split_overlap_stats.csv` in every Stage 3 output (per-split unique
seq_hash / dna_hash counts and cross-split overlap).

---

## 7. Features

Two `feature_source` options:

### 7.1 ESM-2 embeddings (`feature_source: esm2`)

- 1280-dim per protein, mean-pooled over residues.
- Frozen — never fine-tuned.
- Cached once per data version under `data/embeddings/flu/{ver}/master_esm2_embeddings.h5`.
- Has a documented **protein-type subspace offset** (HA and NA
  embeddings live in slightly different subspaces) — cancelled by the
  MLP's per-slot LayerNorm.

### 7.2 K-mers (`feature_source: kmer`, currently active)

- k = 6 nucleotide k-mers, **4096-dim per slot**, **raw integer
  counts**, no normalization at materialization time.
- Sparse storage on disk (CSR), densified per-batch at training time.

**Why no normalization?** Three intersecting reasons:

1. K-mer counts depend on segment length (longer segment → more
   k-mers), which encodes a real biological signal but also a
   length-confound.
2. The MLP path applies per-slot **LayerNorm** before concatenation
   (`slot_transform: slot_norm`, learnable γ/β). This per-row
   standardization removes the gross magnitude differences without
   destroying the directional information.
3. The k-NN baselines use **cosine** distance, which already
   normalizes vector lengths internally inside the distance formula.
   Per-feature `StandardScaler` would actively distort the count
   geometry (would make non-negative count vectors signed; would
   amplify rare k-mers like an implicit TF-IDF).

For LR specifically (linear in features) we **do** apply
`StandardScaler` because L2 regularization is per-feature and
sensitive to scale.

**See also**: `docs/methods/kmer_features.md` (full pipeline,
"Why not normalize counts?" section); `docs/methods/feature_normalization.md`
(model × feature_source defaults matrix); `docs/methods/leakage_definitions.md`
(how cluster leakage interacts with k-mer near-neighbors).

---

## 8. Models

| Model | Role |
|---|---|
| **MLP** | The production classifier. Per-slot LayerNorm + concat → 3-layer MLP → BCE loss. |
| **Logistic regression** | Linear baseline. Tells us how much of the signal is captured by an affine combination of features. |
| **LightGBM** | Tree-based baseline. Captures non-linear interactions without representation learning. |
| **1-NN cosine-margin (`knn1_margin`)** | The *leakage anchor*. Score = cosine to nearest train positive minus cosine to nearest train negative. If MLP ≈ 1-NN on AUC, the MLP is doing soft near-neighbor lookup, not generalization. |
| **k-NN vote (`knn_vote`)** | Smoothed neighborhood baseline. k=5 by default, distance-weighted. |

All five models are trained on the same Stage 3 dataset and produce
the same `test_predicted.csv` schema. Each writes its own run dir
under `models/flu/{ver}/runs/`.

**See also**: `docs/methods/feature_normalization.md` (preprocessing
defaults per model); `docs/methods/leakage_definitions.md` ("biology
learning" criterion: MLP needs to beat 1-NN by ≥0.02 AUC on
sequence-disjoint splits to claim biology generalization).

---

## 9. Evaluation: per-regime heatmap

The headline diagnostic is `aggregate_baselines_vs_mlp.py`, which
produces a **heatmap with one row per model and one column per
regime**:

- **Columns**: `positive` (TPR) + 8 negative regimes (TNR each, in
  ascending hardness from `none_match` to `host_subtype_year`) + optional `unknown_metadata_neg`.
- **Cells**: TPR for the positive column, TNR for the negatives. NaN
  cells (zero-sample regimes, single-class strata) render as grey
  `N/A`.
- **Rows**: typically 5 — the 4 baselines + the MLP. Order = top-to-bottom
  in the heatmap.

### How to read it

The headline column is **`host_subtype_year` TNR** — the hardest
regime where every metadata axis matches between the two sides. The
model has no shortcut here; whatever discrimination it shows must come
from sequence content.

- A high `host_subtype_year` TNR for **1-NN** (e.g., 0.84 on full
  HA/NA) means the test set is densely connected to train in feature
  space — even nearest-neighbor lookup gets it right most of the time.
  This is the **leakage signal**.
- For the MLP to claim "biology learning," it should beat 1-NN's
  `host_subtype_year` TNR by ≥0.02. If MLP ≈ 1-NN, the MLP is doing
  soft memorization with extra steps.
- Comparing across regimes: a model that scores ~1.0 on `none_match`
  but drops sharply on `host_subtype_year` is using metadata
  coincidence as a shortcut.

### The compact aggregated view

A second plot (`level1_neg_regimes_agg.png`, written per model by the
post-hoc analyzer) collapses the 8 metadata-defined regimes into 4
match-count buckets (0 / 1 / 2 / 3 axes matching) plus the unknown
catch-all. Same information, coarser axis — useful when comparing
across many bundles.

### Caveats baked into the caption

The MLP applies *learned* per-slot LayerNorm. The baselines apply each
model's *natural* preprocessing — `StandardScaler` for LR, no scaling
for LightGBM and k-NN. The heatmap caption flags this so reviewers
don't read row-to-row score differences as model-quality differences
when they could partly reflect featurization differences.

**See also**: `docs/methods/leakage_definitions.md` ("biology
learning" criterion); `docs/post_hoc_analysis_design.md` (Level 1 / 2
methodology); `src/analysis/aggregate_baselines_vs_mlp.py` (the
script).

---

## 10. Leakage controls — quick map

The project tracks 5 canonical leakage modes (full taxonomy in
`docs/methods/leakage_definitions.md`):

| # | Mode | Status in this pipeline |
|---|---|---|
| 1 | Same-pair leakage (`pair_key` overlap across splits) | ✅ Mitigated — v2 cross-split assertion raises if any overlap. |
| 2 | Sequence-level label imbalance (a sequence appears only as positive or only as negative) | ✅ Mitigated — v2 coverage phase per slot per `dna_hash`. |
| 3 | Sequence-level leakage (same `seq_hash` / `dna_hash` in different splits) | ⚠️ Confirmed present (~25% seq_hash overlap on HA/NA mixed). Mitigation in plan: `seq_disjoint` / `strict_dedup` split modes. |
| 4 | Cluster leakage (test pair feature-vector cosine-near a train pair) | ⚠️ Suggested but not formally measured. Plan Exp 2 (1-NN baseline) and Exp 5 (mmseqs2 cluster splits) will give a verdict. |
| 5 | Demographic shortcut (model uses metadata coincidence as proxy for "same isolate") | ⚠️ Confirmed present in legacy datasets. Construction-time mitigation in v2: `negative_sampling.regime_targets`. Re-test pending on regime-aware-built datasets via the heatmap. |

Note: full mitigation of mode #5 cannot be achieved by sampling alone
(the model can still pick up the pattern in training); the heatmap is
the test that says whether the mitigation worked.

**See also**: `docs/methods/leakage_definitions.md` for definitions
and the assessment plan.

---

## 11. Cross-references

For each topic, the deepest reference:

| Topic | File |
|---|---|
| K-mer feature pipeline | `docs/methods/kmer_features.md` |
| Leakage taxonomy + biology-learning criterion | `docs/methods/leakage_definitions.md` |
| Per-(model × feature) preprocessing matrix | `docs/methods/feature_normalization.md` |
| Stage 3 builder design | `docs/plans/done/design_dataset_gen_v2.md` |
| Metadata-aware regime sampler design | `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md` |
| Post-hoc analysis methodology | `docs/post_hoc_analysis_design.md` |
| Original metadata-shortcut finding | `docs/results/2026-05-07_metadata_shortcut_negatives.md` |
| Project memory / current state | `.claude/memory.md`, `CLAUDE.md` |
