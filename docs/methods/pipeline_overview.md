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

Several viral-genomics tasks — from data-repository QA to
wastewater-based metagenomic assembly to retrospective surveillance —
share the same primitive: deciding whether two viral protein segments
belong to the same isolate. A model that scores arbitrary segment
pairs for "do these belong together?" supports:

- **Data remediation**: flagging mis-joined or contaminated genome
  records in BV-BRC and similar repositories.
- **Wastewater surveillance**: assembling consensus segments from
  fragmented metagenomic data and asking whether the inferred
  segment-pair memberships are coherent.
- **Reassortment forecasting**: segment pairs the model rates as
  compatible but never co-observed are candidate future
  reassortants — potential outbreak strains worth flagging for
  monitoring.

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
| 2. Embeddings | `src/embeddings/compute_esm2_embeddings.py` (+ k-mer Stage 2b: `src/embeddings/compute_kmer_features.py`) | `master_esm2_embeddings.h5` (1280-dim per protein) and/or `kmer_features_k6_*` | Run once per data version |
| 3. Dataset (pair construction) | `src/datasets/dataset_segment_pairs.py` (CLI) → dispatches to `dataset_segment_pairs_v2.py` (default since 2026-05-11) | `data/datasets/flu/{ver}/runs/dataset_<bundle>_<TS>/{train,val,test}_pairs.{csv,parquet}` plus stats / manifests | Per experiment (~minutes on full Flu A) |
| 4. Train | `src/models/train_pair_classifier.py` (MLP) and `src/models/train_pair_baselines.py` (sklearn baselines) | `models/flu/{ver}/runs/training_<bundle>_<TS>/{best_model.pt,test_predicted.csv,post_hoc/}` | Per experiment, per model |

Configuration is **Hydra bundle-per-experiment**: one YAML under
`conf/bundles/<bundle>.yaml` fully specifies an experiment. Bundles
inherit from a chain of base configs (see `conf/bundles/README.md`).

**See also**: `docs/methods/preprocess.md` (stage 1 detail);
`docs/methods/gto_format_reference.md` (input GTO schema);
`docs/methods/kmer_features.md` (stage 2b k-mer pipeline);
`docs/SEED_SYSTEM.md` (reproducibility seeding).

---

## 4. Filtering and isolate sampling

Stage 3 narrows the per-isolate dataframe in this fixed order before
pair construction:

1. **Metadata enrichment** — joins host, year, hn_subtype, geo_location,
   passage from the parsed metadata.
2. **Metadata filtering** (optional) — `dataset.host/year/hn_subtype/...`
   keep only matching isolates.
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
schema pairs than for diverse ones. Numerical example from the current
production builds (`dataset_flu_ha_na_20260512_*` /
`dataset_flu_pb2_pb1_20260512_*`, 108,530 starting isolates,
`seq_disjoint` routing with `hash_key=seq`):

| Schema pair | unique positive pair_keys (train + val + test) |
|---|---|
| HA / NA | 58,388 |
| PB2 / PB1 | 52,657 |

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
3. **Regime mix** (opt-in): per-bundle target distribution over 8
   mutually-exclusive metadata-match regimes (`none_match`, `host_only`,
   `subtype_only`, `year_only`, `host_subtype_only`, `host_year_only`,
   `subtype_year_only`, `host_subtype_year`). See §5.3. (The legacy 9th
   regime `unknown_metadata_neg` was retired 2026-05-11; null on any
   metadata axis now classifies as no-match on that axis via the
   existing 8-tuple mapping.)

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

### 5.3 The 8 metadata-match regimes

For each candidate negative `(isolate_i, isolate_j)`, classify by
whether the host, hn_subtype, and year_bin axes match between the two
sides:

| Regime | Match condition | Hardness |
|---|---|---|
| `none_match` | host ≠ AND subtype ≠ AND year ≠ | easiest — every shortcut available |
| `host_only` / `subtype_only` / `year_only` | exactly one axis matches | one shortcut |
| `host_subtype_only` / `host_year_only` / `subtype_year_only` | exactly two axes match | two shortcuts |
| `host_subtype_year` | all three match | hardest — model must use sequence content |

Why this taxonomy? Earlier work (`docs/results/2026-05-07_metadata_shortcut_negatives.md`)
showed FP rate climbs **30–50× from `none_match` to `host_subtype_year`**
on legacy random-sampled datasets. The model uses metadata coincidence
as a shortcut. The regime-aware sampler addresses this at construction
time by guaranteeing a minimum population of hard negatives in every
split (default mix has 30% `host_subtype_year`).

**See also**: `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md`
(full design); `docs/methods/leakage.md` (the 5-mode
taxonomy this fits into).

---

## 6. Train / val / test split

### How the split is formed

- **Isolate-disjoint by construction** (`hard_partition_isolates: true`
  in `conf/dataset/default.yaml`, enforced by v2). Train and val/test
  never share an isolate.
- **Routing within the isolate-disjoint partition** is controlled by
  `dataset.split_strategy.mode`. As of 2026-05-11 the production
  default for `flu_ha_na` and `flu_pb2_pb1` is **`seq_disjoint`** with
  `hash_key: seq` — positives are routed so no `seq_hash` appears in
  two splits. This is strictly stronger than isolate-disjoint and
  eliminates mode #3 sequence-level leakage by construction (see
  `docs/methods/leakage.md`).
- Default ratio: 80 / 10 / 10 (`dataset.train_ratio` / `val_ratio`).
- The same `pair_key` cannot appear in two splits (cross-split overlap
  is detected and would raise).

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

Current production HA/NA regime-aware build
(`dataset_flu_ha_na_regimes_20260512_114205`,
`neg_to_pos_ratio: 2.0`, verified from `dataset_stats.json`):

```
train: 140,130 pairs  (pos 46,710, neg 93,420, ratio 2.0)
val  :  17,517 pairs  (pos  5,839, neg 11,678, ratio 2.0)
test :  17,517 pairs  (pos  5,839, neg 11,678, ratio 2.0)
```

PB2/PB1 equivalent (`dataset_flu_pb2_pb1_regimes_20260512_114204`):
train 126,375 / val 15,798 / test 15,798 (same 2.0 ratio).

Non-regime bundles (`flu_ha_na`, `flu_pb2_pb1`) use
`neg_to_pos_ratio: 2.0` as well (set 2026-05-12); achieved neg ratios
drift slightly above 2.0 in val/test because the coverage phase
overshoots `requested_negatives` to honor the per-sequence coverage
minimum.

Negative composition by regime is in `negative_regime_manifest.csv`
(per split). For metric reporting we don't reweight by regime; the
analyzer instead reports per-regime TPR / TNR explicitly (§9).

**See also**: `docs/methods/leakage.md` (modes #1 and #2);
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

- k = 6 nucleotide k-mers, **4,096-dim per slot**, raw integer counts,
  no normalization at materialization time. Storage is sparse CSR
  (1.78 GB on disk for 868,240 segments), densified per-batch at
  training time.
- The same `compute_kmer_features.py` machinery also supports protein
  k-mers via an `alphabet` parameter (added 2026-05-12); no active
  bundle uses this path yet.
- Per-slot transform applied at training time (in the MLP) and at
  feature-materialization time (in `get_kmer_pair_features`, for k-NN
  baselines): **`unit_norm`** (parameter-free L2 row-norm) in current
  HA/NA & PB2/PB1 production. The older gen-3 bundles used `slot_norm`
  (learnable per-slot LayerNorm). LightGBM is scale-invariant and uses
  none; LR uses `StandardScaler` (linear model, scale matters for L2
  regularization).

**See also**: `docs/methods/kmer_features.md` (full pipeline: stride,
storage, ambiguous-base handling, leakage protection);
`docs/methods/feature_normalization.md` (full model × feature defaults
matrix and the slot-transform options); `docs/methods/gto_format_reference.md`
(the GTO contig source the k-mers are computed over).

---

## 8. Models

| Model | Role |
|---|---|
| **MLP** | The production classifier. Per-slot transform (current HA/NA & PB2/PB1: `unit_norm` — parameter-free L2 row-norm; older gen-3 bundles: `slot_norm` = learnable LayerNorm) → pair interaction (current: `unit_diff + prod`; older: `concat`) → 3-layer MLP → BCE loss. |
| **Logistic regression** | Linear baseline. Tells us how much of the signal is captured by an affine combination of features. |
| **LightGBM** | Tree-based baseline. Captures non-linear interactions without representation learning. |
| **1-NN cosine-margin (`knn1_margin`)** | The *leakage anchor*. Score = cosine to nearest train positive minus cosine to nearest train negative. If MLP ≈ 1-NN on AUC, the MLP is doing soft near-neighbor lookup, not generalization. |
| **k-NN vote (`knn_vote`)** | Smoothed neighborhood baseline. k=5 by default, distance-weighted. |

All five models are trained on the same Stage 3 dataset and produce
the same `test_predicted.csv` schema. Each writes its own run dir
under `models/flu/{ver}/runs/`.

**See also**: `docs/methods/feature_normalization.md` (preprocessing
defaults per model); `docs/methods/leakage.md` ("biology
learning" criterion: MLP needs to beat 1-NN by ≥0.02 AUC on
sequence-disjoint splits to claim biology generalization).

---

## 9. Evaluation: per-regime heatmap

The headline diagnostic is `aggregate_baselines_vs_mlp.py`, which
produces a **heatmap with one row per model and one column per
regime**:

- **Columns**: `positive` (TPR) + 8 negative regimes (TNR each, in
  ascending hardness from `none_match` to `host_subtype_year`).
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

Per-slot transform interacts differently with each model. Current
HA/NA & PB2/PB1 production setting `slot_transform: unit_norm` is
**parameter-free** (L2 row-norm), so it applies identically to the MLP
and to the k-NN baseline path (`unit_norm` flows through
`get_kmer_pair_features` unchanged). LR uses its own
`StandardScaler` (per-feature, column-wise — a different operation).
LightGBM uses none (tree splits are scale-invariant). The heatmap
caption flags the mismatch so reviewers don't read row-to-row score
differences as pure model-quality differences when they could partly
reflect different preprocessing.

For the older gen-3 bundles that use `slot_norm` (learnable
LayerNorm), the MLP gets a trainable γ/β while the k-NN baseline
path's coercion at `train_pair_baselines.py:430-435` falls back to
`'none'` for k-mer features (`slot_norm` not supported on non-negative
count vectors). In that setup the MLP and the k-NN baseline see
genuinely different per-slot preprocessing — worth flagging when
comparing.

**See also**: `docs/methods/leakage.md` ("biology
learning" criterion); `docs/post_hoc_analysis_design.md` (Level 1 / 2
methodology); `src/analysis/aggregate_baselines_vs_mlp.py` (the
script).

---

## 10. Leakage controls — quick map

The project tracks 5 canonical leakage modes (full taxonomy in
`docs/methods/leakage.md`):

| # | Mode | Status in this pipeline |
|---|---|---|
| 1 | Same-pair leakage (`pair_key` overlap across splits) | ✅ Mitigated — v2 cross-split assertion raises if any overlap. |
| 2 | Sequence-level label imbalance (a sequence appears only as positive or only as negative) | ✅ Mitigated — v2 coverage phase iterates `(slot, seq_hash)` and `(slot, dna_hash)`; `n_seqs_with_zero_negatives = 0` on every current production build. |
| 3 | Sequence-level leakage (same `seq_hash` / `dna_hash` in different splits) | ✅ **Mitigated** (2026-05-11) — `split_strategy.mode: seq_disjoint` with `hash_key: seq` is the production default for `flu_ha_na` and `flu_pb2_pb1`. Cross-split overlap on the active hash family is 0 by construction. |
| 4 | Cluster leakage (test pair feature-vector cosine-near a train pair) | ⚠️ Partially. Exp 2 (1-NN baseline `knn1_margin`) and Exp 3 (`exp3_cosine_deciles.py`) are implemented; Exp 4 (seq_disjoint) bounds the exact-hash case. Exp 5 (mmseqs2 cluster splits) not yet implemented. Initial seq_disjoint result on PB2/PB1: 1-NN edges MLP on MCC (0.900 vs 0.887). |
| 5 | Demographic shortcut (model uses metadata coincidence as proxy for "same isolate") | ⚠️ Confirmed present in legacy datasets. Construction-time mitigation in v2: `negative_sampling.regime_targets`. Re-test pending on regime-aware-built datasets via the heatmap. |

Note: full mitigation of mode #5 cannot be achieved by sampling alone
(the model can still pick up the pattern in training); the heatmap is
the test that says whether the mitigation worked.

**See also**: `docs/methods/leakage.md` for definitions
and the assessment plan.

---

## 11. Cross-references

For each topic, the deepest reference:

| Topic | File |
|---|---|
| Stage 1 detail (GTO parsing) | `docs/methods/preprocess.md` |
| Input GTO JSON schema (with corpus-wide statistics) | `docs/methods/gto_format_reference.md` |
| K-mer feature pipeline | `docs/methods/kmer_features.md` |
| Leakage taxonomy + biology-learning criterion | `docs/methods/leakage.md` |
| Per-(model × feature) preprocessing matrix | `docs/methods/feature_normalization.md` |
| Stage 3 builder design | `docs/plans/done/design_dataset_gen_v2.md` |
| Metadata-aware regime sampler design | `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md` |
| seq_disjoint routing design (mode #3 mitigation) | `docs/plans/done/2026-05-10_seq_disjoint_routing_plan.md` |
| Codon-aware k-mers feasibility note | `docs/plans/2026-05-12_codon_aware_kmer_features_plan.md` |
| Post-hoc analysis methodology | `docs/post_hoc_analysis_design.md` |
| Original metadata-shortcut finding | `docs/results/2026-05-07_metadata_shortcut_negatives.md` |
| Exp 4a seq_disjoint results (HA/NA) | `docs/results/2026-05-11_exp4a_seq_disjoint_results.md` |
| Project memory / current state | `.claude/memory.md`, `CLAUDE.md` |
