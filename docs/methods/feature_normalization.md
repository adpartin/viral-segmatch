# Feature normalization across models

> Companion reference for choosing input preprocessing per (model,
> feature_source) combination. Linked from each baseline module's
> docstring and from `aggregate_baselines_vs_mlp.py`'s heatmap caption.

This document is the source-of-truth for the project's defaults on
input scaling / normalization, and the rationale for each. When a new
model or feature source is added, update this table first, then make
the code default match.

---

## TL;DR — model × feature defaults

| Model               | ESM-2 embeddings (1280-dim per slot, signed, dense) | Nucleotide k-mer (4096-dim per slot, non-neg counts) | Protein k-mer (code path in production, no active bundle) |
|---|---|---|---|
| **Logistic regression** | StandardScaler                       | StandardScaler                                 | StandardScaler (extrapolated)                |
| **LightGBM**            | none                                  | none                                            | none (extrapolated)                          |
| **1-NN (margin) — `knn1_margin`** | none + slot_norm or unit_norm (recommended)¹ | none, optionally + `unit_norm` per bundle | none (extrapolated)        |
| **k-NN (vote) — `knn_vote`**      | none + slot_norm or unit_norm (recommended)¹ | none, optionally + `unit_norm` per bundle | none (extrapolated)                          |
| **MLP (deep learning)** | learned per-slot LayerNorm (`slot_transform: slot_norm`) — original setup. `unit_norm` (parameter-free L2 row-norm) also supported. | `slot_norm` or `unit_norm`. **Current HA/NA & PB2/PB1 production bundles use `unit_norm + (unit_diff + prod)` per Test 3 (2026-05-12).** | learned per-slot LayerNorm (extrapolated) |

¹ ESM-2 has a "protein-type subspace offset" (see "Features" below).
For cosine k-NN on ESM-2, applying a per-slot transform before the
distance computation is the recommended fix. Two options:
`slot_transform: slot_norm` (numpy LayerNorm — zero-mean / unit-variance
per row) or `slot_transform: unit_norm` (L2 row-norm only). The kmer-only
coercion in `train_pair_baselines.py:430` does NOT fire for ESM-2, so a
bundle that sets either flows through. Not yet measured against `none`
on ESM-2 — the active bundles are k-mer.

Defaults in code (live in `conf/baselines/default.yaml` as of 2026-05-12,
which is the single source of truth via `# @package bundles`):

- LR: `src/models/baselines/logistic.py::feature_scaling_default()` → `'standard'`
- LightGBM: `src/models/baselines/lgbm.py::feature_scaling_default()` → `'none'`
- 1-NN (margin): `src/models/baselines/knn1_margin.py::feature_scaling_default()` → `'none'`
- k-NN (vote): `src/models/baselines/knn_vote.py::feature_scaling_default()` → `'none'`
- MLP slot transform: `conf/training/base.yaml::slot_transform` → `'none'`
  (each bundle opts in: legacy gen-3 bundles use `'slot_norm'`; current
  `flu_ha_na` and `flu_pb2_pb1` use `'unit_norm'`).

---

## Models — what each needs from features

### Logistic regression

Linear in the input features. Two reasons to standardize:

1. **Regularization is per-feature** — L2 penalty `C × ||w||²` treats
   every coefficient on the same scale. If feature `j` has 1000× the
   variance of feature `k`, the optimizer will push `w_j` 1000× smaller
   to keep `|w_j × x_j|` roughly comparable, distorting the regularization.
2. **Convergence** — `lbfgs` and friends converge faster on standardized
   inputs (gradient magnitudes balanced across coordinates).

Standard practice. `StandardScaler` (zero-mean, unit-variance per feature)
is the canonical choice. We fit on train, transform val/test, persist the
scaler under `feature_scaler.joblib`.

### LightGBM (and tree-based models in general)

Tree splits compare each feature to a single threshold:
`x_j > t` is invariant to monotonic per-feature transforms (including
StandardScaler, log, etc.). Standardization neither helps nor hurts model
quality on trees. We default to `none` to:

- save the StandardScaler fit time and the transformed-matrix memory,
- preserve the natural sparse storage benefits when applicable,
- keep the model's saved `best_model.joblib` directly interpretable in
  the original feature space.

Reference: see e.g. *The Elements of Statistical Learning* (Hastie,
Tibshirani, Friedman), §10 on boosting and §9 on trees, for the
scale-invariance argument.

### 1-NN and k-NN with cosine metric

Both `knn1_margin` and `knn_vote` use sklearn's `metric='cosine'`. Cosine
distance is

```
cos_dist(x, y) = 1 − (x · y) / (||x|| × ||y||).
```

The `/||x||/||y||` term **is** a normalization — sklearn applies it to
every distance computation. So when a baseline declares `feature_scaling:
none`, what's actually happening is:

- **No `StandardScaler`** at materialization time (no per-feature shift
  or per-feature std-division).
- **L2 normalization at distance time**, internally, via the cosine
  formula. Equivalent to "L2-normalize both vectors, take dot product."

Pre-applying L2-normalization at materialization time is a no-op under
`metric='cosine'` — sklearn does it anyway. (It would only matter under
`metric='euclidean'`, where Euclidean on L2-normalized vectors gives the
same neighbor ordering as cosine. We don't use that path.)

**Why we do NOT default to `StandardScaler` for k-NN:**

"Cosine is scale-invariant" is shorthand for "cosine is invariant to
multiplying an entire vector by a positive scalar." It is *not*
invariant to per-feature standardization. After `StandardScaler`:

- Non-negative count vectors stop being non-negative (mean-subtraction
  introduces negatives).
- Per-feature std-division amplifies rare features and dampens common
  ones — conceptually similar to TF-IDF reweighting, but applied
  silently and uniformly. The cosine angles between vectors change.

For k-mer counts the natural geometry already plays well with cosine
(k-mer composition vectors point in directions that group same-segment
isolates close together regardless of segment length). Applying
`StandardScaler` would actively distort that geometry. Empirical
support from this project (current production bundles
`flu_ha_na` and `flu_pb2_pb1`, `seq_disjoint` routing + `unit_norm` +
`unit_diff + prod`, single seed, results verified 2026-05-12 from
`results/flu/July_2025/runs/baselines_vs_mlp_*_20260512_*/baselines_vs_mlp.csv`):

| Model | HA/NA AUC-ROC | HA/NA MCC | PB2/PB1 AUC-ROC | PB2/PB1 MCC |
|---|---:|---:|---:|---:|
| LightGBM     | 0.9830 | 0.881 | 0.9824 | 0.879 |
| 1-NN margin  | 0.9771 | 0.892 | 0.9815 | 0.900 |
| k-NN vote    | (in same run) | — | (in same run) | — |
| MLP          | 0.9771 | 0.885 | 0.9760 | 0.887 |

The 1-NN and MLP results are nearly indistinguishable on aggregate
metrics under this setup — and on PB2/PB1 the 1-NN baseline narrowly
edges the MLP on MCC (0.900 vs 0.887). See
`docs/methods/leakage_definitions.md` for the role of the 1-NN result
in the "biology learning" criterion and `docs/results/2026-05-11_exp4a_seq_disjoint_results.md`
for the earlier HA/NA Exp 4a numbers under the looser `hash_key=dna`
routing.

For ESM-2 features the picture changes — see ESM-2 section below.

### MLP

Neural networks benefit from input standardization both for convergence
(gradients balanced across coordinates) and for the model's internal
LayerNorm/BatchNorm to start in a useful regime. The project's MLP path
supports two per-slot transforms applied **before** the interaction
(concat / unit_diff / unit_prod / …) is computed:

- **`slot_norm`** — `nn.LayerNorm(embed_dim)` on each slot
  independently. **Learnable** (`γ` and `β` are trained alongside the
  classifier). This is the gen-3 / original setup; the legacy 28-pair
  sweeps in `paper_outline_v2.md` use it.
- **`unit_norm`** — parameter-free L2 row-norm per slot, added
  2026-05-12. No γ/β; equivalent to applying `Normalizer(norm='l2')`
  per slot. Current HA/NA and PB2/PB1 production bundles use this
  setting (Test 3, paired with `interaction: unit_diff + prod`).

The legacy `slot_norm` differs from the unparameterized numpy
LayerNorm the baseline harness exposes (`_pair_features.py:96
::_apply_slot_norm`) by the γ/β learning. The standardization
(zero-mean, unit-variance per row) is identical; only the trainable
scale/shift on top differs. `unit_norm` has a numpy counterpart at
`_pair_features.py:113::_apply_unit_norm` that mirrors the MLP path
exactly (no parameters to mirror).

Why slot-LEVEL rather than feature-LEVEL standardization? See ESM-2
section below — the MLP needs per-row normalization to remove the
"protein-type subspace offset" before the interaction step.

---

## Features — what each one brings

### ESM-2 embeddings (mean-pooled)

- **Shape**: 1280 dims per slot, dense, signed (typical ranges roughly
  `[-2, +2]` per dim, varying across dims).
- **Geometry quirk — protein-type subspace offset**: raw mean-pooled
  ESM-2 vectors for HA and NA (and presumably other protein-type pairs)
  occupy slightly *different* subspaces of the 1280-dim space. This is
  what makes `unit_diff` work but `concat` collapse on homogeneous data
  (CLAUDE.md "ESM-2 unit_diff > concat on H3N2: AUC 0.96 vs 0.50"). For
  the MLP, `slot_norm` removes the offset enough that `concat` and
  `unit_diff` both work. For ESM-2 cosine k-NN, the same offset would
  bias neighbors toward "same protein type" rather than "same isolate
  pair" — applying `slot_norm` (unparameterized in the baseline path) is
  the recommended fix.

### Nucleotide k-mers (k=6, 4096-dim per slot)

- **Shape**: 4096 dims per slot, sparse-but-densified-at-train-time,
  non-negative integers (raw counts).
- **Length confound**: longer segments have larger total k-mer counts.
  Cosine handles this naturally (the `/||x||` step normalizes the
  composition vs the count). For LR / LightGBM the model has to deal
  with it directly; standardization helps LR, trees handle it natively.
- **Project finding**: k-mer features are *interaction-agnostic* on the
  MLP (`unit_diff ≈ concat` on H3N2 — CLAUDE.md). The bundle still
  applies `slot_norm` because the architecture is configured that way,
  but the empirical benefit is smaller than it is for ESM-2.
- **Documented**: see `docs/methods/kmer_features.md` for the full
  pipeline (Stage 2b script, storage layout, sparse → dense conversion
  cost, why-not-normalize discussion).

### Protein k-mers — in production code, no active bundle

Amino-acid n-grams of protein sequences (vocabulary = 20^k, e.g.,
8,000 for k=3, 160,000 for k=4). As of 2026-05-12,
`sequences_to_sparse_kmer_matrix` accepts an `alphabet` parameter (DNA
default `'ACGT'`, protein `'ACDEFGHIKLMNPQRSTVWY'`), so the same
machinery handles both. Practical ceiling is k≈4 before the exhaustive
`|alphabet|^k` vocabulary becomes impractical to enumerate. Same
mathematical properties as nucleotide k-mers — non-negative counts,
sparse, length-confounded — so the same model × scaling defaults
extrapolate (StandardScaler for LR; none for LightGBM / k-NN; learned
LayerNorm or `unit_norm` for MLP). No active bundle exercises this
path; treat the extrapolations as unverified until benchmarked.

---

## Per-combination defaults — rationale and where they live

### LR + ESM-2

Default: `feature_scaling: standard`. Set in
`baselines/logistic.py::feature_scaling_default()`. Standard practice for
LR; also cancels the protein-type subspace offset (StandardScaler
re-centers each dim across the train distribution). Acceptable
preprocessing for ESM-2 + LR even though it's not the same operation as
slot_norm — for a linear model the two affine transformations of the
input live in the same equivalence class up to a coefficient remapping.

### LR + nucleotide k-mers

Default: `feature_scaling: standard`. Same default as ESM-2; overrides
the natural sparsity of count vectors (densifies when StandardScaler
mean-shifts them to zero). For our matrix size (~104K pair rows × 8,192
dims for the current HA/NA build, ≈ 3.4 GB densified at float32; doubles
under `neg_to_pos_ratio: 2.0`) this is fine. For substantially larger
sweeps a sparse-friendly variant (TF-IDF, or LR with `solver='saga'` on
the sparse matrix without StandardScaler) would be the right swap.

### LightGBM + any feature source

Default: `feature_scaling: none`. Tree split rules are scale-invariant.
LightGBM's histogram-based binning consumes raw values directly. No
preprocessing wins.

### 1-NN / k-NN + ESM-2

Recommended: `feature_scaling: none` AND `slot_transform: slot_norm` on
the bundle. The slot_norm step is critical to remove the protein-type
subspace offset before cosine compares the two slots. The kmer-only
coercion in `train_pair_baselines.py` does not fire for ESM-2, so the
bundle's `training.slot_transform` flows through to the baseline
materialization. NOT yet head-to-head-tested vs `slot_transform: none`
on ESM-2 — flag if testing.

### 1-NN / k-NN + nucleotide k-mers

Default: `feature_scaling: none`. Cosine handles vector-norm
normalization internally (see "Models" section). Per-feature
StandardScaler would distort the count geometry.

The k-mer baseline path **accepts** `slot_transform ∈ {'none',
'unit_norm'}` (`src/utils/kmer_utils.py:110-119`). `slot_norm` is
explicitly rejected for k-mer features (non-negative count vectors
don't benefit from feature-axis LayerNorm). If a bundle sets
`training.slot_transform: slot_norm` for the MLP, the coercion at
`train_pair_baselines.py:430-435` forces the baseline run to use
`'none'` with a printed note. `unit_norm` flows through unchanged —
the current HA/NA and PB2/PB1 bundles use this, so the k-NN baseline
sees the same per-slot transform as the MLP for those runs.

### MLP + ESM-2

Default in active ESM-2 bundles: `slot_transform: slot_norm` (learnable
per-slot LayerNorm). `unit_norm` is also supported but not actively
benchmarked on ESM-2 in current bundles. **Critical** per project
finding (CLAUDE.md): "LayerNorm (slot_norm) is critical for homogeneous
subsets: Without it, raw HA/NA embeddings live in slightly different
subspaces; unit_diff then picks up slot offset rather than biological
signal."

### MLP + nucleotide k-mers

Current production for `flu_ha_na` and `flu_pb2_pb1` (gen-3, 2026-05-12):
`slot_transform: unit_norm` + `interaction: unit_diff + prod` (Test 3
of the four-interaction sweep). All four Tests 1–4 lie within ~0.5% F1
and ~0.1% AUC-ROC on a single-seed run — differences are at seed-noise
level; multiple seeds needed before claiming a winner. The earlier
gen-3 28-pair sweep (`paper_outline_v2.md` §11) used
`slot_transform: slot_norm` + `interaction: concat`; that combination
also works on k-mer.

The benefit of any per-slot transform on k-mer is empirically smaller
than on ESM-2 (see CLAUDE.md "K-mer concat does NOT collapse on H3N2").

**Protein k-mers** for all three model classes (LR, k-NN, MLP):
extrapolated from the nucleotide-k-mer defaults; no active bundle
exercises this path. See the TL;DR table at the top for the
extrapolation.

---

## Common pitfalls

- **`StandardScaler` is not L2-normalization.** StandardScaler shifts
  each *column* to mean 0 and divides by per-column std. L2-norm
  rescales each *row* (sample) to unit length. They are different
  operations with different geometric effects. Cosine k-NN cares about
  L2 (rows); LR cares about column scales.
- **Cosine metric implicitly L2-normalizes — do NOT apply it twice.**
  `metric='cosine'` in sklearn already normalizes vector lengths inside
  the distance formula. Pre-applying `Normalizer(norm='l2')` is a no-op
  for ranking.
- **`slot_transform` (MLP) and `feature_scaling` (baselines) are
  different concepts.** slot_transform operates per-slot, per-row, on
  the embedding axis (LayerNorm). feature_scaling operates per-feature,
  per-column, across the train distribution (StandardScaler). They
  cannot be substituted for each other; they answer different
  questions.
- **For ESM-2 baselines specifically: slot_transform from the bundle
  flows through.** The kmer-only coercion in
  `train_pair_baselines.py:430-435` is conditional on
  `FEATURE_SOURCE == 'kmer'`, so it does NOT fire for ESM-2. If your
  bundle sets `training.slot_transform: slot_norm` (or `unit_norm`)
  for the MLP, the ESM-2 baselines will also see that transform applied
  (via the unparameterized `_apply_slot_norm` / `_apply_unit_norm`
  in `_pair_features.py:96` / `:113`). This is usually what you want;
  flag it if you're trying to compare scenario-by-scenario.

---

## See also

- `conf/baselines/default.yaml` — single source of truth for baseline
  hyperparameter and `feature_scaling` defaults (centralized
  2026-05-12; wired via `# @package bundles`).
- `docs/methods/kmer_features.md` — full k-mer pipeline; "Why not
  normalize counts?" section discusses the length confound.
- `docs/methods/leakage_definitions.md` — uses 1-NN as the leakage
  anchor for the "biology learning" criterion.
- `src/models/_pair_features.py` — the baseline-side feature loader;
  `_apply_slot_norm` (line 96, numpy LayerNorm) and `_apply_unit_norm`
  (line 113, L2 row-norm) live here. The k-mer slot_norm rejection
  itself lives in `src/utils/kmer_utils.py:114-119`.
- `src/models/train_pair_baselines.py` — the multi-baseline runner;
  shared materialization, per-baseline scaling, k-mer slot_transform
  coercion at lines 430–435 (slot_transform forced to `'none'` for
  k-mer baselines unless it's already `'none'` or `'unit_norm'`).
- `src/analysis/aggregate_baselines_vs_mlp.py` — cross-model heatmap;
  default caption flags the featurization mismatch. Outputs at
  `results/{virus}/{data_version}/runs/baselines_vs_mlp_*`.
- `CLAUDE.md` — recurring "Key Findings" section enumerating the
  ESM-2 vs k-mer × interaction × slot_transform results that motivate
  these defaults.
