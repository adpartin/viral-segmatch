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

| Model               | ESM-2 embeddings (1280-dim per slot, signed, dense) | Nucleotide k-mer (4096-dim per slot, non-neg counts) | Protein k-mer (not yet implemented) |
|---|---|---|---|
| **Logistic regression** | StandardScaler                       | StandardScaler                                 | StandardScaler (extrapolated)                |
| **LightGBM**            | none                                  | none                                            | none (extrapolated)                          |
| **1-NN (margin) — `knn1_margin`** | none + slot_norm (recommended)¹ | none (sklearn's `metric='cosine'` normalizes vector lengths internally) | none (extrapolated)        |
| **k-NN (vote) — `knn_vote`**      | none + slot_norm (recommended)¹ | none (same as 1-NN)                              | none (extrapolated)                          |
| **MLP (deep learning)** | learned per-slot LayerNorm (`slot_transform: slot_norm`) | learned per-slot LayerNorm (`slot_transform: slot_norm`) | learned per-slot LayerNorm (extrapolated) |

¹ ESM-2 has a "protein-type subspace offset" (see "Features" below).
For cosine k-NN on ESM-2, applying `slot_transform: slot_norm` before
concat is the recommended fix; today the kmer-only coercion in
`train_pair_baselines.py` does NOT fire for ESM-2, so a bundle that
sets `training.slot_transform: slot_norm` will get it on ESM-2 k-NN
runs as expected. Not yet measured against `none` on ESM-2 — the active
bundle is k-mer.

Defaults in code:

- LR: `src/models/baselines/logistic.py::feature_scaling_default()` → `'standard'`
- LightGBM: `src/models/baselines/lgbm.py::feature_scaling_default()` → `'none'`
- 1-NN (margin): `src/models/baselines/knn1_margin.py::feature_scaling_default()` → `'none'`
- k-NN (vote): `src/models/baselines/knn_vote.py::feature_scaling_default()` → `'none'`
- MLP slot transform: `conf/training/base.yaml::slot_transform` → `'none'` (each
  bundle that wants `slot_norm` opts in via `training.slot_transform: slot_norm`;
  the active gen-3 bundles do).

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
support from this project: with `feature_scaling: none`, both
`knn1_margin` and `knn_vote` reach test AUC ≈ 0.984 on
`flu_ha_na_neg_regimes` (k-mer, k=6, full Flu-A) — competitive with
LightGBM (AUC ≈ 0.988) and a tight upper bound for what near-neighbor
lookup can produce. See
`docs/methods/leakage_definitions.md` for the role of the 1-NN result
in the "biology learning" criterion.

For ESM-2 features the picture changes — see ESM-2 section below.

### MLP

Neural networks benefit from input standardization both for convergence
(gradients balanced across coordinates) and for the model's internal
LayerNorm/BatchNorm to start in a useful regime. The project's MLP path
applies `slot_transform: slot_norm`, which is `nn.LayerNorm(embed_dim)`
on each slot independently *before* the interaction (concat / unit_diff /
…) is computed.

`slot_norm` is **learnable** — `γ` and `β` are trained alongside the
classifier. This differs from the unparameterized numpy LayerNorm the
baseline harness exposes (`_pair_features.py::_apply_slot_norm`) by the
γ/β learning. The standardization (zero-mean, unit-variance per row) is
identical; only the trainable scale/shift on top differs.

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

### Protein k-mers — not currently used

A possible future feature source: amino-acid n-grams of protein
sequences (vocabulary = 20^k, e.g., 8000 for k=3, 160000 for k=4). Same
mathematical properties as nucleotide k-mers — non-negative counts,
sparse, length-confounded — so the same model × scaling defaults
extrapolate (StandardScaler for LR; none for LightGBM / k-NN; learned
LayerNorm for MLP). Not yet measured.

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
mean-shifts them to zero). For our matrix size (~141K rows × 8192 dims
= ~9 GB densified) this is fine. For substantially larger sweeps a
sparse-friendly variant (TF-IDF, or LR with `solver='saga'` on the
sparse matrix without StandardScaler) would be the right swap.

### LR + protein k-mers

Same default as nucleotide; extrapolated. Unverified.

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
StandardScaler would distort the count geometry. The kmer-slot_norm
coercion in `train_pair_baselines.py` forces `slot_transform: none` here
because the kmer baseline path explicitly rejects slot_norm
(`_pair_features.py:298-303`) — non-negative count vectors don't benefit
from feature-axis LayerNorm. Bundle setting `training.slot_transform:
slot_norm` (for the MLP) is silently coerced back to `none` for the
baseline run with a printed note.

### 1-NN / k-NN + protein k-mers

Same default as nucleotide; extrapolated. Unverified.

### MLP + ESM-2

Default: `slot_transform: slot_norm` (learnable per-slot LayerNorm).
**Critical** per project finding (CLAUDE.md): "LayerNorm (slot_norm) is
critical for homogeneous subsets: Without it, raw HA/NA embeddings live
in slightly different subspaces; unit_diff then picks up slot offset
rather than biological signal."

### MLP + nucleotide k-mers

Default in active bundles: `slot_transform: slot_norm`. The benefit on
k-mer is empirically smaller than on ESM-2 (see CLAUDE.md "K-mer concat
does NOT collapse on H3N2"), but the bundle is configured uniformly so
the MLP architecture is the same across feature sources.

### MLP + protein k-mers

Same default as nucleotide; extrapolated. Unverified.

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
  `train_pair_baselines.py` does NOT fire for ESM-2, so if your bundle
  sets `training.slot_transform: slot_norm` for the MLP, the ESM-2
  baselines will also get slot_norm applied (via the unparameterized
  `_apply_slot_norm` in `_pair_features.py`). This is usually what you
  want; flag it if you're trying to compare scenario-by-scenario.

---

## See also

- `docs/methods/kmer_features.md` — full k-mer pipeline; "Why not
  normalize counts?" section discusses the length confound.
- `docs/methods/leakage_definitions.md` — uses 1-NN as the leakage
  anchor for the "biology learning" criterion.
- `src/models/_pair_features.py` — the baseline-side feature loader;
  `_apply_slot_norm` (numpy LayerNorm) and the kmer-rejection of
  slot_norm live here.
- `src/models/train_pair_baselines.py` — the multi-baseline runner;
  shared materialization, per-baseline scaling, kmer-slot_norm
  coercion.
- `src/analysis/aggregate_baselines_vs_mlp.py` — cross-model heatmap;
  default caption flags the featurization mismatch.
- `CLAUDE.md` — recurring "Key Findings" section enumerating the
  ESM-2 vs k-mer × interaction × slot_transform results that motivate
  these defaults.
