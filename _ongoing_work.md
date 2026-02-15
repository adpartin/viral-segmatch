# Ongoing Work

## Interaction Formulas

Given two D-dimensional embeddings `A = emb_a` and `B = emb_b` (D=1280 for ESM-2):

| Interaction | Formula | Output dim | Notes |
|-------------|---------|-----------|-------|
| `concat` | `[A, B]` | 2D | Preserves both embeddings fully; order-sensitive |
| `diff` | `\|A - B\|` (element-wise absolute value) | D | Removes direction sign, keeps magnitude per dimension |
| `unit_diff` | `(A - B) / (‖A - B‖₂ + ε)` where ε=1e-8 | D | L2-normalized signed diff; strips magnitude, retains direction only |
| `prod` | `A ⊙ B` (element-wise product) | D | Captures co-activation patterns |

Combinations (e.g., `concat+diff`) concatenate the respective outputs.

---

## Magnitude vs Direction

This relates to geometric properties of any embedding space (not a transformer architecture specifically).

**Example** with 4-dim embeddings:

```
Pair 1 (positive): emb_a = [10, -4, 7, 6],  emb_b = [12, -8, -1, 4]
  diff = [-2, 4, 8, 2]
  ||diff|| = 9.17
  unit_diff = [-0.22, 0.44, 0.87, 0.22]

Pair 2 (negative): emb_a = [5, 3, 1, 0],  emb_b = [2, -6, -3, 8]
  diff = [3, 9, 4, -8]
  ||diff|| = 13.04
  unit_diff = [0.23, 0.69, 0.31, -0.61]
```

- **Magnitude** (scalar): 9.17 vs 13.04. A magnitude-based model just learns: "if `||diff|| < threshold`, predict positive." This is a 1-D decision boundary.

- **Direction** (1280-dim unit vector): Encodes **which dimensions** differ and by how much proportionally. A direction-based model learns: "when dimensions 37, 210, 945 shift in a coordinated pattern, predict positive." This is a rich, high-dimensional decision boundary.

**The concern is about generalization, not correctness on the training set.** If the model relies on magnitude:
- It works when positive pairs are systematically closer than negatives (which might be the case in our specific training data)
- It **fails** when encountering a hard negative that happens to have small `||diff||` (e.g., two very similar but non-co-occurring proteins)
- It **fails** when encountering a true positive with large `||diff||` (e.g., reassortant where HA and NA are from divergent lineages)

**Bottom line**: Magnitude is a legitimate but **shallow** feature. Direction carries richer information (which residue positions drive the compatibility).

### Experimental Results: `concat` vs `unit_diff` Across Metadata Filters

Architecture: `pre_mlp_mode: slot_norm`, `pre_mlp_dims: null` (per-slot LayerNorm only, no linear layers). Schema-ordered HA → NA.

| Filter | `concat` F1 | `concat` AUC | `unit_diff` F1 | `unit_diff` AUC |
|--------|-------------|--------------|----------------|-----------------|
| None (general) | 0.929 | 0.975 | 0.917 | 0.966 |
| year=2024 | 0.770 | 0.888 | 0.771 | 0.891 |
| geo=Illinois | 0.845 | 0.939 | 0.860 | 0.946 |
| host=Human | 0.915 | 0.964 | 0.897 | 0.952 |
| subtype=H3N2 | **0.570** | **0.498** | **0.877** | **0.962** |

**Key findings**:

1. `concat` and `unit_diff` perform comparably on diverse data (general, 2024, Illinois, Human).
2. **`concat` completely fails on H3N2** (F1=0.57, AUC=0.50 = random). When isolates are homogeneous (same subtype), the magnitude/ordering shortcuts that `concat` exploits in diverse data vanish, and the model has nothing to learn.
3. **`unit_diff` succeeds on H3N2** (F1=0.88, AUC=0.96), confirming that the **direction of the difference vector carries genuine biological signal** that is robust even in homogeneous populations.
4. The year=2024 filter reduces performance for both interactions (~0.77 F1), likely due to smaller dataset size and reduced temporal diversity rather than a methodological issue.
5. **`concat` + H3N2 + patience=100 still fails** -- confirmed that even with 100 epochs, `concat` cannot learn on homogeneous H3N2 data. The failure is architectural, not an early-stopping artifact.

### Ablation: LayerNorm Contribution (`slot_norm` vs `none`)

**Rationale**: The `slot_norm` pre-MLP mode applies per-slot `LayerNorm(1280)` before the interaction. This re-centers and re-scales each slot's embeddings, which could help `unit_diff` by neutralizing systematic offset between HA and NA embedding distributions. To isolate LayerNorm's contribution, we tested `pre_mlp_mode: none` (raw ESM-2 embeddings go directly into `unit_diff`).

| Bundle | pre_mlp_mode | Filter | Result |
|--------|-------------|--------|--------|
| `flu_schema_raw_slot_norm_unit_diff` | slot_norm | None | High score (F1=0.917) |
| `flu_schema_raw_none_unit_diff` | none | None | High score |
| `flu_schema_raw_slot_norm_unit_diff_h3n2` | slot_norm | H3N2 | High score (F1=0.877, delayed learning) |
| `flu_schema_raw_none_unit_diff_h3n2` | none | H3N2 | **Failed** (~0.69 val_loss, 100 epochs patience) |

**Key finding**: LayerNorm is **not needed** for diverse data (general case works with or without it). But LayerNorm is **critical for homogeneous data** (H3N2). Without it, the raw HA and NA embeddings live in slightly different subspaces (different mean/variance per dimension). The difference `emb_a - emb_b` mixes genuine biological signal with this systematic slot offset. After L2 normalization (`unit_diff`), this offset dominates the direction, drowning the subtle within-subtype signal. LayerNorm neutralizes that offset, giving `unit_diff` a cleaner directional signal to work with.

### Delayed Learning Phenomenon (H3N2 + unit_diff)

In the `flu_schema_raw_slot_norm_unit_diff_h3n2` runs, the model exhibited a characteristic plateau-then-breakthrough pattern:

| Seed | Plateau duration | Transition epoch | Final F1 |
|------|-----------------|-----------------|----------|
| 42 (default) | ~9 epochs | ~10 | 0.877 |
| 123 | ~31 epochs | ~32 | 0.877 |

During the plateau: loss flat at ~0.693 (= -ln(0.5), random guessing), F1 oscillating around 0.0-0.58, AUC ~0.5. After the transition: loss drops rapidly, F1 jumps to ~0.8+, and the model converges to similar final performance regardless of seed.

**Seed experiment confirms**: The plateau duration is **seed-dependent** (9 vs 31 epochs) but the phase transition and final performance are **problem-dependent**. With seed=123, the default patience=20 was insufficient -- the model was killed before the breakthrough. Increasing patience to 40 allowed the model to reach the transition at epoch ~32 and achieve the same final F1=0.877.

**Why this happens -- analysis**:

1. **Loss landscape with unit vectors is harder to navigate**. When you normalize `a - b` to unit length, all input vectors lie on a hypersphere. The loss landscape on a hypersphere has different geometry than in unconstrained Euclidean space -- gradients are effectively constrained to tangent planes, and the model must learn to distinguish angular patterns rather than exploiting scale.

2. **The H3N2-only data is more homogeneous**. Within a single subtype, embedding differences are more subtle. Combined with L2 normalization (which removes the "easy" magnitude signal), the model starts with no obvious gradient direction. The LayerNorm layers (from `slot_norm`) may need several epochs to calibrate their running statistics before useful gradients flow.

3. **Phase transition = the model found a useful feature subspace**. The sharp transition is characteristic of a "grokking"-like phenomenon: the model suddenly discovers a low-dimensional structure in the high-dimensional unit-sphere manifold that separates positives from negatives. Before that point, it was exploring a flat region of the loss surface. This is common in settings where the signal-to-noise ratio is low and the model must learn a non-trivial transformation before gradient descent can make progress.

4. **The full dataset (all subtypes) doesn't show this delay** because inter-subtype differences provide stronger angular separation even on the unit sphere -- the model finds useful gradients immediately.

**Practical implications**:
- If using `unit_diff` on homogeneous subsets, **increase patience** (e.g., 40+) to avoid early termination before the phase transition.
- The plateau duration is highly sensitive to weight initialization (seed). Consider running multiple seeds if a run appears to "fail."
- The delayed learning is not a bug -- it's evidence that the model is learning a harder, more genuine pattern rather than a shortcut.

---


## Systematic Bias and Population-Level Confounders

### The Concern

Positive pairs are within-isolate (same host/subtype/year/geography), negatives are cross-isolate (potentially different host/subtype/year). The model might learn "same population" pattern rather than residue-based compatibility. --> **TODO:** Confirm this explanation! Maybe this? "The model might learn same/different population pattern rather than same/different isolate pattern."

**It's fine if**: The magnitude/direction difference reflects true viral compatibility (e.g., HA and NA must co-evolve, so co-occurring pairs have coordinated mutations).

**It's a problem if**: The difference reflects dataset construction artifacts -- the model learns "same host signature" rather than "protein compatibility."

Current negatives are "easy" because cross-isolate pairs may differ in subtype, year, or host, making `||HA_A - NA_B||` large for superficial reasons. The FP pattern (more FP than FN, i.e., model over-predicts positive) suggests some cross-isolate pairs "look compatible" -- likely because proteins from similar isolates (same subtype/year) produce small `||diff||`.

### Validation Approaches

**A) FP/FN Metadata Analysis** (lightweight, using existing files):

`false_positives_detailed.csv` and `false_negatives_detailed.csv` have `assembly_id_a`/`assembly_id_b`. Join with `isolate_metadata.csv` (has `assembly_id`, `host`, `hn_subtype`, `year`, `geo_location_clean`) to answer:
- FPs: What fraction share the same subtype/host/year? If concentrated among "biologically close" pairs, the model can't distinguish "close but non-co-occurring" from "truly co-occurring."
- FNs: Are they enriched for divergent-lineage pairs (reassortants)? If so, the model expects co-occurring proteins to be "similar" and fails when they're not.

**B) Strict Metadata Filtering** (YAML-only config change):

Filter by multiple metadata fields (e.g., `host=Human, hn_subtype=H3N2, year=2024`) so all isolates share the same population confounders. The model **cannot** exploit host/subtype/year as shortcuts.
- Performance holds → signal is genuinely sequence-level.
- Performance collapses → model was relying on confounders.
- **Caveat**: May produce very small datasets. Need >500 total pairs to be trainable.

**C) Hard Negative Sampling** (requires code change):

Modify `create_negative_pairs` so negatives are drawn from isolate pairs that **share** the same subtype/year/host ("super-hard" negatives). Preserves dataset size but forces the model to rely on sequence-level signal.

- B and C address the same concern differently.
- **B** is stricter, simpler (YAML change), but smaller datasets.
- **C** preserves size but requires modifying negative sampling logic.
- **Recommendation**: Start with B. If dataset too small, consider C.

### Status

Not yet implemented. Next step: create a strict-filter bundle (e.g., `flu_schema_raw_slot_norm_h3n2_human_2024`) and check pair counts. If viable, train and compare.

---


## Multi-Interaction PCA Plots -- IMPLEMENTED

New function `plot_pair_interactions()` in `visualize_dataset_stats.py`:
- Loads raw `emb_a`, `emb_b` once, then computes all interactions from them.
- Default interactions: `["concat", "diff", "prod", "unit_diff"]`.
- Supports combinations (e.g., `"concat+diff"` → `pair_pca_concat_diff.png`).
- Per-interaction diagnostics: `corr(||feature||, label)`, PCA variance, mean norms by label.
- All saved to `pair_interaction_diagnostics.json`.
- Reuses `_plot_pair_embedding_splits_2d` (split colors + label styling, filter textbox).
- PCA only (no UMAP -- 4 UMAP runs would be too slow).

`plot_pair_embeddings_splits_overlap()` marked **DEPRECATED** in docstring and no longer called from the main visualization flow.

Architecture-dependent interactions (slot_norm, pre_mlp) require trained model weights and are currently out of scope for dataset-time visualization.

### Output files (per dataset run)

```
plots/
  pair_pca_concat.png
  pair_pca_diff.png
  pair_pca_prod.png
  pair_pca_unit_diff.png
  pair_interaction_diagnostics.json
```

---


## Individual Embedding Plots -- PENDING

`plot_sequence_embeddings_by_confounders_from_pairs()` exists but hasn't been used. Could be extended to overlay emb_a vs emb_b with different markers and color by metadata (host, subtype, etc.). Not yet implemented.

---

## Train/Val/Test Balance (Q9)

**Current setup**: Train is balanced (positive_ratio ≈ 0.5); val and test are typically imbalanced (e.g., 0.34, 0.30).

**Impact on performance analysis**:
- **Train balance**: Good for learning. Balanced training avoids the model defaulting to the majority class.
- **Val imbalance**: Early stopping and threshold selection are done on val. If val is imbalanced, the optimal threshold may shift (e.g., toward 0.4 instead of 0.5) to maximize F1 on that distribution. This is acceptable as long as we report test metrics with the chosen threshold.
- **Test imbalance**: Realistic. In deployment, we expect varying positive rates. Reporting F1, AUC, Brier, and precision/recall gives a full picture regardless of balance.

**Recommendation**: Keep train balanced. Val/test imbalance is fine; developers control train/val splits, but test should reflect real-world distribution. Use `positive_ratio` (rounded to 3 decimal places in `dataset_stats.json`) to document each split's class balance.

---

