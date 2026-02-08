# Ongoing Work

## 1) Magnitude vs Direction: Concrete Explanation

This has nothing to do with transformer architecture specifically -- it's a geometric property of any embedding space.

**Concrete example** with 4-dim embeddings:

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

- **Magnitude** (scalar): 9.17 vs 13.04. A magnitude-based model just learns: "if `||diff|| < threshold`, predict positive." This is a 1D decision boundary.

- **Direction** (1280-dim unit vector): Encodes **which dimensions** differ and by how much proportionally. A direction-based model learns: "when dimensions 37, 210, 945 shift in a coordinated pattern, predict positive." This is a rich, high-dimensional decision boundary.

**The concern is about generalization, not correctness on the training set.** If the model relies on magnitude:
- It works when positive pairs are systematically closer than negatives (which they might be in your training data)
- It **fails** when encountering a hard negative that happens to have small `||diff||` (e.g., two very similar but non-co-occurring proteins)
- It **fails** when encountering a true positive with large `||diff||` (e.g., reassortant where HA and NA are from divergent lineages)

**Bottom line**: Magnitude is a legitimate but **shallow** feature. Direction carries richer biological information (which residue positions drive the compatibility).

### Experimental Results: `unit_diff` (direction only) vs `diff` (magnitude + direction)

We implemented `interaction: unit_diff` which L2-normalizes the diff vector: `(emb_a - emb_b) / ||emb_a - emb_b||`, stripping all magnitude information and retaining only the direction. Two bundles were tested:

| Bundle | Interaction | Result |
|--------|------------|--------|
| `flu_schema_raw_slot_norm_unit_diff` | `unit_diff` (direction only) | Very high score |
| `flu_schema_raw_slot_norm_h3n2_unit_diff` | `unit_diff` (direction only, H3N2) | Very high score (with delayed learning) |

**Key finding**: Performance with `unit_diff` is comparable to `diff`, confirming that the **direction of the difference vector carries the discriminative signal**, not the magnitude. This is a positive result -- it suggests the model is learning genuine biological compatibility patterns in the embedding space, not just exploiting "how far apart" the proteins are.

### Delayed Learning Phenomenon (H3N2 + unit_diff)

In the `flu_schema_raw_slot_norm_h3n2_unit_diff` run, the model exhibited a striking pattern:
- **Epochs 1-9**: Loss flat at ~0.6920 (equivalent to random guessing on a balanced binary task, since -ln(0.5) = 0.693). Train and val F1 oscillating around 0.5. The model was effectively stuck at the random baseline.
- **Epoch ~10**: Sharp phase transition -- loss drops rapidly, F1 jumps from ~0.5 to ~0.8, AUC-ROC surges from ~0.5 to ~0.9.
- **Epochs 10+**: Continued improvement, reaching very high final performance.

**Why this happens -- analysis**:

1. **Loss landscape with unit vectors is harder to navigate**. When you normalize `a - b` to unit length, all input vectors lie on a hypersphere. The loss landscape on a hypersphere has different geometry than in unconstrained Euclidean space -- gradients are effectively constrained to tangent planes, and the model must learn to distinguish angular patterns rather than exploiting scale.

2. **The H3N2-only data is more homogeneous**. Within a single subtype, embedding differences are more subtle. Combined with L2 normalization (which removes the "easy" magnitude signal), the model starts with no obvious gradient direction. The LayerNorm layers (from `slot_norm`) may need several epochs to calibrate their running statistics before useful gradients flow.

3. **Phase transition = the model found a useful feature subspace**. The sharp transition at epoch ~10 is characteristic of a "grokking"-like phenomenon: the model suddenly discovers a low-dimensional structure in the high-dimensional unit-sphere manifold that separates positives from negatives. Before that point, it was exploring a flat region of the loss surface. This is common in settings where the signal-to-noise ratio is low and the model must learn a non-trivial transformation before gradient descent can make progress.

4. **The full dataset (all subtypes) doesn't show this delay** because inter-subtype differences provide stronger angular separation even on the unit sphere -- the model finds useful gradients immediately.

**Practical implications**:
- If using `unit_diff` on homogeneous subsets, consider **warming up with a higher learning rate** or **longer patience** for early stopping to avoid terminating before the phase transition.
- The delayed learning is not a bug -- it's evidence that the model is learning a harder, more genuine pattern rather than a shortcut.

---

## 3) Systematic Bias: When Is It a Problem?

You're right to push back. **If co-occurring proteins are genuinely more similar, that IS biological signal.** The concern is more subtle:

**It's fine if**: The magnitude difference reflects true viral compatibility (e.g., HA and NA must co-evolve, so co-occurring pairs have coordinated mutations).

**It's a problem if**: The magnitude difference reflects **dataset construction artifacts**:
- Positives = within-isolate (same host, same time, same geography)
- Negatives = cross-isolate (potentially different host/time/geography)
- The model might learn "same host signature" rather than "protein compatibility"

**Practical test**: Look at your FP errors. If the model's false positives are cross-isolate pairs from the **same subtype and year**, those are "hard negatives" where magnitude is small but the label is 0. If the model gets these wrong at a higher rate, it's over-relying on magnitude.

Your `false_negatives_detailed.csv` and `false_positives_detailed.csv` files are perfect for this analysis.

---

## C) Random Negative Baseline

Yes, your negatives ARE cross-isolate. But they are "easy" in the sense that:
- Isolate A and Isolate B may differ in subtype, year, host
- So the magnitude of `HA_A - NA_B` is large just because the proteins come from different biological contexts

**Your FP observation is telling**: If you have more FP than FN, it means the model is **over-predicting positive** -- i.e., some cross-isolate pairs "look compatible" to the model. This could mean:
1. Proteins from similar isolates (same subtype/year) produce cross-isolate pairs with small `||diff||`
2. The model hasn't learned enough beyond magnitude

**What I was suggesting earlier** (C) is different from what you do: create negatives by pairing HA from isolate X with NA from isolate Y **where X and Y are from the same subtype/year/host**. These are "super-hard" negatives that force the model to rely on sequence-level signal rather than population-level confounders.

---

## D) Direction Analysis

Yes, exactly. The idea is:

```python
# Current: model sees both magnitude AND direction
feature = emb_a - emb_b  # shape (1280,)

# Proposed diagnostic: model sees ONLY direction
feature = (emb_a - emb_b) / (||emb_a - emb_b|| + eps)  # unit vector, shape (1280,)
```

**If performance stays similar**: the signal is in the direction (good -- biological).
**If performance drops significantly**: the model was heavily relying on magnitude (concerning).

This isn't something to deploy permanently, it's a **diagnostic experiment** to understand the source of the diff's "magic."

You could add it as `interaction: "diff_normed"` or `interaction: "unit_diff"` -- a small addition to the existing code.

---

## Task (a): Multi-Interaction PCA/UMAP Plots

### Current Architecture

`plot_pair_embeddings_splits_overlap` does everything in one monolithic function:
1. Loads pairs, samples them
2. Calls `create_pair_embeddings_concatenation(use_concat=..., use_diff=..., use_prod=...)` to build ONE feature matrix
3. Runs PCA/UMAP on that feature matrix
4. Runs extensive diagnostics (norm correlations, swap analysis, etc.)
5. Plots by split+label, by seg_pair, by func_pair

**Problem**: It builds one combined feature vector (e.g., `concat+diff`), but you want **separate** plots for each interaction mode to understand them independently.

### Proposed Approach

**New function**: `plot_pair_interactions_comparison`

```
Inputs:
  - run_dir, embeddings_file, output_dir
  - interactions: list of str = ["concat", "diff", "prod"]
  - max_per_label_per_split, random_state

Flow:
  1. Load pairs once, sample once
  2. Load raw emb_a, emb_b once (the raw embeddings, not precomputed features)
  3. For each interaction in ["concat", "diff", "prod"]:
     a. Compute feature: concat([a,b]), |a-b|, |a*b|
     b. PCA -> 2D
     c. Plot using _plot_pair_embedding_splits_2d (reuse existing helper)
     d. Print/save diagnostics (norm-vs-label, etc.)
  4. Save all diagnostics to one JSON
```

**Key design decisions:**

1. **Load raw embeddings directly** (not through `create_pair_embeddings_concatenation`): This avoids the legacy `use_concat/use_diff/use_prod` flags entirely and gives us `emb_a` and `emb_b` separately.

2. **Keep `_plot_pair_embedding_splits_2d` as the core scatter helper**: It's already well-designed with split colors, label styles, filter textbox, and smart legend placement.

3. **Where to put it**: Same file (`visualize_dataset_stats.py`) since it's a dataset-level visualization. But it replaces `plot_pair_embeddings_splits_overlap` rather than augmenting it.

4. **Deprecation path**: The new function doesn't use `use_concat/use_diff/use_prod`. The old function stays but gets called only for legacy bundles.

### What About `plot_sequence_embeddings_by_confounders_from_pairs`?

This function plots **individual** embeddings (not pairs), colored by confounders. It's relevant to task (b), not task (a). Some reusable pieces:
- Pair loading + sampling logic (same pattern)
- `_collapse_to_top_n` for category collapsing
- `plot_embeddings_by_category` utility

For task (b), this function is close to what you want but it currently:
- Focuses on a single dominant function (e.g., only HA)
- Uses UMAP with PCA fallback
- Doesn't distinguish emb_a vs emb_b on the same plot

You'd want to extend it to overlay emb_a and emb_b with different markers/colors, and that's a natural evolution of this function.

### Implementation Summary

| What | Action |
|------|--------|
| `plot_pair_embeddings_splits_overlap` | Keep for now, but stop calling from new bundles |
| New `plot_pair_interactions_comparison` | Main function for task (a); loops over interaction modes |
| `_plot_pair_embedding_splits_2d` | Reuse as-is (the scatter helper) |
| `create_pair_embeddings_concatenation` | Don't use in new function; load raw embs directly |
| Diagnostics | Compute per-interaction (norm-vs-label, PCA variance) |
| Output | `pair_pca_concat.png`, `pair_pca_diff.png`, `pair_pca_prod.png` |

### Recommendation

Add alongside the existing function (safe, no breakage). Wire it up for new bundles, while keeping the old one for backward compatibility until ready to remove it.
