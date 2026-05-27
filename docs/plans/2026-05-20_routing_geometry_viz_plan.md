# Routing-geometry visualization (k-mer, HA/NA)

**Status: IN PROGRESS**

Inspired by DataSAIL Fig. 4 (2-D, paired-input task). Goal: produce 2-D
embedding plots that visually demonstrate the progressive separation
between train / val / test as the split strategy moves from `random` →
`seq_disjoint` → `cluster_disjoint`. Three independent Stage 3 runs,
each emitting its own per-run plots; visual comparison happens by
opening the PNGs side-by-side.

This is BACKLOG.md **Methodology #2**.

---

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Feature space | **k-mer (nt, k=6)** | Matches `flu_ha_na.yaml` production default; k-mer outperforms ESM-2 on homogeneous slices (see CLAUDE.md "Key Findings"). |
| Pair (protein) scope | **HA / NA only** | One pair at a time keeps the figure honest. PB2/PB1 follow-up later if HA/NA story lands. |
| Routings to compare | `random`, `seq_disjoint`, `cluster_disjoint @ aa id 0.99` | Only `id100` and `id099` are feasible on Flu A under bilateral routing (`clustering_overview.md` §10.2); `id099` is the canonical operating point. Skipping a 4th intermediate (id098 already collapses to 93.7%). |
| Dim reduction | **TruncatedSVD + UMAP** (both per plot) | TruncatedSVD (not PCA) is the canonical reduction for sparse non-negative count features like k-mer counts (LSI/LSA convention). It does not center the data, which preserves the non-negativity that gives the features meaning. PCA is reserved for ESM-2 and other dense, signed embeddings (see `src/utils/dim_reduction_utils.py` module docstring). UMAP uses a TruncatedSVD-50 pre-step for tractability. |
| Plot types per run | **Sequence-level** (Fig. 4 analog) + **pair-level** (model's input) | Two different stories. Sequence-level shows the routing geometry directly; pair-level shows what the model sees. |
| Layout (sequence-level) | **Two subplots, HA on left, NA on right** | Mirrors DataSAIL Fig. 4's "drugs row / proteins row" pattern. HA and NA live in very different k-mer space and would segregate anyway. |
| Color encoding (split) | Train / val / test from `SPLIT_COLORS` | Project convention. |
| Marker encoding (label) | Filled (positive) / hollow (negative) — **pair-level only** | Sequence-level has no label concept. |
| Random-routing leakage rendering | **4th color "appears in multiple splits"** | Honest depiction — the figure's point is to show that leakage exists. Matches DataSAIL Fig. 4's "drop" 3rd color. |
| Output location | Per-run `plots/` dir under each `data/datasets/.../runs/<run>/plots/` | Existing convention; user inspects 3 dirs side-by-side. |

---

## File layout

```
conf/bundles/
  flu_ha_na_random.yaml            # NEW: split_strategy.mode=random
  flu_ha_na_seq_disjoint.yaml      # NEW: explicit alias (mode=seq_disjoint, hash_key=seq)
  flu_ha_na_cluster_id99.yaml      # EXISTING: cluster_disjoint @ aa id 0.99 (kept; renaming to id099 would break ~14 cross-references to its existing dataset/model runs and the 4 bundles that inherit from it)

src/utils/dim_reduction_utils.py
  compute_truncated_svd_reduction(...)  # NEW: SVD without centering, for k-mer features
  # compute_pca_reduction(...)          # EXISTING: keep, used for ESM-2

src/analysis/plot_kmer_routing_geometry.py    # NEW: two clean entry points
  plot_kmer_pair_geometry(...)     # pair-level: TruncatedSVD + UMAP, fill/hollow by label
  plot_kmer_sequence_geometry(...) # sequence-level: TruncatedSVD + UMAP, two subplots HA/NA, 4-color split + "in multiple" for random

src/analysis/visualize_dataset_stats.py
  visualize_dataset_stats(...)     # MODIFIED: replace plot_kmer_pca() call with plot_kmer_routing_geometry()
  # plot_kmer_pca + helpers          # REMOVED: superseded by the new module (was reading `cfg.kmer.{alphabet,k}` wrong via stale glob; had a latent string-vs-tuple key bug; produced only PCA)

docs/plans/2026-05-20_routing_geometry_viz_plan.md   # THIS FILE
```

The previous `plot_kmer_pca` and its private helpers
(`_plot_kmer_pair_concat`, `_plot_kmer_scree`, `_resolve_kmer_k`,
`_build_kmer_lookup_keys`, `_densify_kmer_rows`) have been removed.
The new module supersedes them: it autodetects the bundle's
`cfg.kmer.alphabet` + `cfg.kmer.k`, handles tuple lookup keys
correctly, uses TruncatedSVD instead of PCA, and adds UMAP +
sequence-level panels.

---

## Sequence-level plot details

Each unique sequence (identified by its `seq_hash`, which is the
protein-level md5) is plotted at its own 2-D coordinate. Two subplots
side-by-side: HA on the left, NA on the right.

**Color assignment** (per unique sequence):
- If sequence appears in exactly one split → that split's color (train / val / test).
- If sequence appears in multiple splits → 4th category color ("multiple", neutral gray-ish to match DataSAIL Fig. 4's "drop" category).
- `seq_disjoint` and `cluster_disjoint` routings will have zero "multiple" points by construction.
- `random` routing will show "multiple" prominently — that's the leakage story.

**Sampling**: capped at `max_sequences_per_function=5000` per protein
type (so the HA + NA total is ≤ 10,000 sequences per run). Sampling is
**stratified by split-membership color** with a guaranteed minimum
allocation (up to 500) for the leakage category — this keeps the
"appears in multiple splits" minority visible under random routing
even when it's a small fraction of the corpus.

**Projection**: TruncatedSVD / UMAP fit on the union of all HA + NA
k-mer vectors in the run, then the two subplots filter to their
respective protein type. This ensures both subplots share the same 2-D
coordinate system. UMAP runs on a TruncatedSVD-50 pre-reduction
(`umap_pre_pca_dim=50` — name kept for API back-compat though the
reducer is TruncatedSVD).

**SVD axis labels** intentionally omit per-component variance ratios.
TruncatedSVD orders components by descending singular value, not
projected variance. On uncentered count data (k-mer counts), the
first singular vector aligns with the corpus mean direction, so the
SVD1 variance ratio can be smaller than SVD2's — which would read
backwards relative to the PCA convention people expect. UMAP labels
are just "UMAP 1" / "UMAP 2" by the same logic.

---

## Pair-level plot details

One point per sampled pair (stratified up to `max_per_label_per_split=1000`).
Feature vector = `concat(kmer_a, kmer_b)`. Single plot (not split by
protein type, since each pair is HA-NA by construction).

**Color = split**, **fill = label** (filled = positive, hollow = negative).
Same convention as the existing `_plot_pair_features_splits_2d`.

**Projection**: TruncatedSVD + UMAP, fit on the union of all sampled
pairs in the run. UMAP runs on a TruncatedSVD-50 pre-reduction. SVD
axis labels intentionally omit variance ratios (see sequence-level
section above for the rationale).

---

## Phases

### Phase 1: Bundles + branch

- ✓ Branch `feature/routing-geometry-viz` cut from master.
- Two new bundle YAMLs created; reuse the existing third:
  - `conf/bundles/flu_ha_na_random.yaml` — NEW; sets `split_strategy.mode: random`.
  - `conf/bundles/flu_ha_na_seq_disjoint.yaml` — NEW; explicit alias (same routing as inherited from `flu_ha_na`).
  - `conf/bundles/flu_ha_na_cluster_id99.yaml` — EXISTING; cluster_disjoint @ aa identity 0.99. Kept under its existing name to avoid breaking cross-references in 4 dependent bundles + 11 dataset/model run dirs + 3 docs.

### Phase 2: Plotting module

Develop against an existing run dir as test fixture (e.g.,
`dataset_flu_ha_na_20260513_201046`). Module is self-contained;
reuses `compute_truncated_svd_reduction`, `compute_umap_reduction`,
plot styles.

### Phase 3: Wire-in + run

Modify `visualize_dataset_stats.py` to call the two new functions
instead of `plot_kmer_pca`. Run Stage 3 for the three new bundles
sequentially. Each emits plots into its own run dir.

Estimated compute per run: a few minutes (k-mer cache exists, pair
construction is the main cost). Three runs total ≈ 10–15 min sequential.

### Phase 4: Review + iterate

Open the 12 PNGs (3 runs × 4 output files —
`kmer_{sequence,pair}_{svd,umap}.png` per run) side-by-side. Confirm
the random → seq_disjoint → cluster_disjoint progression looks right.
Tweak plotting (e.g. point size, alpha, legend placement, layer order)
as needed.

Headline figures are the **sequence UMAP** panels (closest analog to
DataSAIL Fig. 4). The SVD panels are linear-baseline sanity checks;
the pair-level panels show what the model actually trains on.

### Phase 5: Cleanup + commit

- ✓ Removed deprecated `plot_kmer_pca` + its private helpers during the wire-in (see file layout above).
- Update `BACKLOG.md` Methodology #2 entry with the actual implementation
  + link to this plan.
- Move this plan to `docs/plans/done/` with status flipped to IMPLEMENTED.
- Single PR back to master.

---

## Reproducibility

All three reducers are deterministic when seeded:
`PCA(svd_solver='randomized', random_state=42)`,
`TruncatedSVD(algorithm='randomized', random_state=42)`,
`umap.UMAP(random_state=42)`. The new module hard-codes
`random_state=42` as the default and the orchestrator calls into it
with the same. Setting `random_state` on UMAP forces `n_jobs=1` (we
see the umap-learn warning during runs — expected, not an error). The
3 runs produced under these settings are bit-for-bit reproducible
given the same source data.

---

## Out of scope for this plan

- ESM-2 visualization (k-mer is the production default; ESM-2 plot can
  be a sibling follow-up if desired).
- PB2/PB1 (HA/NA only for now).
- Performance bars next to the t-SNEs (DataSAIL Fig. 4c/f/i analog) —
  potentially a separate plan once visualization shape is locked.
- DataSAIL-style "drop" category beyond the leakage rendering — we do
  not actually drop pairs; the bicc routing collapses instead.
