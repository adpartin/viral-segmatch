# Split-separation metrics for train/val/test partitions

**Status: L(π) LEG PAUSED · MMD LEG PHASE 1+2 DONE (full-corpus follow-up pending)**

Add two complementary distribution-aware metrics — DataSAIL's L(π) and
MMD on PCA-reduced pair embeddings — to the existing 1-NN cosine
margin diagnostic. Goal: a paper-grade quantitative answer to
"how separated are these splits?" that complements the per-pair 1-NN
view we already have.

Absorbs **BACKLOG DataSAIL #1** ("Compute L(π) on bicc splits at
id100, id099, id095"); cross-refs but does not absorb the
sister-plan `2026-05-20_routing_geometry_viz_plan.md` (qualitative
2-D visualization of the same routing geometry).

---

## Implementation status (2026-05-24)

**L(π) leg (Step 1) — PAUSED on Flu A HA/NA.** Wrapper, synthetic
validation, and first runs on three 1000-isolate-subsample routings
(cluster_disjoint id099, seq_disjoint, random) are done on branch
`feature/lpi-measurement`. Under `similarity='mmseqs'` the ratios
differ widely between routings and slots but the pattern doesn't match
a clean "more leakage ↔ higher L(π)" reading. Under
`similarity='mmseqspp'` all six (routing × slot) ratios collapse to
within ~1 % of each other at ~0.34, which is the partition-shape
constant for 80/10/10. A diagnostic on the mmseqspp similarity matrix
shows a bimodal/zero-inflated distribution (67 % of pairs sim = 0,
16–22 % > 0.9), so the collapse is not a low-variance artifact. Two
interpretations remain (partition leaks at mmseqspp scale, vs
mmseqspp identity scale differs from id099); neither is directly
tested. Full writeup, tables, and reproduce commands:
`docs/results/2026-05-24_datasail_lpi_results.md`.

**MMD leg (Step 2) — Phase 1 + Phase 2 done at 1000-isolate
subsample on Flu A HA/NA, per-slot (S1).** Script:
`src/analysis/mmd_per_slot.py` (runs in `segmatch` env). Phase 1
wiring sanity passed (random per-entity 50/50 splits give MMD² at
the noise floor consistent with the biased estimator's bias term).
Phase 2 with fixed σ = 1.0719 and 500-permutation p-values produced
the ordering random ≤ seq_disjoint < cluster_disjoint id099 on both
slots: cluster_disjoint id099 reaches p ≈ 0.002 on both slots
(0/500 permutations exceeded observed MMD²); seq_disjoint borderline
(p = 0.06–0.17); random not significant (p = 0.64–0.86). HA and NA
agree on ordering; HA's absolute MMD² values are roughly 3× NA's
at the fixed σ (not disentangled — could be biology or σ-tuning).
Not tested: full-corpus scale, pair-level (S2), k-mer features,
downstream generalization gap. Full writeup, tables, reproduce
commands, and explicit "what we can / cannot claim":
`docs/results/2026-05-24_mmd_per_slot_results.md`.

---

## Motivation

Currently the project has one quantitative split-separation signal:
the **1-NN cosine margin** in `docs/results/2026-05-15_cluster_disjoint_nt_results.md`
(computed by `src/analysis/similarity_leakage_aa_vs_nt.py`). For each
test pair we find its closest train pair in embedding space; the
distribution of those nearest-neighbor distances tells us whether
the model could solve test by lookup.

That's an **edge**-style measure — it catches near-clones but doesn't
tell us anything about the *distribution* of pairs in each split.
Two routings can have similar 1-NN distributions but the test pairs
might live in a different region of embedding space (subtype shift,
host shift, year shift). Reviewers will ask "are train and test
distributionally different?" and 1-NN does not directly answer.

The user-proposed framing ("centers / edges / weighted") maps onto
three established families:

| Family | What it measures | Methods | Our status |
|---|---|---|---|
| Edge — nearest-neighbor | "Does any test pair have a near-clone in train?" | 1-NN cosine distance distribution | **Have it** (similarity_leakage_aa_vs_nt.py) |
| Center — centroid | "Are the splits' average locations different?" | Centroid L2/cosine | **Skip** (uninformative on Flu A — tight family, centroids all near each other) |
| Weighted / distributional — over all points | "Are train and test drawn from the same distribution?" | MMD, sliced-Wasserstein, energy distance, L(π) | **Add two: L(π) + MMD** |

L(π) gives the literature-standard scalar leakage number (DataSAIL
paper's primary metric). MMD adds a kernel-based two-sample test
with an asymptotic p-value, paired with PCA-reduced ESM-2 / k-mer
embeddings. The two are complementary: L(π) is a sum of pairwise
*similarities* across split boundaries (operates on the cluster
graph); MMD is a kernel-distance between *distributions* in
embedding space (operates on the embeddings).

---

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Metric #1 | **L(π) via DataSAIL measurement-only mode** | Literature-standard, paper-grade. ~30–60 min compute on existing splits. |
| Metric #2 | **MMD with RBF kernel** | Most informative single scalar in the weighted/distributional family. Has a hypothesis test (Gretton et al. 2012). |
| Skip | Centroid distance | Likely uninformative on Flu A — all HA/NA pairs cluster in a tight region of ESM-2 space; centroid distances will be small regardless of routing. Optional sanity-check, not a primary metric. |
| Skip (this plan) | Sliced-Wasserstein, energy distance | Same family as MMD; pick MMD as the canonical addition. Revisit later if MMD turns out brittle to kernel-bandwidth choice. |
| Feature space for MMD | **PCA-reduced ESM-2 pair embeddings (~50 dims)** | Matches the "learned biology" view. K-mer is a secondary projection if ESM-2 MMD doesn't discriminate routings. |
| Pair encoding for MMD | **slot_a ⊕ slot_b concatenated (then PCA)** | Each pair is one point in 2×1280-dim space → PCA to 50 dims. Mirrors the model's input view. |
| Kernel bandwidth | **Median-heuristic** | Standard MMD practice (Gretton 2012); set σ to the median pairwise L2 distance in the combined sample. Deterministic. |
| Routings to compare | `random`, `seq_disjoint`, `cluster_disjoint @ aa id099`, `cluster_disjoint @ nt id099` | Matches `2026-05-20_routing_geometry_viz_plan.md` so visual + quantitative live in the same coordinate system. |
| Pair-protein scope | **HA/NA only**, then PB2/PB1 if HA/NA story lands | Same as the viz plan. |
| Output | Per-run CSV row + summary table at `results/flu/July_2025/runs/split_separation/` | Aggregator-output convention from CLAUDE.md. |

---

## File layout

```
src/analysis/
  split_separation_metrics.py     # NEW: top-level script
    compute_lpi_on_bicc_splits(...)        # wraps DataSAIL measurement-only
    compute_mmd_on_pair_embeddings(...)    # RBF MMD with median bandwidth
    main()                                  # CLI: --dataset_dir, --embedding_path, --out_dir

  aggregate_split_separation.py   # NEW: cross-routing summary
    main()  # reads per-run CSVs, emits combined table + heatmap PNG

src/analysis/datasail_bakeoff.py  # EXISTING: wrapper for DataSAIL invocation
  # Reuse the wrapper in measurement-only mode (load our cluster IDs
  # as the partition, ask DataSAIL to compute L(π) instead of solving).
  # No code changes if the wrapper already supports this mode; otherwise
  # add a small --measurement_only flag.

src/utils/dim_reduction_utils.py  # EXISTING
  # Reuse compute_pca_reduction() for the ESM-2 PCA step.

docs/plans/2026-05-22_split_separation_metrics_plan.md  # THIS FILE
docs/results/<future>_split_separation_results.md       # Created after first batch of runs
```

---

## Implementation steps

Each step is concrete and scoped. Earlier steps are prerequisites for
later ones.

### Step 1 — L(π) on existing bicc splits (~1–2 h)

1.1. Inspect `src/analysis/datasail_bakeoff.py` to confirm whether
     measurement-only mode is already supported. Likely path: the
     wrapper has a `--solve / --measure` switch, or you pass an
     existing partition file as input.
1.2. If measurement-only mode isn't already there, add a small flag
     `--measurement_only` that loads `train_pairs.csv` / `val_pairs.csv` /
     `test_pairs.csv` from an existing Stage 3 run, constructs the
     DataSAIL partition object, and calls DataSAIL's L(π) computation
     (skipping the solver).
1.3. Run on 4 routings × 1 pair (HA/NA) at id099 baseline. Then add
     id100 + id095 (the latter exposes the ratio-drift collapse from
     §9). Optionally include PB2/PB1 if HA/NA looks clean.
1.4. Write per-run L(π) into `results/flu/July_2025/runs/split_separation/lpi_<routing>_<pair>.csv`.

Expected output: a 4-row × 2-column CSV (routing, L(π) value) per
pair. Lower L(π) = better separation.

### Step 2 — MMD on PCA-reduced ESM-2 pair embeddings (~2–3 h)

2.1. Write `src/analysis/split_separation_metrics.py` skeleton with
     CLI (Hydra or argparse — match the project default; argparse is
     simpler for one-off analyses).
2.2. Load `train_pairs.csv` / `val_pairs.csv` / `test_pairs.csv` from
     a Stage 3 run. For each pair, look up ESM-2 embeddings via
     `src/utils/embedding_utils.py`. Concatenate `[slot_a, slot_b]` to
     a 2×1280-dim vector per pair.
2.3. PCA-reduce the *combined* (train ∪ val ∪ test) pair-embedding
     matrix to 50 dims using `compute_pca_reduction()`.
2.4. Compute MMD² with RBF kernel between train and test
     (and train vs val) on the PCA-reduced embeddings. Use the
     **median heuristic** for kernel bandwidth: σ = median pairwise
     L2 over a random ~10K subsample of the combined matrix.
2.5. Run the asymptotic two-sample test for the p-value (Gretton 2012,
     Theorem 12). MMD² = 0 under H₀ "same distribution"; the
     asymptotic threshold for rejecting H₀ at α = 0.05 is `4·sup K /
     sqrt(N)`.
2.6. Write MMD², bandwidth, threshold, p-value per routing × pair
     combination into per-run CSVs.

Expected output: same shape as Step 1's output. Lower MMD = closer
distributions = harder to argue for "splits are different
distributions".

### Step 3 — Aggregator + comparison (~1 h)

3.1. Write `src/analysis/aggregate_split_separation.py` to combine
     the per-routing per-pair CSVs into one summary table with rows =
     routings, columns = {L(π), MMD², MMD² p-value, 1-NN cosine
     median, 1-NN cosine p05}.
3.2. Plot a small heatmap (routing × metric) saved as
     `split_separation_summary.png`.
3.3. Cross-check: do the metrics agree on the ordering of routings by
     separation? Expected partial agreement (1-NN and MMD measure
     different things). Disagreements are the interesting cells.

### Step 4 — Writeup (~1 h)

4.1. Draft `docs/results/<date>_split_separation_results.md` with the
     summary table + the heatmap + 2–3 paragraphs of interpretation.
4.2. Mark this plan IMPLEMENTED and move to `docs/plans/done/`.

---

## Why not centroid distance (deferred)

Centroid distance was on the table in the chat discussion but
deliberately deferred:

- Cheap to compute (~minutes).
- On Flu A specifically, all HA/NA pair centroids will be close —
  ESM-2 embeds the whole family near a single mode. The signal
  from centroid distance will be drowned by within-family variance.
- If we want a "are the means different?" signal anyway, it falls
  out of the MMD computation (MMD² with a linear kernel is the
  squared centroid distance). So if MMD with linear kernel later
  turns out useful, centroid distance is a free byproduct.

Add later only if MMD-with-RBF turns out brittle and we want a
sanity-check.

---

## Open questions / out of scope

- **Per-axis MMD** (per-host, per-subtype, per-year): more sensitive
  to confounder-driven distribution shift, but multiplies the output
  table size. Not in this plan; if MMD-overall shows clear separation
  but `analyze_predictions_stratified.py` shows host/subtype-driven
  errors, this is a natural follow-up.
- **K-mer feature space**: addable as a parallel projection if ESM-2
  MMD doesn't discriminate routings on this corpus. Same code path,
  different feature loader.
- **Bootstrap CIs**: the asymptotic MMD p-value is conservative for
  small N; permutation-based p-value would be tighter. Skip for now;
  reportable as `MMD² (asymptotic p)`.
- **Negatives**: this plan operates on positives only (the bipartite
  graph that drives cluster_disjoint routing is positive-only).
  Negatives are constructed per split and don't drive separation in
  the same way. Could add an MMD on negatives later as a sanity-check.

---

## Cross-references

- BACKLOG.md DataSAIL #1 (absorbed by Step 1 of this plan).
- `docs/plans/2026-05-20_routing_geometry_viz_plan.md` (qualitative
  visualization companion to this plan's quantitative metrics).
- `docs/results/2026-05-15_cluster_disjoint_nt_results.md` § "1-NN
  cosine margin (leakage upper bound)" (the existing edge-style
  measure this plan complements).
- `docs/plans/2026-05-19_datasail_bakeoff_plan.md` (the paused
  bake-off; this plan reuses its DataSAIL wrapper in measurement-only
  mode rather than re-engaging with the solving mode).
- Gretton, A. et al. (2012). "A Kernel Two-Sample Test." JMLR 13.
  (canonical MMD reference; median-heuristic bandwidth is from
  Section 6 of the same paper.)
