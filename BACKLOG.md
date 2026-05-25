# segmatch — Backlog

Forward-looking todos that aren't roadmap-level and don't yet warrant a
full plan doc. Bullets graduate to `docs/plans/<name>_plan.md` (Status:
IN PROGRESS) when ready to design + implement, then to
`docs/plans/done/` when complete. Conventions:

- Items are numbered per section so they can be referred to as
  "Cluster sweep #2" or "DataSAIL #1" in conversation. Numbering is
  stable: don't renumber when adding new items — append to the end.
  When an item is done, strike it through with `~~done text~~` or
  delete and bump a "DONE on YYYY-MM-DD" note inline.
- Each item names a concrete **next action**, not just an idea.
- Scope estimate inline in parens (`~5 min`, `needs design`, etc.).
- If an item starts wanting sub-headings, promote it to a plan doc.
- Triage periodically; delete dead items rather than letting them
  moulder.

See also: `roadmap_v1.md` / `roadmap_v2.md` (big-picture experimental
plan from meetings) and `docs/plans/` (active and completed plans).

---

## DataSAIL follow-ups

Bake-off paused after Phase 0 (see
`docs/plans/2026-05-19_datasail_bakeoff_plan.md`). These are the
items worth revisiting before deciding to fully retire the bake-off.

1. ~~**Compute L(π) on bicc splits at id100, id099, id095** as a shared
   yardstick metric for the paper.~~ **DONE 2026-05-24** (negative
   result). Tested via `eval_split` from the `datasail` env on three
   1000-isolate-subsample routings (cluster_disjoint aa id099,
   seq_disjoint, random). Under `similarity='mmseqs'` ratios differ
   between routings/slots but don't form a clean "more leakage ↔
   higher L(π)" pattern. Under `similarity='mmseqspp'` all six
   (routing × slot) ratios collapse to ~0.34 — exactly the partition-
   shape constant 1 − (0.8²+0.1²+0.1²) for 80/10/10. Diagnostic on
   the similarity matrix shows bimodal/zero-inflated distribution, so
   not a low-variance artifact. Full writeup:
   `docs/results/2026-05-24_datasail_lpi_results.md`. MMD took over
   as the working leakage/separation metric (see
   `docs/results/2026-05-24_mmd_per_*_results.md` and the single-
   slot sweep).
2. (Tier C, low priority) **Resolve the 1-pair-silently-missing edge
   case** from `solver/overflow.py` pre-assignment. Reproducible
   across all 13 Phase 0 configs; same pair every time. Likely a
   small bug in DataSAIL's overflow handling.
3. (Tier C, low priority) **C2 SCIP-returns-None failure** at
   K=10, ε=0.05 — wrapper crashes with
   `'NoneType' object is not subscriptable`. Either fix the wrapper
   to handle the None return gracefully, or characterize when SCIP
   fails.
4. (Tier C, low priority) **Why does I2's ILP have no objective
   function?** `src/solver/id_2d.py` calls `solve(1, constraints,
   ...)` — minimize the constant 1. This explains the unpredictable
   fold-size drift we observed but isn't documented in the paper.
   Worth a GitHub issue against `kalininalab/DataSAIL` if we ever
   re-engage.

## Single-slot routing follow-ups (post 2026-05-24)

The 2026-05-24 single-slot cluster_disjoint mode (commit `4607050`)
has been exercised only on HA-NA HA-only across id100..id095 with
aa cluster_alphabet. These items extend that footprint.

1. **NA-only sibling sweep on HA-NA at id100..id095** (slot-symmetry
   check). Predicts: NA MMD grows monotonically with id↓, HA MMD
   inherits a coupling-driven shift via the same H↔N subtype channel
   we measured at id098 (Cramér's V = 0.90). If the HA shift under
   NA-only is comparable to the NA shift under HA-only, that's strong
   evidence the coupling channel is symmetric, not HA-driven.
   Concrete next action: copy the 6 `flu_ha_na_cluster_aa_idXXX_HAonly.yaml`
   bundles to `_NAonly.yaml` variants with `single_slot: b`; build
   datasets, run aa k=3 MMD sweep + train MLP/LGBM/1-NN.
   (~3 hours total compute, similar shape to the HA-only batch.)
2. **PB2-PB1 single-slot sweep on at least one slot direction.**
   PB2-PB1 has no subtype coupling (polymerase complex co-conservation
   instead). Predicts: unconstrained-slot MMD shifts MUCH less than
   under HA-NA, because the biological coupling channel is different.
   Falsifies / supports the "subtype coupling explains the HA-NA NA
   shift" story. Feasibility pre-flight is required first (PB2-PB1
   cluster distributions differ from HA-NA). Concrete next action:
   run `single_slot_cluster_disjoint_feasibility.py` on PB2-PB1; if
   id098 is feasible, create one bundle and one sweep.
3. **CV-style fold support for cluster_disjoint single-slot.** Replace
   the current LPT-greedy bin-packing with sklearn's `GroupKFold`
   keyed on `cluster_id_{single_slot}` for the multi-fold path; keep
   LPT-greedy for the single-shot 80/10/10 case where exact ratios
   matter. ~50 lines + tests. Enables true CV (mean ± std across
   folds, both split + training stochasticity) on top of the existing
   model-seed variance.

## Algorithm-switch follow-ups (post 2026-05-22)

The 2026-05-22 switch from asymmetric easy-cluster (aa) + easy-linclust
(nt) to symmetric easy-linclust on both alphabets (see
`docs/results/2026-05-22_aa_cluster_algorithm_validation_results.md`)
invalidated aa cluster artifacts that downstream datasets and ML
results depend on. Three concrete follow-ups:

1. **Rebuild cluster_disjoint datasets under linclust aa artifacts.**
   The HA/NA aa id099 and PB2/PB1 aa id099 datasets currently in
   `data/datasets/flu/July_2025/runs/dataset_flu_*_cluster_id99_*` were
   built against the prior easy-cluster aa cluster parquets. Under
   symmetric linclust the cluster IDs are different. Need to rebuild
   at id099 (and likely id098 too — under linclust id098 is now at
   88–93% largest CC, much closer to feasibility than the prior 94–98%
   under easy-cluster). Concrete next action: bump the relevant Stage 3
   bundles and re-run `scripts/stage3_dataset.sh` against the new
   `clusters_aa/idNN/combined_cluster.parquet`. (~2 h compute + cleanup.)
2. **Re-run downstream LGBM / 1-NN / MLP comparisons on the rebuilt
   datasets.** The headline finding in
   `docs/results/2026-05-15_cluster_disjoint_nt_results.md` ("1-NN cosine
   margin ≥ LGBM at every cluster_disjoint routing") was measured against
   datasets built from the prior easy-cluster aa artifacts. May or may
   not survive the rebuild. Concrete next action: run the same 8-cell
   comparison (HA/NA × PB2/PB1 × {seq_disjoint, aa id100, aa id099, nt id100, nt id099})
   after item #1, with the linclust-era datasets. (~4 h compute on
   Lambda GPUs after #1 is done.)
## Methodology ideas — possible paper contributions

1. **BiCC improvements (boundary-sample drop / CC-splitting /
   absolute-mutation-tolerance clustering)**. Formalized 2026-05-21 in
   `docs/results/2026-05-21_bicc_pair_drop_audit.md` as directions #3,
   #4, and #5. The audit doc quantified the drop sizes needed to
   recover 80/10/10 on real bundles (~7% on PB2/PB1 aa id099, ~18%
   on HA/NA aa id095) and added the principled
   absolute-mutation-tolerance variant (`t_f = 1 − ε/L_f`). Could be
   a genuine methodological contribution. First action: **draft a
   plan doc** picking one direction (start with #5
   absolute-mutation-tolerance + #1 per-function asymmetric thresholds
   on PB2/PB1 aa as a quick-win prototype) + a small-scale
   feasibility test target.
2. **2-D embedding visualization** (PCA / UMAP) of train/val/test
   pairs with confounder overlays (host, hn_subtype, year). Inspired
   by DataSAIL Fig. 3 (which used ECFP fingerprints for small
   molecules; for protein-pair data the analog is ESM-2 embeddings
   or k-mer feature vectors) and P&M's "how representative is your
   test set" framing.

   **Inventory of existing infrastructure** (audited 2026-05-20):
   - `src/utils/dim_reduction_utils.py` provides
     `compute_pca_reduction()` and `compute_umap_reduction()`
     (umap-learn is an optional dep — installed in segmatch env).
   - `src/analysis/visualize_dataset_stats.py:950` —
     `plot_pair_embeddings_splits_overlap()` already produces a PCA
     2-D plot (always) plus a UMAP plot (when umap-learn is
     present). Supports `pre_pca_dim` to PCA-reduce before UMAP.
   - `src/analysis/visualize_dataset_stats.py:766` — `plot_kmer_pca()`
     does PCA on k-mer features specifically.
   - **No t-SNE anywhere in the codebase.** Decision: add t-SNE
     only if we determine UMAP is insufficient (UMAP is generally
     preferred for genomics — better at preserving global structure
     than t-SNE).

   **Observed gap**: for some Stage 3 bundles (e.g.,
   `dataset_flu_ha_na_regimes_ratio3_20260513_211559`), the `plots/`
   dir contains only categorical-distribution + split-composition
   PNGs — NO PCA/UMAP plots. Either the call isn't wired through
   for k-mer-feature bundles, or it's gated on a config flag.

   **First actions**:
   1. Trace the call site for `plot_pair_embeddings_splits_overlap`
      in `visualize_dataset_stats.py` to confirm why it's not
      invoked for the regimes bundle. Likely needs pair embeddings
      passed in (which kmer-bundles may not thread through).
   2. Decide which feature space(s) to project: ESM-2 (1280-dim,
      "learned biology"), k-mer-nt (4096-dim, "string statistics"),
      k-mer-aa (8000-dim), or pair-level joint features (slot_a +
      slot_b interaction).
   3. Decide on coloring scheme: split (train/val/test) + a second
      facet by confounder (host, hn_subtype, year).
   4. After choices made, either fix the existing wiring or write
      a small dedicated analysis script — depending on how invasive
      the fix is.

