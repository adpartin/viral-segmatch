# Leakage definitions and the "biology learning" criterion

This is the persistent reference for the project's terminology around
dataset / evaluation issues that may inflate test metrics, and for the
operational definition of when we say the model has "learned biology"
versus memorized near-neighbors.

The active to-do list and experimental design live in
`docs/plans/2026-05-07_leakage_diagnostics_plan.md`. **This file is the
source-of-truth for *what we mean* by each term**; the plan is the
to-do list for *what we're doing about it*.

When new modes are discovered, add them to the canonical table here
first, then update the plan to reference them.

---

## Five canonical modes

Use these names in commits, plans, results writeups, and conversation.
Don't invent synonyms. If a real new mode appears, add it as row 6 in
this table and link from the plan.

| # | Canonical name | Synonyms | Description | Assessed by | Addressed | Status |
|---|---|---|---|---|---|---|
| 1 | Same-pair **leakage** | pair-key leakage | Same `pair_key` in train and test. | v2 within-split + cross-split protein-pair dedup | ✅ within-split protein-pair dedup (v2 strict mode) + cross-split protein-pair dedup (`forbidden_pair_keys` threading) | ✅ Verified zero `pair_key` overlap across splits via v2 strict-mode assertion (enforced at construction time). |
| 2 | Sequence-level label **imbalance** | slot label imbalance | A sequence appears only as positive (or only as negative) within a split. | v2 coverage phase (per-`seq_hash` and per-`dna_hash` per slot) + protein-level safety raise; Plan Exp 1 split overlap stats; `coverage.<split>.n_seqs_with_zero_negatives` in `dataset_stats.json` (protein level); `rejection_stats.n_dna_uncovered` in run logs (DNA level) | ✅ Protein level (v2 coverage phase enforces ≥1 neg per `seq_hash` per slot, hard-raise on violation at `dataset_segment_pairs_v2.py:819`). ✅ DNA level (implemented 2026-05-08): coverage extended to per-`dna_hash` per slot, best-effort with logging. | Protein level: `n_seqs_with_zero_negatives = 0` on every dataset built (verified across recent HA/NA, PB2/PB1, and `*_regimes` builds in `data/datasets/flu/July_2025/runs/`). DNA level: `n_dna_uncovered = 0` on the same builds. |
| 3 | Sequence-level **leakage** | slot-level leakage | Same `seq_hash` / `dna_hash` appears in different pairs across splits. | Plan Exp 1 (split overlap stats); Plan Exp 4 (seq-disjoint split routing) | ✅ **IMPLEMENTED 2026-05-11** via `seq_disjoint` routing (`split_strategy.mode: seq_disjoint`). Default `hash_key: seq` (protein-level — strictly stronger than DNA-level) as of 2026-05-12. Cross-split overlap on the chosen hash family is 0 by construction with zero pair drops. See `docs/plans/done/2026-05-10_seq_disjoint_routing_plan.md`. `strict_dedup` deferred — `seq_disjoint` achieves the same test with no data loss. | ✅ Eliminated by construction in current production bundles (`flu_ha_na`, `flu_pb2_pb1` both now use `seq_disjoint` mode). Pre-implementation measurement (Plan Exp 1, HA/NA random splits): ~25% `seq_hash` overlap, ~10% `dna_hash` overlap on slot a. Post-implementation finding (Exp 4a, HA/NA `hash_key=dna`, 2026-05-11): MLP `host_subtype_year` TNR drops 0.872 → 0.834 vs random split. Tighter setting (PB2/PB1 `hash_key=seq`, 2026-05-12): 1-NN baseline edges MLP on aggregate MCC (0.900 vs 0.887) — consistent with conserved proteins offering fewer truly-novel eval examples. Full results: `docs/results/2026-05-11_exp4a_seq_disjoint_results.md`. |
| 4 | Cluster leakage | near-neighbor leakage | Test pair's joint feature vector is cosine-near a training pair's, even if no exact hash match. | Plan Exp 2 (1-NN baseline); Plan Exp 3 (cosine deciles); Plan Exp 4 (bounds exact-`seq_hash` / `dna_hash` case only); Plan Exp 5 (mmseqs2 cluster splits) | ⚠️ Partially. **Exp 2 implemented** as `knn1_margin` and `knn_vote` baselines in `train_pair_baselines.py` (defaults centralized in `conf/baselines/default.yaml`). **Exp 3 implemented** as `src/analysis/exp3_cosine_deciles.py`. **Exp 4 (seq_disjoint)** bounds the exact-hash case. **Exp 5 (mmseqs2 cluster-based splits)** not yet implemented; see `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`. | ⚠️ Anchor signal: median nearest-train PB1 cosine = 0.994. Under seq_disjoint + 1-NN baseline (Exp 2 + Exp 4 combined): on PB2/PB1, 1-NN ≈ MLP (MCC 0.900 vs 0.887); on HA/NA, MLP still leads. Cluster-level result via mmseqs2 still pending. |
| 5 | Demographic shortcut leakage | metadata shortcut leakage | Model uses `same_host`, `same_subtype`, `same_year`, etc. as proxy for "same isolate." | Level 1 / Level 2 stratified eval; `analyze_negative_hardness` (match_count, match_pattern); Plan Exp 3 cosine deciles | ⚠️ Construction-time mitigation in v2: metadata-aware negative sampling under `dataset.negative_sampling.regime_targets` (**8 regimes** per `_negative_regime_sampling.REGIME_NAMES`: `none_match`, `host_only`, `subtype_only`, `year_only`, `host_subtype_only`, `host_year_only`, `subtype_year_only`, `host_subtype_year` — configurable mix). See `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md`. Other candidate mitigations (adversarial / gradient-reversal, IRM) tracked in `roadmap_v2.md` Task 12. | ❌ Confirmed present in legacy random-sampled datasets: 30–50× FP-rate climb on match_count (HA/NA mixed; both h=[10] and h=[200]). See `docs/results/2026-05-07_metadata_shortcut_negatives.md`. Re-test pending on regime-aware-built datasets. |

Quick disambiguation:

- **#2 vs #3** — both touch sequence identity. #2 is about how the
  *labels* of a single sequence are distributed (only-positive or
  only-negative is bad). #3 is about whether the *same identifier*
  appears in two splits.
- **#3 vs #4** — both touch cross-split similarity. #3 is exact match
  (same `seq_hash` / `dna_hash`). #4 is approximate match in feature
  space (high cosine, possibly different hash due to synonymous codon
  variation or close phylogenetic neighbors).
- **#5 vs #1–#4** — modes 1–4 are about train/test contamination at
  the data level. Mode 5 is about the model preferring metadata
  correlations to sequence content. It's a shortcut, not a
  contamination.

## Why `pair_key` is at the protein level

`pair_key = canonical(seq_hash_a, seq_hash_b)` enforces pair identity
at the protein level. Because all DNA encodings of a given protein
pair share one `pair_key`, this is **strictly stronger** than
DNA-level pair identity for the purposes of cross-split dedup,
within-split dedup, and negative blocking — any DNA-level violation
implies the corresponding protein-level violation, which v2 already
forbids. So we do not need a separate DNA-level pair key.

DNA-level concerns (per-feature label imbalance, cluster leakage on
k-mer features) are addressed at the **slot** level (per `dna_hash`),
not at the pair level. Specifically:

- Mode #2 fix at DNA level → per-slot per-`dna_hash` coverage in the
  v2 negative-sampling phase (**implemented** 2026-05-08; coverage
  phase iterates `(slot, dna_hash)` tuples, tracked internally as
  `rejection_stats.n_dna_uncovered`; feasibility study at
  `docs/results/2026-05-08_dna_coverage_feasibility_sweep.md`).
- Mode #3 measurement at DNA level → `dna_hash` rows of
  `split_overlap_stats.csv`. Note: even under `hash_key=seq` routing,
  the v2 saver computes both `seq_hash_overlap` and `dna_hash_overlap`
  as diagnostics (only the active family is hard-failed on).
- Mode #3 mitigation at DNA level → `split_strategy.mode: seq_disjoint`
  with `hash_key: dna` (looser than the default `seq` — synonymous-
  mutation variants of the same protein may end up in different splits).
  **Implemented** 2026-05-11.

---

## When we say "the model learned biology"

The model has learned biology-relevant signal **insofar as** its
accuracy on tasks (a) and (b) below is meaningfully higher than a 1-NN
lookup (k-NN with k=1) on the same splits and the same features:

(a) **Generalization to novel sequences** — pairs whose individual
sequences have cosine < 0.95 to all training sequences (or fall in a
test cluster disjoint from any training cluster, per Plan Exp 5 with
mmseqs2 at e.g. 95% identity).

(b) **Generalization across populations** — pairs from a held-out
demographic stratum (subtype, host, year) that does not appear in
training.

If MLP ≈ 1-NN on these splits, the model is doing nothing more than
near-neighbor lookup. If MLP > 1-NN by a meaningful margin (initial
bar: > 0.02 AUC), the MLP is learning a generalizable representation
beyond what memorization captures.

This definition deliberately avoids claiming "the model is mechanistic"
or "the model captures co-evolution." Both are stronger claims that
require interpretability work this project does not commit to.

---

## How we test it

The plan (`docs/plans/2026-05-07_leakage_diagnostics_plan.md`) lays
out five experiments (Exp N — assess a specific mode) and one
background analysis (Anl N — characterize the data, no pass/fail).
Mapping back to the table above, with current implementation status:

| Plan item | Tests | Status |
|---|---|---|
| Exp 1 — Cross-split overlap stats table | Visibility for #1 and #3 | ✅ Done; baked into v2 saver (`*_overlap_full_pairs_<side>` fields in `split_strategy_audit`). |
| Exp 2 — k-NN baseline (k=1) | #4; provides the comparator for the biology criterion | ✅ Done; `knn1_margin` and `knn_vote` baselines in `src/models/train_pair_baselines.py`, defaults in `conf/baselines/default.yaml`. |
| Exp 3 — Stratified accuracy by nearest-train cosine | #4 (and indirectly #5) | ✅ Done; `src/analysis/exp3_cosine_deciles.py`. |
| Exp 4 — Sequence-disjoint splits | #3 directly; bounds #4 at exact-hash level | ✅ Done as `seq_disjoint` routing (2026-05-11); plan moved to `docs/plans/done/2026-05-10_seq_disjoint_routing_plan.md`. Default `hash_key=seq` (tightened 2026-05-12). `strict_dedup` deferred. |
| Exp 5 — mmseqs2 cluster-based splits | #4 at the biological-similarity level | ⏳ Not implemented; design at `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`. |
| Anl 1 — Conservation analysis across representations | Does NOT directly assess any mode. Characterizes how dense the cluster space is per protein and per representation, which informs how to interpret Exp 4 / Exp 5 results. | ⏳ Pending. |

The headline scientific test is **Exp 4** (the splits work) read
alongside **Exp 2** (the k-NN baseline). Status as of 2026-05-12:

- **HA/NA** under `hash_key=dna` routing (Exp 4a, 2026-05-11): splits
  succeeded with zero pair drops; MLP `host_subtype_year` TNR drops
  modestly (0.872 → 0.834). MLP still leads 1-NN on aggregate metrics.
- **PB2/PB1** under tighter `hash_key=seq` routing (2026-05-12):
  splits succeeded (largest connected component ≈ 38% of pairs, still
  hit 80/10/10 within 0.0011%); 1-NN edges MLP on aggregate MCC
  (0.900 vs 0.887). Per the "biology learning" bar (MLP > 1-NN by
  > 0.02 AUC), this is **inside seed noise** — meaning on PB2/PB1
  we cannot currently claim the MLP has learned biology beyond what
  near-neighbor lookup gives us.

The "biology learning" definition above is a binary criterion based
on those two experiments together. We now have one positive case
(HA/NA) and one inconclusive case (PB2/PB1). Cluster leakage (mode #4
via mmseqs2 splits) remains open.

---

## See also

- `docs/plans/2026-05-07_leakage_diagnostics_plan.md` — active plan
  with experiment-by-experiment design.
- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` —
  detailed design for Plan Exp 3 (cosine-controlled splits, ✅
  implemented) and Plan Exp 5 (mmseqs2 clusters, ⏳ pending).
- `docs/plans/done/2026-05-10_seq_disjoint_routing_plan.md` — Exp 4
  implementation plan, moved to `done/` on 2026-05-11.
- `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md` —
  regime-aware negative sampler (mode #5 construction-time mitigation).
- `docs/results/2026-05-07_metadata_shortcut_negatives.md` — the
  finding that triggered the formal taxonomy (mode #5 confirmed).
- `docs/results/2026-05-08_dna_coverage_feasibility_sweep.md` — mode
  #2 DNA-level coverage feasibility study.
- `docs/results/2026-05-11_exp4a_seq_disjoint_results.md` — first
  Exp 4 results (HA/NA, `hash_key=dna`).
- `docs/post_hoc_analysis_design.md` — Level 1 / Level 2 stratified
  eval methodology.
