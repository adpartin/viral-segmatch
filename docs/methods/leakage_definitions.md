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
| 2 | Sequence-level label **imbalance** | slot label imbalance | A sequence appears only as positive (or only as negative) within a split. | v2 coverage phase (per-`dna_hash` per slot) + protein-level safety raise; Plan Exp 1 split overlap stats; `n_dna_uncovered` in `dataset_stats.json` | ✅ Protein level (v2 coverage phase enforces ≥1 neg per `seq_hash` per slot). ✅ DNA level (option C, implemented 2026-05-08): coverage extended to per-`dna_hash` per slot, best-effort with logging. | Protein level: 0 imbalance on every dataset built. DNA level (HA/NA mixed, post-implementation): pos `n_unique` == neg `n_unique` on every `dna_hash` row; `n_dna_uncovered` = 0 across all splits. |
| 3 | Sequence-level **leakage** | slot-level leakage | Same `seq_hash` / `dna_hash` appears in different pairs across splits. | Plan Exp 1 (split overlap stats); Plan Exp 4 (seq-disjoint / strict-dedup re-train) | ❌ Not yet — Plan Exp 4 will add `seq_disjoint` and `strict_dedup` split modes. | ❌ Confirmed present (Plan Exp 1, HA/NA mixed): ~25% `seq_hash` overlap, ~10% `dna_hash` overlap, train vs val/test on slot a. |
| 4 | Cluster leakage | near-neighbor leakage | Test pair's joint feature vector is cosine-near a training pair's, even if no exact hash match. | Plan Exp 2 (1-NN baseline); Plan Exp 3 (cosine deciles); Plan Exp 4 (partial bound — handles exact-DNA case only); Plan Exp 5 (mmseqs2 cluster splits) | ❌ Not yet. | ⚠️ Suggested but not formally measured. Anchor signal: median nearest-train PB1 cosine = 0.994. Awaiting Plan Exp 2 / Exp 3 / Exp 5 for a verdict. |
| 5 | Demographic shortcut leakage | metadata shortcut leakage | Model uses `same_host`, `same_subtype`, `same_year`, etc. as proxy for "same isolate." | Level 1 / Level 2 stratified eval; `analyze_negative_hardness` (match_count, match_pattern); Plan Exp 3 cosine deciles | ❌ Not yet. Candidate mitigations: hard-negative mining, adversarial / gradient-reversal training (DANN-style), Invariant Risk Minimization (IRM). Tracked in `roadmap_v2.md` Task 12. | ❌ Confirmed present: 30–50× FP-rate climb on match_count (HA/NA mixed; both h=[10] and h=[200]). See `docs/results/2026-05-07_metadata_shortcut_negatives.md`. |

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
  v2 negative-sampling phase (planned; see
  `docs/results/2026-05-08_dna_coverage_feasibility_sweep.md`).
- Mode #3 measurement at DNA level → `dna_hash` rows of
  `split_overlap_stats.csv`.
- Mode #3 mitigation at DNA level → split-mode partitioning on
  `dna_hash` (Exp 4 in the plan).

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
Mapping back to the table above:

| Plan item | Tests |
|---|---|
| Exp 1 — Cross-split overlap stats table | Visibility for #1 and #3 |
| Exp 2 — k-NN baseline (k=1) | #4; provides the comparator for the biology criterion |
| Exp 3 — Stratified accuracy by nearest-train cosine | #4 (and indirectly #5) |
| Exp 4 — Sequence-disjoint and strict-dedup splits | #3 directly; bounds #4 |
| Exp 5 — mmseqs2 cluster-based splits | #4 at the biological-similarity level |
| Anl 1 — Conservation analysis across representations | Does NOT directly assess any mode. Characterizes how dense the cluster space is per protein and per representation, which informs how to interpret Exp 4 / Exp 5 results. |

The headline scientific test is **Exp 4** (the splits work) read
alongside **Exp 2** (the k-NN baseline). If Exp 4 splits don't crash
and the MLP beats k-NN, we have positive evidence the model learned
something beyond memorization. If Exp 4 crashes, the current
high-AUC numbers are inflated.

The "biology learning" definition above is a binary criterion based
on those two experiments together. It is not a standalone metric — it
requires running the experiments first.

---

## Conventions for new findings

- New leakage modes go into the table here first. The plan then
  references them by canonical name.
- Status changes (when a NOT-ADDRESSED mode becomes ADDRESSED): update
  the table here with the mitigation, and link to the results doc
  (`docs/results/<date>_<topic>.md`) that demonstrates it.
- The "Assessed by" column points at the experiment(s) — keep these
  pointing at the plan's section headers, not at code paths, so it
  stays stable when files move.

## See also

- `docs/plans/2026-05-07_leakage_diagnostics_plan.md` — active plan
  with experiment-by-experiment design.
- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` —
  detailed design for Plan Exp 3 (cosine-controlled splits) and Plan
  Exp 5 (mmseqs2 clusters).
- `docs/results/2026-05-07_metadata_shortcut_negatives.md` — the
  finding that triggered the formal taxonomy.
- `docs/post_hoc_analysis_design.md` — Level 1 / Level 2 stratified
  eval methodology.
