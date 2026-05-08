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

| # | Canonical name | Synonyms | Description | Assessed by | Status |
|---|---|---|---|---|---|
| 1 | Same-pair **leakage** | pair-key leakage | Same `pair_key` in train and test. | v2 `pair_key` overlap assertion; Exp 1 makes this visible | ✅ ADDRESSED — v2 assertion + `forbidden_pair_keys` threading |
| 2 | Sequence-level label **imbalance** | slot label imbalance | A sequence appears only as positive (or only as negative) in train. | v2 coverage assertion + `seqs_with_zero_negatives` raise | ✅ ADDRESSED — v2 coverage phase + per-sequence raise |
| 3 | Sequence-level **leakage** | slot-level leakage | Same `seq_hash` / `dna_hash` appears in different pairs across splits. | Plan Exp 1 (split overlap stats); Plan Exp 4 (seq-disjoint / strict-dedup re-train) | ❌ NOT ADDRESSED — measured 11–16% on v2 |
| 4 | Cluster leakage | near-neighbor leakage | Test pair's joint feature vector is cosine-near a training pair's, even if no exact hash match. | Plan Exp 2 (cosine deciles); Plan Exp 3 (1-NN baseline); Plan Exp 5 (mmseqs2 cluster splits) | ❌ NOT ADDRESSED — median nearest-train PB1 cosine = 0.994 |
| 5 | Demographic shortcut leakage | metadata shortcut leakage | Model uses `same_host`, `same_subtype`, `same_year`, etc. as proxy for "same isolate." | Level 1 / Level 2 stratified eval; `analyze_negative_hardness` (match_count, match_pattern); Plan Exp 2 cosine deciles | ❌ NOT ADDRESSED — quantified 30×–50× FP-rate climb on match_count (see `docs/results/2026-05-07_metadata_shortcut_negatives.md`) |

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
| Exp 2 — Stratified accuracy by nearest-train cosine | #4 (and indirectly #5) |
| Exp 3 — k-NN baseline (k=1) | #4; provides the comparator for the biology criterion |
| Exp 4 — Sequence-disjoint and strict-dedup splits | #3 directly; bounds #4 |
| Exp 5 — mmseqs2 cluster-based splits | #4 at the biological-similarity level |
| Anl 1 — Conservation analysis across representations | Does NOT directly assess any mode. Characterizes how dense the cluster space is per protein and per representation, which informs how to interpret Exp 4 / Exp 5 results. |

The headline scientific test is **Exp 4** (the splits work) read
alongside **Exp 3** (the k-NN baseline). If Exp 4 splits don't crash
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
  detailed design for Plan Exp 2 (cosine-controlled splits) and Plan
  Exp 5 (mmseqs2 clusters).
- `docs/results/2026-05-07_metadata_shortcut_negatives.md` — the
  finding that triggered the formal taxonomy.
- `docs/post_hoc_analysis_design.md` — Level 1 / Level 2 stratified
  eval methodology.
