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
| 2 | Sequence-level label **imbalance** | slot label imbalance | A sequence appears only as positive (or only as negative) within a split. | v2 coverage phase (per-`seq_hash` and per-`dna_hash` per slot) + protein-level safety raise; Plan Exp 1 split overlap stats; `coverage.<split>.n_seqs_with_zero_negatives` in `dataset_stats.json` (protein level); `rejection_stats.n_dna_uncovered` in run logs (DNA level) | ✅ Protein level (v2 coverage phase enforces ≥1 neg per `seq_hash` per slot, hard-raise on violation at `dataset_segment_pairs_v2.py:819`). ✅ DNA level (implemented 2026-05-08): coverage extended to per-`dna_hash` per slot, best-effort with logging. | Protein level: `n_seqs_with_zero_negatives = 0` on every dataset audited (HA/NA, PB2/PB1, and `*_regimes` builds in `data/datasets/flu/July_2025/runs/`). DNA level: `n_dna_uncovered = 0` on the same builds. |
| 3 | Sequence-level **leakage** | slot-level leakage | Same `seq_hash` / `dna_hash` appears in different pairs across splits. | Plan Exp 1 (split overlap stats); Plan Exp 4 (seq-disjoint split routing) | ✅ **IMPLEMENTED 2026-05-11** via `seq_disjoint` routing (`split_strategy.mode: seq_disjoint`). Default `hash_key: seq` (protein-level — strictly stronger than DNA-level) as of 2026-05-12. Cross-split overlap on the chosen hash family is 0 by construction with zero pair drops. See `docs/plans/done/2026-05-10_seq_disjoint_routing_plan.md`. `strict_dedup` deferred — `seq_disjoint` achieves the same test with no data loss. | ✅ Eliminated by construction in current production bundles (`flu_ha_na`, `flu_pb2_pb1` both now use `seq_disjoint` mode). Pre-implementation measurement (Plan Exp 1, HA/NA random splits): ~25 % `seq_hash` overlap, ~10 % `dna_hash` overlap on slot a. Performance impact under seq_disjoint and per-pair details: `docs/results/2026-05-11_exp4a_seq_disjoint_results.md`. Single-slot `cluster_disjoint` (`split_strategy.single_slot='a'\|'b'`) eliminates overlap on the constrained slot only; unconstrained-slot overlap measured 3.4–7.5 % on HA-NA NA, 5.2–7.2 % on PB2-PB1 PB1 across t100..t095 (`slot_leakage_summary` in `dataset_stats.json`). |
| 4 | Cluster leakage | near-neighbor leakage | Test pair's joint feature vector is cosine-near a training pair's, even if no exact hash match. | Plan Exp 2 (1-NN baseline); Plan Exp 3 (cosine deciles); Plan Exp 4 (bounds exact-`seq_hash` / `dna_hash` case only); Plan Exp 5 (mmseqs2 cluster splits) | ✅ **Exp 2 implemented** as `knn1_margin` and `knn_vote` baselines. **Exp 3 implemented** as `src/analysis/exp3_cosine_deciles.py`. **Exp 4 (seq_disjoint)** bounds the exact-hash case. **Exp 5 (mmseqs2 cluster splits) implemented 2026-05-15** via `dataset.split_strategy.mode: cluster_disjoint` + `cluster_alphabet: aa\|nt`. See `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` § B (aa) and § B-nt (CDS DNA). | ⚠️ Confirmed real on the production seq_disjoint baseline (aa cluster_id099 LGBM drops materially vs seq_disjoint). Nt cluster_disjoint hits the same feasibility ceiling as aa on Flu A (only t100 / t099 operable bilateral — see `splits.md` § 1.9). 1-NN cosine margin ≥ LGBM at every cluster_disjoint routing — cluster_disjoint weakens the near-neighbor signal gradually rather than absolutely. Single-slot cluster_disjoint extends the operable threshold range past the bilateral ceiling; biological coupling (Cramér's V V(HA × NA) ≈ 0.85 → 0.53; V(PB2 × PB1) ≈ 0.76 → 0.41 across t100 → t095) governs how much the unconstrained slot follows the constrained one. Per-pair performance trajectories: `docs/results/2026-05-15_cluster_disjoint_nt_results.md`, `docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md`, `docs/results/2026-05-26_pb2_pb1_PB2only_idXX_sweep.md`, `docs/results/2026-05-28_HAonly_extended_idXX_sweep.md`. |
| 5 | Demographic shortcut leakage | metadata shortcut leakage | Model uses `same_host`, `same_subtype`, `same_year`, etc. as proxy for "same isolate." | Level 1 / Level 2 stratified eval; `analyze_negative_hardness` (match_count, match_pattern); Plan Exp 3 cosine deciles | ⚠️ Construction-time mitigation in v2: metadata-aware negative sampling under `dataset.negative_sampling.regime_targets` (**8 regimes** per `_negative_regime_sampling.REGIME_NAMES`: `none_match`, `host_only`, `subtype_only`, `year_only`, `host_subtype_only`, `host_year_only`, `subtype_year_only`, `host_subtype_year` — configurable mix). See `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md`. Other candidate mitigations (adversarial / gradient-reversal, IRM) tracked in `roadmap_v2.md` Task 12. | ❌ Confirmed present in legacy random-sampled datasets (large FP-rate climb on match_count). See `docs/results/2026-05-07_metadata_shortcut_negatives.md`. Re-test pending on regime-aware-built datasets. |

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
  v2 negative-sampling phase. The coverage phase iterates
  `(slot, dna_hash)` tuples, tracked internally as
  `rejection_stats.n_dna_uncovered`; feasibility study at
  `docs/results/2026-05-08_dna_coverage_feasibility_sweep.md`.
- Mode #3 measurement at DNA level → `dna_hash` rows of
  `split_overlap_stats.csv`. Note: even under `hash_key=seq` routing,
  the v2 saver computes both `seq_hash_overlap` and `dna_hash_overlap`
  as diagnostics (only the active family is hard-failed on).
- Mode #3 mitigation at DNA level → `split_strategy.mode: seq_disjoint`
  with `hash_key: dna` (looser than the default `seq` — synonymous-
  mutation variants of the same protein may end up in different splits).

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

The leakage diagnostics plan
(`docs/plans/2026-05-07_leakage_diagnostics_plan.md`) lays out five
experiments (Exp N — assess a specific mode) and one background
analysis (Anl N — characterize the data). The "Addressed" column of
the canonical-modes table above records which experiments now exist
in code and which routings they unlocked; the "Status" column records
the current empirical reading. Performance trajectories per pair,
alphabet, and routing live in `docs/results/`. The split-method
mechanics — atoms, LPT-greedy, k-fold, feasibility — are in
`splits.md`. The 1-NN comparator definition for the biology-learning
criterion is in the section above.

---

## Relation to prior-art split taxonomies

The segmatch leakage modes (#1 same-pair, #2 sequence-label imbalance,
#3 sequence-level, #4 cluster, #5 demographic-shortcut) and the
corresponding mitigations (`pair_key` dedup, `seq_disjoint`,
`cluster_disjoint`, `regime_aware_coverage`) overlap with — but are
not identical to — two prior-art taxonomies in the data-leakage
literature. The differences matter for paper writeups and for
tooling interop.

### Park & Marcotte 2012 (Nat. Methods Correspondence)

P&M characterize *test pairs* (post-hoc) by component overlap with
the training set, after a CD-HIT 40% redundancy-reduction
preprocessing pass:
- **C1**: test pair shares both components with train.
- **C2**: test pair shares one component with train.
- **C3**: test pair shares neither component with train.

A typical random CV produces ~99% C1 test pairs (P&M main, p2). In
the HIPPIE PPI population, C1/C2/C3 represent 19.2% / 49.2% / 31.6%
of human protein pairs — i.e., random CV is unrepresentative of the
population. P&M is a *diagnostic* framework: it labels each test
pair by class but doesn't prescribe a splitting algorithm.

### DataSAIL (Joeres et al., Nat. Commun. 2025) — OOD framing

DataSAIL frames the leakage question as **out-of-distribution
generalization**: a "leaky" split is one whose test pairs are too
close to its training pairs along the relevant similarity axis,
inflating reported metrics. The mitigation is to *maximize* the
similarity gap between train and test — make test pairs OOD relative
to train. The leakage-diagnostic framing complements P&M's
test-pair-composition framing: P&M labels the test set's structural
composition; DataSAIL prescribes that composition should be C3-like
(no shared components / no high-similarity bridges) for the OOD test
to be meaningful.

DataSAIL's algorithmic *recipes* (R / I1 / I2 / S1 / S2), the
segmatch ↔ DataSAIL ↔ P&M cross-reference table, and the segmatch
naming convention (BiCC-Split, bipartite-CC LPT-greedy) all live in
`splits.md` § 3 — they belong with the split-method implementations,
not with the vocabulary.

## See also

- `docs/methods/clusters.md` — clustering mechanics; prerequisite for
  understanding how mode #4 (cluster leakage) is operationalized.
- `docs/methods/splits.md` — all split-method implementations
  (atoms, LPT-greedy, k-fold, feasibility, prior-art cross-reference).
- `docs/plans/2026-05-07_leakage_diagnostics_plan.md` — active plan
  with experiment-by-experiment design.
