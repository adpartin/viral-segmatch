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
| 4 | Cluster leakage | near-neighbor leakage | Test pair's joint feature vector is cosine-near a training pair's, even if no exact hash match. | Plan Exp 2 (1-NN baseline); Plan Exp 3 (cosine deciles); Plan Exp 4 (bounds exact-`seq_hash` / `dna_hash` case only); Plan Exp 5 (mmseqs2 cluster splits) | ✅ **Exp 2 implemented** as `knn1_margin` and `knn_vote` baselines. **Exp 3 implemented** as `src/analysis/exp3_cosine_deciles.py`. **Exp 4 (seq_disjoint)** bounds the exact-hash case. **Exp 5 (mmseqs2 cluster splits) implemented 2026-05-15** via `dataset.split_strategy.mode: cluster_disjoint` + `cluster_alphabet: aa\|nt`. See `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` § B (aa) and § B-nt (CDS DNA). | ⚠️ Aa cluster_id099 LGBM drops F1 −27 pp on HA/NA and −17 pp on PB2/PB1 vs seq_disjoint — confirms cluster leakage is real and large on the production seq_disjoint baseline. Nt cluster_disjoint hits the same feasibility ceiling as aa (only id100/id099 operable on Flu A — see § "Routing equivalence" below). 1-NN cosine margin >= LGBM at every cluster_disjoint routing (gap widest at aa id099, +16/+7 pp F1 on HA/NA / PB2/PB1): cluster_disjoint weakens the near-neighbor signal gradually, not absolutely. See `docs/results/2026-05-15_cluster_disjoint_nt_results.md`. |
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

## Routing equivalence and mmseqs argument semantics

Four routing modes are implemented for mitigating modes #3 and #4; their
overlap is non-obvious. This section is the reference.

### What's the same across all routings

All four modes partition pairs by **bipartite connected components**
on a (slot_a_key, slot_b_key) graph and bin-pack components into
train / val / test using **LPT-greedy** (longest-processing-time-first
deficit-fill). The only thing that varies is which `key` each slot maps
to.

### What's different across routings

| Mode + alphabet | Slot key per pair | Two pairs end up in the same component iff they share … | Stricter than |
|---|---|---|---|
| `seq_disjoint` `hash_key=seq` (default) | `seq_hash = md5(prot_seq)` exact | identical protein sequence on a slot | — (default) |
| `seq_disjoint` `hash_key=dna` | `dna_hash = md5(contig.dna)` exact | identical full contig (5′ UTR + CDS + intron + 3′ UTR) on a slot | seq_disjoint seq, on Flu A (more unique contigs than unique proteins) |
| `cluster_disjoint` `cluster_alphabet=aa` | mmseqs2 aa cluster id at chosen identity threshold | aa-similar protein on a slot (definition of "similar" set by threshold + coverage) | seq_disjoint seq at threshold = 1.00 |
| `cluster_disjoint` `cluster_alphabet=nt` | mmseqs2 nt cluster id at chosen identity threshold, keyed on `cds_dna_hash` | nt-similar CDS DNA on a slot (UTRs and introns excluded from the comparison) | seq_disjoint seq at threshold = 1.00, but **not** a subset of seq_disjoint dna |

### Equivalences and non-equivalences

**`aa cluster_id100` ≈ `seq_disjoint hash_key=seq`** — almost identical
on Flu A. Both partition on the protein, both require exact match.
Differences come from mmseqs internals only: the `-c 0.8 --cov-mode 0`
coverage rule may merge a longer protein and a fragment that's
covered at ≥80% by the shorter at 100% identity; an md5 hash would
not. On Flu A the length variation within a function is tiny
(std ≤ 2.8 aa per `gto_format_reference.md` §6.5), the X-fraction is
scrubbed by Stage 1, so empirically the two produce essentially the
same partition. Aa id100 component count ≈ seq_disjoint seq component
count.

**`nt cluster_id100` ≠ `seq_disjoint hash_key=dna`** — they cluster
**different sequences**. seq_disjoint dna keys on `dna_hash =
md5(contig.dna)` — the full contig including 5′ UTR + CDS + intron +
3′ UTR. nt cluster_id100 keys on `cds_dna_hash = md5(cds_dna)` — just
the spliced CDS, UTRs and introns excluded. Concrete consequences:

- Two isolates with **identical CDS but different UTRs** → seq_disjoint
  dna puts them in different components (different contig hash);
  nt id100 puts them in the same cluster (same CDS hash).
- For **spliced segments** (M2, NEP), the contig hash includes the
  intron and is sensitive to intron length / sequence variation;
  cds_dna_hash skips the intron entirely. So a synonymous variation
  in the intron flips the contig hash but not the CDS hash.

Neither is a subset of the other on the partition order; they enforce
different leakage definitions.

### mmseqs argument semantics — what each flag actually does

The cluster_disjoint path invokes mmseqs2 via
`src/utils/clustering_utils.py::run_mmseqs_easy_cluster`. The default
production flags are:

```
mmseqs <easy-cluster | easy-linclust> <fasta> <out_prefix> <tmp> \
    --min-seq-id <t> -c 0.8 --cov-mode 0 [--dbtype 2]
```

Mechanic by flag:

- **`--min-seq-id <t>`** — sequence identity threshold, in [0, 1].
  Counted on residues: aa positions for proteins, nt positions for DNA.
  At `t = 1.0`, only exact-identity matches cluster (modulo coverage
  and ambiguity handling, below). At `t = 0.95`, at most ~5% of
  positions may mismatch — for a 759-aa PB2 protein, ~38 aa mutations
  allowed inside one cluster; for a 2280-nt PB2 CDS, ~114 nt mutations.
  Same rule, alphabet-independent — just count residues.
- **`-c 0.8`** — coverage threshold. Mutually applied with
  `--cov-mode 0` (bidirectional): both sequences must cover ≥80% of
  each other. So a 200-residue fragment cannot cluster with a 800-
  residue full sequence at id100 even if every position aligns,
  because the long sequence only has 25% covered. This is the same
  rule for aa and nt; only the unit (aa positions vs nt positions)
  changes.
- **`--cov-mode 0`** — bidirectional coverage, as above. Other
  cov-modes (1 = target-only, 2 = query-only) would let
  fragment-against-full clusters through. We use 0 throughout.
- **`--dbtype <int>`** — alphabet declaration passed to `createdb`
  under the hood. **1 = amino acid** (default; appropriate for
  aa cluster_disjoint, no explicit flag needed). **2 = nucleotide**
  (explicitly passed for nt cluster_disjoint). This is the only flag
  that genuinely differs between aa and nt; everything above applies
  to both.
- **Ambiguity codes** — alphabet-specific in what counts as ambiguous,
  but consistently handled. Aa: `X` is the ambiguous residue. Nt:
  IUPAC codes `N, R, Y, S, W, K, M, B, D, H, V`. mmseqs scores both
  against its substitution matrix (`blosum62` for aa,
  `nucleotide.out` for nt) — `--alph-size aa:21,nucl:5` is mmseqs's
  internal alphabet size assumption and isn't user-tunable. Stage 1's
  `prepare_sequences_for_esm2` keeps X-fraction below 10% on aa; nt
  IUPAC codes flow through untouched (mmseqs handles them natively).
- **`--search-type`** — *not* a flag on `easy-cluster` / `easy-linclust`
  in mmseqs 18; only on `search`. The original B-nt plan referenced it
  but the current code uses `--dbtype 2` instead. Passing `--search-type
  3` to `easy-cluster` raises "Unrecognized parameter" in mmseqs 18.

### easy-cluster vs easy-linclust

`run_mmseqs_easy_cluster` takes an `algorithm` parameter:

- **`cluster`** (default) — `mmseqs easy-cluster`, the sensitive
  cascaded path. Uses a k-mer prefilter, alignment, and reassignment.
  Best sensitivity at lower identity thresholds; slow on long
  sequences (the prefilter dominates).
- **`linclust`** — `mmseqs easy-linclust`, linear-time clustering.
  Less sensitive but much faster on long sequences. For the nt CDS
  sweep on the full Flu A corpus, linclust is ~8× faster than
  easy-cluster and produces within-noise different cluster counts at
  our thresholds. Used as the default for the nt redundancy sweep
  (`docs/results/2026-05-15_protein_redundancy_per_function_nt.md`).

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
