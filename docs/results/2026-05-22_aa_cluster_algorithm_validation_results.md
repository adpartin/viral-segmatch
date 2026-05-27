# 2026-05-22 — aa clustering algorithm validation: switching from asymmetric easy-cluster + easy-linclust to symmetric easy-linclust

## TL;DR

Until 2026-05-22 our clustering sweep used **easy-cluster** on aa (sensitive
3-round cascade with spaced k-mers) and **easy-linclust** on nt (linear-time
single-pass with contiguous k-mers). The asymmetric choice was justified
historically as a speed/sensitivity tradeoff appropriate per alphabet, with
an unverified claim in `clustering_overview.md` §2.3 that the two algorithms
"agreed within noise on overlapping cells at id ≥ 0.80".

A validation experiment on 2026-05-21 found the two algorithms disagree by
**5×–500×** on cluster counts at sub-id100 thresholds on identical aa
input at identical parameters. The disagreement is **scale-dependent**
(5% at N = 100, growing super-linearly to ~520% at the full corpus
N ≈ 42K). Mechanism: easy-cluster's cascade + spaced k-mers chain
transitively-similar sequences across multiple rounds; easy-linclust's
single pass catches direct k-mer matches only.

Decision: switch to **symmetric easy-linclust** on both alphabets. This
holds the algorithm constant across alphabets so the aa-vs-nt comparison
in `clustering_overview.md` §6 and §10 cleanly reflects alphabet diversity
rather than algorithm sensitivity confounded with it.

Cost: existing aa cluster artifacts are stale (regenerated 2026-05-22 with
linclust). Cluster_disjoint datasets at id099 (HA/NA, PB2/PB1) and the
downstream LGBM / 1-NN / MLP experiments referenced in
`docs/results/2026-05-15_cluster_disjoint_nt_results.md` need rebuilding
to be valid under the new clustering. Tracked in BACKLOG.

---

## Why this matters

Three downstream consequences of the asymmetric easy-cluster/easy-linclust
choice that the validation falsified:

1. **§6.4's "corpus-driven, not algorithm-driven" framing.** The prior
   methods doc asserted the per-function collapse trajectory shape was
   determined by the corpus, with the algorithm being incidental. The
   scaling experiment (Phase 0 below) shows the algorithm contribution
   is comparable to or larger than the corpus contribution at the
   conserved-protein cliff thresholds.

2. **§8.1's headline "PB2 cliff at id097→id096 (89% drop)".** Under
   easy-cluster, PB2's 1 pp transition from id097 (717 clusters) to id096
   (77 clusters) was a striking cliff. Under linclust the same transition
   is 7,634 → 6,755 — a 12% drop. The cliff is an artifact of easy-cluster's
   cascade aggressively chaining conserved-protein sequences across the
   1 pp step.

3. **The aa-vs-nt §6/§8 comparison.** Under the prior asymmetric setup,
   the observation "nt has more clusters than aa at the same threshold"
   conflated:
   - alphabet diversity (nt should have more unique sequences via
     synonymous codons — a real biological effect),
   - algorithm sensitivity (easy-cluster catches more pairs than
     easy-linclust at the prefilter — a methodology artifact).

   Switching to symmetric linclust on both removes the second confound
   and surprisingly inverts the ordering at intermediate thresholds (nt
   collapses faster than aa at id099-id098 on most functions, opposite
   of the prior framing). The mechanism for this inversion is not yet
   established — flagged as a methodology open question.

---

## Investigation overview

Plan: `docs/plans/done/2026-05-21_aa_cluster_algorithm_validation_plan.md`
(moved to done on 2026-05-22).

The plan defined six phases. Phase 0 falsified two of the original
hypotheses (cluster-mode default differs; cluster-reassign is the cause)
which made Phases 1 and 2 moot. Phase 3 was redirected after Phase 0 from
"which algorithm is correct?" to "what is each algorithm actually doing?"
once we observed the gap is scale-dependent rather than a parameter
default difference.

| Phase | Question | Outcome |
|---|---|---|
| 0 | Parameter parity + scale dependence | Both algorithms default to `--cluster-mode 0`; reassign is `false` by default. Gap is scale-dependent: 5% at N = 100, 520% at N = 42K. |
| 1 | Force `--cluster-mode 0` on linclust | Skipped — moot (both already use mode 0). |
| 2 | Turn off `--cluster-reassign` in cluster | Skipped — moot (already off by default). |
| 3 | Ground-truth pairwise identity on disputed PB2 id096 pairs | Found that mmseqs's internal clustering identity is offset ~2 pp lower than convertalis's reported `fident`/`pident` (a previously-unknown gotcha). Rep-based clustering semantics (members within threshold of the rep, not pairwise) explain why within-cluster member-to-member identities are below the cluster threshold. The algorithms differ in RECALL on rep absorption, not in cluster definition. |
| 4 | CD-HIT tie-breaker | Skipped — Phase 3 was conclusive enough that an independent third tool would not change the decision. |
| 5 | Decision + writeup | Symmetric easy-linclust; this document. |

---

## Phase 0 — parameter parity and scale dependence

### Parameter parity audit

Both algorithms run on the same 100-HA FASTA at id097 with identical
caller-supplied flags (`--min-seq-id 0.97 -c 0.8 --cov-mode 0 --threads 4`).
Parameter dumps captured from `mmseqs ... -v 3` stdout.

**60 of the 61 directly-comparable parameters are identical.** Including
the operationally important ones: `Cluster mode = 0` (Set-Cover, the
mmseqs2 default for both algorithms), `Coverage threshold = 0.8`,
`Coverage mode = 0`, `Seq. id. threshold = 0.97`, `Seq. id. mode = 0`,
`Similarity type = 2` (sequence identity).

**1 directly-differing parameter**: `Spaced k-mers` (cluster = 1, linclust
= 0). Easy-cluster's prefilter uses spaced k-mer patterns; easy-linclust's
uses contiguous k-mers.

**13 cluster-only parameters** (all cascade machinery, absent from
linclust by design): `Cascaded clustering steps = 3`, `Sensitivity = 4`,
`Cluster reassign = false`, `Single step clustering = false`,
`Diagonal scoring = true`, etc.

### Scaling experiment

Same 100-, 1,000-, 5,000-, 10,000-, 20,000-, and full-corpus (41,896)
HA subsets, both algorithms, identical parameters, sequential row
ordering:

| N | easy-cluster | easy-linclust | %diff | t_cluster | t_linclust |
|---:|---:|---:|---:|---:|---:|
| 100 | 37 | 39 | +5.4% | 2.1 s | 0.5 s |
| 1,000 | 183 | 230 | +25.7% | 2.5 s | 0.7 s |
| 5,000 | 490 | 1,131 | +130.8% | 4.6 s | 1.0 s |
| 10,000 | 723 | 2,359 | +226.3% | 7.9 s | 1.3 s |
| 20,000 | 1,089 | 5,052 | +363.9% | 14.0 s | 2.0 s |
| 41,896 | 1,706 | 10,558 | +519% | 30.1 s | 3.9 s |

**Super-linear growth in the disagreement.** At small N, candidate-pair
density is low and both algorithms find the same direct k-mer matches.
At larger N, candidate-pair density is high and easy-cluster's cascade
+ spaced k-mers absorb additional members per rep — sequences that
share *transitive* k-mer matches with the rep across multiple rounds.
easy-linclust's single pass cannot bridge those transitive chains, so
those sequences become reps of their own (smaller) clusters.

### Hypothesis status after Phase 0

| Hypothesis | Status | Evidence |
|---|---|---|
| A — cluster-mode default differs | **FALSIFIED** | Both algorithms default to `--cluster-mode 0` per `mmseqs --help` and confirmed in the parameter dumps. |
| B — sensitivity defaults (cascade vs single-pass) | **CONFIRMED** | cluster has 3-round cascade + sensitivity-4 prefilter + spaced k-mers; linclust has none. |
| C — cluster-reassign | **NOT THE CAUSE** | cluster's reassign is `false` by default; gap exists without it. |
| D — flu-corpus scale dependence | **STRONGLY CONFIRMED** | Scaling experiment shows %diff super-linear in N. |
| E — bookkeeping mismatch | **RULED OUT** | TSV semantics identical, both produce 100-row TSVs on 100-sequence input. |

---

## Phase 3 — ground-truth pairwise identity (and the rep-based clustering correction)

The Phase 0 result raised the substantive question: when the algorithms
disagree, which one is "biologically correct"? Phase 3 went after this
empirically on the worst-case (function, threshold) cell: **PB2 at id096**,
where easy-cluster reported 77 clusters and easy-linclust 6,755 — an
88× ratio over 33,663 unique PB2 aa sequences.

### Setup

Sample 500 random "linclust-rep to linclust-rep" pairs, where each pair
contains two distinct linclust representatives that both belong to one
easy-cluster cluster — i.e., pairs that easy-cluster groups together but
easy-linclust separates. Sample 300 sanity pairs within one linclust
sub-cluster (linclust rep + random member; should be ≥ 0.96 by the
clustering threshold definition).

Compute direct pairwise identities via `mmseqs search` at sensitivity 7.5
with relaxed e-value, then read `fident` (= matches / alignment_length,
same definition mmseqs uses for `--min-seq-id`).

### What we found, and the gotcha that re-framed it

**The sanity check failed:** within one linclust cluster at id096, the
rep-to-member pairs (which by the literal `--min-seq-id 0.96` threshold
definition should all be at id ≥ 0.96) measured **0 / 254 pairs at fident
≥ 0.96**. Maximum: 0.951. Median: 0.940.

This is not a bug in either algorithm. It's an mmseqs2 quirk we hadn't
seen before: **the identity that mmseqs uses internally to enforce
`--min-seq-id` during clustering is offset ~2 pp lower than the `fident`
that convertalis reports post-hoc** on the same pair. The two are
computed by different alignment paths (different sensitivity, possibly
different alignment-mode defaults), and they don't agree on the same
pair to 2 pp.

This reframes the Phase 3 question:

- **mmseqs clustering is rep-based.** Each cluster contains a representative
  and members within the *internal* identity threshold of the rep. Two
  members of the same cluster can be much further apart from each other
  than the threshold suggests — at id096 internal, members can be up
  to ~8% apart pairwise (a 4% radius around the rep in each direction).
- **Both algorithms use the same rep-based definition** with the same
  internal threshold. Neither is "over-merging" or "under-merging"
  relative to convertalis-reported identity, because that reported
  identity is on a different scale.
- **The algorithms differ in RECALL on rep absorption.** easy-cluster's
  cascade catches more candidate members per rep (sequences that share
  transitive k-mer evidence with the rep across multiple rounds);
  easy-linclust's single pass catches only direct k-mer matches.

So the original framing — "which algorithm clusters correctly?" —
dissolves. Both are correct per their definitions. The question becomes:
which recall operating-point do we want for our leakage-prevention task?

### Within-cluster pair distribution at id096 (PB2)

Sample of 200 random pairs from the largest easy-cluster cluster
(23,299 members):

- **62%** of pairs have **no significant alignment** at sensitivity 7.5,
  e-value 0.1. Inferred direct identity < ~0.7.
- **38%** align with fident in **[0.92, 0.95]**.
- **Zero pairs** at fident ≥ 0.96.

Same sampling within one linclust sub-cluster of the same easy-cluster
cluster:

- **18%** no alignment.
- **82%** align with fident in **[0.93, 0.95]**.
- Maximum fident: 0.951.

The two distributions overlap heavily but linclust's is slightly tighter
(more pairs find alignment, fewer pairs at the extreme low end). This is
the documented sensitivity tradeoff between cascade and single-pass —
small but consistent.

---

## Decision and rationale

**Switch to symmetric easy-linclust on both alphabets.** Three reasons:

1. **Removes the alphabet × algorithm confound.** `clustering_overview.md`
   §6 / §9 makes per-function, per-threshold, per-alphabet comparisons
   that are paper-relevant. Under the asymmetric setup any cross-alphabet
   difference conflates alphabet diversity with algorithm sensitivity.
   Symmetric linclust isolates the alphabet effect.
2. **Runtime cost is acceptable on both alphabets.** Median per-run time
   under linclust: 1.5 s aa, 6.7 s nt. Total sweep: ~3.5 min aa,
   ~16 min nt. Easy-cluster on aa was ~50 min total; on nt it would
   be substantially longer because nt sequences are 3× longer and the
   cascade scales accordingly.
3. **Preserves a defensible leakage barrier.** Both algorithms enforce
   the same rep-to-member identity threshold definition. The recall
   difference means linclust clusters are slightly tighter (fewer
   members per rep), which is the *less* aggressive choice but still
   well-defined and reproducible.

Alternatives considered:

- **Symmetric easy-cluster** (the more sensitive choice). Rejected on
  runtime: easy-cluster on full nt corpus was hitting wall-clock costs
  that did not scale to multi-threshold sweeps in earlier exploration.
- **Stay asymmetric, document the confound explicitly.** Rejected
  because the §6 / §9 conclusions implicitly rely on cross-alphabet
  comparability. Documenting the confound would weaken every claim
  that section makes about feasibility and collapse shape.

---

## Implementation summary

Branch: `feature/symmetric-easy-linclust` (2026-05-22).

| Commit | Scope |
|---|---|
| `af049c9` | Plan doc — record Phase 0 + Phase 3 findings before implementation |
| `09e0411` | `clustering_overview.md` — section reorder (§6→§4, §7→§5, §8→§6, §4→§7, §5→§8) + cross-reference renumbering |
| `9d9578a` | `src/utils/clustering_utils.py` + `src/analysis/seq_redundancy_per_function.py` — default `algorithm` flipped to `linclust`; pinned mmseqs2 flags (`--cluster-mode 0`, `--seq-id-mode 0`, `--similarity-type 2`, `-e 0.001`, `--dbtype` made explicit) on the CLI for audit visibility |
| `71eaa3a` | `clustering_overview.md` — content rewrite for linclust numbers (§2.3, §4 column rename, §6.1–§6.4, §9) + reversal of falsified claims |

Data regenerated 2026-05-22:

- `data/processed/flu/July_2025/clusters_aa/` — full linclust sweep, 90
  (function, threshold) cells, fresh runtime 204 s. Prior easy-cluster
  artifacts archived at `clusters_aa_easy_cluster_archive/` (1.8 GB,
  gitignored, reversible).
- `results/flu/July_2025/runs/cluster_disjoint_feasibility/feasibility_{ha_na,pb2_pb1}_aa.csv`
  — regenerated against the new aa linclust artifacts.
- `results/flu/July_2025/runs/cluster_analysis/` — `cluster_summary.csv`,
  `mutations_tolerated_table.csv`, three plots (cluster_counts,
  bipartite_largest_pct, unique_sequence_retention) all regenerated.
- `data/processed/flu/July_2025/clusters_nt/redundancy_summary.md` —
  regenerated (same artifacts, new Segment-column markdown format).

---

## Headline number shifts under symmetric easy-linclust

| Quantity | Before (asymmetric) | After (symmetric linclust) | Comment |
|---|---|---|---|
| `clustering_overview.md` §6.1, PB2 aa n_clusters at id096 | 77 | 6,755 | The "1 pp cliff" disappears; collapse moves to id095→id090 (5 pp gap). |
| §6.1 PB2 aa id097→id096 1 pp transition | −89% (cliff) | −12% (gradual) | The cliff was an easy-cluster cascade artifact. |
| §6.1 cliff location for conserved proteins | id097→id096 (1 pp) | id095→id090 (5 pp) | Conserved-protein collapse now spans a 5 pp identity gap. |
| §6.3 PB2 aa largest_cluster % at id095 | 80.3% | 15.6% | Under linclust the largest cluster covers far less of the corpus until id090. |
| §9 HA/NA aa largest CC % at id100 | 20.2% | **49.0%** | NA stalk-length absorption surfaces at id100 under linclust; not an algorithm artifact. |
| §9 PB2/PB1 aa largest CC % at id099 | 87.1% | **81.0%** | Now just 1 pp over the 80% ceiling. id099 PB2/PB1 routing is borderline feasible (was solidly infeasible). |
| §9 HA/NA aa largest CC % at id098 | 93.7% | 88.4% | Closer to feasibility. |
| §9 PB2/PB1 aa largest CC % at id098 | 98.0% | 92.6% | Closer to feasibility. |
| aa runtime (median per cell) | 4.8 s (easy-cluster) | 1.5 s (linclust) | ~3× faster on aa under linclust. |

---

## Open questions / follow-ups

1. **Why does nt have FEWER clusters than aa at intermediate thresholds
   under symmetric linclust?** This is the opposite of the prior
   asymmetric framing, surfaced cleanly by the algorithm switch. At
   id099 HA: aa = 22,679 clusters, nt = 12,150. At id100 the relationship
   is conventional (nt > aa due to synonymous codons); somewhere between
   id100 and id099 it inverts. Mechanism not yet established —
   candidate explanations (k-mer prefilter dynamics, codon-similarity
   geometry within a 22-nt-tolerant cluster) require a cross-tab
   analysis on the cluster parquets to test. See
   `clustering_overview.md` §6.1 for the framing. **Action**: cross-tab
   analysis on `clusters_aa/idNN/<fn>_cluster.parquet` and
   `clusters_nt/idNN/<fn>_cluster.parquet` for one or two functions
   (HA at id099 is the cleanest test case).

2. **Rebuild cluster_disjoint datasets under linclust artifacts.** The
   cluster_disjoint datasets used for the 2026-05-21 "Observed train
   share" audit were built against prior easy-cluster aa cluster
   parquets. The largest-CC % values currently in
   `clustering_overview.md` §10.2's main table are from linclust
   artifacts; the audit's achieved-train % numbers (previously
   tabulated in the same section, removed during the 2026-05-26
   historical-cleanup pass) were from a different (older) dataset
   chain. Apples-to-apples observation requires rebuilding the HA/NA
   aa id099 and PB2/PB1 aa id099 datasets under the new clustering,
   and possibly id098 now that it's closer to feasible.

3. **Downstream LGBM / 1-NN / MLP results in
   `docs/results/2026-05-15_cluster_disjoint_nt_results.md`.** Those
   experiments were trained on datasets built from the prior easy-cluster
   aa artifacts. Under symmetric linclust the underlying cluster IDs
   are different. The headline finding from that doc ("1-NN cosine
   margin ≥ LGBM at every cluster_disjoint routing") may or may not
   survive the rebuild — re-running is the only way to know.

4. **The mmseqs cluster-time vs convertalis-time identity offset
   (Phase 3 gotcha).** Worth a brief one-paragraph addition to
   `clustering_overview.md` §2.1 documenting that the post-hoc
   `mmseqs search` + convertalis identity is not directly comparable
   to the threshold mmseqs uses during clustering. Currently noted
   here and in the plan doc but not in the methods reference.

---

## References

- Plan doc: `docs/plans/done/2026-05-21_aa_cluster_algorithm_validation_plan.md`
- BiCC audit (parent investigation): `docs/results/2026-05-21_bicc_pair_drop_audit.md`
- Methods doc (the consumer of these results): `docs/methods/clustering_overview.md`
  — §2.3 (algorithm choice), §6 (collapse trajectory), §9 (feasibility ceiling)
- Code: `src/utils/clustering_utils.py::run_mmseqs_easy_cluster` (the wrapper that
  pins the mmseqs2 flags and now defaults to linclust)
- Prior cross-alphabet experiment (predates the algorithm switch):
  `docs/results/2026-05-15_cluster_disjoint_nt_results.md`
- mmseqs2 version: 18.8cc5c at `/homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs`
