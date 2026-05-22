# 2026-05-21 — aa clustering algorithm validation: easy-cluster vs easy-linclust

**Status: IN PROGRESS**

## Problem

Our clustering pipeline uses **asymmetric** mmseqs2 algorithms across alphabets:
- aa: `mmseqs easy-cluster` (sensitive, cascaded; default `--cluster-mode 0` = Set-Cover)
- nt: `mmseqs easy-linclust` (linear-time, single-pass; default `--cluster-mode 2` = greedy-by-length)

The asymmetric choice was justified as a speed/sensitivity trade-off appropriate for each
alphabet, with an unverified claim that linclust produces "within-noise different cluster
counts" relative to easy-cluster on this corpus. That claim was tested on aa for the
first time on 2026-05-21, and **the two algorithms disagree by 1–2 orders of magnitude at
sub-id100 thresholds**. This calls three things into question:

1. The aa-vs-nt comparison in `clustering_overview.md` §8 / §9 conflates algorithm
   sensitivity with alphabet diversity. The "collapse is corpus-driven, not
   algorithm-driven" framing may not survive.
2. The §8.1 headline "PB2 717→77 collapse between id097→id096 (89% drop)" is
   algorithm-specific. Under easy-linclust the same trajectory is 7,634 → 6,755 (12%
   drop) — no cliff at all.
3. Existing aa cluster_disjoint id099 datasets (and the LGBM / 1-NN / MLP models trained
   on them) are downstream of easy-cluster artifacts; if those artifacts turn out to be
   "over-merged" relative to the biologically meaningful threshold, the downstream
   conclusions need re-validation.

Goal: explain the gap, decide whether to switch algorithms (and if so to which), and
update the methods doc accordingly.

---

## Empirical findings to date (2026-05-21 sweep)

aa easy-linclust sweep run with **identical** parameters to the production aa easy-cluster
run (same FASTA per function, same thresholds, same coverage rule). Results in
`tmp/clusters_aa_linclust/` (gitignored). Comparison script at
`/tmp/compare_aa_cluster_vs_linclust.py`.

**Wall-clock.** easy-cluster sweep was 3,011 s (80 runs). easy-linclust on the same 90
runs took 526 s — **5.7× speedup**.

**Cluster-count disagreement.** Across 90 (function, threshold) cells:

- Only **11 / 90 cells (12.2%) agree within ±1%** on `n_clusters`.
- Median absolute disagreement: **248%**. Mean: 900%. Max: 24,865%.

**By threshold** — disagreement grows then narrows:

| Threshold | Median |%diff| | Max |%diff| |
|---|---:|---:|
| id100 | 1.4% | 49% (NA) |
| id099 | 91% | 429% (NP) |
| id098 | 363% | 831% (NP) |
| id097 | 759% | 1,414% (PB1) |
| id096 | 736% | 8,673% (PB2) |
| id095 | 789% | 24,865% (PB2) |
| id090 | 584% | 1,100% (PB2) |
| id085 | 93% | 1,133% (HA) |
| id080 | 50% | 1,640% (NS1) |

**Cell highlights (easy-cluster vs easy-linclust):**

| Function | Threshold | easy-cluster | easy-linclust | Ratio |
|---|---|---:|---:|---:|
| PB2 | id095 | 26 | 6,491 | 250× |
| PB2 | id096 | **77** | 6,755 | 88× |
| PA  | id095 | 158 | 8,002 | 51× |
| PB1 | id095 | 50 | 2,033 | 41× |
| NP  | id097 | 153 | 1,750 | 11× |
| HA  | id097 | 1,753 | 11,459 | 6.5× |
| **NA** | **id100** | **37,102** | **18,753** | **0.5×** |

**Direction is one-sided** at sub-id100: easy-linclust always produces *more* clusters
(smaller cluster sizes). Consistent with linclust's documented sensitivity loss —
sequences that the cascade catches as similar get split into separate clusters when the
single-pass prefilter misses the k-mer evidence.

**Surprise — id100 NA.** The two algorithms disagree by 49% at exact-identity. This is
unexpected: both should produce identical clusters at id100. Most likely a difference in
edge-case handling (coverage / overhangs / ties at id100). Worth understanding because
id100 is where we'd assume zero ambiguity.

---

## Hypotheses

In rough order of likelihood:

### A — `--cluster-mode` default differs (very likely)

- easy-cluster default: `--cluster-mode 0` (Set-Cover, greedy)
- easy-linclust default: `--cluster-mode 2` (greedy-by-length)

Same similarity graph + different assignment algorithm → different cluster counts and
sizes. This alone could explain a large fraction of the gap.

### B — Sensitivity defaults differ (likely contributes)

- easy-cluster runs a **cascaded** prefilter (multiple rounds at increasing sensitivity,
  `-s` ramping). Sensitivity-tuned for protein search.
- easy-linclust runs a **single pass**. Linear time but more conservative prefilter.

At high identity (id099+) linclust's prefilter is reportedly highly accurate; the
drop-off at id095 and below could match mmseqs2's documented behavior. But 87× at PB2
id096 is well beyond the few-% range the linclust paper reports for typical protein
corpora.

### C — `--cluster-reassign` (cluster-only feature)

easy-cluster has an optional second-pass re-assignment step that pulls borderline
members into the largest matching cluster. easy-linclust does not. If re-assign is on by
default for cluster and absent in linclust, the conserved-protein mega-cluster signature
could be largely a re-assign artifact.

### D — Flu-specific corpus pathology

Flu's conserved proteins (PB2, PB1, PA, NP, M1) have many sequences pairwise ≥0.96
identical, but where each individual pair only shares a few exact-length k-mers (because
the few SNPs are scattered along the protein). easy-cluster's cascade catches the
indirect similarity via intermediate cluster members; linclust's single-pass prefilter
doesn't. Plausible but needs the ground-truth check (Phase 3).

### E — Bookkeeping discrepancy (unlikely, easy to rule out)

The script's `n_clusters` / `largest_cluster` reporting might count differently between
the two paths (e.g., singleton handling, deduplication). Easy to rule out by reading the
mmseqs `_cluster.tsv` directly.

---

## Investigation plan

Phased, each phase with go/no-go decision. Estimated total: ~1 working day if everything
cooperates. All experiments go in `tmp/` first; nothing committed to production
artifacts until a decision is made.

### Phase 0 — Parameter parity audit (~30 min)

**Question.** Are the two mmseqs invocations using the same parameters where they should
be, and where do their defaults legitimately differ?

**Action.** Run one easy-cluster job and one easy-linclust job on the same FASTA (a
small subset, e.g., 100 HA sequences) at a single threshold (e.g., 0.97). Dump the
full mmseqs parameter summary that mmseqs prints at the start of each run. Build a
side-by-side diff table of effective parameters.

**Specifically check:**
- `--cluster-mode` (cluster=0 default; linclust=2 default — H/A)
- `--cluster-reassign` (cluster default 1; linclust absent — H/C)
- `-s` / `--sensitivity` (cluster cascades; linclust fixed — H/B)
- `--alignment-mode`, `--alignment-output-mode` (could affect what counts as "aligned")
- `--seq-id-mode` (should be 0 for both)
- `--cov-mode`, `-c` (we pass these uniformly — confirm)

**Decision rule.** If the only flagged-as-different parameters are `--cluster-mode` and
the cascade, narrow scope to those before testing Phase 1 / 2. If anything else is
unexpectedly different, fix it first.

### Phase 1 — Same `--cluster-mode` control (~30 min)

**Question.** How much of the gap is just from cluster-mode default differences (H/A)?

**Action.** Re-run easy-linclust with `--cluster-mode 0` explicit (Set-Cover, matching
cluster's default). Use a tight subset: PB2 + HA at id099 / id097 / id095 (6 cells, the
extreme-divergence zone).

**Decision rule.**
- If gap shrinks to <10%: hypothesis A explains most of it. Switch path: use
  easy-linclust everywhere + force `--cluster-mode 0`. Phase 2/3 optional.
- If gap persists at >2×: hypothesis A insufficient. Continue to Phase 2.

### Phase 2 — `--cluster-reassign 0` control (~30 min)

**Question.** Is easy-cluster's optional re-assign step responsible for the
conserved-protein mega-clusters (H/C)?

**Action.** Re-run easy-cluster on PB2 + HA at id099 / id097 / id095 with
`--cluster-reassign 0`. Confirm whether the cluster counts jump (toward easy-linclust's
numbers) when re-assign is off.

**Decision rule.**
- If easy-cluster cluster counts increase substantially with re-assign off: H/C is the
  dominant cause. Then the question becomes "which mode is biologically more
  meaningful" (Phase 3).
- If easy-cluster counts barely move: H/C is not the cause. Continue to Phase 3.

### Phase 3 — Ground-truth spot check (~2 hours)

**Question.** When the algorithms disagree, which one is biologically correct?

**Setup.** Pick the worst case: **PB2 at id096**. easy-cluster reports 77 clusters
covering 33,663 unique sequences; easy-linclust reports 6,755 clusters.

**Action.**

1. **Is easy-cluster over-merging?** Sample 100 random sequence pairs from
   *within one large easy-cluster cluster*. Compute pairwise identity directly using
   `mmseqs search` (no clustering — pairwise alignment only) or BLAST. Histogram the
   identities. If many pairs have identity < 0.96, easy-cluster is making false-positive
   similarity calls — lumping non-similar sequences together. If all pairs are ≥ 0.96,
   easy-cluster is justified.
2. **Is easy-linclust under-merging?** Sample 100 random sequence pairs that lie in
   *different easy-linclust clusters but the same easy-cluster cluster*. Compute
   pairwise identity. If most are ≥ 0.96, easy-linclust missed true matches
   (false negatives).

**Decision rule.** This tells which algorithm is operating at which point on the
precision/recall trade-off. There may be no single "right" answer; the two algorithms
may be valid operating-point choices on the same trade-off curve.

### Phase 4 — Cross-tool reference (~2 hours, optional)

**Question.** What does an independent, mature clustering tool say about the same corpus?

**Action.** Run **CD-HIT** with `-c 0.96` on PB2's unique sequences at id096. Cluster
count from CD-HIT becomes a third data point. CD-HIT uses a different algorithm
(single-pass greedy with banded word matching) and has been a field standard for ~20
years; whichever mmseqs mode CD-HIT agrees with is the one most aligned with
"conventional bioinformatician expectations".

**Decision rule.** Only run this if Phases 0–3 didn't fully resolve. Treat CD-HIT as a
tie-breaker, not ground truth.

### Phase 5 — Decision + writeup (~1 hour)

**Action.** Based on Phases 0–4, pick one:

1. **Switch to easy-linclust on both alphabets + force `--cluster-mode 0`.**
   Symmetric, faster, paper-defensible. Cost: re-run aa cluster sweep (cheap, ~10 min);
   re-validate downstream cluster_disjoint id099 datasets + the LGBM / 1-NN / MLP
   results that depend on them (not cheap).
2. **Switch to easy-cluster on both alphabets** (run easy-cluster on nt). Symmetric,
   more sensitive, methodologically conservative. Cost: nt easy-cluster on the larger
   nt corpus (~868K sequences) — needs runtime estimate. Existing aa results unchanged.
3. **Stay asymmetric, document the bias explicitly.** Acknowledge in §8 / §9 that the
   aa-vs-nt comparison conflates algorithm sensitivity with alphabet diversity, and
   weaken the "corpus-driven not algorithm-driven" claim accordingly. Existing artifacts
   stay valid; methods doc gains an asterisk.

**Writeup.** Update this plan doc with findings, mark `Status: IMPLEMENTED`, move to
`docs/plans/done/`. Update `clustering_overview.md` §8.4 + §2.3 to reflect whichever
decision was made. Update `BACKLOG.md` if the decision spawns follow-up work.

---

## Code walkthrough (supplementary)

Two small scripts to step through the mmseqs2 + comparison logic at the lowest level.
These complement Phases 0–3 with hands-on understanding of the code, not new
experiments.

### Step 1 — `tmp/step1_mmseqs_minimal.py`

**Goal.** Strip away the wrapper. Run easy-cluster on 100 random HA sequences at id097
with `verbose=True`. See the actual mmseqs CLI command, the `*_cluster.tsv` output, and
the parsed cluster dict.

**Approach.** Sample 100 HA sequences from `protein_final.parquet`, write a FASTA, call
`run_mmseqs_easy_cluster` directly. Dump the rep_id ↔ member_id pairs, hand-compute
n_clusters / largest_cluster / fraction_singletons and verify against
`cluster_size_distribution()`. ~30 lines.

### Step 2 — `tmp/step2_cluster_vs_linclust_diff.py`

**Goal.** This is essentially Phase 3 mechanized. Run easy-cluster AND easy-linclust on
identical 100 HA sequences at id097. For pairs the algorithms disagree on (same cluster
under one, different under the other), invoke `mmseqs search` to compute the actual
pairwise identity. Histogram the identities. ~60 lines.

**Output.** Disagreement count, pairwise-identity histogram for disputed pairs. Answers
the over-merging vs under-merging question at small scale before Phase 3's full-corpus
ground-truth check.

### Deferred — Steps 3, 4, 5

Originally proposed (sweep reproducibility, BiCC routing walk-through, Stage 3
instrumentation) — not relevant to the algorithm validation question. Defer indefinitely
or document as separate exploratory work.

---

## Decision tree (summary)

```
Phase 0 (parameter parity)
   ├── only --cluster-mode / cascade differ → Phase 1
   └── unexpected param differences → fix wrapper first, re-run

Phase 1 (force --cluster-mode 0 on linclust)
   ├── gap closes (<10%) → Decision = "switch to linclust + cluster-mode 0", skip 2/3
   └── gap persists (>2×) → Phase 2

Phase 2 (turn off cluster-reassign in cluster)
   ├── cluster counts jump → re-assign is the cause → Phase 3 (which mode is right?)
   └── counts barely move → re-assign not the cause → Phase 3 (sensitivity gap is real)

Phase 3 (ground-truth on PB2 id096)
   ├── easy-cluster over-merges (within-cluster id < 0.96) → linclust wins, switch
   ├── easy-linclust under-merges (cross-cluster id >= 0.96) → cluster wins, keep asym
   └── both valid (different operating points) → Phase 4 or methodology call

Phase 4 (CD-HIT tie-breaker, optional)
   ├── CD-HIT agrees with mmseqs cluster → cluster wins
   └── CD-HIT agrees with mmseqs linclust → linclust wins

Phase 5 — decide + writeup + (optionally) re-generate downstream artifacts.
```

## Scope cuts if a full day is too much

- **Minimum useful**: Phase 0 + Phase 1. If hypothesis A explains the gap, we're done.
- **Next-most-useful**: add Phase 2. If A + C explain it, also done.
- **Full investigation**: through Phase 3 (ground-truth is the highest-info experiment
  if A + C don't suffice).
- **Phase 4 is padding**: only do it if we need an independent reference after Phases 0–3.

---

## References

**Code paths under investigation:**
- mmseqs wrapper: `src/utils/clustering_utils.py::run_mmseqs_easy_cluster` (handles
  both algorithms via the `algorithm` kwarg; never overrides `--cluster-mode` or
  `--cluster-reassign`).
- Sweep driver: `src/analysis/seq_redundancy_per_function.py`.
- Stage 3 consumer: `src/datasets/_pair_helpers.py::attach_cluster_ids` and
  `src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df`.

**Related docs:**
- `docs/results/2026-05-21_bicc_pair_drop_audit.md` — the BiCC audit that exposed
  the asymmetric-algorithm question.
- `docs/methods/clustering_overview.md` §2.3 (algorithm choice), §8 (cluster collapse),
  §9 (feasibility ceiling).
- `docs/results/2026-05-15_cluster_disjoint_nt_results.md` — the nt cluster_disjoint
  empirical results that downstream of this question (id099 datasets, LGBM / 1-NN runs).

**Validation sweep artifacts (gitignored):**
- `tmp/clusters_aa_linclust/` — full sweep output (cluster parquets, redundancy CSV,
  runtime JSON, redundancy markdown).
- `/tmp/compare_aa_cluster_vs_linclust.py` — comparison script.

**mmseqs2 references:**
- `mmseqs easy-cluster --help` and `mmseqs easy-linclust --help` for current
  parameter defaults (mmseqs2 v18.8cc5c installed at
  `/homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs`).
- Steinegger & Söding 2018 (linclust paper) — for the documented sensitivity / speed
  trade-off claims.
