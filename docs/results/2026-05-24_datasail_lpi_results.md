# DataSAIL L(π) measurement on Flu A HA/NA (1000-isolate subsamples)

**Date:** 2026-05-24
**Branch:** `feature/lpi-measurement`
**Plan:** `docs/plans/2026-05-22_split_separation_metrics_plan.md`
(Step 1 — L(π) leg)
**Code:** `src/analysis/datasail_lpi_{measure,validate,diagnose_sim}.py`
(all run in the `datasail` conda env)
**Status:** PAUSED on the L(π) leg pending a decision on follow-up.

---

## Scope of what was tested

To keep claims tight, here is the exact scope of the experiments
behind this writeup.

- **Dataset family:** Flu A HA/NA pair, data version `July_2025`.
- **Three Stage 3 datasets** (all built 2026-05-20, before the
  2026-05-22 switch to symmetric easy-linclust aa clustering):
  - `dataset_flu_ha_na_cluster_id99_20260520_211534`
    (`cluster_disjoint aa id099`)
  - `dataset_flu_ha_na_seq_disjoint_20260520_211109`
    (`seq_disjoint hash_key=seq`)
  - `dataset_flu_ha_na_random_20260520_210647`
    (`random` per-pair)
- **Scale:** each dataset subsampled to 1000 isolates (seed = 0) for
  every run. Full-corpus scale was not tested.
- **DataSAIL similarity methods tested:** `mmseqs` (which DataSAIL
  resolves to `mmseqs easy-cluster` internally and returns an all-1s
  cluster-rep similarity matrix) and `mmseqspp` (which DataSAIL
  resolves to `mmseqs createdb + prefilter + align + convertalis`
  producing continuous per-pair `fident` scores).
- **S1 only** — `eval_split` is one-dimensional, so HA and NA are
  measured independently. S2 (pair-level joint leakage) was not
  implemented.
- **Diagnostic similarity-distribution** extracted only for the
  cluster_id099 dataset, both slots, `mmseqspp` only.

Everything below is empirical observation or a clearly labelled
inference from those observations. Items we did not actually run are
listed under "What was not tested" near the bottom.

---

## Headline observations

1. **Synthetic validation passed exactly.** `eval_split` on a 3-entity
   precomputed 3×3 all-1s similarity matrix returns `0`, `4/9 ≈ 0.444`,
   `6/9 ≈ 0.667` for the three split configurations that hand-compute
   to those values. The function's math is what we expect.
2. **Under `similarity='mmseqs'`, ratios across routings differ widely
   on both slots but the pattern doesn't match a clean "more leakage ↔
   higher L(π)" reading.** cluster_disjoint id099 has L(π) ≈ 0 on HA
   but ≈ 0.53 on NA; seq_disjoint has L(π) ≈ 0.17 on HA but ≈ 0 on NA.
   Random has L(π) ≈ 0 on both — see the wrapper caveat below for why
   the random value at this scale is not directly comparable.
3. **Under `similarity='mmseqspp'`, all three routings have L(π) ratios
   within ~1% of each other on both slots, all near 0.34.** 0.34 is
   exactly the partition-shape constant `1 − (0.8² + 0.1² + 0.1²)`
   for an 80/10/10 split.
4. **The `mmseqspp` similarity distribution on these data is
   bimodal/zero-inflated, not low-variance.** On HA at 1000 isolates,
   67 % of pairs have similarity 0 (no significant alignment) and
   16 % have similarity > 0.9. On NA the same: 67 % zero, 22 % > 0.9.
   Std (~0.38) is comparable to mean (~0.25).

Putting (3) and (4) together arithmetically gives one inference,
called out in its own section below: the L(π) collapse to 0.34 is
**consistent with high-similarity pairs being distributed across
splits without preference for the same split**. We did not directly
measure this; it is an inference from the arithmetic of L(π) and the
empirical similarity distribution.

---

## Setup

### Wrapper and validation

- `src/analysis/datasail_lpi_measure.py` — loads positives from a
  Stage 3 dataset directory, derives per-slot
  `{seq_hash: orig_split}` from the train/val/test CSVs, calls
  `eval_split` for HA and NA, writes a 2-row CSV with
  `lpi_ratio`, `lpi_absolute`, `lpi_total`, plus diagnostic counts.
- `src/analysis/datasail_lpi_validate.py` — three synthetic split
  configurations on a precomputed 3×3 all-1s similarity matrix.
  Passed exactly (`0`, `4/9`, `6/9`).
- `src/analysis/datasail_lpi_diagnose_sim.py` — extracts DataSAIL's
  internal `dataset.cluster_similarity` matrix after `cluster()`
  and reports off-diagonal stats + a histogram PNG.

### A wrapper-level caveat on the random baseline

`eval_split` requires one split label per entity (the
`{entity_name: split}` dict has one value per key). Under
cluster_disjoint and seq_disjoint, each seq_hash lives in exactly one
split by construction, so this contract is satisfied (0 ambiguous
entities). **Under random per-pair partitioning, the same seq_hash
appears in pairs assigned to multiple splits** — on this 1000-isolate
sample we observed 12 (HA) and 23 (NA) seq_hashes spanning multiple
splits. Our wrapper falls back to a "first-seen split" rule, which
collapses the cross-split repetition that is the actual source of
random-routing leakage. **The random L(π) values reported below are
therefore not a clean random baseline** at this measurement layer;
they understate the leakage that random per-pair partitioning
introduces at the pair level.

### A wrapper-level caveat on parallel runs

DataSAIL's `mmseqspp` code path writes `./mmseqs.fasta` and
`./mmseqspp_results/` as hardcoded relative paths in the cwd
(`datasail/cluster/mmseqspp.py:29,33`). Running multiple
`eval_split(..., similarity='mmseqspp')` calls in parallel from the
same cwd stomps on each other and one or more runs error out with
`"Something went wront with mmseqs alignment. The output file does
not exist."`. The runs below were all serialized. A future wrapper
revision could use a per-run temp dir to allow parallelization.

---

## Results: `similarity='mmseqs'`

At 1000-isolate subsample, one row per (routing × slot):

| Routing | Slot | n_unique_entities | lpi_ratio | lpi_absolute | lpi_total | ambiguous_entities | wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| cluster_disjoint aa id099 | HA (a) | 926 | **0.000173** | 9.23 × 10⁶ | 5.33 × 10¹⁰ | 0 | 88.0 |
| cluster_disjoint aa id099 | NA (b) | 904 | **0.5342** | 4.14 × 10¹⁰ | 7.75 × 10¹⁰ | 0 | 90.3 |
| seq_disjoint              | HA (a) | 916 | **0.1677** | 7.96 × 10⁹  | 4.75 × 10¹⁰ | 0 | 84.9 |
| seq_disjoint              | NA (b) | 923 | **0.0023** | 1.73 × 10⁸  | 7.50 × 10¹⁰ | 0 | 68.1 |
| random                    | HA (a) | 926 | **0.000009** | 4.50 × 10⁵ | 5.07 × 10¹⁰ | 12 | 67.4 |
| random                    | NA (b) | 902 | **0.0027**   | 3.49 × 10⁸ | 1.27 × 10¹¹ | 23 | 58.8 |

Observations:

- HA / NA asymmetry flips between routings: cluster_disjoint
  id099 has L(π) ≈ 0 on HA but ≈ 0.53 on NA; seq_disjoint has the
  reverse profile (≈ 0.17 on HA, ≈ 0 on NA). We did not directly
  investigate why, but it is consistent with DataSAIL's internal
  re-clustering producing different cluster boundaries than ours
  do — `similarity='mmseqs'` triggers a binary-search-tuned `-c`
  with default `--min-seq-id`, not the same parameters our id099
  clustering used. The L(π) under this method therefore reflects
  "fraction of DataSAIL's cluster-pairs straddling our partition",
  which depends on how DataSAIL's reclustering differs from ours —
  not directly on the leakage of our partition at our chosen
  identity threshold.
- The random row's L(π) on HA is essentially zero. This is not the
  intuitive ordering (random "should" have the highest leakage).
  Note the ambiguous-entity count (12 / 23) and the wrapper caveat
  above: random's value at this measurement layer is not a clean
  baseline.

---

## Results: `similarity='mmseqspp'`

At 1000-isolate subsample, serialized runs:

| Routing | Slot | n_unique_entities | lpi_ratio | lpi_absolute | lpi_total | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| cluster_disjoint aa id099 | HA (a) | 926 | **0.3352** | 71 621 | 213 651 | 80.8 |
| cluster_disjoint aa id099 | NA (b) | 904 | **0.3449** | 81 452 | 236 149 | 63.2 |
| seq_disjoint              | HA (a) | 916 | **0.3396** | 71 243 | 209 785 | 81.2 |
| seq_disjoint              | NA (b) | 923 | **0.3338** | 80 642 | 241 566 | 62.4 |
| random                    | HA (a) | 926 | **0.3406** | 72 227 | 212 043 | 80.1 |
| random                    | NA (b) | 902 | **0.3451** | 80 679 | 233 812 | 60.9 |

Observations:

- All six ratios lie in [0.334, 0.345] — a range of ~1.1 %.
- 0.34 is exactly the partition-shape constant
  `1 − (0.8² + 0.1² + 0.1²) = 0.34` for an 80/10/10 split. This is
  what L(π) reduces to when the cross-split sum of similarity is a
  constant fraction of the total — i.e., when the partition is
  effectively independent of the similarity distribution.
- Absolute leakage values are also within ~1.5 % across routings on
  both slots (HA: 71.2 k–72.2 k; NA: 80.6 k–81.5 k), so it is not the
  case that one routing has the same ratio as another by coincidence
  of a different total.

---

## Diagnostic: similarity-matrix distribution under `mmseqspp`

Run on the cluster_id099 dataset, both slots, same 1000-isolate
subsample. The off-diagonal upper-triangular cells of
`dataset.cluster_similarity` after DataSAIL's internal `mmseqspp` +
average-linkage clustering:

| Slot | n_entities | n_off_diag_pairs | mean | std | min | p5 | p25 | median | p75 | p95 | max | % == 0 | % > 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 926 | 428,275 | 0.250 | 0.381 | 0 | 0 | 0 | 0 | 0.462 | 0.961 | 0.99 | 67.3 % | 15.8 % |
| NA | 904 | 408,156 | 0.288 | 0.420 | 0 | 0 | 0 | 0 | 0.870 | 0.988 | 1.00 | 66.9 % | 21.5 % |

Histograms at:
- `results/flu/July_2025/runs/datasail_lpi/diag/diag_cluster_id099_HA_mmseqspp_n1000.png`
- `results/flu/July_2025/runs/datasail_lpi/diag/diag_cluster_id099_NA_mmseqspp_n1000.png`

Shape on both slots: two-mode mass. A large spike at 0 (~67 % of
pairs — no significant alignment under DataSAIL's mmseqspp invocation)
and a long right tail concentrated above 0.9 (16–22 % of pairs — near
clones). Median is 0 on both. Standard deviation is comparable to the
mean. This is **not** a low-variance distribution.

The diagonal is exactly 1.0 (self-similarity), as expected.

The `mmseqs` (binary) similarity matrix distribution was not extracted
in this round.

---

## What the arithmetic says (inference, not a measurement)

If similarity values were independent of split membership, the L(π)
ratio would reduce to

```
ratio = sum_{cross-split} sim_{ij} / sum_{all i,j} sim_{ij}
      ≈ E[1_{cross-split}] · E[sim] / E[sim]
      = E[1_{cross-split}]
      = 1 − (0.8² + 0.1² + 0.1²)
      = 0.34   for 80/10/10
```

The empirical mmseqspp ratios (0.334–0.345) match this expression
within 1 %. **One inference consistent with this**: the high-similarity
pairs in the `mmseqspp` matrix are distributed across the train / val /
test splits in roughly the same proportion as low-similarity pairs —
i.e., they are not preferentially placed in the same split by our
partitions. We did not measure this directly. It is an inference from
the arithmetic of L(π) combined with the empirical similarity
distribution.

---

## Two interpretations of that inference (neither tested directly)

The inference above is mechanism-neutral. Two distinct mechanisms could
produce it; both are plausible on this corpus and the runs done so far
do not distinguish them.

**Interpretation (I) — our partition leaks at the `mmseqspp`
identity scale.** If `mmseqspp` correctly identifies many cross-
id099-cluster pairs as > 0.9 similar, and those pairs end up in
different splits because our id099 cluster boundaries put them in
different clusters, then `mmseqspp` is reporting real leakage that
our id099 partition does not prevent. The cluster_id099 dataset used
here was built 2026-05-20 on **easy-cluster** aa cluster IDs; the
2026-05-22 algorithm switch produces different cluster boundaries
under linclust. If the linclust-era partition aligns better with
mmseqspp's similarity boundaries, L(π) under this metric would drop.
Untested.

**Interpretation (II) — `mmseqspp`'s identity scale and our id099
clustering are simply measuring different things.** `mmseqs align -e
inf + convertalis fident` admits very-permissive alignments; the
similarity it assigns to a pair may not correspond directly to
membership in the same id099 cluster, especially in a tight family
where many pair alignments are anchored on a few highly conserved
motifs and pass `-e inf` even at low overall identity. In that case
L(π) under mmseqspp is measuring leakage at a granularity our id099
partition is not designed to prevent, and the collapse to the
partition-shape constant is a property of the corpus + the metric,
not a defect of our partition. Untested.

**A discriminating check** (not run): take the high-sim pairs from
the dumped similarity matrix (those with `fident > 0.9`), look up
both endpoints' id099 cluster IDs in the original partition source,
and count how many high-sim pairs are intra-cluster vs cross-cluster.
- Mostly intra-cluster → favors interpretation (II): mmseqspp is
  finding similarity within our clusters, so cross-cluster pairs at
  fident > 0.9 are the minority; the L(π) signal is dominated by
  within-cluster pair similarity, which is intra-split.
- Mostly cross-cluster → favors interpretation (I): mmseqspp is
  finding similarity across our cluster boundaries, so our partition
  doesn't prevent cross-split similarity at this scale.

Either result would also need re-running on a linclust-era partition
(BACKLOG "Algorithm-switch follow-ups" #1) to separate the algorithm
generation effect from the metric-vs-partition mismatch effect.

---

## What was not tested

The investigation is bounded by what we ran. To avoid implying
otherwise:

- **Full-corpus scale.** Everything above is 1000-isolate subsample
  (~900 unique entities per slot). Scaling could change the picture
  — e.g., the corpus might reach a higher-variance similarity regime
  at larger N, or interpretation (I) might be sample-size-dependent.
- **`mmseqs` (binary) similarity distribution diagnostic.** We
  extracted only the `mmseqspp` similarity matrix for the
  distribution stats.
- **The intra- vs cross-cluster check on high-sim pairs** that would
  distinguish interpretations (I) and (II).
- **Linclust-era partitions.** Both the cluster_disjoint and
  seq_disjoint datasets here were built before the 2026-05-22
  algorithm switch. Rebuilding under symmetric easy-linclust
  (BACKLOG Algorithm-switch #1) and re-running these L(π)
  measurements would separate algorithm-generation effects from
  metric-vs-partition effects.
- **S2 (pair-level joint) L(π).** `eval_split` is 1D; the S2 case is
  not in DataSAIL's API and was not implemented here.
- **Other Flu A pairs** (PB2/PB1 etc.), other viruses.
- **Whether MMD on PCA-reduced pair embeddings shows similar or
  different behavior on this corpus.** MMD is the other proposed
  split-separation metric in the plan; it operates on a different
  feature space (ESM-2 embeddings, not mmseqs alignments) and has
  not been tested.

---

## Reproduce

```bash
# Synthetic validation (passed exactly):
conda run -n datasail python src/analysis/datasail_lpi_validate.py

# mmseqs runs (used in the table above; each ~60-90 s):
for ROUTING_PAIR in \
    "cluster_disjoint_aa_id099:dataset_flu_ha_na_cluster_id99_20260520_211534" \
    "seq_disjoint:dataset_flu_ha_na_seq_disjoint_20260520_211109" \
    "random:dataset_flu_ha_na_random_20260520_210647"; do
  LABEL="${ROUTING_PAIR%%:*}"; DSDIR="${ROUTING_PAIR##*:}"
  conda run -n datasail python src/analysis/datasail_lpi_measure.py \
    --dataset_dir data/datasets/flu/July_2025/runs/${DSDIR} \
    --routing_label ${LABEL} \
    --similarity mmseqs \
    --n_isolates 1000 \
    --out_csv results/flu/July_2025/runs/datasail_lpi/${LABEL}_n1000_mmseqs.csv
done

# mmseqspp runs — MUST be serial (DataSAIL writes hardcoded ./mmseqs.fasta
# + ./mmseqspp_results/ paths in cwd; parallel runs collide):
for ROUTING_PAIR in ...; do
  conda run -n datasail python src/analysis/datasail_lpi_measure.py \
    ... --similarity mmseqspp ...
done

# Diagnostic similarity-distribution (one slot at a time, ~60s):
conda run -n datasail python src/analysis/datasail_lpi_diagnose_sim.py \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_20260520_211534 \
    --slot a --similarity mmseqspp --n_isolates 1000 \
    --routing_label cluster_id099 \
    --out_dir results/flu/July_2025/runs/datasail_lpi/diag
# (repeat with --slot b)
```

Outputs (gitignored under `results/`):

- `results/flu/July_2025/runs/datasail_lpi/{routing}_n1000_{mmseqs,mmseqspp}_*.csv`
- `results/flu/July_2025/runs/datasail_lpi/diag/diag_cluster_id099_{HA,NA}_mmseqspp_n1000.{csv,png}`

---

## See also

- `docs/plans/2026-05-22_split_separation_metrics_plan.md` — parent
  plan; Step 1 (L(π) leg) is the source of this work. MMD leg
  (Step 2) is independent and not addressed here.
- `docs/plans/2026-05-19_datasail_bakeoff_plan.md` — earlier DataSAIL
  bake-off (paused 2026-05-20); the bake-off ran DataSAIL in
  *solving* mode, this work runs it in *measurement-only* mode via
  `eval_split`.
- `docs/results/2026-05-20_datasail_phase0_results.md` — the
  Phase 0 sanity results that originally proposed using `L(π)` as a
  shared yardstick on bicc splits (their "Recommended next steps" #1).
- `docs/results/2026-05-15_cluster_disjoint_nt_results.md` — the
  1-NN cosine margin work that is the existing edge-style
  split-separation diagnostic this L(π) work was meant to
  complement.
- DataSAIL paper: Joeres et al. 2025 (the L(π) metric is Eq. 20).
  Source: `refs/joeres2025_datasail.pdf` and `_supp.pdf`.
- DataSAIL source code (in conda env):
  `/homes/apartin/miniconda3/envs/datasail/lib/python3.10/site-packages/datasail/`
  — `eval.py` (eval_split), `cluster/mmseqs2.py` (cheap path),
  `cluster/mmseqspp.py` (continuous path), `cluster/utils.py`
  (cluster_param_binary_search and its 10-100 target window).
