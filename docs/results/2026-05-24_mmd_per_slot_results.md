# Per-slot MMD on Flu A HA/NA — does MMD discriminate split routings?

**Date:** 2026-05-24
**Branch:** `master` (no separate feature branch was used for this work)
**Plan:** `docs/plans/2026-05-22_split_separation_metrics_plan.md` — Step 2
(MMD leg)
**Code:** `src/analysis/mmd_per_slot.py` (runs in the `segmatch` conda env)
**Status:** Phase 1 (wiring sanity) + Phase 2 (real comparison) done at the
1000-isolate subsample on **both ESM-2 and aa k=3 feature spaces**.
Full-corpus run flagged as a follow-up.

---

## Scope of what was tested

To keep claims tight, here is exactly what these experiments cover.

- **Datasets** — three Stage 3 HA/NA bundles, all built 2026-05-20 on
  easy-cluster-era aa cluster artifacts. Identical pair-CSV row counts
  (116,775 / 14,597 / 14,597) → they share source positives and differ
  only in routing:
  - `dataset_flu_ha_na_random_20260520_210647`
  - `dataset_flu_ha_na_seq_disjoint_20260520_211109`
  - `dataset_flu_ha_na_cluster_id99_20260520_211534`
- **Subsample** — 1000 isolates per dataset with `subsample_seed = 42`
  (fixed across runs).
- **Slots** — HA (`slot a`) and NA (`slot b`), measured independently
  one slot at a time (S1-style; not the pair-level S2 case).
- **Feature spaces** — two protein-level feature spaces compared,
  each PCA-reduced to 50 dims via `compute_pca_reduction`:
  - **ESM-2** (1280-dim per protein, learned embeddings) — the
    project's general-purpose protein representation.
  - **aa k=3** (8000-dim per protein, k-mer count vector) — the
    project's strongest classifier feature on Flu A per memory.md
    "Key Findings".
  Same entity key on both sides (`(assembly_id, brc_fea_id)`), so
  the entity set is identical across feature spaces by construction.
- **Kernel** — RBF. σ is fixed across runs *within* a feature space
  so MMD² values are directly comparable across routings × slots
  inside that feature space. σ values DIFFER between feature spaces
  (ESM-2: σ = 1.0719; aa k=3: σ = 29.3192) because the two feature
  spaces have very different raw-magnitude scales — each σ was the
  median heuristic computed on cluster_id099 HA in its own feature
  space. **Cross-feature-space MMD² values are not directly
  comparable in absolute terms**; the right comparison is within-
  feature-space ratios and p-values.
- **Estimator** — biased MMD² (Gretton 2012 Eq. 4):
  `MMD² = mean(K(X,X)) + mean(K(Y,Y)) − 2·mean(K(X,Y))`.
- **Significance** — permutation test, 500 train/test label shuffles per
  run, p-value with add-one smoothing (Phipson & Smyth 2010) so the
  minimum reportable p is 1/501 ≈ 0.002.
- **Two phases**:
  - **Phase 1** (wiring sanity): random per-entity 50/50 splits on HA
    with 10 different seeds. No permutation test — the N random seeds
    *are* the null distribution.
  - **Phase 2** (real comparison): use each dataset's own train vs test
    labels, with the permutation test on top.

**What was not tested in this round**: full-corpus scale; other Flu A
pairs (PB2/PB1, etc.); nt k-mer feature spaces (only aa k=3 covered);
pair-level (S2) MMD; downstream model performance to validate whether
MMD-detected shift translates to a generalization gap; any other virus.

---

## Headline observations

1. **Wiring sanity (Phase 1) passed.** Random per-entity 50/50 splits
   on HA, 10 seeds: mean MMD² = +0.00225, std = 0.0015, range
   [+0.00071, +0.00537]. Magnitude matches the biased estimator's
   positive O(1/N) finite-sample bias (1/467 ≈ 0.002).
2. **Ordering observed on both slots (Phase 2)** under fixed σ:
   random ≤ seq_disjoint < cluster_disjoint id099.
3. **Permutation p-values cleanly differentiate the three routings.**
   - cluster_disjoint id099 — **0 of 500 permutations exceeded the
     observed MMD² on either slot** → p = 1/501 ≈ 0.002 (the smallest
     reportable under add-one smoothing on 500 shuffles).
   - seq_disjoint — borderline: HA p = 0.166, NA p = 0.056.
   - random — not significant: HA p = 0.639, NA p = 0.864.
4. **HA and NA agree on ordering and significance.** No analogue of the
   HA / NA asymmetry reported in `2026-05-24_datasail_lpi_results.md`
   under DataSAIL's `mmseqs` similarity. Under ESM-2, HA's absolute
   MMD² values are roughly 3× NA's; not disentangled (see "What is
   not tested").

5. **Feature-space robustness check (aa k=3) passes the headline test
   and partially refines the picture.** All cluster_disjoint id099 and
   random p-values agree across ESM-2 and aa k=3 (cluster_disjoint
   p ≈ 0.002 everywhere, random p > 0.6 everywhere). seq_disjoint
   diverges on HA: ESM-2 calls it not significant (p = 0.166), aa k=3
   calls it significant (p = 0.004). On NA both feature spaces agree
   on a borderline call (p ≈ 0.06). The aa-k3 advantage on HA
   seq_disjoint is consistent with the project's existing "K-mer
   dominates ESM-2 on H3N2" finding (memory.md), but the effect is
   slot-dependent — not a universal claim that aa k=3 is more
   sensitive than ESM-2 for MMD on this corpus.

---

## Setup details

### Per-slot data extraction

`load_unique_slot_entities`:

1. Reads `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv`; tags
   each row with `orig_split ∈ {train, val, test}`.
2. Filters to positives (`label == 1`).
3. Subsamples 1000 isolates by `assembly_id_a` with the fixed seed.
4. Dedups by `seq_hash_{slot}` so each unique protein appears once.
5. Flags `is_ambiguous` for entities whose pairs cross multiple splits
   (only happens under random per-pair routing; entity-disjoint
   routings have zero ambiguous entities by construction).

For each (routing × slot) the resulting entity count is in the 913–942
range (slightly different across datasets — see Phase 2 table; the
differences are likely from minor variations in positive labeling
across the three Stage 3 builds).

### Feature pipeline

- Look up ESM-2 embeddings via `(assembly_id, brc_fea_id)` from the
  master HDF5 cache at
  `data/embeddings/flu/July_2025/master_esm2_embeddings.h5`. Embedding
  load dominates wall time (~110 s per run for ~900 entities).
- PCA to 50 dims, `random_state = subsample_seed = 42`.

### Bandwidth choice

- **Phase 1**: median heuristic computed on the PCA-50 matrix → σ =
  1.0719.
- **Phase 2 (this work)**: σ = 1.0719 hard-coded via `--sigma 1.0719`
  for all six runs. Originally each Phase 2 run picked its own median
  bandwidth (1.07, 1.21, 1.24 across the three routings — see "Earlier
  run with per-run σ" below); fixing gave directly comparable MMD²
  values across routings.

### Permutation test

- 500 shuffles of the train / test label assignment per run.
- For each shuffle, recomputes biased MMD² with the same σ.
- p-value =  `(n_extreme + 1) / (n_permutations + 1)` (Phipson & Smyth
  2010 add-one smoothing) — avoids reporting `p = 0` and gives the
  minimum p of 1/501 ≈ 0.002 under 500 shuffles.
- Cost: ~30–45 s per run (dominated by the 500 RBF kernel matrices on
  ~860-entity samples). Adding the permutation test to a run roughly
  doubles wall time (the ESM-2 lookup still dominates).

### Wrapper details that matter for interpretation

- **val excluded from the MMD comparison.** MMD is a 2-sample test;
  we compare train vs test. val entities are present in the PCA basis
  fit and in the σ subsample but not in the MMD numerator/denominator.
- **Ambiguous entities filtered before MMD.** Only affects the random
  dataset (entity-disjoint routings have 0 ambiguous). Filter is
  cheaper than picking a tiebreaking rule.
- **σ fixed across runs (Phase 2).** This is the lever that makes the
  numbers in the Phase 2 table directly comparable. Earlier Phase 2
  runs without `--sigma` produced slightly different σ per routing —
  the ordering was preserved but the absolute MMD² values weren't
  cleanly comparable.

---

## Results

### Phase 1 — wiring sanity (random per-entity 50/50 splits on HA)

Dataset: `dataset_flu_ha_na_cluster_id99_20260520_211534` (we ignore
its labels and resample). σ = 1.0719 (median heuristic).
n_entities = 934.

| split_seed | n_A | n_B | MMD² |
|---:|---:|---:|---:|
| 0 | 467 | 467 | +0.001033 |
| 1 | 467 | 467 | +0.000945 |
| 2 | 467 | 467 | +0.002972 |
| 3 | 467 | 467 | +0.002392 |
| 4 | 467 | 467 | +0.000713 |
| 5 | 467 | 467 | +0.004420 |
| 6 | 467 | 467 | +0.001916 |
| 7 | 467 | 467 | +0.001081 |
| 8 | 467 | 467 | +0.001659 |
| 9 | 467 | 467 | +0.005368 |

**Summary**: mean = +0.00225, std = 0.00150, min = +0.00071, max =
+0.00537. All values positive, all small. Consistent with H0 "same
distribution" plus the biased estimator's bias.

### Phase 2 — dataset labels with fixed σ and 500-permutation p-value

σ = 1.0719 for all six runs.

| Slot | Routing | n_entities | n_ambig | n_train | n_test | MMD² | × random | n_extreme/500 | p-value |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | random | 942 | 11 | 766 | 97 | +0.003450 | 1.0× | 319 | 0.639 |
| HA | seq_disjoint | 933 | 0 | 756 | 99 | +0.010454 | 3.0× | 82 | 0.166 |
| HA | cluster_disjoint id099 | 934 | 0 | 758 | 98 | +0.041859 | 12.1× | **0** | **0.002** |
| NA | random | 941 | 22 | 747 | 79 | +0.001111 | 1.0× | 432 | 0.864 |
| NA | seq_disjoint | 919 | 0 | 742 | 99 | +0.004965 | 4.5× | 27 | 0.056 |
| NA | cluster_disjoint id099 | 913 | 0 | 738 | 98 | +0.014557 | 13.1× | **0** | **0.002** |

`× random` is the multiplier vs the row's own random baseline on the
same slot.

### Feature-space robustness — aa k=3 cross-check

Same script, same datasets, same subsample seed, same partitions —
only the feature space and σ change. aa k=3 uses the project's
k-mer cache at `data/embeddings/flu/July_2025/kmer_features_aa_k3.npz`,
keyed by the same `(assembly_id, brc_fea_id)` composite. Entity sets
are identical to the ESM-2 runs (0 cache misses on both feature
spaces — verified at run time).

**Phase 1 wiring sanity (aa k=3, HA, 10 random per-entity 50/50 splits):**
mean MMD² = +0.00124, std = 0.00045, range [+0.00067, +0.00235].
Passes the same noise-floor sanity check as ESM-2.
σ measured here = 29.3192 (vs ESM-2's 1.0719 — k-mer count vectors
have larger raw magnitudes, so the PCA-50 median pairwise distance
is much larger).

**Phase 2 (aa k=3, fixed σ = 29.3192, 500-permutation):**

| Slot | Routing | n_train | n_test | MMD² | n_extreme/500 | p-value |
|---|---|---:|---:|---:|---:|---:|
| HA | random | 766 | 97 | +0.002888 | 308 | 0.617 |
| HA | seq_disjoint | 756 | 99 | +0.020771 | 1 | **0.004** |
| HA | cluster_disjoint id099 | 758 | 98 | +0.053951 | 0 | **0.002** |
| NA | random | 747 | 79 | +0.002586 | 322 | 0.645 |
| NA | seq_disjoint | 742 | 99 | +0.006719 | 30 | 0.062 |
| NA | cluster_disjoint id099 | 738 | 98 | +0.022333 | 0 | **0.002** |

**Side-by-side p-value comparison (ESM-2 vs aa k=3):**

| Slot | Routing | ESM-2 p | aa k=3 p | Agreement |
|---|---|---:|---:|---|
| HA | random | 0.639 | 0.617 | both non-significant |
| HA | seq_disjoint | 0.166 | 0.004 | **diverge** — aa k=3 sig, ESM-2 not |
| HA | cluster_disjoint id099 | 0.002 | 0.002 | both highly significant (floor) |
| NA | random | 0.864 | 0.645 | both non-significant |
| NA | seq_disjoint | 0.056 | 0.062 | both borderline (agree) |
| NA | cluster_disjoint id099 | 0.002 | 0.002 | both highly significant (floor) |

Observations from the side-by-side:

- The **cluster_disjoint id099 result is robust to feature space** on
  both slots (p ≈ 0.002 everywhere; 0 of 500 permutations exceeded
  observed MMD² in every cell).
- The **random baseline is robust to feature space** on both slots
  (p > 0.6 everywhere).
- The **seq_disjoint signal is feature-space-dependent on HA but
  consistent on NA**. HA shows a real disagreement (ESM-2 misses
  what aa k=3 catches at α = 0.05). NA shows agreement at the
  borderline level.
- Whether the HA seq_disjoint divergence reflects a genuine
  feature-space sensitivity difference, or PCA-50 capturing different
  structure on the 8000-dim aa-k3 vs 1280-dim ESM-2 inputs, is not
  disentangled here.

### Earlier Phase 2 run with per-run σ (kept for traceability)

Before fixing σ, each Phase 2 run picked its own median bandwidth.
The MMD² values were similar in ordering and ratio but not directly
comparable across runs (different kernel). Numbers below; full CSVs
are in `results/flu/July_2025/runs/split_separation_mmd/`.

| Slot | Routing | σ (per-run) | MMD² | n_train | n_test |
|---|---|---:|---:|---:|---:|
| HA | random | 1.2088 | +0.002809 | 766 | 97 |
| HA | seq_disjoint | 1.2351 | +0.008208 | 756 | 99 |
| HA | cluster_disjoint id099 | 1.0719 | +0.041858 | 758 | 98 |
| NA | (per-run σ not measured for NA in this round) | — | — | — | — |

The per-run σ values themselves varied by ~15 % even though entity
sets overlapped ~99 %, which suggests the PCA-50 basis is sensitive
to the ~1–10 entities of difference between datasets. The fixed-σ
Phase 2 above bypasses this.

---

## Interpretation — what we can and cannot claim

### What the empirical results show

- On the three 1000-isolate Flu A HA/NA datasets tested with the
  ESM-2 PCA-50 feature space:
  - MMD² for cluster_disjoint id099 sits 12× (HA) or 13× (NA) above
    its corresponding random baseline.
  - The permutation test rejects the null "same distribution" at p ≈
    0.002 on both slots for cluster_disjoint id099 (no shuffled
    MMD² out of 500 reached the observed value).
  - seq_disjoint produces a borderline result (p = 0.166 on HA, 0.056
    on NA) — the metric notices something but not at the conventional
    α = 0.05 threshold on HA.
  - Random produces no significant signal (p = 0.639 on HA, 0.864 on
    NA). Random NA's high p indicates the observed value is at the
    low end of its own null distribution; not concerning.
- HA and NA agree on routing ordering. No HA / NA asymmetry analogous
  to the L(π) result on the same datasets.
- Repeating the same six runs on aa k=3 features (same datasets, same
  partitions, same subsample seed, different feature space and σ)
  preserves the cluster_disjoint id099 result (p ≈ 0.002 on both
  slots) and the random baseline (p > 0.6 on both slots). The
  seq_disjoint p-value diverges between feature spaces on HA only
  (ESM-2 p = 0.166, aa k=3 p = 0.004). On NA both feature spaces
  give a borderline result (p ≈ 0.06).

### What this does not establish

- **Generalization gap.** We have not measured downstream model
  performance on these splits. MMD detecting "train and test
  distributions are distinguishable" does not by itself prove a
  generalization gap; it only shows the partition has a property MMD
  is designed to detect.
- **Absolute MMD² interpretability.** The MMD² value 0.042 is not
  meaningful in isolation — it depends on σ, PCA dimension, and the
  feature space. The defensible interpretation is the ratio (× the
  random floor) plus the permutation p-value. Different σ or feature
  space would give a different MMD² value with different ratios.
- **Sample-size invariance.** All numbers are at the 1000-isolate
  subsample (~900 entities per slot). Full-corpus (~20,000 entities
  per slot on cluster_id099 HA) was not run. The ordering may hold or
  may shift at scale.
- **HA vs NA absolute values.** HA's MMD² is roughly 3× NA's at fixed
  σ = 1.0719. This could be (a) genuinely more distribution-shift
  structure in HA's ESM-2 cloud at this id099 routing, (b) σ = 1.0719
  being closer to HA's natural bandwidth than NA's, or (c) some
  combination. We did not run NA with its own natural σ; this is
  unresolved.
- **MMD vs L(π) ranking on this corpus.** We can say L(π) collapsed
  to the partition-shape constant on this data while MMD discriminates
  the routings. We cannot say MMD is "better" in some absolute sense —
  the two metrics measure different things (similarity-graph leakage
  vs distribution shift between train and test entity clouds). Both
  measure properties of the partition; whether a property is desirable
  depends on the use case (cluster_disjoint deliberately maximizes the
  property MMD detects).
- **Generalizability beyond Flu A HA/NA, ESM-2 + aa k=3, and id099.**
  Untested. Other k-mer variants (nt k=3, nt k=6) are also untested.
- **What drives the HA seq_disjoint divergence between feature
  spaces.** Could be a genuine feature-space sensitivity difference
  (aa k=3 catching exact-residue patterns that ESM-2 smooths over),
  or a PCA-50 artifact (8000-dim aa-k3 reduced to 50 may capture
  different variance than 1280-dim ESM-2 reduced to 50). Not
  disentangled here.

---

## Reproduce

```bash
# Phase 1 (wiring sanity, ~2 min):
conda run -n segmatch python src/analysis/mmd_per_slot.py \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_20260520_211534 \
    --slot a \
    --partition_mode fresh_random \
    --n_split_seeds 10 \
    --out_csv results/flu/July_2025/runs/split_separation_mmd/phase1_random_HA.csv

# Phase 2 ESM-2 (dataset labels, fixed sigma = 1.0719, permutation test).
# Six runs, ~2-3 min each.
for SLOT in a b; do
  for PAIR in \
    "random:dataset_flu_ha_na_random_20260520_210647" \
    "seq_disjoint:dataset_flu_ha_na_seq_disjoint_20260520_211109" \
    "cluster_disjoint_aa_id099:dataset_flu_ha_na_cluster_id99_20260520_211534"; do
    LABEL="${PAIR%%:*}"; DSDIR="${PAIR##*:}"
    SNAME=$([ "$SLOT" = "a" ] && echo HA || echo NA)
    conda run -n segmatch python src/analysis/mmd_per_slot.py \
      --dataset_dir data/datasets/flu/July_2025/runs/${DSDIR} \
      --slot ${SLOT} --partition_mode dataset_labels --routing_label ${LABEL} \
      --sigma 1.0719 --n_permutations 500 \
      --out_csv results/flu/July_2025/runs/split_separation_mmd/phase2_perm_${LABEL}_${SNAME}.csv
  done
done

# Phase 2 aa k=3 (same datasets and partitions, sigma = 29.3192).
# Six runs, ~1 min each (faster than ESM-2 — no HDF5 load).
for SLOT in a b; do
  for PAIR in \
    "random:dataset_flu_ha_na_random_20260520_210647" \
    "seq_disjoint:dataset_flu_ha_na_seq_disjoint_20260520_211109" \
    "cluster_disjoint_aa_id099:dataset_flu_ha_na_cluster_id99_20260520_211534"; do
    LABEL="${PAIR%%:*}"; DSDIR="${PAIR##*:}"
    SNAME=$([ "$SLOT" = "a" ] && echo HA || echo NA)
    conda run -n segmatch python src/analysis/mmd_per_slot.py \
      --dataset_dir data/datasets/flu/July_2025/runs/${DSDIR} \
      --slot ${SLOT} --feature_space kmer_aa --kmer_k 3 \
      --partition_mode dataset_labels --routing_label ${LABEL} \
      --sigma 29.3192 --n_permutations 500 \
      --out_csv results/flu/July_2025/runs/split_separation_mmd/phase2_perm_${LABEL}_${SNAME}_kmer_aa_k3.csv
  done
done
```

Outputs (gitignored under `results/`):

- `results/flu/July_2025/runs/split_separation_mmd/phase1_random_HA.csv` — ESM-2 Phase 1.
- `results/flu/July_2025/runs/split_separation_mmd/phase1_random_HA_kmer_aa_k3.csv` — aa k=3 Phase 1.
- `results/flu/July_2025/runs/split_separation_mmd/phase2_perm_{random,seq_disjoint,cluster_disjoint_aa_id099}_{HA,NA}.csv` — ESM-2 Phase 2.
- `results/flu/July_2025/runs/split_separation_mmd/phase2_perm_{...}_{HA,NA}_kmer_aa_k3.csv` — aa k=3 Phase 2.
- `results/flu/July_2025/runs/split_separation_mmd/phase2_fixed_sigma_{random,seq_disjoint,cluster_id099}_{HA,NA}.csv`
  (earlier no-permutation ESM-2 runs; still on disk for traceability)

---

## See also

- `docs/results/2026-05-24_datasail_lpi_results.md` — L(π) leg of the
  same plan; null result on the same three datasets at the same
  scale. Provides the context for why this MMD work was framed as
  complementary.
- `docs/plans/2026-05-22_split_separation_metrics_plan.md` — parent
  plan, both legs.
- `docs/results/2026-05-15_cluster_disjoint_nt_results.md` — 1-NN
  cosine margin baseline; the existing per-pair edge-style
  diagnostic this work complements.
- Gretton, A. et al. (2012). "A Kernel Two-Sample Test." JMLR 13.
  (MMD definition, biased estimator, median heuristic.)
- Phipson, B. & Smyth, G. K. (2010). "Permutation P-values should
  never be zero." SAGMB 9(1). (Add-one smoothing convention.)
