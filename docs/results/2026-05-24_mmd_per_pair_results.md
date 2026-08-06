# Per-pair (S2) MMD on Flu A HA-NA — does the per-slot signal carry through to the model's pair representation?

S2 sibling to `2026-05-24_mmd_per_slot_results.md`. S1 measured train vs
test distribution shift on a single slot (HA only or NA only). S2 asks
the same question on the **pair feature the MLP actually sees**: the
HA and NA slot embeddings combined via the production interaction
(Test 3 = `slot_transform=unit_norm`, `interaction=unit_diff+prod`).

Two questions this round:

1. Does the random < seq_disjoint < cluster_disjoint id099 trajectory
   that S1 found at the slot level carry through to the pair
   representation? (Direct relevance to the model's input distribution.)
2. Does pair-level mixing amplify or dampen the slot-level shift
   signal?

Same three feature spaces as S1 — ESM-2, aa k=3, nt k=6 — and same
three Flu A HA-NA datasets (random, seq_disjoint, cluster_disjoint
id099). Subsample seed is fixed across all S1 and S2 runs so the same
underlying isolate set is being probed.

## Scope of what was tested

- One virus, one bundle pair: Flu A HA-NA. Generalization to PB2-PB1
  or to non-Flu viruses is not addressed.
- Three split routings: random (per-pair), seq_disjoint (per-entity
  hash), cluster_disjoint id099 (mmseqs2 aa easy-linclust at 99%
  identity). idXX-sweep across 100/099/098/097/095 is the planned
  next experiment, not done here.
- Three feature spaces: ESM-2 (1280-dim, protein-level), aa k=3
  (8000-dim, protein-level), nt k=6 (4096-dim, DNA-level). All three
  go through the same Test 3 pair-interaction pipeline and PCA-50.
- One pair interaction: Test 3 only. Other interactions (plain
  `concat`, `unit_diff` alone, etc.) not measured.
- One subsample: 1000 isolates, fixed seed (42). The S1 doc already
  argued the entity set is stable enough; no resampling was added at
  S2.
- Positives only (label==1) — same convention as S1. Negative pairs
  are constructed by the dataset builder and their distribution
  depends on the negative-sampling regime; the MMD question we are
  asking is about the biological signal in real co-occurring pairs,
  not about negative-construction noise.

## Headline observations

1. **Same trajectory in all three feature spaces.** random ≪
   seq_disjoint < cluster_disjoint id099 in MMD² magnitude and in
   permutation p-value, for ESM-2, aa k=3, and nt k=6.

2. **Pair-level amplifies the seq_disjoint signal.** ESM-2 seq_disjoint
   was borderline at S1 (HA p=0.166, NA p=0.056) but is significant at
   S2 (pair p=0.034). For aa k=3 and nt k=6 the slot-level seq_disjoint
   signal was already at the p≈0.002 floor on HA and is also at the
   floor at the pair level.

3. **cluster_disjoint id099 is at the p-value floor in every cell**
   (0/500 permutations exceeded observed). Consistent with S1 — the
   cluster split produces a distribution shift that is detectable at
   either resolution and in any feature space we tried.

4. **Pair-level discrimination ratios are larger or equal to
   slot-level ratios.** For ESM-2 the cluster/random ratio goes from
   12.1× (HA) / 13.1× (NA) at S1 to 20.7× at S2. For k-mer spaces the
   pair-level ratios are 20–23×, similar to or slightly above the
   slot-level numbers.

5. **All three feature spaces had σ in the same order of magnitude at
   the pair level** (ESM-2 0.36, nt k=6 0.83, aa k=3 1.07), unlike the
   slot level where σ spanned 50× (ESM-2 1.07 vs nt k=6 50.84). Test 3
   normalizes per slot before the interaction; the resulting pair
   features have similar magnitude across feature spaces. Within-row
   ratios and p-values remain the defensible cross-feature comparison
   regardless.

## Setup details

### Per-pair data extraction

- Load `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv` from the
  Stage 3 dataset dir.
- Restrict to positives (`label == 1`).
- Subsample 1000 isolates by `assembly_id_a` with `subsample_seed=42`
  (fixed across all S1/S2 runs).
- Dedup by pair hash:
  - ESM-2 and aa k=3 → dedup by `(seq_hash_a, seq_hash_b)`
    (protein-level; same identical-AA-sequence pair collapses).
  - nt k=6 → dedup by `(dna_hash_a, dna_hash_b)` (DNA-level; allows
    distinct synonymous-codon encodings of the same protein pair to
    remain as separate entities).
- Pair-level ambiguity: if a `(hash_a, hash_b)` combination shows up
  in more than one split (only possible under random per-pair
  routing), it is flagged ambiguous and filtered in `dataset_labels`
  mode. Zero under seq_disjoint and cluster_disjoint by construction.

### Feature pipeline

Same Test 3 transform on every feature space:

```
u, v               = slot_a embedding, slot_b embedding
u_n, v_n           = u / ||u||,  v / ||v||                  # slot_transform=unit_norm
unit_diff          = |u_n - v_n| / ||  |u_n - v_n|  ||     # element-wise abs, then L2-normalize
prod               = u_n * v_n                              # element-wise product
pair_feature       = concat[unit_diff, prod]                # 2D-dim per pair
```

Per-slot dimensions:

| Feature | per-slot D | pair_feature D (= 2D) |
|---|---:|---:|
| ESM-2  |   1280 |  2560 |
| aa k=3 |   8000 | 16000 |
| nt k=6 |   4096 |  8192 |

PCA to 50 dims (`random_state=subsample_seed=42`) is applied to the
pair feature matrix before MMD.

### Bandwidth choice

Median heuristic σ computed on the **Phase 1 cluster_id099 pair set**
for each feature space and held fixed across all Phase 2 runs of that
feature space (matches S1's setup). σ values:

| Feature | σ (Phase 1) | per-slot σ at S1 |
|---|---:|---:|
| ESM-2  | 0.3588 | 1.0719 |
| aa k=3 | 1.0720 | 29.3192 |
| nt k=6 | 0.8269 | 50.8379 |

Pair σ ≪ slot σ for k-mer because Test 3 first L2-normalizes each slot
(so each slot has unit length) then divides `unit_diff` by its own
norm, so each component of the pair feature is in [0,1]-ish range
regardless of the raw feature magnitude. The cross-feature σ span
collapses from ~50× at the slot level to ~3× at the pair level.

### Permutation test

500 train/test label shuffles per Phase 2 run (`permutation_seed=0`).
p-value uses add-one smoothing — minimum reportable p = 1/501 ≈ 0.002.

## Results

### Phase 1 — wiring sanity (fresh-random per-pair 50/50 splits on cluster_id099)

10 random per-pair 50/50 splits on the cluster_id099 pair set, one
feature space per run. All entities found in cache (0 misses).

| Feature | dedup | n_pairs | σ (median) | mean MMD² | std MMD² | range |
|---|---|---:|---:|---:|---:|---:|
| ESM-2  | (seq_hash_a, seq_hash_b) | 1000 | 0.3588 | +0.00145 | 0.00089 | [+0.00048, +0.00339] |
| aa k=3 | (seq_hash_a, seq_hash_b) | 1000 | 1.0720 | +0.00126 | 0.00062 | [+0.00059, +0.00278] |
| nt k=6 | (dna_hash_a, dna_hash_b) | 1000 | 0.8269 | +0.00135 | 0.00057 | [+0.00069, +0.00262] |

All three feature spaces pass the noise-floor sanity check at S2.
Mean MMD² and std are within a factor of ~2 of each other across
feature spaces — closer agreement than at the slot level, consistent
with the σ-collapse observation.

n_pairs = 1000 for all three feature spaces — at this isolate
subsample, each isolate's (HA, NA) pair appears to be unique enough
that dedup by protein and dedup by DNA both yield 1000 distinct pairs.

### Phase 2 — dataset labels with fixed σ and 500-permutation p-value

All Phase 2 runs use the same train (n=823) and test (n=99) pair
counts after the val-exclude and ambiguous-filter. The ambiguous
filter dropped zero pairs in seq_disjoint and cluster_disjoint and a
small number in random (≤1% of the pair set; matches the S1
slot-level pattern).

**Phase 2 (ESM-2 + Test 3, fixed σ = 0.3588):**

| Routing | n_train | n_test | MMD² | × random | n_extreme/500 | p-value |
|---|---:|---:|---:|---:|---:|---:|
| random | 823 | 99 | +0.002733 | 1.0× | 352 | 0.7046 |
| seq_disjoint | 823 | 99 | +0.012670 | 4.6× | 16 | **0.0339** |
| cluster_disjoint id099 | 823 | 99 | +0.056723 | 20.7× | **0** | **0.0020** |

**Phase 2 (aa k=3 + Test 3, fixed σ = 1.0720):**

| Routing | n_train | n_test | MMD² | × random | n_extreme/500 | p-value |
|---|---:|---:|---:|---:|---:|---:|
| random | 823 | 99 | +0.002559 | 1.0× | 358 | 0.7166 |
| seq_disjoint | 823 | 99 | +0.017997 | 7.0× | **0** | **0.0020** |
| cluster_disjoint id099 | 823 | 99 | +0.051194 | 20.0× | **0** | **0.0020** |

**Phase 2 (nt k=6 + Test 3, fixed σ = 0.8269):**

| Routing | n_train | n_test | MMD² | × random | n_extreme/500 | p-value |
|---|---:|---:|---:|---:|---:|---:|
| random | 823 | 99 | +0.002927 | 1.0× | 370 | 0.7405 |
| seq_disjoint | 823 | 99 | +0.022074 | 7.5× | **0** | **0.0020** |
| cluster_disjoint id099 | 823 | 99 | +0.068907 | 23.5× | **0** | **0.0020** |

### Side-by-side comparison across feature spaces

| Routing | ESM-2 p | aa k=3 p | nt k=6 p | Agreement |
|---|---:|---:|---:|---|
| random | 0.7046 | 0.7166 | 0.7405 | all non-significant |
| seq_disjoint | **0.0339** | **0.0020** | **0.0020** | sensitivity ordering: ESM-2 < aa k=3 ≈ nt k=6 (k-mer at floor) |
| cluster_disjoint id099 | **0.0020** | **0.0020** | **0.0020** | all at floor |

| Routing | ESM-2 | aa k=3 | nt k=6 |
|---|---:|---:|---:|
| seq_disjoint / random | 4.6× | 7.0× | **7.5×** |
| cluster_disjoint / random | 20.7× | 20.0× | **23.5×** |

(Within-feature-space ratios. Absolute MMD² across feature spaces is
not comparable since σ differs.)

### Comparison to slot-level (S1) numbers

Side-by-side of S2 pair-level vs S1 slot-level results, using the
ESM-2 baseline as the reference (k-mer spaces show the same pattern).

| Routing | S1 HA p | S1 NA p | S2 pair p |
|---|---:|---:|---:|
| random | 0.639 | 0.864 | 0.7046 |
| seq_disjoint | 0.166 | 0.056 | **0.0339** |
| cluster_disjoint id099 | 0.002 | 0.002 | 0.0020 |

| Routing | S1 HA ratio | S1 NA ratio | S2 pair ratio |
|---|---:|---:|---:|
| seq_disjoint / random | 3.0× | 4.5× | 4.6× |
| cluster_disjoint / random | 12.1× | 13.1× | 20.7× |

ESM-2 pair-level pushes the seq_disjoint signal across the standard
significance threshold (HA p=0.166 → pair p=0.034) and pushes the
cluster_disjoint discrimination ratio higher (12×–13× → 21×). The
slot-level signal is preserved and concentrated at the pair level,
which is the resolution the model actually trains on.

The same comparison for aa k=3 and nt k=6 is less dramatic on
seq_disjoint because both k-mer spaces were already at the p-floor
on HA at S1 — there is no headroom for amplification at p-value
level — but cluster_disjoint ratios still grow:

| Feature | Slot ratio (HA cluster/random) | Pair ratio (cluster/random) |
|---|---:|---:|
| aa k=3 | 18.6× | 20.0× |
| nt k=6 | 23.1× | 23.5× |

## Interpretation — what we can and cannot claim

### What the empirical results show

- For Flu A HA-NA, the same train-vs-test distribution shift that S1
  detected at the single-slot level is **also detected at the pair
  level the model trains on**, in all three feature spaces, with the
  same routing ordering.
- The ESM-2 pair representation under Test 3 amplifies the
  seq_disjoint signal enough to cross the 0.05 threshold (p=0.034 at
  S2 vs p=0.166 at S1 HA), and roughly doubles the cluster_disjoint /
  random discrimination ratio (12× → 21×). This is direct evidence
  that the production model's input distribution differs between
  train and test under both entity-disjoint routings, with the shift
  being severe under cluster_disjoint id099.
- The cluster_disjoint id099 effect is at the p-value floor (0/500
  permutations) for every (feature × resolution) cell measured.
- nt k=6 has the largest cluster/random ratio at the pair level
  (23.5×), matching its slot-level behavior. We did not disentangle
  whether this comes from synonymous-codon information, DNA-level
  entity-set size, or kernel/PCA interaction effects.

### What this does not establish

- **Not a model-performance claim.** MMD measures distribution shift
  of features; it does not directly predict generalization gap, AUC
  drop, or any downstream metric. The planned idXX-sweep experiment
  is what would connect MMD trajectories to actual model performance.
- **Single subsample, single seed.** Subsample size = 1000 isolates,
  `subsample_seed=42`, `permutation_seed=0`. No resampling of either.
  The Phase 1 random-baseline std numbers give a rough sense of MMD²
  noise from the partition step, but not from the subsample step.
- **No PB2-PB1 or other bundle tested at S2.** The story is currently
  about HA-NA only.
- **Test 3 only.** Other interactions and slot transforms not tested;
  it is possible plain concat or `diff` alone would produce different
  MMD trajectories. Test 3 was chosen because it is what the active
  production HA/NA and PB2/PB1 bundles use.
- **Cluster split ceiling.** Per Experiment B-nt findings
  (`docs/results/2026-05-15_cluster_disjoint_nt_results.md`), only
  id100 and id099 are operable on the full Flu A corpus; id095 and
  below collapse most pairs into one bipartite component. The
  symmetric easy-linclust switch (BACKLOG Algorithm-switch #1) may
  unlock lower thresholds; that is part of the planned next
  experiment.

## Reproduce

```bash
mkdir -p results/flu/July_2025/runs/split_separation_mmd

# Phase 1 wiring sanity (10 random 50/50 splits on cluster_id099),
# one run per feature space:

python src/analysis/mmd_per_pair.py \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_20260520_211534 \
    --partition_mode fresh_random \
    --feature_space esm2 \
    --out_csv results/flu/July_2025/runs/split_separation_mmd/phase1_random_HA_NA_pair_esm2_test3.csv

python src/analysis/mmd_per_pair.py \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_20260520_211534 \
    --partition_mode fresh_random \
    --feature_space kmer_aa --kmer_k 3 \
    --out_csv results/flu/July_2025/runs/split_separation_mmd/phase1_random_HA_NA_pair_kmer_aa_k3_test3.csv

python src/analysis/mmd_per_pair.py \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_cluster_id99_20260520_211534 \
    --partition_mode fresh_random \
    --feature_space kmer_nt --kmer_k 6 \
    --out_csv results/flu/July_2025/runs/split_separation_mmd/phase1_random_HA_NA_pair_kmer_nt_k6_test3.csv

# Phase 2 (dataset labels, fixed sigma per feature space, 500-permutation).
# Three routings per feature space. Pass each feature-space sigma from Phase 1.
# ESM-2 sigma = 0.3588, aa k=3 sigma = 1.0720, nt k=6 sigma = 0.8269.

for routing_pair in \
    "random:dataset_flu_ha_na_random_20260520_210647" \
    "seq_disjoint:dataset_flu_ha_na_seq_disjoint_20260520_211109" \
    "cluster_disjoint_id099:dataset_flu_ha_na_cluster_id99_20260520_211534"; do
  label="${routing_pair%%:*}"
  dirname="${routing_pair##*:}"

  python src/analysis/mmd_per_pair.py \
      --dataset_dir data/datasets/flu/July_2025/runs/$dirname \
      --partition_mode dataset_labels --routing_label $label \
      --feature_space esm2 --sigma 0.3588 --n_permutations 500 \
      --out_csv results/flu/July_2025/runs/split_separation_mmd/phase2_perm_${label}_HA_NA_pair_esm2_test3.csv

  python src/analysis/mmd_per_pair.py \
      --dataset_dir data/datasets/flu/July_2025/runs/$dirname \
      --partition_mode dataset_labels --routing_label $label \
      --feature_space kmer_aa --kmer_k 3 --sigma 1.0720 --n_permutations 500 \
      --out_csv results/flu/July_2025/runs/split_separation_mmd/phase2_perm_${label}_HA_NA_pair_kmer_aa_k3_test3.csv

  python src/analysis/mmd_per_pair.py \
      --dataset_dir data/datasets/flu/July_2025/runs/$dirname \
      --partition_mode dataset_labels --routing_label $label \
      --feature_space kmer_nt --kmer_k 6 --sigma 0.8269 --n_permutations 500 \
      --out_csv results/flu/July_2025/runs/split_separation_mmd/phase2_perm_${label}_HA_NA_pair_kmer_nt_k6_test3.csv
done
```

ESM-2 runs take ~4 min each (HDF5 load dominates); k-mer runs take
~1 min each.

## See also

- `docs/results/2026-05-24_mmd_per_slot_results.md` — S1 per-slot
  baseline this S2 work compares against.
- `docs/results/2026-05-24_datasail_lpi_results.md` — DataSAIL L(π)
  leakage metric, negative result on this corpus.
- `docs/plans/2026-05-22_split_separation_metrics_plan.md` — overall
  plan for quantifying split separation.
- `src/analysis/mmd_per_pair.py` — code that produced these results.
- `src/analysis/mmd_per_slot.py` — S1 code; shares the kernel / MMD /
  permutation primitives.
