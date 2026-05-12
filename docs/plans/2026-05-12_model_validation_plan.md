# Model validation — plan

**Status: MIXED**
- **Part A (training-mechanics sanity checks):** ✅ IMPLEMENTED — lives in
  `src/utils/learning_verification_utils.py`, exercised by the
  `flu_a_learning_test` bundle.
- **Part B (signal-source adversarial ablations):** ⏳ DRAFT — needs
  discussion before implementation.

**Date:** 2026-05-12 (merge of the older `LEARNING_VERIFICATION.md`
reference doc + the `baseline_validation_experiments_plan.md` draft).

## Why this matters

We need two independent sources of evidence that the model is learning
something real about isolate-level co-occurrence:

1. **The training pipeline works mechanically.** Neural networks can fail
   silently — preprocessing bugs (flipped labels), wrong loss, bad
   initialization, data leakage. Part A is a set of cheap Karpathy-style
   checks that catch this class of failure.
2. **The model relies on the signal we care about, not a shortcut.**
   Even with healthy training metrics, the model could be exploiting
   protein-identity (slot A = HA → 1), metadata coincidence (same host /
   subtype / year), or distributional statistics rather than
   isolate-specific sequence signal. Part B is a set of *negative-control
   ablations* that corrupt the signal in a known way and check that the
   model fails the way we'd expect a real learner to fail.

The two parts target different failure modes and are complementary;
passing one does not substitute for passing the other.

---

# Part A — Training-mechanics sanity checks (IMPLEMENTED)

Karpathy-style verification that the training pipeline is mechanically
correct. Source: [Karpathy's "Neural network training
recipe"](https://karpathy.github.io/2019/04/25/recipe/).

Implementation: `src/utils/learning_verification_utils.py` (exports
`check_initialization_loss`, `compute_baseline_metrics`,
`plot_learning_curves`). Driven from `train_pair_classifier.py` when
the `flu_a_learning_test` bundle is active.

### A1. Initialization-loss check

- For balanced binary classification, untrained-model loss should be
  ≈ `−log(0.5) ≈ 0.693`.
- A wildly different starting loss signals data-imbalance or
  model-setup bugs (wrong final layer, miscomputed loss, etc.).
- Implementation: `check_initialization_loss()`.

### A2. Small-dataset overfit test

- Bundle: `flu_a_learning_test.yaml` (100 isolates, 10 epochs,
  patience=3).
- If the model **can** overfit a tiny dataset, training is wired up
  correctly. If it can't, something deeper is broken.
- An overfitting pattern (training loss low, val loss high) is the
  *desired* outcome at this scale — confirms learning.

### A3. Baseline comparison

- Random-classifier F1 and majority-class F1 are computed at training
  start (`compute_baseline_metrics()`).
- Model must beat both. If it can't, training is producing nothing
  useful.

### A4. Learning curves

- `plot_learning_curves()` writes `learning_curves.png` to the run
  directory with train/val loss + val F1 + val AUC per epoch.
- Used as a quick visual: are curves moving in the right direction?

### What "pass" looks like

- Initialization loss ≈ 0.693.
- Training loss decreases over epochs.
- Validation F1 improves before early stopping.
- Model beats both baselines.

### What "fail" looks like

- Initialization loss off by >2× expected.
- Training loss flat or rising.
- Validation F1 stays at ~0.5 (random) or matches majority class.
- All predictions are the same class.

### Status

Implementation has been in place since gen-3. Used as a pre-flight
check for new bundle configurations; not run on every production
training run (the gen-3 production bundles have stable mechanics now).

---

# Part B — Signal-source adversarial ablations (DRAFT)

The Part-A checks tell us the model is *learning something*; they do
**not** tell us *what* signal it's using. The model could be acing the
mechanical checks while exploiting:

- **Protein identity** — slot A is always HA, slot B is always NA;
  some learned mapping `(HA, NA) → 1` could dominate.
- **Distributional shortcuts** — gross per-protein statistics
  (mean / variance per dim) might suffice without isolate-level
  resolution.
- **Slot-direction artifacts** — for asymmetric interactions like
  `unit_diff`, A↔B swaps could be informative or destructive depending
  on what the model learned.

The Part-B ablations probe each of these by deliberately destroying
the candidate signal and re-training. If accuracy stays high, the
model is using a shortcut. If it collapses to chance, the destroyed
signal was the one being used.

### Existing baseline (already in use)

- **Label shuffle** (bundle `flu_pb2_pb1_pa_5ks_label_shuffle`):
  permutes labels in train. Confirms the model needs *some* real
  signal to perform — but doesn't tell us *which* signal.

### Proposed new ablations

#### B1. Embedding shuffle within protein (highest priority)

- **What.** Keep labels correct. For each protein function (e.g., HA),
  randomly permute embeddings across isolates. Each HA embedding is
  still a real HA embedding, just from a different isolate than the
  one it's paired with.
- **Tests.** Whether the model needs isolate-specific sequence signal
  beyond protein-identity-as-a-feature.
- **Expected.** Performance drops toward chance. If it doesn't, the
  model is using slot identity as the shortcut.
- **Bundle name (proposed).** `flu_ha_na_shuffle_emb`.

#### B2. Mean embedding per protein

- **What.** Replace every HA embedding with the global HA mean vector
  (and likewise for NA). Every isolate's HA looks identical.
- **Tests.** Stronger version of B1 — removes all within-protein
  variation.
- **Expected.** Chance.
- **Bundle name (proposed).** `flu_ha_na_mean_emb`.

#### B3. Random embeddings preserving per-protein Gaussian

- **What.** Replace each embedding with `Normal(mean_protein,
  std_protein)`. Preserves per-protein first-and-second-order
  statistics; destroys all sequence-derived structure.
- **Tests.** Whether the model needs any real structure beyond gross
  per-protein statistics.
- **Expected.** Chance. Mostly confirms B2.
- **Bundle name (proposed).** `flu_ha_na_random_emb`.

#### B4. Swap-slot test

- **What.** For positive pairs, swap which embedding goes in slot A
  vs slot B (HA → slot B, NA → slot A).
- **Tests.** Whether the direction of the interaction term carries
  signal. For `unit_diff` it should — the abs-based diff is
  symmetric, but the *learned* downstream MLP is not necessarily
  invariant under slot swap.
- **Expected.** For `unit_diff + prod` (current production): both
  terms are symmetric in `a, b`, so a clean implementation should be
  invariant. A drop here would indicate the model learned a
  slot-position-dependent feature that shouldn't exist under the
  current interaction.
- **Bundle name (proposed).** `flu_ha_na_swap_slot`.

### Priority order

1. **B1** (embedding shuffle within protein) — most informative single
   experiment.
2. **B2** (mean embedding per protein) — quick sanity check on B1.
3. **B4** (swap-slot) — validates the interaction-symmetry property
   of the current production setting.
4. **B3** (random embeddings) — nice-to-have; mostly confirms B2.

### Implementation approach

Three options were discussed in the source draft:

- **Option A: config flags in training / dataset config.** Add flags
  like `training.ablation.shuffle_embeddings: true`; embedding-load
  code applies the ablation. Pros: reproducible via bundle,
  integrates with existing pipeline. Cons: adds complexity to
  production code.
- **Option B: standalone post-hoc script.** Separate script that
  loads a trained model + dataset, applies the perturbation, and
  re-evaluates. Pros: no production-code changes. Cons: doesn't test
  whether the model can *learn* from corrupted data — only tests a
  trained model's robustness to it.
- **Option C: dataset-level perturbation (at Stage 3).** Bake the
  shuffle into the dataset files. Pros: clean separation. Cons:
  creates a separate dataset directory per ablation.

**Recommendation: Option A for B1 and B2.** The question is "can the
model learn from this signal?" — Option B answers a different
question. Option A is preferred over Option C because we don't want
ablation datasets cluttering `data/datasets/runs/`.

### Open questions

- **Train-only vs everywhere.** Apply the ablation to train + val +
  test, or only to train (eval on real test)?
  - *Recommendation:* corrupt everything. We want to test whether the
    model can learn from corrupted input; if train is corrupted but
    test is real, we're measuring a different thing.
- **Fixed vs per-epoch shuffle.** For embedding shuffle, fix the
  permutation or re-shuffle each epoch?
  - *Recommendation:* fixed (simpler, deterministic, easier to debug).
- **k-mer features.** Run the same ablations on k-mer features too?
  - *Recommendation:* yes, eventually. If k-mer passes the shuffle
    test, it confirms the k-mer signal is also isolate-level rather
    than protein-level.

### What pass/fail means at this level

- **Pass** (model is genuinely learning sequence-level co-occurrence):
  B1, B2, B3 all collapse to chance; B4 stays at full accuracy if
  the interaction is symmetric.
- **Fail (likely scenarios):**
  - B1 stays high → model uses slot identity as shortcut.
  - B2 stays high → model uses per-protein distributional statistics.
  - B4 drops on a symmetric interaction → bug in interaction
    implementation, or downstream MLP picks up a slot-position
    artifact.

---

## Cross-references

- `src/utils/learning_verification_utils.py` — Part A implementation.
- `docs/methods/leakage_definitions.md` — broader leakage taxonomy;
  this plan's Part B targets mode #5 (demographic shortcut leakage)
  and mode #4 (cluster leakage) from a different angle than the
  existing post-hoc heatmap.
- `docs/methods/feature_normalization.md` — current production setting
  (`unit_norm + unit_diff + prod`) is the configuration these
  ablations should be run against.
- `roadmap_v2.md` — Task 12 (adversarial / IRM mitigations) covers a
  separate mitigation direction; this plan is *measurement*, that
  one is *intervention*.
