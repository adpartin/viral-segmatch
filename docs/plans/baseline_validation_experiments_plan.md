# Plan: Baseline Validation Experiments

**Status: DRAFT** — needs discussion before implementation

## Motivation

We need to confirm the model is learning from actual amino-acid / nucleotide-level features
rather than exploiting shortcuts. The concern: the model might learn "slot A = HA, slot B = NA → 1"
(protein-identity pattern) instead of sequence-level co-occurrence signal.

## Existing baseline

- **Label shuffle** (`flu_pb2_pb1_pa_5ks_label_shuffle`): Randomizes labels, expects chance-level
  performance. Confirms the model needs *some* real signal but doesn't tell us *what* signal.

## Proposed baselines

### 1. Embedding shuffle within protein (highest priority)

**What**: Keep labels correct. For each protein (e.g., HA), randomly permute embeddings across
isolates. Each HA embedding is still a real HA embedding, but it no longer corresponds to the
isolate it's paired with.

**Tests**: Whether the model needs isolate-specific sequence information (not just protein identity).

**Expected result**: Performance drops to ~chance. If not, the model is using slot identity as a shortcut.

**Implementation**: Add a `shuffle_embeddings: true` flag to the dataset or training config.
At pair-construction time (or embedding-load time), permute the embedding matrix rows within
each protein group independently.

### 2. Mean embedding per protein

**What**: Replace all HA embeddings with the HA mean vector, all NA with the NA mean.
Every isolate's HA looks identical.

**Tests**: Whether per-isolate variation matters at all.

**Expected result**: Chance-level. This is a stronger version of (1) — it removes *all*
within-protein variation.

**Implementation**: After loading embeddings, compute per-protein mean and broadcast.
Could be a `mean_embeddings: true` flag.

### 3. Random embeddings (preserve distribution)

**What**: Replace each embedding with a random vector sampled from N(μ_protein, σ_protein).
Preserves per-protein statistics but destroys all sequence information.

**Tests**: Whether the model needs any real structure in embeddings beyond gross statistics.

**Expected result**: Chance-level.

**Implementation**: `random_embeddings: true` flag. Sample from per-protein Gaussian.

### 4. Swap-slot test

**What**: For positive pairs, swap which embedding goes in slot A vs slot B
(i.e., put NA in slot A and HA in slot B).

**Tests**: Whether `unit_diff` direction carries signal (it should — A-B ≠ B-A).

**Expected result for unit_diff**: Performance may drop since the sign of the difference flips.
If it doesn't drop, the model learned magnitude, not direction.

**Expected result for concat**: Should be unaffected if the model has learned symmetric features,
or may drop if it's learned slot-position-dependent features.

**Implementation**: Swap columns in the pair dataframe before embedding lookup.

## Priority order

1. **Embedding shuffle within protein** — most informative single experiment
2. **Mean embedding per protein** — quick sanity check, confirms (1)
3. **Swap-slot test** — validates the unit_diff direction hypothesis
4. **Random embeddings** — nice-to-have, mostly confirms (2)

## Implementation approach options

**Option A: Config flags in training/dataset config**
- Add flags like `training.ablation.shuffle_embeddings: true`
- Modify embedding loading code to apply the ablation
- Pro: reproducible via bundle, integrates with existing pipeline
- Con: adds complexity to production code

**Option B: Standalone ablation script**
- Separate script that loads a trained model's dataset, applies the perturbation, and re-evaluates
- Pro: no changes to production code
- Con: doesn't test whether the model *learns* from corrupted data (only tests a trained model)

**Option C: Dataset-level perturbation (at Stage 3)**
- Apply the shuffle/mean/random at pair-creation time, bake into the dataset files
- Pro: clean separation; Stage 4 doesn't need to know about ablations
- Con: creates separate dataset directories for each ablation

**Recommendation**: Option A for (1) and (2) — we want to test whether the model can *learn*
from corrupted features, not just whether a trained model is robust to them. The flag should
apply during training so we train from scratch on corrupted data.

## Open questions

- Should we apply the ablation to train+val+test, or only to train (and evaluate on real test)?
  Training on corrupted + evaluating on real would show a different thing than fully corrupted.
  Recommendation: corrupt everything — the question is "can the model learn from this signal?"
- Should embedding shuffle be done once (fixed permutation) or re-shuffled each epoch?
  Recommendation: fixed permutation (simpler, deterministic).
- For k-mer features: do we run the same baselines, or only ESM-2?
  If k-mer also passes the shuffle test, it confirms the k-mer signal is also sequence-level.

## Bundles to create (after design is finalized)

- `flu_schema_raw_slot_norm_unit_diff_shuffle_emb`
- `flu_schema_raw_slot_norm_unit_diff_mean_emb`
- `flu_schema_raw_slot_norm_unit_diff_swap_slot`
- (optionally k-mer variants of each)
