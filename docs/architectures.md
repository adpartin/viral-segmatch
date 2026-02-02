# Architectures: slot-specific subnetworks in schema-ordered datasets

This note captures two related ideas discussed while validating **schema-ordered** datasets (e.g., HA→NA where slot A is always HA and slot B is always NA).

Note on formatting:
- Some renderers do not support LaTeX math. To keep this doc portable, equations are written in plain text below (e.g., "a' = f_A(a)").

## Original suggestion (user): per-slot subnetworks before interaction

**Idea**: Before computing an interaction feature (concat/diff/prod), apply a **different MLP per slot**:

- a' = f_A(a)  (e.g., HA-specific MLP)
- b' = f_B(b)  (e.g., NA-specific MLP)
- then compute interaction features from a', b':
  - concat: [a', b']
  - diff: |a' - b'|
  - prod: a' ⊙ b'

**Schematic (example dims)**:

  - a (1280) -> MLP_A: Linear(1280, 512) -> ReLU -> Linear(512, 256) -> a' (256)
  - b (1280) -> MLP_B: Linear(1280, 512) -> ReLU -> Linear(512, 256) -> b' (256)
  - interaction: concat([a', b']) -> MLP_head -> logits

**Motivation**: HA and NA are different proteins; their embeddings may require different re-parameterizations. Expect \(f_A\) and \(f_B\) to learn different weights.

**Where it fits best**:
- Works naturally in **schema-ordered mode**, where slot identity is semantic by construction.
- Conceptually similar to **domain-specific heads** or **two-tower models** where each tower specializes for a different input domain.

**Main risks**:
- More capacity ⇒ higher overfitting risk (can amplify dataset quirks).
- Less tolerant to any upstream slot mistakes (if A/B are swapped, performance may degrade sharply).

## Suggested refinement (assistant): shared trunk + small slot-specific adapters / normalization

**Idea**: Keep most parameters shared, but allow slot-specific specialization cheaply:

### Option 1 — Shared trunk + slot-specific adapters (residual)
- z_a = g(a), z_b = g(b) with a shared trunk g
- a' = z_a + h_A(z_a), b' = z_b + h_B(z_b) with small adapters h_A, h_B
- Interaction computed on a', b'

**Why**: preserves most sharing (better data efficiency), but still lets HA/NA learn different tweaks.

**Schematic (example dims)**:

  - a (1280) -> g: Linear(1280, 512) -> ReLU -> z_a (512)
  - b (1280) -> g: Linear(1280, 512) -> ReLU -> z_b (512)
  - a' = z_a + Adapter_A(z_a)   # Adapter_A: Linear(512, 128) -> ReLU -> Linear(128, 512)
  - b' = z_b + Adapter_B(z_b)   # Adapter_B: Linear(512, 128) -> ReLU -> Linear(128, 512)
  - interaction: diff(|a' - b'|) or prod(a' ⊙ b') or concat([a', b']) -> MLP_head -> logits

**Related concepts**:
- **Adapters** in NLP (small bottleneck modules)
- **Residual adapters** for efficient specialization
- Related to **mixture-of-experts (MoE)** in spirit (slot-conditioned pathways), but here the routing is fixed by slot, not learned.

### Option 2 — Shared trunk + slot-specific normalization/affine
Use shared \(g\), but allow per-slot LayerNorm / affine scaling (very low parameter count), then interact.

**Why**: often captures “slot identity” distribution shifts with minimal overfitting risk.

**Schematic (example dims)**:

  - a (1280) -> g -> z_a (512) -> LayerNorm_A -> a'
  - b (1280) -> g -> z_b (512) -> LayerNorm_B -> b'
  - interaction -> MLP_head -> logits

**Related concepts**:
- **Conditional normalization** (e.g., FiLM-style affine layers)
- **Domain-specific normalization** for distribution shifts

## Validation notes / expectations

- In schema-ordered datasets, the model is **not required** to be swap-invariant under directed features like [emb_a, emb_b]. Swapping A/B breaks the schema semantics.
- If you compare architectures, keep an eye on:
  - performance on the “hard” subsets (e.g., H3N2, human-only)
  - calibration / PR-AUC (not only accuracy)
  - sensitivity to swapped inputs (expected to degrade in schema mode)

## Recommended ablation ladder (minimal)

1. Baseline: interact on raw embeddings (current)
2. Shared pre-MLP: a' = g(a), b' = g(b)
3. Two-slot MLPs: a' = f_A(a), b' = f_B(b)
4. Shared trunk + adapters (or slot-specific norm) as a “middle ground”

## Where to implement in this codebase

**Short answer**: implement in the **model**, not the dataset.

- **SegmentPairDataset** should keep returning raw embeddings (a, b) as it does today.
- **MLPClassifier / model code** should own the architecture:
  - Add optional pre-MLP blocks (shared or per-slot).
  - Add optional adapters / slot-specific norm.
  - Then compute the interaction feature(s) and pass to the existing classifier head.

**Why not in dataset**:
- Dataset should remain a data loader, not a model.
- Keeping architecture in the model makes ablations easy (config toggles) without regenerating data.

