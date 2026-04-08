# Stage 4 Training Speed-Up Analysis

Baseline: k-mer k=6, slot_norm + concat, full dataset (~111K isolates), 100 epochs forced.
Per-epoch baseline: **97.1s mean** (Lambda, single V100/A100 GPU).

> **See also:** `docs/hardware_notes.md` for the *why* behind the configuration choices
> below — especially `pin_memory`, `num_workers`, and the multi-process effects on Polaris
> that this document (single-fold Lambda benchmarks) does not cover.

---

## 1. Skip full training metrics re-evaluation

**Problem:** Every epoch runs a second full forward pass over the entire training set
(lines 597–623 in `train_pair_classifier.py`) to compute train F1, AUC, precision, recall,
Brier. This roughly **doubles epoch time** — the training forward/backward pass and the
train eval pass are similar cost.

**Fix:** Add `training.eval_train_metrics: true|false` config flag (default `true` for
backward compat). When `false`, skip the second pass. `train_loss` is still tracked from
the training loop itself — the only loss is per-epoch train F1/AUC/etc.

**When to use each mode:**
- `true` (default): Understanding model behavior, debugging, early experiments.
- `false`: Production/HPC runs where the model configuration is already validated.

**Estimated speedup:** ~45–50% of epoch time (second pass is nearly identical cost to
the first, minus the backward pass).

**Status: WORKS.** Tested on 5K dataset (flu_6mer_5k, 20 epochs). Clear per-epoch speedup
even at small scale; impact will be more significant with larger datasets and more epochs.

---

## 2. Larger inference batch size (`infer_batch_size`)

**Problem:** Eval-mode forward passes (train metrics, val metrics, test metrics) use the
same `batch_size` as training. During inference (`torch.no_grad()`), there's no gradient
memory, so 2–4x larger batches fit in VRAM. Larger batches reduce DataLoader overhead and
GPU kernel launch overhead.

**Fix:** Add `training.infer_batch_size: null` config param. When `null`, falls back to
`batch_size`. Use it for all eval passes (train, val, test).

**Independent of optimization 1** — applies even when `eval_train_metrics` is enabled,
and also speeds up val/test evaluation.

**How large can `infer_batch_size` be?** The limit is GPU VRAM: `model weights + activations
+ batch tensors` must fit. For our MLP (a few MB of weights) with k-mer inputs (4096-dim,
~32KB per pair for two slots), even the full training set (~90K pairs ≈ 2.9GB) fits in a
single batch on a 32GB V100 or 40GB A100. `infer_batch_size: 8192` or `16384` is safe for
all current scenarios. Returns plateau once there are only a few batches per eval pass.

**Estimated speedup:** 5–15% on eval passes (depends on batch size headroom).

**Status: IMPLEMENTED.** Tested on 5K dataset (flu_6mer_5k, 20 epochs). Per-epoch speed up.

---

## 3. DataLoader: `num_workers` + `pin_memory`

**Problem:** Current DataLoaders (lines 1126–1128) use `num_workers=0` (main thread loads
data) and no `pin_memory`. GPU sits idle during CPU→GPU data transfers.

**Fix:** `DataLoader(..., num_workers=N, pin_memory=True)`

### `pin_memory=True`
Allocates CPU tensors in page-locked memory, enabling async CPU→GPU DMA transfers.
Almost always beneficial on GPU. Only downside is pinned RAM consumption — not an issue
on Lambda (plenty of RAM) or Polaris (512GB/node). **Use everywhere.**

### `num_workers`
How hardware and workload affect the optimal value:

| Factor | Effect | Guidance |
|--------|--------|----------|
| Data already in RAM | Embeddings/k-mers preloaded as numpy arrays. `__getitem__` is array indexing + tensor conversion — very cheap. | Fewer workers needed (2–4) |
| Batch size | Larger batches → fewer batches/epoch → less DataLoader overhead | Less critical at large batch sizes |
| Sample size | ESM-2: 1280-dim, k-mer: 4096-dim — small tensors | Lightweight; workers help less than with images |
| CPU cores | Lambda: 32+ cores. Polaris: 32-core AMD EPYC/node | Plenty of headroom |
| Shared filesystem | Data preloaded in RAM, not read per-batch from NFS | No NFS bottleneck |
| Multi-fold on same node | 10 folds on 4 GPUs (Polaris): workers compete for CPU cores | Use `num_workers=2` to avoid contention |

**Recommendation (validated):** `num_workers=0`. `pin_memory=True` only on single-process
Lambda runs; `pin_memory=False` on multi-process Polaris ensemble packing (see Section 8
below and `docs/hardware_notes.md` §1). The "set num_workers to CPU count" rule of thumb
does *not* apply here — it assumes a disk-bound image pipeline, not in-memory numerical data.

**Config:** `training.num_workers: 0` (also hard-coded in `train_pair_classifier.py` for
correctness — see comment there about `torch.from_numpy` + forked workers).

**Estimated speedup:** 5–15% (modest because `__getitem__` is lightweight).

**Status: `num_workers` HURT, `pin_memory` SPEED UP as compared to baseline.
** `num_workers=2` with `pin_memory=true` increased per-epoch time on 5K dataset.
Our `__getitem__` is just numpy array indexing — the IPC overhead (process spawning,
pickling, queue management) exceeds the benefit since there is no disk I/O to
overlap with GPU computation. `num_workers > 0` pays off for expensive `__getitem__`
(e.g., loading images from disk), not in-memory lookups. Keeping `num_workers: 0`
as default. `pin_memory` alone helps in isolation (async DMA, no IPC overhead).

---

## 4. Automatic Mixed Precision (AMP)

**Problem:** All computation runs in float32. A100s and V100s have Tensor Cores that
accelerate float16 matrix multiplications.

**Fix:** `torch.amp.autocast('cuda')` around the forward pass + `GradScaler` for training.
Inference needs `autocast` only (no scaler).

### Hardware differences
- **A100 (Polaris):** ~2x throughput with AMP (3rd gen Tensor Cores, native BF16/FP16).
- **V100 (Lambda):** ~1.3–1.5x (1st gen Tensor Cores).

### Effect on predictions
For an MLP (Linear + ReLU + Dropout + LayerNorm), the impact is negligible. Float16 has
~3.3 decimal digits of precision — more than sufficient. `autocast` keeps loss computation
(`BCEWithLogitsLoss`) in float32 by default.

### Use in training and inference?
**Both.** Training uses autocast + GradScaler. Inference uses autocast only. Free speedup,
near-zero risk for MLP classifiers.

### Reasons NOT to use
- Numerically sensitive operations (small differences, large reductions) — not our case.
- Debugging NaN issues becomes slightly harder.
- BCEWithLogitsLoss with extreme logits — handled by autocast keeping loss in float32.

**Config:** `training.use_amp: false` (default off, opt-in for production).

**Estimated speedup:** 20–30% on forward/backward (more on A100 than V100).

**Status: HURT (V100, small MLP, batch_size=16).** AMP caused a small slowdown on
5K dataset with V100. Disabling AMP restored original per-epoch time, confirming the
overhead is from AMP itself (not a code regression). Root cause: our MLP is too small
and memory-bound for AMP to help.
- **Small matrices**: Layers [4096→512→256→64→1] with batch_size=16 produce tiny GEMMs
  that don't saturate Tensor Cores. Dtype casting overhead (float32↔float16) dominates.
- **Memory-bound, not compute-bound**: Small MLPs are bottlenecked by memory bandwidth,
  not arithmetic. AMP only accelerates arithmetic.
- **GradScaler overhead**: Scaling, unscaling, and inf-checking every step adds cost
  that exceeds savings on cheap forward/backward passes.
- **V100 first-gen Tensor Cores**: Higher overhead than A100, less efficient on small ops.

May be worth retesting on A100 with larger batch sizes on the full dataset, but unlikely
to help for this model architecture. AMP pays off for large models (transformers, deep
convnets) with large batches and matrix dimensions.

---

## 5. `torch.compile`

**What it does:** Traces the model's forward pass, fuses operations (e.g., Linear → ReLU →
Dropout into one kernel), eliminates Python overhead, and optimizes memory access patterns.
JIT compiler for the computation graph.

**Fix:** `model = torch.compile(model)` after `model.to(device)`.

### Reasons NOT to use
- **Compilation overhead:** First call takes 10–30s (one-time). Negligible for 100-epoch
  runs; annoying for 5-epoch debug runs.
- **PyTorch version:** Requires 2.0+. Check installed version on both systems.
- **Dynamic shapes:** Recompilation on shape changes. Batches are fixed size (except last
  batch per epoch) — minor.
- **Debugging:** Compiled model tracebacks are harder to read.
- **Compatibility:** Our MLP ops (Linear, ReLU, Dropout, LayerNorm) are fully supported.

**Config:** `training.compile_model: false` (default off, opt-in for production).

**Estimated speedup:** 10–20% for MLPs.

**Status: SKIPPED.**

---

## 6. Batch GPU→CPU transfers

**Problem:** Eval loops call `.cpu().numpy()` 3 times per batch (lines 611–613, 640–642):
```python
train_probs.extend(torch.sigmoid(preds).cpu().numpy())
train_preds.extend((torch.sigmoid(preds) > 0.5).float().cpu().numpy())
train_labels.extend(batch_y.cpu().numpy())
```

**Fix:** Collect tensors on GPU, transfer once after the loop:
```python
all_logits.append(preds)
all_labels.append(batch_y)
# After loop:
all_logits = torch.cat(all_logits)
all_probs = torch.sigmoid(all_logits).cpu().numpy()
```

### Reasons NOT to do it
None. GPU memory for holding all predictions is trivial (~60KB for 15K test pairs × 1 float).
Pure code improvement.

**No config flag needed** — always better.

**Estimated speedup:** 5–10% on eval passes.

**Status: WORKS (small improvement).** Collect logits and labels on GPU via `torch.cat`,
transfer once after the loop. Applied to all three eval loops (train eval, val eval,
`evaluate_on_split`). Small runtime improvement on 5K dataset; expected to be more
significant with the full dataset where eval passes process more batches.

---

## Summary

| # | Optimization | Config flag | Default | Result | Notes |
|---|---|---|---|---|---|
| 1 | Skip train metrics re-pass | `eval_train_metrics` | `false` | WORKS | ~45–50% epoch savings |
| 2 | Larger inference batch size | `infer_batch_size` | `null` (= batch_size) | WORKS | Speedup on eval passes |
| 3 | `num_workers` + `pin_memory` | `num_workers`, `pin_memory` | `0`, `false` | `num_workers` HURT; `pin_memory` HELPED on single-fold Lambda but HURT (~300×) under multi-fold Polaris ensemble packing — see `docs/hardware_notes.md` §1 | In-memory data makes workers overhead-only |
| 4 | AMP (mixed precision) | `use_amp` | `false` | HURT | MLP too small, memory-bound not compute-bound |
| 5 | `torch.compile` | `compile_model` | `false` | SKIPPED | |
| 6 | Batch GPU→CPU transfers | (none) | Always on | WORKS (small) | Expect bigger gain on full dataset |
| -- | Larger `batch_size` | `batch_size` | `16` | **BEST** | 60% faster at bs=128; also a HP |

---

## Empirical Results (5K dataset, 20 epochs, Lambda V100)

All runs use `flu_6mer_5k` bundle (k-mer k=6, HA/NA, 5K isolates).

| Run | batch_size | eval_train | infer_bs | num_workers | pin_mem | use_amp | Mean epoch | vs baseline |
|-----|-----------|------------|----------|------------|---------|---------|-----------|-------------|
| `_150845` | 16 | true | 16 | 0 | false | false | 4.26s | baseline |
| `_163314` | 16 | true | 16 | 0 | false | false | 4.25s | baseline |
| `_171110` | 16 | true | 16 | 0 | false | false | 4.13s | 3% faster (method 6 code) |
| `_154800` | 16 | true | 16 | 0 | **true** | false | 3.51s | 18% faster |
| `_151638` | 16 | **false** | **256** | 0 | false | false | 3.22s | 24% faster |
| **`_171746`** | **128** | true | 128 | 0 | false | false | **1.68s** | **60% faster** |
| `_162802` | 16 | true | 16 | 0 | false | **true** | 5.30s | 25% slower |
| `_154042` | 16 | true | 16 | **2** | true | false | 7.97s | 87% slower |

### Key takeaways

- **`batch_size=128` is the single biggest win** — 60% faster (1.68s vs 4.26s). Reduces
  optimizer steps and DataLoader iterations by 8x, improves GPU utilization with larger
  matrix operations. Note: batch_size is also a hyperparameter that affects convergence
  (larger batches may need higher learning rate).
- **`eval_train_metrics=false` + `infer_batch_size=256`** — 24% faster. Best pure-speed
  optimization with no HP implications.
- **`pin_memory=true`** — 18% faster. Free win, no downsides.
- **Method 6 (batched GPU→CPU transfers)** — ~3% improvement. Small on 5K, expected to
  scale with larger datasets.
- **`use_amp=true`** — 25% slower. MLP too small for Tensor Core benefits.
- **`num_workers=2`** — 87% slower. IPC overhead dominates for in-memory data.

### Recommended production config

Combining the winners for maximum speed:
```yaml
training:
  batch_size: 128          # Or tune as HP (64/128/256)
  eval_train_metrics: false
  infer_batch_size: 8192
  pin_memory: true         # Single-fold workstation only.
                           # Set FALSE for multi-fold ensemble packing on HPC
                           # (Polaris 4 folds/node) — cudaHostAlloc serializes
                           # across processes. See docs/hardware_notes.md §1.
  num_workers: 0
  use_amp: false
```

### Note on batch size alignment and GPU Tensor Cores

The convention of using powers of 2 for batch sizes comes from GPU Tensor Core tile sizes.
V100 Tensor Cores operate on 8×8 tiles; A100 on 16×16. Matrix dimensions that are multiples
of 8 (V100) or 16 (A100) map cleanly to these tiles — non-aligned dimensions get padded,
wasting cycles.

- **`batch_size`**: Matters for training — feeds into matmuls (batch × features @ features ×
  hidden). Multiples of 8 are sufficient; powers of 2 are not required. `batch_size=96` is
  as efficient as 128 on Tensor Cores. Avoid dimensions like 65 or 500 that require padding.
- **Hidden layer dims**: Same principle. `[512, 256, 64]` (current) are ideal. `[384, 192, 96]`
  would also be fine (multiples of 8). `[500, 250, 65]` would not.
- **`infer_batch_size`**: Alignment doesn't matter in practice. The MLP forward pass is so
  cheap that tile padding effects are negligible. Any large round number works.

---

## Full Dataset Results (~111K isolates, 100 epochs, Lambda V100)

All runs use `flu_6mer_full` bundle (k-mer k=6, HA/NA, full dataset). The baseline
(batch_size=16) uses default params. The optimized runs (128/256/512) additionally use
`eval_train_metrics=false`, `infer_batch_size=1024`, `pin_memory=true`.

| Run | batch_size | Mean epoch | Total runtime | vs baseline | F1 | AUC-ROC | PR-AUC |
|-----|-----------|-----------|---------------|-------------|-----|---------|--------|
| `_181009` (baseline) | 16 | 89.71s | 2h 30m 35s | — | 0.9612 | 0.9927 | 0.9742 |
| `_181120` | **128** | **29.21s** | **49m 48s** | **67% faster** | 0.9656 | 0.9940 | 0.9798 |
| `_183118` | 256 | 29.45s | 50m 17s | 67% faster | 0.9630 | 0.9932 | 0.9772 |
| `_183350` | 512 | 29.19s | 50m 03s | 67% faster | 0.9643 | 0.9933 | 0.9799 |

### Key takeaways (full dataset)

- **Combined optimizations deliver 67% speedup** — from 89.71s to ~29s per epoch
  (2h 31m → 50m total). Confirms 5K findings scale to the full dataset.
- **batch_size 128/256/512 produce identical runtime** (~29s/epoch). GPU is saturated
  at batch_size=128; larger batches offer no further speed benefit.
- **Prediction performance is equivalent or slightly better** across all batch sizes.
  batch_size=128 actually has the best F1 (0.9656) and AUC-ROC (0.9940). No convergence
  degradation from larger batches at this scale.
- **For HPC walltime planning**: A single full-dataset fold at 100 epochs ≈ 50 min with
  optimized config (vs 2h 31m baseline). With early stopping (~30–50 epochs), expect
  ~15–25 min per fold.

---

## Experiment Plan

Full dataset benchmarking complete on Lambda (V100) and Polaris (A100). See Section 8.

**Hardware:** Lambda cluster (V100 GPUs) and Polaris (A100 GPUs).

---

## 8. Polaris production validation (April 2026, A100, ensemble-packed)

Task 11 Phase 3: 28 protein pairs × 12-fold CV × 100 epochs on the full Flu A dataset
(~111K isolates), k-mer k=6, slot_norm + concat. Each Polaris node ran 4 folds concurrently
(one per A100), 28 nodes in parallel. Manifest: `allpairs_prod_20260408_063203/`.

**Per-epoch breakdown (median across 334 completed folds, ~33,400 epochs total):**

| Phase | Time |
|-------|------|
| `data_time` | 3.0 s |
| `compute_time` | 20.4 s |
| `eval_time` | 1.0 s |
| **`epoch_time`** | **25.0 s** |

- **Per-fold runtime:** median 44 min at 100 epochs (vs ~50 min on Lambda).
- **Per-epoch:** ~25 s on Polaris A100 vs ~29 s on Lambda V100 — comparable, despite 4-way
  ensemble packing on the same node.

### What it took to get here

The first Phase 3 attempt failed catastrophically: `data_time` was **515 s/epoch** (170×
slower than the Lambda single-fold expectation) and one fold per node OOMed at
`model.to(device)`. Two interacting bugs were diagnosed via Exp A/B/C scaling experiments
(1 fold → 2 folds → 4 folds at full scale, 4 epochs each):

1. **`pin_memory=True` + 4 concurrent fold processes → cudaHostAlloc serialization.**
   The CUDA driver lock for pinned-memory allocation is shared across processes on the
   same node. With 4 folds each pinning every batch, they spent essentially all their
   time waiting on the driver. **Fix: `pin_memory: false` in the master bundle.**
   Restored 1.6 s data_time (322× improvement).
2. **TensorFlow GPU pre-allocation via transitive HuggingFace import.** TF is in the
   Polaris system conda env; HF transformers' `is_tf_available()` check loads it on import,
   and TF eagerly allocates all GPU memory. **Fix: `TF_FORCE_GPU_ALLOW_GROWTH=true` set
   before any import in `train_pair_classifier.py`.**

Full diagnosis in `docs/hardware_notes.md` §1, §3.

### Key takeaways

- **Lambda single-fold benchmarks do not predict HPC ensemble-packed behavior.** On Lambda
  `pin_memory=True` is an 18% win; on Polaris with 4 concurrent folds it is a 300× loss.
  Always re-profile at the concurrency level you will actually deploy.
- **The 67% speedup combo from Section 7 (batch_size=128, eval_train_metrics=false,
  infer_batch_size=8192) transfers cleanly to A100.** Per-epoch time is comparable to V100
  on the same dataset.
- **Per-fold runtime was the planning unit.** 12 folds / 4 GPUs = 3 waves × ~44 min ≈
  2.5 h per pair. All 28 pairs in parallel on 28 nodes → ~3.5 h wall-clock for the full
  Task 11 production run, well within the medium-prod 6 h walltime.
