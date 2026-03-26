# Stage 4 Training Speed-Up Analysis

Baseline: k-mer k=6, slot_norm + concat, full dataset (~111K isolates), 100 epochs forced.
Per-epoch baseline: **97.1s mean** (Lambda, single V100/A100 GPU).

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

**Estimated speedup:** 5–15% on eval passes (depends on batch size headroom).

**Status: IMPLEMENTED.** Tested on 5K dataset (flu_6mer_5k, 20 epochs). Per-epoch steep up.

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

**Recommendation:** `num_workers=2, pin_memory=True` is safe everywhere. Since data is
pre-loaded and samples are small, benefit is modest.

**Config:** `training.num_workers: 2`

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

**Status: NOT YET TESTED.**

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

**Status: NOT YET TESTED.**

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

**Status: NOT YET IMPLEMENTED.**

---

## Summary

| # | Optimization | Config flag | Default | Risk | Est. speedup |
|---|---|---|---|---|---|
| 1 | Skip train metrics re-pass | `eval_train_metrics` | `true` | None | ~45–50% of epoch |
| 2 | Larger inference batch size | `infer_batch_size` | `null` (= batch_size) | None | 5–15% on eval |
| 3 | `num_workers` + `pin_memory` | `num_workers` | `2` | None | 5–15% |
| 4 | AMP (mixed precision) | `use_amp` | `false` | Negligible | 20–30% |
| 5 | `torch.compile` | `compile_model` | `false` | None | 10–20% |
| 6 | Batch GPU→CPU transfers | (none) | Always on | None | 5–10% on eval |

### Combined estimate (full dataset, per epoch)

| Scenario | Est. epoch time | vs baseline |
|---|---|---|
| Baseline | ~97s | — |
| Skip train re-pass only | ~50–55s | ~45% faster |
| All optimizations, eval_train_metrics=true | ~40–50s | ~50% faster |
| All optimizations, eval_train_metrics=false | ~25–35s | ~65–75% faster |

---

## Experiment Plan

Benchmark each optimization individually against the baseline using the full dataset
(bundle: `flu_6mer_full.yaml`, inheriting from `flu.yaml`). Child bundles per optimization
to be created for systematic comparison.

**Hardware:** Lambda cluster (V100 GPUs) and Polaris (A100 GPUs).
