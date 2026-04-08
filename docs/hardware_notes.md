# Hardware / Software Interaction Notes

Purpose: capture the *why* behind code choices in this project that are driven by
hardware behavior (GPUs, CPU caches, OS/driver semantics). Aimed at ML engineers /
data scientists who want to understand how a small training script can go from
5 s/epoch to 540 s/epoch when moved from a single-GPU workstation to a multi-GPU
HPC node, and what to look for.

Topics tagged `[Extra]` are **not** currently exercised by a bug we hit, but are
essential background for anyone maintaining a PyTorch training loop on HPC.

---

## 1. `pin_memory` and `cudaHostAlloc` serialization

**Code**: `conf/bundles/flu_28_major_protein_pairs_master.yaml` → `training.pin_memory: false`

**What it does**: `pin_memory=True` tells the DataLoader's `pin_memory_thread` to
copy every batch into CUDA page-locked (pinned) host memory via `cudaHostAlloc`,
which enables faster async H→D DMA transfers.

**Why we disabled it**: On a single GPU (Lambda), this gave ~18% speedup. On
Polaris with 4 concurrent training processes on the same node (one per GPU), it
caused a **~300x slowdown in data loading** (1.6 s → 515 s per epoch).

**Root cause**: `cudaHostAlloc` takes a driver-level lock that is **serialized
across processes on the same node**. With 4 workers each trying to pin every
batch, they spend nearly all their time waiting on the driver lock instead of
feeding the GPU. The effect is invisible with 1 process and grows superlinearly
with concurrency.

**Evidence** (Exp A/B/C, full dataset, 4 epochs):

| Exp | Folds | `pin_memory` | Data loading/epoch | Total/epoch |
|-----|-------|--------------|--------------------|-------------|
| A   | 1     | false        | 1.6 s              | 4.6 s       |
| B   | 2     | false        | 1.6 s              | 5.0 s       |
| C   | 4     | false        | 1.7 s              | 5.1 s       |
| —   | 4     | true         | 515 s              | 540 s       |
| **Phase 3 (production)** | **4 × 28 nodes × 100 epochs** | **false** | **3.0 s (median)** | **25.0 s (median)** |

The Phase 3 row is the median across **334 completed folds × ~100 epochs** (~33,400 epoch
observations). Mean `data_time` 3.01 s, max 5.45 s. Confirms the Exp A/B/C extrapolation
holds at production scale and over many hours of runtime.

**Takeaway**: pinned memory is a per-process optimization that assumes the GPU is
the bottleneck and the driver is uncontended. Neither holds when you ensemble-pack
multiple training processes on one node. Always benchmark at the *concurrency
level you will actually run at*.

---

## 2. `num_workers=0`

**Code**: `src/models/train_pair_classifier.py` → `NUM_WORKERS = 0` (hard-coded)

Two independent reasons:

**(a) Performance — IPC overhead dominates for in-memory data**

`num_workers>0` forks worker processes that pickle tensors through a queue to the
main process. That overhead is only worth paying when `__getitem__` does real
work (disk I/O, decode, augmentation) that can overlap with GPU compute. Our
dataset is already a numpy matrix in RAM; `__getitem__` is one indexing op. The
IPC cost exceeds the indexing cost, so `num_workers>0` measured ~87% slower on
Lambda (see `speed_up.md`).

**(b) Correctness — `torch.from_numpy` + forked workers is unsafe**

`KmerPairDataset.__getitem__` uses `torch.from_numpy(...)` for zero-copy tensor
construction. This shares the underlying numpy buffer. Forked worker processes
inherit the parent's pages copy-on-write; depending on access patterns this has
caused segfaults / corruption in PyTorch. If you ever need `num_workers>0`,
first switch to `torch.tensor(...)` (which copies).

**Takeaway**: `num_workers=0` is optimal *for this workload* because there is no
latency to hide. The common advice "set it to the number of CPU cores" is
assuming a disk-bound image pipeline, not an in-memory numerical one.

---

## 3. TensorFlow GPU pre-allocation via transitive import

**Code**: top of `src/models/train_pair_classifier.py` sets
`TF_FORCE_GPU_ALLOW_GROWTH=true` and `TF_CPP_MIN_LOG_LEVEL=3` before any import.

**What happened**: On Polaris, fold 2 OOMed at `model.to(device)` even though the
model was tiny and the GPU was nominally free. The Polaris system conda env has
TensorFlow installed. HuggingFace `transformers` (imported via `esm2_utils`)
does a feature check `is_tf_available()` on import, which triggers TF to
initialize — and TF's default is to **eagerly allocate all GPU memory** on the
first visible device. Our PyTorch process then finds <1 GB free and OOMs.

**Fix**: set the env vars *before* any import that can transitively load TF.
`TF_FORCE_GPU_ALLOW_GROWTH=true` makes TF's allocator grow on demand instead of
grabbing everything.

**Diagnostic**: we also log `torch.cuda.mem_get_info()` immediately before
`model.to(device)` so an OOM here is distinguishable from an OOM caused by an
actually too-large model or by another user sharing the node.

**Takeaway**: a library you never explicitly imported can still take your GPU.
On shared HPC environments, assume any ML framework installed in the base env
may get pulled in through a compatibility check. Always set growth / visibility
env vars before imports.

---

## 4. L3 cache vs. working-set size (why Phase 1 worked and Phase 3 didn't)

**Phase 1 (5K isolates)**: per-fold k-mer matrix ≈ 50 MB. Fits inside the CPU L3
cache of an EPYC node (tens to hundreds of MB). Random access during shuffled
DataLoader iteration is nearly free because every read is an L3 hit. The
`pin_memory` overhead existed but was tiny relative to anything else, so nothing
looked wrong.

**Phase 3 (111K isolates, full dataset)**: per-fold matrix ≈ 3.5 GB; 4 folds ×
3.5 GB = 14 GB of working set under random access. This spills L3 into DRAM for
every batch, and crucially it **magnifies any per-batch overhead** (like
`cudaHostAlloc`) because the batch cost is no longer dominated by compute.

**Takeaway**: performance results from small-data prototypes do not extrapolate.
When scaling dataset size, you cross cache-hierarchy thresholds (L1 → L2 → L3 →
DRAM → NVMe) and both latency *and* contention characteristics change
discontinuously. Always re-profile at production scale.

---

## 5. Level 1 / Level 2 profiling methodology

**Level 1 — per-epoch wall-clock breakdown** (always on):
`train_pair_classifier.py` writes `data_time`, `compute_time`, `eval_time`, and
`epoch_time` to `training_history.csv`. This is enough to localize *which phase*
of an epoch is slow (loading vs forward/backward vs evaluation).

**Level 2 — micro-benchmark first 10 batches** (diagnostic, commented out):
A 20-line block in `src/models/train_pair_classifier.py` (search for
`Level 2 diagnostic`, currently around line 630) that builds two DataLoaders — one
with `pin_memory=True`, one with `False` — and times 10 batches each, printing
ms/batch. This is what let us isolate `pin_memory` as the culprit. Uncomment the
block to re-enable; it adds ~5 s to fold startup.

**Takeaway**: keep lightweight profiling permanently on (Level 1) so you can
notice regressions; keep heavier diagnostics (Level 2) parked in comments so
they can be flipped on in minutes when something looks off.

---

## 6. `[Extra]` CUDA async dispatch and why wall-clock timing lies

PyTorch CUDA ops are **asynchronous**: `loss.backward()` returns before the GPU
has actually finished. Naïvely timing `t0 = time.time(); forward(); backward();
print(time.time() - t0)` measures the *launch* time, not the *compute* time.

To get honest compute timings you need `torch.cuda.synchronize()` before
`t0` and before the final `time.time()`. Alternatively, use
`torch.cuda.Event(enable_timing=True)` with `record()` + `elapsed_time()` which
times GPU-side directly.

Implication for this project: our Level 1 timings are honest only because the
DataLoader step and the epoch boundary both implicitly synchronize (the loader
waits for the previous batch to be consumed, and metric computation reads
tensors back to CPU). If you ever move metrics to GPU-side, add explicit
`synchronize()` before timing reads.

---

## 7. `[Extra]` DataLoader shuffling and RNG determinism across processes

When launching N folds as separate processes, each process needs a **different**
shuffle order but the same seed chain should be reproducible. PyTorch's
`DataLoader(shuffle=True)` uses a per-epoch generator; if two processes share
the same master seed *and* the same fold index logic, they will traverse data
identically and you lose the statistical independence CV is supposed to give.

Mitigation: derive per-fold seeds from `master_seed + fold_id` and pass to both
numpy and torch generators (see `src/utils/seed_utils.py`). Log the effective
seeds in the run dir so any suspicious result can be reproduced.

---

## 8. `[Extra]` `CUDA_VISIBLE_DEVICES` and logical-vs-physical device IDs

When you launch child processes with `CUDA_VISIBLE_DEVICES=2 python train.py`,
inside the child **`cuda:0` is physical GPU 2**. The child cannot see GPUs 0, 1,
or 3 at all. This is why our `determine_device()` helper returns `cuda:0`
unconditionally for GPU mode: the remapping already happened at the env-var
level.

Common footgun: code that hard-codes `cuda:2` or `torch.cuda.set_device(2)`
inside the child will crash with "invalid device ordinal" because from its point
of view there is only one device.

---

## 9. `[Extra]` Pinned memory is a finite, global resource

Pinned memory on Linux is allocated from kernel-locked pages (`mlock`-class).
The kernel has a hard limit (`RLIMIT_MEMLOCK`) and the total across all
processes on the node is a fixed budget — typically a few GB. If you enable
`pin_memory=True` in many concurrent DataLoaders on a fat node, you can exhaust
the pool and get cryptic `cudaErrorMemoryAllocation` errors that look like GPU
OOM but are actually host OOM.

This is a second, independent reason (beyond cudaHostAlloc serialization) to be
cautious with `pin_memory` under ensemble-packed training.

---

## 10. `[Extra]` AMP / Tensor Cores: when mixed precision is *not* a win

`use_amp: false` in our bundles is not an oversight. AMP / `torch.autocast` only
helps when:
1. The model is large enough that the FP16/BF16 matmul throughput gain exceeds
   the autocast bookkeeping overhead, AND
2. Your GPU has Tensor Cores (Volta+), AND
3. Shapes are multiples of 8 (or 16 on Ampere+) so Tensor Cores actually fire.

Our MLP is tiny (~few MB of weights) and batch dimensions aren't aligned. We
measured AMP as neutral-to-slightly-slower. For large transformer training AMP
is almost always a win; for small MLPs it often isn't. Profile before assuming.

---

## 11. `[Extra]` Ensemble packing vs PBS job arrays

Two ways to run N folds on M GPUs on Polaris:

- **PBS job array**: N separate PBS jobs, each requesting 1 GPU. Simplest
  isolation, but scheduler overhead and queue wait apply N times.
- **Ensemble packing**: 1 PBS job requesting 1 node with 4 GPUs, launching 4
  `python` processes from a bash script with different `CUDA_VISIBLE_DEVICES`.
  One queue wait, but processes share CPU RAM, L3 cache, PCIe, and the CUDA
  driver — which is exactly how the `pin_memory` bug hid.

We use ensemble packing for Task 11 CV because it minimizes queue time and the
12-fold × 28-pair work fits cleanly on 4-GPU nodes. The lesson from the
`pin_memory` incident is: ensemble packing couples processes in subtle ways, so
profile under the actual packing you will deploy.

---

## 12. `[Extra]` Lustre vs node-local /tmp vs NVMe

Polaris uses Eagle Lustre as the project filesystem (`/lus/eagle/projects/...`). Lustre
is optimized for **large sequential I/O** by many clients in parallel — exactly what HDF5
embedding caches and CSV datasets look like. It is *not* optimized for many small files
(directory metadata operations are slow), and it has variable per-client throughput
depending on overall cluster load.

For our workload Lustre is fine: each fold loads one 3.5 GB k-mer matrix once at startup,
then serves batches from RAM. We never re-read from disk during training. If you ever
move to a workload that re-reads many small files per epoch (e.g., on-the-fly tokenization
of millions of FASTA records), consider:

- **Stage to node-local `/tmp`** at job start (one large `cp` from Lustre, then read from
  /tmp during training). Polaris compute nodes have local NVMe.
- **Pack into HDF5 / Zarr / WebDataset** so the access pattern is one large sequential
  stream rather than many small random reads.
- **Avoid `os.listdir()` / `glob.glob()` loops over large dirs** during training — these
  hit Lustre metadata servers and can stall every fold simultaneously.

The general rule: Lustre rewards **few large reads**, punishes **many small ones**. Match
your data layout to that.

---

## See also
- `speed_up.md` — original Lambda single-fold benchmark numbers (batch size,
  eval_train_metrics, etc.). Predates the Polaris concurrency findings.
- `polaris_plan.md` — Task 11 Polaris execution plan (phases, queues).
- Inline comments in `src/models/train_pair_classifier.py` (NUM_WORKERS,
  TF env vars, GPU memory diagnostic).
- Inline comment in `conf/bundles/flu_28_major_protein_pairs_master.yaml`
  (`pin_memory: false`).
