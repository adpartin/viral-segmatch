# Hardware / Software Interaction Notes

Purpose: record the hardware-driven reasons behind code choices in this project — GPU driver
behavior, CPU caches, filesystem, and shared-node effects. Audience: ML engineers who need to
know why a small training script can slow from a few seconds per epoch on a workstation to
hundreds of seconds per epoch on a multi-GPU HPC node, and what to check.

Machine: **Polaris** — 4x NVIDIA A100 40 GB per node, AMD EPYC "Milan" 32-core CPU, 512 GB host
RAM, HPE Slingshot interconnect, PBS Pro scheduler, **Eagle** Lustre project filesystem (ALCF).
Terms below are defined in the Glossary and match the vendor docs in Sources.

Sections tagged `[Extra]` are background, not tied to a specific bug we hit.

---

## Glossary (confirmed terms)

- **Pinned (page-locked, non-pageable) host memory** — CPU memory locked in place so the OS cannot
  swap or move it. Allocated by CUDA `cudaHostAlloc`. Required for asynchronous host-to-device
  copies; from ordinary pageable memory an "async" copy silently runs synchronously. [NVIDIA]
- **DMA (Direct Memory Access)** — a copy engine moves data over PCIe without using the compute
  engine, so a copy can overlap with compute — but only from pinned memory. [NVIDIA]
- **H2D / D2H** — host-to-device / device-to-host memory copy.
- **`pin_memory` (PyTorch DataLoader)** — pins each fetched batch, in a background thread, for
  faster H2D transfer; costs pinned host RAM. [PyTorch]
- **GPU memory growth (TensorFlow)** — TF maps nearly all GPU memory on startup by default;
  `TF_FORCE_GPU_ALLOW_GROWTH=true` makes it grow on demand instead. [TensorFlow]
- **Ensemble packing** — running several independent training processes on one node (here 4, one
  per A100). They share CPU RAM, L3 cache, PCIe, and the CUDA driver.
- **Load balancing / load imbalance** — spreading work evenly across parallel workers so none
  becomes the bottleneck. Imbalance is measured by `MAX = max work / mean work` (1.0 = perfect).
  [HPC-Wiki]
- **Makespan** — for a set of independent tasks, the time from the start of the first to the finish
  of the last. Minimizing it is the goal when packing many tasks onto limited workers.
- **`num_workers` (DataLoader)** — number of worker processes that prefetch batches; 0 means the
  main process loads data.
- **SM utilization / occupancy** — how busy and how full the GPU's compute units are; low
  utilization at high wall-time means the GPU is starved, not compute-bound.
- **CUDA async execution** — GPU kernels return control to Python before finishing; honest GPU
  timing needs `torch.cuda.synchronize()` or CUDA events. [NVIDIA]

---

## 1. `pin_memory` and `cudaHostAlloc` serialization

**Code:** `conf/bundles/flu_28_major_protein_pairs_master.yaml` → `training.pin_memory: false`.

`pin_memory=True` makes the DataLoader copy every batch into pinned host memory via
`cudaHostAlloc`, which enables fast asynchronous H2D transfers. On one GPU this gave ~18% speedup.
On Polaris with 4 training processes per node it caused a **~322x slowdown** in data loading
(1.6 s → 515 s per epoch).

Cause: pinning memory takes a driver-level operation that serializes across processes on the node.
With 4 processes pinning every batch, they spend almost all their time waiting on the driver
instead of feeding the GPU. The effect is invisible with one process and worsens sharply with
concurrency. (The slowdown is measured; driver-level serialization of `cudaHostAlloc` is the
diagnosis consistent with the evidence below.)

Evidence (Exp A/B/C, full dataset, 4 epochs):

| Exp | Folds | `pin_memory` | Data load/epoch | Total/epoch |
|-----|-------|--------------|-----------------|-------------|
| A | 1 | false | 1.6 s | 4.6 s |
| B | 2 | false | 1.6 s | 5.0 s |
| C | 4 | false | 1.7 s | 5.1 s |
| — | 4 | true | 515 s | 540 s |
| Phase 3 (production) | 4 × 28 nodes × 100 epochs | false | 3.0 s (median) | 25.0 s (median) |

The Phase 3 row is the median over 334 folds × ~100 epochs (~33,400 epochs).

**Takeaway:** pinned memory assumes the GPU is the bottleneck and the driver is uncontended.
Neither holds under ensemble packing. Benchmark at the concurrency you will actually run.
(See also §9: pinned memory is a finite host resource.)

---

## 2. `num_workers=0`

**Code:** `src/models/train_pair_classifier.py` → `NUM_WORKERS = 0` (hard-coded). Two reasons.

**(a) Performance.** `num_workers>0` forks worker processes that pickle batches through a queue to
the main process. That pays off only when `__getitem__` does real work (disk I/O, decode,
augmentation) that can overlap GPU compute. Our dataset is a numpy matrix already in RAM;
`__getitem__` is one array index. The queue and pickling cost exceeds the indexing cost, so
`num_workers=2` measured ~87% slower on Lambda (`speed_up.md`).

**(b) Correctness.** `KmerPairDataset.__getitem__` uses `torch.from_numpy(...)`, which shares the
numpy buffer (zero copy). Forked workers inherit those pages copy-on-write, which has caused
segfaults in PyTorch. If you ever need `num_workers>0`, switch to `torch.tensor(...)` (which
copies) first.

**Takeaway:** `num_workers=0` is right for in-memory data — there is no I/O latency to hide. The
"set it to the CPU count" rule assumes a disk-bound image pipeline.

---

## 3. TensorFlow GPU pre-allocation via transitive import

**Code:** top of `src/models/train_pair_classifier.py` sets `TF_FORCE_GPU_ALLOW_GROWTH=true` and
`TF_CPP_MIN_LOG_LEVEL=3` before any import.

On Polaris, one fold OOMed at `model.to(device)` even though the model is tiny. Cause: the Polaris
base environment has TensorFlow installed. HuggingFace `transformers` (pulled in by `esm2_utils`)
calls `is_tf_available()` on import, which loads TF, and TF maps nearly all GPU memory on startup
by default. PyTorch then finds <1 GB free and OOMs.

Fix: set the env vars before any import that can load TF. `TF_FORCE_GPU_ALLOW_GROWTH=true` makes TF
grow memory on demand. We also log `torch.cuda.mem_get_info()` just before `model.to(device)` so an
OOM here is distinguishable from a genuinely too-large model.

**Takeaway:** on shared HPC environments, a framework you never imported can still take the GPU. Set
memory-growth and visibility env vars before imports.

---

## 4. L3 cache vs working-set size

**Phase 1 (5K isolates):** the per-fold k-mer matrix is ~50 MB and fits in the CPU L3 cache, so
shuffled random access is nearly free (L3 hits) and any per-batch overhead is hidden.

**Phase 3 (full dataset, ~111K isolates):** the per-fold matrix is ~3.5 GB; 4 folds is ~14 GB under
random access. This spills L3 to DRAM on every batch and magnifies any per-batch overhead (like
`cudaHostAlloc`), because the batch cost is no longer dominated by compute.

**Takeaway:** small-data prototypes do not predict production. Growing the dataset crosses
cache-hierarchy thresholds (L1 → L2 → L3 → DRAM), and both latency and contention change sharply.
Re-profile at production scale.

---

## 5. Level 1 / Level 2 profiling

**Level 1 (always on):** `train_pair_classifier.py` writes `data_time`, `compute_time`,
`eval_time`, and `epoch_time` to `training_history.csv`. Enough to localize which phase of an epoch
is slow (loading vs forward/backward vs evaluation).

**Level 2 (diagnostic, commented out):** a block near line 630 builds two DataLoaders
(`pin_memory=True` vs `False`) and times 10 batches each. This isolated the `pin_memory` bug.
Uncomment to re-enable (~5 s added to fold startup).

**Takeaway:** keep light profiling always on to catch regressions; park heavier diagnostics in
comments to flip on in minutes.

---

## 6. `[Extra]` CUDA async execution and why naive timing lies

CUDA kernels are asynchronous: `loss.backward()` returns before the GPU finishes. Timing with
`time.time()` around forward/backward measures launch time, not compute time.

For honest GPU timing, call `torch.cuda.synchronize()` before reading the clock, or use
`torch.cuda.Event(enable_timing=True)`. Our Level 1 timings are honest only because the DataLoader
step and metric read-back implicitly synchronize. If you move metrics to GPU-side, add an explicit
`synchronize()` before the timing read.

---

## 7. `[Extra]` DataLoader shuffling and per-process RNG

When N folds run as separate processes, each needs a different shuffle order but a reproducible
seed. If two processes share the master seed and fold logic, they traverse data identically and
lose the independence CV needs. Fix: derive per-fold seeds as `master_seed + fold_id` for both
numpy and torch, and log them (`src/utils/seed_utils.py`).

---

## 8. `[Extra]` `CUDA_VISIBLE_DEVICES` and logical vs physical device IDs

With `CUDA_VISIBLE_DEVICES=2 python train.py`, inside the child process `cuda:0` is physical GPU 2,
and GPUs 0/1/3 are invisible. This is why `determine_device()` returns `cuda:0` for GPU mode — the
remapping already happened at the env-var level. Hard-coding `cuda:2` or `set_device(2)` in the
child crashes with "invalid device ordinal".

---

## 9. `[Extra]` Pinned memory is a finite host resource

Pinned memory comes from kernel-locked pages, capped by `RLIMIT_MEMLOCK`, and the total across all
node processes is a fixed budget (a few GB). Many concurrent `pin_memory=True` DataLoaders can
exhaust it and raise `cudaErrorMemoryAllocation`, which looks like GPU OOM but is host OOM. A second
reason, beyond the §1 driver serialization, to avoid `pin_memory` under ensemble packing.

---

## 10. `[Extra]` AMP / Tensor Cores: when mixed precision is not a win

`use_amp: false` is deliberate. AMP (`torch.autocast`) helps only when the model is large enough
that FP16/BF16 matmul throughput beats the autocast overhead, the GPU has Tensor Cores, and matrix
shapes are multiples of 8 (16 on Ampere). Our MLP is tiny and its shapes are not aligned, so AMP
measured neutral-to-slower. For large transformers AMP is almost always a win. Details in
`speed_up.md` §4.

---

## 11. `[Extra]` Ensemble packing vs PBS job arrays

Two ways to run N folds on M GPUs:

- **PBS job array:** N separate 1-GPU jobs. Simple isolation, but scheduler overhead and queue wait
  apply N times.
- **Ensemble packing:** one job on one 4-GPU node launching 4 processes with different
  `CUDA_VISIBLE_DEVICES`. One queue wait, but processes share CPU RAM, L3, PCIe, and the CUDA driver
  — which is how the `pin_memory` bug hid.

We use ensemble packing for Task 11 CV: it minimizes queue time, and 12 folds pack cleanly on
4-GPU nodes. Lesson from the `pin_memory` incident: packing couples processes in subtle ways, so
profile under the packing you will deploy.

---

## 12. `[Extra]` Lustre vs node-local /tmp vs NVMe

Polaris uses Eagle Lustre for project storage. Lustre is fast for large sequential reads by many
clients (HDF5 caches, CSVs), but slow for many small files (metadata operations) and varies with
cluster load.

For us Lustre is fine: each fold reads one 3.5 GB k-mer matrix once, then serves batches from RAM.
If you move to a workload that re-reads many small files per epoch (e.g., on-the-fly tokenization):

- **Stage to node-local `/tmp`** (NVMe) at job start with one large copy, then read from /tmp.
- **Pack into HDF5 / Zarr / WebDataset** for one sequential stream instead of many small reads.
- **Avoid `glob` / `listdir`** over large dirs during training — these hit Lustre metadata servers
  and can stall every fold at once.

Rule: Lustre rewards few large reads, punishes many small ones.

---

## Polaris quick facts

- **Profiling tools installed** (via the NVIDIA HPC SDK, on PATH): `nsys` (Nsight Systems, timeline
  profiler), `ncu` (Nsight Compute, per-kernel counters), `dcgmi` (DCGM). Note: `ncu` needs GPU
  performance counters enabled for non-root users — check on a compute node before relying on it.
- **Queues:** `debug` (up to 24 nodes, 1 h walltime), `debug-scaling` (up to 10 nodes, 1 h, one job
  per user), `prod` (routing queue → small/medium/large by node count; medium = 25-99 nodes, 6 h).
- **Filesystems flag:** PBS jobs need `-l filesystems=home:eagle` or they are rejected.

## Sources

- NVIDIA — pinned/page-locked memory and async transfers: "How to Optimize Data Transfers in CUDA
  C/C++" (https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) and the CUDA C++
  Programming Guide.
- PyTorch — `pin_memory` / DataLoader:
  https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
- TensorFlow — GPU memory growth / `TF_FORCE_GPU_ALLOW_GROWTH`:
  https://www.tensorflow.org/guide/gpu
- ALCF — Polaris machine overview and running jobs: https://docs.alcf.anl.gov/polaris/

## See also

- `speed_up.md` — training speed-ups (batch size, `eval_train_metrics`, etc.) and Polaris §8.
- `polaris_plan.md` — Task 11 Polaris execution plan (phases, queues).
- `docs/plans/hpc_scaling_profiling_parsl_plan.md` — scaling/profiling/Parsl plan and its Glossary.
