# HPC Scaling, Profiling, and Parsl Port — Execution Plan

**Status: IN PROGRESS**

Take the existing 28-protein-pair CV sweep on Polaris and turn it into (a) a set of
**reproducible, defensible scaling numbers**, (b) a **profiling exercise** that teaches
the standard GPU/HPC metrics and vocabulary, and (c) a **Parsl port** of the launcher.
The scientific results are already produced by the working `mpiexec` launcher; nothing
here changes the science. This is about measurement rigor, understanding, and a
reproducible, documented artifact.

## Goals (why this plan exists)

1. **Defensible numbers.** Every scaling/perf number we report must be re-derivable
   from artifacts that still exist on disk, with the
   measured-vs-projected boundary stated explicitly.
2. **Knowledge.** Learn the standard terminology and how to *read* the metrics
   (utilization, occupancy, arithmetic intensity, roofline, speedup, parallel efficiency,
   load imbalance, makespan). Terms are confirmed against NVIDIA / PyTorch / TensorFlow /
   ALCF / HPC-Wiki sources (see Sources) and kept verbatim.
3. **Tooling.** Port the launcher from `mpiexec`-as-dispatcher to **Parsl** (ALCF-native
   many-task workflow engine), validated to reproduce the `mpiexec` results. Ray is a
   possible later extension, not in scope here.

## Non-goals / honesty boundaries

- This is an **embarrassingly parallel** sweep of independent, single-GPU MLP jobs. It is
  **not** distributed training of one model, and there is **no** inter-GPU communication
  (no NCCL collectives, no tensor/data parallelism). The `mpiexec` layer is a *dispatcher*
  (`-n 1 --ppn 1` per node), not a parallel program. Do not describe it as "MPI-parallel
  training."
- The 336-run sweep used **k-mer (k=6)** features, not ESM-2. The ESM-2 path was
  benchmarked separately at smaller scale. Do not conflate.
- Machine/fabric named every time: **Polaris = 4× A100 40GB per node, HPE Slingshot 11
  interconnect, PBS Pro scheduler**. Ampere → no FP8.

---

## Terminology & metrics (confirmed; learn these)

Tags: **[sched]** scheduler/HPC · **[gpu]** GPU/driver · **[scale]** parallel-scaling ·
**[prof]** profiling. Each computed quantity uses previously-defined ones.

### Scaling `[scale]`
- **Strong scaling** — fixed total problem size, increase the number of processing
  elements (here: fixed 336 folds, more GPUs). Ideal: time halves when GPUs double.
  (Contrast **weak scaling**: grow problem *and* processors together, constant work per
  unit.) This sweep is a strong-scaling story. [HPC-Wiki]
- **Speedup (S)** — `S = t(1) / t(N)`: serial time on one GPU divided by parallel time on
  N. [HPC-Wiki]
- **Parallel efficiency (E)** — `E = S / N = t(1) / (N · t(N)) ≤ 1`. Fraction of the ideal
  N× you actually achieved. [HPC-Wiki]
- **Amdahl's law** — for a fixed problem, speedup is capped by the non-parallelizable part
  of the work, which includes serial code, communication, *and load imbalance*. [HPC-Wiki]
- **Load imbalance** — uneven work across parallel units so the job waits for the slowest.
  Standard metric: `MAX = (max work over units) / (mean work per unit)`; `1.0` is perfect
  balance, larger is worse. [HPC-Wiki]
- **Makespan / bin-packing** — for *independent* tasks, the scheduling problem is to pack
  tasks onto workers to minimize when the *last* task finishes (the makespan). This — not
  classic Amdahl — is the right model for this sweep (see next section).

### GPU / driver `[gpu]`
- **Pinned (page-locked, non-pageable) host memory** — CPU memory locked in place so the OS
  cannot swap or move it. Allocated via CUDA `cudaHostAlloc`. Required for **asynchronous**
  host→device (H2D) DMA copies; from ordinary *pageable* memory the async copy silently
  reverts to synchronous. [NVIDIA CUDA guide; NVIDIA "Optimize Data Transfers"]
- **DMA (Direct Memory Access)** — the GPU's copy engine moves data over PCIe without
  occupying the compute engine, so a copy can overlap with compute — but only from pinned
  memory. [NVIDIA]
- **PyTorch `pin_memory=True`** — DataLoader pins each fetched batch (in a background
  `pin_memory_thread`) for faster H2D transfer; costs pinned RAM. [PyTorch pinmem tutorial]
- **TensorFlow GPU memory growth** — TF's default is to pre-allocate (nearly) all GPU
  memory on init; `TF_FORCE_GPU_ALLOW_GROWTH=true` (or `set_memory_growth`) makes it grow
  on demand instead. [TensorFlow GPU guide]
- **Ensemble packing** — running several independent training processes on one node (here
  4, one per A100), sharing CPU RAM, L3 cache, PCIe, and the CUDA driver. Cheaper queueing
  than one PBS job per fold, but couples the processes (this is how the pinned-memory bug
  hid). [hardware_notes.md §11]

### Profiling `[prof]`
- **SM / compute utilization** — fraction of time the GPU's streaming multiprocessors are
  busy. Low utilization with high wall-time means the GPU is *starved* (waiting on data or
  CPU), not compute-bound.
- **Occupancy** — ratio of active warps to the hardware maximum; a measure of how well a
  kernel fills the GPU.
- **Arithmetic intensity** — FLOPs performed per byte of memory traffic. Low intensity →
  memory-bandwidth-bound; high → compute-bound.
- **Roofline** — plot of achievable FLOP/s vs arithmetic intensity; the "roof" is
  `min(peak compute, peak bandwidth × intensity)`. Locates a kernel as memory- or
  compute-bound.
- **Nsight Systems (`nsys`)** — system-wide timeline profiler (CPU, GPU kernels, H2D/D2H
  copies, gaps). Answers "where does the wall-clock go?"
- **DCGM / `nvidia-smi dmon`** — sampling of GPU utilization, memory use, power over time.
- **CUDA async caveat** — CUDA kernels are asynchronous; naive `time.time()` around a
  forward/backward measures *launch* time, not GPU time. Use `torch.cuda.synchronize()` or
  CUDA events for honest GPU timing. [hardware_notes.md §6]

---

## The scaling model for THIS job (read before quoting any number)

The 336 folds are **independent, single-GPU tasks**. So:

- This is **not** a classic Amdahl parallel-region problem (one coupled computation split
  across ranks, bounded by communication). It is a **makespan-minimization / bin-packing**
  problem: pack 336 tasks onto 112 GPUs to minimize when the last finishes. The right
  vocabulary is *scheduling and load balancing*, not *MPI collectives*.
- **Within a node**, 12 folds ÷ 4 GPUs = exactly **3 full waves**, and folds within a pair
  have near-equal size, so within-node GPU packing is near-perfect — little idle GPU time
  *during training*.
- Therefore the efficiency shortfall below 100% is dominated by, in order:
  1. **Serial preamble** — Stage-3 dataset generation runs before that node's folds start.
  2. **Serial postamble** — final cross-pair aggregation after all nodes finish.
  3. **Cross-node load imbalance** — different protein pairs have different isolate counts →
     different per-fold time → the whole job waits for the slowest node (the `MAX` metric).
  4. **Idle tail** — nodes that finish early sit idle; billing is whole-node.

**This decomposition is the scaling story** and it is exactly what Phase 0 measures.
Saying "62% end-to-end efficiency, but training-phase GPU packing is near-ideal; the loss
is a serial dataset-gen preamble plus cross-node load imbalance" is far stronger — and more
honest — than a bare "62% parallel efficiency."

### Current (soft) numbers — to be re-derived in Phase 0
From `speed_up.md` §8 and `docs/hardware_notes.md` §1 (k-mer k=6, full dataset, 100 epochs):

| Quantity | Value | Status |
|---|---|---|
| Per-epoch median (production) | data 3.0s / compute 20.4s / eval 1.0s / **total 25.0s** | in-repo, verified |
| Per-fold runtime (median, 100 ep) | ~44 min | in-repo; cross-checked on Polaris (Apr-14 run: per-pair median 134.3 min ÷ 3 waves = 44.8 min) |
| Folds completed | 334 / 336 | from cited manifest (local clone) |
| Whole-job wall-clock `t(N)` | ~3.5 h | in `speed_up.md` §8 as a **planning estimate**; primary artifact (`pbs_job.log` Elapsed) not on Polaris |
| Serial time `t(1)` (projected) | 334 × 44 min ≈ **245 GPU-h** (~10 days) | **projected, not measured** |
| Speedup `S` | ≈ 70× | derived from projected `t(1)` and estimated `t(N)` |
| Parallel efficiency `E = S/112` | ≈ **62%** | derived; **hangs entirely on the 3.5 h figure** |

**Correction to an earlier claim:** the "~96% of epoch time is data loading" note describes
the **failed** first attempt (`data_time` 515 s of a 540 s epoch). The **successful**
production run is **compute-time-dominated** (compute 20.4 s of 25 s ≈ 82%; data 12%).
Note "compute_time-dominated" (a coarse wall-clock bucket) is **not** the same as
"GPU compute-bound" — the MLP may still be memory-bandwidth-bound on the GPU, and the bucket
can hide CPU-side work. Which it is, is an open question Phase 1 resolves.

---

## Phase 0 — Lock down defensible numbers (highest priority, ~half day)

**Purpose:** convert `S`/`E` from a projection anchored on a deleted run into numbers
re-derivable from live artifacts, and produce the efficiency decomposition.

**Do:**
1. **Anchor `t(1)` empirically.** Run **one fold solo** on a single A100 (no node-sharing):
   `CUDA_VISIBLE_DEVICES=0 python -m src.models.train_pair_classifier ...` on one pair/fold,
   full dataset, 100 epochs. Record per-epoch and per-fold time. This is the true
   single-GPU per-fold time (no ensemble contention).
2. **Measure the contention factor.** Run the *same* fold once solo (from step 1) and once
   while 3 sibling folds run on the other 3 GPUs of the node. `contention = t_4share / t_solo`.
   Report it — it is the honest per-GPU cost of ensemble packing, and it tells you whether
   the projected `t(1) = 334 × t_fold` uses a contention-inflated `t_fold` (it does; state so).
3. **Decompose the wall-clock** of a run whose artifacts exist on Polaris (the Apr-14
   `val_unfilt` k-mer run, or a fresh re-run). Break the makespan into: dataset-gen
   preamble, training waves, aggregation postamble, idle tail. Use per-pair start/end from
   the logs + `check_allpairs_status.py` timing. Compute the load-imbalance `MAX` across
   nodes.
4. **Re-derive `S` and `E`** from measured `t(N)` and the step-1-anchored `t(1)`, reporting
   both "training-phase efficiency" and "end-to-end efficiency" separately.
5. **Fix the wording.** State k-mer (drop the "(it was actually k-mer)" parenthetical);
   always pair any speedup with "(throughput; serial baseline anchored by one measured fold,
   remainder projected)."

**Record:** `t_solo`, `t_4share`, contention factor, wall-clock decomposition table, `MAX`,
re-derived `S`/`E`, and the exact run dir + commands that produce them.

**Done when:** every scaling number traces to an artifact on disk; measured-vs-projected is
explicit; the efficiency decomposition figure exists.

### Phase 0a — executed 2026-07-10: wall-clock decomposition (no queue)

**Goal:** reconstruct where the end-to-end wall-clock goes, from surviving Polaris
artifacts, and reconcile the "3.5 h vs 2.3 h" question.

**What I used.** The cited 04-08 run was deleted, so I used the `val_unfilt` k-mer sweep. It
was run as **two jobs**: Apr-13 generated the datasets (Stage 3 only), Apr-14 trained on them
(Stage 4 only, datasets reused). Sources: per-pair `runtime.json`, per-fold
`training_history.csv`, and the Apr-13 Stage-3 pair logs.

**Reproduce.**
- Per-pair training time + load imbalance (all 28 pairs start together at `080618`):
  ```python
  # over models/flu/July_2025/cv_runs/cv_flu_28p_*_val_unfilt_*/runtime.json
  # runtime.json = {"hours","minutes","seconds"} -> minutes; group by run day;
  # MAX (load imbalance) = max(per-pair) / mean(per-pair)
  ```
- Per-fold + per-epoch: `training_flu_28p_ha_na_val_unfilt_fold0_20260414_080618/training_history.csv`
  (columns `epoch_time_sec,data_time_sec,compute_time_sec,eval_time_sec`).
- Dataset-gen time: `grep "Elapsed Time" allpairs_prod_val_unfilt_20260413_151649/flu_28p_ha_na.log`
  → `01:14:37`.

**Findings.**

| Quantity | Value | Source |
|---|---|---|
| Per-fold training (100 epochs) | **42.7 min** (ha_na fold 0) | training_history.csv |
| Per-epoch median | **25.4 s**: data 3.1 s (12%) / compute 20.9 s (82%) / eval 1.4 s (6%) | training_history.csv |
| Per-pair training (median of 26 completed) | **134.3 min** (max 136.0) | runtime.json |
| Load imbalance across pairs | **MAX ≈ 1.02** (~2%) | runtime.json |
| Dataset generation (Stage 3, per pair) | **~74.6 min** (1h14m37s), ha_na, CPU-bound | Apr-13 log |
| Training makespan (all pairs start together) | **136 min ≈ 2.27 h** | runtime.json |
| End-to-end reconstructed (gen + train) | **~1.24 h + 2.27 h ≈ 3.5 h** | matches `speed_up.md` §8 |

**Reconciliation (the key result).** Both wall-clocks are correct at different scopes. The
Apr-14 training job reused datasets, so 2.27 h is **training only**. The 04-08 baseline
generated datasets in the same job, so its ~3.5 h = **~1.24 h dataset generation + ~2.27 h
training**. Dataset generation uses **no GPU**, so during that ~1.24 h all 4 GPUs per node
sit **idle**.

**Efficiency, decomposed:**
- Useful GPU work: 28 pairs × 12 folds × 42.7 min ≈ **239 GPU-hours**.
- Training-only allocation: 112 GPUs × 2.27 h ≈ 254 GPU-h → **training-phase efficiency ≈ 94%**
  (balanced load, near-perfect within-node packing, no communication).
- End-to-end allocation: 112 GPUs × 3.5 h ≈ 392 GPU-h → **end-to-end GPU utilization ≈ 61%**,
  matching the reported ~62%.
- **The entire ~38% gap is the GPU-idle CPU dataset-generation preamble** — not load imbalance
  (~2%), not communication (none), not scheduling.

**Actionable:** (1) generate datasets in a **separate CPU-only job** so no GPU-hours are billed
while GPUs idle — the Apr-13/Apr-14 split already does this; the 04-08 single-job design is what
created the idle GPU-hours. (2) Most of the ~75 min is per-fold diagnostic-plot I/O to Lustre
(many `Saved: .../plots/...` lines per fold); skipping plots in production would cut the
preamble sharply.

**Honesty caveats:** the ~75 min dataset-gen is one pair (ha_na), representative but a single
sample; the 3.5 h end-to-end is **reconstructed** by adding two separate jobs' surviving times
(the single 04-08 `pbs_job.log` is gone) and it matches the §8 estimate. Per-fold/per-epoch
numbers are from this run's own fold artifacts. Two pairs (pb1_ha, pb2_pa) failed at ~9 min,
idling their nodes for the rest of the job (~17 GPU-h) — an availability loss, separate from
the efficiency story.

**Still needs 0b (interactive GPU):** an empirical single-GPU `t(1)` anchor and the
ensemble-packing **contention factor** (solo vs 4-share per-epoch time). 0a used the packed
per-fold time (42.7 min) as the `t(1)` per-fold unit; 0b tests whether a solo fold runs faster,
which would lower the honest speedup.

### Phase 0b — executed 2026-07-10: solo vs 4-share (interactive `debug` node)

**Method.** Same fold (ha_na fold_0, full dataset, 177,265 train pairs, 1385 batches/epoch) run
**solo** on GPU 0, then **4 folds on GPUs 0-3 at once**. Compare per-epoch median (skip warmup) and
`nvidia-smi dmon` GPU utilization. Reproduce: `logs/phase0b/run_4share.sh` + the solo command
above; analysis `scratchpad/p0b2.py`.

**Result 1 — ensemble packing is NOT the bottleneck.**

| | per-epoch | data | compute | eval | GPU `sm%` |
|---|---|---|---|---|---|
| Solo (1 fold) | 4.8 s | 1.6 | 2.9 | 0.3 | 39% |
| 4-share (per fold) | 5.0 s | 1.7 | 3.0 | 0.3 | 38-39% (all 4 GPUs) |

- **Contention factor = 1.06×** → packing is **94% efficient**, 3.77× node throughput vs 1 fold.
- **GPU only ~39% utilized even solo** (mem-controller `mem%` ~8%): the workload is **host-bound**
  (CPU batch prep + H2D for 1385 tiny batches), not GPU-compute-bound and not GPU-memory-bound.
  Confirms why AMP never helped (§10).

**Result 2 (the real finding) — a 5× gap between single-node now and the April 28-node run, same
everything.**

| | per-epoch | data | compute | eval | per-fold (100 ep) |
|---|---|---|---|---|---|
| April production (28 nodes) | 25.4 s | 3.1 | 20.9 | 1.4 | 42.7 min |
| Now (single node) | 5.0 s | 1.7 | 3.0 | 0.4 | ~8-9 min |

- Identical dataset, bundle (k-mer k=6, bs=128), A100. `git log` shows **no compute-path change**
  since April (only eval / CSV-parser / post-hoc fixes). So the 5× is **environmental /
  cluster-scale**, present at 28 nodes, absent single-node. It slowed *all* phases (compute 7×,
  data 1.8×, eval 3.5×) — a system-wide effect, most likely **shared Eagle Lustre I/O contention**
  from ~112 concurrent processes writing logs/progress. **Unconfirmed** — needs a multi-node test.

**Honest implication (do not overclaim).** Single-node a fold is ~8-9 min, not 44 min. But
**single-node speed does not prove 28-node speed**: if the 5× is cluster-scale I/O contention, a
fresh 28-node run may still hit ~25 s/epoch. So the April numbers (44 min/fold, ~239 GPU-h,
~62% end-to-end) stand *for that run*; whether a re-run reproduces them or the ~5×-faster
single-node rate is the open question.

**Correction of a prior hypothesis.** The earlier guess — that 4-fold packing caused the slowdown —
is **wrong**. Measured directly (solo vs 4-share, same session), packing is 94% efficient. The gap
is single-node-vs-cluster, not within-node packing. Catching this is exactly why we measure rather
than reason.

### Phase 0c — executed 2026-07-11: node-count scaling test (batch, 8-node debug-scaling)

**Method.** Same fold, 4-packed per node, at K = 2, 4, 8 nodes in one batch job
(`logs/phase0c/phase0c.pbs` → `run_scaling.sh` → `node_4share.sh`), 10 epochs. Per-epoch from
each fold's `training_history.csv`. Analysis `scratchpad/p0c.py`.

| K (nodes) | processes | per-epoch | data | compute | eval | startup |
|---|---|---|---|---|---|---|
| 1 (phase0b, bash loop) | 4 | **5.0 s** | 1.6 | 2.9 | 0.3 | — |
| 2 (mpiexec) | 8 | **25.0 s** | 2.9 | 19.9 | 2.2 | 134 s |
| 4 (mpiexec) | 16 | **25.1 s** | 3.0 | 20.0 | 2.0 | 129 s |
| 8 (mpiexec) | 32 | **25.1 s** | 3.1 | 20.0 | 1.9 | 129 s |
| 28 (April prod, mpiexec) | 112 | 25.4 s | 3.1 | 20.9 | 1.4 | — |

**Finding: the 5× is a step function at 2 nodes, then flat** through 8 (and equal to April's 28).
Flatness across node count **rules out** progressive shared-resource contention (Lustre bandwidth,
network) — those grow with node count. Startup is flat ~130 s at every K, so the matrix load isn't
it either. So the earlier "cluster-scale Lustre I/O" guess (0b) is **wrong**.

**The one thing that differs between K=1 and K≥2:** phase0b (K=1, 5 s) launched the 4 folds with a
plain `bash` loop; phase0c and the prod launcher launch each node via **`mpiexec -n 1 --ppn 1`
without `--depth`**. Same process tree otherwise (shell → 4 background python).

**Leading hypothesis: mpiexec default CPU binding.** Without `--depth`, PALS `mpiexec` pins each
node's rank (and its 4 child folds) to a small CPU subset. The workload is **host-bound** (GPU only
39% used — Phase 0b), so starving it of CPU cores slows every epoch ~5×. The prod launcher uses the
same `mpiexec` pattern → explains April's 25 s. The ALCF Parsl config pointedly sets
`MpiExecLauncher(overrides="--depth=64 --ppn 1")` and a `cpu_affinity` list — i.e., ALCF expects you
to hand the rank all cores.

**Fix to confirm:** `mpiexec --depth=64 --cpu-bind depth` (or `--cpu-bind none`) so each rank gets
the whole node's cores. Confirmation test: re-run the 2-node wave with the flag; expect ~5 s/epoch.

**Impact if confirmed:** the prod sweep is ~5× faster — per-fold ~44 min → ~9 min, training makespan
~2.3 h → ~0.5 h — for a one-line launcher change. This, not the 62% efficiency, is the headline of
Phase 0.

**CONFIRMED 2026-07-12 (`logs/phase0c/phase0conf.pbs`, 2-node A/B, same fold under 3 binding modes):**

| mpiexec mode | per-epoch | compute |
|---|---|---|
| `default` (current prod) | 25.2 s | 20.3 |
| `--cpu-bind none` | 5.2 s (4.8×) | 3.0 |
| `--depth=64 --cpu-bind depth` | 5.2 s (4.9×) | 3.0 |

Root cause **proven**: `mpiexec`'s default CPU binding pins each node's rank + its 4 folds to a
core subset, starving the host-bound workload; giving the rank all cores restores single-node speed.
`--cpu-bind none` and `--depth=64 --cpu-bind depth` are identical (5.2 s), so it is unambiguously CPU
binding. **Fix:** add `--depth=64 --cpu-bind depth` to the `mpiexec` call in
`scripts/run_allpairs_polaris_prod.sh` (the per-pair dispatch, ~line 400). Then the full 28-pair
sweep runs ~5× faster and becomes the clean, saved re-run.

### Phase 0d — executed 2026-07-12: fix validated through the real launcher (2 pairs, 2 nodes)

Ran the **fixed** `run_allpairs_polaris_prod.sh` on 2 pairs (ha_na, pb2_pb1) / 2 `debug` nodes,
reusing the April datasets (`DATASET_MANIFEST`, no Stage 3), full production config (100 epochs,
12-fold CV). Exercises the real path `mpiexec --depth=64 --cpu-bind depth` → `run_cv_lambda.py`
→ 4 Popen folds. Wrapper: `logs/phase0d/validate_2pair.pbs`.

| | per-epoch | per-fold | 2-pair run |
|---|---|---|---|
| April prod (default binding) | 25.4 s | 42.7 min | — |
| Fixed launcher (this run) | **5.0 s** | **8.5 min** | **28m51s** |

- **5.0× speedup confirmed end-to-end through the production launcher**; 24/24 folds succeeded.
- Correctness preserved: AUC 0.993 (HA/NA), 0.995 (PB2/PB1), matching April — the fix changes only
  speed, not results.
- Run dir `allpairs_prod_20260712_194501/` with `pbs_job.log` saved — the first properly-archived
  timing artifact.

**Phase 0 conclusion.** The headline is not "62% efficiency"; it is: the sweep ran ~5× slower than
the hardware allows because the launcher's `mpiexec` used default CPU binding on a host-bound
workload. One-line fix, validated end-to-end. Green light for the full 28-pair sweep — training
makespan ~2.3 h → ~0.5 h.

### Full k-mer baseline re-run — done 2026-07-13 (`logs/sweep/full_sweep_kmer.pbs`)

Fixed launcher, all 28 pairs / 28 nodes, reusing the April datasets. **28/28 pairs, 336/336 folds,
29m32s** (`allpairs_prod_20260713_171445/`, `pbs_job.log` saved). Per-epoch **5.0 s** (range
4.8-5.3), per-fold **8.4 min** — the fix held at 28-node scale (**5.1×** vs April's 25.4 s /
42.7 min). `data_time` stayed flat at 1.7 s, so the 112-process concurrent Lustre matrix load was a
non-issue. Includes the two pairs that failed in April → the first complete, fast, saved unfiltered
k-mer baseline. Metrics match April (AUC 0.99x), so the fix changed only speed.

### Reproduce & CAR

**Reproduce the k-mer baseline (verified 2026-07-13):**
```bash
qsub logs/sweep/full_sweep_kmer.pbs
```
Fixed launcher, all 28 pairs / 28 nodes, reusing the April datasets via `DATASET_MANIFEST`
(`allpairs_prod_val_unfilt_20260413_151649/dataset_manifest.json`). On disk: `protein_final.csv`,
`kmer_features_k6.npz` (+ `master_esm2_embeddings.h5` for the ESM-2 variant); 28/28 datasets with
12 folds. Output: `allpairs_prod_<ts>/` (28-pair summary, heatmaps, `pbs_job.log`) + 336 per-fold
model dirs (`best_model.pt`). Last run: 28/28, 336/336 folds, 29m32s, 8.4 min/fold.

**CAR (segmatch 28-pair sweep — HPC debugging):**
- **Challenge:** a 28-protein-pair × 12-fold CV sweep (336 GPU training jobs) ran ~5× slower on
  Polaris than a single-GPU baseline predicted (25 s vs 5 s/epoch) — ~44 min/fold, 3.5 h/sweep.
- **Action:** profiled with `nvidia-smi dmon` → workload is host-bound (GPU util ~39%, not compute-
  or memory-bound); a 1→2→4→8-node scaling test showed a step-function 5× at the 2-node (mpiexec)
  boundary and *flat* thereafter — ruling out cluster-scale contention and falsifying my first
  hypothesis (fold packing); a 3-way mpiexec CPU-binding A/B isolated the cause: default binding
  pinned each node's rank and its 4 host-bound folds to a core subset.
- **Result:** one-line launcher fix (`mpiexec --depth=64 --cpu-bind depth`), validated end-to-end —
  42.7 → 8.4 min/fold (5.1×), 336/336 folds, metrics unchanged, provenance saved. Reframed the
  team's "62% efficiency" number: the loss was a missing CPU-affinity flag, not fundamental
  inefficiency.

---

## Phase 1 — Profiling (learn to read the metrics, ~1 day)

**Purpose:** learn the standard GPU-profiling tools and vocabulary, and resolve whether the
"compute-dominated" epoch is real GPU compute or memory/CPU-bound.

**Status:** the host-bound verdict is already established (GPU ~39% util — Phase 0b/0c). Remaining
deliverable: a `torch.profiler` **trace** that shows it visually (GPU-idle gaps, H2D copies,
DataLoader time) + a short tool-choice note.

**Integrated:** `train_pair_classifier.py` now has an opt-in `--profile_steps N` flag (profiles N
real training steps, writes a trace to `<run>/profile/`, prints the op table, then exits). Run one
fold on a `debug` node:
```bash
source scripts/polaris_env.sh
CUDA_VISIBLE_DEVICES=0 python3 src/models/train_pair_classifier.py \
  --config_bundle flu_28p_ha_na --cuda_name cuda:0 \
  --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_28p_ha_na_val_unfilt_20260413_151650/fold_0 \
  --profile_steps 20 --run_output_subdir phase1_profile
```

**Why `torch.profiler`, not `ncu` (state this in the write-up):** `ncu` profiles *inside* individual
kernels and is blind to the between-kernel host stall (DataLoader + H2D + launch gaps) that dominates
this host-bound epoch; its per-kernel counters describe the tiny GEMMs that are not the bottleneck.
`torch.profiler` (low-overhead, whole-timeline) shows where the epoch time actually goes. `ncu` is
reserved for the compute-bound GenSLM case (where it profiles the GEMM/tensor-core efficiency that is
the bottleneck).

**Result (2026-07-14, `phase1_profile`, 20 steps).** GPU **idle 82%** of the profiled window (busy
18% = kernels 18.4 ms + H2D 4.0 ms over a 126 ms window) — the host-bound signature at the trace
level. The dominant CPU cost is **batch collation**: `aten::stack` (21.5 ms) + `aten::cat` (17.4 ms)
assembling the 4096-dim k-mer vectors into batches, then `aten::to`/`copy_` (~23 ms) for the H2D
copy; the MLP GEMMs (`addmm`/`mm`) are only ~16 ms. Fix direction is collation/transfer (GPU-side
collate, or indexing one contiguous tensor), not the GPU math. Makes the `ncu` point concrete: it
could only see the 18% (kernels); the 82% that dominates is invisible to it. (Absolute times are
profiler-inflated; the ratios are the signal.) Trace: `phase1_profile/profile/*.pt.trace.json`.

**Do:**
1. **GPU utilization timeline.** During the solo fold, sample `nvidia-smi dmon -s um`
   (utilization, memory) or DCGM at ~1 Hz to a CSV; plot SM utilization and memory-used vs
   time. Near-idle SMs with high wall-time ⇒ starved GPU (memory/CPU-bound); high SM ⇒
   compute-bound. This is the honest test of the "compute-bound?" question.
2. **One `nsys` trace** of a few training steps:
   `nsys profile -t cuda,osrt,nvtx -o fold_trace python ...` (few epochs). Read the timeline:
   fraction in kernels vs H2D copies vs gaps; whether copies overlap compute (they only can
   from pinned memory — and this run uses `pin_memory=false`, so expect non-overlapped
   copies; note that in the write-up).
3. **Roofline framing.** Estimate the MLP's arithmetic intensity (FLOPs per byte for
   `[4096→512→256→64→1]` at batch 128) and place it on an A100 roofline. Expect
   memory-bandwidth-bound (low intensity) — the correct, honest characterization, consistent
   with AMP not helping (`speed_up.md` §4).
4. **Rigor artifacts:** write `versions.txt` (torch, CUDA, driver, NCCL, Python, git SHA),
   a controlled-variables list, and a short profiling write-up with the two figures.

**Honesty:** use CUDA-event or `torch.cuda.synchronize()` timing for any GPU-time claim
(async caveat, hardware_notes §6). MBU/MFU are **not** meaningful for this tiny fp32 MLP —
do not compute them here (they belong to Phase 2). Say so.

**Done when:** utilization plot + one nsys timeline + a one-paragraph memory-vs-compute-bound
verdict with the arithmetic-intensity number, plus `versions.txt`.

---

## Phase 2 — ESM-2 extraction inference throughput (the honest inference story, ~half day)

**Purpose:** the ESM-2 (650M) embedding-extraction step is a *genuine inference workload*
(forward passes over ~880K protein sequences), and the one place where inference-throughput
vocabulary honestly applies — unlike the MLP. Currently listed as PENDING (unquantified).

**Do:** re-run extraction over a representative slice with logging: **sequences/s**,
**tokens/s**, GPU utilization (dmon), peak memory, across a small **batch-size sweep**.
Optionally estimate a caveated **MFU** for the encoder forward pass (state assumptions;
it is only meaningful because this is a 650M model, not the MLP).

**Record:** throughput table + one figure (throughput vs batch size), GPU-util during
extraction, the config, and `versions.txt`.

**Done when:** a measured extraction-throughput number exists with its controlled variables,
replacing the PENDING line.

---

## Phase 3 — Port the launcher to Parsl (tooling, ~2-3 days)

**Progress (2026-07-14):** parsl 2026.07.06 installed in the venv; `scripts/run_allpairs_parsl.py`
written (v1). Each fold is a `bash_app` over the unchanged training entrypoint; `HighThroughputExecutor`
(`available_accelerators=4` -> one GPU/worker, auto `CUDA_VISIBLE_DEVICES`) + `PBSProProvider` +
`MpiExecLauncher(--depth=64)` + NUMA-aware `cpu_affinity` (the same binding the mpiexec launcher was
missing) + `retries=2`. The config constructs cleanly against the installed version (kwargs verified).
**Next:** validate on 2 pairs / 2 nodes vs the mpiexec numbers (expect ~5 s/epoch, ~8-9 min/fold):
```bash
python3 scripts/run_allpairs_parsl.py --pairs flu_28p_ha_na flu_28p_pb2_pb1 \
  --nodes 2 --queue debug --epochs 20 --tag parslval \
  --dataset_manifest models/flu/July_2025/allpairs_prod_val_unfilt_20260413_151649/dataset_manifest.json
```

**Validated (2026-07-14).** Ran end-to-end: **24/24 folds, per-epoch 5.3 s** (range 5.2-5.4), ~9
min/fold at 100 ep — matching the mpiexec 5.0 s / 8.5 min. So Parsl's NUMA-aware `cpu_affinity`
delivers the same speed as the hand-tuned `--depth=64` *and* avoids the default-binding trap: the
"Parsl would have prevented the bug" thesis, confirmed. PBS submission, worker startup under the venv,
GPU packing, and retries all worked. The first attempt exposed two v1 bugs, both fixed: (1) missing
`-l filesystems=home:eagle` (Polaris rejects the job without it), and (2) a module global (`PROJECT`)
referenced inside the `bash_app` — apps must be self-contained, so pass values as arguments.


**Purpose:** replace the bespoke bash + hand-rolled `wait_any` GPU pool with **Parsl**
(ALCF-native many-task engine), for failure isolation, retries, and a documented workflow.
Validate it reproduces the `mpiexec` numbers before trusting it. Ray is a later extension.

**Why Parsl here (user-chosen):** understanding the system (scheduling, packing, retries,
tradeoffs) matters more than the tool; and it refreshes Parsl mechanics shared with another
project. Parsl is purpose-built for PBS/Polaris (least bootstrap friction vs Ray's
head/worker cluster-on-PBS setup).

**Do:**
1. **Model the sweep as Parsl apps.** Each fold = one `@python_app` (or `@bash_app`
   wrapping the existing training entrypoint) with `retries=N` for failure isolation. The 336
   apps are submitted; Parsl's `HighThroughputExecutor` handles placement and GPU packing —
   replacing *both* the per-node `mpiexec` dispatch and the intra-node `wait_any` pool.
2. **Use the ALCF Polaris config** (confirm exact values against the live ALCF doc before
   running — values below are the documented skeleton):
   - `HighThroughputExecutor(available_accelerators=4, max_workers_per_node=4,
     cpu_affinity="list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39", ...)` — one worker
     per A100 with NUMA-aware CPU binding.
   - `PBSProProvider(select_options="ngpus=4", nodes_per_block=<N>, max_blocks=1,
     cpus_per_node=32, queue=..., account="IMPROVE_Aim1", ...)`.
   - `MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1")`.
   - `worker_init="export TMPDIR=/tmp; <module load / venv activate>"` — the `TMPDIR=/tmp`
     is required to avoid an AF_UNIX "path too long" error (ALCF known issue, Sep 2025+).
   - Enable Parsl checkpointing + `retries` so a failed fold re-runs without re-running the
     whole sweep.
3. **Validate against `mpiexec`.** Run 2 pairs (24 folds) under Parsl; confirm the CV
   metrics (AUC/F1/Brier per fold) match the `mpiexec` run within run-to-run noise, and that
   GPU packing (4 concurrent folds/node) actually happens (check `nvidia-smi`/dmon).
4. **Write the comparison.** `mpiexec`-dispatch vs Parsl: lines of code, failure handling
   (Parsl retries vs manual re-run commands), observability, and the packing mechanism.

**Keep the `mpiexec` launcher** as the reference until Parsl matches it. Do not delete or
gate the science on the port.

**Done when:** a Parsl launcher runs a 2-pair subset on Polaris, reproduces the `mpiexec`
CV metrics within noise, retries a killed fold automatically, and the comparison write-up
exists.

---

## Honesty guardrails (carry into every write-up)

- **Independent single-GPU tasks; no collectives.** Never imply distributed training or MPI
  communication. The parallelism is task-level fan-out.
- **k-mer, not ESM-2**, for the 336-run sweep.
- **Measured vs projected** stated every time (esp. `t(1)`, `S`, `E`).
- **Name the machine and fabric** (Polaris, A100, Slingshot 11, PBS Pro; no FP8 on Ampere).
- **A synonym must not upgrade a claim.** "Ran"/"measured"/"profiled" mean exactly that.
- **MBU/MFU only where a large model makes them meaningful** (Phase 2 extraction), never for
  the MLP.

---

## Sources (terminology confirmed against these)

- NVIDIA CUDA C++ Programming Guide — pinned/page-locked memory, async execution.
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA Technical Blog — "How to Optimize Data Transfers in CUDA C/C++" (pinned memory,
  async DMA). https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
- PyTorch — "A guide on good usage of `non_blocking` and `pin_memory()`" +
  `torch.utils.data` DataLoader docs.
  https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
- TensorFlow — "Use a GPU" (memory growth, `TF_FORCE_GPU_ALLOW_GROWTH`).
  https://www.tensorflow.org/guide/gpu
- ALCF — "Parsl on Polaris" (HighThroughputExecutor / PBSProProvider / MpiExecLauncher,
  one-task-per-GPU, `TMPDIR=/tmp`). https://docs.alcf.anl.gov/polaris/workflows/parsl/
- ALCF — "Example Job Scripts" / "Using GPUs on Polaris".
  https://docs.alcf.anl.gov/running-jobs/example-job-scripts/
- HPC Wiki — "Scaling" (strong/weak scaling, speedup, parallel efficiency, Amdahl, load
  imbalance). https://hpc-wiki.info/hpc/Scaling
- In-repo: `speed_up.md` §8, `docs/hardware_notes.md` §1/§3/§6/§11.
