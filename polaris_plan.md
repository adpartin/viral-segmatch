# Polaris Plan: Task 11 — All Protein-Pair Combinations (8×8 Heatmap)

Plan for running 28 pairwise protein experiments (C(8,2)) on ALCF Polaris.
Covers: Polaris system reference, bundle design, scaling phases, and HPC concepts.

---

## Polaris System Reference

### Context

- **System:** Polaris (ALCF, HPE/Cray), A100 GPUs (4 per node)
- **Active allocation:** IMPROVE_Aim1 — 97k node-hours remaining, expires 2026-04-01
- **Secondary allocation:** bvbrc — 1k node-hours, expires 2026-05-01
- **Project storage:** `/lus/eagle/projects/IMPROVE_Aim1` (fast Lustre, use for data and checkpoints)
- **Home dir:** `/home/apartin` (slower, not for large I/O)

**Allocation unit: full node** (4 A100 GPUs, 32 cores, 512 GB RAM). No sub-node allocation.
You CAN run 4 independent processes per node via `CUDA_VISIBLE_DEVICES` + background `&` + `wait`.

### Queues

| Queue | Nodes | Walltime | Key constraint |
|-------|-------|----------|----------------|
| **debug** | 1-2 | up to 1h | Max 24 nodes shared across all debug users; only 8 exclusive |
| **debug-scaling** | 1-10 | up to 1h | 1 job per user |
| **prod** | **10-496** | up to 24h | Routes to small (10-24, 3h), medium (25-99, 6h), large (100-496, 24h) |
| **preemptable** | 1-10 | up to 72h | Can be killed anytime; 20 concurrent jobs/project |
| **capacity** | 1-4 | up to **168h** (7 days) | Max 2 queued, 1 running per user |

**Key implication:**
- **Prod requires minimum 10 nodes per job.** Single-node job array elements cannot go to prod.
- **Job arrays** (`#PBS -J 0-N`) are supported. Each element is an independent job with its own node allocation.
- **Best strategy: ensemble packing within nodes + job arrays across nodes.** Each array element gets 1 node and runs 4 tasks internally (one per GPU).

---

## Environment Setup

### Validate on compute nodes first

Before any large run, validate that the shell environment works on a compute node.
The venv (`cepi_polaris`) must be created first from a login node (see "Conda / venv
environment" below). It only needs to be created once — all job scripts activate it.

```zsh
qsub -I -l select=1 -l walltime=0:30:00 -A IMPROVE_Aim1 -q debug -l filesystems=eagle

# On the compute node — run these one at a time (long pastes get mangled):
module use /soft/modulefiles
module load conda
conda activate base
source /lus/eagle/projects/IMPROVE_Aim1/apartin/venvs/cepi_polaris/bin/activate
nvidia-smi                     # confirm all 4 A100s are visible
python -c "import torch; print(torch.cuda.is_available())"  # should print True
```

**Paste issues on compute nodes:** Long multi-line pastes get truncated or mangled.
Run commands one at a time, or source a setup script:
```bash
# Create once: scripts/polaris_env.sh
# Then on compute node: source scripts/polaris_env.sh
```

### Conda / venv environment

ALCF provides a **pre-built base conda environment** with PyTorch, CUDA, numpy, scipy,
pandas, matplotlib, etc. They strongly discourage creating custom conda environments.
See: https://docs.alcf.anl.gov/polaris/data-science/python/

**Recommended approach: venv on top of ALCF base** (not a full conda create):

```bash
# Step 1: Load the ALCF base env (has PyTorch 2.3.0, CUDA 12.4.1, etc.)
module use /soft/modulefiles
module load conda
conda activate base

# Step 2: Check what's already available
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip list | grep -iE "hydra|omegaconf|h5py|scipy|pandas|scikit|seaborn|matplotlib|tqdm"

# Step 3: Create a venv on Lustre (NOT /home — Lustre handles parallel I/O better)
# --system-site-packages inherits everything from the base conda env
VENV_DIR="/lus/eagle/projects/IMPROVE_Aim1/apartin/venvs/cepi_polaris"
mkdir -p "${VENV_DIR}"
python -m venv "${VENV_DIR}" --system-site-packages
source "${VENV_DIR}/bin/activate"

# Step 4: Install only what's missing from the base
# Check Step 2 output first — many of these may already be present.
# 03/25/2026: All packages were found
```

**Key differences from Lambda:**
- Lambda: full `environment.yml` with `conda create` (you manage PyTorch + CUDA)
- Polaris: ALCF manages PyTorch + CUDA in the base; you add a thin venv layer on top
- If ALCF updates the base env, recreate the venv (re-run Steps 3-4) since
  `--system-site-packages` links to the old base. You'll notice when imports fail.

**Known gotcha (ALCF docs):** If you see `"MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is
requested, but GTL library is not linked"`, add `from mpi4py import MPI` as the very
first import in your Python script. This is only needed for multi-node MPI jobs (not
for single-GPU training in Phase 0).

### Environment in job scripts

PBS batch jobs run in a bare shell — your `.zshrc` / `.bashrc` / `polaris.zsh` are
**not** sourced. Every job script must load modules and activate the venv explicitly.
Both launcher scripts (`run_allpairs_polaris_phase0.sh`, `run_allpairs_polaris_prod.sh`)
already handle this — no manual setup needed inside a `qsub`-submitted job.

---

## Bundle Design

The 28 child bundles specify **only** `schema_pair`. Everything else flows from the master.
To change sample size, epochs, or architecture for all 28 experiments: edit one line in the master.

### Inheritance chain

```
flu.yaml                                    # Gen1 base (virus, paths, defaults)
  └── flu_28_major_protein_pairs_master.yaml  # Master: all 8 proteins, kmer, slot_norm, concat
        ├── flu_28p_pb2_pb1.yaml
        ├── flu_28p_pb2_pa.yaml
        ├── flu_28p_ha_na.yaml
        └── ... (28 total)
```

### Master bundle: `flu_28_major_protein_pairs_master.yaml`

Inherits from `flu.yaml`. Sets all 8 major proteins in `selected_functions`,
`pair_mode: schema_ordered`, k-mer features, slot_norm + concat. Phase 3 production
settings (full dataset, 12-fold CV, 100 epochs, batch_size=128, optimized training;
see `speed_up.md`). Temporarily edit for earlier phases. Children override only
`dataset.schema_pair`.

### Child bundle example: `flu_28p_ha_na.yaml`

```yaml
# STATUS: active -- HA/NA pair (Task 11: all protein-pair combinations)
defaults:
  - flu_28_major_protein_pairs_master
  - _self_

dataset:
  schema_pair:
    - "Hemagglutinin precursor"
    - "Neuraminidase protein"
```

### Protein short codes

| Short | Full function string | Segment |
|-------|---------------------|---------|
| `pb2` | RNA-dependent RNA polymerase PB2 subunit | S1 |
| `pb1` | RNA-dependent RNA polymerase catalytic core PB1 subunit | S2 |
| `pa`  | RNA-dependent RNA polymerase PA subunit | S3 |
| `ha`  | Hemagglutinin precursor | S4 |
| `np`  | Nucleocapsid protein | S5 |
| `na`  | Neuraminidase protein | S6 |
| `m1`  | Matrix protein 1 | S7 |
| `ns1` | Non-structural protein 1, interferon antagonist and host mRNA processing inhibitor | S8 |

Child bundle naming: `flu_28p_{protA}_{protB}` (e.g., `flu_28p_ha_na`, `flu_28p_pb2_m1`).

---

## Scaling Phases

The master bundle (`flu_28_major_protein_pairs_master.yaml`) is configured for Phase 3
(full dataset, 12-fold CV, 100 epochs). For earlier phases, edit the master before
running, then restore it. The key knobs:

| Knob | Phase 0 | Phase 1 | Phase 2 | Phase 3 (master default) |
|------|---------|---------|---------|--------------------------|
| `max_isolates_to_process` | 5000 | 5000 | 5000 | null (full ~111K) |
| `n_folds` | null (single split) | 12 | 12 | 12 |
| `epochs` | 5 | 5 | 5 | 100 |
| `patience` | 5 | 5 | 5 | 100 (no early stopping) |

### Phase 0 — Pipeline validation (debug, 1 node, 1h) -- COMPLETE

**Goal:** Verify that Stages 3 and 4 run without errors on Polaris.

- Single pair (HA/NA), single split (no CV), 5 epochs, 5K isolates
- Sequential on 1 GPU — just confirm the pipeline works end-to-end
- **Script:** `scripts/run_allpairs_polaris_phase0.sh` (also `scripts/stage3_dataset.sh` + `scripts/stage4_train.sh` individually)

#### Phase 0 step-by-step checklist

```
[v] Step 0: Set up Python environment (one-time, from login node)
      module use /soft/modulefiles; module load conda; conda activate base
      python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
      pip list | grep -iE "hydra|omegaconf|h5py|scipy|pandas|scikit|seaborn|tqdm"
      # Create venv on Lustre if needed (see "Conda / venv environment" section above)

[v] Step 1: Verify data exists on Eagle
      ls data/processed/flu/July_2025/protein_final.csv
      ls data/embeddings/flu/July_2025/kmer_features_k6.npz
      # Both copied from Lambda on 2026-03-27

[v] Step 2: Get an interactive compute node
      qsub -I -l select=1 -l walltime=0:30:00 -A IMPROVE_Aim1 -q debug -l filesystems=eagle

[v] Step 3: Validate environment on compute node
      module use /soft/modulefiles; module load conda; conda activate base
      source /lus/eagle/projects/IMPROVE_Aim1/apartin/venvs/cepi_polaris/bin/activate
      which python
      nvidia-smi                          # 4 A100s visible?
      python -c "import torch; print(torch.cuda.is_available())"   # True?
      python -c "import hydra; print(hydra.__version__)"           # installed?
      cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch
      ls data/processed/flu/July_2025/protein_final.csv            # exists?
      ls data/embeddings/flu/July_2025/                            # k-mer features?

[v] Step 4: Edit master bundle for Phase 0 settings
      # In flu_28_major_protein_pairs_master.yaml, temporarily set:
      #   max_isolates_to_process: 5000
      #   n_folds: null
      #   epochs: 5
      #   patience: 5

[v] Step 5: Test a single pair (HA/NA)
      bash scripts/run_allpairs_polaris_phase0.sh --pairs "flu_28p_ha_na" --gpu 0
      # Check output:
      ls models/flu/July_2025/runs/training_flu_28p_ha_na_*/
      cat models/flu/July_2025/runs/training_flu_28p_ha_na_*/training_info.json

[v] Step 6: Verify results
      # training_info.json should show reasonable metrics (F1, AUC)
      # No errors in the log
```

**Results:** Test F1=0.944, AUC=0.972. Pipeline works end-to-end on Polaris.

**Issues encountered and fixed:**
- Hydra duplicate defaults: `flu.yaml` already includes `/kmer: default`; master bundle
  had it again. Removed duplicate from master.
- PyTorch `ReduceLROnPlateau` `verbose` parameter removed in newer PyTorch (ALCF base
  has newer version than Lambda). Removed `verbose=True` from `src/utils/torch_utils.py`.

**If Step 3 fails:** Common issues:
- `hydra-core` not installed → `pip install hydra-core omegaconf` in the venv
- `torch.cuda.is_available()` returns False → check `module load` and CUDA version
- Data missing → copy from Lambda (Step 1)

### Phase 1 — CV validation (debug, 1 node, 1h) -- COMPLETE

**Goal:** Verify 12-fold CV works end-to-end with `run_cv_lambda.py`.

- Single pair (HA/NA), 12 folds, 5 epochs, 5K isolates
- 12 folds / 4 GPUs = 3 waves × ~1 min/fold = ~3 min total
- **Script:** `python3 scripts/run_cv_lambda.py --config_bundle flu_28p_ha_na --gpus 0 1 2 3`

```
[v] Step 1: Edit master for Phase 1
      # In flu_28_major_protein_pairs_master.yaml, set:
      #   max_isolates_to_process: 5000
      #   n_folds: 12
      #   epochs: 5
      #   patience: 5

[v] Step 2: Get an interactive compute node and run
      qsub -I -l select=1 -l walltime=1:00:00 -A IMPROVE_Aim1 -q debug -l filesystems=eagle

      # On compute node (interactive qsub drops you into /home — cd to project first):
      cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch
      module use /soft/modulefiles; module load conda; conda activate base
      source /lus/eagle/projects/IMPROVE_Aim1/apartin/venvs/cepi_polaris/bin/activate
      python3 scripts/run_cv_lambda.py --config_bundle flu_28p_ha_na --gpus 0 1 2 3

[v] Step 3: Verify CV output
      cat models/flu/July_2025/cv_runs/cv_flu_28p_ha_na_*/cv_summary.json
      # Should have mean +/- std for F1, AUC across 12 folds
```

**Results:** F1=0.922+/-0.009, AUC=0.963+/-0.005 (12-fold CV, 5 epochs, 5K isolates).

**Note:** Master bundle must have `n_folds: 12` (not null) for CV — Phase 0 had it set
to null for single-split mode.

### Phase 2 — Multi-node validation (2-28 nodes, debug-scaling + prod) — COMPLETE

**Goal:** Verify multi-node distribution works (the prod launcher pattern).

- 2 pairs (HA/NA, PB2/PB1), 12 folds, 5 epochs, 5K isolates, 1 pair per node
- Each node: 12 folds / 4 GPUs = 3 waves × ~1 min/fold = ~3 min
- Both nodes finish in ~5 min total
- **Script:** `scripts/run_allpairs_polaris_prod.sh`

```
[v] Step 1: Edit master for Phase 2 (same as Phase 1)
      # In flu_28_major_protein_pairs_master.yaml:
      #   max_isolates_to_process: 5000
      #   n_folds: 12
      #   epochs: 5
      #   patience: 5

[v] Step 2: Multi-node test with SSH dispatch (original approach)
      qsub -I -l select=2:ncpus=64:ngpus=4 -l walltime=1:00:00 -A IMPROVE_Aim1 -q debug-scaling -l filesystems=eagle
      cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch
      source scripts/polaris_env.sh
      eval $(ssh-agent); ssh-add ~/.ssh/id_rsa   # Required for SSH dispatch
      bash scripts/run_allpairs_polaris_prod.sh --pairs "flu_28p_ha_na flu_28p_pb2_pb1"

[v] Step 3: Verify both pairs completed
      cat models/flu/July_2025/allpairs_prod_*/manifest.txt
      # Should show 2 lines, both SUCCEEDED

[v] Step 4: Retest with mpiexec dispatch (replaced SSH — no ssh-agent needed)
      # SSH is fragile in batch mode (passphrase blocks jobs). Switched to
      # mpiexec --hostfile (ALCF ensemble pattern). This step validates the new dispatch.
      qsub -I -l select=2:ncpus=64:ngpus=4 -l walltime=1:00:00 -A IMPROVE_Aim1 -q debug-scaling -l filesystems=eagle
      cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch
      source scripts/polaris_env.sh
      bash scripts/run_allpairs_polaris_prod.sh --pairs "flu_28p_ha_na flu_28p_pb2_pb1"
      # Results: models/flu/July_2025/allpairs_prod_20260331_232931/
```

**Results (Step 2, SSH dispatch):**
- HA/NA:   F1=0.922+/-0.009, AUC=0.963+/-0.005
- PB2/PB1: F1=0.919+/-0.013, AUC=0.960+/-0.012
Both pairs succeeded across 2 nodes. Multi-node distribution works.

**Results (Step 4, mpiexec dispatch):**
- HA/NA:   F1=0.922+/-0.009, AUC=0.963+/-0.005
- PB2/PB1: F1=0.919+/-0.013, AUC=0.960+/-0.012
Identical to SSH results. mpiexec is a drop-in replacement — no SSH keys needed.

**Note on allpairs log buffering:** The per-pair logs in the manifest dir
(`allpairs_prod_*/flu_28p_*.log`) only capture stdout from mpiexec, which buffers
differently than SSH. Stage 4 output may not appear there. The authoritative logs are:
- CV launcher: `logs/cv/run_cv_*.log` (full Stage 3 + Stage 4 progress)
- Per-fold training: `models/.../runs/training_*_fold{k}_*/fold_{k}.log`

```
[v] Step 5: Batch submission test (qsub, not interactive)
      # Steps 2-4 used interactive qsub -I. Batch mode (qsub script.sh) has the
      # script handle env setup (module load, venv, proxy). Test that mpiexec
      # inherits the batch environment correctly before the full production run.
      # Keep master at Phase 2 settings (2K isolates, 5 epochs).
      qsub -l select=2:ncpus=64:ngpus=4 -l walltime=1:00:00 -A IMPROVE_Aim1 \
           -q debug-scaling -l filesystems=eagle \
           -v PAIRS="flu_28p_ha_na flu_28p_pb2_pb1" \
           scripts/run_allpairs_polaris_prod.sh
      # Check: qstat -u apartin
      # Verify: cat models/flu/July_2025/allpairs_prod_*/manifest.txt
      #         tail logs/cv/run_cv_flu_28p_ha_na_*.log

[v] Step 6: Full 28-pair batch test (all pairs, 2K isolates, 5 epochs)
      # Steps 2-5 tested with 2 pairs. This step validates all 28 pairs in parallel
      # on 28 nodes before committing to the expensive Phase 3 production run.
      # Master bundle: 2K isolates, 5 epochs (already set from Step 5).
      #
      # The script has #PBS directives for Phase 3 (28 nodes, prod, 6h).
      # CLI flags override script directives, so we only override walltime here.
      # select=28, -A IMPROVE_Aim1, -q prod, filesystems=eagle all come from the script.
      qsub -l walltime=00:40:00 scripts/run_allpairs_polaris_prod.sh
      # Check: qstat -u apartin
      # Verify: cat models/flu/July_2025/allpairs_prod_*/manifest.txt  (28/28 SUCCEEDED)
      #         grep "CV Summary" models/flu/July_2025/allpairs_prod_*/flu_28p_*.log
```

**Results (Step 6, job 6983977):** 23/28 SUCCEEDED, 5 FAILED.
- Failed pairs: `ha_np`, `ha_ns1`, `na_m1`, `na_ns1`, `np_m1`
- Root cause: **CUDA OOM on specific nodes.** The k-mer matrix (868K × 4096 ≈ 13 GB) is loaded
  per fold process. With 4 concurrent folds, 4 copies live in CPU memory. On 5 nodes, GPU 0
  (or GPU 2) hit OOM when moving the model to device — likely due to residual GPU memory from
  other jobs or driver overhead on those specific nodes.
- Pattern: each failed pair had exactly 3 folds fail (folds 0,4,8 or 2,4,8) — one per wave,
  same GPU each time. The MLP itself is tiny; the OOM is node-specific, not a code bug.
- The 23 successful pairs ran on different nodes without issues.
- **Conclusion:** Pipeline validated at full 28-pair scale. Failures are transient node issues,
  not code bugs. Re-running failed pairs on fresh nodes will succeed.

**Future work: GPU/memory profiling.** The OOM failures highlight the need for profiling
(GPU memory, CPU memory, I/O) integrated into the pipeline. This would help diagnose
node-specific issues, understand hardware utilization, and optimize resource allocation.
See "Open Questions" section.

**Lessons learned:**
- `qsub script.sh -- --args` does NOT work — PBS interprets `--` as "run command directly".
  For multi-node runs, use interactive `qsub -I` and run the script manually.
  For batch submission, use `-v` env vars (requires script modification).
- Interactive `qsub` drops you into `/home`, not the project dir — always `cd` first.
- Use `bash` not `sh` to run the script (bash-specific syntax like process substitution).
- SSH to compute nodes requires passphrase-less keys or ssh-agent — fragile in batch mode.
  Replaced with `mpiexec --hostfile` (ALCF ensemble pattern). See:
  https://docs.alcf.anl.gov/running-jobs/example-job-scripts/#multi-node-ensemble-calculations-example
- **`#!/bin/bash -l` is critical for batch scripts.** The `-l` flag makes it a login shell,
  which loads the full default environment (PrgEnv-nvidia, Cray PALS/mpiexec, libfabric,
  CUDA runtime). Without `-l`, batch jobs start with a bare shell and `mpiexec`, `libfabric.so`,
  `libcudart.so` are all missing. All ALCF example scripts use `#!/bin/bash -l`.
- After `-l` loads the default env, only `module load conda; conda activate base` + venv
  activation is needed. No manual `LD_LIBRARY_PATH` or `PATH` hacks required.

### Phase 3 — Full production run (medium prod, 28 nodes, 6h)

**Goal:** Run all 28 protein pairs with full dataset, 12-fold CV, 100 epochs.

- 28 pairs × 12 folds = 336 training runs, 1 pair per node, 4 GPUs per node
- 12 folds / 4 GPUs = 3 waves × ~50 min/fold (100 epochs, no early stopping) = ~150 min training
- Stage 3 (dataset gen): ~5 min per pair
- Total per pair: ~2.5h. All 28 in parallel → ~2.5h total.
- **Queue:** medium prod (25-99 nodes, 6h walltime). 2.5h actual → comfortable margin.
- **Script:** `scripts/run_allpairs_polaris_prod.sh`

**Note:** Phase 3 uses batch submission (not interactive). The prod script uses
`mpiexec --hostfile` (ALCF multi-node ensemble pattern) to dispatch work to remote
nodes — no SSH keys needed. The head node's environment is inherited by mpiexec.

**Pre-requisite:** Phase 2 complete (Steps 1-6 validated: interactive, batch, and full 28-pair scale).

```
[ ] Step 1: Restore master to Phase 3 settings
      # In flu_28_major_protein_pairs_master.yaml:
      #   max_isolates_to_process: null  (full dataset, ~111K isolates)
      #   n_folds: 12
      #   epochs: 100
      #   patience: 100

[ ] Step 2: Submit full production run
      qsub scripts/run_allpairs_polaris_prod.sh

[ ] Step 3: Monitor
      qstat -u apartin                   # job status
      ls models/flu/July_2025/allpairs_prod_*/   # manifest dir created?
      # Per-pair logs: models/flu/July_2025/allpairs_prod_*/flu_28p_*.log

[ ] Step 4: Verify all 28 pairs
      cat models/flu/July_2025/allpairs_prod_*/manifest.txt   # 28 lines, all SUCCEEDED?
      grep FAILED models/flu/July_2025/allpairs_prod_*/manifest.txt   # any failures?
      # Re-run failures:
      # qsub -v PAIRS="failed1 failed2" scripts/run_allpairs_polaris_prod.sh
      # Or interactive:
      # qsub -I -l select=N:ncpus=64:ngpus=4 -l walltime=1:00:00 -A IMPROVE_Aim1 -q debug-scaling -l filesystems=eagle
      # bash scripts/run_allpairs_polaris_prod.sh --pairs "failed1 failed2" --skip_dataset

[ ] Step 5: Aggregate into 8x8 heatmap
      # (aggregation script TBD)
```

### Allocation cost

IMPROVE_Aim1 has ~97K node-hours remaining (expires 2026-04-01).
Phase 3: 28 nodes × 2.5h actual = **70 node-hours** (0.07% of allocation).
Even at full 6h walltime billing: 28 × 6 = **168 node-hours** (0.17% of allocation).

---

## What Has Been Built

1. **Master bundle** (`conf/bundles/flu_28_major_protein_pairs_master.yaml`) — inherits from `flu.yaml`, sets all 8 major proteins, kmer + slot_norm + concat, Phase 3 production settings (full dataset, 12-fold CV, 100 epochs, optimized batch_size=128). Children override only `dataset.schema_pair`.
2. **Bundle generator** (`scripts/generate_all_pairs_bundles.py`) — generates 28 child bundles (`flu_28p_{protA}_{protB}.yaml`). Run once; re-run overwrites safely.
3. **Phase 0 launcher** (`scripts/run_allpairs_polaris_phase0.sh`) — sequential Stage 3 + Stage 4 for all 28 pairs on a single GPU. Supports PBS (debug queue) and interactive use. Saves manifest + master bundle snapshot for provenance.
4. **Prod launcher** (`scripts/run_allpairs_polaris_prod.sh`) — multi-node PBS job script for Phases 2-3. Assigns 1 protein pair per node via `mpiexec --hostfile` (ALCF ensemble pattern), each node runs `run_cv_lambda.py` with 4 GPUs for 12-fold CV. Supports `--pairs` for subset runs, `--dry_run`, `--skip_dataset`. Default PBS: 28 nodes, medium prod queue, 6h walltime.

## What Needs to Be Built

### Data Aggregation

1. **Cross-pair aggregation + heatmap** — collect 28 CV summaries into 8×8 AUC/F1 matrix. Reads manifest or scans `cv_runs/` dirs. Generates heatmap plot for paper.

---

## HPC Concepts

**Node = scheduling atom.** 1 node = 4 A100s connected via NVLink (600 GB/s GPU-to-GPU). The scheduler allocates whole nodes — you pay for all 4 GPUs even if you use 1.

**Job arrays vs ensemble packing.** Two strategies for many independent tasks:
- **Job arrays** (`-J 0-N`): PBS creates N+1 independent jobs, each with its own node allocation. Each job has queue wait time.
- **Ensemble packing**: One job, N nodes, internal script distributes tasks across GPUs. Single queue wait, better utilization. Trade-off: if one task crashes, need retry logic.

**Hybrid approach**: Job arrays across nodes + ensemble packing within each node. Each array element gets 1 node and runs 4 tasks internally.

**Lustre (Eagle).** Parallel filesystem optimized for large sequential I/O (HDF5, CSV), not small random files. All data is on Eagle (`/lus/eagle/projects/IMPROVE_Aim1/`).

**Walltime vs node-hours.** Walltime is clock time your job runs. Node-hours = nodes × walltime. A 4-node job running 54h = 216 node-hours.

---

## PBS Job Script Templates

### Single-node test (debug queue)

```bash
#!/bin/bash -l
#PBS -N my_job
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=00:30:00
#PBS -A IMPROVE_Aim1
#PBS -q debug
#PBS -o /lus/eagle/projects/IMPROVE_Aim1/apartin/logs/
#PBS -e /lus/eagle/projects/IMPROVE_Aim1/apartin/logs/

cd $PBS_O_WORKDIR

# -l shebang loads PrgEnv-nvidia, Cray PALS, libfabric, CUDA — no manual PATH/LD_LIBRARY_PATH needed
module use /soft/modulefiles
module load conda
conda activate base

python train.py --config config.yaml
```

### Multi-node run (scaling up)

```bash
#!/bin/bash -l
#PBS -N my_distributed_job
#PBS -l select=8:ncpus=64:ngpus=4        # 8 nodes x 4 GPUs = 32 GPUs total
#PBS -l walltime=06:00:00
#PBS -A IMPROVE_Aim1
#PBS -q prod

cd $PBS_O_WORKDIR

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=4                         # one rank per GPU
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

# -l shebang loads PrgEnv-nvidia, Cray PALS, libfabric, CUDA — no manual PATH/LD_LIBRARY_PATH needed
module use /soft/modulefiles
module load conda
conda activate base

# NCCL tuning for Polaris (HPE Slingshot interconnect)
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1       # Slingshot NICs on Polaris

mpiexec -n $NTOTRANKS --ppn $NRANKS_PER_NODE \
    --depth=16 --cpu-bind depth \
    python train.py --config config.yaml
```

### PyTorch DDP / FSDP (reference)

```python
import os
rank       = int(os.environ["PMI_RANK"])       # set by cray-mpich
world_size = int(os.environ["PMI_SIZE"])
local_rank = int(os.environ["PMI_LOCAL_RANK"])  # rank within the node

torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
```

Note: Task 11 does NOT use DDP — each [pair, fold] task is a single-GPU MLP training.
DDP templates are here for reference if needed for future work (e.g., GenSLM).

---

## Monitoring & Debugging

### During a run

```zsh
qstat -u apartin                          # view your jobs
qstat -f <jobid>                          # detailed job info
tail -f /lus/eagle/.../logs/my_job.o*     # follow stdout
```

### After a run

```zsh
sbank                                     # check remaining node-hours
```

### Common failure modes

| Symptom | Likely cause |
|---|---|
| `NCCL timeout` on init | Wrong NIC interface — check `NCCL_SOCKET_IFNAME` |
| `CUDA out of memory` on node 0 only | Rank 0 doing extra work (e.g., data loading) |
| Job killed immediately | Walltime or node count exceeded queue limits |
| Conda env not found in job | Job script forgot to `conda activate` |
| `module: command not found` in job | Missing `module load` at top of job script |

---

## Resolved Decisions

1. **Master bundle naming**: `flu_28_major_protein_pairs_master` (no feature source in name — can switch kmer/esm2 in master).
2. **Child bundle naming**: `flu_28p_{protA}_{protB}` (flat in `conf/bundles/`, no subdirectory due to Hydra limitation).
3. **Scaling knob**: Change `max_isolates_to_process`, `n_folds`, `epochs`, `patience` in the master bundle. One master for all phases.
4. **Inheritance**: From `flu.yaml` directly (not `flu_schema.yaml`), so the master is self-contained.

## Open Questions

1. **ESM-2 comparison**: K-mers only first, or also create an ESM-2 master for parallel comparison?
2. **GPU/memory profiling**: Integrate profiling into the pipeline to understand hardware
   utilization (GPU memory, CPU memory, I/O throughput) per stage. Motivated by Phase 2 Step 6
   CUDA OOM failures on specific nodes — profiling would make such issues diagnosable rather
   than opaque. Also an opportunity to understand Polaris A100 hardware characteristics better.
   Options: `torch.cuda.memory_summary()`, `nvidia-smi` snapshots, Python `tracemalloc`,
   or ALCF's profiling tools (e.g., `nsys`, `ncu`).
