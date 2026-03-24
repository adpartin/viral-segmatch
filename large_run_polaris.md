# Large-Scale Run on Polaris: Approach & Checklist

## Context

- **System:** Polaris (ALCF, HPE/Cray), A100 GPUs (4 per node)
- **Active allocation:** IMPROVE_Aim1 — 97k node-hours remaining, expires 2026-04-01
- **Secondary allocation:** bvbrc — 1k node-hours, expires 2026-05-01
- **Project storage:** `/lus/eagle/projects/IMPROVE_Aim1` (fast Lustre, use for data and checkpoints)
- **Home dir:** `/home/apartin` (slower, not for large I/O)

---

## Phase 1: Validate Environment on Compute Nodes

Before any large run, validate that the shell environment works correctly on a compute node.
The dotfiles handle login nodes well; compute nodes need explicit verification.

### Start a debug session

```zsh
qsub -I -l select=1 -l walltime=0:30:00 -A IMPROVE_Aim1 -q debug
```

### Check dotfiles / shell

```zsh
echo $PBS_ENVIRONMENT          # should print PBS_INTERACTIVE
echo $http_proxy               # should be set (proxy from polaris.zsh)
module list                    # confirm cuda/12.9 was loaded by polaris.zsh
conda activate <your-env>      # confirm lazy-load works on compute node
python -c "import torch; print(torch.cuda.is_available())"  # confirm GPU visible
nvidia-smi                     # confirm all 4 A100s are visible
```

### If anything fails here, fix it before scaling up.

---

## Phase 2: Environment for Distributed Training

### Conda environment

- Use the `covid` env or create a dedicated env for the project.
- **Install into Lustre, not home:** Conda envs on `/home` are on a slower filesystem
  and can cause issues under heavy parallel I/O from many nodes.

  ```zsh
  conda create --prefix /lus/eagle/projects/IMPROVE_Aim1/apartin/envs/my_env python=3.11
  conda activate /lus/eagle/projects/IMPROVE_Aim1/apartin/envs/my_env
  ```

- Install PyTorch with the right CUDA version (matches `cuda/12.9` on Polaris):
  ```zsh
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
  (check ALCF docs for their recommended PyTorch wheel for Polaris — they sometimes
  provide pre-built wheels optimized for the A100s)

### Modules in job scripts

The `polaris.zsh` in dotfiles loads `cuda/12.9` automatically on compute nodes.
**However**, PBS job scripts run in a non-interactive shell — dotfiles are NOT sourced.
Every job script must load modules explicitly:

```bash
module load cuda/12.9
module load cray-mpich
```

This is a critical separation: **dotfiles handle interactive shells; job scripts handle batch.**

---

## Phase 3: PBS Job Script Structure

### Single-node test (debug queue)

```bash
#!/bin/bash
#PBS -N my_job
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=00:30:00
#PBS -A IMPROVE_Aim1
#PBS -q debug
#PBS -o /lus/eagle/projects/IMPROVE_Aim1/apartin/logs/
#PBS -e /lus/eagle/projects/IMPROVE_Aim1/apartin/logs/

cd $PBS_O_WORKDIR

module load cuda/12.9
module load cray-mpich

conda activate /lus/eagle/projects/IMPROVE_Aim1/apartin/envs/my_env

python train.py --config config.yaml
```

### Multi-node run (scaling up)

```bash
#!/bin/bash
#PBS -N my_distributed_job
#PBS -l select=8:ncpus=64:ngpus=4        # 8 nodes x 4 GPUs = 32 GPUs total
#PBS -l walltime=06:00:00
#PBS -A IMPROVE_Aim1
#PBS -q prod

cd $PBS_O_WORKDIR

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=4                         # one rank per GPU
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

module load cuda/12.9
module load cray-mpich

conda activate /lus/eagle/projects/IMPROVE_Aim1/apartin/envs/my_env

# NCCL tuning for Polaris (HPE Slingshot interconnect)
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1       # Slingshot NICs on Polaris

mpiexec -n $NTOTRANKS --ppn $NRANKS_PER_NODE \
    --depth=16 --cpu-bind depth \
    python train.py --config config.yaml
```

### Queues

| Queue   | Max nodes | Max walltime | Use for                   |
|---------|-----------|--------------|---------------------------|
| `debug` | 2         | 1h           | Testing, env validation   |
| `prod`  | 496       | 24h          | Production runs           |
| `preemptable` | 496 | 72h        | Long runs, can be killed  |

---

## Phase 4: Distributed Training Code Checklist

### PyTorch DDP / FSDP

- Use `torchrun` or `mpiexec` as the launcher (not `python` directly).
- Set the right env vars for rank/world size — with `mpiexec` + cray-mpich:

  ```python
  import os
  rank       = int(os.environ["PMI_RANK"])       # set by cray-mpich
  world_size = int(os.environ["PMI_SIZE"])
  local_rank = int(os.environ["PMI_LOCAL_RANK"])  # rank within the node
  ```

- Initialize process group:
  ```python
  torch.distributed.init_process_group(backend="nccl")
  torch.cuda.set_device(local_rank)
  ```

### Data loading

- **All data must be on Lustre** (`/lus/eagle/...`), not on home.
- Lustre performs well for large sequential reads but poorly for many small files.
  If your dataset is many small files, consider packing them into HDF5 or WebDataset.
- Use `num_workers > 0` in DataLoader but don't go too high (each node has 64 CPUs,
  shared across 4 ranks, so ~8-16 workers per rank is a reasonable starting point).

### Checkpointing

- Save checkpoints to Lustre, not home.
- On multi-node runs, only rank 0 should write checkpoints to avoid conflicts:
  ```python
  if rank == 0:
      torch.save(model.state_dict(), checkpoint_path)
  ```

---

## Phase 5: Monitoring & Debugging

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

## Connection to Dotfiles

| What dotfiles handle | What job scripts must handle separately |
|---|---|
| Interactive shell env (zsh, prompt, aliases) | Module loads |
| Conda lazy-load for interactive use | `conda activate` |
| Proxy for interactive pip/git | `http_proxy` export if job needs internet |
| `cuda/12.9` on compute nodes (interactive) | `module load cuda/12.9` |
| `cdimprove` alias, `$IMPROVE_DIR` | `cd $PBS_O_WORKDIR` |

**Key rule:** Never rely on dotfiles inside a PBS job script. The job runs in a
non-interactive shell where `.zshrc` is not sourced. Everything the job needs
must be explicit in the job script.

---

## Recommended First Sequence

1. `qsub -I ... -q debug` → validate env on compute node
2. Run single-node training with 4 GPUs — confirm DDP init, loss goes down
3. Scale to 2 nodes (still in debug queue) — confirm NCCL communication
4. Submit a modest production run (4–8 nodes) with checkpointing
5. Scale to full run
