# Installation Guide

## Prerequisites

- conda (Miniconda or Anaconda)
- A CUDA GPU only for the ESM-2 embeddings (Stage 2) and the MLP. The per-site codon features
  and LightGBM used for the paper run on CPU.

## Environment setup

### Linux

`environment.yml` is the exact `segmatch` environment, with Linux build strings pinned:

```bash
conda env create -f environment.yml
conda activate segmatch
```

Keep bioconda tools (mmseqs2, DataSAIL, BLAST, ...) out of this environment. They pull a
different `libhdf5` build and break `h5py`; install each in its own environment.

### macOS and other platforms

`environment.yml` cannot be used as is, because its build strings and its CUDA packages
(`torch==2.6.0+cu118`, `nvidia-*`) exist only on Linux. Create a smaller environment with the
same versions of the packages the pipeline imports. This recipe has not been tested on macOS.

```bash
conda create -n segmatch -c conda-forge python=3.9 numpy=1.26.4 pandas=2.2.3 \
    scikit-learn=1.6.1 scipy=1.13.1 pyarrow=19.0.1 lightgbm=4.6.0 hydra-core=1.3.2 \
    omegaconf=2.3.0 matplotlib=3.9.2 seaborn=0.13.2 joblib=1.4.2 psutil=7.0.0 \
    pyyaml=6.0.2 tqdm=4.67.1 networkx=3.2.1 h5py=3.12.1 pytest=8.4.2
conda activate segmatch
pip install torch==2.6.0      # CPU build; train_pair_baselines imports torch
```

### Verify the environment

```bash
python -c "from src.utils.config_hydra import get_virus_config_hydra; print('Config system OK')"
python -m pytest -q
```

## Project structure

```
viral-segmatch/
├── src/                    # Python source code
├── scripts/                # Shell scripts for the 4-stage pipeline
├── conf/                   # Hydra configuration (bundles, virus, dataset, training, baselines)
├── data/                   # Created by the pipeline (not in git; symlink raw GTOs in)
├── models/                 # Created by Stage 4 training
├── results/                # Created by post-hoc analysis
└── docs/                   # Setup, methods, plans and results documentation
```

For the source-tree breakdown, see [`architecture.md`](architecture.md).

## Data setup

### 1. Create the data directories

```bash
mkdir -p data/{raw,processed,embeddings,datasets}
mkdir -p models results logs
```

### 2. Drop in the raw GTO files

Stage 1 expects BV-BRC Genome Typed Object (GTO) JSON files under
`data/raw/<run_dir>/`. For the production Flu-A July 2025 dataset this
is `data/raw/Full_Flu_Annos/July_2025/*.gto`. See
[`methods/gto_format_reference.md`](methods/gto_format_reference.md)
for the GTO schema.

To start from processed data instead of raw GTOs, copy the untracked inputs listed under
"Moving to another machine" in [`../.claude/memory.md`](../.claude/memory.md).

### 3. ESM-2 models download automatically

The ESM-2 model (`esm2_t33_650M_UR50D`, ~2.5 GB) downloads on first use
of Stage 2.

## Per-machine git setup (one-time)

```bash
git config pull.rebase true   # avoid "need to reconcile divergent branches" on git pull
```

## Common installation issues

- **CUDA not available** — check `python -c "import torch; print(torch.cuda.is_available())"`. If False, verify the GPU drivers. Only Stage 2 and the MLP need CUDA.
- **ESM-2 download fails** — confirm internet connectivity and disk space; the model lives at HuggingFace (`facebook/esm2_t33_650M_UR50D`).
- **Permission errors on data/** — ensure your user has write access to the data directory.
