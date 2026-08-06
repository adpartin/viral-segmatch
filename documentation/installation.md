# Installation Guide

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for Stage 2 ESM-2 embeddings)
- ~50 GB free disk space (the full-Flu-A data version produces a ~1.78 GB
  k-mer NPZ and a similarly-sized ESM-2 HDF5 cache, plus per-experiment
  dataset and model outputs)

## Environment setup

### 1. Create a conda environment

```bash
conda create -n cepi python=3.9
conda activate cepi
```

### 2. Install dependencies

```bash
# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers fair-esm

# Data processing
pip install pandas numpy scikit-learn h5py
pip install matplotlib seaborn

# Configuration
pip install hydra-core omegaconf

# Sklearn baselines (LightGBM, k-NN)
pip install lightgbm

# Optional: Jupyter
pip install jupyter ipykernel
```

### 3. Verify installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import fair_esm; print('ESM-2 available')"
python -c "from src.utils.config_hydra import get_virus_config_hydra; print('Config system OK')"
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
├── docs/                   # Methods / technical / plans documentation
└── documentation/          # User guides (this directory)
```

For the source-tree breakdown, see
[`development/code-structure.md`](development/code-structure.md) and
[`../CLAUDE.md`](../CLAUDE.md).

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
[`../docs/methods/gto_format_reference.md`](../docs/methods/gto_format_reference.md)
for the GTO schema.

### 3. ESM-2 models download automatically

The ESM-2 model (`esm2_t33_650M_UR50D`, ~2.5 GB) downloads on first use
of Stage 2.

## Per-machine git setup (one-time)

```bash
git config pull.rebase true   # avoid "need to reconcile divergent branches" on git pull
```

## Common installation issues

- **CUDA not available** — check `python -c "import torch; print(torch.cuda.is_available())"`. If False, verify GPU drivers and that you installed the CUDA-bundled PyTorch wheel.
- **ESM-2 download fails** — confirm internet connectivity and disk space; the model lives at HuggingFace (`facebook/esm2_t33_650M_UR50D`).
- **Permission errors on data/** — ensure your user has write access to the data directory.

More issues + fixes: [`troubleshooting.md`](troubleshooting.md).

## Next step

[`quick-start.md`](quick-start.md) — run your first experiment.
