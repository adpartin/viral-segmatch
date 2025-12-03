# Installation Guide

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 50GB+ free disk space for datasets and models

## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n cepi python=3.9
conda activate cepi
```

### 2. Install Dependencies
```bash
# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install fair-esm

# Data processing
pip install pandas numpy scikit-learn
pip install matplotlib seaborn

# Configuration
pip install hydra-core omegaconf

# Optional: Jupyter for analysis
pip install jupyter ipykernel
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import fair_esm; print('ESM-2 available')"
```

## Project Structure

```
viral-segmatch/
├── src/                    # Python source code
├── scripts/               # Shell scripts for experiments
├── conf/                  # Hydra configuration files
├── data/                  # Data directories (created automatically)
├── models/                # Trained models (created automatically)
├── results/               # Analysis results (created automatically)
└── documentation/         # This documentation
```

## Data Setup

### 1. Create Data Directories
```bash
mkdir -p data/{raw,processed,embeddings,datasets}
mkdir -p models results logs
```

### 2. Download ESM-2 Models (Automatic)
The ESM-2 models will be downloaded automatically on first use (~2GB).

### 3. Prepare Your Data
Place your viral protein data in `data/raw/` following the expected format.

## Verification

Run a quick test to verify everything works:

```bash
# Test configuration loading
python -c "from src.utils.config_hydra import get_virus_config_hydra; print('Config system OK')"

# Test path utilities
python -c "from src.utils.path_utils import build_training_paths; print('Path utilities OK')"
```

## Troubleshooting

### Common Issues

**CUDA not available:**
- Install CUDA toolkit
- Verify GPU drivers
- Check PyTorch CUDA installation

**ESM-2 download fails:**
- Check internet connection
- Verify disk space
- Try manual download from Hugging Face

**Permission errors:**
- Check file permissions
- Ensure write access to data directories

### Getting Help

- Check [Troubleshooting Guide](troubleshooting.md)
- Review existing [docs/](../docs/) for technical notes
- Check script logs in `logs/` directory
