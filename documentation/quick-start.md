# Quick Start Guide

Get up and running with the viral-segmatch project in minutes.

**Note**: For detailed technical documentation and research results, see [`../docs/`](../docs/).

## 🚀 Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 50GB+ free disk space

## ⚡ 5-Minute Setup

### 1. Environment Setup
```bash
# Create and activate environment
conda create -n cepi python=3.9
conda activate cepi

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers fair-esm pandas numpy scikit-learn matplotlib seaborn hydra-core omegaconf h5py
```

### 2. Verify Installation
```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import fair_esm; print('ESM-2 available')"
python -c "from src.utils.config_hydra import get_virus_config_hydra; print('Config system OK')"
```

### 3. Create Data Directories
```bash
mkdir -p data/{raw,processed,embeddings,datasets}
mkdir -p models results logs
```

## 🎯 Run Your First Experiment

### Option 1: Complete Pipeline (Recommended)

For Flu experiments, preprocessing is typically already done. Run embeddings, dataset creation, and training:

```bash
# Stage 2: Compute embeddings (if not already done)
./scripts/stage2_esm2.sh flu --cuda_name cuda:7

# Stage 3: Create dataset
./scripts/stage3_dataset.sh flu_ha_na_5ks

# Find the dataset directory
ls -lt data/datasets/flu/July_2025/runs/ | head -3

# Stage 4: Train model
./scripts/stage4_train.sh flu_ha_na_5ks --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS
```

### Reuse a Dataset with Different Training Bundles

Stage 3/4 are decoupled: create a dataset once, then train multiple times with different bundles:

```bash
# Same dataset, different training bundle (e.g., concat interaction)
./scripts/stage4_train.sh flu_schema_raw_slot_norm_concat --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS
```

### Option 2: Individual Stages

```bash
# Just embeddings
./scripts/stage2_esm2.sh flu --cuda_name cuda:7

# Just dataset creation
./scripts/stage3_dataset.sh flu_ha_na_5ks

# Just training (after finding dataset directory)
./scripts/stage4_train.sh flu_ha_na_5ks --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS
```

## 📊 Current Experiment Results

### **Key Finding: Conservation Hypothesis Confirmed** ✅

Performance directly correlates with protein conservation levels:
- **Variable segments (HA-NA)**: 92.3% accuracy, **91.6% F1**, 0.953 AUC
- **Mixed segments (PB2-HA-NA)**: 85.4% accuracy, **85.5% F1**, 0.920 AUC  
- **Conserved segments (PB2-PB1-PA)**: 71.9% accuracy, **75.3% F1**, 0.750 AUC

*For detailed analysis, see [`../docs/EXPERIMENT_RESULTS_ANALYSIS.md`](../docs/EXPERIMENT_RESULTS_ANALYSIS.md)*

### **Expected Output Structure:**
```
data/
├── processed/
│   ├── bunya/April_2025/                      # Shared preprocessing
│   └── flu/July_2025/                         # Shared preprocessing
├── embeddings/
│   ├── bunya/April_2025/                      # Shared embeddings (master cache)
│   └── flu/July_2025/                         # Shared embeddings (master cache)
├── datasets/
│   ├── bunya/April_2025/
│   │   └── runs/                              # All dataset experiments
│   │       └── dataset_bunya_YYYYMMDD_HHMMSS/
│   └── flu/July_2025/
│       └── runs/                              # All dataset experiments
│           ├── dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS/
│           └── dataset_flu_pb2_pb1_pa_5ks_YYYYMMDD_HHMMSS/

models/
├── bunya/April_2025/
│   └── runs/                                  # All training experiments
│       └── training_bunya_YYYYMMDD_HHMMSS/
└── flu/July_2025/
    └── runs/                                  # All training experiments
        ├── training_flu_ha_na_5ks_YYYYMMDD_HHMMSS/
        └── training_flu_pb2_pb1_pa_5ks_YYYYMMDD_HHMMSS/
```

**Key Features:**
- Preprocessing and embeddings are **shared** per virus/data_version
- Datasets and models are **experiment-specific** in `runs/` subdirectories
- Config bundle name is **automatically included** in directory names

## 📊 Analyze Results

### Comprehensive Analysis
```bash
# Detailed ML analysis (automatically finds latest training run)
python src/analysis/analyze_stage4_train.py --config_bundle flu_ha_na_5ks

# Or specify model directory explicitly
python src/analysis/analyze_stage4_train.py --config_bundle flu_ha_na_5ks \
    --model_dir models/flu/July_2025/runs/training_flu_ha_na_5ks_YYYYMMDD_HHMMSS
```

### Presentation Plots
```bash
# Clean, publication-ready plots
python src/analysis/create_presentation_plots.py --config_bundle flu_ha_na_5ks
```

## 🔧 Configuration

### Available Configurations
- **`bunya`**: Bunyavirus baseline (all segments)
- **`flu_ha_na_5ks`**: Flu, HA-NA only (variable segments), 5K isolates
- **`flu_pb2_pb1_pa_5ks`**: Flu, PB2-PB1-PA only (conserved segments), 5K isolates
- **`flu_pb2_ha_na_5ks`**: Flu, PB2-HA-NA (mixed), 5K isolates
- **`flu_overfit_5ks`**: Flu, overfitting capacity test (5K isolates, 1% train)

*For detailed configuration guide, see [`../docs/CONFIGURATION_GUIDE.md`](../docs/CONFIGURATION_GUIDE.md)*

### Custom Configuration
```yaml
# Create conf/bundles/my_experiment.yaml
defaults:
  - /virus: flu
  - /paths: flu
  - /embeddings: default
  - /dataset: default
  - /training: base
  - _self_

run_suffix: null  # Always null - config bundle name is used instead

master_seed: 42

virus:
  selected_functions:
    - "Hemagglutinin precursor"
    - "Neuraminidase protein"

dataset:
  use_selected_only: true
  neg_to_pos_ratio: 1.0
  allow_same_func_negatives: false
  max_isolates_to_process: 5000  # null = full dataset

training:
  batch_size: 16
  learning_rate: 0.001
  patience: 15
  early_stopping_metric: 'f1'
```

## 📁 Output Structure

```
data/
├── processed/flu/July_2025/                    # Preprocessed data (shared)
├── embeddings/flu/July_2025/                  # ESM-2 embeddings (shared, master cache)
│   ├── master_esm2_embeddings.h5
│   └── master_esm2_embeddings.parquet
├── datasets/flu/July_2025/
│   └── runs/                                  # Experiment-specific datasets
│       └── dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS/
│           ├── train_pairs.csv
│           ├── val_pairs.csv
│           └── test_pairs.csv

models/flu/July_2025/
└── runs/                                      # Experiment-specific models
    └── training_flu_ha_na_5ks_YYYYMMDD_HHMMSS/
        ├── best_model.pt
        ├── test_predicted.csv
        └── training_history.csv

results/flu/July_2025/flu_ha_na_5ks/          # Analysis results
├── training_analysis/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics.csv
│   └── ...
└── presentation_plots/
    ├── performance_summary.png
    └── biological_insights.png
```

## 🎯 Key Parameters

### Data Sampling
```yaml
max_isolates_to_process: 5000    # Number of isolates to process (null = full dataset)
master_seed: 42                   # Random seed for reproducibility
```

### Training
```yaml
training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 100
  patience: 15
  early_stopping_metric: 'f1'
```

## 🔍 Troubleshooting

### Common Issues

**1. CUDA not available**
```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Use CPU fallback
export CUDA_VISIBLE_DEVICES=""
```

**2. Missing files**
```bash
# Check if preprocessing completed
ls -la data/processed/flu/July_2025/

# Check if embeddings exist
ls -la data/embeddings/flu/July_2025/master_esm2_embeddings.h5

# Run missing stages
./scripts/stage2_esm2.sh flu --cuda_name cuda:7
```

**3. Configuration errors**
```bash
# Test configuration
python -c "from src.utils.config_hydra import get_virus_config_hydra; config = get_virus_config_hydra('flu_ha_na_5ks'); print('OK')"
```

**4. Can't find dataset directory**
```bash
# List all dataset runs
ls -lt data/datasets/flu/July_2025/runs/ | head -5

# Search for specific bundle
ls -d data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_*
```

*For more troubleshooting help, see [Troubleshooting Guide](troubleshooting.md)*

## 📚 Next Steps

### Learn More
- [Pipeline Overview](pipeline-overview.md) - Understand the 4-stage pipeline
- [Configuration Guide](../docs/CONFIGURATION_GUIDE.md) - Detailed configuration documentation
- [Results Analysis](analysis/results-analysis.md) - Interpret outputs

### Advanced Usage
- [Adding New Viruses](development/adding-viruses.md) - Extend to new species (if exists)
- [Custom Analysis](analysis/presentation-plots.md) - Create custom plots
- [Troubleshooting](troubleshooting.md) - Solve common issues

### Research & Technical Details
- [`../docs/EXP_RESULTS_STATUS.md`](../docs/EXP_RESULTS_STATUS.md) - Project status and research roadmap
- [`../docs/EXPERIMENT_RESULTS_ANALYSIS.md`](../docs/EXPERIMENT_RESULTS_ANALYSIS.md) - Detailed experiment results

## 🎉 Success!

You should now have:
- ✅ Working environment
- ✅ Complete pipeline run
- ✅ Analysis results
- ✅ Presentation plots

Check the `results/` directory for your analysis outputs!
