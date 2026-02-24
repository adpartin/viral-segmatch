# Viral-SegMatch

A machine learning pipeline for analyzing segmented RNA viruses to determine if two viral segments belong to the same isolate. Uses ESM-2 protein embeddings and binary classification to solve a key challenge in viral genomics.

## 🎯 Key Features

- **Primary focus**: Influenza A (Bunyavirales support exists but is not actively maintained)
- **ESM-2 embeddings**: State-of-the-art protein language model with master cache system
- **Hydra configuration**: Flexible, reproducible experiments
- **Conservation hypothesis confirmed**: Variable segments achieve 91.6% F1, conserved segments 75.3% F1
- **Performance optimized**: F1 scores up to 0.92+ with segment-specific models

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/adpartin/viral-segmatch.git
cd viral-segmatch
conda create -n cepi python=3.9
conda activate cepi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers fair-esm pandas numpy scikit-learn matplotlib seaborn hydra-core omegaconf h5py
```

### Run Complete Pipeline

**Bunya (all segments):**
```bash
# Stage 2: Compute embeddings (if not already done)
./scripts/stage2_esm2.sh bunya --cuda_name cuda:7

# Stage 3: Create dataset
./scripts/stage3_dataset.sh bunya

# Find dataset directory
ls -lt data/datasets/bunya/April_2025/runs/ | head -3

# Stage 4: Train model
./scripts/stage4_train.sh bunya --cuda_name cuda:7 \
    --dataset_dir data/datasets/bunya/April_2025/runs/dataset_bunya_YYYYMMDD_HHMMSS
```

**Flu A (HA-NA, variable segments, 5K isolates):**
```bash
# Stage 2: Compute embeddings (if not already done)
./scripts/stage2_esm2.sh flu --cuda_name cuda:7

# Stage 3: Create dataset
./scripts/stage3_dataset.sh flu_ha_na_5ks

# Find dataset directory
ls -lt data/datasets/flu/July_2025/runs/ | head -3

# Stage 4: Train model
./scripts/stage4_train.sh flu_ha_na_5ks --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS
```

*For detailed setup and usage, see [Quick Start Guide](documentation/quick-start.md)*

## 📊 Performance Results

| Virus | Configuration | Accuracy | F1 Score | AUC | Key Finding |
|-------|---------------|----------|----------|-----|-------------|
| **Bunya** | All segments (full dataset) | 91.0% | **91.1%** | 0.927 | Baseline performance |
| **Flu A** | HA-NA (variable, 5K isolates) | 92.3% | **91.6%** | 0.953 | Variable segments excel |
| **Flu A** | PB2-PB1-PA (conserved, 5K isolates) | 71.9% | **75.3%** | 0.750 | Conserved segments limited |
| **Flu A** | PB2-HA-NA (mixed, 5K isolates) | 85.4% | **85.5%** | 0.920 | Mixed performance |

**🔑 Key Discovery: Conservation Hypothesis Confirmed** ✅

Performance directly correlates with protein conservation levels:
- **Variable segments (HA-NA)**: Achieve excellent performance (91.6% F1) - comparable to Bunya
- **Conserved segments (PB2-PB1-PA)**: Limited by biological constraints (75.3% F1)
- **Segment-specific models**: Recommended for optimal performance

*For detailed analysis, see [Experiment Results](docs/EXPERIMENT_RESULTS_ANALYSIS.md)*

## 📁 Project Structure

```
viral-segmatch/
├── CLAUDE.md               # Project context for Claude Code sessions
├── src/                    # Python source code
│   ├── embeddings/         # ESM-2 embedding generation
│   ├── datasets/           # Dataset creation
│   ├── models/             # Model training
│   ├── analysis/           # Results analysis
│   └── utils/              # Utilities
├── scripts/                # Shell scripts for automation
│   ├── stage2_esm2.sh      # Compute embeddings
│   ├── stage3_dataset.sh   # Create datasets
│   └── stage4_train.sh     # Train models
├── conf/                   # Hydra configuration
│   ├── bundles/            # Experiment bundles (one per named experiment)
│   ├── virus/              # Virus configurations
│   └── training/           # Training configs
├── eda/                    # Exploratory analysis scripts (not pipeline)
├── examples/               # HuggingFace reference scripts (not pipeline)
├── old_scripts/            # Superseded scripts (not maintained)
├── data/                   # Data directories
│   ├── processed/          # Preprocessed data (shared)
│   ├── embeddings/         # ESM-2 embeddings (shared, master cache)
│   └── datasets/           # Segment pair datasets (experiment-specific)
├── models/                 # Trained models (experiment-specific)
├── results/                # Analysis results
├── docs/                   # Technical documentation
└── documentation/          # User guides
```

## 📁 Output Structure

```
data/
├── processed/
│   ├── bunya/April_2025/              # Shared preprocessing
│   └── flu/July_2025/                  # Shared preprocessing
├── embeddings/
│   ├── bunya/April_2025/               # Shared embeddings (master cache)
│   │   ├── master_esm2_embeddings.h5
│   │   └── master_esm2_embeddings.parquet
│   └── flu/July_2025/                  # Shared embeddings (master cache)
├── datasets/
│   ├── bunya/April_2025/
│   │   └── runs/                       # Experiment-specific datasets
│   │       └── dataset_bunya_YYYYMMDD_HHMMSS/
│   └── flu/July_2025/
│       └── runs/
│           ├── dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS/
│           └── dataset_flu_pb2_pb1_pa_5ks_YYYYMMDD_HHMMSS/

models/
├── bunya/April_2025/
│   └── runs/                           # Experiment-specific models
│       └── training_bunya_YYYYMMDD_HHMMSS/
└── flu/July_2025/
    └── runs/
        ├── training_flu_ha_na_5ks_YYYYMMDD_HHMMSS/
        └── training_flu_pb2_pb1_pa_5ks_YYYYMMDD_HHMMSS/
```

**Key Principle**: Preprocessing and embeddings are **shared** per virus/data_version. Datasets and models are **experiment-specific** in `runs/` subdirectories.

## ⚙️ Configuration

The pipeline uses Hydra for configuration management with a bundle system:

### Available Configurations

| Bundle | Proteins | Isolates | Purpose |
|--------|----------|----------|---------|
| `bunya` | L, M, S | Full | Bunya baseline |
| `flu_ha_na_5ks` | HA, NA | 5K | Variable segments (expect BETTER) |
| `flu_pb2_pb1_pa_5ks` | PB2, PB1, PA | 5K | Conserved segments (expect WORSE) |
| `flu_pb2_ha_na_5ks` | PB2, HA, NA | 5K | Mixed segments |
| `flu_overfit_5ks` | PB2, PB1, PA | 5K | Overfitting capacity test (1% train) |

### Key Parameters

- `max_isolates_to_process`: Sampling control (null = full dataset)
- `neg_to_pos_ratio`: Class balance (virus-specific, default: 1.0 for Flu, 3.0 for Bunya)
- `selected_functions`: Protein selection
- `allow_same_func_negatives`: Control same-function negative pairs

*For detailed configuration guide, see [Configuration Guide](docs/CONFIGURATION_GUIDE.md)*

## 🔬 Pipeline Stages

1. **Preprocessing** - GTO file processing, protein extraction, segment assignment
   - Output: `data/processed/{virus}/{data_version}/protein_final.csv` (shared)
   
2. **Embeddings** - ESM-2 protein embedding computation (master cache)
   - Output: `data/embeddings/{virus}/{data_version}/master_esm2_embeddings.h5` (shared)
   - Script: `./scripts/stage2_esm2.sh {virus} --cuda_name cuda:X`
   
3. **Dataset Creation** - Segment pair generation with co-occurrence blocking
   - Output: `data/datasets/{virus}/{data_version}/runs/dataset_{bundle}_{timestamp}/` (experiment-specific)
   - Script: `./scripts/stage3_dataset.sh {bundle}`
   
4. **Training** - Binary classifier training with frozen ESM-2
   - Output: `models/{virus}/{data_version}/runs/training_{bundle}_{timestamp}/` (experiment-specific)
   - Script: `./scripts/stage4_train.sh {bundle} --cuda_name cuda:X --dataset_dir {path}`

*For detailed pipeline overview, see [Pipeline Overview](documentation/pipeline-overview.md)*

## 🧬 Supported Viruses

- **Influenza A** *(primary, actively maintained)*: 8-segment viruses; focus on PB1, PB2, PA, HA, NA proteins
- **Bunyavirales** *(not actively maintained)*: 3-segment viruses (S, M, L segments); pipeline exists but may lag behind current Flu A conventions — update when needed

## 📈 Analysis Tools

```bash
# Comprehensive results analysis (automatically finds latest training run)
python src/analysis/analyze_stage4_train.py --config_bundle flu_ha_na_5ks

# Or specify model directory explicitly
python src/analysis/analyze_stage4_train.py --config_bundle flu_ha_na_5ks \
    --model_dir models/flu/July_2025/runs/training_flu_ha_na_5ks_YYYYMMDD_HHMMSS

# Presentation-ready plots
python src/analysis/create_presentation_plots.py --config_bundle flu_ha_na_5ks
```

*For detailed analysis guide, see [Results Analysis](documentation/analysis/results-analysis.md)*

## 📚 Documentation

### User Guides (`documentation/`)
- **[Quick Start Guide](documentation/quick-start.md)** - Complete setup and usage
- **[Pipeline Overview](documentation/pipeline-overview.md)** - Understanding the 4-stage pipeline
- **[Configuration Guide](documentation/configuration.md)** - Configuration overview
- **[Troubleshooting](documentation/troubleshooting.md)** - Common issues and solutions
- **[Results Analysis](documentation/analysis/results-analysis.md)** - Understanding model performance

### Technical Documentation (`docs/`)
- **[Configuration Guide](docs/CONFIGURATION_GUIDE.md)** - Comprehensive configuration documentation
- **[Experiment Results](docs/EXPERIMENT_RESULTS_ANALYSIS.md)** - Detailed experiment analysis
- **[Project Status](docs/EXP_RESULTS_STATUS.md)** - Research status and roadmap
- **[Seed System](docs/SEED_SYSTEM.md)** - Reproducibility and seed management
- **[Experiment Tracking](docs/EXPERIMENT_TRACKING_GUIDE.md)** - Tracking experiments

## 🔬 Research Findings

### Conservation Hypothesis ✅ Confirmed

The experiments definitively confirm that protein conservation directly impacts model performance:

1. **Variable segments (HA-NA)**: Achieve 92.3% accuracy, 91.6% F1, 0.953 AUC
   - Comparable to Bunya performance (91.0% accuracy, 91.1% F1)
   - Sufficient signal for excellent discrimination

2. **Conserved segments (PB2-PB1-PA)**: Achieve 71.9% accuracy, 75.3% F1, 0.750 AUC
   - Biological limitation: High conservation limits ESM-2's ability to distinguish isolates
   - F1 > Accuracy suggests some recoverable signal despite embedding overlap

3. **Recommendation**: Deploy segment-specific models with realistic performance expectations

*For comprehensive analysis, see [Experiment Results Analysis](docs/EXPERIMENT_RESULTS_ANALYSIS.md)*

## 🛠️ Development

- **Code Structure**: See [Code Structure](documentation/development/code-structure.md)
- **Model Improvements**: See [Model Improvements](documentation/model-improvements.md)
- **Adding New Viruses**: See [Configuration Guide](docs/CONFIGURATION_GUIDE.md#creating-a-new-experiment)

## 📄 License

TODO

## 🙏 Acknowledgments

- ESM-2 protein language model by Meta AI
- Hydra configuration framework
- PyTorch deep learning framework
