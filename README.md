# Viral-SegMatch

A machine learning pipeline for analyzing segmented RNA viruses to determine if two viral segments belong to the same isolate. Uses ESM-2 protein embeddings and binary classification to solve a key challenge in viral genomics.

## 🎯 Key Features

- **Primary focus**: Influenza A (Bunyavirales support exists but is not actively maintained)
- **ESM-2 embeddings** with pairwise interaction features (`unit_diff`, `concat`)
- **K-mer genome features** (experimental; matches or exceeds ESM-2 on mixed-subtype data)
- **LayerNorm (`slot_norm`)** enables learning on homogeneous subtype subsets (e.g., H3N2-only)
- **Cross-validation support** (`n_folds`/`fold_id` in dataset config)
- **Decoupled Stage 3/4**: create a dataset once, train with different bundles against it
- **Hydra configuration**: one bundle per reproducible experiment

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

**Flu A — Gen3 bundle (slot_norm + unit_diff, current best ESM-2):**
```bash
# Stage 2: Compute embeddings (run once per virus/data_version)
./scripts/stage2_esm2.sh flu --cuda_name cuda:7

# Stage 3: Create dataset (run once; reusable across training bundles)
./scripts/stage3_dataset.sh flu_schema_raw_slot_norm_unit_diff

# Find the dataset directory
ls -lt data/datasets/flu/July_2025/runs/ | head -3

# Stage 4: Train model (pass dataset_dir explicitly — decoupled from Stage 3)
./scripts/stage4_train.sh flu_schema_raw_slot_norm_unit_diff --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_schema_raw_slot_norm_unit_diff_YYYYMMDD_HHMMSS
```

**Re-train with a different bundle against the same dataset:**
```bash
# Same dataset_dir, different training bundle (e.g., concat interaction)
./scripts/stage4_train.sh flu_schema_raw_slot_norm_concat --cuda_name cuda:7 \
    --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_schema_raw_slot_norm_unit_diff_YYYYMMDD_HHMMSS
```

*For detailed setup and usage, see [Quick Start Guide](documentation/quick-start.md)*

## 📊 Performance Results

### Gen3 results (mixed-subtype HA-NA, 5K isolates)

| Feature Source | Interaction | Accuracy | F1 | AUC | Notes |
|---|---|---|---|---|---|
| ESM-2 | unit_diff | 92.5% | 0.917 | 0.966 | Gen3 baseline |
| ESM-2 | concat | 93.7% | 0.929 | 0.975 | |
| K-mer (k=6) | unit_diff | 96.2% | 0.957 | 0.982 | Current best (experimental) |
| K-mer (k=6) | concat | 95.6% | 0.951 | 0.982 | |

### Key findings

- **`unit_diff` > `concat` on homogeneous data**: On H3N2-only subsets, `concat` collapses to AUC 0.50 while `unit_diff` achieves AUC 0.96. The direction of the embedding difference carries genuine biological signal; magnitude is a shortcut that fails when subtypes are held constant.
- **LayerNorm (`slot_norm`) is critical**: Without per-slot normalization, raw HA/NA embeddings live in slightly different subspaces and `unit_diff` picks up slot offset rather than biological signal.
- **K-mer (k=6, 4096-dim) matches or exceeds ESM-2** on mixed-subtype HA-NA data. Both interactions work with k-mers. This result is experimental and needs cross-validation confirmation.
- **Conservation hypothesis** (Gen1): Variable segments (HA-NA) achieve 91.6% F1; conserved segments (PB2-PB1-PA) are limited to 75.3% F1 by biological constraints.

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
│   │   └── README.md       # Bundle conventions and inheritance
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
│           ├── dataset_flu_schema_raw_slot_norm_unit_diff_YYYYMMDD_HHMMSS/
│           └── dataset_flu_pb2_pb1_pa_5ks_YYYYMMDD_HHMMSS/

models/
├── bunya/April_2025/
│   └── runs/                           # Experiment-specific models
│       └── training_bunya_YYYYMMDD_HHMMSS/
└── flu/July_2025/
    └── runs/
        ├── training_flu_schema_raw_slot_norm_unit_diff_YYYYMMDD_HHMMSS/
        └── training_flu_pb2_pb1_pa_5ks_YYYYMMDD_HHMMSS/
```

**Key Principle**: Preprocessing and embeddings are **shared** per virus/data_version. Datasets and models are **experiment-specific** in `runs/` subdirectories.

## ⚙️ Configuration

The pipeline uses Hydra for configuration management with a bundle system:

### Available Configurations

| Bundle | Description |
|--------|-------------|
| `flu_schema_raw_slot_norm_unit_diff` | Current best ESM-2 (slot_norm + unit_diff) |
| `flu_schema_raw_slot_norm_concat` | ESM-2 + concat interaction comparison |
| `flu_schema_raw_kmer_k6_slot_norm_unit_diff` | K-mer baseline (experimental) |
| `flu_schema_raw_slot_norm_unit_diff_h3n2` | H3N2-only filtered example |
| `flu_schema_raw_slot_norm_unit_diff_cv5` | 5-fold cross-validation |

See [`conf/bundles/README.md`](conf/bundles/README.md) for the full list and bundle inheritance conventions.

### Key Parameters

- `max_isolates_to_process`: Sampling control (null = full dataset)
- `neg_to_pos_ratio`: Class balance (virus-specific, default: 1.0 for Flu, 3.0 for Bunya)
- `selected_functions`: Protein selection
- `allow_same_func_negatives`: Control same-function negative pairs

*For detailed configuration guide, see [Configuration Guide](docs/CONFIGURATION_GUIDE.md)*

## 🔬 Pipeline Stages

1. **Preprocessing** — GTO file processing, protein extraction, segment assignment
   - Output: `data/processed/{virus}/{data_version}/protein_final.csv` (shared)

2. **Embeddings (ESM-2)** — Protein embedding computation (master cache)
   - Output: `data/embeddings/{virus}/{data_version}/master_esm2_embeddings.h5` (shared)
   - Script: `./scripts/stage2_esm2.sh {virus} --cuda_name cuda:X`

2b. **Features (k-mer)** — DNA k-mer feature extraction (experimental)
   - Output: `data/embeddings/{virus}/{data_version}/kmer_k{K}_features.npz` (shared)
   - Script: `python src/embeddings/compute_kmer_features.py`

3. **Dataset Creation** — Segment pair generation with co-occurrence blocking
   - Output: `data/datasets/{virus}/{data_version}/runs/dataset_{bundle}_{timestamp}/` (experiment-specific)
   - Script: `./scripts/stage3_dataset.sh {bundle}`

4. **Training** — Binary classifier with frozen embeddings; supports ESM-2 or k-mer `feature_source`
   - Output: `models/{virus}/{data_version}/runs/training_{bundle}_{timestamp}/` (experiment-specific)
   - Script: `./scripts/stage4_train.sh {bundle} --cuda_name cuda:X --dataset_dir {path}`
   - Provenance tracked in `training_info.json` saved to the training output directory

*For detailed pipeline overview, see [Pipeline Overview](documentation/pipeline-overview.md)*

## 🧬 Supported Viruses

- **Influenza A** *(primary, actively maintained)*: 8-segment viruses; focus on PB1, PB2, PA, HA, NA proteins
- **Bunyavirales** *(not actively maintained)*: 3-segment viruses (S, M, L segments); pipeline exists but may lag behind current Flu A conventions — update when needed

## 📈 Analysis Tools

```bash
# Comprehensive results analysis (automatically finds latest training run)
python src/analysis/analyze_stage4_train.py --config_bundle flu_schema_raw_slot_norm_unit_diff

# Or specify model directory explicitly
python src/analysis/analyze_stage4_train.py --config_bundle flu_schema_raw_slot_norm_unit_diff \
    --model_dir models/flu/July_2025/runs/training_flu_schema_raw_slot_norm_unit_diff_YYYYMMDD_HHMMSS

# Presentation-ready plots
python src/analysis/create_presentation_plots.py --config_bundle flu_schema_raw_slot_norm_unit_diff
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

### Conservation Hypothesis (Gen1) ✅ Confirmed

Protein conservation directly impacts model performance:
- **Variable segments (HA-NA)**: 91.6% F1, 0.953 AUC — sufficient signal for excellent discrimination
- **Conserved segments (PB2-PB1-PA)**: 75.3% F1, 0.750 AUC — limited by biological constraints

### Interaction and Normalization (Gen3)

- **`unit_diff` > `concat` on homogeneous data**: `concat` collapses to chance (AUC 0.50) on H3N2-only subsets; `unit_diff` succeeds (AUC 0.96). Embedding difference direction carries genuine biological signal.
- **LayerNorm (`slot_norm`) is critical** for homogeneous subsets: raw HA/NA embeddings occupy slightly different subspaces, and `unit_diff` picks up the slot offset without normalization.
- **Delayed learning on H3N2 + unit_diff**: characteristic plateau then breakthrough (~epochs 10–32, seed-dependent). Set `patience` to 40+ for H3N2-only runs.

### K-mer Features (Experimental)

- **K-mer (k=6, 4096-dim) matches or exceeds ESM-2** on mixed-subtype HA-NA (AUC 0.982 vs 0.966–0.975). Both interactions work. Needs cross-validation confirmation before publication.

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
