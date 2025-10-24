# Viral-SegMatch

A machine learning pipeline for analyzing segmented RNA viruses to determine if two viral segments belong to the same isolate. Uses ESM-2 protein embeddings and binary classification to solve a key challenge in viral genomics.

## Key Features

- **Multi-virus support**: Bunyavirales and Influenza A
- **ESM-2 embeddings**: State-of-the-art protein language model
- **Hydra configuration**: Flexible, reproducible experiments
- **Performance optimized**: F1 scores up to 0.87+ with virus-specific tuning

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/adpartin/viral-segmatch.git
cd viral-segmatch
conda env create -f environment.yml
conda activate cepi
```

### Run Complete Pipeline

**Bunya (3 segments):**
```bash
./scripts/preprocess_bunya_protein.sh
./scripts/esm2_bunya.sh
./scripts/dataset_bunya.sh
./scripts/classifier_bunya.sh
```

**Flu A (3 proteins, 1k isolates):**
```bash
./scripts/preprocess_flu_protein.sh
./scripts/esm2_flu_3p_1ks.sh
./scripts/dataset_flu_3p_1ks.sh
./scripts/classifier_flu_3p_1ks.sh
```

## 📊 Performance Results

| Virus | Configuration | F1 Score | Key Finding |
|-------|---------------|----------|-------------|
| **Bunya** | Default settings | **0.8829** | `neg_to_pos_ratio: 3.0` tested |
| **Flu A** | Optimized | **0.8708** | `neg_to_pos_ratio: 1.0` tested |

**🔑 Critical Discovery:** Different viruses may require different class balance ratios for optimal performance.

## 📁 Output Structure

```
data/
├── processed/          # Preprocessed protein data
├── embeddings/         # ESM-2 embeddings
└── datasets/           # Segment pair datasets

models/                 # Trained classifiers
└── {virus}/{version}/  # Model checkpoints and results
```

## ⚙️ Configuration

The pipeline uses Hydra for configuration management:

- **`conf/bundles/bunya.yaml`** - Bunyavirales configuration
- **`conf/bundles/flu_a_3p_1ks.yaml`** - Flu A, 3 proteins, 1k isolates
- **`conf/bundles/flu_a.yaml`** - Flu A full dataset

Key parameters:
- `max_isolates_to_process`: Sampling control
- `neg_to_pos_ratio`: Class balance (virus-specific)
- `selected_functions`: Protein selection

## 📚 Documentation

- **[Quick Start Guide](documentation/quick-start.md)** - Complete setup and usage
- **[Pipeline Overview](documentation/pipeline-overview.md)** - Understanding the 4-stage pipeline
- **[Configuration Guide](documentation/configuration.md)** - Customizing experiments

## 🔬 Pipeline Stages

1. **Preprocessing** - GTO file processing, protein extraction, segment assignment
2. **Embeddings** - ESM-2 protein embedding computation
3. **Dataset Creation** - Segment pair generation with balanced sampling
4. **Training** - Binary classifier training with frozen ESM-2

## 🧬 Supported Viruses

- **Bunyavirales**: 3-segment viruses (S, M, L segments)
- **Influenza A**: 8-segment viruses (focus on PB1, PB2, PA proteins)

## 📈 Analysis Tools

```bash
# Comprehensive results analysis
python src/postprocess/segment_classifier_results.py --config_bundle bunya --model_dir ./models/bunya/April_2025

# Presentation-ready plots
python src/postprocess/presentation_plots.py --config_bundle bunya --model_dir ./models/bunya/April_2025
```

## 📄 License

TODO
