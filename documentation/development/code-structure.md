# Code Structure

Understanding the organization and architecture of the viral-segmatch project.

**Note**: For detailed technical documentation and research results, see [`../../docs/`](../../docs/).

## 🏗️ Project Architecture

```
viral-segmatch/
├── src/                          # Python source code
│   ├── embeddings/               # ESM-2 embedding generation
│   ├── datasets/                 # Dataset creation
│   ├── models/                   # Model training
│   ├── analysis/                 # Analysis and visualization
│   └── utils/                    # Utility functions
├── scripts/                      # Shell scripts for automation
├── conf/                         # Hydra configuration files
├── data/                         # Data directories (auto-created)
├── models/                       # Trained models (auto-created)
├── results/                      # Analysis results (auto-created)
├── logs/                         # Execution logs (auto-created)
├── docs/                         # Technical documentation and research logs
└── documentation/                # User guides and tutorials
```

## 📁 Source Code Organization

### Core Pipeline (`src/`)

#### 1. Embeddings (`src/embeddings/`)
- **`compute_esm2_embeddings.py`**: Generate ESM-2 embeddings
- **Purpose**: Convert protein sequences to ESM-2 embeddings
- **Input**: `protein_final.csv` from preprocessing
- **Output**: Master embeddings cache (`master_esm2_embeddings.h5` + `.parquet`)

#### 2. Datasets (`src/datasets/`)
- **`dataset_segment_pairs.py`**: Create segment pair datasets
- **Purpose**: Generate training/validation/test datasets with co-occurrence blocking
- **Input**: ESM-2 embeddings + protein metadata
- **Output**: Segment pair datasets (CSV files in `runs/` subdirectories)

#### 3. Models (`src/models/`)
- **`train_pair_classifier.py`**: Train ESM-2 classifier
- **Purpose**: Train frozen ESM-2 classifier on segment pairs
- **Input**: Segment pair datasets
- **Output**: Trained model + predictions + analysis

#### 4. Analysis (`src/analysis/`)
- **`analyze_stage4_train.py`**: Comprehensive analysis
- **`create_presentation_plots.py`**: Publication-ready plots
- **Purpose**: Analyze model performance and generate visualizations

### Utilities (`src/utils/`)

#### Configuration Management
- **`config_hydra.py`**: Hydra configuration loading
- **`path_utils.py`**: Path generation and management
- **`seed_utils.py`**: Hierarchical seed management
- **`torch_utils.py`**: PyTorch utilities (optimizers, schedulers)

#### Data Processing
- **`data_utils.py`**: Data loading and processing utilities
- **`protein_utils.py`**: Protein-related utilities
- **`learning_verification_utils.py`**: Learning verification functions

#### Experiment Management
- **`experiment_registry.py`**: Experiment tracking and registration

## 🔧 Configuration System

### Hydra Configuration (`conf/`)

#### Bundle System
```
conf/bundles/
├── bunya.yaml                 # Bunyavirus baseline
├── flu.yaml                   # Flu base config
├── flu_ha_na_5ks.yaml         # Flu: HA-NA (variable segments)
├── flu_pb2_pb1_pa_5ks.yaml    # Flu: PB2-PB1-PA (conserved segments)
└── flu_pb2_ha_na_5ks.yaml     # Flu: PB2-HA-NA (mixed)
```

#### Default Configurations
```
conf/
├── virus/                     # Virus biological facts
│   ├── flu.yaml
│   └── bunya.yaml
├── paths/                     # Path configurations
│   ├── flu.yaml
│   └── bunya.yaml
├── embeddings/
│   └── default.yaml           # ESM-2 embedding defaults
├── dataset/
│   └── default.yaml           # Dataset creation defaults
└── training/
    └── base.yaml              # Training defaults
```

### Configuration Loading
```python
# Load configuration
from src.utils.config_hydra import get_virus_config_hydra
config = get_virus_config_hydra('flu_ha_na_5ks')

# Access parameters
print(f"Virus: {config.virus.virus_name}")
print(f"Max isolates: {config.max_isolates_to_process}")
```

*For detailed configuration guide, see [`../../docs/CONFIGURATION_GUIDE.md`](../../docs/CONFIGURATION_GUIDE.md)*

## 🚀 Automation Scripts

### Shell Scripts (`scripts/`)

#### Standardized Stage Scripts
- **`stage2_esm2.sh`**: ESM-2 embedding generation
- **`stage3_dataset.sh`**: Dataset creation
- **`stage4_train.sh`**: Model training

#### Legacy Scripts (for reference)
- **`preprocess_*_protein.sh`**: Preprocessing scripts
- **`esm2_*.sh`**: Legacy embedding scripts
- **`dataset_*.sh`**: Legacy dataset scripts
- **`classifier_*.sh`**: Legacy training scripts

#### Script Features
- **Error handling**: Comprehensive error checking
- **Logging**: Automatic logging to `logs/` directory
- **Path management**: Automatic path resolution
- **Configuration**: Hydra integration
- **Experiment registry**: Automatic experiment tracking

## 📊 Data Flow Architecture

### Stage 1: Preprocessing
```
Raw GTO Files → Preprocessing Script → protein_final.csv
Location: data/processed/{virus}/{data_version}/ (shared)
```

### Stage 2: Embeddings
```
protein_final.csv → ESM-2 Embeddings → master_esm2_embeddings.h5 + .parquet
Location: data/embeddings/{virus}/{data_version}/ (shared, master cache)
```

### Stage 3: Dataset Creation
```
ESM-2 embeddings → Segment Pairs → train/val/test datasets
Location: data/datasets/{virus}/{data_version}/runs/dataset_{bundle}_{timestamp}/
```

### Stage 4: Training
```
Segment Pairs → ESM-2 Classifier → Trained Model + Predictions
Location: models/{virus}/{data_version}/runs/training_{bundle}_{timestamp}/
```

### Stage 5: Analysis
```
Predictions → Analysis Scripts → Results + Plots
Location: results/{virus}/{data_version}/runs/{training_run_id}/
```

## 🔧 Key Design Patterns

### 1. Configuration-Driven
- All scripts use Hydra configuration
- Consistent parameter management
- Easy experiment customization

### 2. Path Management
- Centralized path generation (`path_utils.py`)
- Consistent naming conventions
- Automatic directory creation
- Shared vs experiment-specific paths

### 3. Error Handling
- Comprehensive error checking
- Detailed logging
- Graceful failure handling

### 4. Modularity
- Each stage is independent
- Clear input/output contracts
- Reusable components

### 5. Master Cache System
- Embeddings computed once, reused everywhere
- Efficient HDF5 + Parquet storage
- O(1) lookup via parquet index

## 📁 Data Directory Structure

### Automatic Path Generation
```
data/
├── raw/                          # Original data
├── processed/
│   └── {virus}/
│       └── {data_version}/      # Preprocessed data (shared)
├── embeddings/
│   └── {virus}/
│       └── {data_version}/       # ESM-2 embeddings (shared, master cache)
│           ├── master_esm2_embeddings.h5
│           └── master_esm2_embeddings.parquet
├── datasets/
│   └── {virus}/
│       └── {data_version}/
│           └── runs/             # Experiment-specific datasets
│               └── dataset_{bundle}_{timestamp}/
└── models/
    └── {virus}/
        └── {data_version}/
            └── runs/             # Experiment-specific models
                └── training_{bundle}_{timestamp}/
```

**Key Principle**: Preprocessing and embeddings are **shared** per virus/data_version. Datasets and models are **experiment-specific** in `runs/` subdirectories.

## 🔍 Key Components

### 1. Path Utilities (`src/utils/path_utils.py`)
```python
def build_training_paths(project_root, virus_name, data_version, run_suffix, config):
    """Generate training-specific paths."""
    return {
        'dataset_dir': dataset_dir,
        'embeddings_file': embeddings_file,
        'output_dir': output_dir
    }
```

### 2. Configuration Loading (`src/utils/config_hydra.py`)
```python
def get_virus_config_hydra(config_bundle, config_path='conf'):
    """Load Hydra configuration for virus."""
    # Load and merge configuration
    # Return flattened DictConfig
```

### 3. Master Embeddings (`src/embeddings/compute_esm2_embeddings.py`)
```python
# Master cache format:
# - master_esm2_embeddings.h5: (N, 1280) embedding array
# - master_esm2_embeddings.parquet: brc_fea_id → row index mapping
```

## 🎯 Extension Points

### Adding New Viruses
1. **Create virus config** in `conf/virus/`
2. **Create paths config** in `conf/paths/`
3. **Create bundle config** in `conf/bundles/`
4. **Add preprocessing script** (if needed)

### Adding New Analysis
1. **Create analysis script** in `src/analysis/`
2. **Follow existing patterns**
3. **Add to documentation**

### Customizing Experiments
1. **Modify configuration files** in `conf/bundles/`
2. **Adjust sampling parameters**
3. **Run individual stages**

## 📝 Development Guidelines

### Code Style
- **Python**: Follow PEP 8
- **Shell**: Use consistent formatting
- **Documentation**: Clear docstrings

### Testing
- **Unit tests**: Test individual functions
- **Integration tests**: Test pipeline stages
- **End-to-end tests**: Test complete workflows

### Documentation
- **Code comments**: Explain complex logic
- **Docstrings**: Document functions and classes
- **README files**: Document each module

## 🔧 Maintenance

### Regular Tasks
- **Update dependencies**: Keep packages current
- **Review logs**: Check for errors and issues
- **Clean up**: Remove old data and logs
- **Backup**: Backup important results

### Monitoring
- **Disk space**: Monitor data directory usage
- **Performance**: Track execution times
- **Errors**: Review error logs regularly
- **Results**: Validate output quality

## 📚 Related Documentation

- **[Configuration Guide](../../docs/CONFIGURATION_GUIDE.md)** - Detailed configuration documentation
- **[Experiment Results](../../docs/EXPERIMENT_RESULTS_ANALYSIS.md)** - Current experiment results
- **[Project Status](../../docs/EXP_RESULTS_STATUS.md)** - Research status and roadmap
- **[Pipeline Overview](../pipeline-overview.md)** - Pipeline overview
