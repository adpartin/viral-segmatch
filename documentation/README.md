# Viral Segment Match - Documentation

This directory contains comprehensive user-facing documentation for the viral-segmatch project.

**Note**: For technical documentation, research results, and development logs, see [`../docs/`](../docs/).

## 📚 Documentation Structure

### **Getting Started**
- [Installation Guide](installation.md) - Setup and dependencies
- [Quick Start](quick-start.md) - Run your first experiment
- [Pipeline Overview](pipeline-overview.md) - Understanding the 4-stage pipeline

### **Configuration**
- [Configuration Guide](configuration.md) - Brief overview (see [`../docs/CONFIGURATION_GUIDE.md`](../docs/CONFIGURATION_GUIDE.md) for details)
- [Experiment Setup](../docs/CONFIGURATION_GUIDE.md#creating-a-new-experiment) - Creating new experiments

### **Pipeline Stages**
- [Preprocessing](stages/preprocessing.md) - Raw data to protein sequences
- [Embeddings](../docs/CONFIGURATION_GUIDE.md#stage-2-embeddings) - ESM-2 protein embeddings
- [Dataset Creation](../docs/CONFIGURATION_GUIDE.md#stage-3-dataset-creation) - Segment pair datasets
- [Training](../docs/CONFIGURATION_GUIDE.md#stage-4-training) - ESM-2 frozen classifier training

### **Analysis & Postprocessing**
- [Results Analysis](analysis/results-analysis.md) - Understanding model performance
- [Presentation Plots](analysis/presentation-plots.md) - Creating publication-ready figures
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

### **Development**
- [Code Structure](development/code-structure.md) - Project organization
- [Model Improvements](model-improvements.md) - Training improvements and recommendations

## 🚀 Quick Links

- **Main README**: [../README.md](../README.md)
- **Technical Docs**: [../docs/](../docs/) - Technical notes, research logs, and detailed documentation
- **Configuration Guide**: [../docs/CONFIGURATION_GUIDE.md](../docs/CONFIGURATION_GUIDE.md) - Comprehensive configuration documentation
- **Experiment Results**: [../docs/EXPERIMENT_RESULTS_ANALYSIS.md](../docs/EXPERIMENT_RESULTS_ANALYSIS.md) - Detailed experiment analysis
- **Project Status**: [../docs/EXP_RESULTS_STATUS.md](../docs/EXP_RESULTS_STATUS.md) - Research status and roadmap
- **Scripts**: [../scripts/](../scripts/) - Shell scripts for running experiments
- **Configuration**: [../conf/](../conf/) - YAML configuration files

## 📝 Documentation Organization

### `documentation/` (This Directory)
**Purpose**: User-facing guides and tutorials
- Installation and setup
- Quick start guides
- Pipeline overview
- Troubleshooting
- How-to guides

### `docs/` (Technical Documentation)
**Purpose**: Technical notes, research logs, and detailed documentation
- Configuration system details
- Experiment results analysis
- Research status and roadmap
- Technical deep-dives
- Development history

## 🎯 Current System Overview

### Pipeline Scripts
- **`stage2_esm2.sh`**: Compute ESM-2 embeddings (master cache)
- **`stage3_dataset.sh`**: Create segment pair datasets
- **`stage4_train.sh`**: Train models

### Current Experiments
- **`bunya`**: Bunyavirus baseline (all segments)
- **`flu_ha_na_5ks`**: Flu, HA-NA only (variable segments), 5K isolates
- **`flu_pb2_pb1_pa_5ks`**: Flu, PB2-PB1-PA only (conserved segments), 5K isolates
- **`flu_pb2_ha_na_5ks`**: Flu, PB2-HA-NA (mixed), 5K isolates

### Key Results
- **Variable segments (HA-NA)**: 92.3% accuracy, 91.6% F1, 0.953 AUC
- **Conserved segments (PB2-PB1-PA)**: 71.9% accuracy, 75.3% F1, 0.750 AUC

*For detailed results, see [`../docs/EXPERIMENT_RESULTS_ANALYSIS.md`](../docs/EXPERIMENT_RESULTS_ANALYSIS.md)*

## 🔍 Finding Information

### Need to...
- **Get started quickly?** → [Quick Start](quick-start.md)
- **Understand the pipeline?** → [Pipeline Overview](pipeline-overview.md)
- **Configure experiments?** → [Configuration Guide](../docs/CONFIGURATION_GUIDE.md)
- **Troubleshoot issues?** → [Troubleshooting](troubleshooting.md)
- **Understand results?** → [Results Analysis](analysis/results-analysis.md)
- **See research status?** → [Project Status](../docs/EXP_RESULTS_STATUS.md)
- **Review experiment results?** → [Experiment Results](../docs/EXPERIMENT_RESULTS_ANALYSIS.md)

## 📚 Additional Resources

- **Technical Documentation**: [`../docs/`](../docs/) - Comprehensive technical documentation
- **Configuration System**: [`../docs/CONFIGURATION_GUIDE.md`](../docs/CONFIGURATION_GUIDE.md) - Full configuration guide
- **Research Status**: [`../docs/EXP_RESULTS_STATUS.md`](../docs/EXP_RESULTS_STATUS.md) - Current research status
- **Experiment Results**: [`../docs/EXPERIMENT_RESULTS_ANALYSIS.md`](../docs/EXPERIMENT_RESULTS_ANALYSIS.md) - Detailed analysis
