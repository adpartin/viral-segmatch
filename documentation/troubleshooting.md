# Troubleshooting Guide

Common issues and solutions for the viral-segmatch project.

**Note**: For detailed technical documentation and research results, see [`../docs/`](../docs/).

## 🚨 Common Errors

### 1. Division by Zero in Analysis
**Error**: `ZeroDivisionError: division by zero` in `analyze_fp_fn_errors`

**Cause**: No false negatives (FN) in the dataset, causing division by zero when calculating FP/FN ratio.

**Solution**: ✅ **Fixed in latest version** - The script now handles zero FN cases gracefully.

### 2. File Not Found Errors
**Error**: `FileNotFoundError: Test results file not found`

**Cause**: Model directory path is incorrect or model hasn't been trained.

**Solutions**:
1. **Check model directory**:
   ```bash
   ls -la models/flu/July_2025/runs/training_flu_ha_na_5ks_*/
   ```

2. **Use explicit model directory**:
   ```bash
   python src/analysis/analyze_stage4_train.py \
     --config_bundle flu_ha_na_5ks \
     --model_dir models/flu/July_2025/runs/training_flu_ha_na_5ks_YYYYMMDD_HHMMSS
   ```

3. **Train model first**:
   ```bash
   ./scripts/stage4_train.sh flu_ha_na_5ks --cuda_name cuda:7 \
       --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_YYYYMMDD_HHMMSS
   ```

### 3. Missing Configuration Files
**Error**: `ConfigKeyError: Missing key data_version`

**Cause**: Incorrect configuration access or missing keys.

**Solution**:
```python
# Wrong:
data_version = config['data_version']

# Correct:
data_version = config.virus.data_version
```

## 🔧 Configuration Issues

### 1. Hydra Configuration Errors
**Error**: `ConfigKeyError: Missing key virus_name`

**Cause**: Incorrect bundle name or missing configuration.

**Solutions**:
1. **Check available bundles**:
   ```bash
   ls conf/bundles/
   ```

2. **Verify bundle content**:
   ```python
   from src.utils.config_hydra import get_virus_config_hydra
   config = get_virus_config_hydra('flu_ha_na_5ks')
   print(config.virus.virus_name)
   ```

3. **Use correct bundle name**:
   ```bash
   # Correct
   ./scripts/stage3_dataset.sh flu_ha_na_5ks
   
   # Incorrect
   ./scripts/stage3_dataset.sh flu_ha_na_5k  # Missing 's'
   ```

### 2. Path Resolution Issues
**Error**: Paths not found or incorrect

**Solutions**:
1. **Check path generation**:
   ```python
   from src.utils.path_utils import build_training_paths
   from src.utils.config_hydra import get_virus_config_hydra
   config = get_virus_config_hydra('flu_ha_na_5ks')
   paths = build_training_paths(
       project_root=Path('.'),
       virus_name=config.virus.virus_name,
       data_version=config.virus.data_version,
       run_suffix='',
       config=config
   )
   print(paths)
   ```

2. **Verify directory structure**:
   ```bash
   ls -la data/embeddings/flu/July_2025/
   ls -la data/datasets/flu/July_2025/runs/
   ```

3. **Create missing directories** (usually auto-created):
   ```bash
   mkdir -p data/{embeddings,datasets}/flu/July_2025/
   ```

## 💾 Data Issues

### 1. Missing Input Files
**Error**: `FileNotFoundError: Input file not found`

**Solutions**:
1. **Check preprocessing output**:
   ```bash
   ls -la data/processed/flu/July_2025/protein_final.csv
   ```

2. **Run preprocessing first** (if needed):
   ```bash
   ./scripts/preprocess_flu_protein.sh
   ```

3. **Check embeddings exist**:
   ```bash
   ls -la data/embeddings/flu/July_2025/master_esm2_embeddings.h5
   ```

4. **Run embeddings if missing**:
   ```bash
   ./scripts/stage2_esm2.sh flu --cuda_name cuda:7
   ```

### 2. Insufficient Data
**Error**: "No data found" or empty datasets

**Solutions**:
1. **Check data filtering**:
   ```python
   from src.utils.config_hydra import get_virus_config_hydra
   config = get_virus_config_hydra('flu_ha_na_5ks')
   print(config.virus.selected_functions)
   ```

2. **Increase sampling size**:
   ```yaml
   max_isolates_to_process: 5000  # Instead of 1000
   ```

3. **Check data quality**:
   ```bash
   head data/processed/flu/July_2025/protein_final.csv
   ```

### 3. Can't Find Dataset Directory
**Error**: Dataset directory not found when running training

**Solutions**:
1. **List all dataset runs**:
   ```bash
   ls -lt data/datasets/flu/July_2025/runs/ | head -5
   ```

2. **Search for specific bundle**:
   ```bash
   ls -d data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_*
   ```

3. **Use most recent run**:
   ```bash
   DATASET_DIR=$(ls -td data/datasets/flu/July_2025/runs/dataset_flu_ha_na_5ks_* | head -1)
   ./scripts/stage4_train.sh flu_ha_na_5ks --cuda_name cuda:7 --dataset_dir "$DATASET_DIR"
   ```

## 🐍 Python Environment Issues

### 1. Import Errors
**Error**: `ModuleNotFoundError: No module named 'fair_esm'`

**Solutions**:
1. **Activate correct environment**:
   ```bash
   conda activate cepi
   ```

2. **Install missing packages**:
   ```bash
   pip install fair-esm transformers h5py
   ```

3. **Check Python path**:
   ```python
   import sys
   print(sys.path)
   ```

### 2. CUDA Issues
**Error**: CUDA not available or GPU memory errors

**Solutions**:
1. **Check CUDA availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   ```

2. **Use CPU fallback**:
   ```bash
   # Set CUDA device to CPU
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 16  # Instead of 32
   ```

## 📊 Analysis Issues

### 1. Empty Results
**Error**: No plots generated or empty analysis

**Solutions**:
1. **Check input data**:
   ```python
   import pandas as pd
   df = pd.read_csv('models/flu/July_2025/runs/training_flu_ha_na_5ks_*/test_predicted.csv')
   print(f"Data shape: {df.shape}")
   print(f"Columns: {df.columns.tolist()}")
   ```

2. **Verify model predictions**:
   ```python
   print(f"Unique labels: {df['label'].unique()}")
   print(f"Unique predictions: {df['pred_label'].unique()}")
   ```

3. **Check file permissions**:
   ```bash
   ls -la results/flu/July_2025/flu_ha_na_5ks/
   ```

### 2. Plot Generation Errors
**Error**: Matplotlib or plotting errors

**Solutions**:
1. **Check display backend**:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For headless environments
   ```

2. **Verify data types**:
   ```python
   print(df.dtypes)
   print(df['pred_prob'].min(), df['pred_prob'].max())
   ```

## 🔍 Debugging Tips

### 1. Enable Verbose Logging
```bash
# Check script logs
tail -f logs/datasets/dataset_segment_pairs_flu_ha_na_5ks_*.log
tail -f logs/training/train_esm2_frozen_pair_classifier_flu_ha_na_5ks_*.log
```

### 2. Check Intermediate Files
```bash
# Verify each stage output
ls -la data/processed/flu/July_2025/
ls -la data/embeddings/flu/July_2025/
ls -la data/datasets/flu/July_2025/runs/ | head -5
ls -la models/flu/July_2025/runs/ | head -5
```

### 3. Test with Small Datasets
```yaml
# Use smaller sampling for testing
max_isolates_to_process: 100  # Instead of 5000
```

### 4. Verify Configuration
```bash
# Test configuration loading
python -c "from src.utils.config_hydra import get_virus_config_hydra; config = get_virus_config_hydra('flu_ha_na_5ks'); print('OK')"
```

## 📞 Getting Help

### 1. Check Existing Documentation
- Review [Pipeline Overview](pipeline-overview.md)
- Check [Configuration Guide](../docs/CONFIGURATION_GUIDE.md)
- Look at existing [`../docs/`](../docs/) for technical notes

### 2. Verify System Requirements
- Python 3.9+
- CUDA-capable GPU (recommended)
- Sufficient disk space
- Internet connection for model downloads

### 3. Test Basic Functionality
```bash
# Test configuration loading
python -c "from src.utils.config_hydra import get_virus_config_hydra; print('OK')"

# Test path utilities
python -c "from src.utils.path_utils import build_training_paths; print('OK')"

# Test ESM-2 availability
python -c "import fair_esm; print('OK')"
```

### 4. Check Recent Changes
- Review git history for recent modifications
- Check if issues are related to recent updates
- Verify all dependencies are up to date

## 📚 Related Documentation

- **[Configuration Guide](../docs/CONFIGURATION_GUIDE.md)** - Detailed configuration documentation
- **[Experiment Results](../docs/EXPERIMENT_RESULTS_ANALYSIS.md)** - Current experiment results
- **[Project Status](../docs/EXP_RESULTS_STATUS.md)** - Research status and roadmap
