# Random Seed System

## Overview

This project uses a hierarchical seed system to ensure reproducibility across all pipeline stages while maintaining flexibility for experimentation.

## Seed Hierarchy

### Three-Level System

1. **Master Seed** (highest level)
   - Single seed that generates all process-specific seeds
   - Set in bundle config: `master_seed: 42`
   - Ensures full pipeline reproducibility from a single value

2. **Process Seeds** (middle level)
   - Separate seeds for each pipeline stage
   - Can override master seed for specific stages
   - Set in bundle config: `process_seeds.preprocessing: 123`

3. **No Seed** (lowest level)
   - `null` means truly random behavior
   - Useful for exploration and non-reproducible experiments

### Process Types

- `preprocessing`: Data sampling and preprocessing
- `embeddings`: ESM-2 embedding computation
- `dataset`: Dataset splitting and augmentation
- `training`: Model training and initialization
- `evaluation`: Evaluation sampling and metrics

## Resolution Logic

```python
# Precedence (highest to lowest):
1. Explicit process seed     → process_seeds.preprocessing: 123
2. Derived from master seed   → master_seed: 42 → preprocessing gets derived seed
3. Random (no seed)          → master_seed: null → truly random
```

## Configuration Examples

### Example 1: Fully Deterministic Pipeline (Recommended for Production)

```yaml
# conf/bundles/flu_a.yaml
master_seed: 42
max_files_to_process: null  # Full dataset
process_seeds:
  preprocessing: null  # Derives from master_seed
  embeddings: null     # Derives from master_seed
  dataset: null        # Derives from master_seed
  training: null       # Derives from master_seed
  evaluation: null     # Derives from master_seed
```

**Result:**
- Directory: `processed/flu_a/July_2025/` (no suffix - full dataset)
- All stages use deterministic seeds derived from `master_seed: 42`
- Fully reproducible pipeline
- **Use case:** Production runs, final experiments, paper results

---

### Example 2: Deterministic Subset Sampling

```yaml
master_seed: 42
max_files_to_process: 500
process_seeds:
  preprocessing: null  # Derives from master_seed
  embeddings: null     # Derives from master_seed
  dataset: null        # Derives from master_seed
  training: null       # Derives from master_seed
  evaluation: null     # Derives from master_seed
```

**Result:**
- Directory: `processed/flu_a/July_2025_seed_42_GTOs_500/`
- Preprocessing samples 500 files deterministically using derived seed
- All stages use seeds derived from `master_seed: 42`
- Fully reproducible with subset of data
- **Use case:** Fast iteration during development, subset experiments

---

### Example 3: Different Preprocessing Samples, Same Downstream Seeds

```yaml
master_seed: 42
max_files_to_process: 500
process_seeds:
  preprocessing: 123  # Different sample than master_seed
  embeddings: null    # Uses master_seed
  dataset: null       # Uses master_seed
  training: null      # Uses master_seed
  evaluation: null    # Uses master_seed
```

**Result:**
- Directory: `processed/flu_a/July_2025_seed_123_GTOs_500/`
- Preprocessing uses seed 123 (different data sample)
- Embeddings/training use seeds derived from master_seed 42
- **Use case:** Data variation experiments, cross-validation with different data splits

---

### Example 4: Random Exploration

```yaml
master_seed: null
max_files_to_process: 100
process_seeds:
  preprocessing: null
  embeddings: null
  dataset: null
  training: null
  evaluation: null
```

**Result:**
- Directory: `processed/flu_a/July_2025_random_20251013_143522_GTOs_100/`
- All stages use random seeds (includes timestamp to prevent collisions)
- Not reproducible (useful for initial exploration)
- **Use case:** Quick exploratory analysis, sanity checks

---

### Example 5: Manual Directory Naming

```yaml
master_seed: 42
max_files_to_process: 500
run_suffix: "_special_experiment"
process_seeds:
  preprocessing: null
  embeddings: null
  dataset: null
  training: null
  evaluation: null
```

**Result:**
- Directory: `processed/flu_a/July_2025_special_experiment/`
- Seeds derived from master_seed, but custom directory name
- **Use case:** Custom experiments with specific naming requirements, backward compatibility

---

## Directory Naming

The run suffix is automatically generated based on sampling parameters:

| Scenario | `max_files_to_process` | `preprocessing_seed` | Directory Suffix | Example Path |
|----------|------------------------|----------------------|------------------|--------------|
| Full dataset | `null` | any | `` (no suffix) | `processed/flu_a/July_2025/` |
| Deterministic sample | `500` | `42` | `_seed_42_GTOs_500` | `processed/flu_a/July_2025_seed_42_GTOs_500/` |
| Random sample | `100` | `null` | `_random_<timestamp>_GTOs_100` | `processed/flu_a/July_2025_random_20251013_143522_GTOs_100/` |
| Manual override | any | any | `<custom>` | `processed/flu_a/July_2025_special_experiment/` |

### Auto-Generation Logic

```python
# In src/utils/path_utils.py
def generate_run_suffix(max_files, seed, timestamp=True):
    if max_files is None:
        return ""  # Full dataset
    
    if seed is not None:
        return f"_seed_{seed}_GTOs_{max_files}"  # Deterministic
    else:
        # Random with timestamp to prevent collisions
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"_random_{ts}_GTOs_{max_files}"
```

### Manual Override

You can manually set `run_suffix` in config to override auto-generation:

```yaml
run_suffix: "_special_experiment"
```

**Use this for:**
- Custom experiments with specific naming requirements
- Backward compatibility with old directory structures
- When you want a simpler or more descriptive name

---

## Pipeline Propagation

The `run_suffix` automatically propagates through the entire pipeline:

```
1. Preprocessing (generates suffix)
   ↓
   processed/flu_a/July_2025_seed_42_GTOs_500/
   
2. Embeddings (reads suffix from config)
   ↓
   embeddings/flu_a/July_2025_seed_42_GTOs_500/
   
3. Training (reads suffix from config)
   ↓
   models/flu_a/July_2025_seed_42_GTOs_500/
```

All downstream scripts read `config.run_suffix` to use matching directories. No manual path management needed!

---

## Command-Line Overrides

Hydra allows overriding any config value from the command line:

```bash
# Override master seed
python preprocess_flu_protein.py --virus_name flu_a master_seed=123

# Override preprocessing seed specifically
python preprocess_flu_protein.py --virus_name flu_a process_seeds.preprocessing=999

# Override max_files_to_process
python preprocess_flu_protein.py --virus_name flu_a max_files_to_process=1000

# Multiple overrides
python preprocess_flu_protein.py --virus_name flu_a master_seed=42 max_files_to_process=500

# Override run_suffix manually
python preprocess_flu_protein.py --virus_name flu_a run_suffix="_custom_run"
```

---

## Implementation Details

### Seed Derivation

Process-specific seeds are derived from the master seed using a hash-based approach:

```python
from src.utils.seed_utils import get_process_seed

preprocessing_seed = get_process_seed(master_seed=42, process_name='preprocessing')
# Returns: 3844670132 (deterministic, derived from master_seed=42)
```

This ensures:
- **Different processes get different seeds** (no correlation between stages)
- **Same master seed always produces same process seeds** (fully reproducible)
- **Deterministic and collision-free** (hash-based derivation)

### Setting Seeds in Code

```python
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds

# Resolve seed (handles precedence logic: explicit > derived > random)
seed = resolve_process_seed(config, 'preprocessing')

# Set all random number generators (Python, NumPy, PyTorch)
if seed is not None:
    set_deterministic_seeds(seed, cuda_deterministic=True)
    print(f'Using seed: {seed}')
else:
    print('No seed set - using random behavior')
```

### Path Generation

```python
from src.utils.path_utils import resolve_run_suffix, build_preprocessing_paths

# Auto-generate or use manual override
run_suffix = resolve_run_suffix(
    config=config,
    max_files=MAX_FILES_TO_PROCESS,
    seed=RANDOM_SEED,
    auto_timestamp=True
)

# Build standard paths
paths = build_preprocessing_paths(
    project_root=project_root,
    virus_name='flu_a',
    data_version='July_2025',
    run_suffix=run_suffix
)
```

---

## Best Practices

### ✅ Do:

- **Use `master_seed`** for full pipeline reproducibility
- **Set `master_seed: null`** only for initial exploration
- **Use explicit `process_seeds`** for targeted experiments (e.g., data variation)
- **Document your seed choices** in experiment notes or paper
- **Use `max_files_to_process`** for faster iteration during development
- **Let directory names auto-generate** (they encode sampling strategy)
- **Check directory names** to understand how data was sampled

### ❌ Don't:

- **Hard-code seeds** in Python scripts (always use config)
- **Reuse the same seed** for different purposes (use derived seeds)
- **Forget to set `max_files_to_process`** when sampling (defaults to full dataset)
- **Manually create directory names** (use auto-generation for consistency)
- **Ignore seed warnings** (they indicate non-reproducible behavior)
- **Mix random and deterministic seeds** within same experiment (unless intentional)

---

## Troubleshooting

### "Directory already exists"

**Cause:** You've run with the same `max_files_to_process` and seed before.

**Solutions:**
- Change seed: `master_seed=123`
- Change sample size: `max_files_to_process=1000`
- Use manual suffix: `run_suffix="_run2"`
- Delete old directory if rerunning

---

### "Results not reproducible"

**Causes and solutions:**

1. **Seed not set:**
   - Check: `master_seed` is not `null`
   - Check: `process_seeds` are all `null` (deriving from master)

2. **CUDA non-determinism:**
   - Some CUDA operations are inherently non-deterministic
   - Set `cuda_deterministic=True` in `set_deterministic_seeds()`
   - Note: This can be slower (up to 20-30% performance impact)

3. **Multi-GPU training:**
   - Distributed training may have non-determinism
   - Use single GPU for fully reproducible results

4. **Data loading order:**
   - Ensure `shuffle=False` or use seeded shuffle
   - Check DataLoader workers are seeded

5. **External randomness:**
   - Some transformers operations use random sampling
   - Check library-specific seed settings

---

### "Directory name doesn't match seed"

**Causes:**
- Manual `run_suffix` override in config
- Preprocessing seed differs from master seed (explicit override)
- Random sampling (timestamp makes each run unique)

**To check:**
```bash
# View the config that was used
cat processed/flu_a/July_2025_seed_42_GTOs_500/.hydra/config.yaml
```

---

### "Embeddings directory not found"

**Cause:** Preprocessing and embeddings scripts using different `run_suffix`.

**Solution:**
- Ensure both scripts use the same config bundle
- Check that `run_suffix` is consistently set (or `null` for auto-generation)
- Verify `max_files_to_process` matches between runs

---

## Advanced Usage

### Cross-Validation with Different Data Splits

```yaml
# Split 1
master_seed: 42
max_files_to_process: 5000
process_seeds:
  preprocessing: 1  # Different data sample
  embeddings: null  # Same downstream seeds
  training: null

# Split 2
master_seed: 42
max_files_to_process: 5000
process_seeds:
  preprocessing: 2  # Different data sample
  embeddings: null  # Same downstream seeds
  training: null
```

Creates:
- `processed/flu_a/July_2025_seed_1_GTOs_5000/`
- `processed/flu_a/July_2025_seed_2_GTOs_5000/`

---

### Ensemble Models with Different Initializations

```yaml
# Model 1
master_seed: 42
process_seeds:
  training: 1  # Different initialization

# Model 2
master_seed: 42
process_seeds:
  training: 2  # Different initialization
```

Same data preprocessing, different model initializations.

---

## See Also

- **Implementation:** `src/utils/seed_utils.py` - Seed resolution and setting functions
- **Path utilities:** `src/utils/path_utils.py` - Directory naming functions
- **Configs:** `conf/bundles/` - Bundle configurations with seed examples
- **Scripts:** `src/preprocess/preprocess_flu_protein.py` - Preprocessing with dynamic paths
- **Embeddings:** `src/embeddings/compute_esm2_embeddings.py` - Embeddings with seed management

