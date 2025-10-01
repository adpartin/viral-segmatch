# Data Directory Structure

## Original Directories (from Jim, the biologist)

These directories contain the original data with Jim's naming convention:

- **`Anno_Updates/`** - Bunyavirales genome annotations
- **`Full_Flu_Annos/`** - Influenza A genome annotations

**⚠️ DO NOT modify or rename these directories** - they preserve the original naming for reference.

---

## Symlinks (for code clarity)

For better code readability, we've created symlinks with clearer names:

- **`Bunya/`** → `Anno_Updates/`
- **`Flu_A/`** → `Full_Flu_Annos/`

**Use these symlinks in your code** for clarity and maintainability.

---

## Dataset Structure

### Bunyavirales (Bunya)
```
Bunya/ (symlink → Anno_Updates/)
└── April_2025/
    └── bunya-from-datasets/
        └── Quality_GTOs/
            └── *.qual.gto (genome annotation files)
```

### Influenza A (Flu A)
```
Flu_A/ (symlink → Full_Flu_Annos/)
├── July_2025/                    # Full dataset (111,797 GTO files)
│   └── *.gto
└── July_2025_subset_5k/          # Development subset (5,000 files)
    └── *.gto
```

---

## Development Subsets

For faster iteration during development, we maintain smaller subsets:

### Flu A Subset (5K files)
- **Location:** `Flu_A/July_2025_subset_5k/`
- **Size:** 5,000 files (first 5K files from the full dataset)
- **Purpose:** Fast development and debugging
- **Creation date:** October 1, 2025

**To create additional subsets:**
```bash
# Create a 10K subset
cd Flu_A/July_2025
mkdir -p ../July_2025_subset_10k
find . -maxdepth 1 -name "*.gto" -type f | sort | head -10000 | xargs -I {} cp {} ../July_2025_subset_10k/
```

---

## Usage in Code

### Using Symlinks (Recommended)
```python
# Clear and maintainable
raw_data_dir = project_root / 'data' / 'raw' / 'Flu_A' / 'July_2025'
raw_data_dir = project_root / 'data' / 'raw' / 'Bunya' / 'April_2025'
```

### Switching Between Full Dataset and Subset
```python
# In preprocess_flu_protein.py
USE_SUBSET = True  # Set to False for production runs

if USE_SUBSET:
    raw_data_dir = main_data_dir / 'raw' / 'Flu_A' / 'July_2025_subset_5k'
else:
    raw_data_dir = main_data_dir / 'raw' / 'Flu_A' / 'July_2025'
```

---

## File Counts

| Dataset | Directory | Files | Size |
|---------|-----------|-------|------|
| Flu A (full) | `Flu_A/July_2025/` | 111,797 | ~4 GB |
| Flu A (subset) | `Flu_A/July_2025_subset_5k/` | 5,000 | ~180 MB |
| Bunya | `Bunya/April_2025/` | TBD | TBD |

---

## Performance Notes

**⚠️ Performance Tip:** Globbing 111K files with `glob('*.gto')` takes several seconds. 

**Solutions:**
1. Use development subsets during iteration
2. Cache file lists for production runs
3. Avoid repeated `glob()` calls in tight loops

---

**Last Updated:** October 1, 2025

