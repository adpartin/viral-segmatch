# Data Directory Structure

## Original Directories (from Jim)

These directories contain the original curated data from Jim:

- **`Anno_Updates/`** - Bunyavirales genome annotations
- **`Full_Flu_Annos/`** - Influenza A genome annotations

**⚠️  DO NOT modify or rename these directories** - they preserve the original naming for reference.

---

## Symlinks (for code clarity)

For better code readability, we've created symlinks with shorter names:

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
└── July_2025/          # Full dataset (111,797 GTO files)
    └── *.gto
```

---

## Usage in Code

### Using Symlinks (Recommended)
```python
# Clear and maintainable
raw_data_dir = project_root / 'data' / 'raw' / 'Flu_A' / 'July_2025'
raw_data_dir = project_root / 'data' / 'raw' / 'Bunya' / 'April_2025'
```

---

## File Counts

| Dataset      | Directory                         | Files   | Size    |
|--------------|-----------------------------------|---------|---------|
| Flu A (full) | `Flu_A/July_2025/`                | 111,797 | ~5.2 GB |
| Bunya        | `Bunya/April_2025/.../*.qual.gto` | 1683    | < 58 MB |

---

**Last Updated:** October 29, 2025

