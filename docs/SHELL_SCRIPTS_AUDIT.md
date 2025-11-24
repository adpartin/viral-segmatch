# Shell Scripts Audit & Consistency Check

**Date**: 2025-10-14  
**Status**: ✅ Completed

---

## Overview

This document verifies that all shell scripts follow a consistent structure and use the updated `--config_bundle` parameter system.

---

## Scripts Audited

### **Preprocessing Scripts (3)**
1. ✅ `scripts/preprocess_flu_a_protein.sh`
2. ✅ `scripts/preprocess_bunya_protein.sh`
3. ✅ `scripts/preprocess_flu_a_pb1_pb2.sh`

### **Embeddings Scripts (2)**
1. ✅ `scripts/compute_esm2_embeddings_flu_a.sh`
2. ✅ `scripts/compute_esm2_embeddings_bunya.sh`

---

## Consistency Checklist

All scripts now follow this standard structure:

### **1. Shebang & Error Handling**
```bash
#!/bin/bash
set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure
```
✅ **Status**: Consistent across all scripts

---

### **2. Project Root Detection**
```bash
# Get project root (auto-detect from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
```
✅ **Status**: Consistent across all scripts

---

### **3. Configuration Section**
```bash
# Configuration
CONFIG_BUNDLE="flu_a"  # or "bunya", "flu_a_pb1_pb2", etc.
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/{preprocess|embeddings}"
LOG_FILE="$LOG_DIR/{script_name}_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
```
✅ **Status**: All scripts use `CONFIG_BUNDLE` (no `VIRUS_NAME` variable)
✅ **Status**: Embeddings scripts also include `CUDA_NAME` configuration

---

### **4. Log Directory Creation**
```bash
# Create log directory
mkdir -p "$LOG_DIR"
```
✅ **Status**: Consistent across all scripts

---

### **5. Logging Helper Function**
```bash
# Helper function for logging to both console and file
log() {
    echo "$@" | tee -a "$LOG_FILE"
}
```
✅ **Status**: Consistent across all scripts

---

### **6. Header with Metadata**
```bash
# Print header
log "=============================================================================="
log "{Script Description}"
log "=============================================================================="
log "Config bundle: $CONFIG_BUNDLE"
log "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
log "Host:          $(hostname)"
log "User:          $(whoami)"
log "Python:        $(which python)"
log "Log file:      $LOG_FILE"
log "=============================================================================="
```
✅ **Status**: Consistent across all scripts
✅ **Status**: All use 78-character `=` separator lines
✅ **Status**: Embeddings scripts include `log "CUDA device:   $CUDA_NAME"`

---

### **7. Git Provenance**
```bash
# Capture git info for provenance
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")
GIT_DIRTY=$(git status --porcelain 2>/dev/null | wc -l)

log ""
log "Git commit:    $GIT_COMMIT"
log "Git branch:    $GIT_BRANCH"
log "Git dirty:     $([[ $GIT_DIRTY -gt 0 ]] && echo "Yes ($GIT_DIRTY changes)" || echo "No")"
log ""
```
✅ **Status**: Consistent across all scripts

---

### **8. Main Command Execution**

#### **Preprocessing Scripts:**
```bash
# Run preprocessing
log "Starting preprocessing..."
log "Command: python src/preprocess/preprocess_{virus}_protein.py --config_bundle $CONFIG_BUNDLE"
log ""

python src/preprocess/preprocess_{virus}_protein.py \
    --config_bundle "$CONFIG_BUNDLE" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
```
✅ **Status**: Consistent across all preprocessing scripts
✅ **Status**: All use `--config_bundle` parameter only (no `--virus_name`)

#### **Embeddings Scripts:**
```bash
# Run ESM-2 embeddings computation
log "Starting ESM-2 embeddings computation..."
log "Command: python src/embeddings/compute_esm2_embeddings.py --config_bundle $CONFIG_BUNDLE --use_selected_only --cuda_name $CUDA_NAME"
log ""

python "$PROJECT_ROOT/src/embeddings/compute_esm2_embeddings.py" \
    --config_bundle "$CONFIG_BUNDLE" \
    --use_selected_only \
    --cuda_name "$CUDA_NAME" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
```
✅ **Status**: Consistent across all embeddings scripts
✅ **Status**: All use `--config_bundle` parameter (no `--virus_name`)
✅ **Status**: All use `--use_selected_only` flag
✅ **Status**: All use `--cuda_name` parameter

---

### **9. Footer with Exit Status**
```bash
# Print footer
log ""
log "=============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    log "{Success message}"
else
    log "{Failure message}"
fi

log "Exit code:     $EXIT_CODE"
log "Finished:      $(date '+%Y-%m-%d %H:%M:%S')"
log "Log saved to:  $LOG_FILE"
log "=============================================================================="
```
✅ **Status**: Consistent across all scripts
✅ **Status**: All use 78-character `=` separator lines

---

### **10. Symlink to Latest Log**
```bash
# Create symlink to latest log for easy access
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/{script_type}_${CONFIG_BUNDLE}_latest.log"
log "Symlink:       ${LOG_DIR}/{script_type}_${CONFIG_BUNDLE}_latest.log"
```
✅ **Status**: Consistent across all scripts

---

### **11. Exit with Status**
```bash
exit $EXIT_CODE
```
✅ **Status**: Consistent across all scripts

---

## Script-Specific Details

### **Preprocessing Scripts**

#### **1. `preprocess_flu_a_protein.sh`**
```bash
CONFIG_BUNDLE="flu_a"
LOG_FILE="$LOG_DIR/preprocess_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
python src/preprocess/preprocess_flu_protein.py --config_bundle "$CONFIG_BUNDLE"
```
✅ Uses: `conf/bundles/flu_a.yaml`
✅ Calls: `src/preprocess/preprocess_flu_protein.py`

#### **2. `preprocess_bunya_protein.sh`**
```bash
CONFIG_BUNDLE="bunya"
LOG_FILE="$LOG_DIR/preprocess_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
python src/preprocess/preprocess_bunya_protein.py --config_bundle "$CONFIG_BUNDLE"
```
✅ Uses: `conf/bundles/bunya.yaml`
⚠️ Calls: `src/preprocess/preprocess_bunya_protein.py` (older script, needs updating)
📝 **Note**: The Python script `preprocess_bunya_protein.py` still uses hardcoded configs and doesn't support `--config_bundle` yet. This should be updated in the future.

#### **3. `preprocess_flu_a_pb1_pb2.sh`**
```bash
CONFIG_BUNDLE="flu_a_pb1_pb2"
LOG_FILE="$LOG_DIR/preprocess_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
python src/preprocess/preprocess_flu_protein.py --config_bundle "$CONFIG_BUNDLE"
```
✅ Uses: `conf/bundles/flu_a_pb1_pb2.yaml`
✅ Calls: `src/preprocess/preprocess_flu_protein.py`
✅ Special config: Filters to PB1 and PB2 proteins only

---

### **Embeddings Scripts**

#### **1. `compute_esm2_embeddings_flu_a.sh`**
```bash
CONFIG_BUNDLE="flu_a"
CUDA_NAME="cuda:0"
LOG_FILE="$LOG_DIR/compute_esm2_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
python src/embeddings/compute_esm2_embeddings.py \
    --config_bundle "$CONFIG_BUNDLE" \
    --use_selected_only \
    --cuda_name "$CUDA_NAME"
```
✅ Uses: `conf/bundles/flu_a.yaml`
✅ CUDA device: `cuda:0`
✅ Filters to selected functions only

#### **2. `compute_esm2_embeddings_bunya.sh`**
```bash
CONFIG_BUNDLE="bunya"
CUDA_NAME="cuda:6"
LOG_FILE="$LOG_DIR/compute_esm2_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
python src/embeddings/compute_esm2_embeddings.py \
    --config_bundle "$CONFIG_BUNDLE" \
    --use_selected_only \
    --cuda_name "$CUDA_NAME"
```
✅ Uses: `conf/bundles/bunya.yaml`
✅ CUDA device: `cuda:6`
✅ Filters to selected functions only

---

## Key Configuration Parameters

### **All Scripts Use:**
- `CONFIG_BUNDLE` - Single source of truth for experiment config
- No `VIRUS_NAME` variable (derived from config)
- No hardcoded paths (derived from config)

### **Embeddings Scripts Also Use:**
- `CUDA_NAME` - GPU device specification

---

## Log Files

### **Naming Convention:**
```
logs/{stage}/{script_type}_{config_bundle}_{timestamp}.log
```

### **Examples:**
```
logs/preprocess/preprocess_flu_a_20251014_143022.log
logs/preprocess/preprocess_flu_a_pb1_pb2_20251014_143022.log
logs/embeddings/compute_esm2_flu_a_20251014_150045.log
```

### **Symlinks:**
```
logs/preprocess/preprocess_flu_a_latest.log -> preprocess_flu_a_20251014_143022.log
logs/embeddings/compute_esm2_flu_a_latest.log -> compute_esm2_flu_a_20251014_150045.log
```

---

## Configuration Bundles Used

| Script | Config Bundle | Python Script |
|--------|---------------|---------------|
| `preprocess_flu_a_protein.sh` | `flu_a` | `preprocess_flu_protein.py` |
| `preprocess_bunya_protein.sh` | `bunya` | `preprocess_bunya_protein.py` ⚠️ |
| `preprocess_flu_a_pb1_pb2.sh` | `flu_a_pb1_pb2` | `preprocess_flu_protein.py` |
| `compute_esm2_embeddings_flu_a.sh` | `flu_a` | `compute_esm2_embeddings.py` |
| `compute_esm2_embeddings_bunya.sh` | `bunya` | `compute_esm2_embeddings.py` |

⚠️ = Script needs updating to support `--config_bundle`

---

## Future Improvements

### **1. Update `preprocess_bunya_protein.py`**
The Bunya preprocessing script still uses hardcoded configs. It should be refactored to:
- Accept `--config_bundle` parameter
- Use Hydra config system
- Match the structure of `preprocess_flu_protein.py`

### **2. Create Template Script**
Create a template shell script that can be easily copied and customized:
```bash
scripts/templates/preprocessing_template.sh
scripts/templates/embeddings_template.sh
```

### **3. Add Script Generator**
Create a tool to auto-generate shell scripts from bundle configs:
```bash
python tools/generate_shell_script.py --bundle flu_a_pb1_pb2 --stage preprocessing
```

### **4. Add Validation Tool**
Create a tool to validate shell script consistency:
```bash
bash scripts/validate_shell_scripts.sh
```

---

## Summary

✅ **All 5 shell scripts reviewed and updated**
✅ **Consistent structure across all scripts**
✅ **All use `--config_bundle` parameter**
✅ **No hardcoded `VIRUS_NAME` variables**
✅ **Consistent logging and error handling**
✅ **Git provenance tracking in all scripts**
✅ **Consistent separator line formatting (78 chars)**

**Exception:**
⚠️ `preprocess_bunya_protein.py` (Python script) still needs updating to support the new config system. The shell script is ready, but the Python script doesn't accept `--config_bundle` yet.

---

## Testing

### **Verify Each Script:**
```bash
# Test preprocessing scripts
bash scripts/preprocess_flu_a_protein.sh
bash scripts/preprocess_flu_a_pb1_pb2.sh
# bash scripts/preprocess_bunya_protein.sh  # Will fail - Python script needs update

# Test embeddings scripts
bash scripts/compute_esm2_embeddings_flu_a.sh
bash scripts/compute_esm2_embeddings_bunya.sh
```

### **Check Log Files:**
```bash
# View latest logs
cat logs/preprocess/preprocess_flu_a_latest.log
cat logs/embeddings/compute_esm2_flu_a_latest.log
```

---

**Audit Complete!** 🎉

