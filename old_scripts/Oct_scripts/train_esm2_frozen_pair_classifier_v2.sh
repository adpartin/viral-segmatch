#!/bin/bash
# ESM-2 Frozen Pair Classifier Training Script
# 
# Usage: ./scripts/train_esm2_frozen_pair_classifier_v2.sh
# 
# This script trains the ESM-2 frozen pair classifier using the v2 training script
# with Hydra configuration and optional path overrides.

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Get project root (auto-detect from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
# CONFIG_BUNDLE="bunya"
CONFIG_BUNDLE="flu_a"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/training"
LOG_FILE="$LOG_DIR/train_esm2_frozen_pair_classifier_${CONFIG_BUNDLE}_${TIMESTAMP}.log"

# Optional path overrides (leave empty to use config defaults)
# Bunya (default):
# DATASET_DIR=""
# EMBEDDINGS_DIR=""
# OUTPUT_DIR=""
#
# Flu A (default):
# DATASET_DIR=""
# EMBEDDINGS_DIR=""
# OUTPUT_DIR=""
#
# Bunya (specified paths):
# DATASET_DIR="$PROJECT_ROOT/data/datasets/bunya/April_2025_v2"
# EMBEDDINGS_DIR="$PROJECT_ROOT/data/embeddings/bunya/April_2025_v2"
# OUTPUT_DIR="$PROJECT_ROOT/data/models/bunya/April_2025_v2"
#
# Flu A (specified paths):
DATASET_DIR="$PROJECT_ROOT/data/datasets/flu_a/July_2025_v2"
EMBEDDINGS_DIR="$PROJECT_ROOT/data/embeddings/flu_a/July_2025_v2"
OUTPUT_DIR="$PROJECT_ROOT/data/models/flu_a/July_2025_v2"

CUDA_NAME="cuda:7"     # Default CUDA device

# Create log directory
mkdir -p "$LOG_DIR"

# Logging helper
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Header
log "========================================================================"
log "ESM-2 Frozen Pair Classifier Training"
log "========================================================================"
log "Timestamp: $(date)"
log "Config Bundle: $CONFIG_BUNDLE"
log "CUDA Device: $CUDA_NAME"
log "Project Root: $PROJECT_ROOT"
log "Log File: $LOG_FILE"
log ""
log "Path Overrides:"
log "  Dataset Dir:    $DATASET_DIR"
log "  Embeddings Dir: $EMBEDDINGS_DIR"
log "  Output Dir:     $OUTPUT_DIR"
log ""

# Git provenance
if command -v git >/dev/null 2>&1; then
    log "Git Information:"
    log "  Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    log "  Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    log "  Status: $(git status --porcelain 2>/dev/null | wc -l) modified files"
    log ""
fi

# Path overrides are set above in the main configuration section

# Build command with optional path overrides
CMD="python $PROJECT_ROOT/src/models/train_esm2_frozen_pair_classifier_v2.py --config_bundle $CONFIG_BUNDLE --cuda_name $CUDA_NAME"

if [ -n "$DATASET_DIR" ]; then
    CMD="$CMD --dataset_dir $DATASET_DIR"
fi

if [ -n "$EMBEDDINGS_DIR" ]; then
    CMD="$CMD --embeddings_dir $EMBEDDINGS_DIR"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Run the training script
log "Starting ESM-2 frozen pair classifier with config bundle: $CONFIG_BUNDLE"
log "Command: $CMD"
log ""

set +e  # Temporarily disable exit on error
eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e  # Re-enable exit on error

# Footer
log ""
log "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    log "✅ ESM-2 frozen pair classifier training completed successfully!"
else
    log "❌ ESM-2 frozen pair classifier training failed with exit code: $EXIT_CODE"
fi
log "End time: $(date)"
log "Total runtime: $SECONDS seconds"
log "========================================================================"

# Create symlink to latest log
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/train_esm2_frozen_pair_classifier_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE





