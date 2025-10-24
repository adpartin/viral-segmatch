#!/bin/bash
# ESM-2 Frozen Pair Classifier Training for Bunya
# Usage: ./scripts/classifier_bunya.sh
# This script trains the ESM-2 frozen pair classifier for the Bunya dataset
# using the v2 training script with Hydra configuration and Bunya paths.

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Get project root (auto-detect from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
CONFIG_BUNDLE="bunya"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/training"
LOG_FILE="$LOG_DIR/train_esm2_frozen_pair_classifier_${CONFIG_BUNDLE}_${TIMESTAMP}.log"

# Path overrides - use Bunya directories
DATASET_DIR="$PROJECT_ROOT/data/datasets/bunya/April_2025"
EMBEDDINGS_DIR="$PROJECT_ROOT/data/embeddings/bunya/April_2025"
OUTPUT_DIR="$PROJECT_ROOT/models/bunya/April_2025"

CUDA_NAME="cuda:7"  # CUDA device

# Create log directory
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Header
log "========================================================================"
log "ESM-2 Frozen Pair Classifier Training (Bunya)"
log "========================================================================"
log "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
log "Config bundle: $CONFIG_BUNDLE"
log "Host:          $(hostname)"
log "User:          $(whoami)"
log "Python:        $(which python)"
log "Log file:      $LOG_FILE"
log ""
log "Overrides:"
log "  CUDA device:    $CUDA_NAME"
log "  Dataset Dir:    $DATASET_DIR"
log "  Embeddings Dir: $EMBEDDINGS_DIR"
log "  Output Dir:     $OUTPUT_DIR"
log ""

# Capture git info for provenance
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")
GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l)"
log ""
log "Git commit:    $GIT_COMMIT"
log "Git branch:    $GIT_BRANCH"
log "Git dirty:     $([[ $GIT_DIRTY -gt 0 ]] && echo "Yes ($GIT_DIRTY changes)" || echo "No")"
log ""

# Build command with path overrides
CMD="python $PROJECT_ROOT/src/models/train_esm2_frozen_pair_classifier.py --config_bundle $CONFIG_BUNDLE --cuda_name $CUDA_NAME"

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
log "Starting ESM-2 frozen pair classifier training with config bundle: $CONFIG_BUNDLE"
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
