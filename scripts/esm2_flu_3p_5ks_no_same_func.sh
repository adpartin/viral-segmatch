#!/bin/bash
# ESM-2 Embeddings Computation for Flu A 3p_5ks_no_same_func
# Usage: ./scripts/esm2_flu_3p_5ks_no_same_func.sh
# 3p_5ks: 3 major proteins (PB1, PB2, PA) and 5k isolates sampled from the full dataset.
# No same-function negatives: allow_same_func_negatives=false

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Get project root (auto-detect from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
CONFIG_BUNDLE="flu_a_3p_5ks_no_same_func"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/embeddings"
LOG_FILE="$LOG_DIR/compute_esm2_${CONFIG_BUNDLE}_${TIMESTAMP}.log"

# Path overrides - use existing preprocessing, save to 3p_5ks directories
INPUT_FILE="$PROJECT_ROOT/data/processed/flu_a/July_2025/protein_final.csv"  # Use preprocessed data
OUTPUT_DIR=""  # Let config determine the path (will be July_2025_seed_42_isolates_5000)

FORCE_RECOMPUTE=""  # Set to "--force-recompute" to bypass cache
CUDA_NAME="cuda:5"  # CUDA device

# Create log directory
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Print header
log "========================================================================"
log "ESM-2 Embeddings Computation (Flu A 3p_5ks_no_same_func)"
log "========================================================================"
log "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
log "Config bundle: $CONFIG_BUNDLE"
log "Host:          $(hostname)"
log "User:          $(whoami)"
log "Python:        $(which python)"
log "Log file:      $LOG_FILE"
log ""
log "Overrides:"
log "  CUDA device:     $CUDA_NAME"
log "  Force recompute: $([[ $FORCE_RECOMPUTE == "--force-recompute" ]] && echo "Yes" || echo "No")"
log "  Input file:      $INPUT_FILE"
log "  Output dir:      $OUTPUT_DIR"
log "========================================================================"

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
CMD="python $PROJECT_ROOT/src/embeddings/compute_esm2_embeddings.py --config_bundle $CONFIG_BUNDLE --cuda_name $CUDA_NAME $FORCE_RECOMPUTE"

if [ -n "$INPUT_FILE" ]; then
    CMD="$CMD --input_file $INPUT_FILE"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Run ESM-2 embeddings computation
log "Starting embeddings computation with config bundle: $CONFIG_BUNDLE"
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
    log "✅ ESM-2 embeddings computation completed successfully!"
else
    log "❌ ESM-2 embeddings computation failed with exit code: $EXIT_CODE"
fi
log "Exit code:     $EXIT_CODE"
log "Finished:      $(date '+%Y-%m-%d %H:%M:%S')"
log "Log saved to:  $LOG_FILE"
log "========================================================================"

# Create symlink to latest log
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/compute_esm2_${CONFIG_BUNDLE}_latest.log"
log "Symlink:       ${LOG_DIR}/compute_esm2_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE

