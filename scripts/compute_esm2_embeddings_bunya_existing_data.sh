#!/bin/bash
# Compute ESM-2 embeddings for Bunyavirales using existing processed data

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Get project root (auto-detect from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
CONFIG_BUNDLE="bunya"
CUDA_NAME="cuda:6"
EXISTING_INPUT_FILE="data/processed/bunya/April_2025/protein_final.csv"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/embeddings"
LOG_FILE="$LOG_DIR/compute_esm2_${CONFIG_BUNDLE}_existing_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

# Print header
log "========================================================================"
log "Bunyavirales ESM-2 Embeddings (Using Existing Processed Data)"
log "========================================================================"
log "Config bundle: $CONFIG_BUNDLE"
log "CUDA device:   $CUDA_NAME"
log "Input file:    $EXISTING_INPUT_FILE"
log "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
log "Host:          $(hostname)"
log "User:          $(whoami)"
log "Python:        $(which python)"
log "Log file:      $LOG_FILE"
log "========================================================================"

# Capture git info for provenance
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")
GIT_DIRTY=$(git status --porcelain 2>/dev/null | wc -l)

log ""
log "Git commit:    $GIT_COMMIT"
log "Git branch:    $GIT_BRANCH"
log "Git dirty:     $([[ $GIT_DIRTY -gt 0 ]] && echo "Yes ($GIT_DIRTY changes)" || echo "No")"
log ""

# Run embeddings computation
log "Starting ESM-2 embeddings computation..."
log "Command: python src/embeddings/compute_esm2_embeddings.py --config_bundle $CONFIG_BUNDLE --input_file $EXISTING_INPUT_FILE --cuda_name $CUDA_NAME"
log ""

python src/embeddings/compute_esm2_embeddings.py \
    --config_bundle "$CONFIG_BUNDLE" \
    --input_file "$EXISTING_INPUT_FILE" \
    --cuda_name "$CUDA_NAME" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# Print footer
log ""
log "========================================================================"
log "ESM-2 embeddings computation completed"
log "Exit code:     $EXIT_CODE"
log "Finished:      $(date '+%Y-%m-%d %H:%M:%S')"
log "Log saved to:  $LOG_FILE"
log "========================================================================"

# Create symlink to latest log for easy access
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/compute_esm2_${CONFIG_BUNDLE}_existing_latest.log"
log "Symlink:       ${LOG_DIR}/compute_esm2_${CONFIG_BUNDLE}_existing_latest.log"

exit $EXIT_CODE
