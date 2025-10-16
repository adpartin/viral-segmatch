#!/bin/bash
# ============================================================================
# ESM-2 Embeddings Computation for Flu A: PB1 and PB2 proteins only
# ============================================================================
# This script computes ESM-2 embeddings for a subset of Flu A proteins
# (PB1 and PB2) using the flu_a_pb1_pb2 bundle configuration. The bundle
# defines:
# - run_suffix: "_seed_42_GTOs_2000_pb1_pb2" (manual directory naming)
# - max_files_to_process: 2000 (subset for faster iteration)
# - master_seed: 42 (reproducible sampling and processing)
# - selected_functions: Only PB1 and PB2 proteins
#
# Input directory:
#   data/processed/flu_a/July_2025_seed_42_GTOs_2000_pb1_pb2/
# Output directory:
#   data/embeddings/flu_a/July_2025_seed_42_GTOs_2000_pb1_pb2/
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Get project root (auto-detect from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
CONFIG_BUNDLE="flu_a_pb1_pb2"
CUDA_NAME="cuda:0"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/embeddings"
LOG_FILE="$LOG_DIR/compute_esm2_${CONFIG_BUNDLE}_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

# Print header
log "========================================================================"
log "ESM-2 Embeddings Computation: Flu A PB1 and PB2 Proteins Only"
log "========================================================================"
log "Config bundle: $CONFIG_BUNDLE"
log "CUDA device:   $CUDA_NAME"
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

# Run ESM-2 embeddings computation
log "Starting ESM-2 embeddings computation..."
log "Command: python src/embeddings/compute_esm2_embeddings.py --config_bundle $CONFIG_BUNDLE --cuda_name $CUDA_NAME"
log ""

python "$PROJECT_ROOT/src/embeddings/compute_esm2_embeddings.py" \
    --config_bundle "$CONFIG_BUNDLE" \
    --cuda_name "$CUDA_NAME" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# Print footer
log ""
log "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    log "ESM-2 embeddings computation completed successfully!"
else
    log "ESM-2 embeddings computation failed!"
fi

log "Exit code:     $EXIT_CODE"
log "Finished:      $(date '+%Y-%m-%d %H:%M:%S')"
log "Log saved to:  $LOG_FILE"
log "========================================================================"

# Create symlink to latest log for easy access
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/compute_esm2_${CONFIG_BUNDLE}_latest.log"
log "Symlink:       ${LOG_DIR}/compute_esm2_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE
