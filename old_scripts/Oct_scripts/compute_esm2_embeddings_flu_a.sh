#!/bin/bash
# Compute ESM-2 embeddings for protein data

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
CUDA_NAME="cuda:0"
FORCE_RECOMPUTE=""  # Set to "--force-recompute" to bypass cache
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/embeddings"
LOG_FILE="$LOG_DIR/compute_esm2_${CONFIG_BUNDLE}_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Print header
log "========================================================================"
log "ESM-2 Embeddings Computation"
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
log "Starting embeddings computation with config bundle: $CONFIG_BUNDLE"
log "Command: python src/embeddings/compute_esm2_embeddings.py --config_bundle $CONFIG_BUNDLE --cuda_name $CUDA_NAME $FORCE_RECOMPUTE"
log ""

python "$PROJECT_ROOT/src/embeddings/compute_esm2_embeddings.py" \
    --config_bundle "$CONFIG_BUNDLE" \
    --cuda_name "$CUDA_NAME" \
    $FORCE_RECOMPUTE \
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
