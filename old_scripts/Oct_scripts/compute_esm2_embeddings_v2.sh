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
CONFIG_BUNDLE="bunya"
# CONFIG_BUNDLE="flu_a"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/embeddings"
LOG_FILE="$LOG_DIR/compute_esm2_${CONFIG_BUNDLE}_${TIMESTAMP}.log"

# Optional path overrides (leave empty to use config defaults)
# Bunya (default):
#
# INPUT_FILE=""
# OUTPUT_DIR=""
#
# Flu A (default):
# INPUT_FILE=""
# OUTPUT_DIR=""
#
# Bunya (specified input and output):
INPUT_FILE="$PROJECT_ROOT/data/processed/bunya/April_2025/protein_final.csv"
OUTPUT_DIR="$PROJECT_ROOT/data/embeddings/bunya/April_2025_v2"
#
# Flu A (specified input and output):
# INPUT_FILE="$PROJECT_ROOT/data/processed/flu_a/July_2025/protein_final.csv"
# OUTPUT_DIR="$PROJECT_ROOT/data/embeddings/flu_a/July_2025_v2"

FORCE_RECOMPUTE=""  # Set to "--force-recompute" to bypass cache
CUDA_NAME="cuda:7"

# Create log directory
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Print header
log "========================================================================"
log "ESM-2 Embeddings Computation"
log "========================================================================"
log "Config bundle: $CONFIG_BUNDLE"
log "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
log "Host:          $(hostname)"
log "User:          $(whoami)"
log "Python:        $(which python)"
log "Log file:      $LOG_FILE"
log ""
log "CUDA device:     $CUDA_NAME"
log "Force recompute: $([[ $FORCE_RECOMPUTE == "--force-recompute" ]] && echo "Yes" || echo "No")"
log "Input file:      $INPUT_FILE"
log "Output dir:      $OUTPUT_DIR"
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

# Build command with optional path overrides
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

# Print footer
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

# Create symlink to latest log for easy access
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/esm2_${CONFIG_BUNDLE}_latest.log"
log "Symlink:       ${LOG_DIR}/esm2_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE
