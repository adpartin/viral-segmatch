#!/bin/bash
# Dataset Segment Pairs Creation for Bunya
# Usage: ./scripts/dataset_bunya.sh
# This script creates segment pairs for the Bunya dataset
# using existing preprocessing data and saving to datasets directory.

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
LOG_DIR="$PROJECT_ROOT/logs/datasets"
LOG_FILE="$LOG_DIR/dataset_segment_pairs_${CONFIG_BUNDLE}_${TIMESTAMP}.log"

# Path overrides - use existing preprocessing, save to datasets directory
INPUT_FILE="$PROJECT_ROOT/data/processed/bunya/April_2025/protein_final.csv" # Use preprocessed data
OUTPUT_DIR="$PROJECT_ROOT/data/datasets/bunya/April_2025"

# Create log directory
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Print header
log "========================================================================"
log "Dataset Segment Pairs Creation (Bunya)"
log "========================================================================"
log "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
log "Config bundle: $CONFIG_BUNDLE"
log "Host:          $(hostname)"
log "User:          $(whoami)"
log "Python:        $(which python)"
log "Log file:      $LOG_FILE"
log ""
log "Overrides:"
log "  Input file:  $INPUT_FILE"
log "  Output dir:  $OUTPUT_DIR"
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
CMD="python $PROJECT_ROOT/src/datasets/dataset_segment_pairs.py --config_bundle $CONFIG_BUNDLE"

if [ -n "$INPUT_FILE" ]; then
    CMD="$CMD --input_file $INPUT_FILE"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Run dataset segment pairs creation
log "Starting dataset segment pairs creation with config bundle: $CONFIG_BUNDLE"
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
    log "✅ Dataset segment pairs creation completed successfully!"
else
    log "❌ Dataset segment pairs creation failed with exit code: $EXIT_CODE"
fi
log "Exit code:     $EXIT_CODE"
log "Finished:      $(date '+%Y-%m-%d %H:%M:%S')"
log "Log saved to:  $LOG_FILE"
log "========================================================================"

# Create symlink to latest log
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/dataset_segment_pairs_${CONFIG_BUNDLE}_latest.log"
log "Symlink:       ${LOG_DIR}/dataset_segment_pairs_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE
