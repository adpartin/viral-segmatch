#!/bin/bash

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
# CONFIG_BUNDLE="flu_a"  # Default config bundle
CONFIG_BUNDLE="flu_a_pb1_pb2"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/datasets"
LOG_FILE="$LOG_DIR/dataset_segment_pairs_${CONFIG_BUNDLE}_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging helper
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Header
log "========================================================================"
log "Dataset Segment Pairs Creation"
log "========================================================================"
log "Timestamp: $(date)"
log "Config Bundle: $CONFIG_BUNDLE"
log "Project Root: $PROJECT_ROOT"
log "Log File: $LOG_FILE"
log ""

# Git provenance
if command -v git >/dev/null 2>&1; then
    log "Git Information:"
    log "  Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    log "  Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    log "  Status: $(git status --porcelain 2>/dev/null | wc -l) modified files"
    log ""
fi

# Run the dataset creation script
log "Starting dataset segment pairs creation..."
python src/datasets/dataset_segment_pairs_new.py --config_bundle "$CONFIG_BUNDLE" 2>&1 | tee -a "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# Footer
log ""
log "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    log "✅ Dataset segment pairs creation completed successfully!"
else
    log "❌ Dataset segment pairs creation failed with exit code: $EXIT_CODE"
fi
log "End time: $(date)"
log "Total runtime: $SECONDS seconds"
log "========================================================================"

# Create symlink to latest log
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/dataset_segment_pairs_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE
