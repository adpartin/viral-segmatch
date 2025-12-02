#!/bin/bash
# Stage 2: ESM-2 Embeddings Computation
# Usage: ./scripts/stage2_esm2.sh <config_bundle> [--cuda_name CUDA] [--force_recompute]
# Example: ./scripts/stage2_esm2.sh bunya --cuda_name cuda:7
#          ./scripts/stage2_esm2.sh flu_a --cuda_name cuda:5
#          ./scripts/stage2_esm2.sh flu_a --cuda_name cuda:0 --force_recompute

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Get project root (auto-detect from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
CONFIG_BUNDLE="${1:-}"
if [ -z "$CONFIG_BUNDLE" ]; then
    echo "Error: Config bundle required"
    echo "Usage: $0 <config_bundle> [--cuda_name CUDA] [--force_recompute]"
    echo "Examples:"
    echo "  $0 bunya --cuda_name cuda:7"
    echo "  $0 flu_a --cuda_name cuda:5"
    exit 1
fi

shift  # Remove config_bundle from arguments

# Parse optional arguments
CUDA_NAME="cuda:0"  # Default CUDA device
FORCE_RECOMPUTE=""
INPUT_FILE=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda_name)
            CUDA_NAME="$2"
            shift 2
            ;;
        --force_recompute)
            FORCE_RECOMPUTE="--force-recompute"
            shift
            ;;
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 <config_bundle> [--cuda_name CUDA] [--force_recompute] [--input_file FILE] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/embeddings"
LOG_FILE="$LOG_DIR/compute_esm2_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Print header
log "========================================================================"
log "Stage 2: ESM-2 Embeddings Computation"
log "========================================================================"
log "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
log "Config bundle: $CONFIG_BUNDLE"
log "Host:          $(hostname)"
log "User:          $(whoami)"
log "Python:        $(which python)"
log "Log file:      $LOG_FILE"
log ""
log "Configuration:"
log "  CUDA device:     $CUDA_NAME"
log "  Force recompute: $([[ -n "$FORCE_RECOMPUTE" ]] && echo "Yes" || echo "No")"
log "  Input file:      ${INPUT_FILE:-<config default>}"
log "  Output dir:      ${OUTPUT_DIR:-<config default>}"
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

# Build command
CMD="python $PROJECT_ROOT/src/embeddings/compute_esm2_embeddings.py --config_bundle $CONFIG_BUNDLE --cuda_name $CUDA_NAME"

if [ -n "$FORCE_RECOMPUTE" ]; then
    CMD="$CMD $FORCE_RECOMPUTE"
fi

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

