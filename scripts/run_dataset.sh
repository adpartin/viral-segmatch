#!/bin/bash
# Generic Dataset Creation Script
# Usage: ./scripts/run_dataset.sh <config_bundle> [--input_file INPUT] [--output_dir OUTPUT]
# Example: ./scripts/run_dataset.sh flu_a_learning_test
#          ./scripts/run_dataset.sh flu_a_overfit_test --input_file /path/to/input.csv

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
    echo "Usage: $0 <config_bundle> [--input_file INPUT] [--output_dir OUTPUT]"
    exit 1
fi

shift  # Remove config_bundle from arguments

# Parse optional arguments
INPUT_FILE=""
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "Usage: $0 <config_bundle> [--input_file INPUT] [--output_dir OUTPUT]"
            exit 1
            ;;
    esac
done

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="dataset_${CONFIG_BUNDLE}_${TIMESTAMP}"
LOG_DIR="$PROJECT_ROOT/logs/datasets"
LOG_FILE="$LOG_DIR/dataset_segment_pairs_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Print header
log "========================================================================"
log "Dataset Segment Pairs Creation"
log "========================================================================"
log "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
log "Config bundle: $CONFIG_BUNDLE"
log "Host:          $(hostname)"
log "User:          $(whoami)"
log "Python:        $(which python)"
log "Log file:      $LOG_FILE"
log "Run ID:        $RUN_ID"
log ""
if [ -n "$INPUT_FILE" ]; then
    log "Input file override: $INPUT_FILE"
fi
if [ -n "$OUTPUT_DIR" ]; then
    log "Output dir override: $OUTPUT_DIR"
fi
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
CMD="python $PROJECT_ROOT/src/datasets/dataset_segment_pairs.py --config_bundle $CONFIG_BUNDLE"

if [ -n "$INPUT_FILE" ]; then
    CMD="$CMD --input_file $INPUT_FILE"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
else
    CMD="$CMD --run_output_subdir $RUN_ID"
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

# Register experiment in central registry
log ""
log "Registering experiment in central registry..."
REG_LOG_FILE="${LOG_FILE#$PROJECT_ROOT/}"
OUTPUT_DIR_REL=""
OUTPUT_DIR_LINE=$(grep -m1 'output_dir:' "$LOG_FILE" 2>/dev/null || true)
if [ -n "$OUTPUT_DIR_LINE" ]; then
    OUTPUT_DIR_ABS=$(echo "$OUTPUT_DIR_LINE" | sed -E 's/.*output_dir:\s*//')
    OUTPUT_DIR_ABS=$(echo "$OUTPUT_DIR_ABS" | xargs)
    if [[ "$OUTPUT_DIR_ABS" == $PROJECT_ROOT/* ]]; then
        OUTPUT_DIR_REL="${OUTPUT_DIR_ABS#$PROJECT_ROOT/}"
    else
        OUTPUT_DIR_REL="$OUTPUT_DIR_ABS"
    fi
fi
REGISTER_CMD=(python "$PROJECT_ROOT/src/utils/experiment_registry.py" --register
    --config_bundle "$CONFIG_BUNDLE"
    --stage "dataset"
    --command "$CMD"
    --exit_code "$EXIT_CODE"
    --log_file "$REG_LOG_FILE")
if [ -n "$OUTPUT_DIR_REL" ]; then
    REGISTER_CMD+=(--output_dir "$OUTPUT_DIR_REL")
fi
"${REGISTER_CMD[@]}" 2>/dev/null || log "⚠️  Failed to register experiment (non-critical)"

exit $EXIT_CODE

