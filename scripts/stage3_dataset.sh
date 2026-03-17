#!/bin/bash
# Stage 3: Dataset Segment Pairs Creation
# Usage: ./scripts/stage3_dataset.sh <config_bundle> [options]
# 
# Examples:
#   ./scripts/stage3_dataset.sh flu_schema_raw_slot_norm_unit_diff
#   ./scripts/stage3_dataset.sh flu --input_file data/processed/flu/July_2025/protein_final.csv
#   ./scripts/stage3_dataset.sh flu --output_dir /custom/path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Parse arguments ---
CONFIG_BUNDLE="${1:-}"
if [ -z "$CONFIG_BUNDLE" ]; then
    echo "Usage: $0 <config_bundle> [--input_file FILE] [--output_dir DIR]"
    exit 1
fi
shift

INPUT_FILE=""
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)  INPUT_FILE="$2"; shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Logging ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="dataset_${CONFIG_BUNDLE}_${TIMESTAMP}"
LOG_DIR="$PROJECT_ROOT/logs/datasets"
LOG_FILE="$LOG_DIR/dataset_segment_pairs_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# --- Build command ---
CMD="python src/datasets/dataset_segment_pairs.py --config_bundle $CONFIG_BUNDLE"
[ -n "$INPUT_FILE" ] && CMD="$CMD --input_file $INPUT_FILE"
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
else
    CMD="$CMD --run_output_subdir $RUN_ID"
fi

# --- Run ---
echo "Config:  $CONFIG_BUNDLE"
echo "Run ID:  $RUN_ID"
echo "Command: $CMD"
echo "Log:     $LOG_FILE"
echo ""

$CMD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# --- Copy log into the dataset output dir for co-location with artifacts ---
if [ $EXIT_CODE -eq 0 ]; then
    ACTUAL_OUTPUT_DIR=""
    if [ -n "$OUTPUT_DIR" ]; then
        ACTUAL_OUTPUT_DIR="$OUTPUT_DIR"
    else
        OUTPUT_DIR_LINE=$(grep -m1 'Output dir:' "$LOG_FILE" 2>/dev/null || true)
        if [ -n "$OUTPUT_DIR_LINE" ]; then
            ACTUAL_OUTPUT_DIR=$(echo "$OUTPUT_DIR_LINE" | sed -E 's/.*Output dir:\s*//' | xargs)
        fi
    fi
    # Copy log into dataset output dir if it exists
    if [ -n "$ACTUAL_OUTPUT_DIR" ] && [ -d "$ACTUAL_OUTPUT_DIR" ]; then
        cp "$LOG_FILE" "$ACTUAL_OUTPUT_DIR/stage3_dataset.log"
    fi
fi

# --- Symlink latest log ---
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/dataset_segment_pairs_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE
