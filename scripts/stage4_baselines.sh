#!/bin/bash
# Stage 4 (baseline): sklearn-style baseline runner alongside the MLP path.
# Usage: ./scripts/stage4_baselines.sh <config_bundle> --baseline NAME --dataset_dir DIR [options]
#
# Examples:
#   ./scripts/stage4_baselines.sh flu_ha_na --baseline logistic --dataset_dir data/datasets/.../runs/dataset_...
#
# Output: models/{virus}/{data_version}/runs/baseline_<name>_<bundle>_<TS>/
# Sibling to MLP runs in the same runs/ dir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Parse arguments ---
CONFIG_BUNDLE="${1:-}"
if [ -z "$CONFIG_BUNDLE" ]; then
    echo "Usage: $0 <config_bundle> --baseline NAME --dataset_dir DIR [--output_dir DIR]"
    exit 1
fi
shift

BASELINE=""
DATASET_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)    BASELINE="$2";    shift 2 ;;
        --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$BASELINE" ]; then
    echo "Error: --baseline is required (e.g., --baseline logistic)"
    exit 1
fi
if [ -z "$DATASET_DIR" ]; then
    echo "Error: --dataset_dir is required"
    exit 1
fi

# --- Logging ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="baseline_${BASELINE}_${CONFIG_BUNDLE}_${TIMESTAMP}"
LOG_DIR="$PROJECT_ROOT/logs/training"
LOG_FILE="$LOG_DIR/train_pair_baseline_${BASELINE}_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# --- Build command ---
CMD="python src/models/train_pair_baselines.py"
CMD="$CMD --config_bundle $CONFIG_BUNDLE --baseline $BASELINE --dataset_dir $DATASET_DIR"
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
else
    CMD="$CMD --run_output_subdir $RUN_ID"
fi

# --- Run ---
echo "Config:      $CONFIG_BUNDLE"
echo "Baseline:    $BASELINE"
echo "Dataset dir: $DATASET_DIR"
echo "Run ID:      $RUN_ID"
echo "Command:     $CMD"
echo "Log:         $LOG_FILE"
echo ""

$CMD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# --- Co-locate log with run artifacts ---
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
    if [ -n "$ACTUAL_OUTPUT_DIR" ] && [ -d "$ACTUAL_OUTPUT_DIR" ]; then
        cp "$LOG_FILE" "$ACTUAL_OUTPUT_DIR/stage4_baseline.log"
    fi
fi

# --- Symlink latest log ---
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/train_pair_baseline_${BASELINE}_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE
