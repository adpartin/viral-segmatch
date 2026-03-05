#!/bin/bash
# Stage 4: Model Training (decoupled from dataset creation)
# Usage: ./scripts/stage4_train.sh <config_bundle> --dataset_dir DIR [options]
#
# The config bundle controls training settings (training.*, embeddings.*).
# The dataset_dir points to a Stage 3 output — it can come from any bundle.
# Provenance is tracked via training_info.json saved by the Python script.
#
# Examples:
#   ./scripts/stage4_train.sh flu_schema_raw_slot_norm_unit_diff --dataset_dir data/datasets/flu/.../runs/dataset_...
#   ./scripts/stage4_train.sh flu_schema_raw_concat --dataset_dir data/datasets/flu/.../runs/dataset_... --cuda_name cuda:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Parse arguments ---
CONFIG_BUNDLE="${1:-}"
if [ -z "$CONFIG_BUNDLE" ]; then
    echo "Usage: $0 <config_bundle> --dataset_dir DIR [--cuda_name CUDA] [--embeddings_file FILE] [--output_dir DIR] [--skip_postprocessing]"
    exit 1
fi
shift

CUDA_NAME="cuda:0"
DATASET_DIR=""
EMBEDDINGS_FILE=""
OUTPUT_DIR=""
SKIP_POSTPROCESSING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda_name)           CUDA_NAME="$2"; shift 2 ;;
        --dataset_dir)         DATASET_DIR="$2"; shift 2 ;;
        --embeddings_file)     EMBEDDINGS_FILE="$2"; shift 2 ;;
        --output_dir)          OUTPUT_DIR="$2"; shift 2 ;;
        --skip_postprocessing) SKIP_POSTPROCESSING=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET_DIR" ]; then
    echo "Error: --dataset_dir is required"
    echo "Usage: $0 <config_bundle> --dataset_dir DIR [options]"
    exit 1
fi

# --- Logging ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="training_${CONFIG_BUNDLE}_${TIMESTAMP}"
LOG_DIR="$PROJECT_ROOT/logs/training"
LOG_FILE="$LOG_DIR/train_esm2_frozen_pair_classifier_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# --- Build command ---
CMD="python src/models/train_esm2_frozen_pair_classifier.py"
CMD="$CMD --config_bundle $CONFIG_BUNDLE --cuda_name $CUDA_NAME --dataset_dir $DATASET_DIR"
[ -n "$EMBEDDINGS_FILE" ] && CMD="$CMD --embeddings_file $EMBEDDINGS_FILE"
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
else
    CMD="$CMD --run_output_subdir $RUN_ID"
fi

# --- Run training ---
echo "Config:      $CONFIG_BUNDLE"
echo "Dataset dir: $DATASET_DIR"
echo "CUDA:        $CUDA_NAME"
echo "Run ID:      $RUN_ID"
echo "Command:     $CMD"
echo "Log:         $LOG_FILE"
echo ""

$CMD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# --- Postprocessing ---
if [ $EXIT_CODE -eq 0 ] && [ "$SKIP_POSTPROCESSING" = false ]; then
    # Extract output directory from training log
    ACTUAL_OUTPUT_DIR=""
    if [ -n "$OUTPUT_DIR" ]; then
        ACTUAL_OUTPUT_DIR="$OUTPUT_DIR"
    else
        OUTPUT_DIR_LINE=$(grep -m1 'Output dir:' "$LOG_FILE" 2>/dev/null || true)
        if [ -n "$OUTPUT_DIR_LINE" ]; then
            ACTUAL_OUTPUT_DIR=$(echo "$OUTPUT_DIR_LINE" | sed -E 's/.*Output dir:\s*//' | xargs)
        fi
    fi

    if [ -n "$ACTUAL_OUTPUT_DIR" ]; then
        echo ""
        echo "Running postprocessing..."

        python src/analysis/analyze_stage4_train.py \
            --config_bundle "$CONFIG_BUNDLE" --model_dir "$ACTUAL_OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE" || echo "WARNING: analyze_stage4_train.py failed"

        python src/analysis/create_presentation_plots.py \
            --config_bundle "$CONFIG_BUNDLE" --model_dir "$ACTUAL_OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE" || echo "WARNING: create_presentation_plots.py failed"
    else
        echo "WARNING: Could not detect output directory, skipping postprocessing"
    fi
fi

# --- Symlink latest log ---
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/train_esm2_frozen_pair_classifier_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE
