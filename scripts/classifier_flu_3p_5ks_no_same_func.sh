#!/bin/bash
# ESM-2 Frozen Pair Classifier Training for Flu A 3p_5ks_no_same_func
# Usage: ./scripts/classifier_flu_3p_5ks_no_same_func.sh
# 3p_5ks: 3 major proteins (PB1, PB2, PA) and 5k isolates sampled from the full dataset.
# No same-function negatives: allow_same_func_negatives=false

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Get project root (auto-detect from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
CONFIG_BUNDLE="flu_a_3p_5ks_no_same_func"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/training"
LOG_FILE="$LOG_DIR/train_esm2_frozen_pair_classifier_${CONFIG_BUNDLE}_${TIMESTAMP}.log"

# Path overrides - use 3p_5ks directories
DATASET_DIR="$PROJECT_ROOT/data/datasets/flu_a/July_2025_seed_42_isolates_5000"
EMBEDDINGS_DIR="$PROJECT_ROOT/data/embeddings/flu_a/July_2025_seed_42_isolates_5000"
OUTPUT_DIR="$PROJECT_ROOT/models/flu_a/July_2025_seed_42_isolates_5000"

CUDA_NAME="cuda:5"  # CUDA device

# Create log directory
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Header
log "========================================================================"
log "ESM-2 Frozen Pair Classifier Training (Flu A 3p_5ks_no_same_func)"
log "========================================================================"
log "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
log "Config bundle: $CONFIG_BUNDLE"
log "Host:          $(hostname)"
log "User:          $(whoami)"
log "Python:        $(which python)"
log "Log file:      $LOG_FILE"
log ""
log "Overrides:"
log "  CUDA device:    $CUDA_NAME"
log "  Dataset Dir:    $DATASET_DIR"
log "  Embeddings Dir: $EMBEDDINGS_DIR"
log "  Output Dir:     $OUTPUT_DIR"
log ""

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
CMD="python $PROJECT_ROOT/src/models/train_esm2_frozen_pair_classifier.py --config_bundle $CONFIG_BUNDLE --cuda_name $CUDA_NAME"

if [ -n "$DATASET_DIR" ]; then
    CMD="$CMD --dataset_dir $DATASET_DIR"
fi

if [ -n "$EMBEDDINGS_DIR" ]; then
    CMD="$CMD --embeddings_dir $EMBEDDINGS_DIR"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Run the training script
log "Starting ESM-2 frozen pair classifier training with config bundle: $CONFIG_BUNDLE"
log "Command: $CMD"
log ""

set +u  # Temporarily disable exit on undefined variable
set +e  # Temporarily disable exit on error
eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e  # Re-enable exit on error
set -u  # Re-enable exit on undefined variable

# Footer
log ""
log "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    log "✅ ESM-2 frozen pair classifier training completed successfully!"
    
    # Run postprocessing if training succeeded
    log ""
    log "========================================================================"
    log "POSTPROCESSING: Segment Classifier Results Analysis"
    log "========================================================================"
    
    POSTPROC_EXIT_CODE=0
    
    # 1. Run segment_classifier_results.py
    log "Running segment_classifier_results.py..."
    POSTPROC_CMD="python $PROJECT_ROOT/src/postprocess/segment_classifier_results.py --config_bundle $CONFIG_BUNDLE --model_dir $OUTPUT_DIR"
    log "Command: $POSTPROC_CMD"
    log ""
    
    set +e  # Temporarily disable exit on error
    eval "$POSTPROC_CMD" 2>&1 | tee -a "$LOG_FILE"
    SEGMENT_EXIT_CODE=${PIPESTATUS[0]}
    set -e  # Re-enable exit on error
    
    if [ $SEGMENT_EXIT_CODE -eq 0 ]; then
        log "✅ Segment classifier results analysis completed successfully!"
    else
        log "❌ Segment classifier results analysis failed with exit code: $SEGMENT_EXIT_CODE"
        POSTPROC_EXIT_CODE=$SEGMENT_EXIT_CODE
    fi
    log ""
    
    # 2. Run presentation_plots.py
    log "Running presentation_plots.py..."
    PRESENTATION_CMD="python $PROJECT_ROOT/src/postprocess/presentation_plots.py --config_bundle $CONFIG_BUNDLE --model_dir $OUTPUT_DIR"
    log "Command: $PRESENTATION_CMD"
    log ""
    
    set +e  # Temporarily disable exit on error
    eval "$PRESENTATION_CMD" 2>&1 | tee -a "$LOG_FILE"
    PRESENTATION_EXIT_CODE=${PIPESTATUS[0]}
    set -e  # Re-enable exit on error
    
    if [ $PRESENTATION_EXIT_CODE -eq 0 ]; then
        log "✅ Presentation plots generation completed successfully!"
    else
        log "❌ Presentation plots generation failed with exit code: $PRESENTATION_EXIT_CODE"
        POSTPROC_EXIT_CODE=$PRESENTATION_EXIT_CODE
    fi
    
    log ""
    log "========================================================================"
    if [ $POSTPROC_EXIT_CODE -eq 0 ]; then
        log "✅ All postprocessing steps completed successfully!"
    else
        log "⚠️  Postprocessing completed with errors (exit code: $POSTPROC_EXIT_CODE)"
    fi
    log "========================================================================"
else
    log "❌ ESM-2 frozen pair classifier training failed with exit code: $EXIT_CODE"
    log "Skipping postprocessing due to training failure."
fi
log ""
log "========================================================================"
log "Training and Postprocessing Summary"
log "========================================================================"
log "Training exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    log "Postprocessing exit code: $POSTPROC_EXIT_CODE"
fi
log "End time: $(date)"
log "Total runtime: $SECONDS seconds"
log "========================================================================"

# Create symlink to latest log
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/train_esm2_frozen_pair_classifier_${CONFIG_BUNDLE}_latest.log"

# Exit with training exit code (postprocessing errors don't fail the script)
exit $EXIT_CODE

