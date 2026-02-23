#!/bin/bash
# Stage 4: Model Training
# Usage: ./scripts/stage4_train.sh [<config_bundle>] [--cuda_name CUDA] [--dataset_dir DIR] [--embeddings_file FILE] [--output_dir DIR] [--allow_bundle_mismatch]
# 
# Config bundle can be provided via:
#   1. CLI argument: <config_bundle>
#   2. Extracted from --dataset_dir (if directory name follows pattern: dataset_{config_bundle}_{timestamp})
#
# If both are provided, they must match (validation), unless --allow_bundle_mismatch is set.
# If only --dataset_dir is provided, config bundle is extracted automatically.
#
# Examples:
#   ./scripts/stage4_train.sh bunya --cuda_name cuda:7
#   ./scripts/stage4_train.sh flu_overfit --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_overfit_...
#   ./scripts/stage4_train.sh --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_overfit_...  # Config extracted from dataset dir

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Get project root (auto-detect from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
# Config bundle is optional if --dataset_dir is provided (will be extracted from dataset dir)
CONFIG_BUNDLE_CLI="${1:-}"

# Check if first argument looks like an option (starts with --)
if [[ "${1:-}" == --* ]]; then
    # First arg is an option, so no CLI config bundle provided
    CONFIG_BUNDLE_CLI=""
else
    # First arg is config bundle
    shift  # Remove config_bundle from arguments
fi

# Parse optional arguments
CUDA_NAME="cuda:0"  # Default CUDA device
DATASET_DIR=""
EMBEDDINGS_FILE=""
OUTPUT_DIR=""
SKIP_POSTPROCESSING=false
ALLOW_BUNDLE_MISMATCH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda_name)
            CUDA_NAME="$2"
            shift 2
            ;;
        --dataset_dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --embeddings_file)
            EMBEDDINGS_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_postprocessing)
            SKIP_POSTPROCESSING=true
            shift
            ;;
        --allow_bundle_mismatch)
            ALLOW_BUNDLE_MISMATCH=true
            shift
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Usage: $0 [<config_bundle>] [--cuda_name CUDA] [--dataset_dir DIR] [--embeddings_file FILE] [--output_dir DIR] [--skip_postprocessing] [--allow_bundle_mismatch]"
            echo ""
            echo "Examples:"
            echo "  $0 bunya --cuda_name cuda:7"
            echo "  $0 flu_overfit --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_overfit_..."
            echo "  $0 --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_overfit_...  # Config bundle extracted from dataset dir"
            echo "  $0 flu_schema_raw_shared --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_schema_... --allow_bundle_mismatch"
            exit 1
            ;;
    esac
done

# Extract config bundle from dataset directory if provided (dataset dir is source of truth)
EXTRACTED_BUNDLE=""
if [ -n "$DATASET_DIR" ]; then
    # Get basename of dataset directory (e.g., "dataset_flu_overfit_20251202_140140")
    DATASET_BASENAME=$(basename "$DATASET_DIR")
    
    # Pattern: dataset_{config_bundle}_{timestamp}
    # Extract config bundle by removing "dataset_" prefix and timestamp suffix (YYYYMMDD_HHMMSS)
    if [[ "$DATASET_BASENAME" == dataset_*_????????_?????? ]]; then
        # Remove "dataset_" prefix
        EXTRACTED_BUNDLE="${DATASET_BASENAME#dataset_}"
        # Remove timestamp suffix (pattern: _YYYYMMDD_HHMMSS)
        EXTRACTED_BUNDLE="${EXTRACTED_BUNDLE%_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]}"
        
        if [ -z "$EXTRACTED_BUNDLE" ]; then
            echo "❌ Error: Could not extract config bundle from dataset directory name: $DATASET_BASENAME"
            echo "   Expected pattern: dataset_{config_bundle}_{timestamp}"
            exit 1
        fi
    else
        echo "❌ Error: Dataset directory name doesn't match expected pattern: dataset_{config_bundle}_{timestamp}"
        echo "   Found: $DATASET_BASENAME"
        exit 1
    fi
fi

# Determine final config bundle with validation
if [ -n "$EXTRACTED_BUNDLE" ] && [ -n "$CONFIG_BUNDLE_CLI" ]; then
    # Both provided: validate they match
    if [ "$EXTRACTED_BUNDLE" != "$CONFIG_BUNDLE_CLI" ]; then
        if [ "$ALLOW_BUNDLE_MISMATCH" = true ]; then
            CONFIG_BUNDLE="$CONFIG_BUNDLE_CLI"
            echo "⚠️  Bundle mismatch allowed (--allow_bundle_mismatch)"
            echo "   CLI bundle:        '$CONFIG_BUNDLE_CLI'"
            echo "   Dataset bundle:    '$EXTRACTED_BUNDLE' (extracted from: $DATASET_DIR)"
        else
            echo "❌ Error: Config bundle mismatch!"
            echo "   CLI bundle:        '$CONFIG_BUNDLE_CLI'"
            echo "   Dataset bundle:    '$EXTRACTED_BUNDLE' (extracted from: $DATASET_DIR)"
            echo ""
            echo "   The dataset was created with config bundle '$EXTRACTED_BUNDLE'."
            echo "   Please use the same config bundle for training, or omit the CLI bundle to use the extracted one."
            echo ""
            echo "   Correct usage:"
            echo "     $0 $EXTRACTED_BUNDLE --dataset_dir $DATASET_DIR"
            echo "   Or:"
            echo "     $0 --dataset_dir $DATASET_DIR"
            echo "   Or (override):"
            echo "     $0 $CONFIG_BUNDLE_CLI --dataset_dir $DATASET_DIR --allow_bundle_mismatch"
            exit 1
        fi
    fi
    # They match - use extracted (which equals CLI)
    if [ -z "${CONFIG_BUNDLE:-}" ]; then
        CONFIG_BUNDLE="$EXTRACTED_BUNDLE"
    fi
    echo "✅ Config bundle validated: '$CONFIG_BUNDLE' (matches dataset directory)"
elif [ -n "$EXTRACTED_BUNDLE" ]; then
    # Only dataset dir provided: use extracted bundle
    CONFIG_BUNDLE="$EXTRACTED_BUNDLE"
    echo "📦 Using config bundle extracted from dataset directory: '$CONFIG_BUNDLE'"
elif [ -n "$CONFIG_BUNDLE_CLI" ]; then
    # Only CLI provided: use CLI bundle
    CONFIG_BUNDLE="$CONFIG_BUNDLE_CLI"
else
    # Neither provided: error
    echo "❌ Error: Config bundle required"
    echo "   Provide either:"
    echo "     1. CLI argument: $0 <config_bundle> [options...]"
    echo "     2. --dataset_dir with extractable config bundle in directory name"
    echo ""
    echo "   Examples:"
    echo "     $0 bunya --cuda_name cuda:7"
    echo "     $0 --dataset_dir data/datasets/flu/July_2025/runs/dataset_flu_overfit_..."
    exit 1
fi
# CONFIG_BUNDLE is now the final validated value

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="training_${CONFIG_BUNDLE}_${TIMESTAMP}"
LOG_DIR="$PROJECT_ROOT/logs/training"
LOG_FILE="$LOG_DIR/train_esm2_frozen_pair_classifier_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Helper function for logging to both console and file
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Print header
log "========================================================================"
log "Stage 4: ESM-2 Frozen Pair Classifier Training"
log "========================================================================"
log "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
log "Config bundle: $CONFIG_BUNDLE"
log "Host:          $(hostname)"
log "User:          $(whoami)"
log "Python:        $(which python)"
log "Log file:      $LOG_FILE"
log "Run ID:        $RUN_ID"
log ""
log "Configuration:"
log "  CUDA device:      $CUDA_NAME"
if [ -n "$DATASET_DIR" ]; then
    log "  Dataset dir override: $DATASET_DIR"
fi
if [ -n "$EMBEDDINGS_FILE" ]; then
    log "  Embeddings file override: $EMBEDDINGS_FILE"
fi
if [ -n "$OUTPUT_DIR" ]; then
    log "  Output dir override: $OUTPUT_DIR"
fi
if [ "$SKIP_POSTPROCESSING" = true ]; then
    log "  Postprocessing: SKIPPED"
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
CMD="python $PROJECT_ROOT/src/models/train_esm2_frozen_pair_classifier.py --config_bundle $CONFIG_BUNDLE --cuda_name $CUDA_NAME"

if [ -n "$DATASET_DIR" ]; then
    CMD="$CMD --dataset_dir $DATASET_DIR"
fi

if [ -n "$EMBEDDINGS_FILE" ]; then
    CMD="$CMD --embeddings_file $EMBEDDINGS_FILE"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
else
    CMD="$CMD --run_output_subdir $RUN_ID"
fi

# Run the training script
log "Starting ESM-2 frozen pair classifier training with config bundle: $CONFIG_BUNDLE"
log "  Run ID: $RUN_ID"
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
    
    # Run postprocessing if training succeeded and not skipped
    if [ "$SKIP_POSTPROCESSING" = false ]; then
        log ""
        log "========================================================================"
        log "POSTPROCESSING: Segment Classifier Results Analysis"
        log "========================================================================"
        
        POSTPROC_EXIT_CODE=0
        
        # Determine output directory (use override or extract from training log)
        if [ -n "$OUTPUT_DIR" ]; then
            ACTUAL_OUTPUT_DIR="$OUTPUT_DIR"
        else
            # Extract output_dir from training log (training script prints "Output dir:       /path/...")
            # Try multiple patterns to be robust (handles both "Output dir:" and "output_dir:")
            OUTPUT_DIR_LINE=$(grep -m1 -i 'Output.*dir:' "$LOG_FILE" 2>/dev/null || true)
            if [ -n "$OUTPUT_DIR_LINE" ]; then
                # Extract path after the colon (handles both "Output dir:" and "output_dir:")
                ACTUAL_OUTPUT_DIR=$(echo "$OUTPUT_DIR_LINE" | sed -E 's/.*:\s*//' | xargs)
                log "Detected output directory from training: $ACTUAL_OUTPUT_DIR"
            else
                # Fallback: try to find the most recent training run directory
                VIRUS_NAME=$(grep -m1 "Virus:" "$LOG_FILE" 2>/dev/null | sed -E 's/.*Virus:\s*//' | xargs || echo "")
                DATA_VERSION=$(grep -m1 "Data version:" "$LOG_FILE" 2>/dev/null | sed -E 's/.*Data version:\s*//' | xargs || echo "")
                if [ -n "$VIRUS_NAME" ] && [ -n "$DATA_VERSION" ]; then
                    # Look for most recent training run
                    RUNS_DIR="$PROJECT_ROOT/models/$VIRUS_NAME/$DATA_VERSION/runs"
                    if [ -d "$RUNS_DIR" ]; then
                        ACTUAL_OUTPUT_DIR=$(ls -td "$RUNS_DIR"/training_* 2>/dev/null | head -1)
                        if [ -n "$ACTUAL_OUTPUT_DIR" ]; then
                            log "Found most recent training run: $ACTUAL_OUTPUT_DIR"
                        fi
                    fi
                fi
                if [ -z "$ACTUAL_OUTPUT_DIR" ]; then
                    log "⚠️  Could not detect output directory from training log"
                fi
            fi
        fi
        
        # 1. Run analyze_stage4_train.py
        log "Running analyze_stage4_train.py..."
        POSTPROC_CMD="python $PROJECT_ROOT/src/analysis/analyze_stage4_train.py --config_bundle $CONFIG_BUNDLE"
        if [ -n "$ACTUAL_OUTPUT_DIR" ]; then
            POSTPROC_CMD="$POSTPROC_CMD --model_dir $ACTUAL_OUTPUT_DIR"
        fi
        log "Command: $POSTPROC_CMD"
        log ""
        
        set +e  # Temporarily disable exit on error
        eval "$POSTPROC_CMD" 2>&1 | tee -a "$LOG_FILE"
        SEGMENT_EXIT_CODE=${PIPESTATUS[0]}
        set -e  # Re-enable exit on error

        if [ $SEGMENT_EXIT_CODE -eq 0 ]; then
            log "✅ Training analysis completed successfully!"
        else
            log "❌ Training analysis failed with exit code: $SEGMENT_EXIT_CODE"
            POSTPROC_EXIT_CODE=$SEGMENT_EXIT_CODE
        fi
        log ""

        # 2. Run create_presentation_plots.py
        log "Running create_presentation_plots.py..."
        PRESENTATION_CMD="python $PROJECT_ROOT/src/analysis/create_presentation_plots.py --config_bundle $CONFIG_BUNDLE"
        if [ -n "$ACTUAL_OUTPUT_DIR" ]; then
            PRESENTATION_CMD="$PRESENTATION_CMD --model_dir $ACTUAL_OUTPUT_DIR"
        fi
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
    fi
else
    log "❌ ESM-2 frozen pair classifier training failed with exit code: $EXIT_CODE"
    log "Skipping postprocessing due to training failure."
fi
log ""
log "========================================================================"
log "Training and Postprocessing Summary"
log "========================================================================"
log "Training exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ] && [ "$SKIP_POSTPROCESSING" = false ]; then
    log "Postprocessing exit code: $POSTPROC_EXIT_CODE"
fi
log "End time: $(date)"
log "Total runtime: $SECONDS seconds"
log "========================================================================"

# Create symlink to latest log
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/train_esm2_frozen_pair_classifier_${CONFIG_BUNDLE}_latest.log"

# Register experiment in central registry
log ""
log "Registering experiment in central registry..."
REG_LOG_FILE="${LOG_FILE#$PROJECT_ROOT/}"
OUTPUT_DIR_REL=""
# Use same pattern as postprocessing extraction
OUTPUT_DIR_LINE=$(grep -m1 -i 'Output.*dir:' "$LOG_FILE" 2>/dev/null || true)
if [ -n "$OUTPUT_DIR_LINE" ]; then
    OUTPUT_DIR_ABS=$(echo "$OUTPUT_DIR_LINE" | sed -E 's/.*:\s*//' | xargs)
    if [[ "$OUTPUT_DIR_ABS" == $PROJECT_ROOT/* ]]; then
        OUTPUT_DIR_REL="${OUTPUT_DIR_ABS#$PROJECT_ROOT/}"
    else
        OUTPUT_DIR_REL="$OUTPUT_DIR_ABS"
    fi
fi
REGISTER_CMD=(python "$PROJECT_ROOT/src/utils/experiment_registry.py" --register
    --config_bundle "$CONFIG_BUNDLE"
    --stage "training"
    --command "$CMD"
    --exit_code "$EXIT_CODE"
    --log_file "$REG_LOG_FILE")
if [ -n "$OUTPUT_DIR_REL" ]; then
    REGISTER_CMD+=(--output_dir "$OUTPUT_DIR_REL")
fi
"${REGISTER_CMD[@]}" 2>/dev/null || log "⚠️  Failed to register experiment (non-critical)"

# Exit with training exit code (postprocessing errors don't fail the script)
exit $EXIT_CODE

