#!/bin/bash
# Stage 1: Preprocess Flu protein + genome data from GTO files.
# Usage: ./scripts/stage1_preprocess_flu.sh <config_bundle> [options]
# Example: ./scripts/stage1_preprocess_flu.sh flu
#          ./scripts/stage1_preprocess_flu.sh flu --max_files 100 --force_reprocess

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Parse arguments ---
CONFIG_BUNDLE="${1:-}"
if [ -z "$CONFIG_BUNDLE" ]; then
    echo "Usage: $0 <config_bundle> [--max_files N] [--force_reprocess]"
    exit 1
fi
shift

MAX_FILES=""
FORCE_REPROCESS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_files)      MAX_FILES="$2"; shift 2 ;;
        --force_reprocess) FORCE_REPROCESS="--force-reprocess"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Logging ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/preprocessing"
LOG_FILE="$LOG_DIR/preprocess_flu_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# --- Build command ---
CMD="python src/preprocess/preprocess_flu.py --config_bundle $CONFIG_BUNDLE"
[ -n "$MAX_FILES" ]       && CMD="$CMD --max_files_to_preprocess $MAX_FILES"
[ -n "$FORCE_REPROCESS" ] && CMD="$CMD $FORCE_REPROCESS"

# --- Run ---
echo "Config:  $CONFIG_BUNDLE"
echo "Command: $CMD"
echo "Log:     $LOG_FILE"
echo ""

$CMD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# --- Symlink latest log ---
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/preprocess_flu_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE
