#!/bin/bash
# Stage 2b: Compute k-mer features for genome segments (CPU-only).
# Usage: ./scripts/stage2b_kmer.sh <config_bundle> [options]
# Example: ./scripts/stage2b_kmer.sh flu
#          ./scripts/stage2b_kmer.sh flu --force_recompute
#          ./scripts/stage2b_kmer.sh flu --input_file data/processed/flu/July_2025/genome_final.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Parse arguments ---
CONFIG_BUNDLE="${1:-}"
if [ -z "$CONFIG_BUNDLE" ]; then
    echo "Usage: $0 <config_bundle> [--force_recompute] [--input_file FILE] [--output_dir DIR]"
    exit 1
fi
shift

FORCE_RECOMPUTE=""
INPUT_FILE=""
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --force_recompute) FORCE_RECOMPUTE="--force-recompute"; shift ;;
        --input_file)      INPUT_FILE="$2"; shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Logging ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs/embeddings"
LOG_FILE="$LOG_DIR/compute_kmer_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# --- Build command ---
CMD="python src/embeddings/compute_kmer_features.py --config_bundle $CONFIG_BUNDLE"
[ -n "$FORCE_RECOMPUTE" ] && CMD="$CMD $FORCE_RECOMPUTE"
[ -n "$INPUT_FILE" ]      && CMD="$CMD --input_file $INPUT_FILE"
[ -n "$OUTPUT_DIR" ]      && CMD="$CMD --output_dir $OUTPUT_DIR"

# --- Run ---
echo "Config:  $CONFIG_BUNDLE"
echo "Command: $CMD"
echo "Log:     $LOG_FILE"
echo ""

$CMD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# --- Symlink latest log ---
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/compute_kmer_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE
