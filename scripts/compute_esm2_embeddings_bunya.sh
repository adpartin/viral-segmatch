#!/bin/bash
# Compute ESM-2 embeddings for Bunyavirales protein data

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Setup
PROJECT_ROOT="/nfs/lambda_stor_01/data/apartin/projects/cepi/viral-segmatch"
cd "$PROJECT_ROOT"

# Configuration
VIRUS_NAME="bunya"
DATA_VERSION="April_2025"
CUDA_NAME="cuda:6"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs/embeddings"
LOG_FILE="${LOG_DIR}/compute_esm2_${VIRUS_NAME}_${TIMESTAMP}.log"

# Input and output paths
INPUT_FILE="${PROJECT_ROOT}/data/processed/${VIRUS_NAME}/${DATA_VERSION}/protein_final.csv"
OUTPUT_DIR="${PROJECT_ROOT}/data/embeddings/${VIRUS_NAME}/${DATA_VERSION}_new"

# Create log directory
mkdir -p "$LOG_DIR"

# Capture metadata
echo "========================================" | tee "$LOG_FILE"
echo "ESM-2 Embeddings Computation: $VIRUS_NAME" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Host: $(hostname)" | tee -a "$LOG_FILE"
echo "User: $(whoami)" | tee -a "$LOG_FILE"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')" | tee -a "$LOG_FILE"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')" | tee -a "$LOG_FILE"
echo "Python: $(which python)" | tee -a "$LOG_FILE"
echo "CUDA device: $CUDA_NAME" | tee -a "$LOG_FILE"
echo "Input file: $INPUT_FILE" | tee -a "$LOG_FILE"
echo "Output dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

# Run ESM-2 embeddings computation
echo "" | tee -a "$LOG_FILE" # New line
python "${PROJECT_ROOT}/src/embeddings/compute_esm2_embeddings.py" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --virus_name "$VIRUS_NAME" \
    --use_selected_only \
    --cuda_name "$CUDA_NAME" 2>&1 | tee -a "$LOG_FILE"

# Capture exit status
EXIT_CODE=${PIPESTATUS[0]}

# Summary
echo "========================================" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "ESM-2 embeddings computation completed successfully!" | tee -a "$LOG_FILE"
    echo "Output saved to: $OUTPUT_DIR/esm2_embeddings.h5" | tee -a "$LOG_FILE"
    
    # Show file size
    if [ -f "$OUTPUT_DIR/esm2_embeddings.h5" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_DIR/esm2_embeddings.h5" | cut -f1)
        echo "File size: $FILE_SIZE" | tee -a "$LOG_FILE"
    fi
    
    # Show CSV file size if it exists
    if [ -f "$OUTPUT_DIR/esm2_embeddings.csv" ]; then
        CSV_SIZE=$(du -h "$OUTPUT_DIR/esm2_embeddings.csv" | cut -f1)
        echo "CSV file size: $CSV_SIZE" | tee -a "$LOG_FILE"
    fi
else
    echo "ESM-2 embeddings computation failed!" | tee -a "$LOG_FILE"
fi

echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Create symlink to latest log
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/compute_esm2_${VIRUS_NAME}_latest.log"

exit $EXIT_CODE
