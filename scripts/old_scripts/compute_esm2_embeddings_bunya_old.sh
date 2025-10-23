#!/bin/bash
# Compute ESM-2 embeddings for Bunyavirales using the old script

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Setup
PROJECT_ROOT="/nfs/lambda_stor_01/data/apartin/projects/cepi/viral-segmatch"
cd "$PROJECT_ROOT"

# Configuration
VIRUS_NAME="bunya"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs/embeddings"
LOG_FILE="${LOG_DIR}/compute_esm2_${VIRUS_NAME}_old_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Capture metadata
echo "========================================" | tee "$LOG_FILE"
echo "ESM-2 Embeddings Computation (OLD): $VIRUS_NAME" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Host: $(hostname)" | tee -a "$LOG_FILE"
echo "User: $(whoami)" | tee -a "$LOG_FILE"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')" | tee -a "$LOG_FILE"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')" | tee -a "$LOG_FILE"
echo "Python: $(which python)" | tee -a "$LOG_FILE"
echo "Script: compute_esm2_embeddings_old.py" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Run ESM-2 embeddings computation using the old script
echo "" | tee -a "$LOG_FILE" # New line
python "${PROJECT_ROOT}/src/embeddings/compute_esm2_embeddings_old.py" 2>&1 | tee -a "$LOG_FILE"

# Capture exit status
EXIT_CODE=${PIPESTATUS[0]}

# Summary
echo "========================================" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "ESM-2 embeddings computation (old script) completed successfully!" | tee -a "$LOG_FILE"
else
    echo "ESM-2 embeddings computation (old script) failed!" | tee -a "$LOG_FILE"
fi

echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Create symlink to latest log
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/compute_esm2_${VIRUS_NAME}_old_latest.log"

exit $EXIT_CODE
