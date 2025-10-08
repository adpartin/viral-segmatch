#!/bin/bash
# Preprocess Flu A protein data from GTO files

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Setup
PROJECT_ROOT="/nfs/lambda_stor_01/data/apartin/projects/cepi/viral-segmatch"
cd "$PROJECT_ROOT"

VIRUS_NAME="flu_a"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs/preprocess"
LOG_FILE="${LOG_DIR}/preprocess_${VIRUS_NAME}_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Capture metadata
echo "========================================" | tee "$LOG_FILE"
echo "Preprocessing: $VIRUS_NAME" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Host: $(hostname)" | tee -a "$LOG_FILE"
echo "User: $(whoami)" | tee -a "$LOG_FILE"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')" | tee -a "$LOG_FILE"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')" | tee -a "$LOG_FILE"
echo "Python: $(which python)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Run preprocessing
python src/preprocess/preprocess_flu_protein.py --virus_name "$VIRUS_NAME" 2>&1 | tee -a "$LOG_FILE"

# Capture exit status
EXIT_CODE=${PIPESTATUS[0]}

# Summary
echo "========================================" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Create symlink to latest log
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/preprocess_${VIRUS_NAME}_latest.log"

exit $EXIT_CODE
