#!/bin/bash
# Stage 3 (CC builder): 2D-CD cluster-disjoint connected-component K-fold pair datasets.
#
# Wraps the standalone CC builder src/datasets/dataset_pairs_cc.py — NOT the v2 Stage-3 CLI
# (dataset_segment_pairs.py), which rejects mode=cluster_disjoint_cc. Use this for bundles
# with split_strategy.mode=cluster_disjoint_cc (e.g. flu_ha_na_cc_aa, flu_ha_na_cc_nt_cds,
# flu_ha_na_cc_nt_ctg). Run inside the segmatch conda env.
#
# Unlike the v2 CLI, the CC builder takes a full --out_dir (no --run_output_subdir), so this
# wrapper builds the timestamped run dir itself: data/datasets/flu/<version>/runs/dataset_<bundle>_<ts>.
#
# Usage: ./scripts/stage3_cc.sh <config_bundle> [--out_dir DIR] [--data_version V]
#                               [--protein_final FILE] [--override k=v ...]
# Examples:
#   ./scripts/stage3_cc.sh flu_ha_na_cc_aa
#   ./scripts/stage3_cc.sh flu_ha_na_cc_nt_cds --override dataset.split_strategy.negative_scope=within_fold
#   ./scripts/stage3_cc.sh flu_ha_na_cc_aa --out_dir /custom/path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Parse arguments ---
CONFIG_BUNDLE="${1:-}"
if [ -z "$CONFIG_BUNDLE" ]; then
    echo "Usage: $0 <config_bundle> [--out_dir DIR] [--data_version V] [--protein_final FILE] [--override k=v ...]"
    exit 1
fi
shift

OUT_DIR=""
DATA_VERSION="July_2025"
PROTEIN_FINAL=""
OVERRIDES=()  # Hydra-style dotlist overrides; collected here and passed as --override <a> <b> ...
while [[ $# -gt 0 ]]; do
    case $1 in
        --out_dir)        OUT_DIR="$2"; shift 2 ;;
        --data_version)   DATA_VERSION="$2"; shift 2 ;;
        --protein_final)  PROTEIN_FINAL="$2"; shift 2 ;;
        --override)
            # Accept repeated --override key=val, or one --override key=val key=val ...
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do OVERRIDES+=("$1"); shift; done
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Default out_dir: data/datasets/flu/<version>/runs/dataset_<bundle>_<ts> ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="dataset_${CONFIG_BUNDLE}_${TIMESTAMP}"
[ -z "$OUT_DIR" ] && OUT_DIR="data/datasets/flu/${DATA_VERSION}/runs/${RUN_ID}"

# --- Logging ---
LOG_DIR="$PROJECT_ROOT/logs/datasets"
LOG_FILE="$LOG_DIR/dataset_pairs_cc_${CONFIG_BUNDLE}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# --- Build command ---
CMD="python src/datasets/dataset_pairs_cc.py --config_bundle $CONFIG_BUNDLE --out_dir $OUT_DIR"
[ -n "$PROTEIN_FINAL" ] && CMD="$CMD --protein_final $PROTEIN_FINAL"
if [ ${#OVERRIDES[@]} -gt 0 ]; then
    CMD="$CMD --override"
    for kv in "${OVERRIDES[@]}"; do CMD="$CMD $kv"; done
fi

# --- Run ---
echo "Config:   $CONFIG_BUNDLE"
echo "Out dir:  $OUT_DIR"
echo "Command:  $CMD"
echo "Log:      $LOG_FILE"
echo ""

$CMD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# --- Co-locate the log with the dataset artifacts (out_dir is known directly) ---
if [ $EXIT_CODE -eq 0 ] && [ -d "$OUT_DIR" ]; then
    cp "$LOG_FILE" "$OUT_DIR/stage3_cc.log"
fi

# --- Symlink latest log ---
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/dataset_pairs_cc_${CONFIG_BUNDLE}_latest.log"

exit $EXIT_CODE
