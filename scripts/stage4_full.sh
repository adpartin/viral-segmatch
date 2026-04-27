#!/bin/bash
# Stage 4 (full): run the MLP trainer + every baseline listed under
# `baselines:` in the bundle config. Each runs against the same Stage 3
# dataset directory and writes its own flat run directory.
#
# Usage:
#   ./scripts/stage4_full.sh <config_bundle> --dataset_dir DIR [other args passed through to stage4_train.sh]
#
# Behavior:
# - Runs stage4_train.sh first (MLP). If it fails, the whole script exits.
# - Reads `config.baselines` from the bundle (a list of baseline names).
#   Empty/null -> only the MLP runs.
# - For each name, invokes stage4_baselines.sh with the same dataset_dir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG_BUNDLE="${1:-}"
if [ -z "$CONFIG_BUNDLE" ]; then
    echo "Usage: $0 <config_bundle> --dataset_dir DIR [other args passed through to stage4_train.sh]"
    exit 1
fi
shift
ARGS=("$@")

# Find --dataset_dir in the passthrough args (we need it for baseline calls too).
DATASET_DIR=""
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [ "${ARGS[$i]}" = "--dataset_dir" ]; then
        DATASET_DIR="${ARGS[$((i+1))]}"
        break
    fi
done
if [ -z "$DATASET_DIR" ]; then
    echo "Error: --dataset_dir is required"
    exit 1
fi

# --- Run MLP ---
echo "=== [stage4_full] MLP: ./scripts/stage4_train.sh $CONFIG_BUNDLE ${ARGS[*]} ==="
"$SCRIPT_DIR/stage4_train.sh" "$CONFIG_BUNDLE" "${ARGS[@]}"

# --- Read bundle.baselines ---
BASELINES=$(python -c "
import sys
sys.path.insert(0, '.')
from src.utils.config_hydra import get_virus_config_hydra
c = get_virus_config_hydra(sys.argv[1], config_path='./conf')
bls = list(getattr(c, 'baselines', None) or [])
print(' '.join(bls))
" "$CONFIG_BUNDLE")

if [ -z "$BASELINES" ]; then
    echo ""
    echo "=== [stage4_full] No baselines configured for bundle '$CONFIG_BUNDLE' (config.baselines is null/empty). Done. ==="
    exit 0
fi

# --- Run each baseline ---
echo ""
echo "=== [stage4_full] Baselines to run: $BASELINES ==="
for B in $BASELINES; do
    echo ""
    echo "=== [stage4_full] Baseline '$B': ./scripts/stage4_baselines.sh $CONFIG_BUNDLE --baseline $B --dataset_dir $DATASET_DIR ==="
    "$SCRIPT_DIR/stage4_baselines.sh" "$CONFIG_BUNDLE" --baseline "$B" --dataset_dir "$DATASET_DIR"
done

echo ""
echo "=== [stage4_full] Done. ==="
