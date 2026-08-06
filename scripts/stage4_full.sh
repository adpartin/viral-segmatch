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
# Accepts two shapes of the `baselines` block in the bundle YAML:
#   (a) `baselines: [lgbm, knn1_margin]`  (a plain list)
#   (b) `baselines: {enabled: [lgbm, knn1_margin]}`  (dict with enabled)
# Both are in use across the bundles. The previous version of this
# block called `list()` on a DictConfig under shape (b), which returns
# dict KEYS (['enabled']) and tried to run a baseline called "enabled".
# This fix reads `.enabled` if present, falls back to the value
# itself if it is already a sequence.
BASELINES=$(python -c "
import sys
sys.path.insert(0, '.')
from src.utils.config_hydra import get_virus_config_hydra
c = get_virus_config_hydra(sys.argv[1], config_path='./conf')
b = getattr(c, 'baselines', None)
if b is None:
    bls = []
elif hasattr(b, 'enabled') and b.enabled is not None:
    bls = list(b.enabled)
elif isinstance(b, (list, tuple)):
    bls = list(b)
else:
    # Last-ditch: try iterating; DictConfig.values() would also work
    # for the dict shape, but we've already handled enabled above.
    try:
        bls = list(b)
    except TypeError:
        bls = []
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
