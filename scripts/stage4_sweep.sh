#!/bin/bash
# Stage 4 sweep: train MLP + (optional) baselines on multiple Stage 3
# datasets in parallel, optionally across multiple seeds.
#
# Designed for the single-slot cluster_disjoint idXX sweep but accepts
# any dataset-dir pattern. Per (threshold, seed) it runs one MLP on one
# GPU and (sequentially after the MLP) the configured baselines on CPU.
# Threshold dimension runs in parallel across GPUs; seed dimension runs
# sequentially (one batch per seed).
#
# Why this exists: the inline bash loop in the writeup's Reproduce
# section is verbose, hits a zsh-vs-bash array-indexing gotcha when run
# from zsh, and gets re-copied every sweep. This wraps that pattern in
# a small, portable script. Compatible with both bash and zsh (uses
# `for thr in <values>` with a separate counter for the GPU index).
#
# Usage:
#   scripts/stage4_sweep.sh \
#       --bundle <training_bundle> \
#       --thresholds "100 099 098 097 096 095" \
#       --dataset_pattern "dataset_flu_ha_na_cluster_aa_id{thr}_HAonly_*" \
#       [--seeds "42 43 44"] \
#       [--start_gpu 1] \
#       [--baselines "lgbm knn1_margin"] \
#       [--output_prefix training_flu_ha_na_kmer_aa_k3_HAonly] \
#       [--dataset_root data/datasets/flu/July_2025/runs] \
#       [--models_root models/flu/July_2025/runs] \
#       [--log_dir logs/training/sweep] \
#       [--no_baselines]
#
# {thr} in --dataset_pattern is substituted per threshold. The dataset
# dir is then resolved by globbing `<dataset_root>/<substituted_pattern>`
# and taking the most-recently-modified match.
#
# Outputs land under <models_root>/, named with seed suffix when seed
# is not 42 (so the new dirs sort cleanly alongside legacy single-seed
# runs that pre-date the seed-suffix convention).
#
# Seed override: passed through as --override master_seed=N to the
# Stage 4 trainer (Hydra dotlist).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ----- defaults -----
BUNDLE=""
THRESHOLDS=""
DATASET_PATTERN=""
SEEDS="42"
START_GPU=1
BASELINES_ARG=""        # empty → use bundle's baselines.enabled; "none" → skip baselines
OUTPUT_PREFIX=""
DATASET_ROOT="data/datasets/flu/July_2025/runs"
MODELS_ROOT="models/flu/July_2025/runs"
LOG_DIR="logs/training/sweep"
NO_BASELINES=false

# ----- parse args -----
while [[ $# -gt 0 ]]; do
    case $1 in
        --bundle)           BUNDLE="$2"; shift 2 ;;
        --thresholds)       THRESHOLDS="$2"; shift 2 ;;
        --dataset_pattern)  DATASET_PATTERN="$2"; shift 2 ;;
        --seeds)            SEEDS="$2"; shift 2 ;;
        --start_gpu)        START_GPU="$2"; shift 2 ;;
        --baselines)        BASELINES_ARG="$2"; shift 2 ;;
        --output_prefix)    OUTPUT_PREFIX="$2"; shift 2 ;;
        --dataset_root)     DATASET_ROOT="$2"; shift 2 ;;
        --models_root)      MODELS_ROOT="$2"; shift 2 ;;
        --log_dir)          LOG_DIR="$2"; shift 2 ;;
        --no_baselines)     NO_BASELINES=true; shift ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$BUNDLE" ] || [ -z "$THRESHOLDS" ] || [ -z "$DATASET_PATTERN" ]; then
    echo "Error: --bundle, --thresholds, and --dataset_pattern are required."
    echo "Run with --help for usage."
    exit 1
fi

# Default output_prefix: derive from bundle name (strip 'flu_' prefix
# isn't generic enough; just use bundle name plus a placeholder).
if [ -z "$OUTPUT_PREFIX" ]; then
    OUTPUT_PREFIX="training_${BUNDLE}"
fi

mkdir -p "$LOG_DIR"

# ----- resolve baselines list once (so we honor the bundle YAML or
#       the --baselines override consistently) -----
if [ "$NO_BASELINES" = true ]; then
    BASELINES_LIST=""
elif [ -n "$BASELINES_ARG" ]; then
    if [ "$BASELINES_ARG" = "none" ]; then
        BASELINES_LIST=""
    else
        BASELINES_LIST="$BASELINES_ARG"
    fi
else
    # Read from bundle. Mirrors the fixed stage4_full.sh logic.
    BASELINES_LIST=$(python -c "
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
    try:
        bls = list(b)
    except TypeError:
        bls = []
print(' '.join(bls))
" "$BUNDLE")
fi

if [ -n "$BASELINES_LIST" ]; then
    echo "Baselines to run after each MLP: $BASELINES_LIST"
else
    echo "Baselines: (none — MLP only)"
fi
echo "Bundle:        $BUNDLE"
echo "Thresholds:    $THRESHOLDS"
echo "Seeds:         $SEEDS"
echo "Start GPU:     $START_GPU"
echo "Logs:          $LOG_DIR"
echo ""

# ----- main loop -----
for SEED in $SEEDS; do
    echo ""
    echo "########## seed=${SEED} batch ##########"
    GPU=$START_GPU
    for THR in $THRESHOLDS; do
        # Resolve dataset dir from pattern.
        PATTERN="${DATASET_PATTERN//\{thr\}/$THR}"
        DS=$(ls -d "$DATASET_ROOT"/$PATTERN 2>/dev/null | head -1)
        if [ -z "$DS" ]; then
            echo "ERROR: no dataset matching ${DATASET_ROOT}/${PATTERN}"
            exit 1
        fi

        TS=$(date +%Y%m%d_%H%M%S)
        # Only add _seedN_ to dir name when seed != 42 (the historical
        # default). This keeps legacy seed=42 dirs name-compatible.
        if [ "$SEED" = "42" ]; then
            SEED_SEG=""
        else
            SEED_SEG="_seed${SEED}"
        fi
        OUT_MLP="${MODELS_ROOT}/${OUTPUT_PREFIX}_id${THR}${SEED_SEG}_${TS}"
        LOG="${LOG_DIR}/${OUTPUT_PREFIX}_id${THR}${SEED_SEG}_${TS}.log"

        echo "  id${THR} seed=${SEED} cuda:${GPU}  log=${LOG}"

        # Subshell: MLP on the assigned GPU, then baselines sequentially on CPU.
        (
            source ~/.bashrc 2>/dev/null
            conda activate segmatch
            bash "$SCRIPT_DIR/stage4_train.sh" "$BUNDLE" \
                --cuda_name "cuda:${GPU}" \
                --dataset_dir "$DS" \
                --output_dir "$OUT_MLP" \
                --override "master_seed=${SEED}"

            for B in $BASELINES_LIST; do
                OUT_B="${MODELS_ROOT}/baseline_${B}_${OUTPUT_PREFIX#training_}_id${THR}${SEED_SEG}_${TS}"
                bash "$SCRIPT_DIR/stage4_baselines.sh" "$BUNDLE" \
                    --baseline "$B" \
                    --dataset_dir "$DS" \
                    --output_dir "$OUT_B"
            done
        ) > "$LOG" 2>&1 &

        GPU=$((GPU + 1))
        sleep 1   # stagger so log timestamps are unique
    done

    wait    # block until this seed batch finishes
    echo "########## seed=${SEED} batch DONE ##########"
done

echo ""
echo "ALL SEEDS DONE."
