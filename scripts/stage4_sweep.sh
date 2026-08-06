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

# Activate the segmatch conda env if not already active. The python
# invocations inside the per-(threshold, seed) subshells need the env;
# without it they fail with `conda activate` errors or ModuleNotFoundError
# when run non-interactively. Source conda's init script directly --
# sourcing ~/.bashrc is not enough on systems where conda init lives in
# ~/.zshrc only, and `bash -c` subshells don't inherit shell functions
# unless conda.sh has been sourced in the parent shell.
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "segmatch" ]; then
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
        if [ -f "$d/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$d/etc/profile.d/conda.sh"
            break
        fi
    done
    conda activate segmatch
fi

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
# Baselines (stage4_baselines.sh) don't accept a master_seed override, so
# they'd produce byte-identical output for every seed in $SEEDS. To avoid
# the redundant baseline runs, only run them on the first seed. Captures
# the first whitespace-delimited token of SEEDS (works for both bash and zsh).
FIRST_SEED=$(echo "$SEEDS" | awk '{print $1}')

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

        # Auto-detect fold_*/ subdirs (Phase 6 of the k-fold variance
        # plan). When present, the dataset dir is a k-fold parent and the
        # inner subshell loops over folds sequentially within this
        # (threshold, seed) GPU slot. When absent, the existing holdout
        # flow runs.
        FOLD_IDS=""
        for fd in "$DS"/fold_*; do
            [ -d "$fd" ] || continue
            fid=$(basename "$fd" | sed 's/^fold_//')
            FOLD_IDS="$FOLD_IDS $fid"
        done
        FOLD_IDS=$(echo "$FOLD_IDS" | xargs)

        if [ -n "$FOLD_IDS" ]; then
            echo "  id${THR} seed=${SEED} cuda:${GPU}  folds=[${FOLD_IDS}]  log=${LOG}"
        else
            echo "  id${THR} seed=${SEED} cuda:${GPU}  log=${LOG}"
        fi

        # Subshell: MLP on the assigned GPU, then baselines sequentially on CPU.
        # The segmatch env is already activated at the top of this script;
        # the subshell inherits CONDA_PREFIX/PATH so `python` resolves
        # correctly inside stage4_train.sh / stage4_baselines.sh.
        #
        # k-fold case: loop over folds sequentially within this slot; each
        # fold gets its own output dir suffixed _fold${FOLD}. Baselines are
        # run per-fold (one set per fold) when SEED == FIRST_SEED, since
        # baselines don't take a seed override (so seed dimension is
        # redundant for them, but fold dimension is not).
        (
            if [ -n "$FOLD_IDS" ]; then
                for FOLD in $FOLD_IDS; do
                    FOLD_DS="$DS/fold_${FOLD}"
                    OUT_MLP_FOLD="${OUT_MLP}_fold${FOLD}"
                    bash "$SCRIPT_DIR/stage4_train.sh" "$BUNDLE" \
                        --cuda_name "cuda:${GPU}" \
                        --dataset_dir "$FOLD_DS" \
                        --output_dir "$OUT_MLP_FOLD" \
                        --override "master_seed=${SEED}"

                    if [ "$SEED" = "$FIRST_SEED" ]; then
                        for B in $BASELINES_LIST; do
                            OUT_B_FOLD="${MODELS_ROOT}/baseline_${B}_${OUTPUT_PREFIX#training_}_id${THR}_fold${FOLD}_${TS}"
                            bash "$SCRIPT_DIR/stage4_baselines.sh" "$BUNDLE" \
                                --baseline "$B" \
                                --dataset_dir "$FOLD_DS" \
                                --output_dir "$OUT_B_FOLD"
                        done
                    fi
                done
            else
                bash "$SCRIPT_DIR/stage4_train.sh" "$BUNDLE" \
                    --cuda_name "cuda:${GPU}" \
                    --dataset_dir "$DS" \
                    --output_dir "$OUT_MLP" \
                    --override "master_seed=${SEED}"

                if [ "$SEED" = "$FIRST_SEED" ]; then
                    for B in $BASELINES_LIST; do
                        OUT_B="${MODELS_ROOT}/baseline_${B}_${OUTPUT_PREFIX#training_}_id${THR}_${TS}"
                        bash "$SCRIPT_DIR/stage4_baselines.sh" "$BUNDLE" \
                            --baseline "$B" \
                            --dataset_dir "$DS" \
                            --output_dir "$OUT_B"
                    done
                fi
            fi
        ) > "$LOG" 2>&1 &

        GPU=$((GPU + 1))
        sleep 1   # stagger so log timestamps are unique
    done

    wait    # block until this seed batch finishes
    echo "########## seed=${SEED} batch DONE ##########"
done

echo ""
echo "ALL SEEDS DONE."
