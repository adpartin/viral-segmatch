#!/bin/bash
# Phase 0b: 4-fold contention run. Run on a Polaris COMPUTE node (needs 4 GPUs).
# Self-contained: defines its own DS, because a child bash does NOT inherit shell
# variables set in the interactive shell (that was the /fold_0/train_pairs.csv bug).

# Step 1: Enter interactive queue
# qsub -I -l select=1:ncpus=64:ngpus=4 -l filesystems=home:eagle -l walltime=01:00:00 -q debug -A IMPROVE_Aim1

# Step 2: cd to project dir
# cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch

set -uo pipefail            # -u: unset var is a hard error (would have caught the DS bug)

# Fail fast if the project env is not active. The venv PATH is inherited from the
# parent shell, so you must source the env there first:  source scripts/polaris_env.sh
python3 -c "import h5py, torch" 2>/dev/null \
  || { echo "ERROR: project env not active. First run: source scripts/polaris_env.sh"; exit 1; }

DS=data/datasets/flu/July_2025/runs/dataset_flu_28p_ha_na_val_unfilt_20260413_151650
EPOCHS=12                   # fits the remaining debug walltime; we only need the per-epoch median
mkdir -p logs/phase0b

nvidia-smi dmon -s um -o DT -f logs/phase0b/dmon_4share.csv &
DMON=$!

pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python3 src/models/train_pair_classifier.py \
    --config_bundle flu_28p_ha_na --cuda_name cuda:0 \
    --dataset_dir "$DS/fold_$g" \
    --override training.epochs=$EPOCHS --skip_post_hoc \
    --run_output_subdir phase0b_share4_g$g > logs/phase0b/train_g$g.log 2>&1 &
  pids+=($!)
done

wait "${pids[@]}"           # wait ONLY for the 4 training jobs, not dmon
kill "$DMON" 2>/dev/null
echo "4-share done. logs: logs/phase0b/train_g*.log ; dmon: logs/phase0b/dmon_4share.csv"
