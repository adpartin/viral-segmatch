#!/bin/bash
# Phase 0c per-node worker: run 4 folds (one per GPU) of ha_na on THIS node.
# Launched once per node by run_scaling.sh via mpiexec. Arg $1 = K (node count of
# this wave), used only to keep output names unique across waves.
set -uo pipefail
cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch

K="${1:?need K (node count) as arg 1}"
NODE="$(hostname -s)"
DS=data/datasets/flu/July_2025/runs/dataset_flu_28p_ha_na_val_unfilt_20260413_151650
mkdir -p logs/phase0c

python3 -c "import h5py, torch" 2>/dev/null \
  || { echo "ERROR [$NODE]: project env not active (mpiexec env not propagated)"; exit 1; }

pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python3 src/models/train_pair_classifier.py \
    --config_bundle flu_28p_ha_na --cuda_name cuda:0 \
    --dataset_dir "$DS/fold_$g" \
    --override training.epochs=10 --skip_post_hoc \
    --run_output_subdir "phase0c_${K}n_${NODE}_g$g" \
    > "logs/phase0c/${K}n_${NODE}_g$g.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "[$NODE] K=$K done"
