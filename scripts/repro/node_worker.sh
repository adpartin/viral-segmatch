#!/bin/bash
# Per-node worker for the CPU-binding A/B test. Arg $1 = MODE label (used in output
# names so the three mpiexec binding modes don't collide). Runs the 4-fold packing.
set -uo pipefail
cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch

MODE="${1:?need mode label as arg 1}"
NODE="$(hostname -s)"
DS=data/datasets/flu/July_2025/runs/dataset_flu_28p_ha_na_val_unfilt_20260413_151650
mkdir -p logs/phase0c

python3 -c "import h5py, torch" 2>/dev/null \
  || { echo "ERROR [$NODE]: project env not active"; exit 1; }

pids=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python3 src/models/train_pair_classifier.py \
    --config_bundle flu_28p_ha_na --cuda_name cuda:0 \
    --dataset_dir "$DS/fold_$g" \
    --override training.epochs=8 --skip_post_hoc \
    --run_output_subdir "phase0conf_${MODE}_${NODE}_g$g" \
    > "logs/phase0c/conf_${MODE}_${NODE}_g$g.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "[$NODE] MODE=$MODE done"
