#!/bin/bash
# Phase 0c head script: node-count scaling probe for the 5x mystery.
# Run inside a debug-scaling allocation (select=8). Runs the production 4-fold-per-node
# packing at K = 2, 4, 8 nodes and records per-epoch time vs node count. Reproduces
# production's shared-Lustre logging I/O pattern (per-fold stdout -> Eagle .log files).
#
# Usage (on the compute node, after `source scripts/polaris_env.sh`):
#   bash logs/phase0c/run_scaling.sh
#
# Uses the proven prod pattern: one backgrounded mpiexec per node (single-node
# hostfile, -n 1 --ppn 1), then wait for the whole wave. mpiexec propagates the
# sourced venv env to each node.
set -uo pipefail
cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch

source scripts/polaris_env.sh

python3 -c "import h5py, torch" 2>/dev/null \
  || { echo "ERROR: project env not active. First run: source scripts/polaris_env.sh"; exit 1; }
mkdir -p logs/phase0c

[ -n "${PBS_NODEFILE:-}" ] || { echo "ERROR: no PBS_NODEFILE - run inside a PBS job"; exit 1; }
mapfile -t NODES < <(sort -u "$PBS_NODEFILE")
echo "Allocated ${#NODES[@]} nodes: ${NODES[*]}"

for K in 2 4 8; do
  [ "$K" -le "${#NODES[@]}" ] || { echo "skip K=$K (only ${#NODES[@]} nodes)"; continue; }
  echo "=== running K=$K nodes ==="
  pids=()
  for i in $(seq 0 $((K - 1))); do
    echo "${NODES[$i]}" > "logs/phase0c/host_${K}_$i"
    mpiexec -n 1 --ppn 1 --hostfile "logs/phase0c/host_${K}_$i" \
      bash logs/phase0c/node_4share.sh "$K" &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "K=$K wave complete"
done
echo "phase0c done. Per-epoch data in models/flu/July_2025/runs/phase0c_*"
