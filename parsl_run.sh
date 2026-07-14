#!/bin/bash
# Validate the Parsl port of the all-pairs launcher on 2 pairs / 2 nodes.
#
# HOW TO RUN
# ----------
# Run this from a LOGIN node (polaris-login-*).  Do NOT enter an interactive queue
# (no `qsub -I`): the Parsl script submits its OWN PBS job via PBSProProvider and
# manages it.  This process must stay alive through the queue wait AND the run.
#
# RECOMMENDED -- run inside tmux, so it survives SSH disconnects and you can detach:
#   tmux new -s parsl            # start a session
#   bash parsl_run.sh            # run inside it; watch the live output
#   # detach: Ctrl-b then d   |   reattach later: tmux attach -t parsl
#
# Alternative -- background it and tail the log:
#   nohup bash parsl_run.sh > logs/parsl/run.log 2>&1 &
#   tail -f logs/parsl/run.log
#
# Monitor the PBS job Parsl submits:   qstat -u apartin
#
# Allocation charged: --account (default IMPROVE_Aim1); add --account <name> to change.
# The filesystems (-l filesystems=home:eagle) and CPU affinity are set in the Parsl config.

cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch
source scripts/polaris_env.sh
mkdir -p logs/parsl

# # Two pairs (debug run)
# python3 scripts/run_allpairs_parsl.py \
#     --pairs flu_28p_ha_na flu_28p_pb2_pb1 \
#     --nodes 2 \
#     --queue debug --epochs 20 --tag parsl_debug_2_pairs \
#     --dataset_manifest models/flu/July_2025/allpairs_prod_val_unfilt_20260413_151649/dataset_manifest.json

# 28 pairs (full run)
python3 scripts/run_allpairs_parsl.py \
  --nodes 28 \
  --queue prod --walltime 01:00:00 --tag parsl_28 \
  --dataset_manifest models/flu/July_2025/allpairs_prod_val_unfilt_20260413_151649/dataset_manifest.json


