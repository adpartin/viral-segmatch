#!/bin/bash
# One-shot launcher for PB2/PB1 single-slot PB2-only Phase 4 (MMD sweep)
# + Phase 5 (training sweep). Mirrors the HA-NA HA-only flow:
#
#   1. Verify all 6 datasets exist (id100..id095, PB2-only).
#   2. Probe RBF bandwidth sigma on id099 PB2-only:
#        - sigma_S1: PB2 slot (slot a)  kmer_aa k=3, PCA-50
#        - sigma_S2: HA-NA-style pair representation (Test 3)
#      Probe values are printed and saved to logs/.
#   3. Launch 3 MMD sweeps in serial inside one background subshell:
#        --label_filter 1     -> positives only      (no out_suffix)
#        --label_filter 0     -> negatives only      (out_suffix=_neg)
#        --label_filter both  -> all pairs           (out_suffix=_both)
#   4. Launch the Stage 4 training sweep (MLP + LGBM + 1-NN, 3 seeds)
#      in a separate background subshell.
#
# The two sweeps run in parallel: MMD is CPU-bound (~10-15 min), training
# is GPU-bound (~2-3 hours per seed batch). Logs land under
#   logs/sweep/pb2_pb1_PB2only_phase4_mmd.log
#   logs/training/sweep/training_flu_pb2_pb1_kmer_aa_k3_PB2only_id*_*.log
#
# Run with no args. The script self-activates the segmatch conda env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ----- conda env -----
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

DATASET_ROOT="data/datasets/flu/July_2025/runs"
SWEEP_LOG_DIR="logs/sweep"
TRAIN_LOG_DIR="logs/training/sweep"
mkdir -p "$SWEEP_LOG_DIR" "$TRAIN_LOG_DIR"

# ----- 1. verify all 6 datasets exist -----
echo "[1/4] Verifying datasets ..."
MISSING=0
for THR in 100 099 098 097 096 095; do
    PATTERN="dataset_flu_pb2_pb1_cluster_aa_id${THR}_PB2only_*"
    DS=$(ls -d "$DATASET_ROOT"/$PATTERN 2>/dev/null | head -1)
    if [ -z "$DS" ]; then
        echo "  MISSING: ${DATASET_ROOT}/${PATTERN}"
        MISSING=$((MISSING + 1))
    else
        echo "  OK: $(basename "$DS")"
    fi
done
if [ "$MISSING" -gt 0 ]; then
    echo "ERROR: $MISSING dataset(s) missing. Aborting."
    exit 1
fi

# ----- 2. probe sigma on id099 PB2-only -----
echo ""
echo "[2/4] Probing RBF bandwidths on id099 PB2-only (kmer_aa k=3, PCA-50) ..."
PROBE_DS=$(ls -d "$DATASET_ROOT"/dataset_flu_pb2_pb1_cluster_aa_id099_PB2only_* | head -1)
PROBE_LOG="$SWEEP_LOG_DIR/pb2_pb1_PB2only_sigma_probe.log"

# S1 probe: PB2 slot (slot a). Use 1 permutation so it runs fast — we
# only want the printed sigma. mmd_per_slot prints "Median bandwidth
# sigma = X.XXXX  (computed)" before the permutation test starts.
python -m src.analysis.mmd_per_slot \
    --dataset_dir "$PROBE_DS" --slot a \
    --partition_mode dataset_labels --routing_label probe_s1 \
    --feature_space kmer_aa --kmer_k 3 \
    --n_permutations 1 \
    --label_filter 1 \
    --out_csv /tmp/sigma_probe_pb2_s1.csv 2>&1 | tee "$PROBE_LOG"
SIGMA_S1=$(grep "Median bandwidth sigma" "$PROBE_LOG" | tail -1 | awk '{print $5}')

# S2 probe: pair representation (Test 3). Append to same log.
python -m src.analysis.mmd_per_pair \
    --dataset_dir "$PROBE_DS" \
    --partition_mode dataset_labels --routing_label probe_s2 \
    --feature_space kmer_aa --kmer_k 3 \
    --n_permutations 1 \
    --label_filter 1 \
    --out_csv /tmp/sigma_probe_pb2_s2.csv 2>&1 | tee -a "$PROBE_LOG"
SIGMA_S2=$(grep "Median bandwidth sigma" "$PROBE_LOG" | tail -1 | awk '{print $5}')

# Belt-and-braces: verify both sigmas parsed as numeric floats.
if ! [[ "$SIGMA_S1" =~ ^[0-9]+\.?[0-9]*$ ]] || ! [[ "$SIGMA_S2" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "ERROR: sigma parse produced non-numeric values."
    echo "  SIGMA_S1='$SIGMA_S1'  SIGMA_S2='$SIGMA_S2'"
    echo "  Probe log: $PROBE_LOG"
    exit 1
fi

echo ""
echo "  SIGMA_S1 (per-slot, aa k=3) = $SIGMA_S1"
echo "  SIGMA_S2 (pair Test 3, aa k=3) = $SIGMA_S2"
echo "  (HA-NA reference for context: sigma_S1=29.3192, sigma_S2=1.0720)"
echo ""

# ----- 3. MMD sweep (3 label filters, in serial inside one bg subshell) -----
echo "[3/4] Launching MMD sweep (pos/neg/both) in background ..."
MMD_LOG="$SWEEP_LOG_DIR/pb2_pb1_PB2only_phase4_mmd.log"
(
    for SPEC in "1::" "0::_neg" "both::_both"; do
        LF="${SPEC%%::*}"
        OS="${SPEC##*::}"
        echo ""
        echo "=================== label_filter=$LF  out_suffix='$OS' ==================="
        bash scripts/mmd_sweep.sh \
            --thresholds "100 099 098 097 096 095" \
            --dataset_pattern "dataset_flu_pb2_pb1_cluster_aa_id{thr}_PB2only_*" \
            --routing_label_pattern "cluster_aa_id{thr}_PB2only" \
            --feature_space kmer_aa --kmer_k 3 \
            --sigma_s1 "$SIGMA_S1" --sigma_s2 "$SIGMA_S2" \
            --label_filter "$LF" --out_suffix "$OS"
    done
    echo ""
    echo "ALL MMD SWEEPS DONE."
) > "$MMD_LOG" 2>&1 &
MMD_PID=$!
echo "  MMD sweep PID=$MMD_PID  log=$MMD_LOG"

# ----- 4. training sweep (Stage 4: MLP + baselines, 3 seeds) -----
echo ""
echo "[4/4] Launching Stage 4 training sweep (MLP + LGBM + 1-NN, seeds 42 43 44) ..."
TRAIN_LOG="$TRAIN_LOG_DIR/pb2_pb1_PB2only_phase5_train.log"
(
    bash scripts/stage4_sweep.sh \
        --bundle flu_pb2_pb1_kmer_aa_k3 \
        --thresholds "100 099 098 097 096 095" \
        --dataset_pattern "dataset_flu_pb2_pb1_cluster_aa_id{thr}_PB2only_*" \
        --seeds "42 43 44" \
        --baselines "lgbm knn1_margin" \
        --output_prefix training_flu_pb2_pb1_kmer_aa_k3_PB2only
) > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "  Training sweep PID=$TRAIN_PID  log=$TRAIN_LOG"

echo ""
echo "Both sweeps launched. Tail the logs to watch progress:"
echo "  tail -f $MMD_LOG"
echo "  tail -f $TRAIN_LOG"
echo ""
echo "When both finish, aggregate with:"
echo "  python -m src.analysis.aggregate_mmd_single_slot_sweep \\"
echo "      --pair pb2_pb1 --direction PB2only --feature_space kmer_aa --kmer_k 3"
echo "  (verify the aggregator accepts these args; HA-NA defaults assume ha_na/HAonly)"
