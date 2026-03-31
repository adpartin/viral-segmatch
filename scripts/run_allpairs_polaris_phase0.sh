#!/bin/bash
# =============================================================================
# Phase 0: All 28 protein-pair experiments — sequential on a single Polaris node
# =============================================================================
#
# PURPOSE
# -------
# Validate that all 28 protein-pair bundles (Task 11) run end-to-end on Polaris
# before scaling to multi-node job arrays. Runs Stage 3 (dataset generation) and
# Stage 4 (MLP training) for each pair sequentially on a single GPU.
#
# This is NOT for production results — it uses small defaults (5K isolates,
# 10 epochs, single split) set in the master bundle:
#   conf/bundles/flu_28_major_protein_pairs_master.yaml
#
# TIMING
# ------
# With Phase 0 defaults (5K isolates, 10 epochs, single split):
#   Stage 3: ~30s per pair (dataset generation, CPU-only)
#   Stage 4: ~1 min per pair (MLP training on 1 GPU)
#   Total:   ~1-2 min per pair × 28 pairs ≈ 30-60 min
# Fits comfortably in the 1h debug queue walltime.
#
# USAGE
# -----
# Interactive (on a Polaris compute node after `qsub -I`):
#   bash scripts/run_allpairs_polaris_phase0.sh
#   bash scripts/run_allpairs_polaris_phase0.sh --dry_run
#   bash scripts/run_allpairs_polaris_phase0.sh --gpu 2
#   bash scripts/run_allpairs_polaris_phase0.sh --pairs "flu_28p_ha_na flu_28p_pb2_pb1"
#
# Batch (submit to PBS scheduler):
#   qsub scripts/run_allpairs_polaris_phase0.sh
#
# OPTIONS
# -------
#   --dry_run   Print commands without executing them
#   --gpu N     Use GPU index N (default: 0). Polaris nodes have GPUs 0-3.
#   --pairs "b1 b2 ..."  Run only these bundles (space-separated). Useful for
#                         testing a subset before running all 28.
#
# PREREQUISITES
# -------------
#   1. Master bundle configured: conf/bundles/flu_28_major_protein_pairs_master.yaml
#   2. 28 child bundles generated: conf/bundles/flu_28p_*.yaml
#      (run once: python3 scripts/generate_all_pairs_bundles.py)
#   3. Preprocessed data: data/processed/flu/July_2025/protein_final.csv
#   4. K-mer features: data/embeddings/flu/July_2025/ (from Stage 2b)
#   5. Conda env with PyTorch, hydra-core, h5py, etc. (see polaris_plan.md)
#
# OUTPUT STRUCTURE
# ----------------
# Each pair produces two directories (same timestamp for traceability):
#   data/datasets/flu/July_2025/runs/dataset_flu_28p_{protA}_{protB}_{ts}/
#     └── train.csv, val.csv, test.csv, dataset_stats.json, ...
#   models/flu/July_2025/runs/training_flu_28p_{protA}_{protB}_{ts}/
#     └── best_model.pt, test_predicted.csv, training_info.json, ...
#
# A manifest directory tracks the full run:
#   models/flu/July_2025/allpairs_phase0_{ts}/
#     ├── manifest.txt                  # bundle → dataset_run_id → training_run_id
#     ├── master_bundle_snapshot.yaml   # frozen copy of master settings
#     └── phase0.log                    # copy of the full log
#
# =============================================================================

# --- PBS directives ---
# These are only read by the PBS scheduler (qsub). Ignored when run interactively.
# Requests 1 node with 4 A100 GPUs in the debug queue (max 1h walltime).
#PBS -N allpairs_phase0
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=01:00:00
#PBS -A IMPROVE_Aim1
#PBS -q debug
#PBS -l filesystems=eagle
#PBS -o /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch/logs/allpairs/
#PBS -e /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch/logs/allpairs/

# Exit immediately on any command failure; treat unset variables as errors;
# fail a pipeline if any command in it fails (not just the last one).
set -euo pipefail

# --- Resolve project root ---
# In PBS batch mode, $PBS_O_WORKDIR is set to the directory from which qsub
# was invoked. In interactive mode, we derive it from the script's location.
if [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_ROOT="$PBS_O_WORKDIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi
cd "$PROJECT_ROOT"

# --- PBS environment setup ---
# In batch mode, dotfiles (.zshrc, polaris.zsh) are NOT sourced. We must
# explicitly load modules and activate the conda environment.
# In interactive mode (qsub -I), dotfiles handle this — skip to avoid conflicts.
if [ -n "${PBS_JOBID:-}" ]; then
    # Load the ALCF-provided conda base environment (includes PyTorch, CUDA, etc.)
    # See: https://docs.alcf.anl.gov/polaris/data-science/python/
    module use /soft/modulefiles 2>/dev/null || true
    module load conda 2>/dev/null || true
    conda activate base 2>/dev/null || true

    # If you have a custom venv on top of the base env, activate it here:
    # source /lus/eagle/projects/IMPROVE_Aim1/apartin/venvs/<name>/bin/activate

    # Proxy for any network access (pip, git) from compute nodes
    export http_proxy="http://proxy.alcf.anl.gov:3128"
    export https_proxy="http://proxy.alcf.anl.gov:3128"
fi

# --- Parse arguments ---
DRY_RUN=false   # If true, print commands but don't execute them
GPU=0           # Which GPU to use (0-3 on Polaris). Phase 0 uses only one.
PAIR_FILTER=""  # Empty = all 28 pairs. Set via --pairs to run a subset.

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry_run)    DRY_RUN=true; shift ;;
        --gpu)        GPU="$2"; shift 2 ;;
        --pairs)      PAIR_FILTER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Build list of bundles ---
# Each bundle is a Hydra config name (e.g., "flu_28p_ha_na") corresponding to
# a YAML file in conf/bundles/. The list is either user-specified (--pairs)
# or auto-discovered by globbing the child bundle files.
if [ -n "$PAIR_FILTER" ]; then
    # User-specified subset (space-separated bundle names)
    BUNDLES=($PAIR_FILTER)
else
    # All 28 pairs: glob the generated child bundles
    BUNDLES=()
    for f in conf/bundles/flu_28p_*.yaml; do
        # Extract bundle name from filename: flu_28p_ha_na.yaml → flu_28p_ha_na
        BUNDLE=$(basename "$f" .yaml)
        BUNDLES+=("$BUNDLE")
    done
fi

NUM_BUNDLES=${#BUNDLES[@]}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# --- Logging ---
# All stdout/stderr from Stage 3 and Stage 4 subprocesses is appended to this
# log file. Console output shows only progress messages (pair N/28, pass/fail).
LOG_DIR="$PROJECT_ROOT/logs/allpairs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/allpairs_phase0_${TIMESTAMP}.log"

echo "============================================================"
echo "Phase 0: All protein-pair experiments"
echo "============================================================"
echo "Bundles:   $NUM_BUNDLES"
echo "GPU:       $GPU"
echo "Dry run:   $DRY_RUN"
echo "Log:       $LOG_FILE"
echo "Timestamp: $TIMESTAMP"
echo "Master:    conf/bundles/flu_28_major_protein_pairs_master.yaml"
echo "============================================================"

# --- Manifest (track what was run) ---
# The manifest directory co-locates provenance info with results. It records:
#   - Which bundles were run (manifest.txt: bundle → dataset_run_id → training_run_id)
#   - The master bundle settings at the time of the run (frozen snapshot)
#   - A copy of the full log
# Datasets are saved under: data/datasets/flu/July_2025/runs/
# Models are saved under:   models/flu/July_2025/runs/
MANIFEST_DIR="$PROJECT_ROOT/models/flu/July_2025/allpairs_phase0_${TIMESTAMP}"
mkdir -p "$MANIFEST_DIR"

# Freeze the master bundle so we know exactly what settings were used,
# even if the master is later modified for Phase 1/2/3.
cp conf/bundles/flu_28_major_protein_pairs_master.yaml "$MANIFEST_DIR/master_bundle_snapshot.yaml"

SUCCEEDED=0
FAILED=0
FAILED_BUNDLES=""

# --- Run each pair sequentially ---
# For each of the 28 protein pairs:
#   1. Stage 3 (CPU): Generate dataset (positive/negative pairs, train/val/test split)
#   2. Stage 4 (GPU): Train MLP classifier on the generated dataset
# Both stages read the same Hydra bundle, so config is consistent.
for i in "${!BUNDLES[@]}"; do
    BUNDLE="${BUNDLES[$i]}"
    IDX=$((i + 1))
    echo ""
    echo "---- [$IDX/$NUM_BUNDLES] $BUNDLE ----"

    # Run IDs include the bundle name and timestamp for traceability.
    # These become subdirectory names under data/datasets/.../runs/ and models/.../runs/.
    DATASET_RUN_ID="dataset_${BUNDLE}_${TIMESTAMP}"
    TRAINING_RUN_ID="training_${BUNDLE}_${TIMESTAMP}"

    # Stage 3: Create segment pairs dataset
    # Reads protein_final.csv + k-mer features, creates train/val/test CSVs.
    # --config_bundle: which Hydra bundle to use (sets schema_pair, max_isolates, etc.)
    # --run_output_subdir: subdirectory name under the default output path
    STAGE3_CMD="python3 src/datasets/dataset_segment_pairs.py \
        --config_bundle $BUNDLE \
        --run_output_subdir $DATASET_RUN_ID"

    # Stage 4: Train MLP classifier
    # CUDA_VISIBLE_DEVICES: restrict to one GPU (avoids PyTorch seeing all 4)
    # --config_bundle: same bundle as Stage 3 (training hyperparams come from here)
    # --dataset_dir: points to the Stage 3 output (explicit, not auto-derived)
    # --run_output_subdir: subdirectory name under the default models output path
    DATASET_DIR="data/datasets/flu/July_2025/runs/$DATASET_RUN_ID"
    STAGE4_CMD="CUDA_VISIBLE_DEVICES=$GPU python3 src/models/train_pair_classifier.py \
        --config_bundle $BUNDLE \
        --dataset_dir $DATASET_DIR \
        --run_output_subdir $TRAINING_RUN_ID"

    if [ "$DRY_RUN" = true ]; then
        echo "  [dry-run] Stage 3: $STAGE3_CMD"
        echo "  [dry-run] Stage 4: $STAGE4_CMD"
        continue
    fi

    # Run Stage 3 (output goes to log file; console shows only pass/fail)
    echo "  Stage 3: generating dataset..."
    if eval "$STAGE3_CMD" >> "$LOG_FILE" 2>&1; then
        echo "  Stage 3: done."
    else
        echo "  ERROR: Stage 3 failed for $BUNDLE. See $LOG_FILE"
        FAILED=$((FAILED + 1))
        FAILED_BUNDLES="$FAILED_BUNDLES $BUNDLE"
        # Continue to next pair — don't abort the whole run for one failure
        continue
    fi

    # Run Stage 4 (output goes to log file; console shows only pass/fail)
    echo "  Stage 4: training..."
    if eval "$STAGE4_CMD" >> "$LOG_FILE" 2>&1; then
        echo "  Stage 4: done."
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "  ERROR: Stage 4 failed for $BUNDLE. See $LOG_FILE"
        FAILED=$((FAILED + 1))
        FAILED_BUNDLES="$FAILED_BUNDLES $BUNDLE"
        continue
    fi

    # Append to manifest: one line per successful pair
    # Format: <bundle_name> <dataset_run_id> <training_run_id>
    echo "$BUNDLE $DATASET_RUN_ID $TRAINING_RUN_ID" >> "$MANIFEST_DIR/manifest.txt"
done

# --- Summary ---
echo ""
echo "============================================================"
echo "Phase 0 complete."
echo "  Succeeded: $SUCCEEDED / $NUM_BUNDLES"
if [ $FAILED -gt 0 ]; then
    echo "  Failed:    $FAILED ($FAILED_BUNDLES)"
fi
echo "  Manifest:  $MANIFEST_DIR/manifest.txt"
echo "  Log:       $LOG_FILE"
echo "============================================================"

# Copy the log into the manifest directory so everything is co-located
cp "$LOG_FILE" "$MANIFEST_DIR/phase0.log" 2>/dev/null || true
