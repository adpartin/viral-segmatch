#!/bin/bash -l
# =============================================================================
# Task 11: Multi-node launcher for all 28 protein-pair CV experiments
# =============================================================================
#
# PURPOSE
# -------
# Run 28 protein-pair experiments in parallel across Polaris nodes. Each node
# gets one protein pair and runs 12-fold CV using all 4 GPUs (via run_cv_lambda.py).
#
# This is the production launcher for Phase 3 (full dataset, 100 epochs, 12-fold CV).
# Also usable for Phase 2 (subset of pairs, debug-scaling queue) via --pairs.
#
# ARCHITECTURE
# ------------
# Head node (this script) → mpiexec --hostfile per node → run_cv_lambda.py
#   - Each node runs one protein pair
#   - run_cv_lambda.py handles: Stage 3 (dataset) → Stage 4 (12 folds on 4 GPUs) → aggregation
#   - 12 folds / 4 GPUs = 3 waves per node (each fold is single-GPU MLP training)
#
# Uses mpiexec with per-node hostfiles (ALCF multi-node ensemble pattern).
# This avoids SSH key/passphrase issues in batch mode. See:
#   https://docs.alcf.anl.gov/running-jobs/example-job-scripts/#multi-node-ensemble-calculations-example
#
# TIMING (Phase 3: full dataset, 12-fold CV, 100 epochs, patience=15)
# -------------------------------------------------------------------
# Stage 3: ~5 min per pair (full dataset, 12 folds)
# Stage 4: ~50 min per fold at 100 epochs; with early stopping (~30-50 epochs) → ~15-25 min/fold
#          12 folds / 4 GPUs = 3 waves × 25 min = ~75 min training
# Total per pair: ~80-90 min
# All 28 pairs in parallel: ~90 min (fits easily in 6h medium prod walltime)
#
# USAGE
# -----
# Phase 3 (full run, 28 nodes, prod queue — batch submission):
#   qsub scripts/run_allpairs_polaris_prod.sh
#
# Phase 2 / subset runs (interactive — recommended for debugging):
#   qsub -I -l select=2:ncpus=64:ngpus=4 -l walltime=1:00:00 -A IMPROVE_Aim1 -q debug-scaling -l filesystems=eagle
#   # On compute node:
#   cd /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch
#   source scripts/polaris_env.sh
#   bash scripts/run_allpairs_polaris_prod.sh --pairs "flu_28p_ha_na flu_28p_pb2_pb1"
#
# NOTE: `qsub script.sh -- --args` does NOT work with PBS (it interprets -- as
# "run command directly"). For batch submission with subset pairs, use -v:
#   qsub -v PAIRS="flu_28p_ha_na flu_28p_pb2_pb1" scripts/run_allpairs_polaris_prod.sh
#
# Dry-run (print commands without executing):
#   bash scripts/run_allpairs_polaris_prod.sh --dry_run
#
# Interactive (on a single compute node, runs pairs sequentially):
#   bash scripts/run_allpairs_polaris_prod.sh --pairs "flu_28p_ha_na"
#
# OPTIONS
# -------
#   --dry_run          Print commands without executing
#   --pairs "b1 b2"   Run only these bundles (default: all 28)
#   --skip_dataset     Pass --skip_dataset to run_cv_lambda.py (reuse existing datasets)
#   --skip_aggregate   Skip per-pair CV aggregation
#
# PREREQUISITES
# -------------
#   1. Master bundle: conf/bundles/flu_28_major_protein_pairs_master.yaml (Phase 3 settings)
#   2. 28 child bundles: conf/bundles/flu_28p_*.yaml
#   3. Data on Eagle: data/processed/flu/July_2025/protein_final.csv
#   4. K-mer features on Eagle: data/embeddings/flu/July_2025/kmer_k6_*
#   5. Python env: ALCF base + cepi_polaris venv (see polaris_plan.md)
#
# OUTPUT
# ------
# Per-pair (created by run_cv_lambda.py):
#   data/datasets/flu/July_2025/runs/dataset_flu_28p_{protA}_{protB}_{ts}/fold_0/ ... fold_11/
#   models/flu/July_2025/runs/training_flu_28p_{protA}_{protB}_fold{k}_{ts}/
#   models/flu/July_2025/cv_runs/cv_flu_28p_{protA}_{protB}_{ts}/cv_summary.json
#
# This launcher:
#   models/flu/July_2025/allpairs_prod_{ts}/
#     ├── manifest.txt                  # bundle → node → status
#     ├── master_bundle_snapshot.yaml   # frozen master settings
#     └── *.log                         # per-pair logs
#
# =============================================================================

# --- PBS directives (Phase 3: 28 nodes, medium prod, 6h) ---
#PBS -N allpairs_prod
#PBS -l select=28:ncpus=64:ngpus=4
#PBS -l walltime=06:00:00
#PBS -A IMPROVE_Aim1
#PBS -q prod
#PBS -l filesystems=eagle
#PBS -o /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch/logs/allpairs/
#PBS -e /lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch/logs/allpairs/

set -euo pipefail

# --- Resolve project root ---
if [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_ROOT="$PBS_O_WORKDIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi
cd "$PROJECT_ROOT"

# --- PBS environment setup ---
# The #!/bin/bash -l shebang makes this a login shell, so the full default
# environment (PrgEnv-nvidia, Cray PALS/mpiexec, libfabric, CUDA libs) is
# already loaded. We just need conda + project venv on top.
# See: https://docs.alcf.anl.gov/running-jobs/example-job-scripts/
if [ -n "${PBS_JOBID:-}" ]; then
    module use /soft/modulefiles 2>/dev/null || true
    module load conda 2>/dev/null || true
    conda activate base 2>/dev/null || true

    # Activate project venv (if it exists)
    VENV_DIR="/lus/eagle/projects/IMPROVE_Aim1/apartin/venvs/cepi_polaris"
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    fi

    export http_proxy="http://proxy.alcf.anl.gov:3128"
    export https_proxy="http://proxy.alcf.anl.gov:3128"
fi

# --- Parse arguments ---
DRY_RUN=false
PAIR_FILTER="${PAIRS:-}"  # Accept PAIRS env var (for batch: qsub -v PAIRS="b1 b2" ...)
SKIP_DATASET=false
SKIP_AGGREGATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry_run)         DRY_RUN=true; shift ;;
        --pairs)           PAIR_FILTER="$2"; shift 2 ;;
        --skip_dataset)    SKIP_DATASET=true; shift ;;
        --skip_aggregate)  SKIP_AGGREGATE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Build list of bundles ---
if [ -n "$PAIR_FILTER" ]; then
    BUNDLES=($PAIR_FILTER)
else
    BUNDLES=()
    for f in conf/bundles/flu_28p_*.yaml; do
        BUNDLES+=("$(basename "$f" .yaml)")
    done
fi

NUM_BUNDLES=${#BUNDLES[@]}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# --- Get unique nodes from PBS ---
if [ -n "${PBS_NODEFILE:-}" ]; then
    mapfile -t NODES < <(sort -u "$PBS_NODEFILE")
else
    # Interactive mode: use localhost (pairs run sequentially)
    NODES=("localhost")
fi
NUM_NODES=${#NODES[@]}

if [ "$NUM_BUNDLES" -gt "$NUM_NODES" ]; then
    echo "ERROR: More bundles ($NUM_BUNDLES) than nodes ($NUM_NODES)."
    echo "  Each node runs one protein pair. Request at least $NUM_BUNDLES nodes."
    echo "  Or use --pairs to select a subset."
    exit 1
fi

# --- Logging and manifest ---
LOG_DIR="$PROJECT_ROOT/logs/allpairs"
mkdir -p "$LOG_DIR"

MANIFEST_DIR="$PROJECT_ROOT/models/flu/July_2025/allpairs_prod_${TIMESTAMP}"
mkdir -p "$MANIFEST_DIR"
cp conf/bundles/flu_28_major_protein_pairs_master.yaml \
   "$MANIFEST_DIR/master_bundle_snapshot.yaml"

echo "============================================================"
echo "Task 11: All protein-pair experiments (multi-node)"
echo "============================================================"
echo "Bundles:     $NUM_BUNDLES"
echo "Nodes:       $NUM_NODES"
echo "Dry run:     $DRY_RUN"
echo "Timestamp:   $TIMESTAMP"
echo "Manifest:    $MANIFEST_DIR"
if [ -n "${PBS_JOBID:-}" ]; then
    echo "PBS Job ID:  $PBS_JOBID"
fi
echo "============================================================"
echo ""

# --- Environment setup command for remote nodes ---
# mpiexec on Polaris inherits the head node's environment, so we only need
# ENV_SETUP for the interactive localhost fallback (non-PBS mode).
ENV_SETUP="module use /soft/modulefiles 2>/dev/null; module load conda 2>/dev/null; conda activate base 2>/dev/null"
VENV_DIR="/lus/eagle/projects/IMPROVE_Aim1/apartin/venvs/cepi_polaris"
if [ -d "$VENV_DIR" ]; then
    ENV_SETUP="$ENV_SETUP; source $VENV_DIR/bin/activate"
fi
ENV_SETUP="$ENV_SETUP; export http_proxy=http://proxy.alcf.anl.gov:3128; export https_proxy=http://proxy.alcf.anl.gov:3128"

# --- Create per-node hostfiles (ALCF multi-node ensemble pattern) ---
# Each pair gets its own hostfile pointing to a single node.
# mpiexec reads the hostfile to know which node to run on.
HOSTFILE_DIR="$MANIFEST_DIR/hostfiles"
if [ -n "${PBS_NODEFILE:-}" ]; then
    mkdir -p "$HOSTFILE_DIR"
    for i in "${!NODES[@]}"; do
        echo "${NODES[$i]}" > "$HOSTFILE_DIR/node_$i"
    done
fi

# --- Build run_cv_lambda.py flags ---
CV_FLAGS="--gpus 0 1 2 3"
if [ "$SKIP_DATASET" = true ]; then
    CV_FLAGS="$CV_FLAGS --skip_dataset"
fi
if [ "$SKIP_AGGREGATE" = true ]; then
    CV_FLAGS="$CV_FLAGS --skip_aggregate"
fi

# --- Launch one pair per node ---
PIDS=()        # PID of each background mpiexec/bash process
PID_BUNDLE=()  # Bundle name corresponding to each PID (same index)
PID_NODE=()    # Node hostname corresponding to each PID
PID_LOG=()     # Log file path corresponding to each PID

for i in "${!BUNDLES[@]}"; do
    BUNDLE="${BUNDLES[$i]}"
    NODE="${NODES[$i]}"
    PAIR_LOG="$MANIFEST_DIR/${BUNDLE}.log"

    # run_cv_lambda.py handles Stage 3 → Stage 4 (12 folds on 4 GPUs) → aggregation
    CV_CMD="cd $PROJECT_ROOT && python3 $PROJECT_ROOT/scripts/run_cv_lambda.py --config_bundle $BUNDLE $CV_FLAGS"

    echo "[$((i+1))/$NUM_BUNDLES] $BUNDLE → $NODE"

    if [ "$DRY_RUN" = true ]; then
        if [ "$NODE" = "localhost" ]; then
            echo "  [dry-run] bash -c \"$CV_CMD\""
        else
            echo "  [dry-run] mpiexec --hostfile $HOSTFILE_DIR/node_$i -n 1 --ppn 1 bash -c \"$CV_CMD\""
        fi
        continue
    fi

    if [ "$NODE" = "localhost" ]; then
        # Interactive mode: run directly (sequentially if only 1 node)
        bash -c "$CV_CMD" > "$PAIR_LOG" 2>&1 &
    else
        # Batch mode: use mpiexec to dispatch to the allocated compute node.
        # Each pair gets 1 rank on 1 node; run_cv_lambda.py manages the 4 GPUs internally.
        mpiexec --hostfile "$HOSTFILE_DIR/node_$i" \
            -n 1 --ppn 1 \
            bash -c "$CV_CMD" > "$PAIR_LOG" 2>&1 &
    fi

    PIDS+=($!)
    PID_BUNDLE+=("$BUNDLE")
    PID_NODE+=("$NODE")
    PID_LOG+=("$PAIR_LOG")
done

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[dry-run] No processes launched."
    exit 0
fi

# --- Wait for all processes and collect results ---
echo ""
echo "All $NUM_BUNDLES pairs launched. Waiting for completion..."
echo "  Per-pair logs: $MANIFEST_DIR/<bundle>.log"
echo ""

SUCCEEDED=0
FAILED=0
FAILED_BUNDLES=""

for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    BUNDLE="${PID_BUNDLE[$i]}"
    NODE="${PID_NODE[$i]}"
    PAIR_LOG="${PID_LOG[$i]}"

    if wait "$PID"; then
        echo "  DONE:   $BUNDLE (node: $NODE)"
        SUCCEEDED=$((SUCCEEDED + 1))
        echo "$BUNDLE $NODE SUCCEEDED" >> "$MANIFEST_DIR/manifest.txt"
    else
        RC=$?
        echo "  FAILED: $BUNDLE (node: $NODE, exit=$RC). See $PAIR_LOG"
        FAILED=$((FAILED + 1))
        FAILED_BUNDLES="$FAILED_BUNDLES $BUNDLE"
        echo "$BUNDLE $NODE FAILED(exit=$RC)" >> "$MANIFEST_DIR/manifest.txt"
    fi
done

# --- Cross-pair aggregation ---
echo ""
echo "Running cross-pair aggregation..."
python3 "$PROJECT_ROOT/src/analysis/aggregate_allpairs_results.py" \
    --output_dir "$MANIFEST_DIR" 2>&1 || echo "WARNING: Cross-pair aggregation failed (non-fatal)."

# --- Summary ---
echo ""
echo "============================================================"
echo "Task 11 complete."
echo "  Succeeded:  $SUCCEEDED / $NUM_BUNDLES"
if [ $FAILED -gt 0 ]; then
    echo "  Failed:     $FAILED ($FAILED_BUNDLES )"
    echo "  Re-run failed pairs with:"
    echo "    --pairs \"$FAILED_BUNDLES\" --skip_dataset"
fi
echo "  Manifest:   $MANIFEST_DIR/manifest.txt"
echo "  Pair logs:  $MANIFEST_DIR/<bundle>.log"
echo "  Summary:    $MANIFEST_DIR/allpairs_summary.csv"
if [ -n "${PBS_JOBID:-}" ]; then
    echo "  PBS Job ID: $PBS_JOBID"
fi
echo "============================================================"
