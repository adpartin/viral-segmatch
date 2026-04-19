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
# All options can be passed as CLI flags (interactive) or PBS env vars (batch).
# The env-var fallback exists because PBS `qsub script.sh` does not forward
# CLI arguments — only -v env vars survive. This lets the same script serve
# both modes without per-sweep wrapper scripts.
#
#   CLI flag             Env var (-v)         Description
#   --dry_run            (n/a)                Print commands without executing
#   --pairs "b1 b2"      PAIRS="b1 b2"       Run only these bundles (default: all 28)
#   --skip_dataset        SKIP_DATASET=true    Reuse existing datasets (skip Stage 3)
#   --skip_training       SKIP_TRAINING=true   Generate datasets only (skip Stage 4)
#   --skip_aggregate      SKIP_AGGREGATE=true  Skip per-pair CV aggregation
#   --filter k=v          FILTERS_CSV="k=v,…"  Hydra dotlist overrides (repeatable CLI,
#                                              comma-separated env). See filter docs below.
#   --filter-tag tag      FILTER_TAG="tag"     Short tag for dir names (auto-derived if absent)
#   --dataset_manifest f  DATASET_MANIFEST=f   JSON file mapping bundle → dataset dir path.
#                                              Written automatically by Stage 3 runs.
#                                              Bypasses glob-based auto-discovery, allowing
#                                              dataset reuse across training configs.
#
# DATASET REUSE
# -------------
# Stage 3 (--skip_training) writes a dataset_manifest.json to the manifest dir.
# Stage 4 can consume it via --dataset_manifest / DATASET_MANIFEST to reuse
# those exact datasets with different training settings (e.g., k-mer vs ESM-2).
# This decouples dataset creation from training without relying on glob patterns
# or tag naming conventions.
#
# Example: create H3N2 datasets once, train with both k-mer and ESM-2:
#   # Stage 3 (once):
#   qsub -v FILTERS_CSV="dataset.hn_subtype=H3N2",SKIP_TRAINING=true ...
#   # → writes models/flu/July_2025/allpairs_prod_h3n2_<ts>/dataset_manifest.json
#
#   # Stage 4 with k-mer (default):
#   qsub -v FILTERS_CSV="dataset.hn_subtype=H3N2",SKIP_DATASET=true ...
#
#   # Stage 4 with ESM-2 (reusing same datasets):
#   qsub -v FILTERS_CSV="training.feature_source=esm2",FILTER_TAG="h3n2_esm2",\
#        DATASET_MANIFEST="models/flu/.../allpairs_prod_h3n2_<ts>/dataset_manifest.json",\
#        SKIP_DATASET=true ...
#
# FILTER MECHANISM
# ----------------
# Filters apply Hydra dotlist overrides to Stage 3 and Stage 4 without creating
# new bundles. A tag suffix is injected into every output directory so filtered
# runs don't collide with the unfiltered baseline. See:
#   docs/allpairs_filter_sweep_runbook.md
#
# Examples:
#   # Interactive (CLI flags):
#   bash scripts/run_allpairs_polaris_prod.sh --filter dataset.hn_subtype=H3N2
#
#   # Batch (env vars):
#   qsub -v FILTERS_CSV="dataset.hn_subtype=H3N2" scripts/run_allpairs_polaris_prod.sh
#
#   # Multiple filters:
#   qsub -v FILTERS_CSV="dataset.hn_subtype=H3N2,dataset.host=human" ...
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

JOB_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
JOB_START_SEC=$(date +%s)

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
# Defaults from env vars (PBS -v fallback). CLI flags below override these.
# Rationale: PBS `qsub script.sh` cannot forward CLI args — only env vars
# passed via -v survive into the job. By reading env vars here, the same
# script works for both interactive and batch submissions.
DRY_RUN=false
PAIR_FILTER="${PAIRS:-}"                      # -v PAIRS="b1 b2"
SKIP_DATASET="${SKIP_DATASET:-false}"         # -v SKIP_DATASET=true
SKIP_TRAINING="${SKIP_TRAINING:-false}"       # -v SKIP_TRAINING=true
SKIP_AGGREGATE="${SKIP_AGGREGATE:-false}"     # -v SKIP_AGGREGATE=true
FILTER_TAG="${FILTER_TAG:-}"                  # -v FILTER_TAG="h3n2"
DATASET_MANIFEST="${DATASET_MANIFEST:-}"      # -v DATASET_MANIFEST="/path/to/dataset_manifest.json"
FILTERS=()                                    # populated from FILTERS_CSV or --filter

# Parse FILTERS_CSV env var (list of Hydra overrides).
# Use | as separator to avoid PBS -v comma conflicts. Comma also accepted
# for backward compatibility when running interactively.
# PBS examples (each -v sets one variable to avoid comma ambiguity):
#   qsub -v FILTERS_CSV="dataset.hn_subtype=H3N2|dataset.host=human" \
#        -v FILTER_TAG=h3n2_human -v SKIP_DATASET=true script.sh
if [ -n "${FILTERS_CSV:-}" ]; then
    if [[ "$FILTERS_CSV" == *"|"* ]]; then
        IFS='|' read -ra FILTERS <<< "$FILTERS_CSV"
    else
        IFS=',' read -ra FILTERS <<< "$FILTERS_CSV"
    fi
fi

# CLI flags (override env vars when both are present)
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry_run)         DRY_RUN=true; shift ;;
        --pairs)           PAIR_FILTER="$2"; shift 2 ;;
        --skip_dataset)    SKIP_DATASET=true; shift ;;
        --skip_training)   SKIP_TRAINING=true; shift ;;
        --skip_aggregate)  SKIP_AGGREGATE=true; shift ;;
        --filter)          FILTERS+=("$2"); shift 2 ;;
        --filter-tag)      FILTER_TAG="$2"; shift 2 ;;
        --dataset_manifest) DATASET_MANIFEST="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-derive tag from filter values if not explicitly set.
# Takes the RHS of each key=value, lowercases, strips non-alnum, sorts, joins with _.
# Example: dataset.hn_subtype=H3N2 dataset.host=human → h3n2_human
if [ ${#FILTERS[@]} -gt 0 ] && [ -z "$FILTER_TAG" ]; then
    _parts=()
    for f in "${FILTERS[@]}"; do
        _val="${f#*=}"
        _val=$(echo "$_val" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')
        _parts+=("$_val")
    done
    FILTER_TAG=$(printf '%s\n' "${_parts[@]}" | sort | paste -sd_ -)
fi

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

TAG_INFIX=""
if [ -n "$FILTER_TAG" ]; then
    TAG_INFIX="_${FILTER_TAG}"
fi
MANIFEST_DIR="$PROJECT_ROOT/models/flu/July_2025/allpairs_prod${TAG_INFIX}_${TIMESTAMP}"
mkdir -p "$MANIFEST_DIR"
cp conf/bundles/flu_28_major_protein_pairs_master.yaml \
   "$MANIFEST_DIR/master_bundle_snapshot.yaml"

echo "============================================================"
echo "All protein-pair experiments (multi-node)"
echo "============================================================"
echo "Bundles:       $NUM_BUNDLES"
echo "Nodes:         $NUM_NODES"
echo "Dry run:       $DRY_RUN"
echo "Skip dataset:  $SKIP_DATASET"
echo "Skip training: $SKIP_TRAINING"
echo "Timestamp:     $TIMESTAMP"
echo "Manifest:      $MANIFEST_DIR"
if [ -n "${PBS_JOBID:-}" ]; then
    echo "PBS Job ID:    $PBS_JOBID"
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
# These flags are forwarded to every run_cv_lambda.py invocation (one per pair).
CV_FLAGS="--gpus 0 1 2 3"
if [ "$SKIP_DATASET" = true ]; then
    CV_FLAGS="$CV_FLAGS --skip_dataset"
fi
if [ "$SKIP_TRAINING" = true ]; then
    # Stage 3 only: generate datasets, then exit without training.
    # Useful for parallelizing dataset creation across 28 nodes before
    # submitting a separate training job with --skip_dataset.
    CV_FLAGS="$CV_FLAGS --skip_training"
fi
if [ "$SKIP_AGGREGATE" = true ]; then
    CV_FLAGS="$CV_FLAGS --skip_aggregate"
fi
if [ -n "$FILTER_TAG" ]; then
    CV_FLAGS="$CV_FLAGS --tag $FILTER_TAG"
fi
if [ ${#FILTERS[@]} -gt 0 ]; then
    CV_FLAGS="$CV_FLAGS --override ${FILTERS[*]}"
fi

echo "Filters:     ${FILTERS[*]:-<none>}"
echo "Filter tag:  ${FILTER_TAG:-<none>}"
echo "Dataset manifest: ${DATASET_MANIFEST:-<none> (auto-discovery)}"

# --- Validate dataset manifest (if provided) ---
# The manifest is a JSON file mapping bundle names to dataset directory paths,
# written automatically by this script at the end of Stage 3. Using a manifest
# for Stage 4 decouples dataset creation from training: you can create datasets
# once (with one set of filters), then train with different feature sources or
# training configs by pointing to the same manifest. No glob/tag/naming-convention
# assumptions — just explicit paths.
if [ -n "$DATASET_MANIFEST" ] && [ ! -f "$DATASET_MANIFEST" ]; then
    echo "ERROR: Dataset manifest not found: $DATASET_MANIFEST"
    exit 1
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
    PAIR_CV_FLAGS="$CV_FLAGS"

    # When a dataset manifest is provided, resolve the dataset dir for this pair
    # and pass it explicitly via --dataset_run_dir. This bypasses run_cv_lambda's
    # tag-based auto-discovery, allowing dataset reuse across different training
    # configs (e.g., same H3N2 datasets for both k-mer and ESM-2 training).
    if [ -n "$DATASET_MANIFEST" ]; then
        DATASET_DIR=$(python3 -c "import json; print(json.load(open('$DATASET_MANIFEST'))['$BUNDLE'])" 2>/dev/null)
        if [ -z "$DATASET_DIR" ]; then
            echo "ERROR: Bundle '$BUNDLE' not found in dataset manifest: $DATASET_MANIFEST"
            continue
        fi
        PAIR_CV_FLAGS="$PAIR_CV_FLAGS --dataset_run_dir $DATASET_DIR"
    fi

    CV_CMD="cd $PROJECT_ROOT && python3 $PROJECT_ROOT/scripts/run_cv_lambda.py --config_bundle $BUNDLE $PAIR_CV_FLAGS"

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

# --- Write dataset manifest ---
# After Stage 3 completes (whether standalone or combined with Stage 4), scan for
# the dataset dirs and write a JSON manifest mapping bundle → dataset path. This
# manifest enables dataset reuse across training configs (e.g., k-mer vs ESM-2)
# via --dataset_manifest / DATASET_MANIFEST in subsequent Stage 4 runs.
if [ "$SKIP_DATASET" != true ] && [ "$SUCCEEDED" -gt 0 ]; then
    DATASET_MANIFEST_OUT="$MANIFEST_DIR/dataset_manifest.json"
    python3 -c "
import json, sys
from pathlib import Path

project_root = Path('$PROJECT_ROOT')
bundles = '${BUNDLES[*]}'.split()
tag_suffix = '_$FILTER_TAG' if '$FILTER_TAG' else ''
manifest = {}

for bundle in bundles:
    # Find latest dataset dir matching this bundle + tag
    datasets_base = project_root / 'data' / 'datasets' / 'flu' / 'July_2025' / 'runs'
    pattern = f'dataset_{bundle}{tag_suffix}_*'
    candidates = sorted(datasets_base.glob(pattern))
    if candidates:
        manifest[bundle] = str(candidates[-1])
    else:
        print(f'WARNING: No dataset dir found for {bundle} (pattern: {pattern})', file=sys.stderr)

with open('$DATASET_MANIFEST_OUT', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'Saved dataset manifest ({len(manifest)} pairs): $DATASET_MANIFEST_OUT')
" 2>&1
fi

# --- Cross-pair aggregation ---
echo ""
echo "Running cross-pair aggregation..."
AGG_FLAGS="--output_dir $MANIFEST_DIR"
if [ -n "$FILTER_TAG" ]; then
    AGG_FLAGS="$AGG_FLAGS --tag $FILTER_TAG"
fi
python3 "$PROJECT_ROOT/src/analysis/aggregate_allpairs_results.py" \
    $AGG_FLAGS 2>&1 || echo "WARNING: Cross-pair aggregation failed (non-fatal)."

# --- Summary ---
JOB_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
JOB_END_SEC=$(date +%s)
JOB_ELAPSED=$(( JOB_END_SEC - JOB_START_SEC ))
JOB_ELAPSED_MIN=$(( JOB_ELAPSED / 60 ))
JOB_ELAPSED_SEC=$(( JOB_ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "Task 11 complete."
echo "  Started:    $JOB_START_TIME"
echo "  Finished:   $JOB_END_TIME"
echo "  Elapsed:    ${JOB_ELAPSED_MIN}m ${JOB_ELAPSED_SEC}s"
echo "  Succeeded:  $SUCCEEDED / $NUM_BUNDLES"
if [ $FAILED -gt 0 ]; then
    echo "  Failed:     $FAILED ($FAILED_BUNDLES )"
    echo ""
    echo "  Re-run entire failed pairs:"
    echo "    --pairs \"$FAILED_BUNDLES\" --skip_dataset"
    echo ""
    echo "  Re-run only failed folds (per pair):"
    for FB in $FAILED_BUNDLES; do
        PAIR_LOG="$MANIFEST_DIR/${FB}.log"
        # Extract failed fold numbers from run_cv_lambda.py output: "Training failed for folds: [6, 11]"
        # run_cv_lambda.py prints a ready-to-use re-run command on failure
        RERUN_CMD=$(grep -oP 'Re-run with: \K.*' "$PAIR_LOG" 2>/dev/null | tail -1)
        if [ -n "$RERUN_CMD" ]; then
            echo "    $RERUN_CMD"
        else
            echo "    # $FB: check $PAIR_LOG for details"
        fi
    done
fi
echo "  Manifest:   $MANIFEST_DIR/manifest.txt"
echo "  Pair logs:  $MANIFEST_DIR/<bundle>.log"
echo "  Summary:    $MANIFEST_DIR/allpairs_summary.csv"
if [ -n "${PBS_JOBID:-}" ]; then
    echo "  PBS Job ID: $PBS_JOBID"

    # Copy the PBS stdout log into the results dir for easier discovery.
    # PBS writes to logs/allpairs/{JOBID}.OU after completion, but during
    # execution we can't predict the full hostname suffix. Instead, save the
    # job ID so the user (or check_allpairs_status.py) can find it later.
    echo "$PBS_JOBID" > "$MANIFEST_DIR/pbs_job_id.txt"
    PBS_LOG_DIR="$PROJECT_ROOT/logs/allpairs"
    PBS_LOG_SRC=$(ls "$PBS_LOG_DIR/${PBS_JOBID}"*.OU 2>/dev/null | head -1)
    if [ -n "$PBS_LOG_SRC" ] && [ -f "$PBS_LOG_SRC" ]; then
        cp "$PBS_LOG_SRC" "$MANIFEST_DIR/pbs_job.log"
        echo "  PBS log:    $MANIFEST_DIR/pbs_job.log"
    else
        echo "  PBS log:    (will appear in $PBS_LOG_DIR/${PBS_JOBID}*.OU after job ends)"
    fi
fi
echo "============================================================"
