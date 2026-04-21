#!/bin/bash
# =============================================================================
# Batch post-hoc analysis runner for an all-pairs sweep.
# =============================================================================
#
# PURPOSE
# -------
# Runs src/analysis/analyze_stage4_train.py over every training dir produced by
# an allpairs_prod sweep for a given TAG. Each invocation writes per-fold
# post-hoc artifacts (confusion matrix, ROC/PR curves, FP/FN stratification,
# metadata breakdowns) into <training_run>/post_hoc/.
#
# WHY A SEPARATE SCRIPT (vs. inline in train_pair_classifier.py)
# --------------------------------------------------------------
# train_pair_classifier.py now invokes analyze_stage4_train.py at the end of
# each training run (guardrailed — failure never breaks training). This batch
# script exists to cover three remaining cases:
#
#   1. BACKFILL legacy sweeps that ran before the in-train integration
#      (e.g., the April 2026 allpairs_prod_val_* runs).
#   2. REFRESH post_hoc artifacts after analyze_stage4_train.py evolves
#      (new plots, new stratifications, changed metrics) — no retraining needed.
#   3. RECOVER from in-train post-hoc failures without re-training. Missing
#      post_hoc/ dirs are flagged by aggregate_allpairs_results.py.
#
# USAGE
# -----
#   bash scripts/run_allpairs_post_hoc.sh <TAG>
#
# Examples (the six completed April 2026 allpairs_prod_* sweeps):
#   bash scripts/run_allpairs_post_hoc.sh val_unfilt        # allpairs_prod_val_unfilt_20260414_080617
#   bash scripts/run_allpairs_post_hoc.sh val_unfilt_esm2   # allpairs_prod_val_unfilt_esm2_20260416_182049
#   bash scripts/run_allpairs_post_hoc.sh val_h3n2          # allpairs_prod_val_h3n2_20260417_224216
#   bash scripts/run_allpairs_post_hoc.sh val_h3n2_esm2     # allpairs_prod_val_h3n2_esm2_20260419_012308
#   bash scripts/run_allpairs_post_hoc.sh val_h3n2_human    # allpairs_prod_val_h3n2_human_20260419_142439
#   bash scripts/run_allpairs_post_hoc.sh val_human         # allpairs_prod_val_human_20260419_201014
#
# NOTES
# -----
# - Dirs without test_predicted.csv (e.g., failed Polaris stubs) are skipped.
# - val_unfilt has 2 Lambda re-run dirs whose names don't carry the tag;
#   they are appended explicitly below (tracked via cv_run_manifest.json).
# - Per-dir log: <training_run>/post_hoc_run.log
# - No GPU required — analyze_stage4_train.py is CPU-only.
# =============================================================================

set -euo pipefail

TAG="${1:?usage: bash scripts/run_allpairs_post_hoc.sh <TAG>}"

# Resolve project root (works whether called from repo root or elsewhere).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

mapfile -t DIRS < <(ls -d models/flu/July_2025/runs/training_flu_28p_*_${TAG}_fold*_* 2>/dev/null)

# val_unfilt: 2 Lambda re-runs of Polaris-failed folds (pb1_ha/11, pb2_pa/6)
# were saved without the val_unfilt tag in their names. Add explicitly.
if [ "$TAG" = "val_unfilt" ]; then
    DIRS+=("models/flu/July_2025/runs/training_flu_28p_pb1_ha_fold11_20260414_145440")
    DIRS+=("models/flu/July_2025/runs/training_flu_28p_pb2_pa_fold6_20260414_151853")
fi

n=${#DIRS[@]}
echo "TAG=$TAG  total=$n"
i=0; skipped=0
for d in "${DIRS[@]}"; do
    i=$((i+1))
    if [ ! -f "$d/test_predicted.csv" ]; then
        echo "[$i/$n] SKIP (no test_predicted.csv): $(basename $d)"
        skipped=$((skipped+1))
        continue
    fi
    # Strip tagged form first, then fall back to bare _foldN_<ts> (re-run dirs).
    bundle=$(basename "$d" | sed -E "s/^training_//; s/_${TAG}_fold.*//; s/_fold[0-9]+_.*//")
    echo "[$i/$n] $bundle  $(basename $d)"
    python3 src/analysis/analyze_stage4_train.py \
        --config_bundle "$bundle" --model_dir "$d" \
        > "$d/post_hoc_run.log" 2>&1
done
echo "Done. Processed $((n-skipped))/$n; skipped $skipped (no test_predicted.csv)."
