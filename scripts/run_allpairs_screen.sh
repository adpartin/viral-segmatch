#!/bin/bash
# Run one all-pairs screen end to end: build, audit, fit, optionally redraw, and report.
#
# Each step is an existing command, run in this order:
#   1. build    scripts/run_allpairs_datasets.py          (writes the dataset manifest)
#   2. audit    src/analysis/audit_pair_datasets.py
#   3. fit      scripts/run_allpairs_baselines.py         (writes training_run_manifest.json)
#   4. redraw   scripts/run_allpairs_balanced_redraws.py  (only with --redraw-seeds)
#   5. report   src/analysis/aggregate_allpairs_results.py
# A failed step stops the script, so no later step reads an incomplete result. To continue after
# a failure, run the remaining steps by hand; the fit and redraw steps resume with --resume and the
# run manifest each one printed.
#
# Each step runs under GNU time (/usr/bin/time -v). Its wall time and peak memory go to
# timings.tsv in the run directory. The peak memory is that of the largest single process in the
# step, not the sum over processes running at the same time.
#
# With --final, the report requires all 28 pairs and every fit at the datasets' commit with a
# clean worktree, and this script also requires
# `git status --porcelain --untracked-files=all -- src scripts conf` to be empty before step 1 and
# after step 5.
#
# Usage:
#   bash scripts/run_allpairs_screen.sh \
#       --experiment-spec conf/experiments/human_h3n2_2024_28p_codon_balanced_cv5.yaml \
#       --dataset-prefix h3n2_2024_28p_codon_cv5_20261010_090000_ \
#       --capacity-csv results/flu/July_2025/pair_capacity_8_proteins/pair_capacity.csv \
#       --cv-dir-bundle 'flu_28p_{pair}_codon_cv5' \
#       --reference-dataset-prefix exp2_28p_codon_balanced_20261007_180037_ \
#       --redraw-seeds "42 43 44 45 46 47 48 49 50 51" --final
#
# Options:
#   --experiment-spec PATH            required
#   --dataset-prefix PREFIX_          required; ends in _ and includes YYYYMMDD_HHMMSS
#   --capacity-csv PATH               required
#   --cv-dir-bundle PATTERN           required; CV directory name with {pair} and {bundle}
#   --bundle-prefix PREFIX            default flu_28p_codon_
#   --pairs "a b"                     default all 28
#   --reference-dataset-prefix PREFIX passed to the audit
#   --redraw-seeds "s1 s2 ..."        run the redraw step with these seeds (at least two)
#   --report-dir DIR                  default results/flu/July_2025/{dataset prefix}report
#   --build-workers N                 concurrent dataset builds (default 4)
#   --final                           a final report; see above
#
# The Python interpreter is $PYTHON, default `python`; run with the segmatch env active, or set
# PYTHON to its interpreter.
#
# Output: logs/allpairs_screen/{dataset prefix}{timestamp}/  one log per step, timings.tsv
# ($LOG_ROOT replaces logs/allpairs_screen when set.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python}"

SPEC="" PREFIX="" CAPACITY="" CV_DIR_BUNDLE="" BUNDLE_PREFIX="flu_28p_codon_" PAIRS=""
REFERENCE="" SEEDS="" REPORT_DIR="" BUILD_WORKERS=4 FINAL=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment-spec)          SPEC="$2";          shift 2 ;;
        --dataset-prefix)           PREFIX="$2";        shift 2 ;;
        --capacity-csv)             CAPACITY="$2";      shift 2 ;;
        --cv-dir-bundle)            CV_DIR_BUNDLE="$2"; shift 2 ;;
        --bundle-prefix)            BUNDLE_PREFIX="$2"; shift 2 ;;
        --pairs)                    PAIRS="$2";         shift 2 ;;
        --reference-dataset-prefix) REFERENCE="$2";     shift 2 ;;
        --redraw-seeds)             SEEDS="$2";         shift 2 ;;
        --report-dir)               REPORT_DIR="$2";    shift 2 ;;
        --build-workers)            BUILD_WORKERS="$2"; shift 2 ;;
        --final)                    FINAL=1;            shift ;;
        *) echo "ERROR: unknown option $1"; exit 1 ;;
    esac
done
for required in SPEC PREFIX CAPACITY CV_DIR_BUNDLE; do
    if [ -z "${!required}" ]; then
        echo "ERROR: missing a required option; see the header of $0"
        exit 1
    fi
done
if [[ "$PREFIX" != *_ ]]; then
    echo "ERROR: --dataset-prefix must end in _"
    exit 1
fi
if ! "$PYTHON" -c "import omegaconf, lightgbm" 2>/dev/null; then
    echo "ERROR: $PYTHON cannot import omegaconf and lightgbm; activate segmatch or set PYTHON"
    exit 1
fi

REPORT_DIR="${REPORT_DIR:-results/flu/July_2025/${PREFIX}report}"
DATASET_MANIFEST="data/datasets/flu/July_2025/runs/${PREFIX%_}_manifest.json"
RUN_DIR="${LOG_ROOT:-logs/allpairs_screen}/${PREFIX}$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
TIMINGS="$RUN_DIR/timings.tsv"
printf 'step\texit\twall_clock\tmax_rss_kb\n' > "$TIMINGS"

# Fails the final run unless the code and config the screen uses have no uncommitted changes.
check_clean() {
    local changes
    changes="$(git status --porcelain --untracked-files=all -- src scripts conf)"
    if [ -n "$changes" ]; then
        echo "ERROR: $1: uncommitted changes in src, scripts or conf:"
        echo "$changes"
        exit 1
    fi
}

# Runs one step under GNU time, keeps its log, appends its timing, and stops on failure.
run_step() {
    local step="$1"; shift
    echo "[$(date '+%H:%M:%S')] $step: $*"
    local status=0
    /usr/bin/time -v -o "$RUN_DIR/$step.time" "$@" > "$RUN_DIR/$step.log" 2>&1 || status=$?
    local wall rss
    wall="$(sed -n 's/.*Elapsed (wall clock) time (h:mm:ss or m:ss): //p' "$RUN_DIR/$step.time")"
    rss="$(sed -n 's/.*Maximum resident set size (kbytes): //p' "$RUN_DIR/$step.time")"
    printf '%s\t%s\t%s\t%s\n' "$step" "$status" "$wall" "$rss" >> "$TIMINGS"
    if [ "$status" -ne 0 ]; then
        echo "ERROR: $step exited $status; see $RUN_DIR/$step.log"
        tail -n 20 "$RUN_DIR/$step.log"
        exit "$status"
    fi
    echo "Done. $step in $wall"
}

# The path a runner printed on its "Run manifest:" line.
run_manifest_from() {
    sed -n 's/^Run manifest: //p' "$RUN_DIR/$1.log" | head -n 1
}

echo "Run directory: $RUN_DIR"
git log -1 --format='Commit: %H' | tee "$RUN_DIR/commit.txt"
if [ "$FINAL" -eq 1 ]; then
    check_clean "before the build"
fi

# shellcheck disable=SC2086  # PAIRS and SEEDS are space-separated lists
run_step build "$PYTHON" scripts/run_allpairs_datasets.py \
    --dataset-prefix "$PREFIX" --experiment-spec "$SPEC" --capacity-csv "$CAPACITY" \
    --bundle-prefix "$BUNDLE_PREFIX" --max-workers "$BUILD_WORKERS" ${PAIRS:+--pairs $PAIRS}

run_step audit "$PYTHON" -m src.analysis.audit_pair_datasets \
    --dataset_manifest "$DATASET_MANIFEST" --experiment_spec "$SPEC" \
    ${REFERENCE:+--reference_dataset_prefix "$REFERENCE"}

run_step fit "$PYTHON" scripts/run_allpairs_baselines.py \
    --dataset_manifest "$DATASET_MANIFEST" --experiment_spec "$SPEC" \
    --cv_dir_bundle "$CV_DIR_BUNDLE"
TRAINING_RUN_MANIFEST="$(run_manifest_from fit)"

REDRAW_ARGS=()
if [ -n "$SEEDS" ]; then
    # shellcheck disable=SC2086
    run_step redraw "$PYTHON" scripts/run_allpairs_balanced_redraws.py \
        --dataset_manifest "$DATASET_MANIFEST" --experiment_spec "$SPEC" --seeds $SEEDS
    REDRAW_ARGS=(--redraw_run_manifest "$(run_manifest_from redraw)")
fi

REPORT_ARGS=(--training_run_manifest "$TRAINING_RUN_MANIFEST"
             ${REDRAW_ARGS[@]+"${REDRAW_ARGS[@]}"} --output_dir "$REPORT_DIR")
if [ "$FINAL" -eq 1 ]; then
    REPORT_ARGS+=(--final)
fi
run_step report "$PYTHON" src/analysis/aggregate_allpairs_results.py "${REPORT_ARGS[@]}"

if [ "$FINAL" -eq 1 ]; then
    check_clean "after the report"
fi
echo
column -t -s $'\t' "$TIMINGS"
echo "Done. Report: $REPORT_DIR"
