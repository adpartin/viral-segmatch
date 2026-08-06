#!/bin/bash
# MMD sweep: run S1 HA, S1 NA, and S2 pair MMD across multiple Stage 3
# datasets for one (feature_space, label_filter) configuration.
#
# Designed for the single-slot cluster_disjoint idXX sweep but accepts
# any dataset-dir pattern. Sequential — k-mer load is fast (<10s); the
# bottleneck is the 500-permutation test (~40-60s/run). One full
# 6-threshold × 3-role sweep takes ~10-15 min for k-mer, ~75 min for
# ESM-2.
#
# Usage:
#   scripts/mmd_sweep.sh \
#       --thresholds "100 099 098 097 096 095" \
#       --dataset_pattern "dataset_flu_ha_na_cluster_aa_id{thr}_HAonly_*" \
#       --routing_label_pattern "cluster_aa_id{thr}_HAonly" \
#       --feature_space kmer_aa --kmer_k 3 \
#       --sigma_s1 29.3192 --sigma_s2 1.0720 \
#       [--label_filter pos] \
#       [--n_permutations 500] \
#       [--dataset_root data/datasets/flu/July_2025/runs] \
#       [--out_dir results/flu/July_2025/runs/split_separation_mmd] \
#       [--out_suffix ""]
#
# {thr} placeholders in --dataset_pattern and --routing_label_pattern
# are substituted per threshold. The MMD output CSV names follow the
# existing convention used by the aggregator:
#
#   slot:  phase2_perm_<routing_label>_<HA|NA>_<fs_suffix>[<out_suffix>].csv
#   pair:  phase2_perm_<routing_label>_HA_NA_pair_<fs_suffix>_test3[<out_suffix>].csv
#
# Where <fs_suffix> is `esm2`, `kmer_aa_k3`, or `kmer_nt_k6`. For
# negative-only or both-labels MMD use --label_filter 0 with
# --out_suffix "_neg" or --label_filter both with --out_suffix "_both".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate the segmatch conda env if not already active. The python
# invocations below need pandas / scipy / etc.; without an active env
# they fail with ModuleNotFoundError. Source conda's init script
# directly — sourcing ~/.bashrc is not enough on systems where conda
# init lives in ~/.zshrc only.
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

THRESHOLDS=""
DATASET_PATTERN=""
ROUTING_LABEL_PATTERN=""
FEATURE_SPACE=""
KMER_K="3"
SIGMA_S1=""
SIGMA_S2=""
LABEL_FILTER="1"        # '0' | '1' | 'both' (CLI strings the MMD scripts accept)
N_PERMUTATIONS="500"
DATASET_ROOT="data/datasets/flu/July_2025/runs"
OUT_DIR="results/flu/July_2025/runs/split_separation_mmd"
OUT_SUFFIX=""
N_ISOLATES=""       # empty → use python script default (1000)
PCA_DIM=""          # empty → use python script default (50)

while [[ $# -gt 0 ]]; do
    case $1 in
        --thresholds)             THRESHOLDS="$2"; shift 2 ;;
        --dataset_pattern)        DATASET_PATTERN="$2"; shift 2 ;;
        --routing_label_pattern)  ROUTING_LABEL_PATTERN="$2"; shift 2 ;;
        --feature_space)          FEATURE_SPACE="$2"; shift 2 ;;
        --kmer_k)                 KMER_K="$2"; shift 2 ;;
        --sigma_s1)               SIGMA_S1="$2"; shift 2 ;;
        --sigma_s2)               SIGMA_S2="$2"; shift 2 ;;
        --label_filter)           LABEL_FILTER="$2"; shift 2 ;;
        --n_permutations)         N_PERMUTATIONS="$2"; shift 2 ;;
        --n_isolates)             N_ISOLATES="$2"; shift 2 ;;
        --pca_dim)                PCA_DIM="$2"; shift 2 ;;
        --dataset_root)           DATASET_ROOT="$2"; shift 2 ;;
        --out_dir)                OUT_DIR="$2"; shift 2 ;;
        --out_suffix)             OUT_SUFFIX="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build optional --n_isolates / --pca_dim args once (empty → omitted).
EXTRA_ARGS=""
[ -n "$N_ISOLATES" ] && EXTRA_ARGS="$EXTRA_ARGS --n_isolates $N_ISOLATES"
[ -n "$PCA_DIM" ]    && EXTRA_ARGS="$EXTRA_ARGS --pca_dim $PCA_DIM"

if [ -z "$THRESHOLDS" ] || [ -z "$DATASET_PATTERN" ] || \
   [ -z "$ROUTING_LABEL_PATTERN" ] || [ -z "$FEATURE_SPACE" ] || \
   [ -z "$SIGMA_S1" ] || [ -z "$SIGMA_S2" ]; then
    echo "Error: --thresholds, --dataset_pattern, --routing_label_pattern,"
    echo "       --feature_space, --sigma_s1, and --sigma_s2 are required."
    echo "Run with --help for usage."
    exit 1
fi

# Map feature_space -> file-suffix convention used by the aggregator.
case "$FEATURE_SPACE" in
    esm2)     FS_SUFFIX="esm2" ;;
    kmer_aa)  FS_SUFFIX="kmer_aa_k${KMER_K}" ;;
    kmer_nt)  FS_SUFFIX="kmer_nt_k${KMER_K}" ;;
    *) echo "Unknown feature_space: $FEATURE_SPACE"; exit 1 ;;
esac

mkdir -p "$OUT_DIR"

echo "Feature space:    $FEATURE_SPACE  (suffix: $FS_SUFFIX)"
echo "Label filter:     $LABEL_FILTER  (output suffix: '$OUT_SUFFIX')"
echo "Thresholds:       $THRESHOLDS"
echo "sigma S1 / S2:    $SIGMA_S1 / $SIGMA_S2"
echo "Permutations:     $N_PERMUTATIONS"
echo "Out dir:          $OUT_DIR"
echo ""

for THR in $THRESHOLDS; do
    PATTERN="${DATASET_PATTERN//\{thr\}/$THR}"
    DS=$(ls -d "$DATASET_ROOT"/$PATTERN 2>/dev/null | head -1)
    if [ -z "$DS" ]; then
        echo "ERROR: no dataset matching ${DATASET_ROOT}/${PATTERN}"
        exit 1
    fi
    LABEL="${ROUTING_LABEL_PATTERN//\{thr\}/$THR}"

    echo "############ $LABEL ############"

    # S1 HA
    OUT="${OUT_DIR}/phase2_perm_${LABEL}_HA_${FS_SUFFIX}${OUT_SUFFIX}.csv"
    echo "-- S1 HA --  -> $(basename "$OUT")"
    python -m src.analysis.mmd_per_slot \
        --dataset_dir "$DS" --slot a \
        --partition_mode dataset_labels --routing_label "$LABEL" \
        --feature_space "$FEATURE_SPACE" --kmer_k "$KMER_K" \
        --sigma "$SIGMA_S1" --n_permutations "$N_PERMUTATIONS" \
        --label_filter "$LABEL_FILTER" $EXTRA_ARGS \
        --out_csv "$OUT" 2>&1 | tail -2

    # S1 NA
    OUT="${OUT_DIR}/phase2_perm_${LABEL}_NA_${FS_SUFFIX}${OUT_SUFFIX}.csv"
    echo "-- S1 NA --  -> $(basename "$OUT")"
    python -m src.analysis.mmd_per_slot \
        --dataset_dir "$DS" --slot b \
        --partition_mode dataset_labels --routing_label "$LABEL" \
        --feature_space "$FEATURE_SPACE" --kmer_k "$KMER_K" \
        --sigma "$SIGMA_S1" --n_permutations "$N_PERMUTATIONS" \
        --label_filter "$LABEL_FILTER" $EXTRA_ARGS \
        --out_csv "$OUT" 2>&1 | tail -2

    # S2 pair (Test 3 interaction)
    OUT="${OUT_DIR}/phase2_perm_${LABEL}_HA_NA_pair_${FS_SUFFIX}_test3${OUT_SUFFIX}.csv"
    echo "-- S2 pair --  -> $(basename "$OUT")"
    python -m src.analysis.mmd_per_pair \
        --dataset_dir "$DS" \
        --partition_mode dataset_labels --routing_label "$LABEL" \
        --feature_space "$FEATURE_SPACE" --kmer_k "$KMER_K" \
        --sigma "$SIGMA_S2" --n_permutations "$N_PERMUTATIONS" \
        --label_filter "$LABEL_FILTER" $EXTRA_ARGS \
        --out_csv "$OUT" 2>&1 | tail -2

    echo ""
done

echo "ALL MMD SWEEPS DONE."
