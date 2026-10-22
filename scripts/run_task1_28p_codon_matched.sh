#!/bin/bash
# Task 1 of docs/plans/2026-10-08_paper_plan.md: experiments 1A and 1B on Human-H3N2-2024.
# 
# Runs scripts/run_allpairs_screen.sh with the Task 1 configuration.
#
# Dataset build, audit, negative redraw, and model fitting (not the report):
#   - Use conf/experiments/human_h3n2_2024_28p_codon_matched_cv5.yaml.
#   - Its 5-fold setting overrides the bundles' 4-fold setting.
#
# The dataset build is shared by experiments 1A and 1B:
#   - Build from all 28 flu_28p_codon_* child bundles by default.
#   - Sample negatives with matched_balanced_usage_negatives, configured by the master
#     bundle's negative_sampling block.
#   - Use the 2026-09-22 capacity CSV from the reported runs.
#
# Experiment 1A:
#   - Name each CV output directory after its bundle.
#
# Experiment 1B:
#   - Redraw training negatives using seeds 42-51: 10 draws per fold.
#
# Datasets are built before fitting. Each 1A fit reads one fold's train_pairs.csv,
# val_pairs.csv and test_pairs.csv. For 1B, the redraw step reads those files and
# writes a seed directory holding a new train_pairs.csv beside copies of val_pairs.csv
# and test_pairs.csv; each 1B fit reads one seed directory.
#
# Usage:
#   bash scripts/run_task1_28p_codon_matched.sh                 # all 28 pairs, final report
#   PAIRS='pb1_ha' bash scripts/run_task1_28p_codon_matched.sh  # one pair, to step through
#   PREFIX='my_prefix_' bash scripts/run_task1_28p_codon_matched.sh
#
# PAIRS limits the build to a subset of specific schema pairs. Later steps read the dataset
# manifest the build wrote, so one named pair restricts the whole chain to 5 fits and 50
# redraw fits, against 140 and 1,400 for all 28. PAIRS also drops --final, which a pair-filtered
# build can never satisfy: a final report requires the datasets to cover every pair.
#
# PREFIX defaults to a new timestamp. It must name datasets that do not exist yet: the builder
# never writes into an existing dataset directory, and model run IDs are built from its name.
#
# Run with the segmatch environment active, or set PYTHON to its interpreter.
#
# The reported run used prefix task1_28p_codon_matched_20261015_194607_ at commit ba9b108, and its
# report is in results/flu/July_2025/task1_28p_codon_matched_20261015_194607_report/. A rerun gets
# a new prefix, a new report and whatever commit it runs at.

set -euo pipefail

# Get the directory containing this script
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Resolve it to an absolute path without changing our working directory.
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"

# Use PREFIX from the environment if nonempty; otherwise generate a timestamped prefix.
PREFIX="${PREFIX:-task1_28p_codon_matched_$(date +%Y%m%d_%H%M%S)_}"

# Use PAIRS from the environment; default to empty which would run all 28 pairs.
PAIRS="${PAIRS:-}"

ARGS=(--experiment-spec conf/experiments/human_h3n2_2024_28p_codon_matched_cv5.yaml
      --dataset-prefix "$PREFIX"
      --capacity-csv results/flu/July_2025/pair_capacity_8_proteins/pair_capacity.csv
      --cv-dir-bundle '{bundle}'
      --bundle-prefix flu_28p_codon_
      --redraw-seeds "42 43 44 45 46 47 48 49 50 51")

if [ -n "$PAIRS" ]; then
    ARGS+=(--pairs "$PAIRS")
else
    ARGS+=(--final)
fi

exec bash "$SCRIPT_DIR/run_allpairs_screen.sh" "${ARGS[@]}"
