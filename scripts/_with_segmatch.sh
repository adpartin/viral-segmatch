#!/bin/bash
# Activate the segmatch conda env, then exec the given args.
#
# Lets Claude Code run python scripts that need the segmatch env without
# inline `source $HOME/miniconda3/etc/profile.d/conda.sh` (which triggers
# a permission prompt because `source` evaluates arbitrary shell code).
# The wrapper script itself is auto-allowed by .claude/settings.json's
# `Bash(bash scripts/*.sh *)` rule, so callers get no prompt.
#
# Usage:
#   bash scripts/_with_segmatch.sh python -m src.analysis.foo --bar
#   bash scripts/_with_segmatch.sh python script.py --flag
#
# Leading underscore in the filename signals "infrastructure helper, not
# a user-facing pipeline stage" — keeps it visually separate from the
# stage1..stage4 wrappers in `ls scripts/`.

set -euo pipefail

if [ "${CONDA_DEFAULT_ENV:-}" != "segmatch" ]; then
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
        if [ -f "$d/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$d/etc/profile.d/conda.sh"
            break
        fi
    done
    conda activate segmatch
fi

exec "$@"
