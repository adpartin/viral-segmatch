#!/usr/bin/env bash
# Claude Code PostToolUse hook — lint the just-edited Python file with ruff.
# Wired in .claude/settings.json (PostToolUse, matcher "Edit|Write|MultiEdit").
#
# On violations it surfaces ruff's output back to Claude (exit 2) so Claude fixes it
# in a follow-up edit. Anything that could break the edit loop (non-.py file, missing
# file, no python, ruff not found) -> silent pass (exit 0). Covers CLAUDE's edits only;
# manual edits are gated by pre-commit instead.
#
# No jq dependency: the PostToolUse JSON payload is parsed with python3. ruff is
# resolved in order: $RUFF_BIN, `ruff` on PATH, `python -m ruff`, then the project's
# segmatch env on shared NFS (this repo's canonical env). None found -> no-op.

input=$(cat)

# python (for JSON parse); none -> no-op.
PY=$(command -v python3 || command -v python) || exit 0

# Pull the edited file path out of the PostToolUse payload.
file=$(printf '%s' "$input" | "$PY" -c \
  'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' \
  2>/dev/null) || exit 0

case "$file" in *.py) ;; *) exit 0 ;; esac  # Python files only
[ -f "$file" ] || exit 0

# Resolve a ruff invocation. First hit wins; none found -> no-op.
SEGMATCH_RUFF="/nfs/lambda_stor_01/homes/apartin/miniconda3/envs/segmatch/bin/ruff"
if [ -n "${RUFF_BIN:-}" ] && command -v "$RUFF_BIN" >/dev/null 2>&1; then
  RUFF=("$RUFF_BIN")
elif command -v ruff >/dev/null 2>&1; then
  RUFF=(ruff)
elif "$PY" -m ruff --version >/dev/null 2>&1; then
  RUFF=("$PY" -m ruff)
elif [ -x "$SEGMATCH_RUFF" ]; then
  RUFF=("$SEGMATCH_RUFF")
else
  exit 0                                    # ruff not available -> no-op
fi

if ! out=$("${RUFF[@]}" check "$file" 2>&1); then
  printf 'ruff flagged %s — fix before continuing:\n%s\n' "$file" "$out" >&2
  exit 2                                    # PostToolUse exit 2 -> stderr fed back to Claude
fi
exit 0
