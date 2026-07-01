#!/usr/bin/env bash
# Claude Code PostToolUse hook — lint the just-edited Python file with ruff.
# Wire it in .claude/settings.json (see the snippet in the drafting message), e.g.:
#   "PostToolUse": [{ "matcher": "Edit|Write|MultiEdit",
#     "hooks": [{ "type": "command",
#                 "command": "bash \"$CLAUDE_PROJECT_DIR/.claude/hooks/ruff_check.sh\"" }] }]
#
# On violations it surfaces ruff's output back to Claude (exit 2) so Claude fixes it
# in a follow-up edit. Anything that could break the edit loop (non-.py file, missing
# file, ruff/jq not installed) -> silent pass (exit 0). Covers CLAUDE's edits only;
# your own manual edits are gated by pre-commit instead.

input=$(cat)
command -v jq >/dev/null 2>&1 || exit 0
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

case "$file" in *.py) ;; *) exit 0 ;; esac      # Python files only
[ -f "$file" ] || exit 0

if command -v ruff >/dev/null 2>&1; then
  RUFF=(ruff)
elif python -m ruff --version >/dev/null 2>&1; then
  RUFF=(python -m ruff)
else
  exit 0                                          # ruff not installed -> no-op
fi

if ! out=$("${RUFF[@]}" check "$file" 2>&1); then
  printf 'ruff flagged %s — fix before continuing:\n%s\n' "$file" "$out" >&2
  exit 2                                          # PostToolUse exit 2 -> stderr fed back to Claude
fi
exit 0
