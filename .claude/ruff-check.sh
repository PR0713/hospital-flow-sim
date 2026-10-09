#!/bin/bash
# PostToolUse hook: lint the edited Python file; on errors, exit 2 so ruff's output is fed back to Claude.
f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')
case "$f" in
  *.py)
    out=$("${CLAUDE_PROJECT_DIR:-.}/.venv/bin/ruff" check "$f" 2>&1) || { echo "$out" >&2; exit 2; }
    ;;
esac
exit 0
