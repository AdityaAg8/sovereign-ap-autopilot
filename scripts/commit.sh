#!/usr/bin/env bash
# Commits the staged files without the "hook modified files" failure: runs the pre-commit hooks
# on the staged files once so the auto-fixers (ruff format/--fix, end-of-file, whitespace) rewrite
# them, re-stages those files, then runs `git commit "$@"`, where the hooks now only verify.
# A real hook failure (mypy, headers, secrets) still stops the commit. Usage: scripts/commit.sh -m "msg"
set -euo pipefail

mapfile -d '' files < <(git diff --cached --name-only --diff-filter=ACMR -z)
if [ "${#files[@]}" -eq 0 ] && [ ! -f "$(git rev-parse --git-dir)/MERGE_HEAD" ]; then
  echo "nothing staged" >&2
  exit 1
fi
if [ "${#files[@]}" -gt 0 ]; then
  # First pass may "fail" only because a fixer changed a file; the commit below re-checks.
  uv run python -m pre_commit run --files "${files[@]}" >/dev/null 2>&1 || true
  git add -- "${files[@]}"
fi
git commit "$@"
