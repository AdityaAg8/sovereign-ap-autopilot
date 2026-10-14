#!/usr/bin/env bash
# Verifies the project foundations: lint, typecheck, unit tests, header check, web
# lint/typecheck/build all green, and every foundation artifact exists.
# Prints a PASS/FAIL table and exits nonzero if any gate fails; wired to `make verify-m0`.
set +e
set +u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# results: array of "gate|status|detail" (pipes in detail sanitized to "/")
declare -a RESULTS=()

sanitize() {
  printf '%s' "$1" | tr '|' '/' | tr '\n' ' '
}

record() {
  local gate="$1" status="$2" detail="$3"
  RESULTS+=("$(sanitize "$gate")|$status|$(sanitize "$detail")")
}

# Runs "$2..." in directory "$1", records PASS/FAIL under name "$2", and never aborts
# the script. Pass "." for the repo root.
run_gate_in() {
  local dir="$1" name="$2"
  shift 2
  local out
  out="$(cd "$REPO_ROOT/$dir" && "$@" 2>&1)"
  local status=$?
  if [ "$status" -eq 0 ]; then
    record "$name" "PASS" ""
  else
    record "$name" "FAIL" "$(printf '%s' "$out" | tail -n 3)"
  fi
}

run_gate() {
  local name="$1"
  shift
  run_gate_in "." "$name" "$@"
}

# Checks that a file exists at "$2", recording under name "$1".
check_file_exists() {
  local name="$1" path="$2"
  if [ -f "$path" ]; then
    record "$name" "PASS" ""
  else
    record "$name" "FAIL" "missing: $path"
  fi
}

# Checks that at least one file matches a glob, recording under name "$1".
check_glob_exists() {
  local name="$1" pattern="$2"
  # shellcheck disable=SC2086
  local matches=($pattern)
  if [ -e "${matches[0]}" ]; then
    record "$name" "PASS" ""
  else
    record "$name" "FAIL" "no file matches: $pattern"
  fi
}

echo "==> Python: lint, format, typecheck, test"
run_gate "ruff check" uv run ruff check .
run_gate "ruff format --check" uv run ruff format --check .
run_gate "mypy --strict" uv run mypy
run_gate "pytest -m 'not gpu'" uv run pytest -m "not gpu"

echo "==> file headers"
run_gate "check_headers --all" uv run python scripts/check_headers.py --all

echo "==> Web: lint, typecheck, build"
if [ -d "$REPO_ROOT/web" ]; then
  run_gate_in "web" "web lint" pnpm lint
  run_gate_in "web" "web typecheck" pnpm typecheck
  run_gate_in "web" "web build" pnpm build
else
  record "web lint" "FAIL" "web/ directory not found"
  record "web typecheck" "FAIL" "web/ directory not found"
  record "web build" "FAIL" "web/ directory not found"
fi

echo "==> Foundation artifacts present"
check_file_exists "reports/env.md" "$REPO_ROOT/reports/env.md"
check_file_exists "docs/architecture.md" "$REPO_ROOT/docs/architecture.md"
check_glob_exists "docs/adr/0001-*" "$REPO_ROOT/docs/adr/0001-*"
check_glob_exists "docs/adr/0002-*" "$REPO_ROOT/docs/adr/0002-*"
check_glob_exists "docs/adr/0003-*" "$REPO_ROOT/docs/adr/0003-*"
check_glob_exists "docs/adr/0004-*" "$REPO_ROOT/docs/adr/0004-*"
check_glob_exists "docs/adr/0005-*" "$REPO_ROOT/docs/adr/0005-*"
check_file_exists "infra/compose.yaml" "$REPO_ROOT/infra/compose.yaml"
check_file_exists "infra/models.lock.json" "$REPO_ROOT/infra/models.lock.json"
check_file_exists "reports/smoke.md" "$REPO_ROOT/reports/smoke.md"

echo
echo "Foundation verification results:"
printf '%-28s %-6s %s\n' "GATE" "STATUS" "DETAIL"
overall=0
for row in "${RESULTS[@]}"; do
  IFS='|' read -r gate status detail <<<"$row"
  printf '%-28s %-6s %s\n' "$gate" "$status" "$detail"
  if [ "$status" = "FAIL" ]; then
    overall=1
  fi
done

echo
if [ "$overall" -eq 0 ]; then
  echo "PASS: all foundation gates green."
else
  echo "FAIL: one or more foundation gates failed (see table above)."
fi
exit "$overall"
