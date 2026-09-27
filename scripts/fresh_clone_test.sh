#!/usr/bin/env bash
# Fresh-clone check (PRD §10 M5.6): clones a committed ref into a temporary directory and proves
# the tracked tree alone installs, lints, type-checks, passes the unit tests, validates the
# Compose file and builds the apx image -- nothing untracked (planning files, data/, models/,
# a local .venv or .env) is needed. Writes reports/fresh_clone.md. Usage: fresh_clone_test.sh [ref]
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ref="${1:-HEAD}"
sha="$(git -C "$repo" rev-parse --verify "$ref^{commit}")"
work="$(mktemp -d)"
report="$repo/reports/fresh_clone.md"
declare -a rows=()
status=0

cleanup() { rm -rf -- "$work"; }
trap cleanup EXIT

step() {
  local name="$1"
  shift
  local started=$SECONDS
  if (cd "$work/repo" && "$@") >"$work/step.log" 2>&1; then
    rows+=("| $name | pass | $((SECONDS - started)) |")
  else
    rows+=("| $name | **FAIL** | $((SECONDS - started)) |")
    status=1
    echo "---- $name failed; last lines:" >&2
    tail -n 25 "$work/step.log" >&2
  fi
}

echo "==> cloning $sha into $work"
git clone --quiet --no-local "$repo" "$work/repo"
git -C "$work/repo" checkout --quiet --detach "$sha"

step "uv sync --frozen" uv sync --frozen
# --ignore-workspace: a pnpm workspace in a parent directory (e.g. the home dir) must not
# capture the clone.
step "pnpm install --frozen-lockfile" bash -c "cd web && pnpm install --frozen-lockfile --prefer-offline --ignore-workspace"
step "ruff check + format" bash -c "uv run ruff check . && uv run ruff format --check ."
step "mypy --strict" uv run mypy
step "web lint + typecheck" bash -c "cd web && pnpm lint && pnpm typecheck"
# pyproject's addopts already deselect gpu/integration/artifacts/external/perf tests.
step "unit tests" uv run pytest -q -p no:cacheprovider --no-cov tests/unit
# Compose has no default secrets (REVIEW C-02), so generate them the way `make up` does first.
step "compose config" bash -c "uv run python scripts/init_env.py && cd infra && docker compose --env-file ../.env config --quiet"
step "apx image build" docker build -q -f infra/docker/apx/Dockerfile -t apx-fresh-clone-check .

mkdir -p "$(dirname "$report")"
{
  echo "# Fresh-clone check"
  echo
  echo "Commit \`$sha\`, cloned into an empty directory with no untracked files."
  echo
  echo "| Step | Result | Seconds |"
  echo "|---|---|---|"
  printf '%s\n' "${rows[@]}"
} >"$report"
echo "==> wrote $report"
exit "$status"
