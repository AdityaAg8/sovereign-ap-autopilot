#!/usr/bin/env bash
# Runs every job of .github/workflows/ci.yml (python, web, headers, gitleaks) on a clean clone of
# a git ref, for when GitHub Actions cannot start jobs (e.g. an account billing block). Writes
# logs/ci-local/<ref>.md and exits non-zero if any job fails; used as `make ci-local REF=<branch>`.
set -uo pipefail

REF="${1:-HEAD}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAFE_REF="$(printf '%s' "$REF" | tr '/:' '__')"
OUT_DIR="$ROOT/logs/ci-local"
WORK="$(mktemp -d)"
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/$SAFE_REF.md"
SHA="$(git -C "$ROOT" rev-parse "$REF")" || { echo "unknown ref $REF"; exit 2; }

git clone --quiet --no-hardlinks "$ROOT" "$WORK/repo" && git -C "$WORK/repo" checkout --quiet "$SHA" \
  || { echo "clone/checkout of $SHA failed"; exit 2; }
cd "$WORK/repo" || exit 2

declare -a ROWS=()
FAILED=0
step() {  # step <job> <name> <command...>
  local job="$1" name="$2"
  shift 2
  local log="$OUT_DIR/$SAFE_REF.$job.log"
  echo "==> [$job] $name"
  if "$@" >>"$log" 2>&1; then
    ROWS+=("| $job | $name | PASS |")
  else
    ROWS+=("| $job | $name | FAIL (see logs/ci-local/$SAFE_REF.$job.log) |")
    FAILED=1
  fi
}

rm -f "$OUT_DIR/$SAFE_REF".*.log
# The pinned PINT AE/SchXslt bundles are sha256-checked on download; reuse the main checkout's
# extraction when present so a flaky network doesn't fail the run (the fetcher skips the network
# once a correctly-shaped extraction exists, as on a CI cache hit).
if [ -d "$ROOT/data/pint_ae" ]; then
  mkdir -p data && cp -r "$ROOT/data/pint_ae" data/pint_ae
fi
export PYTHONUTF8=1
step python "uv sync --locked --all-groups" uv sync --locked --all-groups
step python "ruff check" uv run ruff check .
step python "ruff format --check" uv run ruff format --check .
step python "mypy --strict" uv run python -m mypy
# On Windows the unit suite runs in a Linux container like CI's runner (Smart App Control blocks
# the Playwright driver/Chromium on the host); elsewhere it runs directly.
if [ "${OS:-}" = "Windows_NT" ]; then PYTEST=(bash scripts/pytest_linux.sh); else PYTEST=(uv run python -m pytest); fi
step python "pytest (unit)" "${PYTEST[@]}" -p no:cacheprovider
step python "fetch PINT AE bundles" uv run python -m evals.pint_ae.fetch
step python "pytest -m artifacts" uv run python -m pytest -m artifacts -p no:cacheprovider --no-cov
step web "pnpm install --frozen-lockfile" bash -c "cd web && pnpm install --frozen-lockfile --ignore-workspace"
step web "lint" bash -c "cd web && pnpm lint"
step web "typecheck" bash -c "cd web && pnpm typecheck"
step web "build" bash -c "cd web && pnpm build"
step headers "check_headers --all" uv run python scripts/check_headers.py --all
step gitleaks "gitleaks (pre-commit hook, all files)" uv run pre-commit run gitleaks --all-files

{
  echo "# Local CI run"
  echo
  echo "Ref \`$REF\` at \`$SHA\`, $(date -u +%Y-%m-%dT%H:%MZ), clean clone. Same jobs as .github/workflows/ci.yml."
  echo
  echo "| job | step | result |"
  echo "|---|---|---|"
  printf '%s\n' "${ROWS[@]}"
  echo
  if [ "$FAILED" -eq 0 ]; then echo "**All jobs passed.**"; else echo "**FAILED.**"; fi
} >"$REPORT"
cat "$REPORT"
cd "$ROOT" && rm -rf "$WORK"
exit "$FAILED"
