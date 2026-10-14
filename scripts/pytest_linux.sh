#!/usr/bin/env bash
# Runs pytest for this checkout inside a Linux container, the way CI's ubuntu runner does. Used
# by `make test` and scripts/ci_local.sh on Windows hosts, where Smart App Control blocks the
# unsigned Playwright driver and Chromium binaries that the factory render tests launch. Pytest
# arguments pass through; the checkout is copied (read-only mount), never modified.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
SRC="$ROOT"
if command -v cygpath >/dev/null 2>&1; then
  SRC="$(cygpath -w "$ROOT")"
fi

# Inside the container: copy the checkout (minus host-only and large paths; data/pint_ae is kept
# for the artifacts tests), install what CI installs, run pytest with the given arguments.
# GNU tar exits 1 when a file changes while it is read (e.g. .git under a live editor); only
# exit codes above 1 are errors.
read -r -d '' INNER <<'EOF' || true
set -euo pipefail
mkdir -p /work
set +e
tar -C /src --exclude=./.venv --exclude=./models --exclude=./logs --exclude=./node_modules \
  --exclude=./web/node_modules --exclude=./web/.next --exclude='./.*/worktrees' --exclude=./.env \
  --exclude='./data/*' --exclude='*.gguf' --exclude='*.safetensors' --exclude=__pycache__ \
  -cf - . | tar -C /work -xf -
status=${PIPESTATUS[0]}
set -e
[ "$status" -le 1 ] || { echo "copying the checkout failed (tar $status)"; exit 2; }
if [ -d /src/data/pint_ae ]; then mkdir -p /work/data && cp -r /src/data/pint_ae /work/data/; fi
cd /work
export UV_PROJECT_ENVIRONMENT=/venv UV_LINK_MODE=copy PYTHONUTF8=1
apt-get update -qq >/dev/null && apt-get install -y -qq git >/dev/null
git config --global --add safe.directory /work
uv sync --locked --all-groups --quiet
uv run playwright install --with-deps --only-shell chromium >/tmp/playwright.log 2>&1 \
  || { tail -20 /tmp/playwright.log; exit 3; }
exec uv run python -m pytest "$@"
EOF

MSYS_NO_PATHCONV=1 exec docker run --rm \
  -v "$SRC:/src:ro" \
  -v apx-test-uvcache:/root/.cache/uv \
  -v apx-test-playwright:/root/.cache/ms-playwright \
  "$IMAGE" bash -c "$INNER" pytest "$@"
