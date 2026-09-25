# The apx application image (PRD §10 M5.6): one slim, CPU-only Python 3.12 environment for the
# `api` and `worker` Compose services. Models, OCR and Postgres stay in their own containers and
# are reached over the internal `app` network; dependencies come from uv.lock exactly.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, in their own layer, so a source edit doesn't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# The M6.4 injection guard runs on CPU in the worker (docs/adr/0017-injection-guard-runtime.md):
# CPU-only torch at the toolbox's version (the checkpoint was trained and scored with 2.6.0)
# and the transformers release its config.json was saved with (5.0.0). Outside uv.lock on
# purpose: the Windows host venv never installs torch (Smart App Control, ADR-0006), so these
# pins live in this image only; the sync below is `--inexact` so it keeps them. Build-time
# network only: the checkpoint is a read-only mount and HF is forced offline.
RUN uv pip install --python /opt/venv/bin/python "torch==2.6.0" \
        --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --python /opt/venv/bin/python "transformers==5.0.0"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY apx ./apx
COPY factory ./factory
COPY db ./db
COPY alembic.ini ./
RUN uv sync --frozen --no-dev --inexact

RUN useradd --create-home --uid 10001 apx && mkdir -p /app/data && chown apx /app/data
USER apx
