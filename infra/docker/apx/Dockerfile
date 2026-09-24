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

COPY apx ./apx
COPY factory ./factory
COPY db ./db
COPY alembic.ini ./
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 apx && mkdir -p /app/data && chown apx /app/data
USER apx
