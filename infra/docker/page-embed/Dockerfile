# GPU image for the fraud page-embedding service:
# Qwen3-VL-Embedding-2B over FastAPI via `transformers`, loaded from the read-only models/ mount,
# never downloaded here. Same CUDA/uv/torch layout as infra/docker/parse.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# curl: the ADR-0013 probe (`docker compose exec page-embed curl ...`); no host port by design.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH="/opt/venv/bin:${PATH}"

RUN uv python install 3.12 \
    && uv venv --python 3.12 /opt/venv

RUN uv pip install --python /opt/venv/bin/python \
    "fastapi==0.118.0" "uvicorn==0.37.0" "python-multipart==0.0.20" "pillow>=11.0"

RUN uv pip install --python /opt/venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# The checkpoint's config.json was written by transformers 4.57.1, the first release with the
# `qwen3_vl` model type; pinned to it rather than the parse image's 5.0.0 (PaddleOCR-VL's pin).
RUN uv pip install --python /opt/venv/bin/python "transformers==4.57.1" "accelerate>=1.0"

WORKDIR /app
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
