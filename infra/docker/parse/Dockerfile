# GPU image for apx/parse's scan-path service (PRD §3, §4, §10 M3.1): PaddleOCR-VL-1.6 served
# over FastAPI via `transformers`, loaded from the read-only models/ mount, never downloaded
# here. Mirrors infra/docker/toolbox's CUDA/uv/transformers pins (same model, same pin reasons).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# curl: healthcheck and tests/integration/test_parse_service.py's `docker compose exec` probe
# (the service publishes no host port by design, PRD C1, so exec-in is the only way to reach it).
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

# Pinned exactly, not `>=5.0.0`: see infra/docker/toolbox/Dockerfile's comment -- PaddleOCR-VL-
# 1.6's `paddleocr_vl` model type needs this exact transformers release on this host; newer 5.x
# releases break `PaddleOCRVisionConfig` construction. Revisit together with that pin.
RUN uv pip install --python /opt/venv/bin/python "transformers==5.0.0" "accelerate>=1.0"

WORKDIR /app
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
