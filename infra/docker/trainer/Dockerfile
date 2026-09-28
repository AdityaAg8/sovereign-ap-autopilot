# Extractor B trainer image (ADR-0009 §1.5, ADR-0011): CUDA devel base (Triton compiles kernels),
# the project's locked deps via uv, plus Unsloth and the transformers/TRL versions it supports,
# pinned exactly. Separate from the toolbox because PaddleOCR-VL pins transformers==5.0.0, which
# Unsloth excludes. The repo is bind-mounted at /repo at run time (infra/compose.yaml `trainer`).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN uv python install 3.12 && uv venv --python 3.12 /opt/venv

# Project runtime + dev deps (pytest for `make test-trainer`) from the lockfile; the project
# itself is not installed, since /repo is mounted and on PYTHONPATH.
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# CUDA 12.8 wheels; torchvision 0.26 is the release built against torch 2.11.
RUN uv pip install --python /opt/venv/bin/python "torch==2.11.0" "torchvision==0.26.0" \
    --index-url https://download.pytorch.org/whl/cu128

# Unsloth 2026.9.11 requires transformers <=5.5.0 (not 5.0.0/5.1.0) and trl <=0.24.0.
# ADR-0011 amendment 1 records the move to transformers v5. `tests/trainer/
# test_private_api_pins.py` (`make test-trainer`) must pass after any bump here.
RUN uv pip install --python /opt/venv/bin/python \
    "unsloth==2026.9.11" "unsloth_zoo==2026.9.7" "transformers==5.5.0" "trl==0.24.0" \
    "peft==0.21.0" "accelerate==1.15.0" "bitsandbytes==0.50.2" "mlflow-skinny==3.7.0"

# Qwen3.5's Gated DeltaNet layers (3 of every 4) run on flash-linear-attention's Triton kernels and
# causal-conv1d's CUDA kernels when both import; without them transformers silently falls back to
# a slower, more memory-hungry PyTorch reference path (ADR-0011 amendment 6), and
# `ml.extractor_b.kernels` refuses to train on it. causal-conv1d ships only an sdist: it is built
# here against the torch above for the RTX 4080 Laptop (sm_89), never fetched as a prebuilt binary.
RUN uv pip install --python /opt/venv/bin/python \
    "flash-linear-attention==0.5.2" "fla-core==0.5.2" "einops==0.8.2" \
    "setuptools==84.0.0" "wheel==0.48.0" "ninja==1.13.2" "packaging==26.3" \
    && TORCH_CUDA_ARCH_LIST="8.9" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 \
    uv pip install --python /opt/venv/bin/python --no-build-isolation "causal-conv1d==1.7.0"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    UNSLOTH_DISABLE_STATISTICS=1 DO_NOT_TRACK=1 \
    PYTHONPATH=/repo
WORKDIR /repo
CMD ["python", "-m", "ml.extractor_b.train", "--resume", "auto"]
