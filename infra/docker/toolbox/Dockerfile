# Linux CUDA toolbox for `make smoke`, evals and training (docs/adr/0006
# addendum): Python 3.12 via uv plus CUDA torch/transformers, run in Docker because Windows
# Smart App Control blocks native DLLs from these wheels on the host. Opt-in (`--profile
# dev`); never part of `make up`. The repo is bind-mounted at /repo at container start, so
# nothing here needs the source tree.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv itself, copied from its own image rather than a curl-piped installer script.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

# The venv lives outside the repo mount point so the host's Windows .venv (a bind-mounted
# /repo/.venv, if one exists) is never shadowed or reused here.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH="/opt/venv/bin:${PATH}"

RUN uv python install 3.12 \
    && uv venv --python 3.12 /opt/venv

# scripts/smoke.py's own runtime deps (httpx, Pillow -- versions kept in sync with
# pyproject.toml by hand, since this image installs the extra GPU stack directly instead of
# running `uv sync` for the whole project) plus the CUDA ML stack the parse, guard and
# embedding models need. `ml/` pins these properly in pyproject.toml in a later milestone.
# Split into separate layers (rather than one chained RUN) so bumping the transformers pin
# below doesn't force re-downloading the much larger torch/CUDA layer.
RUN uv pip install --python /opt/venv/bin/python "httpx>=0.27" "pillow>=11.0"

RUN uv pip install --python /opt/venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Pinned exactly (not `>=5.0.0`): PaddleOCR-VL-1.6's shipped `modeling_paddleocr_vl.py`
# (models/PaddlePaddle/PaddleOCR-VL-1.6, trust_remote_code) only builds against the
# transformers release its authors tested -- newer 5.x releases changed how nested vision
# sub-configs populate `rope_parameters`, and PaddleOCRVisionConfig never gets one, so model
# construction raises `AttributeError: 'PaddleOCRVisionConfig' object has no attribute
# 'rope_parameters'`. Revisit this pin (and re-run `make smoke`) whenever PaddleOCR-VL ships
# an updated modeling file, per the "next verified build" rule.
RUN uv pip install --python /opt/venv/bin/python \
    "transformers==5.0.0" "sentence-transformers>=3.3" "accelerate>=1.0"

WORKDIR /repo
CMD ["/bin/bash"]
