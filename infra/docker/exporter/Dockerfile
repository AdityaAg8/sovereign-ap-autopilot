# Extractor B export image (ADR-0009 §1.7, `make export-b`): the trainer image plus a pinned
# llama.cpp for `convert_hf_to_gguf.py` (bf16 text GGUF and f16 mmproj) and `llama-quantize`
# (Q4_K_M). Built FROM the trainer image so the merge runs on the exact training stack, while
# the trainer image (and so the run fingerprint's image digest) stays unchanged.
ARG TRAINER_IMAGE=sovereign-ap/trainer:2026.9.11
FROM ${TRAINER_IMAGE}

# llama.cpp is pinned to the build the served llama-swap image runs (infra/compose.yaml,
# v257-cuda-b11096), so the converter writes tensors that exact llama-server reads. The tag is
# cross-checked against its commit; a moved tag fails the build instead of drifting.
ARG LLAMA_CPP_TAG=b11096
ARG LLAMA_CPP_COMMIT=c550d2f60bde72df19fcef1fef627895095b8ba8

RUN apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMA_CPP_TAG}" https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && test "$(git -C /opt/llama.cpp rev-parse HEAD)" = "${LLAMA_CPP_COMMIT}" \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
    && cmake --build /opt/llama.cpp/build --target llama-quantize -j "$(nproc)" \
    && install -m 0755 /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize \
    && rm -rf /opt/llama.cpp/build /opt/llama.cpp/.git

# The converter's own reader/writer from the same commit. Its other requirements (numpy,
# sentencepiece, protobuf, safetensors, transformers, torch) already come from the trainer
# image; --no-deps keeps those pins untouched.
RUN uv pip install --python /opt/venv/bin/python --no-deps /opt/llama.cpp/gguf-py

ENV LLAMA_CPP_DIR=/opt/llama.cpp \
    LLAMA_CPP_COMMIT=${LLAMA_CPP_COMMIT}
CMD ["python", "-m", "ml.extractor_b.export"]
