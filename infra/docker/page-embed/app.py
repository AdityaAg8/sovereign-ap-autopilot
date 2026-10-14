"""GPU-backed page-embedding service: Qwen3-VL-Embedding-2B turns one rendered page image per
POST /embed into an L2-normalized 2048-d vector for fraud near-duplicate detection
(`apx.llm.embedding.HttpPageEmbedder` is the client). The model lives in a child process that
loads on the first page and is stopped by POST /unload, which returns all of its GPU memory,
so the worker's GPU stage scheduler can hand the card on (ADR-0013). Served here rather than
by llama-swap: the pinned checkpoint is safetensors, with no GGUF."""

from __future__ import annotations

import asyncio
import io
import math
import multiprocessing as mp
import os
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Annotated, Any, Final, Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

# `torch`/`transformers` are imported inside the child-process functions only: they are this
# image's own deps (infra/docker/page-embed/Dockerfile), not the host venv's, so this module
# imports cleanly for host-side unit tests (tests/unit/test_page_embed_service_app.py). The
# HTTP process never initializes CUDA, so stopping the child frees the whole allocation.

MODEL_PATH: Final[str] = "/models/Qwen3-VL-Embedding-2B"
"""Read-only mount of models/Qwen/Qwen3-VL-Embedding-2B (infra/compose.yaml `page-embed`),
pinned in infra/models.lock.json; never downloaded here (`make models` fetches on the host)."""

GPU_MEM_FRACTION_ENV: Final[str] = "PAGE_EMBED_GPU_MEM_FRACTION"
DEFAULT_GPU_MEM_FRACTION: Final[float] = 0.45
"""Cap on the process's CUDA allocator (~5.5 GB of 12 GB): bf16 weights are 4.26 GB. Not yet
measured on the card (ADR-0013 needs table marks page-similarity unmeasured)."""

INSTRUCTION: Final[str] = "Represent the user's input."
"""The model's default instruction (its scripts/qwen3_vl_embedding.py). Changing it changes
every vector, so stored `page_embeddings` rows would need re-embedding."""

_PATCH: Final[int] = 32
MIN_PIXELS: Final[int] = 4 * _PATCH * _PATCH
MAX_PIXELS: Final[int] = 1800 * _PATCH * _PATCH
"""The reference embedder's image bounds (~1.8 MP, at most 1800 visual tokens)."""
_JOIN_S: Final[float] = 20.0

MAX_INPUT_PIXELS: Final[int] = 40_000_000
"""Largest upload decoded at all (an A4 page at 600 dpi is ~35 MP). Bigger is refused with 422
before any pixel data is read, so a decompression bomb cannot exhaust the service's RAM."""
Image.MAX_IMAGE_PIXELS = MAX_INPUT_PIXELS  # Pillow's own bomb guard, at the same bound

_IGNORABLE_KEY_PREFIXES: Final[tuple[str, ...]] = ("lm_head.",)
"""The generation head, never used for embeddings: this checkpoint ties it to the input
embeddings and ships no tensor for it, and a re-export might ship one."""


def check_loading_info(info: dict[str, Any]) -> None:
    """Refuse a load where any base-model weight was left at its random initialization (a
    missing key), a checkpoint tensor went unused (an unexpected key: a prefix mismatch would
    otherwise load nothing and embed noise), or a shape disagreed. The lm_head is ignored. `info` is
    `from_pretrained(..., output_loading_info=True)`'s second value."""
    missing = sorted(
        key for key in info.get("missing_keys") or () if not key.startswith(_IGNORABLE_KEY_PREFIXES)
    )
    unexpected = sorted(
        key
        for key in info.get("unexpected_keys") or ()
        if not key.startswith(_IGNORABLE_KEY_PREFIXES)
    )
    mismatched = list(info.get("mismatched_keys") or ())
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "checkpoint does not match Qwen3VLForConditionalGeneration: "
            f"missing={missing[:5]} ({len(missing)}), unexpected={unexpected[:5]} "
            f"({len(unexpected)}), mismatched={mismatched[:5]} ({len(mismatched)})"
        )


class Engine(Protocol):
    """What the HTTP layer needs from the model process (tests inject a fake)."""

    @property
    def loaded(self) -> bool: ...

    @property
    def peak_vram_gb(self) -> float: ...

    def embed(self, png: bytes) -> list[float]:
        """The page's normalized embedding; RuntimeError on failure."""
        ...

    def unload(self) -> None: ...


def reference_size(height: int, width: int) -> tuple[int, int]:
    """(height, width) the reference embedder resizes a page to before the processor (its
    `qwen_vl_utils` smart_resize: multiples of 32, area within [MIN_PIXELS, MAX_PIXELS], aspect
    kept). The pages are pre-resized here and the processor's own resize is off: measured, the
    processor ignored a per-call `longest_edge` and capped pages at 1260 tokens instead of the
    reference's 1750, so vectors would not have matched the model's published behaviour."""
    if height <= 0 or width <= 0:
        raise ValueError(f"empty image {width}x{height}")
    h_bar = max(_PATCH, round(height / _PATCH) * _PATCH)
    w_bar = max(_PATCH, round(width / _PATCH) * _PATCH)
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h_bar = max(_PATCH, math.floor(height / beta / _PATCH) * _PATCH)
        w_bar = max(_PATCH, math.floor(width / beta / _PATCH) * _PATCH)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / _PATCH) * _PATCH
        w_bar = math.ceil(width * beta / _PATCH) * _PATCH
    return h_bar, w_bar


def last_token_index(attention_mask: Sequence[int]) -> int:
    """Position of the last attended token (the model pools the last token, its
    `1_Pooling/config.json` `lasttoken` mode). Raises ValueError on an all-padding row."""
    for index in range(len(attention_mask) - 1, -1, -1):
        if attention_mask[index]:
            return index
    raise ValueError("attention mask has no attended token")


class ChildProcessEngine:
    """Runs `target` (default `_engine_main`) in a spawned child that owns CUDA. Pages are
    serialized; the child starts on the first page and `unload` stops it and waits for it."""

    def __init__(self, target: Callable[[Connection], None] | None = None) -> None:
        self._target = target or _engine_main
        self._lock = threading.Lock()
        self._proc: BaseProcess | None = None
        self._conn: Connection | None = None
        self._peak_gb = 0.0

    @property
    def loaded(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    @property
    def peak_vram_gb(self) -> float:
        """Highest CUDA-reserved memory any page needed since the service started."""
        return self._peak_gb

    def _start(self) -> Connection:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=self._target, args=(child,), daemon=True)
        proc.start()
        child.close()
        try:
            status = parent.recv()  # ("ready", None) or ("error", message)
        except EOFError as exc:
            proc.join(_JOIN_S)
            raise RuntimeError(f"model process exited while loading: {proc.exitcode}") from exc
        if status[0] != "ready":
            proc.join(_JOIN_S)
            raise RuntimeError(f"model process failed to load: {status[1]}")
        self._proc, self._conn = proc, parent
        return parent

    def embed(self, png: bytes) -> list[float]:
        with self._lock:
            conn = self._conn if self.loaded and self._conn is not None else self._start()
            try:
                conn.send(png)
                kind, payload, peak = conn.recv()
            except (EOFError, OSError) as exc:
                self._stop()
                raise RuntimeError(f"model process died: {exc!r}") from exc
            if kind != "ok":
                raise RuntimeError(str(payload))
            self._peak_gb = max(self._peak_gb, float(peak))
            return [float(x) for x in payload]

    def _stop(self) -> None:
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if conn is not None:
            with suppress(OSError, ValueError):
                conn.send(None)
            conn.close()
        if proc is not None:
            proc.join(_JOIN_S)
            if proc.is_alive():
                proc.kill()
                proc.join(_JOIN_S)

    def unload(self) -> None:
        with self._lock:
            self._stop()


def _engine_main(conn: Connection) -> None:  # pragma: no cover - runs in the GPU child only
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            fraction = float(os.environ.get(GPU_MEM_FRACTION_ENV, DEFAULT_GPU_MEM_FRACTION))
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, local_files_only=True, padding_side="right"
        )
        # The checkpoint's tensors are named `model.visual.*` / `model.language_model.*`, i.e. a
        # Qwen3VLForConditionalGeneration. Loading Qwen3VLModel directly does not strip that
        # prefix (measured: every weight "missing", so a silent random init), so load the full
        # class, which matches key for key, and embed with its `.model` (the same module the
        # reference Qwen3VLForEmbedding wraps). The tied lm_head is never used.
        full, info = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            dtype=torch.bfloat16 if device == "cuda" else "auto",
            output_loading_info=True,
        )
        check_loading_info(info)
        model = full.model.to(device).eval()
        del full
    except Exception as exc:  # reported to the parent as a load failure, then exit
        conn.send(("error", repr(exc)))
        return
    conn.send(("ready", None))
    while True:
        png = conn.recv()
        if png is None:
            return
        try:
            image = Image.open(io.BytesIO(png)).convert("RGB")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            vector = _embed(processor, model, image)
            peak = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
            conn.send(("ok", vector, peak))
        except Exception as exc:  # one bad page must not kill the process
            conn.send(("error", repr(exc), 0.0))


def _embed(processor: Any, model: Any, image: Image.Image) -> list[float]:  # pragma: no cover
    import torch
    import torch.nn.functional as F

    height, width = reference_size(image.height, image.width)
    image = image.resize((width, height))  # Pillow's default filter, as the reference does
    messages = [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        do_resize=False,
    ).to(model.device)
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state
    last = last_token_index(inputs["attention_mask"][0].tolist())
    pooled = F.normalize(hidden[0, last].float(), p=2, dim=-1)
    return [float(x) for x in pooled.cpu().tolist()]


_state: dict[str, Engine] = {}


def get_engine() -> Engine:
    engine = _state.get("engine")
    if engine is None:
        engine = _state["engine"] = ChildProcessEngine()
    return engine


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    engine = _state.pop("engine", None)
    if engine is not None:
        engine.unload()


app = FastAPI(title="page-embed", lifespan=_lifespan)


class EmbedResponse(BaseModel):
    """Matches `apx.llm.embedding._WireEmbedResponse` exactly (it forbids extra fields)."""

    embedding: tuple[float, ...] = Field(min_length=1)


@app.get("/health")
def health() -> dict[str, object]:
    """Up as soon as the HTTP server is; the model loads on the first page (lazy).
    `peak_vram_gb` is the ADR-0013 probe's in-process figure (CUDA reserved, no context)."""
    engine = get_engine()
    return {
        "status": "ok",
        "service": "page-embed",
        "model": MODEL_PATH,
        "loaded": engine.loaded,
        "peak_vram_gb": engine.peak_vram_gb,
    }


@app.post("/unload")
async def unload() -> dict[str, object]:
    """Stop the model process (no-op when not loaded); returns once its GPU memory is free."""
    engine = get_engine()
    await asyncio.to_thread(engine.unload)
    return {"loaded": engine.loaded}


@app.post("/embed", response_model=EmbedResponse)
async def embed_image(image: Annotated[UploadFile, File()]) -> EmbedResponse:
    raw = await image.read()
    try:
        with Image.open(io.BytesIO(raw)) as decoded:
            width, height = decoded.size
            if width * height > MAX_INPUT_PIXELS:
                raise HTTPException(
                    status_code=422,
                    detail=f"image is {width}x{height}, over {MAX_INPUT_PIXELS} pixels",
                )
            decoded.convert("RGB").save(buffer := io.BytesIO(), format="PNG")
    except (OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail=f"not a decodable image: {exc}") from exc
    try:
        vector = await asyncio.to_thread(get_engine().embed, buffer.getvalue())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return EmbedResponse(embedding=tuple(vector))
