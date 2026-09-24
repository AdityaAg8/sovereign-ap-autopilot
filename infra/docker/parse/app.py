"""GPU-backed parse service (PRD §3, §4, §10 M3.1): PaddleOCR-VL-1.6 text spotting for one scanned
page image per POST /parse. The model lives in a child process that loads on the first page and
is stopped by POST /unload, which returns all of its GPU memory (CUDA context included) so the
worker's GPU stage scheduler can hand the card to the extract models (ADR-0013). Only the scan
path calls this; born-digital PDFs are parsed host-side by apx.parse.digital (pdfplumber).
"""

from __future__ import annotations

import asyncio
import io
import multiprocessing as mp
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Annotated, Any, Final, Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

# `torch`/`transformers` are imported lazily, inside the child-process functions that need them,
# not here: they are the parse-service image's own deps (infra/docker/parse/Dockerfile), never
# installed in the host `uv` venv, so this module stays importable for fast host-side unit tests
# (tests/unit/test_parse_service_app.py) with no GPU and no `torch` on the host. The HTTP process
# never initializes CUDA, so stopping the child frees the whole allocation.

MODEL_PATH: Final[str] = "/models/PaddleOCR-VL-1.6"
"""Read-only mount of models/PaddlePaddle/PaddleOCR-VL-1.6 (infra/compose.yaml `parse` service);
never downloaded here -- `make models` resolves and checksums it on the host (PRD C1)."""

GPU_MEM_FRACTION_ENV: Final[str] = "PADDLE_GPU_MEM_FRACTION"
DEFAULT_GPU_MEM_FRACTION: Final[float] = 0.35
"""Fraction of total device memory the model process may allocate (~4.3 GB on a 12 GB card),
comfortably above the ~3.2 GB peak `reports/smoke.md` measured for this model (PRD §2 C2)."""

_SPOTTING_PROMPT: Final[str] = "Spotting:"
_UPSCALE_THRESHOLD_PX: Final[int] = 1500
_MAX_PIXELS: Final[int] = 2048 * 28 * 28
_MIN_PIXELS_FALLBACK: Final[int] = 56 * 56
MAX_NEW_TOKENS_ENV: Final[str] = "PARSE_MAX_NEW_TOKENS"
DEFAULT_MAX_NEW_TOKENS: Final[int] = 1024
"""Decode budget per page. The checkpoint's `config.json` ships `use_cache: false`, which made
decoding crawl at ~1 token/s (reports/parse.md); generation now passes `use_cache=True`, and the
budget is sized from the live measurement in ADR-0013. A page that needs more is cut off (a
partial transcription; Extractor A still sees the page image), never padded or invented."""
_JOIN_S: Final[float] = 20.0

# One recognized text zone is emitted as `<text><|LOC_a|><|LOC_b|><|LOC_c|><|LOC_d|><|LOC_e|>
# <|LOC_f|><|LOC_g|><|LOC_h|>` -- four (x, y) corners of an axis-aligned box, clockwise from
# top-left, each coordinate a 0-1000 bin fraction of the image the model saw. Empirically
# verified against models/PaddlePaddle/PaddleOCR-VL-1.6 on UAE test samples (see reports/parse.md):
# corner 1 is (x0, y0), corner 3 is (x1, y1); corners 2 and 4 are redundant for an axis-aligned
# box. Not part of the model's published docs -- if a newer checkpoint changes this, `_parse_
# spotting_output`'s doctest-style unit test (tests/unit/test_parse_service_app.py) will fail
# loudly rather than silently mis-scoring boxes.
_LOC_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"<\|LOC_(\d+)\|>")
_LOC_RUN_RE: Final[re.Pattern[str]] = re.compile(r"(?:<\|LOC_\d+\|>){8}")


class Engine(Protocol):
    """What the HTTP layer needs from the model process (tests inject a fake)."""

    @property
    def loaded(self) -> bool: ...

    def spot(self, png: bytes) -> tuple[str, float]:
        """Raw `Spotting:` output and peak GPU GB for one page; RuntimeError on failure."""
        ...

    def unload(self) -> None: ...


class ChildProcessEngine:
    """Runs `target` (default `_engine_main`) in a spawned child that owns CUDA. Pages are
    serialized (one at a time on the GPU); the child starts on the first page and `unload`
    stops it and waits for it to exit."""

    def __init__(self, target: Callable[[Connection], None] | None = None) -> None:
        self._target = target or _engine_main
        self._lock = threading.Lock()
        self._proc: BaseProcess | None = None
        self._conn: Connection | None = None

    @property
    def loaded(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

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

    def spot(self, png: bytes) -> tuple[str, float]:
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
            return str(payload), float(peak)

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
    import os

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            fraction = float(os.environ.get(GPU_MEM_FRACTION_ENV, DEFAULT_GPU_MEM_FRACTION))
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        # No `trust_remote_code`: transformers ships `paddleocr_vl` as a native model type; the
        # checkpoint's own modeling files would produce a config/model class mismatch instead
        # (scripts/smoke.py's `_load_and_probe_vlm` hit the same thing first, PRD §10 M0.5).
        processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=torch.bfloat16 if device == "cuda" else "auto"
        )
        model = model.to(device).eval()
        max_new_tokens = int(os.environ.get(MAX_NEW_TOKENS_ENV, DEFAULT_MAX_NEW_TOKENS))
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
            with _vram_peak() as peak_gb:
                text = _run_spotting(processor, model, image, max_new_tokens)
            conn.send(("ok", text, peak_gb()))
        except Exception as exc:  # one bad page must not kill the process
            conn.send(("error", repr(exc), 0.0))


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


app = FastAPI(title="parse", lifespan=_lifespan)


class WireWord(BaseModel):
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)


class WireLine(BaseModel):
    word_indices: tuple[int, ...]


class WireTable(BaseModel):
    cells: tuple[dict[str, object], ...] = ()


class ParseResponse(BaseModel):
    width_px: int
    height_px: int
    words: tuple[WireWord, ...]
    lines: tuple[WireLine, ...]
    tables: tuple[WireTable, ...] = ()
    peak_vram_gb: float


@app.get("/health")
def health() -> dict[str, object]:
    """Up as soon as the HTTP server is; the model loads on the first page (lazy)."""
    return {"status": "ok", "service": "parse", "model": MODEL_PATH, "loaded": get_engine().loaded}


@app.post("/unload")
async def unload() -> dict[str, object]:
    """Stop the model process (no-op when not loaded); returns once its GPU memory is free."""
    engine = get_engine()
    await asyncio.to_thread(engine.unload)
    return {"loaded": engine.loaded}


@app.post("/parse", response_model=ParseResponse)
async def parse_image(image: Annotated[UploadFile, File()]) -> ParseResponse:
    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"not a decodable image: {exc}") from exc
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    try:
        raw_text, peak_gb = await asyncio.to_thread(get_engine().spot, buffer.getvalue())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    zones = _parse_spotting_output(raw_text)

    words: list[WireWord] = []
    lines: list[WireLine] = []
    for text, bbox in zones:
        tokens = text.split()
        if not tokens:
            continue
        indices = list(range(len(words), len(words) + len(tokens)))
        words.extend(WireWord(text=tok, bbox=bbox, confidence=1.0) for tok in tokens)
        lines.append(WireLine(word_indices=tuple(indices)))

    width_px, height_px = pil_image.size
    return ParseResponse(
        width_px=width_px,
        height_px=height_px,
        words=tuple(words),
        lines=tuple(lines),
        tables=(),
        peak_vram_gb=peak_gb,
    )


@contextmanager
def _vram_peak() -> Iterator[Callable[[], float]]:  # pragma: no cover - GPU child only
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    box: list[float] = [0.0]
    yield lambda: box[0]
    if torch.cuda.is_available():
        box[0] = torch.cuda.max_memory_allocated() / 1e9


def _run_spotting(
    processor: Any, model: Any, image: Image.Image, max_new_tokens: int
) -> str:  # pragma: no cover - GPU child only
    import torch

    orig_w, orig_h = image.size
    if orig_w < _UPSCALE_THRESHOLD_PX and orig_h < _UPSCALE_THRESHOLD_PX:
        image = image.resize((orig_w * 2, orig_h * 2), Image.Resampling.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _SPOTTING_PROMPT},
            ],
        }
    ]
    min_pixels = getattr(processor.image_processor, "min_pixels", _MIN_PIXELS_FALLBACK)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": min_pixels, "longest_edge": _MAX_PIXELS}},
    ).to(model.device)

    with torch.no_grad():
        # `use_cache=True` overrides the checkpoint's `use_cache: false` (see
        # DEFAULT_MAX_NEW_TOKENS): without the KV cache every token re-runs the whole prefix.
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    generated = outputs[0][inputs["input_ids"].shape[-1] : -1]
    text: str = processor.decode(generated, skip_special_tokens=False)
    return text


def _parse_spotting_output(raw: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Decode `_SPOTTING_PROMPT` output into `(text, (x0, y0, x1, y1))` zones, coordinates
    normalized to [0, 1]. A line with no trailing 8-token LOC run is skipped (best-effort;
    this path is informational only, PRD §10 M3.1)."""
    zones: list[tuple[str, tuple[float, float, float, float]]] = []
    for entry in raw.split("\n"):
        entry = entry.strip()
        if not entry:
            continue
        match = _LOC_RUN_RE.search(entry)
        if match is None:
            continue
        text = entry[: match.start()].strip()
        nums = [int(n) for n in _LOC_TOKEN_RE.findall(match.group(0))]
        if len(nums) < 8 or not text:
            continue
        x0, y0, x1, y1 = nums[0], nums[1], nums[4], nums[5]
        bbox = (
            max(0.0, min(1.0, x0 / 1000.0)),
            max(0.0, min(1.0, y0 / 1000.0)),
            max(0.0, min(1.0, x1 / 1000.0)),
            max(0.0, min(1.0, y1 / 1000.0)),
        )
        zones.append((text, bbox))
    return zones
