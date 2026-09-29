"""GPU-backed parse service (PRD §3, §4, §10 M3.1): PaddleOCR-VL-1.6 text spotting for one scanned
page image per POST /parse. The model lives in a child process that loads on the first page and
is stopped by POST /unload, which returns all of its GPU memory (CUDA context included) so the
worker's GPU stage scheduler can hand the card to the extract models (ADR-0013). A page whose
decoding loops or runs out of budget is re-read as overlapping horizontal strips with a loop guard
(ADR-0013 amendment 2); an unfinished strip is re-read as two half-bands and the merged zones are
put in geometric reading order (amendment 4). Only the scan path calls this; born-digital PDFs are
parsed host-side by apx.parse.digital (pdfplumber). Every route but /health needs the shared
service token, browser requests are refused, and page images are pixel-capped before decoding
(REVIEW N-06).
"""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import math
import multiprocessing as mp
import os
import re
import sys
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager, suppress
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Annotated, Any, Final, Literal, NamedTuple, Protocol

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
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
DEFAULT_MAX_NEW_TOKENS: Final[int] = 3072
"""Decode budget per page (ADR-0013 amendment 1). A dense invoice table page needs ~1500-2100
tokens (8 LOC tokens per text zone); the old 1024 cut every such page after ~65 zones. The
checkpoint's `config.json` ships `use_cache: false`, which made decoding crawl at ~1 token/s
(reports/parse.md), so generation passes `use_cache=True`. A page that still needs more is cut
off and reported as `truncated`, never padded or invented."""
REPETITION_PENALTY: Final[float] = 1.05
"""ADR-0013 amendment 1: without it the model fell into loops ("jìjìjì...") that spent the whole
budget; 1.05 let dense table pages finish in the live probe. A loop it cannot break is detected
(`_looped`) and the page reported as `truncated`."""
_LOOP_RE: Final[re.Pattern[str]] = re.compile(r"(.{1,16}?)\1{15,}", re.DOTALL)
"""A short cycle repeated 16+ times in a row: a decoding loop, not text on an invoice."""
_JOIN_S: Final[float] = 20.0

# Loop guard and strip re-read (ADR-0013 amendment 2). PaddleOCR-VL-1.6 loops inside one zone's
# text on Arabic-first scans ("ذ.م.م.م.م...", "3000000...", "jìjì..."), usually within the first
# ~20 zones, and then spends the whole page budget on it. Measured on the 32 probe pages (16
# looping, 16 finished): every finished page's zones are <= 60 tokens with no alnum cycle
# repeating over more than 10 tokens; every looping page matches the rule below within 37-967
# tokens.
LOOP_MAX_PERIOD: Final[int] = 32
"""Longest token cycle the guard looks for (one looping page repeats a 14-token phrase)."""
LOOP_MIN_SPAN: Final[int] = 24
"""A cycle is a loop once the zone's last max(24, 3 x period) tokens repeat it exactly."""
LOOP_ZONE_CAP: Final[int] = 256
"""A zone this long (tokens after its last LOC token) is runaway text even with no exact cycle."""
LOC_RUN: Final[int] = 8
STRIP_COUNT: Final[int] = 4
STRIP_OVERLAP_FRAC: Final[float] = 0.06
"""Overlap between adjacent strips as a fraction of the page height (~100 px, ~4 text lines on a
1683 px page), so every text line lies wholly inside at least one strip."""
STRIP_MAX_NEW_TOKENS: Final[int] = 1024
"""Budget per strip: a dense page needs ~2100 tokens in all, so ~600 per strip plus overlap."""
STRIP_MAX_CLOSURES: Final[int] = 4
"""Loops a strip may close before it is stopped (and reported truncated)."""
PAGE_DEADLINE_S: Final[float] = 90.0
"""Wall-clock bound on one page's decoding, first pass plus strips, under the 120 s client
timeout (apx.parse.config) with room for the ~12 s lazy model load on a cold first page."""
MIN_STRIP_PASS_S: Final[float] = 10.0
"""A strip pass (or half-band re-read) that would start with less time than this left is not
attempted."""

# Arabic-script blocks and the RTL threshold, the same as apx.parse.digital's (this image does not
# ship `apx`); used to order a merged row's zones in its reading direction (amendment 4).
_ARABIC_BLOCKS: Final[tuple[tuple[int, int], ...]] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_RTL_LINE_THRESHOLD: Final[float] = 0.3

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


class Spotting(NamedTuple):
    """One page's raw `Spotting:` output and whether decoding stopped at the token budget."""

    text: str
    hit_limit: bool


class Engine(Protocol):
    """What the HTTP layer needs from the model process (tests inject a fake)."""

    @property
    def loaded(self) -> bool: ...

    def spot(self, png: bytes) -> tuple[Spotting, float]:
        """`Spotting:` output and peak GPU GB for one page; RuntimeError on failure."""
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

    def spot(self, png: bytes) -> tuple[Spotting, float]:
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
            text, hit_limit = payload
            return Spotting(str(text), bool(hit_limit)), float(peak)

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
        vocab = _Vocab.from_processor(processor, model)
    except Exception as exc:  # reported to the parent as a load failure, then exit
        conn.send(("error", repr(exc)))
        return
    conn.send(("ready", None))

    def run(image: Image.Image, budget: int, mode: GuardMode, max_time: float | None) -> Pass:
        return _generate(processor, model, vocab, image, budget, mode, max_time)

    while True:
        png = conn.recv()
        if png is None:
            return
        try:
            image = Image.open(io.BytesIO(png)).convert("RGB")
            with _vram_peak() as peak_gb:
                spotting, stats = spot_page(image, run, budget=max_new_tokens)
            print(json.dumps({"page": stats._asdict()}), file=sys.stderr, flush=True)
            conn.send(("ok", tuple(spotting), peak_gb()))
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


# No /docs, /redoc or /openapi.json: every route but /health needs the service token anyway.
app = FastAPI(title="parse", lifespan=_lifespan, docs_url=None, redoc_url=None, openapi_url=None)

SERVICE_TOKEN_ENV: Final[str] = "PARSE_SERVICE_TOKEN"
"""Shared secret every caller but the healthcheck sends as `Authorization: Bearer <token>`
(REVIEW N-06): the gateway exposes this service on 127.0.0.1:8010, where any local process or
a browser page could otherwise reach it. Unset or empty means every guarded request is a 503."""
_OPEN_ROUTES: Final[frozenset[tuple[str, str]]] = frozenset({("GET", "/health")})
_FETCH_SITES_ALLOWED: Final[frozenset[str]] = frozenset({"same-origin", "none"})

MAX_IMAGE_PIXELS: Final[int] = 100_000_000
"""Most pixels a page image may declare in its header, checked before it is decoded (REVIEW
N-06). Mirrors apx.common.raster.MAX_IMAGE_PIXELS, which this image cannot import (its build
context is this directory); tests/unit/test_parse_service_app.py asserts the two are equal."""
_IMAGE_FORMATS: Final[frozenset[str]] = frozenset({"PNG", "JPEG"})


def refusal(method: str, path: str, headers: Mapping[str, str]) -> tuple[int, str] | None:
    """`(status, reason)` when a request must be refused, else None. A browser request (any
    `Origin`, or `Sec-Fetch-Site` other than same-origin/none) is a 403, so a web page can never
    drive this service; any route but GET /health then needs the bearer token (401 when missing
    or wrong, compared in constant time; 503 when the service has none configured)."""
    if (method, path) in _OPEN_ROUTES:
        return None
    site = headers.get("sec-fetch-site")
    if "origin" in headers or (site is not None and site not in _FETCH_SITES_ALLOWED):
        return 403, "browser requests are refused"
    expected = os.environ.get(SERVICE_TOKEN_ENV, "")
    if not expected:
        return 503, f"{SERVICE_TOKEN_ENV} is not configured"
    scheme, _, presented = headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        presented.strip().encode(), expected.encode()
    ):
        return 401, "missing or invalid service token"
    return None


@app.middleware("http")
async def _refuse_untrusted(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Runs before the body is read, so a refused request never uploads or decodes anything."""
    refused = refusal(request.method, request.url.path, request.headers)
    if refused is not None:
        return JSONResponse({"detail": refused[1]}, status_code=refused[0])
    return await call_next(request)


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
    truncated: bool
    """Transcription stopped early (decode budget reached or a repetition loop), so text on the
    page may be missing from `words` (ADR-0013 amendment 1). After a strip re-read, true only when
    some strip still did not finish (amendment 2)."""


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
        # `Image.open` reads the header only; nothing is decoded until `convert`.
        probe = Image.open(io.BytesIO(raw))
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"not a decodable image: {exc}") from exc
    if probe.format not in _IMAGE_FORMATS:
        raise HTTPException(status_code=422, detail=f"not a PNG or JPEG image: {probe.format}")
    width_px, height_px = probe.size
    if width_px * height_px > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"image is {width_px} x {height_px} px; at most {MAX_IMAGE_PIXELS} pixels",
        )
    try:
        pil_image = probe.convert("RGB")
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"not a decodable image: {exc}") from exc
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    try:
        spotting, peak_gb = await asyncio.to_thread(get_engine().spot, buffer.getvalue())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    zones = _parse_spotting_output(spotting.text)

    words: list[WireWord] = []
    lines: list[WireLine] = []
    for text, bbox in zones:
        tokens = text.split()
        if not tokens:
            continue
        indices = list(range(len(words), len(words) + len(tokens)))
        words.extend(WireWord(text=tok, bbox=bbox, confidence=1.0) for tok in tokens)
        lines.append(WireLine(word_indices=tuple(indices)))

    return ParseResponse(
        width_px=width_px,
        height_px=height_px,
        words=tuple(words),
        lines=tuple(lines),
        tables=(),
        peak_vram_gb=peak_gb,
        truncated=spotting.hit_limit or _looped(spotting.text),
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


GuardMode = Literal["first", "strip"]
"""`first`: the unchanged full-page pass; the guard only stops decoding at a loop. `strip`: a
strip re-read; the guard closes a looping zone (forces its LOC run) and decoding goes on."""


class Pass(NamedTuple):
    """One `model.generate` call's result, loop tails already cut out of `text`."""

    text: str
    finished: bool
    """Decoding ended on EOS (not on the token budget, a loop stop or the deadline)."""
    tokens: int
    loops: int


class LoopEvent(NamedTuple):
    """A loop the guard saw: generated-token index where its periodic run starts, and its period
    (0 when the zone-length cap fired, in which case the run starts at the zone's start)."""

    run_start: int
    period: int


def find_loop(
    zone: Sequence[int], alnum: Callable[[Sequence[int]], bool]
) -> tuple[int, int] | None:
    """`(run_start, period)` when `zone` (the tokens after the last LOC token) ends in a decoding
    loop: its last max(LOOP_MIN_SPAN, 3 x p) tokens repeat a cycle of p <= LOOP_MAX_PERIOD tokens
    whose text has a letter or digit, or it is longer than LOOP_ZONE_CAP. `run_start` is where
    the periodic run begins in `zone` (0 for the cap). None otherwise."""
    n = len(zone)
    for period in range(1, LOOP_MAX_PERIOD + 1):
        span = max(LOOP_MIN_SPAN, 3 * period)
        if n < span:
            break
        periodic = all(zone[n - k] == zone[n - k - period] for k in range(1, span - period + 1))
        if periodic and alnum(zone[n - period :]):
            start = n - span
            while start > 0 and zone[start - 1] == zone[start - 1 + period]:
                start -= 1
            return start, period
    if n > LOOP_ZONE_CAP:
        return 0, 0
    return None


def _zone_start(ids: Sequence[int], is_loc: Callable[[int], bool]) -> int:
    """Index of the first token after the last LOC token, looking back at most LOOP_ZONE_CAP + 1
    tokens (a longer zone already trips the cap, so the window always holds the whole zone)."""
    floor = max(0, len(ids) - LOOP_ZONE_CAP - 1)
    for i in range(len(ids) - 1, floor - 1, -1):
        if is_loc(ids[i]):
            return i + 1
    return floor


def _trailing_locs(ids: Sequence[int], is_loc: Callable[[int], bool]) -> int:
    count = 0
    for token in reversed(ids):
        if not is_loc(token) or count == LOC_RUN:
            break
        count += 1
    return count


class LoopGuard:
    """Per-`generate` state machine, stepped once per decoded token with the tokens generated so
    far. With `close=False` it only reports `stop` at the first loop. With `close=True` it answers
    `force` (only LOC tokens may follow until the zone's 8-token LOC run is out) so the looping
    zone ends and decoding moves on, and `stop` once more than `max_closures` loops were seen."""

    def __init__(
        self,
        is_loc: Callable[[int], bool],
        alnum: Callable[[Sequence[int]], bool],
        *,
        close: bool,
        max_closures: int = STRIP_MAX_CLOSURES,
    ) -> None:
        self._is_loc = is_loc
        self._alnum = alnum
        self._close = close
        self._max_closures = max_closures
        self._forcing = False
        self.events: list[LoopEvent] = []
        self.stopped = False

    def step(self, ids: Sequence[int]) -> Literal["ok", "force", "stop"]:
        if self.stopped:
            return "stop"
        if self._forcing:
            if _trailing_locs(ids, self._is_loc) < LOC_RUN:
                return "force"
            self._forcing = False
        start = _zone_start(ids, self._is_loc)
        found = find_loop(ids[start:], self._alnum)
        if found is None:
            return "ok"
        self.events.append(LoopEvent(start + found[0], found[1]))
        if not self._close or len(self.events) > self._max_closures:
            self.stopped = True
            return "stop"
        self._forcing = True
        return "force"


def _drop_partial_word(head: str) -> str:
    """Drop the trailing word of `head`'s last zone (text after its last LOC token and last
    whitespace): the loop began inside it, so it is not a word that was on the page."""
    locs = list(_LOC_TOKEN_RE.finditer(head))
    cut = locs[-1].end() if locs else 0
    return head[:cut] + re.sub(r"\S+$", "", head[cut:])


def assemble(
    ids: Sequence[int],
    events: Sequence[LoopEvent],
    decode: Callable[[Sequence[int]], str],
    is_loc: Callable[[int], bool],
) -> str:
    """Decoded text of `ids` with every loop cut out: from each event's run start to the next LOC
    token (the forced end of that zone) or to the end of output. A word the loop began inside is
    dropped too. With no events this is exactly `decode(ids)`."""
    parts: list[str] = []
    pos = 0
    for event in events:
        start = max(event.run_start, pos)
        end = next((i for i in range(start, len(ids)) if is_loc(ids[i])), len(ids))
        head = decode(ids[pos:start])
        if start < len(ids) and not decode(ids[pos : start + 1])[len(head) :][:1].isspace():
            head = _drop_partial_word(head)
        parts.append(head)
        pos = end
    parts.append(decode(ids[pos:]))
    return "".join(parts)


class Band(NamedTuple):
    """One horizontal strip in page pixel rows: `[top, bottom)` is cropped and read; a zone read
    in it is kept only when its vertical centre lies in `[core_top, core_bottom)`. The cores tile
    the page without gaps or overlap, so each text line is kept from exactly one strip."""

    top: int
    bottom: int
    core_top: float
    core_bottom: float


def strip_bands(
    height: int, count: int = STRIP_COUNT, overlap_frac: float = STRIP_OVERLAP_FRAC
) -> list[Band]:
    """`count` equal strips covering `[0, height)`, adjacent ones sharing `overlap_frac x height`
    rows (fewer strips on a page under 64 px per strip); the core boundary between two strips is
    the middle of their overlap."""
    count = max(1, min(count, height // 64))
    overlap = round(height * overlap_frac) if count > 1 else 0
    size = math.ceil((height + (count - 1) * overlap) / count)
    bands: list[Band] = []
    for k in range(count):
        top = k * (size - overlap)
        bottom = height if k == count - 1 else top + size
        core_top = -math.inf if k == 0 else top + overlap / 2
        core_bottom = math.inf if k == count - 1 else bottom - overlap / 2
        bands.append(Band(top, bottom, core_top, core_bottom))
    return bands


Box = tuple[float, float, float, float]
Zone = tuple[str, Box]


def _to_page(box: Box, band: Band, height: int) -> Box:
    """A box normalized to the strip crop, normalized to the whole page instead."""
    x0, y0, x1, y1 = box
    rows = band.bottom - band.top
    return (x0, (band.top + y0 * rows) / height, x1, (band.top + y1 * rows) / height)


def _iou(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_zone(a: Zone, b: Zone) -> bool:
    """Two readings of one line: boxes overlap by IoU >= 0.5 and one text contains the other."""
    ta, tb = " ".join(a[0].split()), " ".join(b[0].split())
    return _iou(a[1], b[1]) >= 0.5 and (ta in tb or tb in ta)


def half_bands(
    band: Band, height: int, overlap_frac: float = STRIP_OVERLAP_FRAC
) -> tuple[Band, Band]:
    """The two halves an unfinished strip is re-read as (ADR-0013 amendment 4). Each crops half
    the strip plus half a strip overlap past the split, and their cores split the strip's core
    there, so together they keep exactly the lines the strip itself would have kept."""
    mid = (band.top + band.bottom) / 2
    reach = round(height * overlap_frac) / 2
    split = min(max(mid, band.core_top), band.core_bottom)
    upper = Band(band.top, min(band.bottom, math.ceil(mid + reach)), band.core_top, split)
    lower = Band(max(band.top, math.floor(mid - reach)), band.bottom, split, band.core_bottom)
    return upper, lower


def _is_arabic(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _ARABIC_BLOCKS)


def _rtl(text: str) -> bool:
    """Right-to-left reading direction, by the same rule as apx.parse.digital.is_rtl_text (the
    service image does not ship `apx`): at least 30% of the letters are Arabic-script."""
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and sum(map(_is_arabic, letters)) / len(letters) >= _RTL_LINE_THRESHOLD


def reading_order(zones: Sequence[Zone]) -> list[Zone]:
    """`zones` in geometric reading order (ADR-0013 amendment 4), never the model's emission
    order: zones are grouped into rows by vertical centre (a zone joins the row whose running
    mean centre is within half the median zone height), rows run top to bottom, and a row's zones
    run left to right, or right to left when the row's text reads right to left."""
    if not zones:
        return []
    heights = sorted(box[3] - box[1] for _, box in zones)
    tolerance = heights[len(heights) // 2] / 2

    def centre(zone: Zone) -> float:
        return (zone[1][1] + zone[1][3]) / 2

    rows: list[list[Zone]] = []
    means: list[float] = []
    for zone in sorted(zones, key=lambda z: (centre(z), z[1][0])):
        if rows and abs(centre(zone) - means[-1]) <= tolerance:
            rows[-1].append(zone)
            means[-1] += (centre(zone) - means[-1]) / len(rows[-1])
        else:
            rows.append([zone])
            means.append(centre(zone))
    ordered: list[Zone] = []
    for row in rows:
        if _rtl(" ".join(text for text, _ in row)):
            ordered.extend(sorted(row, key=lambda z: (-z[1][2], -z[1][0])))
        else:
            ordered.extend(sorted(row, key=lambda z: (z[1][0], z[1][2])))
    return ordered


def merge_strips(strips: Sequence[tuple[Band, Sequence[Zone]]], height: int) -> list[Zone]:
    """Zones of every strip in page coordinates, in geometric reading order (`reading_order`).
    A zone is kept only from the strip whose core holds its centre, and a second reading of a
    kept zone (see `_same_zone`) only replaces it when its text is longer."""
    merged: list[Zone] = []
    for band, zones in strips:
        for text, box in zones:
            page_box = _to_page(box, band, height)
            centre = (page_box[1] + page_box[3]) / 2 * height
            if not band.core_top <= centre < band.core_bottom:
                continue
            zone = (text, page_box)
            dup = next((i for i, kept in enumerate(merged) if _same_zone(kept, zone)), None)
            if dup is None:
                merged.append(zone)
            elif len(text) > len(merged[dup][0]):
                merged[dup] = zone
    return reading_order(merged)


def encode_zones(zones: Sequence[Zone]) -> str:
    """Zones back in the model's `Spotting:` form (`_parse_spotting_output` reads it back to
    within one 0-1000 bin), so the HTTP layer treats a strip re-read like any other page."""

    def loc(v: float) -> str:
        return f"<|LOC_{max(0, min(1000, round(v * 1000)))}|>"

    lines = []
    for text, (x0, y0, x1, y1) in zones:
        corners = (x0, y0, x1, y0, x1, y1, x0, y1)
        lines.append(text + "".join(loc(v) for v in corners) + "\n")
    return "".join(lines)


class PageStats(NamedTuple):
    """One page's decoding, logged by the model process (one JSON line on stderr per page)."""

    first_tokens: int
    first_finished: bool
    first_loops: int
    first_seconds: float
    """Wall time of the first pass alone, so its decode rate (first_tokens / first_seconds) is
    visible apart from the strips (ADR-0013 amendment 3)."""
    strips_read: int
    strips_finished: int
    strip_tokens: int
    strip_loops: int
    halves_read: int
    """Half-band re-reads of unfinished strips (ADR-0013 amendment 4)."""
    halves_finished: int
    used: Literal["first", "strips"]
    seconds: float


Run = Callable[[Image.Image, int, GuardMode, float | None], Pass]
"""(image, max_new_tokens, guard mode, max seconds or None) -> Pass: one generate call."""


def spot_page(
    image: Image.Image,
    run: Run,
    *,
    budget: int,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Spotting, PageStats]:
    """One page's `Spotting:` output (ADR-0013 amendments 1 and 2). The full page is read once,
    as before; a page that ends on EOS is returned untouched. A page stopped by a loop or the
    budget is re-read as `strip_bands` strips, each with its own STRIP_MAX_NEW_TOKENS budget and
    a closing loop guard, inside PAGE_DEADLINE_S, and the merged strips replace the first pass.
    A strip that does not finish is then re-read as its two `half_bands`, still inside the
    deadline (amendment 4): when both halves finish they replace the strip, otherwise the strip's
    and the halves' zones are pooled (what is recoverable) and the page stays truncated.
    `hit_limit` stays true unless every strip finished or was replaced by finished halves; an
    unfinished strip pass that recovered fewer zones than the first pass returns the first pass
    instead. The merged zones are in geometric reading order (`reading_order`)."""
    started = clock()
    first = run(image, budget, "first", None)
    first_seconds = round(clock() - started, 2)
    halves: list[Pass] = []

    def stats(results: Sequence[Pass], used: Literal["first", "strips"]) -> PageStats:
        return PageStats(
            first_tokens=first.tokens,
            first_finished=first.finished,
            first_loops=first.loops,
            first_seconds=first_seconds,
            strips_read=len(results),
            strips_finished=sum(r.finished for r in results),
            strip_tokens=sum(r.tokens for r in results),
            strip_loops=sum(r.loops for r in results),
            halves_read=len(halves),
            halves_finished=sum(h.finished for h in halves),
            used=used,
            seconds=round(clock() - started, 2),
        )

    if first.finished:
        return Spotting(first.text, hit_limit=False), stats((), "first")
    if PAGE_DEADLINE_S - (clock() - started) < MIN_STRIP_PASS_S:
        return Spotting(first.text, hit_limit=True), stats((), "first")

    width, height = image.size
    bands = strip_bands(height)
    results: list[Pass] = []
    for band in bands:
        remaining = PAGE_DEADLINE_S - (clock() - started)
        if remaining <= 0:
            break
        crop = image.crop((0, band.top, width, band.bottom))
        results.append(run(crop, STRIP_MAX_NEW_TOKENS, "strip", remaining))

    reads: list[tuple[Band, Sequence[Zone]]] = []
    resolved = 0
    for band, strip in zip(bands, results, strict=False):
        strip_zones = _parse_spotting_output(strip.text)
        if strip.finished:
            reads.append((band, strip_zones))
            resolved += 1
            continue
        pair: list[tuple[Band, Pass]] = []
        for half in half_bands(band, height):
            remaining = PAGE_DEADLINE_S - (clock() - started)
            if remaining < MIN_STRIP_PASS_S:
                break
            crop = image.crop((0, half.top, width, half.bottom))
            pair.append((half, run(crop, STRIP_MAX_NEW_TOKENS, "strip", remaining)))
        halves.extend(p for _, p in pair)
        # Halves first, so the merge's dedupe keeps their reading of a line the strip also read.
        reads.extend((half, _parse_spotting_output(p.text)) for half, p in pair)
        if len(pair) == 2 and all(p.finished for _, p in pair):
            resolved += 1
        else:
            reads.append((band, strip_zones))
    complete = len(results) == len(bands) and resolved == len(bands)
    zones = merge_strips(reads, height)
    if not complete and len(zones) < len(_parse_spotting_output(first.text)):
        return Spotting(first.text, hit_limit=True), stats(results, "first")
    return Spotting(encode_zones(zones), hit_limit=not complete), stats(results, "strips")


class _Vocab(NamedTuple):  # pragma: no cover - built from the real tokenizer in the GPU child
    loc_ids: frozenset[int]
    decode: Callable[[Sequence[int]], str]
    eos_ids: frozenset[int]

    def is_loc(self, token: int) -> bool:
        return token in self.loc_ids

    def alnum(self, ids: Sequence[int]) -> bool:
        return any(ch.isalnum() for ch in self.decode(ids))

    @staticmethod
    def from_processor(processor: Any, model: Any) -> _Vocab:
        loc_ids = frozenset(
            i
            for tok, i in processor.tokenizer.get_vocab().items()
            if re.fullmatch(r"<\|LOC_\d+\|>", tok)
        )
        if len(loc_ids) < 1001:
            raise RuntimeError(f"tokenizer has {len(loc_ids)} LOC tokens, expected 1001")
        configured = model.generation_config.eos_token_id
        eos = {
            processor.tokenizer.eos_token_id,
            *(configured if isinstance(configured, list) else [configured]),
        }

        def decode(ids: Sequence[int]) -> str:
            return str(processor.decode(list(ids), skip_special_tokens=False))

        return _Vocab(loc_ids, decode, frozenset(i for i in eos if i is not None))


class _ForceLoc:  # pragma: no cover - torch LogitsProcessor adapter, GPU child only
    """Steps the guard before each token; on `force` only LOC tokens keep a finite score."""

    def __init__(self, guard: LoopGuard, prompt_len: int, loc_ids: frozenset[int]) -> None:
        self._guard, self._prompt_len, self._loc_ids = guard, prompt_len, loc_ids
        self._mask: Any = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        if self._guard.step(input_ids[0, self._prompt_len :].tolist()) != "force":
            return scores
        if self._mask is None:
            mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
            mask[torch.tensor(sorted(self._loc_ids), device=scores.device)] = False
            self._mask = mask
        return scores.masked_fill(self._mask, float("-inf"))


class _GuardStop:  # pragma: no cover - torch StoppingCriteria adapter, GPU child only
    """Stops decoding once the guard has stopped; with `stepping` it also steps the guard (the
    first pass, which has no logits processor)."""

    def __init__(self, guard: LoopGuard, prompt_len: int, *, stepping: bool) -> None:
        self._guard, self._prompt_len, self._stepping = guard, prompt_len, stepping

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        import torch

        if self._stepping:
            self._guard.step(input_ids[0, self._prompt_len :].tolist())
        return torch.full(
            (input_ids.shape[0],), self._guard.stopped, dtype=torch.bool, device=input_ids.device
        )


def _generate(
    processor: Any,
    model: Any,
    vocab: _Vocab,
    image: Image.Image,
    max_new_tokens: int,
    mode: GuardMode,
    max_time: float | None,
) -> Pass:  # pragma: no cover - GPU child only
    import torch
    from transformers import LogitsProcessorList, StoppingCriteriaList

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
    prompt_len = int(inputs["input_ids"].shape[-1])

    # The first pass gets a stopping criterion only, no logits processor: the scores of a page
    # that never loops are untouched, so it decodes exactly as it did before amendment 2.
    guard = LoopGuard(vocab.is_loc, vocab.alnum, close=mode == "strip")
    extra: dict[str, Any] = {
        "stopping_criteria": StoppingCriteriaList(
            [_GuardStop(guard, prompt_len, stepping=mode == "first")]
        )
    }
    if mode == "strip":
        extra["logits_processor"] = LogitsProcessorList(
            [_ForceLoc(guard, prompt_len, vocab.loc_ids)]
        )
    if max_time is not None:
        extra["max_time"] = max_time

    # `inference_mode` rather than `no_grad` (ADR-0013 amendment 3): it also skips autograd's
    # version-counter and view tracking on every op; the kernels and so the tokens are the same.
    with torch.inference_mode():
        # `use_cache=True` overrides the checkpoint's `use_cache: false` (see
        # DEFAULT_MAX_NEW_TOKENS): without the KV cache every token re-runs the whole prefix.
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            repetition_penalty=REPETITION_PENALTY,
            **extra,
        )
    ids = [int(t) for t in outputs[0][prompt_len:]]
    finished = len(ids) > 0 and ids[-1] in vocab.eos_ids
    if finished:
        ids = ids[:-1]
    text = assemble(ids, guard.events, vocab.decode, vocab.is_loc)
    return Pass(text, finished=finished, tokens=len(ids), loops=len(guard.events))


def _looped(raw: str) -> bool:
    """Whether the spotting output (LOC tokens removed) contains a decoding loop: a cycle of 1-16
    characters with a letter or digit in it, repeated 16+ times in a row."""
    text = _LOC_TOKEN_RE.sub("", raw)
    return any(any(ch.isalnum() for ch in m.group(1)) for m in _LOOP_RE.finditer(text))


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
