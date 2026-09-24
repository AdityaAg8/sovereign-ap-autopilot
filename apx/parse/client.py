"""HTTP client for the `parse` service (infra/docker/parse): sends one rendered page image and
gets back PaddleOCR-VL text-spotting words, lines and tables. Used only for scanned pages; a
born-digital page never reaches this client (apx.parse.digital handles it host-side). Typed
errors only, no silent fallback: any transport, timeout or malformed-response failure raises
`ParseError` so the caller decides what to do (PRD §3 apx/parse, §8). An optional `gate` wraps
every page call (the worker passes its GPU stage hold, ADR-0013), and `unload()` asks the service
to free its GPU memory before another stage needs the card."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Any

import httpx
import structlog
from pydantic import Field, ValidationError

from apx.common.errors import ParseError
from apx.common.types import BBox, Frozen, Sha256Hex
from apx.parse.config import ParseClientConfig
from apx.parse.interfaces import Line, Page, Table, TableCell, Word


class _WireWord(Frozen):
    """One entry of the service's `words` array (wire contract, see module docstring)."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)


class _WireTableCell(Frozen):
    row: int
    col: int
    word_indices: tuple[int, ...]
    bbox: tuple[float, float, float, float]


class _WireTable(Frozen):
    cells: tuple[_WireTableCell, ...]


class _WireLine(Frozen):
    """Word indices for one line, already in logical (RTL-aware) reading order."""

    word_indices: tuple[int, ...]


class _WireParseResponse(Frozen):
    """POST /parse response. `words` order carries no meaning; `lines`/`tables` reference it by
    index, and `lines` is already in logical reading order (service does the RTL reordering,
    since it is the only side that ran text spotting on the raster image)."""

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    words: tuple[_WireWord, ...]
    lines: tuple[_WireLine, ...]
    tables: tuple[_WireTable, ...] = ()
    peak_vram_gb: float | None = Field(default=None, ge=0.0)
    """The service's per-page GPU peak (infra/docker/parse `ParseResponse`); telemetry only."""


logger = structlog.get_logger(__name__)

Gate = Callable[[], AbstractAsyncContextManager[None]]


class ParseServiceClient:
    """Async client for the parse service's OCR text-spotting endpoint (scan path only)."""

    def __init__(self, config: ParseClientConfig | None = None, gate: Gate | None = None) -> None:
        self._config = config or ParseClientConfig()
        self._gate: Gate = gate or nullcontext

    async def unload(self) -> None:
        """Free the service's GPU memory (it reloads on the next page). A service that is not
        running at all (connection refused, name not resolved) holds no GPU memory, so that is
        logged and treated as released; any other failure raises `ParseError`."""
        timeout = httpx.Timeout(
            self._config.request_timeout_s, connect=self._config.connect_timeout_s
        )
        try:
            async with httpx.AsyncClient(base_url=self._config.base_url, timeout=timeout) as http:
                response = await http.post("/unload")
                response.raise_for_status()
        except httpx.ConnectError as exc:
            logger.warning("parse_unload_unreachable", error=str(exc))
        except httpx.HTTPError as exc:
            raise ParseError(f"parse service /unload failed: {exc!r}") from exc

    async def parse_page_image(
        self, *, document_sha256: Sha256Hex, page_index: int, png_bytes: bytes
    ) -> Page:
        """Send one rendered page (PNG) to the parse service and return a `Page` with word/line
        boxes normalized to [0, 1] and an empty `tables` tuple unless the service found any.
        Raises `ParseError` on a network failure, timeout, non-2xx response or a response that
        does not match the wire contract; never returns a partial or fabricated `Page`."""
        async with self._gate():
            return await self._parse_page_image(document_sha256, page_index, png_bytes)

    async def _parse_page_image(
        self, document_sha256: Sha256Hex, page_index: int, png_bytes: bytes
    ) -> Page:
        timeout = httpx.Timeout(
            self._config.request_timeout_s, connect=self._config.connect_timeout_s
        )
        try:
            async with httpx.AsyncClient(base_url=self._config.base_url, timeout=timeout) as http:
                response = await http.post(
                    "/parse",
                    files={"image": (f"page{page_index}.png", png_bytes, "image/png")},
                )
                response.raise_for_status()
                payload: Any = response.json()
        except httpx.TimeoutException as exc:
            raise ParseError(
                f"parse service timed out on page {page_index} of {document_sha256}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ParseError(
                f"parse service returned {exc.response.status_code} for page {page_index} of "
                f"{document_sha256}: {exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ParseError(
                f"parse service unreachable for page {page_index} of {document_sha256}: {exc}"
            ) from exc

        try:
            wire = _WireParseResponse.model_validate(payload)
        except ValidationError as exc:
            raise ParseError(
                f"parse service returned a malformed response for page {page_index} of "
                f"{document_sha256}: {exc}"
            ) from exc

        return _to_page(document_sha256, page_index, wire)


def _to_page(document_sha256: Sha256Hex, page_index: int, wire: _WireParseResponse) -> Page:
    word_ids = [f"p{page_index}w{i}" for i in range(len(wire.words))]
    words = tuple(
        Word(
            id=word_ids[i],
            text=w.text,
            bbox=_bbox(page_index, w.bbox),
            line_id="",  # placeholder, replaced below once line ids are known
            confidence=w.confidence,
        )
        for i, w in enumerate(wire.words)
    )
    if any(not (0 <= idx < len(words)) for line in wire.lines for idx in line.word_indices):
        raise ParseError(
            f"parse service referenced an out-of-range word index for {document_sha256} "
            f"page {page_index}"
        )

    if any(not line.word_indices for line in wire.lines):
        raise ParseError(
            f"parse service returned an empty line for {document_sha256} page {page_index}"
        )

    lines: list[Line] = []
    word_line: dict[int, str] = {}
    for line_i, wire_line in enumerate(wire.lines):
        line_id = f"p{page_index}l{line_i}"
        for idx in wire_line.word_indices:
            word_line[idx] = line_id
        line_word_ids = tuple(word_ids[idx] for idx in wire_line.word_indices)
        line_boxes = [words[idx].bbox for idx in wire_line.word_indices]
        line_bbox = _union_bbox(page_index, line_boxes)
        lines.append(Line(id=line_id, word_ids=line_word_ids, bbox=line_bbox))

    words = tuple(
        w.model_copy(update={"line_id": word_line.get(i, "")}) for i, w in enumerate(words)
    )

    tables = tuple(
        Table(
            id=f"p{page_index}t{table_i}",
            page=page_index,
            cells=tuple(
                TableCell(
                    row=cell.row,
                    col=cell.col,
                    word_ids=tuple(word_ids[idx] for idx in cell.word_indices),
                    bbox=_bbox(page_index, cell.bbox),
                )
                for cell in wire_table.cells
            ),
        )
        for table_i, wire_table in enumerate(wire.tables)
    )

    return Page(
        index=page_index,
        width_px=wire.width_px,
        height_px=wire.height_px,
        words=words,
        lines=tuple(lines),
        tables=tables,
        image_png=b"",  # set by the caller, which already holds the encoded page image
    )


def _bbox(page: int, coords: tuple[float, float, float, float]) -> BBox:
    x0, y0, x1, y1 = coords
    return BBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1)


def _union_bbox(page: int, boxes: list[BBox]) -> BBox:
    return BBox(
        page=page,
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )
