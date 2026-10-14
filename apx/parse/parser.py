"""Top-level `DocumentParser`: routes each page to the fast pdfplumber path when it is
born-digital, or to the PaddleOCR-VL service (apx.parse.client) when it is a scan.
This is the only place that decides which path a page takes; the decision itself
uses no model, just the page's own extractable word count. Nothing is rasterised until the
`apx.common.raster` caps have passed."""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Final

import pdfplumber
from PIL import Image, UnidentifiedImageError

from apx.common.errors import IngestError, ParseError
from apx.common.raster import RasterTooLargeError, check_image_bytes, check_pdf_bytes
from apx.common.types import Language, Sha256Hex
from apx.parse.client import ParseServiceClient
from apx.parse.digital import (
    has_extractable_text,
    is_rtl_text,
    parse_digital_page,
    parse_digital_pdf,
    render_page_png,
)
from apx.parse.interfaces import Page, ParsedDocument

_PDF_MAGIC: Final[bytes] = b"%PDF"
SERVICE_PARSER_VERSION: Final[str] = "paddleocr-vl-1.6+pdfplumber/4"


class PaddleOcrDocumentParser:
    """`DocumentParser` implementation: pdfplumber for born-digital pages, the PaddleOCR-VL
    service for scans. Constructed with a `ParseServiceClient` so tests can inject a fake
    transport without a live GPU service (see tests/unit/test_parse_parser.py)."""

    def __init__(self, client: ParseServiceClient | None = None) -> None:
        self._client = client or ParseServiceClient()

    async def parse(self, document_sha256: Sha256Hex, data: bytes) -> ParsedDocument:
        if data.startswith(_PDF_MAGIC):
            pages = await self._parse_pdf(document_sha256, data)
        else:
            png_bytes = _to_png(document_sha256, data)
            pages = (await self._parse_image_bytes(document_sha256, page_index=0, data=png_bytes),)

        if not pages:
            raise ParseError(f"{document_sha256} produced no pages")

        reading_order = tuple(line.id for page in pages for line in page.lines)
        languages = _detect_languages(pages)
        return ParsedDocument(
            document_sha256=document_sha256,
            pages=pages,
            reading_order=reading_order,
            languages=languages,
            parser_version=SERVICE_PARSER_VERSION,
        )

    async def _parse_pdf(self, document_sha256: Sha256Hex, data: bytes) -> tuple[Page, ...]:
        _check_raster_caps(document_sha256, data, check_pdf_bytes)
        try:
            plumber_pdf = pdfplumber.open(io.BytesIO(data))
        except Exception as exc:
            raise ParseError(f"could not open {document_sha256} as a PDF: {exc}") from exc

        try:
            digital_flags = [has_extractable_text(page) for page in plumber_pdf.pages]
        finally:
            plumber_pdf.close()

        if all(digital_flags):
            return parse_digital_pdf(document_sha256, data)

        pages: list[Page] = []
        for index, is_digital in enumerate(digital_flags):
            if is_digital:
                pages.append(parse_digital_page(document_sha256, data, index))
            else:
                png_bytes = render_page_png(data, index)
                pages.append(
                    await self._parse_image_bytes(document_sha256, page_index=index, data=png_bytes)
                )
        return tuple(pages)

    async def _parse_image_bytes(
        self, document_sha256: Sha256Hex, *, page_index: int, data: bytes
    ) -> Page:
        page = await self._client.parse_page_image(
            document_sha256=document_sha256, page_index=page_index, png_bytes=data
        )
        return page.model_copy(update={"image_png": data})


def _to_png(document_sha256: Sha256Hex, data: bytes) -> bytes:
    """Normalize a non-PDF upload (JPEG, TIFF scan, etc.) to PNG so `Page.image_png` always
    holds what its name says, regardless of the source format. The header's dimensions are
    checked against the raster caps before any pixel is decoded."""
    _check_raster_caps(document_sha256, data, check_image_bytes)
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ParseError(
            f"{document_sha256} is neither a PDF nor a decodable image: {exc}"
        ) from exc
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _check_raster_caps(
    document_sha256: Sha256Hex, data: bytes, check: Callable[[bytes], object]
) -> None:
    """Runs a raster-cap `check`; `RasterTooLargeError` passes through (typed, deterministic),
    while an unreadable document becomes the parser's own `ParseError`."""
    try:
        check(data)
    except RasterTooLargeError:
        raise
    except IngestError as exc:
        raise ParseError(f"{document_sha256} could not be size-checked: {exc}") from exc


def _detect_languages(pages: tuple[Page, ...]) -> frozenset[Language]:
    languages: set[Language] = set()
    for page in pages:
        for word in page.words:
            languages.add(_word_language(word.text))
    return frozenset(languages) or frozenset({Language.EN})


def _word_language(text: str) -> Language:
    return Language.AR if is_rtl_text(text) else Language.EN
