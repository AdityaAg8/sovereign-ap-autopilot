"""Tests for `apx.parse.parser.PaddleOcrDocumentParser`: routing born-digital pages to
pdfplumber and scanned pages to the (faked) parse service client, on a mixed-page PDF, plus
language detection and non-PDF image input. No GPU, no network: the service
client is a stand-in, never `apx.parse.client.ParseServiceClient` itself."""

from __future__ import annotations

import io

import pytest
from PIL import Image
from reportlab.pdfgen import canvas

from apx.common.errors import ParseError
from apx.common.types import BBox, Language
from apx.parse.interfaces import Line, Page, Word
from apx.parse.parser import PaddleOcrDocumentParser

_SHA = "a" * 64


def _pdf_with_pages(*page_texts: str | None) -> bytes:
    """`None` renders a blank (graphical-only) page, forcing the scan path for it."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(400, 300))
    for text in page_texts:
        if text is not None:
            c.setFont("Helvetica", 12)
            c.drawString(40, 260, text)
        else:
            c.rect(10, 10, 50, 50)
        c.showPage()
    c.save()
    return buf.getvalue()


class _FakeClient:
    """Stand-in for `ParseServiceClient`: records every call and returns a canned `Page`, so
    tests can assert exactly which pages were routed to the scan path."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def parse_page_image(
        self, *, document_sha256: str, page_index: int, png_bytes: bytes
    ) -> Page:
        self.calls.append(page_index)
        word = Word(
            id=f"p{page_index}w0",
            text="ScannedWord",
            bbox=_bbox(page_index),
            line_id=f"p{page_index}l0",
            confidence=0.8,
        )
        line = Line(id=f"p{page_index}l0", word_ids=(word.id,), bbox=word.bbox)
        return Page(
            index=page_index,
            width_px=100,
            height_px=100,
            words=(word,),
            lines=(line,),
            tables=(),
            image_png=b"",
        )


def _bbox(page: int) -> BBox:
    return BBox(page=page, x0=0.1, y0=0.1, x1=0.2, y1=0.2)


class _RaisingClient:
    """Stand-in that fails the test if the digital path ever calls the scan service."""

    async def parse_page_image(self, **_kwargs: object) -> Page:
        raise AssertionError("born-digital page must never call the parse service")


@pytest.mark.asyncio
async def test_all_digital_pdf_never_calls_the_service() -> None:
    # `has_extractable_text`'s default `min_words=3` needs each page to clear that bar.
    pdf_bytes = _pdf_with_pages("Hello there world", "World says hello")
    parser = PaddleOcrDocumentParser(client=_RaisingClient())  # type: ignore[arg-type]

    doc = await parser.parse(_SHA, pdf_bytes)

    assert doc.document_sha256 == _SHA
    assert len(doc.pages) == 2
    assert [w.text for w in doc.pages[0].words] == ["Hello", "there", "world"]
    assert [w.text for w in doc.pages[1].words] == ["World", "says", "hello"]
    assert doc.parser_version == "paddleocr-vl-1.6+pdfplumber/4"


@pytest.mark.asyncio
async def test_mixed_pdf_routes_each_page_independently() -> None:
    # page 0 clears the min_words bar and stays digital; page 1 is blank, forced to scan.
    pdf_bytes = _pdf_with_pages("Digital text page", None)
    fake_client = _FakeClient()
    parser = PaddleOcrDocumentParser(client=fake_client)  # type: ignore[arg-type]

    doc = await parser.parse(_SHA, pdf_bytes)

    assert fake_client.calls == [1]  # only the blank page went through the service
    assert len(doc.pages) == 2
    assert doc.pages[0].words[0].text == "Digital"
    assert doc.pages[1].words[0].text == "ScannedWord"
    # reading_order concatenates line ids across pages, in page order.
    assert doc.reading_order == (doc.pages[0].lines[0].id, doc.pages[1].lines[0].id)


@pytest.mark.asyncio
async def test_non_pdf_image_input_goes_straight_to_the_service() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), color="white").save(buf, format="JPEG")
    fake_client = _FakeClient()
    parser = PaddleOcrDocumentParser(client=fake_client)  # type: ignore[arg-type]

    doc = await parser.parse(_SHA, buf.getvalue())

    assert fake_client.calls == [0]
    assert len(doc.pages) == 1
    assert doc.pages[0].words[0].text == "ScannedWord"


@pytest.mark.asyncio
async def test_garbage_bytes_raise_parse_error() -> None:
    parser = PaddleOcrDocumentParser(client=_RaisingClient())  # type: ignore[arg-type]
    with pytest.raises(ParseError):
        await parser.parse(_SHA, b"neither a pdf nor an image")


def test_word_language_detects_arabic_and_english() -> None:
    from apx.parse.parser import _word_language

    # chr() code points, not raw glyphs, so this file stays plain ASCII (apx/extract/normalize.py
    # convention): ALEF, BEH, TEH MARBUTA.
    arabic_word = chr(0x0627) + chr(0x0628) + chr(0x0629)
    assert _word_language(arabic_word) == Language.AR
    assert _word_language("Invoice") == Language.EN
