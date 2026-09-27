"""Upload-hardening tests for the raster caps (REVIEW H-03, PRD §11.4): hand-built PDFs of a few
hundred bytes that declare huge pages or huge embedded images, and PNG headers that declare huge
dimensions, are rejected at ingest with a typed error, and every renderer refuses them before
allocating a bitmap. Nothing here decodes a large image, so a regression fails fast, not by OOM.
"""

from __future__ import annotations

import asyncio
import io
import struct
import zlib

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from apx.common.errors import ParseError
from apx.common.raster import (
    MAX_IMAGE_PIXELS,
    MAX_PAGE_SIDE_POINTS,
    MAX_RENDER_DPI,
    MAX_RENDER_PIXELS,
    RasterTooLargeError,
    check_page_size,
    check_pdf_bytes,
    render_pixels,
)
from apx.ingest.validate import UnsupportedMediaTypeError, validate
from apx.parse.parser import PaddleOcrDocumentParser
from apx.scan.render import render_page

MAX_BYTES = 20 * 1024 * 1024
SHA = "a" * 64


def tiny_pdf(width_pt: float, height_pt: float, image_px: tuple[int, int] | None = None) -> bytes:
    """A valid one-page PDF with a correct xref table. With `image_px`, the page paints one
    grayscale image XObject declaring that many pixels (its data stream is empty: only the
    declared dimensions matter to the caps, which never decode)."""
    content = b"q 10 0 0 10 0 0 cm /Im1 Do Q" if image_px else b""
    resources = b" /Resources << /XObject << /Im1 5 0 R >> >>" if image_px else b""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %s %s] /Contents 4 0 R%s >>"
        % (str(width_pt).encode(), str(height_pt).encode(), resources),
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
    ]
    if image_px:
        objects.append(
            b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
            b"/BitsPerComponent 8 /Length 0 >>\nstream\n\nendstream" % image_px
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def png_header(width: int, height: int) -> bytes:
    """A PNG signature, a valid IHDR chunk declaring `width` x `height` and an empty IDAT: enough
    for PIL's lazy open to report the size, far too little to decode any pixel."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    signature = bytes.fromhex("89504e470d0a1a0a")
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b"") + chunk(b"IEND", b"")


def real_png(width: int = 8, height: int = 8) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def real_jpeg(width: int = 8, height: int = 8) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------------------------------
# The review's bomb: 14400 x 14400 pt in a few hundred bytes
# ------------------------------------------------------------------------------------------


def test_the_review_bomb_is_a_few_hundred_bytes_and_is_rejected_at_ingest() -> None:
    bomb = tiny_pdf(14400, 14400)
    assert len(bomb) < 600  # the REVIEW's 553-byte case: tiny on disk
    with pytest.raises(RasterTooLargeError, match="pt per side"):
        validate(bomb, max_bytes=MAX_BYTES, max_pages=30)


def test_a_page_under_the_side_cap_but_over_the_pixel_cap_is_rejected() -> None:
    # 2384 x 2384 pt is inside the per-side cap but renders to ~43.8 MP at 200 dpi.
    side = MAX_PAGE_SIDE_POINTS
    assert render_pixels(side, side, MAX_RENDER_DPI / 72) > MAX_RENDER_PIXELS
    with pytest.raises(RasterTooLargeError, match="renders to"):
        validate(tiny_pdf(side, side), max_bytes=MAX_BYTES, max_pages=30)


@pytest.mark.parametrize(("width", "height"), [(595, 842), (612, 792), (1684, 2384)])
def test_a4_letter_and_a1_pages_are_accepted(width: float, height: float) -> None:
    assert validate(tiny_pdf(width, height), max_bytes=MAX_BYTES, max_pages=30) == (
        "application/pdf"
    )


@pytest.mark.parametrize(("width", "height"), [(0, 842), (595, -1)])
def test_a_degenerate_page_size_is_rejected(width: float, height: float) -> None:
    with pytest.raises(RasterTooLargeError, match="invalid page size"):
        check_page_size(width, height, scale=1.0, where="page 1")


def test_an_oversized_embedded_image_on_a_normal_page_is_rejected() -> None:
    bomb = tiny_pdf(595, 842, image_px=(100_000, 100_000))
    assert len(bomb) < 1000
    with pytest.raises(RasterTooLargeError, match="embedded image"):
        validate(bomb, max_bytes=MAX_BYTES, max_pages=30)


def test_an_embedded_image_within_the_cap_is_accepted() -> None:
    check_pdf_bytes(tiny_pdf(595, 842, image_px=(2000, 3000)))


# ------------------------------------------------------------------------------------------
# Images: header dimensions, before any decode
# ------------------------------------------------------------------------------------------


def test_a_png_header_declaring_a_huge_image_is_rejected_without_decoding() -> None:
    bomb = png_header(40_000, 40_000)
    assert len(bomb) < 64
    with pytest.raises(RasterTooLargeError):
        validate(bomb, max_bytes=MAX_BYTES, max_pages=30)


def test_a_png_just_over_the_pixel_cap_is_rejected() -> None:
    width = 10_000
    height = MAX_IMAGE_PIXELS // width + 1
    with pytest.raises(RasterTooLargeError, match="px"):
        validate(png_header(width, height), max_bytes=MAX_BYTES, max_pages=30)


def test_real_small_png_and_jpeg_are_accepted() -> None:
    assert validate(real_png(), max_bytes=MAX_BYTES, max_pages=30) == "image/png"
    assert validate(real_jpeg(), max_bytes=MAX_BYTES, max_pages=30) == "image/jpeg"


def test_an_image_whose_header_cannot_be_read_is_rejected_as_unsupported() -> None:
    with pytest.raises(UnsupportedMediaTypeError, match="unreadable image"):
        validate(b"\x89PNG\r\n\x1a\nrest", max_bytes=MAX_BYTES, max_pages=30)


# ------------------------------------------------------------------------------------------
# Renderers refuse over-cap pages themselves (defence in depth past ingest)
# ------------------------------------------------------------------------------------------


def test_scan_render_page_refuses_an_oversized_page_before_rendering() -> None:
    pdf = pdfium.PdfDocument(tiny_pdf(6000, 6000))
    try:
        with pytest.raises(RasterTooLargeError):
            render_page(pdf[0])
    finally:
        pdf.close()


def test_scan_render_page_refuses_an_oversized_embedded_image() -> None:
    pdf = pdfium.PdfDocument(tiny_pdf(595, 842, image_px=(50_000, 50_000)))
    try:
        with pytest.raises(RasterTooLargeError):
            render_page(pdf[0])
    finally:
        pdf.close()


def test_scan_render_page_still_renders_a_normal_page() -> None:
    pdf = pdfium.PdfDocument(tiny_pdf(72, 36))
    try:
        image = render_page(pdf[0], dpi=MAX_RENDER_DPI)
    finally:
        pdf.close()
    assert image.shape == (100, 200, 3)
    assert image.dtype == np.uint8


class _UnusedClient:
    async def parse_page_image(self, **_: object) -> None:
        raise AssertionError("an over-cap document must never reach the parse service")


def test_parser_refuses_an_oversized_pdf_page_before_rasterising() -> None:
    parser = PaddleOcrDocumentParser(client=_UnusedClient())  # type: ignore[arg-type]
    with pytest.raises(RasterTooLargeError):
        asyncio.run(parser.parse(SHA, tiny_pdf(14400, 14400)))


def test_parser_refuses_an_oversized_image_before_decoding() -> None:
    parser = PaddleOcrDocumentParser(client=_UnusedClient())  # type: ignore[arg-type]
    with pytest.raises(RasterTooLargeError):
        asyncio.run(parser.parse(SHA, png_header(40_000, 40_000)))


def test_parser_turns_an_unreadable_image_into_a_parse_error() -> None:
    parser = PaddleOcrDocumentParser(client=_UnusedClient())  # type: ignore[arg-type]
    with pytest.raises(ParseError):
        asyncio.run(parser.parse(SHA, b"not an image at all"))
