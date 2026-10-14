"""Upload-hardening tests for the raster caps: hand-built PDFs of a few
hundred bytes that declare huge pages or huge embedded images, and PNG headers that declare huge
dimensions, are rejected at ingest with a typed error, and every renderer refuses them before
allocating a bitmap. Nothing here decodes a large image, so a regression fails fast, not by OOM."""

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
    MAX_DECODED_BYTES,
    MAX_EMBEDDED_PIXELS_PER_PAGE,
    MAX_IMAGE_PIXELS,
    MAX_PAGE_SIDE_POINTS,
    MAX_RENDER_DPI,
    MAX_RENDER_PIXELS,
    RasterTooLargeError,
    check_image_bytes,
    check_page_size,
    check_pdf_bytes,
    render_pixels,
)
from apx.ingest.validate import UnsupportedMediaTypeError, validate
from apx.parse.parser import PaddleOcrDocumentParser
from apx.scan.render import render_page

MAX_BYTES = 20 * 1024 * 1024
SHA = "a" * 64


@pytest.fixture(autouse=True)
def _pillow_pinned_like_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins Pillow's process-global bomb guard to the apx cap, as the worker does at startup
    (`apx.workflows.main`), so these tests see that setting and not whatever another module
    imported earlier left behind (the page-embed service app sets its own, stricter value)."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", MAX_IMAGE_PIXELS)


def tiny_pdf(
    width_pt: float,
    height_pt: float,
    image_px: tuple[int, int] | None = None,
    *,
    image_count: int = 1,
) -> bytes:
    """A valid one-page PDF with a correct xref table. With `image_px`, the page paints one
    grayscale image XObject declaring that many pixels, `image_count` times under distinct
    names (its data stream is empty: only the declared dimensions matter to the caps, which
    never decode)."""
    names = [b"/Im%d" % n for n in range(1, image_count + 1)]
    content = b"q 10 0 0 10 0 0 cm %s Q" % b" ".join(n + b" Do" for n in names) if image_px else b""
    xobjects = b" ".join(n + b" 5 0 R" for n in names)
    resources = b" /Resources << /XObject << %s >> >>" % xobjects if image_px else b""
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


def jpeg_header(width: int, height: int) -> bytes:
    """A real 8 x 8 baseline JPEG whose SOF0 frame header is patched to declare `width` x
    `height`: PIL's lazy open reports the patched size, and nothing large is ever encoded."""
    data = bytearray(real_jpeg())
    sof = data.index(b"\xff\xc0")
    data[sof + 5 : sof + 9] = struct.pack(">HH", height, width)
    return bytes(data)


# ------------------------------------------------------------------------------------------
# REVIEW N-05: legitimate large scans and phone photos pass; the budget still holds
# ------------------------------------------------------------------------------------------


def test_the_pixel_caps_derive_from_a_400_mb_decoded_budget() -> None:
    assert MAX_DECODED_BYTES == 400_000_000
    assert MAX_IMAGE_PIXELS * 4 <= MAX_DECODED_BYTES
    assert MAX_EMBEDDED_PIXELS_PER_PAGE * 4 <= MAX_DECODED_BYTES


@pytest.mark.parametrize(
    ("label", "page_pt", "image_px"),
    [
        ("A3 at 600 dpi", (842, 1191), (7016, 9921)),
        ("US Legal at 600 dpi", (612, 1008), (5100, 8400)),
        ("A4 at 600 dpi", (595, 842), (4961, 7016)),
    ],
)
def test_a_600_dpi_scan_pdf_is_accepted(
    label: str, page_pt: tuple[int, int], image_px: tuple[int, int]
) -> None:
    pdf = tiny_pdf(*page_pt, image_px=image_px)
    assert validate(pdf, max_bytes=MAX_BYTES, max_pages=30) == "application/pdf", label


@pytest.mark.parametrize(
    ("label", "size"),
    [
        ("48 MP phone photo", (8064, 6048)),
        ("50 MP phone photo", (8160, 6144)),
        ("64 MP phone photo", (9248, 6936)),
        ("A3 600 dpi scan", (7016, 9921)),
    ],
)
def test_large_phone_photos_and_scans_are_accepted(label: str, size: tuple[int, int]) -> None:
    assert validate(jpeg_header(*size), max_bytes=MAX_BYTES, max_pages=30) == "image/jpeg", label
    assert validate(png_header(*size), max_bytes=MAX_BYTES, max_pages=30) == "image/png", label


def test_a_108_mp_photo_is_over_the_budget_and_rejected() -> None:
    with pytest.raises(RasterTooLargeError, match="px"):
        validate(jpeg_header(12000, 9000), max_bytes=MAX_BYTES, max_pages=30)


def test_an_image_exactly_at_the_cap_is_accepted_and_one_pixel_row_over_is_not() -> None:
    width = 10_000
    at_cap = MAX_IMAGE_PIXELS // width
    assert check_image_bytes(png_header(width, at_cap)) == (width, at_cap)
    with pytest.raises(RasterTooLargeError):
        check_image_bytes(png_header(width, at_cap + 1))


def test_a_page_whose_images_together_exceed_the_budget_is_rejected() -> None:
    # Two 60 MP images: each under the per-image cap, together over the per-page budget.
    pdf = tiny_pdf(595, 842, image_px=(7746, 7746), image_count=2)
    with pytest.raises(RasterTooLargeError, match="total"):
        validate(pdf, max_bytes=MAX_BYTES, max_pages=30)


def test_pillow_bomb_guard_is_consistent_with_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # The worker pins Image.MAX_IMAGE_PIXELS to the cap: an image at the cap must not trip
    # Pillow's DecompressionBombError (which fires at twice the setting), one far over must be
    # refused by us with a typed error either way.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", MAX_IMAGE_PIXELS)
    assert check_image_bytes(jpeg_header(10_000, MAX_IMAGE_PIXELS // 10_000))
    with pytest.raises(RasterTooLargeError):
        check_image_bytes(jpeg_header(20_000, 20_000))
    # The API process keeps Pillow's default (~89 MP; the error fires at ~179 MP): an at-cap
    # upload still validates there, and its only warning is the one check_image_bytes silences.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 89_478_485)
    assert check_image_bytes(jpeg_header(10_000, MAX_IMAGE_PIXELS // 10_000))


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
