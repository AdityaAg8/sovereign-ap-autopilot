"""Raster size caps shared by upload validation (`apx.ingest.validate`) and every renderer
(`apx.scan.render`, `apx.parse.parser`): a PDF page's size in points, its pixel count at the
render DPI, its embedded images' pixel dimensions and a PNG/JPEG's header dimensions are all
bounded, so a few hundred crafted bytes can never decode to gigabytes, while real A3/Legal
600-dpi scans and 64 MP phone photos still fit."""

from __future__ import annotations

import io
import math
import warnings
from collections.abc import Iterator
from typing import Final

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image, UnidentifiedImageError

from apx.common.errors import IngestError

MAX_RENDER_DPI: Final[int] = 200
"""The highest DPI any apx renderer uses (`apx.scan.render.RENDER_DPI`); ingest checks a page at
this DPI so every later render of an accepted page fits the pixel cap."""

MAX_PAGE_SIDE_POINTS: Final[float] = 2384.0
"""Longest allowed page side in PDF points: A1's long side (841 mm). Invoices are A4/Letter;
the PDF format itself allows 14400 pt (200 in), which is what a bomb uses."""

MAX_RENDER_PIXELS: Final[int] = 40_000_000
"""Most pixels one rendered page may have (A1 at 200 dpi is ~31 MP; 40 MP of RGBA is 160 MB)."""

MAX_DECODED_BYTES: Final[int] = 400_000_000
"""Decoded-memory budget for one image, or for all images embedded in one PDF page: 400 MB,
counted at 4 bytes per pixel (RGBA, the widest mode any apx decoder produces). This is what the
pixel caps below are derived from (cap what is allocated, not a native resolution that
legitimate scans exceed)."""

MAX_IMAGE_PIXELS: Final[int] = MAX_DECODED_BYTES // 4
"""Most pixels one PNG/JPEG upload, or one image embedded in a PDF page, may declare: 100 MP.
Fits A3 at 600 dpi (7016 x 9921, ~70 MP), US Legal at 600 dpi (5100 x 8400, ~43 MP) and 48-64 MP
phone photos (9248 x 6936 is ~64 MP). The worker sets `PIL.Image.MAX_IMAGE_PIXELS` to this value,
so Pillow's own bomb error (at twice it) can never fire before this cap does."""

MAX_EMBEDDED_PIXELS_PER_PAGE: Final[int] = MAX_DECODED_BYTES // 4
"""Most pixels all images embedded in one PDF page may declare together (pdfium decodes each
one it paints), so a page's images share the same 400 MB decoded budget as a single image."""


class RasterTooLargeError(IngestError):
    """A PDF page, an image embedded in one, or an image upload exceeds a raster cap."""


def render_pixels(width_pt: float, height_pt: float, scale: float) -> int:
    """Pixel count pdfium produces for a `width_pt` x `height_pt` page at `scale` (px per pt)."""
    return math.ceil(width_pt * scale) * math.ceil(height_pt * scale)


def check_page_size(width_pt: float, height_pt: float, *, scale: float, where: str) -> None:
    """Raises `RasterTooLargeError` if the page is degenerate, has a side over
    `MAX_PAGE_SIDE_POINTS`, or renders to more than `MAX_RENDER_PIXELS` at `scale`."""
    if not (math.isfinite(width_pt) and math.isfinite(height_pt)) or min(width_pt, height_pt) <= 0:
        raise RasterTooLargeError(f"{where}: invalid page size {width_pt} x {height_pt} pt")
    if max(width_pt, height_pt) > MAX_PAGE_SIDE_POINTS:
        raise RasterTooLargeError(
            f"{where}: page is {width_pt:.0f} x {height_pt:.0f} pt; "
            f"at most {MAX_PAGE_SIDE_POINTS:.0f} pt per side is allowed"
        )
    pixels = render_pixels(width_pt, height_pt, scale)
    if pixels > MAX_RENDER_PIXELS:
        raise RasterTooLargeError(
            f"{where}: page renders to {pixels} px at {scale * 72:.0f} dpi; "
            f"at most {MAX_RENDER_PIXELS} are allowed"
        )


def check_embedded_images(page: pdfium.PdfPage, *, where: str) -> None:
    """Raises `RasterTooLargeError` if an image painted on `page` (including inside form
    XObjects) declares more than `MAX_IMAGE_PIXELS`, or all of them together more than
    `MAX_EMBEDDED_PIXELS_PER_PAGE`. Reads the image dictionaries only; nothing is decoded."""
    total = 0
    for width, height in _embedded_image_sizes(page, where):
        pixels = int(width) * int(height)
        if pixels > MAX_IMAGE_PIXELS:
            raise RasterTooLargeError(
                f"{where}: embedded image is {width} x {height} px; "
                f"at most {MAX_IMAGE_PIXELS} px are allowed"
            )
        total += pixels
        if total > MAX_EMBEDDED_PIXELS_PER_PAGE:
            raise RasterTooLargeError(
                f"{where}: embedded images total over {MAX_EMBEDDED_PIXELS_PER_PAGE} px"
            )


def _embedded_image_sizes(page: pdfium.PdfPage, where: str) -> Iterator[tuple[int, int]]:
    """Declared `(width, height)` of each image on `page`, one at a time. An unreadable object
    list or size raises `RasterTooLargeError`: sizes unknown, so never rendered (fail closed)."""
    try:
        for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)):
            if isinstance(obj, pdfium.PdfImage):
                yield obj.get_px_size()
    except pdfium.PdfiumError as exc:
        raise RasterTooLargeError(f"{where}: embedded images unreadable: {exc}") from exc


def check_pdf_page(page: pdfium.PdfPage, *, scale: float, where: str) -> None:
    """Every cap for one PDF page rendered at `scale`: size, pixel count, embedded images."""
    width_pt, height_pt = page.get_size()
    check_page_size(width_pt, height_pt, scale=scale, where=where)
    check_embedded_images(page, where=where)


def check_pdf_document(pdf: pdfium.PdfDocument, *, dpi: int = MAX_RENDER_DPI) -> None:
    """`check_pdf_page` for every page of an open document at `dpi`."""
    scale = dpi / 72.0
    for index in range(len(pdf)):
        page = pdf[index]
        try:
            check_pdf_page(page, scale=scale, where=f"page {index + 1}")
        finally:
            page.close()


def check_pdf_bytes(data: bytes, *, dpi: int = MAX_RENDER_DPI) -> None:
    """Opens `data` as a PDF and runs `check_pdf_document`. Raises `IngestError` if it cannot be
    opened: its page sizes are then unknown, so it is never rendered."""
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise IngestError(f"PDF could not be opened to check its page sizes: {exc}") from exc
    try:
        check_pdf_document(pdf, dpi=dpi)
    finally:
        pdf.close()


def check_image_bytes(data: bytes) -> tuple[int, int]:
    """Reads a PNG/JPEG header (no pixel decode) and returns its `(width, height)`. Raises
    `RasterTooLargeError` if it declares more than `MAX_IMAGE_PIXELS`, and `IngestError` if the
    header cannot be read at all (its size is then unknown, so it is never accepted)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
    except Image.DecompressionBombError as exc:
        raise RasterTooLargeError(f"image exceeds the decoder's pixel cap: {exc}") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise IngestError(f"image header could not be read: {exc}") from exc
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise RasterTooLargeError(
            f"image is {width} x {height} px; at most {MAX_IMAGE_PIXELS} px are allowed"
        )
    return width, height
