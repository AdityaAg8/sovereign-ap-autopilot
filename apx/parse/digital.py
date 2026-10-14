"""Born-digital PDF fast path: word/line/table boxes straight from the PDF's own text layer via
pdfplumber, with no model call. This is the only path the IoU≥0.5 gate on
≥98% of labeled fields runs against; PaddleOCR-VL (apx.parse.client) is for scanned pages only.
Field boxes built this way satisfy I1: they come from parse words, never a model.
Word text is emitted in logical reading order (Arabic un-reversed, apx.parse.bidi)."""

from __future__ import annotations

import io
import unicodedata
from typing import Final, NotRequired, TypedDict, cast

import numpy as np
import pdfplumber
import pypdfium2 as pdfium
from numpy.typing import NDArray
from PIL import Image

from apx.common.errors import ParseError
from apx.common.types import BBox
from apx.parse.bidi import VisualWord, logical_line
from apx.parse.interfaces import Line, Page, Table, TableCell, Word


class _RawChar(TypedDict):
    """The subset of a pdfplumber char dict `_visual_text` reads."""

    text: str
    x0: float
    x1: float


class _RawWord(TypedDict):
    """The subset of pdfplumber's `extract_words()` dict this module reads, typed explicitly
    since pdfplumber's own stubs return untyped `object` values per key."""

    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    chars: NotRequired[list[_RawChar]]


# Arabic-script Unicode blocks (Arabic, Supplement, Extended-A, Presentation Forms A/B). A line
# is treated as RTL when at least this fraction of its letter characters fall in these ranges.
_ARABIC_BLOCKS: Final[tuple[tuple[int, int], ...]] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_RTL_LINE_THRESHOLD: Final[float] = 0.3
"""Arabic-letter fraction at or above which a line is read as a right-to-left paragraph."""

RENDER_SCALE: Final[float] = 2.0
"""~144 DPI page raster used for `Page.image_png` and, unchanged, as the scan-path input."""

_MARK_EPS: Final[float] = 0.01
"""Points of float noise tolerated when a zero-width mark sits exactly on a glyph boundary (its
origin equals the next glyph's x0 up to rounding): it belongs to the glyph starting there."""

_INK_THRESHOLD: Final[int] = 128
_INK_SEARCH_MARGIN: Final[float] = 0.006
"""Vertical search margin (page fraction) padded around pdfplumber's own top/bottom before
scanning the rendered page raster for actual ink rows. Needed because pdfminer computes a
word's top/bottom from the embedded font's declared ascent/descent, not the glyphs' visual
ink extent -- for some embedded fonts (observed on the UAE factory's PDFs, reports/parse.md)
that under- or over-shoots the real text height enough to fail the IoU gate on its own, even
though the horizontal (x0/x1) extent pdfplumber reports is accurate. This refinement keeps
pdfplumber as the source of words/text/x-position (I1) and only corrects the
vertical extent against the same raster already rendered for `Page.image_png`."""


def is_arabic_char(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _ARABIC_BLOCKS)


def arabic_fraction(text: str) -> float:
    """Fraction of `text`'s alphabetic characters that are Arabic-script. Used both to decide a
    line's reading direction here and, in `apx.parse.parser`, to tag a word's `Language`."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if is_arabic_char(ch)) / len(letters)


def is_rtl_text(text: str) -> bool:
    return arabic_fraction(text) >= _RTL_LINE_THRESHOLD


def parse_digital_pdf(document_sha256: str, data: bytes) -> tuple[Page, ...]:
    """Extract every page of a born-digital PDF via pdfplumber. Raises `ParseError` if the file
    cannot be opened or a page yields no extractable text (callers should route such a page
    through the scan path instead of calling this function on it)."""
    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception as exc:  # pdfplumber wraps pdfminer's varied parse failures
        raise ParseError(f"could not open {document_sha256} as a PDF: {exc}") from exc

    try:
        page_images = _render_all_pages(data)
        if len(page_images) != len(pdf.pages):
            raise ParseError(
                f"{document_sha256}: pdfplumber found {len(pdf.pages)} pages but pypdfium2 "
                f"rendered {len(page_images)}"
            )
        return tuple(
            _parse_digital_page(document_sha256, index, plumber_page, page_images[index])
            for index, plumber_page in enumerate(pdf.pages)
        )
    finally:
        pdf.close()


def parse_digital_page(document_sha256: str, data: bytes, page_index: int) -> Page:
    """Single-page variant of `parse_digital_pdf`, for a document that mixes born-digital and
    scanned pages: `apx.parse.parser` routes each page independently and only calls this for
    the pages `has_extractable_text` accepted."""
    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception as exc:  # pdfplumber wraps pdfminer's varied parse failures
        raise ParseError(f"could not open {document_sha256} as a PDF: {exc}") from exc
    try:
        plumber_page = pdf.pages[page_index]
        image_png = render_page_png(data, page_index)
        return _parse_digital_page(document_sha256, page_index, plumber_page, image_png)
    finally:
        pdf.close()


def has_extractable_text(plumber_page: pdfplumber.page.Page, *, min_words: int = 3) -> bool:
    """True if a page's own text layer has enough words to trust the fast path; a page below
    this (a pure scan, or one with only stray metadata text) should go through the scan path."""
    return len(plumber_page.extract_words()) >= min_words


def _parse_digital_page(
    document_sha256: str, index: int, plumber_page: pdfplumber.page.Page, image_png: bytes
) -> Page:
    raw_words = cast(
        "list[_RawWord]",
        plumber_page.extract_words(use_text_flow=False, keep_blank_chars=False, return_chars=True),
    )
    if not raw_words:
        raise ParseError(f"{document_sha256} page {index} has no extractable text")

    width, height = plumber_page.width, plumber_page.height
    if width <= 0 or height <= 0:
        raise ParseError(f"{document_sha256} page {index} has a degenerate page size")

    lines_of_words = _cluster_lines(raw_words)
    ink = _grayscale_ink(image_png)

    words: list[Word] = []
    lines: list[Line] = []
    word_index = 0
    for line_i, visual_words in enumerate(lines_of_words):
        # The text layer holds glyphs in visual order (Arabic letters reversed, as presentation
        # forms): recover logical word order and text, keeping each word's own box.
        line_height = max(w["bottom"] - w["top"] for w in visual_words)
        logical = logical_line(
            [VisualWord(_visual_text(w), w["x0"], w["x1"]) for w in visual_words],
            rtl=is_rtl_text("".join(w["text"] for w in visual_words)),
            line_height=line_height,
        )

        provisional = [
            (
                text,
                max(0.0, min(1.0, raw["x0"] / width)),
                max(0.0, min(1.0, raw["top"] / height)),
                max(0.0, min(1.0, raw["x1"] / width)),
                max(0.0, min(1.0, raw["bottom"] / height)),
            )
            for raw, text in ((visual_words[i], text) for i, text in logical)
        ]
        line_y0, line_y1 = _refine_line_vertical_extent(
            ink,
            x0=min(p[1] for p in provisional),
            x1=max(p[3] for p in provisional),
            approx_y0=min(p[2] for p in provisional),
            approx_y1=max(p[4] for p in provisional),
        )

        line_id = f"p{index}l{line_i}"
        line_word_ids: list[str] = []
        for text, word_x0, _approx_y0, word_x1, _approx_y1 in provisional:
            word_id = f"p{index}w{word_index}"
            word_index += 1
            bbox = BBox(page=index, x0=word_x0, y0=line_y0, x1=word_x1, y1=line_y1)
            words.append(Word(id=word_id, text=text, bbox=bbox, line_id=line_id, confidence=1.0))
            line_word_ids.append(word_id)
        lines.append(
            Line(id=line_id, word_ids=tuple(line_word_ids), bbox=_line_bbox(words, line_word_ids))
        )

    tables = _extract_tables(plumber_page, index, words, width, height)

    return Page(
        index=index,
        width_px=round(width * RENDER_SCALE),
        height_px=round(height * RENDER_SCALE),
        words=tuple(words),
        lines=tuple(lines),
        tables=tables,
        image_png=image_png,
    )


def _is_mark(text: str) -> bool:
    """A glyph is a combining mark when its text is all non-spacing marks. pdfplumber gives one
    char per glyph, but a ligature (a font mapping one glyph to "fi") carries several code points
    and an unmapped glyph none, which `unicodedata.bidirectional` (one code point only) cannot
    take: both are bases. For a single code point this is exactly `bidirectional(c) == "NSM"`."""
    return bool(text) and all(unicodedata.bidirectional(ch) == "NSM" for ch in text)


def _visual_text(raw: _RawWord) -> str:
    """The word's glyphs left to right, with each combining mark placed directly after the base
    glyph it is drawn on. Marks are zero-width and positioned at or inside their base's span,
    so pdfplumber's x-sort can land a mark after the wrong neighbour (a damma read onto the
    letter beside it); this re-attaches each mark to the base whose [x0, x1) holds it."""
    chars = raw.get("chars")
    if not chars or not any(_is_mark(c["text"]) for c in chars):
        return raw["text"]
    bases = sorted((c for c in chars if not _is_mark(c["text"])), key=lambda c: c["x0"])
    if not bases:
        return raw["text"]
    marks: list[list[str]] = [[] for _ in bases]
    for mark in (c for c in chars if _is_mark(c["text"])):
        owner = next(
            (
                i
                for i, b in enumerate(bases)
                if b["x0"] - _MARK_EPS <= mark["x0"] < b["x1"] - _MARK_EPS
            ),
            min(
                range(len(bases)),
                key=lambda i: abs((bases[i]["x0"] + bases[i]["x1"]) / 2 - mark["x0"]),
            ),
        )
        marks[owner].append(mark["text"])
    return "".join(b["text"] + "".join(m) for b, m in zip(bases, marks, strict=True))


def _grayscale_ink(image_png: bytes) -> NDArray[np.uint8]:
    """Decode the page's own rendered raster (already produced for `Page.image_png`) to
    grayscale, for `_refine_line_vertical_extent`."""
    with Image.open(io.BytesIO(image_png)) as image:
        return np.asarray(image.convert("L"))


def _refine_line_vertical_extent(
    ink: NDArray[np.uint8], *, x0: float, x1: float, approx_y0: float, approx_y1: float
) -> tuple[float, float]:
    """Correct a line's top/bottom against the actual rendered ink in its column band, within
    `_INK_SEARCH_MARGIN` of pdfplumber's own estimate. Falls back to the pdfplumber estimate if
    the band is empty (e.g. a degenerate box) or no dark pixels are found there."""
    height, width = ink.shape
    x0_px = max(0, min(width - 1, int(x0 * width)))
    x1_px = max(x0_px + 1, min(width, int(x1 * width) + 1))
    y0_px = max(0, int((approx_y0 - _INK_SEARCH_MARGIN) * height))
    y1_px = min(height, int((approx_y1 + _INK_SEARCH_MARGIN) * height) + 1)
    if y1_px <= y0_px:
        return approx_y0, approx_y1

    band = ink[y0_px:y1_px, x0_px:x1_px]
    dark_rows = np.nonzero((band < _INK_THRESHOLD).any(axis=1))[0]
    if dark_rows.size == 0:
        return approx_y0, approx_y1

    refined_y0 = (y0_px + int(dark_rows.min())) / height
    refined_y1 = (y0_px + int(dark_rows.max()) + 1) / height
    return refined_y0, refined_y1


def _line_bbox(words: list[Word], word_ids: list[str]) -> BBox:
    by_id = {w.id: w for w in words}
    boxes = [by_id[wid].bbox for wid in word_ids]
    page = boxes[0].page
    return BBox(
        page=page,
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def _cluster_lines(raw_words: list[_RawWord]) -> list[list[_RawWord]]:
    """Group words into visual text lines by vertical position. pdfplumber gives no line
    grouping of its own; words within `y_tolerance` of the running line's mean `top` join it,
    matching how a reader would group glyphs on the same baseline into one line."""
    ordered = sorted(raw_words, key=lambda w: w["top"])
    heights = [w["bottom"] - w["top"] for w in ordered]
    tolerance = (sum(heights) / len(heights)) * 0.5 if heights else 2.0

    lines: list[list[_RawWord]] = []
    line_tops: list[float] = []
    for word in ordered:
        top = word["top"]
        if lines and abs(top - line_tops[-1]) <= tolerance:
            lines[-1].append(word)
            line_tops[-1] = (line_tops[-1] * (len(lines[-1]) - 1) + top) / len(lines[-1])
        else:
            lines.append([word])
            line_tops.append(top)
    return lines


def _extract_tables(
    plumber_page: pdfplumber.page.Page,
    page_index: int,
    words: list[Word],
    width: float,
    height: float,
) -> tuple[Table, ...]:
    try:
        found = plumber_page.find_tables()
    except Exception as exc:  # pdfplumber/pdfminer table detection can raise on odd layouts
        raise ParseError(f"table detection failed on page {page_index}: {exc}") from exc

    tables: list[Table] = []
    for table_i, table in enumerate(found):
        cells: list[TableCell] = []
        for row_i, row in enumerate(table.rows):
            for col_i, cell_bbox in enumerate(row.cells):
                if cell_bbox is None:
                    continue
                x0, top, x1, bottom = cell_bbox
                cell_word_ids = tuple(
                    w.id
                    for w in words
                    if w.bbox.x0 * width >= x0 - 1
                    and w.bbox.x1 * width <= x1 + 1
                    and w.bbox.y0 * height >= top - 1
                    and w.bbox.y1 * height <= bottom + 1
                )
                cells.append(
                    TableCell(
                        row=row_i,
                        col=col_i,
                        word_ids=cell_word_ids,
                        bbox=BBox(
                            page=page_index,
                            x0=max(0.0, min(1.0, x0 / width)),
                            y0=max(0.0, min(1.0, top / height)),
                            x1=max(0.0, min(1.0, x1 / width)),
                            y1=max(0.0, min(1.0, bottom / height)),
                        ),
                    )
                )
        tables.append(Table(id=f"p{page_index}t{table_i}", page=page_index, cells=tuple(cells)))
    return tuple(tables)


def render_page_png(data: bytes, page_index: int) -> bytes:
    """Raster one PDF page via pypdfium2 at `RENDER_SCALE`, as PNG bytes. Public: also used by
    `apx.parse.parser` to render a scanned page before sending it to the parse service."""
    pdf = pdfium.PdfDocument(data)
    try:
        return _render_open_page(pdf, page_index)
    finally:
        pdf.close()


def _render_open_page(pdf: pdfium.PdfDocument, page_index: int) -> bytes:
    bitmap = pdf[page_index].render(scale=RENDER_SCALE)
    image = bitmap.to_pil()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _render_all_pages(data: bytes) -> list[bytes]:
    """Raster every page of `data` once via pypdfium2, so `parse_digital_pdf` never reopens the
    PDF per page. `pdfplumber.Page.pdf.stream` is a `BytesIO` with no filesystem path, so this
    renders straight from the original bytes rather than trying to reuse pdfplumber's handle."""
    pdf = pdfium.PdfDocument(data)
    try:
        return [_render_open_page(pdf, index) for index in range(len(pdf))]
    finally:
        pdf.close()
