"""factory/render.py: Playwright-Chromium rendering for the M2.3 UAE invoice factory. Turns one
`factory.layouts`-rendered HTML document into a digital PDF plus `data-field` DOM-box labels in
PDF-page coordinates: it paginates the DOM itself (real spacer elements reserve each page's
margins and push any row or landmark that would straddle a page cut), measures the labels, then
prints that same paginated DOM with a zero print margin, so the PDF matches the measured layout
by construction. Optionally also returns a clean per-page PNG screenshot for the
30% of invoices that get an Augraphy scan variant."""

from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass
from io import BytesIO
from typing import cast

from PIL import Image, UnidentifiedImageError
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from factory.labels import LabelSet, RawPagination, build_label_set

MM_PER_INCH = 25.4
CSS_PX_PER_INCH = 96.0

# Chromium stamps a wall-clock `/CreationDate` and `/ModDate` into every exported PDF, which is
# the only source of non-determinism in an otherwise pure function of the HTML/CSS layout (PRD
# §10 M2.3: "deterministic from seed"). `_normalize_pdf_bytes` overwrites both with a fixed
# placeholder of the exact same byte length, so it never shifts the PDF's xref byte offsets.
_PDF_TIMESTAMP_RE = re.compile(rb"\(D:\d{14}[+-]\d{2}'\d{2}'\)")
_FIXED_PDF_TIMESTAMP = b"(D:20260101000000+00'00')"


def _normalize_pdf_bytes(pdf_bytes: bytes) -> bytes:
    def _replace(match: re.Match[bytes]) -> bytes:
        original = match.group(0)
        if len(original) != len(_FIXED_PDF_TIMESTAMP):
            raise ValueError(
                "factory.render: Chromium's PDF timestamp format changed length; "
                "_FIXED_PDF_TIMESTAMP must be updated to match or the xref table will break"
            )
        return _FIXED_PDF_TIMESTAMP

    return _PDF_TIMESTAMP_RE.sub(_replace, pdf_bytes)


# Must match factory/templates/_shared.html.j2's `@page { size: A4; margin: 14mm; }` (checked at
# import time by `test_factory_render_page_geometry_matches_css_contract`). Kept as explicit
# constants here, and passed explicitly to `page.pdf()`, rather than relying on Chromium reading
# the CSS `@page` rule, so this module's page geometry never silently drifts from what it thinks
# it produced.
PAGE_WIDTH_MM = 210.0
PAGE_HEIGHT_MM = 297.0
MARGIN_MM = 14.0


def _mm_to_css_px(mm: float) -> float:
    return mm * CSS_PX_PER_INCH / MM_PER_INCH


PAGE_WIDTH_PX = _mm_to_css_px(PAGE_WIDTH_MM)
PAGE_HEIGHT_PX = _mm_to_css_px(PAGE_HEIGHT_MM)
MARGIN_PX = _mm_to_css_px(MARGIN_MM)

# One self-contained pagination pass, run in the page via `page.evaluate`:
#   1. Inserts a top spacer worth one page margin, so page 1's content starts where the real
#      PDF's top margin would put it.
#   2. Walks every table row (`<tr>`) plus every `header`/`section`/`footer` landmark, in DOM
#      order; an element whose real print rendering would straddle a page boundary is pushed down
#      to the top of the next page. A table row's spacer is itself a `<tr>` (a plain sibling
#      `<div>` is invalid inside a `<tbody>`); every other element gets a plain spacer `<div>`
#      inserted as its previous sibling. This covers every `factory/templates/*.j2` layout's
#      fragmentable blocks: the shared CSS only marks `tr.ap-row` explicit `break-inside: avoid`,
#      but browsers do not split a table row, or a short landmark block like the letterhead
#      header or a totals/parties section, across a page boundary either way -- so all of them
#      are treated as break-atomic here, not just the one class the CSS says so about.
#   3. Re-measures every `[data-field]` element's `getClientRects()` (one rect per wrapped text
#      line) against the now page-break-adjusted layout, plus which `.ap-row` it belongs to and
#      that row's 0-based index within its own line-items table.
# Steps 2-3 use real DOM layout, not a hand-rolled approximation. `RenderWorker.render` then
# prints this *same* spacer-adjusted DOM with a zero print margin, so Chromium's page cuts fall
# at exact multiples of the page height, where the spacers guarantee no row or landmark sits:
# nothing is left for its fragmentation to move. M2.4's pdfplumber self-check verifies it.
_PAGINATE_JS = r"""
({ pageHeightPx, marginPx }) => {
  const sheet = document.querySelector(".sheet");
  if (!sheet) {
    throw new Error("factory.render: no .sheet root element found in the rendered HTML");
  }

  const topSpacer = document.createElement("div");
  topSpacer.setAttribute("data-ap-spacer", "top");
  topSpacer.style.cssText = "height:" + marginPx + "px;margin:0;padding:0;";
  sheet.insertBefore(topSpacer, sheet.firstChild);

  // Document order (not selector order): a `<header>`/`<section>`/`<footer>` that itself
  // straddles a boundary is checked, and if needed pushed, before its own descendant `<tr>`s are
  // checked, so a short landmark that gets pushed whole never needs its inner rows pushed again.
  const units = Array.from(document.querySelectorAll("tr, header, section, footer"));
  // `pageHeightPx`/`marginPx` are irrational (mm->96dpi-px conversions), so a unit whose true
  // bottom edge sits *exactly* on the content boundary lands a few thousandths of a px on either
  // side of it by sheer float-rounding coincidence, not a real layout difference -- confirmed
  // against a live repro: a 30+-row table's boundary row measured 0.003px past a page's content
  // boundary here while the real PDF placed that same row's bottom edge 8+pt *inside* the
  // previous page, comfortably fitting; treating that as a straddle pushed it (and the whole rest
  // of the document after it, including the footer) down by a full row's height for nothing. A
  // tiny epsilon absorbs that noise. It must still catch a genuine sub-pixel straddle: a
  // `<footer>` protected by `.ap-row`'s sibling rule `header, section, footer { break-inside:
  // avoid }` cannot be split at all, so Chromium pushes it whole to the next page even when it
  // overflows by a fraction of a px (confirmed against a second live repro: a footer measured
  // only 0.47px past this same boundary, yet the real PDF gave it an entire page to itself) --
  // two orders of magnitude above the float-noise case above, so one small epsilon well between
  // the two (not the earlier 24px-for-landmarks/0px-for-rows split, which got both of these
  // backwards at once) resolves both without re-breaking the other.
  const OVERFLOW_EPSILON_PX = 0.1;
  for (const unit of units) {
    const rect = unit.getBoundingClientRect();
    const pageOfTop = Math.floor(rect.top / pageHeightPx);
    // A page's *printable* content zone is only `pageHeightPx - 2*marginPx` tall -- the top and
    // bottom `marginPx` of every `pageHeightPx` slot are blank print margin, not flowing content
    // (Chromium's real print engine fits that much on-screen content per physical page, then
    // frames it with margin whitespace; the margin itself never "consumes" content-flow space).
    // A straddle is a unit whose bottom edge reaches past *this page's own content boundary*,
    // not merely past the next whole multiple of `pageHeightPx` -- using the latter (as an
    // earlier version of this script did) under-detects by up to `marginPx`+`marginPx` per page,
    // silently leaving the last row or two of *every* page that overflows a page attributed to
    // the page before the one it actually prints on (PRD §10 M2.4: this exact drift was caught by
    // the pdfplumber self-check, not by eye -- Chromium still renders a plausible-looking PDF
    // either way, just not at the coordinates this script predicted).
    const contentEndOfThisPage = (pageOfTop + 1) * pageHeightPx - marginPx;
    if (rect.bottom > contentEndOfThisPage + OVERFLOW_EPSILON_PX) {
      const targetTop = (pageOfTop + 1) * pageHeightPx + marginPx;
      // Inserting a new (zero-margin) sibling before `unit` shifts `unit` and everything after it
      // down by exactly the sibling's own height, however `unit`'s own margin-block-start is
      // composed -- that margin still applies in full between the spacer and `unit`, exactly as
      // it did between `unit` and whatever preceded it before the spacer was inserted, so it
      // contributes nothing to how far the shift needs to be. A real browser's print engine does
      // trim a fragment-starting box's own block-start margin to zero (CSS Fragmentation: "the
      // used value of a margin that would appear at the start of a fragment is zero"), but that
      // trim already happened to the *real* PDF's positions this label set is trying to match:
      // the target position is the destination page's own content-zone top, full stop, and a
      // plain `targetTop - rect.top` gets `unit` there regardless of its margin. Subtracting the
      // margin again here (an earlier version of this script did, reasoning that the trim needed
      // to be replicated by hand) overcorrects by exactly that margin -- confirmed against a live
      // repro: a `<section class="totals">` with an unrelated 4mm `margin-block-start` landed its
      // labels 4mm short of the real PDF's totals row positions once it was the one pushed.
      const gap = targetTop - rect.top;
      if (gap > 0) {
        const isRow = unit.tagName === "TR";
        const spacer = document.createElement(isRow ? "tr" : "div");
        spacer.setAttribute("data-ap-spacer", isRow ? "row" : "block");
        if (isRow) {
          const td = document.createElement("td");
          td.colSpan = Math.max(1, unit.children.length);
          td.style.cssText = "height:" + gap + "px;margin:0;padding:0;border:0;";
          spacer.appendChild(td);
        } else {
          spacer.style.cssText = "height:" + gap + "px;margin:0;padding:0;border:0;";
        }
        unit.parentElement.insertBefore(spacer, unit);
      }
    }
  }

  const fieldEls = Array.from(document.querySelectorAll("[data-field]"));
  const fields = fieldEls.map((el, index) => {
    const rowEl = el.closest("tr.ap-row");
    let lineIndex = null;
    if (rowEl) {
      const tbody = rowEl.closest("tbody");
      const siblingRows = tbody
        ? Array.from(tbody.querySelectorAll(":scope > tr.ap-row"))
        : [rowEl];
      lineIndex = siblingRows.indexOf(rowEl);
    }
    // A Range over the field's own rendered text (not `el.getClientRects()` on the element)
    // so the box always comes from the actual inline text run, never a block/cell ancestor's
    // box, however this field happens to be wrapped (plain text, or `ltr_isolate`'s nested
    // `<bdi>`) -- PRD §10 M2.4 / M3.1's IoU gate against pdfplumber's own tight word boxes.
    const range = document.createRange();
    range.selectNodeContents(el);
    const rects = Array.from(range.getClientRects()).map((r) => ({
      left: r.left,
      top: r.top,
      width: r.width,
      height: r.height,
    }));
    return {
      elementIndex: index,
      field: el.getAttribute("data-field"),
      text: el.textContent,
      lineIndex: lineIndex,
      rects: rects,
    };
  });

  const totalHeight = sheet.getBoundingClientRect().bottom;
  const totalPages = Math.max(1, Math.ceil(totalHeight / pageHeightPx));

  return { fields: fields, totalPages: totalPages };
}
"""


_LOAD_FONTS_JS = """
async () => {
  const faces = Array.from(document.fonts);
  if (faces.length === 0) {
    throw new Error("factory.render: no @font-face rules registered (would fall back)");
  }
  await Promise.all(faces.map((face) => face.load()));
  await document.fonts.ready;
}
"""

_ZERO_MARGIN_PAGE_CSS = "@page { size: A4; margin: 0 !important; }"
"""Overrides the templates' own `@page { margin: 14mm }` for the final print only; the
pagination script's spacers stand in for that margin (see `RenderWorker.render`)."""


@dataclass(frozen=True)
class RenderedInvoice:
    """One rendered invoice document: the digital PDF (source of truth), its DOM-box labels in
    PDF-page coordinates, and — only when requested — one clean PNG screenshot per page for the
    Augraphy scan-variant path."""

    pdf_bytes: bytes
    labels: LabelSet
    page_pngs: tuple[bytes, ...]


# Observed in practice: under many concurrent Chromium processes, `page.screenshot()`
# occasionally returns a PNG with a corrupt `IDAT` chunk (a libpng "CRC error" logged to stderr,
# not a Python exception) -- and a corrupted-but-still-decodable chunk can render as a plausible-
# looking image, so decoding without an exception is not enough to trust the bytes. A page's
# *content* can legitimately be sparse (e.g. a mostly-blank second page with only the totals
# spilling over), so corruption is detected structurally -- every PNG chunk's own CRC32, which is
# exactly what libpng's warning is about -- rather than by any pixel-content heuristic.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_SCREENSHOT_ATTEMPTS = 3


def _png_chunk_crcs_are_valid(data: bytes) -> bool:
    if not data.startswith(_PNG_SIGNATURE):
        return False
    offset = len(_PNG_SIGNATURE)
    size = len(data)
    saw_iend = False
    while offset + 8 <= size:
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunk_type = data[offset + 4 : offset + 8]
        data_end = offset + 8 + length
        crc_end = data_end + 4
        if crc_end > size:
            return False  # truncated mid-chunk
        (stored_crc,) = struct.unpack(">I", data[data_end:crc_end])
        computed_crc = zlib.crc32(data[offset + 4 : data_end]) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            return False
        if chunk_type == b"IEND":
            saw_iend = True
            break
        offset = crc_end
    return saw_iend


def _capture_page_screenshot(page: Page, *, y: float) -> bytes:
    """Screenshots one page-sized region and retries if the PNG comes back corrupt.
    Raises `RuntimeError` if every attempt is corrupt."""
    last_error: Exception | None = None
    for _attempt in range(_MAX_SCREENSHOT_ATTEMPTS):
        data = page.screenshot(
            clip={"x": 0, "y": y, "width": PAGE_WIDTH_PX, "height": PAGE_HEIGHT_PX},
            scale="css",
            type="png",
        )
        if not _png_chunk_crcs_are_valid(data):
            last_error = ValueError("screenshot PNG failed a chunk CRC32 check")
            continue
        try:
            with Image.open(BytesIO(data)) as image:
                image.load()
        except (OSError, UnidentifiedImageError) as exc:
            last_error = exc
            continue
        return data
    raise RuntimeError(
        f"factory.render: page screenshot at y={y} was corrupt after "
        f"{_MAX_SCREENSHOT_ATTEMPTS} attempts"
    ) from last_error


class RenderWorker:
    """One Chromium browser + context, reused across many `render()` calls. Not thread-safe;
    create one instance per worker process when parallelizing
    ("parallelize rendering... multiple browser contexts/processes"), typically via
    `factory.build`'s
    process-pool initializer."""

    def __init__(self, *, scale: float = 2.0) -> None:
        self._scale = scale
        self._playwright: Playwright = sync_playwright().start()
        self._browser: Browser = self._playwright.chromium.launch()
        self._context: BrowserContext = self._browser.new_context(
            viewport={"width": round(PAGE_WIDTH_PX), "height": round(PAGE_HEIGHT_PX * 4)},
            device_scale_factor=scale,
        )

    def close(self) -> None:
        self._context.close()
        self._browser.close()
        self._playwright.stop()

    def __enter__(self) -> RenderWorker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def render(
        self,
        html: str,
        *,
        layout_id: str,
        header_field_names: frozenset[str],
        line_field_names: frozenset[str],
        capture_pages: bool = False,
    ) -> RenderedInvoice:
        """Renders `html` (from `factory.render_html.render_template`) to a `RenderedInvoice`.
        `capture_pages=True` also returns one clean PNG per page (needed only for the 30% of
        invoices that get an Augraphy scan variant; skipped otherwise to save time)."""
        page = self._context.new_page()
        try:
            page.set_content(html, wait_until="load")
            # The bundled Noto faces are `font-display: swap` data URIs that load lazily: "load"
            # can fire before they are decoded, and measuring or printing then uses a fallback
            # system font (on Linux, one with no usable Arabic ToUnicode map). Load every face
            # explicitly so the labels and the PDF both come from the Noto layout.
            page.evaluate(_LOAD_FONTS_JS)

            natural_height = cast(float, page.evaluate("document.documentElement.scrollHeight"))
            # Headroom for the spacers the pagination script is about to insert: a small top
            # margin plus, pessimistically, up to a few whole pages' worth of forced row breaks.
            viewport_height = round(natural_height + PAGE_HEIGHT_PX * 5)
            page.set_viewport_size({"width": round(PAGE_WIDTH_PX), "height": viewport_height})

            raw = cast(
                RawPagination,
                page.evaluate(
                    _PAGINATE_JS,
                    {"pageHeightPx": PAGE_HEIGHT_PX, "marginPx": MARGIN_PX},
                ),
            )
            # Print the *paginated* DOM, not the original: its spacers already reserve every
            # page's top/bottom margin and push every unit clear of each page cut, so with a
            # zero print margin Chromium cuts at exact multiples of the page height and has
            # nothing left to re-break. PDF coordinates then equal the measured DOM coordinates
            # by construction instead of by simulating Chromium's fragmentation.
            page.add_style_tag(content=_ZERO_MARGIN_PAGE_CSS)
            pdf_bytes = page.pdf(
                width=f"{PAGE_WIDTH_MM}mm",
                height=f"{PAGE_HEIGHT_MM}mm",
                margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"},
                print_background=True,
            )
            pdf_bytes = _normalize_pdf_bytes(pdf_bytes)
            label_set = build_label_set(
                raw,
                layout_id=layout_id,
                page_width_px=PAGE_WIDTH_PX,
                page_height_px=PAGE_HEIGHT_PX,
                header_field_names=header_field_names,
                line_field_names=line_field_names,
            )

            page_pngs: tuple[bytes, ...] = ()
            if capture_pages:
                page_pngs = tuple(
                    _capture_page_screenshot(page, y=page_index * PAGE_HEIGHT_PX)
                    for page_index in range(label_set.n_pages)
                )

            return RenderedInvoice(pdf_bytes=pdf_bytes, labels=label_set, page_pngs=page_pngs)
        finally:
            page.close()


def css_page_geometry_matches(css_text: str) -> bool:
    """True if `css_text` (the contents of a stylesheet) declares the same A4-size, 14mm-margin
    `@page` rule this module's constants assume. Used by a test to catch silent drift if
    `factory/templates/_shared.html.j2`'s `@page` rule ever changes."""
    return "size: A4" in css_text and f"margin: {MARGIN_MM:.0f}mm" in css_text
