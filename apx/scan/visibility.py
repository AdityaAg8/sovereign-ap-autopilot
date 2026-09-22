"""Compares pdfplumber's parsed characters against a pypdfium2 render of the same page to find
text a human would never see: invisible fill, sub-4pt font, off-page/clipped positioning and
text covered by a later-drawn image or opaque shape (PRD §3, §6.3 A01-A04, M6.2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pdfplumber.page

from apx.common.types import BBox
from apx.scan.colors import RGB, color_distance, contrast_ratio, normalize_color
from apx.scan.interfaces import ScanFlag, ScanFlagKind
from apx.scan.render import RGBImage
from apx.scan.text import build_evidence

PdfChar = dict[str, Any]
"""pdfplumber's per-character dict (`page.chars` items); see `apx.scan.controls.PdfChar`."""

TINY_FONT_PT = 4.0
"""Font size below which text is considered unreadable by a human (PRD §3 A02)."""

INVISIBLE_CONTRAST_RATIO = 1.2
"""WCAG relative-luminance contrast ratio, between a glyph's actual rendered ink and its actual
rendered background, below which text counts as invisible (A01). 1.0 is identical luminance
(perfect invisibility, e.g. true white-on-white); WCAG's own floors are far higher -- 3:1 for
non-text elements, 4.5:1 for AA body text, and even the palest WCAG-AA-legal "secondary text"
grey (#767676 on white) sits at ~4.5:1. 1.2 leaves headroom only for anti-aliasing and render
rounding noise near the 1.0 floor, so it never catches a legitimate low-contrast design choice
(muted grey text, text on a tinted callout box) while still catching near-white-on-white."""

INK_FOUND_DISTANCE = 90.0
"""Max RGB distance from the declared ink colour, at any sampled pixel inside the glyph's own
box, that still counts as "the glyph is visible there". Above this, nothing resembling the
declared ink ever appears in the render, so the glyph must be covered by something opaque.
Reused to recognise ring pixels that are plausibly the glyph's own ink when sampling the
background (see `_sample_background`)."""

MIN_PAGE_OVERLAP = 0.9
"""Minimum fraction of a character's box that must fall inside the page for it to count as
fully on-page; anything less is off-page or clipped (A03)."""

MIN_ADVANCE_WIDTH_PT = 0.5
"""Below this, a character has no real glyph box (a zero-width space or joiner). Those are
`apx.scan.controls`'s job (A05/A06), not a visibility technique."""

_MARGIN_PX = 2
"""Raster margin, in pixels, of the first ring sampled just outside a glyph's box for the
background colour."""

_MARGIN_STEP_PX = 2
"""How far the ring expands outward, in pixels, each time every sampled point still looks like
the glyph's own ink (see `_sample_background`)."""

_MAX_MARGIN_PX = 8
"""Largest ring margin tried before giving up and treating the glyph as boxed in by same-
coloured content on every side."""

_EDGE_SAMPLES = 8
"""Points sampled along each edge of the ring (including corners). More than the 4 corners
alone: a glyph's declared box commonly touches, or is touched by, a neighbouring glyph's own
antialiased ink at one or two points on the ring, especially for bold/kerned text or fonts whose
declared ascent/descent metrics run taller than their real ink; spreading samples along the full
edge finds a genuine gap between letters that a lone corner sample would miss."""

_FOREGROUND_MIN_RELATIVE_DISTANCE = 0.6
"""A glyph's own box is mostly empty space for many characters (a period's ink is a handful of
pixels near the baseline inside a box as tall as the font's full ascent-descent), so a fixed
*fraction of the box's pixels* would dilute the ink estimate with background padding. Instead,
keep every pixel at least this fraction of the way from the background to the single most
extreme pixel found, then take their median. That keeps the estimate anchored to genuine ink
regardless of how small a share of the box it occupies, while still "accounting for anti-
aliasing": partially-blended edge pixels (below the cutoff) are excluded, and the median absorbs
any single-pixel render noise at the extreme end."""


def scan_page_visibility(
    page: pdfplumber.page.Page, page_index: int, rendered: RGBImage, dpi: int
) -> list[ScanFlag]:
    """Flag hidden text on one page. `rendered` is the page rasterised at `dpi` (see
    `apx.scan.render.render_page`). Never raises for a page with no hidden text (returns [])."""
    flags: list[ScanFlag] = []
    scale = dpi / 72.0
    width, height = float(page.width), float(page.height)
    img_h, img_w = rendered.shape[0], rendered.shape[1]

    for char in page.chars:
        text = str(char["text"])
        if not text.strip():
            continue
        if float(char["x1"]) - float(char["x0"]) < MIN_ADVANCE_WIDTH_PT:
            continue  # zero/near-zero width glyph: not a visibility technique (A05/A06 instead)
        size = float(char["size"])
        bbox = _normalized_bbox(char, page_index, width, height)

        if size < TINY_FONT_PT:
            flags.append(
                _flag(
                    ScanFlagKind.HIDDEN_TEXT, page_index, bbox, f"tiny font ({size:.1f}pt): {text}"
                )
            )
            continue

        overlap = _page_overlap_fraction(char, width, height)
        if overlap < MIN_PAGE_OVERLAP:
            flags.append(
                _flag(ScanFlagKind.HIDDEN_TEXT, page_index, bbox, f"off-page/clipped: {text}")
            )
            continue

        px0, py0, px1, py1 = _pixel_box(char, scale, img_w, img_h)
        if px1 <= px0 or py1 <= py0:
            continue  # degenerate (zero-area) box on an otherwise on-page glyph; nothing to sample

        ink = normalize_color(char.get("non_stroking_color"))

        # Covered-by-image/shape (A04) first: if the glyph's *declared* colour never appears
        # anywhere in its own rendered box, something opaque was painted over it after the text
        # -- there is no rendered ink to measure a contrast ratio from at all.
        if not _ink_visible_in_render(rendered, px0, py0, px1, py1, ink):
            flags.append(
                _flag(ScanFlagKind.HIDDEN_TEXT, page_index, bbox, f"covered by image/shape: {text}")
            )
            continue

        # Invisible fill (A01): the glyph *was* rendered in (approximately) its declared colour,
        # so judge visibility from what actually got rendered -- the background around the glyph
        # and the ink inside it -- rather than trusting the declared colour for both sides of the
        # comparison (PRD M6.2 fix: declared colour vs. a background sample that can land back on
        # the glyph's own ink gave false positives on ordinary coloured/gray text).
        background = _sample_background(rendered, px0, py0, px1, py1, img_w, img_h, ink)
        foreground = _measure_foreground(rendered, px0, py0, px1, py1, background)
        ratio = contrast_ratio(foreground, background)
        if ratio < INVISIBLE_CONTRAST_RATIO:
            flags.append(
                _flag(
                    ScanFlagKind.INVISIBLE_TEXT,
                    page_index,
                    bbox,
                    f"invisible fill: {text} fg={foreground} bg={background} contrast={ratio:.2f}",
                )
            )

    return flags


def _flag(kind: ScanFlagKind, page_index: int, bbox: BBox | None, excerpt: str) -> ScanFlag:
    return ScanFlag(
        kind=kind, page=page_index, score=1.0, evidence=build_evidence(kind, excerpt), bbox=bbox
    )


def _normalized_bbox(char: PdfChar, page_index: int, width: float, height: float) -> BBox | None:
    if width <= 0 or height <= 0:
        return None
    x0 = max(0.0, min(1.0, float(char["x0"]) / width))
    x1 = max(0.0, min(1.0, float(char["x1"]) / width))
    y0 = max(0.0, min(1.0, float(char["top"]) / height))
    y1 = max(0.0, min(1.0, float(char["bottom"]) / height))
    return BBox(page=page_index, x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1))


def _page_overlap_fraction(char: PdfChar, width: float, height: float) -> float:
    x0, x1 = float(char["x0"]), float(char["x1"])
    top, bottom = float(char["top"]), float(char["bottom"])
    area = (x1 - x0) * (bottom - top)
    if area <= 0:
        # Zero-width/zero-height glyph (e.g. a zero-width space): there is no box to intersect,
        # so judge it by whether its anchor point falls on the page.
        return 1.0 if (0.0 <= x0 <= width and 0.0 <= top <= height) else 0.0
    ix0, ix1 = max(x0, 0.0), min(x1, width)
    iy0, iy1 = max(top, 0.0), min(bottom, height)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / area


def _pixel_box(char: PdfChar, scale: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x0 = max(0, int(float(char["x0"]) * scale))
    x1 = min(img_w, round(float(char["x1"]) * scale) + 1)
    y0 = max(0, int(float(char["top"]) * scale))
    y1 = min(img_h, round(float(char["bottom"]) * scale) + 1)
    return x0, y0, x1, y1


def _sample_background(
    image: RGBImage, x0: int, y0: int, x1: int, y1: int, img_w: int, img_h: int, ink: RGB
) -> RGB:
    """Sample the true rendered background around a glyph's box.

    Expands a ring of points just outside the box and discards any point that looks like the
    glyph's own ink (an antialiased edge, or a neighbouring glyph's ink in the same colour --
    common in bold or tightly kerned text, and in fonts whose declared ascent/descent run taller
    than their real ink). If a ring is entirely ink-like, it widens the ring and tries again, up
    to `_MAX_MARGIN_PX`. Only if the glyph is boxed in by same-coloured content at every margin
    does it fall back to that ring (the closest observable thing to "background" left)."""
    margin = _MARGIN_PX
    ring: list[RGB] = []
    while True:
        ring = _ring_pixels(image, x0, y0, x1, y1, margin, img_w, img_h)
        distinct = [pixel for pixel in ring if color_distance(pixel, ink) > INK_FOUND_DISTANCE]
        if distinct:
            return _median_rgb(distinct)
        if margin >= _MAX_MARGIN_PX:
            return _median_rgb(ring)
        margin += _MARGIN_STEP_PX


def _ring_pixels(
    image: RGBImage, x0: int, y0: int, x1: int, y1: int, margin: int, img_w: int, img_h: int
) -> list[RGB]:
    """Pixels along the perimeter of the glyph's box expanded by `margin`, sampled at
    `_EDGE_SAMPLES` points per edge rather than just the four corners."""
    ex0 = max(0, x0 - margin)
    ey0 = max(0, y0 - margin)
    ex1 = min(img_w - 1, x1 + margin)
    ey1 = min(img_h - 1, y1 + margin)
    xs = _spread(ex0, ex1, _EDGE_SAMPLES)
    ys = _spread(ey0, ey1, _EDGE_SAMPLES)
    points = {(x, ey0) for x in xs} | {(x, ey1) for x in xs}
    points |= {(ex0, y) for y in ys} | {(ex1, y) for y in ys}
    return [(int(image[y, x][0]), int(image[y, x][1]), int(image[y, x][2])) for x, y in points]


def _spread(low: int, high: int, count: int) -> list[int]:
    """`count` integer positions evenly spread across `[low, high]`, deduplicated."""
    if high <= low:
        return [low]
    return sorted({low + round(i * (high - low) / (count - 1)) for i in range(count)})


def _median_rgb(pixels: Sequence[RGB]) -> RGB:
    reds = sorted(p[0] for p in pixels)
    greens = sorted(p[1] for p in pixels)
    blues = sorted(p[2] for p in pixels)
    mid = len(pixels) // 2
    return (reds[mid], greens[mid], blues[mid])


def _measure_foreground(
    image: RGBImage, x0: int, y0: int, x1: int, y1: int, background: RGB
) -> RGB:
    """The glyph's own actual rendered ink colour, measured from the render rather than trusted
    from its declared fill colour: the (robust, anti-aliasing-tolerant) colour of the pixels
    inside its box that differ the most from the measured background. If every pixel matches the
    background, the glyph truly has no visible ink and this returns the background itself."""
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return background
    pixels = crop.reshape(-1, 3).astype(np.int64)
    bg = np.array(background, dtype=np.int64)
    distances = ((pixels - bg) ** 2).sum(axis=-1)
    max_distance = int(distances.max())
    if max_distance <= 0:
        return background  # every pixel in the box exactly matches the background: no ink at all
    most_different = pixels[distances >= max_distance * _FOREGROUND_MIN_RELATIVE_DISTANCE]
    median = np.median(most_different, axis=0)
    return (int(median[0]), int(median[1]), int(median[2]))


def _ink_visible_in_render(image: RGBImage, x0: int, y0: int, x1: int, y1: int, ink: RGB) -> bool:
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return True
    diff = crop.astype("int32") - [[ink]]
    distances = (diff**2).sum(axis=-1) ** 0.5
    return bool(distances.min() <= INK_FOUND_DISTANCE)
