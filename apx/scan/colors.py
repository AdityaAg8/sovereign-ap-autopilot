"""Colour helpers for the hidden-content scan: normalises pdfplumber's per-glyph fill colour
(gray, RGB or CMYK, in any 0..1 form) to 8-bit RGB, measures a raw colour distance used to spot
a glyph's own ink in a render, and computes WCAG relative luminance/contrast so
`apx.scan.visibility` can judge whether rendered text is actually visible against its rendered
background (PRD §3 M6.2)."""

from __future__ import annotations

RGB = tuple[int, int, int]

_BLACK: RGB = (0, 0, 0)


def normalize_color(value: object, default: RGB = _BLACK) -> RGB:
    """Convert a pdfplumber `non_stroking_color`/`stroking_color` value to 8-bit RGB.

    Handles the shapes pdfplumber (via pdfminer) produces: `None` (no colour set, PDF default
    is black), a single number (DeviceGray, 0..1), a 3-tuple (DeviceRGB, 0..1 each) or a 4-tuple
    (DeviceCMYK, 0..1 each). Any other shape falls back to `default`.
    """
    if value is None:
        return default
    if isinstance(value, int | float):
        gray = _clamp_unit(value)
        level = round(gray * 255)
        return (level, level, level)
    if isinstance(value, tuple | list):
        if len(value) == 1:
            return normalize_color(value[0], default)
        if len(value) == 3:
            r, g, b = (round(_clamp_unit(c) * 255) for c in value)
            return (r, g, b)
        if len(value) == 4:
            c, m, y, k = (_clamp_unit(v) for v in value)
            r = round(255 * (1 - c) * (1 - k))
            g = round(255 * (1 - m) * (1 - k))
            b = round(255 * (1 - y) * (1 - k))
            return (r, g, b)
    return default


def color_distance(a: RGB, b: RGB) -> float:
    """Euclidean distance between two RGB colours (0.0 for an exact match, ~441.7 max). Used to
    recognise "this pixel is plausibly the glyph's own ink colour", not to judge visibility --
    see `contrast_ratio` for that."""
    return float(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5)


def relative_luminance(rgb: RGB) -> float:
    """WCAG 2.x relative luminance of an sRGB colour, in [0.0, 1.0] (0 = black, 1 = white)."""
    r, g, b = (_linearize(channel / 255.0) for channel in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: RGB, b: RGB) -> float:
    """WCAG 2.x contrast ratio between two sRGB colours, in [1.0, 21.0]. 1.0 means the two
    colours have identical luminance (perceptually indistinguishable); 21.0 is black on white."""
    lum_a, lum_b = relative_luminance(a), relative_luminance(b)
    lighter, darker = (lum_a, lum_b) if lum_a >= lum_b else (lum_b, lum_a)
    return (lighter + 0.05) / (darker + 0.05)


def _linearize(channel: float) -> float:
    """sRGB -> linear-light gamma decoding, the first step of the WCAG luminance formula."""
    return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
