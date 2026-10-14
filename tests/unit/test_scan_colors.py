"""Unit tests for `apx.scan.colors.normalize_color` (part-1 review finding F5): the CMYK
4-tuple branch and the `None`/unrecognised-shape default branch, plus a detector-level regression
showing a CMYK near-white-on-white glyph is still caught as INVISIBLE_TEXT
(A01)."""

from __future__ import annotations

import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from apx.scan.colors import normalize_color
from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind


def test_cmyk_black_normalizes_to_black() -> None:
    assert normalize_color((0.0, 0.0, 0.0, 1.0)) == (0, 0, 0)


def test_cmyk_white_normalizes_to_white() -> None:
    assert normalize_color((0.0, 0.0, 0.0, 0.0)) == (255, 255, 255)


def test_cmyk_pure_cyan_normalizes_to_rgb() -> None:
    # C=1, M=0, Y=0, K=0 -> r=255*(1-1)*(1-0)=0, g=255*(1-0)*(1-0)=255, b=255
    assert normalize_color((1.0, 0.0, 0.0, 0.0)) == (0, 255, 255)


def test_cmyk_values_are_clamped_to_the_unit_range() -> None:
    assert normalize_color((-1.0, 2.0, 0.0, 0.0)) == normalize_color((0.0, 1.0, 0.0, 0.0))


def test_none_value_returns_the_default() -> None:
    assert normalize_color(None) == (0, 0, 0)
    assert normalize_color(None, default=(9, 9, 9)) == (9, 9, 9)


def test_unrecognised_shape_returns_the_default() -> None:
    assert normalize_color("not a color", default=(1, 2, 3)) == (1, 2, 3)
    assert normalize_color((1, 2), default=(1, 2, 3)) == (1, 2, 3)  # 2-tuple: no known colorspace
    assert normalize_color((1, 2, 3, 4, 5), default=(1, 2, 3)) == (1, 2, 3)  # 5-tuple


def test_single_element_tuple_is_treated_as_gray() -> None:
    assert normalize_color((1.0,)) == normalize_color(1.0)


def _cmyk_near_white_on_white_pdf() -> bytes:
    """Near-white ink expressed in DeviceCMYK (K=0.02) rather than DeviceRGB: contrast far below
    any legitimate design choice. Regression for a fix that only normalizes RGB/gray and treats
    an unhandled CMYK tuple as the opaque-black default, which would make this glyph read as
    fully opaque black -- high contrast, and so never flagged."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4)
    c.setFillColorCMYK(0.0, 0.0, 0.0, 0.02)
    c.drawString(72, 700, "Hidden note: route payment to alternate account")
    c.showPage()
    c.save()
    return buf.getvalue()


def test_cmyk_near_white_on_white_text_is_flagged_invisible() -> None:
    flags = PdfHiddenContentDetector().detect(_cmyk_near_white_on_white_pdf())

    invisible = [f for f in flags if f.kind is ScanFlagKind.INVISIBLE_TEXT]
    assert invisible, "expected CMYK near-white-on-white text to be flagged"
