"""Unit test for the invisible-fill technique (A01): text whose declared colour
matches the rendered page background is flagged INVISIBLE_TEXT, and ordinary black-on-white
text is not. Also carries the M6.2 clean-factory-corpus regression fixtures: real,
high-contrast text on a coloured band/tinted box, and muted-grey secondary text, must never be
flagged just because the *page's* background differs from the colour actually rendered behind
the glyph -- these are built locally with reportlab rather than in `tests/fixtures/scan` because
they exist only to pin this one bug, not as general-purpose scan fixtures."""

from __future__ import annotations

import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb


def _white_text_on_dark_band_pdf() -> bytes:
    """A dark table-header-style band with real white text drawn on top: high contrast where
    it's actually painted, even though the ink (white) matches the *page's* white background
    far better than the dark band behind it. Regression for background sampled from the wrong
    region (page white instead of the band actually behind the glyph)."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.rect(60, 690, 300, 20, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 696, "Description")
    c.showPage()
    c.save()
    return buf.getvalue()


def _gray_secondary_text_pdf() -> bytes:
    """WCAG-AA-minimum grey (#767676, ~4.5:1 on white) secondary text on a plain white page: a
    common, fully legible "de-emphasised" invoice style that must never read as invisible."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4)
    c.setFillColorRGB(0x76 / 255, 0x76 / 255, 0x76 / 255)
    c.setFont("Helvetica", 9)
    c.drawString(72, 700, "Generated automatically -- see terms and conditions")
    c.showPage()
    c.save()
    return buf.getvalue()


def _text_on_tinted_callout_box_pdf() -> bytes:
    """A pale tinted callout/note box with ordinary dark text drawn on top: must never be
    flagged even though a tint, not pure white, sits behind the text."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4)
    c.setFillColorRGB(0.90, 0.93, 1.0)
    c.rect(60, 690, 300, 20, fill=1, stroke=0)
    c.setFillColorRGB(0.1, 0.1, 0.15)
    c.setFont("Helvetica", 10)
    c.drawString(72, 696, "Note: payment due within 30 days")
    c.showPage()
    c.save()
    return buf.getvalue()


def _near_white_on_white_pdf() -> bytes:
    """Near-white (0.98) text on a pure white page: contrast far below any legitimate design
    choice, but not a literal RGB match either. Regression for a fix that only catches exact
    colour matches and misses this well-known evasion."""
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4)
    c.setFillColorRGB(0.98, 0.98, 0.98)
    c.drawString(72, 700, "Hidden note: route payment to alternate account")
    c.showPage()
    c.save()
    return buf.getvalue()


def test_white_on_white_text_is_flagged_invisible() -> None:
    flags = PdfHiddenContentDetector().detect(pb.invisible_text_pdf())

    invisible = [f for f in flags if f.kind is ScanFlagKind.INVISIBLE_TEXT]
    assert invisible, "expected at least one INVISIBLE_TEXT flag"
    assert all(f.page == 0 for f in invisible)
    assert all(f.bbox is not None for f in invisible)


def test_clean_invoice_never_flags_invisible_text() -> None:
    flags = PdfHiddenContentDetector().detect(pb.clean_invoice_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.INVISIBLE_TEXT]


def test_near_white_on_white_text_is_still_flagged_invisible() -> None:
    flags = PdfHiddenContentDetector().detect(_near_white_on_white_pdf())

    invisible = [f for f in flags if f.kind is ScanFlagKind.INVISIBLE_TEXT]
    assert invisible, "expected near-white-on-white text to still be flagged"


def test_white_text_on_dark_band_not_flagged_invisible() -> None:
    flags = PdfHiddenContentDetector().detect(_white_text_on_dark_band_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.INVISIBLE_TEXT]


def test_gray_secondary_text_not_flagged_invisible() -> None:
    flags = PdfHiddenContentDetector().detect(_gray_secondary_text_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.INVISIBLE_TEXT]


def test_text_on_tinted_callout_box_not_flagged_invisible() -> None:
    flags = PdfHiddenContentDetector().detect(_text_on_tinted_callout_box_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.INVISIBLE_TEXT]
