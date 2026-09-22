"""Unit test for the tiny-font technique (PRD §6.3 A02): text below 4pt is flagged HIDDEN_TEXT,
and normal 12pt body text is not (M6.2)."""

from __future__ import annotations

from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb


def test_sub_4pt_text_is_flagged_hidden() -> None:
    flags = PdfHiddenContentDetector().detect(pb.tiny_font_pdf(size=2.0))

    hidden = [f for f in flags if f.kind is ScanFlagKind.HIDDEN_TEXT]
    assert hidden, "expected at least one HIDDEN_TEXT flag"
    assert all("tiny font" in f.evidence.text for f in hidden)


def test_font_just_under_the_threshold_is_flagged() -> None:
    flags = PdfHiddenContentDetector().detect(pb.tiny_font_pdf(size=3.9))

    assert any(f.kind is ScanFlagKind.HIDDEN_TEXT and "tiny font" in f.evidence.text for f in flags)


def test_clean_invoice_never_flags_tiny_font() -> None:
    flags = PdfHiddenContentDetector().detect(pb.clean_invoice_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.HIDDEN_TEXT]
