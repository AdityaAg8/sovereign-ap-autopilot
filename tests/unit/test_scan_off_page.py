"""Unit test for the off-page/clipped technique (PRD §6.3 A03): text drawn outside the page or
mostly clipped by its edge is flagged HIDDEN_TEXT (M6.2)."""

from __future__ import annotations

from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb


def test_off_page_text_is_flagged_hidden() -> None:
    flags = PdfHiddenContentDetector().detect(pb.off_page_text_pdf())

    hidden = [f for f in flags if f.kind is ScanFlagKind.HIDDEN_TEXT]
    assert hidden, "expected at least one HIDDEN_TEXT flag"
    assert all("off-page" in f.evidence.text for f in hidden)


def test_clipped_text_is_flagged_hidden() -> None:
    flags = PdfHiddenContentDetector().detect(pb.clipped_text_pdf())

    assert any(f.kind is ScanFlagKind.HIDDEN_TEXT and "off-page" in f.evidence.text for f in flags)


def test_clean_invoice_never_flags_off_page_text() -> None:
    flags = PdfHiddenContentDetector().detect(pb.clean_invoice_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.HIDDEN_TEXT]
