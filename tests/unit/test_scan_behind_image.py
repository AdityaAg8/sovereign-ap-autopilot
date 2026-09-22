"""Unit test for the covered-by-image/opaque-shape technique (PRD §6.3 A04): text painted over
by a later, opaque rectangle is flagged HIDDEN_TEXT even though its declared colour has high
contrast against the page (M6.2)."""

from __future__ import annotations

from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb


def test_text_covered_by_a_later_rectangle_is_flagged_hidden() -> None:
    flags = PdfHiddenContentDetector().detect(pb.covered_text_pdf())

    covered = [
        f for f in flags if f.kind is ScanFlagKind.HIDDEN_TEXT and "covered" in f.evidence.text
    ]
    assert covered, "expected at least one HIDDEN_TEXT flag for the covered text"
    assert all(f.page == 0 for f in covered)


def test_clean_invoice_never_flags_covered_text() -> None:
    flags = PdfHiddenContentDetector().detect(pb.clean_invoice_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.HIDDEN_TEXT]
