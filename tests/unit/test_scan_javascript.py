"""Unit tests for the active-content technique (PRD §6.3 A10): `/OpenAction`,
`/Names/JavaScript` and a page-level `/AA` are all flagged PDF_JAVASCRIPT (M6.2)."""

from __future__ import annotations

from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb


def test_open_action_javascript_is_flagged() -> None:
    flags = PdfHiddenContentDetector().detect(pb.javascript_open_action_pdf())

    assert any(
        f.kind is ScanFlagKind.PDF_JAVASCRIPT and "OpenAction" in f.evidence.text for f in flags
    )


def test_names_javascript_is_flagged() -> None:
    flags = PdfHiddenContentDetector().detect(pb.javascript_names_pdf())

    assert any(
        f.kind is ScanFlagKind.PDF_JAVASCRIPT and "JavaScript" in f.evidence.text for f in flags
    )


def test_page_additional_actions_are_flagged() -> None:
    flags = PdfHiddenContentDetector().detect(pb.page_additional_actions_pdf())

    js = [f for f in flags if f.kind is ScanFlagKind.PDF_JAVASCRIPT]
    assert any("/AA" in f.evidence.text for f in js)
    assert any(f.page == 0 for f in js)


def test_clean_invoice_never_flags_javascript() -> None:
    flags = PdfHiddenContentDetector().detect(pb.clean_invoice_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.PDF_JAVASCRIPT]
