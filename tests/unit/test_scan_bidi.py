"""Unit test for the bidi-override technique (PRD §6.3 A06): a right-to-left override that can
reorder digits is always flagged BIDI_CONTROL, unlike normal RTL/Arabic bidi marks (M6.2)."""

from __future__ import annotations

from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb

RLO_CODEPOINT = 0x202E
"""U+202E RIGHT-TO-LEFT OVERRIDE: the character `pb.bidi_override_pdf` embeds."""


def test_rlo_override_is_always_flagged() -> None:
    flags = PdfHiddenContentDetector().detect(pb.bidi_override_pdf())

    bidi = [f for f in flags if f.kind is ScanFlagKind.BIDI_CONTROL]
    assert bidi, "expected at least one BIDI_CONTROL flag"
    assert all(f.page == 0 for f in bidi)

    escaped = "\\u" + format(RLO_CODEPOINT, "04x")
    assert any(escaped in f.evidence.text for f in bidi)


def test_clean_invoice_never_flags_bidi_control() -> None:
    flags = PdfHiddenContentDetector().detect(pb.clean_invoice_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.BIDI_CONTROL]
