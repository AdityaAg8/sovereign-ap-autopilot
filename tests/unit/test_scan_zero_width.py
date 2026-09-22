"""Unit test for the zero-width character technique (PRD §6.3 A05): zero-width spaces stitched
into Latin text are flagged ZERO_WIDTH, and plain text is not (M6.2)."""

from __future__ import annotations

from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb

ZERO_WIDTH_SPACE_CODEPOINT = 0x200B
"""U+200B: the character `pb.zero_width_pdf` stitches into its payload text."""


def test_zero_width_space_in_latin_text_is_flagged() -> None:
    flags = PdfHiddenContentDetector().detect(pb.zero_width_pdf())

    zero_width = [f for f in flags if f.kind is ScanFlagKind.ZERO_WIDTH]
    assert zero_width, "expected at least one ZERO_WIDTH flag"
    assert all(f.page == 0 for f in zero_width)

    escaped = "\\u" + format(ZERO_WIDTH_SPACE_CODEPOINT, "04x")
    assert all(escaped in f.evidence.text for f in zero_width)


def test_clean_invoice_never_flags_zero_width() -> None:
    flags = PdfHiddenContentDetector().detect(pb.clean_invoice_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.ZERO_WIDTH]
