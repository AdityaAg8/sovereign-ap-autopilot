"""Arabic false-positive tests for the control-character technique: normal
RTL text, ZWJ inside an Arabic word, and RLM bracketing a number in Arabic-script context must
never flag; the same code points outside that context (or a bidi override, in any context)
still do."""

from __future__ import annotations

from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb


def test_arabic_shaping_and_rlm_next_to_a_number_never_flags() -> None:
    """A realistic Arabic invoice line: ZWJ inside a word, RLM around a digit run. Zero flags,
    of any kind -- not just zero control-character flags."""
    flags = PdfHiddenContentDetector().detect(pb.arabic_clean_pdf())

    assert flags == []


def test_zwj_outside_arabic_script_still_flags() -> None:
    """The same code point (ZWJ) with no Arabic neighbour is not normal shaping."""
    flags = PdfHiddenContentDetector().detect(pb.zwj_in_latin_text_pdf())

    zero_width = [f for f in flags if f.kind is ScanFlagKind.ZERO_WIDTH]
    assert zero_width, "expected a ZERO_WIDTH flag for a ZWJ with no Arabic neighbour"


def test_bidi_override_inside_arabic_text_still_flags() -> None:
    """RLO/LRO are never exempt, even embedded in genuine Arabic-script text."""
    flags = PdfHiddenContentDetector().detect(pb.bidi_override_in_arabic_pdf())

    bidi = [f for f in flags if f.kind is ScanFlagKind.BIDI_CONTROL]
    assert bidi, "expected a BIDI_CONTROL flag for an RLO even inside Arabic text"
