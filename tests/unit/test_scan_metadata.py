"""Unit tests for the metadata-injection technique (A10): a long free-text Info dict
value, a zero-width-carrying XMP stream, and an annotation are all flagged PDF_METADATA; the
standard Producer/CreationDate/... boilerplate every PDF writer sets is never flagged.
Also covers two hardening cases: an undecodable XMP stream must fail closed with
`ScanError` rather than being treated as empty/clean, and a baseline Info dict key (Producer,
Creator, ...) is exempt from the length heuristic but never from the suspicious-codepoint check."""

from __future__ import annotations

import pytest

from apx.common.errors import ScanError
from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb


def test_long_keywords_value_is_flagged_metadata() -> None:
    flags = PdfHiddenContentDetector().detect(pb.metadata_injection_pdf())

    metadata = [f for f in flags if f.kind is ScanFlagKind.PDF_METADATA]
    assert metadata, "expected at least one PDF_METADATA flag"
    assert any("Keywords" in f.evidence.text for f in metadata)


def test_xmp_stream_with_hidden_characters_is_flagged_metadata() -> None:
    flags = PdfHiddenContentDetector().detect(pb.xmp_metadata_pdf())

    metadata = [f for f in flags if f.kind is ScanFlagKind.PDF_METADATA]
    assert metadata, "expected at least one PDF_METADATA flag"
    assert any("XMP" in f.evidence.text for f in metadata)


def test_annotation_contents_are_flagged_metadata() -> None:
    flags = PdfHiddenContentDetector().detect(pb.annotation_pdf())

    metadata = [f for f in flags if f.kind is ScanFlagKind.PDF_METADATA]
    assert metadata, "expected at least one PDF_METADATA flag"
    assert any(f.page == 0 for f in metadata)
    assert any("annotation" in f.evidence.text for f in metadata)


def test_clean_invoice_never_flags_metadata() -> None:
    """reportlab's own Producer/Creator/CreationDate/ModDate/Trapped/Title/Subject boilerplate
    must never trigger PDF_METADATA on its own."""
    flags = PdfHiddenContentDetector().detect(pb.clean_invoice_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.PDF_METADATA]


def test_undecodable_xmp_stream_fails_closed_with_scan_error() -> None:
    """F1: an XMP stream pdfminer cannot decode must raise `ScanError`, with the underlying
    decode failure chained as its cause, never fall back to an empty/clean scan (SEC-02)."""
    with pytest.raises(ScanError) as excinfo:
        PdfHiddenContentDetector().detect(pb.corrupted_xmp_pdf())

    assert excinfo.value.__cause__ is not None


def test_zero_width_in_baseline_creator_field_is_flagged_metadata() -> None:
    """F2: `Creator` is a baseline key exempt from the length heuristic, but a zero-width
    character stitched into it must still flag -- no legitimate writer does this."""
    flags = PdfHiddenContentDetector().detect(pb.baseline_metadata_key_zero_width_pdf())

    metadata = [f for f in flags if f.kind is ScanFlagKind.PDF_METADATA]
    assert metadata, "expected the zero-width Creator value to be flagged"
    assert any("Info/Creator" in f.evidence.text for f in metadata)


def test_bidi_override_in_baseline_producer_field_is_flagged_metadata() -> None:
    """F2: same as above for a bidi override stitched into `Producer`."""
    flags = PdfHiddenContentDetector().detect(pb.baseline_metadata_key_bidi_pdf())

    metadata = [f for f in flags if f.kind is ScanFlagKind.PDF_METADATA]
    assert metadata, "expected the bidi-override Producer value to be flagged"
    assert any("Info/Producer" in f.evidence.text for f in metadata)


def test_normal_producer_and_creator_strings_are_not_flagged() -> None:
    """F2 regression: ordinary real-world baseline values ("Skia/PDF m120", "Microsoft Word")
    carry neither a suspicious codepoint nor the length heuristic's exemption removed, so they
    must never flag."""
    flags = PdfHiddenContentDetector().detect(pb.normal_producer_and_creator_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.PDF_METADATA]
