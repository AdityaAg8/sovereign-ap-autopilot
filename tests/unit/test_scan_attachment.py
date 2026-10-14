"""Unit test for the embedded-attachment technique (A10): a file in the document's
`/Names/EmbeddedFiles` tree is flagged PDF_ATTACHMENT."""

from __future__ import annotations

from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind
from tests.fixtures.scan import pdf_builder as pb


def test_embedded_file_is_flagged_attachment() -> None:
    flags = PdfHiddenContentDetector().detect(pb.attachment_pdf(filename="remit_to.txt"))

    attachments = [f for f in flags if f.kind is ScanFlagKind.PDF_ATTACHMENT]
    assert attachments, "expected at least one PDF_ATTACHMENT flag"
    assert any("remit_to.txt" in f.evidence.text for f in attachments)


def test_clean_invoice_never_flags_attachment() -> None:
    flags = PdfHiddenContentDetector().detect(pb.clean_invoice_pdf())

    assert not [f for f in flags if f.kind is ScanFlagKind.PDF_ATTACHMENT]
