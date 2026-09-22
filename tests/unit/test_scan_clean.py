"""Baseline and fail-closed tests for `PdfHiddenContentDetector` (PRD §6.3 SEC-02, M6.2): a
plain multi-page invoice never flags, and a document that cannot be parsed raises `ScanError`
rather than silently reporting zero flags (a failed scan must never look clean)."""

from __future__ import annotations

import pytest

from apx.common.errors import ScanError
from apx.scan.detector import PdfHiddenContentDetector
from tests.fixtures.scan import pdf_builder as pb


def test_clean_multi_page_invoice_has_zero_flags() -> None:
    flags = PdfHiddenContentDetector().detect(pb.clean_invoice_pdf(pages=3))

    assert flags == []


def test_unparseable_bytes_raise_scan_error_not_an_empty_list() -> None:
    with pytest.raises(ScanError):
        PdfHiddenContentDetector().detect(b"this is not a pdf")
