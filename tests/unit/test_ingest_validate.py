"""Unit tests for `apx.ingest.validate`, the upload hardening shared by the watcher and the API
upload path (PRD §10 M5.3 limits, M5.5): magic-byte sniffing ignores the claimed content type,
the size and page caps are enforced, and an encrypted or malformed PDF is rejected. No database
or network is needed.
"""

from __future__ import annotations

import io

import pypdfium2 as pdfium
import pytest
from hypothesis import given
from hypothesis import strategies as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.pdfgen import canvas

from apx.ingest.validate import (
    EncryptedDocumentError,
    PayloadTooLargeError,
    TooManyPagesError,
    UnsupportedMediaTypeError,
    check_pdf_pages,
    check_size,
    detect_media_type,
    validate,
)

PDF_MAGIC = b"%PDF-"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def _make_pdf(pages: int = 1, *, encrypted: bool = False) -> bytes:
    buf = io.BytesIO()
    encrypt = StandardEncryption("userpw", ownerPassword="ownerpw") if encrypted else None
    c = canvas.Canvas(buf, pagesize=letter, encrypt=encrypt)
    for _ in range(pages):
        c.drawString(100, 700, "hello")
        c.showPage()
    c.save()
    return buf.getvalue()


# ------------------------------------------------------------------------------------------
# detect_media_type
# ------------------------------------------------------------------------------------------


def test_detect_media_type_uses_magic_bytes_not_extension() -> None:
    assert detect_media_type(_make_pdf()) == "application/pdf"
    assert detect_media_type(PNG_MAGIC + b"rest") == "image/png"
    assert detect_media_type(JPEG_MAGIC + b"rest") == "image/jpeg"
    assert detect_media_type(b"not a real document") is None


def test_detect_media_type_rejects_tiff_not_yet_supported() -> None:
    # apx.ingest.interfaces.IngestedDocument reserves "image/tiff" for a future parser; no
    # magic-byte sniffing is implemented for it yet, so it must be rejected as unsupported
    # rather than silently mis-detected as something else (PRD §8: no silent fallbacks).
    tiff_magic = b"II*\x00"
    assert detect_media_type(tiff_magic + b"rest") is None


@given(st.binary(max_size=64))
def test_detect_media_type_never_raises_on_arbitrary_bytes(data: bytes) -> None:
    result = detect_media_type(data)
    assert result is None or result in {"application/pdf", "image/png", "image/jpeg"}


@given(
    st.binary(min_size=1, max_size=32).filter(lambda b: not b.startswith((b"%", b"\x89", b"\xff")))
)
def test_detect_media_type_returns_none_for_bytes_without_a_known_magic(data: bytes) -> None:
    assert detect_media_type(data) is None


# ------------------------------------------------------------------------------------------
# check_size
# ------------------------------------------------------------------------------------------


def test_check_size_accepts_exactly_at_the_limit() -> None:
    check_size(b"x" * 10, max_bytes=10)


def test_check_size_rejects_one_byte_over_the_limit() -> None:
    with pytest.raises(PayloadTooLargeError):
        check_size(b"x" * 11, max_bytes=10)


@given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
def test_check_size_raises_iff_over_the_limit(size: int, max_bytes: int) -> None:
    data = b"x" * size
    if size > max_bytes:
        with pytest.raises(PayloadTooLargeError):
            check_size(data, max_bytes)
    else:
        check_size(data, max_bytes)  # must not raise


# ------------------------------------------------------------------------------------------
# check_pdf_pages / validate
# ------------------------------------------------------------------------------------------


def test_rejects_pdf_with_too_many_pages() -> None:
    with pytest.raises(TooManyPagesError):
        check_pdf_pages(_make_pdf(pages=3), max_pages=2)


def test_accepts_pdf_at_exactly_the_page_limit() -> None:
    check_pdf_pages(_make_pdf(pages=2), max_pages=2)  # must not raise


def test_rejects_encrypted_pdf() -> None:
    with pytest.raises(EncryptedDocumentError):
        check_pdf_pages(_make_pdf(encrypted=True), max_pages=30)


def test_rejects_malformed_pdf_bytes() -> None:
    with pytest.raises((EncryptedDocumentError, pdfium.PdfiumError)):
        check_pdf_pages(b"%PDF-1.7\nnot actually a pdf body", max_pages=30)


def test_validate_accepts_a_small_pdf_and_returns_its_media_type() -> None:
    assert validate(_make_pdf(), max_bytes=20 * 1024 * 1024, max_pages=30) == "application/pdf"


def test_validate_rejects_a_file_named_pdf_whose_bytes_are_not_a_document() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        validate(b"<html>not a pdf</html>", max_bytes=20 * 1024 * 1024, max_pages=30)


def test_validate_checks_size_before_media_type() -> None:
    with pytest.raises(PayloadTooLargeError):
        validate(_make_pdf(), max_bytes=1, max_pages=30)


def test_validate_checks_pdf_pages_only_for_pdfs() -> None:
    # A PNG well under the page-relevant code path never opens pdfium at all.
    assert validate(PNG_MAGIC + b"rest", max_bytes=20 * 1024 * 1024, max_pages=1) == "image/png"
