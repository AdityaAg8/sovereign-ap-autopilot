"""Upload hardening shared by every ingest path (watcher and API upload): magic-byte media-type
sniffing (never the client's declared content type or a filename extension), the size cap and
the PDF page/encryption checks (PRD §10 M5.3 limits, M5.5; docs/security.md B1). The only home
of this logic: the API upload route and the inbox watcher both reach it through `Ingestor`.
"""

from __future__ import annotations

from typing import Literal

import pypdfium2 as pdfium

from apx.common.errors import IngestError

MediaType = Literal["application/pdf", "image/png", "image/jpeg"]
"""Media types this validator can detect and accept. `apx.ingest.interfaces.IngestedDocument`
also allows `image/tiff` for a future parser; sniffing for it is not implemented yet, so a TIFF
upload is rejected as unsupported, consistent with the M5.3 limits this module started from."""

_PDF_MAGIC = b"%PDF-"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


class UnsupportedMediaTypeError(IngestError):
    """The document's magic bytes do not match PDF, PNG or JPEG."""


class PayloadTooLargeError(IngestError):
    """The document exceeds the configured maximum byte size."""


class TooManyPagesError(IngestError):
    """The PDF has more pages than the configured maximum."""


class EncryptedDocumentError(IngestError):
    """The PDF is password-protected or otherwise could not be opened for validation."""


def detect_media_type(data: bytes) -> MediaType | None:
    """Sniffs the media type from magic bytes only -- never from a filename extension or a
    client-supplied `Content-Type` header, both of which are attacker-controlled (§6.2 I1
    spirit: provenance comes from the bytes, not from a claim)."""
    if data.startswith(_PDF_MAGIC):
        return "application/pdf"
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    return None


def check_size(data: bytes, max_bytes: int) -> None:
    """Raises `PayloadTooLargeError` if `data` exceeds `max_bytes`."""
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"upload is {len(data)} bytes; at most {max_bytes} are allowed")


def check_pdf_pages(data: bytes, max_pages: int) -> None:
    """Raises `EncryptedDocumentError` if the PDF cannot be opened without a password (or is
    otherwise malformed), or `TooManyPagesError` if it has more than `max_pages` pages."""
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise EncryptedDocumentError(
            f"PDF could not be opened (encrypted or malformed): {exc}"
        ) from exc
    try:
        page_count = len(pdf)
    finally:
        pdf.close()
    if page_count > max_pages:
        raise TooManyPagesError(f"PDF has {page_count} pages; at most {max_pages} are allowed")


def validate(data: bytes, *, max_bytes: int, max_pages: int) -> MediaType:
    """Runs every hardening rule against `data` and returns its detected media type. Raises the
    first typed `IngestError` subclass hit: size, then media type, then (for PDFs) page count
    and encryption."""
    check_size(data, max_bytes)
    media_type = detect_media_type(data)
    if media_type is None:
        raise UnsupportedMediaTypeError(
            "unsupported file type: only PDF, PNG and JPEG are accepted"
        )
    if media_type == "application/pdf":
        check_pdf_pages(data, max_pages)
    return media_type
