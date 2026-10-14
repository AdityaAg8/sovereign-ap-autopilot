"""Upload hardening shared by every ingest path (watcher and API upload): magic-byte media-type
sniffing (never the client's declared content type or a filename extension), the size cap, the PDF
page/encryption checks and the raster caps (docs/security.md B1). The only home of this logic: the
API upload route and the inbox watcher both reach it through `Ingestor`."""

from __future__ import annotations

from typing import Literal

import pypdfium2 as pdfium

from apx.common.errors import IngestError
from apx.common.raster import (
    RasterTooLargeError,
    check_image_bytes,
    check_pdf_document,
)

__all__ = [
    "EncryptedDocumentError",
    "MediaType",
    "PayloadTooLargeError",
    "RasterTooLargeError",
    "TooManyPagesError",
    "UnsupportedMediaTypeError",
    "check_image",
    "check_pdf_pages",
    "check_size",
    "detect_media_type",
    "validate",
]

MediaType = Literal["application/pdf", "image/png", "image/jpeg"]
"""Media types this validator can detect and accept. `apx.ingest.interfaces.IngestedDocument`
also allows `image/tiff` for a future parser; sniffing for it is not implemented yet, so a TIFF
upload is rejected as unsupported, consistent with the upload limits this module started from."""

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
    client-supplied `Content-Type` header, both of which are attacker-controlled
    (I1 spirit: provenance comes from the bytes, not from a claim)."""
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
    otherwise malformed), `TooManyPagesError` if it has more than `max_pages` pages, or
    `RasterTooLargeError` if any page is too large in points, in pixels at the highest render
    DPI, or in its embedded images (`apx.common.raster`). Nothing is rendered here."""
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise EncryptedDocumentError(
            f"PDF could not be opened (encrypted or malformed): {exc}"
        ) from exc
    try:
        page_count = len(pdf)
        if page_count > max_pages:
            raise TooManyPagesError(f"PDF has {page_count} pages; at most {max_pages} are allowed")
        check_pdf_document(pdf)
    finally:
        pdf.close()


def check_image(data: bytes) -> None:
    """Raises `RasterTooLargeError` if a PNG/JPEG declares too many pixels, or
    `UnsupportedMediaTypeError` if its header cannot be read (size unknown, never accepted)."""
    try:
        check_image_bytes(data)
    except RasterTooLargeError:
        raise
    except IngestError as exc:
        raise UnsupportedMediaTypeError(f"unreadable image: {exc}") from exc


def validate(data: bytes, *, max_bytes: int, max_pages: int) -> MediaType:
    """Runs every hardening rule against `data` and returns its detected media type. Raises the
    first typed `IngestError` subclass hit: size, then media type, then (for PDFs) encryption,
    page count and page size, or (for images) header dimensions."""
    check_size(data, max_bytes)
    media_type = detect_media_type(data)
    if media_type is None:
        raise UnsupportedMediaTypeError(
            "unsupported file type: only PDF, PNG and JPEG are accepted"
        )
    if media_type == "application/pdf":
        check_pdf_pages(data, max_pages)
    else:
        check_image(data)
    return media_type
