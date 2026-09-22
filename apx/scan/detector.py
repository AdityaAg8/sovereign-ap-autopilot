"""Concrete hidden-content detector: opens a PDF once with pdfplumber and once with pypdfium2
and runs every M6.2 technique (visibility, control characters, active content) per page and
per document (PRD §3, §6.3, M6.2). Never returns [] on failure -- raises `ScanError` instead,
so a scan that could not run is never mistaken for a clean document (SEC-02)."""

from __future__ import annotations

import io

import pdfplumber
import pypdfium2 as pdfium

from apx.common.errors import ScanError
from apx.scan.active_content import scan_document_active_content
from apx.scan.controls import scan_page_controls
from apx.scan.interfaces import ScanFlag
from apx.scan.render import RENDER_DPI, render_page
from apx.scan.visibility import scan_page_visibility


class PdfHiddenContentDetector:
    """`HiddenContentDetector` (see `apx.scan.interfaces`) over pdfplumber's parsed content and
    a pypdfium2 render of the same pages."""

    def __init__(self, dpi: int = RENDER_DPI) -> None:
        self._dpi = dpi

    def detect(self, pdf_bytes: bytes) -> list[ScanFlag]:
        """Run every M6.2 technique. Raises `ScanError` if the PDF cannot be parsed or
        rendered; a failed scan is never reported as an empty (clean) flag list."""
        try:
            return self._detect(pdf_bytes)
        except ScanError:
            raise
        except Exception as exc:
            raise ScanError(f"hidden-content scan failed: {exc}") from exc

    def _detect(self, pdf_bytes: bytes) -> list[ScanFlag]:
        flags: list[ScanFlag] = []
        with (
            pdfplumber.open(io.BytesIO(pdf_bytes)) as pl_doc,
            pdfium.PdfDocument(pdf_bytes) as fpdf_doc,
        ):
            if len(pl_doc.pages) != len(fpdf_doc):
                raise ScanError(
                    f"page count mismatch: pdfplumber={len(pl_doc.pages)} pypdfium2={len(fpdf_doc)}"
                )
            for page_index, page in enumerate(pl_doc.pages):
                rendered = render_page(fpdf_doc[page_index], dpi=self._dpi)
                flags.extend(scan_page_visibility(page, page_index, rendered, self._dpi))
                flags.extend(scan_page_controls(page, page_index))
            flags.extend(scan_document_active_content(pl_doc))
        return flags
