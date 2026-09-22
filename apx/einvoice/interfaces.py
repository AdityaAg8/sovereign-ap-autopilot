"""Interfaces for PINT AE: building UBL 2.1 invoices and credit notes from a validated Invoice
and schematron validation against the pinned rule set (PRD §3, §5 schematron gate)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol

from apx.common.invoice import Invoice
from apx.common.types import Frozen, InvoiceId, Sha256Hex


class DocumentTypeCode(StrEnum):
    INVOICE = "380"
    CREDIT_NOTE = "381"


class EInvoiceDocument(Frozen):
    invoice_id: InvoiceId
    type_code: DocumentTypeCode
    xml: bytes
    xml_sha256: Sha256Hex
    spec_version: str
    """Pinned PINT AE version the XML targets."""


class SchematronFailure(Frozen):
    rule_id: str
    flag: Literal["fatal", "warning"]
    location: str
    message: str


class SchematronResult(Frozen):
    passed: bool
    failures: tuple[SchematronFailure, ...]
    rules_version: str


class UblBuilder(Protocol):
    def build(self, invoice: Invoice, type_code: DocumentTypeCode) -> EInvoiceDocument:
        """Build PINT AE UBL 2.1 XML. Raises EInvoiceError on unmappable data."""
        ...


class SchematronValidator(Protocol):
    def validate(self, xml: bytes) -> SchematronResult:
        """Run the compiled PINT AE schematron (saxonche + SchXslt). Raises EInvoiceError if the
        validator cannot run; rule failures are data."""
        ...
