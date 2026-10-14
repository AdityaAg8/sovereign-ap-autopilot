"""Interfaces for PINT AE: building UBL 2.1 invoices and credit notes from a validated Invoice
and schematron validation against the pinned rule set."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol

from pydantic import ConfigDict

from apx.common.invoice import Invoice
from apx.common.types import Frozen, InvoiceId, Sha256Hex


class DocumentTypeCode(StrEnum):
    INVOICE = "380"
    CREDIT_NOTE = "381"
    OUT_OF_SCOPE_INVOICE = "480"
    """PINT AE 'Out of scope of VAT' invoice: required when every line is Exempt (E) and/or Not
    subject to VAT (O), since `ibr-151-ae` rejects such a document as a 380 Commercial invoice."""
    OUT_OF_SCOPE_CREDIT_NOTE = "81"
    """The credit-note counterpart of `OUT_OF_SCOPE_INVOICE` (PINT AE `ibr-122-ae`/`ibr-157-ae`)."""

    @property
    def is_invoice(self) -> bool:
        """True for the document types serialised as a UBL `Invoice` (380, 480); false for the
        `CreditNote` ones (381, 81)."""
        return self in (DocumentTypeCode.INVOICE, DocumentTypeCode.OUT_OF_SCOPE_INVOICE)


class EInvoiceDocument(Frozen):
    # `xml` crosses the Temporal activity boundary as JSON: base64 keeps any byte sequence
    # (non-UTF-8 encodings, a BOM) lossless, where pydantic's default UTF-8 mode raises.
    model_config = ConfigDict(
        frozen=True, extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

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
    def build(
        self,
        invoice: Invoice,
        type_code: DocumentTypeCode,
        *,
        verified_payment_iban: str | None,
    ) -> EInvoiceDocument:
        """Build PINT AE UBL 2.1 XML paying `verified_payment_iban` (the vendor master's
        admin-verified account, never the invoice's own IBAN). Raises EInvoiceError on unmappable
        data or, for an Invoice, a missing verified IBAN."""
        ...


class SchematronValidator(Protocol):
    def validate(self, xml: bytes) -> SchematronResult:
        """Run the compiled PINT AE schematron (saxonche + SchXslt). Raises EInvoiceError if the
        validator cannot run; rule failures are data."""
        ...
