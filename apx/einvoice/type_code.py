"""The PINT AE document type code (BT-3) for an invoice: 380/381 from what the document says it
is, turned into 480/81 when every line is Exempt or Out of scope (ibr-151-ae). Shared by the
workflow's e-invoice activity and the factory's gold XML (ADR-0012)."""

from __future__ import annotations

from collections.abc import Iterable

from apx.common.errors import EInvoiceError
from apx.common.invoice import InvoiceKind, VatCategory
from apx.einvoice.interfaces import DocumentTypeCode

_OUT_OF_SCOPE: frozenset[VatCategory] = frozenset({VatCategory.EXEMPT, VatCategory.OUT_OF_SCOPE})


def document_type_code(
    kind: InvoiceKind | None, categories: Iterable[VatCategory | None]
) -> DocumentTypeCode:
    """invoice -> 380 (480 when every category is E or O); credit_note -> 381 (81 likewise).
    A None kind, a None category or no lines at all is an `EInvoiceError`: nothing is guessed."""
    cats = tuple(categories)
    if kind is None:
        raise EInvoiceError("document type is unknown; review must set invoice or credit note")
    if not cats:
        raise EInvoiceError("cannot derive a document type code without invoice lines")
    if any(c is None for c in cats):
        raise EInvoiceError("a line VAT category is unknown; review must set it")
    out_of_scope = all(c in _OUT_OF_SCOPE for c in cats)
    if kind is InvoiceKind.INVOICE:
        return DocumentTypeCode.OUT_OF_SCOPE_INVOICE if out_of_scope else DocumentTypeCode.INVOICE
    return (
        DocumentTypeCode.OUT_OF_SCOPE_CREDIT_NOTE if out_of_scope else DocumentTypeCode.CREDIT_NOTE
    )
