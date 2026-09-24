"""Canonical, normalised invoice model shared by validate, fraud, policy and einvoice. Built
from accepted or reviewer-corrected fields, never directly from model output (PRD §3)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from apx.common.types import Frozen, InvoiceId, Language, Money, Sha256Hex, VendorId


class VatCategory(StrEnum):
    """UAE VAT categories (UNCL5305 codes used by PINT AE)."""

    STANDARD = "S"
    ZERO_RATED = "Z"
    EXEMPT = "E"
    REVERSE_CHARGE = "AE"
    OUT_OF_SCOPE = "O"


class InvoiceKind(StrEnum):
    """What the document says it is (BT-3 as titled, ADR-0012). Amounts are always stored as
    printed, never negated: a credit note is identified by this kind, not by a negative total."""

    INVOICE = "invoice"
    CREDIT_NOTE = "credit_note"


class Party(Frozen):
    """Supplier or buyer."""

    name: str
    trn: str | None
    address: str | None
    country: str = "AE"


class InvoiceLine(Frozen):
    """One invoice line."""

    description: str
    quantity: Decimal
    unit_price: Money
    net_amount: Money
    vat_category: VatCategory | None
    """None when the extractors did not agree on it (ADR-0012 §1.4); the validator flags it."""
    vat_rate: Decimal | None
    """None when the rate is missing or unreadable; the validator flags it."""
    po_line_ref: str | None = None


class Invoice(Frozen):
    """Normalised invoice. `iban` is what the document claims; payment uses only the vendor
    master's verified account (P1)."""

    id: InvoiceId
    document_sha256: Sha256Hex
    language: Language
    number: str
    issue_date: date
    due_date: date | None
    supplier: Party
    buyer: Party
    vendor_id: VendorId | None
    po_number: str | None
    lines: tuple[InvoiceLine, ...]
    subtotal: Money
    vat_total: Money
    total: Money
    iban: str | None
    kind: InvoiceKind | None = None
    """None when the extractors did not agree on the type (ADR-0012 §1.4); the validator flags
    it and the e-invoice type code refuses it."""
