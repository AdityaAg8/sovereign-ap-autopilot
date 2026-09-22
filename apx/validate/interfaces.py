"""Interfaces for validation: VAT math, TRN, IBAN, mandatory fields and the invoice/PO/GRN
3-way match. Violations are data returned to the policy engine (PRD §3)."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Protocol

from apx.common.invoice import Invoice

# PoLine/PurchaseOrder live in apx.common (fraud needs them too and must stay common-only); the
# `as`-aliased re-export keeps `apx.validate.interfaces.PurchaseOrder` importable for compatibility.
from apx.common.purchasing import PoLine as PoLine
from apx.common.purchasing import PurchaseOrder as PurchaseOrder
from apx.common.types import Frozen


class ViolationCode(StrEnum):
    MISSING_MANDATORY = "missing_mandatory"
    VAT_MATH = "vat_math"
    TOTALS_MISMATCH = "totals_mismatch"
    TRN_INVALID = "trn_invalid"
    IBAN_INVALID = "iban_invalid"
    DATE_INVALID = "date_invalid"
    CURRENCY_INVALID = "currency_invalid"
    PO_NOT_FOUND = "po_not_found"
    THREE_WAY_MISMATCH = "three_way_mismatch"


class Violation(Frozen):
    code: ViolationCode
    field: str | None
    message: str
    severity: Literal["error", "warning"]


class GrnLine(Frozen):
    po_line_ref: str
    quantity_received: Decimal


class GoodsReceipt(Frozen):
    number: str
    po_number: str
    lines: tuple[GrnLine, ...]


class MatchTolerance(Frozen):
    quantity: Decimal = Decimal("0")
    price_pct: Decimal = Decimal("0.01")


class ValidationReport(Frozen):
    violations: tuple[Violation, ...]

    @property
    def passed(self) -> bool:
        """True when there are no error-severity violations."""
        raise NotImplementedError


class PurchasingLookup(Protocol):
    def purchase_order(self, number: str) -> PurchaseOrder | None: ...
    def receipts_for(self, po_number: str) -> Sequence[GoodsReceipt]: ...


class Validator(Protocol):
    def validate(self, invoice: Invoice) -> ValidationReport: ...


def is_valid_trn(trn: str) -> bool:
    """UAE TRN: 15 digits (format check only)."""
    raise NotImplementedError


def is_valid_iban(iban: str) -> bool:
    """ISO 13616 IBAN with mod-97 checksum; AE IBANs are 23 characters."""
    raise NotImplementedError


def check_vat(invoice: Invoice) -> list[Violation]:
    """Line and total VAT math per category (standard 5%, zero-rated, exempt, reverse charge)."""
    raise NotImplementedError


def three_way_match(
    invoice: Invoice,
    po: PurchaseOrder | None,
    receipts: Sequence[GoodsReceipt],
    tolerance: MatchTolerance,
) -> list[Violation]:
    """Compare invoice lines to PO lines and received quantities within tolerance."""
    raise NotImplementedError
