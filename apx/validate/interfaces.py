"""Interfaces for validation: VAT math, TRN, IBAN, mandatory fields and the invoice/PO/GRN
3-way match. Violations are data returned to the policy engine."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Final, Literal, Protocol

from apx.common.invoice import Invoice, InvoiceKind, VatCategory

# PoLine/PurchaseOrder live in apx.common (fraud needs them too and must stay common-only); the
# `as`-aliased re-export keeps `apx.validate.interfaces.PurchaseOrder` importable for compatibility.
from apx.common.purchasing import PoLine as PoLine
from apx.common.purchasing import PurchaseOrder as PurchaseOrder
from apx.common.types import Frozen, Money
from apx.validate import ids as _ids
from apx.validate.address import resolve_emirate


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
    # --- M4.2: 3-way match (apx.validate.match), additive to the M4.1 codes above -----------
    GRN_NOT_FOUND = "grn_not_found"
    UNKNOWN_LINE = "unknown_line"
    DUPLICATE_LINE = "duplicate_line"
    OVER_BILLING = "over_billing"
    QUANTITY_OVER_RECEIVED = "quantity_over_received"
    PRICE_OVER_TOLERANCE = "price_over_tolerance"
    CURRENCY_MISMATCH = "currency_mismatch"
    # --- ADR-0012: classified values the extractors did not agree on, never defaulted ---------
    INVOICE_TYPE_MISSING = "invoice_type_missing"
    VAT_CATEGORY_MISSING = "vat_category_missing"
    VAT_RATE_MISSING = "vat_rate_missing"
    VAT_CATEGORY_RATE_MISMATCH = "vat_category_rate_mismatch"
    # --- ADR-0012 amendment 2: the buyer is our own organisation (WORKER_BUYER_TRN) ----------
    BUYER_TRN_MISMATCH = "buyer_trn_mismatch"


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
        return not any(v.severity == "error" for v in self.violations)


class PurchasingLookup(Protocol):
    def purchase_order(self, number: str) -> PurchaseOrder | None: ...
    def receipts_for(self, po_number: str) -> Sequence[GoodsReceipt]: ...


class Validator(Protocol):
    def validate(self, invoice: Invoice) -> ValidationReport: ...


def is_valid_trn(trn: str) -> bool:
    """UAE TRN: 15 digits (format check only). Delegates to `apx.validate.ids`."""
    return _ids.is_valid_trn(trn)


def is_valid_iban(iban: str) -> bool:
    """ISO 13616 IBAN with mod-97 checksum; AE IBANs are 23 characters. Delegates to
    `apx.validate.ids`."""
    return _ids.is_valid_iban(iban)


STANDARD_VAT_RATE: Final[Decimal] = Decimal("5")
"""The UAE standard VAT rate (5%), used to check `VatCategory.STANDARD` lines."""

NO_CURRENCY: Final[str] = "XXX"
"""ISO 4217 "no currency": what the invoice builder records when none was extracted (ADR-0012)."""

SUPPLIER_ADDRESS_VIOLATION_FIELD: Final[str] = "supplier.address"
"""`Violation.field` of the supplier-address MISSING_MANDATORY error (ADR-0012 amendment 4)."""

_ZERO: Final[Decimal] = Decimal("0")
_CENTS: Final[Decimal] = Decimal("0.01")

_NO_VAT_CATEGORIES: Final[frozenset[VatCategory]] = frozenset(
    {VatCategory.ZERO_RATED, VatCategory.EXEMPT, VatCategory.OUT_OF_SCOPE}
)
"""Categories whose VAT rate must be 0 (S needs 5; AE may state 0 or 5, see below).
`VatCategory.OUT_OF_SCOPE` also covers UAE designated (free) zone supplies: PINT AE's VAT
category codelist has no separate code for them, so Executive-Regulation-conditioned free zone
supplies are declared "O" like any other not-subject-to-VAT supply."""

_REVERSE_CHARGE_RATES: Final[frozenset[Decimal]] = frozenset({_ZERO, STANDARD_VAT_RATE})
"""PINT AE 1.0.4 (official example `Doc-level-charge-AE-category.xml`): an AE line states 0% or
the 5% the buyer self-accounts for; the VAT this invoice collects on it is 0 either way
(`ibr-162-ae`, `aligned-ibrp-ae-09-ae`)."""


def vat_rate_fits_category(category: VatCategory, rate: Decimal) -> bool:
    """ADR-0012's category/rate rule (S needs 5; Z, E and O need 0; AE states 0 or 5), the one
    `check_vat` reports as `VAT_CATEGORY_RATE_MISMATCH`. It checks a rate, never supplies one."""
    if category is VatCategory.STANDARD:
        return rate == STANDARD_VAT_RATE
    if category is VatCategory.REVERSE_CHARGE:
        return rate in _REVERSE_CHARGE_RATES
    return rate == _ZERO


def _round_aed(amount: Decimal) -> Decimal:
    """Round a Decimal AED amount to 2 decimal places (fils) with commercial half-up rounding."""
    return amount.quantize(_CENTS, rounding=ROUND_HALF_UP)


def check_vat(invoice: Invoice) -> list[Violation]:
    """Line and total VAT math per category (standard 5%, zero-rated, exempt, reverse charge,
    designated free zone / out-of-scope) with AED rounding.

    Reverse-charge (`VatCategory.REVERSE_CHARGE`) lines state 0% or 5% and collect no VAT on
    this invoice (PINT AE 1.0.4 `ibr-162-ae`, `aligned-ibrp-ae-09-ae`). A line whose category or
    rate is None gets a `VAT_CATEGORY_MISSING`/`VAT_RATE_MISSING` error and is skipped by the VAT
    math, and the VAT total is then not reconciled (it cannot be computed)."""
    violations: list[Violation] = []
    expected_vat_total = _ZERO
    expected_subtotal = _ZERO
    vat_unknown = False

    for i, line in enumerate(invoice.lines):
        expected_line_net = _round_aed(line.quantity * line.unit_price.amount)
        if line.net_amount.amount != expected_line_net:
            violations.append(
                Violation(
                    code=ViolationCode.TOTALS_MISMATCH,
                    field=f"lines[{i}].net_amount",
                    message=(
                        f"line {i}: net amount {line.net_amount.amount} does not equal "
                        f"quantity * unit price ({expected_line_net})"
                    ),
                    severity="error",
                )
            )

        category, rate = line.vat_category, line.vat_rate
        if category is None:
            violations.append(
                Violation(
                    code=ViolationCode.VAT_CATEGORY_MISSING,
                    field=f"lines[{i}].vat_category",
                    message=f"line {i}: VAT category is missing or the extractors disagree",
                    severity="error",
                )
            )
        if rate is None:
            violations.append(
                Violation(
                    code=ViolationCode.VAT_RATE_MISSING,
                    field=f"lines[{i}].vat_rate",
                    message=f"line {i}: VAT rate is missing or unreadable",
                    severity="error",
                )
            )
        if category is None or rate is None:
            # The line already carries an error; its VAT cannot be computed (ADR-0012).
            expected_subtotal += line.net_amount.amount
            vat_unknown = True
            continue
        if rate < _ZERO:
            violations.append(
                Violation(
                    code=ViolationCode.VAT_MATH,
                    field=f"lines[{i}].vat_rate",
                    message=f"line {i}: VAT rate {rate} is negative",
                    severity="error",
                )
            )
        elif category is VatCategory.STANDARD and rate != STANDARD_VAT_RATE:
            violations.append(
                Violation(
                    code=ViolationCode.VAT_CATEGORY_RATE_MISMATCH,
                    field=f"lines[{i}].vat_rate",
                    message=(
                        f"line {i}: standard-rated line must use {STANDARD_VAT_RATE}% VAT, "
                        f"got {rate}%"
                    ),
                    severity="error",
                )
            )
        elif category is VatCategory.REVERSE_CHARGE and rate not in _REVERSE_CHARGE_RATES:
            violations.append(
                Violation(
                    code=ViolationCode.VAT_CATEGORY_RATE_MISMATCH,
                    field=f"lines[{i}].vat_rate",
                    message=f"line {i}: reverse-charge line must state 0% or 5%, got {rate}%",
                    severity="error",
                )
            )
        elif category in _NO_VAT_CATEGORIES and rate != _ZERO:
            violations.append(
                Violation(
                    code=ViolationCode.VAT_CATEGORY_RATE_MISMATCH,
                    field=f"lines[{i}].vat_rate",
                    message=(
                        f"line {i}: {category.name.lower()} line must use 0% VAT, got {rate}%"
                    ),
                    severity="error",
                )
            )

        line_vat = (
            _round_aed(line.net_amount.amount * rate / Decimal(100))
            if category is VatCategory.STANDARD
            else _ZERO
        )
        expected_vat_total += line_vat
        expected_subtotal += line.net_amount.amount

    if invoice.subtotal.amount != expected_subtotal:
        violations.append(
            Violation(
                code=ViolationCode.TOTALS_MISMATCH,
                field="subtotal",
                message=(
                    f"subtotal {invoice.subtotal.amount} does not equal the sum of line net "
                    f"amounts ({expected_subtotal})"
                ),
                severity="error",
            )
        )
    if not vat_unknown and invoice.vat_total.amount != expected_vat_total:
        violations.append(
            Violation(
                code=ViolationCode.TOTALS_MISMATCH,
                field="vat_total",
                message=(
                    f"VAT total {invoice.vat_total.amount} does not equal the computed VAT "
                    f"({expected_vat_total})"
                ),
                severity="error",
            )
        )
    expected_total = _round_aed(invoice.subtotal.amount + invoice.vat_total.amount)
    if invoice.total.amount != expected_total:
        violations.append(
            Violation(
                code=ViolationCode.TOTALS_MISMATCH,
                field="total",
                message=(
                    f"total {invoice.total.amount} does not equal subtotal + VAT ({expected_total})"
                ),
                severity="error",
            )
        )
    return violations


def check_trn(invoice: Invoice) -> list[Violation]:
    """Format-validate the supplier and buyer TRNs when present (UAE TRN = 15 digits; there is
    no published check-digit scheme). Presence is `check_mandatory_fields`'s job, not this
    one's."""
    violations: list[Violation] = []
    if invoice.supplier.trn is not None and not _ids.is_valid_trn(invoice.supplier.trn):
        violations.append(
            Violation(
                code=ViolationCode.TRN_INVALID,
                field="supplier.trn",
                message=(
                    f"supplier TRN {invoice.supplier.trn!r} is not a valid TRN "
                    "(15 digits, starting 1, ending 03)"
                ),
                severity="error",
            )
        )
    if invoice.buyer.trn is not None and not _ids.is_valid_trn(invoice.buyer.trn):
        violations.append(
            Violation(
                code=ViolationCode.TRN_INVALID,
                field="buyer.trn",
                message=(
                    f"buyer TRN {invoice.buyer.trn!r} is not a valid TRN "
                    "(15 digits, starting 1, ending 03)"
                ),
                severity="error",
            )
        )
    return violations


def check_iban(invoice: Invoice) -> list[Violation]:
    """Format- and checksum-validate the invoice-claimed AE IBAN (ISO 13616 / MOD 97-10) when
    present. Payment always uses the vendor master's verified account (P1); this only flags a
    malformed claim for review."""
    if invoice.iban is None or _ids.is_valid_ae_iban(invoice.iban):
        return []
    return [
        Violation(
            code=ViolationCode.IBAN_INVALID,
            field="iban",
            message=f"{invoice.iban!r} is not a valid AE IBAN (ISO 13616 / MOD 97-10)",
            severity="error",
        )
    ]


def check_dates(invoice: Invoice, *, today: date | None = None) -> list[Violation]:
    """The issue date must not be in the future and the due date, when given, must not precede
    the issue date. PINT AE's VAT point date (BT-7) defaults to the issue date when absent, and
    the canonical `Invoice` carries no separate tax-point field, so validating `issue_date` also
    validates the tax point."""
    today = today if today is not None else date.today()
    violations: list[Violation] = []
    if invoice.issue_date > today:
        violations.append(
            Violation(
                code=ViolationCode.DATE_INVALID,
                field="issue_date",
                message=f"issue date {invoice.issue_date} is in the future (today is {today})",
                severity="error",
            )
        )
    if invoice.due_date is not None and invoice.due_date < invoice.issue_date:
        violations.append(
            Violation(
                code=ViolationCode.DATE_INVALID,
                field="due_date",
                message=(
                    f"due date {invoice.due_date} is before the issue date {invoice.issue_date}"
                ),
                severity="error",
            )
        )
    return violations


def check_currency(invoice: Invoice) -> list[Violation]:
    """Every `Money` amount on the invoice must share one currency code. PINT AE also requires a
    tax-accounting exchange rate (BTAE-04, `ibr-159-ae`) whenever that currency is not AED; the
    canonical `Invoice` carries no exchange-rate field and the PINT AE builder refuses a non-AED
    document, so a non-AED invoice is an error (routes to review, blocks the build) rather than
    a warning that would let approval through only for the build to fail (ADR-0012)."""
    amounts: list[tuple[str, Money]] = [
        ("subtotal", invoice.subtotal),
        ("vat_total", invoice.vat_total),
        ("total", invoice.total),
    ]
    for i, line in enumerate(invoice.lines):
        amounts.append((f"lines[{i}].unit_price", line.unit_price))
        amounts.append((f"lines[{i}].net_amount", line.net_amount))

    currencies = {money.currency for _, money in amounts}
    if len(currencies) > 1:
        by_currency = {field: money.currency for field, money in amounts}
        return [
            Violation(
                code=ViolationCode.CURRENCY_INVALID,
                field="currency",
                message=f"invoice mixes currency codes across amounts: {by_currency}",
                severity="error",
            )
        ]

    (currency,) = currencies
    if currency == NO_CURRENCY:
        return [
            Violation(
                code=ViolationCode.CURRENCY_INVALID,
                field="currency",
                message="invoice currency is missing (ISO 4217 XXX); it is never assumed to be AED",
                severity="error",
            )
        ]
    if currency != "AED":
        return [
            Violation(
                code=ViolationCode.CURRENCY_INVALID,
                field="currency",
                message=(
                    f"invoice currency is {currency}, not AED; PINT AE requires a tax "
                    "accounting exchange rate (BTAE-04) that the canonical invoice does not "
                    "carry, so no e-invoice can be built for it"
                ),
                severity="error",
            )
        ]
    return []


def check_document_type(invoice: Invoice) -> list[Violation]:
    """ADR-0012: the document type (invoice or credit note) must be known, and amounts are stored
    as printed: PINT AE credit notes (381/81) carry positive amounts, so a negative subtotal, VAT
    total, total or line net is an error on either kind."""
    violations: list[Violation] = []
    if invoice.kind is None:
        violations.append(
            Violation(
                code=ViolationCode.INVOICE_TYPE_MISSING,
                field="kind",
                message="document type (380 invoice / 381 credit note) is missing or disputed",
                severity="error",
            )
        )
    label = "a credit note" if invoice.kind is InvoiceKind.CREDIT_NOTE else "an invoice"
    amounts: list[tuple[str, Decimal]] = [
        ("subtotal", invoice.subtotal.amount),
        ("vat_total", invoice.vat_total.amount),
        ("total", invoice.total.amount),
    ]
    amounts += [
        (f"lines[{i}].net_amount", ln.net_amount.amount) for i, ln in enumerate(invoice.lines)
    ]
    for field, amount in amounts:
        if amount < _ZERO:
            violations.append(
                Violation(
                    code=ViolationCode.TOTALS_MISMATCH,
                    field=field,
                    message=(
                        f"{field} {amount} is negative on {label}; amounts are stored as printed "
                        "(a credit note carries positive amounts and is typed 381)"
                    ),
                    severity="error",
                )
            )
    return violations


def check_mandatory_fields(invoice: Invoice, *, buyer_trn: str | None = None) -> list[Violation]:
    """Presence checks for the canonical `Invoice`'s own required business terms. The raw
    51-mandatory-field PINT AE checklist (`apx.extract.schema.HEADER_FIELD_BT`) is extraction's
    concern; by the time an `Invoice` exists here it has already passed the trust gate,
    so this only catches values that type-check as present but are semantically empty, plus the
    invoice-line cardinality rule (an Invoice line group, IBG-25, must occur at least once).
    The supplier address must name a UAE emirate (`apx.validate.address.resolve_emirate`, the
    lookup the UBL builder uses for the seller city and subdivision).
    With `buyer_trn` (our own organisation's configured TRN) the document's buyer TRN is not
    required: the e-invoice takes the configured one and `check_buyer_trn` cross-checks."""
    violations: list[Violation] = []

    def _require(field: str, value: str | None) -> None:
        if value is None or not value.strip():
            violations.append(
                Violation(
                    code=ViolationCode.MISSING_MANDATORY,
                    field=field,
                    message=f"{field} is required and must not be blank",
                    severity="error",
                )
            )

    _require("number", invoice.number)
    _require("supplier.name", invoice.supplier.name)
    _require("supplier.trn", invoice.supplier.trn)
    _require("buyer.name", invoice.buyer.name)
    if resolve_emirate(invoice.supplier.address) is None:
        # ADR-0012 amendment 4: the UBL builder derives the seller city and country subdivision
        # from the same lookup, and PINT AE requires both (`ibr-143-ae`), so an address naming
        # no emirate would pass here and fail at the build, after approval.
        violations.append(
            Violation(
                code=ViolationCode.MISSING_MANDATORY,
                field=SUPPLIER_ADDRESS_VIOLATION_FIELD,
                message=(
                    "supplier address names no UAE emirate; PINT AE needs the seller city and "
                    "country subdivision (ibr-143-ae), so correct the supplier address"
                ),
                severity="error",
            )
        )
    if buyer_trn is None:
        # The buyer's TRN is the e-invoice's buyer electronic address (BT-49, `cbc:EndpointID`),
        # which the PINT AE builder never fabricates.
        _require("buyer.trn", invoice.buyer.trn)

    if not invoice.lines:
        violations.append(
            Violation(
                code=ViolationCode.MISSING_MANDATORY,
                field="lines",
                message="an invoice must have at least one line (IBG-25)",
                severity="error",
            )
        )
    for i, line in enumerate(invoice.lines):
        _require(f"lines[{i}].description", line.description)

    return violations


def check_buyer_trn(invoice: Invoice, buyer_trn: str) -> list[Violation]:
    """ADR-0012 amendment 2: the buyer is our own organisation, so the e-invoice's buyer TRN is
    the configured `buyer_trn`. A document that names a different buyer TRN is an error (it may
    be addressed to someone else) and goes to review; it is never silently overwritten. A
    document with no buyer TRN is fine: the configured one is used."""
    printed = invoice.buyer.trn
    if printed is None or not printed.strip() or printed.strip() == buyer_trn.strip():
        return []
    return [
        Violation(
            code=ViolationCode.BUYER_TRN_MISMATCH,
            field="buyer.trn",
            message=(
                "the document's buyer TRN differs from our configured buyer TRN; the e-invoice "
                "uses the configured TRN, so confirm the invoice is addressed to us"
            ),
            severity="error",
        )
    ]


def validate_invoice(
    invoice: Invoice, *, today: date | None = None, buyer_trn: str | None = None
) -> ValidationReport:
    """Runs every M4.1 field-level check and merges the violations. 3-way match
     is layered on separately via `three_way_match`, once a `PurchasingLookup` is wired
    in. `buyer_trn` is our organisation's configured TRN (`WorkerSettings.buyer_trn`); without
    it the document's own buyer TRN is required instead."""
    violations = (
        *check_document_type(invoice),
        *check_mandatory_fields(invoice, buyer_trn=buyer_trn),
        *([] if buyer_trn is None else check_buyer_trn(invoice, buyer_trn)),
        *check_vat(invoice),
        *check_trn(invoice),
        *check_iban(invoice),
        *check_dates(invoice, today=today),
        *check_currency(invoice),
    )
    return ValidationReport(violations=violations)


class InvoiceValidator:
    """Concrete `Validator`: field-level checks only. 3-way match needs a `PurchasingLookup`
    that M4.2 wires in separately. The worker passes `buyer_trn` (our own organisation's TRN,
    required config); None keeps the document-only checks (evals, fixtures)."""

    def __init__(self, *, buyer_trn: str | None = None) -> None:
        self._buyer_trn = buyer_trn

    def validate(self, invoice: Invoice) -> ValidationReport:
        return validate_invoice(invoice, buyer_trn=self._buyer_trn)


def three_way_match(
    invoice: Invoice,
    po: PurchaseOrder | None,
    receipts: Sequence[GoodsReceipt],
    tolerance: MatchTolerance,
) -> list[Violation]:
    """Compare invoice lines to PO lines and received quantities within tolerance."""
    raise NotImplementedError
