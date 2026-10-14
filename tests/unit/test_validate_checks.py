"""Unit tests for the non-VAT `apx.validate.interfaces` checks: TRN, AE IBAN, dates, currency
and mandatory-field presence, plus `ValidationReport.passed` and the `InvoiceValidator` wiring."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from apx.common.invoice import Invoice, InvoiceKind, InvoiceLine, Party, VatCategory
from apx.common.types import InvoiceId, Language, Money, Sha256Hex, VendorId
from apx.validate.ids import make_ae_iban
from apx.validate.interfaces import (
    InvoiceValidator,
    MatchTolerance,
    ValidationReport,
    Violation,
    ViolationCode,
    check_currency,
    check_dates,
    check_iban,
    check_mandatory_fields,
    check_trn,
    is_valid_iban,
    is_valid_trn,
    three_way_match,
    validate_invoice,
)

_SHA = "c" * 64
_VALID_TRN_1 = "100200300400503"
_VALID_TRN_2 = "100600700800903"
_VALID_IBAN = make_ae_iban("014", "0000000000001234")


def _party(trn: str | None = _VALID_TRN_1, name: str = "Party") -> Party:
    return Party(name=name, trn=trn, address="Some Street, Dubai", country="AE")


def _invoice(
    *,
    number: str = "INV-0001",
    issue_date: date = date(2025, 1, 15),
    due_date: date | None = date(2025, 2, 15),
    supplier: Party | None = None,
    buyer: Party | None = None,
    lines: tuple[InvoiceLine, ...] | None = None,
    iban: str | None = _VALID_IBAN,
    currency: str = "AED",
) -> Invoice:
    if lines is None:
        lines = (
            InvoiceLine(
                description="Widget",
                quantity=Decimal("1"),
                unit_price=Money(amount=Decimal("100.00"), currency=currency),
                net_amount=Money(amount=Decimal("100.00"), currency=currency),
                vat_category=VatCategory.STANDARD,
                vat_rate=Decimal("5"),
            ),
        )
    subtotal = sum((line.net_amount.amount for line in lines), Decimal("0"))
    return Invoice(
        id=InvoiceId("INV"),
        document_sha256=Sha256Hex(_SHA),
        language=Language.EN,
        number=number,
        issue_date=issue_date,
        due_date=due_date,
        supplier=supplier if supplier is not None else _party(_VALID_TRN_1, "Supplier"),
        buyer=buyer if buyer is not None else _party(_VALID_TRN_2, "Buyer"),
        vendor_id=VendorId("VEND"),
        po_number=None,
        lines=lines,
        subtotal=Money(amount=subtotal, currency=currency),
        vat_total=Money(amount=Decimal("5.00"), currency=currency),
        total=Money(amount=subtotal + Decimal("5.00"), currency=currency),
        iban=iban,
        kind=InvoiceKind.INVOICE,
    )


# --- ValidationReport --------------------------------------------------------------------------


def test_report_passed_true_with_no_violations() -> None:
    assert ValidationReport(violations=()).passed


def test_report_passed_false_with_error_violation() -> None:
    report = ValidationReport(
        violations=(
            Violation(
                code=ViolationCode.DATE_INVALID, field="issue_date", message="x", severity="error"
            ),
        )
    )
    assert not report.passed


def test_report_passed_true_with_only_warnings() -> None:
    report = ValidationReport(
        violations=(
            Violation(
                code=ViolationCode.CURRENCY_INVALID,
                field="currency",
                message="x",
                severity="warning",
            ),
        )
    )
    assert report.passed


# --- TRN / IBAN delegation -----------------------------------------------------------------------


def test_is_valid_trn_delegates() -> None:
    assert is_valid_trn(_VALID_TRN_1)
    assert not is_valid_trn("123")


def test_is_valid_iban_delegates() -> None:
    assert is_valid_iban(_VALID_IBAN)
    assert not is_valid_iban("not-an-iban")


def test_check_trn_accepts_valid_trns() -> None:
    assert check_trn(_invoice()) == []


def test_check_trn_flags_malformed_supplier_trn() -> None:
    invoice = _invoice(supplier=_party("bad-trn", "Supplier"))
    violations = check_trn(invoice)
    assert [v.code for v in violations] == [ViolationCode.TRN_INVALID]
    assert violations[0].field == "supplier.trn"


def test_check_trn_ignores_absent_trn() -> None:
    invoice = _invoice(buyer=_party(None, "Buyer"))
    assert check_trn(invoice) == []


# --- IBAN ----------------------------------------------------------------------------------------


def test_check_iban_accepts_valid_ae_iban() -> None:
    assert check_iban(_invoice()) == []


def test_check_iban_accepts_absent_iban() -> None:
    assert check_iban(_invoice(iban=None)) == []


_IBAN_CHECKSUM_FLIPPED = _VALID_IBAN[:-1] + ("1" if _VALID_IBAN[-1] != "1" else "2")


@pytest.mark.parametrize(
    "bad_iban",
    ["SA" + _VALID_IBAN[2:], _VALID_IBAN + "0", _IBAN_CHECKSUM_FLIPPED],
)
def test_check_iban_flags_bad_iban(bad_iban: str) -> None:
    violations = check_iban(_invoice(iban=bad_iban))
    assert [v.code for v in violations] == [ViolationCode.IBAN_INVALID]


# --- Dates -----------------------------------------------------------------------------------


def test_check_dates_accepts_valid_dates() -> None:
    invoice = _invoice(issue_date=date(2025, 1, 15), due_date=date(2025, 2, 15))
    assert check_dates(invoice, today=date(2025, 6, 1)) == []


def test_check_dates_flags_future_issue_date() -> None:
    invoice = _invoice(issue_date=date(2026, 1, 1), due_date=None)
    violations = check_dates(invoice, today=date(2025, 1, 1))
    assert [v.code for v in violations] == [ViolationCode.DATE_INVALID]
    assert violations[0].field == "issue_date"


def test_check_dates_flags_due_before_issue() -> None:
    invoice = _invoice(issue_date=date(2025, 6, 1), due_date=date(2025, 5, 1))
    violations = check_dates(invoice, today=date(2025, 6, 15))
    assert [v.code for v in violations] == [ViolationCode.DATE_INVALID]
    assert violations[0].field == "due_date"


def test_check_dates_allows_missing_due_date() -> None:
    invoice = _invoice(issue_date=date(2025, 6, 1), due_date=None)
    assert check_dates(invoice, today=date(2025, 6, 15)) == []


# --- Currency --------------------------------------------------------------------------------


def test_check_currency_accepts_consistent_aed() -> None:
    assert check_currency(_invoice()) == []


def test_check_currency_flags_non_aed_as_error() -> None:
    """Error, not warning: the PINT AE builder refuses a non-AED document, so a warning would let
    an approval through only for the e-invoice build to fail afterwards (ADR-0012)."""
    invoice = _invoice(currency="USD")
    violations = check_currency(invoice)
    assert [v.code for v in violations] == [ViolationCode.CURRENCY_INVALID]
    assert violations[0].severity == "error"
    assert not validate_invoice(invoice, today=date(2025, 6, 1)).passed


def test_check_currency_flags_mixed_currencies_as_error() -> None:
    line = InvoiceLine(
        description="Widget",
        quantity=Decimal("1"),
        unit_price=Money(amount=Decimal("100.00"), currency="USD"),
        net_amount=Money(amount=Decimal("100.00"), currency="USD"),
        vat_category=VatCategory.STANDARD,
        vat_rate=Decimal("5"),
    )
    invoice = _invoice(lines=(line,))  # header stays AED while the line is USD
    violations = check_currency(invoice)
    assert [v.code for v in violations] == [ViolationCode.CURRENCY_INVALID]
    assert violations[0].severity == "error"


# --- Mandatory fields --------------------------------------------------------------------------


def test_check_mandatory_fields_accepts_complete_invoice() -> None:
    assert check_mandatory_fields(_invoice()) == []


def test_check_mandatory_fields_flags_blank_number() -> None:
    violations = check_mandatory_fields(_invoice(number=""))
    assert [v.code for v in violations] == [ViolationCode.MISSING_MANDATORY]
    assert violations[0].field == "number"


def test_check_mandatory_fields_flags_missing_supplier_trn() -> None:
    invoice = _invoice(supplier=_party(None, "Supplier"))
    violations = check_mandatory_fields(invoice)
    assert [v.code for v in violations] == [ViolationCode.MISSING_MANDATORY]
    assert violations[0].field == "supplier.trn"


def test_check_mandatory_fields_flags_missing_buyer_trn() -> None:
    """The buyer TRN is the e-invoice's buyer electronic address (BT-49), never fabricated."""
    invoice = _invoice(buyer=_party(None, "Buyer"))
    violations = check_mandatory_fields(invoice)
    assert [v.code for v in violations] == [ViolationCode.MISSING_MANDATORY]
    assert violations[0].field == "buyer.trn"
    assert violations[0].severity == "error"


def test_check_mandatory_fields_flags_no_lines() -> None:
    invoice = _invoice(lines=())
    violations = check_mandatory_fields(invoice)
    assert [v.code for v in violations] == [ViolationCode.MISSING_MANDATORY]
    assert violations[0].field == "lines"


# --- Aggregate validator -----------------------------------------------------------------------


def test_validate_invoice_passes_a_clean_invoice() -> None:
    report = validate_invoice(_invoice(), today=date(2025, 6, 1))
    assert report.passed
    assert report.violations == ()


def test_invoice_validator_matches_validate_invoice() -> None:
    invoice = _invoice()
    validator = InvoiceValidator()
    expected = validate_invoice(invoice, today=date.today())
    assert validator.validate(invoice).violations == expected.violations


def test_three_way_match_not_yet_implemented() -> None:
    """`apx.validate.match` implements this; the field-level module only reserves the interface."""
    with pytest.raises(NotImplementedError):
        three_way_match(_invoice(), None, (), MatchTolerance())
