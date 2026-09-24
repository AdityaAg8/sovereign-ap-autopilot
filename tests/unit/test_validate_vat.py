"""Hypothesis property tests for `apx.validate.interfaces.check_vat`: for any UAE VAT category
and any quantity/price, a correctly computed invoice has no VAT violations, and perturbing the
stated VAT or the stated total by a nonzero amount is always caught (PRD §10 M4.1 acceptance)."""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from hypothesis import given
from hypothesis import strategies as st

from apx.common.invoice import Invoice, InvoiceKind, InvoiceLine, Party, VatCategory
from apx.common.types import InvoiceId, Language, Money, Sha256Hex, VendorId
from apx.validate.interfaces import STANDARD_VAT_RATE, ViolationCode, check_vat

_SHA = "b" * 64
_SUPPLIER = Party(name="Supplier", trn=None, address=None, country="AE")
_BUYER = Party(name="Buyer", trn=None, address=None, country="AE")


def _cents(amount: Decimal) -> Decimal:
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _rate_for(category: VatCategory) -> Decimal:
    if category is VatCategory.STANDARD:
        return STANDARD_VAT_RATE
    if category is VatCategory.REVERSE_CHARGE:
        return STANDARD_VAT_RATE  # informational rate; VAT actually charged is still zero
    return Decimal("0")


def _correct_invoice(category: VatCategory, quantity: Decimal, unit_price: Decimal) -> Invoice:
    net_amount = _cents(quantity * unit_price)
    rate = _rate_for(category)
    line_vat = (
        _cents(net_amount * rate / Decimal(100))
        if category is VatCategory.STANDARD
        else Decimal("0")
    )
    total = _cents(net_amount + line_vat)
    return Invoice(
        id=InvoiceId("INV-PROP"),
        document_sha256=Sha256Hex(_SHA),
        language=Language.EN,
        number="INV-PROP-0001",
        issue_date=date(2025, 1, 1),
        due_date=None,
        supplier=_SUPPLIER,
        buyer=_BUYER,
        vendor_id=VendorId("VEND-PROP"),
        po_number=None,
        lines=(
            InvoiceLine(
                description="line",
                quantity=quantity,
                unit_price=Money(amount=unit_price),
                net_amount=Money(amount=net_amount),
                vat_category=category,
                vat_rate=rate,
            ),
        ),
        subtotal=Money(amount=net_amount),
        vat_total=Money(amount=line_vat),
        total=Money(amount=total),
        iban=None,
        kind=InvoiceKind.INVOICE,
    )


_categories = st.sampled_from(list(VatCategory))
_quantities = st.decimals(
    min_value=Decimal("0.01"), max_value=Decimal("1000"), places=2, allow_nan=False
)
_unit_prices = st.decimals(
    min_value=Decimal("0.01"), max_value=Decimal("100000"), places=2, allow_nan=False
)
_nonzero_deltas = st.decimals(
    min_value=Decimal("0.01"), max_value=Decimal("500"), places=2, allow_nan=False
)


@given(category=_categories, quantity=_quantities, unit_price=_unit_prices)
def test_correctly_computed_invoice_has_no_vat_violations(
    category: VatCategory, quantity: Decimal, unit_price: Decimal
) -> None:
    invoice = _correct_invoice(category, quantity, unit_price)
    assert check_vat(invoice) == []


@given(
    category=_categories,
    quantity=_quantities,
    unit_price=_unit_prices,
    delta=_nonzero_deltas,
)
def test_corrupted_vat_total_is_always_caught(
    category: VatCategory, quantity: Decimal, unit_price: Decimal, delta: Decimal
) -> None:
    invoice = _correct_invoice(category, quantity, unit_price)
    corrupted = invoice.model_copy(
        update={"vat_total": Money(amount=invoice.vat_total.amount + delta)}
    )
    violations = check_vat(corrupted)
    assert violations, "a nonzero VAT total delta must always be caught"
    assert any(
        v.code is ViolationCode.TOTALS_MISMATCH and v.field == "vat_total" for v in violations
    )


@given(
    category=_categories,
    quantity=_quantities,
    unit_price=_unit_prices,
    delta=_nonzero_deltas,
)
def test_corrupted_total_is_always_caught(
    category: VatCategory, quantity: Decimal, unit_price: Decimal, delta: Decimal
) -> None:
    invoice = _correct_invoice(category, quantity, unit_price)
    corrupted = invoice.model_copy(update={"total": Money(amount=invoice.total.amount + delta)})
    violations = check_vat(corrupted)
    assert violations, "a nonzero total delta must always be caught"
    assert any(v.code is ViolationCode.TOTALS_MISMATCH and v.field == "total" for v in violations)


@given(
    quantity=_quantities,
    unit_price=_unit_prices,
    bad_rate=st.decimals(
        min_value=Decimal("0.01"), max_value=Decimal("99"), places=2, allow_nan=False
    ),
)
def test_standard_line_with_wrong_rate_is_caught(
    quantity: Decimal, unit_price: Decimal, bad_rate: Decimal
) -> None:
    if bad_rate == STANDARD_VAT_RATE:
        return
    invoice = _correct_invoice(VatCategory.STANDARD, quantity, unit_price)
    net_amount = invoice.lines[0].net_amount.amount
    bad_vat = _cents(net_amount * bad_rate / Decimal(100))
    corrupted = invoice.model_copy(
        update={
            "lines": (invoice.lines[0].model_copy(update={"vat_rate": bad_rate}),),
            "vat_total": Money(amount=bad_vat),
            "total": Money(amount=_cents(net_amount + bad_vat)),
        }
    )
    violations = check_vat(corrupted)
    assert any(v.code is ViolationCode.VAT_CATEGORY_RATE_MISMATCH for v in violations)


@given(
    quantity=_quantities,
    unit_price=_unit_prices,
    bad_rate=st.decimals(
        min_value=Decimal("0.01"), max_value=Decimal("99"), places=2, allow_nan=False
    ),
)
def test_zero_rate_category_with_nonzero_rate_is_caught(
    quantity: Decimal, unit_price: Decimal, bad_rate: Decimal
) -> None:
    invoice = _correct_invoice(VatCategory.ZERO_RATED, quantity, unit_price)
    corrupted = invoice.model_copy(
        update={"lines": (invoice.lines[0].model_copy(update={"vat_rate": bad_rate}),)}
    )
    violations = check_vat(corrupted)
    assert any(
        v.code is ViolationCode.VAT_CATEGORY_RATE_MISMATCH and v.field == "lines[0].vat_rate"
        for v in violations
    )
