"""Unit and property tests for `apx.validate.match`: the 3-way match between an invoice, its
purchase order and goods receipts, including tolerance boundaries and partial deliveries."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from apx.common.invoice import Invoice, InvoiceLine, Party, VatCategory
from apx.common.types import InvoiceId, Language, Money, VendorId
from apx.validate.interfaces import (
    GoodsReceipt,
    GrnLine,
    MatchTolerance,
    PoLine,
    PurchaseOrder,
    Violation,
    ViolationCode,
)
from apx.validate.match import DEFAULT_MATCH_TOLERANCE, match_invoice, three_way_match
from factory.erp import generate_erp

_SUPPLIER = Party(name="Al Falah Trading LLC", trn="100123456789012", address="Dubai, UAE")
_BUYER = Party(name="Buyer LLC", trn="100987654321098", address="Abu Dhabi, UAE")


def _money(amount: str, currency: str = "AED") -> Money:
    return Money(amount=Decimal(amount), currency=currency)


def _po_line(ref: str, quantity: str, unit_price: str, description: str = "Widgets") -> PoLine:
    return PoLine(
        line_ref=ref,
        description=description,
        quantity=Decimal(quantity),
        unit_price=_money(unit_price),
    )


def _grn(number: str, po_number: str, lines: Sequence[GrnLine]) -> GoodsReceipt:
    return GoodsReceipt(number=number, po_number=po_number, lines=tuple(lines))


def _invoice_line(
    quantity: str,
    unit_price: str,
    po_line_ref: str | None = "L1",
    description: str = "Widgets",
    currency: str = "AED",
) -> InvoiceLine:
    qty = Decimal(quantity)
    price = _money(unit_price, currency)
    return InvoiceLine(
        description=description,
        quantity=qty,
        unit_price=price,
        net_amount=Money(amount=qty * price.amount, currency=currency),
        vat_category=VatCategory.STANDARD,
        vat_rate=Decimal("0.05"),
        po_line_ref=po_line_ref,
    )


def _invoice(lines: Sequence[InvoiceLine], po_number: str | None = "PO-1") -> Invoice:
    subtotal = sum((line.net_amount.amount for line in lines), Decimal(0))
    return Invoice(
        id=InvoiceId("INV-1"),
        document_sha256="0" * 64,
        language=Language.EN,
        number="INV-1",
        issue_date=date(2026, 1, 15),
        due_date=None,
        supplier=_SUPPLIER,
        buyer=_BUYER,
        vendor_id=VendorId("VEND-1"),
        po_number=po_number,
        lines=tuple(lines),
        subtotal=_money(str(subtotal)),
        vat_total=_money(str(subtotal * Decimal("0.05"))),
        total=_money(str(subtotal * Decimal("1.05"))),
        iban=None,
    )


def _codes(violations: Sequence[Violation]) -> set[ViolationCode]:
    return {v.code for v in violations}


# --- clean match ------------------------------------------------------------------------------


def test_clean_match_passes() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "100", "10.00"),)
    )
    receipts = [
        _grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("100"))])
    ]
    invoice = _invoice([_invoice_line("100", "10.00")])

    result = match_invoice(invoice, po, receipts, DEFAULT_MATCH_TOLERANCE)

    assert result.passed
    assert result.violations == ()
    assert result.lines[0].quantity_received_cumulative == Decimal("100")


def test_three_way_match_wrapper_matches_match_invoice() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "100", "10.00"),)
    )
    receipts = [
        _grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("100"))])
    ]
    invoice = _invoice([_invoice_line("100", "10.00")])

    assert three_way_match(invoice, po, receipts, DEFAULT_MATCH_TOLERANCE) == []


# --- missing PO ---------------------------------------------------------------------------------


def test_missing_po_is_a_single_error() -> None:
    invoice = _invoice([_invoice_line("10", "10.00")], po_number="PO-GHOST")

    result = match_invoice(invoice, None, [], DEFAULT_MATCH_TOLERANCE)

    assert not result.passed
    assert len(result.violations) == 1
    assert result.violations[0].code == ViolationCode.PO_NOT_FOUND
    assert result.lines == ()


# --- missing GRN --------------------------------------------------------------------------------


def test_missing_grn_is_flagged() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "100", "10.00"),)
    )
    invoice = _invoice([_invoice_line("10", "10.00")])

    result = match_invoice(invoice, po, [], DEFAULT_MATCH_TOLERANCE)

    assert ViolationCode.GRN_NOT_FOUND in _codes(result.violations)


# --- over-billing vs quantity over received are distinct ----------------------------------------


def test_over_billing_exceeds_ordered_quantity() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "100", "10.00"),)
    )
    # Received (150) covers the invoice, but the PO only ever ordered 100.
    receipts = [
        _grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("150"))])
    ]
    invoice = _invoice([_invoice_line("150", "10.00")])

    result = match_invoice(invoice, po, receipts, DEFAULT_MATCH_TOLERANCE)

    codes = _codes(result.violations)
    assert ViolationCode.OVER_BILLING in codes
    assert ViolationCode.QUANTITY_OVER_RECEIVED not in codes


def test_partial_grn_caps_billable_quantity() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "100", "10.00"),)
    )
    # Two partial deliveries summing to 60 of the 100 ordered.
    receipts = [
        _grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("35"))]),
        _grn("GRN-2", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("25"))]),
    ]

    ok_invoice = _invoice([_invoice_line("60", "10.00")])
    ok_result = match_invoice(ok_invoice, po, receipts, DEFAULT_MATCH_TOLERANCE)
    assert ok_result.passed
    assert ok_result.lines[0].quantity_received_cumulative == Decimal("60")

    over_invoice = _invoice([_invoice_line("61", "10.00")])
    over_result = match_invoice(over_invoice, po, receipts, DEFAULT_MATCH_TOLERANCE)
    codes = _codes(over_result.violations)
    assert ViolationCode.QUANTITY_OVER_RECEIVED in codes
    assert ViolationCode.OVER_BILLING not in codes  # still within the ordered 100


def test_previously_invoiced_reduces_available_quantity() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "100", "10.00"),)
    )
    receipts = [_grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("60"))])]
    prior = {"L1": Decimal("40")}

    ok = match_invoice(
        _invoice([_invoice_line("20", "10.00")]), po, receipts, DEFAULT_MATCH_TOLERANCE, prior
    )
    assert ok.passed

    over = match_invoice(
        _invoice([_invoice_line("21", "10.00")]), po, receipts, DEFAULT_MATCH_TOLERANCE, prior
    )
    assert ViolationCode.QUANTITY_OVER_RECEIVED in _codes(over.violations)


# --- price tolerance -----------------------------------------------------------------------------


def test_price_within_tolerance_passes() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "10", "100.00"),)
    )
    receipts = [_grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("10"))])]
    invoice = _invoice([_invoice_line("10", "100.40")])  # 0.4% over, within default 0.5%

    result = match_invoice(invoice, po, receipts, DEFAULT_MATCH_TOLERANCE)
    assert ViolationCode.PRICE_OVER_TOLERANCE not in _codes(result.violations)


def test_price_over_tolerance_is_flagged() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "10", "100.00"),)
    )
    receipts = [_grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("10"))])]
    invoice = _invoice([_invoice_line("10", "100.60")])  # 0.6% over, beyond default 0.5%

    result = match_invoice(invoice, po, receipts, DEFAULT_MATCH_TOLERANCE)
    assert ViolationCode.PRICE_OVER_TOLERANCE in _codes(result.violations)


def test_currency_mismatch_skips_price_check() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "10", "100.00"),)
    )
    receipts = [_grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("10"))])]
    invoice = _invoice([_invoice_line("10", "100.00", currency="USD")])

    result = match_invoice(invoice, po, receipts, DEFAULT_MATCH_TOLERANCE)
    codes = _codes(result.violations)
    assert ViolationCode.CURRENCY_MISMATCH in codes
    assert ViolationCode.PRICE_OVER_TOLERANCE not in codes


# --- line matching: ref, description fallback, unknown, duplicate --------------------------------


def test_description_fallback_matches_when_ref_is_missing() -> None:
    po = PurchaseOrder(
        number="PO-1",
        vendor_id=VendorId("VEND-1"),
        lines=(_po_line("L1", "10", "100.00", description="Steel Beams"),),
    )
    receipts = [_grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("10"))])]
    invoice = _invoice(
        [_invoice_line("10", "100.00", po_line_ref=None, description="  steel   beams ")]
    )

    result = match_invoice(invoice, po, receipts, DEFAULT_MATCH_TOLERANCE)
    assert result.passed
    assert result.lines[0].po_line_ref == "L1"


def test_unknown_line_when_no_ref_or_description_matches() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "10", "100.00"),)
    )
    receipts = [_grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("10"))])]
    invoice = _invoice(
        [_invoice_line("10", "100.00", po_line_ref="L9", description="Completely Different Item")]
    )

    result = match_invoice(invoice, po, receipts, DEFAULT_MATCH_TOLERANCE)
    assert ViolationCode.UNKNOWN_LINE in _codes(result.violations)


def test_duplicate_line_flagged_when_two_lines_match_same_po_line() -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "100", "10.00"),)
    )
    receipts = [
        _grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("100"))])
    ]
    invoice = _invoice(
        [
            _invoice_line("40", "10.00", po_line_ref="L1"),
            _invoice_line("40", "10.00", po_line_ref="L1"),
        ]
    )

    result = match_invoice(invoice, po, receipts, DEFAULT_MATCH_TOLERANCE)
    assert ViolationCode.DUPLICATE_LINE in _codes(result.violations)


# --- realistic fixtures from the synthetic ERP factory -------------------------------------------


def test_realistic_erp_purchase_orders_match_their_own_receipts() -> None:
    erp = generate_erp(seed=4242, n_vendors=5, n_users=4)
    assert erp.purchase_orders, "factory produced no purchase orders to exercise"

    checked_a_partial_delivery = False
    for po in erp.purchase_orders:
        grns = [g for g in erp.goods_receipts if g.po_number == po.number]
        received: dict[str, Decimal] = {}
        for grn in grns:
            for line in grn.lines:
                received[line.po_line_ref] = (
                    received.get(line.po_line_ref, Decimal(0)) + line.quantity_received
                )

        invoice_lines = [
            _invoice_line(
                str(received.get(pl.line_ref, Decimal(0))),
                str(pl.unit_price.amount),
                po_line_ref=pl.line_ref,
            )
            for pl in po.lines
            if received.get(pl.line_ref, Decimal(0)) > 0
        ]
        if not invoice_lines:
            continue
        invoice = _invoice(invoice_lines, po_number=po.number)

        result = match_invoice(invoice, po, grns, DEFAULT_MATCH_TOLERANCE)
        assert result.passed, result.violations

        if any(received.get(pl.line_ref, Decimal(0)) < pl.quantity for pl in po.lines):
            checked_a_partial_delivery = True

    assert checked_a_partial_delivery, (
        "seed produced no partial delivery to exercise the ERP partial-delivery mode"
    )


# --- hypothesis: tolerance boundary exactness -----------------------------------------------------


@given(
    po_price=st.decimals(min_value="0.01", max_value="100000", places=2, allow_nan=False),
    tolerance_pct=st.decimals(min_value="0.0001", max_value="0.05", places=4, allow_nan=False),
    over_by=st.decimals(min_value="0.01", max_value="5", places=2, allow_nan=False),
)
def test_price_tolerance_boundary_is_exact(
    po_price: Decimal, tolerance_pct: Decimal, over_by: Decimal
) -> None:
    po = PurchaseOrder(
        number="PO-1", vendor_id=VendorId("VEND-1"), lines=(_po_line("L1", "1", str(po_price)),)
    )
    receipts = [_grn("GRN-1", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=Decimal("1"))])]
    tolerance = MatchTolerance(quantity=Decimal("0"), price_pct=tolerance_pct)
    allowed_deviation = po_price * tolerance_pct

    at_boundary_price = po_price + allowed_deviation
    at_boundary = match_invoice(
        _invoice([_invoice_line("1", str(at_boundary_price))]), po, receipts, tolerance
    )
    assert ViolationCode.PRICE_OVER_TOLERANCE not in _codes(at_boundary.violations)

    beyond_boundary_price = at_boundary_price + over_by
    beyond_boundary = match_invoice(
        _invoice([_invoice_line("1", str(beyond_boundary_price))]), po, receipts, tolerance
    )
    assert ViolationCode.PRICE_OVER_TOLERANCE in _codes(beyond_boundary.violations)


# --- hypothesis: partial GRN sums are exact -------------------------------------------------------


@given(
    shares=st.lists(st.integers(min_value=1, max_value=1000).map(Decimal), min_size=1, max_size=6)
)
def test_partial_grn_quantities_sum_exactly(shares: list[Decimal]) -> None:
    total_received = sum(shares, Decimal(0))
    po = PurchaseOrder(
        number="PO-1",
        vendor_id=VendorId("VEND-1"),
        lines=(_po_line("L1", str(total_received + Decimal("1000")), "10.00"),),
    )
    receipts = [
        _grn(f"GRN-{i}", "PO-1", [GrnLine(po_line_ref="L1", quantity_received=share)])
        for i, share in enumerate(shares)
    ]

    exact = match_invoice(
        _invoice([_invoice_line(str(total_received), "10.00")]),
        po,
        receipts,
        DEFAULT_MATCH_TOLERANCE,
    )
    assert exact.lines[0].quantity_received_cumulative == total_received
    assert ViolationCode.QUANTITY_OVER_RECEIVED not in _codes(exact.violations)

    over = match_invoice(
        _invoice([_invoice_line(str(total_received + Decimal("1")), "10.00")]),
        po,
        receipts,
        DEFAULT_MATCH_TOLERANCE,
    )
    assert ViolationCode.QUANTITY_OVER_RECEIVED in _codes(over.violations)
