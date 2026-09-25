"""Invoice-line -> PO-line attribution (`apx.common.purchasing.attribute_po_line`) and its use by
the grounded invoice builder and the `price_variance_po` fraud rule: no positional links, a clean
seeded-ERP invoice raises no variance, a real price change does, an unmatchable line is reported
as low-severity `po_line_unmatched` (PRD §10 M4.2 / M8.1, ADR-0012 no money-path defaults)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pytest

from apx.common.invoice import Invoice, InvoiceLine, Party, VatCategory
from apx.common.master import BankDetailChange
from apx.common.purchasing import PoLine, PurchaseOrder, attribute_po_line
from apx.common.types import FieldValue, InvoiceId, Language, Money, Sha256Hex, VendorId
from apx.extract.interfaces import (
    ExtractionResult,
    ExtractorId,
    ExtractorOutput,
    FieldName,
    InvoiceTypeCode,
    RawLineItem,
)
from apx.fraud.interfaces import FraudSeverity
from apx.fraud.rules import RULE_ID_PO_LINE_UNMATCHED, PriceVarianceRule
from apx.workflows.components import GroundedInvoiceBuilder
from apx.workflows.interfaces import InvoiceWorkflowInput
from factory.erp import Erp, generate_erp
from factory.layouts import LAYOUTS, TaxCase, build_invoice_context

VARIANCE = "price_variance_po"


def _money(amount: str | Decimal) -> Money:
    return Money(amount=Decimal(amount), currency="AED")


def _po_line(ref: str, description: str, quantity: str, price: str) -> PoLine:
    return PoLine(
        line_ref=ref, description=description, quantity=Decimal(quantity), unit_price=_money(price)
    )


def _line(description: str, quantity: str, price: str, ref: str | None = None) -> InvoiceLine:
    qty, unit = Decimal(quantity), Decimal(price)
    return InvoiceLine(
        description=description,
        quantity=qty,
        unit_price=_money(unit),
        net_amount=_money(qty * unit),
        vat_category=VatCategory.STANDARD,
        vat_rate=Decimal("5"),
        po_line_ref=ref,
    )


def _po(*lines: PoLine, number: str = "PO-1") -> PurchaseOrder:
    return PurchaseOrder(number=number, vendor_id=VendorId("V1"), lines=lines)


def _invoice(lines: Sequence[InvoiceLine], po_number: str = "PO-1") -> Invoice:
    subtotal = sum((line.net_amount.amount for line in lines), Decimal(0))
    party = Party(name="Acme LLC", trn=None, address=None, country="AE")
    sha = hashlib.sha256(repr(lines).encode()).hexdigest()
    return Invoice(
        id=InvoiceId(sha),
        document_sha256=sha,
        language=Language.EN,
        number="INV-1",
        issue_date=date(2026, 1, 15),
        due_date=None,
        supplier=party,
        buyer=party,
        vendor_id=VendorId("V1"),
        po_number=po_number,
        lines=tuple(lines),
        subtotal=_money(subtotal),
        vat_total=_money("0"),
        total=_money(subtotal),
        iban=None,
    )


@dataclass
class _History:
    """`InvoiceHistory` with only the purchase-order lookup populated."""

    pos: dict[str, PurchaseOrder]

    def recent_for_vendor(self, vendor_id: VendorId, days: int) -> Sequence[Invoice]:
        return []

    def bank_changes(self, vendor_id: VendorId, days: int) -> Sequence[BankDetailChange]:
        return []

    def vendors_using_iban(self, iban: str) -> Sequence[VendorId]:
        return []

    def purchase_order(self, number: str) -> PurchaseOrder | None:
        return self.pos.get(number)


def _rule_ids(invoice: Invoice, po: PurchaseOrder) -> list[str]:
    flags = PriceVarianceRule().evaluate(invoice, _History({po.number: po}))
    return [flag.rule_id for flag in flags]


# --- attribute_po_line --------------------------------------------------------------------


def test_a_stated_ref_is_authoritative() -> None:
    po = _po(_po_line("1", "Cement", "10", "20"), _po_line("2", "Sand", "5", "8"))
    assert attribute_po_line(_line("Sand", "5", "8", ref="1"), po) == po.lines[0]
    assert attribute_po_line(_line("Cement", "10", "20", ref="9"), po) is None


def test_without_a_ref_description_and_quantity_must_both_match() -> None:
    po = _po(_po_line("1", "Cement, OPC  42.5N", "10", "20"), _po_line("2", "Sand", "5", "8"))
    assert attribute_po_line(_line("  cement, opc 42.5n ", "10", "99"), po) == po.lines[0]
    assert attribute_po_line(_line("Cement, OPC 42.5N", "11", "20"), po) is None
    assert attribute_po_line(_line("Gravel", "10", "20"), po) is None


def test_an_ambiguous_description_and_quantity_is_unmatched() -> None:
    po = _po(_po_line("1", "Cement", "10", "20"), _po_line("2", "Cement", "10", "25"))
    assert attribute_po_line(_line("Cement", "10", "20"), po) is None


# --- price_variance_po --------------------------------------------------------------------


def test_invoice_billing_its_po_exactly_raises_nothing() -> None:
    po = _po(_po_line("1", "Cement", "10", "20"), _po_line("2", "Sand", "5", "8"))
    invoice = _invoice([_line("Sand", "5", "8"), _line("Cement", "10", "20")])
    assert _rule_ids(invoice, po) == []


def test_real_price_change_on_a_matched_line_is_a_variance() -> None:
    po = _po(_po_line("1", "Cement", "10", "20"), _po_line("2", "Sand", "5", "8"))
    invoice = _invoice([_line("Cement", "10", "20"), _line("Sand", "5", "10")])
    flags = PriceVarianceRule().evaluate(invoice, _History({"PO-1": po}))
    assert [flag.rule_id for flag in flags] == [VARIANCE]
    assert "line 2 price 8" in flags[0].evidence
    assert flags[0].score >= 0.5


def test_unmatchable_line_is_unmatched_not_a_variance() -> None:
    po = _po(_po_line("1", "Cement", "10", "20"))
    invoice = _invoice([_line("Cement", "10", "20"), _line("Diesel generator service", "1", "900")])
    flags = PriceVarianceRule().evaluate(invoice, _History({"PO-1": po}))
    assert [flag.rule_id for flag in flags] == [RULE_ID_PO_LINE_UNMATCHED]
    assert flags[0].severity is FraudSeverity.LOW
    assert flags[0].score < 0.5  # informational: below the fraud_review routing line
    assert "1 of 2 line(s)" in flags[0].evidence
    assert "Diesel generator service" in flags[0].evidence


# --- builder + seeded ERP (the M6.6 smoke finding) ----------------------------------------


def _db_shaped(po: PurchaseOrder) -> PurchaseOrder:
    """The PO as `PgInvoiceHistory` returns it after `factory.seed_db`: refs are "1".."n"."""
    lines = tuple(
        line.model_copy(update={"line_ref": str(i)}) for i, line in enumerate(po.lines, start=1)
    )
    return po.model_copy(update={"lines": lines})


class _Vendors:
    def __init__(self, erp: Erp) -> None:
        self._by_trn = {vendor.trn: vendor.id for vendor in erp.vendors}

    async def vendor_id_for_trn(self, trn: str) -> VendorId | None:
        return self._by_trn.get(trn)


def _extraction(context: dict[str, object], sha: Sha256Hex) -> ExtractionResult:
    values = {
        FieldName.INVOICE_NUMBER: context["invoice_number"],
        FieldName.ISSUE_DATE: context["issue_date_iso"],
        FieldName.DUE_DATE: context["due_date_iso"],
        FieldName.SUPPLIER_NAME: context["supplier_name_en"],
        FieldName.SUPPLIER_TRN: context["supplier_trn"],
        FieldName.BUYER_NAME: context["buyer_name_en"],
        FieldName.BUYER_TRN: context["buyer_trn"],
        FieldName.CURRENCY: context["currency"],
        FieldName.SUBTOTAL: context["subtotal"],
        FieldName.VAT_TOTAL: context["vat_total"],
        FieldName.TOTAL: context["total"],
        FieldName.PO_NUMBER: context["po_number"],
    }
    grounded = {
        name: FieldValue(name=name.value, raw=str(raw), normalized=str(raw))
        for name, raw in values.items()
    }
    raw_lines = context["lines"]
    assert isinstance(raw_lines, tuple)
    lines = tuple(
        RawLineItem(
            description=line.description_en,
            quantity=str(line.quantity),
            unit_price=str(line.unit_price),
            net_amount=str(line.net_amount),
            vat_rate=f"{line.vat_rate * 100}%",
            vat_category=VatCategory.STANDARD,
        )
        for line in raw_lines
    )

    def output(which: ExtractorId) -> ExtractorOutput:
        return ExtractorOutput(
            extractor=which,
            model=which.value,
            fields=(),
            line_items=lines,
            logprobs={},
            valid_json=True,
            invoice_type_code=InvoiceTypeCode.INVOICE,
        )

    return ExtractionResult(
        document_sha256=sha,
        a=output(ExtractorId.A),
        b=output(ExtractorId.B),
        arbiter=None,
        disputed=frozenset(),
        grounded=grounded,
    )


@pytest.fixture(scope="module")
def erp() -> Erp:
    return generate_erp(2026, n_vendors=20, n_users=8)


@pytest.mark.parametrize("vendor_index", range(12))
async def test_clean_seeded_invoice_raises_no_price_variance(erp: Erp, vendor_index: int) -> None:
    """Before the fix the builder linked line i to PO line str(i), so these fired on nearly every
    clean invoice. Factory lines are drawn independently of PO lines (factory/layouts.py), so
    they are reported as unmatched, never as a variance."""
    context = build_invoice_context(
        LAYOUTS[vendor_index % len(LAYOUTS)],
        erp,
        vendor_index=vendor_index,
        tax_case=TaxCase.STANDARD,
        n_lines=8,
    )
    sha = Sha256Hex(hashlib.sha256(str(vendor_index).encode()).hexdigest())
    inp = InvoiceWorkflowInput(invoice_id=InvoiceId(sha), document_sha256=sha, filename="i.pdf")
    invoice = await GroundedInvoiceBuilder(_Vendors(erp)).build(_extraction(context, sha), inp)
    po = next(p for p in erp.purchase_orders if p.number == context["po_number"])

    assert all(line.po_line_ref is None for line in invoice.lines)
    rule_ids = _rule_ids(invoice, _db_shaped(po))
    assert VARIANCE not in rule_ids
    assert rule_ids == [RULE_ID_PO_LINE_UNMATCHED]
