"""Unit tests for `factory.gold_xml`: the `truth.json` -> `InvoiceFields` mapping builds gold
PINT AE XML that passes the runtime schematron for every UAE factory tax case (including reverse
charge and designated zone), always uses the vendor's ERP-verified IBAN rather than the invoice's
own claimed one (P1), and is deterministic; `render_report` shapes the gold XML markdown report."""

from __future__ import annotations

from typing import Any

import pytest

from apx.common.master import BankAccount
from apx.einvoice.interfaces import DocumentTypeCode
from apx.einvoice.schematron import PintAeSchematronValidator
from factory.erp import VendorRecord, generate_erp
from factory.gold_xml import (
    GoldXmlError,
    GoldXmlResult,
    build_gold_xml,
    build_vendor_index,
    document_type_for,
    render_report,
    truth_to_invoice_fields,
)
from factory.layouts import BUYER_TRADE_LICENSE
from tests.fixtures.pint_ae_data import require_pint_ae_data_root

_ERP = generate_erp(seed=101, n_vendors=3, n_users=6)
_VENDOR = next(v for v in _ERP.vendors if any(a.verified for a in v.bank_accounts))
_VERIFIED_IBAN = next(a.iban for a in _VENDOR.bank_accounts if a.verified)

_BUYER_TRN = "100234567800003"
_BUYER_ADDRESS_EN = (
    "Warehouse 12, Al Quoz Industrial Area 3, Dubai, United Arab Emirates, P.O. Box 45210"
)

# `factory/layouts.py` (`m2-uae-factory`, out of scope here): `TAX_CASE_RATE` per `TaxCase`,
# stored in `truth.json` as the *fraction* applied to `net_amount` (see
# `factory.gold_xml._vat_rate_percent`), not as a percentage.
_TAX_CASE_RATE_FRACTION: dict[str, str] = {
    "standard": "0.05",
    "zero_rated": "0",
    "exempt": "0",
    "reverse_charge": "0",
    "designated_zone": "0",
}


def _line(index: int, *, net_amount: str, rate_fraction: str) -> dict[str, Any]:
    return {
        "index": index,
        "description_en": f"Item {index}",
        "description_ar": f"بند {index}",
        "quantity": "10",
        "unit_price": "100.00",
        "net_amount": net_amount,
        "vat_rate": rate_fraction,
        "vat_amount": str(float(net_amount) * float(rate_fraction)),
    }


def _truth_for(
    tax_case: str,
    *,
    vendor: VendorRecord = _VENDOR,
    n_lines: int = 2,
    claimed_iban: str = "AE070331234567890123456",
) -> dict[str, Any]:
    rate = _TAX_CASE_RATE_FRACTION[tax_case]
    lines = [_line(i, net_amount="1000.00", rate_fraction=rate) for i in range(1, n_lines + 1)]
    subtotal = 1000.00 * n_lines
    vat_total = subtotal * float(rate)
    return {
        "sha256": f"fake-sha-{tax_case}",
        "tax_case": tax_case,
        "split": "train",
        "context": {
            "invoice_number": f"INV-{tax_case}-0001",
            "issue_date_iso": "2026-06-01",
            "due_date_iso": "2026-06-15",
            "currency": "AED",
            "po_number": "PO-2026-000001",
            "supplier_name_en": vendor.name,
            "supplier_trn": vendor.trn,
            "buyer_name_en": "Sovereign Foods Trading LLC",
            "buyer_trn": _BUYER_TRN,
            "buyer_address_en": _BUYER_ADDRESS_EN,
            "iban": claimed_iban,
            "lines": lines,
            "subtotal": f"{subtotal:.2f}",
            "vat_total": f"{vat_total:.2f}",
            "total": f"{subtotal + vat_total:.2f}",
        },
    }


def _validate(xml: bytes) -> tuple[bool, tuple[str, ...]]:
    data_root = require_pint_ae_data_root()
    with PintAeSchematronValidator(DocumentTypeCode.INVOICE, data_root=data_root) as validator:
        result = validator.validate(xml)
    return result.passed, tuple(f.rule_id for f in result.failures if f.flag == "fatal")


@pytest.mark.artifacts
@pytest.mark.parametrize(
    "tax_case", ["standard", "zero_rated", "exempt", "reverse_charge", "designated_zone"]
)
def test_every_tax_case_maps_to_schematron_valid_gold_xml(tax_case: str) -> None:
    truth = _truth_for(tax_case)
    vendors = build_vendor_index(_ERP)
    xml = build_gold_xml(truth, vendors)
    passed, fatal_rule_ids = _validate(xml)
    assert passed, f"tax_case {tax_case} failed schematron: {fatal_rule_ids}"


@pytest.mark.parametrize("tax_case", ["exempt", "designated_zone"])
def test_exempt_and_designated_zone_are_out_of_scope_invoices_with_truth_lines_only(
    tax_case: str,
) -> None:
    """`ibr-151-ae` rejects an all-E/O 380 invoice, so these are 480 'Out of scope of VAT'
    invoices: exactly the ERP-truth lines (no invented line), plus the buyer legal registration
    identifier `ibr-136-ae` requires for type 480."""
    truth = _truth_for(tax_case, n_lines=3)
    vendors = build_vendor_index(_ERP)
    fields, _ = truth_to_invoice_fields(truth, vendors)
    assert document_type_for(tax_case) is DocumentTypeCode.OUT_OF_SCOPE_INVOICE
    assert fields.invoice_type_code == "480"
    assert len(fields.line_items) == 3
    assert fields.buyer_legal_registration_id == BUYER_TRADE_LICENSE
    assert fields.tax_exclusive_amount == "3000.00"
    assert b"<cbc:InvoiceTypeCode>480</cbc:InvoiceTypeCode>" in build_gold_xml(truth, vendors)


@pytest.mark.parametrize("tax_case", ["standard", "zero_rated", "reverse_charge"])
def test_other_tax_cases_are_commercial_invoices_with_truth_lines_only(tax_case: str) -> None:
    truth = _truth_for(tax_case, n_lines=3)
    vendors = build_vendor_index(_ERP)
    fields, _ = truth_to_invoice_fields(truth, vendors)
    assert fields.invoice_type_code == "380"
    assert len(fields.line_items) == 3


def test_uses_verified_vendor_iban_never_the_invoice_claimed_iban() -> None:
    claimed = "AE070331234567890123456"
    assert claimed != _VERIFIED_IBAN
    truth = _truth_for("standard", claimed_iban=claimed)
    vendors = build_vendor_index(_ERP)

    fields, verified_iban = truth_to_invoice_fields(truth, vendors)
    assert verified_iban == _VERIFIED_IBAN

    xml = build_gold_xml(truth, vendors)
    assert _VERIFIED_IBAN.encode() in xml
    assert claimed.encode() not in xml
    # `InvoiceFields.payment_account_iban` still records the invoice's own claimed text, for
    # fidelity to what a real extractor would see -- it is simply never used to pay anything.
    assert fields.payment_account_iban == claimed


def test_supplier_trn_is_the_erp_trn_verbatim_and_already_pint_ae_shaped() -> None:
    """TRNs are generated in `ibr-132-ae`'s shape at source (factory.ids), so the gold XML uses
    the printed TRN unchanged -- no rewriting that the extracted value could never match."""
    truth = _truth_for("standard")
    vendors = build_vendor_index(_ERP)
    fields, _ = truth_to_invoice_fields(truth, vendors)
    assert _VENDOR.trn is not None
    assert fields.supplier_trn == _VENDOR.trn
    assert len(_VENDOR.trn) == 15 and _VENDOR.trn.startswith("1") and _VENDOR.trn.endswith("03")


def test_unknown_tax_case_raises_gold_xml_error() -> None:
    truth = _truth_for("standard")
    truth["tax_case"] = "not_a_real_tax_case"
    vendors = build_vendor_index(_ERP)
    with pytest.raises(GoldXmlError):
        truth_to_invoice_fields(truth, vendors)


def test_unknown_supplier_trn_raises_gold_xml_error() -> None:
    truth = _truth_for("standard")
    truth["context"]["supplier_trn"] = "999999999999999"
    vendors = build_vendor_index(_ERP)
    with pytest.raises(GoldXmlError):
        truth_to_invoice_fields(truth, vendors)


def test_vendor_with_no_verified_account_raises_gold_xml_error() -> None:
    unverified_vendor = _VENDOR.model_copy(
        update={
            "bank_accounts": (
                BankAccount(
                    iban=_VERIFIED_IBAN,
                    verified=False,
                    verified_by=None,
                    verified_at=None,
                    source="unverified",
                ),
            )
        }
    )
    erp = _ERP.model_copy(update={"vendors": (unverified_vendor,)})
    truth = _truth_for("standard", vendor=unverified_vendor)
    vendors = build_vendor_index(erp)
    with pytest.raises(GoldXmlError):
        truth_to_invoice_fields(truth, vendors)


def test_same_truth_builds_byte_identical_xml() -> None:
    truth = _truth_for("standard", n_lines=4)
    vendors = build_vendor_index(_ERP)
    first = build_gold_xml(truth, vendors)
    second = build_gold_xml(truth, vendors)
    assert first == second


def test_render_report_counts_and_pass_rate() -> None:
    results = [
        GoldXmlResult("a", "train", "standard", True, ()),
        GoldXmlResult("b", "train", "exempt", True, ()),
        GoldXmlResult("c", "cal", "designated_zone", False, ("ibr-151-ae",)),
        GoldXmlResult("d", "test", "reverse_charge", True, ()),
    ]
    report = render_report(results, erp_seed=2026)
    assert "3/4 (75.00%)" in report
    assert "0 of 4 documents are credit notes" in report
    assert "| ibr-151-ae | 1 |" in report
    assert "| train | 2 | 2 | 100.00% |" in report
    assert "| cal | 1 | 0 | 0.00% |" in report


def test_render_report_with_no_failures_says_so() -> None:
    results = [GoldXmlResult("a", "train", "standard", True, ())]
    report = render_report(results, erp_seed=2026)
    assert "None -- every document passed" in report
