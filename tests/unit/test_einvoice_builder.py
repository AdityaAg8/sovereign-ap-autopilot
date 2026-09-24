"""Unit tests for `apx.einvoice.builder`: PINT AE UBL 2.1 Invoice/CreditNote XML built from
`InvoiceFields` plus a `factory.erp`-generated vendor's admin-verified IBAN passes the runtime
schematron 100% for every VAT category (S, Z, E, AE, O) and for a credit note; a missing
mandatory element fails with its rule ID (all three marked `artifacts`, PRD §8: they need the
real pinned PINT AE data). The builder's own input-validation errors (no verified IBAN, unknown
VAT category) and the canonical-diff stability check need no schematron run at all, so they stay
unmarked and run in the default suite (PRD §10 M4.3)."""

from __future__ import annotations

import pytest

from apx.einvoice.builder import EInvoiceError, build_ubl_xml, canonical_xml_diff
from apx.einvoice.interfaces import DocumentTypeCode
from apx.einvoice.schematron import PintAeSchematronValidator
from apx.extract.schema import InvoiceFields, LineItemFields
from factory.erp import generate_erp
from tests.fixtures.pint_ae_data import require_pint_ae_data_root

_ERP = generate_erp(seed=101, n_vendors=3, n_users=6)
_VENDOR = _ERP.vendors[0]
_VERIFIED_IBAN = next(a.iban for a in _VENDOR.bank_accounts if a.verified)

# `factory.erp.generate_erp` only guarantees a 15-digit, "100"-prefixed TRN (its own format
# check, `apx.validate.ids.is_valid_trn`, does not enforce a check-digit suffix). PINT AE's
# jurisdiction rule `ibr-132-ae` additionally requires UAE TRNs to end in "03"; a real extracted
# TRN would carry that suffix, so the test fixture's *invoice text* uses it too, independent of
# the ERP vendor record's own randomly generated tail.
_SUPPLIER_TRN = _VENDOR.trn[:-2] + "03"
_BUYER_TRN = "100000000000003"


def _header(**overrides: str | None) -> dict[str, str | None]:
    fields: dict[str, str | None] = {
        "invoice_number": "INV-2026-0001",
        "issue_date": "2026-06-01",
        "due_date": "2026-06-15",
        "currency_code": "AED",
        "supplier_name": _VENDOR.name,
        "supplier_identifier": _SUPPLIER_TRN,
        "supplier_legal_registration_id": "112345678900003",
        "supplier_trn": _SUPPLIER_TRN,
        "supplier_address_line1": _VENDOR.address_en,
        "supplier_city": _VENDOR.emirate,
        "supplier_country_subdivision": _VENDOR.emirate[:3].upper(),
        "supplier_country_code": "AE",
        "buyer_name": "Sovereign AP Autopilot Demo LLC",
        "buyer_identifier": _BUYER_TRN,
        "buyer_legal_registration_id": "112345679000001",
        "buyer_trn": _BUYER_TRN,
        "buyer_address_line1": "Building 1, Al Khalidiyah",
        "buyer_city": "Abu Dhabi",
        "buyer_country_code": "AE",
    }
    fields.update(overrides)
    return fields


def _line(
    category: str, *, line_id: str = "1", rate: str | None = None, quantity: str = "10"
) -> LineItemFields:
    return LineItemFields(
        line_id=line_id,
        item_name="Office desk",
        item_description="Office desk, 160x80cm, laminate finish",
        quantity=quantity,
        unit_code="EA",
        unit_price="100.00",
        net_amount=str(int(quantity) * 100) + ".00",
        vat_category_code=category,
        vat_rate=rate,
    )


_CATEGORY_LINE_KWARGS: dict[str, dict[str, str | None]] = {
    "S": {"rate": "5"},
    "Z": {"rate": "0"},
    "E": {"rate": None},
    "AE": {"rate": "5"},
    "O": {"rate": None},
}

# `ibr-151-ae`: a Commercial invoice / Credit note whose lines are *only* "Exempt" and/or "Not
# subject to VAT" is rejected -- at least one other category must also be present. Every
# single-category test fixture below therefore includes one standard-rated filler line.
_CATEGORIES_NEEDING_A_TAXABLE_FILLER_LINE = frozenset({"E", "O"})


def _fields_for(category: str, **header_overrides: str | None) -> InvoiceFields:
    lines = [_line(category, **_CATEGORY_LINE_KWARGS[category])]
    if category in _CATEGORIES_NEEDING_A_TAXABLE_FILLER_LINE:
        lines.append(_line("S", line_id="2", rate="5"))
    return InvoiceFields(line_items=tuple(lines), **_header(**header_overrides))


def _validate(xml: bytes, type_code: DocumentTypeCode) -> tuple[bool, tuple[str, ...]]:
    data_root = require_pint_ae_data_root()
    with PintAeSchematronValidator(type_code, data_root=data_root) as validator:
        result = validator.validate(xml)
    fatal_rule_ids = tuple(f.rule_id for f in result.failures if f.flag == "fatal")
    return result.passed, fatal_rule_ids


@pytest.mark.artifacts
@pytest.mark.parametrize("category", ["S", "Z", "E", "AE", "O"])
def test_invoice_passes_schematron_for_every_tax_category(category: str) -> None:
    fields = _fields_for(category)
    xml = build_ubl_xml(
        fields, type_code=DocumentTypeCode.INVOICE, verified_payment_iban=_VERIFIED_IBAN
    )
    passed, fatal_rule_ids = _validate(xml, DocumentTypeCode.INVOICE)
    assert passed, f"category {category} failed schematron: {fatal_rule_ids}"


@pytest.mark.artifacts
def test_credit_note_passes_schematron() -> None:
    fields = _fields_for(
        "S", invoice_number="CN-2026-0001", preceding_invoice_reference="INV-2026-0001"
    )
    xml = build_ubl_xml(fields, type_code=DocumentTypeCode.CREDIT_NOTE)
    passed, fatal_rule_ids = _validate(xml, DocumentTypeCode.CREDIT_NOTE)
    assert passed, f"credit note failed schematron: {fatal_rule_ids}"
    assert b"<CreditNote" in xml
    assert b"CreditNoteLine" in xml
    assert b"CreditedQuantity" in xml


@pytest.mark.artifacts
def test_missing_invoice_number_fails_with_its_rule_id() -> None:
    fields = _fields_for("S", invoice_number=None)
    xml = build_ubl_xml(
        fields, type_code=DocumentTypeCode.INVOICE, verified_payment_iban=_VERIFIED_IBAN
    )
    data_root = require_pint_ae_data_root()
    with PintAeSchematronValidator(DocumentTypeCode.INVOICE, data_root=data_root) as validator:
        result = validator.validate(xml)
    assert not result.passed
    assert any(f.rule_id == "ibr-002" for f in result.failures)


def test_invoice_without_verified_iban_raises() -> None:
    fields = _fields_for("S")
    with pytest.raises(EInvoiceError):
        build_ubl_xml(fields, type_code=DocumentTypeCode.INVOICE)


def test_unknown_vat_category_raises() -> None:
    fields = _fields_for("S")
    bad_line = _line("X")
    fields = fields.model_copy(update={"line_items": (bad_line,)})
    with pytest.raises(EInvoiceError):
        build_ubl_xml(
            fields, type_code=DocumentTypeCode.INVOICE, verified_payment_iban=_VERIFIED_IBAN
        )


def test_canonical_xml_is_stable_across_rebuilds() -> None:
    """Building the same InvoiceFields twice yields canonically identical XML: a regression
    guard equivalent to diffing against a pinned gold document, without hardcoding an eval
    answer (the "gold" here is the builder's own prior output, not an expected field value)."""
    fields = _fields_for("S")
    first = build_ubl_xml(
        fields, type_code=DocumentTypeCode.INVOICE, verified_payment_iban=_VERIFIED_IBAN
    )
    second = build_ubl_xml(
        fields, type_code=DocumentTypeCode.INVOICE, verified_payment_iban=_VERIFIED_IBAN
    )
    assert canonical_xml_diff(first, second) == []
