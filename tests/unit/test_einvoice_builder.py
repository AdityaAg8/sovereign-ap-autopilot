"""Unit tests for `apx.einvoice.builder`: PINT AE UBL 2.1 Invoice/CreditNote XML built from
`InvoiceFields` plus a `factory.erp`-generated vendor's admin-verified IBAN passes the runtime
schematron 100% for every VAT category (S, Z, E, AE, O) and for a credit note; a missing
mandatory element fails with its rule ID (all three marked `artifacts`, they need the real
pinned PINT AE data). The builder's own input-validation errors (no verified IBAN, unknown VAT
category, missing or non-AED currency, missing VAT rate or category) and the canonical-diff
stability check need no schematron run at all, so they stay unmarked and run in the default
suite."""

from __future__ import annotations

from decimal import Decimal

import pytest
from lxml import etree  # type: ignore[import-untyped]

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
_CAC = "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2"
_CBC = "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2"


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


# ADR-0012, "no silent fallbacks": currency and line VAT rate are never defaulted.


def _build_invoice(fields: InvoiceFields) -> bytes:
    return build_ubl_xml(
        fields, type_code=DocumentTypeCode.INVOICE, verified_payment_iban=_VERIFIED_IBAN
    )


@pytest.mark.parametrize("currency", [None, "", "  ", "XXX"])
def test_missing_currency_raises_never_defaults_to_aed(currency: str | None) -> None:
    with pytest.raises(EInvoiceError, match="never assumed to be AED"):
        _build_invoice(_fields_for("S", currency_code=currency))


@pytest.mark.parametrize("currency", ["USD", "EUR", "SAR"])
def test_non_aed_currency_raises_never_relabelled_as_aed(currency: str) -> None:
    with pytest.raises(EInvoiceError, match="is not AED"):
        _build_invoice(_fields_for("S", currency_code=currency))


def test_non_aed_credit_note_also_raises() -> None:
    fields = _fields_for("S", currency_code="USD")
    with pytest.raises(EInvoiceError, match="is not AED"):
        build_ubl_xml(fields, type_code=DocumentTypeCode.CREDIT_NOTE)


def test_document_currency_and_every_amount_are_aed() -> None:
    root = etree.fromstring(_build_invoice(_fields_for("S")))
    assert root.findtext(f"{{{_CBC}}}DocumentCurrencyCode") == "AED"
    currency_ids = {el.get("currencyID") for el in root.iter() if el.get("currencyID")}
    assert currency_ids == {"AED"}


@pytest.mark.parametrize("category", ["S", "Z", "AE"])
def test_rate_bearing_category_without_rate_raises_never_defaults_to_5(category: str) -> None:
    fields = InvoiceFields(line_items=(_line(category, rate=None),), **_header())
    with pytest.raises(EInvoiceError, match=r"line_items\[1\]: .*requires a vat_rate"):
        _build_invoice(fields)


@pytest.mark.parametrize("category", [None, "", " "])
@pytest.mark.parametrize("rate", [None, "5"])
def test_line_without_category_raises(category: str | None, rate: str | None) -> None:
    line = _line("S", rate=rate).model_copy(update={"vat_category_code": category})
    fields = InvoiceFields(line_items=(line,), **_header())
    with pytest.raises(EInvoiceError, match="vat_category_code"):
        _build_invoice(fields)


@pytest.mark.parametrize(
    ("category", "rate", "expected_percent", "expected_line_vat"),
    [
        ("S", "5", "5", "50.00"),
        ("Z", "0", "0", "0.00"),
        ("AE", "5", "5", "0.00"),
        ("AE", "0", "0", "0.00"),
        ("E", None, None, None),  # ibr-163-ae: no line VAT amount at all for Exempt
        ("O", None, None, "0.00"),
        ("E", "0", None, None),
        ("O", "0", None, "0.00"),
    ],
)
def test_category_rate_mapping_carries_only_the_stated_rate(
    category: str, rate: str | None, expected_percent: str | None, expected_line_vat: str | None
) -> None:
    """S/Z/AE carry exactly the stated rate (no statutory fill-in); E/O never carry one
    (`aligned-ibrp-e-05`, `aligned-ibrp-o-05`); only S collects VAT (ADR-0012 amendment). The
    expected values restate those PINT AE rules, not builder output."""
    fields = InvoiceFields(line_items=(_line(category, rate=rate, quantity="10"),), **_header())
    root = etree.fromstring(_build_invoice(fields))
    line = root.find(f"{{{_CAC}}}InvoiceLine")
    assert line is not None
    classified = line.find(f"{{{_CAC}}}Item/{{{_CAC}}}ClassifiedTaxCategory")
    assert classified is not None
    assert classified.findtext(f"{{{_CBC}}}ID") == category
    assert classified.findtext(f"{{{_CBC}}}Percent") == expected_percent
    line_vat = line.findtext(f"{{{_CAC}}}ItemPriceExtension/{{{_CAC}}}TaxTotal/{{{_CBC}}}TaxAmount")
    assert line_vat == expected_line_vat


@pytest.mark.parametrize("category", ["E", "O"])
@pytest.mark.parametrize("rate", ["5", "0.01"])
def test_no_vat_category_with_non_zero_rate_raises_never_drops_the_vat(
    category: str, rate: str
) -> None:
    fields = InvoiceFields(line_items=(_line(category, rate=rate),), **_header())
    with pytest.raises(EInvoiceError, match="category and rate disagree"):
        _build_invoice(fields)


@pytest.mark.parametrize("party", ["supplier", "buyer"])
def test_party_without_identifier_or_trn_raises_never_fabricates_an_endpoint(party: str) -> None:
    fields = _fields_for("S", **{f"{party}_identifier": None, f"{party}_trn": None})
    with pytest.raises(EInvoiceError, match=f"{party}: .*never fabricated"):
        _build_invoice(fields)


@pytest.mark.parametrize("party", ["supplier", "buyer"])
def test_endpoint_falls_back_to_the_parties_own_trn(party: str) -> None:
    fields = _fields_for("S", **{f"{party}_identifier": None})
    root = etree.fromstring(_build_invoice(fields))
    tag = "AccountingSupplierParty" if party == "supplier" else "AccountingCustomerParty"
    endpoint = root.findtext(f"{{{_CAC}}}{tag}/{{{_CAC}}}Party/{{{_CBC}}}EndpointID")
    assert endpoint == (_SUPPLIER_TRN if party == "supplier" else _BUYER_TRN)


def _first_line(xml: bytes) -> etree._Element:
    line = etree.fromstring(xml).find(f"{{{_CAC}}}InvoiceLine")
    assert line is not None
    return line


def test_explicit_zero_quantity_is_kept_not_turned_into_one() -> None:
    line = _line("S", rate="5").model_copy(update={"quantity": "0", "net_amount": "0.00"})
    xml = _build_invoice(InvoiceFields(line_items=(line,), **_header()))
    assert _first_line(xml).findtext(f"{{{_CBC}}}InvoicedQuantity") == "0"


def test_zero_quantity_without_unit_price_raises() -> None:
    line = _line("S", rate="5").model_copy(update={"quantity": "0", "unit_price": None})
    with pytest.raises(EInvoiceError, match="quantity is 0"):
        _build_invoice(InvoiceFields(line_items=(line,), **_header()))


def test_missing_unit_price_is_net_amount_over_quantity() -> None:
    line = _line("S", rate="5", quantity="4").model_copy(update={"unit_price": None})
    xml = _build_invoice(InvoiceFields(line_items=(line,), **_header()))
    price = _first_line(xml).findtext(f"{{{_CAC}}}Price/{{{_CBC}}}PriceAmount")
    assert price is not None
    assert Decimal(price) == Decimal("100")
