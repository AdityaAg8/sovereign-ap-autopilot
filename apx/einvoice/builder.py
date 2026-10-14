"""Builds PINT AE 1.0.4 UBL 2.1 Invoice/CreditNote XML from `InvoiceFields` plus the
ERP-verified vendor payment IBAN (never the invoice's own claimed IBAN, per P1 "injection-proof
payments"), for VAT categories S/Z/E/AE/O. Presence checks are
deliberately shallow: a header/line field left `None` is simply omitted from the XML, so the
runtime schematron validator (`apx.einvoice.schematron`) is the single source of truth for which
omissions are fatal, and reports the violated rule ID. The money path is the exception: currency
and line VAT rate are never defaulted (ADR-0012), so a missing/non-AED currency or a rate-bearing
line without a rate raises `EInvoiceError`."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from lxml import etree  # type: ignore[import-untyped]
from lxml.builder import ElementMaker  # type: ignore[import-untyped]

from apx.common.errors import EInvoiceError
from apx.common.fields import InvoiceFields
from apx.common.normalize import NormalizationError, normalize_amount, normalize_date
from apx.common.xml import parse_untrusted_xml
from apx.einvoice.interfaces import DocumentTypeCode

_CAC_URI = "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2"
_CBC_URI = "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2"
_INVOICE_URI = "urn:oasis:names:specification:ubl:schema:xsd:Invoice-2"
_CREDIT_NOTE_URI = "urn:oasis:names:specification:ubl:schema:xsd:CreditNote-2"

CAC = ElementMaker(namespace=_CAC_URI)
CBC = ElementMaker(namespace=_CBC_URI)

_CUSTOMIZATION_ID = "urn:peppol:pint:billing-1@ae-1"
"""BT-24 Specification identifier: fixed by the PINT AE profile we always emit into, not
extracted per invoice."""

_PROFILE_ID = "urn:peppol:bis:billing"
"""BT-23 Business process type: fixed, see `_CUSTOMIZATION_ID`."""

_PROFILE_EXECUTION_ID = "00000000"
"""UAE transaction-type code: "no special transaction type" (avoids the free-trade-zone/exports/
deemed-supply conditional rules, e.g. `ibr-007-ae`, that only fire for other bit patterns)."""

_EAS_SCHEME_ID = "0060"
"""D-U-N-S Number (ISO 6523 ICD "0060"), used for every `cbc:EndpointID`. Chosen over UAE's own
"0235" scheme to avoid that scheme's cluster of conditional jurisdiction rules (`ibr-149-ae`,
`ibr-150-ae`, `ibr-173-ae`, `ibr-180/181/183-ae`, `ibr-135-ae`, which only fire when `@schemeID =
"0235"`), and over GLN ("0088") to avoid its GS1 check-digit assertion (`ibr-068`), which would
otherwise require generating a real, checksum-valid GLN instead of an opaque identifier. "0060"
is a valid CEF EAS code (`ibr-cl-25`) with no further format rule in the PINT AE rule set."""

_UUID_NAMESPACE = uuid.UUID("6f6c6465-6e64-4275-696c-646572000000")
"""Fixed namespace for deriving a deterministic `cbc:UUID` (BTAE-07) from a document's own
identity, so the same `InvoiceFields` always builds byte-identical XML (needed for the
canonical-diff regression test) without a random `uuid4()` on every build."""

_CITY_COUNTRY_SUBENTITY: dict[str, str] = {
    "abu dhabi": "AUH",
    "dubai": "DXB",
    "sharjah": "SHJ",
    "ajman": "AJM",
    "umm al quwain": "UAQ",
    "ras al khaimah": "RAK",
    "fujairah": "FUJ",
}
"""UAE emirate (city) name -> the `cbc:CountrySubentity` code PINT AE examples use, for parties
whose `InvoiceFields` carries no dedicated subdivision text (there is no `buyer_country_
subdivision` business term in the 51-field schema; BT-39 only exists for the seller)."""

_TL_SCHEME_AGENCY_ID = "TL"
"""Trade License, one of the UAE-allowed legal-registration-id scheme agencies."""

_VAT_SCHEME_ID = "VAT"
_DEFAULT_UNIT_CODE = "C62"
"""UN/CEFACT Recommendation 20 code for "one" (piece), used when a line has no unit code."""

_CREDIT_TRANSFER_CODE = "30"
"""UNCL4461 payment means code for a bank credit transfer."""

_DOCUMENT_CURRENCY = "AED"
"""The only document currency (BT-5) this builder emits. A non-AED invoice needs the VAT
accounting currency (BT-6), a tax-currency `cac:TaxTotal` and the tax accounting exchange rate
(BTAE-04, `ibr-159-ae`), none of which `InvoiceFields` carries, so it is refused rather than
relabelled as AED (see `apx.validate.interfaces`, which flags it for review for the same
reason)."""

_ZERO = Decimal("0")
_CENTS = Decimal("0.01")

_S, _Z, _E, _AE, _O = "S", "Z", "E", "AE", "O"
_VALID_VAT_CATEGORIES = frozenset({_S, _Z, _E, _AE, _O})
_LINE_RATE_CATEGORIES = frozenset({_S, _Z, _AE})
"""Categories whose line/breakdown `cbc:Percent` MUST be present (`aligned-ibrp-{s,z,ae}-05*`).
`E` and `O` lines/breakdowns MUST NOT carry a rate (`aligned-ibrp-e-05`, `aligned-ibrp-o-05`)."""

_DEFAULT_EXEMPTION_REASON_CODE = "VATEX-AE-25-1"
"""Aligned-TaxExemptionCodes.gc entry: financial services, the most common UAE exempt supply."""
_DEFAULT_EXEMPTION_REASON_TEXT = "Exempt supply"

_AE_DEFAULT_NATURE_CODE = "DL8.48.3.1"
"""BTAE-09 "type of goods or services" for a Reverse charge line; a UAE FTA classification code,
required (`ibr-166-ae`) whenever `ClassifiedTaxCategory/ID` is "AE"."""

_CREDIT_NOTE_DISCREPANCY_RESPONSE_CODE = "DL8.61.1.E"
"""A concrete UAE credit-reason code (CreditReason.gc), used whenever a credit note references a
preceding invoice. `ibr-055-ae` requires `cac:DiscrepancyResponse/cbc:ResponseCode` to be "VD"
precisely when there is *no* `cac:BillingReference` (a standalone credit note); this module uses
"VD" for that case and this code otherwise, matching `preceding_invoice_reference`'s presence."""
_CREDIT_NOTE_VOID_RESPONSE_CODE = "VD"


def _round2(amount: Decimal) -> Decimal:
    return amount.quantize(_CENTS, rounding=ROUND_HALF_UP)


def _fmt(amount: Decimal) -> str:
    return format(amount, "f")


def _text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _amount(value: str | None, *, field: str) -> Decimal | None:
    raw = _text(value)
    if raw is None:
        return None
    try:
        return normalize_amount(raw)
    except NormalizationError as exc:
        raise EInvoiceError(f"{field}: {exc}") from exc


def _parse_date(value: str | None, *, field: str) -> date | None:
    raw = _text(value)
    if raw is None:
        return None
    try:
        return normalize_date(raw)
    except NormalizationError as exc:
        raise EInvoiceError(f"{field}: {exc}") from exc


def _money(amount: Decimal) -> etree._Element:
    return CBC.Amount(_fmt(_round2(amount)), currencyID=_DOCUMENT_CURRENCY)


def _endpoint_id(identifier: str | None, trn: str | None, *, party: str) -> etree._Element:
    """The party's electronic address (seller BT-34 / buyer BT-49), mandatory in PINT: its
    identifier, else its TRN. Never fabricated: with neither, the document cannot be addressed."""
    value = _text(identifier) or _text(trn)
    if value is None:
        bt = "BT-34" if party == "supplier" else "BT-49"
        raise EInvoiceError(
            f"{party}: no identifier or TRN to use as the electronic address ({bt}); "
            "an EndpointID is never fabricated"
        )
    return CBC.EndpointID(value, schemeID=_EAS_SCHEME_ID)


def _country_subentity(city: str | None, explicit: str | None) -> str | None:
    if (given := _text(explicit)) is not None:
        return given
    city_text = _text(city)
    if city_text is not None:
        return _CITY_COUNTRY_SUBENTITY.get(city_text.lower())
    return None


def _postal_address(
    *,
    address_line1: str | None,
    city: str | None,
    postal_zone: str | None,
    subdivision: str | None,
    country: str | None,
) -> etree._Element:
    children = []
    if (line1 := _text(address_line1)) is not None:
        children.append(CBC.StreetName(line1))
    if (city_name := _text(city)) is not None:
        children.append(CBC.CityName(city_name))
    if (zone := _text(postal_zone)) is not None:
        children.append(CBC.PostalZone(zone))
    if (subentity := _country_subentity(city, subdivision)) is not None:
        children.append(CBC.CountrySubentity(subentity))
    children.append(CAC.Country(CBC.IdentificationCode(_text(country) or "AE")))
    return CAC.PostalAddress(*children)


def _party_tax_scheme(trn: str | None) -> etree._Element | None:
    trn_text = _text(trn)
    if trn_text is None:
        return None
    return CAC.PartyTaxScheme(CBC.CompanyID(trn_text), CAC.TaxScheme(CBC.ID(_VAT_SCHEME_ID)))


def _party_legal_entity(
    name: str | None, legal_registration_id: str | None
) -> etree._Element | None:
    registration_name = _text(name)
    if registration_name is None:
        return None
    children = [CBC.RegistrationName(registration_name)]
    if (legal_id := _text(legal_registration_id)) is not None:
        children.append(
            CBC.CompanyID(
                legal_id,
                schemeAgencyID=_TL_SCHEME_AGENCY_ID,
                schemeAgencyName="Trade License issuing Authority",
            )
        )
    return CAC.PartyLegalEntity(*children)


def _party(
    *,
    role: str,
    identifier: str | None,
    name: str | None,
    legal_registration_id: str | None,
    trn: str | None,
    address_line1: str | None,
    city: str | None,
    postal_zone: str | None,
    subdivision: str | None,
    country: str | None,
) -> etree._Element:
    children = [
        _endpoint_id(identifier, trn, party=role),
        _postal_address(
            address_line1=address_line1,
            city=city,
            postal_zone=postal_zone,
            subdivision=subdivision,
            country=country,
        ),
    ]
    if (tax_scheme := _party_tax_scheme(trn)) is not None:
        children.append(tax_scheme)
    if (legal_entity := _party_legal_entity(name, legal_registration_id)) is not None:
        children.append(legal_entity)
    return CAC.Party(*children)


class _LineTax:
    """Resolved per-line VAT presentation, keyed off the line's raw `vat_category_code`."""

    __slots__ = (
        "category",
        "exemption_reason",
        "exemption_reason_code",
        "line_vat_amount",
        "net_amount",
        "rate",
    )

    def __init__(
        self,
        *,
        category: str,
        rate: Decimal | None,
        net_amount: Decimal,
        exemption_reason_code: str | None,
        exemption_reason: str | None,
    ) -> None:
        self.category = category
        self.rate = rate
        self.net_amount = net_amount
        self.exemption_reason_code = exemption_reason_code
        self.exemption_reason = exemption_reason
        if category == _S:
            self.line_vat_amount: Decimal | None = _round2(net_amount * (rate or _ZERO) / 100)
        elif category == _E:
            self.line_vat_amount = None  # ibr-163-ae: must not be present at all
        else:  # Z, AE, O: always zero when present (ibr-165-ae, ibr-162-ae, and O by 0% rate)
            self.line_vat_amount = _ZERO


def _resolve_line_tax(
    *, category_raw: str | None, rate_raw: str | None, net_amount: Decimal, line_ref: str
) -> _LineTax:
    category = (_text(category_raw) or "").upper()
    if category not in _VALID_VAT_CATEGORIES:
        raise EInvoiceError(
            f"{line_ref}: vat_category_code {category_raw!r} is not one of "
            f"{sorted(_VALID_VAT_CATEGORIES)}"
        )
    rate = _amount(rate_raw, field=f"{line_ref}.vat_rate")
    if category in _LINE_RATE_CATEGORIES:
        if rate is None:
            # ADR-0012: a missing rate is never assumed (no 5% fallback); the validator routes
            # it to review as VAT_RATE_MISSING and the reviewer supplies it.
            raise EInvoiceError(
                f"{line_ref}: vat_category_code {category!r} requires a vat_rate (BT-152), "
                "and none was given; a VAT rate is never assumed"
            )
    else:
        # E, O: the line MUST NOT carry a rate at all. A stated non-zero rate contradicts the
        # category (the validator's VAT_CATEGORY_RATE_MISMATCH); dropping it would silently turn
        # a VAT-bearing line into a 0-VAT one, so it is refused instead.
        if rate is not None and rate != _ZERO:
            raise EInvoiceError(
                f"{line_ref}: vat_category_code {category!r} carries no VAT, but vat_rate is "
                f"{rate}; the category and rate disagree"
            )
        rate = None

    exemption_code = exemption_text = None
    if category == _E:
        exemption_code = _DEFAULT_EXEMPTION_REASON_CODE
        exemption_text = _DEFAULT_EXEMPTION_REASON_TEXT

    return _LineTax(
        category=category,
        rate=rate,
        net_amount=net_amount,
        exemption_reason_code=exemption_code,
        exemption_reason=exemption_text,
    )


def _classified_tax_category(tax: _LineTax) -> etree._Element:
    children = [CBC.ID(tax.category)]
    if tax.rate is not None:
        children.append(CBC.Percent(_fmt(tax.rate)))
    if tax.exemption_reason_code is not None:
        children.append(CBC.TaxExemptionReasonCode(tax.exemption_reason_code))
    if tax.exemption_reason is not None:
        children.append(CBC.TaxExemptionReason(tax.exemption_reason))
    children.append(CAC.TaxScheme(CBC.ID(_VAT_SCHEME_ID)))
    return CAC.ClassifiedTaxCategory(*children)


def _item_price_extension(tax: _LineTax) -> etree._Element:
    # ibr-194-ae: BTAE-10 (line amount incl. VAT, in AED) is always mandatory.
    gross = tax.net_amount + (tax.line_vat_amount or _ZERO)
    children: list[etree._Element] = [_money(gross)]
    if tax.line_vat_amount is not None:  # ibr-163-ae: omit entirely for Exempt lines
        children.append(
            CAC.TaxTotal(
                CBC.TaxAmount(_fmt(_round2(tax.line_vat_amount)), currencyID=_DOCUMENT_CURRENCY)
            )
        )
    return CAC.ItemPriceExtension(*children)


def _invoice_line(
    *,
    line_tag: str,
    quantity_tag: str,
    line_id: str,
    quantity: Decimal,
    unit_code: str | None,
    description: str | None,
    name: str | None,
    tax: _LineTax,
    unit_price: Decimal,
    price_base_quantity: Decimal,
) -> etree._Element:
    item_children = []
    if (desc := _text(description)) is not None:
        item_children.append(CBC.Description(desc))
    item_children.append(CBC.Name(_text(name) or _text(description) or "Item"))
    if tax.category == _AE:
        # ibr-166-ae/ibr-174-ae: a Reverse charge line MUST identify what is being supplied,
        # via a standard item identifier and a "nature of goods/services" code (BTAE-09).
        item_children.append(
            CAC.StandardItemIdentification(CBC.ID(f"SKU-{line_id}", schemeID="0160"))
        )
        item_children.append(CAC.CommodityClassification(CBC.NatureCode(_AE_DEFAULT_NATURE_CODE)))
    item_children.append(_classified_tax_category(tax))

    line = getattr(CAC, line_tag)
    quantity_el = getattr(CBC, quantity_tag)
    return line(
        CBC.ID(line_id),
        quantity_el(_fmt(quantity), unitCode=unit_code or _DEFAULT_UNIT_CODE),
        CBC.LineExtensionAmount(_fmt(_round2(tax.net_amount)), currencyID=_DOCUMENT_CURRENCY),
        CAC.Item(*item_children),
        CAC.Price(
            CBC.PriceAmount(_fmt(unit_price), currencyID=_DOCUMENT_CURRENCY),
            CBC.BaseQuantity(_fmt(price_base_quantity), unitCode=unit_code or _DEFAULT_UNIT_CODE),
            # ibr-126-ae: Item price base quantity (IBT-149) and Item Gross Price (IBT-148, the
            # AllowanceCharge/BaseAmount below) MUST both be there; ChargeIndicator=false and
            # Amount=0 mean "no line-level price discount" rather than an actual allowance.
            CAC.AllowanceCharge(
                CBC.ChargeIndicator("false"),
                CBC.Amount(_fmt(_ZERO), currencyID=_DOCUMENT_CURRENCY),
                CBC.BaseAmount(_fmt(unit_price), currencyID=_DOCUMENT_CURRENCY),
            ),
        ),
        _item_price_extension(tax),
    )


def _group_tax_amount(category: str, rate: Decimal | None, taxable_amount: Decimal) -> Decimal:
    """The VAT category tax amount (IBT-117) for one (category, rate) breakdown group. Only `S`
    actually collects VAT; every other category's is 0 by construction
    (`aligned-ibrp-{z,e,ae,o}-09*`)."""
    return _round2(taxable_amount * rate / 100) if category == _S and rate else _ZERO


def _tax_subtotal(
    category: str, rate: Decimal | None, taxable_amount: Decimal, tax_amount: Decimal
) -> etree._Element:
    """One `cac:TaxSubtotal` (IBG-23) for one (category, rate) group."""
    category_children = [CBC.ID(category)]
    if rate is not None:
        category_children.append(CBC.Percent(_fmt(rate)))
    category_children.append(CAC.TaxScheme(CBC.ID(_VAT_SCHEME_ID)))
    return CAC.TaxSubtotal(
        CBC.TaxableAmount(_fmt(_round2(taxable_amount)), currencyID=_DOCUMENT_CURRENCY),
        CBC.TaxAmount(_fmt(tax_amount), currencyID=_DOCUMENT_CURRENCY),
        CAC.TaxCategory(*category_children),
    )


def _payment_means(iban: str | None) -> etree._Element | None:
    if iban is None:
        return None
    return CAC.PaymentMeans(
        CBC.PaymentMeansCode(_CREDIT_TRANSFER_CODE),
        CAC.PayeeFinancialAccount(CBC.ID(iban)),
    )


def _require_document_currency(currency_raw: str | None) -> None:
    """Refuse, typed, any currency other than AED: never default a missing one to AED and never
    label a foreign-currency invoice's amounts as AED (ADR-0012; "XXX" is ISO 4217 "no
    currency", what the extraction wiring emits for a missing one)."""
    currency = (_text(currency_raw) or "").upper()
    if currency in ("", "XXX"):
        raise EInvoiceError(
            "currency_code (BT-5) is missing; the document currency is never assumed to be AED"
        )
    if currency != _DOCUMENT_CURRENCY:
        raise EInvoiceError(
            f"currency_code {currency_raw!r} is not AED; a non-AED PINT AE invoice needs a VAT "
            "accounting currency and tax exchange rate (BT-6, BTAE-04, ibr-159-ae) that "
            "InvoiceFields does not carry, so it cannot be built"
        )


def build_ubl_xml(
    fields: InvoiceFields,
    *,
    type_code: DocumentTypeCode,
    verified_payment_iban: str | None = None,
) -> bytes:
    """Build PINT AE 1.0.4 UBL 2.1 XML (`Invoice` or `CreditNote`) from an
    `apx.common.fields.InvoiceFields` (`fields`) plus the ERP vendor master's admin-verified
    IBAN (`verified_payment_iban`) -- never `fields.payment_account_iban`, which is untrusted,
    attacker-reachable text (P1). Field-level omissions in `fields` are not pre-checked here:
    an absent header/line value is simply left out of the XML and the runtime schematron
    validator (`apx.einvoice.schematron`) reports the specific violated rule ID.

    Raises `EInvoiceError` when data cannot be structurally represented at all: a missing or
    non-AED `currency_code` (see `_DOCUMENT_CURRENCY`), an unrecognised or missing
    `vat_category_code`, a missing `vat_rate` on an S/Z/AE line or a non-zero one on an E/O
    line, a supplier or buyer with neither identifier nor TRN (no EndpointID), an unparsable
    amount/date, a zero quantity with no unit price, or (for an Invoice document, per PINT AE
    `ibr-191-ae`/`ibr-192-ae`) a missing `verified_payment_iban`, since an Invoice without a
    payment account would otherwise silently fail the money path further downstream."""
    _require_document_currency(fields.currency_code)
    if type_code.is_invoice and verified_payment_iban is None:
        raise EInvoiceError(
            "an Invoice requires a verified_payment_iban (ERP vendor master, never the "
            "invoice's own claimed IBAN); pass one from the vendor's admin-verified bank account"
        )

    lines = fields.line_items
    if not lines:
        raise EInvoiceError("an invoice/credit note must have at least one line (IBG-25)")

    line_taxes: list[_LineTax] = []
    line_elements: list[etree._Element] = []
    line_tag = "InvoiceLine" if type_code.is_invoice else "CreditNoteLine"
    quantity_tag = "InvoicedQuantity" if type_code.is_invoice else "CreditedQuantity"

    for i, line in enumerate(lines, start=1):
        line_ref = f"line_items[{i}]"
        net_amount = _amount(line.net_amount, field=f"{line_ref}.net_amount")
        if net_amount is None:
            raise EInvoiceError(f"{line_ref}.net_amount is required to build a line")
        quantity = _amount(line.quantity, field=f"{line_ref}.quantity")
        if quantity is None:
            quantity = Decimal(1)
        unit_price = _amount(line.unit_price, field=f"{line_ref}.unit_price")
        if unit_price is None:
            if quantity == _ZERO:
                raise EInvoiceError(
                    f"{line_ref}: unit_price is missing and quantity is 0, so it cannot be "
                    "derived from net_amount"
                )
            unit_price = net_amount / quantity
        price_base_quantity = _amount(line.price_base_quantity, field=f"{line_ref}.base_qty")
        if price_base_quantity is None:
            price_base_quantity = Decimal(1)

        tax = _resolve_line_tax(
            category_raw=line.vat_category_code,
            rate_raw=line.vat_rate,
            net_amount=net_amount,
            line_ref=line_ref,
        )
        line_taxes.append(tax)
        line_elements.append(
            _invoice_line(
                line_tag=line_tag,
                quantity_tag=quantity_tag,
                line_id=_text(line.line_id) or str(i),
                quantity=quantity,
                unit_code=_text(line.unit_code),
                description=line.item_description,
                name=line.item_name,
                tax=tax,
                unit_price=unit_price,
                price_base_quantity=price_base_quantity,
            )
        )

    groups: dict[tuple[str, str], Decimal] = {}
    group_rate: dict[tuple[str, str], Decimal | None] = {}
    for tax in line_taxes:
        rate_key = _fmt(tax.rate) if tax.rate is not None else ""
        key = (tax.category, rate_key)
        groups[key] = groups.get(key, _ZERO) + tax.net_amount
        group_rate[key] = tax.rate
    tax_subtotals = []
    subtotal_tax_amounts = []
    for (category, rate_key), taxable_amount in groups.items():
        rate = group_rate[(category, rate_key)]
        tax_amount = _group_tax_amount(category, rate, taxable_amount)
        subtotal_tax_amounts.append(tax_amount)
        tax_subtotals.append(_tax_subtotal(category, rate, taxable_amount, tax_amount))
    vat_total = sum(subtotal_tax_amounts, _ZERO)

    line_extension_amount = sum((_round2(t.net_amount) for t in line_taxes), _ZERO)
    tax_exclusive_amount = line_extension_amount
    tax_inclusive_amount = _round2(tax_exclusive_amount + vat_total)
    payable_amount = tax_inclusive_amount

    issue_date = _parse_date(fields.issue_date, field="issue_date")
    due_date = _parse_date(fields.due_date, field="due_date")

    supplier_party = _party(
        role="supplier",
        identifier=fields.supplier_identifier,
        name=fields.supplier_name,
        legal_registration_id=fields.supplier_legal_registration_id,
        trn=fields.supplier_trn,
        address_line1=fields.supplier_address_line1,
        city=fields.supplier_city,
        postal_zone=fields.supplier_post_code,
        subdivision=fields.supplier_country_subdivision,
        country=fields.supplier_country_code,
    )
    buyer_party = _party(
        role="buyer",
        identifier=fields.buyer_identifier,
        name=fields.buyer_name,
        legal_registration_id=fields.buyer_legal_registration_id,
        trn=fields.buyer_trn,
        address_line1=fields.buyer_address_line1,
        city=fields.buyer_city,
        postal_zone=fields.buyer_post_code,
        subdivision=None,
        country=fields.buyer_country_code,
    )

    header: list[etree._Element] = [
        CBC.CustomizationID(_CUSTOMIZATION_ID),
        CBC.ProfileID(_PROFILE_ID),
        CBC.ProfileExecutionID(_PROFILE_EXECUTION_ID),
    ]
    if (number := _text(fields.invoice_number)) is not None:
        header.append(CBC.ID(number))
    document_uuid = uuid.uuid5(
        _UUID_NAMESPACE,
        f"{type_code.value}:{fields.invoice_number or ''}:{fields.issue_date or ''}",
    )
    header.append(CBC.UUID(str(document_uuid)))  # ibr-193-ae: BTAE-07, always mandatory
    if issue_date is not None:
        header.append(CBC.IssueDate(issue_date.isoformat()))
    preceding = _text(fields.preceding_invoice_reference)
    if type_code.is_invoice:
        if due_date is not None:
            header.append(CBC.DueDate(due_date.isoformat()))
        header.append(CBC.InvoiceTypeCode(type_code.value))
    else:
        header.append(CBC.CreditNoteTypeCode(type_code.value))
        # ibr-158-ae: a Credit note always needs a reason code; ibr-055-ae: that code MUST be
        # "VD" (void) exactly when there is no preceding-invoice BillingReference, and MUST NOT
        # be "VD" when there is one.
        response_code = (
            _CREDIT_NOTE_DISCREPANCY_RESPONSE_CODE
            if preceding is not None
            else _CREDIT_NOTE_VOID_RESPONSE_CODE
        )
        header.append(CAC.DiscrepancyResponse(CBC.ResponseCode(response_code)))
    if (note := _text(fields.invoice_note)) is not None:
        header.append(CBC.Note(note))
    header.append(CBC.DocumentCurrencyCode(_DOCUMENT_CURRENCY))
    if (buyer_ref := _text(fields.buyer_reference)) is not None:
        header.append(CBC.BuyerReference(buyer_ref))
    if preceding is not None:
        header.append(CAC.BillingReference(CAC.InvoiceDocumentReference(CBC.ID(preceding))))

    header.append(CAC.AccountingSupplierParty(supplier_party))
    header.append(CAC.AccountingCustomerParty(buyer_party))

    payment_means = _payment_means(verified_payment_iban)
    if payment_means is not None:
        header.append(payment_means)
    if (terms := _text(fields.payment_terms)) is not None:
        header.append(CAC.PaymentTerms(CBC.Note(terms)))

    header.append(
        CAC.TaxTotal(
            CBC.TaxAmount(_fmt(_round2(vat_total)), currencyID=_DOCUMENT_CURRENCY),
            *tax_subtotals,
        )
    )
    header.append(
        CAC.LegalMonetaryTotal(
            CBC.LineExtensionAmount(_fmt(line_extension_amount), currencyID=_DOCUMENT_CURRENCY),
            CBC.TaxExclusiveAmount(_fmt(tax_exclusive_amount), currencyID=_DOCUMENT_CURRENCY),
            CBC.TaxInclusiveAmount(_fmt(tax_inclusive_amount), currencyID=_DOCUMENT_CURRENCY),
            CBC.PayableAmount(_fmt(payable_amount), currencyID=_DOCUMENT_CURRENCY),
        )
    )
    header.extend(line_elements)

    root_uri = _INVOICE_URI if type_code.is_invoice else _CREDIT_NOTE_URI
    root_tag = "Invoice" if type_code.is_invoice else "CreditNote"
    nsmap = {None: root_uri, "cac": _CAC_URI, "cbc": _CBC_URI}
    root_maker = ElementMaker(namespace=root_uri, nsmap=nsmap)
    root = getattr(root_maker, root_tag)(*header)
    xml: bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return xml


def canonical_xml_diff(a: bytes, b: bytes) -> list[str]:
    """Canonicalise (C14N 1.1) `a` and `b` and return the unified-diff lines between them
    (empty when they are canonically identical). Used to compare a freshly built document
    against a pinned "gold" document irrespective of attribute order or insignificant
    whitespace."""
    import difflib

    def _c14n_lines(xml: bytes) -> list[str]:
        tree = parse_untrusted_xml(xml)
        buf: bytes = etree.tostring(tree, method="c14n2")
        lines: list[str] = buf.decode("utf-8").splitlines(keepends=True)
        return lines

    return list(
        difflib.unified_diff(_c14n_lines(a), _c14n_lines(b), fromfile="built", tofile="gold")
    )
