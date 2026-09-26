"""ADR-0012 amendment 4 (M5.6 e2e findings): a supplier address naming no UAE emirate is a
validation error the reviewer can correct, and the correction reaches `apply_review` and the
UBL; amounts that do not add up (`totals_mismatch`) are recorded with the routing decision so
the API refuses the approval, and `apply_review` refuses them too."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from lxml import etree  # type: ignore[import-untyped]
from pydantic import JsonValue

from apx.api.service import corrections_for_rows, refuse_totals_mismatch
from apx.common.invoice import Invoice
from apx.common.types import Money
from apx.db.enums import FieldSource
from apx.db.models import Field as FieldRow
from apx.einvoice.interfaces import DocumentTypeCode
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.interfaces import (
    PolicyRule,
    PolicyThresholds,
    RoutingInput,
    ValidationFinding,
    routing_record,
)
from apx.validate.address import UAE_EMIRATES, resolve_emirate
from apx.validate.interfaces import (
    SUPPLIER_ADDRESS_VIOLATION_FIELD,
    ValidationReport,
    Violation,
    ViolationCode,
    validate_invoice,
)
from apx.workflows.components import PintAeUblBuilder
from apx.workflows.corrections import (
    SUPPLIER_ADDRESS_FIELD,
    MissingCorrectionsError,
    ReviewCorrections,
    TotalsMismatchError,
    apply_corrections,
    corrections_from_fields,
)
from tests.unit._policy_factories import (
    make_fraud_report,
    make_scan_report,
    make_trust_decision,
    make_vendor,
    money,
)
from tests.unit.test_review_corrections_workflow import (
    CORRECTIONS,
    _Audit,
    _real,
    _Records,
    _Rules,
    _unresolved_invoice,
)

_NS = {
    "cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
}

OCR_ADDRESS = "Warehouse 4, Ras Al Khalaimah, United Arab Emirates"
"""The e2e run's OCR reading of fbf2cc1d12b3's supplier address (one letter off)."""
FIXED_ADDRESS = "Warehouse 4, Al Nakheel, Ras Al Khaimah"
SUPPLIER_TRN = "100000000000003"
BUYER_TRN = "100000000000103"


def _with_address(address: str | None) -> Invoice:
    invoice = _unresolved_invoice()
    return invoice.model_copy(
        update={"supplier": invoice.supplier.model_copy(update={"address": address})}
    )


def _address_errors(invoice: Invoice) -> list[str]:
    report = validate_invoice(invoice, today=date(2026, 9, 26), buyer_trn=BUYER_TRN)
    return [
        v.code.value
        for v in report.violations
        if v.field == SUPPLIER_ADDRESS_VIOLATION_FIELD and v.severity == "error"
    ]


# --- 1. one emirate lookup, shared by the validator and the UBL builder ----------------------


@pytest.mark.parametrize("emirate", UAE_EMIRATES)
def test_every_emirate_resolves_case_insensitively(emirate: str) -> None:
    assert resolve_emirate(f"Office 12, {emirate.upper()}, UAE") == emirate


@pytest.mark.parametrize("address", [None, "", "   ", OCR_ADDRESS, "Muscat, Oman"])
def test_an_address_naming_no_emirate_resolves_to_none(address: str | None) -> None:
    assert resolve_emirate(address) is None


@pytest.mark.parametrize("address", [OCR_ADDRESS, FIXED_ADDRESS, "Plot 7, Umm Al Quwain", None])
def test_the_validator_flags_exactly_what_the_ubl_builder_cannot_place(
    address: str | None,
) -> None:
    """One lookup: the seller subdivision (ibr-143-ae) is missing from the UBL exactly when
    the validator reports the supplier address."""
    invoice = apply_corrections(_with_address(address), CORRECTIONS)
    builder = PintAeUblBuilder(buyer_trn=BUYER_TRN, buyer_trade_license="TL-1")
    xml = builder.build(invoice, DocumentTypeCode.CREDIT_NOTE, verified_payment_iban=None).xml
    root = etree.fromstring(xml)
    seller = root.find(".//cac:AccountingSupplierParty//cac:PostalAddress", _NS)
    assert seller is not None
    placed = seller.find("cbc:CountrySubentity", _NS) is not None
    assert placed is not bool(_address_errors(invoice))
    assert placed is (resolve_emirate(address) is not None)


@pytest.mark.parametrize("address", [None, "", OCR_ADDRESS])
def test_validator_flags_a_supplier_address_with_no_emirate(address: str | None) -> None:
    assert _address_errors(_with_address(address)) == [ViolationCode.MISSING_MANDATORY.value]


def test_validator_accepts_a_supplier_address_naming_an_emirate() -> None:
    assert _address_errors(_with_address(FIXED_ADDRESS)) == []


# --- 2. the reviewer's address fix travels in the corrections --------------------------------


def _fields(address: str | None) -> dict[str, str | None]:
    return {
        "invoice_type_code": "380",
        "currency": "AED",
        "line_items[1].vat_category": "Z",
        "line_items[1].vat_rate": "0",
        SUPPLIER_ADDRESS_FIELD: address,
    }


def _extracted(address: str | None) -> dict[str, str | None]:
    return {"supplier_trn": SUPPLIER_TRN, "currency": "AED", SUPPLIER_ADDRESS_FIELD: address}


@pytest.mark.parametrize("address", [None, OCR_ADDRESS])
def test_approval_is_refused_until_the_supplier_address_names_an_emirate(
    address: str | None,
) -> None:
    with pytest.raises(MissingCorrectionsError) as err:
        corrections_from_fields(_fields(address), extracted=_extracted(address))
    message = str(err.value)
    assert "no UAE emirate" in message and "ibr-143-ae" in message
    # The web client highlights the fields listed after this prefix, so the name stands alone.
    assert message.endswith("correct these fields first: " + SUPPLIER_ADDRESS_FIELD)


def test_a_reviewer_address_is_carried_and_an_unchanged_one_is_not() -> None:
    fixed = corrections_from_fields(_fields(FIXED_ADDRESS), extracted=_extracted(OCR_ADDRESS))
    assert fixed.supplier_address == FIXED_ADDRESS
    same = corrections_from_fields(_fields(FIXED_ADDRESS), extracted=_extracted(FIXED_ADDRESS))
    assert same.supplier_address is None


def _row(
    row_id: int, name: str, raw: str | None, normalized: str | None, source: FieldSource
) -> FieldRow:
    return FieldRow(
        id=row_id,
        invoice_id="inv",
        name=name,
        value=raw,
        normalized_value=normalized,
        source=source,
    )


def _model_rows(address_raw: str, address_normalized: str | None) -> list[FieldRow]:
    base = [
        (name, value, value)
        for name, value in _fields(None).items()
        if name != SUPPLIER_ADDRESS_FIELD
    ]
    base += [("supplier_trn", SUPPLIER_TRN, SUPPLIER_TRN)]
    base += [(SUPPLIER_ADDRESS_FIELD, address_raw, address_normalized)]
    return [
        _row(i, name, raw, norm, FieldSource.EXTRACTOR_A)
        for i, (name, raw, norm) in enumerate(base, start=1)
    ]


def test_corrections_for_rows_carries_the_reviewer_address_edit() -> None:
    rows = _model_rows(OCR_ADDRESS, OCR_ADDRESS)
    with pytest.raises(MissingCorrectionsError, match=SUPPLIER_ADDRESS_FIELD):
        corrections_for_rows(rows)
    edit = _row(99, SUPPLIER_ADDRESS_FIELD, FIXED_ADDRESS, FIXED_ADDRESS, FieldSource.HUMAN)
    later_model = _row(100, SUPPLIER_ADDRESS_FIELD, OCR_ADDRESS, None, FieldSource.EXTRACTOR_A)
    assert corrections_for_rows([*rows, edit, later_model]).supplier_address == FIXED_ADDRESS


def test_corrections_for_rows_reads_the_raw_address_the_invoice_was_built_from() -> None:
    """`GroundedInvoiceBuilder` reads the raw address, so a differing normalized value on the
    model's own row is not a reviewer correction."""
    rows = _model_rows("Al Quoz,  Dubai ", "Al Quoz, Dubai")
    assert corrections_for_rows(rows).supplier_address is None


def test_old_review_signals_without_an_address_still_deserialize() -> None:
    payload = CORRECTIONS.model_dump(mode="json")
    payload.pop("supplier_address")
    assert ReviewCorrections.model_validate(payload).supplier_address is None


def test_apply_corrections_sets_the_supplier_address_only_when_given() -> None:
    invoice = _with_address(OCR_ADDRESS)
    assert apply_corrections(invoice, CORRECTIONS).supplier.address == OCR_ADDRESS
    fixed = CORRECTIONS.model_copy(update={"supplier_address": FIXED_ADDRESS})
    corrected = apply_corrections(invoice, fixed)
    assert corrected.supplier.address == FIXED_ADDRESS
    assert corrected.supplier.trn == invoice.supplier.trn


# --- 3. apply_review re-checks after the corrections -----------------------------------------


async def test_apply_review_passes_a_corrected_address_into_the_ubl() -> None:
    real = _real(_Audit(), _Records(), _Rules(), _Rules())
    fixed = CORRECTIONS.model_copy(update={"supplier_address": FIXED_ADDRESS})
    corrected = await real.apply_review(_with_address(OCR_ADDRESS), fixed)
    assert corrected.supplier.address == FIXED_ADDRESS
    builder = PintAeUblBuilder(buyer_trn=BUYER_TRN, buyer_trade_license="TL-1")
    xml = builder.build(
        corrected, DocumentTypeCode.CREDIT_NOTE, verified_payment_iban=None
    ).xml.decode()
    assert "<cbc:CityName>Ras Al Khaimah</cbc:CityName>" in xml
    assert "<cbc:CountrySubentity>RAK</cbc:CountrySubentity>" in xml


async def test_apply_review_still_refuses_an_uncorrected_address() -> None:
    real = _real(_Audit(), _Records(), _Rules(), _Rules())
    with pytest.raises(MissingCorrectionsError, match="missing_mandatory"):
        await real.apply_review(_with_address(OCR_ADDRESS), CORRECTIONS)


async def test_apply_review_refuses_amounts_that_do_not_add_up() -> None:
    invoice = _unresolved_invoice()
    off = Money(amount=Decimal("999.00"), currency="AED")
    invoice = invoice.model_copy(update={"subtotal": off, "total": off})
    real = _real(_Audit(), _Records(), _Rules(), _Rules())
    with pytest.raises(MissingCorrectionsError, match="totals_mismatch"):
        await real.apply_review(invoice, CORRECTIONS)


# --- 4. totals_mismatch is recorded at routing and refuses the approval ----------------------


def _route_reasons(
    invoice: Invoice, report: ValidationReport | None = None
) -> list[dict[str, JsonValue]]:
    if report is None:
        report = validate_invoice(invoice, today=date(2026, 9, 26), buyer_trn=BUYER_TRN)
    engine = DefaultPolicyEngine(
        PolicyThresholds(
            approval_limit=money("5000"), fraud_review=0.5, scan_review=0.5, attack_block=0.9
        )
    )
    decision = engine.route(
        RoutingInput(
            invoice=invoice,
            vendor=make_vendor(),
            trust=make_trust_decision(),
            validation=report,
            fraud=make_fraud_report(),
            scan=make_scan_report(),
            at=datetime(2026, 9, 26, tzinfo=UTC),
        )
    )
    # the JSON the `policy_route` decision row stores and the API reads back
    return [r for r in routing_record(decision) if isinstance(r, dict)]


def _mismatched_invoice() -> Invoice:
    invoice = _with_address(FIXED_ADDRESS)
    line = invoice.lines[0].model_copy(
        update={"net_amount": Money(amount=Decimal("990.00"), currency="AED")}
    )
    return invoice.model_copy(update={"lines": (line,)})


def test_the_validation_reason_records_its_error_findings() -> None:
    reasons = _route_reasons(_with_address(OCR_ADDRESS))
    validation = [r for r in reasons if r.get("rule") == PolicyRule.VALIDATION.value]
    assert len(validation) == 1
    findings = validation[0]["violations"]
    assert isinstance(findings, list)
    assert {"code": "missing_mandatory", "field": "supplier.address"} in findings
    assert "missing_mandatory (supplier.address)" in str(validation[0]["message"])


def test_approval_is_refused_while_totals_do_not_add_up() -> None:
    reasons = _route_reasons(_mismatched_invoice())
    with pytest.raises(TotalsMismatchError, match=r"lines\[0\]\.net_amount") as err:
        refuse_totals_mismatch(reasons)
    assert "reject the invoice" in str(err.value)
    assert "correct these fields first" not in str(err.value)
    assert issubclass(TotalsMismatchError, MissingCorrectionsError)  # the API's 422


def test_other_validation_errors_and_legacy_decisions_do_not_trigger_the_totals_refusal() -> None:
    refuse_totals_mismatch(_route_reasons(_with_address(OCR_ADDRESS)))
    refuse_totals_mismatch(None)
    refuse_totals_mismatch(
        [{"rule": "validation", "route": "review", "message": "legacy, no findings"}]
    )


def test_only_error_severity_violations_become_findings() -> None:
    invoice = _with_address(FIXED_ADDRESS)
    warning = Violation(
        code=ViolationCode.TOTALS_MISMATCH, field="total", message="w", severity="warning"
    )
    error = Violation(code=ViolationCode.IBAN_INVALID, field="iban", message="e", severity="error")
    warned = _route_reasons(invoice, ValidationReport(violations=(warning,)))
    assert all(r.get("rule") != PolicyRule.VALIDATION.value for r in warned)
    refuse_totals_mismatch(warned)
    both = _route_reasons(invoice, ValidationReport(violations=(warning, error)))
    (validation,) = [r for r in both if r.get("rule") == PolicyRule.VALIDATION.value]
    assert validation["violations"] == [
        ValidationFinding(code=ViolationCode.IBAN_INVALID, field="iban").model_dump(mode="json")
    ]
