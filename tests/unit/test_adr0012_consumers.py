"""ADR-0012 consumer spec: the invoice builder takes the document kind and line VAT categories
from the A/B-agreed extraction (no 5%/AED defaults), the validator routes every unresolved value
to review, `document_type_code` derives 380/381/480/81, fraud compares only same-kind documents,
and each tax category plus a credit note survives builder -> UBL -> PINT AE schematron."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from apx.common.errors import EInvoiceError
from apx.common.invoice import Invoice, InvoiceKind, VatCategory
from apx.common.master import BankAccount, Vendor
from apx.common.types import FieldValue, InvoiceId, Money, Route, Sha256Hex, VendorId
from apx.db.models import Invoice as InvoiceRow
from apx.einvoice.interfaces import DocumentTypeCode, SchematronResult
from apx.einvoice.schematron import PintAeSchematronValidator
from apx.einvoice.type_code import document_type_code
from apx.extract.interfaces import (
    ExtractionResult,
    ExtractorId,
    ExtractorOutput,
    FieldName,
    InvoiceTypeCode,
    RawLineItem,
)
from apx.fraud.interfaces import InvoiceHistory
from apx.fraud.rules import DuplicateInvoiceRule, SplitInvoiceRule
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.interfaces import PolicyRule, PolicyThresholds, RoutingInput
from apx.policy.writer import invoice_for_digest
from apx.validate.interfaces import ValidationReport, ViolationCode, validate_invoice
from apx.workflows.activities import DefaultInvoiceActivities, InvoiceActivityDeps
from apx.workflows.components import GroundedInvoiceBuilder, PintAeUblBuilder
from apx.workflows.interfaces import InvoiceWorkflowInput
from apx.workflows.sql import HistorySnapshot
from tests.fixtures.pint_ae_data import require_pint_ae_data_root
from tests.unit._policy_factories import (
    make_fraud_report,
    make_scan_report,
    make_trust_decision,
    money,
)

SHA = Sha256Hex("c" * 64)
VENDOR = VendorId("V-ADR12")
SUPPLIER_TRN = "100000000000003"
BUYER_TRN = "100000000000103"
VERIFIED_IBAN = "AE070331234567890123456"
INP = InvoiceWorkflowInput(invoice_id=InvoiceId(SHA), document_sha256=SHA, filename="inv.pdf")
TODAY = date(2026, 9, 23)
ALL_CATEGORIES = tuple(VatCategory)
S, Z, E, AE, OS = (
    VatCategory.STANDARD,
    VatCategory.ZERO_RATED,
    VatCategory.EXEMPT,
    VatCategory.REVERSE_CHARGE,
    VatCategory.OUT_OF_SCOPE,
)
REPO_ROOT = Path(__file__).resolve().parents[2]


class _Vendors:
    async def vendor_id_for_trn(self, trn: str) -> VendorId | None:
        return VENDOR if trn == SUPPLIER_TRN else None


def _fields(category: VatCategory, **overrides: str | None) -> dict[FieldName, FieldValue]:
    vat = "50.00" if category is S else "0.00"
    total = "1050.00" if category is S else "1000.00"
    values: dict[FieldName, str | None] = {
        FieldName.INVOICE_NUMBER: "INV-12",
        FieldName.ISSUE_DATE: "2026-03-05",
        FieldName.DUE_DATE: "2026-04-04",
        FieldName.SUPPLIER_NAME: "Gulf Steel Trading LLC",
        FieldName.SUPPLIER_TRN: SUPPLIER_TRN,
        FieldName.SUPPLIER_ADDRESS: "Building 4, Al Quoz, Dubai",
        FieldName.BUYER_NAME: "Sovereign Holdings",
        FieldName.BUYER_TRN: BUYER_TRN,
        FieldName.CURRENCY: "AED",
        FieldName.SUBTOTAL: "1000.00",
        FieldName.VAT_TOTAL: vat,
        FieldName.TOTAL: total,
    }
    values.update({FieldName(k): v for k, v in overrides.items()})
    return {
        name: FieldValue(name=name.value, raw=raw, normalized=raw)
        for name, raw in values.items()
        if raw is not None
    }


def _line(category: VatCategory | None, rate: str | None) -> RawLineItem:
    return RawLineItem(
        description="Steel beams",
        quantity="10",
        unit_price="100.00",
        net_amount="1000.00",
        vat_rate=rate,
        vat_category=category,
    )


def _output(
    which: ExtractorId, type_code: InvoiceTypeCode | None, lines: tuple[RawLineItem, ...]
) -> ExtractorOutput:
    return ExtractorOutput(
        extractor=which,
        model=which.value,
        fields=(),
        line_items=lines,
        logprobs={},
        valid_json=True,
        invoice_type_code=type_code,
    )


def _result(
    grounded: dict[FieldName, FieldValue],
    a_lines: tuple[RawLineItem, ...],
    *,
    a_type: InvoiceTypeCode | None = InvoiceTypeCode.INVOICE,
    b_type: InvoiceTypeCode | None = InvoiceTypeCode.INVOICE,
    b_lines: tuple[RawLineItem, ...] | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        document_sha256=SHA,
        a=_output(ExtractorId.A, a_type, a_lines),
        b=_output(ExtractorId.B, b_type, a_lines if b_lines is None else b_lines),
        arbiter=None,
        disputed=frozenset(),
        grounded=grounded,
    )


def _rate(category: VatCategory) -> str:
    return "5%" if category is S else "0%"


async def _build(result: ExtractionResult) -> Invoice:
    return await GroundedInvoiceBuilder(_Vendors()).build(result, INP)


def _codes(report: ValidationReport) -> set[ViolationCode]:
    return {v.code for v in report.violations if v.severity == "error"}


def _route(invoice: Invoice, report: ValidationReport) -> tuple[Route, set[PolicyRule]]:
    engine = DefaultPolicyEngine(
        PolicyThresholds(
            approval_limit=money("5000"),
            fraud_review=0.5,
            scan_review=0.5,
            attack_block=0.9,
            sod_cooldown_days=30,
            bank_change_review_days=30,
        )
    )
    decision = engine.route(
        RoutingInput(
            invoice=invoice,
            vendor=None,
            trust=make_trust_decision(all_accepted=True),
            validation=report,
            fraud=make_fraud_report(),
            scan=make_scan_report(),
            at=datetime(2026, 9, 23, tzinfo=UTC),
            bank_changes=(),
        )
    )
    return decision.route, {r.rule for r in decision.reasons}


# --- document_type_code ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "categories", "expected"),
    [
        (InvoiceKind.INVOICE, (S,), DocumentTypeCode.INVOICE),
        (InvoiceKind.INVOICE, (Z,), DocumentTypeCode.INVOICE),
        (InvoiceKind.INVOICE, (AE,), DocumentTypeCode.INVOICE),
        (InvoiceKind.INVOICE, (E,), DocumentTypeCode.OUT_OF_SCOPE_INVOICE),
        (InvoiceKind.INVOICE, (OS,), DocumentTypeCode.OUT_OF_SCOPE_INVOICE),
        (InvoiceKind.INVOICE, (E, OS), DocumentTypeCode.OUT_OF_SCOPE_INVOICE),
        (InvoiceKind.INVOICE, (E, S), DocumentTypeCode.INVOICE),
        (InvoiceKind.CREDIT_NOTE, (S,), DocumentTypeCode.CREDIT_NOTE),
        (InvoiceKind.CREDIT_NOTE, (Z, E), DocumentTypeCode.CREDIT_NOTE),
        (InvoiceKind.CREDIT_NOTE, (E, OS), DocumentTypeCode.OUT_OF_SCOPE_CREDIT_NOTE),
    ],
)
def test_document_type_code(
    kind: InvoiceKind, categories: tuple[VatCategory, ...], expected: DocumentTypeCode
) -> None:
    assert document_type_code(kind, categories) is expected


@pytest.mark.parametrize(
    ("kind", "categories"),
    [(None, (S,)), (InvoiceKind.INVOICE, (S, None)), (InvoiceKind.CREDIT_NOTE, ())],
)
def test_document_type_code_refuses_unknowns(
    kind: InvoiceKind | None, categories: tuple[VatCategory | None, ...]
) -> None:
    with pytest.raises(EInvoiceError):
        document_type_code(kind, categories)


# --- builder + validator + routing ----------------------------------------------------------


@pytest.mark.parametrize("category", ALL_CATEGORIES)
async def test_agreed_category_and_type_build_a_clean_invoice(category: VatCategory) -> None:
    invoice = await _build(_result(_fields(category), (_line(category, _rate(category)),)))
    assert invoice.kind is InvoiceKind.INVOICE
    assert invoice.lines[0].vat_category is category
    assert invoice.lines[0].vat_rate == (Decimal("5") if category is S else Decimal("0"))
    assert validate_invoice(invoice, today=TODAY).violations == ()


async def test_credit_note_keeps_printed_positive_amounts() -> None:
    invoice = await _build(
        _result(
            _fields(S),
            (_line(S, "5%"),),
            a_type=InvoiceTypeCode.CREDIT_NOTE,
            b_type=InvoiceTypeCode.CREDIT_NOTE,
        )
    )
    assert invoice.kind is InvoiceKind.CREDIT_NOTE
    assert invoice.total.amount == Decimal("1050.00")
    assert validate_invoice(invoice, today=TODAY).violations == ()


@pytest.mark.parametrize(
    ("a_type", "b_type"),
    [
        (InvoiceTypeCode.INVOICE, InvoiceTypeCode.CREDIT_NOTE),
        (InvoiceTypeCode.INVOICE, None),
        (None, None),
    ],
)
async def test_type_disagreement_or_null_routes_to_review(
    a_type: InvoiceTypeCode | None, b_type: InvoiceTypeCode | None
) -> None:
    invoice = await _build(_result(_fields(S), (_line(S, "5%"),), a_type=a_type, b_type=b_type))
    assert invoice.kind is None
    report = validate_invoice(invoice, today=TODAY)
    assert _codes(report) == {ViolationCode.INVOICE_TYPE_MISSING}
    route, rules = _route(invoice, report)
    assert route is Route.REVIEW
    assert PolicyRule.VALIDATION in rules


async def test_category_disagreement_routes_to_review_and_keeps_the_line() -> None:
    invoice = await _build(_result(_fields(S), (_line(S, "5%"),), b_lines=(_line(Z, "5%"),)))
    assert len(invoice.lines) == 1
    assert invoice.lines[0].vat_category is None
    report = validate_invoice(invoice, today=TODAY)
    assert _codes(report) == {ViolationCode.VAT_CATEGORY_MISSING}
    assert _route(invoice, report)[0] is Route.REVIEW


@pytest.mark.parametrize("rate", [None, "", "five-ish"])
async def test_missing_or_unreadable_rate_is_none_not_five(rate: str | None) -> None:
    invoice = await _build(_result(_fields(S), (_line(S, rate),)))
    assert len(invoice.lines) == 1
    assert invoice.lines[0].vat_rate is None
    report = validate_invoice(invoice, today=TODAY)
    assert _codes(report) == {ViolationCode.VAT_RATE_MISSING}
    assert _route(invoice, report)[0] is Route.REVIEW


async def test_missing_currency_is_xxx_never_aed_and_routes_to_review() -> None:
    invoice = await _build(_result(_fields(S, currency=None), (_line(S, "5%"),)))
    assert invoice.total.currency == "XXX"
    assert invoice.lines[0].net_amount.currency == "XXX"
    report = validate_invoice(invoice, today=TODAY)
    assert _codes(report) == {ViolationCode.CURRENCY_INVALID}
    assert _route(invoice, report)[0] is Route.REVIEW


@pytest.mark.parametrize(
    ("category", "rate"), [(S, "0%"), (Z, "5%"), (E, "5%"), (AE, "7%"), (OS, "5%")]
)
async def test_category_rate_mismatch(category: VatCategory, rate: str) -> None:
    fields = _fields(category, vat_total="0.00", total="1000.00")
    invoice = await _build(_result(fields, (_line(category, rate),)))
    report = validate_invoice(invoice, today=TODAY)
    assert ViolationCode.VAT_CATEGORY_RATE_MISMATCH in _codes(report)


@pytest.mark.parametrize("kind", [InvoiceKind.INVOICE, InvoiceKind.CREDIT_NOTE])
async def test_negative_total_is_an_error_on_either_kind(kind: InvoiceKind) -> None:
    code = InvoiceTypeCode.INVOICE if kind is InvoiceKind.INVOICE else InvoiceTypeCode.CREDIT_NOTE
    fields = _fields(Z, subtotal="-1000.00", total="-1000.00")
    line = RawLineItem(
        description="Refund",
        quantity="1",
        unit_price="-1000.00",
        net_amount="-1000.00",
        vat_rate="0%",
        vat_category=Z,
    )
    invoice = await _build(_result(fields, (line,), a_type=code, b_type=code))
    report = validate_invoice(invoice, today=TODAY)
    negative = [v for v in report.violations if v.field == "total" and "negative" in v.message]
    assert negative and all(v.severity == "error" for v in negative)
    assert _route(invoice, report)[0] is Route.REVIEW


def test_components_has_no_silent_money_defaults() -> None:
    source = (REPO_ROOT / "apx" / "workflows" / "components.py").read_text(encoding="utf-8")
    assert '"AED"' not in source
    assert 'Decimal("5")' not in source
    assert "_STANDARD_RATE" not in source


async def test_ubl_builder_refuses_an_unclassified_line() -> None:
    invoice = await _build(_result(_fields(S), (_line(S, None),)))
    with pytest.raises(EInvoiceError):
        PintAeUblBuilder().build(
            invoice, DocumentTypeCode.INVOICE, verified_payment_iban=VERIFIED_IBAN
        )


# --- fraud: same-kind comparisons -----------------------------------------------------------


class _History:
    def __init__(self, *invoices: Invoice) -> None:
        self._invoices = invoices

    def recent_for_vendor(self, vendor_id: VendorId, days: int) -> list[Invoice]:
        return [i for i in self._invoices if i.vendor_id == vendor_id]


async def _pair(kind_a: InvoiceKind | None, kind_b: InvoiceKind | None) -> tuple[Invoice, Invoice]:
    base = await _build(_result(_fields(S), (_line(S, "5%"),)))
    first = base.model_copy(update={"id": InvoiceId("a" * 64), "kind": kind_a})
    second = base.model_copy(update={"id": InvoiceId("b" * 64), "kind": kind_b})
    return first, second


@pytest.mark.parametrize(
    ("kind_a", "kind_b", "flagged"),
    [
        (InvoiceKind.INVOICE, InvoiceKind.INVOICE, True),
        (InvoiceKind.CREDIT_NOTE, InvoiceKind.CREDIT_NOTE, True),
        (InvoiceKind.INVOICE, InvoiceKind.CREDIT_NOTE, False),
        (InvoiceKind.INVOICE, None, True),
    ],
)
async def test_duplicate_rule_compares_same_kind_only(
    kind_a: InvoiceKind, kind_b: InvoiceKind | None, flagged: bool
) -> None:
    first, second = await _pair(kind_a, kind_b)
    history = cast(InvoiceHistory, _History(first))
    assert bool(DuplicateInvoiceRule().evaluate(second, history)) is flagged


async def test_split_rule_ignores_a_credit_note_sibling() -> None:
    first, second = await _pair(InvoiceKind.CREDIT_NOTE, InvoiceKind.INVOICE)
    rule = SplitInvoiceRule(approval_limit=Money(amount=Decimal("2000")))
    assert rule.evaluate(second, cast(InvoiceHistory, _History(first))) == []
    same = first.model_copy(update={"kind": InvoiceKind.INVOICE, "number": "INV-13"})
    assert rule.evaluate(second, cast(InvoiceHistory, _History(same)))


# --- build_einvoice activity picks the type code and rule-set family -------------------------


class _Directory:
    async def lookup(self, vendor_id: VendorId | None) -> tuple[Vendor, tuple[()]]:
        account = BankAccount(
            iban=VERIFIED_IBAN,
            verified=True,
            verified_by=None,
            verified_at=None,
            source="admin_callback",
        )
        vendor = Vendor(id=VENDOR, name="Gulf Steel", trn=SUPPLIER_TRN, bank_accounts=(account,))
        return vendor, ()


class _Schematron:
    def __init__(self) -> None:
        self.seen: list[bytes] = []

    def validate(self, xml: bytes) -> SchematronResult:
        self.seen.append(xml)
        return SchematronResult(passed=True, failures=(), rules_version="test")


class _Audit:
    async def append(self, event: object) -> None:
        return None


def _activities(invoice_rules: _Schematron, credit_rules: _Schematron | None) -> Any:
    unused = cast(Any, None)
    deps = InvoiceActivityDeps(
        documents=unused,
        scanner=unused,
        parser=unused,
        parsed=unused,
        extraction=unused,
        invoices=unused,
        feature_builder=unused,
        trust_gate=unused,
        validator=unused,
        fraud_engine=unused,
        vendors=cast(Any, _Directory()),
        policy_engine=unused,
        ubl_builder=PintAeUblBuilder(buyer_trade_license="TL-1"),
        schematron=cast(Any, invoice_rules),
        credit_note_schematron=cast(Any, credit_rules),
        asp_client=unused,
        archive=unused,
        audit=cast(Any, _Audit()),
        records=unused,
        session_factory=unused,
    )
    return DefaultInvoiceActivities(deps)


@pytest.mark.parametrize(
    ("type_code", "category", "expected", "credit"),
    [
        (InvoiceTypeCode.INVOICE, S, DocumentTypeCode.INVOICE, False),
        (InvoiceTypeCode.INVOICE, E, DocumentTypeCode.OUT_OF_SCOPE_INVOICE, False),
        (InvoiceTypeCode.CREDIT_NOTE, S, DocumentTypeCode.CREDIT_NOTE, True),
        (InvoiceTypeCode.CREDIT_NOTE, OS, DocumentTypeCode.OUT_OF_SCOPE_CREDIT_NOTE, True),
    ],
)
async def test_build_einvoice_derives_the_type_code(
    type_code: InvoiceTypeCode, category: VatCategory, expected: DocumentTypeCode, credit: bool
) -> None:
    invoice = await _build(
        _result(
            _fields(category),
            (_line(category, _rate(category)),),
            a_type=type_code,
            b_type=type_code,
        )
    )
    invoice_rules, credit_rules = _Schematron(), _Schematron()
    document = await _activities(invoice_rules, credit_rules).build_einvoice(invoice)
    assert document.type_code is expected
    assert (len(credit_rules.seen), len(invoice_rules.seen)) == ((1, 0) if credit else (0, 1))


async def test_build_einvoice_refuses_unknown_kind_and_unwired_credit_rules() -> None:
    unknown = await _build(_result(_fields(S), (_line(S, "5%"),), a_type=None, b_type=None))
    with pytest.raises(EInvoiceError):
        await _activities(_Schematron(), _Schematron()).build_einvoice(unknown)
    credit = unknown.model_copy(update={"kind": InvoiceKind.CREDIT_NOTE})
    with pytest.raises(EInvoiceError):
        await _activities(_Schematron(), None).build_einvoice(credit)


# --- real PINT AE schematron (artifacts) ----------------------------------------------------


def _schematron_result(invoice: Invoice) -> tuple[DocumentTypeCode, SchematronResult]:
    type_code = document_type_code(invoice.kind, (ln.vat_category for ln in invoice.lines))
    document = PintAeUblBuilder(buyer_trade_license="TL-1", buyer_address="Marina, Dubai").build(
        invoice, type_code, verified_payment_iban=VERIFIED_IBAN
    )
    with PintAeSchematronValidator(
        DocumentTypeCode.INVOICE if type_code.is_invoice else DocumentTypeCode.CREDIT_NOTE,
        data_root=require_pint_ae_data_root(),
    ) as validator:
        return type_code, validator.validate(document.xml)


@pytest.mark.artifacts
@pytest.mark.parametrize("category", ALL_CATEGORIES)
async def test_each_category_passes_pint_ae_schematron(category: VatCategory) -> None:
    invoice = await _build(_result(_fields(category), (_line(category, _rate(category)),)))
    type_code, result = _schematron_result(invoice)
    expected = (
        DocumentTypeCode.OUT_OF_SCOPE_INVOICE if category in (E, OS) else DocumentTypeCode.INVOICE
    )
    assert type_code is expected
    assert result.passed, [f.message for f in result.failures]


@pytest.mark.artifacts
@pytest.mark.parametrize(
    ("category", "expected"),
    [(S, DocumentTypeCode.CREDIT_NOTE), (E, DocumentTypeCode.OUT_OF_SCOPE_CREDIT_NOTE)],
)
async def test_credit_note_round_trip_passes_pint_ae_schematron(
    category: VatCategory, expected: DocumentTypeCode
) -> None:
    code = InvoiceTypeCode.CREDIT_NOTE
    invoice = await _build(
        _result(_fields(category), (_line(category, _rate(category)),), a_type=code, b_type=code)
    )
    type_code, result = _schematron_result(invoice)
    assert type_code is expected
    assert result.passed, [f.message for f in result.failures]


# --- demo scenarios (scripts/demo.py): agreed extraction of their truths changes nothing -----


def _demo_truths() -> list[dict[str, Any]]:
    import scripts.demo as demo

    data = REPO_ROOT / "data"
    wanted = {s.sha for s in demo.scenarios(data)}
    truths = []
    for path in sorted((data / "uae" / "test").glob("*.truth.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        pdf = path.with_name(f"{truth['sha256']}.pdf")
        if pdf.exists() and _sha(pdf) in wanted:
            truths.append(truth)
    return truths


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


_ADR12_CODES = {
    ViolationCode.INVOICE_TYPE_MISSING,
    ViolationCode.VAT_CATEGORY_MISSING,
    ViolationCode.VAT_RATE_MISSING,
    ViolationCode.VAT_CATEGORY_RATE_MISMATCH,
}


@pytest.mark.external  # reads the local factory split (data/uae/test), not PINT AE bundles
async def test_demo_scenarios_raise_no_adr12_violation_when_a_and_b_agree() -> None:
    from ml.extractor_b.mappings import uae_classification

    truths = _demo_truths()
    assert truths, "demo scenarios found no factory truths under data/uae/test"
    for truth in truths:
        ctx = truth["context"]
        type_code, category = uae_classification(truth["tax_case"])
        lines = tuple(
            RawLineItem(
                description=ln["description_en"],
                quantity=ln["quantity"],
                unit_price=ln["unit_price"],
                net_amount=ln["net_amount"],
                vat_rate=ln["vat_rate"],
                vat_category=category,
            )
            for ln in ctx["lines"]
        )
        fields = _fields(
            category,
            invoice_number=ctx["invoice_number"],
            issue_date=ctx["issue_date_iso"],
            currency=ctx["currency"],
            subtotal=ctx["subtotal"],
            vat_total=ctx["vat_total"],
            total=ctx["total"],
        )
        invoice = await _build(_result(fields, lines, a_type=type_code, b_type=type_code))
        assert invoice.kind is InvoiceKind.INVOICE
        codes = {v.code for v in validate_invoice(invoice, today=TODAY).violations}
        assert not codes & _ADR12_CODES, (truth["sha256"], codes)
        assert re.fullmatch(r"[A-Z]{3}", invoice.total.currency) and invoice.total.currency != "XXX"


# --- persistence: invoices.kind feeds the fraud history -------------------------------------


def _row(invoice: Invoice, kind: str | None) -> InvoiceRow:
    return InvoiceRow(
        id=invoice.id,
        document_sha256=invoice.document_sha256,
        vendor_id=invoice.vendor_id,
        language=invoice.language,
        number=invoice.number,
        issue_date=invoice.issue_date,
        due_date=invoice.due_date,
        po_number=invoice.po_number,
        currency=invoice.total.currency,
        subtotal=invoice.subtotal.amount,
        vat_total=invoice.vat_total.amount,
        total=invoice.total.amount,
        iban=invoice.iban,
        kind=kind,
    )


@pytest.mark.parametrize(
    ("stored", "expected"),
    [("invoice", InvoiceKind.INVOICE), ("credit_note", InvoiceKind.CREDIT_NOTE), (None, None)],
)
async def test_history_row_loader_carries_kind(
    stored: str | None, expected: InvoiceKind | None
) -> None:
    subject, other = await _pair(InvoiceKind.INVOICE, None)
    loaded = invoice_for_digest(_row(other, stored))
    assert loaded.kind is expected
    snapshot = HistorySnapshot(subject, (subject, loaded), (), (), None)
    flagged = bool(DuplicateInvoiceRule().evaluate(subject, snapshot))
    assert flagged is (expected is not InvoiceKind.CREDIT_NOTE)


# --- AE rate: PINT AE 1.0.4 official example -------------------------------------------------

_UBL_NS = {
    "cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
}


@pytest.mark.artifacts
async def test_official_ae_example_line_validates_and_builds() -> None:
    """`Doc-level-charge-AE-category.xml` states its AE line at 5% with 0 VAT; the validator
    accepts that line as-is and our UBL for it passes the PINT AE schematron."""
    from lxml import etree  # type: ignore[import-untyped]

    example = (
        require_pint_ae_data_root()
        / "1.0.4"
        / "resources"
        / "trn-invoice"
        / "example"
        / "Doc-level-charge-AE-category.xml"
    )
    root = etree.parse(str(example)).getroot()
    line = root.find("cac:InvoiceLine", _UBL_NS)
    assert line is not None
    category = line.findtext("cac:Item/cac:ClassifiedTaxCategory/cbc:ID", namespaces=_UBL_NS)
    rate = line.findtext("cac:Item/cac:ClassifiedTaxCategory/cbc:Percent", namespaces=_UBL_NS)
    quantity = line.findtext("cbc:InvoicedQuantity", namespaces=_UBL_NS)
    price = line.findtext("cac:Price/cbc:PriceAmount", namespaces=_UBL_NS)
    net = line.findtext("cbc:LineExtensionAmount", namespaces=_UBL_NS)
    assert (category, rate) == ("AE", "5")
    assert quantity and price and net

    fields = _fields(AE, subtotal=net, vat_total="0.00", total=net)
    raw = RawLineItem(
        description="Reverse-charge supply",
        quantity=quantity,
        unit_price=price,
        net_amount=net,
        vat_rate=f"{rate}%",
        vat_category=AE,
    )
    invoice = await _build(_result(fields, (raw,)))
    assert invoice.lines[0].vat_rate == Decimal("5")
    assert validate_invoice(invoice, today=TODAY).violations == ()
    type_code, result = _schematron_result(invoice)
    assert type_code is DocumentTypeCode.INVOICE
    assert result.passed, [f.message for f in result.failures]
