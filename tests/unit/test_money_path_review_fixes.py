"""CR-money (PR #48 review): the money-path fixes around ADR-0014 and ADR-0012 amendment 2. The
buyer TRN is our configured one and the document's is only cross-checked; a missing supplier TRN
or a currency relabel is refused before approval; the payee P1 chose is pinned at routing,
persisted for readers and re-checked at the e-invoice build (TOCTOU)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, cast

import pytest
from temporalio.exceptions import ApplicationError

from apx.agent.interfaces import AgentContext
from apx.agent.prompts import DefaultAgentPromptBuilder
from apx.api.service import corrections_for_rows
from apx.audit.interfaces import AuditEvent
from apx.common.errors import ConfigError, PayeeChangedError, PayeeRefusal
from apx.common.invoice import Invoice, InvoiceKind, InvoiceLine, Party, VatCategory
from apx.common.master import Vendor
from apx.common.types import InvoiceId, Language, Money, Role, Route, Sha256Hex, VendorId
from apx.db.enums import FieldSource
from apx.db.models import Field as FieldRow
from apx.einvoice.interfaces import DocumentTypeCode, EInvoiceDocument, SchematronResult
from apx.policy.engine import IBAN_EDIT_HINT, DefaultPolicyEngine
from apx.policy.interfaces import (
    ActionRequest,
    Capability,
    PayeeMatch,
    PolicyRule,
    PolicyThresholds,
    RoutedPayee,
    RoutingDecision,
    RoutingInput,
    pin_payee,
    routing_record,
    select_payee,
)
from apx.trust.interfaces import TrustDecision
from apx.validate.interfaces import InvoiceValidator, ViolationCode, validate_invoice
from apx.workflows.activities import DefaultInvoiceActivities, InvoiceActivityDeps
from apx.workflows.components import PintAeUblBuilder
from apx.workflows.corrections import (
    CurrencyRelabelError,
    LineCorrection,
    MissingCorrectionsError,
    ReviewCorrections,
    SupplierTrnMissingError,
    apply_corrections,
    corrections_from_fields,
)
from tests.unit._policy_factories import (
    make_bank_account,
    make_fraud_report,
    make_scan_report,
    make_trust_decision,
    make_user,
    make_validation_report,
    make_vendor,
    money,
)

OUR_TRN = "100000000000103"
OTHER_TRN = "100999999999903"
SUPPLIER_TRN = "100000000000003"
IBAN_A = "AE070331234567890123456"
IBAN_B = "AE460331234567890654321"
AT = datetime(2026, 3, 1, tzinfo=UTC)


def _invoice(
    *,
    buyer_trn: str | None = OUR_TRN,
    supplier_trn: str | None = SUPPLIER_TRN,
    iban: str | None = None,
    currency: str = "AED",
    vendor_id: str = "vendor-1",
) -> Invoice:
    net = Money(amount=Decimal("1000.00"), currency=currency)
    return Invoice(
        id=InvoiceId("inv-cr"),
        document_sha256="0" * 64,
        language=Language.EN,
        number="INV-7",
        issue_date=date(2026, 2, 1),
        due_date=date(2026, 3, 3),
        supplier=Party(name="Gulf Steel", trn=supplier_trn, address="Al Quoz, Dubai"),
        buyer=Party(name="Sovereign Holdings", trn=buyer_trn, address="Marina, Dubai"),
        vendor_id=VendorId(vendor_id),
        po_number=None,
        lines=(
            InvoiceLine(
                description="Steel beams",
                quantity=Decimal("10"),
                unit_price=Money(amount=Decimal("100.00"), currency=currency),
                net_amount=net,
                vat_category=VatCategory.STANDARD,
                vat_rate=Decimal("5"),
            ),
        ),
        subtotal=net,
        vat_total=Money(amount=Decimal("50.00"), currency=currency),
        total=Money(amount=Decimal("1050.00"), currency=currency),
        iban=iban,
        kind=InvoiceKind.INVOICE,
    )


# --- 1. buyer TRN from config; the document's is only cross-checked --------------------------


def _codes(invoice: Invoice, buyer_trn: str | None) -> dict[str | None, ViolationCode]:
    report = validate_invoice(invoice, today=date(2026, 2, 2), buyer_trn=buyer_trn)
    return {v.field: v.code for v in report.violations if v.severity == "error"}


def test_configured_buyer_trn_makes_a_missing_document_buyer_trn_no_error() -> None:
    assert _codes(_invoice(buyer_trn=None), None)["buyer.trn"] is ViolationCode.MISSING_MANDATORY
    assert "buyer.trn" not in _codes(_invoice(buyer_trn=None), OUR_TRN)
    assert "buyer.trn" not in _codes(_invoice(buyer_trn=OUR_TRN), OUR_TRN)


def test_a_different_document_buyer_trn_is_an_error_that_routes_review() -> None:
    validator = InvoiceValidator(buyer_trn=OUR_TRN)
    report = validator.validate(_invoice(buyer_trn=OTHER_TRN))
    mismatch = [v for v in report.violations if v.code is ViolationCode.BUYER_TRN_MISMATCH]
    assert len(mismatch) == 1 and mismatch[0].severity == "error"
    engine = DefaultPolicyEngine(_thresholds())
    decision = engine.route(_routing_input(_invoice(buyer_trn=OTHER_TRN), validation=report))
    assert decision.route is Route.REVIEW
    assert PolicyRule.VALIDATION in {r.rule for r in decision.reasons}


def test_ubl_takes_the_configured_buyer_trn_never_the_printed_one() -> None:
    builder = PintAeUblBuilder(buyer_trn=OUR_TRN, buyer_trade_license="TL-1")
    for printed in (None, OTHER_TRN):
        document = builder.build(
            _invoice(buyer_trn=printed), DocumentTypeCode.INVOICE, verified_payment_iban=IBAN_A
        )
        assert OUR_TRN.encode() in document.xml
        assert OTHER_TRN.encode() not in document.xml


def test_ubl_builder_refuses_a_blank_configured_buyer_trn() -> None:
    with pytest.raises(ConfigError):
        PintAeUblBuilder(buyer_trn="  ")


@pytest.mark.parametrize("value", [None, "", "12345", "100000000000100"])
def test_worker_refuses_to_start_without_a_valid_buyer_trn(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    from apx.workflows.main import load_worker_settings

    if value is None:
        monkeypatch.delenv("WORKER_BUYER_TRN", raising=False)
    else:
        monkeypatch.setenv("WORKER_BUYER_TRN", value)
    with pytest.raises(ConfigError, match="WORKER_BUYER_TRN"):
        load_worker_settings()


def test_worker_settings_accept_a_valid_buyer_trn(monkeypatch: pytest.MonkeyPatch) -> None:
    from apx.workflows.main import load_worker_settings

    monkeypatch.setenv("WORKER_BUYER_TRN", f" {OUR_TRN} ")
    assert load_worker_settings().buyer_trn == OUR_TRN


# --- 1b/4. approval refusals no review can fix -----------------------------------------------

_COMPLETE = {
    "invoice_type_code": "380",
    "currency": "AED",
    "line_items[1].vat_category": "S",
    "line_items[1].vat_rate": "5",
    "supplier_address": "Al Quoz, Dubai",
}


def _row(row_id: int, name: str, value: str | None, source: FieldSource) -> FieldRow:
    return FieldRow(
        id=row_id,
        invoice_id="inv-cr",
        name=name,
        value=value,
        normalized_value=value,
        source=source,
    )


def _rows(extra: list[tuple[str, str | None, FieldSource]]) -> list[FieldRow]:
    base = [(name, value, FieldSource.EXTRACTOR_A) for name, value in _COMPLETE.items()]
    return [_row(i, n, v, s) for i, (n, v, s) in enumerate([*base, *extra], start=1)]


def test_approval_refused_when_the_document_has_no_supplier_trn() -> None:
    with pytest.raises(SupplierTrnMissingError, match="supplier TRN"):
        corrections_for_rows(_rows([]))
    # A reviewer's edit never reaches the workflow's invoice, so it does not unlock approval.
    with pytest.raises(SupplierTrnMissingError):
        corrections_for_rows(_rows([("supplier_trn", SUPPLIER_TRN, FieldSource.HUMAN)]))
    assert issubclass(SupplierTrnMissingError, MissingCorrectionsError)  # the API's 422
    ok = corrections_for_rows(_rows([("supplier_trn", SUPPLIER_TRN, FieldSource.EXTRACTOR_A)]))
    assert ok.currency == "AED"


async def test_apply_review_refuses_a_missing_supplier_trn() -> None:
    activities, _ = _activities(make_vendor())
    corrections = ReviewCorrections(
        kind=InvoiceKind.INVOICE,
        currency="AED",
        lines=(LineCorrection(index=1, vat_category=VatCategory.STANDARD, vat_rate=Decimal("5")),),
    )
    with pytest.raises(MissingCorrectionsError, match="missing_mandatory"):
        await activities.apply_review(_invoice(supplier_trn=None), corrections)


@pytest.mark.parametrize("extracted", ["USD", "eur"])
def test_a_human_currency_override_to_aed_of_a_valid_code_is_refused(extracted: str) -> None:
    rows = _rows(
        [
            ("supplier_trn", SUPPLIER_TRN, FieldSource.EXTRACTOR_A),
            ("currency", extracted, FieldSource.EXTRACTOR_A),
            ("currency", "AED", FieldSource.HUMAN),
        ]
    )
    # the first "currency" row (AED, from _COMPLETE) is superseded by the later model row
    with pytest.raises(CurrencyRelabelError, match=extracted.upper()):
        corrections_for_rows(rows)
    assert issubclass(CurrencyRelabelError, MissingCorrectionsError)


@pytest.mark.parametrize("extracted", [None, "", "XXX", "AED", "US"])
def test_resolving_a_missing_or_unreadable_currency_is_allowed(extracted: str | None) -> None:
    values = dict(_COMPLETE)
    corrections = corrections_from_fields(
        values,
        extracted={
            "supplier_trn": SUPPLIER_TRN,
            "currency": extracted,
            "supplier_address": values["supplier_address"],
        },
    )
    assert corrections.currency == "AED"


def test_apply_corrections_refuses_relabelling_usd_amounts_as_aed() -> None:
    corrections = ReviewCorrections(
        kind=InvoiceKind.INVOICE,
        currency="AED",
        lines=(LineCorrection(index=1, vat_category=VatCategory.STANDARD, vat_rate=Decimal("5")),),
    )
    with pytest.raises(MissingCorrectionsError, match="USD"):
        apply_corrections(_invoice(currency="USD"), corrections)
    xxx = apply_corrections(_invoice(currency="XXX"), corrections)
    assert xxx.total.currency == "AED"


# --- 5. an IBAN edit cannot select a payee: the refusal says so ------------------------------


def _thresholds() -> PolicyThresholds:
    return PolicyThresholds(
        approval_limit=money("5000"), fraud_review=0.5, scan_review=0.5, attack_block=0.9
    )


def _routing_input(
    invoice: Invoice, *, vendor: Vendor | None = None, validation: Any = None
) -> RoutingInput:
    return RoutingInput(
        invoice=invoice,
        vendor=vendor if vendor is not None else make_vendor(),
        trust=make_trust_decision(),
        validation=validation if validation is not None else make_validation_report(),
        fraud=make_fraud_report(),
        scan=make_scan_report(),
        at=AT,
    )


def _two_accounts() -> Vendor:
    return make_vendor(
        accounts=(make_bank_account(iban=IBAN_A), make_bank_account(iban=IBAN_B)),
    )


@pytest.mark.parametrize(
    ("claimed", "refusal"),
    [
        (None, PayeeRefusal.INVOICE_IBAN_MISSING),
        ("AE07 0331-2345", PayeeRefusal.INVOICE_IBAN_MALFORMED),
        ("AE120331234567890999999", PayeeRefusal.NO_MATCH),
    ],
)
def test_invoice_iban_refusals_tell_the_reviewer_an_edit_does_not_help(
    claimed: str | None, refusal: PayeeRefusal
) -> None:
    engine = DefaultPolicyEngine(_thresholds())
    decision = engine.route(_routing_input(_invoice(iban=claimed), vendor=_two_accounts()))
    p1 = [r for r in decision.reasons if r.payee_refusal is refusal]
    assert len(p1) == 1
    assert "edit of the IBAN field does not change the selection" in p1[0].message
    assert "reject the invoice" in p1[0].message
    assert len(p1[0].message) <= 300  # the agent's SystemText bound
    assert decision.payee is None

    authz = engine.authorize(
        ActionRequest(
            actor=make_user(roles=frozenset({Role.APPROVER})),
            action=Capability.APPROVE_PAYMENT,
            invoice=_invoice(iban=claimed),
            vendor=_two_accounts(),
            at=AT,
            routing=decision,
        )
    )
    assert not authz.allowed
    assert any(IBAN_EDIT_HINT in r.message for r in authz.reasons)


def test_a_refusal_the_invoice_iban_cannot_cause_has_no_edit_hint() -> None:
    engine = DefaultPolicyEngine(_thresholds())
    vendor = make_vendor(accounts=(make_bank_account(verified=False, source="unverified"),))
    decision = engine.route(_routing_input(_invoice(), vendor=vendor))
    [p1] = [r for r in decision.reasons if r.payee_refusal is not None]
    assert p1.payee_refusal is PayeeRefusal.NO_VERIFIED_ACCOUNT
    assert IBAN_EDIT_HINT not in p1.message


# --- 2/6. the routed payee is pinned, persisted masked, and audited ---------------------------


def test_route_pins_select_payee_and_the_record_never_holds_a_raw_iban() -> None:
    engine = DefaultPolicyEngine(_thresholds())
    decision = engine.route(_routing_input(_invoice(iban=IBAN_B), vendor=_two_accounts()))
    expected = pin_payee(select_payee(_two_accounts(), invoice_iban=IBAN_B))
    assert decision.payee == expected
    assert decision.payee is not None and decision.payee.masked_iban == "****4321"
    record = routing_record(decision)
    blob = json.dumps(record)
    assert IBAN_B not in blob and IBAN_A not in blob
    assert decision.payee.iban_sha256 not in blob
    assert record[-1] == {
        "payee": {"masked_iban": "****4321", "match": "matched", "verified_count": 2}
    }
    assert all("rule" in r for r in record[:-1] if isinstance(r, dict))


def test_routing_record_carries_the_refusal_when_p1_refused() -> None:
    engine = DefaultPolicyEngine(_thresholds())
    decision = engine.route(_routing_input(_invoice(), vendor=_two_accounts()))
    assert routing_record(decision)[-1] == {"payee": {"refusal": "invoice_iban_missing"}}


@dataclass
class _Directory:
    vendor: Vendor

    async def lookup(self, vendor_id: VendorId | None) -> tuple[Vendor, tuple[()]]:
        return self.vendor, ()


@dataclass
class _Records:
    routed_calls: list[RoutingDecision] = field(default_factory=list)
    kinds: list[InvoiceKind] = field(default_factory=list)

    async def routed(
        self, invoice_id: InvoiceId, routing: RoutingDecision, trust: TrustDecision
    ) -> None:
        self.routed_calls.append(routing)

    async def set_kind(self, invoice_id: InvoiceId, kind: InvoiceKind) -> None:
        self.kinds.append(kind)


@dataclass
class _Audit:
    events: list[AuditEvent] = field(default_factory=list)

    async def append(self, event: AuditEvent) -> None:
        self.events.append(event)


@dataclass
class _Ubl:
    paid: list[str | None] = field(default_factory=list)

    def build(
        self, invoice: Invoice, type_code: DocumentTypeCode, *, verified_payment_iban: str | None
    ) -> EInvoiceDocument:
        self.paid.append(verified_payment_iban)
        return EInvoiceDocument(
            invoice_id=invoice.id,
            type_code=type_code,
            xml=b"<Invoice/>",
            xml_sha256=Sha256Hex("0" * 64),
            spec_version="test",
        )


class _Rules:
    def validate(self, xml: bytes) -> SchematronResult:
        return SchematronResult(passed=True, failures=(), rules_version="test")


@dataclass
class _Harness:
    directory: _Directory
    records: _Records
    audit: _Audit
    ubl: _Ubl


def _activities(vendor: Vendor) -> tuple[DefaultInvoiceActivities, _Harness]:
    unused = cast(Any, None)
    harness = _Harness(_Directory(vendor), _Records(), _Audit(), _Ubl())
    deps = InvoiceActivityDeps(
        documents=unused,
        scanner=unused,
        parser=unused,
        parsed=unused,
        extraction=unused,
        invoices=unused,
        feature_builder=unused,
        trust_gate=unused,
        validator=InvoiceValidator(buyer_trn=OUR_TRN),
        fraud_engine=unused,
        vendors=cast(Any, harness.directory),
        policy_engine=DefaultPolicyEngine(_thresholds()),
        ubl_builder=harness.ubl,
        schematron=_Rules(),
        credit_note_schematron=_Rules(),
        asp_client=unused,
        archive=unused,
        audit=cast(Any, harness.audit),
        records=cast(Any, harness.records),
        session_factory=unused,
        clock=lambda: AT,
    )
    return DefaultInvoiceActivities(deps), harness


async def _route(activities: DefaultInvoiceActivities, invoice: Invoice) -> RoutingDecision:
    return await activities.route(
        invoice,
        make_trust_decision(),
        make_validation_report(),
        make_fraud_report(),
        make_scan_report(),
    )


async def test_route_activity_persists_and_audits_the_payee_selection() -> None:
    activities, h = _activities(_two_accounts())
    decision = await _route(activities, _invoice(iban=IBAN_A))
    assert h.records.routed_calls == [decision] and decision.payee is not None
    [event] = [e for e in h.audit.events if e.action == "activity:route"]
    assert event.payload["payee_match"] == PayeeMatch.MATCHED.value
    assert event.payload["payee_refusal"] is None
    assert event.payload["payee"] == "****3456"
    assert IBAN_A not in json.dumps(event.payload)

    activities, h = _activities(_two_accounts())
    await _route(activities, _invoice(iban=None))
    [event] = [e for e in h.audit.events if e.action == "activity:route"]
    assert event.payload["payee_refusal"] == PayeeRefusal.INVOICE_IBAN_MISSING.value
    assert event.payload["payee_match"] is None


# --- 3. TOCTOU: the build pays only the payee pinned at routing ------------------------------


async def test_build_pays_the_pinned_payee_when_the_master_is_unchanged() -> None:
    activities, h = _activities(_two_accounts())
    invoice = _invoice(iban=IBAN_B)
    decision = await _route(activities, invoice)
    await activities.build_einvoice(invoice, decision.payee)
    assert h.ubl.paid == [IBAN_B]


def _assert_refused(exc: pytest.ExceptionInfo[ApplicationError], h: _Harness) -> None:
    assert exc.value.type == PayeeChangedError.__name__
    assert exc.value.non_retryable
    assert isinstance(exc.value.__cause__, PayeeChangedError)
    assert h.ubl.paid == []
    assert [e.action for e in h.audit.events].count("activity:payee_changed") == 1
    assert not any(e.action == "activity:build_einvoice" for e in h.audit.events)


async def test_master_change_between_approval_and_build_is_refused() -> None:
    """Routed (and approved) with one verified account; before the build an admin swaps it for
    another verified account. The live selection differs from the pinned one: refused."""
    activities, h = _activities(make_vendor(accounts=(make_bank_account(iban=IBAN_A),)))
    invoice = _invoice(iban=None)
    decision = await _route(activities, invoice)
    assert decision.payee is not None
    h.directory.vendor = make_vendor(accounts=(make_bank_account(iban=IBAN_B),))
    with pytest.raises(ApplicationError) as exc:
        await activities.build_einvoice(invoice, decision.payee)
    _assert_refused(exc, h)
    changed = next(e for e in h.audit.events if e.action == "activity:payee_changed")
    assert changed.payload["pinned"] == "****3456" and changed.payload["live"] == "****4321"


async def test_a_second_account_after_routing_changes_the_match_and_is_refused() -> None:
    activities, h = _activities(make_vendor(accounts=(make_bank_account(iban=IBAN_A),)))
    invoice = _invoice(iban=None)
    decision = await _route(activities, invoice)
    h.directory.vendor = _two_accounts()
    with pytest.raises(ApplicationError) as exc:
        await activities.build_einvoice(invoice, decision.payee)
    _assert_refused(exc, h)


async def test_a_payee_that_appears_after_a_refused_routing_is_refused() -> None:
    unverified = make_bank_account(iban=IBAN_A, verified=False, source="unverified")
    activities, h = _activities(make_vendor(accounts=(unverified,)))
    invoice = _invoice(iban=IBAN_A)
    decision = await _route(activities, invoice)
    assert decision.payee is None
    h.directory.vendor = make_vendor(accounts=(make_bank_account(iban=IBAN_A),))
    with pytest.raises(ApplicationError) as exc:
        await activities.build_einvoice(invoice, decision.payee)
    _assert_refused(exc, h)


async def test_a_forged_pin_is_refused() -> None:
    activities, h = _activities(make_vendor(accounts=(make_bank_account(iban=IBAN_A),)))
    invoice = _invoice(iban=None)
    decision = await _route(activities, invoice)
    assert decision.payee is not None
    forged = RoutedPayee(
        masked_iban=decision.payee.masked_iban,
        match=decision.payee.match,
        verified_count=1,
        iban_sha256=Sha256Hex("f" * 64),
    )
    with pytest.raises(ApplicationError) as exc:
        await activities.build_einvoice(invoice, forged)
    _assert_refused(exc, h)


# --- the pinned digest stays out of the agent's prompt ---------------------------------------


def test_agent_prompt_carries_the_masked_payee_but_not_its_digest() -> None:
    engine = DefaultPolicyEngine(_thresholds())
    decision = engine.route(_routing_input(_invoice(iban=IBAN_B), vendor=_two_accounts()))
    assert decision.payee is not None
    request = DefaultAgentPromptBuilder().build(
        AgentContext(invoice_id="inv-cr", routing=decision, state={}, quoted_document=()),
        (),
        "exception-agent",
    )
    text = json.dumps(request.model_dump(mode="json"))
    assert decision.payee.iban_sha256 not in text
    assert IBAN_B not in text
    assert "****4321" in text


# --- the agent reads the persisted route decision --------------------------------------------


async def test_get_invoice_reads_policy_rules_and_the_payee_refusal_from_the_view() -> None:
    from apx.agent.interfaces import GetInvoiceParams
    from apx.agent.sql import SQL_GET_INVOICE
    from apx.agent.toolbox import PgAgentToolbox
    from tests.unit._agent_factories import FakeReader, SpySink, chat_caller, invoice_row, reviewer

    engine = DefaultPolicyEngine(_thresholds())
    decision = engine.route(_routing_input(_invoice(), vendor=_two_accounts()))
    record = routing_record(decision)
    row = invoice_row(
        payee_masked=None,
        document_iban_matches_payee=None,
        reasons_json=[r for r in record if isinstance(r, dict) and "payee" not in r],
    )
    row["payee_refusal"] = "invoice_iban_missing"
    reader = FakeReader({SQL_GET_INVOICE: [row]})
    result = await PgAgentToolbox(reader=reader, sink=SpySink()).get_invoice(
        GetInvoiceParams(invoice_id=InvoiceId("inv-1")), chat_caller(reviewer())
    )
    assert result.payee is None and result.payee_refusal == "invoice_iban_missing"
    assert [r.rule for r in result.reasons] == ["p1"]
