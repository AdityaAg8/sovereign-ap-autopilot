"""ADR-0012 amendment, end to end in the time-skipping Temporal env: an invoice whose extractors
disagreed on the kind and line VAT category routes to review, the approval carries the reviewer's
corrections, and the real `apply_review` + `build_einvoice` activities turn it into a PINT AE
document. Without corrections, `build_einvoice` fails once, non-retryably."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, cast

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from apx.audit.interfaces import AuditEvent
from apx.common.invoice import Invoice, InvoiceKind, InvoiceLine, Party, VatCategory
from apx.common.master import BankAccount, Vendor
from apx.common.types import InvoiceId, Language, Money, Route, VendorId
from apx.einvoice.interfaces import EInvoiceDocument, SchematronResult
from apx.extract.interfaces import ExtractionResult
from apx.policy.interfaces import RoutedPayee, RoutingDecision, pin_payee, select_payee
from apx.validate.interfaces import InvoiceValidator
from apx.workflows.activities import DefaultInvoiceActivities, InvoiceActivityDeps
from apx.workflows.components import PintAeUblBuilder
from apx.workflows.corrections import (
    LineCorrection,
    MissingCorrectionsError,
    ReviewCorrections,
    apply_corrections,
    corrections_from_fields,
)
from apx.workflows.interfaces import InvoiceStatus, InvoiceWorkflowInput
from apx.workflows.workflow import InvoiceWorkflow, ReviewDecision
from tests.unit.test_invoice_workflow import (
    INVOICE_ID,
    TASK_QUEUE,
    WORKFLOW_INPUT,
    FakeInvoiceActivities,
    _wait_until,
    temporal_env,
)

__all__ = ["temporal_env"]  # the fixture, re-exported so pytest finds it here

VENDOR = VendorId("V-REVIEW")
IBAN = "AE070331234567890123456"


def _unresolved_invoice() -> Invoice:
    """What `GroundedInvoiceBuilder` produces when A and B disagreed on the type and category."""
    net = Money(amount=Decimal("1000.00"))
    return Invoice(
        id=INVOICE_ID,
        document_sha256=INVOICE_ID,
        language=Language.EN,
        number="INV-9",
        issue_date=date(2026, 3, 5),
        due_date=date(2026, 4, 4),
        supplier=Party(name="Gulf Steel", trn="100000000000003", address="Al Quoz, Dubai"),
        buyer=Party(name="Sovereign Holdings", trn="100000000000103", address="Marina, Dubai"),
        vendor_id=VENDOR,
        po_number=None,
        lines=(
            InvoiceLine(
                description="Steel beams",
                quantity=Decimal("10"),
                unit_price=Money(amount=Decimal("100.00")),
                net_amount=net,
                vat_category=None,
                vat_rate=Decimal("0"),
            ),
        ),
        subtotal=net,
        vat_total=Money(amount=Decimal("0.00")),
        total=net,
        iban=None,
        kind=None,
    )


CORRECTIONS = ReviewCorrections(
    kind=InvoiceKind.CREDIT_NOTE,
    currency="AED",
    lines=(LineCorrection(index=1, vat_category=VatCategory.ZERO_RATED, vat_rate=Decimal("0")),),
)


class _Directory:
    async def lookup(self, vendor_id: VendorId | None) -> tuple[Vendor, tuple[()]]:
        account = BankAccount(
            iban=IBAN, verified=True, verified_by=None, verified_at=None, source="admin_callback"
        )
        return Vendor(id=VENDOR, name="Gulf Steel", trn=None, bank_accounts=(account,)), ()


def _review_routing() -> RoutingDecision:
    """REVIEW, with the payee the engine would pin for `_Directory`'s sole account."""
    account = BankAccount(
        iban=IBAN, verified=True, verified_by=None, verified_at=None, source="admin_callback"
    )
    vendor = Vendor(id=VENDOR, name="Gulf Steel", trn=None, bank_accounts=(account,))
    payee = pin_payee(select_payee(vendor, invoice_iban=None))
    return RoutingDecision(route=Route.REVIEW, reasons=(), payee=payee)


@dataclass
class _Rules:
    seen: list[bytes] = field(default_factory=list)

    def validate(self, xml: bytes) -> SchematronResult:
        self.seen.append(xml)
        return SchematronResult(passed=True, failures=(), rules_version="test")


@dataclass
class _Records:
    kinds: list[InvoiceKind] = field(default_factory=list)

    async def set_kind(self, invoice_id: InvoiceId, kind: InvoiceKind) -> None:
        self.kinds.append(kind)


@dataclass
class _Audit:
    events: list[AuditEvent] = field(default_factory=list)

    async def append(self, event: AuditEvent) -> None:
        self.events.append(event)


@dataclass
class _Fakes(FakeInvoiceActivities):
    einvoice_attempts: int = 0

    @activity.defn(name="build_invoice")
    async def build_invoice(self, result: ExtractionResult, inp: InvoiceWorkflowInput) -> Invoice:
        self.calls.append("build_invoice")
        return _unresolved_invoice()


def _real(audit: _Audit, records: _Records, invoice_rules: _Rules, credit_rules: _Rules) -> Any:
    unused = cast(Any, None)
    return DefaultInvoiceActivities(
        InvoiceActivityDeps(
            documents=unused,
            scanner=unused,
            parser=unused,
            parsed=unused,
            extraction=unused,
            invoices=unused,
            feature_builder=unused,
            trust_gate=unused,
            validator=InvoiceValidator(),
            fraud_engine=unused,
            vendors=cast(Any, _Directory()),
            policy_engine=unused,
            ubl_builder=PintAeUblBuilder(buyer_trn="100000000000103", buyer_trade_license="TL-1"),
            schematron=invoice_rules,
            credit_note_schematron=credit_rules,
            asp_client=unused,
            archive=unused,
            audit=cast(Any, audit),
            records=cast(Any, records),
            session_factory=unused,
        )
    )


def _activities(fake: _Fakes, real: Any) -> Sequence[Callable[..., object]]:
    @activity.defn(name="build_einvoice")
    async def build_einvoice(invoice: Invoice, pinned: RoutedPayee | None) -> EInvoiceDocument:
        fake.calls.append("build_einvoice")
        fake.einvoice_attempts += 1
        return cast(EInvoiceDocument, await real.build_einvoice(invoice, pinned))

    @activity.defn(name="apply_review")
    async def apply_review(invoice: Invoice, corrections: ReviewCorrections) -> Invoice:
        fake.calls.append("apply_review")
        return cast(Invoice, await real.apply_review(invoice, corrections))

    return [
        fake.scan,
        fake.parse,
        fake.extract,
        fake.trust_gate,
        fake.build_invoice,
        fake.validate,
        fake.fraud,
        fake.route,
        fake.review_hold,
        fake.review_note,
        fake.confirm_review,
        apply_review,
        build_einvoice,
        fake.send_to_asp,
        fake.archive,
        fake.mark_failed,
    ]


async def _run(env: WorkflowEnvironment, fake: _Fakes, real: Any, decision: ReviewDecision) -> Any:
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[InvoiceWorkflow],
        activities=_activities(fake, real),
    ):
        handle = await env.client.start_workflow(
            InvoiceWorkflow.run, WORKFLOW_INPUT, id=INVOICE_ID, task_queue=TASK_QUEUE
        )

        async def _awaiting() -> bool:
            state = await handle.query(InvoiceWorkflow.status)
            return bool(state.status is InvoiceStatus.AWAITING_REVIEW)

        await _wait_until(_awaiting)
        await handle.signal(InvoiceWorkflow.review, decision)
        return await handle.result()


@pytest.mark.asyncio
async def test_disagreement_review_approve_with_corrections_builds_the_einvoice(
    temporal_env: WorkflowEnvironment,
) -> None:
    fake = _Fakes(route_decision=_review_routing())
    audit, records, invoice_rules, credit_rules = _Audit(), _Records(), _Rules(), _Rules()
    real = _real(audit, records, invoice_rules, credit_rules)

    result = await _run(
        temporal_env, fake, real, ReviewDecision(approved=True, corrections=CORRECTIONS)
    )

    assert result.status is InvoiceStatus.ARCHIVED
    assert fake.calls.index("apply_review") < fake.calls.index("build_einvoice")
    assert records.kinds == [InvoiceKind.CREDIT_NOTE]
    assert len(credit_rules.seen) == 1 and invoice_rules.seen == []
    assert b"<cbc:CreditNoteTypeCode>381</cbc:CreditNoteTypeCode>" in credit_rules.seen[0]
    applied = [e for e in audit.events if e.action == "activity:apply_review"]
    assert applied and applied[0].payload["corrections"] == CORRECTIONS.model_dump(mode="json")


@pytest.mark.asyncio
async def test_approve_without_corrections_fails_once_non_retryably(
    temporal_env: WorkflowEnvironment,
) -> None:
    fake = _Fakes(route_decision=_review_routing())
    real = _real(_Audit(), _Records(), _Rules(), _Rules())

    with pytest.raises(WorkflowFailureError):
        await _run(temporal_env, fake, real, ReviewDecision(approved=True))

    assert fake.einvoice_attempts == 1  # EInvoiceError is non-retryable
    assert "send_to_asp" not in fake.calls
    # A failed post-review build must not leave the invoice at APPROVED in the database.
    assert fake.calls[-1] == "mark_failed"


# --- pure corrections helpers ---------------------------------------------------------------

_ADDRESS = "Al Quoz, Dubai"
_EXTRACTED: dict[str, str | None] = {
    "supplier_trn": "100000000000003",
    "currency": None,
    "supplier_address": _ADDRESS,
}
"""The model-sourced values the workflow's invoice was built from (supplier TRN present, and a
supplier address naming an emirate)."""


def test_corrections_from_fields_names_every_missing_value() -> None:
    with pytest.raises(MissingCorrectionsError) as err:
        corrections_from_fields(
            {
                "invoice_type_code": None,
                "currency": "XXX",
                "line_items[1].vat_category": "S",
                "line_items[1].vat_rate": None,
                "line_items[2].vat_category": "bogus",
                "line_items[2].vat_rate": "0",
            },
            extracted=_EXTRACTED,
        )
    message = str(err.value)
    for name in (
        "invoice_type_code",
        "currency",
        "line_items[1].vat_rate",
        "line_items[2].vat_category",
    ):
        assert name in message


def test_corrections_from_fields_parses_a_complete_set() -> None:
    corrections = corrections_from_fields(
        {
            "invoice_type_code": "380",
            "currency": "aed",
            "line_items[1].vat_category": "AE",
            "line_items[1].vat_rate": "5%",
            "supplier_address": _ADDRESS,
        },
        extracted=_EXTRACTED,
    )
    assert corrections.kind is InvoiceKind.INVOICE
    assert corrections.currency == "AED"
    assert corrections.lines == (
        LineCorrection(index=1, vat_category=VatCategory.REVERSE_CHARGE, vat_rate=Decimal("5")),
    )
    assert corrections.supplier_address is None  # the extracted address is kept as is


_COMPLETE_FIELDS: dict[str, str | None] = {
    "invoice_type_code": "380",
    "currency": "AED",
    "line_items[1].vat_category": "S",
    "line_items[1].vat_rate": "5",
    "supplier_address": _ADDRESS,
}


@pytest.mark.parametrize("currency", ["USD", "eur"])
def test_corrections_from_fields_refuses_a_non_aed_currency(currency: str) -> None:
    """The approval path (`ApiService._review_corrections`, before `record_approval`) refuses
    a currency the PINT AE builder would refuse, instead of failing after the approval."""
    with pytest.raises(MissingCorrectionsError, match=r"currency \(must be AED"):
        corrections_from_fields({**_COMPLETE_FIELDS, "currency": currency}, extracted=_EXTRACTED)


@pytest.mark.parametrize(
    ("category", "rate"),
    [("E", "5"), ("O", "0.01"), ("S", "0"), ("S", "10"), ("Z", "5"), ("AE", "10")],
)
def test_corrections_from_fields_refuses_a_rate_that_does_not_fit_the_category(
    category: str, rate: str
) -> None:
    fields = {
        **_COMPLETE_FIELDS,
        "line_items[1].vat_category": category,
        "line_items[1].vat_rate": rate,
    }
    with pytest.raises(MissingCorrectionsError, match=r"line_items\[1\]\.vat_rate \("):
        corrections_from_fields(fields, extracted=_EXTRACTED)


@pytest.mark.parametrize(
    ("category", "rate"), [("S", "5"), ("Z", "0"), ("E", "0"), ("O", "0"), ("AE", "0"), ("AE", "5")]
)
def test_corrections_from_fields_accepts_every_fitting_category_rate(
    category: str, rate: str
) -> None:
    fields = {
        **_COMPLETE_FIELDS,
        "line_items[1].vat_category": category,
        "line_items[1].vat_rate": rate,
    }
    corrections = corrections_from_fields(fields, extracted=_EXTRACTED)
    assert corrections.lines == (
        LineCorrection(index=1, vat_category=VatCategory(category), vat_rate=Decimal(rate)),
    )


def test_apply_corrections_refuses_a_partial_line_set() -> None:
    invoice = _unresolved_invoice()
    with pytest.raises(MissingCorrectionsError):
        apply_corrections(invoice, CORRECTIONS.model_copy(update={"lines": ()}))
    fixed = apply_corrections(invoice, CORRECTIONS)
    assert fixed.kind is InvoiceKind.CREDIT_NOTE
    assert fixed.lines[0].vat_category is VatCategory.ZERO_RATED


async def test_apply_review_refuses_a_still_unresolved_currency() -> None:
    invoice = _unresolved_invoice()
    xxx = Money(amount=Decimal("1000.00"), currency="XXX")
    invoice = invoice.model_copy(update={"subtotal": xxx, "total": xxx})
    real = _real(_Audit(), _Records(), _Rules(), _Rules())
    bad = CORRECTIONS.model_copy(update={"currency": "XXX"})
    with pytest.raises(MissingCorrectionsError):
        await real.apply_review(invoice, bad)


async def test_apply_review_refuses_a_category_rate_mismatch() -> None:
    """An exempt line at 5% contradicts its category and the builder would refuse it, so the
    approved review is stopped here, before the e-invoice build (VAT_CATEGORY_RATE_MISMATCH)."""
    real = _real(_Audit(), _Records(), _Rules(), _Rules())
    mismatch = LineCorrection(index=1, vat_category=VatCategory.EXEMPT, vat_rate=Decimal("5"))
    bad = CORRECTIONS.model_copy(update={"lines": (mismatch,)})
    with pytest.raises(MissingCorrectionsError, match="vat_category_rate_mismatch"):
        await real.apply_review(_unresolved_invoice(), bad)


async def test_apply_review_refuses_a_non_aed_currency() -> None:
    real = _real(_Audit(), _Records(), _Rules(), _Rules())
    bad = CORRECTIONS.model_copy(update={"currency": "USD"})
    with pytest.raises(MissingCorrectionsError, match="currency_invalid"):
        await real.apply_review(_unresolved_invoice(), bad)
