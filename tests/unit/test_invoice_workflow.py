"""Unit tests for `apx.workflows.workflow.InvoiceWorkflow` using Temporal's time-skipping test
environment: the workflow itself is the unit under test, `FakeInvoiceActivities` (a canned,
in-memory stand-in for every pipeline module) is its only dependency (PRD §10 M5.2)."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError, WorkflowHandle
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from apx.asp_mock.interfaces import ArchiveRecord, AspReceipt
from apx.common.errors import ModelServingError, ParsedArtifactIntegrityError
from apx.common.invoice import Invoice, Party
from apx.common.types import IdempotencyKey, InvoiceId, Language, Money, Route, make_idempotency_key
from apx.einvoice.interfaces import DocumentTypeCode, EInvoiceDocument
from apx.extract.interfaces import ExtractionResult, ExtractorId, ExtractorOutput
from apx.fraud.interfaces import FraudReport
from apx.parse.interfaces import ParsedDocumentRef
from apx.policy.interfaces import PolicyReason, PolicyRule, RoutedPayee, RoutingDecision
from apx.scan.interfaces import ScanReport
from apx.trust.interfaces import TrustDecision
from apx.validate.interfaces import ValidationReport
from apx.workflows.interfaces import InvoiceStatus, InvoiceWorkflowInput, InvoiceWorkflowState, Step
from apx.workflows.workflow import InvoiceWorkflow, ReviewDecision

TASK_QUEUE = "test-invoice-workflow"

INVOICE_ID = InvoiceId(hashlib.sha256(b"fake-invoice").hexdigest())
WORKFLOW_INPUT = InvoiceWorkflowInput(
    invoice_id=INVOICE_ID, document_sha256=INVOICE_ID, filename="invoice.pdf"
)


def _scan_report() -> ScanReport:
    return ScanReport(document_sha256=INVOICE_ID, flags=(), injection_score=0.0)


def _parsed_ref() -> ParsedDocumentRef:
    return ParsedDocumentRef(
        document_sha256=INVOICE_ID,
        content_sha256=hashlib.sha256(b"fake-parse").hexdigest(),
        parser_version="fake-1",
        page_count=5,
    )


def _extraction_result() -> ExtractionResult:
    output_a = ExtractorOutput(
        extractor=ExtractorId.A,
        model="fake",
        fields=(),
        line_items=(),
        logprobs={},
        valid_json=True,
    )
    output_b = output_a.model_copy(update={"extractor": ExtractorId.B})
    return ExtractionResult(
        document_sha256=INVOICE_ID,
        a=output_a,
        b=output_b,
        arbiter=None,
        disputed=frozenset(),
        grounded={},
    )


def _trust_decision() -> TrustDecision:
    return TrustDecision(fields=(), all_accepted=True)


def _invoice() -> Invoice:
    party = Party(name="Acme Supplies LLC", trn=None, address=None, country="AE")
    total = Money(amount=Decimal("100.00"), currency="AED")
    zero = Money(amount=Decimal("0.00"), currency="AED")
    return Invoice(
        id=INVOICE_ID,
        document_sha256=INVOICE_ID,
        language=Language.EN,
        number="INV-1",
        issue_date=date(2026, 1, 1),
        due_date=None,
        supplier=party,
        buyer=party,
        vendor_id=None,
        po_number=None,
        lines=(),
        subtotal=total,
        vat_total=zero,
        total=total,
        iban=None,
    )


def _validation_report() -> ValidationReport:
    return ValidationReport(violations=())


def _fraud_report() -> FraudReport:
    return FraudReport(flags=(), max_score=0.0)


def _einvoice_document() -> EInvoiceDocument:
    xml = b"<Invoice/>"
    return EInvoiceDocument(
        invoice_id=INVOICE_ID,
        type_code=DocumentTypeCode.INVOICE,
        xml=xml,
        xml_sha256=hashlib.sha256(xml).hexdigest(),
        spec_version="1.0",
    )


def _asp_receipt(key: IdempotencyKey) -> AspReceipt:
    return AspReceipt(
        receipt_id="receipt-1",
        idempotency_key=key,
        status="accepted",
        received_at=datetime.now(UTC),
        message="ok",
    )


def _archive_record(key: IdempotencyKey) -> ArchiveRecord:
    return ArchiveRecord(
        idempotency_key=key,
        xml_sha256=hashlib.sha256(b"<Invoice/>").hexdigest(),
        archived_at=datetime.now(UTC),
        retain_until=date(2033, 1, 1),
    )


@dataclass
class FakeInvoiceActivities:
    """Canned `InvoiceActivities`: one fixed result per step, a call-order log and the
    idempotency keys `send_to_asp`/`archive` actually received, so tests can assert the
    workflow calls activities in order and builds `invoice_id:step` keys correctly."""

    route_decision: RoutingDecision
    calls: list[str] = field(default_factory=list)
    received_keys: dict[str, IdempotencyKey] = field(default_factory=dict)
    review_note_fails: bool = False
    received_refs: dict[str, ParsedDocumentRef] = field(default_factory=dict)
    extract_error: Exception | None = None
    failures: list[tuple[str, str]] = field(default_factory=list)

    @activity.defn(name="scan")
    async def scan(self, inp: InvoiceWorkflowInput) -> ScanReport:
        self.calls.append("scan")
        return _scan_report()

    @activity.defn(name="parse")
    async def parse(self, inp: InvoiceWorkflowInput) -> ParsedDocumentRef:
        self.calls.append("parse")
        return _parsed_ref()

    @activity.defn(name="extract")
    async def extract(self, parsed: ParsedDocumentRef) -> ExtractionResult:
        self.calls.append("extract")
        self.received_refs["extract"] = parsed
        if self.extract_error is not None:
            raise self.extract_error
        return _extraction_result()

    @activity.defn(name="trust_gate")
    async def trust_gate(
        self, result: ExtractionResult, parsed: ParsedDocumentRef
    ) -> TrustDecision:
        self.calls.append("trust_gate")
        self.received_refs["trust_gate"] = parsed
        return _trust_decision()

    @activity.defn(name="build_invoice")
    async def build_invoice(self, result: ExtractionResult, inp: InvoiceWorkflowInput) -> Invoice:
        self.calls.append("build_invoice")
        return _invoice()

    @activity.defn(name="validate")
    async def validate(self, invoice: Invoice) -> ValidationReport:
        self.calls.append("validate")
        return _validation_report()

    @activity.defn(name="fraud")
    async def fraud(self, invoice: Invoice, parsed: ParsedDocumentRef) -> FraudReport:
        self.calls.append("fraud")
        self.received_refs["fraud"] = parsed
        return _fraud_report()

    @activity.defn(name="route")
    async def route(
        self,
        invoice: Invoice,
        trust: TrustDecision,
        validation: ValidationReport,
        fraud: FraudReport,
        scan: ScanReport,
    ) -> RoutingDecision:
        self.calls.append("route")
        return self.route_decision

    @activity.defn(name="review_note")
    async def review_note(self, invoice: Invoice, routing: RoutingDecision) -> bool:
        self.calls.append("review_note")
        if self.review_note_fails:
            raise RuntimeError("agent crashed")
        return True

    @activity.defn(name="build_einvoice")
    async def build_einvoice(
        self, invoice: Invoice, pinned_payee: RoutedPayee | None
    ) -> EInvoiceDocument:
        self.calls.append("build_einvoice")
        return _einvoice_document()

    @activity.defn(name="send_to_asp")
    async def send_to_asp(self, key: IdempotencyKey, document: EInvoiceDocument) -> AspReceipt:
        self.calls.append("send_to_asp")
        self.received_keys["send_to_asp"] = key
        return _asp_receipt(key)

    @activity.defn(name="archive")
    async def archive(self, key: IdempotencyKey, document: EInvoiceDocument) -> ArchiveRecord:
        self.calls.append("archive")
        self.received_keys["archive"] = key
        return _archive_record(key)

    @activity.defn(name="mark_failed")
    async def mark_failed(self, invoice_id: str, error: str) -> None:
        self.calls.append("mark_failed")
        self.failures.append((invoice_id, error))


def _activities(fake: FakeInvoiceActivities) -> Sequence[Callable[..., object]]:
    return [
        fake.scan,
        fake.parse,
        fake.extract,
        fake.trust_gate,
        fake.build_invoice,
        fake.validate,
        fake.fraud,
        fake.route,
        fake.review_note,
        fake.build_einvoice,
        fake.send_to_asp,
        fake.archive,
        fake.mark_failed,
    ]


@pytest.fixture
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    # `TEMPORAL_TEST_SERVER_PATH`, if set, points at an already-downloaded test-server binary
    # and skips the SDK's own download -- needed on hosts where the ephemeral-server download
    # is unreachable but a binary was fetched out of band. Unset, this is the SDK's own
    # (unmodified) lazy-download behaviour.
    existing_path = os.environ.get("TEMPORAL_TEST_SERVER_PATH")
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter, test_server_existing_path=existing_path
    ) as env:
        yield env


async def _run_workflow(
    temporal_env: WorkflowEnvironment, fake: FakeInvoiceActivities
) -> tuple[WorkflowHandle[InvoiceWorkflow, InvoiceWorkflowState], InvoiceWorkflowState]:
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[InvoiceWorkflow],
        activities=_activities(fake),
    ):
        handle = await temporal_env.client.start_workflow(
            InvoiceWorkflow.run,
            WORKFLOW_INPUT,
            id=INVOICE_ID,
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
    return handle, result


@pytest.mark.asyncio
async def test_auto_route_completes_through_archive(temporal_env: WorkflowEnvironment) -> None:
    fake = FakeInvoiceActivities(route_decision=RoutingDecision(route=Route.AUTO, reasons=()))
    _handle, result = await _run_workflow(temporal_env, fake)

    assert result.status is InvoiceStatus.ARCHIVED
    assert result.route is Route.AUTO
    assert result.error is None
    assert fake.calls == [
        "scan",
        "parse",
        "extract",
        "trust_gate",
        "build_invoice",
        "validate",
        "fraud",
        "route",
        "build_einvoice",
        "send_to_asp",
        "archive",
    ]


@pytest.mark.asyncio
async def test_parse_ref_reaches_every_parse_consumer(temporal_env: WorkflowEnvironment) -> None:
    # The parse crosses activity boundaries only as its small stored-parse ref (page PNGs would
    # break JSON payloads); every consumer must get exactly the ref `parse` returned.
    fake = FakeInvoiceActivities(route_decision=RoutingDecision(route=Route.AUTO, reasons=()))
    await _run_workflow(temporal_env, fake)

    assert fake.received_refs == {
        "extract": _parsed_ref(),
        "trust_gate": _parsed_ref(),
        "fraud": _parsed_ref(),
    }


@pytest.mark.asyncio
async def test_idempotency_keys_are_invoice_id_and_step(temporal_env: WorkflowEnvironment) -> None:
    fake = FakeInvoiceActivities(route_decision=RoutingDecision(route=Route.AUTO, reasons=()))
    await _run_workflow(temporal_env, fake)

    expected_asp_key = make_idempotency_key(INVOICE_ID, Step.ASP_SEND.value)
    expected_archive_key = make_idempotency_key(INVOICE_ID, Step.ARCHIVE.value)
    assert fake.received_keys["send_to_asp"] == expected_asp_key
    assert fake.received_keys["archive"] == expected_archive_key


@pytest.mark.asyncio
async def test_block_route_ends_blocked_without_einvoice(temporal_env: WorkflowEnvironment) -> None:
    reason = PolicyReason(rule=PolicyRule.P4_FLAGS, route=Route.BLOCK, message="blocked for test")
    block_decision = RoutingDecision(route=Route.BLOCK, reasons=(reason,))
    fake = FakeInvoiceActivities(route_decision=block_decision)
    _handle, result = await _run_workflow(temporal_env, fake)

    assert result.status is InvoiceStatus.BLOCKED
    assert result.route is Route.BLOCK
    assert "build_einvoice" not in fake.calls
    assert "send_to_asp" not in fake.calls
    assert "archive" not in fake.calls


@pytest.mark.asyncio
async def test_review_route_waits_then_approved_completes(
    temporal_env: WorkflowEnvironment,
) -> None:
    fake = FakeInvoiceActivities(route_decision=RoutingDecision(route=Route.REVIEW, reasons=()))
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[InvoiceWorkflow],
        activities=_activities(fake),
    ):
        handle = await temporal_env.client.start_workflow(
            InvoiceWorkflow.run,
            WORKFLOW_INPUT,
            id=INVOICE_ID,
            task_queue=TASK_QUEUE,
        )

        async def _awaiting_review() -> bool:
            state = await handle.query(InvoiceWorkflow.status)
            return bool(state.status is InvoiceStatus.AWAITING_REVIEW)

        await _wait_until(_awaiting_review)
        mid_state = await handle.query(InvoiceWorkflow.status)
        assert mid_state.status is InvoiceStatus.AWAITING_REVIEW
        assert mid_state.route is Route.REVIEW
        assert "build_einvoice" not in fake.calls

        await handle.signal(InvoiceWorkflow.review, ReviewDecision(approved=True))
        result = await handle.result()

    assert result.status is InvoiceStatus.ARCHIVED
    assert fake.calls.index("review_note") == fake.calls.index("route") + 1


@pytest.mark.asyncio
async def test_failing_review_note_never_blocks_the_human_review(
    temporal_env: WorkflowEnvironment,
) -> None:
    fake = FakeInvoiceActivities(
        route_decision=RoutingDecision(route=Route.REVIEW, reasons=()), review_note_fails=True
    )
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[InvoiceWorkflow],
        activities=_activities(fake),
    ):
        handle = await temporal_env.client.start_workflow(
            InvoiceWorkflow.run,
            WORKFLOW_INPUT,
            id=INVOICE_ID,
            task_queue=TASK_QUEUE,
        )

        async def _both_note_attempts_done() -> bool:
            return fake.calls.count("review_note") == 2

        await _wait_until(_both_note_attempts_done)
        await handle.signal(InvoiceWorkflow.review, ReviewDecision(approved=True))
        result = await handle.result()

    assert result.status is InvoiceStatus.ARCHIVED


@pytest.mark.asyncio
async def test_review_route_rejected_ends_blocked(temporal_env: WorkflowEnvironment) -> None:
    fake = FakeInvoiceActivities(route_decision=RoutingDecision(route=Route.REVIEW, reasons=()))
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[InvoiceWorkflow],
        activities=_activities(fake),
    ):
        handle = await temporal_env.client.start_workflow(
            InvoiceWorkflow.run,
            WORKFLOW_INPUT,
            id=INVOICE_ID,
            task_queue=TASK_QUEUE,
        )

        async def _awaiting_review() -> bool:
            state = await handle.query(InvoiceWorkflow.status)
            return bool(state.status is InvoiceStatus.AWAITING_REVIEW)

        await _wait_until(_awaiting_review)
        await handle.signal(InvoiceWorkflow.review, ReviewDecision(approved=False))
        result = await handle.result()

    assert result.status is InvoiceStatus.BLOCKED
    assert "build_einvoice" not in fake.calls
    assert "archive" not in fake.calls


async def _wait_until(
    predicate: Callable[[], Awaitable[bool]], timeout_s: float = 10.0, interval_s: float = 0.05
) -> None:
    """Polls `predicate` (a zero-arg async callable) until it is true, real wall-clock time --
    this is client-side polling of a query, not workflow code, so real `asyncio.sleep` is fine."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(interval_s)
    raise AssertionError("condition not met before timeout")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        # Deterministic for the same ref: retrying would only re-run OCR and fail again.
        (ParsedArtifactIntegrityError("stored parse does not match its ref"), 1),
        # Transient: keeps the default five attempts.
        (ModelServingError("llama-swap timed out"), 5),
    ],
    ids=["integrity-error-not-retried", "serving-error-retried"],
)
async def test_deterministic_errors_are_not_retried(
    temporal_env: WorkflowEnvironment, error: Exception, attempts: int
) -> None:
    fake = FakeInvoiceActivities(
        route_decision=RoutingDecision(route=Route.AUTO, reasons=()), extract_error=error
    )

    with pytest.raises(WorkflowFailureError):
        await _run_workflow(temporal_env, fake)

    assert fake.calls.count("extract") == attempts
    assert "trust_gate" not in fake.calls
    # The API reads `invoices.status`, so the failure is written there too, exactly once.
    assert fake.calls[-1] == "mark_failed"
    assert [invoice_id for invoice_id, _ in fake.failures] == [INVOICE_ID]
    # The typed cause, not the bare "Activity task failed" of the ActivityError around it.
    assert f"{type(error).__name__}: {error}" in fake.failures[0][1]


@pytest.mark.asyncio
async def test_llm_activities_run_only_on_the_bounded_llm_queue(
    temporal_env: WorkflowEnvironment,
) -> None:
    """ADR-0013: with `llm_task_queue` set, extract runs on the LLM worker (capped at one
    concurrent activity here) and never on the workflow's own queue."""
    fake = FakeInvoiceActivities(route_decision=RoutingDecision(route=Route.AUTO, reasons=()))
    queues: list[str] = []

    @activity.defn(name="extract")
    async def extract(parsed: ParsedDocumentRef) -> ExtractionResult:
        queues.append(activity.info().task_queue)
        return await fake.extract(parsed)

    llm_queue = f"{TASK_QUEUE}-llm"
    main_activities = [a for a in _activities(fake) if a not in (fake.extract, fake.review_note)]
    async with (
        Worker(
            temporal_env.client,
            task_queue=TASK_QUEUE,
            workflows=[InvoiceWorkflow],
            activities=main_activities,
        ),
        Worker(
            temporal_env.client,
            task_queue=llm_queue,
            activities=[extract, fake.review_note],
            max_concurrent_activities=1,
        ),
    ):
        result = await temporal_env.client.execute_workflow(
            InvoiceWorkflow.run,
            WORKFLOW_INPUT.model_copy(update={"llm_task_queue": llm_queue}),
            id=f"{INVOICE_ID}-llm-queue",
            task_queue=TASK_QUEUE,
        )
    assert result.status is InvoiceStatus.ARCHIVED
    assert queues == [llm_queue]
