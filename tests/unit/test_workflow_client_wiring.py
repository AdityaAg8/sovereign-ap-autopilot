"""Unit tests for the API's Temporal wiring (PRD §10 M5.2/M5.3, M5 review F3/F4): the workflow
starter uses the document sha256 as the workflow id and treats an already-started workflow as
success, the review signaller delivers `ReviewDecision` and turns delivery failures into a typed
`WorkflowSignalError`, and both fail loudly when `API_TEMPORAL_ADDRESS` is unset. A stub client
stands in for the Temporal server; the units under test are real.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode

from apx.api.deps import get_review_signaller, get_workflow_starter
from apx.api.settings import ApiSettings
from apx.api.signals import (
    TemporalReviewSignaller,
    UnconfiguredReviewSignaller,
    WorkflowSignalError,
)
from apx.common.errors import ConfigError
from apx.common.types import InvoiceId, Sha256Hex
from apx.ingest.interfaces import IngestedDocument
from apx.workflows.interfaces import InvoiceWorkflowInput
from apx.workflows.starter import LazyTemporalClient, TemporalWorkflowStarter
from apx.workflows.worker import LLM_TASK_QUEUE
from apx.workflows.workflow import ReviewDecision

SHA = Sha256Hex("b" * 64)


class _Handle:
    def __init__(self, client: _StubClient, workflow_id: str) -> None:
        self._client = client
        self._workflow_id = workflow_id

    async def signal(self, _signal: Any, arg: ReviewDecision) -> None:
        if self._client.signal_error is not None:
            raise self._client.signal_error
        self._client.signals.append((self._workflow_id, arg))


class _StubClient:
    def __init__(self) -> None:
        self.started: list[tuple[InvoiceWorkflowInput, str, str]] = []
        self.signals: list[tuple[str, ReviewDecision]] = []
        self.start_error: Exception | None = None
        self.signal_error: Exception | None = None

    async def start_workflow(
        self, _run: Any, arg: InvoiceWorkflowInput, *, id: str, task_queue: str
    ) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started.append((arg, id, task_queue))

    def get_workflow_handle(self, workflow_id: str) -> _Handle:
        return _Handle(self, workflow_id)


class _StubLazyClient(LazyTemporalClient):
    def __init__(self, client: _StubClient) -> None:
        super().__init__("unused:7233")
        self._stub = client

    async def get(self) -> Any:
        return self._stub


def _document() -> IngestedDocument:
    return IngestedDocument(
        sha256=SHA,
        filename="invoice.pdf",
        media_type="application/pdf",
        size_bytes=10,
        received_at=datetime.now(UTC),
        source="upload",
    )


def _settings(address: str | None) -> ApiSettings:
    return ApiSettings(jwt_secret="unit-test-secret-value-long-enough", temporal_address=address)  # type: ignore[arg-type]


async def test_starter_uses_the_sha256_as_workflow_id() -> None:
    stub = _StubClient()
    starter = TemporalWorkflowStarter(_StubLazyClient(stub), task_queue="q")

    workflow_id = await starter.start_invoice_workflow(_document())

    assert workflow_id == SHA
    [(inp, started_id, queue)] = stub.started
    assert (started_id, queue) == (SHA, "q")
    assert inp == InvoiceWorkflowInput(
        invoice_id=InvoiceId(SHA),
        document_sha256=SHA,
        filename="invoice.pdf",
        llm_task_queue=LLM_TASK_QUEUE,  # extract/review_note go to the bounded LLM worker
    )


async def test_starter_treats_an_already_started_workflow_as_success() -> None:
    stub = _StubClient()
    stub.start_error = WorkflowAlreadyStartedError(SHA, "InvoiceWorkflow")
    starter = TemporalWorkflowStarter(_StubLazyClient(stub))

    assert await starter.start_invoice_workflow(_document()) == SHA


async def test_starter_propagates_other_failures() -> None:
    stub = _StubClient()
    stub.start_error = RuntimeError("temporal unreachable")
    starter = TemporalWorkflowStarter(_StubLazyClient(stub))

    with pytest.raises(RuntimeError):
        await starter.start_invoice_workflow(_document())


async def test_signaller_delivers_the_review_decision() -> None:
    stub = _StubClient()
    signaller = TemporalReviewSignaller(_StubLazyClient(stub))

    await signaller.signal_review(InvoiceId(SHA), approved=False)

    assert stub.signals == [(SHA, ReviewDecision(approved=False))]


async def test_signaller_raises_a_typed_error_when_delivery_fails() -> None:
    stub = _StubClient()
    stub.signal_error = RPCError("workflow not found", RPCStatusCode.NOT_FOUND, b"")
    signaller = TemporalReviewSignaller(_StubLazyClient(stub))

    with pytest.raises(WorkflowSignalError, match="workflow not found"):
        await signaller.signal_review(InvoiceId(SHA), approved=True)


async def test_unset_temporal_address_fails_loudly() -> None:
    signaller = get_review_signaller(_settings(None))
    assert isinstance(signaller, UnconfiguredReviewSignaller)
    with pytest.raises(ConfigError):
        await signaller.signal_review(InvoiceId(SHA), approved=True)
    with pytest.raises(ConfigError):
        await get_workflow_starter(_settings(None)).start_invoice_workflow(_document())


def test_configured_address_wires_temporal_implementations() -> None:
    assert isinstance(get_review_signaller(_settings("temporal:7233")), TemporalReviewSignaller)
    assert isinstance(get_workflow_starter(_settings("temporal:7233")), TemporalWorkflowStarter)
