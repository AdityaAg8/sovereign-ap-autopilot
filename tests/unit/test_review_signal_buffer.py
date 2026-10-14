"""`InvoiceWorkflow` buffers a `review` signal that lands before
its review wait (the API may approve as soon as `route` commits AWAITING_REVIEW), still acts on it
only once `confirm_review` backs it, and ends FAILED when a recorded approval went stale. The
`workflow.patched` guard keeps histories recorded before the fix replaying deterministically:
an old history whose early signal was dropped is replayed against today's workflow."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from temporalio import activity, workflow
from temporalio.client import Client, WorkflowFailureError, WorkflowHistory
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker

from apx.common.invoice import Invoice
from apx.common.types import Route
from apx.fraud.interfaces import FraudReport
from apx.parse.interfaces import ParsedDocumentRef
from apx.policy.interfaces import RoutingDecision
from apx.workflows.corrections import ReviewCorrections
from apx.workflows.interfaces import InvoiceStatus, InvoiceWorkflowInput, InvoiceWorkflowState
from apx.workflows.review_guard import StaleApprovalError
from apx.workflows.workflow import REVIEW_BUFFER_PATCH, InvoiceWorkflow, ReviewDecision
from tests.unit.test_invoice_workflow import (
    INVOICE_ID,
    TASK_QUEUE,
    WORKFLOW_INPUT,
    FakeInvoiceActivities,
    _activities,
    _wait_until,
    temporal_env,  # noqa: F401  (pytest fixture)
)


@workflow.defn(name="InvoiceWorkflow")
class _PreBufferInvoiceWorkflow(InvoiceWorkflow):
    """`InvoiceWorkflow` before the signal buffer: identical except that the `review` handler
    drops any signal that arrives before the in-memory status is AWAITING_REVIEW. Used only to
    record a history the way a running pre-fix execution recorded it."""

    @workflow.run
    async def run(self, inp: InvoiceWorkflowInput) -> InvoiceWorkflowState:
        return await super().run(inp)

    @workflow.signal(name="review")
    def review(self, decision: ReviewDecision) -> None:
        if self._status is InvoiceStatus.AWAITING_REVIEW and self._review is None:
            self._review = decision


@dataclass
class _SignalDuring(FakeInvoiceActivities):
    """Sends a `review` signal from inside one pipeline activity, i.e. before the workflow has
    processed that activity's result (and so before its review wait)."""

    client: Client | None = None
    signal_in: str = "review_hold"
    approved: bool = True

    async def _signal(self, step: str) -> None:
        if step != self.signal_in:
            return
        assert self.client is not None
        handle = self.client.get_workflow_handle(INVOICE_ID)
        await handle.signal(InvoiceWorkflow.review, ReviewDecision(approved=self.approved))

    @activity.defn(name="review_hold")
    async def review_hold(self, invoice_id: str, route: Route) -> Route:
        self.calls.append("review_hold")
        await self._signal("review_hold")
        return route

    @activity.defn(name="fraud")
    async def fraud(self, invoice: Invoice, parsed: ParsedDocumentRef) -> FraudReport:
        report = await FakeInvoiceActivities.fraud(self, invoice, parsed)
        await self._signal("fraud")
        return report


def _review_route() -> RoutingDecision:
    return RoutingDecision(route=Route.REVIEW, reasons=())


async def _result_status(env: WorkflowEnvironment, fake: FakeInvoiceActivities) -> str:
    async with Worker(
        env.client, task_queue=TASK_QUEUE, workflows=[InvoiceWorkflow], activities=_activities(fake)
    ):
        handle = await env.client.start_workflow(
            InvoiceWorkflow.run, WORKFLOW_INPUT, id=INVOICE_ID, task_queue=TASK_QUEUE
        )
        try:
            return (await handle.result()).status.value
        except WorkflowFailureError as exc:
            return f"failed: {exc.cause!r}"


@pytest.mark.asyncio
async def test_an_approval_signalled_before_fraud_completes_is_buffered_for_the_wait(
    temporal_env: WorkflowEnvironment,  # noqa: F811
) -> None:
    fake = _SignalDuring(
        route_decision=_review_route(), client=temporal_env.client, signal_in="fraud"
    )
    assert await _result_status(temporal_env, fake) == InvoiceStatus.ARCHIVED.value
    # Acted on only after the review note and a backing check, exactly once.
    assert fake.calls.index("review_note") < fake.calls.index("confirm_review")
    assert fake.confirmed == [True]
    assert fake.calls.count("send_to_asp") == 1


@pytest.mark.asyncio
async def test_an_early_unbacked_signal_is_checked_and_the_invoice_keeps_waiting(
    temporal_env: WorkflowEnvironment,  # noqa: F811
) -> None:
    """The DB-backing check holds for a buffered signal: unbacked, it is dropped and a later backed
    decision is what completes the invoice."""
    fake = _SignalDuring(
        route_decision=_review_route(), client=temporal_env.client, unbacked_signals=1
    )
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[InvoiceWorkflow],
        activities=_activities(fake),
    ):
        handle = await temporal_env.client.start_workflow(
            InvoiceWorkflow.run, WORKFLOW_INPUT, id=INVOICE_ID, task_queue=TASK_QUEUE
        )

        async def _first_checked() -> bool:
            return fake.calls.count("confirm_review") == 1

        await _wait_until(_first_checked)
        state = await handle.query(InvoiceWorkflow.status)
        assert state.status is InvoiceStatus.AWAITING_REVIEW
        assert "build_einvoice" not in fake.calls
        await handle.signal(InvoiceWorkflow.review, ReviewDecision(approved=True))
        result = await handle.result()
    assert result.status is InvoiceStatus.ARCHIVED
    assert fake.confirmed == [True, True]


@pytest.mark.asyncio
async def test_an_early_signal_is_never_read_when_the_route_is_auto(
    temporal_env: WorkflowEnvironment,  # noqa: F811
) -> None:
    fake = _SignalDuring(
        route_decision=RoutingDecision(route=Route.AUTO, reasons=()),
        client=temporal_env.client,
        approved=False,
    )
    assert await _result_status(temporal_env, fake) == InvoiceStatus.ARCHIVED.value
    assert "confirm_review" not in fake.calls


@dataclass
class _StaleApproval(FakeInvoiceActivities):
    @activity.defn(name="confirm_review")
    async def confirm_review(
        self,
        invoice_id: str,
        approved: bool,
        corrections: ReviewCorrections | None = None,
        check_corrections: bool = False,
    ) -> bool:
        self.calls.append("confirm_review")
        raise StaleApprovalError("approval digest does not match the current invoice or payee")


@pytest.mark.asyncio
async def test_a_stale_recorded_approval_fails_the_invoice_with_its_reason(
    temporal_env: WorkflowEnvironment,  # noqa: F811
) -> None:
    """Not APPROVED forever. One attempt (non-retryable), then FAILED with the reason."""
    fake = _StaleApproval(route_decision=_review_route())
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[InvoiceWorkflow],
        activities=_activities(fake),
    ):
        handle = await temporal_env.client.start_workflow(
            InvoiceWorkflow.run, WORKFLOW_INPUT, id=INVOICE_ID, task_queue=TASK_QUEUE
        )

        async def _awaiting() -> bool:
            state = await handle.query(InvoiceWorkflow.status)
            return bool(state.status is InvoiceStatus.AWAITING_REVIEW)

        await _wait_until(_awaiting)
        await handle.signal(InvoiceWorkflow.review, ReviewDecision(approved=True))
        with pytest.raises(WorkflowFailureError):
            await handle.result()
        state = await handle.query(InvoiceWorkflow.status)
    assert state.status is InvoiceStatus.FAILED
    assert fake.calls.count("confirm_review") == 1
    assert "build_einvoice" not in fake.calls
    assert len(fake.failures) == 1
    assert "StaleApprovalError" in fake.failures[0][1]


async def _record_history(
    env: WorkflowEnvironment,
    workflow_cls: type[InvoiceWorkflow],
    fake: _SignalDuring,
    *,
    unsandboxed: bool,
    later_approved: bool = True,
) -> WorkflowHistory:
    """Runs one REVIEW execution whose first approval lands during `review_hold`; if the
    workflow dropped it, a second decision (`later_approved`) is sent once it waits."""
    runner = {"workflow_runner": UnsandboxedWorkflowRunner()} if unsandboxed else {}
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[workflow_cls],
        activities=_activities(fake),
        **runner,  # type: ignore[arg-type]
    ):
        handle = await env.client.start_workflow(
            InvoiceWorkflow.run, WORKFLOW_INPUT, id=INVOICE_ID, task_queue=TASK_QUEUE
        )

        async def _settled() -> bool:
            state = await handle.query(InvoiceWorkflow.status)
            return bool(
                state.status in (InvoiceStatus.ARCHIVED, InvoiceStatus.BLOCKED)
                or (state.status is InvoiceStatus.AWAITING_REVIEW and "review_note" in fake.calls)
            )

        await _wait_until(_settled)
        state = await handle.query(InvoiceWorkflow.status)
        if state.status is InvoiceStatus.AWAITING_REVIEW:
            await handle.signal(InvoiceWorkflow.review, ReviewDecision(approved=later_approved))
        await handle.result()
        return await handle.fetch_history()


def _has_marker(history: WorkflowHistory, patch_id: str) -> bool:
    return any(
        event.HasField("marker_recorded_event_attributes")
        and patch_id.encode() in event.marker_recorded_event_attributes.SerializeToString()
        for event in history.events
    )


@pytest.mark.asyncio
async def test_a_pre_fix_history_with_a_dropped_early_signal_replays_deterministically(
    temporal_env: WorkflowEnvironment,  # noqa: F811
) -> None:
    """A running execution recorded by the pre-buffer worker dropped an early approval, then
    was rejected once it waited. Today's workflow must replay that history identically (the
    `workflow.patched` guard returns False for it); acting on the early approval instead would
    go on to build the e-invoice where the history ends BLOCKED, a nondeterminism error. With
    the guard replaced by True, this replay fails (checked when the test was written)."""
    old_fake = _SignalDuring(route_decision=_review_route(), client=temporal_env.client)
    old = await _record_history(
        temporal_env, _PreBufferInvoiceWorkflow, old_fake, unsandboxed=True, later_approved=False
    )
    # The pre-fix run dropped the early approval: only the later rejection was checked.
    assert old_fake.confirmed == [False]
    assert "build_einvoice" not in old_fake.calls
    assert not _has_marker(old, REVIEW_BUFFER_PATCH)

    replayer = Replayer(workflows=[InvoiceWorkflow], data_converter=pydantic_data_converter)
    await replayer.replay_workflow(old)


@pytest.mark.asyncio
async def test_a_buffered_signal_history_records_the_patch_and_replays(
    temporal_env: WorkflowEnvironment,  # noqa: F811
) -> None:
    fake = _SignalDuring(route_decision=_review_route(), client=temporal_env.client)
    history = await _record_history(temporal_env, InvoiceWorkflow, fake, unsandboxed=False)
    assert _has_marker(history, REVIEW_BUFFER_PATCH)
    assert fake.confirmed == [True]

    replayer = Replayer(workflows=[InvoiceWorkflow], data_converter=pydantic_data_converter)
    await replayer.replay_workflow(history)
