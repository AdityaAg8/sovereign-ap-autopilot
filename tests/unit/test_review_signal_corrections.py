"""KNOWN_ISSUES KI-02: `InvoiceWorkflow` hands the corrections carried by a `review` signal to
`confirm_review` and asks it to check them, so the activity can refuse a signal whose corrections
are not the ones recorded with the approval. The `workflow.patched` marker keeps histories
recorded before the change replaying: a pre-change history, whose `confirm_review` got only the
invoice id and the decision, replays deterministically against today's workflow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

import pytest
from temporalio import activity, workflow
from temporalio.client import WorkflowHistory
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker

from apx.common.invoice import Invoice, InvoiceKind, VatCategory
from apx.common.types import InvoiceId, Route
from apx.policy.interfaces import RoutingDecision
from apx.workflows.corrections import LineCorrection, ReviewCorrections
from apx.workflows.interfaces import InvoiceStatus, InvoiceWorkflowInput, InvoiceWorkflowState
from apx.workflows.workflow import (
    REVIEW_GUARD_PATCH,
    REVIEW_SIGNAL_CORRECTIONS_PATCH,
    InvoiceWorkflow,
    ReviewDecision,
)
from tests.unit.test_invoice_workflow import (
    INVOICE_ID,
    TASK_QUEUE,
    WORKFLOW_INPUT,
    FakeInvoiceActivities,
    _activities,
    _wait_until,
    temporal_env,
)

__all__ = ["temporal_env"]  # the fixture, re-exported so pytest finds it here

_TIMEOUT = timedelta(minutes=5)

CORRECTIONS = ReviewCorrections(
    kind=InvoiceKind.INVOICE,
    currency="AED",
    lines=(LineCorrection(index=1, vat_category=VatCategory.STANDARD, vat_rate=Decimal("5")),),
)


@workflow.defn(name="InvoiceWorkflow")
class _PreCorrectionsCheckWorkflow(InvoiceWorkflow):
    """`InvoiceWorkflow` before the corrections check: identical except that `confirm_review` gets
    only the invoice id and the decision. Used only to record a history the way a running
    pre-change execution recorded it."""

    @workflow.run
    async def run(self, inp: InvoiceWorkflowInput) -> InvoiceWorkflowState:
        return await super().run(inp)

    async def _await_backed_review(self, invoice_id: InvoiceId) -> ReviewDecision:
        guarded = workflow.patched(REVIEW_GUARD_PATCH)
        while True:
            await workflow.wait_condition(lambda: self._review is not None)
            decision = self._review
            assert decision is not None
            if not guarded:
                return decision
            backed: bool = await workflow.execute_activity(
                "confirm_review",
                args=[invoice_id, decision.approved],
                result_type=bool,
                start_to_close_timeout=_TIMEOUT,
            )
            if backed:
                return decision
            self._review = None


@dataclass
class _WithApplyReview(FakeInvoiceActivities):
    applied: list[ReviewCorrections] | None = None

    @activity.defn(name="apply_review")
    async def apply_review(self, invoice: Invoice, corrections: ReviewCorrections) -> Invoice:
        self.calls.append("apply_review")
        if self.applied is None:
            self.applied = []
        self.applied.append(corrections)
        return invoice


def _with_apply(fake: _WithApplyReview) -> Sequence[Callable[..., object]]:
    return [*_activities(fake), fake.apply_review]


def _fake() -> _WithApplyReview:
    return _WithApplyReview(route_decision=RoutingDecision(route=Route.REVIEW, reasons=()))


async def _record(
    env: WorkflowEnvironment,
    workflow_cls: type[InvoiceWorkflow],
    fake: _WithApplyReview,
    *,
    unsandboxed: bool,
) -> WorkflowHistory:
    """One REVIEW execution approved with `CORRECTIONS` once it waits for review."""
    runner = {"workflow_runner": UnsandboxedWorkflowRunner()} if unsandboxed else {}
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[workflow_cls],
        activities=_with_apply(fake),
        **runner,  # type: ignore[arg-type]
    ):
        handle = await env.client.start_workflow(
            InvoiceWorkflow.run, WORKFLOW_INPUT, id=INVOICE_ID, task_queue=TASK_QUEUE
        )

        async def _waiting() -> bool:
            state = await handle.query(InvoiceWorkflow.status)
            waiting = state.status is InvoiceStatus.AWAITING_REVIEW
            return bool(waiting and "review_note" in fake.calls)

        await _wait_until(_waiting)
        await handle.signal(
            InvoiceWorkflow.review, ReviewDecision(approved=True, corrections=CORRECTIONS)
        )
        result = await handle.result()
        assert result.status is InvoiceStatus.ARCHIVED
        return await handle.fetch_history()


def _has_marker(history: WorkflowHistory, patch_id: str) -> bool:
    return any(
        event.HasField("marker_recorded_event_attributes")
        and patch_id.encode() in event.marker_recorded_event_attributes.SerializeToString()
        for event in history.events
    )


@pytest.mark.asyncio
async def test_confirm_review_receives_the_signal_corrections_and_is_asked_to_check_them(
    temporal_env: WorkflowEnvironment,
) -> None:
    fake = _fake()
    history = await _record(temporal_env, InvoiceWorkflow, fake, unsandboxed=False)
    # The activity sees exactly what `apply_review` then applies, and is told to check it.
    assert fake.confirm_corrections == [(CORRECTIONS, True)]
    assert fake.applied == [CORRECTIONS]
    assert fake.calls.index("confirm_review") < fake.calls.index("apply_review")
    assert _has_marker(history, REVIEW_SIGNAL_CORRECTIONS_PATCH)

    replayer = Replayer(workflows=[InvoiceWorkflow], data_converter=pydantic_data_converter)
    await replayer.replay_workflow(history)


@pytest.mark.asyncio
async def test_a_rejection_passes_no_corrections_to_check(
    temporal_env: WorkflowEnvironment,
) -> None:
    fake = _fake()
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[InvoiceWorkflow],
        activities=_with_apply(fake),
    ):
        handle = await temporal_env.client.start_workflow(
            InvoiceWorkflow.run, WORKFLOW_INPUT, id=INVOICE_ID, task_queue=TASK_QUEUE
        )

        async def _waiting() -> bool:
            state = await handle.query(InvoiceWorkflow.status)
            waiting = state.status is InvoiceStatus.AWAITING_REVIEW
            return bool(waiting and "review_note" in fake.calls)

        await _wait_until(_waiting)
        await handle.signal(InvoiceWorkflow.review, ReviewDecision(approved=False))
        result = await handle.result()
    assert result.status is InvoiceStatus.BLOCKED
    assert fake.confirm_corrections == [(None, True)]
    assert "apply_review" not in fake.calls


@pytest.mark.asyncio
async def test_a_pre_change_history_replays_deterministically(
    temporal_env: WorkflowEnvironment,
) -> None:
    """A history from a worker without the check scheduled `confirm_review` with two arguments
    and no marker. Today's workflow must replay it (the `workflow.patched` guard returns False
    for it) without the new arguments' marker being expected."""
    old_fake = _fake()
    old = await _record(temporal_env, _PreCorrectionsCheckWorkflow, old_fake, unsandboxed=True)
    # The pre-change worker never asked the activity to check the corrections.
    assert old_fake.confirm_corrections == [(None, False)]
    assert not _has_marker(old, REVIEW_SIGNAL_CORRECTIONS_PATCH)

    replayer = Replayer(workflows=[InvoiceWorkflow], data_converter=pydantic_data_converter)
    await replayer.replay_workflow(old)
