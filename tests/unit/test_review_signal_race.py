"""The `route` activity commits AWAITING_REVIEW to Postgres
before the workflow's in-memory status says so (it changes only after `route` and `review_hold`
complete), and the `review` signal handler drops any signal that arrives before that. A reviewer
(or a scripted approver) acting in that window gets a 204, the approval row is written, and the
workflow keeps waiting forever with the invoice APPROVED in the database. Emulated here by
delivering the signal while `review_hold` is still running."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from temporalio import activity
from temporalio.client import Client, WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from apx.common.types import Route
from apx.policy.interfaces import RoutingDecision
from apx.workflows.interfaces import InvoiceStatus
from apx.workflows.workflow import InvoiceWorkflow, ReviewDecision
from tests.unit.test_invoice_workflow import (
    INVOICE_ID,
    TASK_QUEUE,
    WORKFLOW_INPUT,
    FakeInvoiceActivities,
    _activities,
    temporal_env,  # noqa: F401  (pytest fixture)
)


@dataclass
class _ApproveDuringHold(FakeInvoiceActivities):
    """The API's approval lands after `route` committed AWAITING_REVIEW (what the review UI and
    `_apply_approval` read) but before the workflow has processed `review_hold`'s result."""

    client: Client | None = None

    @activity.defn(name="review_hold")
    async def review_hold(self, invoice_id: str, route: Route) -> Route:
        self.calls.append("review_hold")
        assert self.client is not None
        handle = self.client.get_workflow_handle(invoice_id)
        await handle.signal(InvoiceWorkflow.review, ReviewDecision(approved=True))
        return route


@pytest.mark.asyncio
async def test_an_approval_signalled_while_routing_completes_is_not_lost(
    temporal_env: WorkflowEnvironment,  # noqa: F811
) -> None:
    fake = _ApproveDuringHold(
        route_decision=RoutingDecision(route=Route.REVIEW, reasons=()), client=temporal_env.client
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
        outcome: str
        try:
            # Time-skipping: a workflow blocked forever on the review wait runs straight into
            # the test server's execution timeout instead of hanging the test.
            outcome = (await handle.result()).status.value
        except WorkflowFailureError as exc:
            outcome = f"hung on the review wait until {type(exc.cause).__name__}"
    assert outcome == InvoiceStatus.ARCHIVED.value, (
        f"the approval signal was accepted by Temporal but dropped by the workflow ({outcome}; "
        f"confirm_review calls={fake.calls.count('confirm_review')}): the database says "
        "APPROVED and nothing ever pays or fails the invoice"
    )
