"""Delivers a human review outcome to the invoice's Temporal `InvoiceWorkflow` (its `review`
signal, PRD §10 M5.3). The API records the decision through `apx.policy` first, then signals, so
a REVIEW-routed execution resumes instead of waiting forever while the database says approved.
A signal to a workflow that has already finished is a typed 409, not a retryable 503.
"""

from __future__ import annotations

from typing import Any, Protocol

from temporalio.client import WorkflowExecutionStatus, WorkflowHandle
from temporalio.service import RPCError, RPCStatusCode

from apx.common.errors import ApxError, ConfigError
from apx.common.types import InvoiceId
from apx.workflows.corrections import ReviewCorrections
from apx.workflows.starter import LazyTemporalClient
from apx.workflows.workflow import InvoiceWorkflow, ReviewDecision


class WorkflowSignalError(ApxError):
    """The review decision was recorded but the workflow could not be signalled (Temporal down,
    or no workflow for this invoice). Recovery is to re-send the invoice's decision (`approve` or
    `reject`): the payment-state write is idempotent and the workflow ignores a duplicate `review`
    signal. That also covers an executed approve_payment proposal, which cannot run twice."""


class WorkflowAlreadyCompletedError(ApxError):
    """The review decision was recorded, but the invoice's workflow has already finished
    (completed, failed or ended), so no workflow is waiting for it. Re-sending cannot help: the
    API answers 409 (not the retryable 503 of `WorkflowSignalError`) and the reviewer sees where
    the invoice now stands."""


class ReviewSignaller(Protocol):
    async def signal_review(
        self,
        invoice_id: InvoiceId,
        approved: bool,
        corrections: ReviewCorrections | None = None,
    ) -> None:
        """Signal the invoice's workflow (workflow id = invoice id). Raises
        `WorkflowAlreadyCompletedError` when that workflow has already finished, and
        `WorkflowSignalError` on any other delivery failure; never silently drops a decision."""
        ...


class UnconfiguredReviewSignaller:
    """Default until `API_TEMPORAL_ADDRESS` is set: fails loudly (PRD §8) rather than pretend
    the workflow was told."""

    async def signal_review(
        self,
        invoice_id: InvoiceId,
        approved: bool,
        corrections: ReviewCorrections | None = None,
    ) -> None:
        raise ConfigError(
            "API_TEMPORAL_ADDRESS is not set; cannot signal the review decision for invoice "
            f"{invoice_id} to its workflow"
        )


class TemporalReviewSignaller:
    """`ReviewSignaller` over a lazily-connected, reused Temporal `Client`."""

    def __init__(self, client: LazyTemporalClient) -> None:
        self._client = client

    async def signal_review(
        self,
        invoice_id: InvoiceId,
        approved: bool,
        corrections: ReviewCorrections | None = None,
    ) -> None:
        try:
            client = await self._client.get()
            handle = client.get_workflow_handle(invoice_id)
            try:
                await handle.signal(
                    InvoiceWorkflow.review,
                    ReviewDecision(approved=approved, corrections=corrections),
                )
            except RPCError as exc:
                # Temporal answers NOT_FOUND both for no workflow and for a closed one; only a
                # describe tells them apart (the message differs between server builds).
                if exc.status is RPCStatusCode.NOT_FOUND and await _is_closed(handle):
                    raise WorkflowAlreadyCompletedError(
                        f"the decision for invoice {invoice_id} is recorded, but its workflow "
                        "has already finished; there is nothing to notify"
                    ) from exc
                raise
        except (RPCError, RuntimeError, OSError) as exc:
            raise WorkflowSignalError(
                f"could not signal the review decision for invoice {invoice_id}: {exc}"
            ) from exc


async def _is_closed(handle: WorkflowHandle[Any, Any]) -> bool:
    """True when the workflow exists and is no longer running; False when there is none."""
    try:
        description = await handle.describe()
    except RPCError as exc:
        if exc.status is RPCStatusCode.NOT_FOUND:
            return False
        raise
    return description.status is not WorkflowExecutionStatus.RUNNING
