"""Delivers a human review outcome to the invoice's Temporal `InvoiceWorkflow` (its `review`
signal, PRD §10 M5.3). The API records the decision through `apx.policy` first, then signals, so
a REVIEW-routed execution resumes instead of waiting forever while the database says approved.
"""

from __future__ import annotations

from typing import Protocol

from temporalio.service import RPCError

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


class ReviewSignaller(Protocol):
    async def signal_review(
        self,
        invoice_id: InvoiceId,
        approved: bool,
        corrections: ReviewCorrections | None = None,
    ) -> None:
        """Signal the invoice's workflow (workflow id = invoice id). Raises
        `WorkflowSignalError` on any delivery failure; never silently drops a decision."""
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
            await handle.signal(
                InvoiceWorkflow.review, ReviewDecision(approved=approved, corrections=corrections)
            )
        except (RPCError, RuntimeError, OSError) as exc:
            raise WorkflowSignalError(
                f"could not signal the review decision for invoice {invoice_id}: {exc}"
            ) from exc
