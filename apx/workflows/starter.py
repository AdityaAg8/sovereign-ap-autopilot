"""Temporal-backed `apx.ingest.interfaces.WorkflowStarter` and the lazily-connected client
shared with the API's review signaller: one `InvoiceWorkflow` per ingested document, with the
workflow id (= invoice id) set to the document's sha256 (PRD §3, §10 M5.2). The id is never
reused, even after its workflow has closed (REVIEW L-06), so exactly-once does not rest on the
idempotency keys alone."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

from apx.common.types import InvoiceId
from apx.ingest.interfaces import IngestedDocument
from apx.workflows.interfaces import InvoiceWorkflowInput
from apx.workflows.worker import LLM_TASK_QUEUE, TASK_QUEUE, connect_client
from apx.workflows.workflow import InvoiceWorkflow

INVOICE_ID_REUSE_POLICY = WorkflowIDReusePolicy.REJECT_DUPLICATE
"""A closed invoice workflow's id can never start a new run (Temporal's default allows it), for
as long as the namespace retains its history."""


class LazyTemporalClient:
    """Connects on first use and reuses the connection; safe under concurrent first calls."""

    def __init__(self, target_host: str) -> None:
        self._target_host = target_host
        self._client: Client | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> Client:
        async with self._lock:
            if self._client is None:
                self._client = await connect_client(self._target_host)
            return self._client


class TemporalWorkflowStarter:
    """Starts `InvoiceWorkflow` on `task_queue`. Starting a document whose workflow already
    exists -- running, or closed in any way (`REJECT_DUPLICATE`) -- is a no-op that returns the
    same id, so a retried or re-dropped ingest never runs the pipeline twice. Any other failure
    propagates; `apx.ingest.ingestor.Ingestor` wraps it in a typed `WorkflowStartError`."""

    def __init__(
        self,
        client: LazyTemporalClient,
        task_queue: str = TASK_QUEUE,
        llm_task_queue: str | None = LLM_TASK_QUEUE,
    ) -> None:
        self._client = client
        self._task_queue = task_queue
        self._llm_task_queue = llm_task_queue

    async def start_invoice_workflow(self, document: IngestedDocument) -> str:
        invoice_id = InvoiceId(document.sha256)
        client = await self._client.get()
        # Exactly once: a document whose workflow already ran or is running keeps that one.
        with suppress(WorkflowAlreadyStartedError):
            await client.start_workflow(
                InvoiceWorkflow.run,
                InvoiceWorkflowInput(
                    invoice_id=invoice_id,
                    document_sha256=document.sha256,
                    filename=document.filename,
                    llm_task_queue=self._llm_task_queue,
                ),
                id=invoice_id,
                task_queue=self._task_queue,
                id_reuse_policy=INVOICE_ID_REUSE_POLICY,
            )
        return invoice_id
