"""`TemporalWorkflowStarter` id reuse (REVIEW L-06, PRD §6.4): against Temporal's time-skipping
test server, re-ingesting a document whose invoice workflow has already closed returns the same
id and starts no new run, and a running one is left alone. No worker runs, so the started
workflow simply waits on its first activity until the test terminates it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.common import WorkflowIDReusePolicy
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

from apx.common.types import Sha256Hex
from apx.ingest.interfaces import IngestedDocument
from apx.workflows.starter import (
    INVOICE_ID_REUSE_POLICY,
    LazyTemporalClient,
    TemporalWorkflowStarter,
)

TASK_QUEUE = "starter-test-no-worker"


class _EnvClient(LazyTemporalClient):
    """`LazyTemporalClient` over an already-connected test-server client."""

    def __init__(self, client: Client) -> None:
        super().__init__("unused:0")
        self._client = client


def _document(sha: str) -> IngestedDocument:
    return IngestedDocument(
        sha256=Sha256Hex(sha),
        filename="invoice.pdf",
        media_type="application/pdf",
        size_bytes=100,
        received_at=datetime.now(UTC),
        source="upload",
    )


async def test_a_closed_invoice_workflow_id_is_never_started_again() -> None:
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        starter = TemporalWorkflowStarter(
            _EnvClient(env.client), task_queue=TASK_QUEUE, llm_task_queue=None
        )
        document = _document("c" * 64)

        workflow_id = await starter.start_invoice_workflow(document)
        handle = env.client.get_workflow_handle(workflow_id)
        first_run = (await handle.describe()).run_id

        # Running: a second ingest keeps the running workflow.
        assert await starter.start_invoice_workflow(document) == workflow_id
        assert (await handle.describe()).run_id == first_run

        # Closed: Temporal's default policy would start a fresh run here; ours must not.
        await handle.terminate(reason="test")
        assert await starter.start_invoice_workflow(document) == workflow_id
        latest = await env.client.get_workflow_handle(workflow_id).describe()
        assert latest.run_id == first_run
        assert latest.status is WorkflowExecutionStatus.TERMINATED


def test_the_policy_is_reject_duplicate() -> None:
    assert INVOICE_ID_REUSE_POLICY is WorkflowIDReusePolicy.REJECT_DUPLICATE
