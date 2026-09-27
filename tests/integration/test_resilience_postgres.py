"""PRD §11.2 "DB restart": Postgres goes down while `send_to_asp` or `archive` is in flight (after
the external call, before its idempotency row is written) and comes back; the retried activity
must still leave one ASP receipt, one archive record and one ARCHIVED transition. A third case
restarts Postgres under the worker's pooled connections and checks the pool recovers."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Literal

import pytest
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

from apx.asp_mock.interfaces import AspReceipt, AspSubmission
from apx.common.types import InvoiceId, Route
from apx.db.enums import DbInvoiceStatus
from apx.workflows.interfaces import InvoiceStatus, InvoiceWorkflowInput
from apx.workflows.worker import TASK_QUEUE, build_worker
from apx.workflows.workflow import InvoiceWorkflow
from tests.integration.resilience_support import (
    PgContainer,
    SessionFactory,
    activity_deps,
    final_attempt,
    idempotency_rows,
    invoice_status,
    seed_invoice,
    session_factory_for,
)
from tests.integration.test_workflow_crash import (
    FakeArchive,
    FakeAspClient,
    FakeAuditLog,
    FakeExtractionPipeline,
    FakeParser,
)

pytestmark = pytest.mark.integration

PG = PgContainer(name="apx-test-postgres-resilience-db", port=55442)
OUTAGE_S = 2.0
"""How long Postgres stays stopped after the in-flight call is released: long enough that at
least one retry finds it down, well inside the default policy's ~15 s of retry backoff."""

Step = Literal["send_to_asp", "archive"]
Mode = Literal["outage", "restart"]


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    PG.create()
    try:
        yield PG.url
    finally:
        PG.remove()


@pytest.fixture
async def session_factory(postgres_url: str) -> AsyncIterator[SessionFactory]:
    async with session_factory_for(postgres_url) as factory:
        yield factory
    await PG.ensure_running()  # a failed case must not leave the next one without a database


class GatedAspClient(FakeAspClient):
    """`FakeAspClient` (exactly-once by key, like `apx.asp_mock`) whose first `submit` holds
    until the test releases it, so the database can go away while the call is in flight."""

    def __init__(self, entered: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__()
        self._entered = entered
        self._release = release

    async def submit(self, submission: AspSubmission) -> AspReceipt:
        if self.calls == 0:
            self._entered.set()
            await self._release.wait()
        return await super().submit(submission)


@pytest.mark.parametrize(
    ("step", "mode"),
    [("send_to_asp", "outage"), ("archive", "outage"), ("send_to_asp", "restart")],
)
async def test_postgres_restart_mid_workflow_completes_exactly_once(
    session_factory: SessionFactory, tmp_path: Path, step: Step, mode: Mode
) -> None:
    invoice_id = InvoiceId(hashlib.sha256(f"resilience-db-{step}-{mode}".encode()).hexdigest())
    await seed_invoice(session_factory, invoice_id)

    entered = asyncio.Event()
    release = asyncio.Event()
    open_gate = asyncio.Event()
    open_gate.set()
    if step == "send_to_asp":
        asp_client: FakeAspClient = GatedAspClient(entered, release)
        archive = FakeArchive(asyncio.Event(), open_gate)
    else:
        asp_client = FakeAspClient()
        archive = FakeArchive(entered, release)  # its first `store` holds the same way
    audit = FakeAuditLog()
    deps = activity_deps(
        session_factory,
        tmp_path / "parsed",
        parser=FakeParser(),
        extraction=FakeExtractionPipeline(),
        asp_client=asp_client,
        archive=archive,
        audit=audit,
    )

    async with (
        await WorkflowEnvironment.start_local(
            data_converter=pydantic_data_converter,
            dev_server_existing_path=os.environ.get("TEMPORAL_DEV_SERVER_PATH"),
        ) as env,
        build_worker(env.client, deps, task_queue=TASK_QUEUE),
    ):
        handle = await env.client.start_workflow(
            InvoiceWorkflow.run,
            InvoiceWorkflowInput(
                invoice_id=invoice_id, document_sha256=invoice_id, filename="invoice.pdf"
            ),
            id=invoice_id,
            task_queue=TASK_QUEUE,
        )
        await asyncio.wait_for(entered.wait(), timeout=30)
        # `step` has read its idempotency key (none yet) and is inside the external call.
        if mode == "outage":
            await asyncio.to_thread(PG.stop)
            release.set()  # the call returns while the database is down: its write fails
            await asyncio.sleep(OUTAGE_S)
            await PG.start()
        else:
            await PG.restart()  # every pooled connection now points at a dead backend
            release.set()
        result = await asyncio.wait_for(handle.result(), timeout=90)
        attempt = await final_attempt(handle, step)

    assert result.status is InvoiceStatus.ARCHIVED
    assert result.route is Route.AUTO
    assert result.error is None

    if mode == "outage":
        # The write after the external call failed, so the retry repeated that call under the
        # same idempotency key; the ASP/archive answered it from their own key record.
        assert attempt >= 2
        external_calls = asp_client.calls if step == "send_to_asp" else archive.calls
        assert external_calls == 2
    else:
        # `pool_pre_ping` (as in `apx.db.session.get_engine`) replaced the dead connections.
        assert attempt == 1
        assert asp_client.calls == 1

    # Exactly once, whatever was retried.
    assert len(asp_client.receipts) == 1
    assert len(archive.records) == 1
    actions = [e.action for e in audit.events]
    assert actions.count("activity:send_to_asp") == 1
    assert actions.count("activity:archive") == 1
    assert await idempotency_rows(session_factory, invoice_id, "asp_send") == 1
    assert await idempotency_rows(session_factory, invoice_id, "archive") == 1
    assert await invoice_status(session_factory, invoice_id) is DbInvoiceStatus.ARCHIVED
