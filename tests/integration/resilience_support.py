"""Shared harness for the resilience tests (`test_resilience_llama_swap.py`,
`test_resilience_postgres.py`): a throwaway, migrated pgvector Postgres container the test can
stop, start and restart mid-workflow, the `InvoiceActivityDeps` wiring over the M5.2 crash-test
fakes, and helpers that read the exactly-once evidence (history attempts, idempotency rows)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine as _create_engine
from temporalio.client import WorkflowHandle

from apx.asp_mock.interfaces import Archive, AspClient
from apx.common.types import InvoiceId, Language
from apx.db.enums import DbInvoiceStatus
from apx.db.models import IdempotencyRecord
from apx.db.models import Invoice as InvoiceRow
from apx.parse.interfaces import DocumentParser
from apx.parse.store import FileParsedDocumentStore
from apx.workflows.activities import InvoiceActivityDeps
from apx.workflows.idempotency import SqlIdempotencyStore
from apx.workflows.ports import ExtractionPipeline
from apx.workflows.sql import SqlInvoiceRecords
from tests.integration.test_workflow_crash import (
    FakeAuditLog,
    FakeDocumentStore,
    FakeFeatureBuilder,
    FakeFraudEngine,
    FakeInvoiceBuilder,
    FakePolicyEngine,
    FakeScanner,
    FakeSchematronValidator,
    FakeTrustGate,
    FakeUblBuilder,
    FakeValidator,
    FakeVendorDirectory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PG_IMAGE = "pgvector/pgvector:pg16-trixie"
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["docker", *args], capture_output=True, check=check)


async def wait_for_postgres(dsn: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = await asyncpg.connect(dsn, timeout=5)
        except (TimeoutError, OSError, asyncpg.PostgresError) as exc:
            last_error = exc
            await asyncio.sleep(0.25)
            continue
        try:
            await conn.execute("SELECT 1")
        finally:
            await conn.close()
        return
    raise TimeoutError(f"Postgres did not become ready within {timeout_s}s") from last_error


@dataclass(frozen=True)
class PgContainer:
    """One named Postgres container on a fixed host port. Deliberately not `--rm`: a stopped
    container keeps its data directory, so `stop` + `start` is a real database restart."""

    name: str
    port: int

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{self.port}/{PG_DB}"

    @property
    def dsn(self) -> str:
        return f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{self.port}/{PG_DB}"

    def create(self) -> None:
        """Starts a fresh container and migrates it to head."""
        self.remove()
        _docker(
            "run",
            "-d",
            "--name",
            self.name,
            "-p",
            f"127.0.0.1:{self.port}:5432",
            "-e",
            f"POSTGRES_USER={PG_USER}",
            "-e",
            f"POSTGRES_PASSWORD={PG_PASSWORD}",
            "-e",
            f"POSTGRES_DB={PG_DB}",
            PG_IMAGE,
        )
        asyncio.run(wait_for_postgres(self.dsn))
        previous = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = self.url
        try:
            cfg = Config(str(REPO_ROOT / "alembic.ini"))
            cfg.set_main_option("script_location", str(REPO_ROOT / "db" / "migrations"))
            command.upgrade(cfg, "head")
        finally:
            if previous is None:
                del os.environ["DATABASE_URL"]
            else:
                os.environ["DATABASE_URL"] = previous

    def stop(self) -> None:
        """Stops Postgres (SIGTERM, fast shutdown: every open connection is terminated)."""
        _docker("stop", "-t", "10", self.name)

    async def start(self) -> None:
        _docker("start", self.name)
        await wait_for_postgres(self.dsn)

    async def restart(self) -> None:
        _docker("restart", "-t", "10", self.name)
        await wait_for_postgres(self.dsn)

    async def ensure_running(self) -> None:
        """Brings the container back if a failed test left it stopped."""
        state = _docker("inspect", "-f", "{{.State.Running}}", self.name, check=False)
        if state.stdout.strip() != b"true":
            await self.start()

    def remove(self) -> None:
        _docker("rm", "-f", "-v", self.name, check=False)


def production_engine(url: str) -> AsyncEngine:
    """The engine configuration `apx.db.session.get_engine` uses (`pool_pre_ping=True`), so a
    pooled connection that a Postgres restart killed is detected the way the worker detects it."""
    return _create_engine(url, pool_pre_ping=True)


@asynccontextmanager
async def session_factory_for(url: str) -> AsyncIterator[SessionFactory]:
    """Commits on clean exit and rolls back on error, like `apx.db.session.get_session`."""
    engine = production_engine(url)
    maker = async_sessionmaker(bind=engine, expire_on_commit=False)

    @asynccontextmanager
    async def factory() -> AsyncIterator[AsyncSession]:
        async with maker() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    try:
        yield factory
    finally:
        await engine.dispose()


async def seed_invoice(session_factory: SessionFactory, invoice_id: InvoiceId) -> None:
    """The `invoices` row ingest creates before the workflow; `archive` updates its status."""
    async with session_factory() as session:
        session.add(
            InvoiceRow(
                id=invoice_id,
                document_sha256=invoice_id,
                language=Language.EN,
                number="INV-1",
                issue_date=date(2026, 1, 1),
                currency="AED",
                subtotal=Decimal("100.00"),
                vat_total=Decimal("5.00"),
                total=Decimal("105.00"),
                status=DbInvoiceStatus.RECEIVED,
            )
        )


def activity_deps(
    session_factory: SessionFactory,
    parsed_dir: Path,
    *,
    parser: DocumentParser,
    extraction: ExtractionPipeline,
    asp_client: AspClient,
    archive: Archive,
    audit: FakeAuditLog,
) -> InvoiceActivityDeps:
    """Real Postgres-backed records and idempotency store; the named collaborators are the
    test's own, every other pipeline stage is an M5.2 crash-test fake."""
    return InvoiceActivityDeps(
        documents=FakeDocumentStore(),
        scanner=FakeScanner(),
        parser=parser,
        parsed=FileParsedDocumentStore(parsed_dir),
        extraction=extraction,
        invoices=FakeInvoiceBuilder(),
        feature_builder=FakeFeatureBuilder(),
        trust_gate=FakeTrustGate(),
        validator=FakeValidator(),
        fraud_engine=FakeFraudEngine(),
        vendors=FakeVendorDirectory(),
        policy_engine=FakePolicyEngine(),
        ubl_builder=FakeUblBuilder(),
        schematron=FakeSchematronValidator(),
        asp_client=asp_client,
        archive=archive,
        audit=audit,
        records=SqlInvoiceRecords(session_factory),
        session_factory=session_factory,
        idempotency=SqlIdempotencyStore(),
    )


async def final_attempt(handle: WorkflowHandle[Any, Any], activity: str) -> int:
    """The attempt number of `activity`'s last started attempt, from the workflow history
    (Temporal records one started event per activity, carrying the final attempt count)."""
    history = await handle.fetch_history()
    scheduled = {
        event.event_id
        for event in history.events
        if event.HasField("activity_task_scheduled_event_attributes")
        and event.activity_task_scheduled_event_attributes.activity_type.name == activity
    }
    attempts = [
        event.activity_task_started_event_attributes.attempt
        for event in history.events
        if event.HasField("activity_task_started_event_attributes")
        and event.activity_task_started_event_attributes.scheduled_event_id in scheduled
    ]
    if not attempts:
        raise AssertionError(f"activity {activity!r} never started")
    return max(attempts)


async def idempotency_rows(
    session_factory: SessionFactory, invoice_id: InvoiceId, step: str
) -> int:
    async with session_factory() as session:
        rows = await session.scalars(
            select(IdempotencyRecord).where(
                IdempotencyRecord.invoice_id == invoice_id, IdempotencyRecord.step == step
            )
        )
        return len(rows.all())


async def invoice_status(session_factory: SessionFactory, invoice_id: InvoiceId) -> DbInvoiceStatus:
    async with session_factory() as session:
        row = await session.get(InvoiceRow, invoice_id)
        if row is None:
            raise AssertionError(f"invoice {invoice_id} has no row")
        return row.status
