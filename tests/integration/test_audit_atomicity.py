"""Integration tests for audit atomicity (I4) on a real, throwaway
Postgres: an audit entry appended inside a transaction that rolls back leaves no phantom row, and
when `send_to_asp` or `archive` fails at its final audit append, the retry writes that event
exactly once (the idempotency record rolled back with it). The audit log is the real `PgAuditLog`
behind a wrapper that injects one failure after its row is flushed. Marked `integration`."""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import asyncpg  # type: ignore[import-untyped]
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import apx.db.session as db_session
from apx.asp_mock.interfaces import ArchiveRecord, AspReceipt, AspSubmission
from apx.audit.chain import PgAuditLog
from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.audit.verify import PgAuditVerifier
from apx.common.errors import AuditChainError
from apx.common.types import IdempotencyKey, InvoiceId, Language, make_idempotency_key
from apx.db.enums import DbInvoiceStatus
from apx.db.models import AuditLog as AuditLogRow
from apx.db.models import IdempotencyRecord
from apx.db.models import Invoice as InvoiceRow
from apx.einvoice.interfaces import DocumentTypeCode, EInvoiceDocument
from apx.workflows.activities import DefaultInvoiceActivities, InvoiceActivityDeps
from apx.workflows.idempotency import SqlIdempotencyStore
from apx.workflows.sql import SqlInvoiceRecords

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-audit-atomicity"
HOST_PORT = 55446
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def _alembic_config() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "db" / "migrations"))
    return cfg


async def _wait_for_postgres(dsn: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = await asyncpg.connect(dsn)
        except (OSError, asyncpg.PostgresError) as exc:
            last_error = exc
            await asyncio.sleep(0.5)
            continue
        await conn.close()
        return
    raise TimeoutError(f"Postgres did not become ready within {timeout_s}s") from last_error


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, check=False)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"127.0.0.1:{HOST_PORT}:5432",
            "-e",
            f"POSTGRES_USER={PG_USER}",
            "-e",
            f"POSTGRES_PASSWORD={PG_PASSWORD}",
            "-e",
            f"POSTGRES_DB={PG_DB}",
            "pgvector/pgvector:pg16-trixie",
        ],
        check=True,
        capture_output=True,
    )
    previous = os.environ.get("DATABASE_URL")
    try:
        asyncio.run(_wait_for_postgres(ASYNCPG_DSN))
        os.environ["DATABASE_URL"] = DATABASE_URL
        db_session.get_engine.cache_clear()
        db_session._session_factory.cache_clear()
        cfg = _alembic_config()
        try:
            command.upgrade(cfg, "head")
            yield DATABASE_URL
        finally:
            command.downgrade(cfg, "base")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        db_session.get_engine.cache_clear()
        db_session._session_factory.cache_clear()
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture
async def session_factory(postgres_url: str) -> AsyncIterator[SessionFactory]:
    """Commits on clean exit, rolls back on error (`apx.db.session.get_session`'s contract)."""
    engine = create_async_engine(postgres_url)
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
        # `PgAuditLog.append` uses the process engine; bind it afresh to the next test's loop.
        db_session.get_engine.cache_clear()
        db_session._session_factory.cache_clear()


class FailOnceAuditLog:
    """The real `PgAuditLog`, except the first `append_in` of `fail_action` raises right after
    its row is flushed -- the worst moment: the row exists in the transaction when it fails."""

    def __init__(self, fail_action: str) -> None:
        self._inner = PgAuditLog()
        self._fail_action = fail_action
        self.failures = 0

    async def append(self, event: AuditEvent) -> AuditEntry:
        return await self._inner.append(event)

    async def append_in(self, session: AsyncSession, event: AuditEvent) -> AuditEntry:
        entry = await self._inner.append_in(session, event)
        if event.action == self._fail_action and self.failures == 0:
            self.failures += 1
            raise AuditChainError("injected audit failure after the row was flushed")
        return entry


class KeyedAsp:
    """ASP and archive fake, idempotent by key like `apx.asp_mock.service.MockAsp`."""

    def __init__(self) -> None:
        self.receipts: dict[str, AspReceipt] = {}
        self.records: dict[str, ArchiveRecord] = {}
        self.submit_calls = 0
        self.store_calls = 0

    async def submit(self, submission: AspSubmission) -> AspReceipt:
        self.submit_calls += 1
        key = submission.idempotency_key
        if key not in self.receipts:
            self.receipts[key] = AspReceipt(
                receipt_id=f"receipt-{len(self.receipts) + 1}",
                idempotency_key=key,
                status="accepted",
                received_at=datetime.now(UTC),
                message="ok",
            )
        return self.receipts[key]

    async def store(self, key: IdempotencyKey, document: EInvoiceDocument) -> ArchiveRecord:
        self.store_calls += 1
        if key not in self.records:
            self.records[key] = ArchiveRecord(
                idempotency_key=key,
                xml_sha256=document.xml_sha256,
                archived_at=datetime.now(UTC),
                retain_until=date(2033, 1, 1),
            )
        return self.records[key]


async def _seed_invoice(session_factory: SessionFactory, name: str) -> InvoiceId:
    invoice_id = InvoiceId(hashlib.sha256(name.encode()).hexdigest())
    async with session_factory() as session:
        session.add(
            InvoiceRow(
                id=invoice_id,
                document_sha256=invoice_id,
                language=Language.EN,
                number=name,
                issue_date=date(2026, 1, 1),
                currency="AED",
                subtotal=Decimal("100.00"),
                vat_total=Decimal("5.00"),
                total=Decimal("105.00"),
                status=DbInvoiceStatus.APPROVED,
            )
        )
    return invoice_id


def _document(invoice_id: InvoiceId) -> EInvoiceDocument:
    xml = f"<Invoice>{invoice_id}</Invoice>".encode()
    return EInvoiceDocument(
        invoice_id=invoice_id,
        type_code=DocumentTypeCode.INVOICE,
        xml=xml,
        xml_sha256=hashlib.sha256(xml).hexdigest(),
        spec_version="1.0",
    )


def _activities(
    session_factory: SessionFactory, audit: FailOnceAuditLog, asp: KeyedAsp
) -> DefaultInvoiceActivities:
    unused = cast(Any, None)
    return DefaultInvoiceActivities(
        InvoiceActivityDeps(
            documents=unused,
            scanner=unused,
            parser=unused,
            parsed=unused,
            extraction=unused,
            invoices=unused,
            feature_builder=unused,
            trust_gate=unused,
            validator=unused,
            fraud_engine=unused,
            vendors=unused,
            policy_engine=unused,
            ubl_builder=unused,
            schematron=unused,
            asp_client=asp,
            archive=asp,
            audit=audit,
            records=SqlInvoiceRecords(session_factory),
            session_factory=session_factory,
            idempotency=SqlIdempotencyStore(),
        )
    )


async def _audit_actions(session_factory: SessionFactory, invoice_id: str) -> list[str]:
    async with session_factory() as session:
        rows = await session.scalars(
            select(AuditLogRow.action)
            .where(AuditLogRow.entity == f"invoice:{invoice_id}")
            .order_by(AuditLogRow.id)
        )
        return list(rows)


async def _idempotency_rows(session_factory: SessionFactory, invoice_id: str, step: str) -> int:
    async with session_factory() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(IdempotencyRecord)
            .where(IdempotencyRecord.invoice_id == invoice_id, IdempotencyRecord.step == step)
        )
        return int(count or 0)


async def test_an_append_in_a_rolled_back_transaction_leaves_no_phantom_entry(
    session_factory: SessionFactory,
) -> None:
    audit = PgAuditLog()
    with pytest.raises(RuntimeError, match="outer write failed"):
        async with session_factory() as session:
            await audit.append_in(
                session, AuditEvent(actor="t", action="phantom", entity="x:1", payload={})
            )
            raise RuntimeError("outer write failed")
    async with session_factory() as session:
        await audit.append_in(
            session, AuditEvent(actor="t", action="kept", entity="x:2", payload={})
        )

    async with session_factory() as session:
        actions = list(await session.scalars(select(AuditLogRow.action)))
    assert "phantom" not in actions
    assert "kept" in actions
    assert (await PgAuditVerifier().verify()).ok


async def test_send_to_asp_retried_after_a_failed_audit_append_audits_exactly_once(
    session_factory: SessionFactory,
) -> None:
    invoice_id = await _seed_invoice(session_factory, "INV-ASP")
    audit = FailOnceAuditLog("activity:send_to_asp")
    asp = KeyedAsp()
    activities = _activities(session_factory, audit, asp)
    key = make_idempotency_key(invoice_id, "asp_send")
    document = _document(invoice_id)

    with pytest.raises(AuditChainError):
        await activities.send_to_asp(key, document)
    # The idempotency record rolled back with the audit entry, so nothing claims it was sent.
    assert await _idempotency_rows(session_factory, invoice_id, "asp_send") == 0
    assert await _audit_actions(session_factory, invoice_id) == []

    first = await activities.send_to_asp(key, document)  # Temporal's retry
    replay = await activities.send_to_asp(key, document)  # a later retry: cached path

    assert first == replay
    assert len(asp.receipts) == 1  # the ASP holds one submission for the key
    assert await _idempotency_rows(session_factory, invoice_id, "asp_send") == 1
    assert await _audit_actions(session_factory, invoice_id) == ["activity:send_to_asp"]
    async with session_factory() as session:
        row = await session.get(InvoiceRow, invoice_id)
        assert row is not None and row.status is DbInvoiceStatus.SENT
    assert (await PgAuditVerifier().verify()).ok


async def test_archive_retried_after_a_failed_audit_append_audits_exactly_once(
    session_factory: SessionFactory,
) -> None:
    invoice_id = await _seed_invoice(session_factory, "INV-ARCHIVE")
    audit = FailOnceAuditLog("activity:archive")
    asp = KeyedAsp()
    activities = _activities(session_factory, audit, asp)
    key = make_idempotency_key(invoice_id, "archive")
    document = _document(invoice_id)

    with pytest.raises(AuditChainError):
        await activities.archive(key, document)
    assert await _idempotency_rows(session_factory, invoice_id, "archive") == 0
    async with session_factory() as session:
        row = await session.get(InvoiceRow, invoice_id)
        assert row is not None and row.status is DbInvoiceStatus.APPROVED  # rolled back too

    first = await activities.archive(key, document)
    replay = await activities.archive(key, document)

    assert first == replay
    assert len(asp.records) == 1
    assert await _idempotency_rows(session_factory, invoice_id, "archive") == 1
    assert await _audit_actions(session_factory, invoice_id) == ["activity:archive"]
    async with session_factory() as session:
        row = await session.get(InvoiceRow, invoice_id)
        assert row is not None and row.status is DbInvoiceStatus.ARCHIVED
    assert (await PgAuditVerifier().verify()).ok


async def test_mark_failed_audits_in_the_same_transaction_as_its_status_write(
    session_factory: SessionFactory,
) -> None:
    invoice_id = await _seed_invoice(session_factory, "INV-FAILED")
    audit = FailOnceAuditLog("activity:failed")
    activities = _activities(session_factory, audit, KeyedAsp())

    with pytest.raises(AuditChainError):
        await activities.mark_failed(invoice_id, "boom")
    async with session_factory() as session:
        row = await session.get(InvoiceRow, invoice_id)
        assert row is not None and row.status is DbInvoiceStatus.APPROVED  # not FAILED yet

    await activities.mark_failed(invoice_id, "boom")
    async with session_factory() as session:
        row = await session.get(InvoiceRow, invoice_id)
        assert row is not None and row.status is DbInvoiceStatus.FAILED
    assert await _audit_actions(session_factory, invoice_id) == ["activity:failed"]
