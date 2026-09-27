"""M11.1 audit (REVIEW.md F-02, F-03): the review decisions must respect the invoice state
machine. A rejected invoice can never be approved afterwards, and `send_to_review` never pulls
an invoice out of a terminal or in-flight payment state (SENT, ARCHIVED, BLOCKED, FAILED,
APPROVED) back into the review queue, where a later approval would write a second payment
decision for it. Runs against a throwaway Postgres like `test_api_authz.py`, whose helpers it
reuses, on its own container name and port so the two modules can run side by side."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

import apx.db.session as db_session
from apx.api.settings import get_settings
from apx.common.types import Role, Route
from apx.db.enums import DbInvoiceStatus
from apx.db.models import Approval, Invoice
from apx.db.session import get_session
from apx.ingest.settings import get_ingest_settings
from tests.integration.test_api_authz import (
    APPROVAL_LIMIT,
    PG_DB,
    PG_PASSWORD,
    PG_USER,
    _alembic_config,
    _app,
    _auth,
    _create_invoice,
    _create_user,
    _create_verified_vendor,
    _invoice_id,
    _login,
    _RecordingSignaller,
    _seed_roles,
    _wait_for_postgres,
)

pytestmark = pytest.mark.integration

CONTAINER_NAME = "apx-test-postgres-review-sm"
HOST_PORT = 55450
DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
IBAN = "AE070331234567890123456"


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
    try:
        asyncio.run(_wait_for_postgres(ASYNCPG_DSN))
        yield DATABASE_URL
    finally:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture
def api_schema(postgres_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[str]:
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    monkeypatch.setenv("API_JWT_SECRET", "integration-test-secret-value-long-enough-for-hs256")
    monkeypatch.setenv("API_APPROVAL_LIMIT", str(APPROVAL_LIMIT))
    monkeypatch.setenv("API_SOD_COOLDOWN_DAYS", "30")
    monkeypatch.setenv("INGEST_STORE_DIR", str(tmp_path / "documents"))
    db_session.get_engine.cache_clear()
    db_session._session_factory.cache_clear()
    get_settings.cache_clear()
    get_ingest_settings.cache_clear()
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    try:
        yield postgres_url
    finally:
        command.downgrade(cfg, "base")
        db_session.get_engine.cache_clear()
        db_session._session_factory.cache_clear()
        get_settings.cache_clear()
        get_ingest_settings.cache_clear()


async def _seed(invoices: dict[str, tuple[Route, DbInvoiceStatus]]) -> None:
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="rev", password="pw", roles=[Role.REVIEWER])
        await _create_user(session, user_id="adm", password="pw", roles=[Role.ADMIN])
        await _create_verified_vendor(
            session,
            vendor_id="vendorSM",
            iban=IBAN,
            verified_by="adm",
            changed_by="adm",
            changed_at=datetime.now(UTC) - timedelta(days=365),
        )
        for invoice_id, (route, status) in invoices.items():
            await _create_invoice(
                session,
                invoice_id=invoice_id,
                vendor_id="vendorSM",
                total=Decimal("100.00"),
                route=route,
                status=status,
            )


async def _approvals(invoice_id: str) -> int:
    async with get_session() as session:
        rows = await session.execute(select(Approval.id).where(Approval.invoice_id == invoice_id))
        return len(rows.all())


async def _reject_then_approve_scenario() -> None:
    invoice_id = _invoice_id("sm-reject-then-approve")
    await _seed({invoice_id: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW)})
    signaller = _RecordingSignaller()
    transport = ASGITransport(app=_app(signaller))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")
        rejected = await client.post(
            f"/invoices/{invoice_id}/decisions", json={"action": "reject"}, headers=_auth(token)
        )
        assert rejected.status_code == 204, rejected.text
        approved = await client.post(
            f"/invoices/{invoice_id}/decisions", json={"action": "approve"}, headers=_auth(token)
        )
    async with get_session() as session:
        row = await session.get(Invoice, invoice_id)
        assert row is not None
        status = row.status
    count = await _approvals(invoice_id)
    await db_session.get_engine().dispose()
    assert approved.status_code >= 400, (
        f"a rejected invoice was approved afterwards ({approved.status_code}); "
        f"status={status.value}, approval rows={count}"
    )
    assert count == 0
    assert status is not DbInvoiceStatus.AWAITING_REVIEW, "a rejected invoice stays in review"


def test_a_rejected_invoice_cannot_be_approved_afterwards(api_schema: str) -> None:
    asyncio.run(_reject_then_approve_scenario())


_SETTLED = {
    "sent": (Route.AUTO, DbInvoiceStatus.SENT),
    "archived": (Route.AUTO, DbInvoiceStatus.ARCHIVED),
    "failed": (Route.REVIEW, DbInvoiceStatus.FAILED),
    "approved": (Route.AUTO, DbInvoiceStatus.APPROVED),
}


async def _send_to_review_scenario() -> None:
    ids = {name: _invoice_id(f"sm-settled-{name}") for name in _SETTLED}
    await _seed({ids[name]: spec for name, spec in _SETTLED.items()})
    signaller = _RecordingSignaller()
    transport = ASGITransport(app=_app(signaller))
    after: dict[str, tuple[DbInvoiceStatus, int]] = {}
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")
        for name, invoice_id in ids.items():
            await client.post(
                f"/invoices/{invoice_id}/decisions",
                json={"action": "send_to_review"},
                headers=_auth(token),
            )
            await client.post(
                f"/invoices/{invoice_id}/decisions",
                json={"action": "approve"},
                headers=_auth(token),
            )
            async with get_session() as session:
                row = await session.get(Invoice, invoice_id)
                assert row is not None
                after[name] = (row.status, await _approvals(invoice_id))
    await db_session.get_engine().dispose()
    moved = {
        name: (status.value, approvals)
        for name, (status, approvals) in after.items()
        if status is not _SETTLED[name][1] or approvals != 0
    }
    assert not moved, f"send_to_review + approve moved settled invoices: {moved}"


def test_send_to_review_never_reopens_a_settled_invoice(api_schema: str) -> None:
    asyncio.run(_send_to_review_scenario())


async def _reject_in_flight_scenario() -> None:
    """The review page enables Reject for `gated`/`validated`/`fraud_checked` invoices (web
    `decisionsClosed`). The workflow only listens for the review signal while it waits for a
    reviewer, so a rejection accepted here while the invoice is still in flight is lost: if the
    route then comes back AUTO, the invoice is paid although a human rejected it."""
    ids = {
        status: _invoice_id(f"sm-in-flight-{status.value}")
        for status in (
            DbInvoiceStatus.GATED,
            DbInvoiceStatus.VALIDATED,
            DbInvoiceStatus.FRAUD_CHECKED,
        )
    }
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="rev", password="pw", roles=[Role.REVIEWER])
        await _create_user(session, user_id="adm", password="pw", roles=[Role.ADMIN])
        await _create_verified_vendor(
            session,
            vendor_id="vendorSM",
            iban=IBAN,
            verified_by="adm",
            changed_by="adm",
            changed_at=datetime.now(UTC) - timedelta(days=365),
        )
        for status, invoice_id in ids.items():
            await _create_invoice(
                session,
                invoice_id=invoice_id,
                vendor_id="vendorSM",
                total=Decimal("100.00"),
                route=None,
                status=status,
            )
    signaller = _RecordingSignaller()
    transport = ASGITransport(app=_app(signaller))
    lost: dict[str, int] = {}
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")
        for status, invoice_id in ids.items():
            response = await client.post(
                f"/invoices/{invoice_id}/decisions", json={"action": "reject"}, headers=_auth(token)
            )
            async with get_session() as session:
                row = await session.get(Invoice, invoice_id)
                assert row is not None
                stays_payable = row.status is status
            if response.status_code < 400 and stays_payable:
                lost[status.value] = response.status_code
    await db_session.get_engine().dispose()
    assert not lost, f"reject reported success but nothing stops the payment: {lost}"


def test_rejecting_an_in_flight_invoice_is_never_silently_lost(api_schema: str) -> None:
    asyncio.run(_reject_in_flight_scenario())
