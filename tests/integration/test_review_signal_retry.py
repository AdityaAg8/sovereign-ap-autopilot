"""M11.3 audit pass 2: a review decision whose Temporal signal fails (503) must
stay recoverable by re-sending the same decision, as `apx.api.signals.WorkflowSignalError`
promises. The pass-1 fix made reject write a terminal BLOCKED status before signalling, so a retried
reject now answers 409 and the waiting workflow is never told. Runs against a throwaway Postgres
like `test_review_state_machine.py`, on its own container name and port."""

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

import apx.db.session as db_session
from apx.api.settings import get_settings
from apx.api.signals import WorkflowSignalError
from apx.common.types import InvoiceId, Role, Route
from apx.db.enums import DbInvoiceStatus
from apx.db.session import get_session
from apx.ingest.settings import get_ingest_settings
from apx.workflows.corrections import ReviewCorrections
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

CONTAINER_NAME = "apx-test-postgres-review-retry"
HOST_PORT = 55460
DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
IBAN = "AE070331234567890123456"


class _FlakySignaller(_RecordingSignaller):
    """Temporal briefly unavailable: the first signal fails, later ones are delivered."""

    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 1

    async def signal_review(
        self,
        invoice_id: InvoiceId,
        approved: bool,
        corrections: ReviewCorrections | None = None,
    ) -> None:
        if self.failures_left > 0:
            self.failures_left -= 1
            raise WorkflowSignalError(f"could not signal the review decision for {invoice_id}")
        await super().signal_review(invoice_id, approved, corrections)


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
    monkeypatch.setenv("APPROVAL_LIMIT", str(APPROVAL_LIMIT))
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


async def _reject_retry_scenario() -> None:
    invoice_id = _invoice_id("retry-reject-after-503")
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="rev", password="pw", roles=[Role.REVIEWER])
        await _create_user(session, user_id="adm", password="pw", roles=[Role.ADMIN])
        await _create_verified_vendor(
            session,
            vendor_id="vendorRR",
            iban=IBAN,
            verified_by="adm",
            changed_by="adm",
            changed_at=datetime.now(UTC) - timedelta(days=365),
        )
        await _create_invoice(
            session,
            invoice_id=invoice_id,
            vendor_id="vendorRR",
            total=Decimal("100.00"),
            route=Route.REVIEW,
            status=DbInvoiceStatus.AWAITING_REVIEW,
        )
    signaller = _FlakySignaller()
    transport = ASGITransport(app=_app(signaller))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")
        first = await client.post(
            f"/invoices/{invoice_id}/decisions", json={"action": "reject"}, headers=_auth(token)
        )
        retry = await client.post(
            f"/invoices/{invoice_id}/decisions", json={"action": "reject"}, headers=_auth(token)
        )
    await db_session.get_engine().dispose()
    assert first.status_code == 503, first.text
    assert (invoice_id, False) in signaller.sent, (
        "the rejection never reached the waiting workflow: the first signal failed (503) and "
        f"the documented recovery, re-sending reject, answered {retry.status_code}: {retry.text}"
    )


def test_a_reject_whose_signal_failed_can_be_resent(api_schema: str) -> None:
    asyncio.run(_reject_retry_scenario())
