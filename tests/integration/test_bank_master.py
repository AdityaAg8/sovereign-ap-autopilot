"""Integration tests for the bank-master API against a real, throwaway Postgres
container: an admin proposes a change (stored unverified, never selectable by P1), the proposer
cannot verify it (SoD, audited), a different admin verifies it with an out-of-band method and
reference (then `select_payee` pays it), replays are idempotent, stale or malformed requests are
refused, a payee pinned before the change never silently follows it, the writer itself refuses
self-verification even with a grant, the changer is barred from approving that vendor (P2), and
every step is on an intact audit hash chain. Marked `integration`, so it needs Docker
(`pytest -m integration`)."""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import apx.db.session as db_session
from apx.api.app import create_app
from apx.api.deps import get_review_signaller
from apx.api.security import hash_password
from apx.api.settings import get_settings
from apx.audit.chain import PgAuditLog
from apx.audit.verify import iter_chain_entries, verify_chain_stream
from apx.common.errors import (
    PayeeRefusal,
    PayeeRefusedError,
    PolicyDeniedError,
    SegregationOfDutiesError,
)
from apx.common.master import User as MasterUser
from apx.common.types import InvoiceId, Language, Money, Role, Route, UserId, VendorId
from apx.db.enums import BankDetailSource, DbInvoiceStatus
from apx.db.models import (
    AuditLog,
    BankDetail,
    BankDetailHistory,
    BankVerification,
    Invoice,
    RoleDef,
    User,
    UserRole,
    Vendor,
)
from apx.db.session import get_session
from apx.ingest.settings import get_ingest_settings
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.guard import DefaultPolicyGuard
from apx.policy.interfaces import (
    ActionRequest,
    Capability,
    PayeeMatch,
    PolicyThresholds,
    pin_payee,
    select_payee,
)
from apx.policy.writer import SqlPaymentStateWriter, load_vendor
from apx.workflows.corrections import ReviewCorrections

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-bank-master"
HOST_PORT = 55451
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"

OLD_IBAN = "AE070331234567890123456"
NEW_IBAN = "GB82WEST12345698765432"
VENDOR = "vendor-bank"


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
    """Throwaway `pgvector/pgvector:pg16-trixie` container on a host-only port (same pattern as
    `tests/integration/test_api_authz.py`), stopped after the module."""
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
    """Points the API at the container, migrates to `head` and resets caches per test."""
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    monkeypatch.setenv("API_JWT_SECRET", "integration-test-secret-value-long-enough-for-hs256")
    monkeypatch.setenv("API_APPROVAL_LIMIT", "10000.00")
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


class _NoSignaller:
    async def signal_review(
        self,
        invoice_id: InvoiceId,
        approved: bool,
        corrections: ReviewCorrections | None = None,
    ) -> None:
        raise AssertionError("no review signal is expected in these tests")


def _app() -> FastAPI:
    app = create_app()
    app.dependency_overrides[get_review_signaller] = _NoSignaller
    return app


async def _create_user(session: AsyncSession, user_id: str, roles: list[Role]) -> None:
    session.add(
        User(
            id=user_id,
            email=f"{user_id}@example.test",
            display_name=user_id,
            password_hash=hash_password("pw"),
        )
    )
    await session.flush()
    role_ids = (await session.scalars(select(RoleDef.id).where(RoleDef.name.in_(roles)))).all()
    for role_id in role_ids:
        session.add(UserRole(user_id=user_id, role_id=role_id))
    await session.flush()


async def _seed() -> None:
    """Users for every role, and a vendor whose only account was verified a year ago."""
    async with get_session() as session:
        for role in Role:
            session.add(RoleDef(name=role))
        await session.flush()
        await _create_user(session, "reviewer1", [Role.REVIEWER])
        await _create_user(session, "approver1", [Role.APPROVER])
        await _create_user(session, "admin1", [Role.ADMIN, Role.APPROVER])
        await _create_user(session, "admin2", [Role.ADMIN])
        session.add(Vendor(id=VENDOR, name="Bank Vendor"))
        session.add(Vendor(id="vendor-two", name="Two Accounts"))
        await session.flush()
        verification = BankVerification(vendor_id=VENDOR, iban=OLD_IBAN, verified_by="admin2")
        session.add(verification)
        await session.flush()
        session.add(
            BankDetail(
                vendor_id=VENDOR,
                iban=OLD_IBAN,
                source=BankDetailSource.ADMIN_CALLBACK,
                verification_id=verification.id,
                is_current=True,
                changed_by="admin2",
                changed_at=datetime.now(UTC) - timedelta(days=365),
            )
        )
        for iban in ("DE89370400440532013000", "SA0380000000608010167519"):
            session.add(BankDetail(vendor_id="vendor-two", iban=iban, changed_by="admin2"))
        invoice_id = hashlib.sha256(b"bank-master-invoice").hexdigest()
        session.add(
            Invoice(
                id=invoice_id,
                document_sha256=invoice_id,
                vendor_id=VENDOR,
                language=Language.EN,
                number="INV-BANK-1",
                issue_date=date(2026, 9, 1),
                subtotal=Decimal("100.00"),
                vat_total=Decimal("0.00"),
                total=Decimal("100.00"),
                status=DbInvoiceStatus.AWAITING_REVIEW,
                route=Route.REVIEW,
            )
        )


async def _login(client: AsyncClient, user_id: str) -> dict[str, str]:
    response = await client.post(
        "/auth/login", data={"username": f"{user_id}@example.test", "password": "pw"}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def _audit_actions(session: AsyncSession, entity: str) -> list[str]:
    rows = await session.scalars(
        select(AuditLog.action).where(AuditLog.entity == entity).order_by(AuditLog.id)
    )
    return list(rows.all())


def _change(new_iban: str, expected: str | None = OLD_IBAN) -> dict[str, object]:
    return {"new_iban": new_iban, "expected_current_iban": expected}


def _verification(iban: str = NEW_IBAN) -> dict[str, object]:
    return {"iban": iban, "method": "phone_callback_on_file", "reference": "CALL-2026-0917"}


async def _propose_and_verify_scenario() -> None:
    await _seed()
    async with get_session() as session:
        pinned = pin_payee(select_payee(await load_vendor(session, VENDOR), invoice_iban=None))

    changes = f"/vendors/{VENDOR}/bank-details/changes"
    verifications = f"/vendors/{VENDOR}/bank-details/verifications"
    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        reviewer = await _login(client, "reviewer1")
        approver = await _login(client, "approver1")
        admin1 = await _login(client, "admin1")
        admin2 = await _login(client, "admin2")

        # Authz matrix: only a role holding CHANGE_BANK_DETAILS may propose.
        for headers in (reviewer, approver):
            response = await client.post(changes, json=_change(NEW_IBAN), headers=headers)
            assert response.status_code == 403, response.text

        # Refusals that write nothing: bad checksum, stale expectation, unknown vendor, a vendor
        # with several current accounts.
        bad = "GB82WEST12345698765433"
        response = await client.post(changes, json=_change(bad), headers=admin1)
        assert response.status_code == 422, response.text
        assert bad not in response.text
        response = await client.post(
            changes, json=_change(NEW_IBAN, expected="DE89370400440532013000"), headers=admin1
        )
        assert response.status_code == 409, response.text
        response = await client.post(
            "/vendors/nope/bank-details/changes", json=_change(NEW_IBAN), headers=admin1
        )
        assert response.status_code == 404, response.text
        response = await client.post(
            "/vendors/vendor-two/bank-details/changes",
            json=_change(NEW_IBAN, expected="DE89370400440532013000"),
            headers=admin1,
        )
        assert response.status_code == 409, response.text

        # Propose (print form, lower case): stored pending and unverified.
        response = await client.post(
            changes, json=_change("gb82 west 1234 5698 7654 32", expected=OLD_IBAN), headers=admin1
        )
        assert response.status_code == 200, response.text
        proposed = response.json()
        assert proposed["status"] == "pending_verification"
        assert proposed["changed_by"] == "admin1"
        assert proposed["verified_by"] is None
        assert NEW_IBAN not in response.text

        # Replay: idempotent, nothing new written.
        response = await client.post(changes, json=_change(NEW_IBAN), headers=admin1)
        assert response.status_code == 200, response.text
        assert response.json() == proposed

    async with get_session() as session:
        rows = (
            await session.scalars(
                select(BankDetail).where(BankDetail.vendor_id == VENDOR).order_by(BankDetail.id)
            )
        ).all()
        assert [(r.iban, r.is_current, r.verified) for r in rows] == [
            (OLD_IBAN, False, True),
            (NEW_IBAN, True, False),
        ]
        assert rows[1].source is BankDetailSource.UNVERIFIED
        assert rows[1].verification_id is None
        history = (
            await session.scalars(
                select(BankDetailHistory).where(
                    BankDetailHistory.vendor_id == VENDOR, BankDetailHistory.iban == NEW_IBAN
                )
            )
        ).all()
        assert [(h.operation, h.changed_by) for h in history] == [("I", "admin1")]
        # Never auto-verified: P1 refuses to pay the pending account, so the payee pinned at
        # routing no longer matches and the build refuses (ADR-0014 §7).
        with pytest.raises(PayeeRefusedError) as refused:
            select_payee(await load_vendor(session, VENDOR), invoice_iban=NEW_IBAN)
        assert refused.value.refusal is PayeeRefusal.NO_VERIFIED_ACCOUNT

    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as client:
        reviewer = await _login(client, "reviewer1")
        admin1 = await _login(client, "admin1")
        admin2 = await _login(client, "admin2")

        # Authz matrix for verify, then SoD: the proposer cannot verify their own change.
        response = await client.post(verifications, json=_verification(), headers=reviewer)
        assert response.status_code == 403, response.text
        response = await client.post(verifications, json=_verification(), headers=admin1)
        assert response.status_code == 403, response.text
        assert "cannot verify" in response.text

        # Verifying an IBAN that is not a pending current account is a conflict.
        response = await client.post(verifications, json=_verification(OLD_IBAN), headers=admin2)
        assert response.status_code == 409, response.text

        # A different admin verifies with an out-of-band method and reference.
        response = await client.post(verifications, json=_verification(), headers=admin2)
        assert response.status_code == 200, response.text
        verified = response.json()
        assert verified["status"] == "verified"
        assert verified["verified_by"] == "admin2"
        assert verified["verification_method"] == "phone_callback_on_file"

        # Replay: idempotent.
        response = await client.post(verifications, json=_verification(), headers=admin2)
        assert response.status_code == 200, response.text
        assert response.json() == verified

        # P2: the changer (admin1 also holds approver) is barred from approving this vendor.
        invoice_id = hashlib.sha256(b"bank-master-invoice").hexdigest()
        response = await client.post(
            f"/invoices/{invoice_id}/decisions", json={"action": "approve"}, headers=admin1
        )
        assert response.status_code == 403, response.text
        assert "segregation of duties" in response.text

    async with get_session() as session:
        verification = (
            await session.scalars(
                select(BankVerification).where(
                    BankVerification.vendor_id == VENDOR, BankVerification.iban == NEW_IBAN
                )
            )
        ).one()
        assert (verification.verified_by, verification.method, verification.notes) == (
            "admin2",
            "phone_callback_on_file",
            "CALL-2026-0917",
        )
        vendor = await load_vendor(session, VENDOR)
        payee = select_payee(vendor, invoice_iban=None)
        assert (payee.iban, payee.match) == (NEW_IBAN, PayeeMatch.SOLE_UNCLAIMED)
        # The pin taken before the change still names the old account: the build compares and
        # refuses instead of redirecting the in-flight invoice to the new one.
        assert pin_payee(payee).iban_sha256 != pinned.iban_sha256
        assert (
            await session.scalar(
                select(func.count())
                .select_from(BankVerification)
                .where(BankVerification.iban == NEW_IBAN)
            )
            == 1
        )

        actions = await _audit_actions(session, f"vendor:{VENDOR}")
        assert actions == [
            "authorize:change_bank_details",
            "payment_state:bank_change",
            "bank_master:verify_refused",
            "authorize:verify_bank_details",
            "payment_state:verify_bank",
        ]
        result = await verify_chain_stream(iter_chain_entries(session))
        assert result.ok, result

    await db_session.get_engine().dispose()


def test_propose_then_verify_by_another_admin(api_schema: str) -> None:
    asyncio.run(_propose_and_verify_scenario())


async def _writer_sod_scenario() -> None:
    """The writer re-checks four eyes under the vendor lock, so a grant obtained by the changer
    (a race past the service's pre-check) still cannot verify their own account."""
    await _seed()
    async with get_session() as session:
        session.add(Vendor(id="vendor-solo", name="Solo"))
        await session.flush()
        session.add(BankDetail(vendor_id="vendor-solo", iban=NEW_IBAN, changed_by="admin1"))

    audit = PgAuditLog()
    thresholds = PolicyThresholds(
        approval_limit=Money(amount=Decimal("10000.00"), currency="AED"),
        fraud_review=1.0,
        scan_review=1.0,
        attack_block=1.0,
    )
    guard = DefaultPolicyGuard(DefaultPolicyEngine(thresholds), audit)
    writer = SqlPaymentStateWriter(get_session, audit)
    admin1 = MasterUser(id=UserId("admin1"), display_name="admin1", roles=frozenset({Role.ADMIN}))
    async with get_session() as session:
        request = ActionRequest(
            actor=admin1,
            action=Capability.VERIFY_BANK_DETAILS,
            invoice=None,
            vendor=await load_vendor(session, "vendor-solo"),
            at=datetime.now(UTC),
        )
    grant = await guard.require(request)
    with pytest.raises(SegregationOfDutiesError):
        await writer.verify_bank_account(
            VendorId("vendor-solo"), NEW_IBAN, grant, method="in_person", reference="REF-1"
        )
    with pytest.raises(PolicyDeniedError, match="method and reference"):
        await writer.verify_bank_account(
            VendorId("vendor-solo"), NEW_IBAN, grant, method="in_person", reference="  "
        )
    async with get_session() as session:
        row = await session.scalar(select(BankDetail).where(BankDetail.vendor_id == "vendor-solo"))
        assert row is not None and not row.verified
        assert await _audit_actions(session, "vendor:vendor-solo") == [
            "authorize:verify_bank_details"
        ]

    await db_session.get_engine().dispose()


def test_writer_refuses_self_verification_even_with_a_grant(api_schema: str) -> None:
    asyncio.run(_writer_sod_scenario())
