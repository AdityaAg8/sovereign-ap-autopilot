"""Integration tests for the API's authorisation matrix against a real, throwaway
Postgres container (P1-P5): P3 gates an above-limit approval on the
approver role, P2 bars the bank-detail changer from approving that vendor for 30 days, an
admin-only user cannot approve payments, field edits need the reviewer role, a proposal
executes only for its kind's required role, approvals need AWAITING_REVIEW, are idempotent
and signal the invoice's workflow (all via PolicyGuard -> SqlPaymentStateWriter), and every
decision/edit appends to the audit chain
(I4). A SQL-injection payload in the list filter is also proven inert end to end. The
vendor endpoints (masked bank details, the G-3 verified flag, recent invoices) and the
document-serving route (authz matrix across every role plus path-traversal attempts,
docs/security.md B1) are covered here too, reusing the same throwaway Postgres container. Marked
`integration`, so it needs Docker (`pytest -m integration`)."""

from __future__ import annotations

import asyncio
import json
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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import apx.db.session as db_session
from apx.api.app import create_app
from apx.api.deps import get_review_signaller
from apx.api.security import hash_password
from apx.api.settings import get_settings
from apx.common.types import InvoiceId, Language, Role, Route
from apx.db.enums import (
    BankDetailSource,
    DbInvoiceStatus,
    FieldSource,
    ProposalKind,
    ProposalStatus,
)
from apx.db.models import (
    Approval,
    AuditLog,
    BankDetail,
    BankVerification,
    Field,
    Invoice,
    Label,
    Proposal,
    RoleDef,
    User,
    UserRole,
    Vendor,
)
from apx.db.session import get_session
from apx.ingest.settings import get_ingest_settings
from apx.workflows.corrections import ReviewCorrections

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-api"
HOST_PORT = 55434
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"

APPROVAL_LIMIT = Decimal("10000.00")
# Real magic bytes are all `apx.ingest.validate.detect_media_type` (and so the document route)
# ever inspects; this document is never parsed as a real PDF in these tests.
FAKE_PDF_BYTES = b"%PDF-1.4\n% a fake pdf body, only the magic bytes matter here\n%%EOF"


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
    """Starts a throwaway `pgvector/pgvector:pg16-trixie` container on a host-only port (same
    pattern as `tests/integration/test_audit_chain.py`), waits for it, and stops it after every
    test in this module has run."""
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
    """Points `apx.db.session` and `apx.api.settings` at the throwaway container/environment,
    migrates to `head`, and forces fresh caches so every test starts from an identical, empty
    schema on a connection bound to its own event loop."""
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


async def _seed_roles(session: AsyncSession) -> None:
    for role in Role:
        session.add(RoleDef(name=role))
    await session.flush()


async def _create_user(
    session: AsyncSession, *, user_id: str, password: str, roles: list[Role]
) -> None:
    session.add(
        User(
            id=user_id,
            email=f"{user_id}@example.test",
            display_name=user_id,
            password_hash=hash_password(password),
        )
    )
    await session.flush()
    role_ids = (
        (await session.execute(select(RoleDef.id).where(RoleDef.name.in_(roles)))).scalars().all()
    )
    for role_id in role_ids:
        session.add(UserRole(user_id=user_id, role_id=role_id))
    await session.flush()


async def _create_verified_vendor(
    session: AsyncSession,
    *,
    vendor_id: str,
    iban: str,
    verified_by: str,
    changed_by: str,
    changed_at: datetime | None = None,
) -> None:
    session.add(Vendor(id=vendor_id, name=f"Vendor {vendor_id}"))
    await session.flush()
    verification = BankVerification(vendor_id=vendor_id, iban=iban, verified_by=verified_by)
    session.add(verification)
    await session.flush()
    bank_detail = BankDetail(
        vendor_id=vendor_id,
        iban=iban,
        source=BankDetailSource.ADMIN_CALLBACK,
        verification_id=verification.id,
        is_current=True,
        changed_by=changed_by,
    )
    if changed_at is not None:
        bank_detail.changed_at = changed_at
    session.add(bank_detail)
    await session.flush()


# ADR-0012: the classified values an approval needs, as the workflow's records.save writes
# them (NULL when the extractors disagreed; `classified=False`).
_CLASSIFIED_FIELDS: dict[str, str] = {
    "invoice_type_code": "380",
    "line_items[1].vat_category": "Z",
    "line_items[1].vat_rate": "0",
}
# Every model-sourced field row `_create_invoice` seeds: the classified values plus the grounded
# currency and supplier TRN (BT-31), which an approval requires from the extraction itself
# (`SupplierTrnMissingError`: a reviewer's edit cannot supply it), and a supplier address naming
# an emirate (ADR-0012 amendment 4: PINT AE ibr-143-ae).
_SEEDED_FIELDS: dict[str, str] = {
    **_CLASSIFIED_FIELDS,
    "currency": "AED",
    "supplier_trn": "100000000000003",
    "supplier_address": "Al Quoz, Dubai",
}


async def _create_invoice(
    session: AsyncSession,
    *,
    invoice_id: str,
    vendor_id: str | None,
    total: Decimal,
    route: Route | None = Route.REVIEW,
    status: DbInvoiceStatus = DbInvoiceStatus.AWAITING_REVIEW,
    classified: bool = True,
) -> None:
    session.add(
        Invoice(
            id=invoice_id,
            document_sha256=invoice_id,
            vendor_id=vendor_id,
            language=Language.EN,
            number=f"INV-{invoice_id[:8]}",
            issue_date=date(2026, 1, 1),
            subtotal=total,
            vat_total=Decimal("0.00"),
            total=total,
            status=status,
            route=route,
        )
    )
    await session.flush()
    for name, value in _SEEDED_FIELDS.items():
        kept = classified or name not in _CLASSIFIED_FIELDS
        session.add(
            Field(
                invoice_id=invoice_id,
                name=name,
                value=value if kept else None,
                normalized_value=value if kept else None,
                source=FieldSource.EXTRACTOR_A,
            )
        )
    await session.flush()


def _invoice_id(name: str) -> str:
    import hashlib

    return hashlib.sha256(name.encode()).hexdigest()


async def _login(client: AsyncClient, user_id: str, password: str) -> str:
    response = await client.post(
        "/auth/login", data={"username": f"{user_id}@example.test", "password": password}
    )
    assert response.status_code == 200, response.text
    return str(response.json()["access_token"])


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class _RecordingSignaller:
    """Stands in for Temporal: records every `review` signal the API sends (F3)."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, bool]] = []
        self.corrections: list[ReviewCorrections | None] = []

    async def signal_review(
        self,
        invoice_id: InvoiceId,
        approved: bool,
        corrections: ReviewCorrections | None = None,
    ) -> None:
        self.sent.append((invoice_id, approved))
        self.corrections.append(corrections)


def _app(signaller: _RecordingSignaller | None = None) -> FastAPI:
    app = create_app()
    recorder = signaller or _RecordingSignaller()
    app.dependency_overrides[get_review_signaller] = lambda: recorder
    return app


async def _audit_entities(session: AsyncSession, entity: str) -> list[str]:
    rows = (
        (await session.execute(select(AuditLog.action).where(AuditLog.entity == entity)))
        .scalars()
        .all()
    )
    return list(rows)


async def _approval_scenario() -> None:
    signaller = _RecordingSignaller()
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer1", password="pw", roles=[Role.REVIEWER])
        await _create_user(session, user_id="approver1", password="pw", roles=[Role.APPROVER])
        await _create_user(session, user_id="admin1", password="pw", roles=[Role.ADMIN])
        await _create_verified_vendor(
            session,
            vendor_id="vendorA",
            iban="AE070331234567890123456",
            verified_by="admin1",
            changed_by="admin1",
            changed_at=datetime.now(UTC) - timedelta(days=365),
        )
        low_id = _invoice_id("low-amount-invoice")
        high_id = _invoice_id("high-amount-invoice")
        await _create_invoice(
            session, invoice_id=low_id, vendor_id="vendorA", total=Decimal("500.00")
        )
        await _create_invoice(
            session, invoice_id=high_id, vendor_id="vendorA", total=Decimal("50000.00")
        )

    transport = ASGITransport(app=_app(signaller))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reviewer_token = await _login(client, "reviewer1", "pw")
        approver_token = await _login(client, "approver1", "pw")

        # K1: reviewer can approve within the limit.
        response = await client.post(
            f"/invoices/{low_id}/decisions",
            json={"action": "approve"},
            headers=_auth(reviewer_token),
        )
        assert response.status_code == 204, response.text

        # P3: reviewer cannot approve above the limit; the approver role is required.
        response = await client.post(
            f"/invoices/{high_id}/decisions",
            json={"action": "approve"},
            headers=_auth(reviewer_token),
        )
        assert response.status_code == 403, response.text

        response = await client.post(
            f"/invoices/{high_id}/decisions",
            json={"action": "approve"},
            headers=_auth(approver_token),
        )
        assert response.status_code == 204, response.text

    async with get_session() as session:
        assert "decision.approve" in await _audit_entities(session, f"invoice:{low_id}")
        assert "decision.approve" in await _audit_entities(session, f"invoice:{high_id}")
        low_invoice = await session.get(Invoice, low_id)
        high_invoice = await session.get(Invoice, high_id)
        assert low_invoice is not None and low_invoice.status == DbInvoiceStatus.APPROVED
        assert high_invoice is not None and high_invoice.status == DbInvoiceStatus.APPROVED
    assert signaller.sent == [(low_id, True), (high_id, True)]

    await db_session.get_engine().dispose()


def test_reviewer_can_approve_within_limit_but_not_above_it(api_schema: str) -> None:
    asyncio.run(_approval_scenario())


async def _admin_only_cannot_approve_scenario() -> None:
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="adminonly", password="pw", roles=[Role.ADMIN])
        await _create_user(
            session, user_id="adminapprover", password="pw", roles=[Role.ADMIN, Role.APPROVER]
        )
        await _create_verified_vendor(
            session,
            vendor_id="vendorB",
            iban="AE070331234567890123457",
            verified_by="adminonly",
            changed_by="adminonly",
            changed_at=datetime.now(UTC) - timedelta(days=365),
        )
        invoice_id = _invoice_id("admin-only-invoice")
        await _create_invoice(
            session, invoice_id=invoice_id, vendor_id="vendorB", total=Decimal("100.00")
        )

    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        admin_only_token = await _login(client, "adminonly", "pw")
        admin_approver_token = await _login(client, "adminapprover", "pw")

        response = await client.post(
            f"/invoices/{invoice_id}/decisions",
            json={"action": "approve"},
            headers=_auth(admin_only_token),
        )
        assert response.status_code == 403, response.text

        response = await client.post(
            f"/invoices/{invoice_id}/decisions",
            json={"action": "approve"},
            headers=_auth(admin_approver_token),
        )
        assert response.status_code == 204, response.text

    await db_session.get_engine().dispose()


def test_admin_alone_cannot_approve_payments(api_schema: str) -> None:
    asyncio.run(_admin_only_cannot_approve_scenario())


async def _sod_scenario() -> None:
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="admin2", password="pw", roles=[Role.ADMIN])
        await _create_user(session, user_id="changer", password="pw", roles=[Role.APPROVER])
        await _create_user(session, user_id="clean_approver", password="pw", roles=[Role.APPROVER])
        # `changer` (an approver) is recorded as the one who changed vendorC's bank details
        # *today* -- inside the 30-day SoD cooldown.
        await _create_verified_vendor(
            session,
            vendor_id="vendorC",
            iban="AE070331234567890123458",
            verified_by="admin2",
            changed_by="changer",
        )
        invoice_id = _invoice_id("sod-invoice")
        await _create_invoice(
            session, invoice_id=invoice_id, vendor_id="vendorC", total=Decimal("100.00")
        )

    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        changer_token = await _login(client, "changer", "pw")
        clean_token = await _login(client, "clean_approver", "pw")

        response = await client.post(
            f"/invoices/{invoice_id}/decisions",
            json={"action": "approve"},
            headers=_auth(changer_token),
        )
        assert response.status_code == 403, response.text

        response = await client.post(
            f"/invoices/{invoice_id}/decisions",
            json={"action": "approve"},
            headers=_auth(clean_token),
        )
        assert response.status_code == 204, response.text

    await db_session.get_engine().dispose()


def test_sod_bars_the_bank_detail_changer_for_30_days(api_schema: str) -> None:
    asyncio.run(_sod_scenario())


async def _field_edit_scenario() -> None:
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer2", password="pw", roles=[Role.REVIEWER])
        await _create_user(session, user_id="approver2", password="pw", roles=[Role.APPROVER])
        invoice_id = _invoice_id("field-edit-invoice")
        await _create_invoice(session, invoice_id=invoice_id, vendor_id=None, total=Decimal("1"))

    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reviewer_token = await _login(client, "reviewer2", "pw")
        approver_token = await _login(client, "approver2", "pw")

        response = await client.patch(
            f"/invoices/{invoice_id}/fields",
            json={"field": "total", "value": "1234.56", "reason": "typo fix"},
            headers=_auth(approver_token),
        )
        assert response.status_code == 403, response.text

        response = await client.patch(
            f"/invoices/{invoice_id}/fields",
            json={"field": "total", "value": "1234.56", "reason": "typo fix"},
            headers=_auth(reviewer_token),
        )
        assert response.status_code == 204, response.text

    async with get_session() as session:
        field_row = (
            await session.execute(
                select(Field).where(Field.invoice_id == invoice_id, Field.name == "total")
            )
        ).scalar_one()
        assert field_row.value == "1234.56"
        assert field_row.source.value == "human"

        label_row = (
            await session.execute(select(Label).where(Label.invoice_id == invoice_id))
        ).scalar_one()
        assert label_row.corrected_value == "1234.56"
        assert label_row.labeled_by == "reviewer2"

        assert "field_edit" in await _audit_entities(session, f"invoice:{invoice_id}:field:total")

    await db_session.get_engine().dispose()


def test_field_edit_requires_the_reviewer_role(api_schema: str) -> None:
    asyncio.run(_field_edit_scenario())


async def _proposal_scenario() -> None:
    proposal_id = "proposal-1"
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer3", password="pw", roles=[Role.REVIEWER])
        await _create_user(session, user_id="approver3", password="pw", roles=[Role.APPROVER])
        invoice_id = _invoice_id("proposal-invoice")
        await _create_invoice(session, invoice_id=invoice_id, vendor_id=None, total=Decimal("1"))
        session.add(
            Proposal(
                id=proposal_id,
                invoice_id=invoice_id,
                kind=ProposalKind.EDIT_FIELD,
                payload={"field": "total", "value": "42.00"},
                created_by="agent",
                status=ProposalStatus.PENDING,
            )
        )
        await session.flush()

    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        approver_token = await _login(client, "approver3", "pw")
        reviewer_token = await _login(client, "reviewer3", "pw")

        # An edit_field proposal needs the EDIT_FIELD capability; approver lacks it.
        response = await client.post(
            f"/proposals/{proposal_id}/accept", headers=_auth(approver_token)
        )
        assert response.status_code == 403, response.text

        response = await client.post(
            f"/proposals/{proposal_id}/accept", headers=_auth(reviewer_token)
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "executed"

    async with get_session() as session:
        proposal_row = await session.get(Proposal, proposal_id)
        assert proposal_row is not None
        assert proposal_row.status == ProposalStatus.EXECUTED
        assert "proposal.accept" in await _audit_entities(session, f"proposal:{proposal_id}")

    await db_session.get_engine().dispose()


def test_proposal_accept_requires_its_kinds_capability(api_schema: str) -> None:
    asyncio.run(_proposal_scenario())


async def _approval_preconditions_scenario() -> None:
    """Approvals go through `SqlPaymentStateWriter`, so only an
    AWAITING_REVIEW invoice can be approved, a repeated approval is an idempotent no-op (one
    approval row), and every approval -- direct or via an approve_payment proposal -- signals
    the invoice's workflow. A change_bank proposal lands through the same writer."""
    signaller = _RecordingSignaller()
    received_id = _invoice_id("precondition-received")
    blocked_id = _invoice_id("precondition-blocked")
    review_id = _invoice_id("precondition-review")
    proposal_invoice_id = _invoice_id("precondition-proposal")
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="approver5", password="pw", roles=[Role.APPROVER])
        await _create_user(session, user_id="admin5", password="pw", roles=[Role.ADMIN])
        await _create_verified_vendor(
            session,
            vendor_id="vendorE",
            iban="AE070331234567890123459",
            verified_by="admin5",
            changed_by="admin5",
            changed_at=datetime.now(UTC) - timedelta(days=365),
        )
        await _create_invoice(
            session,
            invoice_id=received_id,
            vendor_id="vendorE",
            total=Decimal("100.00"),
            status=DbInvoiceStatus.RECEIVED,
        )
        await _create_invoice(
            session,
            invoice_id=blocked_id,
            vendor_id="vendorE",
            total=Decimal("100.00"),
            route=Route.BLOCK,
            status=DbInvoiceStatus.BLOCKED,
        )
        await _create_invoice(
            session, invoice_id=review_id, vendor_id="vendorE", total=Decimal("100.00")
        )
        await _create_invoice(
            session, invoice_id=proposal_invoice_id, vendor_id="vendorE", total=Decimal("200.00")
        )
        session.add(
            Proposal(
                id="approve-proposal",
                invoice_id=proposal_invoice_id,
                kind=ProposalKind.APPROVE_PAYMENT,
                payload={},
                created_by="agent",
                status=ProposalStatus.PENDING,
            )
        )
        session.add(
            Proposal(
                id="bank-proposal",
                invoice_id=None,
                kind=ProposalKind.CHANGE_BANK,
                payload={"vendor_id": "vendorE", "new_iban": "AE070331234567890123460"},
                created_by="admin5",
                status=ProposalStatus.PENDING,
            )
        )
        await session.flush()

    transport = ASGITransport(app=_app(signaller))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        approver_token = await _login(client, "approver5", "pw")
        admin_token = await _login(client, "admin5", "pw")

        for invoice_id in (received_id, blocked_id):
            response = await client.post(
                f"/invoices/{invoice_id}/decisions",
                json={"action": "approve"},
                headers=_auth(approver_token),
            )
            assert response.status_code == 403, response.text

        for _ in range(2):
            response = await client.post(
                f"/invoices/{review_id}/decisions",
                json={"action": "approve"},
                headers=_auth(approver_token),
            )
            assert response.status_code == 204, response.text

        response = await client.post(
            "/proposals/approve-proposal/accept", headers=_auth(approver_token)
        )
        assert response.status_code == 200, response.text

        response = await client.post("/proposals/bank-proposal/accept", headers=_auth(admin_token))
        assert response.status_code == 200, response.text

    assert signaller.sent == [
        (review_id, True),
        (review_id, True),
        (proposal_invoice_id, True),
    ]
    async with get_session() as session:
        approvals = (
            (await session.execute(select(Approval.invoice_id).order_by(Approval.invoice_id)))
            .scalars()
            .all()
        )
        assert sorted(approvals) == sorted([review_id, proposal_invoice_id])
        for invoice_id, expected in (
            (received_id, DbInvoiceStatus.RECEIVED),
            (blocked_id, DbInvoiceStatus.BLOCKED),
            (review_id, DbInvoiceStatus.APPROVED),
            (proposal_invoice_id, DbInvoiceStatus.APPROVED),
        ):
            row = await session.get(Invoice, invoice_id)
            assert row is not None and row.status == expected
        for proposal_id in ("approve-proposal", "bank-proposal"):
            proposal_row = await session.get(Proposal, proposal_id)
            assert proposal_row is not None and proposal_row.status == ProposalStatus.EXECUTED
        current = (
            (
                await session.execute(
                    select(BankDetail).where(
                        BankDetail.vendor_id == "vendorE", BankDetail.is_current.is_(True)
                    )
                )
            )
            .scalars()
            .all()
        )
        assert [(bd.iban, bd.verified) for bd in current] == [("AE070331234567890123460", False)]
        assert "payment_state:approve" in await _audit_entities(session, f"invoice:{review_id}")

    await db_session.get_engine().dispose()


def test_approval_needs_awaiting_review_is_idempotent_and_signals_the_workflow(
    api_schema: str,
) -> None:
    asyncio.run(_approval_preconditions_scenario())


async def _injection_in_list_filter_scenario() -> None:
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer4", password="pw", roles=[Role.REVIEWER])
        invoice_id = _invoice_id("filter-invoice")
        await _create_invoice(session, invoice_id=invoice_id, vendor_id=None, total=Decimal("1"))

    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "reviewer4", "pw")
        response = await client.get(
            "/invoices",
            params={"q": "'; DROP TABLE invoices; --"},
            headers=_auth(token),
        )
        assert response.status_code == 200, response.text
        assert response.json()["items"] == []

        # The table must still be there and still hold the seeded invoice.
        response = await client.get("/invoices", headers=_auth(token))
        assert response.status_code == 200, response.text
        assert response.json()["total"] == 1

    await db_session.get_engine().dispose()


def test_sql_injection_in_list_filter_is_inert_end_to_end(api_schema: str) -> None:
    asyncio.run(_injection_in_list_filter_scenario())


async def _vendor_endpoints_scenario() -> None:
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer6", password="pw", roles=[Role.REVIEWER])
        await _create_user(session, user_id="admin6", password="pw", roles=[Role.ADMIN])
        await _create_verified_vendor(
            session,
            vendor_id="vendorD",
            iban="AE070331234567890199999",
            verified_by="admin6",
            changed_by="admin6",
        )
        session.add(Vendor(id="vendorE", name="Unverified Vendor E"))
        await session.flush()
        invoice_id = _invoice_id("vendor-recent-invoice")
        await _create_invoice(
            session, invoice_id=invoice_id, vendor_id="vendorD", total=Decimal("250.00")
        )

    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "reviewer6", "pw")

        # Search/paging: `q` matches the vendor name substring.
        response = await client.get("/vendors", params={"q": "vendorD"}, headers=_auth(token))
        assert response.status_code == 200, response.text
        page = response.json()
        assert page["total"] == 1
        assert page["items"][0]["id"] == "vendorD"
        assert page["items"][0]["verified"] is True

        # An unverified vendor has no bank details at all -- G-3's verified flag is false, and
        # there is nothing to mask.
        response = await client.get("/vendors/vendorE", headers=_auth(token))
        assert response.status_code == 200, response.text
        assert response.json()["bank_details"] == []

        # A verified vendor's IBAN is masked to its last 4 characters -- never returned in full,
        # even to this authenticated reviewer -- and its G-3 `verified` generated column reads
        # true; its recent invoices include the one just created.
        response = await client.get("/vendors/vendorD", headers=_auth(token))
        assert response.status_code == 200, response.text
        detail = response.json()
        bank_detail = detail["bank_details"][0]
        assert bank_detail["verified"] is True
        assert bank_detail["verified_at"] is not None
        assert bank_detail["iban_masked"].endswith("9999")
        assert "AE07" not in bank_detail["iban_masked"]
        assert bank_detail["iban_masked"] != "AE070331234567890199999"
        assert [inv["id"] for inv in detail["recent_invoices"]] == [invoice_id]

        response = await client.get("/vendors/does-not-exist", headers=_auth(token))
        assert response.status_code == 404, response.text

    await db_session.get_engine().dispose()


def test_vendor_endpoints_mask_ibans_and_report_the_g3_verified_flag(api_schema: str) -> None:
    asyncio.run(_vendor_endpoints_scenario())


async def _document_route_scenario(store_dir: Path) -> None:
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer5", password="pw", roles=[Role.REVIEWER])
        await _create_user(session, user_id="approver5", password="pw", roles=[Role.APPROVER])
        await _create_user(session, user_id="admin5", password="pw", roles=[Role.ADMIN])
        invoice_id = _invoice_id("document-invoice")
        await _create_invoice(session, invoice_id=invoice_id, vendor_id=None, total=Decimal("1"))

    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / invoice_id).write_bytes(FAKE_PDF_BYTES)

    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # No bearer token at all: 401, not a leaked file.
        response = await client.get(f"/invoices/{invoice_id}/document")
        assert response.status_code == 401, response.text

        # Authz matrix: every one of the three roles can fetch the source document -- it is core
        # review material, not vendor bank master data, so this is the same read boundary as
        # `GET /invoices/{id}`, not a narrower one.
        for user_id in ("reviewer5", "approver5", "admin5"):
            token = await _login(client, user_id, "pw")
            response = await client.get(f"/invoices/{invoice_id}/document", headers=_auth(token))
            assert response.status_code == 200, response.text
            assert response.headers["content-type"] == "application/pdf"
            assert response.headers["content-disposition"] == f'inline; filename="{invoice_id}.pdf"'
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.content == FAKE_PDF_BYTES

        token = await _login(client, "reviewer5", "pw")

        # An invoice that does not exist: 404, never a filesystem error.
        missing_id = _invoice_id("no-such-invoice")
        response = await client.get(f"/invoices/{missing_id}/document", headers=_auth(token))
        assert response.status_code == 404, response.text

        # Path-traversal attempts in the `invoice_id` path segment: none of these resolve to any
        # invoice row (the real path is only ever built from a validated `document_sha256`
        # column, never from this parameter), so every one 404s rather than reading a file
        # outside `store_dir`.
        for traversal_id in (
            "..%2f..%2f..%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ):
            response = await client.get(f"/invoices/{traversal_id}/document", headers=_auth(token))
            assert response.status_code == 404, response.text
            assert "root:" not in response.text

    await db_session.get_engine().dispose()


def test_document_route_authz_matrix_and_traversal_attempts(
    api_schema: str, tmp_path: Path
) -> None:
    asyncio.run(_document_route_scenario(tmp_path / "documents"))


async def _fraud_flags_scenario() -> None:
    from apx.fraud.interfaces import FraudFlag, FraudSeverity, FraudTypology
    from apx.fraud.store import PgFraudFlagStore

    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer4", password="pw", roles=[Role.REVIEWER])
        invoice_id = _invoice_id("flagged-invoice")
        await _create_invoice(session, invoice_id=invoice_id, vendor_id=None, total=Decimal("9"))

    def _flag(rule: str, severity: FraudSeverity, score: float, why: str) -> FraudFlag:
        return FraudFlag(
            rule_id=rule,
            typology=FraudTypology.DUPLICATE,
            severity=severity,
            score=score,
            evidence="internal ids only",
            explanation=why,
        )

    await PgFraudFlagStore().record(
        InvoiceId(invoice_id),
        [
            _flag("price_variance_po", FraudSeverity.MEDIUM, 0.6, "Price is above the PO."),
            _flag("duplicate_invoice", FraudSeverity.HIGH, 1.0, "Same invoice seen last week."),
        ],
    )

    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "reviewer4", "pw")
        response = await client.get(f"/invoices/{invoice_id}", headers=_auth(token))
        assert response.status_code == 200, response.text
        flags = response.json()["flags"]

    assert [(f["rule_id"], f["severity"], f["explanation"]) for f in flags] == [
        ("duplicate_invoice", "high", "Same invoice seen last week."),
        ("price_variance_po", "medium", "Price is above the PO."),
    ]
    assert all("evidence" not in f for f in flags)  # evidence stays in the audit trail
    await db_session.get_engine().dispose()


def test_invoice_detail_returns_stored_fraud_flags_with_explanations(api_schema: str) -> None:
    asyncio.run(_fraud_flags_scenario())


async def _failed_before_invoice_scenario() -> None:
    """A workflow that failed before `build_invoice` persisted its row. The real
    `mark_failed` records it on `ingested_documents` and audits it; the detail endpoint then
    answers FAILED with the error, while a document still in flight stays a 404."""
    from typing import Any, cast

    from apx.audit.chain import PgAuditLog
    from apx.db.models import IngestedDocument
    from apx.workflows.activities import DefaultInvoiceActivities, InvoiceActivityDeps
    from apx.workflows.idempotency import SqlIdempotencyStore
    from apx.workflows.sql import SqlInvoiceRecords

    failed_id = _invoice_id("failed-before-invoice")
    running_id = _invoice_id("still-running")
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer5", password="pw", roles=[Role.REVIEWER])
        for sha in (failed_id, running_id):
            session.add(
                IngestedDocument(
                    sha256=sha,
                    filename=f"{sha[:8]}.pdf",
                    media_type="application/pdf",
                    size_bytes=10,
                    source="watch",
                    workflow_id=sha,
                )
            )

    unused = cast(Any, None)
    activities = DefaultInvoiceActivities(
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
            asp_client=unused,
            archive=unused,
            audit=PgAuditLog(),
            records=SqlInvoiceRecords(get_session),
            session_factory=get_session,
            idempotency=SqlIdempotencyStore(),
        )
    )
    error = "ActivityError: Activity task failed <- ModelOutputError: totals missing"
    await activities.mark_failed(failed_id, error)

    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "reviewer5", "pw")
        failed = await client.get(f"/invoices/{failed_id}", headers=_auth(token))
        running = await client.get(f"/invoices/{running_id}", headers=_auth(token))

    assert failed.status_code == 200, failed.text
    body = failed.json()
    assert body["summary"]["id"] == failed_id
    assert body["summary"]["status"] == "failed"
    assert body["summary"]["route"] is None
    assert body["error"] == error
    assert body["fields"] == []
    assert running.status_code == 404
    async with get_session() as session:
        assert await _audit_entities(session, f"invoice:{failed_id}") == ["activity:failed"]
    await db_session.get_engine().dispose()


def test_a_document_failed_before_its_invoice_row_reads_as_failed(api_schema: str) -> None:
    asyncio.run(_failed_before_invoice_scenario())


async def _ops_dashboard_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every dashboard number equals an independent check -- the hand-derived
    value for this seed, and a recomputation in Python from raw rows (not the dashboard SQL)."""
    from apx.common.types import Stage
    from apx.db.enums import DecisionAction, FieldSource
    from apx.db.models import Decision, Field, Label, LlmCall

    monkeypatch.setenv("OPS_LLM_BASE_URL", "http://127.0.0.1:9")  # llama-swap unreachable
    statuses = {
        "ops-a": DbInvoiceStatus.AWAITING_REVIEW,
        "ops-b": DbInvoiceStatus.APPROVED,
        "ops-c": DbInvoiceStatus.BLOCKED,
        "ops-d": DbInvoiceStatus.AWAITING_REVIEW,
    }
    ids = {name: _invoice_id(name) for name in statuses}
    attack = {"rule": "P4", "route": "block", "message": "scan injection score reached the block"}
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer5", password="pw", roles=[Role.REVIEWER])
        for name, status in statuses.items():
            await _create_invoice(
                session, invoice_id=ids[name], vendor_id=None, total=Decimal("5"), status=status
            )
        await session.flush()
        now = datetime.now(UTC)
        routes = [
            ("ops-a", Route.REVIEW, None, 0),
            ("ops-b", Route.AUTO, None, 0),
            ("ops-c", Route.BLOCK, [attack], 0),
            ("ops-d", Route.AUTO, None, 0),
            ("ops-d", Route.REVIEW, None, 5),  # re-routed later: latest decision wins
        ]
        for name, route, reasons, later in routes:
            session.add(
                Decision(
                    invoice_id=ids[name],
                    action=DecisionAction.POLICY_ROUTE,
                    route=route,
                    reasons=reasons,
                    created_at=now + timedelta(seconds=later),
                )
            )
        for n in range(5):
            session.add(
                Field(
                    invoice_id=ids["ops-a"], name=f"f{n}", value="x", source=FieldSource.EXTRACTOR_A
                )
            )
        session.add(Field(invoice_id=ids["ops-a"], name="f0", value="y", source=FieldSource.HUMAN))
        for previous in (FieldSource.EXTRACTOR_A, FieldSource.EXTRACTOR_A, FieldSource.HUMAN):
            session.add(
                Label(
                    invoice_id=ids["ops-a"],
                    field_name="f0",
                    corrected_value="y",
                    previous_source=previous,
                    source="review",
                )
            )
        for latency in (100, 200, 300, 400, 1000):
            session.add(
                LlmCall(
                    stage=Stage.EXTRACT, model="extractor-a", latency_ms=latency, trace_id="0" * 32
                )
            )

    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "reviewer5", "pw")
        response = await client.get("/ops/snapshot", headers=_auth(token))
        assert response.status_code == 200, response.text
        snap = response.json()
        assert (await client.get("/ops/snapshot")).status_code == 401

    kpis = snap["kpis"]
    # 1. Hand-derived for this seed.
    assert kpis["stp_rate"] == pytest.approx(1 / 4)
    # 2 automated-source labels over the 5 explicit ops-a fields plus the rows every
    # `_create_invoice` seeds (all extractor-sourced).
    automated_fields = 5 + len(statuses) * len(_SEEDED_FIELDS)
    assert kpis["automated_fields"] == automated_fields
    assert kpis["auto_accept_error_rate"] == pytest.approx(2 / automated_fields)
    assert kpis["llm_latency_p50_ms"] == pytest.approx(300)
    assert kpis["llm_latency_p95_ms"] == pytest.approx(880)  # 400 + 0.8 * (1000 - 400)
    assert kpis["attacks_blocked"] == 1 and kpis["queue_size"] == 2
    assert snap["models_reachable"] is False and snap["models"] == []

    # 2. Recomputed from raw rows, independently of the dashboard SQL.
    conn = await asyncpg.connect(ASYNCPG_DSN)
    try:
        decisions = await conn.fetch(
            "SELECT invoice_id, route::text AS route, created_at, id, reasons FROM decisions"
        )
        field_sources = [r[0] for r in await conn.fetch("SELECT source::text FROM fields")]
        label_sources = [r[0] for r in await conn.fetch("SELECT previous_source::text FROM labels")]
        latencies = sorted(r[0] for r in await conn.fetch("SELECT latency_ms FROM llm_calls"))
        status_rows = [r[0] for r in await conn.fetch("SELECT status::text FROM invoices")]
    finally:
        await conn.close()
    latest: dict[str, tuple[datetime, int, str]] = {}
    for row in decisions:
        key = (row["created_at"], row["id"])
        if row["invoice_id"] not in latest or key > latest[row["invoice_id"]][:2]:
            latest[row["invoice_id"]] = (row["created_at"], row["id"], row["route"])
    routes_now = [v[2] for v in latest.values()]
    automated = {"extractor_a", "extractor_b", "arbiter"}
    blocked_by_scan = {
        row["invoice_id"]
        for row in decisions
        for reason in (json.loads(row["reasons"] or "null") or [])
        if reason["rule"] == "P4"
        and reason["route"] == "block"
        and reason["message"].startswith("scan ")
    }
    assert kpis["stp_rate"] == pytest.approx(routes_now.count("auto") / len(routes_now))
    assert kpis["auto_accept_error_rate"] == pytest.approx(
        sum(s in automated for s in label_sources) / sum(s in automated for s in field_sources)
    )
    assert kpis["llm_latency_p50_ms"] == pytest.approx(latencies[len(latencies) // 2])
    assert kpis["attacks_blocked"] == len(blocked_by_scan)
    assert kpis["queue_size"] == status_rows.count("awaiting_review")
    stages = {s["stage"]: s["count"] for s in snap["stages"]}
    for status in set(status_rows):
        assert stages[status] == status_rows.count(status)
    assert {r["id"] for r in snap["recent"]} == set(ids.values())
    await db_session.get_engine().dispose()


def test_ops_dashboard_numbers_equal_independent_checks(
    api_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_ops_dashboard_scenario(monkeypatch))


async def _review_corrections_scenario() -> None:
    """ADR-0012 amendment: an approval while a classified value is unresolved is a 422 that
    records nothing and signals nothing; once the reviewer fixes the fields, the approval
    carries typed corrections to the workflow and the audit log."""
    signaller = _RecordingSignaller()
    invoice_id = _invoice_id("unclassified-invoice")
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer1", password="pw", roles=[Role.REVIEWER])
        await _create_verified_vendor(
            session,
            vendor_id="vendorA",
            iban="AE070331234567890123456",
            verified_by="reviewer1",
            changed_by="reviewer1",
            changed_at=datetime.now(UTC) - timedelta(days=365),
        )
        await _create_invoice(
            session,
            invoice_id=invoice_id,
            vendor_id="vendorA",
            total=Decimal("500.00"),
            classified=False,
        )

    transport = ASGITransport(app=_app(signaller))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "reviewer1", "pw")
        response = await client.post(
            f"/invoices/{invoice_id}/decisions",
            json={"action": "approve"},
            headers=_auth(token),
        )
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert "invoice_type_code" in detail and "line_items[1].vat_category" in detail
        assert signaller.sent == []

        for field_name, value in (
            ("invoice_type_code", "381"),
            ("line_items[1].vat_category", "Z"),
            ("line_items[1].vat_rate", "0"),
        ):
            response = await client.patch(
                f"/invoices/{invoice_id}/fields",
                json={"field": field_name, "value": value, "reason": "per the document"},
                headers=_auth(token),
            )
            assert response.status_code == 204, response.text
        response = await client.post(
            f"/invoices/{invoice_id}/decisions",
            json={"action": "approve"},
            headers=_auth(token),
        )
        assert response.status_code == 204, response.text

    assert signaller.sent == [(invoice_id, True)]
    corrections = signaller.corrections[0]
    assert corrections is not None
    assert corrections.kind.value == "credit_note"
    assert corrections.currency == "AED"
    assert [(c.index, c.vat_category.value, c.vat_rate) for c in corrections.lines] == [
        (1, "Z", Decimal("0"))
    ]
    async with get_session() as session:
        invoice = await session.get(Invoice, invoice_id)
        assert invoice is not None and invoice.status == DbInvoiceStatus.APPROVED
        payload = await session.scalar(
            select(AuditLog.payload_json).where(
                AuditLog.entity == f"invoice:{invoice_id}",
                AuditLog.action == "decision.approve",
            )
        )
        assert payload is not None and payload["corrections"]["kind"] == "credit_note"


def test_approval_needs_resolved_classification_and_carries_corrections(api_schema: str) -> None:
    asyncio.run(_review_corrections_scenario())


async def _address_and_totals_scenario() -> None:
    """ADR-0012 amendment 4: a supplier address naming no emirate refuses the approval until the
    reviewer fixes it, and the fix travels in the corrections; an invoice whose routing recorded
    a `totals_mismatch` cannot be approved at all (422, nothing recorded, no signal)."""
    from apx.db.enums import DecisionAction
    from apx.db.models import Decision

    signaller = _RecordingSignaller()
    address_id = _invoice_id("address-invoice")
    totals_id = _invoice_id("totals-invoice")
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="reviewer1", password="pw", roles=[Role.REVIEWER])
        await _create_verified_vendor(
            session,
            vendor_id="vendorA",
            iban="AE070331234567890123456",
            verified_by="reviewer1",
            changed_by="reviewer1",
            changed_at=datetime.now(UTC) - timedelta(days=365),
        )
        for invoice_id in (address_id, totals_id):
            await _create_invoice(
                session, invoice_id=invoice_id, vendor_id="vendorA", total=Decimal("500.00")
            )
        session.add(
            Field(
                invoice_id=address_id,
                name="supplier_address",
                value="Warehouse 4, Ras Al Khalaimah, United Arab Emirates",
                normalized_value=None,
                source=FieldSource.EXTRACTOR_A,
            )
        )
        session.add(
            Decision(
                invoice_id=totals_id,
                action=DecisionAction.POLICY_ROUTE,
                route=Route.REVIEW,
                reasons=[
                    {
                        "rule": "validation",
                        "route": "review",
                        "message": "validation found error-severity violations: totals_mismatch",
                        "violations": [{"code": "totals_mismatch", "field": "subtotal"}],
                    },
                    {"payee": {"refusal": None}},
                ],
            )
        )

    transport = ASGITransport(app=_app(signaller))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "reviewer1", "pw")
        approve = {"action": "approve"}
        response = await client.post(
            f"/invoices/{totals_id}/decisions", json=approve, headers=_auth(token)
        )
        assert response.status_code == 422, response.text
        assert "subtotal" in response.json()["detail"]
        assert "reject the invoice" in response.json()["detail"]

        response = await client.post(
            f"/invoices/{address_id}/decisions", json=approve, headers=_auth(token)
        )
        assert response.status_code == 422, response.text
        assert response.json()["detail"].endswith("correct these fields first: supplier_address")
        assert signaller.sent == []

        response = await client.patch(
            f"/invoices/{address_id}/fields",
            json={
                "field": "supplier_address",
                "value": "Warehouse 4, Ras Al Khaimah",
                "reason": "per the document",
            },
            headers=_auth(token),
        )
        assert response.status_code == 204, response.text
        response = await client.post(
            f"/invoices/{address_id}/decisions", json=approve, headers=_auth(token)
        )
        assert response.status_code == 204, response.text

    assert signaller.sent == [(address_id, True)]
    corrections = signaller.corrections[0]
    assert corrections is not None
    assert corrections.supplier_address == "Warehouse 4, Ras Al Khaimah"
    async with get_session() as session:
        refused = await session.get(Invoice, totals_id)
        assert refused is not None and refused.status == DbInvoiceStatus.AWAITING_REVIEW


def test_address_fix_reaches_the_corrections_and_totals_mismatch_refuses_approval(
    api_schema: str,
) -> None:
    asyncio.run(_address_and_totals_scenario())
