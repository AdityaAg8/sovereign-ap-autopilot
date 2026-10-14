"""Integration test for `apx.policy.writer.SqlPaymentStateWriter` against a real, throwaway
Postgres container: approval, bank-detail change round-trip on the Postgres schema, idempotent
replay and a stale-grant rejection (I3, ADR-0008). Marked `integration`
(needs Docker), like `tests/integration/test_migrations.py`."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.common.errors import PolicyDeniedError
from apx.common.invoice import Invoice as InvoiceModel
from apx.common.invoice import Party
from apx.common.master import BankAccount, BankDetailChange
from apx.common.master import User as UserModel
from apx.common.master import Vendor as VendorModel
from apx.common.types import InvoiceId, Language, Money, Role, UserId, VendorId
from apx.db.enums import BankDetailSource, DbInvoiceStatus, ProposalKind, ProposalStatus
from apx.db.models import Approval as ApprovalRow
from apx.db.models import BankDetail as BankDetailRow
from apx.db.models import BankVerification as BankVerificationRow
from apx.db.models import Invoice as InvoiceRow
from apx.db.models import Proposal as ProposalRow
from apx.db.models import RoleDef, User, UserRole
from apx.db.models import Vendor as VendorRow
from apx.policy.interfaces import ActionRequest, Capability, Grant, subject_digest
from apx.policy.writer import SqlPaymentStateWriter, proposal_model

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-m5-2-writer"
HOST_PORT = 55437
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"

VENDOR_ID = VendorId("vendor-1")
USER_ID = UserId("user-1")
INVOICE_ID = InvoiceId("a" * 64)
IBAN = "AE070331234567890123456"

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

_FIXTURE_TABLES = (
    "approvals",
    "bank_details_history",
    "bank_details",
    "bank_verifications",
    "invoices",
    "user_roles",
    "roles",
    "vendors",
    "users",
)


class FakeAuditLog:
    """In-memory `AuditLog`: `apx.audit`'s concrete hash-chained implementation lives on its
    own module branch; this writer only needs something that accepts `AuditEvent`s (I4)."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append(self, event: AuditEvent) -> AuditEntry:
        self.events.append(event)
        return AuditEntry(
            id=len(self.events),
            ts=datetime.now(UTC),
            actor=event.actor,
            action=event.action,
            entity=event.entity,
            payload=event.payload,
            prev_hash="0" * 64,
            hash="1" * 64,
        )


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
    try:
        asyncio.run(_wait_for_postgres(ASYNCPG_DSN))
        os.environ["DATABASE_URL"] = DATABASE_URL
        cfg = _alembic_config()
        try:
            command.upgrade(cfg, "head")
            yield DATABASE_URL
        finally:
            command.downgrade(cfg, "base")
            del os.environ["DATABASE_URL"]
    finally:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture
async def session_factory(postgres_url: str) -> AsyncIterator[SessionFactory]:
    """A `SqlPaymentStateWriter`-compatible factory: commits on clean exit, rolls back on
    error (matching `apx.db.session.get_session`'s contract, which a bare `async_sessionmaker`
    call does not provide on its own)."""
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


@pytest.fixture
async def seeded(session_factory: SessionFactory) -> None:
    """One vendor with a single admin-verified IBAN, one approver user and one invoice
    awaiting review -- the minimal fixture every test in this file builds on. Truncates first
    since `postgres_url` is module-scoped (one container reused across this file's tests)."""
    async with session_factory() as session:
        truncate_sql = f"TRUNCATE {', '.join(_FIXTURE_TABLES)} RESTART IDENTITY CASCADE"
        await session.execute(text(truncate_sql))
        session.add(
            User(id=USER_ID, email="a@example.com", display_name="Approver", password_hash="x")
        )
        role = RoleDef(name=Role.APPROVER)
        session.add(role)
        await session.flush()
        session.add(UserRole(user_id=USER_ID, role_id=role.id))

        session.add(VendorRow(id=VENDOR_ID, name="Acme Supplies", trn=None))
        await session.flush()

        verification = BankVerificationRow(vendor_id=VENDOR_ID, iban=IBAN, verified_by=USER_ID)
        session.add(verification)
        await session.flush()
        session.add(
            BankDetailRow(
                vendor_id=VENDOR_ID,
                iban=IBAN,
                source=BankDetailSource.ADMIN_CALLBACK,
                verification_id=verification.id,
                is_current=True,
                changed_by=USER_ID,
            )
        )

        session.add(
            InvoiceRow(
                id=INVOICE_ID,
                document_sha256=INVOICE_ID,
                vendor_id=VENDOR_ID,
                language=Language.EN,
                number="INV-1",
                issue_date=date(2026, 1, 1),
                currency="AED",
                subtotal=Decimal("100.00"),
                vat_total=Decimal("5.00"),
                total=Decimal("105.00"),
                status=DbInvoiceStatus.AWAITING_REVIEW,
            )
        )


def _actor(roles: frozenset[Role] = frozenset({Role.APPROVER})) -> UserModel:
    return UserModel(id=USER_ID, display_name="", roles=roles)


def _seeded_invoice() -> InvoiceModel:
    party = Party(name="", trn=None, address=None, country="AE")
    return InvoiceModel(
        id=INVOICE_ID,
        document_sha256=INVOICE_ID,
        language=Language.EN,
        number="INV-1",
        issue_date=date(2026, 1, 1),
        due_date=None,
        supplier=party,
        buyer=party,
        vendor_id=VENDOR_ID,
        po_number=None,
        lines=(),
        subtotal=Money(amount=Decimal("100.00"), currency="AED"),
        vat_total=Money(amount=Decimal("5.00"), currency="AED"),
        total=Money(amount=Decimal("105.00"), currency="AED"),
        iban=None,
    )


def _seeded_vendor() -> VendorModel:
    return VendorModel(
        id=VENDOR_ID,
        name="Acme Supplies",
        trn=None,
        bank_accounts=(
            BankAccount(
                iban=IBAN,
                verified=True,
                verified_by=USER_ID,
                verified_at=datetime.now(UTC),
                source="admin_callback",
            ),
        ),
    )


def _approval_grant() -> Grant:
    """A `Grant` whose digest matches the `seeded` fixture's current state exactly."""
    request = ActionRequest(
        actor=_actor(),
        action=Capability.APPROVE_PAYMENT,
        invoice=_seeded_invoice(),
        vendor=_seeded_vendor(),
        at=datetime.now(UTC),
    )
    return Grant(
        actor=USER_ID,
        action=Capability.APPROVE_PAYMENT,
        subject_digest=subject_digest(request),
        reasons=(),
        audit_entry_id=1,
        issued_at=datetime.now(UTC),
    )


@pytest.mark.usefixtures("seeded")
async def test_record_approval_creates_approval_and_sets_status(
    session_factory: SessionFactory,
) -> None:
    audit = FakeAuditLog()
    writer = SqlPaymentStateWriter(session_factory, audit)
    grant = _approval_grant()

    await writer.record_approval(INVOICE_ID, grant)

    async with session_factory() as session:
        row = await session.get(InvoiceRow, INVOICE_ID)
        assert row is not None
        assert row.status is DbInvoiceStatus.APPROVED
        approvals = (
            await session.scalars(select(ApprovalRow).where(ApprovalRow.invoice_id == INVOICE_ID))
        ).all()
        assert len(approvals) == 1
        assert approvals[0].role_at_approval is Role.APPROVER
    assert any(e.action == "payment_state:approve" for e in audit.events)


@pytest.mark.usefixtures("seeded")
async def test_record_approval_is_idempotent(
    session_factory: SessionFactory,
) -> None:
    audit = FakeAuditLog()
    writer = SqlPaymentStateWriter(session_factory, audit)
    grant = _approval_grant()

    await writer.record_approval(INVOICE_ID, grant)
    await writer.record_approval(INVOICE_ID, grant)  # replay: must not create a 2nd approval

    async with session_factory() as session:
        approvals = (
            await session.scalars(select(ApprovalRow).where(ApprovalRow.invoice_id == INVOICE_ID))
        ).all()
        assert len(approvals) == 1


@pytest.mark.usefixtures("seeded")
async def test_record_approval_rejects_stale_grant(
    session_factory: SessionFactory,
) -> None:
    audit = FakeAuditLog()
    writer = SqlPaymentStateWriter(session_factory, audit)
    grant = _approval_grant()
    stale = grant.model_copy(update={"subject_digest": "0" * 64})

    with pytest.raises(PolicyDeniedError):
        await writer.record_approval(INVOICE_ID, stale)

    async with session_factory() as session:
        row = await session.get(InvoiceRow, INVOICE_ID)
        assert row is not None
        assert row.status is DbInvoiceStatus.AWAITING_REVIEW  # unchanged


@pytest.mark.usefixtures("seeded")
async def test_record_bank_change_replaces_current_account(
    session_factory: SessionFactory,
) -> None:
    audit = FakeAuditLog()
    writer = SqlPaymentStateWriter(session_factory, audit)
    new_iban = "AE070331234567890199999"
    change = BankDetailChange(
        vendor_id=VENDOR_ID,
        changed_by=USER_ID,
        changed_at=datetime.now(UTC),
        old_iban=IBAN,
        new_iban=new_iban,
    )
    request = ActionRequest(
        actor=_actor(),
        action=Capability.CHANGE_BANK_DETAILS,
        invoice=None,
        vendor=_seeded_vendor(),
        at=datetime.now(UTC),
        bank_change=change,
    )
    grant = Grant(
        actor=USER_ID,
        action=Capability.CHANGE_BANK_DETAILS,
        subject_digest=subject_digest(request),
        reasons=(),
        audit_entry_id=2,
        issued_at=datetime.now(UTC),
    )

    await writer.record_bank_change(change, grant)

    async with session_factory() as session:
        rows = (
            await session.scalars(
                select(BankDetailRow).where(
                    BankDetailRow.vendor_id == VENDOR_ID, BankDetailRow.is_current.is_(True)
                )
            )
        ).all()
        assert len(rows) == 1
        assert rows[0].iban == new_iban
        assert rows[0].verified is False
    assert any(e.action == "payment_state:bank_change" for e in audit.events)


async def _add_approve_proposal(session_factory: SessionFactory) -> ProposalRow:
    async with session_factory() as session:
        row = ProposalRow(
            id="proposal-approve-1",
            invoice_id=INVOICE_ID,
            kind=ProposalKind.APPROVE_PAYMENT,
            payload={"note": "agent suggests approval"},
            created_by="agent",
            status=ProposalStatus.PENDING,
        )
        session.add(row)
        await session.flush()
        await session.refresh(row)
        return row


def _proposal_grant(proposal: ProposalRow) -> Grant:
    """An EXECUTE_PROPOSAL grant: its digest covers the proposal id and payload, exactly as
    `DefaultPolicyGuard` computes it for the API."""
    request = ActionRequest(
        actor=_actor(),
        action=Capability.EXECUTE_PROPOSAL,
        invoice=_seeded_invoice(),
        vendor=_seeded_vendor(),
        at=datetime.now(UTC),
        proposal=proposal_model(proposal),
    )
    return Grant(
        actor=USER_ID,
        action=Capability.EXECUTE_PROPOSAL,
        subject_digest=subject_digest(request),
        reasons=(),
        audit_entry_id=1,
        issued_at=datetime.now(UTC),
    )


@pytest.mark.usefixtures("seeded")
async def test_execute_approve_proposal_matches_a_proposal_bound_grant(
    session_factory: SessionFactory,
) -> None:
    audit = FakeAuditLog()
    writer = SqlPaymentStateWriter(session_factory, audit)
    proposal = await _add_approve_proposal(session_factory)

    await writer.execute_proposal(proposal.id, _proposal_grant(proposal))

    async with session_factory() as session:
        invoice = await session.get(InvoiceRow, INVOICE_ID)
        assert invoice is not None and invoice.status is DbInvoiceStatus.APPROVED
        executed = await session.get(ProposalRow, proposal.id)
        assert executed is not None and executed.status is ProposalStatus.EXECUTED


@pytest.mark.usefixtures("seeded")
async def test_execute_proposal_rejects_a_grant_over_a_different_payload(
    session_factory: SessionFactory,
) -> None:
    audit = FakeAuditLog()
    writer = SqlPaymentStateWriter(session_factory, audit)
    proposal = await _add_approve_proposal(session_factory)
    tampered = proposal_model(proposal).model_copy(update={"payload": {"note": "changed"}})
    request = ActionRequest(
        actor=_actor(),
        action=Capability.EXECUTE_PROPOSAL,
        invoice=_seeded_invoice(),
        vendor=_seeded_vendor(),
        at=datetime.now(UTC),
        proposal=tampered,
    )
    grant = _proposal_grant(proposal).model_copy(update={"subject_digest": subject_digest(request)})

    with pytest.raises(PolicyDeniedError):
        await writer.execute_proposal(proposal.id, grant)

    async with session_factory() as session:
        invoice = await session.get(InvoiceRow, INVOICE_ID)
        assert invoice is not None and invoice.status is DbInvoiceStatus.AWAITING_REVIEW


def _bank_change_grant(new_iban: str) -> tuple[BankDetailChange, Grant]:
    change = BankDetailChange(
        vendor_id=VENDOR_ID,
        changed_by=USER_ID,
        changed_at=datetime.now(UTC),
        old_iban=IBAN,
        new_iban=new_iban,
    )
    request = ActionRequest(
        actor=_actor(),
        action=Capability.CHANGE_BANK_DETAILS,
        invoice=None,
        vendor=_seeded_vendor(),
        at=datetime.now(UTC),
        bank_change=change,
    )
    grant = Grant(
        actor=USER_ID,
        action=Capability.CHANGE_BANK_DETAILS,
        subject_digest=subject_digest(request),
        reasons=(),
        audit_entry_id=3,
        issued_at=datetime.now(UTC),
    )
    return change, grant


@pytest.mark.usefixtures("seeded")
async def test_concurrent_bank_changes_leave_exactly_one_current_account(
    session_factory: SessionFactory,
) -> None:
    """Both grants were issued against the same old IBAN; the vendor-row lock serialises the
    writes, so the second sees the first's new account, fails STALE_GRANT, and the vendor never
    ends up with two current accounts."""
    writer = SqlPaymentStateWriter(session_factory, FakeAuditLog())
    first = _bank_change_grant("AE070331234567890100001")
    second = _bank_change_grant("AE070331234567890100002")

    results = await asyncio.gather(
        writer.record_bank_change(*first),
        writer.record_bank_change(*second),
        return_exceptions=True,
    )

    assert sum(isinstance(r, PolicyDeniedError) for r in results) == 1
    assert sum(r is None for r in results) == 1
    async with session_factory() as session:
        rows = (
            await session.scalars(
                select(BankDetailRow).where(
                    BankDetailRow.vendor_id == VENDOR_ID, BankDetailRow.is_current.is_(True)
                )
            )
        ).all()
        assert len(rows) == 1
