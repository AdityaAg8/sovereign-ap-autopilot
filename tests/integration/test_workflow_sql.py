"""Integration tests for the M5.6 Postgres adapters in `apx.workflows.sql` against a real,
migrated Postgres: vendor lookup by TRN and bank state, idempotent invoice/field persistence,
route + confidence updates, fraud-flag replacement, and the fraud history snapshot."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from apx.api.security import verify_password
from apx.common.invoice import Invoice, Party
from apx.common.types import FieldValue, InvoiceId, Language, Money, Route, Sha256Hex, VendorId
from apx.db.enums import BankDetailSource, DbInvoiceStatus, DecisionAction
from apx.db.models import BankDetail as BankDetailRow
from apx.db.models import BankVerification as BankVerificationRow
from apx.db.models import Decision as DecisionRow
from apx.db.models import Field as FieldRow
from apx.db.models import FraudFlag as FraudFlagRow
from apx.db.models import Invoice as InvoiceRow
from apx.db.models import User
from apx.db.models import Vendor as VendorRow
from apx.extract.interfaces import ExtractionResult, ExtractorId, ExtractorOutput, FieldName
from apx.fraud.interfaces import FraudFlag, FraudSeverity, FraudTypology
from apx.policy.interfaces import PayeeMatch, RoutedPayee, RoutingDecision, payee_iban_digest
from apx.trust.interfaces import FieldDecision, TrustDecision
from apx.workflows.corrections import KIND_FIELD
from apx.workflows.sql import PgInvoiceHistory, PgVendorDirectory, SqlInvoiceRecords
from factory.erp import generate_erp
from factory.seed_db import email_for, seed

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-m5-6-sql"
HOST_PORT = 55441
DATABASE_URL = f"postgresql+asyncpg://apx:apx@127.0.0.1:{HOST_PORT}/apx"
ASYNCPG_DSN = f"postgresql://apx:apx@127.0.0.1:{HOST_PORT}/apx"

VENDOR = VendorId("V-100")
TRN = "100000000000003"
IBAN = "AE070331234567890123456"
FIRST = InvoiceId("d" * 64)
SECOND = InvoiceId("e" * 64)

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


async def _wait_for_postgres(timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            conn = await asyncpg.connect(ASYNCPG_DSN)
        except (OSError, asyncpg.PostgresError):
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.5)
            continue
        await conn.close()
        return


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
            "POSTGRES_USER=apx",
            "-e",
            "POSTGRES_PASSWORD=apx",
            "-e",
            "POSTGRES_DB=apx",
            "pgvector/pgvector:pg16-trixie",
        ],
        check=True,
        capture_output=True,
    )
    try:
        asyncio.run(_wait_for_postgres())
        os.environ["DATABASE_URL"] = DATABASE_URL
        cfg = Config(str(REPO_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(REPO_ROOT / "db" / "migrations"))
        try:
            command.upgrade(cfg, "head")
            yield DATABASE_URL
        finally:
            command.downgrade(cfg, "base")
            del os.environ["DATABASE_URL"]
    finally:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture
async def sessions(postgres_url: str) -> AsyncIterator[SessionFactory]:
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

    async with factory() as session:
        await session.execute(
            text(
                "TRUNCATE invoices, bank_details, bank_details_history, bank_verifications, "
                "vendors, users CASCADE"
            )
        )
        session.add(User(id="admin", email="a@x.ae", display_name="Admin", password_hash="x"))
        session.add(VendorRow(id=VENDOR, name="Gulf Steel", trn=TRN))
        await session.flush()
        verification = BankVerificationRow(vendor_id=VENDOR, iban=IBAN, verified_by="admin")
        session.add(verification)
        await session.flush()
        session.add(
            BankDetailRow(
                vendor_id=VENDOR,
                iban=IBAN,
                source=BankDetailSource.ADMIN_CALLBACK,
                verification_id=verification.id,
                is_current=True,
                changed_by="admin",
            )
        )
    try:
        yield factory
    finally:
        await engine.dispose()


def _invoice(invoice_id: str, total: str = "1050.00") -> Invoice:
    party = Party(name="Gulf Steel", trn=TRN, address=None)
    return Invoice(
        id=InvoiceId(invoice_id),
        document_sha256=Sha256Hex(invoice_id),
        language=Language.EN,
        number="INV-7",
        issue_date=date(2026, 9, 1),
        due_date=None,
        supplier=party,
        buyer=party,
        vendor_id=VENDOR,
        po_number=None,
        lines=(),
        subtotal=Money(amount=Decimal("1000.00")),
        vat_total=Money(amount=Decimal("50.00")),
        total=Money(amount=Decimal(total)),
        iban=IBAN,
    )


def _result(sha: str) -> ExtractionResult:
    out = ExtractorOutput(
        extractor=ExtractorId.A,
        model="qwen-a",
        fields=(),
        line_items=(),
        logprobs={},
        valid_json=True,
    )
    grounded = {
        FieldName.INVOICE_NUMBER: FieldValue(
            name="invoice_number", raw="INV-7", normalized="inv 7"
        ),
        FieldName.TOTAL: FieldValue(name="total", raw="1,050.00", normalized="1050.00"),
    }
    return ExtractionResult(
        document_sha256=Sha256Hex(sha),
        a=out,
        b=out,
        arbiter=None,
        disputed=frozenset(),
        grounded=grounded,
    )


async def test_vendor_directory_resolves_trn_and_bank_state(sessions: SessionFactory) -> None:
    directory = PgVendorDirectory(sessions)
    assert await directory.vendor_id_for_trn(TRN) == VENDOR
    assert await directory.vendor_id_for_trn("100000000000999") is None
    vendor, changes = await directory.lookup(VENDOR)
    assert vendor is not None and [a.iban for a in vendor.bank_accounts if a.verified] == [IBAN]
    assert [c.new_iban for c in changes] == [IBAN]
    assert await directory.lookup(None) == (None, ())


async def test_records_are_idempotent_and_track_the_workflow(sessions: SessionFactory) -> None:
    records = SqlInvoiceRecords(sessions)
    await records.save(_invoice(FIRST, "999.00"), _result(FIRST))
    await records.save(_invoice(FIRST), _result(FIRST))  # a retried activity
    flag = FraudFlag(
        rule_id="duplicate_invoice",
        typology=FraudTypology.DUPLICATE,
        severity=FraudSeverity.HIGH,
        score=0.9,
        evidence="e",
        explanation="x",
    )
    await records.flags(FIRST, [flag, flag])
    await records.flags(FIRST, [flag])  # retried: replaced, not appended
    trust = TrustDecision(
        fields=(FieldDecision(field=FieldName.TOTAL, p_correct=0.42, accepted=False),),
        all_accepted=False,
    )
    payee = RoutedPayee(
        masked_iban="****3456",
        match=PayeeMatch.SOLE_UNCLAIMED,
        verified_count=1,
        iban_sha256=payee_iban_digest(IBAN),
    )
    routing = RoutingDecision(route=Route.REVIEW, reasons=(), payee=payee)
    await records.routed(FIRST, routing, trust)

    async with sessions() as session:
        row = await session.get(InvoiceRow, FIRST)
        assert row is not None
        assert row.total == Decimal("1050.00")
        assert (row.status, row.route) == (DbInvoiceStatus.AWAITING_REVIEW, Route.REVIEW)
        field_rows = (
            await session.scalars(select(FieldRow).where(FieldRow.invoice_id == FIRST))
        ).all()
        fields = {f.name: f for f in field_rows}
        # The grounded fields plus the ADR-0012 classification row (BT-3; no lines here), each
        # exactly once although `save` ran twice.
        assert set(fields) == {"invoice_number", "total", KIND_FIELD}
        assert len(field_rows) == len(fields)
        assert fields[KIND_FIELD].value is None  # the fixture's kind is unresolved
        assert fields["total"].confidence == pytest.approx(0.42)
        assert fields["total"].model_revision == "qwen-a"
        flags = await session.scalar(
            select(func.count()).select_from(FraudFlagRow).where(FraudFlagRow.invoice_id == FIRST)
        )
        assert flags == 1
        # ADR-0014 §7: the route decision is persisted with the masked pinned payee only.
        decision = await session.scalar(
            select(DecisionRow).where(
                DecisionRow.invoice_id == FIRST,
                DecisionRow.action == DecisionAction.POLICY_ROUTE,
            )
        )
        assert decision is not None and decision.route is Route.REVIEW
        assert decision.reasons == [
            {"payee": {"masked_iban": "****3456", "match": "sole_unclaimed", "verified_count": 1}}
        ]
        assert IBAN not in str(decision.reasons)

    await records.set_status(FIRST, DbInvoiceStatus.SENT)
    async with sessions() as session:
        row = await session.get(InvoiceRow, FIRST)
        assert row is not None and row.status is DbInvoiceStatus.SENT


async def test_history_snapshot_loads_vendor_invoices_iban_users_and_changes(
    sessions: SessionFactory,
) -> None:
    records = SqlInvoiceRecords(sessions)
    await records.save(_invoice(FIRST), _result(FIRST))
    await records.save(_invoice(SECOND), _result(SECOND))
    snapshot = await PgInvoiceHistory(sessions).snapshot(_invoice(SECOND))
    assert [i.id for i in snapshot.recent_for_vendor(VENDOR, 14)] == [FIRST]
    assert list(snapshot.vendors_using_iban(IBAN)) == [VENDOR]
    assert [c.new_iban for c in snapshot.bank_changes(VENDOR, 30)] == [IBAN]
    assert snapshot.purchase_order("PO-1") is None


async def test_seed_loads_the_factory_erp(sessions: SessionFactory) -> None:
    erp = generate_erp(2026, n_vendors=5, n_users=4)
    await seed(erp, "pw-for-test", sessions)
    await seed(erp, "pw-for-test", sessions)  # re-seeding replaces, never duplicates
    async with sessions() as session:
        users = (await session.scalars(select(User))).all()
        assert sorted(u.email for u in users) == sorted(email_for(u.id) for u in erp.users)
        assert all(verify_password(u.password_hash, "pw-for-test") for u in users)
        assert await session.scalar(select(func.count()).select_from(VendorRow)) == 5
    directory = PgVendorDirectory(sessions)
    vendor = erp.vendors[0]
    assert await directory.vendor_id_for_trn(vendor.trn or "") == vendor.id
    found, _ = await directory.lookup(vendor.id)
    assert found is not None
    assert {a.iban for a in found.bank_accounts if a.verified} == {
        a.iban for a in vendor.bank_accounts if a.verified
    }
