"""Integration test for `apx.workflows.workflow.InvoiceWorkflow`'s crash-recovery guarantee
(PRD §10 M5.2: "kill the worker mid-invoice, restart: completes exactly once"): a real Temporal
dev server plus a real, throwaway Postgres container back `DefaultInvoiceActivities`'s
idempotency-guarded `send_to_asp`/`archive` steps; every other pipeline stage is a fake (those
modules live on their own branches). Worker 1 is shut down while `archive` is in flight; worker
2 resumes it. Also checks `mark_failed` against the same Postgres: FAILED lands on the row
unless it is already terminal. Marked `integration` (needs Docker), like
`tests/integration/test_migrations.py`.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import asyncpg  # type: ignore[import-untyped]
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

from apx.asp_mock.interfaces import ArchiveRecord, AspReceipt, AspSubmission
from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.common.errors import FailureNotRecordedError
from apx.common.invoice import Invoice, InvoiceKind, InvoiceLine, Party, VatCategory
from apx.common.master import BankDetailChange, Vendor
from apx.common.types import IdempotencyKey, InvoiceId, Language, Money, Route, VendorId
from apx.db.enums import DbInvoiceStatus
from apx.db.models import IdempotencyRecord
from apx.db.models import IngestedDocument as IngestedDocumentRow
from apx.db.models import Invoice as InvoiceRow
from apx.einvoice.interfaces import DocumentTypeCode, EInvoiceDocument, SchematronResult
from apx.extract.interfaces import ExtractionResult, ExtractorId, ExtractorOutput, FieldName
from apx.fraud.interfaces import FraudReport
from apx.ingest.interfaces import IngestedDocument
from apx.parse.interfaces import Page, ParsedDocument
from apx.parse.store import FileParsedDocumentStore
from apx.policy.interfaces import (
    ActionRequest,
    AuthorizationDecision,
    RoutingDecision,
    RoutingInput,
)
from apx.scan.interfaces import ScanReport
from apx.trust.interfaces import FieldFeatures, TrustDecision
from apx.validate.interfaces import ValidationReport
from apx.workflows.activities import DefaultInvoiceActivities, InvoiceActivityDeps
from apx.workflows.idempotency import SqlIdempotencyStore
from apx.workflows.interfaces import InvoiceStatus, InvoiceWorkflowInput
from apx.workflows.sql import SqlInvoiceRecords
from apx.workflows.worker import TASK_QUEUE, build_worker
from apx.workflows.workflow import InvoiceWorkflow

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-m5-2-workflow"
HOST_PORT = 55438
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"

INVOICE_ID = InvoiceId("b" * 64)
WORKFLOW_INPUT = InvoiceWorkflowInput(
    invoice_id=INVOICE_ID, document_sha256=INVOICE_ID, filename="invoice.pdf"
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


SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


@pytest.fixture
async def session_factory(postgres_url: str) -> AsyncIterator[SessionFactory]:
    """Commits on clean exit, rolls back on error, matching `apx.db.session.get_session`'s
    contract that `InvoiceActivityDeps.session_factory` requires."""
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
async def seeded_invoice(session_factory: SessionFactory) -> None:
    """The `invoices` row `apx.ingest` (M5.5) would have created before starting the
    workflow; `archive` updates its `status`, so the FK requires it to already exist."""
    async with session_factory() as session:
        session.add(
            InvoiceRow(
                id=INVOICE_ID,
                document_sha256=INVOICE_ID,
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


class FakeAuditLog:
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


class FakeDocumentStore:
    def get(self, sha256: str) -> bytes:
        return b"%PDF-fake"

    def put(  # pragma: no cover
        self, data: bytes, filename: str, source: Literal["watch", "upload"]
    ) -> IngestedDocument:
        raise NotImplementedError

    def exists(self, sha256: str) -> bool:  # pragma: no cover
        return True


class FakeScanner:
    def scan(self, document_sha256: str, pdf_bytes: bytes) -> ScanReport:
        return ScanReport(document_sha256=document_sha256, flags=(), injection_score=0.0)


class FakeParser:
    async def parse(self, document_sha256: str, data: bytes) -> ParsedDocument:
        return ParsedDocument(
            document_sha256=document_sha256,
            # Non-UTF-8 image bytes: the parse must reach extract/trust/fraud through the stored
            # ref, never as raw bytes in a Temporal payload.
            pages=tuple(
                Page(
                    index=i,
                    width_px=10,
                    height_px=10,
                    words=(),
                    lines=(),
                    tables=(),
                    image_png=b"\x89PNG\r\n\x1a\n\xff" + bytes([i]),
                )
                for i in range(2)
            ),
            reading_order=(),
            languages=frozenset({Language.EN}),
            parser_version="fake-1",
        )


class FakeExtractionPipeline:
    async def run(self, document: ParsedDocument) -> ExtractionResult:
        output_a = ExtractorOutput(
            extractor=ExtractorId.A,
            model="fake",
            fields=(),
            line_items=(),
            logprobs={},
            valid_json=True,
        )
        output_b = output_a.model_copy(update={"extractor": ExtractorId.B})
        return ExtractionResult(
            document_sha256=document.document_sha256,
            a=output_a,
            b=output_b,
            arbiter=None,
            disputed=frozenset(),
            grounded={},
        )


class FakeInvoiceBuilder:
    async def build(self, result: ExtractionResult, inp: InvoiceWorkflowInput) -> Invoice:
        party = Party(name="Acme Supplies LLC", trn=None, address=None, country="AE")
        net = Money(amount=Decimal("100.00"), currency="AED")
        # ADR-0012: the build derives BT-3 from a resolved kind and every line's VAT category
        # and refuses to guess, so an AUTO invoice must carry both.
        line = InvoiceLine(
            description="Steel bolts",
            quantity=Decimal("1"),
            unit_price=net,
            net_amount=net,
            vat_category=VatCategory.STANDARD,
            vat_rate=Decimal("5"),
        )
        return Invoice(
            id=inp.invoice_id,
            document_sha256=inp.document_sha256,
            language=Language.EN,
            number="INV-1",
            issue_date=date(2026, 1, 1),
            due_date=None,
            supplier=party,
            buyer=party,
            vendor_id=None,
            po_number=None,
            lines=(line,),
            subtotal=net,
            vat_total=Money(amount=Decimal("5.00"), currency="AED"),
            total=Money(amount=Decimal("105.00"), currency="AED"),
            iban=None,
            kind=InvoiceKind.INVOICE,
        )


class FakeFeatureBuilder:
    def features(
        self, result: ExtractionResult, document: ParsedDocument
    ) -> dict[FieldName, FieldFeatures]:
        return {}


class FakeTrustGate:
    def decide(self, features: Mapping[FieldName, FieldFeatures]) -> TrustDecision:
        return TrustDecision(fields=(), all_accepted=True)


class FakeValidator:
    def validate(self, invoice: Invoice) -> ValidationReport:
        return ValidationReport(violations=())


class FakeFraudEngine:
    async def assess(self, invoice: Invoice, page_images: Sequence[bytes]) -> FraudReport:
        return FraudReport(flags=(), max_score=0.0)


class FakeVendorDirectory:
    async def lookup(
        self, vendor_id: VendorId | None
    ) -> tuple[Vendor | None, tuple[BankDetailChange, ...]]:
        return None, ()


class FakePolicyEngine:
    def route(self, inp: RoutingInput) -> RoutingDecision:
        return RoutingDecision(route=Route.AUTO, reasons=())

    def authorize(self, request: ActionRequest) -> AuthorizationDecision:  # pragma: no cover
        """Never exercised: the workflow only calls `route`; `authorize` backs the future
        human-approval API layer (M5.3)."""
        raise NotImplementedError


class FakeUblBuilder:
    def build(
        self,
        invoice: Invoice,
        type_code: DocumentTypeCode,
        *,
        verified_payment_iban: str | None,
    ) -> EInvoiceDocument:
        xml = b"<Invoice/>"
        import hashlib

        return EInvoiceDocument(
            invoice_id=invoice.id,
            type_code=type_code,
            xml=xml,
            xml_sha256=hashlib.sha256(xml).hexdigest(),
            spec_version="1.0",
        )


class FakeSchematronValidator:
    def validate(self, xml: bytes) -> SchematronResult:
        return SchematronResult(passed=True, failures=(), rules_version="fake-1")


class FakeAspClient:
    """Exactly-once by key (mirrors the real `apx.asp_mock` contract): tracks both raw calls
    and distinct receipts actually created, so a test can tell "called twice under retry" apart
    from "submitted twice for real"."""

    def __init__(self) -> None:
        self.calls = 0
        self.receipts: dict[IdempotencyKey, AspReceipt] = {}

    async def submit(self, submission: AspSubmission) -> AspReceipt:
        self.calls += 1
        if submission.idempotency_key not in self.receipts:
            self.receipts[submission.idempotency_key] = AspReceipt(
                receipt_id=f"receipt-{len(self.receipts) + 1}",
                idempotency_key=submission.idempotency_key,
                status="accepted",
                received_at=datetime.now(UTC),
                message="ok",
            )
        return self.receipts[submission.idempotency_key]


class FakeArchive:
    """Like `FakeAspClient`; `store`'s *first* call for a not-yet-seen key blocks on
    `block_then_hang` so the test can crash worker 1 while this activity is genuinely
    in flight, then release it (worker 1 is already gone) so worker 2's retry succeeds."""

    def __init__(self, blocked: asyncio.Event, release: asyncio.Event) -> None:
        self.calls = 0
        self.records: dict[IdempotencyKey, ArchiveRecord] = {}
        self._blocked = blocked
        self._release = release

    async def store(self, key: IdempotencyKey, document: EInvoiceDocument) -> ArchiveRecord:
        self.calls += 1
        if key not in self.records:
            if len(self.records) == 0 and self.calls == 1:
                self._blocked.set()
                await self._release.wait()
            self.records[key] = ArchiveRecord(
                idempotency_key=key,
                xml_sha256=document.xml_sha256,
                archived_at=datetime.now(UTC),
                retain_until=date(2033, 1, 1),
            )
        return self.records[key]


async def test_crash_mid_archive_then_restart_completes_exactly_once(
    session_factory: SessionFactory, seeded_invoice: None, tmp_path: Path
) -> None:
    existing_path = os.environ.get("TEMPORAL_DEV_SERVER_PATH")
    async with await WorkflowEnvironment.start_local(
        data_converter=pydantic_data_converter, dev_server_existing_path=existing_path
    ) as env:
        asp_client = FakeAspClient()
        blocked = asyncio.Event()
        release = asyncio.Event()
        archive = FakeArchive(blocked, release)
        audit = FakeAuditLog()
        deps = InvoiceActivityDeps(
            documents=FakeDocumentStore(),
            scanner=FakeScanner(),
            parser=FakeParser(),
            parsed=FileParsedDocumentStore(tmp_path / "parsed"),
            extraction=FakeExtractionPipeline(),
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

        worker1 = build_worker(env.client, deps, task_queue=TASK_QUEUE)
        async with worker1:
            handle = await env.client.start_workflow(
                InvoiceWorkflow.run, WORKFLOW_INPUT, id=INVOICE_ID, task_queue=TASK_QUEUE
            )
            await asyncio.wait_for(blocked.wait(), timeout=15)
            # `archive` is now blocked mid-call on worker 1: send_to_asp already completed.
        # worker 1 is fully shut down here (its in-flight `archive` attempt is abandoned).

        release.set()  # let the abandoned attempt's coroutine unblock, if it ever resumes
        worker2 = build_worker(env.client, deps, task_queue=TASK_QUEUE)
        async with worker2:
            result = await asyncio.wait_for(handle.result(), timeout=30)

    assert result.status is InvoiceStatus.ARCHIVED
    assert result.route is Route.AUTO
    assert result.error is None

    # One ASP submission: send_to_asp fully completed on worker 1 before the crash.
    assert len(asp_client.receipts) == 1
    assert asp_client.calls == 1

    # One archive record, despite `store` being attempted on both workers.
    assert len(archive.records) == 1
    assert archive.calls >= 1

    async with session_factory() as session:
        row = await session.get(InvoiceRow, INVOICE_ID)
        assert row is not None
        assert row.status is DbInvoiceStatus.ARCHIVED  # the one payment-state transition

        asp_rows = (
            await session.scalars(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.invoice_id == INVOICE_ID, IdempotencyRecord.step == "asp_send"
                )
            )
        ).all()
        archive_rows = (
            await session.scalars(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.invoice_id == INVOICE_ID, IdempotencyRecord.step == "archive"
                )
            )
        ).all()
        assert len(asp_rows) == 1
        assert len(archive_rows) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("before", "after"),
    [
        (DbInvoiceStatus.APPROVED, DbInvoiceStatus.FAILED),  # post-review build failed
        (DbInvoiceStatus.SENT, DbInvoiceStatus.SENT),  # the ASP already has it
        (DbInvoiceStatus.BLOCKED, DbInvoiceStatus.BLOCKED),
    ],
)
async def test_mark_failed_writes_failed_but_never_overwrites_a_terminal_status(
    session_factory: SessionFactory,
    before: DbInvoiceStatus,
    after: DbInvoiceStatus,
) -> None:
    # One row per case: the Postgres container is shared by every test in this module.
    invoice_id = InvoiceId(hashlib.sha256(before.value.encode()).hexdigest())
    async with session_factory() as session:
        session.add(
            InvoiceRow(
                id=invoice_id,
                document_sha256=invoice_id,
                language=Language.EN,
                number="INV-F",
                issue_date=date(2026, 1, 1),
                currency="AED",
                subtotal=Decimal("100.00"),
                vat_total=Decimal("5.00"),
                total=Decimal("105.00"),
                status=before,
            )
        )
    audit = FakeAuditLog()
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
            audit=audit,
            records=SqlInvoiceRecords(session_factory),
            session_factory=session_factory,
            idempotency=SqlIdempotencyStore(),
        )
    )

    await activities.mark_failed(invoice_id, "SchematronFailedError: ibr-147-ae")

    async with session_factory() as session:
        row = await session.get(InvoiceRow, invoice_id)
        assert row is not None
        assert row.status is after
    assert [(e.action, e.payload) for e in audit.events] == [
        (
            "activity:failed",
            {"error": "SchematronFailedError: ibr-147-ae", "recorded_on": "invoice"},
        )
    ]


def _failure_activities(
    session_factory: SessionFactory, audit: FakeAuditLog
) -> DefaultInvoiceActivities:
    """`DefaultInvoiceActivities` with only what `mark_failed` touches wired."""
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
            asp_client=unused,
            archive=unused,
            audit=audit,
            records=SqlInvoiceRecords(session_factory),
            session_factory=session_factory,
            idempotency=SqlIdempotencyStore(),
        )
    )


@pytest.mark.asyncio
async def test_mark_failed_before_the_invoice_row_records_the_failure_on_the_document(
    session_factory: SessionFactory,
) -> None:
    # The live M5.6 case: `build_invoice` failed, so only the ingest row exists.
    sha = hashlib.sha256(b"failed-before-invoice").hexdigest()
    async with session_factory() as session:
        session.add(
            IngestedDocumentRow(
                sha256=sha,
                filename="attack.pdf",
                media_type="application/pdf",
                size_bytes=10,
                source="watch",
                workflow_id=sha,
            )
        )
    audit = FakeAuditLog()
    activities = _failure_activities(session_factory, audit)

    await activities.mark_failed(sha, "ModelOutputError: subtotal missing")
    await activities.mark_failed(sha, "a later retry's text")  # idempotent: the first is kept

    async with session_factory() as session:
        document = await session.get(IngestedDocumentRow, sha)
        assert document is not None
        assert document.failed_at is not None
        assert document.failure == "ModelOutputError: subtotal missing"
        assert await session.get(InvoiceRow, sha) is None  # no invoice is invented
    assert [e.payload["recorded_on"] for e in audit.events] == ["document", "document"]


@pytest.mark.asyncio
async def test_mark_failed_with_neither_row_raises_instead_of_dropping_the_failure(
    session_factory: SessionFactory,
) -> None:
    audit = FakeAuditLog()
    activities = _failure_activities(session_factory, audit)
    unknown = hashlib.sha256(b"never-ingested").hexdigest()

    with pytest.raises(FailureNotRecordedError):
        await activities.mark_failed(unknown, "ScanError: boom")
    assert audit.events == []  # audited once a retry records it, not before
