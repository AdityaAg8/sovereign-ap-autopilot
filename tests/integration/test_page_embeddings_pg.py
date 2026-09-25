"""Integration tests for M8.1's pgvector page index against a real, migrated Postgres
(pgvector/pgvector:pg16-trixie, the Compose image): `0010_page_embeddings` up/down, vendor-scoped
cosine top-k and idempotent upserts through `PgPageVectorStore`, typed refusals of malformed
vectors, and a near-duplicate page flagged end to end through `RuleFraudEngine`."""

from __future__ import annotations

import asyncio
import math
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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from apx.common.errors import FraudError
from apx.common.invoice import Invoice, Party
from apx.common.types import InvoiceId, Language, Money, VendorId
from apx.db.models import PAGE_VECTOR_DIMENSIONS
from apx.db.models import Invoice as InvoiceRow
from apx.db.models import Vendor as VendorRow
from apx.fraud.interfaces import FraudTypology, PageVector
from apx.fraud.pg_page_similarity import PgPageSimilarityIndexes, PgPageVectorStore
from apx.fraud.rules import RULE_ID_NEAR_DUPLICATE_EMBEDDING
from apx.workflows.components import RuleFraudEngine
from apx.workflows.sql import HistorySnapshot

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-m8-1-pages"
HOST_PORT = 55443
DATABASE_URL = f"postgresql+asyncpg://apx:apx@127.0.0.1:{HOST_PORT}/apx"
ASYNCPG_DSN = f"postgresql://apx:apx@127.0.0.1:{HOST_PORT}/apx"

VENDOR = VendorId("V-100")
OTHER_VENDOR = VendorId("V-200")
FIRST = InvoiceId("a" * 64)
SECOND = InvoiceId("b" * 64)
FOREIGN = InvoiceId("c" * 64)
NEW = InvoiceId("d" * 64)

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def _unit(*head: float) -> PageVector:
    """A 2048-d vector whose first coordinates are `head`, the rest zero."""
    return tuple(head) + (0.0,) * (PAGE_VECTOR_DIMENSIONS - len(head))


def _alembic() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "db" / "migrations"))
    return cfg


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


async def _fetchval(sql: str) -> object:
    conn = await asyncpg.connect(ASYNCPG_DSN)
    try:
        return await conn.fetchval(sql)
    finally:
        await conn.close()


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    """A throwaway pgvector container on a host-only port, migrated to head; the Compose
    `postgres` is left alone (its data and revision belong to the running stack)."""
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
        try:
            command.upgrade(_alembic(), "head")
            yield DATABASE_URL
        finally:
            command.downgrade(_alembic(), "base")
            del os.environ["DATABASE_URL"]
    finally:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)


def _invoice_row(invoice_id: str, vendor_id: str) -> InvoiceRow:
    return InvoiceRow(
        id=invoice_id,
        document_sha256=invoice_id,
        vendor_id=vendor_id,
        language=Language.EN,
        number=f"INV-{invoice_id[:4]}",
        issue_date=date(2026, 9, 1),
        subtotal=Decimal("100.00"),
        vat_total=Decimal("5.00"),
        total=Decimal("105.00"),
    )


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
        await session.execute(text("TRUNCATE invoices, vendors CASCADE"))
        session.add_all(
            [
                VendorRow(id=VENDOR, name="Gulf Steel", trn=None),
                VendorRow(id=OTHER_VENDOR, name="Desert Paper", trn=None),
            ]
        )
        await session.flush()
        session.add_all(
            [
                _invoice_row(FIRST, VENDOR),
                _invoice_row(SECOND, VENDOR),
                _invoice_row(FOREIGN, OTHER_VENDOR),
                _invoice_row(NEW, VENDOR),
            ]
        )
    try:
        yield factory
    finally:
        await engine.dispose()


def test_migration_0010_up_and_down(postgres_url: str) -> None:
    cfg = _alembic()
    assert asyncio.run(_fetchval("SELECT to_regclass('public.page_embeddings')::text")) == (
        "page_embeddings"
    )
    assert asyncio.run(
        _fetchval(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'page_embeddings'::regclass AND attname = 'embedding'"
        )
    ) == (f"vector({PAGE_VECTOR_DIMENSIONS})")
    command.downgrade(cfg, "0009_invoice_kind_check_name")
    try:
        assert asyncio.run(_fetchval("SELECT to_regclass('public.page_embeddings')")) is None
        assert asyncio.run(
            _fetchval("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
        ) == (0)
    finally:
        command.upgrade(cfg, "head")
    assert asyncio.run(_fetchval("SELECT to_regclass('public.page_embeddings')::text")) == (
        "page_embeddings"
    )


async def test_nearest_is_vendor_scoped_cosine_top_k(sessions: SessionFactory) -> None:
    store = PgPageVectorStore(sessions)
    await store.upsert(FIRST, VENDOR, [_unit(1.0, 0.0), _unit(0.0, 1.0)])
    await store.upsert(SECOND, VENDOR, [_unit(1.0, 1.0)])
    # Identical to the query, but another vendor's page: never a candidate.
    await store.upsert(FOREIGN, OTHER_VENDOR, [_unit(1.0, 0.0)])

    found = await store.nearest(VENDOR, _unit(2.0, 0.0), k=2)

    assert [(p.invoice_id, p.page) for p in found] == [(FIRST, 0), (SECOND, 0)]
    assert found[0].similarity == pytest.approx(1.0)
    assert found[1].similarity == pytest.approx(1 / math.sqrt(2))
    everything = await store.nearest(VENDOR, _unit(1.0, 0.0), k=10)
    assert {p.invoice_id for p in everything} == {FIRST, SECOND}
    assert [p.similarity for p in everything][-1] == pytest.approx(0.0)  # orthogonal page


async def test_upsert_is_idempotent_for_a_retried_step(sessions: SessionFactory) -> None:
    store = PgPageVectorStore(sessions)
    await store.upsert(FIRST, VENDOR, [_unit(1.0, 0.0)])
    await store.upsert(FIRST, VENDOR, [_unit(0.0, 1.0)])

    found = await store.nearest(VENDOR, _unit(0.0, 1.0), k=5)

    assert [(p.invoice_id, p.page) for p in found] == [(FIRST, 0)]
    assert found[0].similarity == pytest.approx(1.0)


async def test_a_retried_step_whose_own_pages_fill_k_still_finds_the_duplicate(
    sessions: SessionFactory,
) -> None:
    """NEW's first attempt stored five pages identical to the query, then the step failed. On
    the retry those rows are the closest candidates and fill the rule's k=5; excluding NEW in
    the query (not after the LIMIT) still surfaces FIRST, the real near-duplicate."""
    store = PgPageVectorStore(sessions)
    await store.upsert(FIRST, VENDOR, [_unit(0.98, 0.1)])
    await store.upsert(NEW, VENDOR, [_unit(1.0, 0.0)] * 5)
    unscoped = await store.nearest(VENDOR, _unit(1.0, 0.0), k=5)
    assert {p.invoice_id for p in unscoped} == {NEW}  # all a post-LIMIT filter would see

    embedder = _Embedder({b"p": _unit(1.0, 0.0)})
    engine = RuleFraudEngine((), _History(), page_index=PgPageSimilarityIndexes(embedder, store))
    report = await engine.assess(_invoice(NEW), [b"p"])

    assert [(f.rule_id, f.related_invoices) for f in report.flags] == [
        (RULE_ID_NEAR_DUPLICATE_EMBEDDING, (FIRST,))
    ]


async def test_malformed_vectors_are_refused_before_the_database(
    sessions: SessionFactory,
) -> None:
    store = PgPageVectorStore(sessions)
    with pytest.raises(FraudError, match="dimensions"):
        await store.upsert(FIRST, VENDOR, [(1.0, 0.0)])
    with pytest.raises(FraudError, match="zero-length"):
        await store.nearest(VENDOR, _unit(), k=1)


class _Embedder:
    def __init__(self, vectors: dict[bytes, PageVector]) -> None:
        self._vectors = vectors
        self.calls: list[bytes] = []

    async def embed_page(self, page_png: bytes) -> PageVector:
        self.calls.append(page_png)
        return self._vectors[page_png]


class _History:
    async def snapshot(self, invoice: Invoice) -> HistorySnapshot:
        return HistorySnapshot(subject=invoice, invoices=(), changes=(), iban_vendors=(), po=None)


def _invoice(invoice_id: InvoiceId) -> Invoice:
    party = Party(name="Gulf Steel", trn=None, address=None)
    return Invoice(
        id=invoice_id,
        document_sha256=invoice_id,
        language=Language.EN,
        number="INV-9",
        issue_date=date(2026, 9, 2),
        due_date=None,
        supplier=party,
        buyer=party,
        vendor_id=VENDOR,
        po_number=None,
        lines=(),
        subtotal=Money(amount=Decimal("100"), currency="AED"),
        vat_total=Money(amount=Decimal("5"), currency="AED"),
        total=Money(amount=Decimal("105"), currency="AED"),
        iban=None,
    )


async def test_engine_flags_a_rescanned_page_and_stores_the_new_pages(
    sessions: SessionFactory,
) -> None:
    store = PgPageVectorStore(sessions)
    await store.upsert(FIRST, VENDOR, [_unit(1.0, 0.0)])
    embedder = _Embedder({b"rescan": _unit(0.99, 0.05)})
    engine = RuleFraudEngine((), _History(), page_index=PgPageSimilarityIndexes(embedder, store))

    report = await engine.assess(_invoice(NEW), [b"rescan"])

    [flag] = report.flags
    assert flag.rule_id == RULE_ID_NEAR_DUPLICATE_EMBEDDING
    assert flag.typology is FraudTypology.NEAR_DUPLICATE
    assert flag.related_invoices == (FIRST,)
    assert "nearly identical" in flag.explanation
    assert embedder.calls == [b"rescan"]  # embedded once for lookup and storage
    stored = await store.nearest(VENDOR, _unit(0.99, 0.05), k=1)
    assert [(p.invoice_id, p.page) for p in stored] == [(NEW, 0)]
