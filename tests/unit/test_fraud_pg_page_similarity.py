"""Unit tests for the pgvector page index and its wiring, with fakes only (no database, no
model): vector validation and distance mapping, `PgPageVectorStore`'s SQL parameters and typed
error wrapping, the per-assessment embedding cache, `RuleFraudEngine` raising the learned
near-duplicate flag and storing pages after the lookup, and `HttpPageEmbedder.unload`."""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import httpx
import pytest
from sqlalchemy.exc import OperationalError

from apx.common.errors import FraudError, ModelServingError
from apx.common.invoice import Invoice, Party
from apx.common.types import InvoiceId, Language, Money, VendorId
from apx.fraud.interfaces import FraudTypology, PageVector, SimilarPage
from apx.fraud.pg_page_similarity import (
    PageIndexUnavailableError,
    PgPageSimilarityIndexes,
    PgPageVectorStore,
    VendorPageSimilarityIndex,
    similarity_from_distance,
    to_vector_literal,
)
from apx.fraud.rules import RULE_ID_NEAR_DUPLICATE_EMBEDDING
from apx.llm.embedding import (
    HttpPageEmbedder,
    PageEmbeddingSettings,
    PageEmbeddingUnavailableError,
)
from apx.workflows.components import RuleFraudEngine
from apx.workflows.sql import HistorySnapshot

VENDOR = VendorId("V1")
DIM = 3

# --- fakes ---------------------------------------------------------------------------------


@dataclass
class _Row:
    invoice_id: str
    page: int
    distance: float


class _Result:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    def all(self) -> list[_Row]:
        return self._rows


class FakeSession:
    def __init__(self, rows: list[_Row] | None = None, fail: bool = False) -> None:
        self.rows = rows or []
        self.fail = fail
        self.calls: list[tuple[str, Any]] = []

    async def execute(self, statement: Any, params: Any) -> _Result:
        if self.fail:
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))
        self.calls.append((str(statement), params))
        return _Result(self.rows)


def _factory(session: FakeSession) -> Any:
    @asynccontextmanager
    async def factory() -> AsyncIterator[FakeSession]:
        yield session

    return factory


class FakeEmbedder:
    def __init__(self, vectors: dict[bytes, PageVector]) -> None:
        self.vectors = vectors
        self.calls: list[bytes] = []

    async def embed_page(self, page_png: bytes) -> PageVector:
        self.calls.append(page_png)
        return self.vectors[page_png]


class DownEmbedder:
    async def embed_page(self, page_png: bytes) -> PageVector:
        raise PageEmbeddingUnavailableError("page-embedding service request failed: refused")


class RecordingIndex:
    """A `VendorPageIndex` fake that answers from a fixed neighbour list and logs calls."""

    def __init__(self, neighbours: list[SimilarPage], log: list[str]) -> None:
        self._neighbours = neighbours
        self._log = log

    async def nearest(self, page_png: bytes, k: int) -> list[SimilarPage]:
        self._log.append("nearest")
        return self._neighbours[:k]

    async def add(self, invoice_id: InvoiceId, page_images: Sequence[bytes]) -> None:
        self._log.append(f"add:{invoice_id}:{len(page_images)}")


class RecordingIndexes:
    def __init__(self, neighbours: list[SimilarPage]) -> None:
        self.neighbours = neighbours
        self.log: list[str] = []
        self.vendors: list[VendorId] = []
        self.excluded: list[InvoiceId | None] = []

    def for_vendor(
        self, vendor_id: VendorId, *, exclude: InvoiceId | None = None
    ) -> RecordingIndex:
        self.vendors.append(vendor_id)
        self.excluded.append(exclude)
        return RecordingIndex(self.neighbours, self.log)


class EmptyHistory:
    async def snapshot(self, invoice: Invoice) -> HistorySnapshot:
        return HistorySnapshot(invoice, (), (), (), None)


def _invoice(seed: str = "new", vendor: VendorId | None = VENDOR) -> Invoice:
    sha = hashlib.sha256(seed.encode()).hexdigest()
    party = Party(name="Acme Supplies LLC", trn=None, address=None)
    total = Money(amount=Decimal("500"), currency="AED")
    return Invoice(
        id=InvoiceId(sha),
        document_sha256=sha,
        language=Language.EN,
        number=f"INV-{seed}",
        issue_date=date(2026, 3, 10),
        due_date=None,
        supplier=party,
        buyer=party,
        vendor_id=vendor,
        po_number=None,
        lines=(),
        subtotal=total,
        vat_total=Money(amount=Decimal("0"), currency="AED"),
        total=total,
        iban=None,
    )


ORIGINAL = InvoiceId("a" * 64)

# --- vector validation and distance mapping ----------------------------------------------


def test_vector_literal_is_pgvector_text_form() -> None:
    assert to_vector_literal((1.0, -0.5, 0.25), dimensions=DIM) == "[1.0,-0.5,0.25]"


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ((1.0, 0.0), "dimensions"),
        ((1.0, math.nan, 0.0), "non-finite"),
        ((1.0, math.inf, 0.0), "non-finite"),
        ((0.0, 0.0, 0.0), "zero-length"),
    ],
)
def test_malformed_vectors_are_a_fraud_error(vector: PageVector, message: str) -> None:
    with pytest.raises(FraudError, match=message):
        to_vector_literal(vector, dimensions=DIM)


@pytest.mark.parametrize(
    ("distance", "similarity"),
    [(0.0, 1.0), (-1e-9, 1.0), (0.05, 0.95), (1.0, 0.0), (2.0, 0.0)],
)
def test_distance_maps_to_a_unit_interval_similarity(distance: float, similarity: float) -> None:
    assert similarity_from_distance(distance) == pytest.approx(similarity)


def test_nan_distance_is_refused() -> None:
    with pytest.raises(FraudError, match="non-finite"):
        similarity_from_distance(math.nan)


# --- PgPageVectorStore ----------------------------------------------------------------------


async def test_nearest_sends_a_vendor_scoped_query_and_maps_rows() -> None:
    session = FakeSession([_Row(ORIGINAL, 1, 0.02), _Row("b" * 64, 0, 0.4)])
    store = PgPageVectorStore(_factory(session), dimensions=DIM)

    found = await store.nearest(VENDOR, (1.0, 0.0, 0.0), k=2)

    assert [(p.invoice_id, p.page) for p in found] == [(ORIGINAL, 1), ("b" * 64, 0)]
    assert [p.similarity for p in found] == pytest.approx([0.98, 0.6])
    [(sql, params)] = session.calls
    assert "WHERE vendor_id = :vendor_id" in sql and "<=>" in sql
    assert "invoice_id <> :exclude" in sql
    assert params == {"query": "[1.0,0.0,0.0]", "vendor_id": VENDOR, "k": 2, "exclude": ""}


async def test_nearest_excludes_the_assessed_invoice_in_the_query() -> None:
    session = FakeSession()
    index = PgPageSimilarityIndexes(
        FakeEmbedder({b"p0": (1.0, 0.0, 0.0)}), PgPageVectorStore(_factory(session), dimensions=DIM)
    ).for_vendor(VENDOR, exclude=ORIGINAL)
    await index.nearest(b"p0", 5)
    [(_, params)] = session.calls
    assert params["exclude"] == ORIGINAL


async def test_upsert_writes_every_page_in_one_statement() -> None:
    session = FakeSession()
    store = PgPageVectorStore(_factory(session), dimensions=DIM)

    await store.upsert(ORIGINAL, VENDOR, [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])

    [(sql, params)] = session.calls
    assert "ON CONFLICT (invoice_id, page)" in sql
    assert [(p["page"], p["vendor_id"], p["embedding"]) for p in params] == [
        (0, VENDOR, "[1.0,0.0,0.0]"),
        (1, VENDOR, "[0.0,1.0,0.0]"),
    ]


async def test_upsert_of_no_pages_touches_nothing() -> None:
    session = FakeSession(fail=True)
    await PgPageVectorStore(_factory(session), dimensions=DIM).upsert(ORIGINAL, VENDOR, [])


async def test_nearest_needs_a_positive_k() -> None:
    store = PgPageVectorStore(_factory(FakeSession()), dimensions=DIM)
    with pytest.raises(FraudError, match="k >= 1"):
        await store.nearest(VENDOR, (1.0, 0.0, 0.0), k=0)


async def test_database_failure_is_a_retryable_typed_error_not_a_pass() -> None:
    store = PgPageVectorStore(_factory(FakeSession(fail=True)), dimensions=DIM)
    with pytest.raises(PageIndexUnavailableError) as caught:
        await store.nearest(VENDOR, (1.0, 0.0, 0.0), k=1)
    # Not a FraudError: the fraud activity lists FraudError as non-retryable.
    assert not isinstance(caught.value, FraudError)
    with pytest.raises(PageIndexUnavailableError):
        await store.upsert(ORIGINAL, VENDOR, [(1.0, 0.0, 0.0)])


# --- VendorPageSimilarityIndex --------------------------------------------------------------


async def test_scoped_index_embeds_each_page_once_for_lookup_and_add() -> None:
    session = FakeSession([_Row(ORIGINAL, 0, 0.0)])
    embedder = FakeEmbedder({b"p0": (1.0, 0.0, 0.0), b"p1": (0.0, 1.0, 0.0)})
    index = PgPageSimilarityIndexes(
        embedder, PgPageVectorStore(_factory(session), dimensions=DIM)
    ).for_vendor(VENDOR)
    assert isinstance(index, VendorPageSimilarityIndex)

    await index.nearest(b"p0", 3)
    await index.nearest(b"p1", 3)
    await index.add(_invoice().id, [b"p0", b"p1"])

    assert embedder.calls == [b"p0", b"p1"]
    assert [params["vendor_id"] for _, params in session.calls[:2]] == [VENDOR, VENDOR]


async def test_scoped_index_stores_nothing_when_the_embedder_fails() -> None:
    session = FakeSession()
    index = PgPageSimilarityIndexes(
        DownEmbedder(), PgPageVectorStore(_factory(session), dimensions=DIM)
    ).for_vendor(VENDOR)
    with pytest.raises(ModelServingError):
        await index.add(_invoice().id, [b"p0"])
    assert session.calls == []


# --- RuleFraudEngine wiring -----------------------------------------------------------------


async def test_engine_raises_the_learned_near_duplicate_flag_then_stores_pages() -> None:
    subject = _invoice()
    indexes = RecordingIndexes([SimilarPage(invoice_id=ORIGINAL, page=0, similarity=0.97)])
    engine = RuleFraudEngine((), EmptyHistory(), page_index=indexes)

    report = await engine.assess(subject, [b"page"])

    [flag] = report.flags
    assert flag.rule_id == RULE_ID_NEAR_DUPLICATE_EMBEDDING
    assert flag.typology is FraudTypology.NEAR_DUPLICATE
    assert flag.related_invoices == (ORIGINAL,)
    assert flag.explanation and "nearly identical" in flag.explanation
    assert report.max_score == pytest.approx(0.97)
    assert indexes.vendors == [VENDOR]
    assert indexes.excluded == [subject.id]
    assert indexes.log == ["nearest", f"add:{subject.id}:1"]


async def test_engine_ignores_neighbours_below_threshold_and_itself() -> None:
    subject = _invoice()
    indexes = RecordingIndexes(
        [
            SimilarPage(invoice_id=subject.id, page=0, similarity=1.0),
            SimilarPage(invoice_id=ORIGINAL, page=0, similarity=0.85),
        ]
    )
    report = await RuleFraudEngine((), EmptyHistory(), page_index=indexes).assess(
        subject, [b"page"]
    )
    assert report.flags == ()


async def test_engine_skips_learned_pages_without_a_resolved_vendor() -> None:
    indexes = RecordingIndexes([SimilarPage(invoice_id=ORIGINAL, page=0, similarity=1.0)])
    report = await RuleFraudEngine((), EmptyHistory(), page_index=indexes).assess(
        _invoice(vendor=None), [b"page"]
    )
    assert report.flags == ()
    assert indexes.vendors == []


async def test_engine_fails_when_the_embedder_is_down_never_passes() -> None:
    store = PgPageVectorStore(_factory(FakeSession()), dimensions=DIM)
    engine = RuleFraudEngine(
        (), EmptyHistory(), page_index=PgPageSimilarityIndexes(DownEmbedder(), store)
    )
    with pytest.raises(PageEmbeddingUnavailableError):
        await engine.assess(_invoice(), [b"page"])


# --- HttpPageEmbedder.unload ----------------------------------------------------------------


def _embedder(handler: Any) -> HttpPageEmbedder:
    return HttpPageEmbedder(
        PageEmbeddingSettings(base_url="http://page-embed.test"),
        transport=httpx.MockTransport(handler),
    )


async def test_unload_posts_to_the_service() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(200, json={"loaded": False})

    await _embedder(handler).unload()
    assert seen == ["POST /unload"]


async def test_unload_of_a_service_that_is_not_running_counts_as_released() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    await _embedder(handler).unload()


async def test_unload_failure_is_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(PageEmbeddingUnavailableError, match="/unload"):
        await _embedder(handler).unload()
