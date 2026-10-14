"""pgvector-backed page similarity: `PgPageVectorStore` keeps one Qwen3-VL-Embedding-2B vector per
invoice page in `page_embeddings` and answers vendor-scoped cosine top-k;
`PgPageSimilarityIndexes.for_vendor` gives the fraud engine a `PageSimilarityIndex` over one
vendor's pages that embeds each page once, for both the lookup and the later `add`.

Typed errors only: a malformed vector (wrong size, non-finite, all zeros) is a `FraudError`, an
unreachable or failing database a retryable `PageIndexUnavailableError`, and an unavailable
embedder propagates as the embedder raised it. None of them ever reads as "no near-duplicate"."""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from apx.common.errors import ApxError, FraudError
from apx.common.types import InvoiceId, VendorId
from apx.db.models import PAGE_VECTOR_DIMENSIONS
from apx.db.session import get_session
from apx.fraud.interfaces import PageEmbedder, PageVector, SimilarPage

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

_UPSERT = text(
    """
    INSERT INTO page_embeddings (invoice_id, page, vendor_id, embedding)
    VALUES (:invoice_id, :page, :vendor_id, CAST(:embedding AS vector))
    ON CONFLICT (invoice_id, page)
    DO UPDATE SET vendor_id = EXCLUDED.vendor_id, embedding = EXCLUDED.embedding
    """
)
# `<=>` is pgvector's cosine distance (1 - cosine similarity). Ties break by invoice id, then
# page, like `InMemoryPageSimilarityIndex`, so results are deterministic. The invoice under
# assessment is excluded here, not after the LIMIT: a retried fraud step has already stored its
# own pages, which would otherwise fill the top k and hide the real neighbours.
_NEAREST = text(
    """
    SELECT invoice_id, page, embedding <=> CAST(:query AS vector) AS distance
    FROM page_embeddings
    WHERE vendor_id = :vendor_id AND invoice_id <> :exclude
    ORDER BY distance, invoice_id, page
    LIMIT :k
    """
)


class PageIndexUnavailableError(ApxError):
    """The page-vector database could not be read or written (connection lost, statement
    failed). Transient from the pipeline's point of view: the fraud step fails and Temporal
    retries it, unlike the deterministic `FraudError`."""


def to_vector_literal(vector: PageVector, *, dimensions: int = PAGE_VECTOR_DIMENSIONS) -> str:
    """pgvector's text form `[x1,x2,...]` of `vector`, after checking it can be compared by
    cosine: exactly `dimensions` finite values, not all zero. Raises `FraudError` otherwise."""
    if len(vector) != dimensions:
        raise FraudError(f"page embedding has {len(vector)} dimensions, expected {dimensions}")
    if not all(math.isfinite(x) for x in vector):
        raise FraudError("page embedding contains a non-finite value")
    if not any(vector):
        raise FraudError("cannot compare a zero-length page embedding")
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def similarity_from_distance(distance: float) -> float:
    """Cosine distance to a [0, 1] similarity score: opposite pages score 0.0 (not similar),
    and float rounding just past an exact match cannot push the score above 1.0."""
    if not math.isfinite(distance):
        raise FraudError(f"pgvector returned a non-finite cosine distance: {distance}")
    return max(0.0, min(1.0, 1.0 - distance))


class PgPageVectorStore:
    """Page vectors in Postgres (`page_embeddings`), always read and written for one vendor."""

    def __init__(
        self,
        session_factory: SessionFactory = get_session,
        *,
        dimensions: int = PAGE_VECTOR_DIMENSIONS,
    ) -> None:
        self._sessions = session_factory
        self._dimensions = dimensions

    async def upsert(
        self, invoice_id: InvoiceId, vendor_id: VendorId, vectors: Sequence[PageVector]
    ) -> None:
        """Store `vectors` as pages 0..n-1 of `invoice_id` in one transaction. Idempotent: a
        retried fraud step overwrites its own rows instead of failing on the primary key."""
        if not vectors:
            return
        rows = [
            {
                "invoice_id": invoice_id,
                "page": page,
                "vendor_id": vendor_id,
                "embedding": to_vector_literal(vector, dimensions=self._dimensions),
            }
            for page, vector in enumerate(vectors)
        ]
        async with self._session() as session:
            await session.execute(_UPSERT, rows)

    async def nearest(
        self,
        vendor_id: VendorId,
        vector: PageVector,
        k: int,
        *,
        exclude: InvoiceId | None = None,
    ) -> list[SimilarPage]:
        """The `k` stored pages of `vendor_id` closest to `vector` by cosine, most similar first.
        Another vendor's pages, and the pages of `exclude`, are never candidates."""
        if k <= 0:
            raise FraudError(f"nearest() needs k >= 1, got {k}")
        query = to_vector_literal(vector, dimensions=self._dimensions)
        async with self._session() as session:
            result = await session.execute(
                _NEAREST,
                # Invoice ids are sha256 hex, so "" excludes nothing.
                {"query": query, "vendor_id": vendor_id, "k": k, "exclude": exclude or ""},
            )
            rows = result.all()
        return [
            SimilarPage(
                invoice_id=InvoiceId(row.invoice_id),
                page=int(row.page),
                similarity=similarity_from_distance(float(row.distance)),
            )
            for row in rows
        ]

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        try:
            async with self._sessions() as session:
                yield session
        except (SQLAlchemyError, OSError) as exc:
            raise PageIndexUnavailableError(f"page-embedding store failed: {exc!r}") from exc


class VendorPageSimilarityIndex:
    """`PageSimilarityIndex` over one vendor's stored pages, for one fraud assessment, never
    returning the assessed invoice's own pages. Each distinct page is embedded once and reused
    by `add`, so the GPU sees every page once."""

    def __init__(
        self,
        embedder: PageEmbedder,
        store: PgPageVectorStore,
        vendor_id: VendorId,
        *,
        exclude: InvoiceId | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._vendor_id = vendor_id
        self._exclude = exclude
        self._vectors: dict[bytes, PageVector] = {}

    async def _embed(self, page_png: bytes) -> PageVector:
        key = hashlib.sha256(page_png).digest()
        vector = self._vectors.get(key)
        if vector is None:
            vector = self._vectors[key] = await self._embedder.embed_page(page_png)
        return vector

    async def nearest(self, page_png: bytes, k: int) -> list[SimilarPage]:
        vector = await self._embed(page_png)
        return await self._store.nearest(self._vendor_id, vector, k, exclude=self._exclude)

    async def add(self, invoice_id: InvoiceId, page_images: Sequence[bytes]) -> None:
        """Store every page of `invoice_id` under this vendor. All pages are embedded before
        any is written, so an embedder failure stores nothing."""
        vectors = [await self._embed(page) for page in page_images]
        await self._store.upsert(invoice_id, self._vendor_id, vectors)


class PgPageSimilarityIndexes:
    """Hands out a fresh `VendorPageSimilarityIndex` per assessment (its embedding cache lives
    only as long as one invoice's fraud step)."""

    def __init__(self, embedder: PageEmbedder, store: PgPageVectorStore | None = None) -> None:
        self._embedder = embedder
        self._store = store or PgPageVectorStore()

    def for_vendor(
        self, vendor_id: VendorId, *, exclude: InvoiceId | None = None
    ) -> VendorPageSimilarityIndex:
        """The index over `vendor_id`'s pages, minus `exclude`'s (the invoice being assessed)."""
        return VendorPageSimilarityIndex(self._embedder, self._store, vendor_id, exclude=exclude)
