"""Embedder-backed `PageSimilarityIndex`: pages are embedded with a
`PageEmbedder` (Qwen3-VL-Embedding-2B), stored with `add`, and `nearest` returns the stored pages
closest by cosine similarity as `SimilarPage`s. The flag decision (threshold 0.9, rule id
`near_duplicate_embedding`) stays with `DuplicateInvoiceRule.evaluate_similarity`.

An unavailable provider is never a pass: `PageEmbeddingUnavailableError` / `ModelOutputError`
propagates out of `add` and `nearest`, and so out of `evaluate_similarity`, the same way an
undecodable page raises `FraudError` in the perceptual-hash check."""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

from apx.common.errors import FraudError
from apx.common.types import InvoiceId
from apx.fraud.interfaces import PageEmbedder, PageVector, SimilarPage


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between `a` and `b`, clamped to [-1, 1]. Raises `FraudError` on a
    dimension mismatch or a zero vector, since either means the vectors are not comparable."""
    if len(a) != len(b):
        raise FraudError(f"embedding dimensions differ ({len(a)} vs {len(b)})")
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise FraudError("cannot compare a zero-length page embedding")
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class InMemoryPageSimilarityIndex:
    """`PageSimilarityIndex` holding page vectors in memory, for the fraud eval and tests; the
    production index keeps the same vectors in pgvector. Like the Protocol it is not scoped by
    vendor or document kind: every stored page is a candidate, including pages of invoices with
    no resolved vendor. An invoice added with no pages contributes no candidates."""

    def __init__(self, embedder: PageEmbedder) -> None:
        self._embedder = embedder
        self._pages: list[tuple[InvoiceId, int, PageVector]] = []

    async def add(self, invoice_id: InvoiceId, page_images: Sequence[bytes]) -> list[PageVector]:
        """Embed and store every page of `invoice_id` (page numbers are 0-based positions).
        All pages are embedded before any is stored, so a provider failure stores nothing."""
        vectors = [await self._embedder.embed_page(page) for page in page_images]
        self._pages.extend((invoice_id, n, v) for n, v in enumerate(vectors))
        return vectors

    async def nearest(self, page_png: bytes, k: int) -> list[SimilarPage]:
        """The `k` stored pages most similar to `page_png`, most similar first (ties by invoice
        id, then page). Negative cosine is reported as 0.0, since `SimilarPage.similarity` is a
        [0, 1] score and an opposite page is simply not similar."""
        if k <= 0:
            raise FraudError(f"nearest() needs k >= 1, got {k}")
        query = await self._embedder.embed_page(page_png)
        scored = (
            (max(0.0, cosine_similarity(query, vector)), invoice_id, page)
            for invoice_id, page, vector in self._pages
        )
        top = heapq.nsmallest(k, scored, key=lambda row: (-row[0], row[1], row[2]))
        return [SimilarPage(invoice_id=i, page=p, similarity=s) for s, i, p in top]
