"""Unit and property tests for the learned page similarity: the cosine math, the
embedder-backed `InMemoryPageSimilarityIndex` driven through the existing
`DuplicateInvoiceRule.evaluate_similarity`, typed provider errors, and the HTTP page-embedding
client driven through an in-process transport (no server, no model)."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from datetime import date
from decimal import Decimal

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from apx.common.errors import FraudError, ModelOutputError, ModelServingError
from apx.common.invoice import Invoice, Party
from apx.common.types import InvoiceId, Language, Money, VendorId
from apx.fraud.page_similarity import InMemoryPageSimilarityIndex, cosine_similarity
from apx.fraud.rules import RULE_ID_NEAR_DUPLICATE_EMBEDDING, DuplicateInvoiceRule
from apx.llm.embedding import (
    HttpPageEmbedder,
    PageEmbeddingSettings,
    PageEmbeddingUnavailableError,
    PageVector,
)

# --- fakes and factories -----------------------------------------------------------------


class FakePageEmbedder:
    """Deterministic `PageEmbedder`: a page listed in `fixed` gets that vector, any other page a
    unit vector derived from its sha256, so the same bytes always embed the same way."""

    def __init__(self, fixed: dict[bytes, PageVector] | None = None, dim: int = 2) -> None:
        self._fixed = fixed or {}
        self._dim = dim

    async def embed_page(self, page_png: bytes) -> PageVector:
        if page_png in self._fixed:
            return self._fixed[page_png]
        digest = hashlib.sha256(page_png).digest()
        raw = [b - 127.5 for b in struct.unpack(f"{self._dim}B", digest[: self._dim])]
        norm = math.sqrt(sum(x * x for x in raw))
        return tuple(x / norm for x in raw)


class DownPageEmbedder:
    async def embed_page(self, page_png: bytes) -> PageVector:
        raise PageEmbeddingUnavailableError("page-embedding service request failed: refused")


def _invoice(seed: str) -> Invoice:
    sha = hashlib.sha256(seed.encode()).hexdigest()
    total = Money(amount=Decimal("500"), currency="AED")
    party = Party(name="Acme Supplies LLC", trn=None, address=None, country="AE")
    return Invoice(
        id=InvoiceId(sha),
        document_sha256=sha,
        language=Language.EN,
        number=f"INV-{seed}",
        issue_date=date(2026, 3, 10),
        due_date=None,
        supplier=party,
        buyer=party,
        vendor_id=VendorId("V1"),
        po_number=None,
        lines=(),
        subtotal=total,
        vat_total=Money(amount=Decimal("0"), currency="AED"),
        total=total,
        iban=None,
    )


def _unit(angle: float) -> PageVector:
    return (math.cos(angle), math.sin(angle))


async def _index_with(
    pages: dict[bytes, PageVector], stored: dict[InvoiceId, list[bytes]]
) -> InMemoryPageSimilarityIndex:
    index = InMemoryPageSimilarityIndex(FakePageEmbedder(pages))
    for invoice_id, images in stored.items():
        await index.add(invoice_id, images)
    return index


# --- cosine math ---------------------------------------------------------------------------

_vectors = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False), min_size=1, max_size=12
).filter(lambda v: math.sqrt(sum(x * x for x in v)) > 1e-3)


@given(_vectors)
def test_cosine_of_vector_with_itself_is_one(v: list[float]) -> None:
    assert cosine_similarity(v, v) == pytest.approx(1.0)


@given(st.data())
def test_cosine_is_symmetric_bounded_and_scale_invariant(data: st.DataObject) -> None:
    a = data.draw(_vectors)
    b = data.draw(
        st.lists(
            st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
            min_size=len(a),
            max_size=len(a),
        ).filter(lambda v: math.sqrt(sum(x * x for x in v)) > 1e-3)
    )
    scale = data.draw(st.floats(min_value=0.01, max_value=100.0))
    sim = cosine_similarity(a, b)
    assert -1.0 <= sim <= 1.0
    assert sim == pytest.approx(cosine_similarity(b, a))
    assert sim == pytest.approx(cosine_similarity([x * scale for x in a], b), abs=1e-9)


def test_cosine_rejects_dimension_mismatch_and_zero_vector() -> None:
    with pytest.raises(FraudError, match="dimensions differ"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
    with pytest.raises(FraudError, match="zero-length"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


# --- index behind the existing evaluate_similarity (threshold 0.9) -------------------------


async def test_evaluate_similarity_flags_a_stored_page_above_0_9() -> None:
    prior = _invoice("old")
    index = await _index_with({b"old": _unit(0.0), b"new": _unit(0.1)}, {prior.id: [b"old"]})

    flags = await DuplicateInvoiceRule().evaluate_similarity(
        _invoice("new"), page_images=(b"new",), similarity_index=index
    )

    assert len(flags) == 1
    assert flags[0].rule_id == RULE_ID_NEAR_DUPLICATE_EMBEDDING
    assert flags[0].related_invoices == (prior.id,)
    assert flags[0].score == pytest.approx(math.cos(0.1))


async def test_evaluate_similarity_ignores_a_stored_page_below_0_9() -> None:
    prior = _invoice("old")
    index = await _index_with(
        {b"old": _unit(0.0), b"new": _unit(math.pi / 4)}, {prior.id: [b"old"]}
    )

    flags = await DuplicateInvoiceRule().evaluate_similarity(
        _invoice("new"), page_images=(b"new",), similarity_index=index
    )

    assert flags == []


async def test_evaluate_similarity_ignores_the_invoices_own_stored_pages() -> None:
    current = _invoice("new")
    index = await _index_with({b"new": _unit(0.0)}, {current.id: [b"new"]})

    flags = await DuplicateInvoiceRule().evaluate_similarity(
        current, page_images=(b"new",), similarity_index=index
    )

    assert flags == []


async def test_unavailable_provider_propagates_through_evaluate_similarity() -> None:
    index = InMemoryPageSimilarityIndex(DownPageEmbedder())

    with pytest.raises(PageEmbeddingUnavailableError):
        await DuplicateInvoiceRule().evaluate_similarity(
            _invoice("new"), page_images=(b"new",), similarity_index=index
        )


async def test_add_with_unavailable_provider_raises_typed_error() -> None:
    index = InMemoryPageSimilarityIndex(DownPageEmbedder())

    with pytest.raises(PageEmbeddingUnavailableError):
        await index.add(_invoice("old").id, [b"p0", b"p1"])


# --- nearest() contract --------------------------------------------------------------------


async def test_nearest_returns_top_k_most_similar_first_with_page_numbers() -> None:
    a, b = _invoice("a").id, _invoice("b").id
    index = await _index_with(
        {
            b"a0": _unit(1.0),
            b"a1": _unit(0.05),
            b"b0": _unit(0.2),
            b"q": _unit(0.0),
        },
        {a: [b"a0", b"a1"], b: [b"b0"]},
    )

    hits = await index.nearest(b"q", k=2)

    assert [(h.invoice_id, h.page) for h in hits] == [(a, 1), (b, 0)]
    assert hits[0].similarity == pytest.approx(math.cos(0.05))
    assert hits[0].similarity >= hits[1].similarity


async def test_nearest_reports_opposite_page_as_zero_and_empty_index_as_no_hits() -> None:
    prior = _invoice("old")
    index = await _index_with({b"old": (1.0, 0.0), b"q": (-1.0, 0.0)}, {prior.id: [b"old"]})

    assert [h.similarity for h in await index.nearest(b"q", k=5)] == [0.0]
    assert await InMemoryPageSimilarityIndex(FakePageEmbedder()).nearest(b"q", k=5) == []


async def test_nearest_rejects_non_positive_k() -> None:
    with pytest.raises(FraudError, match="k >= 1"):
        await InMemoryPageSimilarityIndex(FakePageEmbedder()).nearest(b"q", k=0)


async def test_same_bytes_embed_identically_so_a_resubmitted_page_matches_itself() -> None:
    prior = _invoice("old")
    index = await _index_with({}, {prior.id: [b"scan-bytes"]})

    hits = await index.nearest(b"scan-bytes", k=1)

    assert hits[0].invoice_id == prior.id
    assert hits[0].similarity == pytest.approx(1.0)


# --- HTTP provider -------------------------------------------------------------------------


def _client(handler: httpx.MockTransport) -> HttpPageEmbedder:
    return HttpPageEmbedder(
        PageEmbeddingSettings(base_url="http://page-embed.test:8000"), transport=handler
    )


async def test_http_embedder_posts_png_and_returns_vector() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"embedding": [0.6, 0.8]})

    vector = await _client(httpx.MockTransport(handler)).embed_page(b"\x89PNG-bytes")

    assert vector == (0.6, 0.8)
    assert seen[0].url == "http://page-embed.test:8000/embed"
    assert b"\x89PNG-bytes" in seen[0].content


async def test_http_embedder_uses_configured_base_url_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAGE_EMBED_BASE_URL", "http://embed-from-env:9000")
    assert PageEmbeddingSettings().base_url == "http://embed-from-env:9000"


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_http_embedder_non_2xx_is_unavailable(status: int) -> None:
    client = _client(httpx.MockTransport(lambda _: httpx.Response(status, text="down")))
    with pytest.raises(PageEmbeddingUnavailableError, match=str(status)) as info:
        await client.embed_page(b"png")
    assert isinstance(info.value, ModelServingError)


async def test_http_embedder_connection_failure_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(PageEmbeddingUnavailableError, match="request failed"):
        await _client(httpx.MockTransport(handler)).embed_page(b"png")


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        json.dumps({"embedding": []}),
        json.dumps({"vector": [1.0]}),
        json.dumps({"embedding": ["x"]}),
        '{"embedding": [NaN]}',
    ],
)
async def test_http_embedder_malformed_response_is_model_output_error(body: str) -> None:
    client = _client(httpx.MockTransport(lambda _: httpx.Response(200, text=body)))
    with pytest.raises(ModelOutputError):
        await client.embed_page(b"png")
