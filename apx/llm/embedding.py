"""Page-embedding provider for fraud page similarity (PRD §3 apx/fraud, §4 Qwen3-VL-Embedding-2B,
§10 M8.1): `HttpPageEmbedder`, structurally an `apx.fraud.interfaces.PageEmbedder`, posts one
rendered page PNG to the local embedding service and returns its vector. Typed errors only: an
unreachable, busy or failing service raises `PageEmbeddingUnavailableError`, a malformed answer
`ModelOutputError`, never an empty or made-up vector."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Any

import httpx
from pydantic import Field
from pydantic import ValidationError as PydanticValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from apx.common.errors import ModelOutputError, ModelServingError
from apx.common.types import Frozen

PageVector = tuple[float, ...]
"""One page's embedding. Not normalized by contract; cosine similarity normalizes."""

Gate = Callable[[], AbstractAsyncContextManager[None]]


class PageEmbeddingUnavailableError(ModelServingError):
    """The page-embedding service could not produce a vector (connection refused, timeout,
    non-2xx). Transient from the pipeline's point of view: the fraud step fails and is retried
    or routed to review; it never counts as "no near-duplicate found"."""


class PageEmbeddingSettings(BaseSettings):
    """`PAGE_EMBED_*` settings. The default URL is the in-network service name, following
    `apx.parse.config.ParseClientConfig`; override it per deployment, never with a public host."""

    model_config = SettingsConfigDict(env_prefix="PAGE_EMBED_", extra="forbid")

    base_url: str = "http://page-embed:8000"
    connect_timeout_s: float = Field(default=5.0, gt=0.0)
    request_timeout_s: float = Field(default=120.0, gt=0.0)
    """Covers a cold load of the ~4.5 GB checkpoint (ADR-0013 budget) on the first page."""


class _WireEmbedResponse(Frozen):
    """POST /embed response: one embedding for the one uploaded image."""

    embedding: tuple[float, ...] = Field(min_length=1)


class HttpPageEmbedder:
    """Async `PageEmbedder` over the local page-embedding service (internal network only). An
    optional `gate` wraps every call (the worker's GPU stage hold, ADR-0013); `transport` lets
    tests drive the real HTTP handling without a server."""

    def __init__(
        self,
        settings: PageEmbeddingSettings | None = None,
        *,
        gate: Gate | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or PageEmbeddingSettings()
        self._gate: Gate = gate or nullcontext
        self._transport = transport

    async def embed_page(self, page_png: bytes) -> PageVector:
        async with self._gate():
            payload = await self._post(page_png)
        try:
            vector = _WireEmbedResponse.model_validate(payload).embedding
        except PydanticValidationError as exc:
            raise ModelOutputError(f"malformed page-embedding response: {exc}") from exc
        if not all(math.isfinite(x) for x in vector):
            raise ModelOutputError("page-embedding response contains a non-finite value")
        return vector

    async def _post(self, page_png: bytes) -> Any:
        timeout = httpx.Timeout(
            self._settings.request_timeout_s, connect=self._settings.connect_timeout_s
        )
        try:
            async with httpx.AsyncClient(
                base_url=self._settings.base_url, timeout=timeout, transport=self._transport
            ) as http:
                response = await http.post(
                    "/embed", files={"image": ("page.png", page_png, "image/png")}
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise PageEmbeddingUnavailableError(
                f"page-embedding service returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise PageEmbeddingUnavailableError(
                f"page-embedding service request failed: {exc!r}"
            ) from exc
        try:
            return response.json()
        except ValueError as exc:
            raise ModelOutputError(f"page-embedding service returned non-JSON: {exc}") from exc
