"""Model-call telemetry: `RecordingLlmClient` wraps any `LlmClient`, opens an
OpenTelemetry span per call and writes one `llm_calls` row (model, version, stage, latency,
tokens, trace id). Recording failures are logged and never fail the model call itself: the
telemetry row is a side channel, not part of the money path."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import structlog

from apx.common.types import InvoiceId, Stage
from apx.db.models import LlmCall
from apx.db.session import get_session
from apx.llm.interfaces import CompletionResult, LlmClient, StructuredRequest, ToolRequest
from apx.observability.tracing import current_trace_id, tracer

logger = structlog.get_logger(__name__)


class LlmCallRecorder(Protocol):
    async def record(
        self,
        *,
        model: str,
        stage: Stage,
        latency_ms: int,
        tokens_out: int | None,
        trace_id: str,
        invoice_id: InvoiceId | None,
    ) -> None: ...


class PgLlmCallRecorder:
    async def record(
        self,
        *,
        model: str,
        stage: Stage,
        latency_ms: int,
        tokens_out: int | None,
        trace_id: str,
        invoice_id: InvoiceId | None,
    ) -> None:
        async with get_session() as session:
            session.add(
                LlmCall(
                    invoice_id=invoice_id,
                    stage=stage,
                    model=model,
                    latency_ms=latency_ms,
                    tokens_out=tokens_out,
                    trace_id=trace_id,
                )
            )


class RecordingLlmClient:
    """`LlmClient` decorator for one pipeline stage (and optionally one invoice)."""

    def __init__(
        self,
        inner: LlmClient,
        recorder: LlmCallRecorder,
        stage: Stage,
        invoice_id: InvoiceId | None = None,
    ) -> None:
        self._inner = inner
        self._recorder = recorder
        self._stage = stage
        self._invoice_id = invoice_id

    async def _record(self, model: str, started: float, tokens_out: int | None) -> None:
        try:
            await self._recorder.record(
                model=model,
                stage=self._stage,
                latency_ms=int((time.perf_counter() - started) * 1000),
                tokens_out=tokens_out,
                trace_id=current_trace_id(),
                invoice_id=self._invoice_id,
            )
        except Exception as exc:  # telemetry must never break extraction
            logger.warning("llm_call_record_failed", error=str(exc), stage=self._stage.value)

    async def complete_structured(self, request: StructuredRequest) -> CompletionResult:
        with tracer(__name__).start_as_current_span(
            "llm.complete_structured",
            attributes={"llm.model": request.model, "apx.stage": self._stage.value},
        ):
            started = time.perf_counter()
            result = await self._inner.complete_structured(request)
            await self._record(result.model, started, len(result.token_logprobs) or None)
            return result

    async def complete_with_tools(self, request: ToolRequest) -> CompletionResult:
        with tracer(__name__).start_as_current_span(
            "llm.complete_with_tools",
            attributes={"llm.model": request.model, "apx.stage": self._stage.value},
        ):
            started = time.perf_counter()
            result = await self._inner.complete_with_tools(request)
            await self._record(result.model, started, None)
            return result

    async def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        with tracer(__name__).start_as_current_span(
            "llm.embed", attributes={"llm.model": model, "apx.stage": self._stage.value}
        ):
            started = time.perf_counter()
            vectors = await self._inner.embed(model, texts)
            await self._record(model, started, None)
            return vectors
