"""Unit tests for LLM telemetry: `RecordingLlmClient` records one `llm_calls` row per call
inside a real trace, never lets a telemetry failure break the model call, and the trace id
helper reports the active span. No database, no network."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from apx.common.types import InvoiceId, Stage
from apx.llm.interfaces import (
    ChatMessage,
    CompletionResult,
    StructuredRequest,
    TokenLogprob,
    ToolRequest,
)
from apx.observability.llm_calls import RecordingLlmClient
from apx.observability.tracing import current_trace_id, tracer


class _Inner:
    async def complete_structured(self, request: StructuredRequest) -> CompletionResult:
        return CompletionResult(
            content="{}",
            token_logprobs=(
                TokenLogprob(token="{", logprob=0.0),
                TokenLogprob(token="}", logprob=0.0),
            ),
            finish_reason="stop",
            model="extractor-a",
            latency_ms=5.0,
        )

    async def complete_with_tools(self, request: ToolRequest) -> CompletionResult:
        raise AssertionError("unused")

    async def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _Recorder:
    def __init__(self, fail: bool = False) -> None:
        self.rows: list[dict[str, Any]] = []
        self.fail = fail

    async def record(self, **row: Any) -> None:
        if self.fail:
            raise RuntimeError("database down")
        self.rows.append(row)


def _request() -> StructuredRequest:
    return StructuredRequest(
        model="extractor-a", messages=(ChatMessage(role="user", content="x"),), json_schema={}
    )


async def test_each_call_is_recorded_with_stage_tokens_and_a_real_trace_id() -> None:
    recorder = _Recorder()
    client = RecordingLlmClient(_Inner(), recorder, Stage.EXTRACT, InvoiceId("inv-1"))
    result = await client.complete_structured(_request())
    await client.embed("embedder", ["a", "b"])
    assert result.content == "{}"
    first, second = recorder.rows
    assert first["model"] == "extractor-a" and first["stage"] is Stage.EXTRACT
    assert first["tokens_out"] == 2 and first["invoice_id"] == "inv-1"
    assert first["trace_id"] != "0" * 32 and len(first["trace_id"]) == 32
    assert second["model"] == "embedder" and second["tokens_out"] is None


async def test_a_recording_failure_never_fails_the_model_call() -> None:
    client = RecordingLlmClient(_Inner(), _Recorder(fail=True), Stage.EXTRACT)
    assert (await client.complete_structured(_request())).model == "extractor-a"


def test_current_trace_id_follows_the_active_span() -> None:
    assert current_trace_id() == "0" * 32
    with tracer("test").start_as_current_span("outer") as span:
        assert current_trace_id() == format(span.get_span_context().trace_id, "032x")
