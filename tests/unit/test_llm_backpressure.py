"""Backpressure against llama-swap (ADR-0013): `LlamaSwapClient` retries only 429/503 "busy"
answers, with bounded exponential backoff, then raises `ModelBusyError`; a context overflow is a
non-retryable `ModelContextError`; and `ModelSlots` never lets more calls reach a model than it
has slots, with aliases sharing their target's slots. Regression for the live e2e run where 20
concurrent extractions got 92 immediate HTTP 429s and B's shared KV cache overflowed."""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import httpx
import pytest

from apx.common.errors import ModelBusyError, ModelContextError, ModelServingError
from apx.llm.client import LlamaSwapClient, LlmSettings, busy_backoff_s, classify_http_error
from apx.llm.interfaces import ChatMessage, CompletionResult, StructuredRequest, ToolRequest
from apx.llm.slots import ModelSlots, SlotLimitedLlmClient

_OK = {"model": "extractor-a", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}


def _request(model: str = "extractor-a") -> StructuredRequest:
    return StructuredRequest(
        model=model,
        messages=(ChatMessage(role="user", content="hi"),),
        json_schema={"type": "object"},
        temperature=0.0,
        seed=0,
    )


def _client(responses: list[httpx.Response], sleeps: list[float], retries: int = 3) -> Any:
    calls = iter(responses)

    def handler(_: httpx.Request) -> httpx.Response:
        return next(calls)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    settings = LlmSettings(base_url="http://llama-swap", busy_retries=retries)
    return LlamaSwapClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
        rng=random.Random(0),
    )


def _error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message}})


async def test_busy_answers_are_retried_with_growing_backoff_then_succeed() -> None:
    sleeps: list[float] = []
    client = _client(
        [_error(429, "Too many requests"), _error(503, "loading"), httpx.Response(200, json=_OK)],
        sleeps,
    )
    result = await client.complete_structured(_request())
    assert result.content == "{}"
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] <= 1.25 and 2.0 <= sleeps[1] <= 2.5


async def test_busy_retries_are_bounded_and_end_in_a_typed_error() -> None:
    sleeps: list[float] = []
    client = _client([_error(429, "Too many requests")] * 4, sleeps, retries=3)
    with pytest.raises(ModelBusyError, match="HTTP 429"):
        await client.complete_structured(_request())
    assert len(sleeps) == 3


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (500, "Context size has been exceeded.", ModelContextError),
        (400, "the request exceeds the available context size", ModelContextError),
        (500, "failed to process mtmd chunk", ModelServingError),
    ],
)
async def test_non_busy_errors_are_typed_and_never_retried(
    status: int, message: str, expected: type[Exception]
) -> None:
    sleeps: list[float] = []
    client = _client([_error(status, message)], sleeps)
    with pytest.raises(expected) as info:
        await client.complete_structured(_request())
    assert not isinstance(info.value, ModelBusyError)
    assert sleeps == []


def test_classification_and_backoff_cap() -> None:
    assert isinstance(classify_http_error("/p", 429, ""), ModelBusyError)
    assert type(classify_http_error("/p", 502, json.dumps({"x": 1}))) is ModelServingError
    settings = LlmSettings(busy_backoff_s=1.0, busy_backoff_max_s=30.0)
    rng = random.Random(1)
    assert all(30.0 <= busy_backoff_s(n, settings, rng) <= 37.5 for n in (5, 9, 20))


class _Probe:
    """Fake inner client that records the peak number of concurrent calls per model."""

    def __init__(self) -> None:
        self.active: dict[str, int] = {}
        self.peak: dict[str, int] = {}

    async def _call(self, model: str) -> CompletionResult:
        self.active[model] = self.active.get(model, 0) + 1
        self.peak[model] = max(self.peak.get(model, 0), self.active[model])
        await asyncio.sleep(0.01)
        self.active[model] -= 1
        return CompletionResult(content="{}", finish_reason="stop", model=model, latency_ms=1.0)

    async def complete_structured(self, request: StructuredRequest) -> CompletionResult:
        return await self._call(request.model)

    async def complete_with_tools(self, request: ToolRequest) -> CompletionResult:
        return await self._call(request.model)

    async def embed(self, model: str, texts: Any) -> list[list[float]]:
        await self._call(model)
        return []


async def test_slots_cap_concurrency_per_model_and_aliases_share_slots() -> None:
    probe = _Probe()
    slots = ModelSlots(
        {"extractor-a": 2, "extractor-b": 1}, aliases={"exception-agent": "extractor-a"}
    )
    client = SlotLimitedLlmClient(probe, slots)
    requests = [_request("extractor-a")] * 5 + [_request("exception-agent")] * 5
    requests += [_request("extractor-b")] * 5
    results = await asyncio.gather(*(client.complete_structured(r) for r in requests))
    assert len(results) == 15  # every call completed: queued, never dropped
    assert probe.peak["extractor-b"] == 1
    assert probe.peak.get("extractor-a", 0) + probe.peak.get("exception-agent", 0) <= 2 * 2
    assert slots.capacity("exception-agent") == 2 and slots.waiting("extractor-a") == 0


async def test_aliased_calls_count_against_the_target_model() -> None:
    running = 0
    peak = 0

    class Shared(_Probe):
        async def _call(self, model: str) -> CompletionResult:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return CompletionResult(content="", finish_reason="stop", model=model, latency_ms=1)

    slots = ModelSlots({"extractor-a": 2}, aliases={"exception-agent": "extractor-a"})
    client = SlotLimitedLlmClient(Shared(), slots)
    mixed = [_request("extractor-a"), _request("exception-agent")] * 4
    await asyncio.gather(*(client.complete_structured(r) for r in mixed))
    assert peak == 2


def test_slot_config_is_validated() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        ModelSlots({"extractor-a": 0})
    with pytest.raises(ValueError, match="no slot count"):
        ModelSlots({"extractor-a": 1}, aliases={"agent": "missing"})
