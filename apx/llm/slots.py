"""Worker-side backpressure for llama-swap (ADR-0013): `ModelSlots` holds one semaphore per served
model, sized to that llama-server's `--parallel` slots, and `SlotLimitedLlmClient` makes every
call wait for a free slot before it is sent. Callers queue in-process (logged) instead of being
rejected with HTTP 429 or overflowing a shared KV cache; the call itself is never dropped."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager

import structlog

from apx.llm.interfaces import CompletionResult, LlmClient, StructuredRequest, ToolRequest

logger = structlog.get_logger(__name__)


class ModelSlots:
    """Per-model concurrency limits. `slots` maps a served model id to its slot count;
    `aliases` maps request names llama-swap resolves to that same process (e.g. the exception
    agent's "exception-agent" -> "extractor-a"), so they share its slots. A model with no
    entry gets `default` slots."""

    def __init__(
        self, slots: Mapping[str, int], aliases: Mapping[str, str] | None = None, default: int = 1
    ) -> None:
        if default < 1 or any(n < 1 for n in slots.values()):
            raise ValueError(f"slot counts must be >= 1: {dict(slots)}, default={default}")
        unknown = {target for target in (aliases or {}).values() if target not in slots}
        if unknown:
            raise ValueError(f"aliases point at models with no slot count: {sorted(unknown)}")
        self._capacity = dict(slots)
        self._aliases = dict(aliases or {})
        self._default = default
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._waiting: dict[str, int] = {}

    def resolve(self, model: str) -> str:
        return self._aliases.get(model, model)

    def capacity(self, model: str) -> int:
        return self._capacity.get(self.resolve(model), self._default)

    def waiting(self, model: str) -> int:
        """Callers currently queued for `model`'s slots (for logs and tests)."""
        return self._waiting.get(self.resolve(model), 0)

    @asynccontextmanager
    async def hold(self, model: str) -> AsyncIterator[None]:
        name = self.resolve(model)
        semaphore = self._semaphores.get(name)
        if semaphore is None:
            semaphore = self._semaphores[name] = asyncio.Semaphore(self.capacity(name))
        if semaphore.locked():
            self._waiting[name] = self._waiting.get(name, 0) + 1
            logger.info("llm_slot_wait", model=name, waiting=self._waiting[name])
            started = time.perf_counter()
            try:
                await semaphore.acquire()
            finally:
                self._waiting[name] -= 1
            logger.info(
                "llm_slot_acquired", model=name, waited_s=round(time.perf_counter() - started, 1)
            )
        else:
            await semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()


class SlotLimitedLlmClient:
    """`LlmClient` decorator: each call holds one of its model's slots for its whole duration.
    Share one `ModelSlots` between every client in a process that reaches the same llama-swap."""

    def __init__(self, inner: LlmClient, slots: ModelSlots) -> None:
        self._inner = inner
        self._slots = slots

    async def complete_structured(self, request: StructuredRequest) -> CompletionResult:
        async with self._slots.hold(request.model):
            return await self._inner.complete_structured(request)

    async def complete_with_tools(self, request: ToolRequest) -> CompletionResult:
        async with self._slots.hold(request.model):
            return await self._inner.complete_with_tools(request)

    async def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        async with self._slots.hold(model):
            return await self._inner.embed(model, texts)
