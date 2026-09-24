"""GPU stage scheduling for the worker (ADR-0013): on the 12 GB card with the desktop's share, only
one of the scan parser (PaddleOCR-VL, ~4.3 GB cap), Extractor A (~6.4 GB) and Extractor B (~4.1 GB)
fits at a time (A and B together measured ~10.5 GB). Every GPU call holds its `GpuStage` and
`GpuStageScheduler` runs one stage at a time, so work is batched per model instead of thrashing:
calls of the active stage share it; a switch waits for the stage to drain, then frees its VRAM
through a release hook (llama-swap `/unload`, parse service `/unload`) before the next starts."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from enum import StrEnum

import structlog

from apx.llm.interfaces import CompletionResult, LlmClient, StructuredRequest, ToolRequest

logger = structlog.get_logger(__name__)

ReleaseHook = Callable[[], Awaitable[None]]


class GpuStage(StrEnum):
    PARSE = "parse"
    """PaddleOCR-VL text spotting of scanned pages (parse service)."""
    EXTRACTOR_A = "extractor-a"
    """Extractor A in llama-swap, and every model served as its alias (the exception agent)."""
    EXTRACTOR_B = "extractor-b"
    """Extractor B in llama-swap."""


class GpuStageScheduler:
    """One GPU stage at a time, shared by all concurrent callers of that stage.

    Fairness: the active stage keeps admitting new callers until another stage has waited for
    `max_hold_s` since the active stage started; then it stops admitting and drains. When the
    active stage is idle, the card goes to the waiting stage whose oldest caller arrived first
    (FIFO across stages), so a stage with few callers (scanned pages) cannot be starved by two
    busy ones (A and B) handing the card back and forth. The first entry of any stage releases
    every other stage, since a fresh worker cannot know what an earlier process left resident.
    A release hook that raises propagates to the caller and leaves no stage active, so the next
    caller retries it."""

    def __init__(
        self,
        release: Mapping[GpuStage, ReleaseHook],
        *,
        max_hold_s: float = 300.0,
        idle_grace_s: float = 3.0,
        poll_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._release = dict(release)
        self._max_hold_s = max_hold_s
        self._idle_grace_s = idle_grace_s
        self._idle_since = 0.0
        self._poll_s = poll_s
        self._clock = clock
        self._cond = asyncio.Condition()
        self._active: GpuStage | None = None
        self._started_at = 0.0
        self._holders = 0
        self._waiting: dict[GpuStage, int] = dict.fromkeys(GpuStage, 0)
        self._arrivals: dict[GpuStage, list[int]] = {stage: [] for stage in GpuStage}
        self._seq = 0
        self.switches: list[GpuStage] = []
        """Every stage entered after a switch, in order (for logs and tests)."""

    @property
    def active(self) -> GpuStage | None:
        return self._active

    def _others_waiting(self, stage: GpuStage) -> bool:
        return any(n > 0 for s, n in self._waiting.items() if s is not stage)

    def _hold_expired(self) -> bool:
        return self._clock() - self._started_at >= self._max_hold_s

    def _first_waiting(self, exclude: GpuStage | None) -> GpuStage | None:
        """The stage (other than `exclude`) whose oldest waiting caller arrived first."""
        queued = [(min(seqs), s) for s, seqs in self._arrivals.items() if seqs and s is not exclude]
        return min(queued)[1] if queued else None

    def _can_enter(self, stage: GpuStage) -> bool:
        if self._active is None:
            return self._holders == 0 and self._first_waiting(None) is stage
        if self._active is stage:
            return not (self._others_waiting(stage) and self._hold_expired())
        if self._holders > 0:
            return False
        # The active stage is idle: take over unless its own waiters still have their turn or
        # it went idle only just now (the gap between one scanned page or completion and the
        # next of the same batch: without the grace every page would cost a model swap), and
        # only if this stage has waited longest among the others.
        idle_long_enough = self._clock() - self._idle_since >= self._idle_grace_s
        its_turn_over = (
            self._waiting[self._active] == 0 and idle_long_enough
        ) or self._hold_expired()
        return its_turn_over and self._first_waiting(self._active) is stage

    async def _switch_to(self, stage: GpuStage) -> None:
        previous = self._active
        self._active = None
        for other in GpuStage:
            if other is stage or (previous is not None and other is not previous):
                continue
            hook = self._release.get(other)
            if hook is not None:
                await hook()
        self._active = stage
        self._started_at = self._clock()
        self.switches.append(stage)
        logger.info("gpu_stage_switch", from_stage=previous, to_stage=stage.value)

    @asynccontextmanager
    async def hold(self, stage: GpuStage) -> AsyncIterator[None]:
        async with self._cond:
            self._waiting[stage] += 1
            self._seq += 1
            ticket = self._seq
            self._arrivals[stage].append(ticket)
            try:
                while not self._can_enter(stage):
                    # Timed wait: a hold can expire with nobody notifying, so re-check anyway.
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._cond.wait(), timeout=self._poll_s)
            finally:
                self._waiting[stage] -= 1
                self._arrivals[stage].remove(ticket)
            if self._active is not stage:
                await self._switch_to(stage)
            self._holders += 1
        try:
            yield
        finally:
            async with self._cond:
                self._holders -= 1
                if self._holders == 0:
                    self._idle_since = self._clock()
                self._cond.notify_all()


class SerialStageGate:
    """A gate for a single-model service that serves one request at a time (the parse service
    runs one page on the GPU at once): callers queue here first, then hold `stage`, so a page's
    HTTP timeout covers its own inference, not other pages queued inside the service."""

    def __init__(self, scheduler: GpuStageScheduler, stage: GpuStage) -> None:
        self._scheduler = scheduler
        self._stage = stage
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[None]:
        async with self._lock, self._scheduler.hold(self._stage):
            yield


class StagedLlmClient:
    """`LlmClient` decorator: each call holds its model's stage (`stage_for(model)`) while it
    runs, so llama-swap only ever has that stage's model loaded."""

    def __init__(
        self,
        inner: LlmClient,
        scheduler: GpuStageScheduler,
        stage_for: Callable[[str], GpuStage],
    ) -> None:
        self._inner = inner
        self._scheduler = scheduler
        self._stage_for = stage_for

    async def complete_structured(self, request: StructuredRequest) -> CompletionResult:
        async with self._scheduler.hold(self._stage_for(request.model)):
            return await self._inner.complete_structured(request)

    async def complete_with_tools(self, request: ToolRequest) -> CompletionResult:
        async with self._scheduler.hold(self._stage_for(request.model)):
            return await self._inner.complete_with_tools(request)

    async def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        async with self._scheduler.hold(self._stage_for(model)):
            return await self._inner.embed(model, texts)
