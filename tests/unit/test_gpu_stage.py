"""`GpuStageScheduler` (ADR-0013): the scan parser, Extractor A and Extractor B never hold the GPU
at the same time and are batched rather than interleaved per call; a switch releases the previous
stage's VRAM first (the first entry releases every other stage); the active stage yields to a
waiting stage once its hold expires; and a failing release propagates to the caller and is
retried by the next one. Also checks the parse client's gate and `StagedLlmClient` stages."""

from __future__ import annotations

import asyncio
import hashlib

import httpx
import pytest

from apx.common.errors import ParseError
from apx.common.types import Sha256Hex
from apx.llm.interfaces import ChatMessage, CompletionResult, StructuredRequest, ToolRequest
from apx.parse.client import ParseServiceClient
from apx.parse.config import ParseClientConfig
from apx.workflows.gpu_stage import (
    GpuStage,
    GpuStageScheduler,
    SerialStageGate,
    StagedLlmClient,
)


class _Card:
    """Fake GPU: tracks which stages run and whether two different ones ever ran at once."""

    def __init__(self) -> None:
        self.running: dict[GpuStage, int] = dict.fromkeys(GpuStage, 0)
        self.overlap = False
        self.released: list[GpuStage] = []
        self.fail_next_release = False

    def hook(self, stage: GpuStage):  # type: ignore[no-untyped-def]
        async def release() -> None:
            if self.fail_next_release:
                self.fail_next_release = False
                raise ParseError("unload failed")
            self.released.append(stage)

        return release

    async def work(self, scheduler: GpuStageScheduler, stage: GpuStage, seconds: float) -> None:
        async with scheduler.hold(stage):
            self.running[stage] += 1
            if sum(n > 0 for n in self.running.values()) > 1:
                self.overlap = True
            await asyncio.sleep(seconds)
            self.running[stage] -= 1


def _scheduler(card: _Card, max_hold_s: float = 60.0) -> GpuStageScheduler:
    return GpuStageScheduler(
        {stage: card.hook(stage) for stage in GpuStage},
        max_hold_s=max_hold_s,
        idle_grace_s=0.0,
        poll_s=0.01,
    )


async def test_stages_never_overlap_and_switches_release_the_previous_stage() -> None:
    card = _Card()
    scheduler = _scheduler(card)
    jobs = [card.work(scheduler, GpuStage.EXTRACTOR_A, 0.02) for _ in range(6)]
    jobs += [card.work(scheduler, GpuStage.PARSE, 0.02) for _ in range(3)]
    await asyncio.gather(*jobs)
    assert not card.overlap
    # First entry (A) released every other stage; the later switch to PARSE released A.
    assert card.released[:3] == [GpuStage.PARSE, GpuStage.EXTRACTOR_B, GpuStage.EXTRACTOR_A]
    assert len(scheduler.switches) == len(card.released) - 1


async def test_active_stage_yields_after_its_hold_expires() -> None:
    now = [0.0]
    card = _Card()
    scheduler = GpuStageScheduler(
        {stage: card.hook(stage) for stage in GpuStage},
        max_hold_s=10.0,
        idle_grace_s=0.0,
        poll_s=0.01,
        clock=lambda: now[0],
    )
    order: list[str] = []

    async def llm_stream() -> None:
        for _ in range(20):
            await card.work(scheduler, GpuStage.EXTRACTOR_A, 0.001)
            order.append("llm")
            now[0] += 1.0  # each call "takes" one second of the hold

    async def one_parse() -> None:
        while not order:  # join once the LLM stage is running
            await asyncio.sleep(0.001)
        await card.work(scheduler, GpuStage.PARSE, 0.001)
        order.append("parse")

    await asyncio.gather(llm_stream(), llm_stream(), one_parse())
    parse_at = order.index("parse")
    assert 0 < parse_at < 20  # admitted once the 10 s hold expired, not after all 40 LLM calls
    assert not card.overlap
    assert scheduler.switches == [GpuStage.EXTRACTOR_A, GpuStage.PARSE, GpuStage.EXTRACTOR_A]


async def test_a_and_b_are_batched_not_interleaved_per_call() -> None:
    """Six extractions each needing A then B: stages alternate per batch, not per document."""
    card = _Card()
    scheduler = _scheduler(card)

    async def extraction() -> None:
        await asyncio.gather(
            card.work(scheduler, GpuStage.EXTRACTOR_A, 0.01),
            card.work(scheduler, GpuStage.EXTRACTOR_B, 0.01),
        )

    await asyncio.gather(*(extraction() for _ in range(6)))
    assert not card.overlap
    assert len(scheduler.switches) <= 3


async def test_idle_stage_goes_to_the_longest_waiting_stage() -> None:
    """Regression (live e2e 2026-09-23): A and B handed the card back and forth for 12 minutes
    while the scanned-page parse waited; the stage whose caller arrived first must go next."""
    card = _Card()
    scheduler = _scheduler(card)
    order: list[GpuStage] = []
    a_running = asyncio.Event()

    async def run(stage: GpuStage, seconds: float) -> None:
        async with scheduler.hold(stage):
            order.append(stage)
            if stage is GpuStage.EXTRACTOR_A:
                a_running.set()
            await asyncio.sleep(seconds)

    first_a = asyncio.create_task(run(GpuStage.EXTRACTOR_A, 0.05))
    await a_running.wait()
    parse = asyncio.create_task(run(GpuStage.PARSE, 0.01))
    await asyncio.sleep(0.01)
    later_b = [asyncio.create_task(run(GpuStage.EXTRACTOR_B, 0.01)) for _ in range(3)]
    await asyncio.gather(first_a, parse, *later_b)
    assert order[:2] == [GpuStage.EXTRACTOR_A, GpuStage.PARSE]
    assert not card.overlap


async def test_a_stage_keeps_the_card_across_a_short_gap_between_its_calls() -> None:
    """Pages of one scanned document are separate calls; a waiting stage must not grab the card
    in the instant between two pages (each grab would cost an unload and a reload)."""
    now = [0.0]
    card = _Card()
    scheduler = GpuStageScheduler(
        {stage: card.hook(stage) for stage in GpuStage},
        idle_grace_s=3.0,
        poll_s=0.005,
        clock=lambda: now[0],
    )
    order: list[str] = []

    async def pages() -> None:
        for page in range(3):
            await card.work(scheduler, GpuStage.PARSE, 0.01)
            order.append(f"page{page}")
            await asyncio.sleep(0.02)  # next page arrives while A is already waiting

    async def extract() -> None:
        await asyncio.sleep(0.005)
        await card.work(scheduler, GpuStage.EXTRACTOR_A, 0.0)
        order.append("a")

    async def clock() -> None:
        while "a" not in order:
            await asyncio.sleep(0.01)
            if "page2" in order:
                now[0] += 1.0  # only after the last page does the idle time pass the grace

    await asyncio.gather(pages(), extract(), clock())
    assert order == ["page0", "page1", "page2", "a"]
    assert scheduler.switches == [GpuStage.PARSE, GpuStage.EXTRACTOR_A]


async def test_failed_release_propagates_and_the_next_caller_retries_it() -> None:
    card = _Card()
    card.fail_next_release = True
    scheduler = _scheduler(card)
    with pytest.raises(ParseError):
        await card.work(scheduler, GpuStage.EXTRACTOR_A, 0.0)
    assert scheduler.active is None
    await card.work(scheduler, GpuStage.EXTRACTOR_A, 0.0)
    assert scheduler.active is GpuStage.EXTRACTOR_A
    assert card.released == [GpuStage.PARSE, GpuStage.EXTRACTOR_B]


async def test_staged_llm_client_holds_each_models_own_stage() -> None:
    scheduler = GpuStageScheduler({}, idle_grace_s=0.0, poll_s=0.01)
    seen: list[GpuStage | None] = []

    class Inner:
        async def complete_structured(self, request: StructuredRequest) -> CompletionResult:
            seen.append(scheduler.active)
            return CompletionResult(content="", finish_reason="stop", model="m", latency_ms=1)

        async def complete_with_tools(self, request: ToolRequest) -> CompletionResult:
            raise NotImplementedError

        async def embed(self, model: str, texts: object) -> list[list[float]]:
            raise NotImplementedError

    def stage_for(model: str) -> GpuStage:
        return GpuStage.EXTRACTOR_B if model == "extractor-b" else GpuStage.EXTRACTOR_A

    client = StagedLlmClient(Inner(), scheduler, stage_for)
    for model in ("extractor-b", "exception-agent"):
        await client.complete_structured(
            StructuredRequest(
                model=model,
                messages=(ChatMessage(role="user", content="x"),),
                json_schema={},
                temperature=0.0,
                seed=0,
            )
        )
    assert seen == [GpuStage.EXTRACTOR_B, GpuStage.EXTRACTOR_A]


async def test_parse_page_calls_run_inside_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = GpuStageScheduler({}, poll_s=0.01)
    seen: list[GpuStage | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(scheduler.active)
        return httpx.Response(
            200, json={"width_px": 10, "height_px": 10, "words": [], "lines": [], "tables": []}
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )
    client = ParseServiceClient(
        ParseClientConfig(base_url="http://parse:8000"),
        gate=lambda: scheduler.hold(GpuStage.PARSE),
    )
    sha = Sha256Hex(hashlib.sha256(b"doc").hexdigest())
    await client.parse_page_image(document_sha256=sha, page_index=0, png_bytes=b"png")
    assert seen == [GpuStage.PARSE]


@pytest.mark.parametrize(
    ("response", "raises"),
    [
        (httpx.Response(200, json={"loaded": False}), False),
        (httpx.Response(500, text="boom"), True),
        (None, False),  # connection refused: the service is not running, so it holds nothing
    ],
)
async def test_parse_unload_outcomes(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response | None, raises: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/unload"
        if response is None:
            raise httpx.ConnectError("refused", request=request)
        return response

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )
    client = ParseServiceClient(ParseClientConfig(base_url="http://parse:8000"))
    if raises:
        with pytest.raises(ParseError, match="/unload"):
            await client.unload()
    else:
        await client.unload()


async def test_serial_gate_runs_one_page_at_a_time_inside_the_parse_stage() -> None:
    scheduler = GpuStageScheduler({}, idle_grace_s=0.0, poll_s=0.01)
    gate = SerialStageGate(scheduler, GpuStage.PARSE)
    running = 0
    peak = 0

    async def page() -> None:
        nonlocal running, peak
        async with gate():
            assert scheduler.active is GpuStage.PARSE
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.005)
            running -= 1

    await asyncio.gather(*(page() for _ in range(5)))
    assert peak == 1
