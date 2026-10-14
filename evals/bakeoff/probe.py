"""5-page VRAM probe of every M3.4 bake-off arm (C2, ADR-0013), run before any eval
hours: loads the arm alone from an empty llama-swap, sends its real extraction prompt for 5-page
UAE test invoices at its served concurrency while sampling whole-card nvidia-smi, and records
idle, peak and model (peak - idle) GiB; an arm over budget is disqualified in
reports/bakeoff_probe.json.

Usage: python -m evals.bakeoff.probe [--arms a,q38-27b,...] [--data-dir data/uae/test]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

from apx.common.errors import ModelServingError
from apx.llm.interfaces import CompletionResult, StructuredRequest
from apx.parse.interfaces import ParsedDocument
from evals.bakeoff.arms import (
    ARMS,
    MODEL_BUDGET_GB,
    PROBE_PAGES,
    WHOLE_CARD_BUDGET_GB,
    Arm,
    BakeoffError,
    arm_by_key,
    invoice_prompt_builder,
    served_parallel,
)
from ml.extractor_b.gates import StrictVramSampler, VramUnavailableError

PROBE_PATH: Final[Path] = Path("reports/bakeoff_probe.json")
PROBE_DOCS: Final[int] = 2
PROBE_ROUNDS: Final[int] = 2

ProbeStatus = Literal["ok", "over_budget", "failed"]
CallFn = Callable[[StructuredRequest], Awaitable[CompletionResult]]
UnloadFn = Callable[[], Awaitable[None]]
SampleFn = Callable[[], float | None]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    arm: str
    alias: str
    status: ProbeStatus
    reason: str
    idle_gb: float
    peak_gb: float | None
    """Whole-card nvidia-smi peak (GiB) over load + burst; None when the model never served."""
    model_gb: float | None
    """peak - idle: the arm's models with their working context (the 9.5 GB rule)."""
    concurrency: int
    requests: int
    failed_requests: int
    vram_samples: int
    measured_at: str

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def budget_status(idle_gb: float, peak_gb: float) -> tuple[ProbeStatus, str]:
    """`over_budget` when the whole card exceeds 11.8 GB or the model share exceeds 9.5 GB."""
    model_gb = peak_gb - idle_gb
    reasons = []
    if peak_gb > WHOLE_CARD_BUDGET_GB:
        reasons.append(f"whole-card peak {peak_gb:.2f} GB > {WHOLE_CARD_BUDGET_GB} GB")
    if model_gb > MODEL_BUDGET_GB:
        reasons.append(f"model {model_gb:.2f} GB > {MODEL_BUDGET_GB} GB")
    if reasons:
        return "over_budget", "; ".join(reasons)
    return "ok", (
        f"peak {peak_gb:.2f} GB ≤ {WHOLE_CARD_BUDGET_GB} GB, model {model_gb:.2f} GB ≤ "
        f"{MODEL_BUDGET_GB} GB"
    )


async def probe_arm(
    arm: Arm,
    documents: Sequence[ParsedDocument],
    *,
    call: CallFn,
    unload: UnloadFn,
    sample_fn: SampleFn,
    concurrency: int,
    rounds: int = PROBE_ROUNDS,
    interval_s: float = 0.2,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ProbeResult:
    """One untimed load call, then `rounds` bursts of `concurrency` requests, all inside one
    whole-card sampling window that fails closed on any missed nvidia-smi reading."""
    if not documents:
        raise BakeoffError(f"{arm.key}: no {PROBE_PAGES}-page documents to probe with")
    builder = invoice_prompt_builder(arm)
    requests = [builder.build(document, arm.alias) for document in documents]
    await unload()  # the arm alone on the card, from an empty llama-swap
    idle = sample_fn()
    if idle is None:
        raise VramUnavailableError(f"{arm.key}: nvidia-smi gave no idle reading")
    sent = failed = 0
    error: str | None = None
    served = True
    sampler = StrictVramSampler(sample_fn, interval_s=interval_s)
    try:
        with sampler:
            try:
                await call(requests[0])  # loads the model: its allocation is in the window
            except ModelServingError as exc:
                served, error = False, f"did not serve: {exc}"
            for round_index in range(rounds if served else 0):
                batch = [
                    requests[(round_index * concurrency + i) % len(requests)]
                    for i in range(concurrency)
                ]
                results = await asyncio.gather(*(call(r) for r in batch), return_exceptions=True)
                for result in results:
                    sent += 1
                    if isinstance(result, ModelServingError):
                        failed += 1
                        error = error or f"request failed: {result}"
                    elif isinstance(result, BaseException):
                        raise result
    finally:
        await unload()
    peak = sampler.peak_gb if served else None  # a missed reading raises: no unmeasured pass
    status: ProbeStatus
    if served and not failed and peak is not None:
        status, reason = budget_status(idle, peak)
    else:
        status, reason = "failed", error or "failed"
    return ProbeResult(
        arm=arm.key,
        alias=arm.alias,
        status=status,
        reason=reason,
        idle_gb=round(idle, 3),
        peak_gb=None if peak is None else round(peak, 3),
        model_gb=None if peak is None else round(peak - idle, 3),
        concurrency=concurrency,
        requests=sent,
        failed_requests=failed,
        vram_samples=sampler.samples,
        measured_at=now().isoformat(),
    )


def load_probe(path: Path) -> dict[str, ProbeResult]:
    """Probe results by arm key ({} when no probe has run)."""
    if not path.is_file():
        return {}
    body = json.loads(path.read_text(encoding="utf-8"))
    return {entry["arm"]: ProbeResult(**entry) for entry in body["arms"]}


def write_probe(path: Path, results: Sequence[ProbeResult]) -> None:
    """Merge `results` into the probe file by arm key, keeping arms not probed this time."""
    merged = load_probe(path)
    merged.update({result.arm: result for result in results})
    ordered = [merged[arm.key] for arm in ARMS if arm.key in merged]
    body = {
        "budget": {"whole_card_gb": WHOLE_CARD_BUDGET_GB, "model_gb": MODEL_BUDGET_GB},
        "pages": PROBE_PAGES,
        "arms": [result.as_json() for result in ordered],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8", newline="\n")


def probe_document_ids(
    data_dir: Path, test_ids: Sequence[str], count: int = PROBE_DOCS
) -> list[str]:
    """The first `count` born-digital `PROBE_PAGES`-page UAE test documents in sha order (born-
    digital parses on the CPU, so the parse service never enters the probe window)."""
    chosen: list[str] = []
    for sha in sorted(test_ids):
        truth_path = data_dir / f"{sha}.truth.json"
        if not truth_path.is_file():
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        if truth.get("n_pages") == PROBE_PAGES and not truth.get("has_scan"):
            chosen.append(sha)
            if len(chosen) == count:
                break
    return chosen


def parse_arms(value: str) -> list[Arm]:
    return [arm_by_key(key.strip()) for key in value.split(",") if key.strip()] or list(ARMS)


async def _live(args: argparse.Namespace) -> int:  # pragma: no cover - needs the GPU + llama-swap
    from scripts.smoke import sample_vram_gb

    from apx.llm.client import LlamaSwapClient, LlmSettings
    from apx.parse.parser import PaddleOcrDocumentParser
    from evals.splits import load_splits

    shas = probe_document_ids(args.data_dir, load_splits().sets["uae"]["test"])
    parser = PaddleOcrDocumentParser()
    documents = [
        await parser.parse(sha, (args.data_dir / f"{sha}.pdf").read_bytes()) for sha in shas
    ]
    arms = parse_arms(args.arms)
    client = LlamaSwapClient(
        LlmSettings(base_url=args.base_url, allowed_models=frozenset(a.alias for a in arms))
    )
    results: list[ProbeResult] = []
    try:
        for index, arm in enumerate(arms, start=1):
            print(f"[{index}/{len(arms)}] probe {arm.key} ({arm.alias})", flush=True)
            result = await probe_arm(
                arm,
                documents,
                call=client.complete_structured,
                unload=client.unload,
                sample_fn=sample_vram_gb,
                concurrency=served_parallel(args.llama_swap_config, arm.alias),
            )
            results.append(result)
            write_probe(args.out, results)  # after every arm: an interrupted probe keeps its arms
            print(f"      {result.status}: {result.reason}", flush=True)
    finally:
        await client.aclose()
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(prog="python -m evals.bakeoff.probe", description=__doc__)
    parser.add_argument("--arms", default="", help="comma-separated arm keys (default: all)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/uae/test"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path, default=PROBE_PATH)
    parser.add_argument("--llama-swap-config", type=Path, default=Path("infra/llama-swap.yaml"))
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_live(args))
    except BakeoffError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
