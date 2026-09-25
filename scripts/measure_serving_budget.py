"""Serving-budget probe (ADR-0013): sends the real Extractor A and B prompts for the largest test
invoices (5 pages, born-digital, parsed host-side) to the running llama-swap at the given
concurrency (one stage per run: `--a-concurrency 0` measures B alone) while sampling
`nvidia-smi` for whole-card memory. Prints idle, loaded and peak MiB and each request's prompt
tokens and status, so a ctx/--parallel choice rests on a peak under real load, not idle.
Needs `make up`; reaches llama-swap through the gateway."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from apx.extract.interfaces import extraction_json_schema
from apx.extract.prompt import InvoicePromptBuilder
from apx.extract.prompt_b import ImageOnlyPromptBuilder
from apx.llm.client import _messages_json
from apx.llm.interfaces import StructuredRequest
from apx.parse.interfaces import ParsedDocument
from apx.parse.parser import PaddleOcrDocumentParser

ROOT = Path(__file__).resolve().parents[1]


class GpuSampler:
    """Polls whole-card used memory (MiB) on a background thread."""

    def __init__(self, period_s: float = 0.25) -> None:
        self.samples: list[int] = []
        self._period = period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def read() -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0])

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.read())
            self._stop.wait(self._period)

    def __enter__(self) -> GpuSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


def pick_documents(data: Path, pages: int, count: int) -> list[Path]:
    chosen: list[Path] = []
    for path in sorted((data / "uae" / "test").glob("*.truth.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        if truth["n_pages"] == pages and not truth["has_scan"]:
            chosen.append(path.with_name(path.name.replace(".truth.json", ".pdf")))
        if len(chosen) == count:
            break
    return chosen


async def parse(path: Path) -> ParsedDocument:
    data = path.read_bytes()
    return await PaddleOcrDocumentParser().parse(hashlib.sha256(data).hexdigest(), data)


def body(request: StructuredRequest, max_tokens: int) -> dict[str, Any]:
    return {
        "model": request.model,
        "messages": _messages_json(request.messages),
        "temperature": request.temperature,
        "seed": request.seed,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction",
                "strict": True,
                "schema": extraction_json_schema(),
            },
        },
    }


async def send(http: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = await http.post("/v1/chat/completions", json=payload)
    seconds = time.perf_counter() - started
    row: dict[str, Any] = {"model": payload["model"], "status": response.status_code, "s": seconds}
    if response.status_code == 200:
        usage = response.json().get("usage", {})
        row["prompt_tokens"] = usage.get("prompt_tokens")
        row["completion_tokens"] = usage.get("completion_tokens")
    else:
        row["error"] = response.text[:200]
    return row


async def main_async(args: argparse.Namespace) -> int:
    docs = [await parse(p) for p in pick_documents(args.data, args.pages, args.docs)]
    if not docs:
        print("no documents matched", file=sys.stderr)
        return 2
    a_builder, b_builder = InvoicePromptBuilder(), ImageOnlyPromptBuilder()
    idle = GpuSampler.read()
    async with httpx.AsyncClient(base_url=args.base_url, timeout=900) as http:
        # Warm-up loads each measured model before the burst (one at a time: with the extract
        # group swapping, only the model(s) of the measured stage stay resident).
        if args.a_concurrency:
            await send(http, body(a_builder.build(docs[0], "extractor-a"), args.max_tokens))
        if args.b_concurrency:
            await send(http, body(b_builder.build(docs[0], "extractor-b"), args.max_tokens))
        loaded = GpuSampler.read()
        payloads = [
            body(a_builder.build(docs[i % len(docs)], "extractor-a"), args.max_tokens)
            for i in range(args.a_concurrency)
        ] + [
            body(b_builder.build(docs[i % len(docs)], "extractor-b"), args.max_tokens)
            for i in range(args.b_concurrency)
        ]
        rows: list[dict[str, Any]] = []
        with GpuSampler() as sampler:
            for _ in range(args.rounds):
                rows += await asyncio.gather(*(send(http, p) for p in payloads))
    for row in rows:
        print(json.dumps(row))
    ok = [r for r in rows if r["status"] == 200]

    def of(model: str, key: str) -> list[Any]:
        return [r[key] for r in ok if r["model"] == model]

    summary = {
        "idle_mib": idle,
        "loaded_mib": loaded,
        "peak_mib": max(sampler.samples),
        "samples": len(sampler.samples),
        "requests": len(rows),
        "failed": len(rows) - len(ok),
        "a_prompt_max": max(of("extractor-a", "prompt_tokens"), default=None),
        "b_prompt_max": max(of("extractor-b", "prompt_tokens"), default=None),
        "a_completion_max": max(of("extractor-a", "completion_tokens"), default=None),
        "b_completion_max": max(of("extractor-b", "completion_tokens"), default=None),
        "a_s_median": statistics.median(of("extractor-a", "s") or [0]),
        "b_s_median": statistics.median(of("extractor-b", "s") or [0]),
    }
    print(json.dumps(summary))
    return 0 if len(ok) == len(rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--docs", type=int, default=4)
    parser.add_argument("--a-concurrency", type=int, default=2)
    parser.add_argument("--b-concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
