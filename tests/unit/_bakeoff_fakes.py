"""Fakes for the M3.4 bake-off unit tests: a scripted `LlmClient`, a one-line parsed document,
factory label/truth files for a UAE document and a scripted nvidia-smi reading source. No GPU,
no llama-swap, no parse service."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from apx.common.types import BBox, Language
from apx.extract.interfaces import FieldName
from apx.llm.interfaces import CompletionResult, StructuredRequest, TokenLogprob
from apx.parse.interfaces import Line, Page, ParsedDocument, Word

Reply = str | Exception


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 280), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def document(
    sha: str, texts: Sequence[str] = ("Invoice", "INV-1"), pages: int = 1
) -> ParsedDocument:
    """`pages` pages, each holding one line of `texts` laid left to right."""
    built: list[Page] = []
    order: list[str] = []
    for index in range(pages):
        line_id = f"p{index}l0"
        words = tuple(
            Word(
                id=f"p{index}w{n}",
                text=text,
                bbox=BBox(page=index, x0=0.05 + n * 0.2, y0=0.1, x1=0.2 + n * 0.2, y1=0.12),
                line_id=line_id,
                confidence=1.0,
            )
            for n, text in enumerate(texts)
        )
        line = Line(
            id=line_id,
            word_ids=tuple(w.id for w in words),
            bbox=BBox(page=index, x0=0.0, y0=0.1, x1=1.0, y1=0.12),
        )
        built.append(
            Page(
                index=index,
                width_px=200,
                height_px=280,
                words=words,
                lines=(line,),
                tables=(),
                image_png=png(),
            )
        )
        order.append(line_id)
    return ParsedDocument(
        document_sha256=sha,
        pages=tuple(built),
        reading_order=tuple(order),
        languages=frozenset({Language.EN}),
        parser_version="test",
    )


def invoice_json(**values: str | None) -> str:
    payload: dict[str, Any] = {name.value: None for name in FieldName}
    payload["invoice_type_code"] = "380"
    payload.update(values)
    payload["line_items"] = []
    return json.dumps(payload)


def write_uae_doc(data_dir: Path, sha: str, *, arabic: bool = False, number: str = "INV-1") -> None:
    labels = [
        {"field": "invoice_number", "value": number, "page": 0, "bbox": [0.1, 0.1, 0.1, 0.02]}
    ]
    (data_dir / f"{sha}.labels.json").write_text(json.dumps({"labels": labels}), "utf-8")
    truth = {"arabic_first": arabic, "context": {}, "tax_case": "standard", "n_pages": 1}
    (data_dir / f"{sha}.truth.json").write_text(json.dumps(truth), "utf-8")


class FakeLlm:
    """`LlmClient` whose structured answers come from `reply(request)`; an Exception is raised."""

    def __init__(self, reply: Callable[[StructuredRequest], Reply], latency_ms: float = 1000.0):
        self._reply = reply
        self.latency_ms = latency_ms
        self.calls: list[StructuredRequest] = []

    async def complete_structured(self, request: StructuredRequest) -> CompletionResult:
        self.calls.append(request)
        answer = self._reply(request)
        if isinstance(answer, Exception):
            raise answer
        return CompletionResult(
            content=answer,
            token_logprobs=(TokenLogprob(token=answer, logprob=-0.01),),
            finish_reason="stop",
            model=request.model,
            latency_ms=self.latency_ms,
        )

    async def complete_with_tools(self, request: Any) -> CompletionResult:
        raise NotImplementedError

    async def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class Unloads:
    def __init__(self) -> None:
        self.count = 0

    async def __call__(self) -> None:
        self.count += 1


def readings(values: Sequence[float | None], tail: float | None) -> Callable[[], float | None]:
    """nvidia-smi stand-in: `values` in order, then `tail` forever."""
    it: Iterator[float | None] = iter(values)
    return lambda: next(it, tail)
