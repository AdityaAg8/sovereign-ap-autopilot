"""Tool-free LLM extractor (PRD §3, §10 M3.3): sends one schema-constrained request, parses the
JSON into `RawField`s and line items, and derives per-field confidence from the token logprobs
of each field's value (byte-aligned to the JSON text). Output that does not match the schema is
reported as `valid_json=False` with every field null -- never repaired or retried silently."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from apx.common.errors import ModelOutputError
from apx.common.invoice import VatCategory
from apx.extract.interfaces import (
    INVOICE_TYPE_CODE_KEY,
    LINE_ITEM_KEYS,
    PRINTED_LINE_ITEM_KEYS,
    ExtractorId,
    ExtractorOutput,
    FieldLogprobs,
    FieldName,
    InvoiceTypeCode,
    PromptBuilder,
    RawField,
    RawLineItem,
)
from apx.llm.interfaces import LlmClient, TokenLogprob
from apx.parse.interfaces import ParsedDocument

_WHITESPACE = " \t\n\r"
_DECODER = json.JSONDecoder()


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index] in _WHITESPACE:
        index += 1
    return index


def top_level_value_spans(content: str) -> dict[str, tuple[int, int]]:
    """Character span of each top-level member's value in a JSON object text. Raises
    `ValueError` if `content` is not a single JSON object."""
    spans: dict[str, tuple[int, int]] = {}
    index = _skip_ws(content, 0)
    if index >= len(content) or content[index] != "{":
        raise ValueError("not a JSON object")
    index = _skip_ws(content, index + 1)
    if index < len(content) and content[index] == "}":
        return spans
    while True:
        if index >= len(content) or content[index] != '"':
            raise ValueError(f"expected a key at {index}")
        key, index = _DECODER.raw_decode(content, index)
        index = _skip_ws(content, index)
        if index >= len(content) or content[index] != ":":
            raise ValueError(f"expected ':' at {index}")
        start = _skip_ws(content, index + 1)
        _, end = _DECODER.raw_decode(content, start)
        spans[key] = (start, end)
        index = _skip_ws(content, end)
        if index < len(content) and content[index] == ",":
            index = _skip_ws(content, index + 1)
            continue
        if index < len(content) and content[index] == "}":
            break
        raise ValueError(f"expected ',' or '}}' at {index}")
    if _skip_ws(content, index + 1) != len(content):
        raise ValueError("trailing data after the JSON object")
    return spans


def _token_byte_offsets(content: str, tokens: Sequence[TokenLogprob]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    position = 0
    for token in tokens:
        size = token.nbytes if token.nbytes is not None else len(token.token.encode("utf-8"))
        offsets.append((position, position + size))
        position += size
    if position != len(content.encode("utf-8")):
        raise ModelOutputError(
            f"token logprobs cover {position} bytes but the content has "
            f"{len(content.encode('utf-8'))}; cannot attribute confidence to fields"
        )
    return offsets


def field_logprobs(
    content: str, spans: dict[str, tuple[int, int]], tokens: Sequence[TokenLogprob]
) -> dict[FieldName, FieldLogprobs]:
    """Mean and min logprob over the tokens overlapping each header field's value. A string
    value's quotes are excluded (grammar-forced, near-certain) unless the string is empty."""
    offsets = _token_byte_offsets(content, tokens)
    result: dict[FieldName, FieldLogprobs] = {}
    for name in FieldName:
        if name.value not in spans:
            continue
        start, end = spans[name.value]
        if content[start] == '"' and end - start > 2:
            start, end = start + 1, end - 1
        byte_start = len(content[:start].encode("utf-8"))
        byte_end = len(content[:end].encode("utf-8"))
        values = [
            token.logprob
            for token, (token_start, token_end) in zip(tokens, offsets, strict=True)
            if token_start < byte_end and token_end > byte_start
        ]
        if values:
            result[name] = FieldLogprobs(mean=math.fsum(values) / len(values), min=min(values))
    return result


def _nullable_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


_TYPE_CODES = frozenset(c.value for c in InvoiceTypeCode)
_VAT_CATEGORIES = frozenset(c.value for c in VatCategory)


def _parse_payload(
    content: str,
) -> tuple[tuple[RawField, ...], InvoiceTypeCode | None, tuple[RawLineItem, ...]] | None:
    """Fields, type code and line items if `content` matches the extraction schema exactly (enum
    members included), else None."""
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    expected = {name.value for name in FieldName} | {INVOICE_TYPE_CODE_KEY, "line_items"}
    if not isinstance(payload, dict) or set(payload) != expected:
        return None
    if not all(_nullable_str(payload[name.value]) for name in FieldName):
        return None
    type_code = payload[INVOICE_TYPE_CODE_KEY]
    if type_code is not None and type_code not in _TYPE_CODES:
        return None
    items = payload["line_items"]
    if not isinstance(items, list):
        return None
    line_items: list[RawLineItem] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != set(LINE_ITEM_KEYS):
            return None
        if not all(_nullable_str(item[key]) for key in PRINTED_LINE_ITEM_KEYS):
            return None
        category = item["vat_category"]
        if category is not None and category not in _VAT_CATEGORIES:
            return None
        line_items.append(
            RawLineItem(
                **{key: item[key] for key in PRINTED_LINE_ITEM_KEYS},
                vat_category=None if category is None else VatCategory(category),
            )
        )
    fields = tuple(RawField(name=name, value=payload[name.value]) for name in FieldName)
    return fields, None if type_code is None else InvoiceTypeCode(type_code), tuple(line_items)


def invalid_output(
    extractor: ExtractorId, model: str, finish_reason: str | None = None
) -> ExtractorOutput:
    return ExtractorOutput(
        extractor=extractor,
        model=model,
        fields=tuple(RawField(name=name, value=None) for name in FieldName),
        line_items=(),
        logprobs={},
        valid_json=False,
        finish_reason=finish_reason,
    )


class LlmExtractor:
    """`Extractor` over any `LlmClient`. Raises `ModelServingError` / `ModelOutputError` from the
    client and `ModelOutputError` when logprobs cannot be aligned to the content."""

    def __init__(
        self,
        client: LlmClient,
        prompt_builder: PromptBuilder,
        *,
        model: str,
        extractor: ExtractorId,
    ) -> None:
        self._client = client
        self._prompt_builder = prompt_builder
        self._model = model
        self._extractor = extractor
        self.last_latency_ms: float | None = None

    async def extract(self, document: ParsedDocument) -> ExtractorOutput:
        request = self._prompt_builder.build(document, self._model)
        result = await self._client.complete_structured(request)
        self.last_latency_ms = result.latency_ms
        parsed = _parse_payload(result.content) if result.finish_reason != "length" else None
        finish_reason = result.finish_reason or None
        if parsed is None:
            return invalid_output(self._extractor, result.model, finish_reason)
        fields, type_code, line_items = parsed
        spans = top_level_value_spans(result.content)
        return ExtractorOutput(
            extractor=self._extractor,
            model=result.model,
            fields=fields,
            line_items=line_items,
            logprobs=field_logprobs(result.content, spans, result.token_logprobs),
            valid_json=True,
            invoice_type_code=type_code,
            finish_reason=finish_reason,
        )
