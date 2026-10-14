"""`LlamaSwapClient`: the `apx.llm.interfaces.LlmClient` over llama-swap's OpenAI-compatible API
on the internal network (I2). Structured calls always send the extraction
settings -- thinking off, JSON-schema constrained output, the request's temperature and seed,
logprobs on -- and every transport or protocol failure is a typed error, never an empty result.
The one retry here is bounded, logged backoff on llama-swap's 429/503 "busy" answers
(`ModelBusyError` once exhausted); worker-side slot limits (`apx.llm.slots`) keep those rare."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from functools import lru_cache
from typing import Any

import httpx
import structlog
from pydantic import Field, JsonValue
from pydantic_settings import BaseSettings, SettingsConfigDict

from apx.common.errors import (
    ModelBusyError,
    ModelContextError,
    ModelNotAllowedError,
    ModelOutputError,
    ModelServingError,
)
from apx.llm.interfaces import (
    ChatMessage,
    CompletionResult,
    ImagePart,
    StructuredRequest,
    TextPart,
    TokenLogprob,
    ToolCall,
    ToolRequest,
)

RUNTIME_MODEL_ALIASES: frozenset[str] = frozenset(
    {"extractor-a", "exception-agent", "ops-chat", "extractor-b", "arbiter"}
)
"""The llama-swap aliases the runtime pipeline may call (infra/llama-swap.yaml runtime models).
Offline-only models (bake-off, teacher, red-team) are deliberately absent."""


class LlmSettings(BaseSettings):
    """`LLM_*` environment settings. The default URL is the in-network service name; a host
    process (evals, dev) reaches the same API through the gateway at `http://127.0.0.1:8080`."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    base_url: str = "http://llama-swap:8080"
    timeout_s: float = Field(default=600.0, gt=0)
    """Generous: the first call of a stage cold-loads a GGUF (~90 s for Extractor A)."""
    max_tokens: int = Field(default=4096, gt=0)
    busy_retries: int = Field(default=6, ge=0)
    """Extra attempts after a 429/503 before `ModelBusyError`; with the defaults below the
    waits are ~1, 2, 4, 8, 16, 30 s (plus up to 25% jitter), about a minute in all."""
    busy_backoff_s: float = Field(default=1.0, gt=0)
    busy_backoff_max_s: float = Field(default=30.0, gt=0)
    allowed_models: frozenset[str] = RUNTIME_MODEL_ALIASES
    """Model aliases this client may call. Offline tooling passes its own explicit set."""


logger = structlog.get_logger(__name__)

BUSY_STATUSES: frozenset[int] = frozenset({429, 503})
"""llama-swap's per-model/global request limit (429) and a model still loading (503)."""
_CONTEXT_MARKERS: tuple[str, ...] = (
    "context size has been exceeded",
    "exceeds the available context",
)


def classify_http_error(path: str, status: int, text: str) -> ModelServingError:
    """The typed error for a non-200 llama-swap answer: busy (retryable), context overflow
    (deterministic, never retried) or any other serving failure."""
    message = f"llama-swap {path} returned HTTP {status}: {text[:300]}"
    if status in BUSY_STATUSES:
        return ModelBusyError(message)
    if any(marker in text.lower() for marker in _CONTEXT_MARKERS):
        return ModelContextError(message)
    return ModelServingError(message)


def busy_backoff_s(attempt: int, settings: LlmSettings, rng: random.Random) -> float:
    """Exponential backoff for the `attempt`-th busy retry (0-based), capped, with up to 25%
    added jitter so queued callers do not all return in the same instant."""
    base = min(settings.busy_backoff_max_s, settings.busy_backoff_s * float(2**attempt))
    return base * (1.0 + 0.25 * rng.random())


@lru_cache(maxsize=1)
def get_llm_settings() -> LlmSettings:
    return LlmSettings()


def _content_json(content: str | tuple[TextPart | ImagePart, ...]) -> JsonValue:
    if isinstance(content, str):
        return content
    parts: list[JsonValue] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        else:
            parts.append({"type": "image_url", "image_url": {"url": part.data_url}})
    return parts


def _messages_json(messages: Sequence[ChatMessage]) -> list[JsonValue]:
    out: list[JsonValue] = []
    for message in messages:
        item: dict[str, JsonValue] = {
            "role": message.role,
            "content": _content_json(message.content),
        }
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in message.tool_calls
            ]
        out.append(item)
    return out


def _first_choice(payload: Any) -> dict[str, Any]:
    try:
        choice = payload["choices"][0]
        if not isinstance(choice, dict):
            raise TypeError("choice is not an object")
        return choice
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelOutputError(f"completion response has no choices: {exc}") from exc


def _token_logprobs(choice: dict[str, Any]) -> tuple[TokenLogprob, ...]:
    content = (choice.get("logprobs") or {}).get("content") or []
    try:
        return tuple(
            TokenLogprob(
                token=str(item["token"]),
                logprob=float(item["logprob"]),
                nbytes=len(item["bytes"]) if item.get("bytes") is not None else None,
            )
            for item in content
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelOutputError(f"malformed logprobs in completion response: {exc}") from exc


class LlamaSwapClient:
    """Async `LlmClient`. One `httpx.AsyncClient` per instance; close it with `aclose()`."""

    def __init__(
        self,
        settings: LlmSettings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._settings = settings or get_llm_settings()
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._http = httpx.AsyncClient(
            base_url=self._settings.base_url,
            timeout=self._settings.timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, path: str, body: dict[str, JsonValue]) -> tuple[Any, float]:
        attempt = 0
        while True:
            start = time.perf_counter()
            try:
                response = await self._http.post(path, json=body)
            except httpx.HTTPError as exc:
                raise ModelServingError(f"llama-swap request to {path} failed: {exc!r}") from exc
            latency_ms = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                break
            error = classify_http_error(path, response.status_code, response.text)
            if not isinstance(error, ModelBusyError) or attempt >= self._settings.busy_retries:
                raise error
            delay = busy_backoff_s(attempt, self._settings, self._rng)
            logger.warning(
                "llm_busy_backoff",
                path=path,
                model=body.get("model"),
                status=response.status_code,
                attempt=attempt + 1,
                delay_s=round(delay, 2),
            )
            await self._sleep(delay)
            attempt += 1
        try:
            return response.json(), latency_ms
        except ValueError as exc:
            raise ModelOutputError(f"llama-swap {path} returned non-JSON: {exc}") from exc

    def _check_model(self, model: str) -> None:
        if model not in self._settings.allowed_models:
            raise ModelNotAllowedError(f"model alias {model!r} is not allowed for this client")

    async def complete_structured(self, request: StructuredRequest) -> CompletionResult:
        self._check_model(request.model)
        body: dict[str, JsonValue] = {
            "model": request.model,
            "messages": _messages_json(request.messages),
            "temperature": request.temperature,
            "seed": request.seed,
            "logprobs": request.logprobs,
            "max_tokens": self._settings.max_tokens,
            "chat_template_kwargs": {"enable_thinking": request.enable_thinking},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "strict": True,
                    "schema": request.json_schema,
                },
            },
        }
        payload, latency_ms = await self._post("/v1/chat/completions", body)
        choice = _first_choice(payload)
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise ModelOutputError("structured completion has no text content")
        return CompletionResult(
            content=content,
            token_logprobs=_token_logprobs(choice),
            finish_reason=str(choice.get("finish_reason") or ""),
            model=str(payload.get("model") or request.model),
            latency_ms=latency_ms,
        )

    async def complete_with_tools(self, request: ToolRequest) -> CompletionResult:
        self._check_model(request.model)
        body: dict[str, JsonValue] = {
            "model": request.model,
            "messages": _messages_json(request.messages),
            "temperature": request.temperature,
            "seed": request.seed,
            "max_tokens": self._settings.max_tokens,
            "chat_template_kwargs": {"enable_thinking": request.enable_thinking},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ],
        }
        payload, latency_ms = await self._post("/v1/chat/completions", body)
        choice = _first_choice(payload)
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            try:
                function = raw["function"]
                arguments = json.loads(function.get("arguments") or "{}")
                calls.append(
                    ToolCall(id=str(raw["id"]), name=str(function["name"]), arguments=arguments)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelOutputError(f"malformed tool call in completion: {exc}") from exc
        return CompletionResult(
            content=str(message.get("content") or ""),
            tool_calls=tuple(calls),
            finish_reason=str(choice.get("finish_reason") or ""),
            model=str(payload.get("model") or request.model),
            latency_ms=latency_ms,
        )

    async def running(self) -> list[dict[str, Any]]:
        """Models llama-swap currently holds in memory, with their state (for the live view)."""
        try:
            response = await self._http.get("/running")
            response.raise_for_status()
            rows = response.json().get("running", [])
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise ModelServingError(f"llama-swap /running failed: {exc!r}") from exc
        return [{"model": r.get("model"), "state": r.get("state")} for r in rows]

    async def unload(self) -> None:
        """Unload every model now, freeing VRAM as soon as a stage's work is done instead of
        waiting for the idle TTL."""
        try:
            response = await self._http.get("/unload")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelServingError(f"llama-swap /unload failed: {exc!r}") from exc

    async def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        self._check_model(model)
        payload, _ = await self._post("/v1/embeddings", {"model": model, "input": list(texts)})
        try:
            rows = sorted(payload["data"], key=lambda row: row["index"])
            return [[float(x) for x in row["embedding"]] for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelOutputError(f"malformed embeddings response: {exc}") from exc
