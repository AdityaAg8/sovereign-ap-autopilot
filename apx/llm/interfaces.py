"""Interfaces for the local model client, model registry and GPU stage scheduler. Structured
(tool-free) and tool-calling requests are distinct types so extractors cannot carry tools
(I2, ADR-0002, ADR-0003)."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import Field, JsonValue

from apx.common.types import Frozen, Sha256Hex, Stage


class ModelRole(StrEnum):
    """Runtime model roles."""

    PARSE = "parse"
    GUARD = "guard"
    EXTRACTOR_A = "extractor_a"
    EXTRACTOR_B = "extractor_b"
    ARBITER = "arbiter"
    AGENT = "agent"
    TEXT_EMBED = "text_embed"
    PAGE_EMBED = "page_embed"


class ModelSpec(Frozen):
    """One pinned model from infra/models.lock.json plus its serving name and stage."""

    role: ModelRole
    repo_id: str
    filename: str
    revision: str
    sha256: Sha256Hex
    served_name: str
    stage: Stage
    vram_gb_measured: float | None = None


class TextPart(Frozen):
    type: Literal["text"] = "text"
    text: str


class ImagePart(Frozen):
    """Page image sent to a VLM, as a data URL (never a remote URL)."""

    type: Literal["image_url"] = "image_url"
    data_url: Annotated[str, Field(pattern=r"^data:image/(png|jpeg);base64,")]


class ChatMessage(Frozen):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | tuple[TextPart | ImagePart, ...]
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    """On an assistant message: the calls it made, replayed so each following `tool` message
    answers a call the chat template can see."""


class ToolSpec(Frozen):
    """OpenAI-style function tool definition."""

    name: str
    description: str
    parameters: dict[str, JsonValue]


class StructuredRequest(Frozen):
    """Tool-free, JSON-schema constrained request. Has no `tools` field by
    construction: this is the only request type extractors and the arbiter may build (I2)."""

    model: str
    messages: tuple[ChatMessage, ...]
    json_schema: dict[str, JsonValue]
    temperature: Annotated[float, Field(ge=0.0, le=0.2)] = 0.0
    seed: int = 0
    logprobs: Literal[True] = True
    enable_thinking: Literal[False] = False


class ToolRequest(Frozen):
    """Tool-calling request for the exception agent and ops chat only."""

    model: str
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSpec, ...]
    temperature: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    seed: int = 0
    enable_thinking: bool = False


class TokenLogprob(Frozen):
    token: str
    logprob: float
    nbytes: int | None = None
    """UTF-8 byte length of the token as the server reports it. A multi-byte character can split
    across tokens, so byte lengths (not `len(token)`) align tokens to offsets in the content."""


class ToolCall(Frozen):
    id: str
    name: str
    arguments: dict[str, JsonValue]


class CompletionResult(Frozen):
    content: str
    token_logprobs: tuple[TokenLogprob, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str
    model: str
    latency_ms: float


class LlmClient(Protocol):
    """Async client for llama-swap's OpenAI-compatible API on the internal network only."""

    async def complete_structured(self, request: StructuredRequest) -> CompletionResult:
        """Run a constrained completion. Raises ModelServingError or ModelOutputError."""
        ...

    async def complete_with_tools(self, request: ToolRequest) -> CompletionResult:
        """Run a tool-calling completion. Raises ModelServingError."""
        ...

    async def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per text. Raises ModelServingError."""
        ...


class ModelRegistry(Protocol):
    """Resolves roles to pinned models; Extractor B resolves via MLflow alias."""

    def get(self, role: ModelRole) -> ModelSpec:
        """Return the spec for a role. Raises ConfigError if not pinned."""
        ...

    def extractor_b(self, alias: Literal["champion", "challenger"]) -> ModelSpec:
        """Return the Extractor B build registered under an MLflow alias."""
        ...


class StageScheduler(Protocol):
    """Serialises GPU stages so only one llama-swap group is resident at a time (ADR-0002)."""

    def stage(self, stage: Stage) -> AbstractAsyncContextManager[None]:
        """Async context that loads the stage's group and holds the GPU lock until exit.
        Raises VramBudgetError if the measured peak would exceed 11.5 GB."""
        ...
