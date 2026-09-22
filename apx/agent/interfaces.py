"""Interfaces for the exception agent, ops chat and their tools. Document text reaches the agent
only as UntrustedText and tools can only read state or create proposals
(PRD §6.2 I2, §6.3 P5, ADR-0003)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import JsonValue

from apx.common.master import User
from apx.common.types import Frozen, InvoiceId
from apx.common.untrusted import UntrustedText
from apx.llm.interfaces import ToolRequest, ToolSpec
from apx.policy.interfaces import Proposal, RoutingDecision


class AgentContext(Frozen):
    """Everything the agent sees about one invoice. Trusted fields are system state; any
    document-derived text is in `quoted_document` (I2)."""

    invoice_id: InvoiceId
    routing: RoutingDecision
    state: dict[str, JsonValue]
    quoted_document: tuple[UntrustedText, ...]


class Citation(Frozen):
    source: Literal["invoice", "audit", "vendor", "policy", "document"]
    ref: str


class ChatTurn(Frozen):
    role: Literal["user", "assistant"]
    text: str


class AgentReply(Frozen):
    text: str
    citations: tuple[Citation, ...]
    proposals: tuple[Proposal, ...]


class ToolResult(Frozen):
    content: dict[str, JsonValue]
    untrusted: tuple[UntrustedText, ...] = ()


class AgentTool(Protocol):
    """Read-only or proposal-creating tool. No tool mutates payment state (P5)."""

    spec: ToolSpec
    effect: Literal["read", "propose"]

    async def call(self, arguments: dict[str, JsonValue], actor: User | None) -> ToolResult: ...


class AgentPromptBuilder(Protocol):
    def build(self, context: AgentContext, tools: Sequence[ToolSpec], model: str) -> ToolRequest:
        """Render trusted state plus `UntrustedText.quoted()` blocks only (I2)."""
        ...


class ExceptionAgent(Protocol):
    async def handle(self, context: AgentContext) -> AgentReply: ...


class OpsChat(Protocol):
    async def respond(self, history: Sequence[ChatTurn], user: User) -> AgentReply: ...
