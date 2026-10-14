"""The agent tool loop and its two callers (ADR-0010): the exception agent, which
writes a cited review note for a review-routed invoice, and ops chat, which answers staff
questions. Both run the same bounded loop over the tool dispatcher and return only replies whose
citations trace back to this turn's tool results; anything else becomes the fixed refusal."""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Final, cast

from pydantic import JsonValue

from apx.agent.contracts import check_citations, parse_citations
from apx.agent.interfaces import (
    REFUSAL_TEMPLATES,
    TOOL_CONTRACTS,
    AgentConfig,
    AgentContext,
    AgentPromptBuilder,
    AgentReply,
    ChatTurn,
    RefusalReason,
    ToolCaller,
    ToolDispatcher,
    ToolExecution,
    ToolName,
    ToolRefusal,
    ToolResult,
    ToolTrace,
)
from apx.audit.interfaces import AuditEvent, AuditLog
from apx.common.master import User
from apx.common.types import InvoiceId
from apx.llm.interfaces import LlmClient, ToolCall, ToolRequest
from apx.policy.interfaces import Proposal

StepListener = Callable[[ToolTrace], Awaitable[None]]
"""Called after every tool call, so a UI can show what the agent is doing while it works."""

REVIEW_NOTE_ACTION: Final[str] = "agent.review_note"
CHAT_UNVERIFIABLE_ACTION: Final[str] = "chat.unverifiable"

_PROPOSE_TOOLS: Final[frozenset[str]] = frozenset(
    name for name, contract in TOOL_CONTRACTS.items() if contract.effect == "propose"
)


def _session_id() -> str:
    return secrets.token_hex(12)


def _refused(
    reason: RefusalReason, trace: Sequence[ToolTrace], proposals: Sequence[Proposal]
) -> AgentReply:
    return AgentReply(
        text=REFUSAL_TEMPLATES[reason],
        citations=(),
        proposals=tuple(proposals),
        refusal=reason,
        trace=tuple(trace),
    )


def _proposal_cap_refusal(call: ToolCall) -> ToolExecution:
    tool = cast(ToolName, call.name)
    refusal = ToolRefusal(
        tool=tool, code="rate_limited", message="The proposal limit for one answer is reached."
    )
    trace = ToolTrace(
        call_id=call.id,
        tool=tool,
        arguments=dict(call.arguments),
        outcome="refused",
        latency_ms=0.0,
        refusal="rate_limited",
    )
    return ToolExecution(result=ToolResult(content=refusal.model_dump(mode="json")), trace=trace)


def _proposal(execution: ToolExecution, caller: ToolCaller) -> Proposal:
    """The pending proposal a propose tool just created, rebuilt from its typed receipt."""
    content = execution.result.content
    payload: dict[str, JsonValue] = {
        key: content[key] for key in ("subject", "body", "to_vendor_id", "note") if key in content
    }
    kind = "request_info" if execution.trace.tool == "draft_vendor_email" else "approve_payment"
    return Proposal.model_validate(
        {
            "id": execution.trace.proposal_id,
            "invoice_id": content.get("invoice_id"),
            "kind": kind,
            "payload": payload,
            "created_by": caller.origin,
            "created_at": datetime.now(UTC),
        }
    )


class ToolLoop:
    """Bounded tool-calling loop: at most `config.max_tool_calls` tool calls and
    `config.max_proposals` proposals per answer. Returns the model's final text with its parsed
    citations, unchecked; callers decide what an unverifiable reply becomes."""

    def __init__(
        self,
        llm: LlmClient,
        dispatcher: ToolDispatcher,
        prompts: AgentPromptBuilder,
        config: AgentConfig,
    ) -> None:
        self._llm = llm
        self._dispatcher = dispatcher
        self._prompts = prompts
        self._config = config

    async def run(
        self, request: ToolRequest, caller: ToolCaller, on_step: StepListener | None = None
    ) -> AgentReply:
        request = request.model_copy(
            update={
                "temperature": self._config.temperature,
                "seed": self._config.seed,
                "enable_thinking": self._config.enable_thinking,
            }
        )
        trace: list[ToolTrace] = []
        proposals: list[Proposal] = []
        while True:
            result = await self._llm.complete_with_tools(request)
            if not result.tool_calls:
                text = result.content.strip()
                if not text:
                    return _refused("unverifiable", trace, proposals)
                return AgentReply(
                    text=text,
                    citations=parse_citations(text),
                    proposals=tuple(proposals),
                    trace=tuple(trace),
                )
            if len(trace) + len(result.tool_calls) > self._config.max_tool_calls:
                return _refused("budget_exhausted", trace, proposals)
            executions: list[ToolExecution] = []
            for call in result.tool_calls:
                if call.name in _PROPOSE_TOOLS and len(proposals) >= self._config.max_proposals:
                    execution = _proposal_cap_refusal(call)
                else:
                    execution = await self._dispatcher.dispatch(call, caller)
                    if execution.trace.proposal_id is not None:
                        proposals.append(_proposal(execution, caller))
                trace.append(execution.trace)
                executions.append(execution)
                if on_step is not None:
                    await on_step(execution.trace)
            request = self._prompts.with_tool_results(request, result, executions)


class LlmExceptionAgent:
    """The `ExceptionAgent` implementation: runs as origin `agent`, scoped to one invoice."""

    def __init__(
        self, loop: ToolLoop, dispatcher: ToolDispatcher, prompts: AgentPromptBuilder, model: str
    ) -> None:
        self._loop = loop
        self._dispatcher = dispatcher
        self._prompts = prompts
        self._model = model

    async def handle(self, context: AgentContext) -> AgentReply:
        caller = ToolCaller(
            origin="agent", user=None, session_id=_session_id(), scope_invoice_id=context.invoice_id
        )
        request = self._prompts.build(context, self._dispatcher.specs(caller), self._model)
        reply = await self._loop.run(request, caller)
        if reply.refusal is not None:
            return reply
        # A review note must be grounded: at least one citation, and every citation real.
        if not reply.citations or not check_citations(reply).ok:
            return _refused("unverifiable", reply.trace, reply.proposals)
        return reply


class LlmOpsChat:
    """The `OpsChat` implementation: runs as origin `chat` for the signed-in user. A reply whose
    citations fail the check is replaced by the fixed refusal and audited."""

    def __init__(
        self,
        loop: ToolLoop,
        dispatcher: ToolDispatcher,
        prompts: AgentPromptBuilder,
        audit: AuditLog,
        model: str,
    ) -> None:
        self._loop = loop
        self._dispatcher = dispatcher
        self._prompts = prompts
        self._audit = audit
        self._model = model

    async def respond(
        self, history: Sequence[ChatTurn], user: User, on_step: StepListener | None = None
    ) -> AgentReply:
        caller = ToolCaller(origin="chat", user=user, session_id=_session_id())
        request = self._prompts.build_chat(history, self._dispatcher.specs(caller), self._model)
        reply = await self._loop.run(request, caller, on_step)
        if reply.refusal is not None:
            return reply
        check = check_citations(reply)
        if check.ok:
            return reply
        await self._audit.append(
            AuditEvent(
                actor=f"chat:{user.id}",
                action=CHAT_UNVERIFIABLE_ACTION,
                entity=f"user:{user.id}",
                payload={
                    "invalid": [f"{c.source}:{c.ref}" for c in check.invalid],
                    "tool_calls": len(reply.trace),
                },
            )
        )
        return _refused("unverifiable", reply.trace, reply.proposals)


async def record_review_note(audit: AuditLog, invoice_id: InvoiceId, reply: AgentReply) -> None:
    """Store an exception-agent review note as a hash-chained audit event on the invoice. Refused
    replies are never stored (ADR-0010)."""
    if reply.refusal is not None:
        return
    await audit.append(
        AuditEvent(
            actor="agent",
            action=REVIEW_NOTE_ACTION,
            entity=f"invoice:{invoice_id}",
            payload={
                "text": reply.text,
                "citations": [f"{c.source}:{c.ref}" for c in reply.citations],
                "proposal_ids": [p.id for p in reply.proposals],
            },
        )
    )


__all__ = [
    "CHAT_UNVERIFIABLE_ACTION",
    "REVIEW_NOTE_ACTION",
    "LlmExceptionAgent",
    "LlmOpsChat",
    "StepListener",
    "ToolLoop",
    "record_review_note",
]
