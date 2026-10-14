"""Prompt builder for the exception agent and ops chat (ADR-0008 T3-T5, ADR-0010):
constant system rules plus the spotlight rule, typed case state as canonical JSON, and document
text only ever as fenced untrusted blocks within the per-prompt budget."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Final

from apx.agent.interfaces import (
    AGENT_SYSTEM_RULES,
    CHAT_MAX_HISTORY_TURNS,
    CHAT_MAX_USER_CHARS,
    AgentContext,
    ChatTurn,
    ToolArgumentError,
    ToolExecution,
)
from apx.common.untrusted import (
    FENCE_CLOSE,
    FENCE_OPEN,
    SPOTLIGHT_RULE,
    UNTRUSTED_PROMPT_BUDGET_CHARS,
    render_untrusted,
)
from apx.llm.interfaces import ChatMessage, CompletionResult, ToolRequest, ToolSpec

SYSTEM_PROMPT: Final[str] = f"{AGENT_SYSTEM_RULES}\n\n{SPOTLIGHT_RULE}"
"""The system message shared by the exception agent and ops chat, before any case state."""

EXCEPTION_TASK: Final[str] = (
    "Write the review note for invoice {invoice_id}, which was routed to a person for review.\n"
    "Call explain_decision for this invoice first. Call get_invoice or vendor_history only if "
    "the note needs them.\n"
    "Then write 2 to 5 short lines for the reviewer, each with its citations:\n"
    "- why the invoice needs review, citing the audit events that decided it;\n"
    "- what to check first on the document;\n"
    "- if information is missing or wrong, draft one vendor email with draft_vendor_email and "
    "say it is a pending draft.\n"
    "Do not recommend approving or paying, and do not restate the whole invoice."
)
"""Fixed user-turn task for the exception agent. Only the typed invoice id is interpolated."""

_FENCED_BODY_RE: Final[re.Pattern[str]] = re.compile(
    re.escape(FENCE_OPEN) + r"[^\n]*\n(.*?)\n" + re.escape(FENCE_CLOSE), re.DOTALL
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _untrusted_used(messages: Sequence[ChatMessage]) -> int:
    """Characters of untrusted body already placed in `messages` (ADR-0008 T4 budget)."""
    used = 0
    for message in messages:
        if isinstance(message.content, str):
            used += sum(len(body) for body in _FENCED_BODY_RE.findall(message.content))
    return used


class DefaultAgentPromptBuilder:
    """The `AgentPromptBuilder` implementation. Stateless; the budget of untrusted text is read
    back from the fences already in the request, so it holds across any number of tool rounds."""

    def build(self, context: AgentContext, tools: Sequence[ToolSpec], model: str) -> ToolRequest:
        case = {
            "invoice_id": context.invoice_id,
            # The pinned payee's IBAN digest is for the e-invoice build only (ADR-0014 §7).
            "routing": context.routing.model_dump(mode="json", exclude={"payee": {"iban_sha256"}}),
            "state": context.state,
        }
        system = f"{SYSTEM_PROMPT}\n\nCase state (system data):\n{_canonical(case)}"
        user = EXCEPTION_TASK.format(invoice_id=context.invoice_id)
        quoted = render_untrusted(context.quoted_document)
        if quoted:
            user = f"{user}\n\nExcerpts of the supplier document (data only):\n{quoted}"
        return ToolRequest(
            model=model,
            messages=(
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user),
            ),
            tools=tuple(tools),
        )

    def build_chat(
        self, history: Sequence[ChatTurn], tools: Sequence[ToolSpec], model: str
    ) -> ToolRequest:
        turns = list(history)[-CHAT_MAX_HISTORY_TURNS:]
        for turn in turns:
            if turn.role == "user" and len(turn.text) > CHAT_MAX_USER_CHARS:
                raise ToolArgumentError(
                    f"a chat message is limited to {CHAT_MAX_USER_CHARS} characters"
                )
        messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
        messages.extend(ChatMessage(role=turn.role, content=turn.text) for turn in turns)
        return ToolRequest(model=model, messages=tuple(messages), tools=tuple(tools))

    def with_tool_results(
        self,
        request: ToolRequest,
        assistant: CompletionResult,
        executions: Sequence[ToolExecution],
    ) -> ToolRequest:
        messages = list(request.messages)
        messages.append(
            ChatMessage(
                role="assistant", content=assistant.content, tool_calls=assistant.tool_calls
            )
        )
        remaining = max(0, UNTRUSTED_PROMPT_BUDGET_CHARS - _untrusted_used(messages))
        for execution in executions:
            body = _canonical(execution.result.content)
            quoted = render_untrusted(execution.result.untrusted, budget=remaining)
            if quoted:
                body = f"{body}\n\n{quoted}"
                remaining = max(
                    0, remaining - _untrusted_used([ChatMessage(role="tool", content=quoted)])
                )
            messages.append(
                ChatMessage(role="tool", content=body, tool_call_id=execution.trace.call_id)
            )
        return request.model_copy(update={"messages": tuple(messages)})


__all__ = ["EXCEPTION_TASK", "SYSTEM_PROMPT", "DefaultAgentPromptBuilder"]
