"""`PgToolDispatcher`: the JSON boundary between a tool-calling model and `AgentToolbox`. Looks
up the contract, authorises the caller, validates arguments into the params model, runs the
toolbox method, enforces the `TOOL_RESULT_MAX_BYTES` cap and turns every `AgentToolError` or
`PolicyDeniedError` into a `ToolRefusal` the model can recover from; anything else propagates
(PRD §10 M9.2, ADR-0010 D1, D3, D7)."""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue, ValidationError

from apx.agent.contracts import authorize_tool, tool_spec
from apx.agent.interfaces import (
    MCP_MAX_CALLS_PER_MINUTE,
    TOOL_CONTRACTS,
    TOOL_RESULT_MAX_BYTES,
    AgentToolbox,
    AgentToolError,
    Citation,
    ToolCaller,
    ToolContract,
    ToolExecution,
    ToolName,
    ToolRateLimitError,
    ToolRefusal,
    ToolRefusalCode,
    ToolResult,
    ToolRoleError,
    ToolScopeError,
    ToolTrace,
)
from apx.common.errors import PolicyDeniedError
from apx.llm.interfaces import ToolCall, ToolSpec

_RATE_WINDOW_SECONDS: Final[float] = 60.0


def _now() -> float:
    return time.monotonic()


def _canonical_json_size(content: Mapping[str, object]) -> int:
    return len(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _refs_for(tool: ToolName, result: object) -> tuple[Citation, ...]:
    """Every citable ref `result` contains (ADR-0010 D6): every audit id for `explain_decision`;
    the invoice id, policy rules, proposal ids and untrusted sources for `get_invoice`; the
    vendor id, invoice ids and bank-change audit ids for `vendor_history`. Other tools cite what
    they naturally produced (found invoices, or the proposal just created)."""
    refs: list[Citation] = []
    untrusted = getattr(result, "untrusted", ())
    for block in untrusted:
        refs.append(Citation(source="document", ref=block.source))

    if tool == "explain_decision":
        for event in result.events:  # type: ignore[attr-defined]
            refs.append(Citation(source="audit", ref=str(event.audit_id)))
        refs.append(Citation(source="invoice", ref=str(result.invoice_id)))  # type: ignore[attr-defined]
    elif tool == "get_invoice":
        row = result.row  # type: ignore[attr-defined]
        refs.append(Citation(source="invoice", ref=str(row.invoice_id)))
        for reason in result.reasons:  # type: ignore[attr-defined]
            refs.append(Citation(source="policy", ref=reason.rule))
        for proposal in result.pending_proposals:  # type: ignore[attr-defined]
            refs.append(Citation(source="proposal", ref=proposal.proposal_id))
    elif tool == "vendor_history":
        refs.append(Citation(source="vendor", ref=str(result.vendor_id)))  # type: ignore[attr-defined]
        for invoice in result.invoices:  # type: ignore[attr-defined]
            refs.append(Citation(source="invoice", ref=str(invoice.invoice_id)))
        for change in result.bank_changes:  # type: ignore[attr-defined]
            if change.audit_id is not None:
                refs.append(Citation(source="audit", ref=str(change.audit_id)))
    elif tool == "search_invoices":
        for row in result.rows:  # type: ignore[attr-defined]
            refs.append(Citation(source="invoice", ref=str(row.invoice_id)))
    elif tool in ("request_approval", "draft_vendor_email"):
        refs.append(Citation(source="proposal", ref=str(result.proposal_id)))  # type: ignore[attr-defined]
        refs.append(Citation(source="invoice", ref=str(result.invoice_id)))  # type: ignore[attr-defined]

    # De-duplicate while keeping first-seen order (a result can repeat a ref, e.g. the same
    # invoice id from both `row` and a proposal).
    seen: set[tuple[str, str]] = set()
    deduped: list[Citation] = []
    for ref in refs:
        key = (ref.source, ref.ref)
        if key not in seen:
            seen.add(key)
            deduped.append(ref)
    return tuple(deduped)


class PgToolDispatcher:
    """The `ToolDispatcher` implementation. Its only dependencies are an `AgentToolbox` and the
    `TOOL_CONTRACTS` table (ADR-0010 D2 K3: no `AuditLog`, `Grant`, `PolicyGuard` or payment
    writer). MCP-read and refusal auditing (D7, D8) is wired by the MCP server entrypoint
    (M9.4), which sits above this class and already holds its own `AuditLog`."""

    def __init__(
        self,
        toolbox: AgentToolbox,
        contracts: Mapping[ToolName, ToolContract] = TOOL_CONTRACTS,
    ) -> None:
        self._toolbox = toolbox
        self._contracts = contracts
        self._mcp_calls: dict[str, deque[float]] = {}

    def specs(self, caller: ToolCaller) -> tuple[ToolSpec, ...]:
        specs: list[ToolSpec] = []
        for contract in self._contracts.values():
            if caller.origin not in contract.origins:
                continue
            if caller.origin in ("chat", "mcp") and (
                caller.user is None or not (caller.user.roles & contract.roles)
            ):
                continue
            specs.append(tool_spec(contract))
        return tuple(specs)

    async def dispatch(self, call: ToolCall, caller: ToolCaller) -> ToolExecution:
        start = _now()
        contract = self._contracts.get(call.name)  # type: ignore[call-overload]
        if contract is None:
            return _refusal(call, None, "unknown_tool", start)

        try:
            authorize_tool(contract, caller)
        except ToolRoleError:
            return _refusal(call, contract.name, "role_denied", start)
        except ToolScopeError:
            return _refusal(call, contract.name, "out_of_scope", start)
        # ConfigError (a malformed ToolCaller) propagates: that is a runtime bug, not a refusal.

        if caller.origin == "mcp":
            try:
                self._check_rate_limit(caller.session_id)
            except ToolRateLimitError:
                return _refusal(call, contract.name, "rate_limited", start)

        try:
            params = contract.params.model_validate(call.arguments)
        except ValidationError:
            return _refusal(call, contract.name, "invalid_arguments", start)

        method = getattr(self._toolbox, contract.name)
        try:
            result = await method(params, caller)
        except AgentToolError as exc:
            return _refusal(call, contract.name, exc.code, start, params.model_dump(mode="json"))
        except PolicyDeniedError:
            return _refusal(
                call, contract.name, "policy_denied", start, params.model_dump(mode="json")
            )

        content = result.model_dump(mode="json", exclude={"untrusted"})
        if _canonical_json_size(content) > TOOL_RESULT_MAX_BYTES:
            return _refusal(
                call, contract.name, "result_too_large", start, params.model_dump(mode="json")
            )

        trace = ToolTrace(
            call_id=call.id,
            tool=contract.name,
            arguments=params.model_dump(mode="json"),
            outcome="ok",
            latency_ms=_elapsed_ms(start),
            refs=_refs_for(contract.name, result),
            proposal_id=getattr(result, "proposal_id", None),
            truncated=result.truncated,
        )
        return ToolExecution(
            result=ToolResult(content=content, untrusted=result.untrusted), trace=trace
        )

    def _check_rate_limit(self, session_id: str) -> None:
        now = _now()
        calls = self._mcp_calls.setdefault(session_id, deque())
        while calls and now - calls[0] > _RATE_WINDOW_SECONDS:
            calls.popleft()
        if len(calls) >= MCP_MAX_CALLS_PER_MINUTE:
            raise ToolRateLimitError(f"more than {MCP_MAX_CALLS_PER_MINUTE} calls in one minute")
        calls.append(now)


def _elapsed_ms(start: float) -> float:
    return (_now() - start) * 1000.0


def _refusal(
    call: ToolCall,
    tool: ToolName | None,
    code: ToolRefusalCode,
    start: float,
    arguments: dict[str, JsonValue] | None = None,
) -> ToolExecution:
    message = _REFUSAL_MESSAGES[code]
    trace = ToolTrace(
        call_id=call.id,
        tool=tool,
        arguments=arguments if arguments is not None else dict(call.arguments),
        outcome="refused",
        latency_ms=_elapsed_ms(start),
        refusal=code,
    )
    refusal_content = ToolRefusal(tool=tool, code=code, message=message).model_dump(mode="json")
    return ToolExecution(result=ToolResult(content=refusal_content), trace=trace)


_REFUSAL_MESSAGES: Final[Mapping[ToolRefusalCode, str]] = {
    "unknown_tool": "There is no tool by that name.",
    "invalid_arguments": "Those arguments do not fit this tool; check the required fields.",
    "role_denied": "Your role cannot use this tool.",
    "out_of_scope": "That is outside what this session can look at.",
    "not_found": "No matching record was found.",
    "precondition_failed": "This tool cannot run for the record in its current state.",
    "rate_limited": "Too many tool calls; wait a moment and try again.",
    "policy_denied": "Policy refused this request.",
    "result_too_large": "The result was too large to return; narrow the query.",
}


__all__ = ["PgToolDispatcher"]
