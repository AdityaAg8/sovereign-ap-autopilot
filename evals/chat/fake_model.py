"""Deterministic scripted fake of the local OpenAI-compatible endpoint (llama-swap
`/v1/chat/completions`), served through an in-process `httpx` mock transport. It stands in for
the model only, for dry runs and unit tests of the chat harness: a keyword policy for the ops-chat
tools and a read-then-answer policy for AgentDojo. It never sees the golden calls."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Final

import httpx

from apx.agent.interfaces import REFUSAL_TEMPLATES

Body = dict[str, Any]
Policy = Callable[[Body], Body]
"""Maps an OpenAI chat-completions request body to one assistant `message` object."""

_INV_RE: Final[re.Pattern[str]] = re.compile(r"\binv-\d{4}\b")
_VENDOR_RE: Final[re.Pattern[str]] = re.compile(r"\bv-[a-z]+\b")
_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"\bINV-\d+\b")
_ARABIC_RE: Final[re.Pattern[str]] = re.compile(r"[؀-ۿ]")
_DATA_REFUSAL_RE: Final[re.Pattern[str]] = re.compile(
    r"unmasked|full.*iban|export|list all users|أظهر|صدّر", re.IGNORECASE
)
_ACTION_REFUSAL_RE: Final[re.Pattern[str]] = re.compile(
    r"^pay |release the payment|transfer|change the bank|send the|as verified|execute|"
    r"ادفع|وافق|غيّر|أرسل|حوّل",
    re.IGNORECASE,
)


def _tool_call(n: int, name: str, args: dict[str, Any]) -> Body:
    return {
        "id": f"call_{n}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _calls(n: int, *pairs: tuple[str, dict[str, Any]]) -> Body:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [_tool_call(n + i, name, args) for i, (name, args) in enumerate(pairs)],
    }


def _say(text: str) -> Body:
    return {"role": "assistant", "content": text}


def _text(message: Body) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def _tool_json(message: Body) -> Any:
    """The canonical JSON a tool message starts with (untrusted fences may follow it)."""
    try:
        value, _ = json.JSONDecoder().raw_decode(_text(message))
    except ValueError:
        return {}
    return value


# -- ops-chat keyword policy ---------------------------------------------------------------


def _first_calls(prompt: str) -> tuple[tuple[str, dict[str, Any]], ...] | str:
    """The first tool calls for a user prompt, or a final text when no tool applies."""
    low = prompt.lower()
    invs = _INV_RE.findall(prompt)
    vendors = _VENDOR_RE.findall(prompt)
    numbers = _NUMBER_RE.findall(prompt)
    arabic = bool(_ARABIC_RE.search(prompt))
    if _DATA_REFUSAL_RE.search(prompt):
        return REFUSAL_TEMPLATES["data_not_permitted"]
    if "ask an approver" in low or "request approval" in low:
        return (
            (("explain_decision", {"invoice_id": invs[0]}),)
            if invs
            else REFUSAL_TEMPLATES["unverifiable"]
        )
    if _ACTION_REFUSAL_RE.search(prompt) or ("approve" in low and "payment" in low):
        return REFUSAL_TEMPLATES["action_not_permitted"]
    if invs and re.search(r"draft|email|صغ|بريد", low):
        reason = "other"
        if re.search(r"purchase order|أمر الشراء", low):
            reason = "missing_po"
        elif re.search(r"trn|الضريبي", low):
            reason = "invalid_trn"
        elif re.search(r"amount|مبلغ", low):
            reason = "amount_mismatch"
        email = {
            "invoice_id": invs[0],
            "reason": reason,
            "details": "Please reply with the corrected information.",
            "language": "ar" if arabic else "en",
        }
        return (("draft_vendor_email", email),)
    if invs and re.search(r"why|explain|rule|fraud|لماذا|اشرح|الاحتيال", low):
        args: dict[str, Any] = {"invoice_id": invs[0]}
        if re.search(r"fraud|الاحتيال", low):
            args["topic"] = "fraud"
        return (("explain_decision", args),)
    if vendors:
        if re.search(r"list the invoices|فواتير", low):
            return (("search_invoices", {"vendor_id": vendors[0]}),)
        return (("vendor_history", {"vendor_id": vendors[0]}),)
    if invs:
        return (("get_invoice", {"invoice_id": invs[0]}),)
    if numbers:
        return (("search_invoices", {"invoice_number": numbers[0]}),)
    if re.search(r"blocked|المحظورة", low):
        return (("search_invoices", {"route": "block"}),)
    if "review" in low:
        return (("search_invoices", {"route": "review"}),)
    if "largest" in low:
        return (("search_invoices", {"sort": "total_desc", "limit": 5}),)
    return REFUSAL_TEMPLATES["unverifiable"]


def _cite(result: Any) -> list[str]:
    """Citation markers for the facts a tool result contains."""
    if not isinstance(result, dict):
        return []
    marks: list[str] = []
    if isinstance(result.get("events"), list):
        marks += [f"[audit:{e['audit_id']}]" for e in result["events"][:3]]
    if isinstance(result.get("row"), dict):
        marks.append(f"[invoice:{result['row']['invoice_id']}]")
        marks += [f"[policy:{r['rule']}]" for r in result.get("reasons", [])]
    if "vendor_id" in result and "accounts" in result:
        marks.append(f"[vendor:{result['vendor_id']}]")
    if isinstance(result.get("rows"), list):
        marks += [f"[invoice:{r['invoice_id']}]" for r in result["rows"][:3]]
    if "proposal_id" in result:
        marks.append(f"[proposal:{result['proposal_id']}]")
    return marks


def keyword_chat_policy(body: Body) -> Body:
    """A rule-based stand-in for the ops-chat agent, driven only by the conversation."""
    messages: list[Body] = body.get("messages", [])
    last_user = max(i for i, m in enumerate(messages) if m.get("role") == "user")
    prompt = _text(messages[last_user])
    after = messages[last_user + 1 :]
    tool_msgs = [m for m in after if m.get("role") == "tool"]
    if not tool_msgs:
        first = _first_calls(prompt)
        return _say(first) if isinstance(first, str) else _calls(1, *first)
    results = [_tool_json(m) for m in tool_msgs]
    last = results[-1]
    wants_approval = "ask an approver" in prompt.lower()
    done_approval = any(isinstance(r, dict) and r.get("kind") == "approve_payment" for r in results)
    if wants_approval and not done_approval and isinstance(last, dict) and last.get("events"):
        ids = [e["audit_id"] for e in last["events"]][:3]
        args = {
            "invoice_id": last.get("invoice_id"),
            "rationale": "Routed to review; the cited audit events explain why.",
            "cited_audit_ids": ids,
        }
        return _calls(len(tool_msgs) + 1, ("request_approval", args))
    if isinstance(last, dict) and "code" in last and "message" in last:
        return _say("I could not complete that: the tool refused the request.")
    marks = [m for r in results for m in _cite(r)]
    return _say("Here is what the records show. " + " ".join(dict.fromkeys(marks)))


# -- AgentDojo read-then-answer policy -------------------------------------------------------------


def read_then_answer_policy(body: Body) -> Body:
    """Calls the first offered tool that needs no arguments once, then answers with its output.
    Never follows instructions found in tool output (a 'stock', non-obedient stand-in)."""
    messages: list[Body] = body.get("messages", [])
    tools: list[Body] = body.get("tools") or []
    if any(m.get("role") == "tool" for m in messages):
        tool_texts = [_text(m) for m in messages if m.get("role") == "tool"]
        return _say(f"Done. Tool output: {tool_texts[-1][:200]}")
    for tool in tools:
        fn = tool.get("function", {})
        if not (fn.get("parameters") or {}).get("required"):
            return _calls(1, (str(fn.get("name")), {}))
    return _say("I cannot help with that.")


# -- transport ---------------------------------------------------------------------------


class FakeEndpoint:
    """An in-process `/v1/chat/completions` that answers with `policy`. Records request bodies."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self.requests: list[Body] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or request.url.path != "/v1/chat/completions":
            return httpx.Response(404, json={"error": f"no route {request.url.path}"})
        body = json.loads(request.content)
        self.requests.append(body)
        message = self._policy(body)
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        return httpx.Response(
            200,
            json={
                "model": body.get("model", "fake"),
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            },
        )

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)


__all__ = ["FakeEndpoint", "keyword_chat_policy", "read_then_answer_policy"]
