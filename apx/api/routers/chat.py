"""Ops chat and audit-trail endpoints (PRD §10 M9.3). `POST /chat` streams server-sent events: one
`step` per tool call while the agent works, then a `reply` whose citations carry links to the
cited records (audit citations resolve to their invoice's audit tab). `GET /invoices/{id}/audit`
returns that tab: the invoice's hash-chained audit events with only scalar facts exposed."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from datetime import datetime
from functools import lru_cache
from typing import Annotated, Any, Final, Literal

from fastapi import APIRouter, Depends
from pydantic import Field
from sqlalchemy import select
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from apx.agent.interfaces import (
    AUDIT_FACT_KEYS,
    AgentConfig,
    AgentReply,
    ChatTurn,
    Citation,
    OpsChat,
    ToolArgumentError,
    ToolTrace,
)
from apx.agent.runner import REVIEW_NOTE_ACTION
from apx.api.deps import CurrentUser
from apx.common.errors import ApxError, ModelServingError
from apx.common.types import Frozen, Stage
from apx.db.models import AuditLog
from apx.db.session import get_session

router = APIRouter(tags=["chat"])

CHAT_MODEL_ENV: Final[str] = "APX_CHAT_MODEL"
DEFAULT_CHAT_MODEL: Final[str] = "ops-chat"
AUDIT_TAB_MAX_EVENTS: Final[int] = 200

TOOL_LABELS: Final[dict[str, str]] = {
    "search_invoices": "Searching invoices",
    "get_invoice": "Reading the invoice",
    "explain_decision": "Checking the audit trail",
    "vendor_history": "Looking at the vendor's history",
    "request_approval": "Drafting an approval request",
    "draft_vendor_email": "Drafting a vendor email",
}
"""Plain-language step labels for the chat panel; unknown tools fall back to a generic label."""


class ChatTurnIn(Frozen):
    role: Literal["user", "assistant"]
    text: Annotated[str, Field(min_length=1, max_length=4000)]


class ChatRequest(Frozen):
    messages: Annotated[tuple[ChatTurnIn, ...], Field(min_length=1, max_length=40)]


class AuditEventOut(Frozen):
    id: int
    ts: datetime
    actor: str
    action: str
    facts: dict[str, str | int | float | bool | None]
    note: str | None = None
    """The review-note text, for `agent.review_note` events only."""


@lru_cache(maxsize=1)
def _default_chat() -> OpsChat:
    """Production wiring: Extractor A behind the `ops-chat` alias, recorded in llm_calls; tools
    over the read-only agent role; proposals through the audited sink."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from apx.agent.dispatch import PgToolDispatcher
    from apx.agent.prompts import DefaultAgentPromptBuilder
    from apx.agent.runner import LlmOpsChat, ToolLoop
    from apx.agent.sql import PgSqlReader, get_agent_database_url
    from apx.agent.toolbox import PgAgentToolbox
    from apx.audit.chain import PgAuditLog
    from apx.llm.client import LlamaSwapClient
    from apx.observability.llm_calls import PgLlmCallRecorder, RecordingLlmClient
    from apx.policy.proposals import DefaultProposalSink, PgProposalStore

    model = os.environ.get(CHAT_MODEL_ENV, DEFAULT_CHAT_MODEL)
    audit = PgAuditLog()
    reader = PgSqlReader(create_async_engine(get_agent_database_url()))
    dispatcher = PgToolDispatcher(
        PgAgentToolbox(reader, DefaultProposalSink(PgProposalStore(), audit))
    )
    llm = RecordingLlmClient(LlamaSwapClient(), PgLlmCallRecorder(), Stage.AGENT)
    prompts = DefaultAgentPromptBuilder()
    loop = ToolLoop(llm, dispatcher, prompts, AgentConfig(model=model))
    return LlmOpsChat(loop, dispatcher, prompts, audit, model)


def get_ops_chat() -> OpsChat:
    return _default_chat()


OpsChatDep = Annotated[OpsChat, Depends(get_ops_chat)]


async def _audit_entities(ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    async with get_session() as session:
        rows = await session.execute(
            select(AuditLog.id, AuditLog.entity).where(AuditLog.id.in_(ids))
        )
        return {row.id: row.entity for row in rows}


def _href(citation: Citation, entities: dict[int, str]) -> str | None:
    if citation.source == "invoice":
        return f"/invoices/{citation.ref}"
    if citation.source == "vendor":
        return f"/vendors/{citation.ref}"
    if citation.source == "audit" and citation.ref.isdigit():
        entity = entities.get(int(citation.ref), "")
        if entity.startswith("invoice:"):
            return f"/invoices/{entity.removeprefix('invoice:')}?tab=audit&event={citation.ref}"
    return None


async def reply_payload(reply: AgentReply) -> dict[str, Any]:
    audit_ids = [int(c.ref) for c in reply.citations if c.source == "audit" and c.ref.isdigit()]
    entities = await _audit_entities(audit_ids)
    return {
        "text": reply.text,
        "refusal": reply.refusal,
        "citations": [
            {"source": c.source, "ref": c.ref, "href": _href(c, entities)} for c in reply.citations
        ],
        "proposals": [{"id": p.id, "kind": p.kind} for p in reply.proposals],
    }


def _step_payload(trace: ToolTrace) -> dict[str, Any]:
    return {
        "tool": trace.tool,
        "label": TOOL_LABELS.get(trace.tool or "", "Using a tool"),
        "outcome": trace.outcome,
    }


def _event(name: str, data: dict[str, Any]) -> ServerSentEvent:
    return ServerSentEvent(event=name, data=json.dumps(data, sort_keys=True))


@router.post("/chat")
async def chat(body: ChatRequest, user: CurrentUser, ops_chat: OpsChatDep) -> EventSourceResponse:
    history = [ChatTurn(role=turn.role, text=turn.text) for turn in body.messages]
    queue: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()

    async def on_step(trace: ToolTrace) -> None:
        await queue.put(_event("step", _step_payload(trace)))

    async def run() -> None:
        try:
            reply = await ops_chat.respond(history, user, on_step)
            await queue.put(_event("reply", await reply_payload(reply)))
        except ApxError as exc:
            # Typed failures (model server down, oversized message) end the stream with a
            # message the panel can show; nothing is retried or silently swallowed.
            await queue.put(_event("error", {"message": _error_message(exc)}))
        finally:
            await queue.put(None)

    async def stream() -> AsyncIterator[ServerSentEvent]:
        task = asyncio.create_task(run())
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(stream())


def _error_message(exc: ApxError) -> str:
    if isinstance(exc, ModelServingError):
        return "The assistant model is not reachable right now. Try again in a minute."
    if isinstance(exc, ToolArgumentError):
        return str(exc)
    return "The assistant could not answer that. Try a narrower question."


@router.get("/invoices/{invoice_id}/audit", response_model=list[AuditEventOut])
async def invoice_audit(invoice_id: str, _user: CurrentUser) -> list[AuditEventOut]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(AuditLog)
                .where(AuditLog.entity == f"invoice:{invoice_id}")
                .order_by(AuditLog.id.desc())
                .limit(AUDIT_TAB_MAX_EVENTS)
            )
        ).scalars()
        events: list[AuditEventOut] = []
        for row in rows:
            payload = row.payload_json or {}
            facts = {
                key: value
                for key, value in payload.items()
                if key in AUDIT_FACT_KEYS
                and (value is None or isinstance(value, str | int | float | bool))
            }
            note = payload.get("text") if row.action == REVIEW_NOTE_ACTION else None
            events.append(
                AuditEventOut(
                    id=row.id,
                    ts=row.ts,
                    actor=row.actor,
                    action=row.action,
                    facts=facts,
                    note=note if isinstance(note, str) else None,
                )
            )
        return events
