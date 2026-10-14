"""Unit tests for the `review_note` workflow activity: a grounded note is stored as
an `agent.review_note` audit event on the invoice; a refused note or an unreachable model is
recorded as its own audit event and never raises, so the human review is never held up."""

from __future__ import annotations

from typing import Any, cast

import pytest

from apx.agent.interfaces import AgentContext, AgentReply, Citation
from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.common.errors import ModelServingError
from apx.common.types import Route
from apx.policy.interfaces import RoutingDecision
from apx.workflows.activities import DefaultInvoiceActivities, InvoiceActivityDeps
from tests.unit._policy_factories import make_invoice

_UNUSED: Any = None
_ROUTING = RoutingDecision(route=Route.REVIEW, reasons=())


class _Audit:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append(self, event: AuditEvent) -> AuditEntry:
        self.events.append(event)
        return AuditEntry.model_validate(
            {
                **event.model_dump(),
                "id": len(self.events),
                "ts": "2026-09-23T00:00:00Z",
                "prev_hash": "0" * 64,
                "hash": f"{len(self.events):064x}",
            }
        )


class _Agent:
    def __init__(self, reply: AgentReply | None = None, error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error
        self.contexts: list[AgentContext] = []

    async def handle(self, context: AgentContext) -> AgentReply:
        self.contexts.append(context)
        if self._error is not None:
            raise self._error
        assert self._reply is not None
        return self._reply


def _activities(agent: _Agent | None, audit: _Audit) -> DefaultInvoiceActivities:
    deps = InvoiceActivityDeps(
        documents=_UNUSED,
        scanner=_UNUSED,
        parser=_UNUSED,
        parsed=_UNUSED,
        extraction=_UNUSED,
        invoices=_UNUSED,
        feature_builder=_UNUSED,
        trust_gate=_UNUSED,
        validator=_UNUSED,
        fraud_engine=_UNUSED,
        vendors=_UNUSED,
        policy_engine=_UNUSED,
        ubl_builder=_UNUSED,
        schematron=_UNUSED,
        asp_client=_UNUSED,
        archive=_UNUSED,
        audit=audit,
        records=_UNUSED,
        session_factory=_UNUSED,
        idempotency=_UNUSED,
        exception_agent=cast(Any, agent),
    )
    return DefaultInvoiceActivities(deps)


async def test_grounded_note_is_stored_on_the_invoice() -> None:
    reply = AgentReply(
        text="Fraud score over its limit [audit:7].",
        citations=(Citation(source="audit", ref="7"),),
        proposals=(),
    )
    agent, audit = _Agent(reply=reply), _Audit()
    stored = await _activities(agent, audit).review_note(make_invoice(), _ROUTING)
    assert stored is True
    assert agent.contexts[0].invoice_id == "inv-1"
    assert agent.contexts[0].state["route"] == "review"
    assert agent.contexts[0].quoted_document == ()
    assert [(e.action, e.entity) for e in audit.events] == [("agent.review_note", "invoice:inv-1")]
    assert audit.events[0].payload["text"] == reply.text


@pytest.mark.parametrize(
    ("agent", "action", "code"),
    [
        (
            _Agent(reply=AgentReply(text="x", citations=(), proposals=(), refusal="unverifiable")),
            "agent.review_note_refused",
            "unverifiable",
        ),
        (
            _Agent(error=ModelServingError("llama-swap down")),
            "agent.review_note_unavailable",
            "ModelServingError",
        ),
    ],
)
async def test_no_note_is_recorded_as_its_own_event_and_never_raises(
    agent: _Agent, action: str, code: str
) -> None:
    audit = _Audit()
    stored = await _activities(agent, audit).review_note(make_invoice(), _ROUTING)
    assert stored is False
    assert [(e.action, e.payload) for e in audit.events] == [(action, {"code": code})]


async def test_without_an_agent_the_activity_is_a_no_op() -> None:
    audit = _Audit()
    assert await _activities(None, audit).review_note(make_invoice(), _ROUTING) is False
    assert audit.events == []
