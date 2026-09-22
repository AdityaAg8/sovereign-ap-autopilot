"""Tests for DefaultProposalSink (P5): agent, chat and MCP proposals stay inside
AGENT_PROPOSAL_KINDS and can never be `change_bank`; every accepted proposal is audited before
being persisted, and a proposal never executes itself (PRD §6.3 P5, ADR-0008 K3, M6.5)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.common.errors import PolicyDeniedError
from apx.policy.interfaces import AGENT_PROPOSAL_KINDS, Proposal
from apx.policy.proposals import DefaultProposalSink

_AT = datetime(2026, 1, 1, tzinfo=UTC)
_ALL_KINDS = ("edit_field", "route", "request_info", "approve_payment", "change_bank")
_NON_HUMAN_ORIGINS = ("agent", "chat", "mcp")


class FakeAuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append(self, event: AuditEvent) -> AuditEntry:
        self.events.append(event)
        return AuditEntry(
            id=len(self.events),
            ts=_AT,
            actor=event.actor,
            action=event.action,
            entity=event.entity,
            payload=event.payload,
            prev_hash="0" * 64,
            hash="1" * 64,
        )


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[Proposal] = []

    async def save(self, proposal: Proposal) -> None:
        self.saved.append(proposal)


def _proposal(kind: str, created_by: str, status: str = "pending") -> Proposal:
    return Proposal(
        id=f"{kind}-{created_by}",
        invoice_id=None,
        kind=kind,  # type: ignore[arg-type]
        payload={},
        created_by=created_by,  # type: ignore[arg-type]
        created_at=_AT,
        status=status,  # type: ignore[arg-type]
    )


@given(origin=st.sampled_from(_NON_HUMAN_ORIGINS))
def test_non_human_origin_can_never_propose_change_bank(origin: str) -> None:
    async def run() -> None:
        audit, store = FakeAuditLog(), FakeStore()
        sink = DefaultProposalSink(store, audit)
        with pytest.raises(PolicyDeniedError):
            await sink.submit(_proposal("change_bank", origin))
        assert store.saved == []

    asyncio.run(run())


@given(
    origin=st.sampled_from(_NON_HUMAN_ORIGINS),
    kind=st.sampled_from(sorted(AGENT_PROPOSAL_KINDS)),
)
def test_non_human_origin_may_propose_agent_kinds(origin: str, kind: str) -> None:
    async def run() -> None:
        audit, store = FakeAuditLog(), FakeStore()
        sink = DefaultProposalSink(store, audit)
        result = await sink.submit(_proposal(kind, origin))
        assert result.status == "pending"
        assert store.saved == [result]
        assert len(audit.events) == 1

    asyncio.run(run())


def test_a_human_may_propose_change_bank() -> None:
    """P5 restricts agent/chat/mcp only; a human-authored proposal is not an agent action."""

    async def run() -> None:
        audit, store = FakeAuditLog(), FakeStore()
        sink = DefaultProposalSink(store, audit)
        result = await sink.submit(_proposal("change_bank", "human-user-1"))
        assert result.status == "pending"

    asyncio.run(run())


def test_submitting_a_non_pending_proposal_is_refused() -> None:
    async def run() -> None:
        audit, store = FakeAuditLog(), FakeStore()
        sink = DefaultProposalSink(store, audit)
        executed = _proposal("edit_field", "agent", status="executed")
        with pytest.raises(PolicyDeniedError):
            await sink.submit(executed)
        assert store.saved == []

    asyncio.run(run())


def test_a_submitted_proposal_never_executes_itself() -> None:
    """P5: ProposalSink has no execute method; submitting only ever leaves status pending."""

    async def run() -> None:
        audit, store = FakeAuditLog(), FakeStore()
        sink = DefaultProposalSink(store, audit)
        result = await sink.submit(_proposal("edit_field", "agent"))
        assert result.status == "pending"

    asyncio.run(run())
    assert not hasattr(DefaultProposalSink, "execute")
