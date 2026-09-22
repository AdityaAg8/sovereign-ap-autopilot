"""Default ProposalSink: the only write dependency apx.agent, chat and MCP receive. Enforces P5
(agent/chat/mcp proposals stay inside AGENT_PROPOSAL_KINDS) and audits every accepted proposal
before it is persisted (ADR-0008 K3)."""

from __future__ import annotations

from typing import Protocol

from apx.audit.interfaces import AuditEvent, AuditLog
from apx.common.errors import PolicyDeniedError
from apx.policy.interfaces import AGENT_PROPOSAL_KINDS, Proposal

_NON_HUMAN_ORIGINS: frozenset[str] = frozenset({"agent", "chat", "mcp"})


class ProposalStore(Protocol):
    """Where an accepted proposal is persisted. Kept separate from `DefaultProposalSink` so the
    sink's own policy logic (P5) can be tested without a real database."""

    async def save(self, proposal: Proposal) -> None: ...


class DefaultProposalSink:
    """The `ProposalSink` implementation. Never executes a proposal: `submit` only validates,
    audits and persists it as `status="pending"` (P5)."""

    def __init__(self, store: ProposalStore, audit: AuditLog) -> None:
        self._store = store
        self._audit = audit

    async def submit(self, proposal: Proposal) -> Proposal:
        """Persist a pending proposal and audit it. Raises PolicyDeniedError (rule P5) if the
        origin is agent/chat/mcp and the kind is not in AGENT_PROPOSAL_KINDS, or if the status
        is not pending. Never executes anything."""
        if proposal.status != "pending":
            raise PolicyDeniedError(
                f"proposal {proposal.id} must be submitted as pending, not {proposal.status!r}"
            )
        if proposal.created_by in _NON_HUMAN_ORIGINS and proposal.kind not in AGENT_PROPOSAL_KINDS:
            raise PolicyDeniedError(f"{proposal.created_by} may not propose {proposal.kind!r} (P5)")

        await self._audit.append(
            AuditEvent(
                actor=str(proposal.created_by),
                action="proposal.submit",
                entity=f"proposal:{proposal.id}",
                payload={"kind": proposal.kind, "invoice_id": proposal.invoice_id},
            )
        )
        await self._store.save(proposal)
        return proposal
