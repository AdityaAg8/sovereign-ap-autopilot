"""Shared builders for apx.agent tests: a fake `SqlReader` keyed by the exact `SQL_*` statement
constant (a "recording reader with fixture rows"), a spy `ProposalSink`, and minimal invoice/
vendor/audit row fixtures shaped exactly like `v_agent_invoices`/`v_agent_vendor_accounts`/
`v_agent_audit` columns. Not collected by pytest (no test_ prefix)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from apx.agent.interfaces import SqlParam, SqlStatement, ToolCaller
from apx.common.master import User
from apx.common.types import InvoiceId, Role, UserId, VendorId
from apx.policy.interfaces import Proposal


class FakeReader:
    """A `SqlReader` returning canned rows per registered `SQL_*` statement, and recording every
    call it received (statement identity, bound params, requested limit)."""

    def __init__(self, fixtures: Mapping[SqlStatement, list[dict[str, Any]]] | None = None) -> None:
        self.fixtures: dict[SqlStatement, list[dict[str, Any]]] = dict(fixtures or {})
        self.calls: list[tuple[SqlStatement, dict[str, SqlParam], int]] = []

    async def fetch_all(
        self, statement: SqlStatement, params: Mapping[str, SqlParam], limit: int
    ) -> list[dict[str, Any]]:
        self.calls.append((statement, dict(params), limit))
        return list(self.fixtures.get(statement, []))[:limit]


class SpySink:
    """A `ProposalSink` that accepts every submission (no P5 enforcement -- that belongs to
    `apx.policy.proposals.DefaultProposalSink`, not the toolbox) and records what it received,
    so a test can assert exactly one proposal was written and nothing else changed."""

    def __init__(self) -> None:
        self.submitted: list[Proposal] = []

    async def submit(self, proposal: Proposal) -> Proposal:
        self.submitted.append(proposal)
        return proposal


class DenyingSink:
    """A `ProposalSink` that must never be called; used by read-tool tests to prove they never
    write (ADR-0010 D2 K3)."""

    async def submit(self, proposal: Proposal) -> Proposal:
        raise AssertionError(f"a read tool must never submit a proposal, got {proposal!r}")


def reviewer(user_id: str = "u1") -> User:
    return User(id=UserId(user_id), display_name="Reviewer", roles=frozenset({Role.REVIEWER}))


def approver(user_id: str = "u2") -> User:
    return User(id=UserId(user_id), display_name="Approver", roles=frozenset({Role.APPROVER}))


def admin(user_id: str = "u3") -> User:
    return User(id=UserId(user_id), display_name="Admin", roles=frozenset({Role.ADMIN}))


def chat_caller(user: User, session_id: str = "s" * 8) -> ToolCaller:
    return ToolCaller(origin="chat", user=user, session_id=session_id)


def mcp_caller(user: User, session_id: str = "m" * 8) -> ToolCaller:
    return ToolCaller(origin="mcp", user=user, session_id=session_id)


def agent_caller(scope_invoice_id: str, session_id: str = "a" * 8) -> ToolCaller:
    return ToolCaller(
        origin="agent",
        user=None,
        session_id=session_id,
        scope_invoice_id=InvoiceId(scope_invoice_id),
    )


def invoice_row(
    *,
    invoice_id: str = "inv-1",
    status: str = "awaiting_review",
    route: str | None = "review",
    vendor_id: str | None = "v1",
    vendor_name: str | None = "Acme Supplies LLC",
    invoice_number_raw: str | None = "INV-1001",
    po_number_raw: str | None = None,
    currency: str = "AED",
    subtotal: Decimal = Decimal("100.00"),
    vat_total: Decimal = Decimal("5.00"),
    total: Decimal = Decimal("105.00"),
    payee_masked: str | None = "****1234",
    document_iban_matches_payee: bool | None = True,
    fields_json: list[dict[str, Any]] | None = None,
    proposals_json: list[dict[str, Any]] | None = None,
    reasons_json: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "invoice_id": InvoiceId(invoice_id),
        "status": status,
        "route": route,
        "vendor_id": VendorId(vendor_id) if vendor_id else None,
        "vendor_name": vendor_name,
        "invoice_number_raw": invoice_number_raw,
        "issue_date": date(2026, 1, 5),
        "due_date": date(2026, 2, 4),
        "po_number_raw": po_number_raw,
        "currency": currency,
        "subtotal": subtotal,
        "vat_total": vat_total,
        "total": total,
        "received_at": datetime(2026, 1, 5, 9, 0, tzinfo=UTC),
        "payee_masked": payee_masked,
        "document_iban_matches_payee": document_iban_matches_payee,
        "fields_json": fields_json if fields_json is not None else [],
        "proposals_json": proposals_json if proposals_json is not None else [],
        "reasons_json": reasons_json if reasons_json is not None else [],
    }


def audit_event_row(
    *,
    audit_id: int = 501,
    action: str = "policy.route",
    actor: str = "system",
    payload_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "audit_id": audit_id,
        "ts": datetime(2026, 1, 5, 9, 1, tzinfo=UTC),
        "actor": actor,
        "action": action,
        "payload_json": payload_json or {"rule": "fraud_score", "route": "review", "score": 0.91},
    }
