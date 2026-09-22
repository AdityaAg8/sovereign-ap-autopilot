"""Interfaces for the REST + SSE API: JWT claims, request/response bodies and the service
boundary the FastAPI routes call (PRD §3, §10 M5.3)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Literal, Protocol

from pydantic import JsonValue

from apx.common.master import User
from apx.common.types import FieldValue, Frozen, InvoiceId, Role, Route, UserId
from apx.policy.interfaces import Proposal
from apx.workflows.interfaces import InvoiceStatus


class TokenClaims(Frozen):
    sub: UserId
    roles: frozenset[Role]
    exp: datetime


class InvoiceSummary(Frozen):
    id: InvoiceId
    status: InvoiceStatus
    route: Route | None
    supplier_name: str | None
    total: str | None
    received_at: datetime


class InvoiceDetail(Frozen):
    summary: InvoiceSummary
    fields: tuple[FieldValue, ...]
    accepted: dict[str, bool]
    reasons: tuple[str, ...]
    proposals: tuple[Proposal, ...]


class FieldEdit(Frozen):
    field: str
    value: str
    reason: str


class DecisionRequest(Frozen):
    action: Literal["approve", "reject", "send_to_review", "execute_proposal"]
    proposal_id: str | None = None
    comment: str = ""


class SseEvent(Frozen):
    event: str
    data: dict[str, JsonValue]


class TokenVerifier(Protocol):
    def verify(self, token: str) -> TokenClaims:
        """Verify a JWT. Raises PolicyDeniedError on invalid or expired tokens."""
        ...


class InvoiceService(Protocol):
    async def list(self, status: InvoiceStatus | None, limit: int) -> Sequence[InvoiceSummary]: ...
    async def get(self, invoice_id: InvoiceId) -> InvoiceDetail: ...
    async def edit_field(self, invoice_id: InvoiceId, edit: FieldEdit, user: User) -> None: ...
    async def decide(self, invoice_id: InvoiceId, request: DecisionRequest, user: User) -> None:
        """Signal the workflow after policy authorisation. Raises PolicyDeniedError."""
        ...

    def events(self, invoice_id: InvoiceId | None) -> AsyncIterator[SseEvent]: ...
