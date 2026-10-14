"""Interfaces for the REST + SSE API: JWT claims, request/response bodies and the service
boundary the FastAPI routes call."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal, Protocol

from pydantic import Field, JsonValue

from apx.common.master import User
from apx.common.types import BBox, Frozen, InvoiceId, Role, Route, Sha256Hex, UserId, VendorId
from apx.db.enums import FieldSource
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
    vendor_id: VendorId | None
    supplier_name: str | None
    total: str | None
    received_at: datetime


class InvoiceFieldView(Frozen):
    """One field as shown to a reviewer: value, confidence, grounding box and provenance.
    `value` is masked for `name == "iban"` unless the caller is an admin
    (docs/security.md B5, ADR-0008 T2: IBANs are always shown masked to the last 4 characters
    except to the role that manages vendor bank master data)."""

    name: str
    value: str | None
    confidence: float | None
    boxes: tuple[BBox, ...]
    source: FieldSource


class FlagView(Frozen):
    """A stored fraud flag as review shows it: which rule, how severe, and why in plain words.
    Evidence stays in the audit trail, not the review card."""

    id: int
    rule_id: str
    typology: str
    severity: Literal["low", "medium", "high"]
    score: float
    explanation: str


class InvoiceDetail(Frozen):
    summary: InvoiceSummary
    fields: tuple[InvoiceFieldView, ...]
    accepted: dict[str, bool]
    reasons: tuple[str, ...]
    proposals: tuple[Proposal, ...]
    flags: tuple[FlagView, ...] = ()
    error: str | None = None
    """Why the workflow failed, when it failed before its invoice row existed (status FAILED,
    no fields); None otherwise."""


class InvoicePage(Frozen):
    """One page of `InvoiceSummary` rows plus the total match count, for list/filter/paginate."""

    items: tuple[InvoiceSummary, ...]
    total: int
    limit: int
    offset: int


class VendorBankDetailView(Frozen):
    """One `bank_details` row as shown on the vendor view: the IBAN is always
    masked to its last 4 characters here regardless of role -- unlike the admin exception on
    `InvoiceFieldView.value` above, this is the vendor master-data screen, not the one-off
    document-verification view ADR-0008 T2 carves the exception for (ADR-0008 line 85: "always
    shown masked to the last 4 characters"). `verified` is read straight from the G-3
    `Computed` column: it can only be true if `verification_id` names a real
    `bank_verifications` row."""

    id: int
    iban_masked: str
    verified: bool
    verified_at: datetime | None
    source: str
    is_current: bool
    changed_at: datetime


class VendorSummary(Frozen):
    id: VendorId
    name: str
    trn: str | None
    verified: bool
    """True if the vendor has at least one current, G-3-verified bank account."""


class VendorDetail(Frozen):
    id: VendorId
    name: str
    trn: str | None
    bank_details: tuple[VendorBankDetailView, ...]
    recent_invoices: tuple[InvoiceSummary, ...]


class VendorPage(Frozen):
    """One page of `VendorSummary` rows plus the total match count, for list/search/paginate."""

    items: tuple[VendorSummary, ...]
    total: int
    limit: int
    offset: int


class InvoiceDocument(Frozen):
    """A stored source document ready to be served by `GET /invoices/{id}/document`.
    `media_type` is sniffed from the stored bytes' magic number, never trusted from
    a stale database column, so the `Content-Type` the client sees always matches what is
    actually inside the response body -- required for `X-Content-Type-Options: nosniff` (set by
    `apx.api.app`) to be safe rather than misleading. `filename` is built only from the
    content-addressed sha256 and a matching extension, never from `Invoice.number` or any other
    value extracted from the document itself, so it can never carry a header-injection or
    path-traversal payload into `Content-Disposition`."""

    data: bytes
    media_type: Literal["application/pdf", "image/png", "image/jpeg"]
    filename: str


FIELD_VALUE_MAX_LENGTH = 1000
"""Longest value a reviewer's field edit may carry (422 beyond it); the supplier address has its
own, tighter cap in `apx.workflows.corrections` (ADR-0012 amendment 4)."""


class FieldEdit(Frozen):
    field: str
    value: str = Field(max_length=FIELD_VALUE_MAX_LENGTH)
    reason: str


class DecisionRequest(Frozen):
    action: Literal["approve", "reject", "send_to_review", "execute_proposal"]
    proposal_id: str | None = None
    comment: str = ""


class SseEvent(Frozen):
    event: str
    data: dict[str, JsonValue]


class TokenResponse(Frozen):
    """Response body of `POST /auth/login`."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class UploadResponse(Frozen):
    """Response body of `POST /uploads`: the stored document's content address and, for a new
    (non-duplicate) document, the `InvoiceWorkflow` id `apx.ingest.ingestor.Ingestor` started
    for it. `workflow_id` is `None` when `duplicate` is `True`: the document was
    already ingested, so no new workflow was started."""

    sha256: Sha256Hex
    filename: str
    media_type: Literal["application/pdf", "image/png", "image/jpeg", "image/tiff"]
    size_bytes: int
    duplicate: bool
    workflow_id: str | None = None


class ProposalAcceptResponse(Frozen):
    """Response body of `POST /proposals/{id}/accept` (P5)."""

    id: str
    status: Literal["executed"]


class TokenVerifier(Protocol):
    def verify(self, token: str) -> TokenClaims:
        """Verify a JWT. Raises PolicyDeniedError on invalid or expired tokens."""
        ...


class InvoiceService(Protocol):
    async def list(
        self,
        status: InvoiceStatus | None,
        vendor_id: VendorId | None,
        q: str | None,
        limit: int,
        offset: int,
    ) -> InvoicePage: ...
    async def get(self, invoice_id: InvoiceId, user: User) -> InvoiceDetail: ...
    async def edit_field(self, invoice_id: InvoiceId, edit: FieldEdit, user: User) -> None: ...
    async def decide(self, invoice_id: InvoiceId, request: DecisionRequest, user: User) -> None:
        """Signal the workflow after policy authorisation. Raises PolicyDeniedError."""
        ...

    def events(self, invoice_id: InvoiceId | None) -> AsyncIterator[SseEvent]: ...
