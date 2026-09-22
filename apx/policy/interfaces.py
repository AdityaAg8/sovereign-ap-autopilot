"""Interfaces for the policy engine: role capabilities, segregation of duties, auto/review/block
routing (P1-P5), authorisation grants for payment-affecting actions and agent proposals
(PRD §6.2 I3, §6.3, ADR-0003, ADR-0008)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, Protocol

from pydantic import JsonValue

from apx.common.errors import PolicyDeniedError
from apx.common.invoice import Invoice
from apx.common.master import BankAccount, BankDetailChange, User, Vendor
from apx.common.types import (
    Frozen,
    InvoiceId,
    Money,
    Role,
    Route,
    Sha256Hex,
    UserId,
    VendorId,
)
from apx.fraud.interfaces import FraudReport
from apx.scan.interfaces import ScanReport
from apx.trust.interfaces import TrustDecision
from apx.validate.interfaces import ValidationReport

ProposalKind = Literal["edit_field", "route", "request_info", "approve_payment", "change_bank"]
ProposalOrigin = Literal["agent", "chat", "mcp"]


class Capability(StrEnum):
    EDIT_FIELD = "edit_field"
    ROUTE_EXCEPTION = "route_exception"
    APPROVE_PAYMENT = "approve_payment"
    APPROVE_ABOVE_LIMIT = "approve_above_limit"
    CHANGE_BANK_DETAILS = "change_bank_details"
    VERIFY_BANK_DETAILS = "verify_bank_details"
    MANAGE_USERS = "manage_users"
    EXECUTE_PROPOSAL = "execute_proposal"


ROLE_CAPABILITIES: Final[Mapping[Role, frozenset[Capability]]] = MappingProxyType(
    {
        Role.REVIEWER: frozenset(
            {
                Capability.EDIT_FIELD,
                Capability.ROUTE_EXCEPTION,
                Capability.APPROVE_PAYMENT,
                Capability.EXECUTE_PROPOSAL,
            }
        ),
        Role.APPROVER: frozenset(
            {
                Capability.ROUTE_EXCEPTION,
                Capability.APPROVE_PAYMENT,
                Capability.APPROVE_ABOVE_LIMIT,
                Capability.EXECUTE_PROPOSAL,
            }
        ),
        Role.ADMIN: frozenset(
            {
                Capability.CHANGE_BANK_DETAILS,
                Capability.VERIFY_BANK_DETAILS,
                Capability.MANAGE_USERS,
                Capability.EXECUTE_PROPOSAL,
            }
        ),
    }
)
"""Role to capability matrix (ADR-0008). Admin alone cannot approve payments; a user holding
several roles gets the union, and P2 segregation of duties still applies."""

PAYMENT_AFFECTING: Final[frozenset[Capability]] = frozenset(
    {
        Capability.APPROVE_PAYMENT,
        Capability.APPROVE_ABOVE_LIMIT,
        Capability.CHANGE_BANK_DETAILS,
        Capability.VERIFY_BANK_DETAILS,
    }
)
"""Capabilities whose use changes who is paid, how much, or whether a non-auto invoice is paid.
EXECUTE_PROPOSAL is payment-affecting when the proposal kind maps into this set (I3)."""

PROPOSAL_CAPABILITY: Final[Mapping[ProposalKind, Capability]] = MappingProxyType(
    {
        "edit_field": Capability.EDIT_FIELD,
        "route": Capability.ROUTE_EXCEPTION,
        "request_info": Capability.ROUTE_EXCEPTION,
        "approve_payment": Capability.APPROVE_PAYMENT,
        "change_bank": Capability.CHANGE_BANK_DETAILS,
    }
)
"""Capability the executing human must hold, in addition to EXECUTE_PROPOSAL (P5)."""

AGENT_PROPOSAL_KINDS: Final[frozenset[ProposalKind]] = frozenset(
    {"edit_field", "route", "request_info", "approve_payment"}
)
"""Kinds an agent, chat or MCP client may create. Bank-detail changes originate only from an
admin in the vendor-master screen, never from a model (ADR-0008)."""


class PolicyRule(StrEnum):
    P1_VERIFIED_IBAN = "P1"
    P2_BANK_CHANGE = "P2"
    P3_APPROVAL_LIMIT = "P3"
    P4_FLAGS = "P4"
    P5_PROPOSALS_ONLY = "P5"
    TRUST_GATE = "trust_gate"
    VALIDATION = "validation"
    CAPABILITY = "capability"
    STALE_GRANT = "stale_grant"


class PolicyThresholds(Frozen):
    approval_limit: Money
    fraud_review: float
    scan_review: float
    attack_block: float
    sod_cooldown_days: int = 30
    bank_change_review_days: int = 30


class PolicyReason(Frozen):
    rule: PolicyRule
    route: Route
    message: str
    """System-written template text. Never interpolates document-derived free text (ADR-0008)."""


class RoutingInput(Frozen):
    invoice: Invoice
    vendor: Vendor | None
    trust: TrustDecision
    validation: ValidationReport
    fraud: FraudReport
    scan: ScanReport
    at: datetime
    """Evaluation time; injected so routing is deterministic under property tests."""
    bank_changes: tuple[BankDetailChange, ...] = ()
    """The vendor's bank-detail changes; one inside bank_change_review_days means REVIEW (P2)."""


class RoutingDecision(Frozen):
    """Most severe route across reasons wins (block > review > auto)."""

    route: Route
    reasons: tuple[PolicyReason, ...]


class Proposal(Frozen):
    """An agent-, chat- or MCP-drafted action, or one drafted by a user. Never executes by
    itself (P5)."""

    id: str
    invoice_id: InvoiceId | None
    kind: ProposalKind
    payload: dict[str, JsonValue]
    created_by: ProposalOrigin | UserId
    created_at: datetime
    status: Literal["pending", "executed", "rejected"] = "pending"


class ActionRequest(Frozen):
    """A human attempt to perform a payment-affecting or state-changing action. Built by the
    API from the JWT-authenticated user and current database state, never by apx.agent."""

    actor: User
    action: Capability
    invoice: Invoice | None
    vendor: Vendor | None
    at: datetime
    """Evaluation time for the P2 cooldown window."""
    routing: RoutingDecision | None = None
    """Current routing of `invoice`; required (fail closed) for approvals."""
    bank_changes: tuple[BankDetailChange, ...] = ()
    """The vendor's bank-detail changes; input to the P2 segregation-of-duties bar."""
    bank_change: BankDetailChange | None = None
    """Target change for CHANGE_BANK_DETAILS and VERIFY_BANK_DETAILS."""
    proposal: Proposal | None = None


class AuthorizationDecision(Frozen):
    allowed: bool
    reasons: tuple[PolicyReason, ...]


class Grant(Frozen):
    """Proof that PolicyGuard allowed one ActionRequest. Payment-state writers accept only a
    Grant, and only one whose `subject_digest` still matches current state (ADR-0008)."""

    actor: UserId
    action: Capability
    subject_digest: Sha256Hex
    reasons: tuple[PolicyReason, ...]
    audit_entry_id: int
    issued_at: datetime


class PolicyEngine(Protocol):
    def route(self, inp: RoutingInput) -> RoutingDecision:
        """Pure, deterministic P1-P4 + trust gate + validation routing. P4: injection_score at
        or above attack_block gives BLOCK; any scan flag at or above scan_review or fraud flag
        at or above fraud_review gives REVIEW. AUTO only when no reason is more severe."""
        ...

    def authorize(self, request: ActionRequest) -> AuthorizationDecision:
        """Pure capability + P1-P4 + segregation of duties check (I3). Denies approvals when
        routing is missing or BLOCK. Callers raise PolicyDeniedError on `allowed=False`; there
        is no override path."""
        ...


class PolicyGuard(Protocol):
    async def require(self, request: ActionRequest) -> Grant:
        """Authorize through PolicyEngine, append an audit event for allowed and denied outcomes
        (I4) and return a Grant bound to `subject_digest(request)`. Raises
        SegregationOfDutiesError for P2 bars and PolicyDeniedError for every other denial."""
        ...


class PaymentStateWriter(Protocol):
    """The only mutators of payment state. Each method checks `grant.action`, recomputes the
    subject digest from current state in the same transaction and raises PolicyDeniedError
    (rule STALE_GRANT) on mismatch. Must not be imported by apx.agent (ADR-0008)."""

    async def record_approval(self, invoice_id: InvoiceId, grant: Grant) -> None: ...

    async def record_bank_change(self, change: BankDetailChange, grant: Grant) -> None:
        """Store the change as an unverified account (P1) and start the P2 window."""
        ...

    async def verify_bank_account(self, vendor_id: VendorId, iban: str, grant: Grant) -> None:
        """Record the admin callback verification: `verified=True`, `source="admin_callback"`."""
        ...

    async def execute_proposal(self, proposal_id: str, grant: Grant) -> None: ...


class ProposalSink(Protocol):
    """The only write dependency agent, chat and MCP tools receive (P5)."""

    async def submit(self, proposal: Proposal) -> Proposal:
        """Persist a pending proposal and audit it. Raises PolicyDeniedError (rule P5) if the
        origin is agent/chat/mcp and the kind is not in AGENT_PROPOSAL_KINDS, or if the status
        is not pending. Never executes anything."""
        ...


def capabilities_for(roles: frozenset[Role]) -> frozenset[Capability]:
    """Union of ROLE_CAPABILITIES over the given roles."""
    caps: set[Capability] = set()
    for role in roles:
        caps |= ROLE_CAPABILITIES.get(role, frozenset())
    return frozenset(caps)


_APPROVAL_CAPABILITIES: Final[frozenset[Capability]] = frozenset(
    {Capability.APPROVE_PAYMENT, Capability.APPROVE_ABOVE_LIMIT}
)


def is_approval_action(request: ActionRequest) -> bool:
    """True for a direct payment approval, or an EXECUTE_PROPOSAL that resolves to one through
    PROPOSAL_CAPABILITY. P1, the routing gate and P2 SoD apply only to these; CHANGE_BANK_DETAILS
    and VERIFY_BANK_DETAILS need the capability only (ADR-0008 K5)."""
    if request.action in _APPROVAL_CAPABILITIES:
        return True
    if request.action is Capability.EXECUTE_PROPOSAL and request.proposal is not None:
        return PROPOSAL_CAPABILITY.get(request.proposal.kind) in _APPROVAL_CAPABILITIES
    return False


def required_capabilities(
    request: ActionRequest, thresholds: PolicyThresholds
) -> frozenset[Capability]:
    """Capabilities the actor must hold: `request.action`; for EXECUTE_PROPOSAL also
    PROPOSAL_CAPABILITY[proposal.kind]; for a payment approval (direct or via proposal) also
    APPROVE_ABOVE_LIMIT when the invoice total exceeds approval_limit or is in another
    currency (P3, fail closed)."""
    caps = {request.action}
    if request.action is Capability.EXECUTE_PROPOSAL and request.proposal is not None:
        mapped = PROPOSAL_CAPABILITY.get(request.proposal.kind)
        if mapped is not None:
            caps.add(mapped)
    if is_approval_action(request):
        invoice = request.invoice
        if invoice is None or _above_limit(invoice.total, thresholds.approval_limit):
            caps.add(Capability.APPROVE_ABOVE_LIMIT)
    return frozenset(caps)


def _above_limit(total: Money, limit: Money) -> bool:
    """True when `total` cannot be covered by the plain approval capability (P3): a different
    currency is always treated as above limit, since the two amounts cannot be compared."""
    return total.currency != limit.currency or total.amount > limit.amount


def is_payment_affecting(request: ActionRequest) -> bool:
    """True if the action, or the executed proposal's mapped capability, is in PAYMENT_AFFECTING."""
    if request.action in PAYMENT_AFFECTING:
        return True
    if request.action is Capability.EXECUTE_PROPOSAL and request.proposal is not None:
        return PROPOSAL_CAPABILITY.get(request.proposal.kind) in PAYMENT_AFFECTING
    return False


def select_payee(vendor: Vendor | None) -> BankAccount:
    """P1: the vendor's unique account with `verified=True` and `source="admin_callback"`.
    Raises PolicyDeniedError (rule P1) if the vendor is unknown, no account qualifies or more
    than one does. Never reads `Invoice.iban`."""
    if vendor is None:
        raise PolicyDeniedError("no vendor: cannot select a payee (P1)")
    verified = [
        account
        for account in vendor.bank_accounts
        if account.verified and account.source == "admin_callback"
    ]
    if len(verified) != 1:
        raise PolicyDeniedError(
            f"vendor {vendor.id} has {len(verified)} admin-verified accounts, need exactly 1 (P1)"
        )
    return verified[0]


def subject_digest(request: ActionRequest) -> Sha256Hex:
    """SHA-256 of canonical JSON (sorted keys, compact separators, UTF-8) over the payment
    subject: action, invoice id, total, currency, vendor id, payee IBAN from `select_payee` for
    approvals, target bank change, proposal id and payload. Excludes actor and time, so any edit
    to these values between grant and write is detected (TOCTOU)."""
    invoice = request.invoice
    vendor = request.vendor
    subject: dict[str, object] = {
        "action": request.action.value,
        "invoice_id": invoice.id if invoice is not None else None,
        "total_amount": str(invoice.total.amount) if invoice is not None else None,
        "total_currency": invoice.total.currency if invoice is not None else None,
        "vendor_id": vendor.id if vendor is not None else None,
    }
    if is_approval_action(request):
        subject["payee_iban"] = select_payee(vendor).iban
    change = request.bank_change
    subject["bank_change"] = (
        None
        if change is None
        else {
            "vendor_id": change.vendor_id,
            "old_iban": change.old_iban,
            "new_iban": change.new_iban,
        }
    )
    proposal = request.proposal
    subject["proposal_id"] = proposal.id if proposal is not None else None
    subject["proposal_payload"] = proposal.payload if proposal is not None else None
    blob = json.dumps(subject, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
