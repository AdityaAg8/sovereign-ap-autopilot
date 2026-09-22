"""Interfaces for the policy engine: role capabilities, segregation of duties, auto/review/block
routing (P1-P5) and authorisation of payment-affecting actions and agent proposals
(PRD §6.2 I3, §6.3)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import JsonValue

from apx.common.invoice import Invoice
from apx.common.master import User, Vendor
from apx.common.types import Frozen, InvoiceId, Money, Role, Route, UserId
from apx.fraud.interfaces import FraudReport
from apx.scan.interfaces import ScanReport
from apx.trust.interfaces import TrustDecision
from apx.validate.interfaces import ValidationReport


class Capability(StrEnum):
    EDIT_FIELD = "edit_field"
    ROUTE_EXCEPTION = "route_exception"
    APPROVE_PAYMENT = "approve_payment"
    APPROVE_ABOVE_LIMIT = "approve_above_limit"
    CHANGE_BANK_DETAILS = "change_bank_details"
    VERIFY_BANK_DETAILS = "verify_bank_details"
    MANAGE_USERS = "manage_users"
    EXECUTE_PROPOSAL = "execute_proposal"


class PolicyRule(StrEnum):
    P1_VERIFIED_IBAN = "P1"
    P2_BANK_CHANGE = "P2"
    P3_APPROVAL_LIMIT = "P3"
    P4_FLAGS = "P4"
    P5_PROPOSALS_ONLY = "P5"
    TRUST_GATE = "trust_gate"
    VALIDATION = "validation"


class PolicyThresholds(Frozen):
    approval_limit: Money
    fraud_review: float
    scan_review: float
    attack_block: float
    sod_cooldown_days: int = 30


class PolicyReason(Frozen):
    rule: PolicyRule
    route: Route
    message: str


class RoutingInput(Frozen):
    invoice: Invoice
    vendor: Vendor | None
    trust: TrustDecision
    validation: ValidationReport
    fraud: FraudReport
    scan: ScanReport


class RoutingDecision(Frozen):
    """Most severe route across reasons wins (block > review > auto)."""

    route: Route
    reasons: tuple[PolicyReason, ...]


class Proposal(Frozen):
    """An agent- or chat-drafted action. Never executes by itself (P5)."""

    id: str
    invoice_id: InvoiceId | None
    kind: Literal["edit_field", "route", "request_info", "approve_payment", "change_bank"]
    payload: dict[str, JsonValue]
    created_by: Literal["agent", "chat"] | UserId
    created_at: datetime
    status: Literal["pending", "executed", "rejected"] = "pending"


class ActionRequest(Frozen):
    """A human attempt to perform a payment-affecting or state-changing action."""

    actor: User
    action: Capability
    invoice: Invoice | None
    vendor: Vendor | None
    proposal: Proposal | None = None


class AuthorizationDecision(Frozen):
    allowed: bool
    reasons: tuple[PolicyReason, ...]


class PolicyEngine(Protocol):
    def route(self, inp: RoutingInput) -> RoutingDecision: ...

    def authorize(self, request: ActionRequest) -> AuthorizationDecision:
        """Capability + P1-P3 + segregation of duties (I3). Callers raise PolicyDeniedError on
        `allowed=False`; there is no override path."""
        ...


def capabilities_for(roles: frozenset[Role]) -> frozenset[Capability]:
    """Union of capabilities granted by the given roles."""
    raise NotImplementedError
