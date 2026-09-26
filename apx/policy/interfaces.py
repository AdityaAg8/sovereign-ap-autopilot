"""Interfaces for the policy engine: role capabilities, segregation of duties, auto/review/block
routing (P1-P5), authorisation grants for payment-affecting actions and agent proposals
(PRD §6.2 I3, §6.3, ADR-0003, ADR-0008)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, Protocol

from pydantic import JsonValue

from apx.common.errors import PayeeRefusal, PayeeRefusedError
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
from apx.validate.interfaces import ValidationReport, ViolationCode

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


class PayeeMatch(StrEnum):
    """How the invoice IBAN related to the payee `select_payee` chose (ADR-0014)."""

    MATCHED = "matched"
    """The invoice IBAN equals the selected account."""
    SOLE_UNCLAIMED = "sole_unclaimed"
    """Exactly one verified account and the invoice IBAN is None or blank."""
    SOLE_MISMATCH = "sole_mismatch"
    """Exactly one verified account and the invoice IBAN differs from it or is malformed."""


class RoutedPayee(Frozen):
    """The P1 payee pinned at routing (ADR-0014 §7): `select_payee`'s choice when the invoice
    was routed, carried through the workflow and persisted with the route decision. It holds no
    raw IBAN: the build compares `iban_sha256` with a fresh selection, and readers (agent view,
    review UI) only ever see `masked_iban`."""

    masked_iban: str
    """`"****"` plus the last 4 characters of `Payee.iban`."""
    match: PayeeMatch
    verified_count: int
    iban_sha256: Sha256Hex
    """SHA-256 (hex) of `Payee.iban` in electronic format, UTF-8."""


class ValidationFinding(Frozen):
    """An error-severity violation behind a VALIDATION reason: the validator's code and field
    path, both system text (never document text or amounts). Persisted with the routing
    decision so the API can refuse an approval no correction can fix (ADR-0012 amendment 4)."""

    code: ViolationCode
    field: str | None


class PolicyReason(Frozen):
    rule: PolicyRule
    route: Route
    message: str
    """System-written template text. Never interpolates document-derived free text (ADR-0008)."""
    payee_refusal: PayeeRefusal | None = None
    """Set on a P1 reason when `select_payee` refused (ADR-0014)."""
    payee_match: PayeeMatch | None = None
    """Set on a P1 reason raised for a non-trivial payee selection (ADR-0014)."""
    violations: tuple[ValidationFinding, ...] = ()
    """Set on a VALIDATION reason: its error-severity violations (ADR-0012 amendment 4). The
    default keeps route results recorded before it deserializable on workflow replay."""


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
    payee: RoutedPayee | None = None
    """P1's selection at routing time; None when `select_payee` refused (the P1 reason then
    carries `payee_refusal`) or for a decision rebuilt from a stored route (`apx.api`)."""


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


class Payee(Frozen):
    """The P1 payee (ADR-0014). `account` is always one of `vendor.bank_accounts` with
    `verified=True` and `source="admin_callback"`; the invoice IBAN only selected it."""

    account: BankAccount
    iban: str
    """`electronic_iban(account.iban)`: the value paid, digested and written into the UBL."""
    match: PayeeMatch
    verified_count: int
    """Number of verified admin-callback account records (>= 1)."""


_ELECTRONIC_IBAN_RE: Final = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}")


def electronic_iban(raw: str) -> str | None:
    """ISO 13616 electronic format, or None if `raw` is not a well-formed IBAN (ADR-0014 §2).

    ASCII only (checked before case-folding), U+0020 SPACE is the only separator removed, then
    upper-case and `[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}`. Tabs, NBSP, hyphens, zero-width characters,
    non-ASCII digits and lookalike letters are rejected, never stripped or transliterated, so
    untrusted text cannot become a verified IBAN through characters a reviewer cannot see.
    Mod-97 is deliberately not checked: selection is exact equality with an admin-verified record.
    Not the same as `apx.fraud.normalize.normalize_iban`, which strips separators for fuzzy fraud
    matching and must never be used for payment; the two are kept apart on purpose."""
    if not raw.isascii():
        return None
    candidate = raw.replace(" ", "").upper()
    if _ELECTRONIC_IBAN_RE.fullmatch(candidate) is None:
        return None
    return candidate


def select_payee(vendor: Vendor | None, *, invoice_iban: str | None) -> Payee:
    """P1 (ADR-0014): the payee among the vendor's admin-verified accounts. Raises
    PayeeRefusedError with a PayeeRefusal. The invoice IBAN only selects, never introduces.

    One verified account is always the payee (match records how the invoice relates to it).
    Two or more need the invoice IBAN to equal exactly one of them in electronic format."""
    if vendor is None:
        raise PayeeRefusedError(
            PayeeRefusal.UNKNOWN_VENDOR, "no vendor: cannot select a payee (P1)"
        )
    verified: list[tuple[BankAccount, str]] = []
    for account in vendor.bank_accounts:
        if not (account.verified and account.source == "admin_callback"):
            continue
        normalized = electronic_iban(account.iban)
        if normalized is None:
            raise PayeeRefusedError(
                PayeeRefusal.MASTER_IBAN_MALFORMED,
                f"vendor {vendor.id} has an admin-verified account with a malformed IBAN (P1)",
            )
        verified.append((account, normalized))

    if not verified:
        raise PayeeRefusedError(
            PayeeRefusal.NO_VERIFIED_ACCOUNT,
            f"vendor {vendor.id} has no admin-verified account (P1)",
        )

    blank = invoice_iban is None or invoice_iban.strip() == ""
    claimed = None if invoice_iban is None or blank else electronic_iban(invoice_iban)

    if len(verified) == 1:
        account, iban = verified[0]
        if blank:
            match = PayeeMatch.SOLE_UNCLAIMED
        elif claimed == iban:
            match = PayeeMatch.MATCHED
        else:
            match = PayeeMatch.SOLE_MISMATCH
        return Payee(account=account, iban=iban, match=match, verified_count=1)

    count = len(verified)
    if blank:
        raise PayeeRefusedError(
            PayeeRefusal.INVOICE_IBAN_MISSING,
            f"vendor {vendor.id} has {count} admin-verified accounts and the invoice names "
            "none (P1)",
        )
    if claimed is None:
        raise PayeeRefusedError(
            PayeeRefusal.INVOICE_IBAN_MALFORMED,
            f"vendor {vendor.id} has {count} admin-verified accounts and the invoice IBAN "
            "is malformed (P1)",
        )
    matches = [(account, iban) for account, iban in verified if iban == claimed]
    if not matches:
        raise PayeeRefusedError(
            PayeeRefusal.NO_MATCH,
            f"vendor {vendor.id} has {count} admin-verified accounts and none equals the "
            "invoice IBAN (P1)",
        )
    if len(matches) > 1:
        raise PayeeRefusedError(
            PayeeRefusal.AMBIGUOUS_MATCH,
            f"vendor {vendor.id} has {len(matches)} admin-verified records equal to the "
            "invoice IBAN (P1)",
        )
    account, iban = matches[0]
    return Payee(account=account, iban=iban, match=PayeeMatch.MATCHED, verified_count=count)


def payee_iban_digest(iban: str) -> Sha256Hex:
    """SHA-256 (hex) of an electronic-format IBAN; how a pinned payee is compared (§7)."""
    return Sha256Hex(hashlib.sha256(iban.encode("utf-8")).hexdigest())


def pin_payee(payee: Payee) -> RoutedPayee:
    """The `RoutedPayee` for `payee`: masked, digested, never the raw IBAN (ADR-0014 §7)."""
    return RoutedPayee(
        masked_iban="****" + payee.iban[-4:],
        match=payee.match,
        verified_count=payee.verified_count,
        iban_sha256=payee_iban_digest(payee.iban),
    )


def reasons_payload(reasons: tuple[PolicyReason, ...]) -> list[JsonValue]:
    """Audit/record form of `reasons`; the typed P1 payee fields are written only when set
    (ADR-0014)."""
    payload: list[JsonValue] = []
    for r in reasons:
        item: dict[str, JsonValue] = {
            "rule": r.rule.value,
            "route": r.route.value,
            "message": r.message,
        }
        if r.payee_refusal is not None:
            item["payee_refusal"] = r.payee_refusal.value
        if r.payee_match is not None:
            item["payee_match"] = r.payee_match.value
        if r.violations:
            item["violations"] = [
                {"code": finding.code.value, "field": finding.field} for finding in r.violations
            ]
        payload.append(item)
    return payload


def payee_refusal_of(decision: RoutingDecision) -> PayeeRefusal | None:
    """The typed P1 refusal among `decision.reasons`, if `select_payee` refused at routing."""
    return next((r.payee_refusal for r in decision.reasons if r.payee_refusal is not None), None)


def payee_record(decision: RoutingDecision) -> dict[str, JsonValue]:
    """The persisted, reader-safe form of the routed payee: masked IBAN, match and count, or the
    refusal. Never the raw IBAN or its digest (ADR-0014 §7, migration 0008)."""
    payee = decision.payee
    if payee is None:
        refusal = payee_refusal_of(decision)
        return {"refusal": None if refusal is None else refusal.value}
    return {
        "masked_iban": payee.masked_iban,
        "match": payee.match.value,
        "verified_count": payee.verified_count,
    }


def routing_record(decision: RoutingDecision) -> list[JsonValue]:
    """`decisions.reasons` for a `policy_route` row: every reason, then one `{"payee": ...}`
    element (`payee_record`). `v_agent_invoices` splits the two apart (migration 0008)."""
    return [*reasons_payload(decision.reasons), {"payee": payee_record(decision)}]


def subject_digest(request: ActionRequest) -> Sha256Hex:
    """SHA-256 of canonical JSON (sorted keys, compact separators, UTF-8) over the payment
    subject: action, invoice id, total, currency, vendor id, for approvals the normalized payee
    IBAN and payee match from `select_payee` (ADR-0014 §6), target bank change, proposal id and
    payload. Excludes actor and time, so any edit to these values between grant and write is
    detected (TOCTOU). Raises PayeeRefusedError when an approval's payee selection refuses."""
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
        payee = select_payee(vendor, invoice_iban=invoice.iban if invoice is not None else None)
        subject["payee_iban"] = payee.iban
        subject["payee_match"] = payee.match.value
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
