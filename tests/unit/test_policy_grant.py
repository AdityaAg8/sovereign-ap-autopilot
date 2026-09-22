"""Tests for DefaultPolicyGuard: every authorize outcome is audited (I4), an allowed request
returns a Grant bound to subject_digest, and denials raise the right typed error
(ADR-0008 K6, M6.5)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime

import pytest

from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.common.errors import PolicyDeniedError, SegregationOfDutiesError
from apx.common.types import Role, Route
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.guard import DefaultPolicyGuard
from apx.policy.interfaces import (
    ActionRequest,
    Capability,
    PolicyThresholds,
    RoutingDecision,
    subject_digest,
)
from tests.unit._policy_factories import (
    make_bank_change,
    make_invoice,
    make_user,
    make_vendor,
    money,
)

_AT = datetime(2026, 1, 1, tzinfo=UTC)


class FakeAuditLog:
    """In-memory AuditLog fake: no chain hashing, just records what was appended."""

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


def _thresholds() -> PolicyThresholds:
    return PolicyThresholds(
        approval_limit=money("5000"), fraud_review=0.5, scan_review=0.5, attack_block=0.9
    )


def _approval_request(*, actor_roles: frozenset[Role], vendor=None, routing=None) -> ActionRequest:
    vendor = vendor if vendor is not None else make_vendor()
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    return ActionRequest(
        actor=make_user(user_id="u1", roles=actor_roles),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=routing if routing is not None else RoutingDecision(route=Route.AUTO, reasons=()),
    )


def test_allowed_request_returns_a_grant_bound_to_subject_digest() -> None:
    audit = FakeAuditLog()
    guard = DefaultPolicyGuard(DefaultPolicyEngine(_thresholds()), audit)
    request = _approval_request(actor_roles=frozenset({Role.APPROVER}))

    grant = asyncio.run(guard.require(request))

    assert grant.subject_digest == subject_digest(request)
    assert grant.actor == request.actor.id
    assert grant.action == request.action
    assert grant.audit_entry_id == 1
    assert len(audit.events) == 1
    assert audit.events[0].payload["allowed"] is True


def test_denied_request_is_audited_and_raises_policy_denied() -> None:
    audit = FakeAuditLog()
    guard = DefaultPolicyGuard(DefaultPolicyEngine(_thresholds()), audit)
    request = _approval_request(actor_roles=frozenset())  # no capability at all

    with pytest.raises(PolicyDeniedError):
        asyncio.run(guard.require(request))

    assert len(audit.events) == 1
    assert audit.events[0].payload["allowed"] is False


def test_sod_denial_is_audited_and_raises_segregation_of_duties_error() -> None:
    audit = FakeAuditLog()
    guard = DefaultPolicyGuard(DefaultPolicyEngine(_thresholds()), audit)
    vendor = make_vendor()
    change = make_bank_change(vendor_id=str(vendor.id), changed_by="u1", changed_at=_AT)
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    request = ActionRequest(
        actor=make_user(user_id="u1", roles=frozenset({Role.APPROVER})),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=RoutingDecision(route=Route.AUTO, reasons=()),
        bank_changes=(change,),
    )

    with pytest.raises(SegregationOfDutiesError):
        asyncio.run(guard.require(request))

    assert len(audit.events) == 1
    assert audit.events[0].payload["allowed"] is False


def test_subject_digest_changes_when_the_invoice_total_changes() -> None:
    """The mechanism behind K6's stale-grant check: editing a digest input after a grant was
    issued must be detectable by recomputing subject_digest."""
    vendor = make_vendor()
    request_a = _approval_request(actor_roles=frozenset({Role.APPROVER}), vendor=vendor)
    invoice_b = make_invoice(total="999", vendor_id=vendor.id)
    request_b = request_a.model_copy(update={"invoice": invoice_b})

    assert subject_digest(request_a) != subject_digest(request_b)


def test_subject_digest_is_stable_for_an_unchanged_request() -> None:
    request = _approval_request(actor_roles=frozenset({Role.APPROVER}))
    assert subject_digest(request) == subject_digest(request)


def test_subject_digest_omits_payee_iban_for_a_non_approval_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F8: CHANGE_BANK_DETAILS needs only the capability, not the routing/P1 payee gate
    (ADR-0008 K5) -- it is not an approval action, so `subject_digest` must never resolve or
    include a payee IBAN for it, even with no vendor at all. `select_payee(None)` raises
    `PolicyDeniedError` (see test_policy_payee.py), so a real, unmocked call would fail this
    request outright; monkeypatching confirms it is never reached."""

    def _must_not_be_called(vendor: object) -> None:
        raise AssertionError("select_payee must not be called for a non-approval action")

    monkeypatch.setattr("apx.policy.interfaces.select_payee", _must_not_be_called)

    change = make_bank_change(vendor_id="vendor-1", changed_by="admin-1", changed_at=_AT)
    request = ActionRequest(
        actor=make_user(user_id="admin-1", roles=frozenset({Role.ADMIN})),
        action=Capability.CHANGE_BANK_DETAILS,
        invoice=None,
        vendor=None,
        at=_AT,
        bank_change=change,
    )

    digest = subject_digest(request)  # must not raise: proves select_payee was never invoked

    expected_subject = {
        "action": "change_bank_details",
        "invoice_id": None,
        "total_amount": None,
        "total_currency": None,
        "vendor_id": None,
        "bank_change": {
            "vendor_id": change.vendor_id,
            "old_iban": change.old_iban,
            "new_iban": change.new_iban,
        },
        "proposal_id": None,
        "proposal_payload": None,
    }
    assert "payee_iban" not in expected_subject
    blob = json.dumps(expected_subject, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    assert digest == hashlib.sha256(blob).hexdigest()
