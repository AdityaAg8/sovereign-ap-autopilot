"""Segregation-of-duties matrix tests (P2): the user who changed a vendor's bank details is
barred from approving that vendor's payments for `sod_cooldown_days`, including through the
proposal-execution path and for users holding several roles (P2, ADR-0008 K5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from apx.common.types import Role, Route
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.interfaces import (
    ActionRequest,
    AuthorizationDecision,
    Capability,
    PolicyRule,
    PolicyThresholds,
    Proposal,
    RoutingDecision,
)
from tests.unit._policy_factories import (
    make_bank_change,
    make_invoice,
    make_user,
    make_vendor,
    money,
)

_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _thresholds() -> PolicyThresholds:
    return PolicyThresholds(
        approval_limit=money("5000"), fraud_review=0.5, scan_review=0.5, attack_block=0.9
    )


def _is_barred(decision: AuthorizationDecision) -> bool:
    return any(r.rule is PolicyRule.P2_BANK_CHANGE for r in decision.reasons)


def _request(
    *,
    actor_roles: frozenset[Role],
    actor_id: str,
    changed_by: str,
    days_ago: int,
    proposal: Proposal | None = None,
    action: Capability = Capability.APPROVE_PAYMENT,
) -> ActionRequest:
    vendor = make_vendor()
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    change = make_bank_change(
        vendor_id=str(vendor.id), changed_by=changed_by, changed_at=_AT - timedelta(days=days_ago)
    )
    return ActionRequest(
        actor=make_user(user_id=actor_id, roles=actor_roles),
        action=action,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=RoutingDecision(route=Route.AUTO, reasons=()),
        bank_changes=(change,),
        proposal=proposal,
    )


def test_day_29_is_barred() -> None:
    request = _request(
        actor_roles=frozenset({Role.APPROVER}),
        actor_id="changer-1",
        changed_by="changer-1",
        days_ago=29,
    )
    decision = DefaultPolicyEngine(_thresholds()).authorize(request)
    assert not decision.allowed
    assert _is_barred(decision)


def test_day_30_is_allowed() -> None:
    request = _request(
        actor_roles=frozenset({Role.APPROVER}),
        actor_id="changer-1",
        changed_by="changer-1",
        days_ago=30,
    )
    decision = DefaultPolicyEngine(_thresholds()).authorize(request)
    assert decision.allowed
    assert not _is_barred(decision)


def test_a_different_approver_is_allowed_on_day_zero() -> None:
    request = _request(
        actor_roles=frozenset({Role.APPROVER}),
        actor_id="other-approver",
        changed_by="changer-1",
        days_ago=0,
    )
    decision = DefaultPolicyEngine(_thresholds()).authorize(request)
    assert decision.allowed


def test_admin_and_approver_roles_together_are_still_barred() -> None:
    request = _request(
        actor_roles=frozenset({Role.ADMIN, Role.APPROVER}),
        actor_id="changer-1",
        changed_by="changer-1",
        days_ago=5,
    )
    decision = DefaultPolicyEngine(_thresholds()).authorize(request)
    assert not decision.allowed
    assert _is_barred(decision)


def test_sod_bar_applies_through_the_proposal_execution_path() -> None:
    proposal = Proposal(
        id="p1",
        invoice_id=None,
        kind="approve_payment",
        payload={},
        created_by="agent",
        created_at=_AT,
    )
    request = _request(
        actor_roles=frozenset({Role.APPROVER}),
        actor_id="changer-1",
        changed_by="changer-1",
        days_ago=5,
        proposal=proposal,
        action=Capability.EXECUTE_PROPOSAL,
    )
    decision = DefaultPolicyEngine(_thresholds()).authorize(request)
    assert not decision.allowed
    assert _is_barred(decision)


def test_sod_bar_does_not_apply_to_a_different_vendors_history() -> None:
    thresholds = _thresholds()
    vendor = make_vendor(vendor_id="vendor-1")
    other_vendor_change = make_bank_change(
        vendor_id="vendor-2", changed_by="changer-1", changed_at=_AT
    )
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    request = ActionRequest(
        actor=make_user(user_id="changer-1", roles=frozenset({Role.APPROVER})),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=RoutingDecision(route=Route.AUTO, reasons=()),
        bank_changes=(other_vendor_change,),
    )
    decision = DefaultPolicyEngine(thresholds).authorize(request)
    assert decision.allowed
