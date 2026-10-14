"""Property tests for DefaultPolicyEngine.authorize: `allowed` implies every K5 condition holds
(capability, P1 payee, routing present and not BLOCK, P2 SoD), and a minimal allowed case exists
(non-vacuous) (I3, ADR-0008 K5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

from apx.common.errors import PayeeRefusal
from apx.common.master import Vendor
from apx.common.types import Role, Route
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.interfaces import (
    ActionRequest,
    Capability,
    PolicyRule,
    PolicyThresholds,
    RoutingDecision,
    capabilities_for,
    required_capabilities,
    select_payee,
)
from tests.unit._policy_factories import (
    make_bank_account,
    make_bank_change,
    make_invoice,
    make_user,
    make_vendor,
    money,
)

_AT = datetime(2026, 1, 1, tzinfo=UTC)

_ROLE_SETS = st.frozensets(st.sampled_from(list(Role)), min_size=0, max_size=3)
_ROUTES_OR_MISSING = st.sampled_from([Route.AUTO, Route.REVIEW, Route.BLOCK, None])


def _thresholds() -> PolicyThresholds:
    return PolicyThresholds(
        approval_limit=money("5000"), fraud_review=0.5, scan_review=0.5, attack_block=0.9
    )


@given(
    roles=_ROLE_SETS,
    total=st.decimals(
        min_value="1", max_value="10000", places=2, allow_nan=False, allow_infinity=False
    ),
    days_ago=st.integers(min_value=0, max_value=60),
    same_actor_changed_bank=st.booleans(),
    routing_route=_ROUTES_OR_MISSING,
    vendor_has_payee=st.booleans(),
)
def test_allowed_approval_implies_every_k5_condition(
    roles: frozenset[Role],
    total,
    days_ago: int,
    same_actor_changed_bank: bool,
    routing_route,
    vendor_has_payee: bool,
) -> None:
    thresholds = _thresholds()
    vendor = make_vendor(accounts=() if not vendor_has_payee else None)
    invoice = make_invoice(total=str(total), vendor_id=vendor.id)
    routing = None if routing_route is None else RoutingDecision(route=routing_route, reasons=())
    change = make_bank_change(
        vendor_id=str(vendor.id),
        changed_by="actor-1" if same_actor_changed_bank else "someone-else",
        changed_at=_AT - timedelta(days=days_ago),
    )
    request = ActionRequest(
        actor=make_user(user_id="actor-1", roles=roles),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=routing,
        bank_changes=(change,),
    )

    decision = DefaultPolicyEngine(thresholds).authorize(request)

    if decision.allowed:
        assert required_capabilities(request, thresholds) <= capabilities_for(roles)
        assert select_payee(vendor, invoice_iban=invoice.iban) is not None  # must not raise (P1)
        assert routing is not None and routing.route != Route.BLOCK
        assert not (same_actor_changed_bank and days_ago < thresholds.sod_cooldown_days)


def test_a_minimal_allowed_case_exists() -> None:
    """Non-vacuous: the property above would hold trivially if nothing were ever allowed."""
    thresholds = _thresholds()
    vendor = make_vendor()
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    request = ActionRequest(
        actor=make_user(user_id="approver-1", roles=frozenset({Role.APPROVER})),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=RoutingDecision(route=Route.AUTO, reasons=()),
    )
    decision = DefaultPolicyEngine(thresholds).authorize(request)
    assert decision.allowed
    assert decision.reasons == ()


def test_missing_capability_is_denied() -> None:
    thresholds = _thresholds()
    vendor = make_vendor()
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    request = ActionRequest(
        actor=make_user(roles=frozenset()),  # no roles at all
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=RoutingDecision(route=Route.AUTO, reasons=()),
    )
    decision = DefaultPolicyEngine(thresholds).authorize(request)
    assert not decision.allowed


def test_above_limit_requires_approve_above_limit_capability() -> None:
    thresholds = _thresholds()
    vendor = make_vendor()
    invoice = make_invoice(total="9000", vendor_id=vendor.id)  # above the 5000 limit
    reviewer_request = ActionRequest(
        actor=make_user(roles=frozenset({Role.REVIEWER})),  # has APPROVE_PAYMENT, not ABOVE_LIMIT
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=RoutingDecision(route=Route.REVIEW, reasons=()),
    )
    approver_request = reviewer_request.model_copy(
        update={"actor": make_user(roles=frozenset({Role.APPROVER}))}
    )
    assert not DefaultPolicyEngine(thresholds).authorize(reviewer_request).allowed
    assert DefaultPolicyEngine(thresholds).authorize(approver_request).allowed


def test_missing_routing_denies_an_approval_fail_closed() -> None:
    thresholds = _thresholds()
    vendor = make_vendor()
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    request = ActionRequest(
        actor=make_user(roles=frozenset({Role.APPROVER})),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=None,
    )
    assert not DefaultPolicyEngine(thresholds).authorize(request).allowed


def test_blocked_routing_denies_an_approval_with_no_override() -> None:
    thresholds = _thresholds()
    vendor = make_vendor()
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    request = ActionRequest(
        actor=make_user(roles=frozenset({Role.APPROVER})),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=RoutingDecision(route=Route.BLOCK, reasons=()),
    )
    assert not DefaultPolicyEngine(thresholds).authorize(request).allowed


def test_change_bank_details_needs_only_the_capability() -> None:
    """K5: CHANGE_BANK_DETAILS and VERIFY_BANK_DETAILS skip P1/P2/routing entirely."""
    thresholds = _thresholds()
    request = ActionRequest(
        actor=make_user(roles=frozenset({Role.ADMIN})),
        action=Capability.CHANGE_BANK_DETAILS,
        invoice=None,
        vendor=None,
        at=_AT,
        routing=None,
    )
    decision = DefaultPolicyEngine(thresholds).authorize(request)
    assert decision.allowed


def _approval(vendor: Vendor, iban: str | None) -> ActionRequest:
    invoice = make_invoice(total="100", vendor_id=vendor.id).model_copy(update={"iban": iban})
    return ActionRequest(
        actor=make_user(user_id="approver-1", roles=frozenset({Role.APPROVER})),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=RoutingDecision(route=Route.REVIEW, reasons=()),
    )


def test_a_payee_refusal_denies_approval_with_the_typed_reason() -> None:
    """ADR-0014: two verified accounts and an invoice IBAN matching neither refuses P1."""
    vendor = make_vendor(
        accounts=(
            make_bank_account(iban="AE070331234567890000001"),
            make_bank_account(iban="AE070331234567890000002"),
        )
    )
    decision = DefaultPolicyEngine(_thresholds()).authorize(
        _approval(vendor, "AE460090000000123456789")
    )
    assert not decision.allowed
    p1 = [r for r in decision.reasons if r.rule is PolicyRule.P1_VERIFIED_IBAN]
    assert [r.payee_refusal for r in p1] == [PayeeRefusal.NO_MATCH]


def test_two_verified_accounts_with_a_match_can_be_approved() -> None:
    vendor = make_vendor(
        accounts=(
            make_bank_account(iban="AE070331234567890000001"),
            make_bank_account(iban="AE070331234567890000002"),
        )
    )
    decision = DefaultPolicyEngine(_thresholds()).authorize(
        _approval(vendor, "ae07 0331 2345 6789 0000 002")
    )
    assert decision.allowed
    assert decision.reasons == ()


def test_a_sole_verified_account_with_a_mismatching_invoice_iban_can_be_approved() -> None:
    """ADR-0014 §4: authorize does not deny SOLE_MISMATCH; a human approving it is the review."""
    vendor = make_vendor(accounts=(make_bank_account(iban="AE070331234567890000001"),))
    decision = DefaultPolicyEngine(_thresholds()).authorize(
        _approval(vendor, "AE460090000000123456789")
    )
    assert decision.allowed
