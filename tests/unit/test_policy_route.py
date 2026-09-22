"""Property tests for DefaultPolicyEngine.route: P1-P4 + trust gate + validation routing is
deterministic, and the most severe reason wins (PRD §6.3 P1-P4, ADR-0008 K7, M6.5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from apx.common.types import Route
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.interfaces import PolicyThresholds, RoutingInput
from tests.unit._policy_factories import (
    make_bank_change,
    make_fraud_report,
    make_invoice,
    make_scan_report,
    make_trust_decision,
    make_validation_report,
    make_vendor,
    money,
)

_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _thresholds() -> PolicyThresholds:
    return PolicyThresholds(
        approval_limit=money("5000"),
        fraud_review=0.5,
        scan_review=0.5,
        attack_block=0.9,
        sod_cooldown_days=30,
        bank_change_review_days=30,
    )


def _clean_input(
    thresholds: PolicyThresholds,
    *,
    scan_score: float = 0.0,
    fraud_score: float = 0.0,
    injection_score: float = 0.0,
    all_accepted: bool = True,
    has_validation_error: bool = False,
    total: str = "1000",
    currency: str = "AED",
    bank_changes: tuple = (),
    vendor=None,
    no_vendor: bool = False,
) -> RoutingInput:
    if no_vendor:
        vendor = None
    elif vendor is None:
        vendor = make_vendor()
    return RoutingInput(
        invoice=make_invoice(
            total=total, currency=currency, vendor_id=vendor.id if vendor else None
        ),
        vendor=vendor,
        trust=make_trust_decision(all_accepted=all_accepted),
        validation=make_validation_report(has_error=has_validation_error),
        fraud=make_fraud_report(max_flag_score=fraud_score),
        scan=make_scan_report(max_flag_score=scan_score, injection_score=injection_score),
        at=_AT,
        bank_changes=bank_changes,
    )


def test_auto_when_everything_clears() -> None:
    thresholds = _thresholds()
    decision = DefaultPolicyEngine(thresholds).route(_clean_input(thresholds))
    assert decision.route == Route.AUTO
    assert decision.reasons == ()


@given(injection_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_block_exactly_at_or_above_the_attack_block_threshold(injection_score: float) -> None:
    thresholds = _thresholds()
    decision = DefaultPolicyEngine(thresholds).route(
        _clean_input(thresholds, injection_score=injection_score)
    )
    assert (decision.route == Route.BLOCK) == (injection_score >= thresholds.attack_block)


@given(
    scan_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    fraud_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    injection_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_no_route_to_auto_when_any_flag_is_at_or_above_threshold(
    scan_score: float, fraud_score: float, injection_score: float
) -> None:
    thresholds = _thresholds()
    decision = DefaultPolicyEngine(thresholds).route(
        _clean_input(
            thresholds,
            scan_score=scan_score,
            fraud_score=fraud_score,
            injection_score=injection_score,
        )
    )
    any_above = (
        scan_score >= thresholds.scan_review
        or fraud_score >= thresholds.fraud_review
        or injection_score >= thresholds.attack_block
    )
    if any_above:
        assert decision.route != Route.AUTO
        assert decision.reasons


def test_auto_only_when_every_condition_clears() -> None:
    thresholds = _thresholds()
    decision = DefaultPolicyEngine(thresholds).route(_clean_input(thresholds))
    assert decision.route == Route.AUTO
    # Perturb exactly one condition at a time; each alone must break AUTO.
    assert (
        DefaultPolicyEngine(thresholds).route(_clean_input(thresholds, all_accepted=False)).route
        != Route.AUTO
    )
    assert (
        DefaultPolicyEngine(thresholds)
        .route(_clean_input(thresholds, has_validation_error=True))
        .route
        != Route.AUTO
    )
    assert (
        DefaultPolicyEngine(thresholds).route(_clean_input(thresholds, no_vendor=True)).route
        != Route.AUTO
    )


def test_review_when_trust_gate_rejects_a_field() -> None:
    thresholds = _thresholds()
    decision = DefaultPolicyEngine(thresholds).route(_clean_input(thresholds, all_accepted=False))
    assert decision.route == Route.REVIEW


def test_review_when_validation_has_an_error() -> None:
    thresholds = _thresholds()
    decision = DefaultPolicyEngine(thresholds).route(
        _clean_input(thresholds, has_validation_error=True)
    )
    assert decision.route == Route.REVIEW


def test_review_when_vendor_has_no_unique_verified_account() -> None:
    thresholds = _thresholds()
    decision = DefaultPolicyEngine(thresholds).route(_clean_input(thresholds, no_vendor=True))
    assert decision.route == Route.REVIEW


@given(days_ago=st.integers(min_value=0, max_value=60))
def test_bank_change_review_window_boundary(days_ago: int) -> None:
    thresholds = _thresholds()
    vendor = make_vendor()
    change = make_bank_change(vendor_id=str(vendor.id), changed_at=_AT - timedelta(days=days_ago))
    decision = DefaultPolicyEngine(thresholds).route(
        _clean_input(thresholds, vendor=vendor, bank_changes=(change,))
    )
    if days_ago < thresholds.bank_change_review_days:
        assert decision.route == Route.REVIEW
    else:
        assert decision.route == Route.AUTO


@given(
    extra=st.decimals(
        min_value="0.01", max_value="1000", places=2, allow_nan=False, allow_infinity=False
    )
)
def test_above_the_approval_limit_routes_to_review(extra: Decimal) -> None:
    thresholds = _thresholds()
    total = thresholds.approval_limit.amount + extra
    decision = DefaultPolicyEngine(thresholds).route(_clean_input(thresholds, total=str(total)))
    assert decision.route == Route.REVIEW


def test_a_different_currency_routes_to_review_even_under_the_limit() -> None:
    thresholds = _thresholds()
    decision = DefaultPolicyEngine(thresholds).route(
        _clean_input(thresholds, total="10", currency="USD")
    )
    assert decision.route == Route.REVIEW
