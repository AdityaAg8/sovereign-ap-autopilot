"""Default PolicyEngine: pure P1-P4 routing and capability + segregation-of-duties
authorisation, with no hidden state and no override path (PRD §6.3, ADR-0008 K5/K7)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from apx.common.errors import PolicyDeniedError
from apx.common.types import Money, Route
from apx.policy.interfaces import (
    ActionRequest,
    AuthorizationDecision,
    PolicyReason,
    PolicyRule,
    PolicyThresholds,
    RoutingDecision,
    RoutingInput,
    capabilities_for,
    is_approval_action,
    required_capabilities,
    select_payee,
)


def _most_severe(reasons: Sequence[PolicyReason]) -> Route:
    """Block beats review beats auto (RoutingDecision docstring)."""
    routes = {reason.route for reason in reasons}
    if Route.BLOCK in routes:
        return Route.BLOCK
    if Route.REVIEW in routes:
        return Route.REVIEW
    return Route.AUTO


class DefaultPolicyEngine:
    """The `PolicyEngine` implementation: `route` and `authorize` are pure functions of their
    argument and `thresholds` only (ADR-0008)."""

    def __init__(self, thresholds: PolicyThresholds) -> None:
        self._thresholds = thresholds

    def route(self, inp: RoutingInput) -> RoutingDecision:
        """P1-P4 + trust gate + validation routing (K7). Validation pass/fail is read from
        `violations` directly rather than the `ValidationReport.passed` property, which belongs
        to apx.validate and is out of this task's scope."""
        t = self._thresholds
        reasons: list[PolicyReason] = []

        if not inp.trust.all_accepted:
            reasons.append(
                PolicyReason(
                    rule=PolicyRule.TRUST_GATE,
                    route=Route.REVIEW,
                    message="not every field was accepted by the trust gate",
                )
            )

        if any(v.severity == "error" for v in inp.validation.violations):
            reasons.append(
                PolicyReason(
                    rule=PolicyRule.VALIDATION,
                    route=Route.REVIEW,
                    message="validation found an error-severity violation",
                )
            )

        if inp.scan.injection_score >= t.attack_block:
            reasons.append(
                PolicyReason(
                    rule=PolicyRule.P4_FLAGS,
                    route=Route.BLOCK,
                    message="scan injection score reached the block threshold",
                )
            )
        elif any(flag.score >= t.scan_review for flag in inp.scan.flags):
            reasons.append(
                PolicyReason(
                    rule=PolicyRule.P4_FLAGS,
                    route=Route.REVIEW,
                    message="a scan flag reached the review threshold",
                )
            )

        if any(flag.score >= t.fraud_review for flag in inp.fraud.flags):
            reasons.append(
                PolicyReason(
                    rule=PolicyRule.P4_FLAGS,
                    route=Route.REVIEW,
                    message="a fraud flag reached the review threshold",
                )
            )

        try:
            select_payee(inp.vendor)
        except PolicyDeniedError:
            reasons.append(
                PolicyReason(
                    rule=PolicyRule.P1_VERIFIED_IBAN,
                    route=Route.REVIEW,
                    message="vendor has no unique admin-verified account",
                )
            )

        cooldown = timedelta(days=t.bank_change_review_days)
        if any((inp.at - change.changed_at) < cooldown for change in inp.bank_changes):
            reasons.append(
                PolicyReason(
                    rule=PolicyRule.P2_BANK_CHANGE,
                    route=Route.REVIEW,
                    message="vendor had a bank-detail change inside the review window",
                )
            )

        if _above_limit(inp.invoice.total, t.approval_limit):
            reasons.append(
                PolicyReason(
                    rule=PolicyRule.P3_APPROVAL_LIMIT,
                    route=Route.REVIEW,
                    message="invoice total is above the approval limit or in another currency",
                )
            )

        return RoutingDecision(route=_most_severe(reasons), reasons=tuple(reasons))

    def authorize(self, request: ActionRequest) -> AuthorizationDecision:
        """Pure capability + P1-P4 + segregation of duties check (I3, K5). Denies approvals when
        routing is missing or BLOCK; CHANGE_BANK_DETAILS and VERIFY_BANK_DETAILS need the
        capability only. Never raises; callers (PolicyGuard) raise on `allowed=False`."""
        t = self._thresholds
        reasons: list[PolicyReason] = []

        needed = required_capabilities(request, t)
        held = capabilities_for(request.actor.roles)
        missing = needed - held
        if missing:
            names = ", ".join(sorted(cap.value for cap in missing))
            reasons.append(
                PolicyReason(
                    rule=PolicyRule.CAPABILITY,
                    route=Route.REVIEW,
                    message=f"actor is missing capabilities: {names}",
                )
            )

        if is_approval_action(request):
            try:
                select_payee(request.vendor)
            except PolicyDeniedError:
                reasons.append(
                    PolicyReason(
                        rule=PolicyRule.P1_VERIFIED_IBAN,
                        route=Route.REVIEW,
                        message="vendor has no unique admin-verified account",
                    )
                )

            if request.routing is None:
                reasons.append(
                    PolicyReason(
                        rule=PolicyRule.P4_FLAGS,
                        route=Route.REVIEW,
                        message="no routing decision for this invoice (fail closed)",
                    )
                )
            elif request.routing.route is Route.BLOCK:
                reasons.append(
                    PolicyReason(
                        rule=PolicyRule.P4_FLAGS,
                        route=Route.BLOCK,
                        message="invoice is routed BLOCK; there is no override",
                    )
                )

            if request.vendor is not None:
                cooldown = timedelta(days=t.sod_cooldown_days)
                barred = any(
                    change.vendor_id == request.vendor.id
                    and change.changed_by == request.actor.id
                    and (request.at - change.changed_at) < cooldown
                    for change in request.bank_changes
                )
                if barred:
                    reasons.append(
                        PolicyReason(
                            rule=PolicyRule.P2_BANK_CHANGE,
                            route=Route.REVIEW,
                            message=(
                                "actor changed this vendor's bank details inside the "
                                "segregation-of-duties window"
                            ),
                        )
                    )

        return AuthorizationDecision(allowed=not reasons, reasons=tuple(reasons))


def _above_limit(total: Money, limit: Money) -> bool:
    """True when `total` needs APPROVE_ABOVE_LIMIT (P3): a different currency is always treated
    as above limit, since the two amounts cannot be compared directly."""
    return total.currency != limit.currency or total.amount > limit.amount
