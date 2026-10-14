"""Default PolicyEngine: pure P1-P4 routing and capability + segregation-of-duties
authorisation, with no hidden state and no override path (ADR-0008 K5/K7)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from apx.common.errors import PayeeRefusal, PayeeRefusedError
from apx.common.types import Money, Route
from apx.policy.interfaces import (
    ActionRequest,
    AuthorizationDecision,
    Payee,
    PayeeMatch,
    PolicyReason,
    PolicyRule,
    PolicyThresholds,
    RoutingDecision,
    RoutingInput,
    ValidationFinding,
    capabilities_for,
    is_approval_action,
    pin_payee,
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

        errors = tuple(
            ValidationFinding(code=v.code, field=v.field)
            for v in inp.validation.violations
            if v.severity == "error"
        )
        if errors:
            # Codes and field paths are the validator's own text, never the document's.
            listed = ", ".join(
                f.code.value if f.field is None else f"{f.code.value} ({f.field})" for f in errors
            )
            reasons.append(
                PolicyReason(
                    rule=PolicyRule.VALIDATION,
                    route=Route.REVIEW,
                    message=f"validation found error-severity violations: {listed}",
                    violations=errors,
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

        payee_reasons, payee = _payee_route_reasons(inp)
        reasons.extend(payee_reasons)

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

        return RoutingDecision(
            route=_most_severe(reasons),
            reasons=tuple(reasons),
            payee=None if payee is None else pin_payee(payee),
        )

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
            invoice_iban = request.invoice.iban if request.invoice is not None else None
            try:
                select_payee(request.vendor, invoice_iban=invoice_iban)
            except PayeeRefusedError as refused:
                reasons.append(_payee_refusal_reason(refused))

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


_INVOICE_IBAN_REFUSALS = frozenset(
    {PayeeRefusal.INVOICE_IBAN_MISSING, PayeeRefusal.INVOICE_IBAN_MALFORMED, PayeeRefusal.NO_MATCH}
)
"""Refusals where the extracted invoice IBAN failed to select among several verified accounts."""

IBAN_EDIT_HINT = (
    "; a reviewer's edit of the IBAN field does not change the selection (only the extracted "
    "IBAN selects), so reject the invoice and ask the supplier for a corrected one"
)
"""Appended to an invoice-IBAN refusal (ADR-0014 Consequences): `edit_field` only appends a
field row, and `select_payee` reads the extracted `invoices.iban`, so a misread IBAN on a
multi-account vendor cannot be fixed in review."""


def _payee_refusal_reason(refused: PayeeRefusedError) -> PolicyReason:
    """P1 REVIEW reason for a refused payee selection; the message is a fixed template plus the
    enum value, never document text (ADR-0008, ADR-0014)."""
    hint = IBAN_EDIT_HINT if refused.refusal in _INVOICE_IBAN_REFUSALS else ""
    return PolicyReason(
        rule=PolicyRule.P1_VERIFIED_IBAN,
        route=Route.REVIEW,
        message=(
            f"no payee could be selected from the admin-verified accounts: {refused.refusal}{hint}"
        ),
        payee_refusal=refused.refusal,
    )


def _payee_route_reasons(inp: RoutingInput) -> tuple[list[PolicyReason], Payee | None]:
    """P1 routing (ADR-0014 §4-5): a refusal, a sole account the invoice IBAN contradicts, and
    any selection among several verified accounts each route REVIEW. Only a sole verified
    account that the invoice matches or does not name adds no reason. Also returns the selected
    payee (None on refusal), which the route decision pins."""
    try:
        payee = select_payee(inp.vendor, invoice_iban=inp.invoice.iban)
    except PayeeRefusedError as refused:
        return [_payee_refusal_reason(refused)], None
    if payee.verified_count >= 2:
        return [
            PolicyReason(
                rule=PolicyRule.P1_VERIFIED_IBAN,
                route=Route.REVIEW,
                message=(
                    "vendor has several admin-verified accounts; the invoice IBAN chose the payee"
                ),
                payee_match=payee.match,
            )
        ], payee
    if payee.match is PayeeMatch.SOLE_MISMATCH:
        return [
            PolicyReason(
                rule=PolicyRule.P1_VERIFIED_IBAN,
                route=Route.REVIEW,
                message=(
                    "the invoice IBAN differs from the vendor's only admin-verified account; "
                    "payment goes to the verified account"
                ),
                payee_match=payee.match,
            )
        ], payee
    return [], payee


def _above_limit(total: Money, limit: Money) -> bool:
    """True when `total` needs APPROVE_ABOVE_LIMIT (P3): a different currency is always treated
    as above limit, since the two amounts cannot be compared directly."""
    return total.currency != limit.currency or total.amount > limit.amount
