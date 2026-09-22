"""Default PolicyGuard: authorizes an ActionRequest through a PolicyEngine, audits every
outcome (I4) and issues a Grant bound to the current subject digest (ADR-0008 K6)."""

from __future__ import annotations

from pydantic import JsonValue

from apx.audit.interfaces import AuditEvent, AuditLog
from apx.common.errors import PolicyDeniedError, SegregationOfDutiesError
from apx.policy.interfaces import (
    ActionRequest,
    Grant,
    PolicyEngine,
    PolicyReason,
    PolicyRule,
    subject_digest,
)


def _entity_for(request: ActionRequest) -> str:
    if request.invoice is not None:
        return f"invoice:{request.invoice.id}"
    if request.vendor is not None:
        return f"vendor:{request.vendor.id}"
    return "action"


def _reasons_payload(reasons: tuple[PolicyReason, ...]) -> list[JsonValue]:
    return [{"rule": r.rule.value, "route": r.route.value, "message": r.message} for r in reasons]


class DefaultPolicyGuard:
    """The `PolicyGuard` implementation: the only path allowed to turn an authorized
    `ActionRequest` into a `Grant` a `PaymentStateWriter` will accept."""

    def __init__(self, engine: PolicyEngine, audit: AuditLog) -> None:
        self._engine = engine
        self._audit = audit

    async def require(self, request: ActionRequest) -> Grant:
        """Authorize through `PolicyEngine`, append an audit event for allowed and denied
        outcomes (I4) and return a `Grant` bound to `subject_digest(request)`. Raises
        `SegregationOfDutiesError` for P2 bars and `PolicyDeniedError` for every other denial."""
        decision = self._engine.authorize(request)
        entry = await self._audit.append(
            AuditEvent(
                actor=str(request.actor.id),
                action=f"authorize:{request.action.value}",
                entity=_entity_for(request),
                payload={
                    "allowed": decision.allowed,
                    "reasons": _reasons_payload(decision.reasons),
                },
            )
        )

        if not decision.allowed:
            message = "; ".join(r.message for r in decision.reasons) or "denied"
            if any(r.rule is PolicyRule.P2_BANK_CHANGE for r in decision.reasons):
                raise SegregationOfDutiesError(
                    f"actor {request.actor.id} is barred by segregation of duties: {message}"
                )
            raise PolicyDeniedError(f"action {request.action.value} denied: {message}")

        return Grant(
            actor=request.actor.id,
            action=request.action,
            subject_digest=subject_digest(request),
            reasons=decision.reasons,
            audit_entry_id=entry.id,
            issued_at=request.at,
        )
