"""The review-state activities `InvoiceWorkflow` runs around a human decision: `review_hold`
turns an AUTO route into REVIEW when a reviewer sent the invoice to review before routing, and
`confirm_review` checks that a `review` signal is backed by the database before the workflow acts
on it (an approval row whose digest matches the current invoice, or a recorded rejection). A raw
Temporal signal with no such record never leads to payment: it is audited and the workflow keeps
waiting (REVIEW.md C-03, H-01, H-02). A recorded approval that went stale fails the invoice
instead (`StaleApprovalError`, N-13)."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from pydantic import JsonValue
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from temporalio import activity

from apx.audit.chain import append_in_transaction
from apx.audit.interfaces import AuditEvent, AuditLog
from apx.common.errors import ApxError, PayeeRefusedError
from apx.common.types import Route
from apx.db.enums import DbInvoiceStatus, DecisionAction
from apx.db.models import Approval as ApprovalRow
from apx.db.models import Decision as DecisionRow
from apx.db.models import Invoice as InvoiceRow
from apx.policy.writer import approval_digest, invoice_for_digest, load_vendor

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


class StaleApprovalError(ApxError):
    """A reviewer's approval is recorded (APPROVED, an approvals row with a digest) but the
    invoice or its vendor payee moved since, so the digest no longer matches (or the payee is
    now refused). Deterministic for the same rows: non-retryable, and the workflow ends the
    invoice FAILED with this reason instead of waiting for a decision the API cannot take
    again (an APPROVED invoice can be neither rejected nor re-approved; REVIEW.md N-13)."""


class _Stale:
    """`_refusal`'s verdict for an approval the API did record but that no longer holds."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


class ReviewGuardActivities:
    """Activities over `apx.db` for the review step. `session_factory` must behave like
    `apx.db.session.get_session`: commit on clean exit, roll back on error."""

    def __init__(self, session_factory: SessionFactory, audit: AuditLog) -> None:
        self._session_factory = session_factory
        self._audit = audit

    async def _record(
        self, session: AsyncSession, action: str, invoice_id: str, payload: dict[str, JsonValue]
    ) -> None:
        """Audits inside `session`'s transaction, so the event and any status write it explains
        commit together (M-02)."""
        await append_in_transaction(
            self._audit,
            session,
            AuditEvent(
                actor="workflow", action=action, entity=f"invoice:{invoice_id}", payload=payload
            ),
        )

    @activity.defn(name="review_hold")
    async def review_hold(self, invoice_id: str, route: Route) -> Route:
        """The route the workflow follows: REVIEW instead of AUTO when a `send_to_review`
        decision (the API's pre-route hold) exists, with the status written back to
        AWAITING_REVIEW so the review queue shows it and an approval is possible. BLOCK and
        REVIEW are returned unchanged. Idempotent."""
        if route is not Route.AUTO:
            return route
        async with self._session_factory() as session:
            held = await session.scalar(
                select(DecisionRow.id)
                .where(
                    DecisionRow.invoice_id == invoice_id,
                    DecisionRow.action == DecisionAction.SEND_TO_REVIEW,
                )
                .limit(1)
            )
            if held is None:
                return route
            # Only the APPROVED that `routed` just wrote for AUTO, which no human approved:
            # APPROVED with an approvals row is a reviewer's approval (this is a retry after the
            # first attempt committed and the reviewer then approved) and is never reopened, nor
            # is any other status (AWAITING_REVIEW from that first attempt, BLOCKED after a
            # rejection, FAILED). Audited only when this attempt moved the row (N-09).
            human_approval = (
                select(ApprovalRow.id).where(ApprovalRow.invoice_id == invoice_id).exists()
            )
            reopened = await session.scalar(
                update(InvoiceRow)
                .where(
                    InvoiceRow.id == invoice_id,
                    InvoiceRow.status == DbInvoiceStatus.APPROVED,
                    ~human_approval,
                )
                .values(status=DbInvoiceStatus.AWAITING_REVIEW)
                .returning(InvoiceRow.id)
            )
            if reopened is not None:
                await self._record(
                    session, "review.hold_honoured", invoice_id, {"policy_route": route.value}
                )
        return Route.REVIEW

    @activity.defn(name="confirm_review")
    async def confirm_review(self, invoice_id: str, approved: bool) -> bool:
        """True when the `review` signal is backed: for an approval, the invoice is APPROVED,
        was never rejected, and its newest `approvals` row carries the `approval_digest` of the
        invoice and vendor payee as they are now; for a rejection, a REJECT decision exists (the
        status is then made BLOCKED if the API's write has not landed). Otherwise the signal is
        audited as `review.signal_unbacked` and False returned, and the workflow keeps waiting.
        An approval the API recorded (with a digest) that no longer matches the invoice or payee
        is audited as `review.approval_stale` and raises `StaleApprovalError` (N-13)."""
        async with self._session_factory() as session:
            refusal = await self._refusal(session, invoice_id, approved=approved)
            if isinstance(refusal, _Stale):
                await self._record(
                    session, "review.approval_stale", invoice_id, {"reason": refusal.reason}
                )
            elif refusal is not None:
                await self._record(
                    session,
                    "review.signal_unbacked",
                    invoice_id,
                    {"approved": approved, "reason": refusal},
                )
        # Raised after the session committed the audit entry (an exception inside would roll it
        # back); the workflow's failure path then writes FAILED with this reason.
        if isinstance(refusal, _Stale):
            raise StaleApprovalError(
                f"invoice {invoice_id}: the recorded approval no longer holds: {refusal.reason}"
            )
        return refusal is None

    async def _refusal(
        self, session: AsyncSession, invoice_id: str, *, approved: bool
    ) -> str | _Stale | None:
        row = await session.get(InvoiceRow, invoice_id, with_for_update=True)
        if row is None:
            return "no invoice row"
        rejected = await session.scalar(
            select(DecisionRow.id)
            .where(
                DecisionRow.invoice_id == invoice_id, DecisionRow.action == DecisionAction.REJECT
            )
            .limit(1)
        )
        if not approved:
            if rejected is None:
                return "no reject decision recorded"
            if row.status is DbInvoiceStatus.AWAITING_REVIEW:
                row.status = DbInvoiceStatus.BLOCKED
            return None
        if rejected is not None:
            return "the invoice was rejected"
        if row.status is not DbInvoiceStatus.APPROVED:
            return f"invoice status is {row.status.value}, not approved"
        approval = await session.scalar(
            select(ApprovalRow)
            .where(ApprovalRow.invoice_id == invoice_id)
            .order_by(ApprovalRow.id.desc())
            .limit(1)
        )
        if approval is None:
            return "no approval row"
        if approval.invoice_digest is None:
            # Written before migration 0012: never trusted, but not proof of a moved invoice.
            return "approval row has no digest"
        try:
            current = approval_digest(
                invoice_for_digest(row), await load_vendor(session, row.vendor_id)
            )
        except PayeeRefusedError as exc:
            return _Stale(f"payee refused: {exc.refusal.value}")
        if approval.invoice_digest != current:
            return _Stale("approval digest does not match the current invoice or payee")
        return None
