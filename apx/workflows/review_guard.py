"""The review-state activities `InvoiceWorkflow` runs around a human decision: `review_hold`
turns an AUTO route into REVIEW when a reviewer sent the invoice to review before routing, and
`confirm_review` checks that a `review` signal is backed by the database before the workflow acts
on it (an approval row whose digest matches the current invoice, or a recorded rejection). A raw
Temporal signal with no such record never leads to payment: it is audited and the workflow keeps
waiting (REVIEW.md C-03, H-01, H-02)."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from pydantic import JsonValue
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from temporalio import activity

from apx.audit.chain import append_in_transaction
from apx.audit.interfaces import AuditEvent, AuditLog
from apx.common.errors import PayeeRefusedError
from apx.common.types import Route
from apx.db.enums import DbInvoiceStatus, DecisionAction
from apx.db.models import Approval as ApprovalRow
from apx.db.models import Decision as DecisionRow
from apx.db.models import Invoice as InvoiceRow
from apx.policy.writer import approval_digest, invoice_for_digest, load_vendor

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


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
            # Only the status `routed` just wrote for AUTO; anything else (a retry after the
            # reviewer already approved) is left alone.
            await session.execute(
                update(InvoiceRow)
                .where(
                    InvoiceRow.id == invoice_id,
                    InvoiceRow.status == DbInvoiceStatus.APPROVED,
                )
                .values(status=DbInvoiceStatus.AWAITING_REVIEW)
            )
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
        audited as `review.signal_unbacked` and False returned, and the workflow keeps waiting."""
        async with self._session_factory() as session:
            refusal = await self._refusal(session, invoice_id, approved=approved)
            if refusal is not None:
                await self._record(
                    session,
                    "review.signal_unbacked",
                    invoice_id,
                    {"approved": approved, "reason": refusal},
                )
        return refusal is None

    async def _refusal(
        self, session: AsyncSession, invoice_id: str, *, approved: bool
    ) -> str | None:
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
        try:
            current = approval_digest(
                invoice_for_digest(row), await load_vendor(session, row.vendor_id)
            )
        except PayeeRefusedError as exc:
            return f"payee refused: {exc.refusal.value}"
        if approval.invoice_digest != current:
            return "approval digest does not match the current invoice"
        return None
