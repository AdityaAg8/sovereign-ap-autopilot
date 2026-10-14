"""Server-side reconciler for lost review signals (KNOWN_ISSUES): the API records a decision and
then signals the invoice's workflow, so a signal lost to a Temporal outage leaves the database
decided (APPROVED with a digest-matched approval, or BLOCKED by a REJECT decision) while the
workflow waits in review forever. Run periodically in the worker process, it finds those invoices
and re-sends the decision. Idempotent (the workflow ignores a signal once it holds a decision, and
`confirm_review` still backs every signal against the database), audited (`review.signal_resent`),
and bounded per pass and per invoice. An approval that went stale (payee or invoice moved since)
is re-sent too, so `confirm_review` ends the invoice FAILED with its typed reason instead of
leaving it APPROVED forever."""

from __future__ import annotations

import asyncio
import enum
import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import JsonValue
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.service import RPCError, RPCStatusCode

from apx.audit.chain import append_in_transaction
from apx.audit.interfaces import AuditEvent, AuditLog
from apx.common.errors import PayeeRefusedError
from apx.common.types import InvoiceId
from apx.db.enums import DbInvoiceStatus, DecisionAction
from apx.db.models import Approval as ApprovalRow
from apx.db.models import AuditLog as AuditLogRow
from apx.db.models import Decision as DecisionRow
from apx.db.models import Field as FieldRow
from apx.db.models import Invoice as InvoiceRow
from apx.policy.writer import approval_digest, invoice_for_digest, load_vendor
from apx.review.corrections import corrections_for_rows
from apx.workflows.corrections import MissingCorrectionsError, ReviewCorrections
from apx.workflows.interfaces import InvoiceStatus
from apx.workflows.workflow import InvoiceWorkflow, ReviewDecision

log = logging.getLogger("apx.reconciler")

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

RESENT_ACTION: Final[str] = "review.signal_resent"
"""Audit action of every re-send attempt; its count per invoice bounds the attempts."""
_ACTOR: Final[str] = "reconciler"


class WorkflowReviewState(enum.Enum):
    """Where an invoice's workflow stands, as far as a re-send is concerned."""

    WAITING = "waiting"
    """Running, in review (the only state a re-send is for)."""
    BUSY = "busy"
    """Running, not (yet) in review; check again next pass."""
    CLOSED = "closed"
    """Completed, failed, terminated...: it never waits again."""
    MISSING = "missing"
    """No workflow with this id."""


@dataclass(frozen=True)
class Candidate:
    """An invoice whose recorded decision may not have reached its workflow."""

    invoice_id: InvoiceId
    approved: bool


@dataclass(frozen=True)
class Resend:
    """The signal to re-send for a candidate, and why it is expected to fail the invoice
    rather than advance it (`stale`), which the re-send's audit entry records."""

    decision: ReviewDecision
    stale: str | None = None


class ReviewDecisionStore(Protocol):
    async def candidates(self, *, grace: timedelta, window: timedelta) -> Sequence[Candidate]:
        """APPROVED invoices with an approval, and BLOCKED ones with a REJECT decision, whose
        decision is older than `grace` and newer than `window`; oldest decision first."""
        ...

    async def resend_counts(
        self, invoice_ids: Sequence[InvoiceId], *, window: timedelta
    ) -> dict[InvoiceId, int]:
        """The number of re-sends already audited for each invoice within `window`."""
        ...

    async def decision(self, candidate: Candidate) -> Resend | None:
        """The signal to re-send, as the API would send it, or None when the database records
        no such decision (status moved, no approval or rejection). A recorded approval that no
        longer matches the invoice or payee is still re-sent, marked `stale`."""
        ...

    async def record_resend(
        self,
        candidate: Candidate,
        *,
        delivered: bool,
        error: str | None,
        stale: str | None = None,
    ) -> None:
        """Audits one re-send attempt (`RESENT_ACTION`)."""
        ...


class ReviewDelivery(Protocol):
    async def state(self, invoice_id: InvoiceId) -> WorkflowReviewState: ...

    async def signal(self, invoice_id: InvoiceId, decision: ReviewDecision) -> None:
        """Delivers the `review` signal; raises on failure."""
        ...


class TemporalReviewDelivery:
    """`ReviewDelivery` over a Temporal `Client` carrying the Pydantic data converter."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def state(self, invoice_id: InvoiceId) -> WorkflowReviewState:
        handle = self._client.get_workflow_handle(invoice_id)
        try:
            description = await handle.describe()
        except RPCError as exc:
            if exc.status is RPCStatusCode.NOT_FOUND:
                return WorkflowReviewState.MISSING
            raise
        if description.status is not WorkflowExecutionStatus.RUNNING:
            return WorkflowReviewState.CLOSED
        state = await handle.query(InvoiceWorkflow.status)
        if state.status is InvoiceStatus.AWAITING_REVIEW:
            return WorkflowReviewState.WAITING
        return WorkflowReviewState.BUSY

    async def signal(self, invoice_id: InvoiceId, decision: ReviewDecision) -> None:
        await self._client.get_workflow_handle(invoice_id).signal(InvoiceWorkflow.review, decision)


class PgReviewDecisionStore:
    """`ReviewDecisionStore` over `apx.db`. `session_factory` behaves like
    `apx.db.session.get_session` (commit on clean exit, roll back on error)."""

    def __init__(self, session_factory: SessionFactory, audit: AuditLog) -> None:
        self._session_factory = session_factory
        self._audit = audit

    async def candidates(self, *, grace: timedelta, window: timedelta) -> Sequence[Candidate]:
        now = func.now()
        approved_at = (
            select(ApprovalRow.invoice_id, func.max(ApprovalRow.approved_at).label("at"))
            .group_by(ApprovalRow.invoice_id)
            .subquery()
        )
        rejected_at = (
            select(DecisionRow.invoice_id, func.max(DecisionRow.created_at).label("at"))
            .where(DecisionRow.action == DecisionAction.REJECT)
            .group_by(DecisionRow.invoice_id)
            .subquery()
        )
        async with self._session_factory() as session:
            approved = (
                await session.execute(
                    select(InvoiceRow.id, approved_at.c.at)
                    .join(approved_at, approved_at.c.invoice_id == InvoiceRow.id)
                    .where(
                        InvoiceRow.status == DbInvoiceStatus.APPROVED,
                        approved_at.c.at < now - grace,
                        approved_at.c.at > now - window,
                    )
                )
            ).all()
            rejected = (
                await session.execute(
                    select(InvoiceRow.id, rejected_at.c.at)
                    .join(rejected_at, rejected_at.c.invoice_id == InvoiceRow.id)
                    .where(
                        InvoiceRow.status == DbInvoiceStatus.BLOCKED,
                        rejected_at.c.at < now - grace,
                        rejected_at.c.at > now - window,
                    )
                )
            ).all()
        dated = [(at, Candidate(InvoiceId(i), True)) for i, at in approved]
        dated += [(at, Candidate(InvoiceId(i), False)) for i, at in rejected]
        dated.sort(key=lambda pair: (pair[0], pair[1].invoice_id))
        return [candidate for _, candidate in dated]

    async def resend_counts(
        self, invoice_ids: Sequence[InvoiceId], *, window: timedelta
    ) -> dict[InvoiceId, int]:
        if not invoice_ids:
            return {}
        entities = {f"invoice:{invoice_id}": invoice_id for invoice_id in invoice_ids}
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(AuditLogRow.entity, func.count(AuditLogRow.id))
                    .where(
                        AuditLogRow.action == RESENT_ACTION,
                        AuditLogRow.ts > func.now() - window,
                        AuditLogRow.entity.in_(entities),
                    )
                    .group_by(AuditLogRow.entity)
                )
            ).all()
        return {entities[entity]: int(count) for entity, count in rows}

    async def decision(self, candidate: Candidate) -> Resend | None:
        async with self._session_factory() as session:
            row = await session.get(InvoiceRow, candidate.invoice_id)
            if row is None:
                return None
            if not candidate.approved:
                rejected = await session.scalar(
                    select(DecisionRow.id)
                    .where(
                        DecisionRow.invoice_id == candidate.invoice_id,
                        DecisionRow.action == DecisionAction.REJECT,
                    )
                    .limit(1)
                )
                backed = row.status is DbInvoiceStatus.BLOCKED and rejected is not None
                return Resend(ReviewDecision(approved=False)) if backed else None
            if row.status is not DbInvoiceStatus.APPROVED:
                return None
            approval = await session.scalar(
                select(ApprovalRow)
                .where(ApprovalRow.invoice_id == candidate.invoice_id)
                .order_by(ApprovalRow.id.desc())
                .limit(1)
            )
            if approval is None:
                return None
            current: str | PayeeRefusedError
            try:
                current = approval_digest(
                    invoice_for_digest(row), await load_vendor(session, row.vendor_id)
                )
            except PayeeRefusedError as exc:
                current = exc
            rows = (
                await session.scalars(
                    select(FieldRow)
                    .where(FieldRow.invoice_id == candidate.invoice_id)
                    .order_by(FieldRow.id)
                )
            ).all()
        return approval_resend(approval.invoice_digest, current, rows)

    async def record_resend(
        self,
        candidate: Candidate,
        *,
        delivered: bool,
        error: str | None,
        stale: str | None = None,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "approved": candidate.approved,
            "delivered": delivered,
            "error": error,
            "stale": stale,
        }
        async with self._session_factory() as session:
            await append_in_transaction(
                self._audit,
                session,
                AuditEvent(
                    actor=_ACTOR,
                    action=RESENT_ACTION,
                    entity=f"invoice:{candidate.invoice_id}",
                    payload=payload,
                ),
            )


def approval_resend(
    recorded: str | None, current: str | PayeeRefusedError, rows: Sequence[FieldRow]
) -> Resend:
    """The approval to re-send for an APPROVED invoice, given the newest approval's
    `invoice_digest` (`recorded`), the digest of the invoice and payee now (or why the payee is
    refused) and the invoice's `fields` rows. Always a re-send: when the approval no longer
    holds, `confirm_review` raises `StaleApprovalError` and the workflow ends the invoice FAILED
    with that reason, the outcome the lost signal would have had; withholding it would
    leave the invoice APPROVED and its workflow waiting forever. Corrections are those
    a repeated approve would send; when the rows no longer derive complete ones, none are sent
    and `confirm_review` finds the approval stale on its corrections."""
    stale: str | None = None
    if isinstance(current, PayeeRefusedError):
        stale = f"payee refused: {current.refusal.value}"
    elif recorded is None:
        stale = "approval row has no digest"
    elif recorded != current:
        stale = "approval digest does not match the current invoice or payee"
    corrections: ReviewCorrections | None
    try:
        corrections = corrections_for_rows(rows)
    except MissingCorrectionsError as exc:
        corrections = None
        stale = stale or f"the invoice's corrections are incomplete: {exc}"
    return Resend(ReviewDecision(approved=True, corrections=corrections), stale=stale)


@dataclass(frozen=True)
class ReconcilerSettings:
    interval: timedelta = timedelta(minutes=1)
    """Time between passes."""
    grace: timedelta = timedelta(minutes=2)
    """A decision younger than this is left to the API's own signal (and its retry)."""
    window: timedelta = timedelta(days=7)
    """Decisions older than this are not reconsidered, which bounds every pass's query."""
    batch: int = 50
    """Most re-sends per pass. Only workflows found WAITING count, so candidates whose workflow
    is busy (e.g. an approved invoice retrying `send_to_asp`) never starve a lost signal."""
    max_describes: int = 500
    """Most workflows described per pass, which bounds a pass's Temporal calls. Each pass
    starts where the previous one stopped, so more candidates than this are all visited in
    turn."""
    max_resends: int = 5
    """Most re-sends per invoice within `window`; an invoice whose signal keeps being refused
    (audited `review.signal_unbacked` by `confirm_review`) is not re-sent forever."""
    settled_cache: int = 100_000
    """Most invoice ids remembered as settled (workflow closed or missing) in this process."""


class ReviewSignalReconciler:
    """One pass (`reconcile_once`) or a loop (`run`) re-sending lost review decisions."""

    def __init__(
        self,
        store: ReviewDecisionStore,
        delivery: ReviewDelivery,
        settings: ReconcilerSettings | None = None,
    ) -> None:
        self._store = store
        self._delivery = delivery
        self._settings = settings or ReconcilerSettings()
        self._settled: OrderedDict[InvoiceId, None] = OrderedDict()
        self._cursor = 0

    def _settle(self, invoice_id: InvoiceId) -> None:
        self._settled[invoice_id] = None
        while len(self._settled) > self._settings.settled_cache:
            self._settled.popitem(last=False)

    async def reconcile_once(self) -> list[InvoiceId]:
        """One bounded pass; returns the invoices whose decision was re-sent (delivered)."""
        s = self._settings
        found = await self._store.candidates(grace=s.grace, window=s.window)
        open_ = [c for c in found if c.invoice_id not in self._settled]
        counts = await self._store.resend_counts([c.invoice_id for c in open_], window=s.window)
        todo = [c for c in open_ if counts.get(c.invoice_id, 0) < s.max_resends]
        start = self._cursor % len(todo) if todo else 0
        ordered = todo[start:] + todo[:start]
        resent: list[InvoiceId] = []
        described = 0
        for candidate in ordered:
            if len(resent) >= s.batch or described >= s.max_describes:
                break
            described += 1
            try:
                if await self._reconcile(candidate):
                    resent.append(candidate.invoice_id)
            except (RPCError, RuntimeError, OSError) as exc:
                # Temporal or the network is unreachable: not this invoice's fault, so it is not
                # counted against `max_resends` (an outage is what the reconciler exists for) and
                # stays a candidate for the next pass.
                log.warning("could not reconcile invoice %s: %s", candidate.invoice_id, exc)
            except Exception as exc:
                # Anything specific to this invoice (its query fails or is rejected, e.g. a
                # workflow task stuck on nondeterminism; its rows cannot be read) is audited and
                # counted, so it is bounded by `max_resends` and never aborts the pass for the
                # others.
                log.warning(
                    "reconciling invoice %s failed: %s: %s",
                    candidate.invoice_id,
                    type(exc).__name__,
                    exc,
                )
                await self._store.record_resend(
                    candidate, delivered=False, error=f"{type(exc).__name__}: {exc}"
                )
        # The next pass starts after the last candidate looked at, so a pass's worth of busy
        # workflows at the head of the list cannot hide the ones behind it.
        self._cursor = start + described
        return resent

    async def _reconcile(self, candidate: Candidate) -> bool:
        """Re-sends `candidate`'s decision if its workflow still waits in review; True if sent."""
        state = await self._delivery.state(candidate.invoice_id)
        if state in (WorkflowReviewState.CLOSED, WorkflowReviewState.MISSING):
            self._settle(candidate.invoice_id)
            return False
        if state is not WorkflowReviewState.WAITING:
            return False
        resend = await self._store.decision(candidate)
        if resend is None:
            return False
        try:
            await self._delivery.signal(candidate.invoice_id, resend.decision)
        except (RPCError, RuntimeError, OSError) as exc:
            log.warning("re-send to invoice %s failed: %s", candidate.invoice_id, exc)
            await self._store.record_resend(
                candidate, delivered=False, error=str(exc), stale=resend.stale
            )
            return False
        await self._store.record_resend(candidate, delivered=True, error=None, stale=resend.stale)
        log.info(
            "re-sent the %s decision to invoice %s",
            "approve" if candidate.approved else "reject",
            candidate.invoice_id,
        )
        return True

    async def run(self) -> None:
        """Passes every `interval` until cancelled. A failed pass (Postgres or Temporal down) is
        logged and retried at the next interval; it never stops the worker."""
        while True:
            try:
                await self.reconcile_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("review signal reconciliation pass failed")
            await asyncio.sleep(self._settings.interval.total_seconds())
