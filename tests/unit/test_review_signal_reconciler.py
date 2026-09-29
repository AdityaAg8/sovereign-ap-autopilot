"""REVIEW.md N-01/N-02 hardening: `ReviewSignalReconciler` re-delivers a recorded review decision
whose signal never reached a workflow still waiting in review (a Temporal outage after the API
committed, or a real signal dropped while an unbacked one was being checked). Fake stores and
deliveries drive the pass logic; a time-skipping Temporal server runs the dropped-signal case
end to end through the real `TemporalReviewDelivery` and `InvoiceWorkflow`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import pytest
from temporalio.client import WorkflowQueryFailedError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from apx.common.errors import PayeeRefusal, PayeeRefusedError
from apx.common.types import InvoiceId, Route
from apx.db.enums import FieldSource
from apx.db.models import Field as FieldRow
from apx.policy.interfaces import RoutingDecision
from apx.review.corrections import corrections_for_rows
from apx.workflows.corrections import SUPPLIER_ADDRESS_FIELD
from apx.workflows.interfaces import InvoiceStatus
from apx.workflows.reconciler import (
    Candidate,
    ReconcilerSettings,
    Resend,
    ReviewSignalReconciler,
    TemporalReviewDelivery,
    WorkflowReviewState,
    approval_resend,
)
from apx.workflows.workflow import InvoiceWorkflow, ReviewDecision
from tests.unit.test_invoice_workflow import (
    INVOICE_ID,
    TASK_QUEUE,
    WORKFLOW_INPUT,
    FakeInvoiceActivities,
    _activities,
    _wait_until,
    temporal_env,
)

__all__ = ["temporal_env"]  # the fixture, re-exported so pytest finds it here

A = InvoiceId("a" * 64)
B = InvoiceId("b" * 64)
C = InvoiceId("c" * 64)


@dataclass
class _Store:
    found: list[Candidate]
    decisions: dict[InvoiceId, ReviewDecision | Resend | None]
    counts: dict[InvoiceId, int] = field(default_factory=dict)
    resends: list[tuple[InvoiceId, bool, str | None]] = field(default_factory=list)
    stale: list[tuple[InvoiceId, str | None]] = field(default_factory=list)

    async def candidates(self, *, grace: timedelta, window: timedelta) -> Sequence[Candidate]:
        return self.found

    async def resend_counts(
        self, invoice_ids: Sequence[InvoiceId], *, window: timedelta
    ) -> dict[InvoiceId, int]:
        return {i: self.counts[i] for i in invoice_ids if i in self.counts}

    async def decision(self, candidate: Candidate) -> Resend | None:
        found = self.decisions.get(candidate.invoice_id)
        return Resend(found) if isinstance(found, ReviewDecision) else found

    async def record_resend(
        self,
        candidate: Candidate,
        *,
        delivered: bool,
        error: str | None,
        stale: str | None = None,
    ) -> None:
        self.resends.append((candidate.invoice_id, delivered, error))
        self.stale.append((candidate.invoice_id, stale))
        self.counts[candidate.invoice_id] = self.counts.get(candidate.invoice_id, 0) + 1


@dataclass
class _Delivery:
    states: dict[InvoiceId, WorkflowReviewState | Exception]
    fail_signal: set[InvoiceId] = field(default_factory=set)
    signals: list[tuple[InvoiceId, ReviewDecision]] = field(default_factory=list)
    described: list[InvoiceId] = field(default_factory=list)

    async def state(self, invoice_id: InvoiceId) -> WorkflowReviewState:
        self.described.append(invoice_id)
        state = self.states[invoice_id]
        if isinstance(state, Exception):
            raise state
        return state

    async def signal(self, invoice_id: InvoiceId, decision: ReviewDecision) -> None:
        if invoice_id in self.fail_signal:
            raise RPCError("temporal unavailable", RPCStatusCode.UNAVAILABLE, b"")
        self.signals.append((invoice_id, decision))


APPROVE = ReviewDecision(approved=True)
REJECT = ReviewDecision(approved=False)


async def test_a_waiting_workflow_gets_its_recorded_decision_again() -> None:
    store = _Store([Candidate(A, True), Candidate(B, False)], {A: APPROVE, B: REJECT})
    delivery = _Delivery({A: WorkflowReviewState.WAITING, B: WorkflowReviewState.WAITING})

    resent = await ReviewSignalReconciler(store, delivery).reconcile_once()

    assert resent == [A, B]
    assert delivery.signals == [(A, APPROVE), (B, REJECT)]
    assert store.resends == [(A, True, None), (B, True, None)]


async def test_only_a_workflow_in_its_review_wait_is_signalled() -> None:
    store = _Store(
        [Candidate(A, True), Candidate(B, True), Candidate(C, True)],
        {A: APPROVE, B: APPROVE, C: APPROVE},
    )
    delivery = _Delivery(
        {
            A: WorkflowReviewState.BUSY,
            B: WorkflowReviewState.CLOSED,
            C: WorkflowReviewState.MISSING,
        }
    )
    reconciler = ReviewSignalReconciler(store, delivery)

    assert await reconciler.reconcile_once() == []
    assert delivery.signals == [] and store.resends == []
    # A closed or missing workflow never waits again: it is not described on later passes,
    # while a busy one is looked at again.
    delivery.described.clear()
    await reconciler.reconcile_once()
    assert delivery.described == [A]


async def test_a_decision_the_database_no_longer_backs_is_not_sent() -> None:
    store = _Store([Candidate(A, True)], {A: None})
    delivery = _Delivery({A: WorkflowReviewState.WAITING})

    assert await ReviewSignalReconciler(store, delivery).reconcile_once() == []
    assert delivery.signals == []


async def test_re_sends_are_bounded_per_invoice_and_failures_audited() -> None:
    store = _Store([Candidate(A, True)], {A: APPROVE})
    delivery = _Delivery({A: WorkflowReviewState.WAITING}, fail_signal={A})
    reconciler = ReviewSignalReconciler(store, delivery, ReconcilerSettings(max_resends=2))

    for _ in range(4):
        assert await reconciler.reconcile_once() == []

    assert store.resends == [(A, False, "temporal unavailable")] * 2


async def test_one_unreadable_workflow_does_not_starve_the_others() -> None:
    store = _Store([Candidate(A, True), Candidate(B, True)], {A: APPROVE, B: APPROVE})
    delivery = _Delivery(
        {
            A: RPCError("deadline exceeded", RPCStatusCode.DEADLINE_EXCEEDED, b""),
            B: WorkflowReviewState.WAITING,
        }
    )

    assert await ReviewSignalReconciler(store, delivery).reconcile_once() == [B]


async def test_a_workflow_whose_query_fails_is_audited_bounded_and_never_aborts_the_pass() -> None:
    # A query that fails inside the workflow (e.g. a task stuck on nondeterminism) raises a
    # TemporalError that is not an RPCError; it must not abort the pass, and since it is this
    # invoice's own failure it counts against `max_resends`.
    failure = WorkflowQueryFailedError("nondeterminism")
    store = _Store([Candidate(A, True), Candidate(B, True)], {A: APPROVE, B: APPROVE})
    delivery = _Delivery({A: failure, B: WorkflowReviewState.WAITING})
    reconciler = ReviewSignalReconciler(store, delivery, ReconcilerSettings(max_resends=2))

    assert await reconciler.reconcile_once() == [B]
    delivery.described.clear()
    for _ in range(3):
        await reconciler.reconcile_once()

    failed = [r for r in store.resends if r[0] == A]
    assert failed == [(A, False, "WorkflowQueryFailedError: nondeterminism")] * 2
    assert delivery.described.count(A) == 1  # bounded: not described after max_resends


async def test_an_unreachable_temporal_does_not_use_up_the_resend_budget() -> None:
    store = _Store([Candidate(A, True)], {A: APPROVE})
    delivery = _Delivery({A: RPCError("unavailable", RPCStatusCode.UNAVAILABLE, b"")})
    reconciler = ReviewSignalReconciler(store, delivery, ReconcilerSettings(max_resends=2))

    for _ in range(4):
        assert await reconciler.reconcile_once() == []

    assert store.resends == []
    assert delivery.described == [A] * 4


async def test_a_pass_is_capped_at_the_batch_size() -> None:
    store = _Store([Candidate(A, True), Candidate(B, True)], {A: APPROVE, B: APPROVE})
    delivery = _Delivery({A: WorkflowReviewState.WAITING, B: WorkflowReviewState.WAITING})

    resent = await ReviewSignalReconciler(
        store, delivery, ReconcilerSettings(batch=1)
    ).reconcile_once()

    assert resent == [A]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "final"),
    [(APPROVE, InvoiceStatus.ARCHIVED), (REJECT, InvoiceStatus.BLOCKED)],
)
async def test_a_dropped_signal_is_redelivered_and_the_workflow_finishes(
    temporal_env: WorkflowEnvironment, decision: ReviewDecision, final: InvoiceStatus
) -> None:
    """The API recorded the decision but its signal never arrived: the workflow waits in review
    until the reconciler, reading the real workflow through Temporal, delivers it."""
    fake = FakeInvoiceActivities(route_decision=RoutingDecision(route=Route.REVIEW, reasons=()))
    async with Worker(
        temporal_env.client,
        task_queue=TASK_QUEUE,
        workflows=[InvoiceWorkflow],
        activities=_activities(fake),
    ):
        handle = await temporal_env.client.start_workflow(
            InvoiceWorkflow.run, WORKFLOW_INPUT, id=INVOICE_ID, task_queue=TASK_QUEUE
        )
        delivery = TemporalReviewDelivery(temporal_env.client)
        invoice_id = InvoiceId(INVOICE_ID)

        async def _waiting() -> bool:
            return await delivery.state(invoice_id) is WorkflowReviewState.WAITING

        await _wait_until(_waiting)
        store = _Store([Candidate(invoice_id, decision.approved)], {invoice_id: decision})
        reconciler = ReviewSignalReconciler(store, delivery)

        assert await reconciler.reconcile_once() == [invoice_id]
        result = await handle.result()
        assert result.status is final
        assert fake.confirmed == [decision.approved]
        # Once the workflow has ended, a further pass leaves it alone.
        assert await delivery.state(invoice_id) is WorkflowReviewState.CLOSED
        assert await reconciler.reconcile_once() == []
        assert len(store.resends) == 1


async def test_busy_workflows_never_crowd_a_lost_signal_out_of_the_batch() -> None:
    """P3-01(b): candidates whose workflow is past its review wait (an approved invoice retrying
    `send_to_asp`) are oldest-first but must not use up the batch: only WAITING ones count."""
    store = _Store(
        [Candidate(A, True), Candidate(B, True), Candidate(C, True)],
        {A: APPROVE, B: APPROVE, C: APPROVE},
    )
    delivery = _Delivery(
        {A: WorkflowReviewState.BUSY, B: WorkflowReviewState.BUSY, C: WorkflowReviewState.WAITING}
    )

    resent = await ReviewSignalReconciler(
        store, delivery, ReconcilerSettings(batch=1)
    ).reconcile_once()

    assert resent == [C]


async def test_passes_rotate_through_more_candidates_than_one_pass_describes() -> None:
    """P3-01(b): describes per pass are capped, and the next pass starts where the last one
    stopped, so busy workflows at the head of the list cannot hide a waiting one forever."""
    store = _Store(
        [Candidate(A, True), Candidate(B, True), Candidate(C, True)],
        {A: APPROVE, B: APPROVE, C: APPROVE},
    )
    delivery = _Delivery(
        {A: WorkflowReviewState.BUSY, B: WorkflowReviewState.BUSY, C: WorkflowReviewState.WAITING}
    )
    reconciler = ReviewSignalReconciler(store, delivery, ReconcilerSettings(max_describes=2))

    assert await reconciler.reconcile_once() == []
    assert delivery.described == [A, B]
    assert await reconciler.reconcile_once() == [C]
    assert delivery.described == [A, B, C, A]  # the second pass starts at C, then wraps


def _rows() -> list[FieldRow]:
    values = {
        "invoice_type_code": "380",
        "currency": "AED",
        "supplier_trn": "100000000000003",
        "line_items[1].vat_category": "S",
        "line_items[1].vat_rate": "5",
        SUPPLIER_ADDRESS_FIELD: "Office 12, Al Quoz, Dubai",
    }
    return [
        FieldRow(
            id=i,
            invoice_id="inv",
            name=name,
            value=value,
            normalized_value=value,
            source=FieldSource.EXTRACTOR_A,
        )
        for i, (name, value) in enumerate(values.items(), start=1)
    ]


def test_a_matching_approval_is_resent_as_the_api_would_send_it() -> None:
    rows = _rows()
    resend = approval_resend("d" * 64, "d" * 64, rows)
    assert resend == Resend(ReviewDecision(approved=True, corrections=corrections_for_rows(rows)))


@pytest.mark.parametrize(
    ("recorded", "current", "reason"),
    [
        ("d" * 64, "e" * 64, "approval digest does not match"),
        (
            "d" * 64,
            PayeeRefusedError(PayeeRefusal.NO_VERIFIED_ACCOUNT, "no verified account"),
            "payee refused",
        ),
    ],
)
def test_a_stale_approval_is_still_resent_so_the_workflow_can_fail_it(
    recorded: str, current: str | PayeeRefusedError, reason: str
) -> None:
    """P3-01(a), N-13: withholding the signal would leave the invoice APPROVED and its workflow
    waiting forever; sent, `confirm_review` raises `StaleApprovalError` and the run ends
    FAILED. The re-send is audited as stale."""
    resend = approval_resend(recorded, current, _rows())
    assert resend.decision.approved is True
    assert resend.stale is not None and reason in resend.stale


def test_incomplete_corrections_are_resent_without_corrections_and_marked_stale() -> None:
    rows = [
        *_rows(),
        FieldRow(
            id=99,
            invoice_id="inv",
            name="currency",
            value=None,
            normalized_value=None,
            source=FieldSource.HUMAN,
        ),
    ]
    resend = approval_resend("d" * 64, "d" * 64, rows)
    assert resend.decision == ReviewDecision(approved=True, corrections=None)
    assert resend.stale is not None and "incomplete" in resend.stale


async def test_a_stale_resend_is_delivered_and_audited_as_stale() -> None:
    stale = Resend(APPROVE, stale="approval digest does not match the current invoice or payee")
    store = _Store([Candidate(A, True)], {A: stale})
    delivery = _Delivery({A: WorkflowReviewState.WAITING})

    assert await ReviewSignalReconciler(store, delivery).reconcile_once() == [A]
    assert delivery.signals == [(A, APPROVE)]
    assert store.stale == [(A, stale.stale)]
