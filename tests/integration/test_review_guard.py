"""M11.2 (REVIEW.md C-01, C-03, H-01, H-02, M-06, L-04) against a throwaway Postgres: every review
transition through the API (reject only while awaiting review, send-to-review only before routing,
both 409 otherwise), the workflow's `review_hold` and `confirm_review` activities over the real
rows (a signal with no digest-matched approval is never backed), proposal acceptance that either
acts or is refused, and the role an approval is recorded under. M11.3: a stale recorded approval
raises `StaleApprovalError` (N-13), a retried hold never reopens an approval (N-09), a repeated
reject re-sends its signal (N-02), and decisions, edits and accepted proposals commit with their
audit entry (N-07). M11.4: an approval signal's corrections must match the ones recorded with the
approval, and an edit after the approval makes it stale (N-03). Reuses `test_api_authz.py`'s
helpers on its own container name and port."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

import apx.db.session as db_session
from apx.api.interfaces import DecisionRequest, FieldEdit
from apx.api.service import DbInvoiceService
from apx.api.settings import get_settings
from apx.audit.chain import PgAuditLog
from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.common.errors import AuditChainError
from apx.common.invoice import InvoiceKind
from apx.common.master import User
from apx.common.types import InvoiceId, Money, Role, Route, UserId
from apx.db.enums import DbInvoiceStatus, DecisionAction, FieldSource, ProposalKind, ProposalStatus
from apx.db.models import Approval, AuditLog, Decision, Invoice, Proposal
from apx.db.models import Field as FieldRow
from apx.db.session import get_session
from apx.ingest.settings import get_ingest_settings
from apx.policy.writer import SqlPaymentStateWriter
from apx.review.corrections import corrections_digest
from apx.workflows.reconciler import Candidate, PgReviewDecisionStore, Resend
from apx.workflows.review_guard import ReviewGuardActivities, StaleApprovalError
from apx.workflows.workflow import ReviewDecision
from tests.integration.test_api_authz import (
    APPROVAL_LIMIT,
    PG_DB,
    PG_PASSWORD,
    PG_USER,
    _alembic_config,
    _app,
    _auth,
    _create_invoice,
    _create_user,
    _create_verified_vendor,
    _invoice_id,
    _login,
    _RecordingSignaller,
    _seed_roles,
    _wait_for_postgres,
)

pytestmark = pytest.mark.integration

CONTAINER_NAME = "apx-test-postgres-review-guard"
HOST_PORT = 55462
DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
IBAN = "AE070331234567890123456"
VENDOR = "vendorRG"


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, check=False)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"127.0.0.1:{HOST_PORT}:5432",
            "-e",
            f"POSTGRES_USER={PG_USER}",
            "-e",
            f"POSTGRES_PASSWORD={PG_PASSWORD}",
            "-e",
            f"POSTGRES_DB={PG_DB}",
            "pgvector/pgvector:pg16-trixie",
        ],
        check=True,
        capture_output=True,
    )
    try:
        asyncio.run(_wait_for_postgres(ASYNCPG_DSN))
        yield DATABASE_URL
    finally:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture
def api_schema(postgres_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[str]:
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    monkeypatch.setenv("API_JWT_SECRET", "integration-test-secret-value-long-enough-for-hs256")
    monkeypatch.setenv("API_APPROVAL_LIMIT", str(APPROVAL_LIMIT))
    monkeypatch.setenv("API_SOD_COOLDOWN_DAYS", "30")
    monkeypatch.setenv("INGEST_STORE_DIR", str(tmp_path / "documents"))
    db_session.get_engine.cache_clear()
    db_session._session_factory.cache_clear()
    get_settings.cache_clear()
    get_ingest_settings.cache_clear()
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    try:
        yield postgres_url
    finally:
        command.downgrade(cfg, "base")
        db_session.get_engine.cache_clear()
        db_session._session_factory.cache_clear()
        get_settings.cache_clear()
        get_ingest_settings.cache_clear()


async def _seed(
    invoices: dict[str, tuple[Route | None, DbInvoiceStatus, Decimal]],
) -> None:
    async with get_session() as session:
        await _seed_roles(session)
        await _create_user(session, user_id="rev", password="pw", roles=[Role.REVIEWER])
        await _create_user(session, user_id="appr", password="pw", roles=[Role.APPROVER])
        await _create_user(
            session, user_id="both", password="pw", roles=[Role.REVIEWER, Role.APPROVER]
        )
        await _create_user(session, user_id="adm", password="pw", roles=[Role.ADMIN])
        await _create_verified_vendor(
            session,
            vendor_id=VENDOR,
            iban=IBAN,
            verified_by="adm",
            changed_by="adm",
            changed_at=datetime.now(UTC) - timedelta(days=365),
        )
        for invoice_id, (route, status, total) in invoices.items():
            await _create_invoice(
                session,
                invoice_id=invoice_id,
                vendor_id=VENDOR,
                total=total,
                route=route,
                status=status,
            )


async def _status(invoice_id: str) -> DbInvoiceStatus:
    async with get_session() as session:
        row = await session.get(Invoice, invoice_id)
        assert row is not None
        return row.status


async def _decisions(invoice_id: str) -> list[DecisionAction]:
    async with get_session() as session:
        rows = await session.scalars(
            select(Decision.action).where(Decision.invoice_id == invoice_id).order_by(Decision.id)
        )
        return list(rows.all())


async def _audit_actions(entity: str) -> list[str]:
    async with get_session() as session:
        rows = await session.scalars(
            select(AuditLog.action).where(AuditLog.entity == entity).order_by(AuditLog.id)
        )
        return list(rows.all())


_HUNDRED = Decimal("100.00")


async def _transitions_scenario() -> None:
    review = _invoice_id("rg-review")
    in_flight = _invoice_id("rg-in-flight")
    await _seed(
        {
            review: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED),
            in_flight: (None, DbInvoiceStatus.FRAUD_CHECKED, _HUNDRED),
        }
    )
    signaller = _RecordingSignaller()
    transport = ASGITransport(app=_app(signaller))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")

        async def decide(invoice_id: str, action: str) -> int:
            response = await client.post(
                f"/invoices/{invoice_id}/decisions", json={"action": action}, headers=_auth(token)
            )
            return response.status_code

        # Awaiting review: send_to_review is not a pre-route state (409); reject lands.
        assert await decide(review, "send_to_review") == 409
        assert await decide(review, "reject") == 204
        assert await _status(review) is DbInvoiceStatus.BLOCKED
        assert await _decisions(review) == [DecisionAction.REJECT]
        assert signaller.sent == [(review, False)]
        # H-01: the rejection is terminal: a reopen and an approval refuse. N-02: repeating the
        # reject records nothing new and only re-sends its signal.
        assert await decide(review, "reject") == 204
        assert signaller.sent == [(review, False), (review, False)]
        assert await decide(review, "send_to_review") == 409
        assert await decide(review, "approve") == 403
        assert await _decisions(review) == [DecisionAction.REJECT]
        assert await _status(review) is DbInvoiceStatus.BLOCKED
        assert (await _audit_actions(f"invoice:{review}")).count("decision.reject") == 2

        # In flight: reject is refused (C-01); send_to_review is a durable hold only.
        assert await decide(in_flight, "reject") == 409
        assert await decide(in_flight, "send_to_review") == 204
        assert await _status(in_flight) is DbInvoiceStatus.FRAUD_CHECKED
        assert await _decisions(in_flight) == [DecisionAction.SEND_TO_REVIEW]
        assert signaller.sent == [(review, False), (review, False)]
    await db_session.get_engine().dispose()


def test_review_decisions_follow_the_state_machine(api_schema: str) -> None:
    asyncio.run(_transitions_scenario())


async def _confirm_scenario() -> None:
    approved = _invoice_id("rg-approved")
    unapproved = _invoice_id("rg-unapproved")
    rejected = _invoice_id("rg-rejected")
    await _seed(
        {
            approved: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED),
            unapproved: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED),
            rejected: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED),
        }
    )
    guard = ReviewGuardActivities(get_session, PgAuditLog())
    transport = ASGITransport(app=_app(_RecordingSignaller()))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")
        for invoice_id, action in ((approved, "approve"), (rejected, "reject")):
            response = await client.post(
                f"/invoices/{invoice_id}/decisions", json={"action": action}, headers=_auth(token)
            )
            assert response.status_code == 204, response.text

    # Backed decisions go through.
    assert await guard.confirm_review(approved, True) is True
    assert await guard.confirm_review(rejected, False) is True
    # C-03: a raw approval signal with no approval row, or a raw reject with no recorded
    # rejection, is not backed; the invoice stays awaiting review and the attempt is audited.
    assert await guard.confirm_review(unapproved, True) is False
    assert await guard.confirm_review(unapproved, False) is False
    assert await _status(unapproved) is DbInvoiceStatus.AWAITING_REVIEW
    assert (await _audit_actions(f"invoice:{unapproved}")).count("review.signal_unbacked") == 2
    # A rejected invoice is never confirmed as approved.
    assert await guard.confirm_review(rejected, True) is False

    # The approval row's digest must still match. N-13: a recorded approval whose invoice moved
    # is not merely unbacked (nobody could re-approve it): it raises, audited with its reason.
    async with get_session() as session:
        await session.execute(
            update(Invoice).where(Invoice.id == approved).values(total=Decimal("99999.00"))
        )
    with pytest.raises(StaleApprovalError, match="digest does not match"):
        await guard.confirm_review(approved, True)
    assert "review.approval_stale" in await _audit_actions(f"invoice:{approved}")
    async with get_session() as session:
        await session.execute(update(Invoice).where(Invoice.id == approved).values(total=_HUNDRED))
    assert await guard.confirm_review(approved, True) is True
    # A row with no digest (written before migration 0012) is not trusted either.
    async with get_session() as session:
        await session.execute(
            update(Approval).where(Approval.invoice_id == approved).values(invoice_digest=None)
        )
    assert await guard.confirm_review(approved, True) is False
    await db_session.get_engine().dispose()


def test_confirm_review_needs_a_digest_matched_approval_or_a_recorded_rejection(
    api_schema: str,
) -> None:
    asyncio.run(_confirm_scenario())


async def _corrections_scenario() -> None:
    """N-03 (KI-02): an approval records the digest of the corrections it signals; the guard
    backs a signal only with exactly those corrections, and an edit of the corrections after
    the approval makes it stale. An accepted approve_payment proposal records its digest too."""
    approved = _invoice_id("rg-corr-approved")
    legacy = _invoice_id("rg-corr-legacy")
    proposed = _invoice_id("rg-corr-proposed")
    await _seed(
        {
            approved: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED),
            legacy: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED),
            proposed: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED),
        }
    )
    async with get_session() as session:
        session.add(
            Proposal(
                id="p-approve",
                invoice_id=proposed,
                kind=ProposalKind.APPROVE_PAYMENT,
                payload={},
                created_by="agent",
                status=ProposalStatus.PENDING,
            )
        )
    signaller = _RecordingSignaller()
    transport = ASGITransport(app=_app(signaller))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")
        for invoice_id in (approved, legacy):
            response = await client.post(
                f"/invoices/{invoice_id}/decisions",
                json={"action": "approve"},
                headers=_auth(token),
            )
            assert response.status_code == 204, response.text
        response = await client.post("/proposals/p-approve/accept", headers=_auth(token))
        assert response.status_code == 200, response.text
    signalled = dict(zip([i for i, _ in signaller.sent], signaller.corrections, strict=True))
    sent = signalled[approved]
    assert sent is not None

    async def recorded(invoice_id: str) -> str | None:
        async with get_session() as session:
            return await session.scalar(
                select(Approval.corrections_digest).where(Approval.invoice_id == invoice_id)
            )

    assert await recorded(approved) == corrections_digest(sent)
    proposed_sent = signalled[proposed]
    assert proposed_sent is not None
    assert await recorded(proposed) == corrections_digest(proposed_sent)

    guard = ReviewGuardActivities(get_session, PgAuditLog())
    assert await guard.confirm_review(approved, True, sent, True) is True
    assert await guard.confirm_review(proposed, True, proposed_sent, True) is True
    # A raw signal whose corrections differ from the approved ones (or carry none) is unbacked:
    # audited, and the invoice stays approved for the real signal.
    forged = sent.model_copy(update={"kind": InvoiceKind.CREDIT_NOTE})
    assert await guard.confirm_review(approved, True, forged, True) is False
    assert await guard.confirm_review(approved, True, None, True) is False
    assert await _status(approved) is DbInvoiceStatus.APPROVED
    assert (await _audit_actions(f"invoice:{approved}")).count("review.signal_unbacked") == 2

    # A row written before migration 0014 has no corrections digest: the current rows stand in.
    async with get_session() as session:
        await session.execute(
            update(Approval).where(Approval.invoice_id == legacy).values(corrections_digest=None)
        )
    legacy_sent = signalled[legacy]
    assert legacy_sent is not None
    assert await guard.confirm_review(legacy, True, legacy_sent, True) is True
    assert await guard.confirm_review(legacy, True, forged, True) is False

    # An edit of a correction after the approval invalidates it, whichever signal arrives.
    transport = ASGITransport(app=_app(_RecordingSignaller()))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")
        response = await client.patch(
            f"/invoices/{approved}/fields",
            json={
                "field": "supplier_address",
                "value": "Office 7, Al Quoz, Dubai",
                "reason": "after approval",
            },
            headers=_auth(token),
        )
        assert response.status_code == 204, response.text
    with pytest.raises(StaleApprovalError, match="edited after the approval"):
        await guard.confirm_review(approved, True, sent, True)
    assert "review.approval_stale" in await _audit_actions(f"invoice:{approved}")
    # A history scheduled before the patch does not ask for the check (unchanged behaviour).
    assert await guard.confirm_review(approved, True) is True
    await db_session.get_engine().dispose()


def test_confirm_review_checks_the_signal_corrections_against_the_approval(
    api_schema: str,
) -> None:
    asyncio.run(_corrections_scenario())


async def _reconciler_scenario() -> None:
    """N-01/N-02 follow-up: the reconciler's store finds an APPROVED invoice and a rejected
    BLOCKED one whose decision is older than the grace period, rebuilds the signal the API would
    send (a backed one: `confirm_review` accepts it), skips undecided or fresh decisions, and
    counts its audited re-sends."""
    approved = InvoiceId(_invoice_id("rc-approved"))
    rejected = InvoiceId(_invoice_id("rc-rejected"))
    fresh = _invoice_id("rc-fresh")
    waiting = InvoiceId(_invoice_id("rc-waiting"))
    await _seed(
        {
            invoice_id: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED)
            for invoice_id in (approved, rejected, fresh, waiting)
        }
    )
    signaller = _RecordingSignaller()
    transport = ASGITransport(app=_app(signaller))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")
        for invoice_id, action in ((approved, "approve"), (rejected, "reject"), (fresh, "approve")):
            response = await client.post(
                f"/invoices/{invoice_id}/decisions", json={"action": action}, headers=_auth(token)
            )
            assert response.status_code == 204, response.text
    an_hour_ago = datetime.now(UTC) - timedelta(hours=1)
    async with get_session() as session:
        await session.execute(
            update(Approval).where(Approval.invoice_id == approved).values(approved_at=an_hour_ago)
        )
        await session.execute(
            update(Decision)
            .where(Decision.invoice_id == rejected, Decision.action == DecisionAction.REJECT)
            .values(created_at=an_hour_ago)
        )

    store = PgReviewDecisionStore(get_session, PgAuditLog())
    found = await store.candidates(grace=timedelta(minutes=2), window=timedelta(days=7))
    assert set(found) == {Candidate(approved, True), Candidate(rejected, False)}

    resend = await store.decision(Candidate(approved, True))
    assert resend is not None and resend.stale is None
    approval = resend.decision
    assert approval.approved
    sent = dict(zip([i for i, _ in signaller.sent], signaller.corrections, strict=True))
    assert approval.corrections == sent[approved]
    guard = ReviewGuardActivities(get_session, PgAuditLog())
    assert await guard.confirm_review(approved, True, approval.corrections, True) is True
    assert await store.decision(Candidate(rejected, False)) == Resend(
        ReviewDecision(approved=False)
    )
    # A candidate the database does not back (never decided) yields no signal.
    assert await store.decision(Candidate(waiting, True)) is None
    assert await store.decision(Candidate(waiting, False)) is None

    await store.record_resend(Candidate(approved, True), delivered=True, error=None)
    await store.record_resend(Candidate(approved, True), delivered=False, error="unreachable")
    counts = await store.resend_counts([approved, rejected], window=timedelta(days=7))
    assert counts == {approved: 2}

    # P3-01: an approval gone stale (its digest no longer matches) is still re-sent, marked
    # stale, so `confirm_review` fails the invoice instead of leaving it APPROVED forever.
    async with get_session() as session:
        await session.execute(
            update(Approval).where(Approval.invoice_id == fresh).values(invoice_digest="0" * 64)
        )
    stale = await store.decision(Candidate(InvoiceId(fresh), True))
    assert stale is not None and stale.decision.approved
    assert stale.stale is not None and "does not match" in stale.stale
    with pytest.raises(StaleApprovalError):
        await guard.confirm_review(fresh, True, stale.decision.corrections, True)
    await db_session.get_engine().dispose()


def test_the_reconciler_store_finds_and_rebuilds_lost_decisions(api_schema: str) -> None:
    asyncio.run(_reconciler_scenario())


async def _hold_scenario() -> None:
    held = _invoice_id("rg-held")
    free = _invoice_id("rg-free")
    await _seed(
        {
            held: (None, DbInvoiceStatus.FRAUD_CHECKED, _HUNDRED),
            free: (None, DbInvoiceStatus.FRAUD_CHECKED, _HUNDRED),
        }
    )
    transport = ASGITransport(app=_app(_RecordingSignaller()))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")
        response = await client.post(
            f"/invoices/{held}/decisions", json={"action": "send_to_review"}, headers=_auth(token)
        )
        assert response.status_code == 204, response.text
    # What the `route` activity's `routed` write leaves for an AUTO route.
    async with get_session() as session:
        await session.execute(
            update(Invoice)
            .where(Invoice.id.in_([held, free]))
            .values(route=Route.AUTO, status=DbInvoiceStatus.APPROVED)
        )
    guard = ReviewGuardActivities(get_session, PgAuditLog())

    assert await guard.review_hold(held, Route.AUTO) is Route.REVIEW
    assert await _status(held) is DbInvoiceStatus.AWAITING_REVIEW
    assert "review.hold_honoured" in await _audit_actions(f"invoice:{held}")
    assert await guard.review_hold(free, Route.AUTO) is Route.AUTO
    assert await _status(free) is DbInvoiceStatus.APPROVED
    # A hold never softens BLOCK.
    assert await guard.review_hold(held, Route.BLOCK) is Route.BLOCK

    # N-09: a retried hold (its first attempt committed) after the reviewer approved never
    # reopens the approval, and the hold is audited once.
    transport = ASGITransport(app=_app(_RecordingSignaller()))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "appr", "pw")
        response = await client.post(
            f"/invoices/{held}/decisions", json={"action": "approve"}, headers=_auth(token)
        )
        assert response.status_code == 204, response.text
    assert await _status(held) is DbInvoiceStatus.APPROVED
    assert await guard.review_hold(held, Route.AUTO) is Route.REVIEW
    assert await _status(held) is DbInvoiceStatus.APPROVED
    assert (await _audit_actions(f"invoice:{held}")).count("review.hold_honoured") == 1
    await db_session.get_engine().dispose()


def test_a_pre_route_hold_turns_auto_into_review(api_schema: str) -> None:
    asyncio.run(_hold_scenario())


async def _role_scenario() -> None:
    above = _invoice_id("rg-above-limit")
    within = _invoice_id("rg-within-limit")
    await _seed(
        {
            above: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, APPROVAL_LIMIT + 1),
            within: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED),
        }
    )
    transport = ASGITransport(app=_app(_RecordingSignaller()))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        both = await _login(client, "both", "pw")
        reviewer = await _login(client, "rev", "pw")
        for invoice_id, token in ((above, both), (within, reviewer)):
            response = await client.post(
                f"/invoices/{invoice_id}/decisions",
                json={"action": "approve"},
                headers=_auth(token),
            )
            assert response.status_code == 204, response.text
    async with get_session() as session:
        rows = await session.execute(select(Approval.invoice_id, Approval.role_at_approval))
        roles = {invoice_id: role for invoice_id, role in rows.all()}
    # L-04: above the limit only the approver role grants the approval, so that is recorded.
    assert roles == {above: Role.APPROVER, within: Role.REVIEWER}
    await db_session.get_engine().dispose()


def test_role_at_approval_is_the_role_that_granted_it(api_schema: str) -> None:
    asyncio.run(_role_scenario())


async def _proposal_scenario() -> None:
    invoice_id = _invoice_id("rg-proposals")
    await _seed({invoice_id: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED)})
    proposals = {
        "p-route": (ProposalKind.ROUTE, {"route": "review"}),
        "p-info": (ProposalKind.REQUEST_INFO, {"subject": "Missing PO", "body": "Please send"}),
        "p-edit": (ProposalKind.EDIT_FIELD, {"field": "po_number", "value": "PO-77"}),
        "p-bad-edit": (ProposalKind.EDIT_FIELD, {"field": "po_number"}),
    }
    async with get_session() as session:
        for proposal_id, (kind, payload) in proposals.items():
            session.add(
                Proposal(
                    id=proposal_id,
                    invoice_id=invoice_id,
                    kind=kind,
                    payload=payload,
                    created_by="agent",
                    status=ProposalStatus.PENDING,
                )
            )
    transport = ASGITransport(app=_app(_RecordingSignaller()))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _login(client, "rev", "pw")
        codes = {
            proposal_id: (
                await client.post(f"/proposals/{proposal_id}/accept", headers=_auth(token))
            ).status_code
            for proposal_id in proposals
        }
    assert codes == {"p-route": 409, "p-info": 409, "p-edit": 200, "p-bad-edit": 409}
    async with get_session() as session:
        status_rows = await session.execute(select(Proposal.id, Proposal.status))
        statuses = {proposal_id: status for proposal_id, status in status_rows.all()}
        edited = (
            await session.scalars(
                select(FieldRow.value).where(
                    FieldRow.invoice_id == invoice_id,
                    FieldRow.name == "po_number",
                    FieldRow.source == FieldSource.HUMAN,
                )
            )
        ).all()
    # M-06: refused proposals stay pending; the accepted edit really happened, audited.
    assert statuses == {
        "p-route": ProposalStatus.PENDING,
        "p-info": ProposalStatus.PENDING,
        "p-edit": ProposalStatus.EXECUTED,
        "p-bad-edit": ProposalStatus.PENDING,
    }
    assert list(edited) == ["PO-77"]
    assert "field_edit" in await _audit_actions(f"invoice:{invoice_id}:field:po_number")
    assert await _audit_actions("proposal:p-route") == []
    await db_session.get_engine().dispose()


def test_accepting_a_proposal_acts_or_is_refused(api_schema: str) -> None:
    asyncio.run(_proposal_scenario())


class _FailingAudit(PgAuditLog):
    """A chain whose in-transaction append fails, as a lost connection or lock timeout would."""

    async def append_in(self, session: AsyncSession, event: AuditEvent) -> AuditEntry:
        raise AuditChainError("injected audit failure")


async def _audit_rollback_scenario() -> None:
    invoice_id = _invoice_id("rg-audit-rollback")
    await _seed({invoice_id: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED)})
    service = DbInvoiceService(signaller=_RecordingSignaller())
    service._writer = SqlPaymentStateWriter(
        get_session,
        _FailingAudit(),
        approval_limit=Money(amount=APPROVAL_LIMIT, currency="AED"),
    )
    reviewer = User(id=UserId("rev"), display_name="rev", roles=frozenset({Role.REVIEWER}))

    with pytest.raises(AuditChainError):
        await service.decide(InvoiceId(invoice_id), DecisionRequest(action="approve"), reviewer)

    # M-02: the approval and its `payment_state:approve` entry commit together or not at all.
    async with get_session() as session:
        approvals = (
            await session.scalars(select(Approval.id).where(Approval.invoice_id == invoice_id))
        ).all()
    assert approvals == []
    assert await _status(invoice_id) is DbInvoiceStatus.AWAITING_REVIEW
    assert "payment_state:approve" not in await _audit_actions(f"invoice:{invoice_id}")
    await db_session.get_engine().dispose()


def test_a_failed_audit_append_rolls_back_the_payment_state_write(api_schema: str) -> None:
    asyncio.run(_audit_rollback_scenario())


async def _decision_audit_rollback_scenario() -> None:
    rejected = _invoice_id("rg-audit-rollback-reject")
    await _seed({rejected: (Route.REVIEW, DbInvoiceStatus.AWAITING_REVIEW, _HUNDRED)})
    async with get_session() as session:
        session.add(
            Proposal(
                id="p-audit-edit",
                invoice_id=rejected,
                kind=ProposalKind.EDIT_FIELD,
                payload={"field": "po_number", "value": "PO-9"},
                created_by="agent",
                status=ProposalStatus.PENDING,
            )
        )
    signaller = _RecordingSignaller()
    service = DbInvoiceService(signaller=signaller)
    service._audit = _FailingAudit()
    reviewer = User(id=UserId("rev"), display_name="rev", roles=frozenset({Role.REVIEWER}))
    invoice_id = InvoiceId(rejected)

    # N-07: each change and its audit entry commit together or not at all.
    with pytest.raises(AuditChainError):
        await service.decide(invoice_id, DecisionRequest(action="reject"), reviewer)
    assert await _status(rejected) is DbInvoiceStatus.AWAITING_REVIEW
    assert await _decisions(rejected) == []
    assert signaller.sent == []

    with pytest.raises(AuditChainError):
        await service.edit_field(
            invoice_id, FieldEdit(field="po_number", value="PO-1", reason="typo"), reviewer
        )
    with pytest.raises(AuditChainError):
        await service.accept_proposal("p-audit-edit", reviewer)
    async with get_session() as session:
        human_fields = (
            await session.scalars(
                select(FieldRow.id).where(
                    FieldRow.invoice_id == rejected, FieldRow.source == FieldSource.HUMAN
                )
            )
        ).all()
        proposal = await session.get(Proposal, "p-audit-edit")
    assert human_fields == []
    assert proposal is not None and proposal.status == ProposalStatus.PENDING
    assert await _decisions(rejected) == []
    # Only the policy guard's own authorization entry (a separate log) exists: no decision,
    # edit or acceptance entry was written for a change that did not commit.
    assert await _audit_actions(f"invoice:{rejected}") == ["authorize:execute_proposal"]
    assert await _audit_actions(f"invoice:{rejected}:field:po_number") == []
    assert await _audit_actions("proposal:p-audit-edit") == []
    await db_session.get_engine().dispose()


def test_decisions_edits_and_accepted_proposals_commit_with_their_audit_entry(
    api_schema: str,
) -> None:
    asyncio.run(_decision_audit_rollback_scenario())
