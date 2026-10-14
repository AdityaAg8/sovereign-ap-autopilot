"""Tests for DefaultPolicyGuard: every authorize outcome is audited (I4), an allowed request
returns a Grant bound to subject_digest, and denials raise the right typed error
(ADR-0008 K6)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.common.errors import (
    PayeeRefusal,
    PayeeRefusedError,
    PolicyDeniedError,
    SegregationOfDutiesError,
)
from apx.common.types import Language, Role, Route
from apx.db.enums import BankDetailSource, DbInvoiceStatus
from apx.db.models import Invoice as InvoiceRow
from apx.db.models import Vendor as VendorRow
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.guard import DefaultPolicyGuard
from apx.policy.interfaces import (
    ActionRequest,
    Capability,
    Grant,
    PolicyThresholds,
    RoutingDecision,
    subject_digest,
)
from apx.policy.writer import SqlPaymentStateWriter
from tests.unit._policy_factories import (
    make_bank_account,
    make_bank_change,
    make_invoice,
    make_user,
    make_vendor,
    money,
)

_AT = datetime(2026, 1, 1, tzinfo=UTC)


class FakeAuditLog:
    """In-memory AuditLog fake: no chain hashing, just records what was appended."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append(self, event: AuditEvent) -> AuditEntry:
        self.events.append(event)
        return AuditEntry(
            id=len(self.events),
            ts=_AT,
            actor=event.actor,
            action=event.action,
            entity=event.entity,
            payload=event.payload,
            prev_hash="0" * 64,
            hash="1" * 64,
        )


def _thresholds() -> PolicyThresholds:
    return PolicyThresholds(
        approval_limit=money("5000"), fraud_review=0.5, scan_review=0.5, attack_block=0.9
    )


def _approval_request(*, actor_roles: frozenset[Role], vendor=None, routing=None) -> ActionRequest:
    vendor = vendor if vendor is not None else make_vendor()
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    return ActionRequest(
        actor=make_user(user_id="u1", roles=actor_roles),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=routing if routing is not None else RoutingDecision(route=Route.AUTO, reasons=()),
    )


def test_allowed_request_returns_a_grant_bound_to_subject_digest() -> None:
    audit = FakeAuditLog()
    guard = DefaultPolicyGuard(DefaultPolicyEngine(_thresholds()), audit)
    request = _approval_request(actor_roles=frozenset({Role.APPROVER}))

    grant = asyncio.run(guard.require(request))

    assert grant.subject_digest == subject_digest(request)
    assert grant.actor == request.actor.id
    assert grant.action == request.action
    assert grant.audit_entry_id == 1
    assert len(audit.events) == 1
    assert audit.events[0].payload["allowed"] is True


def test_denied_request_is_audited_and_raises_policy_denied() -> None:
    audit = FakeAuditLog()
    guard = DefaultPolicyGuard(DefaultPolicyEngine(_thresholds()), audit)
    request = _approval_request(actor_roles=frozenset())  # no capability at all

    with pytest.raises(PolicyDeniedError):
        asyncio.run(guard.require(request))

    assert len(audit.events) == 1
    assert audit.events[0].payload["allowed"] is False


def test_sod_denial_is_audited_and_raises_segregation_of_duties_error() -> None:
    audit = FakeAuditLog()
    guard = DefaultPolicyGuard(DefaultPolicyEngine(_thresholds()), audit)
    vendor = make_vendor()
    change = make_bank_change(vendor_id=str(vendor.id), changed_by="u1", changed_at=_AT)
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    request = ActionRequest(
        actor=make_user(user_id="u1", roles=frozenset({Role.APPROVER})),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_AT,
        routing=RoutingDecision(route=Route.AUTO, reasons=()),
        bank_changes=(change,),
    )

    with pytest.raises(SegregationOfDutiesError):
        asyncio.run(guard.require(request))

    assert len(audit.events) == 1
    assert audit.events[0].payload["allowed"] is False


def test_subject_digest_changes_when_the_invoice_total_changes() -> None:
    """The mechanism behind K6's stale-grant check: editing a digest input after a grant was
    issued must be detectable by recomputing subject_digest."""
    vendor = make_vendor()
    request_a = _approval_request(actor_roles=frozenset({Role.APPROVER}), vendor=vendor)
    invoice_b = make_invoice(total="999", vendor_id=vendor.id)
    request_b = request_a.model_copy(update={"invoice": invoice_b})

    assert subject_digest(request_a) != subject_digest(request_b)


def test_subject_digest_is_stable_for_an_unchanged_request() -> None:
    request = _approval_request(actor_roles=frozenset({Role.APPROVER}))
    assert subject_digest(request) == subject_digest(request)


def test_subject_digest_omits_payee_iban_for_a_non_approval_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F8: CHANGE_BANK_DETAILS needs only the capability, not the routing/P1 payee gate
    (ADR-0008 K5) -- it is not an approval action, so `subject_digest` must never resolve or
    include a payee IBAN for it, even with no vendor at all. `select_payee(None)` raises
    `PolicyDeniedError` (see test_policy_payee.py), so a real, unmocked call would fail this
    request outright; monkeypatching confirms it is never reached."""

    def _must_not_be_called(vendor: object, *, invoice_iban: object) -> None:
        raise AssertionError("select_payee must not be called for a non-approval action")

    monkeypatch.setattr("apx.policy.interfaces.select_payee", _must_not_be_called)

    change = make_bank_change(vendor_id="vendor-1", changed_by="admin-1", changed_at=_AT)
    request = ActionRequest(
        actor=make_user(user_id="admin-1", roles=frozenset({Role.ADMIN})),
        action=Capability.CHANGE_BANK_DETAILS,
        invoice=None,
        vendor=None,
        at=_AT,
        bank_change=change,
    )

    digest = subject_digest(request)  # must not raise: proves select_payee was never invoked

    expected_subject = {
        "action": "change_bank_details",
        "invoice_id": None,
        "total_amount": None,
        "total_currency": None,
        "vendor_id": None,
        "bank_change": {
            "vendor_id": change.vendor_id,
            "old_iban": change.old_iban,
            "new_iban": change.new_iban,
        },
        "proposal_id": None,
        "proposal_payload": None,
    }
    assert "payee_iban" not in expected_subject
    assert "payee_match" not in expected_subject
    blob = json.dumps(expected_subject, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    assert digest == hashlib.sha256(blob).hexdigest()


# --- ADR-0014 §6: the grant binds the payee selection ------------------------------------------

_A = "AE070331234567890000001"
_B = "AE070331234567890000002"
_C = "AE460090000000123456789"


def _with_iban(request: ActionRequest, iban: str | None) -> ActionRequest:
    assert request.invoice is not None
    return request.model_copy(update={"invoice": request.invoice.model_copy(update={"iban": iban})})


def test_subject_digest_binds_the_normalized_payee_iban_and_match() -> None:
    vendor = make_vendor(accounts=(make_bank_account(iban=_A),))
    request = _with_iban(
        _approval_request(actor_roles=frozenset({Role.APPROVER}), vendor=vendor), None
    )
    # The digest uses the electronic form, so a print-form invoice IBAN that still matches the
    # same account with the same match gives the same digest.
    matched = _with_iban(request, _A)
    assert subject_digest(matched) == subject_digest(
        _with_iban(request, "ae07 0331 2345 6789 0000 001")
    )
    # A match, an absent claim and a mismatch are three different subjects.
    unclaimed = subject_digest(request)
    mismatched = subject_digest(_with_iban(request, _C))
    assert len({subject_digest(matched), unclaimed, mismatched}) == 3


def test_subject_digest_changes_when_a_second_verified_account_is_selected() -> None:
    """Admin verifies the account the invoice names between grant and write: the payee moves
    from the sole verified A (SOLE_MISMATCH) to B (MATCHED), so the old grant goes stale."""
    before_vendor = make_vendor(
        accounts=(
            make_bank_account(iban=_A),
            make_bank_account(iban=_B, verified=False, source="unverified"),
        )
    )
    after_vendor = make_vendor(accounts=(make_bank_account(iban=_A), make_bank_account(iban=_B)))
    before = _with_iban(
        _approval_request(actor_roles=frozenset({Role.APPROVER}), vendor=before_vendor), _B
    )
    after = before.model_copy(update={"vendor": after_vendor})
    assert subject_digest(before) != subject_digest(after)


def test_subject_digest_changes_when_the_invoice_iban_stops_matching() -> None:
    vendor = make_vendor(accounts=(make_bank_account(iban=_A),))
    request = _approval_request(actor_roles=frozenset({Role.APPROVER}), vendor=vendor)
    assert subject_digest(_with_iban(request, _A)) != subject_digest(_with_iban(request, _C))


def test_subject_digest_changes_when_the_invoice_iban_selects_another_verified_account() -> None:
    vendor = make_vendor(accounts=(make_bank_account(iban=_A), make_bank_account(iban=_B)))
    request = _approval_request(actor_roles=frozenset({Role.APPROVER}), vendor=vendor)
    assert subject_digest(_with_iban(request, _A)) != subject_digest(_with_iban(request, _B))


def test_subject_digest_raises_the_typed_refusal_for_an_approval() -> None:
    vendor = make_vendor(accounts=(make_bank_account(iban=_A), make_bank_account(iban=_B)))
    request = _with_iban(
        _approval_request(actor_roles=frozenset({Role.APPROVER}), vendor=vendor), None
    )
    with pytest.raises(PayeeRefusedError) as refused:
        subject_digest(request)
    assert refused.value.refusal is PayeeRefusal.INVOICE_IBAN_MISSING


def test_denied_payee_refusal_is_audited_with_its_typed_reason() -> None:
    audit = FakeAuditLog()
    guard = DefaultPolicyGuard(DefaultPolicyEngine(_thresholds()), audit)
    vendor = make_vendor(accounts=(make_bank_account(iban=_A), make_bank_account(iban=_B)))
    request = _with_iban(
        _approval_request(actor_roles=frozenset({Role.APPROVER}), vendor=vendor), _C
    )

    with pytest.raises(PolicyDeniedError):
        asyncio.run(guard.require(request))

    reasons = audit.events[0].payload["reasons"]
    assert isinstance(reasons, list)
    p1 = [r for r in reasons if isinstance(r, dict) and r["rule"] == "P1"]
    assert p1 == [
        {
            "rule": "P1",
            "route": "review",
            "message": p1[0]["message"],
            "payee_refusal": "no_match",
        }
    ]
    others = [r for r in reasons if isinstance(r, dict) and r["rule"] != "P1"]
    assert all("payee_refusal" not in r and "payee_match" not in r for r in others)


def test_allowed_grant_audit_carries_no_payee_fields_without_a_p1_reason() -> None:
    audit = FakeAuditLog()
    guard = DefaultPolicyGuard(DefaultPolicyEngine(_thresholds()), audit)
    asyncio.run(guard.require(_approval_request(actor_roles=frozenset({Role.APPROVER}))))
    assert audit.events[0].payload["reasons"] == []


class _FakeWriterSession:
    """The few AsyncSession calls `SqlPaymentStateWriter._apply_approval` makes, over in-memory
    rows: enough to drive the real digest recomputation without a database."""

    def __init__(self, invoice_row: SimpleNamespace, vendor_row: SimpleNamespace) -> None:
        self.rows = {InvoiceRow: invoice_row, VendorRow: vendor_row}
        self.added: list[object] = []

    async def get(self, model: type, key: object, **_: object) -> SimpleNamespace | None:
        return self.rows.get(model)

    async def scalars(self, _statement: object) -> SimpleNamespace:
        return SimpleNamespace(all=lambda: [Role.APPROVER])

    async def scalar(self, _statement: object) -> None:
        return None  # the REJECT-decision lookup: this invoice was never rejected

    def add(self, row: object) -> None:
        self.added.append(row)

    async def execute(self, _statement: object) -> None:
        return None


def _db_state(invoice_iban: str | None, accounts: list[tuple[str, bool]]) -> _FakeWriterSession:
    vendor = make_vendor()
    invoice = make_invoice(total="100", vendor_id=vendor.id)
    invoice_row = SimpleNamespace(
        id=invoice.id,
        document_sha256=invoice.document_sha256,
        language=Language.EN,
        number=invoice.number,
        issue_date=date(2026, 1, 1),
        due_date=None,
        vendor_id=str(vendor.id),
        po_number=None,
        total=Decimal("100"),
        subtotal=Decimal("100"),
        vat_total=Decimal("0"),
        currency="AED",
        iban=invoice_iban,
        kind=None,
        status=DbInvoiceStatus.AWAITING_REVIEW,
    )
    bank_details = [
        SimpleNamespace(
            iban=iban,
            verified=ok,
            verification=SimpleNamespace(verified_by="admin-1", verified_at=_AT) if ok else None,
            source=BankDetailSource.ADMIN_CALLBACK if ok else BankDetailSource.UNVERIFIED,
            is_current=True,
        )
        for iban, ok in accounts
    ]
    vendor_row = SimpleNamespace(
        id=str(vendor.id), name=vendor.name, trn=None, bank_details=bank_details
    )
    return _FakeWriterSession(invoice_row, vendor_row)


def _grant_for(session: _FakeWriterSession) -> Grant:
    """Issue a grant through the real guard from the same projection the writer rebuilds."""
    from apx.policy.writer import invoice_for_digest, load_vendor

    row = session.rows[InvoiceRow]
    vendor = asyncio.run(load_vendor(session, row.vendor_id))  # type: ignore[arg-type]
    request = ActionRequest(
        actor=make_user(user_id="u1", roles=frozenset({Role.APPROVER})),
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice_for_digest(row),  # type: ignore[arg-type]
        vendor=vendor,
        at=_AT,
        routing=RoutingDecision(route=Route.REVIEW, reasons=()),
    )
    return asyncio.run(
        DefaultPolicyGuard(DefaultPolicyEngine(_thresholds()), FakeAuditLog()).require(request)
    )


def _writer(session: _FakeWriterSession) -> SqlPaymentStateWriter:
    @asynccontextmanager
    async def factory() -> AsyncIterator[_FakeWriterSession]:
        yield session

    return SqlPaymentStateWriter(factory, FakeAuditLog())  # type: ignore[arg-type]


def test_writer_accepts_the_grant_when_the_payee_selection_is_unchanged() -> None:
    """Positive control for the stale cases below: same state, the approval is written."""
    session = _db_state(_B, [(_A, True), (_B, True)])
    grant = _grant_for(session)
    asyncio.run(_writer(session).record_approval(session.rows[InvoiceRow].id, grant))
    assert len(session.added) == 1


def test_writer_rejects_a_stale_grant_when_a_second_account_is_verified() -> None:
    session = _db_state(_B, [(_A, True), (_B, False)])
    grant = _grant_for(session)  # sole verified A, SOLE_MISMATCH
    session.rows[VendorRow].bank_details[1].verified = True  # admin verifies B: now MATCHED B
    session.rows[VendorRow].bank_details[1].source = BankDetailSource.ADMIN_CALLBACK
    session.rows[VendorRow].bank_details[1].verification = SimpleNamespace(
        verified_by="admin-1", verified_at=_AT
    )
    with pytest.raises(PolicyDeniedError, match="stale_grant"):
        asyncio.run(_writer(session).record_approval(session.rows[InvoiceRow].id, grant))
    assert session.added == []


def test_writer_rejects_a_stale_grant_when_the_invoice_iban_selects_another_account() -> None:
    session = _db_state(_A, [(_A, True), (_B, True)])
    grant = _grant_for(session)
    session.rows[InvoiceRow].iban = _B
    with pytest.raises(PolicyDeniedError, match="stale_grant"):
        asyncio.run(_writer(session).record_approval(session.rows[InvoiceRow].id, grant))
    assert session.added == []


def test_writer_rejects_a_stale_grant_when_a_match_flips_to_a_mismatch() -> None:
    session = _db_state(_A, [(_A, True)])
    grant = _grant_for(session)
    session.rows[InvoiceRow].iban = _C
    with pytest.raises(PolicyDeniedError, match="stale_grant"):
        asyncio.run(_writer(session).record_approval(session.rows[InvoiceRow].id, grant))
    assert session.added == []


def test_writer_propagates_the_typed_refusal_when_the_selection_now_refuses() -> None:
    session = _db_state(_A, [(_A, True), (_B, True)])
    grant = _grant_for(session)
    session.rows[InvoiceRow].iban = None
    with pytest.raises(PayeeRefusedError) as refused:
        asyncio.run(_writer(session).record_approval(session.rows[InvoiceRow].id, grant))
    assert refused.value.refusal is PayeeRefusal.INVOICE_IBAN_MISSING
    assert session.added == []
