"""`SqlPaymentStateWriter`: the concrete, only mutator of payment state on the Postgres
schema (approvals, bank details, proposal execution). Every method re-derives the current
subject from the database inside its own transaction and compares it against `grant.
subject_digest`, raising `PolicyDeniedError` (rule STALE_GRANT) if reality moved since the
grant was issued; idempotent replays (a retried API call, a duplicated signal) are no-ops
(I3, ADR-0008). Each approval stores `approval_digest`, which the
workflow re-checks before paying, and an invoice rejected once is never approved."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime

from pydantic import JsonValue
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from apx.audit.chain import append_in_transaction
from apx.audit.interfaces import AuditEvent, AuditLog
from apx.common.errors import PolicyDeniedError, SegregationOfDutiesError
from apx.common.invoice import Invoice, InvoiceKind, Party
from apx.common.master import BankAccount, BankDetailChange, User, Vendor
from apx.common.types import InvoiceId, Money, Role, Sha256Hex, UserId, VendorId
from apx.db.enums import BankDetailSource, DbInvoiceStatus, DecisionAction, ProposalStatus
from apx.db.models import Approval as ApprovalRow
from apx.db.models import BankDetail as BankDetailRow
from apx.db.models import BankVerification as BankVerificationRow
from apx.db.models import Decision as DecisionRow
from apx.db.models import Invoice as InvoiceRow
from apx.db.models import Proposal as ProposalRow
from apx.db.models import RoleDef, UserRole
from apx.db.models import Vendor as VendorRow
from apx.policy.interfaces import (
    ROLE_CAPABILITIES,
    ActionRequest,
    Capability,
    Grant,
    PolicyRule,
    above_limit,
    is_approval_action,
    subject_digest,
)
from apx.policy.interfaces import Proposal as ProposalModel

_PLACEHOLDER = User(id=UserId(""), display_name="", roles=frozenset())
"""`subject_digest` excludes `actor`, so the `ActionRequest.actor` this module
builds to recompute a digest never needs to be the real approver -- only `grant.actor` (used
below to resolve `role_at_approval` and to attribute the write) matters."""


def _stale(rule: PolicyRule = PolicyRule.STALE_GRANT) -> PolicyDeniedError:
    return PolicyDeniedError(
        f"grant subject digest no longer matches current state (rule={rule.value})"
    )


async def load_vendor(session: AsyncSession, vendor_id: str | None) -> Vendor | None:
    """Projects the vendor's *current* bank accounts (mirrors `apx.common.master.Vendor`)."""
    if vendor_id is None:
        return None
    row = await session.get(
        VendorRow,
        vendor_id,
        options=[selectinload(VendorRow.bank_details).selectinload(BankDetailRow.verification)],
    )
    if row is None:
        return None
    accounts = tuple(
        BankAccount(
            iban=bd.iban,
            verified=bd.verified,
            verified_by=UserId(bd.verification.verified_by) if bd.verification else None,
            verified_at=bd.verification.verified_at if bd.verification else None,
            source=bd.source.value,
        )
        for bd in row.bank_details
        if bd.is_current
    )
    return Vendor(id=VendorId(row.id), name=row.name, trn=row.trn, bank_accounts=accounts)


def invoice_for_digest(row: InvoiceRow) -> Invoice:
    """Minimal projection of `row` sufficient for `subject_digest` and the fraud history (id,
    total, vendor, number, dates, kind):
    not a general invoice loader. `supplier`/`buyer`/`lines` are placeholders `Invoice` requires
    but `subject_digest` never reads."""
    placeholder_party = Party(name="", trn=None, address=None, country="AE")
    total = Money(amount=row.total, currency=row.currency)
    return Invoice(
        id=InvoiceId(row.id),
        document_sha256=row.document_sha256,
        language=row.language,
        number=row.number,
        issue_date=row.issue_date,
        due_date=row.due_date,
        supplier=placeholder_party,
        buyer=placeholder_party,
        vendor_id=VendorId(row.vendor_id) if row.vendor_id else None,
        po_number=row.po_number,
        lines=(),
        subtotal=Money(amount=row.subtotal, currency=row.currency),
        vat_total=Money(amount=row.vat_total, currency=row.currency),
        total=total,
        iban=row.iban,
        kind=None if row.kind is None else InvoiceKind(row.kind),
    )


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def approval_digest(invoice: Invoice, vendor: Vendor | None) -> Sha256Hex:
    """The payment subject an approval covers: `subject_digest` of a direct APPROVE_PAYMENT over
    `invoice` (id, total, currency) and `vendor` (id, the P1-selected payee IBAN and match).
    Stored on the `approvals` row and re-derived by the workflow's `confirm_review` activity from
    the then-current rows, so the two agree only if nothing that decides the payment moved.
    Raises `PayeeRefusedError` when P1 refuses a payee (no approval can exist then)."""
    request = ActionRequest(
        actor=_PLACEHOLDER,
        action=Capability.APPROVE_PAYMENT,
        invoice=invoice,
        vendor=vendor,
        at=_EPOCH,
    )
    return subject_digest(request)


async def _resolve_role(
    session: AsyncSession, actor: UserId, required: frozenset[Capability]
) -> Role:
    """The actor's current role that grants every capability in `required`, for
    `approvals.role_at_approval`; with several, the first by name, so the record never depends on
    set iteration order. Fails closed if the actor holds no such role any more
    (their roles changed after the grant was issued -- a case the digest itself cannot catch,
    since roles are not part of it)."""
    rows = await session.scalars(
        select(RoleDef.name)
        .join(UserRole, UserRole.role_id == RoleDef.id)
        .where(UserRole.user_id == actor)
    )
    for role in sorted(frozenset(rows.all()), key=lambda r: r.value):
        if required <= ROLE_CAPABILITIES.get(role, frozenset()):
            return role
    names = ", ".join(sorted(c.value for c in required))
    raise PolicyDeniedError(f"actor {actor} no longer holds a role granting {names}")


class SqlPaymentStateWriter:
    """`apx.policy.interfaces.PaymentStateWriter` over `apx.db`. Never imported by `apx.agent`
    (ADR-0008); the only write path for `approvals`, `bank_details`/`bank_verifications` and
    `proposals.status`. `approval_limit` decides which role an approval is recorded under (P3:
    above it, only a role holding APPROVE_ABOVE_LIMIT); None treats every invoice as above the
    limit, failing closed like `required_capabilities` does without an invoice."""

    def __init__(
        self,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        audit: AuditLog,
        *,
        approval_limit: Money | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._audit = audit
        self._approval_limit = approval_limit

    async def _log(
        self,
        session: AsyncSession,
        actor: UserId,
        action: str,
        entity: str,
        payload: dict[str, JsonValue],
    ) -> None:
        """Appends the audit event inside `session`'s transaction: the state write and
        its audit entry commit together or not at all. Called last, after the row locks."""
        await append_in_transaction(
            self._audit,
            session,
            AuditEvent(actor=str(actor), action=action, entity=entity, payload=payload),
        )

    async def record_approval(
        self,
        invoice_id: InvoiceId,
        grant: Grant,
        *,
        corrections_digest: Sha256Hex | None = None,
    ) -> None:
        request = ActionRequest(
            actor=_PLACEHOLDER, action=grant.action, invoice=None, vendor=None, at=grant.issued_at
        )
        if not is_approval_action(request):
            raise PolicyDeniedError(
                f"record_approval requires an approval grant, got {grant.action.value}"
            )
        async with self._session_factory() as session:
            await self._apply_approval(
                session, invoice_id, grant, grant.action, corrections_digest=corrections_digest
            )

    async def _apply_approval(
        self,
        session: AsyncSession,
        invoice_id: InvoiceId,
        grant: Grant,
        capability: Capability,
        proposal: ProposalModel | None = None,
        *,
        corrections_digest: Sha256Hex | None = None,
    ) -> None:
        # Row lock: two concurrent approvals of the same invoice serialise here, so the second
        # sees APPROVED and becomes an idempotent no-op instead of a duplicate approval row.
        row = await session.get(InvoiceRow, invoice_id, with_for_update=True)
        if row is None:
            raise PolicyDeniedError(f"invoice {invoice_id} not found")
        if row.status is DbInvoiceStatus.APPROVED:
            return  # idempotent replay: this invoice is already approved
        rejected = await session.scalar(
            select(DecisionRow.id)
            .where(
                DecisionRow.invoice_id == invoice_id,
                DecisionRow.action == DecisionAction.REJECT,
            )
            .limit(1)
        )
        if rejected is not None:
            # A reviewer's rejection is final; no later approval may override it.
            raise PolicyDeniedError(f"invoice {invoice_id} was rejected; cannot approve")
        if row.status is not DbInvoiceStatus.AWAITING_REVIEW:
            raise PolicyDeniedError(
                f"invoice {invoice_id} is {row.status.value}, not awaiting review; cannot approve"
            )

        invoice = invoice_for_digest(row)
        vendor = await load_vendor(session, row.vendor_id)
        request = ActionRequest(
            actor=_PLACEHOLDER,
            action=grant.action,
            invoice=invoice,
            vendor=vendor,
            at=grant.issued_at,
            proposal=proposal,
        )
        if subject_digest(request) != grant.subject_digest:
            raise _stale()

        required = {capability}
        if self._approval_limit is None or above_limit(invoice.total, self._approval_limit):
            required.add(Capability.APPROVE_ABOVE_LIMIT)
        role = await _resolve_role(session, grant.actor, frozenset(required))
        session.add(
            ApprovalRow(
                invoice_id=invoice_id,
                approver_id=grant.actor,
                role_at_approval=role,
                amount=invoice.total.amount,
                currency=invoice.total.currency,
                invoice_digest=approval_digest(invoice, vendor),
                corrections_digest=corrections_digest,
            )
        )
        await session.execute(
            update(InvoiceRow)
            .where(InvoiceRow.id == invoice_id)
            .values(status=DbInvoiceStatus.APPROVED)
        )
        await self._log(
            session,
            grant.actor,
            "payment_state:approve",
            f"invoice:{invoice_id}",
            {
                "role": role.value,
                "amount": str(invoice.total.amount),
                "currency": invoice.total.currency,
            },
        )

    async def record_bank_change(self, change: BankDetailChange, grant: Grant) -> None:
        if grant.action is not Capability.CHANGE_BANK_DETAILS:
            raise PolicyDeniedError(
                f"record_bank_change requires CHANGE_BANK_DETAILS, got {grant.action.value}"
            )
        async with self._session_factory() as session:
            await self._apply_bank_change(session, change, grant)

    async def _apply_bank_change(
        self,
        session: AsyncSession,
        change: BankDetailChange,
        grant: Grant,
        proposal: ProposalModel | None = None,
    ) -> None:
        # Serialise every bank change for this vendor on the vendor row: locking only the current
        # bank_details row is not enough, because a concurrent change that waited on it re-reads
        # `is_current` after the first commits, finds no row and would insert a second current
        # account.
        # (A column-only select: loading the entity here would park it in the identity map
        # without its bank_details and break `load_vendor`'s eager load below.)
        locked_vendor = await session.scalar(
            select(VendorRow.id).where(VendorRow.id == change.vendor_id).with_for_update()
        )
        if locked_vendor is None:
            raise PolicyDeniedError(f"vendor {change.vendor_id} not found")
        current = await session.scalar(
            select(BankDetailRow).where(
                BankDetailRow.vendor_id == change.vendor_id, BankDetailRow.is_current.is_(True)
            )
        )
        actual_old_iban = current.iban if current is not None else None
        if actual_old_iban == change.new_iban:
            return  # idempotent replay: this change already landed

        # Recompute the digest against the *actual current* old_iban, not the caller-supplied
        # one, so a race that changed the account again after the grant was issued is caught
        # (the caller-supplied `change.old_iban` alone would make this check tautological).
        vendor = await load_vendor(session, change.vendor_id)
        recomputed = BankDetailChange(
            vendor_id=change.vendor_id,
            changed_by=change.changed_by,
            changed_at=change.changed_at,
            old_iban=actual_old_iban,
            new_iban=change.new_iban,
        )
        request = ActionRequest(
            actor=_PLACEHOLDER,
            action=grant.action,
            invoice=None,
            vendor=vendor,
            at=grant.issued_at,
            bank_change=recomputed,
            proposal=proposal,
        )
        if subject_digest(request) != grant.subject_digest:
            raise _stale()

        if current is not None:
            await session.execute(
                update(BankDetailRow).where(BankDetailRow.id == current.id).values(is_current=False)
            )
        session.add(
            BankDetailRow(
                vendor_id=change.vendor_id,
                iban=change.new_iban,
                source=BankDetailSource.UNVERIFIED,
                verification_id=None,
                is_current=True,
                changed_by=grant.actor,
                changed_at=change.changed_at,
            )
        )
        await self._log(
            session,
            grant.actor,
            "payment_state:bank_change",
            f"vendor:{change.vendor_id}",
            {"old_iban": actual_old_iban, "new_iban": change.new_iban},
        )

    async def verify_bank_account(
        self, vendor_id: VendorId, iban: str, grant: Grant, *, method: str, reference: str
    ) -> None:
        if grant.action is not Capability.VERIFY_BANK_DETAILS:
            raise PolicyDeniedError(
                f"verify_bank_account requires VERIFY_BANK_DETAILS, got {grant.action.value}"
            )
        if not method.strip() or not reference.strip():
            raise PolicyDeniedError("a bank verification needs an out-of-band method and reference")
        async with self._session_factory() as session:
            # Same vendor-row lock as `_apply_bank_change`, so a verification never interleaves
            # with a change that retires or replaces the account it is checking (column-only
            # select, for the reason given there).
            locked_vendor = await session.scalar(
                select(VendorRow.id).where(VendorRow.id == vendor_id).with_for_update()
            )
            if locked_vendor is None:
                raise PolicyDeniedError(f"vendor {vendor_id} not found")
            vendor = await load_vendor(session, vendor_id)
            request = ActionRequest(
                actor=_PLACEHOLDER,
                action=Capability.VERIFY_BANK_DETAILS,
                invoice=None,
                vendor=vendor,
                at=grant.issued_at,
            )
            if subject_digest(request) != grant.subject_digest:
                raise _stale()

            current = await session.scalar(
                select(BankDetailRow).where(
                    BankDetailRow.vendor_id == vendor_id,
                    BankDetailRow.iban == iban,
                    BankDetailRow.is_current.is_(True),
                )
            )
            if current is None:
                raise PolicyDeniedError(
                    f"vendor {vendor_id} has no current bank detail for {iban!r}"
                )
            if current.verification_id is not None:
                return  # idempotent replay: already verified
            if current.changed_by == grant.actor:
                # Four eyes (closes R-5): checked here, under the vendor lock, so
                # a same-actor re-change racing the API's own pre-check cannot slip through.
                raise SegregationOfDutiesError(
                    f"actor {grant.actor} recorded this bank detail and cannot verify it"
                )

            verification = BankVerificationRow(
                vendor_id=vendor_id,
                iban=iban,
                verified_by=grant.actor,
                method=method,
                notes=reference,
            )
            session.add(verification)
            await session.flush()  # assigns verification.id
            await session.execute(
                update(BankDetailRow)
                .where(BankDetailRow.id == current.id)
                .values(verification_id=verification.id, source=BankDetailSource.ADMIN_CALLBACK)
            )
            await self._log(
                session,
                grant.actor,
                "payment_state:verify_bank",
                f"vendor:{vendor_id}",
                {"iban": iban, "method": method, "reference": reference},
            )

    async def execute_proposal(
        self, proposal_id: str, grant: Grant, *, corrections_digest: Sha256Hex | None = None
    ) -> None:
        if grant.action is not Capability.EXECUTE_PROPOSAL:
            raise PolicyDeniedError(
                f"execute_proposal requires EXECUTE_PROPOSAL, got {grant.action.value}"
            )
        async with self._session_factory() as session:
            proposal = await session.get(ProposalRow, proposal_id, with_for_update=True)
            if proposal is None:
                raise PolicyDeniedError(f"proposal {proposal_id} not found")
            if proposal.status is ProposalStatus.EXECUTED:
                return  # idempotent replay
            if proposal.status is not ProposalStatus.PENDING:
                raise PolicyDeniedError(
                    f"proposal {proposal_id} is {proposal.status.value}, not pending"
                )

            # The grant was issued for EXECUTE_PROPOSAL over this proposal, so its digest covers
            # the proposal id and payload: recompute it with the same proposal in the request.
            model = proposal_model(proposal)
            if proposal.kind.value == "approve_payment":
                if proposal.invoice_id is None:
                    raise PolicyDeniedError(f"proposal {proposal_id} has no invoice to approve")
                await self._apply_approval(
                    session,
                    InvoiceId(proposal.invoice_id),
                    grant,
                    Capability.APPROVE_PAYMENT,
                    proposal=model,
                    corrections_digest=corrections_digest,
                )
            elif proposal.kind.value == "change_bank":
                change = bank_change_from_payload(proposal.payload, grant.actor)
                await self._apply_bank_change(session, change, grant, proposal=model)
            else:
                raise PolicyDeniedError(
                    f"execute_proposal here only handles payment-affecting kinds "
                    f"(approve_payment, change_bank); {proposal.kind.value} is applied elsewhere"
                )

            await session.execute(
                update(ProposalRow)
                .where(ProposalRow.id == proposal_id)
                .values(status=ProposalStatus.EXECUTED)
            )
            await self._log(
                session,
                grant.actor,
                "payment_state:execute_proposal",
                f"proposal:{proposal_id}",
                {"kind": proposal.kind.value},
            )


def proposal_model(row: ProposalRow) -> ProposalModel:
    """The `apx.policy.interfaces.Proposal` a grant's digest is computed over. Shared with the
    API so the digest it asks `PolicyGuard` for and the one recomputed here always agree."""
    return ProposalModel(
        id=row.id,
        invoice_id=InvoiceId(row.invoice_id) if row.invoice_id else None,
        kind=row.kind.value,
        payload=row.payload,
        created_by=UserId(row.created_by),
        created_at=row.created_at,
        status=row.status.value,
    )


def bank_change_from_payload(payload: dict[str, object], changed_by: UserId) -> BankDetailChange:
    """`Proposal.payload` for a `change_bank` proposal: `{"vendor_id": str, "new_iban": str}`
    (P5/ADR-0008: `change_bank` proposals never originate from a model, only a user through
    K4, but still flow through `execute_proposal` like every other proposal kind)."""
    vendor_id = payload.get("vendor_id")
    new_iban = payload.get("new_iban")
    if not isinstance(vendor_id, str) or not isinstance(new_iban, str):
        raise PolicyDeniedError(
            "change_bank proposal payload must have string vendor_id and new_iban"
        )
    return BankDetailChange(
        vendor_id=VendorId(vendor_id),
        changed_by=changed_by,
        changed_at=datetime.now(UTC),
        old_iban=None,
        new_iban=new_iban,
    )
