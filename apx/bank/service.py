"""`BankMasterService`: the product path for changing and verifying vendor bank details (REVIEW.md
M-01). A change goes PolicyGuard -> `PaymentStateWriter.record_bank_change` and is stored
unverified, so P1 never pays it and P2 starts its window; a different admin then records the
out-of-band verification through `verify_bank_account`, which makes it selectable by
`select_payee` (ADR-0008 K5, ADR-0014)."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from apx.audit.interfaces import AuditEvent, AuditLog
from apx.bank.errors import (
    BankMasterConflictError,
    BankVendorNotFoundError,
    InvalidBankRequestError,
    InvalidIbanError,
)
from apx.bank.iban import validated_iban
from apx.common.errors import SegregationOfDutiesError
from apx.common.master import BankDetailChange, User
from apx.common.types import Frozen, UserId, VendorId
from apx.db.models import BankDetail as BankDetailRow
from apx.db.models import Vendor as VendorRow
from apx.policy.interfaces import (
    ActionRequest,
    Capability,
    PaymentStateWriter,
    PolicyGuard,
    electronic_iban,
)
from apx.policy.writer import load_vendor

MAX_REFERENCE_CHARS = 200
"""Upper bound on a verification reference (a ticket id, call log id or letter number)."""


class VerificationMethod(StrEnum):
    """How the admin confirmed the account outside the system (docs/security.md R-1: call the
    number already on file, never one from the document or the change request)."""

    PHONE_CALLBACK_ON_FILE = "phone_callback_on_file"
    IN_PERSON = "in_person"
    SIGNED_BANK_LETTER = "signed_bank_letter"


class BankAccountStatus(StrEnum):
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"


class BankAccountRecord(Frozen):
    """One current vendor bank account as the API returns it; the IBAN is always masked."""

    vendor_id: VendorId
    masked_iban: str
    status: BankAccountStatus
    changed_by: UserId
    changed_at: datetime
    verified_by: UserId | None
    verified_at: datetime | None
    verification_method: str | None


def mask_iban(iban: str) -> str:
    """All but the last four characters masked (ADR-0008 T2)."""
    return "*" * max(len(iban) - 4, 0) + iban[-4:]


def _record(row: BankDetailRow) -> BankAccountRecord:
    verification = row.verification
    return BankAccountRecord(
        vendor_id=VendorId(row.vendor_id),
        masked_iban=mask_iban(row.iban),
        status=BankAccountStatus.VERIFIED
        if row.verified
        else BankAccountStatus.PENDING_VERIFICATION,
        changed_by=UserId(row.changed_by),
        changed_at=row.changed_at,
        verified_by=UserId(verification.verified_by) if verification else None,
        verified_at=verification.verified_at if verification else None,
        verification_method=verification.method if verification else None,
    )


def _clean_reference(reference: str) -> str:
    cleaned = reference.strip()
    if not cleaned or len(cleaned) > MAX_REFERENCE_CHARS:
        raise InvalidBankRequestError(
            f"verification reference must be 1-{MAX_REFERENCE_CHARS} characters"
        )
    if not cleaned.isprintable():
        raise InvalidBankRequestError("verification reference must not contain control characters")
    return cleaned


class BankMasterService:
    """Propose and verify vendor bank details. Every write goes through `PolicyGuard` (audited
    `authorize:*` event, digest-bound grant) and then the `PaymentStateWriter` (audited
    `payment_state:*` event); refusals are typed and nothing is written on any of them."""

    def __init__(
        self,
        session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        guard: PolicyGuard,
        writer: PaymentStateWriter,
        audit: AuditLog,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_factory = session_factory
        self._guard = guard
        self._writer = writer
        self._audit = audit
        self._clock = clock

    async def propose_change(
        self,
        actor: User,
        vendor_id: VendorId,
        new_iban: str,
        *,
        expected_current_iban: str | None,
    ) -> BankAccountRecord:
        """Replace the vendor's current account with `new_iban`, stored unverified (P1 refuses
        to pay it until verified; in-flight invoices keep their pinned payee, and the build
        refuses rather than pay a changed master, ADR-0014 §7). `expected_current_iban` is the
        account the caller saw (None: none); a mismatch is a 409, so a retried or stale request
        never overwrites a newer change. Re-proposing the current IBAN is an idempotent no-op."""
        new = validated_iban(new_iban)
        expected: str | None = None
        if expected_current_iban is not None:
            expected = electronic_iban(expected_current_iban)
            if expected is None:
                raise InvalidIbanError("expected_current_iban is not in ISO 13616 form")

        now = self._clock()
        async with self._session_factory() as session:
            current = await self._current_rows(session, vendor_id)
            if len(current) > 1:
                raise BankMasterConflictError(
                    f"vendor {vendor_id} has {len(current)} current accounts; a change must "
                    "replace exactly one (retire the others first)"
                )
            row = current[0] if current else None
            if row is not None and electronic_iban(row.iban) == new:
                return _record(row)  # idempotent replay: this change already landed
            current_iban = electronic_iban(row.iban) if row is not None else None
            if expected != current_iban:
                raise BankMasterConflictError(
                    f"vendor {vendor_id}'s current bank detail is not the one the request "
                    "expected; reload and retry"
                )
            change = BankDetailChange(
                vendor_id=vendor_id,
                changed_by=actor.id,
                changed_at=now,
                old_iban=row.iban if row is not None else None,
                new_iban=new,
            )
            request = ActionRequest(
                actor=actor,
                action=Capability.CHANGE_BANK_DETAILS,
                invoice=None,
                vendor=await load_vendor(session, vendor_id),
                at=now,
                bank_change=change,
            )

        grant = await self._guard.require(request)
        await self._writer.record_bank_change(change, grant)
        return await self._current_record(vendor_id, new)

    async def verify(
        self,
        actor: User,
        vendor_id: VendorId,
        iban: str,
        *,
        method: VerificationMethod,
        reference: str,
    ) -> BankAccountRecord:
        """Record `actor`'s out-of-band verification of the vendor's pending current account
        `iban`. Refuses (SegregationOfDutiesError, audited) when `actor` recorded that account:
        a bank change never verifies itself. Verifying an already verified account is an
        idempotent no-op."""
        normalized = electronic_iban(iban)
        if normalized is None:
            raise InvalidIbanError("IBAN is not in ISO 13616 form")
        cleaned_reference = _clean_reference(reference)

        now = self._clock()
        async with self._session_factory() as session:
            current = await self._current_rows(session, vendor_id)
            matching = [row for row in current if electronic_iban(row.iban) == normalized]
            if len(matching) != 1:
                raise BankMasterConflictError(
                    f"vendor {vendor_id} has no single current account with that IBAN to verify"
                )
            row = matching[0]
            if row.verified:
                return _record(row)  # idempotent replay: already verified
            if row.changed_by == actor.id:
                await self._audit.append(
                    AuditEvent(
                        actor=str(actor.id),
                        action="bank_master:verify_refused",
                        entity=f"vendor:{vendor_id}",
                        payload={
                            "reason": "verifier_is_changer",
                            "masked_iban": mask_iban(row.iban),
                        },
                    )
                )
                raise SegregationOfDutiesError(
                    f"actor {actor.id} recorded this bank detail and cannot verify it"
                )
            stored_iban = row.iban
            request = ActionRequest(
                actor=actor,
                action=Capability.VERIFY_BANK_DETAILS,
                invoice=None,
                vendor=await load_vendor(session, vendor_id),
                at=now,
            )

        grant = await self._guard.require(request)
        await self._writer.verify_bank_account(
            vendor_id, stored_iban, grant, method=method.value, reference=cleaned_reference
        )
        return await self._current_record(vendor_id, normalized)

    async def _current_rows(
        self, session: AsyncSession, vendor_id: VendorId
    ) -> list[BankDetailRow]:
        if await session.scalar(select(VendorRow.id).where(VendorRow.id == vendor_id)) is None:
            raise BankVendorNotFoundError(f"vendor {vendor_id} not found")
        rows = await session.scalars(
            select(BankDetailRow)
            .where(BankDetailRow.vendor_id == vendor_id, BankDetailRow.is_current.is_(True))
            .options(selectinload(BankDetailRow.verification))
            .order_by(BankDetailRow.id)
        )
        return list(rows.all())

    async def _current_record(self, vendor_id: VendorId, iban: str) -> BankAccountRecord:
        async with self._session_factory() as session:
            for row in await self._current_rows(session, vendor_id):
                if electronic_iban(row.iban) == iban:
                    return _record(row)
        raise BankMasterConflictError(
            f"vendor {vendor_id}'s bank master changed again before the result could be read"
        )
