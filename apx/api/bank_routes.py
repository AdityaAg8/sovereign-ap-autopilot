"""Vendor bank-master endpoints: an admin proposes a bank-detail change, stored
unverified, and a different admin records the out-of-band verification that makes it payable
under P1. Both go through `apx.bank.BankMasterService` (PolicyGuard -> PaymentStateWriter)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from apx.api.deps import SettingsDep, require_capability
from apx.audit.chain import PgAuditLog
from apx.bank.errors import (
    BankMasterConflictError,
    BankVendorNotFoundError,
    InvalidBankRequestError,
)
from apx.bank.service import (
    MAX_REFERENCE_CHARS,
    BankAccountRecord,
    BankMasterService,
    VerificationMethod,
)
from apx.common.master import User
from apx.common.types import Money, VendorId
from apx.db.session import get_session
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.guard import DefaultPolicyGuard
from apx.policy.interfaces import Capability, PolicyThresholds
from apx.policy.writer import SqlPaymentStateWriter

router = APIRouter(prefix="/vendors/{vendor_id}/bank-details", tags=["bank-details"])

_IBAN_INPUT_MAX = 64
"""Room for a 34-character IBAN in print form (spaces every four characters) and no more."""


class BankChangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_iban: str = Field(min_length=1, max_length=_IBAN_INPUT_MAX)
    expected_current_iban: str | None = Field(default=None, max_length=_IBAN_INPUT_MAX)
    """The current account the caller saw (None: the vendor had none); a stale value is a 409."""


class BankVerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iban: str = Field(min_length=1, max_length=_IBAN_INPUT_MAX)
    method: VerificationMethod
    reference: str = Field(min_length=1, max_length=MAX_REFERENCE_CHARS)


def get_bank_service(settings: SettingsDep) -> BankMasterService:
    audit = PgAuditLog()
    limit = Money(amount=settings.approval_limit, currency="AED")
    thresholds = PolicyThresholds(
        approval_limit=limit,
        # `authorize` for CHANGE/VERIFY_BANK_DETAILS reads only the capability matrix; the
        # routing thresholds are never consulted on this path (the workflow routes).
        fraud_review=1.0,
        scan_review=1.0,
        attack_block=1.0,
        sod_cooldown_days=settings.sod_cooldown_days,
    )
    return BankMasterService(
        get_session,
        DefaultPolicyGuard(DefaultPolicyEngine(thresholds), audit),
        SqlPaymentStateWriter(get_session, audit, approval_limit=limit),
        audit,
    )


BankServiceDep = Annotated[BankMasterService, Depends(get_bank_service)]
ChangerDep = Annotated[User, Depends(require_capability(Capability.CHANGE_BANK_DETAILS))]
VerifierDep = Annotated[User, Depends(require_capability(Capability.VERIFY_BANK_DETAILS))]


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, BankVendorNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, BankMasterConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))


@router.post("/changes", response_model=BankAccountRecord)
async def propose_bank_change(
    vendor_id: str, body: BankChangeRequest, service: BankServiceDep, user: ChangerDep
) -> BankAccountRecord:
    """Stores the new account unverified; it is never paid until another admin verifies it."""
    try:
        return await service.propose_change(
            user,
            VendorId(vendor_id),
            body.new_iban,
            expected_current_iban=body.expected_current_iban,
        )
    except (BankVendorNotFoundError, BankMasterConflictError, InvalidBankRequestError) as exc:
        raise _http_error(exc) from exc


@router.post("/verifications", response_model=BankAccountRecord)
async def verify_bank_detail(
    vendor_id: str, body: BankVerificationRequest, service: BankServiceDep, user: VerifierDep
) -> BankAccountRecord:
    """Records the out-of-band verification; the proposer of the account gets a 403 (SoD)."""
    try:
        return await service.verify(
            user,
            VendorId(vendor_id),
            body.iban,
            method=body.method,
            reference=body.reference,
        )
    except (BankVendorNotFoundError, BankMasterConflictError, InvalidBankRequestError) as exc:
        raise _http_error(exc) from exc
