"""Typed errors of the bank-master service (REVIEW.md M-01); `apx.api.bank_routes` maps each to
an HTTP status. Policy refusals stay `PolicyDeniedError`/`SegregationOfDutiesError` (403)."""

from __future__ import annotations

from apx.common.errors import ApxError


class InvalidBankRequestError(ApxError):
    """A malformed bank-master request (bad IBAN, blank or oversized reference); nothing is
    written (422)."""


class InvalidIbanError(InvalidBankRequestError):
    """The submitted IBAN is malformed or fails mod-97; nothing is written (422)."""


class BankVendorNotFoundError(ApxError):
    """The vendor does not exist (404)."""


class BankMasterConflictError(ApxError):
    """The request no longer fits the vendor's current bank master: the caller's expected
    current IBAN is stale, the vendor has several current accounts, or the IBAN to verify is not
    a pending current account (409). Nothing is written."""
