"""Vendor master and user records: vendors, admin-verified bank accounts, bank-detail changes
and users with roles. Inputs to policy rules P1-P3 and fraud rules (PRD §6.3)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from apx.common.types import Frozen, Role, UserId, VendorId


class User(Frozen):
    """Authenticated human user."""

    id: UserId
    display_name: str
    roles: frozenset[Role]


class BankAccount(Frozen):
    """Vendor bank account. `verified` is set only by an admin callback record (P1), never
    by extraction; `source` makes that provenance explicit in the type."""

    iban: str
    verified: bool
    verified_by: UserId | None
    verified_at: datetime | None
    source: Literal["admin_callback", "unverified"]


class Vendor(Frozen):
    """Vendor master record."""

    id: VendorId
    name: str
    trn: str | None
    bank_accounts: tuple[BankAccount, ...]


class BankDetailChange(Frozen):
    """A change of a vendor's bank details; drives P2 review routing and the 30-day SoD bar."""

    vendor_id: VendorId
    changed_by: UserId
    changed_at: datetime
    old_iban: str | None
    new_iban: str
