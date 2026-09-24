"""Loads the factory's deterministic ERP snapshot (users, roles, vendors, admin-verified bank
accounts and recent bank-detail changes, POs and GRNs) into Postgres, so the worker matches the
same vendors the generated invoices name and the web app has users to sign in as (PRD §10 M5.6).
Re-running replaces the ERP tables (and, by cascade, any invoices). `make seed` runs it."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apx.api.security import hash_password
from apx.common.types import Role
from apx.db.enums import BankDetailSource
from apx.db.models import BankDetail, BankVerification, Grn, GrnLine, RoleDef, User, UserRole
from apx.db.models import PoLine as PoLineRow
from apx.db.models import PurchaseOrder as PoRow
from apx.db.models import Vendor as VendorRow
from apx.db.session import get_session
from factory.erp import Erp, generate_erp

ERP_TABLES = (
    "grn_lines, grns, po_lines, pos, bank_details_history, bank_details, bank_verifications, "
    "vendors, user_roles, roles, users"
)
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
"""`factory.erp`'s snapshot date: verifications and GRNs without their own date carry it."""


def email_for(user_id: str) -> str:
    return f"{user_id.lower()}@apx.local"


async def _users(session: AsyncSession, erp: Erp, password: str) -> None:
    role_rows = {role: RoleDef(name=role) for role in Role}
    session.add_all(role_rows.values())
    digest = hash_password(password)
    for user in erp.users:
        session.add(
            User(
                id=user.id,
                email=email_for(user.id),
                display_name=user.display_name,
                password_hash=digest,
            )
        )
    await session.flush()
    for user in erp.users:
        for role in sorted(user.roles):
            session.add(UserRole(user_id=user.id, role_id=role_rows[role].id))


async def _vendors(session: AsyncSession, erp: Erp) -> None:
    admin = next(u.id for u in erp.users if Role.ADMIN in u.roles)
    changes = {change.new_iban: change for change in erp.bank_detail_changes}
    for vendor in erp.vendors:
        session.add(VendorRow(id=vendor.id, name=vendor.name, trn=vendor.trn))
    await session.flush()
    for vendor in erp.vendors:
        for account in vendor.bank_accounts:
            verification_id = None
            if account.verified and account.verified_by is not None:
                verification = BankVerification(
                    vendor_id=vendor.id,
                    iban=account.iban,
                    verified_by=account.verified_by,
                    verified_at=account.verified_at or EPOCH,
                )
                session.add(verification)
                await session.flush()
                verification_id = verification.id
            change = changes.get(account.iban)
            session.add(
                BankDetail(
                    vendor_id=vendor.id,
                    iban=account.iban,
                    source=BankDetailSource(account.source),
                    verification_id=verification_id,
                    is_current=True,
                    changed_by=change.changed_by if change else (account.verified_by or admin),
                    changed_at=change.changed_at if change else (account.verified_at or EPOCH),
                )
            )


async def _purchasing(session: AsyncSession, erp: Erp) -> None:
    po_rows: dict[str, PoRow] = {}
    line_rows: dict[tuple[str, str], PoLineRow] = {}
    for po in erp.purchase_orders:
        row = PoRow(
            po_number=po.number,
            vendor_id=po.vendor_id,
            currency=po.lines[0].unit_price.currency if po.lines else "AED",
            total=sum((ln.quantity * ln.unit_price.amount for ln in po.lines), Decimal(0)),
        )
        for index, line in enumerate(po.lines, start=1):
            line_row = PoLineRow(
                line_no=index,
                description=line.description,
                quantity=line.quantity,
                unit_price=line.unit_price.amount,
                net_amount=line.quantity * line.unit_price.amount,
                currency=line.unit_price.currency,
            )
            row.lines.append(line_row)
            line_rows[(po.number, line.line_ref)] = line_row
        po_rows[po.number] = row
        session.add(row)
    await session.flush()
    for grn in erp.goods_receipts:
        grn_row = Grn(
            po_id=po_rows[grn.po_number].id, grn_number=grn.number, received_at=EPOCH.date()
        )
        for ln in grn.lines:
            po_line = line_rows[(grn.po_number, ln.po_line_ref)]
            grn_row.lines.append(
                GrnLine(
                    po_line_id=po_line.id,
                    description=po_line.description,
                    quantity=ln.quantity_received,
                )
            )
        session.add(grn_row)


async def seed(
    erp: Erp,
    password: str,
    sessions: Callable[[], AbstractAsyncContextManager[AsyncSession]] = get_session,
) -> None:
    """Replaces the ERP tables with `erp`. The truncate cascades to every row that references
    them, invoices included: seeding starts a clean demo, it never merges into one."""
    async with sessions() as session:
        await session.execute(text(f"TRUNCATE {ERP_TABLES} RESTART IDENTITY CASCADE"))
        await _users(session, erp, password)
        await _vendors(session, erp)
        await _purchasing(session, erp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026, help="factory ERP seed (build.py's)")
    parser.add_argument("--vendors", type=int, default=20)
    parser.add_argument("--users", type=int, default=8)
    parser.add_argument(
        "--credentials",
        type=Path,
        default=Path("data/seed_credentials.json"),
        help="where the generated sign-in emails and shared password are written",
    )
    args = parser.parse_args()
    erp = generate_erp(args.seed, n_vendors=args.vendors, n_users=args.users)
    password = secrets.token_urlsafe(12)
    asyncio.run(seed(erp, password))
    args.credentials.parent.mkdir(parents=True, exist_ok=True)
    users = [
        {"email": email_for(u.id), "roles": sorted(r.value for r in u.roles)} for u in erp.users
    ]
    args.credentials.write_text(
        json.dumps({"password": password, "users": users}, indent=2), encoding="utf-8"
    )
    print(
        f"seeded {len(erp.users)} users, {len(erp.vendors)} vendors, "
        f"{len(erp.purchase_orders)} POs; sign-in details in {args.credentials}"
    )


if __name__ == "__main__":
    main()
