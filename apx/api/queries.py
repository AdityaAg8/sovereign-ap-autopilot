"""Pure SQLAlchemy statement builders for the invoice and vendor list endpoints' filter/paginate
queries. Every filter value is bound as a parameter (never string-formatted into SQL), so a
filter value containing SQL-injection payloads is always inert -- this is what
`test_api_filters.py` checks by compiling these statements and inspecting the bound parameters
(PRD §10 M5.3, M5.4).
"""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy import Select, func, select

from apx.common.types import VendorId
from apx.db.models import BankDetail, Invoice, Vendor
from apx.workflows.interfaces import InvoiceStatus


def invoice_list_stmt(
    *,
    status: InvoiceStatus | None,
    vendor_id: VendorId | None,
    q: str | None,
    limit: int,
    offset: int,
) -> Select[tuple[Invoice, str | None]]:
    """Builds the page query: invoices left-joined to their vendor's name, newest first.
    `q` matches (case-insensitively) against the invoice number or vendor name; `status` and
    `vendor_id` match exactly. Every value is bound via `.where(... == value)` / `.ilike`, so no
    filter value is ever concatenated into the SQL text."""
    # `Vendor.name` is non-nullable in the schema, but the outer join can still produce a NULL
    # for an invoice with no resolved vendor -- `cast` corrects the type SQLAlchemy infers to
    # match what actually comes back at runtime.
    stmt = cast(
        "Select[tuple[Invoice, str | None]]",
        select(Invoice, Vendor.name).outerjoin(Vendor, Invoice.vendor_id == Vendor.id),
    )
    if status is not None:
        stmt = stmt.where(Invoice.status == status)
    if vendor_id is not None:
        stmt = stmt.where(Invoice.vendor_id == vendor_id)
    if q:
        pattern = f"%{q}%"
        stmt = stmt.where((Invoice.number.ilike(pattern)) | (Vendor.name.ilike(pattern)))
    return stmt.order_by(Invoice.created_at.desc()).limit(limit).offset(offset)


def invoice_count_stmt(
    *, status: InvoiceStatus | None, vendor_id: VendorId | None, q: str | None
) -> Select[tuple[int]]:
    """Total row count for the same filters as `invoice_list_stmt`, for pagination metadata."""
    stmt: Select[tuple[int]] = (
        select(func.count(Invoice.id))
        .select_from(Invoice)
        .outerjoin(Vendor, Invoice.vendor_id == Vendor.id)
    )
    if status is not None:
        stmt = stmt.where(Invoice.status == status)
    if vendor_id is not None:
        stmt = stmt.where(Invoice.vendor_id == vendor_id)
    if q:
        pattern = f"%{q}%"
        stmt = stmt.where((Invoice.number.ilike(pattern)) | (Vendor.name.ilike(pattern)))
    return stmt


def _vendor_verified_subquery() -> Select[tuple[int]]:
    """Correlated scalar subquery: the count of a vendor's current, G-3-verified bank accounts.
    `> 0` gives the vendor-level `verified` flag `vendor_list_stmt` and
    `apx.api.service.DbVendorService.get` both use, without ever reading a plain boolean column
    -- `BankDetail.verified` is itself a `Computed` column, so this can only be true when a real
    `bank_verifications` row exists (G-3)."""
    return (
        select(func.count())
        .select_from(BankDetail)
        .where(
            BankDetail.vendor_id == Vendor.id,
            BankDetail.is_current.is_(True),
            BankDetail.verified.is_(True),
        )
        .correlate(Vendor)
    )


def _vendor_search_filter(stmt: Select[Any], q: str | None) -> Select[Any]:
    if q:
        pattern = f"%{q}%"
        return stmt.where((Vendor.name.ilike(pattern)) | (Vendor.trn.ilike(pattern)))
    return stmt


def vendor_list_stmt(*, q: str | None, limit: int, offset: int) -> Select[tuple[Vendor, bool]]:
    """Builds the vendor list/search/paginate query (PRD §10 M5.4): every vendor plus whether it
    has at least one current, verified bank account, ordered by name. `q` matches
    (case-insensitively) the vendor name or TRN; every filter value is bound, never
    string-formatted into the SQL text (same convention as `invoice_list_stmt`)."""
    stmt = cast(
        "Select[tuple[Vendor, bool]]",
        select(Vendor, _vendor_verified_subquery().scalar_subquery() > 0),
    )
    stmt = _vendor_search_filter(stmt, q)
    return stmt.order_by(Vendor.name).limit(limit).offset(offset)


def vendor_count_stmt(*, q: str | None) -> Select[tuple[int]]:
    """Total row count for the same filters as `vendor_list_stmt`, for pagination metadata."""
    stmt: Select[tuple[int]] = select(func.count(Vendor.id))
    return _vendor_search_filter(stmt, q)
