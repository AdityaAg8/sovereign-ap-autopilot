"""In-memory `InvoiceHistory` (PRD §10 M8.2): an append-only invoice stream with vendor bank
changes, vendor-master IBANs and purchase orders, answering every query "as of" a date so a rule
or the anomaly detector only ever sees invoices issued on or before the one being assessed. Used
by the fraud eval and tests; the production history reads the same facts from Postgres."""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, timedelta

from apx.common.invoice import Invoice
from apx.common.master import BankDetailChange
from apx.common.purchasing import PurchaseOrder
from apx.common.types import VendorId
from apx.fraud.normalize import normalize_iban


class InMemoryInvoiceHistory:
    """Append invoices in issue-date order; set `as_of` before assessing an invoice. IBAN
    ownership comes from the vendor master and recorded bank changes only -- never from what an
    invoice claims, or one impersonation attempt would make the victim's own IBAN look shared."""

    def __init__(
        self,
        *,
        vendor_ibans: dict[VendorId, str] | None = None,
        bank_changes: Iterable[BankDetailChange] = (),
        purchase_orders: Iterable[PurchaseOrder] = (),
    ) -> None:
        self.as_of: date = date.min
        self._by_vendor: dict[VendorId, list[Invoice]] = defaultdict(list)
        self._iban_vendors: dict[str, set[VendorId]] = defaultdict(set)
        for vendor, iban in (vendor_ibans or {}).items():
            self._iban_vendors[normalize_iban(iban)].add(vendor)
        self._changes: dict[VendorId, list[BankDetailChange]] = defaultdict(list)
        for change in bank_changes:
            self._changes[change.vendor_id].append(change)
            self._iban_vendors[normalize_iban(change.new_iban)].add(change.vendor_id)
        self._pos = {po.number: po for po in purchase_orders}

    def add(self, invoice: Invoice) -> None:
        """Record an assessed invoice. Raises `ValueError` if it predates the latest one for its
        vendor, since every window query assumes per-vendor issue-date order."""
        if invoice.vendor_id is None:
            return
        rows = self._by_vendor[invoice.vendor_id]
        if rows and rows[-1].issue_date > invoice.issue_date:
            raise ValueError(f"invoice {invoice.id} is older than the vendor's latest invoice")
        rows.append(invoice)

    def recent_for_vendor(self, vendor_id: VendorId, days: int) -> Sequence[Invoice]:
        rows = self._by_vendor.get(vendor_id, [])
        start = self.as_of - timedelta(days=days)
        dates = [row.issue_date for row in rows]
        return rows[bisect.bisect_left(dates, start) : bisect.bisect_right(dates, self.as_of)]

    def bank_changes(self, vendor_id: VendorId, days: int) -> Sequence[BankDetailChange]:
        start = self.as_of - timedelta(days=days)
        return [
            change
            for change in self._changes.get(vendor_id, [])
            if start <= change.changed_at.date() <= self.as_of
        ]

    def vendors_using_iban(self, iban: str) -> Sequence[VendorId]:
        return sorted(self._iban_vendors.get(normalize_iban(iban), set()))

    def purchase_order(self, number: str) -> PurchaseOrder | None:
        return self._pos.get(number)
