"""Purchase order and line data shared by validate (3-way match) and fraud (PO price-variance
rule), kept in `common` so neither module has to import the other (PRD §3, docs/architecture.md
import-direction rule)."""

from __future__ import annotations

from decimal import Decimal

from apx.common.types import Frozen, Money, VendorId


class PoLine(Frozen):
    line_ref: str
    description: str
    quantity: Decimal
    unit_price: Money


class PurchaseOrder(Frozen):
    number: str
    vendor_id: VendorId
    lines: tuple[PoLine, ...]
