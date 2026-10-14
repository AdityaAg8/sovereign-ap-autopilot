"""Purchase order and line data shared by validate (3-way match) and fraud (PO price-variance
rule), plus the deterministic invoice-line -> PO-line attribution the price-variance rule uses,
kept in `common` so neither module has to import the other
(docs/architecture.md import-direction rule)."""

from __future__ import annotations

import re
from decimal import Decimal

from apx.common.invoice import InvoiceLine
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


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_description(text: str) -> str:
    """Case- and whitespace-insensitive key used to compare an invoice line's description with a
    PO line's."""
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def attribute_po_line(line: InvoiceLine, po: PurchaseOrder) -> PoLine | None:
    """The PO line this invoice line provably bills, or None when the link is not established.

    Deterministic rule, no positional guessing:
    1. The line states a PO line reference (`po_line_ref`, only ever set from the document):
       that PO line, or None when the PO has no such line. A stated reference is never
       overridden by a description match.
    2. Otherwise exactly one PO line with the same normalized description AND the same
       quantity. Unit price is deliberately not part of the key, so a changed price on an
       otherwise identical line still attributes (and is then checked for variance); quantity
       is, because a description alone recurs across unrelated lines. No match, or more than
       one, is None: unmatched, never a variance."""
    if line.po_line_ref is not None:
        return next((pl for pl in po.lines if pl.line_ref == line.po_line_ref), None)
    key = normalize_description(line.description)
    candidates = [
        pl
        for pl in po.lines
        if normalize_description(pl.description) == key and pl.quantity == line.quantity
    ]
    return candidates[0] if len(candidates) == 1 else None
