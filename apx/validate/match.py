"""3-way match: reconciles invoice lines against a purchase order and its goods receipts,
enforcing exact quantity against cumulative receipts (minus quantities already invoiced),
configurable unit-price tolerance and per-line currency consistency (PRD §10 M4.2)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal

from apx.common.invoice import Invoice, InvoiceLine
from apx.common.types import Frozen, Money
from apx.validate.interfaces import (
    GoodsReceipt,
    MatchTolerance,
    PoLine,
    PurchaseOrder,
    Violation,
    ViolationCode,
)

DEFAULT_MATCH_TOLERANCE = MatchTolerance(quantity=Decimal("0"), price_pct=Decimal("0.005"))
"""Quantity must match received exactly; unit price may differ by up to ±0.5% (PRD §10 M4.2)."""

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_description(text: str) -> str:
    """Case- and whitespace-insensitive key used for the SKU/description fallback match."""
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


class LineMatch(Frozen):
    """Per-line detail of a 3-way match: what was invoiced against what the PO line allows and
    what the GRNs confirm was received, before quantities already invoiced are subtracted."""

    invoice_line_index: int
    po_line_ref: str | None
    quantity_invoiced: Decimal
    quantity_previously_invoiced: Decimal
    quantity_received_cumulative: Decimal | None
    unit_price_invoice: Money
    unit_price_po: Money | None


class MatchResult(Frozen):
    """Full 3-way match outcome for one invoice: per-line detail plus the flat violation list
    that feeds into `apx.validate.interfaces.ValidationReport`."""

    po_number: str | None
    lines: tuple[LineMatch, ...]
    violations: tuple[Violation, ...]

    @property
    def passed(self) -> bool:
        """True when there are no error-severity violations."""
        return not any(v.severity == "error" for v in self.violations)


def _resolve_po_line(invoice_line: InvoiceLine, po: PurchaseOrder) -> PoLine | None:
    """Match by PO line ref first; fall back to an unambiguous description match."""
    if invoice_line.po_line_ref is not None:
        for po_line in po.lines:
            if po_line.line_ref == invoice_line.po_line_ref:
                return po_line
    target = _normalize_description(invoice_line.description)
    candidates = [pl for pl in po.lines if _normalize_description(pl.description) == target]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _received_by_line_ref(
    po_number: str, receipts: Sequence[GoodsReceipt]
) -> tuple[dict[str, Decimal], bool]:
    """Cumulative received quantity per PO line ref, and whether any GRN exists for this PO."""
    received: dict[str, Decimal] = {}
    has_receipt = False
    for grn in receipts:
        if grn.po_number != po_number:
            continue
        has_receipt = True
        for grn_line in grn.lines:
            received[grn_line.po_line_ref] = (
                received.get(grn_line.po_line_ref, Decimal(0)) + grn_line.quantity_received
            )
    return received, has_receipt


def match_invoice(
    invoice: Invoice,
    po: PurchaseOrder | None,
    receipts: Sequence[GoodsReceipt],
    tolerance: MatchTolerance = DEFAULT_MATCH_TOLERANCE,
    previously_invoiced: Mapping[str, Decimal] | None = None,
) -> MatchResult:
    """Match every invoice line against `po` and its `receipts`.

    `previously_invoiced` maps a PO line ref to the quantity already billed by other invoices
    against that same line; it is subtracted from cumulative receipts to get the quantity still
    available to this invoice. Quantity tolerance is exact by default (PRD §10 M4.2)."""
    prior = previously_invoiced or {}

    if po is None:
        violation = Violation(
            code=ViolationCode.PO_NOT_FOUND,
            field="po_number",
            message=f"No purchase order found for {invoice.po_number!r}.",
            severity="error",
        )
        return MatchResult(po_number=invoice.po_number, lines=(), violations=(violation,))

    violations: list[Violation] = []
    line_results: list[LineMatch] = []
    claimed_refs: set[str] = set()

    received_by_ref, has_receipt = _received_by_line_ref(po.number, receipts)
    if not has_receipt:
        violations.append(
            Violation(
                code=ViolationCode.GRN_NOT_FOUND,
                field="po_number",
                message=f"No goods receipt found for purchase order {po.number}.",
                severity="error",
            )
        )

    for idx, invoice_line in enumerate(invoice.lines):
        field = f"lines[{idx}]"
        po_line = _resolve_po_line(invoice_line, po)

        if po_line is None:
            violations.append(
                Violation(
                    code=ViolationCode.UNKNOWN_LINE,
                    field=field,
                    message=(
                        f"Invoice line {idx} ({invoice_line.description!r}) does not match "
                        f"any line on purchase order {po.number}."
                    ),
                    severity="error",
                )
            )
            line_results.append(
                LineMatch(
                    invoice_line_index=idx,
                    po_line_ref=None,
                    quantity_invoiced=invoice_line.quantity,
                    quantity_previously_invoiced=Decimal(0),
                    quantity_received_cumulative=None,
                    unit_price_invoice=invoice_line.unit_price,
                    unit_price_po=None,
                )
            )
            continue

        po_line_ref = po_line.line_ref
        if po_line_ref in claimed_refs:
            violations.append(
                Violation(
                    code=ViolationCode.DUPLICATE_LINE,
                    field=field,
                    message=(
                        f"Invoice line {idx} duplicates another invoice line already matched "
                        f"to PO line {po_line_ref}."
                    ),
                    severity="error",
                )
            )
        claimed_refs.add(po_line_ref)

        prior_qty = prior.get(po_line_ref, Decimal(0))
        received_cumulative = received_by_ref.get(po_line_ref, Decimal(0)) if has_receipt else None

        total_invoiced = prior_qty + invoice_line.quantity
        if total_invoiced > po_line.quantity + tolerance.quantity:
            violations.append(
                Violation(
                    code=ViolationCode.OVER_BILLING,
                    field=field,
                    message=(
                        f"Line {po_line_ref}: invoiced quantity {total_invoiced} (including "
                        f"{prior_qty} previously invoiced) exceeds the ordered quantity "
                        f"{po_line.quantity}."
                    ),
                    severity="error",
                )
            )

        if received_cumulative is not None:
            available = received_cumulative - prior_qty
            if invoice_line.quantity > available + tolerance.quantity:
                violations.append(
                    Violation(
                        code=ViolationCode.QUANTITY_OVER_RECEIVED,
                        field=field,
                        message=(
                            f"Line {po_line_ref}: invoiced quantity {invoice_line.quantity} "
                            f"exceeds the {available} still available after {received_cumulative} "
                            f"received and {prior_qty} previously invoiced."
                        ),
                        severity="error",
                    )
                )

        if invoice_line.unit_price.currency != po_line.unit_price.currency:
            violations.append(
                Violation(
                    code=ViolationCode.CURRENCY_MISMATCH,
                    field=field,
                    message=(
                        f"Line {po_line_ref}: invoice currency {invoice_line.unit_price.currency} "
                        f"does not match PO currency {po_line.unit_price.currency}."
                    ),
                    severity="error",
                )
            )
        else:
            po_price = po_line.unit_price.amount
            invoice_price = invoice_line.unit_price.amount
            allowed_deviation = po_price * tolerance.price_pct
            actual_deviation = abs(invoice_price - po_price)
            if actual_deviation > allowed_deviation:
                violations.append(
                    Violation(
                        code=ViolationCode.PRICE_OVER_TOLERANCE,
                        field=field,
                        message=(
                            f"Line {po_line_ref}: unit price {invoice_price} deviates from PO "
                            f"price {po_price} by more than the configured tolerance "
                            f"({tolerance.price_pct:%})."
                        ),
                        severity="error",
                    )
                )

        line_results.append(
            LineMatch(
                invoice_line_index=idx,
                po_line_ref=po_line_ref,
                quantity_invoiced=invoice_line.quantity,
                quantity_previously_invoiced=prior_qty,
                quantity_received_cumulative=received_cumulative,
                unit_price_invoice=invoice_line.unit_price,
                unit_price_po=po_line.unit_price,
            )
        )

    return MatchResult(po_number=po.number, lines=tuple(line_results), violations=tuple(violations))


def three_way_match(
    invoice: Invoice,
    po: PurchaseOrder | None,
    receipts: Sequence[GoodsReceipt],
    tolerance: MatchTolerance = DEFAULT_MATCH_TOLERANCE,
    previously_invoiced: Mapping[str, Decimal] | None = None,
) -> list[Violation]:
    """Flat violation list for `apx.validate.interfaces.Validator` consumers; see
    `match_invoice` for the full per-line result."""
    return list(match_invoice(invoice, po, receipts, tolerance, previously_invoiced).violations)
