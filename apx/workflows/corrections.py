"""Reviewer corrections of the ADR-0012 values (document kind, currency, per-line VAT category
and rate). They travel as ordinary `fields` rows, fixed through the existing field-edit flow,
and reach the workflow typed inside the `review` signal. Pure: safe to import from workflow code."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Final

from apx.common.errors import ApxError
from apx.common.invoice import Invoice, InvoiceKind, VatCategory
from apx.common.types import Frozen, Money

KIND_FIELD: Final[str] = "invoice_type_code"
"""Field row holding BT-3 as "380" (invoice) or "381" (credit note)."""

_KIND_VALUES: Final[Mapping[str, InvoiceKind]] = {
    "380": InvoiceKind.INVOICE,
    "381": InvoiceKind.CREDIT_NOTE,
}
CURRENCY_FIELD: Final[str] = "currency"
"""The grounded currency field (`FieldName.CURRENCY`); "XXX" or blank means not extracted."""
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_LINE_FIELD = re.compile(r"^line_items\[(\d+)\]\.(vat_category|vat_rate)$")


def kind_value(kind: InvoiceKind | None) -> str | None:
    """The field-row value for `kind` ("380"/"381"), or None when unresolved."""
    if kind is None:
        return None
    return next(code for code, k in _KIND_VALUES.items() if k is kind)


def line_category_field(index: int) -> str:
    """Field row name for line `index` (1-based over `Invoice.lines`)."""
    return f"line_items[{index}].vat_category"


def line_rate_field(index: int) -> str:
    """Field row name for line `index`'s VAT rate in percent, e.g. "5" or "0"."""
    return f"line_items[{index}].vat_rate"


class MissingCorrectionsError(ApxError):
    """An approval arrived while a classified value is still missing or unreadable; the API
    answers 422 and the workflow is never signalled."""


class LineCorrection(Frozen):
    index: int
    """1-based position in `Invoice.lines`."""
    vat_category: VatCategory
    vat_rate: Decimal


class ReviewCorrections(Frozen):
    """Every classified value the e-invoice needs, as the reviewer finally approved it."""

    kind: InvoiceKind
    currency: str
    lines: tuple[LineCorrection, ...]


def _rate(raw: str) -> Decimal | None:
    text = raw.strip().removesuffix("%").strip()
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return value if value.is_finite() and value >= 0 else None


def corrections_from_fields(values: Mapping[str, str | None]) -> ReviewCorrections:
    """Builds `ReviewCorrections` from the latest value of each classified field row. Raises
    `MissingCorrectionsError` naming every field still missing or unreadable."""
    problems: list[str] = []
    kind = _KIND_VALUES.get((values.get(KIND_FIELD) or "").strip())
    if kind is None:
        problems.append(KIND_FIELD)
    currency = (values.get(CURRENCY_FIELD) or "").strip().upper()
    if not _CURRENCY.match(currency) or currency == "XXX":
        problems.append(CURRENCY_FIELD)
    indices = sorted(
        {int(m.group(1)) for name in values if (m := _LINE_FIELD.match(name)) is not None}
    )
    if not indices:
        problems.append("line_items[*].vat_category")
    lines: list[LineCorrection] = []
    for index in indices:
        category_raw = (values.get(line_category_field(index)) or "").strip()
        rate_raw = values.get(line_rate_field(index)) or ""
        category = VatCategory(category_raw) if category_raw in set(VatCategory) else None
        rate = _rate(rate_raw) if rate_raw.strip() else None
        if category is None:
            problems.append(line_category_field(index))
        if rate is None:
            problems.append(line_rate_field(index))
        if category is not None and rate is not None:
            lines.append(LineCorrection(index=index, vat_category=category, vat_rate=rate))
    if problems or kind is None:
        raise MissingCorrectionsError(
            "cannot approve: correct these fields first: " + ", ".join(problems)
        )
    return ReviewCorrections(kind=kind, currency=currency, lines=tuple(lines))


def apply_corrections(invoice: Invoice, corrections: ReviewCorrections) -> Invoice:
    """`invoice` with the reviewer's kind and line categories/rates. Every line must be covered
    exactly once; anything else is a `MissingCorrectionsError` (never a partial apply)."""
    by_index = {c.index: c for c in corrections.lines}
    expected = set(range(1, len(invoice.lines) + 1))
    if set(by_index) != expected or len(by_index) != len(corrections.lines):
        raise MissingCorrectionsError(
            f"corrections cover lines {sorted(by_index)}, invoice has lines {sorted(expected)}"
        )
    cur = corrections.currency

    def money(value: Money) -> Money:
        return value.model_copy(update={"currency": cur})

    lines = tuple(
        line.model_copy(
            update={
                "vat_category": by_index[i].vat_category,
                "vat_rate": by_index[i].vat_rate,
                "unit_price": money(line.unit_price),
                "net_amount": money(line.net_amount),
            }
        )
        for i, line in enumerate(invoice.lines, start=1)
    )
    return invoice.model_copy(
        update={
            "kind": corrections.kind,
            "lines": lines,
            "subtotal": money(invoice.subtotal),
            "vat_total": money(invoice.vat_total),
            "total": money(invoice.total),
        }
    )
