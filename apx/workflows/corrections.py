"""Reviewer corrections of the ADR-0012 values (document kind, currency, per-line VAT category
and rate) and of a supplier address that names no UAE emirate. They travel as ordinary `fields`
rows, fixed through the existing field-edit flow, and reach the workflow typed inside the
`review` signal. Pure: safe to import from workflow code."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Final

from apx.common.errors import ApxError
from apx.common.invoice import Invoice, InvoiceKind, VatCategory
from apx.common.types import Frozen, Money
from apx.validate.address import resolve_emirate
from apx.validate.interfaces import vat_rate_fits_category

KIND_FIELD: Final[str] = "invoice_type_code"
"""Field row holding BT-3 as "380" (invoice) or "381" (credit note)."""

_KIND_VALUES: Final[Mapping[str, InvoiceKind]] = {
    "380": InvoiceKind.INVOICE,
    "381": InvoiceKind.CREDIT_NOTE,
}
CURRENCY_FIELD: Final[str] = "currency"
"""The grounded currency field (`FieldName.CURRENCY`); "XXX" or blank means not extracted."""
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_DOCUMENT_CURRENCY: Final[str] = "AED"
"""The only currency a PINT AE e-invoice is built in here (`apx.einvoice.builder`)."""
_LINE_FIELD = re.compile(r"^line_items\[(\d+)\]\.(vat_category|vat_rate)$")
SUPPLIER_TRN_FIELD: Final[str] = "supplier_trn"
"""The grounded supplier TRN (`FieldName.SUPPLIER_TRN`): BT-31 and the seller EndpointID."""
SUPPLIER_ADDRESS_FIELD: Final[str] = "supplier_address"
"""The grounded supplier address (`FieldName.SUPPLIER_ADDRESS`): BT-35, and through its emirate
the seller city and country subdivision (BT-37, BT-39; ADR-0012 amendment 4)."""


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


class SupplierTrnMissingError(MissingCorrectionsError):
    """Approval refused: the document has no supplier TRN. The e-invoice needs it (BT-31, the
    BT-34 EndpointID) and a reviewer's field edit cannot supply it (the workflow builds from the
    extracted invoice), so the approval would only fail at the build. 422, nothing recorded."""


class TotalsMismatchError(MissingCorrectionsError):
    """Approval refused: validation found line or document amounts that do not add up
    (`totals_mismatch`). No correction carries amounts and the PINT AE build refuses them
    (`ibr-147-ae` and the totals rules), so the reviewer rejects instead. 422, nothing recorded
    (ADR-0012 amendment 4)."""


class CurrencyRelabelError(MissingCorrectionsError):
    """Approval refused: the reviewer's currency would relabel amounts extracted in another
    valid currency (for example USD) as AED without converting them. 422, nothing recorded."""


def currency_relabel(extracted: str | None, corrected: str) -> str | None:
    """Why `corrected` may not replace `extracted`, else None. A reviewer may resolve a missing
    or `XXX` currency, but changing a valid extracted code to AED would put foreign amounts
    into an AED e-invoice unchanged. Any other non-AED target is refused by the AED-only rule."""
    code = (extracted or "").strip().upper()
    target = corrected.strip().upper()
    if not _CURRENCY.match(code) or code in ("XXX", _DOCUMENT_CURRENCY):
        return None
    if target != _DOCUMENT_CURRENCY:
        return None
    return (
        f"{CURRENCY_FIELD}: the document's amounts are in {code}; relabelling them {target} "
        "without conversion is refused, so reject the invoice"
    )


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
    supplier_address: str | None = None
    """The reviewer's supplier address, set only when it differs from the extracted one; None
    keeps the invoice's own. Optional with a default so review signals recorded before ADR-0012
    amendment 4 still deserialize on replay."""


def _rate(raw: str) -> Decimal | None:
    text = raw.strip().removesuffix("%").strip()
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return value if value.is_finite() and value >= 0 else None


def corrections_from_fields(
    values: Mapping[str, str | None], *, extracted: Mapping[str, str | None]
) -> ReviewCorrections:
    """Builds `ReviewCorrections` from the latest value of each classified field row (`values`,
    a reviewer's row winning). Raises `MissingCorrectionsError` naming every field still missing
    or unreadable, or holding a value the e-invoice build refuses: a non-AED currency, or a line
    rate that does not fit its category (so the approval is refused before anything is
    recorded). `extracted` holds the model-sourced values the workflow's invoice was built from:
    a missing supplier TRN raises `SupplierTrnMissingError`, and a currency override that
    relabels a valid extracted code as AED raises `CurrencyRelabelError`. The supplier address
    (the latest value, a reviewer's winning) must name a UAE emirate; a reviewer's address that
    differs from the extracted one is carried as `supplier_address`."""
    if not (extracted.get(SUPPLIER_TRN_FIELD) or "").strip():
        raise SupplierTrnMissingError(
            "cannot approve: the document has no supplier TRN (BT-31); a reviewer's edit of "
            f"{SUPPLIER_TRN_FIELD} does not reach the e-invoice, so reject the invoice and ask "
            "the supplier for a corrected one"
        )
    problems: list[str] = []
    kind = _KIND_VALUES.get((values.get(KIND_FIELD) or "").strip())
    if kind is None:
        problems.append(KIND_FIELD)
    currency = (values.get(CURRENCY_FIELD) or "").strip().upper()
    if not _CURRENCY.match(currency) or currency == "XXX":
        problems.append(CURRENCY_FIELD)
    elif currency != _DOCUMENT_CURRENCY:
        # The PINT AE builder refuses a non-AED document (ADR-0012 amendment 2), so approving
        # one would only fail at the e-invoice build, after the approval was recorded.
        problems.append(f"{CURRENCY_FIELD} (must be AED, got {currency})")
    elif (relabel := currency_relabel(extracted.get(CURRENCY_FIELD), currency)) is not None:
        raise CurrencyRelabelError(f"cannot approve: {relabel}")
    address = (values.get(SUPPLIER_ADDRESS_FIELD) or "").strip()
    address_problem = resolve_emirate(address) is None
    if address_problem:
        # The builder takes the seller city and subdivision from this emirate (ibr-143-ae), so
        # approving without one would only fail at the e-invoice build.
        problems.append(SUPPLIER_ADDRESS_FIELD)
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
        if category is not None and rate is not None and not vat_rate_fits_category(category, rate):
            # apply_review refuses VAT_CATEGORY_RATE_MISMATCH and the builder refuses an E/O line
            # with a rate, so a mismatch is refused here, before the approval is recorded.
            problems.append(f"{line_rate_field(index)} ({rate}% does not fit category {category})")
        elif category is not None and rate is not None:
            lines.append(LineCorrection(index=index, vat_category=category, vat_rate=rate))
    if problems or kind is None:
        why = (
            "the supplier address names no UAE emirate (PINT AE ibr-143-ae); "
            if address_problem
            else ""
        )
        raise MissingCorrectionsError(
            f"cannot approve: {why}correct these fields first: " + ", ".join(problems)
        )
    corrected_address = (
        address if address != (extracted.get(SUPPLIER_ADDRESS_FIELD) or "").strip() else None
    )
    return ReviewCorrections(
        kind=kind, currency=currency, lines=tuple(lines), supplier_address=corrected_address
    )


def apply_corrections(invoice: Invoice, corrections: ReviewCorrections) -> Invoice:
    """`invoice` with the reviewer's kind, line categories/rates and, when given, supplier
    address. Every line must be covered
    exactly once; anything else is a `MissingCorrectionsError` (never a partial apply). So is a
    currency that would relabel the invoice's valid extracted code as AED (`currency_relabel`):
    raised as the base class, which `apply_review` already lists as non-retryable."""
    by_index = {c.index: c for c in corrections.lines}
    expected = set(range(1, len(invoice.lines) + 1))
    if set(by_index) != expected or len(by_index) != len(corrections.lines):
        raise MissingCorrectionsError(
            f"corrections cover lines {sorted(by_index)}, invoice has lines {sorted(expected)}"
        )
    cur = corrections.currency
    if (relabel := currency_relabel(invoice.total.currency, cur)) is not None:
        raise MissingCorrectionsError(f"invoice {invoice.id}: {relabel}")

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
    supplier = invoice.supplier
    if corrections.supplier_address is not None:
        supplier = supplier.model_copy(update={"address": corrections.supplier_address})
    return invoice.model_copy(
        update={
            "kind": corrections.kind,
            "supplier": supplier,
            "lines": lines,
            "subtotal": money(invoice.subtotal),
            "vat_total": money(invoice.vat_total),
            "total": money(invoice.total),
        }
    )
