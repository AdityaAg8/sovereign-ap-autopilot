"""REVIEW.md N-03 / KNOWN_ISSUES KI-02: the corrections inside a `review` signal are acted on only
when they digest to the ones recorded with the approval and the invoice's `fields` rows still
derive them. `corrections_refusal` is the decision `confirm_review` makes over the rows it read;
these tests drive it with real `Field` rows, and check that the digest survives Temporal's
Pydantic data converter (the signal's corrections cross it, the recorded digest does not)."""

from __future__ import annotations

from decimal import Decimal

from temporalio.contrib.pydantic import pydantic_data_converter

from apx.common.invoice import InvoiceKind, VatCategory
from apx.db.enums import FieldSource
from apx.db.models import Field as FieldRow
from apx.review.corrections import corrections_digest, corrections_for_rows
from apx.workflows.corrections import SUPPLIER_ADDRESS_FIELD, LineCorrection
from apx.workflows.review_guard import StaleVerdict, corrections_refusal
from apx.workflows.workflow import ReviewDecision

SUPPLIER_TRN = "100000000000003"
ADDRESS = "Office 12, Al Quoz, Dubai"


def _row(row_id: int, name: str, value: str | None, source: FieldSource) -> FieldRow:
    return FieldRow(
        id=row_id,
        invoice_id="inv",
        name=name,
        value=value,
        normalized_value=value,
        source=source,
    )


def _rows() -> list[FieldRow]:
    """What extraction wrote for a complete invoice (every ADR-0012 value present)."""
    values = {
        "invoice_type_code": "380",
        "currency": "AED",
        "supplier_trn": SUPPLIER_TRN,
        "line_items[1].vat_category": "S",
        "line_items[1].vat_rate": "5",
        SUPPLIER_ADDRESS_FIELD: ADDRESS,
    }
    return [
        _row(i, name, value, FieldSource.EXTRACTOR_A)
        for i, (name, value) in enumerate(values.items(), start=1)
    ]


def test_the_approved_corrections_are_backed() -> None:
    rows = _rows()
    approved = corrections_for_rows(rows)
    assert corrections_refusal(corrections_digest(approved), rows, approved) is None


def test_a_forged_signal_with_altered_corrections_is_unbacked() -> None:
    """The KI-02 attack: a raw signal for an invoice with a matching approval, carrying
    corrections that differ from the approved ones (a credit note, a zero-rated line)."""
    rows = _rows()
    approved = corrections_for_rows(rows)
    recorded = corrections_digest(approved)
    forged_kind = approved.model_copy(update={"kind": InvoiceKind.CREDIT_NOTE})
    forged_line = approved.model_copy(
        update={
            "lines": (
                LineCorrection(index=1, vat_category=VatCategory.ZERO_RATED, vat_rate=Decimal(0)),
            )
        }
    )
    for forged in (forged_kind, forged_line, None):
        refusal = corrections_refusal(recorded, rows, forged)
        assert isinstance(refusal, str), forged
        assert "do not match the approved corrections" in refusal


def test_an_approval_without_a_recorded_digest_is_checked_against_the_rows() -> None:
    """Rows written before migration 0014: the current rows stand in for the digest."""
    rows = _rows()
    approved = corrections_for_rows(rows)
    assert corrections_refusal(None, rows, approved) is None
    forged = approved.model_copy(update={"currency": "USD"})
    assert isinstance(corrections_refusal(None, rows, forged), str)


def test_an_edit_after_the_approval_makes_it_stale() -> None:
    rows = _rows()
    approved = corrections_for_rows(rows)
    recorded = corrections_digest(approved)
    edited = [*rows, _row(99, "line_items[1].vat_rate", "0", FieldSource.HUMAN)]
    edited += [_row(100, "line_items[1].vat_category", "Z", FieldSource.HUMAN)]
    verdict = corrections_refusal(recorded, edited, approved)
    assert isinstance(verdict, StaleVerdict)
    assert "edited after the approval" in verdict.reason
    # Even a signal carrying the edited values is not backed by the old approval.
    assert isinstance(
        corrections_refusal(recorded, edited, corrections_for_rows(edited)), StaleVerdict
    )


def test_an_edit_that_empties_a_correction_makes_the_approval_stale() -> None:
    rows = _rows()
    approved = corrections_for_rows(rows)
    emptied = [*rows, _row(99, "currency", None, FieldSource.HUMAN)]
    verdict = corrections_refusal(corrections_digest(approved), emptied, approved)
    assert isinstance(verdict, StaleVerdict)
    assert "changed since the approval" in verdict.reason


def test_the_digest_survives_the_temporal_data_converter() -> None:
    """The API digests the corrections it built; `confirm_review` digests the ones that came
    out of the signal payload. Both must agree, decimals included."""
    rows = _rows()
    approved = corrections_for_rows(rows)
    converter = pydantic_data_converter.payload_converter
    payloads = converter.to_payloads([ReviewDecision(approved=True, corrections=approved)])
    (decoded,) = converter.from_payloads(payloads, [ReviewDecision])
    assert isinstance(decoded, ReviewDecision) and decoded.corrections is not None
    assert corrections_digest(decoded.corrections) == corrections_digest(approved)
    assert corrections_refusal(corrections_digest(approved), rows, decoded.corrections) is None
