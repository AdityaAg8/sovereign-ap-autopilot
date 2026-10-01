"""`apx.trust.gate.refuse_truncated_parse` (ADR-0013 amendment 1): a parse with a page the OCR
service cut short accepts no field and names the pages, keeping any refusal the calibrated gate
already made; a complete parse leaves the decision exactly as the gate made it."""

from __future__ import annotations

from apx.extract.interfaces import FieldName
from apx.trust.gate import PARSE_TRUNCATED, refuse_truncated_parse
from apx.trust.interfaces import FieldDecision, TrustDecision


def _accepted(refusal: str | None = None) -> TrustDecision:
    return TrustDecision(
        fields=tuple(FieldDecision(field=n, p_correct=0.99, accepted=True) for n in FieldName),
        all_accepted=refusal is None,
        calibrator_version="cal-1",
        refusal=refusal,
    )


def test_a_complete_parse_leaves_the_decision_unchanged() -> None:
    decision = _accepted()
    assert refuse_truncated_parse(decision, (), 4) is decision


def test_a_truncated_page_accepts_nothing_and_names_the_pages() -> None:
    refused = refuse_truncated_parse(_accepted(), (0, 2), 4)

    assert not refused.all_accepted
    assert not any(f.accepted for f in refused.fields)
    assert [f.p_correct for f in refused.fields] == [0.99] * len(FieldName)
    assert refused.refusal == f"{PARSE_TRUNCATED}: OCR cut short on page(s) 1, 3 of 4"
    assert refused.calibrator_version == "cal-1"


def test_an_earlier_gate_refusal_is_kept() -> None:
    earlier = "calibrator_missing: no active calibrator"
    refused = refuse_truncated_parse(_accepted(earlier), (1,), 2)
    assert refused.refusal == f"{PARSE_TRUNCATED}: OCR cut short on page(s) 2 of 2; {earlier}"
