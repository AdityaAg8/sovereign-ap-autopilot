"""Unit tests for evals/scorers/fatura.py on hand-computed cases."""

from __future__ import annotations

from evals.scorers.fatura import score_fatura


def test_score_fatura_perfect_match() -> None:
    gold = {"inv1": {"invoice_number": "INV-001", "total": "100.00"}}
    predicted = {"inv1": {"invoice_number": "INV-001", "total": "100.00"}}

    result = score_fatura(gold, predicted)

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.f1 == 1.0


def test_score_fatura_hand_computed_partial_match() -> None:
    gold = {
        "inv1": {"invoice_number": "INV-001", "total": "100.00"},
        "inv2": {"invoice_number": "INV-002", "total": "200.00"},
    }
    predicted = {
        "inv1": {"invoice_number": "INV-001", "total": "999.00"},  # total wrong
        "inv2": {"invoice_number": "INV-002", "total": "200.00"},  # both correct
    }

    result = score_fatura(gold, predicted)

    # TP: (inv1,invoice_number), (inv2,invoice_number), (inv2,total) = 3
    # FP: (inv1,total,999.00) = 1
    # FN: (inv1,total,100.00) = 1
    assert result.true_positives == 3
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.precision == 3 / 4
    assert result.recall == 3 / 4
    assert abs(result.f1 - 3 / 4) < 1e-9


def test_score_fatura_missing_document_is_all_false_negatives() -> None:
    gold = {"inv1": {"invoice_number": "INV-001"}}
    predicted: dict[str, dict[str, str]] = {}

    result = score_fatura(gold, predicted)

    assert result.true_positives == 0
    assert result.false_negatives == 1
    assert result.recall == 0.0
    assert result.precision == 1.0  # nothing predicted, vacuously perfect precision
