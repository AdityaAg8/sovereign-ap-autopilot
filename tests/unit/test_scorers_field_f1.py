"""Unit tests for evals/scorers/field_f1.py, the shared normalized-exact-match field F1 core
(PRD §10 M1.3), on hand-computed cases."""

from __future__ import annotations

from evals.scorers.field_f1 import FieldRecord, field_f1, normalize_value


def test_normalize_value_collapses_whitespace() -> None:
    assert normalize_value("  Acme   Corp \n") == "Acme Corp"


def test_normalize_value_folds_unicode_compatibility_forms() -> None:
    # U+FF11 FULLWIDTH DIGIT ONE NFKC-normalizes to ASCII "1".
    assert normalize_value("\uff11") == "1"


def test_field_f1_perfect_match() -> None:
    gold = [FieldRecord("doc1", "total", "100.00"), FieldRecord("doc1", "vat", "5.00")]
    predicted = [FieldRecord("doc1", "total", "100.00"), FieldRecord("doc1", "vat", "5.00")]

    result = field_f1(gold, predicted)

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_field_f1_hand_computed_mixed_case() -> None:
    # 2 docs x 2 fields = 4 gold fields. Predictions: 2 correct, 1 wrong value (false positive
    # + false negative), 1 field not predicted at all (false negative only).
    gold = [
        FieldRecord("doc1", "total", "100.00"),
        FieldRecord("doc1", "vat", "5.00"),
        FieldRecord("doc2", "total", "200.00"),
        FieldRecord("doc2", "vat", "10.00"),
    ]
    predicted = [
        FieldRecord("doc1", "total", "100.00"),  # correct
        FieldRecord("doc1", "vat", "999.00"),  # wrong value
        FieldRecord("doc2", "total", "200.00"),  # correct
        # doc2/vat not predicted at all
    ]

    result = field_f1(gold, predicted)

    # TP = {(doc1,total,100.00), (doc2,total,200.00)} = 2
    # FP = {(doc1,vat,999.00)} = 1
    # FN = {(doc1,vat,5.00), (doc2,vat,10.00)} = 2
    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 2
    assert result.precision == 2 / 3
    assert result.recall == 2 / 4
    expected_f1 = 2 * (2 / 3) * (1 / 2) / ((2 / 3) + (1 / 2))
    assert abs(result.f1 - expected_f1) < 1e-9


def test_field_f1_empty_gold_and_predicted_is_vacuously_perfect() -> None:
    result = field_f1([], [])
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_field_f1_all_false_positives_when_gold_is_empty() -> None:
    predicted = [FieldRecord("doc1", "total", "100.00")]
    result = field_f1([], predicted)
    assert result.precision == 0.0
    assert result.recall == 1.0  # nothing to find, so recall is vacuously perfect
    assert result.f1 == 0.0


def test_field_f1_normalizes_before_matching() -> None:
    gold = [FieldRecord("doc1", "name", "  Acme   Corp  ")]
    predicted = [FieldRecord("doc1", "name", "Acme Corp")]
    result = field_f1(gold, predicted)
    assert result.true_positives == 1
    assert result.f1 == 1.0
