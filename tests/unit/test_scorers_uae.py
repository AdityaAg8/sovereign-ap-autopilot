"""Unit tests for evals/scorers/uae.py, UAE field F1 split by EN/AR (PRD §5 scoreboard row,
§10 M1.3), on hand-computed cases."""

from __future__ import annotations

from evals.scorers.uae import UaeDocument, score_uae


def test_score_uae_splits_by_language_and_pools_overall() -> None:
    gold = [
        UaeDocument("en1", "en", {"total": "100.00"}),
        UaeDocument("en2", "en", {"total": "200.00"}),
        UaeDocument("ar1", "ar", {"total": "300.00"}),
    ]
    predicted = {
        "en1": {"total": "100.00"},  # correct
        "en2": {"total": "999.00"},  # wrong
        "ar1": {"total": "300.00"},  # correct
    }

    result = score_uae(gold, predicted)

    # EN: 1 TP, 1 FP (en2 wrong value), 1 FN (en2 gold value never matched) -> P=0.5, R=0.5
    assert result.en.true_positives == 1
    assert result.en.false_positives == 1
    assert result.en.false_negatives == 1
    assert result.en.precision == 0.5
    assert result.en.recall == 0.5

    # AR: perfect match on its single field.
    assert result.ar.true_positives == 1
    assert result.ar.false_positives == 0
    assert result.ar.false_negatives == 0
    assert result.ar.f1 == 1.0

    # Overall pools both languages: TP=2, FP=1, FN=1.
    assert result.overall.true_positives == 2
    assert result.overall.false_positives == 1
    assert result.overall.false_negatives == 1


def test_score_uae_language_with_no_documents_is_vacuously_perfect() -> None:
    gold = [UaeDocument("en1", "en", {"total": "100.00"})]
    predicted = {"en1": {"total": "100.00"}}

    result = score_uae(gold, predicted)

    assert result.ar.precision == 1.0
    assert result.ar.recall == 1.0
    assert result.ar.f1 == 1.0
    assert result.ar.true_positives == 0


def test_score_uae_missing_prediction_counts_as_false_negative_in_its_language() -> None:
    gold = [UaeDocument("ar1", "ar", {"vendor_name_ar": "شركة"})]
    predicted: dict[str, dict[str, str]] = {}

    result = score_uae(gold, predicted)

    assert result.ar.false_negatives == 1
    assert result.ar.recall == 0.0
    assert result.en.true_positives == 0
    assert result.en.precision == 1.0  # EN subset is empty on both sides
