"""Unit and property tests for `apx.learning.selection.score_candidate`: the deterministic,
fixed-weight active-learning score is bounded, has no learned parameters, and is monotonic in
each of its three signals (PRD §10 M10.1). The database-backed selection query itself is
covered by `tests/integration/test_corrections.py` (needs real Postgres data to exercise
confidence/disagreement/correction-frequency together).
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from apx.learning.selection import (
    CORRECTED_OFTEN_WEIGHT,
    DISAGREEMENT_WEIGHT,
    LOW_CONFIDENCE_WEIGHT,
    score_candidate,
)

_BOOL = st.booleans()


def test_weights_sum_to_one() -> None:
    """The score is meant to land in `[0, 1]`; anchor that on the documented weights rather
    than only on `score_candidate`'s output, so a future edit to the weights that breaks the
    invariant is caught here directly."""
    assert LOW_CONFIDENCE_WEIGHT + DISAGREEMENT_WEIGHT + CORRECTED_OFTEN_WEIGHT == 1.0


def test_no_signal_scores_zero() -> None:
    assert score_candidate(low_confidence=False, disagreement=False, corrected_often=False) == 0.0


def test_every_signal_scores_one() -> None:
    assert score_candidate(low_confidence=True, disagreement=True, corrected_often=True) == 1.0


@given(low_confidence=_BOOL, disagreement=_BOOL, corrected_often=_BOOL)
def test_score_is_bounded(low_confidence: bool, disagreement: bool, corrected_often: bool) -> None:
    score = score_candidate(
        low_confidence=low_confidence, disagreement=disagreement, corrected_often=corrected_often
    )
    assert 0.0 <= score <= 1.0


@given(disagreement=_BOOL, corrected_often=_BOOL)
def test_score_is_monotonic_in_low_confidence(disagreement: bool, corrected_often: bool) -> None:
    without = score_candidate(
        low_confidence=False, disagreement=disagreement, corrected_often=corrected_often
    )
    with_ = score_candidate(
        low_confidence=True, disagreement=disagreement, corrected_often=corrected_often
    )
    assert with_ > without


@given(low_confidence=_BOOL, corrected_often=_BOOL)
def test_score_is_monotonic_in_disagreement(low_confidence: bool, corrected_often: bool) -> None:
    without = score_candidate(
        low_confidence=low_confidence, disagreement=False, corrected_often=corrected_often
    )
    with_ = score_candidate(
        low_confidence=low_confidence, disagreement=True, corrected_often=corrected_often
    )
    assert with_ > without


@given(low_confidence=_BOOL, disagreement=_BOOL)
def test_score_is_monotonic_in_corrected_often(low_confidence: bool, disagreement: bool) -> None:
    without = score_candidate(
        low_confidence=low_confidence, disagreement=disagreement, corrected_often=False
    )
    with_ = score_candidate(
        low_confidence=low_confidence, disagreement=disagreement, corrected_often=True
    )
    assert with_ > without


@given(low_confidence=_BOOL, disagreement=_BOOL, corrected_often=_BOOL)
def test_score_is_pure_and_deterministic(
    low_confidence: bool, disagreement: bool, corrected_often: bool
) -> None:
    first = score_candidate(
        low_confidence=low_confidence, disagreement=disagreement, corrected_often=corrected_often
    )
    second = score_candidate(
        low_confidence=low_confidence, disagreement=disagreement, corrected_often=corrected_often
    )
    assert first == second
