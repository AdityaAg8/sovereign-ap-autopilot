"""Unit tests for ml/guard/metrics.py (PRD §10 M6.4): confusion counts and the cal-split
threshold rule (max F1, ties to the higher threshold, typed errors on bad input)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.guard.metrics import ThresholdSelectionError, confusion, select_threshold


def test_confusion_counts_and_rates() -> None:
    m = confusion([0.9, 0.8, 0.2, 0.6, 0.1], [1, 1, 1, 0, 0], 0.5)
    assert (m.tp, m.fn, m.fp, m.tn) == (2, 1, 1, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert m.fpr == pytest.approx(0.5)


def test_threshold_separates_perfectly_separable_scores_at_the_lowest_positive() -> None:
    assert select_threshold([0.1, 0.3, 0.7, 0.9], [0, 0, 1, 1]) == 0.7


def test_ties_go_to_the_higher_threshold() -> None:
    # F1 is 2/3 at both 0.3 and 0.9; the higher one (fewer false positives) wins.
    scores, labels = [0.1, 0.3, 0.5, 0.7, 0.9], [0, 1, 0, 0, 1]
    assert confusion(scores, labels, 0.3).f1 == pytest.approx(confusion(scores, labels, 0.9).f1)
    assert select_threshold(scores, labels) == 0.9


@pytest.mark.parametrize(
    ("scores", "labels"), [([], []), ([0.1, 0.2], [1, 1]), ([0.1, 0.2], [0, 0]), ([0.1], [0, 1])]
)
def test_bad_calibration_input_raises(scores: list[float], labels: list[int]) -> None:
    with pytest.raises(ThresholdSelectionError):
        select_threshold(scores, labels)


@given(
    st.lists(st.tuples(st.floats(0, 1), st.integers(0, 1)), min_size=2, max_size=40).filter(
        lambda xs: {lbl for _, lbl in xs} == {0, 1}
    )
)
def test_selected_threshold_is_optimal_over_candidates(pairs: list[tuple[float, int]]) -> None:
    scores = [s for s, _ in pairs]
    labels = [lbl for _, lbl in pairs]
    chosen = select_threshold(scores, labels)
    best = max(confusion(scores, labels, t).f1 for t in set(scores))
    assert chosen in scores
    assert confusion(scores, labels, chosen).f1 == best
