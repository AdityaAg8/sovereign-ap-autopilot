"""Tests for Extractor B best-checkpoint selection (ADR-0011 §2): highest dev F1, ties within
epsilon broken by lower dev loss then the earlier step; the last step is not preferred."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from ml.extractor_b.interfaces import EvalResult
from ml.extractor_b.selection import select_best


def gen(step: int, f1: float, loss: float | None = None) -> EvalResult:
    return EvalResult(step=step, kind="generative", dev_f1=f1, dev_loss=loss)


def dev(step: int, loss: float) -> EvalResult:
    return EvalResult(step=step, kind="dev_loss", dev_loss=loss)


def test_highest_f1_wins_and_last_step_is_not_preferred() -> None:
    evals = [gen(229, 80.0), gen(458, 85.0), gen(687, 84.0), gen(916, 83.0)]
    assert select_best(evals, tie_epsilon=0.1) == 458


def test_tie_breaks_on_lower_dev_loss() -> None:
    evals = [gen(229, 85.00, 0.40), gen(458, 85.05, 0.50), gen(687, 80.0, 0.10)]
    assert select_best(evals, tie_epsilon=0.1) == 229


def test_tie_with_equal_loss_prefers_earlier_step() -> None:
    evals = [gen(458, 85.0, 0.3), gen(229, 85.0, 0.3)]
    assert select_best(evals, tie_epsilon=0.1) == 229


def test_generative_eval_borrows_dev_loss_logged_at_same_step() -> None:
    evals = [gen(100, 85.0), dev(100, 0.9), gen(200, 85.05), dev(200, 0.2)]
    assert select_best(evals, tie_epsilon=0.1) == 200


def test_outside_epsilon_is_not_a_tie() -> None:
    evals = [gen(100, 85.0, 0.1), gen(200, 85.2, 0.9)]
    assert select_best(evals, tie_epsilon=0.1) == 200


def test_dev_loss_only_evals_are_ignored() -> None:
    assert select_best([dev(50, 0.1), dev(100, 0.05)], tie_epsilon=0.1) is None
    assert select_best([dev(50, 0.01), gen(229, 70.0, 0.9)], tie_epsilon=0.1) == 229


def test_empty_is_none() -> None:
    assert select_best([], tie_epsilon=0.1) is None


@given(
    st.lists(
        st.tuples(st.integers(1, 1000), st.floats(0, 100), st.floats(0.01, 5)),
        min_size=1,
        max_size=20,
        unique_by=lambda t: t[0],
    )
)
def test_selected_is_within_epsilon_of_max(rows: list[tuple[int, float, float]]) -> None:
    evals = [gen(s, f, loss) for s, f, loss in rows]
    best = select_best(evals, tie_epsilon=0.1)
    chosen = next(e for e in evals if e.step == best)
    top = max(f for _, f, _ in rows)
    assert chosen.dev_f1 is not None and chosen.dev_f1 >= top - 0.1
