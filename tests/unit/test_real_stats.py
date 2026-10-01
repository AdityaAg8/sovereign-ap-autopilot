"""Unit tests for evals.real.stats (ADR-0019 §3): micro F1, the document bootstrap, the paired
bootstrap of a difference, the mean bootstrap and the Wilson interval. Pure CPU."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evals.real.stats import (
    bootstrap_ci,
    f1_metric,
    mean_metric,
    micro_f1,
    paired_delta_ci,
    quantile,
    rate_metric,
    wilson,
)

DOCS = [(3, 1, 0), (2, 0, 1), (5, 2, 2), (0, 1, 3), (4, 0, 0), (1, 1, 1), (6, 1, 0), (2, 2, 2)]


def test_micro_f1_sums_counts_before_dividing() -> None:
    assert micro_f1([(1, 0, 0), (0, 1, 1)]) == pytest.approx(2 / 4)
    assert micro_f1([(0, 0, 0)]) == 1.0
    assert micro_f1([(0, 2, 0)]) == 0.0


def test_bootstrap_ci_contains_the_point_estimate_and_is_deterministic() -> None:
    ci = bootstrap_ci(DOCS)
    assert ci.method == "bootstrap"
    assert ci.low <= micro_f1(DOCS) <= ci.high
    assert bootstrap_ci(DOCS) == ci
    assert bootstrap_ci(DOCS, seed=1) != ci


def test_bootstrap_of_identical_documents_is_a_point() -> None:
    ci = bootstrap_ci([(2, 1, 1)] * 10)
    assert ci.low == pytest.approx(ci.high)


def test_bootstrap_refuses_no_documents() -> None:
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_ci([])


def test_f1_metric_carries_n() -> None:
    metric = f1_metric("field_f1", DOCS)
    assert metric.n_docs == len(DOCS)
    assert metric.value == micro_f1(DOCS)


def test_paired_delta_of_an_arm_with_itself_is_zero() -> None:
    delta = paired_delta_ci(DOCS, DOCS)
    assert delta.value == 0.0
    assert delta.ci.low == delta.ci.high == 0.0
    assert delta.ci.method == "paired_bootstrap"


def test_paired_delta_sign_and_interval() -> None:
    better = [(tp + fn, fp, 0) for tp, fp, fn in DOCS]  # every miss found
    delta = paired_delta_ci(DOCS, better)
    assert delta.value == pytest.approx(micro_f1(better) - micro_f1(DOCS))
    assert 0 < delta.ci.low <= delta.value <= delta.ci.high


def test_paired_delta_refuses_unequal_arms() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        paired_delta_ci(DOCS, DOCS[:-1])


def test_wilson_bounds() -> None:
    assert wilson(0, 50).low == 0.0
    assert wilson(50, 50).high == 1.0
    ci = wilson(40, 50)
    assert ci.low < 0.8 < ci.high
    assert wilson(0, 0).low == 0.0 and wilson(0, 0).high == 1.0
    with pytest.raises(ValueError):
        wilson(3, 2)


@given(st.integers(min_value=1, max_value=500), st.data())
def test_wilson_contains_the_rate(n: int, data: st.DataObject) -> None:
    k = data.draw(st.integers(min_value=0, max_value=n))
    ci = wilson(k, n)
    assert 0.0 <= ci.low <= k / n <= ci.high <= 1.0


def test_rate_metric_and_quantile() -> None:
    assert rate_metric("valid_json_rate", 9, 10).value == 0.9
    assert quantile([], 0.5) is None
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


@settings(max_examples=25, deadline=None)
@given(st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=30))
def test_mean_metric_ci_brackets_the_mean(values: list[float]) -> None:
    metric = mean_metric("nted_accuracy", values)
    assert metric.ci.low - 1e-9 <= metric.value <= metric.ci.high + 1e-9
