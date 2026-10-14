"""Unit and property tests for the Learn-then-Test fixed-sequence threshold (ADR-0009 §6):
`default_grid`, hand-computed `learn_then_test` cases checked against scipy, the skipped prefix,
lambda-hat-None on first-step failure, the label-free k_min testable grid, and monotonicity of
lambda-hat in alpha and delta."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binom

from apx.common.errors import TrustGateError
from apx.extract.interfaces import FieldName
from apx.trust.features import hard_reject
from apx.trust.interfaces import ArbiterVote, FieldFeatures, ThresholdResult, TrustConfig
from apx.trust.ltt import (
    LTT_MIN_ACCEPTED,
    CalibratedTrustGate,
    accepted_counts,
    default_grid,
    learn_then_test,
)

# --------------------------------------------------------------------------------------------
# default_grid
# --------------------------------------------------------------------------------------------


def test_default_grid_has_76_points() -> None:
    assert len(default_grid()) == 76


def test_default_grid_is_strictly_decreasing() -> None:
    grid = default_grid()
    assert all(grid[i] > grid[i + 1] for i in range(len(grid) - 1))


def test_default_grid_starts_at_one_and_ends_near_half() -> None:
    grid = default_grid()
    assert grid[0] == 1.0
    assert 0.49 < grid[-1] < 0.50


def test_default_grid_values_are_in_unit_interval() -> None:
    assert all(0.0 <= lam <= 1.0 for lam in default_grid())


# --------------------------------------------------------------------------------------------
# learn_then_test: hand-computed cases checked against scipy
# --------------------------------------------------------------------------------------------


def _config(grid: tuple[float, ...], alpha: float = 0.1, delta: float = 0.5) -> TrustConfig:
    return TrustConfig(alpha=alpha, delta=delta, grid=grid)


# The hand-computed cases below are far smaller than LTT_MIN_ACCEPTED; they test the binomial walk
# itself, so they set k_min = 1 (only k_j = 0 steps are untestable, the pre-amendment rule).
_WALK_ALL = 1


def test_learn_then_test_matches_scipy_binom_cdf_by_hand() -> None:
    # Hand-picked so every step accepts everything with score >= lambda; walk the grid by hand.
    p = [0.95, 0.9, 0.8, 0.7, 0.6]
    correct = [True, True, True, False, False]
    grid = (0.95, 0.9, 0.8, 0.7, 0.6)
    config = _config(grid, alpha=0.4, delta=0.9)
    result = learn_then_test(p, correct, config, min_accepted=_WALK_ALL)

    # k_j / m_j computed by hand from the nested accept sets:
    expected_k = [1, 2, 3, 4, 5]
    expected_m = [0, 0, 0, 1, 2]
    for step, k_j, m_j in zip(result.steps, expected_k, expected_m, strict=True):
        assert step.accepted == k_j
        assert step.wrong == m_j
        assert step.p_value == pytest.approx(float(binom.cdf(m_j, k_j, config.alpha)))


def test_learn_then_test_lambda_hat_is_last_passing_step() -> None:
    # At lambda=0.9: k=500, m=0 -> BinomCDF(0, 500, 0.01) ~= 0.0066 <= delta=0.05: passes.
    # At lambda=0.5: k=600, m=30 -> BinomCDF(30, 600, 0.01) ~= 1.0 > delta=0.05: fails.
    p = [0.9] * 500 + [0.5] * 100
    correct = [True] * 500 + [True] * 70 + [False] * 30
    grid = (0.9, 0.5)
    config = _config(grid, alpha=0.01, delta=0.05)
    result = learn_then_test(p, correct, config, min_accepted=_WALK_ALL)
    assert result.steps[0].accepted == 500
    assert result.steps[0].wrong == 0
    assert result.steps[0].passed is True
    assert result.steps[1].accepted == 600
    assert result.steps[1].wrong == 30
    assert result.steps[1].passed is False
    assert result.lambda_hat == 0.9


def test_learn_then_test_returns_the_full_grid_with_informational_rows_after_the_stop() -> None:
    """ADR-0009 §6/§7 (F3 fix): the returned step table always spans the whole grid, not just the
    walk's prefix up to the stop. Steps after the first failure are computed from the real LTT
    labels (not placeholders, unlike `skipped` steps) but marked `informational`: even when a
    later step's own p-value would independently pass, it must never move lambda_hat, because the
    fixed-sequence walk already stopped at the earlier failure."""
    p = [0.9] * 100 + [0.7] * 100 + [0.5] * 1000 + [0.3] * 5000
    correct = (
        [True] * 100  # lam=0.9: k=100,  m=0   -> passes, lambda_hat = 0.9
        + [False] * 100  # lam=0.7: k=200,  m=100 -> fails badly, the walk stops here
        + [True] * 1000  # lam=0.5: k=1200, m=100 -> would pass alone; informational only
        + [True] * 5000  # lam=0.3: k=6200, m=100 -> would pass alone; informational only
    )
    grid = (0.9, 0.7, 0.5, 0.3)
    config = _config(grid, alpha=0.1, delta=0.5)
    result = learn_then_test(p, correct, config, min_accepted=_WALK_ALL)

    assert len(result.steps) == len(grid)
    assert result.lambda_hat == 0.9

    passed_step, failed_step, after1, after2 = result.steps
    assert passed_step.passed is True
    assert passed_step.informational is False
    assert failed_step.passed is False
    assert failed_step.informational is False

    for step, k_j in ((after1, 1200), (after2, 6200)):
        assert step.informational is True
        assert step.skipped is False
        assert step.accepted == k_j
        assert step.wrong == 100
        assert step.p_value == pytest.approx(float(binom.cdf(100, k_j, config.alpha)))
        assert step.passed is True  # real, would-be-passing value -- yet lambda_hat stayed 0.9


def test_learn_then_test_lambda_hat_none_on_first_step_failure() -> None:
    p = [0.9] * 10
    correct = [False] * 6 + [True] * 4  # 60% wrong at the only threshold
    grid = (0.9,)
    config = _config(grid, alpha=0.01, delta=0.05)
    result = learn_then_test(p, correct, config, min_accepted=_WALK_ALL)
    assert result.lambda_hat is None
    assert result.steps[0].passed is False
    assert result.steps[0].skipped is False


def test_learn_then_test_skips_prefix_with_zero_accepted() -> None:
    p = [0.5, 0.5, 0.5]
    correct = [True, True, True]
    grid = (0.99, 0.9, 0.5)  # nothing scores >= 0.99 or >= 0.9
    # BinomCDF(0, 3, 0.1) = 0.9**3 = 0.729, which must be <= delta for the final step to pass.
    config = _config(grid, alpha=0.1, delta=0.8)
    result = learn_then_test(p, correct, config, min_accepted=_WALK_ALL)
    assert result.steps[0].skipped is True
    assert result.steps[0].p_value == 1.0
    assert result.steps[0].passed is False
    assert result.steps[1].skipped is True
    assert result.steps[2].skipped is False
    assert result.lambda_hat == 0.5


def test_learn_then_test_empty_input_returns_all_skipped_and_none() -> None:
    grid = default_grid()
    config = _config(grid)
    result = learn_then_test([], [], config)
    assert result.lambda_hat is None
    assert len(result.steps) == len(grid)
    assert all(step.skipped for step in result.steps)


def test_learn_then_test_raises_on_length_mismatch() -> None:
    with pytest.raises(TrustGateError):
        learn_then_test([0.9, 0.8], [True], _config((0.9, 0.8)))


def test_learn_then_test_raises_on_non_decreasing_grid() -> None:
    with pytest.raises(TrustGateError):
        learn_then_test([0.9], [True], _config((0.5, 0.9)))


def test_learn_then_test_raises_on_empty_grid() -> None:
    with pytest.raises(TrustGateError):
        learn_then_test([0.9], [True], _config(()))


def test_learn_then_test_calibrator_version_left_for_the_caller() -> None:
    result = learn_then_test([0.9, 0.8], [True, True], _config((0.9, 0.8), alpha=0.4, delta=0.9))
    assert result.calibrator_version == ""
    filled = result.model_copy(update={"calibrator_version": "abc123abc123"})
    assert filled.calibrator_version == "abc123abc123"


# --------------------------------------------------------------------------------------------
# k_min: the label-free testable grid (ADR-0009 §6 amendment)
# --------------------------------------------------------------------------------------------

_ADR_0009 = next((Path(__file__).resolve().parents[2] / "docs" / "adr").glob("0009-*.md"))


def test_k_min_is_the_named_constant_in_adr_0009() -> None:
    adr_text = _ADR_0009.read_text(encoding="utf-8")
    assert f"LTT_MIN_ACCEPTED = {LTT_MIN_ACCEPTED}" in adr_text
    default = inspect.signature(learn_then_test).parameters["min_accepted"].default
    assert default == LTT_MIN_ACCEPTED


def test_untestable_top_block_is_skipped_and_the_walk_continues() -> None:
    # The original defect: isotonic gives p = 1.0 to a tiny all-correct block. With k_min = 1 that
    # step is tested (k=10, m=0 -> 0.99**10 ~= 0.904 > delta), fails, and lambda-hat is None.
    p = [1.0] * 10 + [0.99] * 2000
    correct = [True] * 10 + [True] * 1995 + [False] * 5
    config = _config((1.0, 0.99), alpha=0.01, delta=0.05)

    old_rule = learn_then_test(p, correct, config, min_accepted=_WALK_ALL)
    assert old_rule.steps[0].skipped is False
    assert old_rule.steps[0].p_value == pytest.approx(0.99**10)
    assert old_rule.lambda_hat is None

    # With k_min = LTT_MIN_ACCEPTED the 10-field step is untestable (skipped, labels unread) and
    # the first testable step is k=2010, m=5: BinomCDF(5, 2010, 0.01) << 0.05, so it passes.
    result = learn_then_test(p, correct, config)
    assert result.steps[0].skipped is True
    assert result.steps[0].accepted == 10
    assert result.steps[0].wrong == 0
    assert result.steps[0].p_value == 1.0
    assert result.steps[1].skipped is False
    assert result.steps[1].accepted == 2010
    assert result.steps[1].wrong == 5
    assert result.steps[1].p_value == pytest.approx(float(binom.cdf(5, 2010, 0.01)))
    assert result.lambda_hat == 0.99


def test_fewer_fields_than_k_min_accepts_nothing() -> None:
    p = [0.999] * 100
    result = learn_then_test(p, [True] * 100, _config((0.999, 0.9), alpha=0.01, delta=0.05))
    assert result.lambda_hat is None
    assert all(step.skipped for step in result.steps)
    assert [step.accepted for step in result.steps] == [100, 100]


def test_min_accepted_below_one_raises() -> None:
    with pytest.raises(TrustGateError):
        learn_then_test([0.9], [True], _config((0.9,)), min_accepted=0)


def test_accepted_counts_matches_a_direct_count() -> None:
    p = [0.95, 0.9, 0.9, 0.5]
    grid = (1.0, 0.9, 0.6, 0.5)
    assert accepted_counts(p, grid) == (0, 3, 3, 4)


@given(
    p=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=60),
    labels=st.data(),
    min_accepted=st.integers(min_value=1, max_value=61),
)
@settings(max_examples=100)
def test_skip_decision_never_reads_labels(
    p: list[float], labels: st.DataObject, min_accepted: int
) -> None:
    """Which steps are skipped (and their recorded k_j) is a function of the scores alone: two
    arbitrary label vectors give the same skipped prefix, equal to the label-free count rule."""
    grid = _make_grid(p)
    config = _config(grid, alpha=0.05, delta=0.3)
    correct_a = [labels.draw(st.booleans()) for _ in p]
    correct_b = [labels.draw(st.booleans()) for _ in p]

    def skipped(result: ThresholdResult) -> list[tuple[float, int, int]]:
        return [(s.lam, s.accepted, s.wrong) for s in result.steps if s.skipped]

    counts = accepted_counts(p, grid)
    expected = [(lam, k, 0) for lam, k in zip(grid, counts, strict=True) if k < min_accepted]
    result_a = learn_then_test(p, correct_a, config, min_accepted=min_accepted)
    result_b = learn_then_test(p, correct_b, config, min_accepted=min_accepted)
    assert skipped(result_a) == skipped(result_b) == expected


# --------------------------------------------------------------------------------------------
# Property tests: monotonicity of lambda_hat in alpha and delta
# --------------------------------------------------------------------------------------------


def _sort_key(lam: float | None) -> float:
    """None (nothing accepted) is the most conservative outcome: worse than any real threshold."""
    return float("inf") if lam is None else lam


def _make_grid(p: list[float]) -> tuple[float, ...]:
    """A strictly decreasing grid covering every score in `p`, with at least two points."""
    values = sorted(set(p), reverse=True)
    if len(values) < 2:
        only = values[0]
        second = only / 2.0 if only > 0.0 else 1.0
        values = sorted({only, second}, reverse=True)
    return tuple(values)


@given(
    p=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=5, max_size=60),
    wrong_bits=st.data(),
    alpha_lo=st.floats(min_value=0.001, max_value=0.2),
    alpha_hi=st.floats(min_value=0.201, max_value=0.4),
)
@settings(max_examples=50)
def test_lambda_hat_is_monotone_in_alpha(
    p: list[float], wrong_bits: st.DataObject, alpha_lo: float, alpha_hi: float
) -> None:
    correct = [wrong_bits.draw(st.booleans()) for _ in p]
    grid = _make_grid(p)

    lo_config = _config(grid, alpha=alpha_lo, delta=0.3)
    hi_config = _config(grid, alpha=alpha_hi, delta=0.3)
    result_lo = learn_then_test(p, correct, lo_config, min_accepted=_WALK_ALL)
    result_hi = learn_then_test(p, correct, hi_config, min_accepted=_WALK_ALL)
    # Larger alpha (more tolerance) can only make the walk go at least as far (an equal or lower,
    # i.e. more permissive, lambda_hat) since BinomCDF(m, k, alpha) is non-increasing in alpha.
    assert _sort_key(result_hi.lambda_hat) <= _sort_key(result_lo.lambda_hat)


@given(
    p=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=5, max_size=60),
    wrong_bits=st.data(),
    delta_lo=st.floats(min_value=0.01, max_value=0.2),
    delta_hi=st.floats(min_value=0.201, max_value=0.6),
)
@settings(max_examples=50)
def test_lambda_hat_is_monotone_in_delta(
    p: list[float], wrong_bits: st.DataObject, delta_lo: float, delta_hi: float
) -> None:
    correct = [wrong_bits.draw(st.booleans()) for _ in p]
    grid = _make_grid(p)

    lo_config = _config(grid, alpha=0.05, delta=delta_lo)
    hi_config = _config(grid, alpha=0.05, delta=delta_hi)
    result_lo = learn_then_test(p, correct, lo_config, min_accepted=_WALK_ALL)
    result_hi = learn_then_test(p, correct, hi_config, min_accepted=_WALK_ALL)
    assert _sort_key(result_hi.lambda_hat) <= _sort_key(result_lo.lambda_hat)


# --------------------------------------------------------------------------------------------
# CalibratedTrustGate
# --------------------------------------------------------------------------------------------


class _FixedCalibrator:
    """A stub Calibrator returning a fixed P(correct) per field, for TrustGate wiring tests
    only -- this is not the unit under test for `learn_then_test`, only for `CalibratedTrustGate`
    routing hard-rejects and the threshold comparison correctly."""

    def __init__(self, scores: Mapping[FieldName, float]) -> None:
        self._scores = scores

    @property
    def method(self) -> str:
        return "fixed"

    @property
    def version(self) -> str:
        return "fixed-version"

    def fit(self, records: object) -> None:
        raise NotImplementedError

    def predict(self, items: list[tuple[FieldName, FieldFeatures]]) -> list[float]:
        return [self._scores[field] for field, _feats in items]


def _features(**overrides: object) -> FieldFeatures:
    base: dict[str, object] = {
        "agreement": 1.0,
        "a_mean_logprob": -0.1,
        "a_min_logprob": -0.2,
        "b_mean_logprob": -0.1,
        "b_min_logprob": -0.2,
        "format_valid": True,
        "ocr_support": 0.8,
        "arbiter_vote": None,
        "arbiter": ArbiterVote.NOT_RUN,
    }
    base.update(overrides)
    return FieldFeatures(**base)  # type: ignore[arg-type]


def test_trust_gate_accepts_only_fields_above_lambda_hat() -> None:
    calibrator = _FixedCalibrator({FieldName.TOTAL: 0.99, FieldName.SUBTOTAL: 0.5})
    gate = CalibratedTrustGate(calibrator, lambda_hat=0.9)  # type: ignore[arg-type]
    decision = gate.decide(
        {FieldName.TOTAL: _features(), FieldName.SUBTOTAL: _features()},
    )
    by_field = {d.field: d for d in decision.fields}
    assert by_field[FieldName.TOTAL].accepted is True
    assert by_field[FieldName.SUBTOTAL].accepted is False
    assert decision.all_accepted is False


def test_trust_gate_hard_rejects_are_never_accepted() -> None:
    calibrator = _FixedCalibrator({FieldName.TOTAL: 0.99})
    gate = CalibratedTrustGate(calibrator, lambda_hat=0.5)  # type: ignore[arg-type]
    hard_rejected_features = _features(a_logprob_missing=True)
    assert hard_reject(hard_rejected_features) is True
    decision = gate.decide({FieldName.TOTAL: hard_rejected_features})
    assert decision.fields[0].accepted is False
    assert decision.fields[0].hard_rejected is True


def test_trust_gate_accepts_nothing_when_lambda_hat_is_none() -> None:
    calibrator = _FixedCalibrator({FieldName.TOTAL: 1.0})
    gate = CalibratedTrustGate(calibrator, lambda_hat=None)  # type: ignore[arg-type]
    decision = gate.decide({FieldName.TOTAL: _features()})
    assert decision.fields[0].accepted is False
    assert decision.all_accepted is False
